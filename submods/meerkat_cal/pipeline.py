"""facetselfcal --meerkatcal: CASA-free calibration of MeerKAT archive data (all fields in one MS).

Stages (each is skipped when it finished in an earlier run in the same directory):
discover   fields, scan intents, band, calibration level, polarisation calibrator checks
prepare    split the calibrator scans, feed swap, flagging
bandpass   delay (treeMF), phaseonly, complexgain (bandpass) and amplitudeonly on the bandpass calibrator(s),
           solution QA, flag cycles
leakage    leakage on the unpolarised bandpass calibrator (level leakage and full)
phasecal   phase calibrator model (short facetselfcal run), delay, phaseonly and scalaramplitude
polangle   cross-hand phase (polarisation angle) with xfsolve (level full)
polcheck   images of the polarisation calibrator(s), polarisation compared with the model (level full)
targets    apply everything to the target scans, parallactic angle, average, flag, concatenate the scans of
           each target in time (--meerkatcal-target-concat)
"""
import glob
import gzip
import hashlib
import json
import os
import re
import shutil

import numpy as np
import tables
from casacore.tables import table  # type: ignore

from submods.applycal import applycal_dp3_steps
from submods.calibrator_models import setjy_dp3
from submods.flagging import aoflagger_column
from submods.h5_helpers.h5_info import set_refant
from submods.h5_helpers.solution_qa import (fix_leakage_gauge, freq_ranges, merge_bandpasses, qa_crosshand,
                                            qa_gains, qa_leakage, qa_real)
from submods.meerkat_cal import diode, plots, polcheck, weightgain
from submods.meerkat_cal.discover import (UNPOLARISED_CALIBRATORS, discover, field_locations, input_mslist,
                                          scan_location, split_input)
from submods.meerkat_cal.ionosphere import make_rm_h5
from submods.meerkat_cal.prepare import (DYSCO_ARGS, DYSCO_ARGS_TARGET, flag_fraction, flag_residual, freq_factor, nchan_divisible, preflag_parset,
                                         prepare_calibrator_scan, set_receptor_angle_zero, snapshot_flags,
                                         split_scan, swap_parset, time_factor)
from submods.meerkat_cal.solve import (apply, combine_xf, concat_h5_time, fold_transfer_terms,
                                       normalise_amplitudes, nthreads,
                                       reference, solve_ddecal, solve_delay, solve_xf, solve_xf_diode,
                                       unwrap_phase)
from submods.ms_fixes import fix_uvw
from submods.ms_info import choose_refant_from_ms
from submods.model_columns import predictsky
from submods.runtime_utils import get_logger, run, terminal_print
from submods.skymodels import set_MeerKAT_bandpass_skymodel

STAGES = ['discover', 'prepare', 'bandpass', 'leakage', 'phasecal', 'polangle', 'polcheck', 'targets']
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMODPATH = os.path.join(REPO, 'submods')
PIXELSCALE = {'UHF': 1.8, 'L': 1.0}  # arcsec, as facetselfcal --auto; S-band 0.5
BANDWIDTH = {'UHF': 544e6, 'L': 856e6, 'S': 875e6}  # Hz, the nominal digitised bands (S: each sub-band)

# solutions applied to the targets, named <role>_<soltype> after the DP3/facetselfcal solution types
BANDPASS_H5 = 'h5_solutions/meerkatcal_bpcal_complexgain.h5'
# the bandpass shape without the delay and X-Y phase offset folded in (plots and report)
BANDPASS_SHAPE_H5 = 'h5_solutions/meerkatcal_bpcal_complexgain_shape.h5'
LEAKAGE_H5 = 'h5_solutions/meerkatcal_bpcal_leakage.h5'
PHASECAL_H5 = {soltype: f'h5_solutions/meerkatcal_phasecal_{soltype}.h5'
               for soltype in ('delay', 'phaseonly', 'scalaramplitude')}
XFSOLVE_H5 = 'h5_solutions/meerkatcal_polcal_xfsolve.h5'
# noise floor of the QA of per-dump gains: MeerKAT gains jump and drift between dumps (which these
# solves absorb), so only deviations above ~10% in amplitude or ~6 deg in phase are outliers
TIME_QA_MINSCALE = 0.02
# noise floor of the QA of the per-scan phase calibrator amplitudes along time: the gains drift between
# scans (elevation, temperature), so only an amplitude more than 5 x 5% = 25% off the running median of
# its neighbouring scans is an outlier (a bad antenna or scan, not a drift)
PHASECAL_AMP_QA_MINSCALE = 0.05
# noise floor (s) of the QA of the delays along time: the two or more solution intervals within one
# calibrator scan are nearly identical, so the noise from neighbouring differences is ~0.001 ns and a
# harmless ~0.01 ns change between scans would be an outlier (flagging the antenna for the target
# scans on both sides); only a delay more than 5 x 0.05 = 0.25 ns off (~80 deg across L-band) is flagged
DELAY_QA_MINSCALE = 0.05e-9


def _marker(stage):
    return f'misc/meerkatcal_{stage}.done'


def _done(stage):
    return os.path.isfile(_marker(stage))


def _finish(stage, summary):
    with open(_marker(stage), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    terminal_print(f'meerkatcal: stage {stage} finished')


def _summary(stage):
    """Summary of a finished stage (the JSON in its marker), or None."""
    return json.load(open(_marker(stage))) if _done(stage) else None


def _weightgain_summary():
    """Reference level and per-scan gain range of --meerkatcal-weightgain (None when it was not used)."""
    if not os.path.isfile(weightgain.REFERENCE_JSON):
        return None
    ref = json.load(open(weightgain.REFERENCE_JSON))
    return dict(wref=ref['wref'], scans=ref.get('scans', {}))


def write_report_json(info, args=None):
    """
    Collect what the HTML report (Bandpass and Polarisation pages) shows in misc/meerkatcal_report.json.
    Missing stages give null entries, so the report works for every level and partial runs.
    """
    def existing(path):
        return path if os.path.isfile(path) else None

    leak, pol, polchk = _summary('leakage'), _summary('polangle'), _summary('polcheck')
    rm = {}
    # RM files of an earlier run with the correction on are not shown when it is off now
    for f in sorted(glob.glob('h5_solutions/RM_*.json')) if args is None or _ionorm(args) else []:
        rm[os.path.basename(f)[3:-5]] = json.load(open(f))
    flagging = {stage: (_summary(stage) or {}).get('flagged') for stage in ('bandpass', 'leakage', 'phasecal', 'targets')}
    if _done('prepare'):
        flagging['prepare'] = _summary('prepare')
    report = dict(
        band=info['band'], level=info['level'], angle=info.get('angle'),
        refant=open('misc/refant.txt').read().strip() if os.path.isfile('misc/refant.txt') else None,
        roles=info['roles'],
        bandpass_h5=existing(BANDPASS_SHAPE_H5),
        leakage=None if leak is None else dict(h5=existing(LEAKAGE_H5),
                                               calibrators=leak['calibrator'], nchan=leak['nchan'],
                                               gauge_c=leak['qa'].get('gauge_c'),
                                               qa=leak['qa'].get(f"h5_solutions/bpcal_leakage_c{_last_cycle(leak['qa'], 'bpcal_leakage')}_merged.h5")),
        xfsolve=None if pol is None or 'skipped' in pol else dict(
            h5=existing(XFSOLVE_H5), calibrators=pol['calibrators'], nchan=pol['nchan'],
            qa=pol['qa'], perscan_median_diff_deg=pol['perscan_median_diff_deg'],
            perscan_unconstrained_mhz=pol.get('perscan_unconstrained_mhz', {}), warnings=pol.get('warnings', []),
            perscan_h5=[sc['xfsolve'] for sc in pol['scans'].values()], diode=pol.get('diode')),
        angle_skipped=None if pol is None else pol.get('skipped'),
        rm=rm, polcheck=polchk, flagging=flagging, weightgain=_weightgain_summary())
    with open('misc/meerkatcal_report.json', 'w') as f:
        json.dump(report, f, indent=1, default=str)


def _last_cycle(qa, prefix):
    """Cycle number of the last merged solution of a solve type in a QA dictionary."""
    cycles = [int(re.match(r'_c(\d+)_', os.path.basename(k)[len(prefix):]).group(1)) for k in qa
              if re.match(re.escape(prefix) + r'_c\d+_', os.path.basename(k))]
    return max(cycles) if cycles else 0


def _warn(message, collect=None):
    """Print a warning, write it to logs/selfcal.log and add it to collect (shown in the HTML report)."""
    terminal_print('WARNING: ' + message)
    logger = get_logger()
    if logger is not None:
        logger.warning(message)
    if collect is not None:
        collect.append(message)


def _mhz_text(ranges):
    if not ranges:
        return 'none'
    return ', '.join(f'{lo:.0f}' + (f'-{hi:.0f}' if hi > lo else '') for lo, hi in ranges) + ' MHz'


def _xf_constraint(sums):
    """rms fraction of the model polarisation in the cross-hands per solution (None for old sums files)."""
    if 'upow' not in sums.files:
        return None
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(sums['upow'][0] / sums['ppow'][0])


# the cross-hand phase criterion (--meerkatcal-xf-minfrac auto): the model angle of the polarisation
# calibrator must be further from a zero of its feed-frame Stokes U than the expected angle error, a model
# error plus an ionospheric RM error times lambda^2 (UHF test data: solutions flipped by 180 deg below about
# 0.2 at 620-710 MHz, none above)
XF_ANGLE_ERROR_DEG = 1.
XF_RM_ERROR = 0.5  # rad/m^2


def xf_min_fraction(freqs, minfrac=None):
    """
    Minimum fraction of the model polarisation in the feed-frame cross-hands (|U|/P) for a cross-hand
    phase solution, per frequency: a fixed minfrac, or (None) sin(2 delta) with
    delta = XF_ANGLE_ERROR_DEG + XF_RM_ERROR * lambda^2, the angle the model may be off before the sign of
    U, and with it the solution, flips by 180 deg (e.g. 0.25 at 650 MHz, 0.14 at 925 MHz, 0.08 at 1.4 GHz).
    """
    freqs = np.asarray(freqs, dtype=float)
    if minfrac is not None:
        return np.full(freqs.shape, float(minfrac))
    delta = np.radians(XF_ANGLE_ERROR_DEG) + XF_RM_ERROR * (299792458. / freqs) ** 2
    return np.sin(np.minimum(2. * delta, 0.5 * np.pi))


def _minfrac_text(minfrac):
    """Text for the threshold of xf_min_fraction (array), e.g. '0.14 (925 MHz) to 0.07 (1627 MHz)'."""
    m = np.asarray(minfrac)
    return f'{m.max():.2f}' if np.ptp(m) < 0.005 else f'{m.max():.2f} to {m.min():.2f} (decreasing with frequency)'


def _scan_ms(name, scan):
    return f'cal/{name}.scan{scan:03d}.ms'


def calibrator_scans(info, role):
    """The prepared scan MSs of the calibrators with a role (bpcal, phasecal, polcal)."""
    return [_scan_ms(n, s) for n in info['roles'][role] for s in info['fields'][n]['scans']]


def _weightgain_h5(name, scan):
    return f'h5_solutions/meerkatcal_weightgain_{name}.scan{scan:03d}.h5'


def use_weightgain(args):
    """
    Whether the array-wide weight gain (--meerkatcal-weightgain) is applied, to the calibrators (prepare stage)
    and the targets alike. It is on when the option is given or when the calibrator scans were prepared with
    it (its reference file exists), so that a rerun without the option keeps the targets consistent with the
    calibrators. Giving the option after calibrator scans were prepared without it is an error.
    """
    applied = os.path.isfile(weightgain.REFERENCE_JSON)
    if args['meerkatcal_weightgain'] and not applied and glob.glob('cal/*.ms.prepared'):
        raise ValueError('--meerkatcal-weightgain must be applied to the calibrator scans as well, but they were '
                         'prepared without it: remove cal/ and the misc/meerkatcal_*.done markers after '
                         'misc/meerkatcal_discover.done to start again from the prepare stage')
    if applied and not args['meerkatcal_weightgain']:
        terminal_print('WARNING: the calibrator scans were prepared with --meerkatcal-weightgain, it is also '
                       'applied to the targets')
    return applied or bool(args['meerkatcal_weightgain'])


def stage_prepare(info, args):
    """Split, swap and flag every calibrator scan."""
    summary = {}
    wgain = use_weightgain(args)
    if wgain:
        terminal_print('WARNING: ' + weightgain.WARNING)
    for role in ('bpcal', 'phasecal', 'polcal'):
        for name in info['roles'][role]:
            for scan in info['fields'][name]['scans']:
                out = _scan_ms(name, scan)
                if os.path.isfile(out + '.prepared'):
                    frac = open(out + '.prepared').read().strip()
                    if frac:
                        summary[out] = float(frac)
                    continue
                split_scan(*scan_location(info, name, scan), scan, out)
                gain_h5 = weightgain.scan_h5(out, input_mslist(info), _weightgain_h5(name, scan)) if wgain else None
                prepare_calibrator_scan(out, info['band'], edgefraction=args['meerkatcal_edgefraction'],
                                        maxnchan=args['meerkatcal_calnchan'], rfimask=args['meerkatcal_rfimask'],
                                        gain_h5=gain_h5)
                summary[out] = flag_fraction(out)
                terminal_print(f'{out}: {100 * summary[out]:.1f}% flagged after preparation')
                with open(out + '.prepared', 'w') as f:
                    f.write(f'{summary[out]}\n')
    if info.get('angle') == 'diode':
        # early check for diode firings, so that a run without them says so at the start (the polangle
        # stage detects them again, its result is the one used)
        _, _, source, nant = find_diode(args, ensure_autocorr_scans(info, args))
        if nant < diode.MIN_ANTENNAS:
            _warn('no noise diode firing found in the autocorrelations of the calibrator scans: the polarisation '
                  'angle will not be calibrated (the targets get level leakage), unless the diode-on times are given '
                  'with --meerkatcal-diode-times')
        else:
            terminal_print(f'Noise diode firings found ({source}) for {nant} antennas')
    return summary


def setup_meerkatcal_refant(info, args):
    """Fix the reference antenna for all solves (--refant, an earlier choice, or the least flagged one)."""
    refantfile = 'misc/refant.txt'
    if args['refant'] is not None:
        refant = args['refant']
    elif os.path.isfile(refantfile):
        refant = open(refantfile).read().strip()
    else:
        refant = choose_refant_from_ms(calibrator_scans(info, 'bpcal'), 'MeerKAT')
    with open(refantfile, 'w') as f:
        f.write(refant + '\n')
    set_refant(refant)
    terminal_print('Reference antenna for all solves:', refant)
    return refant


def _noqa_copy(h5):
    """Copy of an h5parm before its QA (<name>_noqa.h5), for the applies that write FLAG."""
    out = h5.replace('.h5', '_noqa.h5')
    shutil.copy(h5, out)
    return out


def stage_bandpass(info, args):
    """
    delay (treeMF), per-dump phaseonly (one per polarisation over the band), complexgain (one solution per
    channel for the whole scan, the bandpass) and per-dump amplitudeonly on the bandpass calibrator scans, with solution QA and flag cycles
    (the delay is solved in cycle 0 only, later cycles re-solve the gains and the bandpass). The per-dump
    amplitudeonly solutions absorb the broadband gain jumps between dumps, so that the residuals
    (data - model) can be flagged with AOFlagger; they are not transferred to the targets. The
    transferred bandpass (BANDPASS_H5) has the delays and X-Y phase offsets folded in
    (fold_transfer_terms); BANDPASS_SHAPE_H5 is the bandpass without them.
    """
    bpms = calibrator_scans(info, 'bpcal')
    for ms in bpms:
        snapshot_flags(ms, 'bandpass')
        predictsky(ms, set_MeerKAT_bandpass_skymodel(ms), modeldata='MODEL_DATA')
    bmerged_prev = None
    ncycles = max(1, args['meerkatcal_flagcycles'])
    qa = {}
    delay0 = {}  # per scan: the cycle-0 delay h5parm and its copy before QA, used in every cycle
    for cycle in range(ncycles):
        blist, pregains, scanbp, noqa = [], {}, {}, {}
        for ms in bpms:
            tag = os.path.basename(ms).replace('.ms', '')
            incol = 'DATA'
            if bmerged_prev is not None:
                apply(ms, [bmerged_prev], 'DATA', 'DATA_BANDPASS')
                incol = 'DATA_BANDPASS'
            # the applies in place (FLAG is written) use the solutions before their QA (_noqa copies):
            # applying the QA flags would flag the data for good, and a false positive would become a
            # gap the next solve or cycle cannot fill; the QA'd solutions are the ones transferred
            if cycle == 0:
                delayh5 = f'h5_solutions/bpcal_delay_c{cycle}_{tag}.h5'
                solve_delay(ms, delayh5, incol, solint=4)
                reference(delayh5)
                delay_noqa = _noqa_copy(delayh5)
                qa[delayh5] = qa_real(delayh5, 'delay000', minscale=DELAY_QA_MINSCALE)
                delay0[ms] = (delayh5, delay_noqa)
            else:
                # the cycle-0 delays are kept: the bandpass (one solution per channel) already absorbs any
                # constant delay, and a re-solve on the bandpass-corrected data aliases (by ~1.6-3.2 ns per
                # solution interval at L-band) across the large flagged gaps of the static RFI mask, which
                # decorrelates the per-dump gains and flags all data
                delayh5, delay_noqa = delay0[ms]
            apply(ms, [delay_noqa], incol, 'DATA_DELAY', interpolation='nearest')
            phaseh5 = f'h5_solutions/bpcal_phaseonly_c{cycle}_{tag}.h5'
            solve_ddecal(ms, phaseh5, 'phaseonly', 'DATA_DELAY', solint=1, nchan=0, maxchan=128)
            reference(phaseh5)
            phase_noqa = _noqa_copy(phaseh5)
            qa[phaseh5] = qa_gains(phaseh5, along='time', window=9, minscale=TIME_QA_MINSCALE)
            apply(ms, [phase_noqa, delay_noqa], 'DATA', 'DATA_DELAY', interpolation='nearest')
            bph5 = f'h5_solutions/bpcal_complexgain_c{cycle}_{tag}.h5'
            solve_ddecal(ms, bph5, 'complexgain', 'DATA_DELAY', solint=0, nchan=1, timeavg=True)
            reference(bph5)
            bp_noqa = _noqa_copy(bph5)
            qa[bph5] = qa_gains(bph5, along='freq')
            apply(ms, [bp_noqa], 'DATA_DELAY', 'DATA_DELAY_BANDPASS', interpolation='nearest')
            amph5 = f'h5_solutions/bpcal_amplitudeonly_c{cycle}_{tag}.h5'
            solve_ddecal(ms, amph5, 'amplitudeonly', 'DATA_DELAY_BANDPASS', solint=1, nchan=0, maxchan=128)
            amp_noqa = _noqa_copy(amph5)
            qa[amph5] = qa_gains(amph5, along='time', window=9, minscale=TIME_QA_MINSCALE)
            blist.append(bph5)
            pregains[ms] = [amph5, phaseh5, delayh5]
            scanbp[ms] = bph5
            noqa[ms] = [amp_noqa, phase_noqa, delay_noqa, bp_noqa]
        bmerged = f'h5_solutions/bpcal_complexgain_c{cycle}_merged.h5'
        ratios = merge_bandpasses(blist, bmerged)
        qa[bmerged] = qa_gains(bmerged, along='freq')
        qa[bmerged]['amplitude_ratios'] = dict(zip(blist, [float(r) for r in ratios]))
        if cycle < ncycles - 1:
            # flag on the residual of each scan's own calibration: the models of the bandpass calibrators
            # disagree in spectral shape (J0408-6545 and J1939-6342 by up to ~10% across the UHF band), so
            # with the merged bandpass the residual of a bright calibrator is model mismatch, not RFI
            for ms in bpms:
                apply(ms, noqa[ms], 'DATA', 'CORRECTED_DATA', interpolation='nearest')
                flag_residual(ms)
                terminal_print(f'{ms}: {100 * flag_fraction(ms):.1f}% flagged after cycle {cycle}')
        bmerged_prev = bmerged
    shutil.copy(bmerged_prev, BANDPASS_SHAPE_H5)
    transfer = fold_transfer_terms(bmerged_prev, [pregains[ms][2] for ms in bpms], [pregains[ms][1] for ms in bpms],
                                   BANDPASS_H5)
    os.makedirs('plots', exist_ok=True)
    plots.plot_spectrum(BANDPASS_SHAPE_H5, 'plots/' + os.path.basename(BANDPASS_SHAPE_H5).replace('.h5', '.png'),
                        'Bandpass (complexgain)')
    for ms in bpms:
        amph5, phaseh5, delayh5 = pregains[ms]
        plots.plot_timeseries(delayh5, 'delay000', 'plots/' + os.path.basename(delayh5).replace('.h5', '_delay.png'),
                              os.path.basename(delayh5), scale=1e9, unit='ns')
        plots.plot_timeseries(phaseh5, 'phase000', 'plots/' + os.path.basename(phaseh5).replace('.h5', '.png'),
                              os.path.basename(phaseh5))
        plots.plot_timeseries(amph5, 'amplitude000', 'plots/' + os.path.basename(amph5).replace('.h5', '.png'),
                              os.path.basename(amph5))
    return dict(qa=qa, flagged={ms: flag_fraction(ms) for ms in bpms}, pregains=pregains, scanbandpass=scanbp,
                transfer=transfer)


def _nchan_for_bandwidth(ms, bandwidth):
    """Channels per solution for a solution bandwidth in MHz (0: one solution per channel)."""
    if bandwidth is None or bandwidth <= 0:
        return 1
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        chanwidth = abs(float(t.getcol('CHAN_WIDTH')[0][0]))
    return max(1, int(round(bandwidth * 1e6 / chanwidth)))


def _python_step_parset(prefix, module, cls, **keys):
    """DP3 arguments of a PythonDPPP step."""
    return (f'{prefix}.type=PythonDPPP {prefix}.python.module={module} {prefix}.python.class={cls} '
            + ''.join(f'{prefix}.{k}={v} ' for k, v in keys.items()))


def stage_leakage(info, args, pregains, scanbp):
    """
    DDECal leakage on the unpolarised bandpass calibrator: one solution per observation (per channel or
    --meerkatcal-leakage-bandwidth), gauge fix (the common term is not constrained by an unpolarised
    source), QA and flag cycles. Each scan is calibrated with its own bandpass and per-dump gains
    (pregains, scanbp from stage_bandpass), so that the residual flagging sees no model mismatch; the
    leakage differs from one solved on the merged bandpass only in second order.
    """
    names = [n for n in info['roles']['bpcal'] if info['fields'][n]['bandpass_calibrator'] in UNPOLARISED_CALIBRATORS]
    if not names:
        names = info['roles']['bpcal']
        terminal_print('WARNING: no unpolarised calibrator, the leakage is solved on ' + ', '.join(names)
                       + ' (assumed unpolarised, J0408-6545 is polarised at the percent level)')
    scans = [_scan_ms(n, s) for n in names for s in info['fields'][n]['scans']]
    nchan = _nchan_for_bandwidth(scans[0], args['meerkatcal_leakage_bandwidth'])
    ncycles = max(1, args['meerkatcal_flagcycles'])
    qa = {}
    for ms in scans:
        snapshot_flags(ms, 'leakage')
        apply(ms, pregains[ms] + [scanbp[ms]], 'DATA', 'DATA_DELAY_BANDPASS', interpolation='nearest')
    for cycle in range(ncycles):
        dlist = []
        for ms in scans:
            tag = os.path.basename(ms).replace('.ms', '')
            leakh5 = f'h5_solutions/bpcal_leakage_c{cycle}_{tag}.h5'
            solve_ddecal(ms, leakh5, 'leakage', 'DATA_DELAY_BANDPASS', solint=0, nchan=nchan, timeavg=True)
            fix_leakage_gauge(leakh5)
            dlist.append(leakh5)
        dmerged = f'h5_solutions/bpcal_leakage_c{cycle}_merged.h5'
        if len(dlist) > 1:
            merge_bandpasses(dlist, dmerged, normalise=False)
        else:
            shutil.copy(dlist[0], dmerged)
        qa['gauge_c'] = fix_leakage_gauge(dmerged)
        if cycle < ncycles - 1:
            # flag on the residual with the solution before its QA: applying the QA flags here would flag
            # the data for good, and a false positive (e.g. the smooth but large leakage ripple of some
            # antennas) would become a gap the next cycle cannot solve, with more QA flags at its edges
            dflag = dmerged.replace('.h5', '_noqa.h5')
            shutil.copy(dmerged, dflag)
        qa[dmerged] = qa_leakage(dmerged)
        if cycle < ncycles - 1:
            for ms in scans:
                apply(ms, [dflag], 'DATA_DELAY_BANDPASS', 'CORRECTED_DATA')
                flag_residual(ms)
                terminal_print(f'{ms}: {100 * flag_fraction(ms):.1f}% flagged after leakage cycle {cycle}')
    shutil.copy(dmerged, LEAKAGE_H5)
    unwrap_phase(LEAKAGE_H5)  # for the linear interpolation in frequency
    plots.plot_spectrum(LEAKAGE_H5, 'plots/' + os.path.basename(LEAKAGE_H5).replace('.h5', '.png'),
                        'Leakage (' + ', '.join(names) + ')')
    return dict(qa=qa, calibrator=names, nchan=nchan, flagged={ms: flag_fraction(ms) for ms in scans})


def _phasecal_field(ms):
    """The field name of a phase calibrator scan MS (<field>.scanNNN[.B|.BG].ms)."""
    return os.path.basename(ms).rsplit('.scan', 1)[0]


def _first_time(ms):
    with table(ms, ack=False) as t:
        return float(np.min(t.getcol('TIME')))


def _average_copy(ms, out, band, bandnchan):
    """
    Copy of ms (DATA, flags, weights) averaged in frequency to about the resolution of bandnchan channels
    over the full nominal band (BANDWIDTH; e.g. 1024 gives 4x averaging of 4k data, 2x of the 2k archive
    product, also when the archive trimmed the band edges). bandnchan 0: no averaging.
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
        chanwidth = float(np.median(np.abs(t.getcol('CHAN_WIDTH')[0])))
    target = BANDWIDTH.get(band, BANDWIDTH['S']) / bandnchan if bandnchan > 0 else chanwidth
    factor = max(1, min(nchan, int(round(target / chanwidth))))
    nread = nchan_divisible(nchan, factor)
    cmd = (f'DP3 numthreads={nthreads()} msin={ms} msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM '
           f'msout={out} ' + (f'msin.nchan={nread} ' if nread != nchan else '')
           + f'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False {DYSCO_ARGS}'
           + (f'steps=[avg] avg.type=averager avg.freqstep={factor}' if factor > 1 else 'steps=[]'))
    terminal_print('Average for the phase calibrator model:', cmd)
    run(cmd, log=True)
    fix_uvw([out])  # time slots inserted by DP3
    return nread // factor


def _phasecal_model_selfcal(info, msin, workdir='phasecal_selfcal', bandnchan=0):
    """
    Build the model of each phase calibrator with a short facetselfcal DI run on its scans, in its own
    directory <workdir>/<field> (an observation can have several phase calibrators, e.g. alternating
    between target scans). The scans of a field are averaged to about the resolution of bandnchan
    channels over the band (--meerkatcal-phasecal-nchan; a Stokes I model) and concatenated in time order (casacore table
    concatenation, no time slots inserted) into one MS for facetselfcal: a single phase calibrator scan
    is usually shorter than the 21 time slots facetselfcal needs per MS, and facetselfcal splits an MS at
    its time gaps itself (split_irregular_timeaxis, which needs the rows in time order).
    A finished field (<workdir>/<field>/meerkatcal_field_model.json) is kept on a restart.
    Returns {field: image name of the last cycle}.
    """
    byfield = {}
    for ms in msin:
        byfield.setdefault(_phasecal_field(ms), []).append(ms)
    pixelscale = PIXELSCALE.get(info['band'], 0.5)
    imsize = 4800
    refant = open('misc/refant.txt').read().strip()
    models = {}
    for field, mslist in byfield.items():
        fielddir = os.path.join(workdir, field)
        done = os.path.join(fielddir, 'meerkatcal_field_model.json')
        if os.path.isfile(done):
            models[field] = json.load(open(done))['model']
            terminal_print(f'Reusing the model of the phase calibrator {field} ({models[field]})')
            continue
        if os.path.isdir(fielddir):  # an interrupted earlier run
            shutil.rmtree(fielddir)
        os.makedirs(fielddir)
        avg = []
        for ms in sorted(mslist, key=_first_time):
            out = os.path.join(fielddir, os.path.basename(ms).replace('.BG.ms', '.BGavg.ms'))
            nchan = _average_copy(ms, out, 'S' if info['band'].startswith('S') else info['band'], bandnchan)
            avg.append(out)
        concat = os.path.join(fielddir, 'phasecal_concat.ms')
        with table(avg, ack=False) as t:
            t.copy(concat, deep=True)
        for ms in avg:
            shutil.rmtree(ms)
        terminal_print(f'Phase calibrator {field}: {len(avg)} scans with {nchan} channels in {concat}')
        cmd = (f'cd {fielddir} && python {REPO}/facetselfcal.py -i phasecal --imsize={imsize} '
               f'--pixelscale={pixelscale} --channelsout=6 --fitspectralpol=3 --niter=20000 --stop=3 '
               "--soltype-list=\"['scalarphase','scalarcomplexgain']\" --solint-list=\"['1min','10min']\" "
               "--soltypecycles-list=\"[0,1]\" --smoothnessconstraint-list=\"[20.,50.]\" "
               f'--refant={refant} --noarchive --skip-Ateam-plotting ' + os.path.abspath(concat))
        terminal_print(f'facetselfcal run on the phase calibrator {field}:', cmd)
        run(cmd, log=True)
        # facetselfcal moves its images, gzipped, to fits_images/
        images = sorted(glob.glob(f'{fielddir}/fits_images/phasecal_[0-9][0-9][0-9]-0000-model.fits*'))
        if not images:
            raise RuntimeError('No model images from the phase calibrator facetselfcal run in ' + fielddir)
        models[field] = os.path.basename(images[-1]).split('-')[0]
        with open(done, 'w') as fp:
            json.dump(dict(model=models[field], bandnchan=bandnchan), fp)
    return models


def _predict_phasecal_model(name, msmodel, workdir='phasecal_selfcal'):
    """Predict the facetselfcal channel model images <workdir>/fits_images/<name>-NNNN-model.fits[.gz]
    into the MODEL_DATA of msmodel (unzipped first into workdir, for WSClean)."""
    images = sorted(glob.glob(f'{workdir}/fits_images/{name}-[0-9][0-9][0-9][0-9]-model.fits*'))
    for image in images:
        out = os.path.join(workdir, os.path.basename(image).removesuffix('.gz'))
        if image.endswith('.gz'):
            with gzip.open(image, 'rb') as fin, open(out, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copy(image, out)
    prefix = os.path.join(workdir, name)
    for ms in msmodel:
        run(f'wsclean -predict -name {prefix} -channels-out {len(images)} -j {nthreads()} {ms}', log=True)


def _file_digest(files):
    """MD5 digest of the contents of files, in order."""
    md5 = hashlib.md5()
    for f in files:
        with open(f, 'rb') as fp:
            for block in iter(lambda: fp.read(1 << 20), b''):
                md5.update(block)
    return md5.hexdigest()


def _phasecal_solves(ms, label):
    """
    delay (treeMF), phaseonly and scalaramplitude (one solution per scan) of one phase calibrator scan
    against MODEL_DATA. Returns the three h5parms (named phasecal_<soltype>_<label>).

    phaseonly and scalaramplitude have one solution over the band: the frequency structure is in the
    bandpass, and per-channel gains would carry errors of the phase calibrator model to the targets
    (e.g. the steps between the WSClean output channels of the facetselfcal model).
    """
    delayh5, phaseh5, amph5 = (f'h5_solutions/phasecal_{soltype}_{label}.h5' for soltype in PHASECAL_H5)
    solve_delay(ms, delayh5, 'DATA', solint=8)
    reference(delayh5)
    apply(ms, [delayh5], 'DATA', 'DATA_DELAY', interpolation='nearest')
    solve_ddecal(ms, phaseh5, 'phaseonly', 'DATA_DELAY', solint=0, nchan=0, timeavg=True, maxchan=128)
    reference(phaseh5)
    apply(ms, [phaseh5], 'DATA_DELAY', 'DATA_DELAY_PHASEONLY', interpolation='nearest')
    solve_ddecal(ms, amph5, 'scalaramplitude', 'DATA_DELAY_PHASEONLY', solint=0, nchan=0, timeavg=True,
                 maxchan=128)
    return delayh5, phaseh5, amph5


def stage_phasecal(info, args, extra_h5=()):
    """
    Phase calibrator: apply the bandpass (and leakage), then, solve delay and phaseonly
    against a point source and apply them, so that the data imaged by the short facetselfcal run that
    builds the model are coherent; then solve delay, phaseonly and scalaramplitude per scan against
    that model (the time-variable gains transferred to the targets).
    """
    from submods.model_columns import set_constant_model_column
    # the MSs of the work directory carry the bandpass (and leakage) of the run that made them, so the
    # directory, and with it the facetselfcal run, is kept on a restart only when it was completed with
    # the same tables; otherwise it is rebuilt, since an interrupted stage can leave an incomplete MS or
    # facetselfcal state behind
    workdir = 'phasecal_selfcal'
    checkpoint = os.path.join(workdir, 'meerkatcal_model.json')
    inputs = _file_digest([BANDPASS_H5] + list(extra_h5))
    state = json.load(open(checkpoint)) if os.path.isfile(checkpoint) else {}
    selfcal = args['meerkatcal_phasecal_selfcal']
    same = state.get('inputs') == inputs and state.get('selfcal') == selfcal
    if same and state.get('model'):
        terminal_print(f'Reusing the phase calibrator data and model in {workdir} ({state["model"]})')
        msB = state['msB']
    else:
        if same and state.get('msG'):
            # the data were completed (checkpoint written after them), the model run was not
            terminal_print(f'Reusing the phase calibrator data in {workdir}, making the model')
            msB = state['msB']
        else:
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
            os.makedirs(workdir)
            msB = []
            for ms in calibrator_scans(info, 'phasecal'):
                out = os.path.join(workdir, os.path.basename(ms).replace('.ms', '.B.ms'))
                apply(ms, [BANDPASS_H5] + list(extra_h5), 'DATA', 'DATA', msout=out, updateweights=True)
                fix_uvw([out])  # time slots inserted by DP3
                aoflagger_column([out], aoflagger_strategy='defaultMeerKAT_StokesQUV.lua', column='DATA')
                set_constant_model_column(out, 'MODEL_DATA', 1.0)
                msB.append(out)
            state = dict(inputs=inputs, selfcal=selfcal, msB=msB, model=None if selfcal else 'point source')
            if selfcal:
                msG = []
                for ms in msB:
                    tag = os.path.basename(ms).replace('.B.ms', '')
                    delayh5, phaseh5, _ = _phasecal_solves(ms, f'pointsource_{tag}')
                    out = ms.replace('.B.ms', '.BG.ms')
                    apply(ms, [phaseh5, delayh5], 'DATA', 'DATA', msout=out, interpolation='nearest')
                    fix_uvw([out])
                    msG.append(out)
                state['msG'] = msG
                with open(checkpoint, 'w') as fp:
                    json.dump(state, fp)
        if selfcal:
            state['model'] = _phasecal_model_selfcal(info, state['msG'], workdir,
                                                     bandnchan=args['meerkatcal_phasecal_nchan'])
        with open(checkpoint, 'w') as fp:
            json.dump(state, fp)
    for ms in msB:
        snapshot_flags(ms, 'phasecal')  # the flags before the transfer solves, restored on a restart
    model = state['model']
    if isinstance(model, dict):
        for field, name in model.items():
            _predict_phasecal_model(name, [ms for ms in msB if _phasecal_field(ms) == field],
                                    os.path.join(workdir, field))
    elif model != 'point source':  # checkpoint of a run before the per-field model directories
        _predict_phasecal_model(model, msB, workdir)
    qa = {}
    ncycles = max(1, args['meerkatcal_flagcycles'])
    for cycle in range(ncycles):
        h5lists = {soltype: [] for soltype in PHASECAL_H5}
        for ms in msB:
            tag = os.path.basename(ms).replace('.B.ms', '')
            delayh5, phaseh5, amph5 = _phasecal_solves(ms, tag)
            for soltype, h5 in zip(PHASECAL_H5, (delayh5, phaseh5, amph5)):
                h5lists[soltype].append(h5)
            if cycle < ncycles - 1:
                apply(ms, [amph5], 'DATA_DELAY_PHASEONLY', 'CORRECTED_DATA', interpolation='nearest')
                flag_residual(ms)
                terminal_print(f'{ms}: {100 * flag_fraction(ms):.1f}% flagged after cycle {cycle}')
    for soltype, h5 in PHASECAL_H5.items():
        concat_h5_time(h5lists[soltype], h5)
    delayh5, phaseh5, amph5 = PHASECAL_H5.values()
    qa['delay'] = qa_real(delayh5, 'delay000', window=5, minscale=DELAY_QA_MINSCALE)
    # one solution per scan over the band: the phases change between scans with the ionosphere, so they
    # get only the NaN and antenna checks (no outliers along the single channel); the amplitudes are
    # checked along time over the scans (needs at least 3 scans)
    qa['phaseonly'] = qa_gains(phaseh5, along='freq')
    qa['scalaramplitude'] = qa_gains(amph5, along='time', window=5, minscale=PHASECAL_AMP_QA_MINSCALE)
    qa['scalaramplitude_norm'] = normalise_amplitudes(amph5)
    plots.plot_timeseries(delayh5, 'delay000', 'plots/meerkatcal_phasecal_delay.png',
                          'Phase calibrator delays', scale=1e9, unit='ns')
    plots.plot_timeseries(phaseh5, 'phase000', 'plots/meerkatcal_phasecal_phaseonly.png',
                          'Phase calibrator phaseonly')
    plots.plot_timeseries(amph5, 'amplitude000', 'plots/meerkatcal_phasecal_scalaramplitude.png',
                          'Phase calibrator scalaramplitude')
    return dict(model=model, qa=qa)


def _ionorm(args):
    """Whether the ionospheric Faraday rotation is corrected: --meerkatcal-ionorm or an --meerkatcal-rmfile."""
    return bool(args['meerkatcal_ionorm'] or args['meerkatcal_rmfile'])


def _rm_h5(info, args, name):
    """The ionospheric RM h5parm towards a field (made once), or None when the correction is off (_ionorm)."""
    if not _ionorm(args):
        return None
    h5 = f'h5_solutions/RM_{name}.h5'
    if not os.path.isfile(h5):
        f = info['fields'][name]
        locs = field_locations(f, info.get('ms'))
        res = make_rm_h5(h5, [m for m, _, _ in locs], [fid for _, fid, _ in locs], f['direction'],
                         rmfile=args['meerkatcal_rmfile'])
        with open(h5.replace('.h5', '.json'), 'w') as fp:
            json.dump(res, fp)
    return h5


def feed_frame_model(ms, rmh5, incol='MODEL_DATA', outcol='MODEL_FEED'):
    """Corrupt a sky-frame model with the ionospheric Faraday rotation and the parallactic angle."""
    steps, cmd = [], ''
    if rmh5 is not None:
        cmd, nac = applycal_dp3_steps(ms, [rmh5], invert=False, interpolation='linear')
        steps += ['ac%d' % i for i in range(nac)]
    cmd += _python_step_parset('pa', 'parang', 'ParAng', mode='corrupt', receptorangle=0)
    steps.append('pa')
    cmd = (f'DP3 numthreads={nthreads()} msin={ms} msin.datacolumn={incol} msout=. msout.datacolumn={outcol} '
           + cmd + 'steps=[' + ','.join(steps) + ']')
    terminal_print('Feed-frame model:', cmd)
    run(cmd, log=True)


def _diode_scans(info):
    """The calibrator scans searched for noise diode firings: {scan MS: (role, field, scan)}."""
    out = {}
    for role in ('bpcal', 'phasecal', 'polcal'):
        for name in info['roles'][role]:
            for scan in info['fields'][name]['scans']:
                out.setdefault(_scan_ms(name, scan), (role, name, scan))
    return out


def ensure_autocorr_scans(info, args):
    """
    The autocorrelation copies of the calibrator scans for the noise diode (diode.make_autocorr_scan),
    made once (marker <ms>.prepared). Returns {auto MS: (role, field, scan)}.
    """
    out = {}
    for scanms, (role, name, scan) in _diode_scans(info).items():
        ms = diode.auto_ms_name(scanms)
        if not os.path.isfile(ms + '.prepared'):
            diode.make_autocorr_scan(*scan_location(info, name, scan), scan, ms, info['band'],
                                     edgefraction=args['meerkatcal_edgefraction'], maxnchan=args['meerkatcal_calnchan'],
                                     rfimask=args['meerkatcal_rfimask'])
            open(ms + '.prepared', 'w').close()
        out[ms] = (role, name, scan)
    return out


def find_diode(args, automs):
    """
    The noise diode-on time ranges per autocorrelation MS and antenna: from --meerkatcal-diode-times, or
    found in the autocorrelations (diode.detect_diode). Written to misc/meerkatcal_diode.json.
    Returns (ranges, statistics, source, number of antennas with diode-on data).
    """
    if args['meerkatcal_diode_times']:
        ranges = diode.ranges_from_times(list(automs), diode.parse_diode_times(args['meerkatcal_diode_times']))
        stats, source = {}, 'user'
    else:
        ranges, stats = diode.detect_diode(list(automs), minjump=args['meerkatcal_diode_minjump'])
        source = 'detected'
    nant = len({a for r in ranges.values() for a in r})
    for ms in automs:
        r = ranges.get(ms, {})
        terminal_print(f'Noise diode ({source}) in {ms}: ' +
                       (f'on for {len(r)} antennas ({sum(len(v) for v in r.values())} time ranges)' if r else 'not on'))
    with open('misc/meerkatcal_diode.json', 'w') as f:
        json.dump(dict(source=source, n_antennas=nant, ranges=ranges, stats=stats), f, indent=1)
    return ranges, stats, source, nant


def stage_polangle_diode(info, args, pre_h5):
    """
    Cross-hand phase (polarisation angle) from the noise diode (see submods/meerkat_cal/diode.py): on
    every calibrator scan with diode-on dumps, the per-antenna diode X-Y phase of the autocorrelations
    with the calibration applied (bandpass, leakage, and for the phase calibrator scans its time-variable
    gains, as for the targets), then the median over the antennas and scans minus the diode-to-sky
    spline. Without diode firings the angle is not calibrated (the targets get level leakage).
    """
    warnings = []
    if os.path.isfile(XFSOLVE_H5):  # from an earlier attempt: it must not be applied if this one finds no diode
        os.remove(XFSOLVE_H5)
    automs = ensure_autocorr_scans(info, args)
    ranges, stats, source, nant = find_diode(args, automs)
    if nant < diode.MIN_ANTENNAS:
        _warn(f'no noise diode firing found in the autocorrelations of the calibrator scans ({nant} antennas with '
              'diode-on dumps): the polarisation angle (cross-hand phase) is not calibrated, the targets are '
              'calibrated at level leakage. If the diode fired, give its on times with --meerkatcal-diode-times '
              '(or lower --meerkatcal-diode-minjump)', warnings)
        return dict(angle='diode', skipped='no noise diode firing found', warnings=warnings, detection=stats)
    nchan = _nchan_for_bandwidth(next(iter(automs)), args['meerkatcal_xf_bandwidth'])
    have_phasecal = _done('phasecal')
    xlist, scaninfo = [], {}
    for ms, (role, name, scan) in automs.items():
        if ms not in ranges:
            continue
        tag = os.path.basename(ms).replace('.auto.ms', '')
        chain = [BANDPASS_H5] + list(pre_h5)
        if role == 'phasecal' and have_phasecal:
            chain = [PHASECAL_H5[s] for s in ('scalaramplitude', 'phaseonly', 'delay')] + chain
        diodefile = f'misc/diode_on_{tag}.json'
        with open(diodefile, 'w') as f:
            json.dump(ranges[ms], f)
        xfh5 = f'h5_solutions/diode_xfsolve_{tag}.h5'
        solve_xf_diode(ms, xfh5, chain, diodefile, nchan=nchan)
        xlist.append(xfh5)
        scaninfo[ms] = {'field': name, 'role': role, 'xfsolve': xfh5, 'diodefile': diodefile,
                        'n_antennas': len(ranges[ms]), 'applied': chain}
    xall = XFSOLVE_H5
    res = diode.combine_diode(xlist, xall, info['band'], skyspline=args['meerkatcal_diode_skyspline'])
    qa = qa_crosshand(xall, constraint=None, interpolate=args['meerkatcal_xf_interpolate'], fill_missing=True)
    unwrap_phase(xall)
    with tables.open_file(xall) as H:
        theta = np.array(H.root.sol000.phase000.val[0, :, 0, 0, 0])  # axes time, freq, ant, dir, pol
    what = 'interpolated from the neighbouring solutions' if args['meerkatcal_xf_interpolate'] else 'flagged'
    _warn('the noise diode polarisation angle follows the katsdpcal diode-to-sky convention, which is not yet '
          'verified against a celestial polarisation calibrator in this pipeline', warnings)
    if qa['outlier_mhz']:
        _warn(f"diode cross-hand phase solutions at {_mhz_text(qa['outlier_mhz'])} deviate from the smooth "
              f'frequency dependence: {what}', warnings)
    if qa['flagged_mhz']:
        _warn(f"no diode cross-hand phase at {_mhz_text(qa['flagged_mhz'])} (no good solutions on both sides): "
              'these channels are flagged in the targets', warnings)
    if res['outliers']:
        _warn(f"noise diode X-Y phase of {', '.join(res['outliers'])} deviates by more than 30 deg from the "
              'median of the antennas: left out', warnings)
    spread = float(np.nanmedian(res['spread_deg']))
    if spread > 10.:
        _warn(f'the noise diode X-Y phases of the antennas scatter by {spread:.1f} deg (median over the channels '
              'of 1.4826 MAD): check the diode detection and the calibration', warnings)
    # per-scan cross-hand phase solutions (<tag>.theta.h5; the per-scan h5parms of xfsolve hold the
    # per-antenna diode phases), for the consistency check and the report
    perscan = {}
    for ms, sc in scaninfo.items():
        single = sc['xfsolve'].replace('.h5', '.theta.h5')
        th = diode.combine_diode([sc['xfsolve']], single, info['band'], skyspline=args['meerkatcal_diode_skyspline'])['theta']
        sc['diode_phases'], sc['xfsolve'] = sc['xfsolve'], single
        if len(scaninfo) > 1:
            with np.errstate(invalid='ignore'):
                diff = np.degrees(np.abs(np.angle(np.exp(1j * (th - theta)))))
            perscan[single] = float(np.nanmedian(diff))
            if perscan[single] > 5.:
                _warn(f'diode cross-hand phase of {single} differs by {perscan[single]:.1f} deg (median) from the '
                      'combined solution', warnings)
    plots.plot_spectrum(xall, 'plots/' + os.path.basename(xall).replace('.h5', '.png'),
                        'Cross-hand phase, noise diode (' + ', '.join(sorted({s["field"] for s in scaninfo.values()})) + ')')
    diode.plot_diode(xlist, res, 'plots/meerkatcal_diode_xyphase.png', 'Noise diode X-Y phases')
    return dict(angle='diode', calibrators=sorted({s['field'] for s in scaninfo.values()}), nchan=nchan, qa=qa,
                perscan_median_diff_deg=perscan, perscan_unconstrained_mhz={}, warnings=warnings, scans=scaninfo,
                theta_deg=np.degrees(theta).tolist(),
                diode=dict(source=source, n_antennas=len(res['antennas']), outliers=res['outliers'],
                           spread_median_deg=spread, skyspline=args['meerkatcal_diode_skyspline'],
                           convention_verified=False, detection=stats))


def stage_polangle(info, args, pre_h5):
    """
    Cross-hand phase (polarisation angle). On every polarisation calibrator scan: full-polarisation model
    (Perley-Butler 2017 Stokes I, Perley 2026 / Hugo & Perley 2024 polarisation) in the feed frame
    (Faraday rotation and parallactic angle), bandpass and leakage applied, delay and phaseonly
    solved against that model, then xfsolve per scan. The applied solution combines all scans (one per
    observation); the per-scan solutions are a consistency check. With the noise diode as the angle
    source see stage_polangle_diode.
    """
    angle = info['angle']
    if angle == 'diode':
        return stage_polangle_diode(info, args, pre_h5)
    names = info['roles']['polcal']
    if not names:
        raise ValueError('The polarisation angle needs a polarisation calibrator (--meerkatcal-polcal)')
    nchan = None
    xlist, scaninfo = [], {}
    for name in names:
        rmh5 = _rm_h5(info, args, name)
        for scan in info['fields'][name]['scans']:
            ms = _scan_ms(name, scan)
            tag = os.path.basename(ms).replace('.ms', '')
            nchan = _nchan_for_bandwidth(ms, args['meerkatcal_xf_bandwidth'])
            snapshot_flags(ms, 'polangle')
            if angle == 'celestial':
                setjy_dp3(ms)
                feed_frame_model(ms, rmh5)
            apply(ms, [BANDPASS_H5] + list(pre_h5), 'DATA', 'DATA_PREAPPLY')
            delayh5, phaseh5, xfh5 = (f'h5_solutions/polcal_{soltype}_{tag}.h5'
                                      for soltype in ('delay', 'phaseonly', 'xfsolve'))
            modelcol = 'MODEL_FEED' if angle == 'celestial' else None
            if modelcol is None:  # nullv: Stokes I point source for the gains
                from submods.model_columns import set_constant_model_column
                set_constant_model_column(ms, 'MODEL_FEED', 1.0)
            solve_delay(ms, delayh5, 'DATA_PREAPPLY', modelcol='MODEL_FEED', solint=8)
            reference(delayh5)
            apply(ms, [delayh5], 'DATA_PREAPPLY', 'DATA_PREAPPLY_DELAY', interpolation='nearest')
            # 8 channels per solution: well inside the 10 MHz smoothness kernel, ~4x faster than per channel
            # (33 -> 8 min for a 75-dump, 2048-channel scan) with phases within 0.2 deg of the per-channel solve
            solve_ddecal(ms, phaseh5, 'phaseonly', 'DATA_PREAPPLY_DELAY', solint=1, nchan=8, modelcol='MODEL_FEED',
                         smoothness=10.)
            reference(phaseh5)
            apply(ms, [phaseh5], 'DATA_PREAPPLY_DELAY', 'DATA_CAL')
            solve_xf(ms, xfh5, 'DATA_CAL', modelcol='MODEL_FEED', mode='model' if angle == 'celestial' else 'nullv',
                     nchan=nchan)
            xlist.append(xfh5)
            scaninfo[ms] = {'field': name, 'delay': delayh5, 'phaseonly': phaseh5, 'xfsolve': xfh5, 'rm': rmh5}
    xall = XFSOLVE_H5
    combine_xf(xlist, xall)
    # the model fixes the sign of theta only where enough of its polarisation is in the feed-frame
    # cross-hands (Stokes U; its sign changes with frequency at some parallactic angles), elsewhere a
    # model angle error of a few degrees flips the solution by 180 deg. Those solutions, and outliers from
    # a smooth reference, are flagged and interpolated from the neighbouring good solutions
    sums_all = np.load(xall + '.sums.npz')
    minfrac = xf_min_fraction(sums_all['freqs'], args['meerkatcal_xf_minfrac'])
    qa = qa_crosshand(xall, constraint=_xf_constraint(sums_all), min_constraint=minfrac,
                      interpolate=args['meerkatcal_xf_interpolate'])
    if qa['flagged_after'] >= 1.:
        # applying a fully flagged solution would flag all target data: leave the angle uncalibrated instead
        # (level leakage; os.path.isfile(XFSOLVE_H5) is what the later stages test)
        os.rename(xall, xall.replace('.h5', '_unusable.h5'))
        reason = (f'no usable cross-hand phase solution: the model polarisation of {", ".join(names)} in the '
                  f'feed-frame cross-hands is below the minimum ({_minfrac_text(minfrac)}, '
                  f'--meerkatcal-xf-minfrac) at all frequencies (at most {np.nanmax(_xf_constraint(sums_all)):.2f}; '
                  'parallactic angle coverage too small)')
        warnings = []
        _warn(f'{reason}. The polarisation angle is not calibrated (level leakage). A smaller --meerkatcal-xf-minfrac '
              'accepts the solution (risk: a 180 deg cross-hand phase error, 90 deg in the polarisation angle, if the '
              'model angle is off by more than the feed-frame angle), or use --meerkatcal-angle diode', warnings)
        return dict(angle=angle, skipped=reason, calibrators=names, qa=qa, warnings=warnings, scans=scaninfo)
    unwrap_phase(xall)
    with tables.open_file(xall) as H:
        theta = np.array(H.root.sol000.phase000.val[0, :, 0, 0, 0])  # axes time, freq, ant, dir, pol
    warnings = []
    what = 'interpolated from the neighbouring solutions' if args['meerkatcal_xf_interpolate'] else 'flagged'
    if qa['unconstrained_mhz']:
        _warn(f'the cross-hand phase is not constrained by the polarisation calibrator model at '
              f"{_mhz_text(qa['unconstrained_mhz'])} (in all scans less than {_minfrac_text(minfrac)} of the model "
              f'polarisation in the feed-frame cross-hands, Stokes U near zero): {what}. More parallactic angle '
              'coverage of the polarisation calibrator avoids this', warnings)
    if qa['outlier_mhz']:
        _warn(f"cross-hand phase solutions at {_mhz_text(qa['outlier_mhz'])} deviate from the smooth "
              f'frequency dependence: {what}', warnings)
    if qa['flagged_mhz']:
        _warn(f"no cross-hand phase at {_mhz_text(qa['flagged_mhz'])} (no good solutions on both sides): "
              'these channels are flagged in the polarisation calibrator and targets', warnings)
    # per-scan consistency, only where the scan constrains the solution (the scan alone would flip there)
    perscan, perscan_weak = {}, {}
    for xfh5 in xlist:
        th = combine_xf([xfh5], xfh5.replace('.h5', '.single.h5'))
        os.remove(xfh5.replace('.h5', '.single.h5'))
        sums = np.load(xfh5 + '.sums.npz')
        with np.errstate(invalid='ignore', divide='ignore'):
            diff = np.degrees(np.abs(np.angle(np.exp(1j * (th - theta)))))
            constraint = _xf_constraint(sums)
            if constraint is None:  # old sums files: solution strength relative to the median
                strength = np.abs(sums['sum'][0]) / sums['weight'][0]
                weak = ~(strength >= 0.25 * np.nanmedian(strength[np.isfinite(diff)]))
            else:
                weak = ~(constraint >= xf_min_fraction(sums['freqs'], args['meerkatcal_xf_minfrac']))
        perscan[xfh5] = float(np.nanmedian(diff[~weak]))
        bad = np.isfinite(diff) & ~weak & (diff > 30.)
        if perscan[xfh5] > 5. or bad.any():
            _warn(f'cross-hand phase of {xfh5} differs by {perscan[xfh5]:.1f} deg (median) from the '
                  f'combined solution, {bad.sum()} solutions by more than 30 deg: check the ionospheric RM, '
                  'the parallactic angle coverage and the flags', warnings)
        weakfreq = np.isfinite(th) & weak
        perscan_weak[xfh5] = freq_ranges(np.load(xfh5 + '.sums.npz')['freqs'], weakfreq)
        nweak = int((weakfreq & (diff > 30.)).sum())
        if nweak:
            terminal_print(f'{xfh5}: {nweak} solutions differ by more than 30 deg from the combined solution where '
                           f'the model cross-hands of this scan are weak ({_mhz_text(perscan_weak[xfh5])}), '
                           'they have little weight in the combined solution')
    plots.plot_spectrum(xall, 'plots/' + os.path.basename(xall).replace('.h5', '.png'),
                        'Cross-hand phase, xfsolve (' + ', '.join(names) + ')')
    return dict(angle=angle, calibrators=names, nchan=nchan, qa=qa, perscan_median_diff_deg=perscan,
                perscan_unconstrained_mhz=perscan_weak, warnings=warnings, scans=scaninfo,
                theta_deg=np.degrees(theta).tolist())


def stage_polcheck(info, args, pre_h5):
    """Image the calibrated polarisation calibrator(s) per frequency chunk and compare with the model."""
    scans = json.load(open(_marker('polangle')))['scans']
    os.makedirs('polcheck', exist_ok=True)
    results = {}
    for name in info['roles']['polcal']:
        source = info['fields'][name]['polarisation_calibrator']
        calms = []
        for ms, sc in scans.items():
            if sc['field'] != name or 'phaseonly' not in sc:  # the diode solve has no per-scan gains
                continue
            out = os.path.join('polcheck', os.path.basename(ms).replace('.ms', '.cal.ms'))
            chain = [sc['phaseonly'], sc['delay'], BANDPASS_H5] + list(pre_h5)
            chain += [sc['rm']] if sc['rm'] else []
            polcheck.calibrated_scan(ms, chain, out, freqstep=4, nthreads=nthreads())
            calms.append(out)
        if not calms:
            terminal_print(f'Polarisation check {name}: skipped (no delay and phaseonly solutions of this field, '
                           'they are solved only with a celestial angle calibration)')
            continue
        prefix = f'polcheck/{name}'
        nchunks = 8
        polcheck.image_iquv(calms, prefix, nchunks=nchunks, pixelscale=PIXELSCALE.get(info['band'], 0.5),
                            nthreads=nthreads())
        rmjson = f'h5_solutions/RM_{name}.json'  # written by _rm_h5
        rm = json.load(open(rmjson)) if _ionorm(args) and os.path.isfile(rmjson) else None
        res = polcheck.measure(prefix, source, nchunks, rm=rm)
        res['warnings'] = polcheck.verdict(res, info['band'])
        res['thumbnails'] = polcheck.thumbnails(prefix, nchunks, 'plots/polcheck')
        polcheck.plot(res, f'plots/meerkatcal_polcheck_{name}.png')
        for w in res['warnings']:
            terminal_print(f'WARNING polarisation check {name}: {w}')
        if not res['warnings']:
            terminal_print(f'Polarisation check {name}: pass')
        results[name] = res
        for ms in calms:
            shutil.rmtree(ms, ignore_errors=True)
    with open('misc/meerkatcal_polcheck.json', 'w') as f:
        json.dump(results, f, indent=1)
    return results


def stage_targets(info, args, h5chain):
    """
    Apply the solutions to every target scan with ionospheric Faraday rotation (towards the target) and
    parallactic angle correction, average and flag.

    Weights: the archive MS (katdal mvftoms) has WEIGHT_SPECTRUM = N / (A_i A_j), with N the number of
    accumulated spectra (channel width x dump time) and A_i, A_j the autocorrelations (katdal
    vis_flags_weights). That is 1/sigma^2 of the complex visibility, in raw (uncalibrated) units.
    They are scaled here with the applied amplitudes (DP3 updateweights, as CASA applycal calwt=True),
    which gives weights in Jy^-2 but still in the katdal convention: CASA (statwt) uses 1/sigma^2 per
    real or imaginary part, i.e. weights 2x larger (on UHF test data sigma_re^2 x weight was 0.5-0.7). Relative
    weights, all that imaging needs, are right; for combining with data of other telescopes the
    weights should be rescaled (x2, or from the measured noise, e.g. CASA statwt). mvftoms channel and
    time averaging sums the weights, so it keeps this scale.
    """
    os.makedirs('targets', exist_ok=True)
    outputs = []
    wgain = use_weightgain(args)
    if wgain:
        terminal_print('WARNING: ' + weightgain.WARNING)
    for name in info['roles']['targets']:
        rmh5 = _rm_h5(info, args, name) if info['level'] == 'full' else None
        chain = list(h5chain) + ([rmh5] if rmh5 is not None else [])
        final = target_ms(name)
        if args['meerkatcal_target_concat'] and os.path.isfile(final + '.done') and \
                os.path.getmtime(final + '.done') > max(os.path.getmtime(h) for h in chain):
            outputs.append(final)
            continue
        for scan in info['fields'][name]['scans']:
            view = f'targets/{name}.scan{scan:03d}.split.ms'  # temporary copy of the scan, removed below
            out = f'targets/{name}.scan{scan:03d}.cal.ms'
            # resume only when no applied table changed since the scan was done (a rerun of an earlier stage)
            if os.path.isfile(out + '.done') and \
                    os.path.getmtime(out + '.done') > max(os.path.getmtime(h) for h in chain):
                outputs.append(out)
                continue
            split_scan(*scan_location(info, name, scan), scan, view)
            # the array-wide weight gain is measured on the split scan (archive weights) and divided out first
            scanchain = ([weightgain.scan_h5(view, input_mslist(info), _weightgain_h5(name, scan))] if wgain else []) + chain
            swapargs, swapsteps = swap_parset()
            pfargs, pfsteps = preflag_parset(view, info['band'], args['meerkatcal_edgefraction'], include_hi=False,
                                             rfimask=args['meerkatcal_rfimask'])
            # the archive (katdal) weights follow the raw gains (~1/|g_i|^2|g_j|^2): scale them with the
            # applied amplitudes (bandpass, scalaramplitude), as CASA applycal calwt=True
            acargs, nac = applycal_dp3_steps(view, scanchain, missingantennabehavior='flag', interpolation='linear',
                                             updateweights=True)
            with table(view + '::SPECTRAL_WINDOW', ack=False) as t:
                nchan = len(t.getcol('CHAN_FREQ')[0])
            # the top band-edge channels that do not fill a whole averaging group are left out
            freqstep = freq_factor(nchan, args['meerkatcal_nchan'])
            nread = nchan_divisible(nchan, freqstep)
            timestep = time_factor(view, args['meerkatcal_timeres'])
            steps = swapsteps + pfsteps + ['ac%d' % i for i in range(nac)] + ['pa', 'avg']
            if os.path.isdir(out):
                shutil.rmtree(out)
            dysco = DYSCO_ARGS if args['meerkatcal_target_exactweights'] else DYSCO_ARGS_TARGET
            cmd = (f'DP3 numthreads={nthreads()} msin={view} msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM '
                   + (f'msin.nchan={nread} ' if nread != nchan else '')
                   + f'msout={out} ' + dysco + 'msout.uvwcompression=False msout.antennacompression=False '
                   'msout.scalarflags=False ' + swapargs + pfargs + acargs
                   + 'pa.type=PythonDPPP pa.python.module=parang pa.python.class=ParAng pa.mode=correct pa.receptorangle=0 '
                   + f'avg.type=averager avg.timestep={timestep} avg.freqstep={freqstep} '
                   + 'steps=[' + ','.join(steps) + ']')
            terminal_print('Target DP3 run:', cmd)
            run(cmd, log=True)
            set_receptor_angle_zero(out)
            fix_uvw([out])  # mixed MeerKAT UVW signs averaged by DP3 and inserted time slots: recompute all
            run('python ' + SUBMODPATH + '/shadow_flag.py --ms=' + out)
            aoflagger_column([out], aoflagger_strategy='defaultMeerKAT_StokesQUV.lua', column='DATA')
            with table(out, ack=False, readonly=False) as t:
                t.putkeyword('MEERKATCAL_APPLIED', 'feedswap,' + ','.join(os.path.basename(h) for h in scanchain)
                             + ',parang')
                t.putkeyword('MEERKATCAL_FRAME', 'sky')
            shutil.rmtree(view, ignore_errors=True)
            open(out + '.done', 'w').close()
            terminal_print(f'{out}: {100 * flag_fraction(out):.1f}% flagged')
            outputs.append(out)
    if args['meerkatcal_target_concat']:
        outputs = concat_target_outputs(info, outputs)
    return dict(outputs=outputs, flagged={o: flag_fraction(o) for o in outputs}, h5chain=h5chain)


def target_ms(name):
    """The calibrated MS of a target with its scans concatenated in time (--meerkatcal-target-concat)."""
    return f'targets/{name}.cal.ms'


def concat_target(name, scanms):
    """
    Concatenate the calibrated scan MSs of a target in time into target_ms(name) and remove the scan MSs.

    A casacore concatenation of the scans (in scan order, so the time axis is increasing), deep-copied
    with the storage managers of the scans, so Dysco compressed scans give a Dysco compressed MS (the data
    are decoded and encoded again). The subtables are those of the first scan, the same for all scans of
    a field. The MS has time gaps where other fields were observed; facetselfcal splits it at those gaps
    (--timesplitbefore, chosen automatically).

    Parameters
    ----------
    name : str
        Target field name.
    scanms : list of str
        Calibrated scan MSs of the target (targets/<name>.scanNNN.cal.ms).

    Returns
    -------
    str
        The concatenated MS.
    """
    scanms = sorted(scanms)  # scan numbers are zero padded
    final = target_ms(name)
    if os.path.isdir(final):
        shutil.rmtree(final)
    terminal_print(f'Concatenating {len(scanms)} scans of {name} into {final}')
    chains = []
    for ms in scanms:
        with table(ms, ack=False) as t:
            chains.append(t.getkeyword('MEERKATCAL_APPLIED').split(','))
    # the chains are the same except for per-scan tables (--meerkatcal-weightgain), named with scanNNN
    applied = [items[0] if len(set(items)) == 1 else re.sub(r'scan\d+', 'scanNNN', items[0])
               for items in zip(*chains)] if len({len(c) for c in chains}) == 1 else chains[0]
    with table(scanms, ack=False) as t:
        t.copy(final, deep=True)
    with table(final, ack=False, readonly=False) as t:
        t.putkeyword('MEERKATCAL_APPLIED', ','.join(applied))
        t.putkeyword('MEERKATCAL_FRAME', 'sky')
    for ms in scanms:
        shutil.rmtree(ms, ignore_errors=True)
        if os.path.isfile(ms + '.done'):
            os.remove(ms + '.done')
    open(final + '.done', 'w').close()
    terminal_print(f'{final}: {100 * flag_fraction(final):.1f}% flagged')
    return final


def concat_target_outputs(info, outputs):
    """
    Replace the per-scan target MSs in outputs by one concatenated MS per target (concat_target).
    Targets that were already concatenated are kept.
    """
    new = []
    for name in info['roles']['targets']:
        scans = [o for o in outputs if os.path.basename(o).startswith(f'{name}.scan') and os.path.isdir(o)]
        if scans:
            new.append(concat_target(name, scans))
        elif target_ms(name) in outputs or os.path.isfile(target_ms(name) + '.done'):
            new.append(target_ms(name))
        else:
            terminal_print(f'WARNING: no calibrated scans of {name} found to concatenate')
    return new


def run_meerkatcal(mslist, args):
    """
    Run the MeerKAT calibrator-based calibration (1GC) on an archive MS (all fields).

    Parameters
    ----------
    mslist : list of str
        The archive Measurement Set (one MS with all fields), or with already split input (the calibrator
        MSs given with --meerkatcal-bpcal-ms, --meerkatcal-phasecal-ms and --meerkatcal-polcal-ms) the
        target MSs.
    args : dict
        facetselfcal options.
    """
    mslist = [os.path.abspath(m.rstrip('/')) for m in mslist]
    if split_input(args):
        if not args['meerkatcal_bpcal_ms']:
            raise ValueError('Already split input needs the bandpass calibrator MS(s) in --meerkatcal-bpcal-ms')
        named = [o for o in ('bpcal', 'phasecal', 'polcal', 'targets') if args[f'meerkatcal_{o}']]
        if named:
            raise ValueError('With already split MSs the roles are given by MS (--meerkatcal-bpcal-ms, '
                             '--meerkatcal-phasecal-ms, --meerkatcal-polcal-ms, the target MSs as input), not '
                             'by field name: remove ' + ', '.join(f'--meerkatcal-{o}' for o in named))
    elif len(mslist) != 1:
        raise ValueError('--meerkatcal expects one archive MS with all fields; for already split MSs give the '
                         'calibrator MSs with --meerkatcal-bpcal-ms (and --meerkatcal-phasecal-ms, '
                         '--meerkatcal-polcal-ms) and the target MSs as input')
    for d in ('misc', 'cal', 'h5_solutions', 'plots'):
        os.makedirs(d, exist_ok=True)
    stopafter = args['meerkatcal_stopafter']

    if _done('discover'):
        info = json.load(open('misc/meerkatcal_discovery.json'))
    else:
        info = discover(mslist, args)
        _finish('discover', info)
    if stopafter == 'discover':
        return
    # The flocs container sets OMP_NUM_THREADS to the core count and DP3 is linked against MKL, which then
    # starts that many threads inside each DP3 thread: a per-dump phaseonly solve ran ~800 threads and
    # spent most of its time in system calls. DP3, AOFlagger and WSClean use their own thread pools.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        _run_stages(info, args, stopafter)
    finally:
        write_report_json(info, args)


def _run_stages(info, args, stopafter):
    """The stages after discovery, see run_meerkatcal."""
    if not _done('prepare'):
        _finish('prepare', stage_prepare(info, args))
    setup_meerkatcal_refant(info, args)
    if stopafter == 'prepare':
        return

    if not _done('bandpass'):
        _finish('bandpass', stage_bandpass(info, args))
    if stopafter == 'bandpass':
        return

    pre_h5 = []  # leakage and cross-hand phase, applied after the bandpass
    if info['level'] in ('leakage', 'full'):
        if not _done('leakage'):
            bandpass = json.load(open(_marker('bandpass')))
            _finish('leakage', stage_leakage(info, args, bandpass['pregains'], bandpass['scanbandpass']))
        pre_h5.append(LEAKAGE_H5)
    if stopafter == 'leakage':
        return

    have_phasecal = len(info['roles']['phasecal']) > 0
    if not have_phasecal:
        terminal_print('WARNING: no phase (secondary) calibrator: only the bandpass (and leakage/cross-hand phase) '
                       'calibration is transferred to the targets, the flux scale comes from the bandpass calibrator '
                       'and the time-variable gains are left to the facetselfcal self-calibration')
    elif not _done('phasecal'):
        _finish('phasecal', stage_phasecal(info, args, extra_h5=pre_h5))
    if stopafter == 'phasecal':
        return

    if info['level'] == 'full':
        if not _done('polangle'):
            _finish('polangle', stage_polangle(info, args, pre_h5))
        if os.path.isfile(XFSOLVE_H5):
            pre_h5.append(XFSOLVE_H5)
        elif (_summary('polangle') or {}).get('skipped'):
            # no angle calibration (no noise diode firing, no usable Xf): the rest of the run is level leakage. Only in
            # memory, so that a rerun after removing misc/meerkatcal_polangle.done (e.g. with
            # --meerkatcal-diode-times) tries again
            terminal_print(f"Polarisation angle not calibrated ({_summary('polangle')['skipped']}): level leakage")
            info['level'] = 'leakage'
    if stopafter == 'polangle':
        return

    if info['level'] == 'full' and os.path.isfile(XFSOLVE_H5) and not _done('polcheck'):
        try:
            _finish('polcheck', stage_polcheck(info, args, pre_h5))
        except Exception as exc:  # a failed check must not stop the calibration of the targets
            terminal_print(f'WARNING: the polarisation calibrator check failed: {exc}')
    if stopafter == 'polcheck':
        return

    # the corrections are applied from the outermost Jones term inwards: time-variable gains (scalaramplitude,
    # phaseonly, delay), bandpass (complexgain), leakage, cross-hand phase (xfsolve) and then the
    # parallactic angle
    h5chain = [BANDPASS_H5] + pre_h5
    if have_phasecal:
        h5chain = [PHASECAL_H5[s] for s in ('scalaramplitude', 'phaseonly', 'delay')] + h5chain
    if not _done('targets'):
        _finish('targets', stage_targets(info, args, h5chain))
    elif args['meerkatcal_target_concat']:
        # a run that finished with one MS per target scan (before --meerkatcal-target-concat, or with
        # --no-meerkatcal-target-concat): concatenate those scans now, no recalibration
        summary = _summary('targets')
        if any(re.search(r'\.scan\d+\.cal\.ms$', o) for o in summary['outputs']):
            outputs = concat_target_outputs(info, summary['outputs'])
            _finish('targets', dict(summary, outputs=outputs, flagged={o: flag_fraction(o) for o in outputs}))
    terminal_print('meerkatcal finished, calibrated target MSs (sky frame): '
                   + ' '.join(json.load(open(_marker('targets')))['outputs']))
