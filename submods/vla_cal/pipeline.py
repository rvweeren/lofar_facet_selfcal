"""facetselfcal --vlacal: CASA-free calibration of historical (pre-JVLA) VLA data from a CASA importvla MS
(all fields and spectral windows in one MS).

Stages (each is skipped when it finished in an earlier run in the same directory):
discover   band, spectral windows, field roles (calibrator codes and positions), phase calibrator of each
           target, parallactic angle coverage, calibration level
prepare    split the calibrator scans per spectral window, quack, flag zeros, shadowing and (EVLA-transition
           data) the VLA-EVLA baselines, apply the VLA gain curve
fluxcal    Perley-Butler 2017 model, per-integration phaseonly and per-scan complexgain (per channel: the
           bandpass, and for the 1-channel continuum data the gain amplitudes that set the flux scale)
phasecal   phase calibrator model (short facetselfcal run), phaseonly and scalaramplitude per scan
leakage    leakage (D-terms) with the calibrator polarisation (D+QU, dqu.py) on calibrators with enough
           parallactic angle coverage, or on an unpolarised calibrator
polangle   R-L phase (polarisation angle) with xfsolve on 3C286 (or 3C138 at L-band) (level full)
polcheck   images of the polarisation calibrator, polarisation compared with the model (level full)
targets    apply everything to the target scans, parallactic angle correction, flag, one MS per target
           and spectral window
"""
import glob
import gzip
import hashlib
import json
import os
import re
import shutil

import numpy as np
from casacore.tables import table  # type: ignore

from submods.applycal import applycal, applycal_dp3_steps
from submods.calibrator_models import setjy_dp3
from submods.flagging import aoflagger_column
from submods.h5_helpers.h5_info import set_refant
from submods.h5_helpers.solution_qa import merge_bandpasses, qa_crosshand, qa_gains, qa_leakage
from submods.meerkat_cal import plots, polcheck
from submods.meerkat_cal.pipeline import feed_frame_model
from submods.meerkat_cal.prepare import flag_fraction, flag_residual, snapshot_flags, time_factor
from submods.meerkat_cal.solve import (combine_xf, concat_h5_time, normalise_amplitudes, nthreads,
                                       solve_ddecal, solve_xf, unwrap_phase)
from submods.ms_info import choose_refant_from_ms
from submods.model_columns import set_constant_model_column
from submods.runtime_utils import get_logger, run, terminal_print
from submods.vla_cal.discover import CIRCULAR_CORR, choose_level, discover
from submods.vla_cal.dqu import solve_leakage
from submods.vla_cal.prepare import VLACAL_SNAPSHOT_STAGES, clip_residuals, refresh_fake_crosshands, split_scan_spw
from submods.vla_helpers import get_vla_maxconfiguration

STAGES = ['discover', 'prepare', 'fluxcal', 'phasecal', 'leakage', 'polangle', 'polcheck', 'targets']
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TELESCOPE = 'VLA'
# pixel scale (arcsec) in A-configuration, as facetselfcal chooses it for the VLA (x3 per smaller configuration)
PIXELSCALE_A = {'L': 0.3, 'C': 0.075, 'X': 0.045}
CONFIG_FACTOR = {'A': 1., 'B': 3., 'C': 9., 'D': 27.}
# channels at or above which AOFlagger (on the residual) is used in addition to the clipper
AOFLAGGER_MINCHAN = 16


def BANDPASS_H5(spw):
    return f'h5_solutions/vlacal_fluxcal_complexgain_spw{spw}.h5'


def LEAKAGE_H5(spw):
    return f'h5_solutions/vlacal_leakage_spw{spw}.h5'


def XFSOLVE_H5(spw):
    return f'h5_solutions/vlacal_polcal_xfsolve_spw{spw}.h5'


def PHASECAL_H5(name, spw, soltype):
    return f'h5_solutions/vlacal_phasecal_{_safe(name)}_spw{spw}_{soltype}.h5'


def _safe(name):
    """Field name for file names (+ and - kept readable, other special characters replaced)."""
    return re.sub(r'[^A-Za-z0-9_.]', lambda m: {'+': 'p', '-': 'm'}.get(m.group(), '_'), name)


def _marker(stage):
    return f'misc/vlacal_{stage}.done'


def _done(stage):
    return os.path.isfile(_marker(stage))


def _finish(stage, summary):
    with open(_marker(stage), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    terminal_print(f'vlacal: stage {stage} finished')


def _summary(stage):
    return json.load(open(_marker(stage))) if _done(stage) else None


def _warn(message, collect=None):
    terminal_print('WARNING: ' + message)
    logger = get_logger()
    if logger is not None:
        logger.warning(message)
    if collect is not None:
        collect.append(message)


def _qa(function, *args, **kwargs):
    """Run a solution QA function; with very few solutions (one channel, few scans) some checks are not
    defined, so a failure is reported instead of stopping the calibration."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        terminal_print(f'QA {function.__name__} skipped for {args[0]}: {exc}')
        return dict(skipped=str(exc))


def reference(h5parm, telescope=TELESCOPE):
    """
    Reference the phases of an h5parm to the reference antenna of the run, with a fallback where the
    reference antenna has no solution (flagged in that time slot; with per-integration solves a plain
    referencing makes every antenna NaN there, and applying the solutions flags the whole time slot).

    Where the reference antenna is missing, another antenna is used, rotated per frequency, direction
    and polarisation by its mean phase offset from the reference antenna over the times where both have
    solutions. Antennas are tried in the order of how many of the missing solutions they fill. The phase
    common to all antennas is arbitrary per solution (it cancels in the visibilities); what the fallback
    keeps is the R-L phase of the reference, the instrumental R-L phase of the reference antenna, on which
    the leakage and R-L phase (xfsolve) solves rely. The atmospheric phase is common to R and L, so the
    drift of the offset in time does not enter the R-L phase.

    Parameters
    ----------
    h5parm : str
        h5parm with phase000 (modified in place).
    telescope : str, optional
        Telescope name for findrefant_core.

    Returns
    -------
    str
        The reference antenna.
    """
    import tables
    from submods.h5_helpers.h5_info import findrefant_core
    from submods.h5_helpers.solution_qa import _axes, _from_standard, _to_standard
    refant = findrefant_core(h5parm, telescope=telescope)
    with tables.open_file(h5parm, 'r+') as H:
        if 'phase000' not in H.root.sol000._v_children:
            return refant
        st = H.root.sol000.phase000
        axes = _axes(st)
        ants = [a.decode() if isinstance(a, bytes) else str(a) for a in st.ant[:]]
        z = np.exp(1j * _to_standard(st.val[:], axes))  # time, freq, ant, dir, pol; NaN where flagged
        r = ants.index(refant)
        zr = z[:, :, r]
        ref = zr.copy()
        missing = ~np.isfinite(ref)
        nfilled = 0
        if missing.any() and np.isfinite(ref).any():
            gain = np.sum(np.isfinite(z) & missing[:, :, None], axis=(0, 1, 3, 4))
            for c in np.argsort(-gain):
                if c == r or not missing.any():
                    continue
                zc = z[:, :, c]
                both = np.isfinite(zr) & np.isfinite(zc)
                offset = np.sum(np.where(both, zr * np.conj(zc), 0.), axis=0)  # freq, dir, pol
                ok = np.abs(offset) > 0
                offset = np.where(ok, offset / np.where(ok, np.abs(offset), 1.), np.nan)
                fill = missing & np.isfinite(zc) & ok[None]
                ref[fill] = (zc * offset[None])[fill]
                nfilled += int(fill.sum())
                missing &= ~fill
        phase = np.angle(z * np.conj(ref)[:, :, None])
        st.val[:] = _from_standard(phase, axes)
        st.weight[:] = _from_standard(np.where(np.isfinite(phase), 1., 0.) *
                                      (_to_standard(st.weight[:], axes) > 0), axes)
    if nfilled:
        terminal_print(f'{h5parm}: reference antenna {refant} has no solution in {nfilled} cells, referenced via '
                       'other antennas (their mean phase offset from it)')
    unwrap_phase(h5parm)
    return refant


def _scan_ms(name, scan, spw):
    return f'cal/{_safe(name)}.scan{scan:03d}.spw{spw}.ms'


def _tag(ms):
    return os.path.basename(ms).replace('.B.ms', '').replace('.ms', '')


def _field_scans(info, name, spw):
    """Scans of a field in a spectral window (a band-switching scan observes only some windows)."""
    field = info['fields'][name]
    return field['scans_spw'].get(str(spw), []) if 'scans_spw' in field else field['scans']


def _scans(info, names, spw):
    return [_scan_ms(n, s, spw) for n in names for s in _field_scans(info, n, spw)]


def _nfreq(mslist):
    """Number of distinct channel frequencies of a set of MSs (two IF pairs can share one frequency)."""
    freqs = set()
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            freqs.update(np.round(t.getcell('CHAN_FREQ', 0)).tolist())
    return len(freqs)


def _calibrators(info):
    roles = info['roles']
    return sorted(set(roles['fluxcal'] + roles['phasecal'] + roles['polcal'] + list(info['leakcal'])))


def _apply(ms, h5list, incol, outcol, msout='.', updateweights=False, interpolation='linear'):
    """Correct with a list of h5parms (outermost Jones term first); NaN solutions flag the data.
    --vlacal never uses Dysco: the data are small, and the solves need exact data and weights.
    Solutions solved on the same MS are applied with interpolation='nearest' (_selfapply): with
    'linear', DP3 also flags the time slots next to a NaN solution, so every in-place apply of
    per-integration solutions grows the flags by one slot around each flagged slot."""
    if updateweights and msout == '.':
        raise ValueError('_apply: updateweights needs a new output MS, WEIGHT_SPECTRUM is shared by all columns')
    applycal(ms, h5list, msincol=incol, msoutcol=outcol, msout=msout, dysco=False, invert=True,
             missingantennabehavior='flag', metadata_compression=False, interpolation=interpolation,
             updateweights=updateweights)


def _selfapply(ms, h5list, incol, outcol, msout='.'):
    """Apply solutions solved on this MS itself (same time grid): nearest, no flag growth."""
    _apply(ms, h5list, incol, outcol, msout=msout, interpolation='nearest')


def _slots(ms, seconds):
    """Number of time slots in a time interval (at least 1)."""
    return time_factor(ms, seconds)


def _nchan(ms):
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        return len(t.getcell('CHAN_FREQ', 0))


def _fourpol(info):
    """True if the input MS has all four circular correlations (else RR and LL only, and the split MSs
    have fake cross-hands RL = LR = RR - LL, see prepare.split_scan_spw)."""
    return all(list(s['corr']) == CIRCULAR_CORR for s in info['spws'])


def flag_calibrator(ms, datacolumn='CORRECTED_DATA', modelcolumn='MODEL_DATA', crosshands=True):
    """Flag a calibrator on its residual: the robust clipper, and AOFlagger (Stokes QUV) for many channels.
    crosshands=False for RR/LL-only data: the clipper uses Stokes V only, and the fake cross-hands of
    datacolumn are set to RR - LL first (the Stokes I model has RL = LR = 0, so the residual cross-hands
    are then Stokes V)."""
    clip_residuals(ms, datacolumn=datacolumn, modelcolumn=modelcolumn, mode='calibrator', crosshands=crosshands)
    if _nchan(ms) >= AOFLAGGER_MINCHAN:
        if not crosshands:
            refresh_fake_crosshands(ms, datacolumn)
        flag_residual(ms, datacolumn=datacolumn, modelcolumn=modelcolumn, aoflagger_strategy='default_StokesQUV.lua')


def pixelscale(info, mslist):
    """Pixel scale (arcsec) for the band and the VLA configuration, as facetselfcal chooses it."""
    return PIXELSCALE_A[info['band']] * CONFIG_FACTOR.get(get_vla_maxconfiguration(mslist), 1.)


def stage_prepare(info, args):
    """Split every calibrator scan per spectral window, quack, flag and apply the gain curve."""
    summary, gaincurve = {}, {}
    for name in _calibrators(info):
        for spw in info['spws']:
            for scan in _field_scans(info, name, spw['spw']):
                out = _scan_ms(name, scan, spw['spw'])
                if os.path.isfile(out + '.prepared'):
                    summary[out] = json.load(open(out + '.prepared'))
                    continue
                gc = split_scan_spw(info['ms'], info['fields'][name]['field_id'], scan, spw, out,
                                    quack=args['vlacal_quack'], flag_mixed=args['vlacal_flag_mixed_baselines'],
                                    gaincurve=args['vlacal_gaincurve'])
                summary[out] = dict(flagged=flag_fraction(out), gaincurve=gc)
                terminal_print(f'{out}: {100 * summary[out]["flagged"]:.1f}% flagged after preparation')
                with open(out + '.prepared', 'w') as f:
                    json.dump(summary[out], f)
    for out, item in summary.items():
        if item.get('gaincurve'):
            name = os.path.basename(out).split('.scan')[0]
            lo, hi = gaincurve.get(name, (np.inf, -np.inf))
            gaincurve[name] = (min(lo, item['gaincurve']['gain'][0]), max(hi, item['gaincurve']['gain'][1]))
    return dict(flagged={k: v['flagged'] for k, v in summary.items()}, gaincurve=gaincurve)


def setup_refant(info, args):
    """Fix the reference antenna for all solves (--refant, an earlier choice, or the least flagged one)."""
    refantfile = 'misc/refant.txt'
    if args['refant'] is not None:
        refant = args['refant']
    elif os.path.isfile(refantfile):
        refant = open(refantfile).read().strip()
    else:
        # all calibrators: an antenna that is good on the flux calibrator can be flagged for much of the
        # phase calibrator scans (in a test, one antenna was flagged for half of the phase calibrator time slots)
        mslist = [ms for spw in info['spws'] for ms in _scans(info, _calibrators(info), spw['spw'])]
        refant = choose_refant_from_ms(mslist, TELESCOPE)
    with open(refantfile, 'w') as f:
        f.write(refant + '\n')
    set_refant(refant)
    terminal_print('Reference antenna for all solves:', refant)
    return refant


def stage_fluxcal(info, args):
    """
    Flux density (and bandpass) calibrator: Perley-Butler 2017 model (setjy_dp3), then per scan a
    per-integration phaseonly (over all channels) and one complexgain per channel for the whole scan;
    the complexgains of the scans are averaged (merge_bandpasses) per spectral window. With one channel
    per spectral window (continuum mode) this is the gain amplitude per antenna and polarisation that
    sets the flux scale; the phases are taken out by the phaseonly solve and solved on the phase
    calibrator. Solve and flag cycles on the residuals.
    """
    ncycles = max(1, args['vlacal_flagcycles'])
    qa, ratios, pregains = {}, {}, {}
    for spw in info['spws']:
        s = spw['spw']
        mss = _scans(info, info['roles']['fluxcal'], s)
        for ms in mss:
            snapshot_flags(ms, 'fluxcal', VLACAL_SNAPSHOT_STAGES)
            setjy_dp3(ms)
        merged = None
        for cycle in range(ncycles):
            blist = []
            for ms in mss:
                tag = _tag(ms)
                ph5 = f'h5_solutions/fluxcal_phaseonly_c{cycle}_{tag}.h5'
                solve_ddecal(ms, ph5, 'phaseonly', 'DATA', solint=1, nchan=0)
                reference(ph5, telescope=TELESCOPE)
                _selfapply(ms, [ph5], 'DATA', 'DATA_PHASEONLY')
                bp5 = f'h5_solutions/fluxcal_complexgain_c{cycle}_{tag}.h5'
                solve_ddecal(ms, bp5, 'complexgain', 'DATA_PHASEONLY', solint=0, nchan=1, timeavg=True)
                reference(bp5, telescope=TELESCOPE)
                # the flag cycle uses the solution before its QA: applying the QA flags would flag the
                # data for good, and a false positive would become a gap the next cycle cannot solve
                bp5_noqa = bp5.replace('.h5', '_noqa.h5')
                shutil.copy(bp5, bp5_noqa)
                if spw['nchan'] > 1:
                    qa[bp5] = _qa(qa_gains, bp5, along='freq')
                blist.append(bp5)
                pregains[ms] = ph5
                if cycle < ncycles - 1:
                    _selfapply(ms, [ph5, bp5_noqa], 'DATA', 'CORRECTED_DATA')
                    flag_calibrator(ms, crosshands=_fourpol(info))
                    terminal_print(f'{ms}: {100 * flag_fraction(ms):.1f}% flagged after cycle {cycle}')
            merged = f'h5_solutions/fluxcal_complexgain_c{cycle}_spw{s}_merged.h5'
            r = merge_bandpasses(blist, merged)
            ratios[s] = dict(zip(blist, [float(x) for x in r]))
        shutil.copy(merged, BANDPASS_H5(s))
        unwrap_phase(BANDPASS_H5(s))
        os.makedirs('plots', exist_ok=True)
        if spw['nchan'] > 1:
            plots.plot_spectrum(BANDPASS_H5(s), 'plots/' + os.path.basename(BANDPASS_H5(s)).replace('.h5', '.png'),
                                f'Bandpass spw{s} (complexgain)')
        for ms in mss:
            plots.plot_timeseries(pregains[ms], 'phase000',
                                  'plots/' + os.path.basename(pregains[ms]).replace('.h5', '.png'),
                                  os.path.basename(pregains[ms]))
    return dict(qa=qa, amplitude_ratios=ratios, pregains=pregains,
                flagged={ms: flag_fraction(ms) for spw in info['spws']
                         for ms in _scans(info, info['roles']['fluxcal'], spw['spw'])})


def _phasecal_model_selfcal(info, name, msin_per_spw, workdir):
    """
    Model of a phase calibrator from a short facetselfcal DI run on its bandpass-calibrated scans (one
    MS per spectral window, the scans concatenated in time). Returns the image name of the last cycle.
    """
    concat = []
    for s, msin in msin_per_spw.items():
        if not msin:
            continue
        out = os.path.join(workdir, f'{_safe(name)}_spw{s}_concat.ms')
        with table(msin, ack=False) as t:
            t.copy(out, deep=True)
        concat.append(os.path.abspath(out))
    channelsout = max(1, min(6, _nfreq(concat)))
    # no spectral fit for a few channels (each spectral window imaged on its own), facetselfcal needs
    # fitspectralpol < channelsout
    fitspectralpol = 0 if channelsout <= 3 else min(3, channelsout - 1)
    cmd = (f'cd {workdir} && python {REPO}/facetselfcal.py -i phasecal --imsize=1536 '
           f'--pixelscale={pixelscale(info, concat)} --channelsout={channelsout} --fitspectralpol={fitspectralpol} '
           '--niter=20000 --stop=3 '
           "--soltype-list=\"['scalarphase','scalarcomplexgain']\" --solint-list=\"['1min','10min']\" "
           "--soltypecycles-list=\"[0,1]\" --smoothnessconstraint-list=\"[0.,0.]\" "
           f'--refant={open("misc/refant.txt").read().strip()} --noarchive --skip-Ateam-plotting --dysco=False '
           + ' '.join(concat))
    terminal_print('facetselfcal run on the phase calibrator:', cmd)
    run(cmd, log=True)
    models = sorted(glob.glob(f'{workdir}/fits_images/phasecal_[0-9][0-9][0-9]-*model*.fits*'))
    if not models:
        raise RuntimeError('No model images from the phase calibrator facetselfcal run in ' + workdir)
    return os.path.basename(models[-1]).split('-')[0], channelsout


def _predict_phasecal_model(name, channelsout, msmodel, workdir):
    """Predict the facetselfcal model images into MODEL_DATA of msmodel: one WSClean run over all MSs, so
    that its division of the frequency range in output channels is that of the imaging run."""
    if channelsout > 1:
        images = sorted(glob.glob(f'{workdir}/fits_images/{name}-[0-9][0-9][0-9][0-9]-model.fits*'))
    else:
        images = sorted(glob.glob(f'{workdir}/fits_images/{name}-model.fits*'))
    for image in images:
        out = os.path.join(workdir, os.path.basename(image).removesuffix('.gz'))
        if image.endswith('.gz'):
            with gzip.open(image, 'rb') as fin, open(out, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copy(image, out)
    prefix = os.path.join(workdir, name)
    chans = f'-channels-out {channelsout} ' if channelsout > 1 else ''
    run(f'wsclean -predict -name {prefix} {chans}-j {nthreads()} ' + ' '.join(msmodel), log=True)


def _file_digest(files):
    md5 = hashlib.md5()
    for f in files:
        with open(f, 'rb') as fp:
            for block in iter(lambda: fp.read(1 << 20), b''):
                md5.update(block)
    return md5.hexdigest()


def _phasecal_solves(ms, label):
    """
    Gains of one phase calibrator scan against MODEL_DATA, over all channels: phaseonly per scan (the
    phase transferred to the targets), and scalaramplitude per scan on the data corrected with a
    per-integration phaseonly (as CASA gaincal solint='int' calmode='p' before solint='inf' 'ap'): on
    data averaged over a scan with only one phase per scan, the phase drift within the scan
    decorrelates the average and biases the amplitudes low (4-5% on test data). Returns the per-scan
    phaseonly, scalaramplitude and per-integration phaseonly h5parms; DATA_PHASEINT holds the data
    with the per-integration phases applied.
    """
    phaseh5 = f'h5_solutions/phasecal_phaseonly_{label}.h5'
    phaseinth5 = f'h5_solutions/phasecal_phaseonly_int_{label}.h5'
    amph5 = f'h5_solutions/phasecal_scalaramplitude_{label}.h5'
    solve_ddecal(ms, phaseh5, 'phaseonly', 'DATA', solint=0, nchan=0, timeavg=True)
    reference(phaseh5, telescope=TELESCOPE)
    solve_ddecal(ms, phaseinth5, 'phaseonly', 'DATA', solint=1, nchan=0)
    reference(phaseinth5, telescope=TELESCOPE)
    _selfapply(ms, [phaseinth5], 'DATA', 'DATA_PHASEINT')
    solve_ddecal(ms, amph5, 'scalaramplitude', 'DATA_PHASEINT', solint=0, nchan=0, timeavg=True)
    return phaseh5, amph5, phaseinth5


def _flux_density(mslist, column='CORRECTED_DATA'):
    """Vector-averaged Stokes I flux density (Jy) of the calibrated data of a point-like calibrator."""
    num, den = 0., 0.
    for ms in mslist:
        with table(ms, ack=False) as t:
            d = t.getcol(column)
            f = t.getcol('FLAG')
            w = t.getcol('WEIGHT_SPECTRUM')
            a1, a2 = t.getcol('ANTENNA1'), t.getcol('ANTENNA2')
        ok = ~(f[..., 0] | f[..., 3]) & (a1 != a2)[:, None]
        I = 0.5 * (d[..., 0] + d[..., 3])
        wi = 0.5 * (w[..., 0] + w[..., 3])
        num += np.sum((wi * I)[ok])
        den += np.sum(wi[ok])
    return float(abs(num / den)) if den > 0 else float('nan')


def stage_phasecal(info, args):
    """
    Phase calibrators: apply the flux calibrator gains (bandpass), make the model (facetselfcal run, or
    a point source), then solve phaseonly and scalaramplitude per scan (the time-variable gains transferred
    to the targets), with flag cycles. The scalaramplitudes are normalised to a median of 1, so the
    absolute flux scale is that of the flux calibrator; the bootstrapped flux density of each phase
    calibrator is measured from its calibrated data.
    """
    results, msB_all = {}, {}
    ncycles = max(1, args['vlacal_flagcycles'])
    bandpasses = [BANDPASS_H5(spw['spw']) for spw in info['spws']]
    for name in info['roles']['phasecal']:
        workdir = os.path.join('phasecal_selfcal', _safe(name))
        checkpoint = os.path.join(workdir, 'vlacal_model.json')
        inputs = _file_digest(bandpasses)
        state = json.load(open(checkpoint)) if os.path.isfile(checkpoint) else {}
        if state.get('inputs') == inputs and state.get('selfcal') == args['vlacal_phasecal_selfcal']:
            terminal_print(f'Reusing the phase calibrator data and model in {workdir} ({state["model"]})')
        else:
            if os.path.isdir(workdir):
                shutil.rmtree(workdir)
            os.makedirs(workdir)
            msB = {}
            for spw in info['spws']:
                s = spw['spw']
                msB[s] = []
                for ms in _scans(info, [name], s):
                    out = os.path.join(workdir, os.path.basename(ms).replace('.ms', '.B.ms'))
                    _apply(ms, [BANDPASS_H5(s)], 'DATA', 'DATA', msout=out, updateweights=True)
                    set_constant_model_column(out, 'MODEL_DATA', 1.0)
                    msB[s].append(out)
            state = dict(inputs=inputs, selfcal=args['vlacal_phasecal_selfcal'], msB=msB, model='point source',
                         channelsout=1)
            if args['vlacal_phasecal_selfcal']:
                try:
                    msG = {}
                    for s, mslist in msB.items():
                        msG[s] = []
                        for ms in mslist:
                            _, _, phaseinth5 = _phasecal_solves(ms, f'pointsource_{_tag(ms)}')
                            out = ms.replace('.B.ms', '.BG.ms')
                            _selfapply(ms, [phaseinth5], 'DATA', 'DATA', msout=out)
                            msG[s].append(out)
                    model, channelsout = _phasecal_model_selfcal(info, name, msG, workdir)
                    state['model'], state['channelsout'] = model, channelsout
                except Exception as exc:  # noqa: BLE001
                    _warn(f'the facetselfcal run on the phase calibrator {name} failed ({exc}), '
                          'using a point source model')
            with open(checkpoint, 'w') as fp:
                json.dump(state, fp)
        msB = {int(s): v for s, v in state['msB'].items()}
        msB_all[name] = msB
        allms = [ms for v in msB.values() for ms in v]
        for ms in allms:
            snapshot_flags(ms, 'phasecal', VLACAL_SNAPSHOT_STAGES)
        if state['model'] != 'point source':
            _predict_phasecal_model(state['model'], state['channelsout'], allms, workdir)
        qa, norm, flux = {}, {}, {}
        for s, mslist in msB.items():
            if not mslist:
                _warn(f'phase calibrator {name} has no scans in spw{s}: the targets paired with it get no '
                      f'phase calibrator gains in spw{s}')
                continue
            for cycle in range(ncycles):
                h5lists = {'phaseonly': [], 'scalaramplitude': [], 'phaseonly_int': []}
                for ms in mslist:
                    phaseh5, amph5, phaseinth5 = _phasecal_solves(ms, _tag(ms))
                    h5lists['phaseonly'].append(phaseh5)
                    h5lists['scalaramplitude'].append(amph5)
                    h5lists['phaseonly_int'].append(phaseinth5)
                    if cycle < ncycles - 1:
                        _selfapply(ms, [amph5], 'DATA_PHASEINT', 'CORRECTED_DATA')
                        flag_calibrator(ms, crosshands=_fourpol(info))
                        terminal_print(f'{ms}: {100 * flag_fraction(ms):.1f}% flagged after cycle {cycle}')
            for soltype, h5list in h5lists.items():
                concat_h5_time(h5list, PHASECAL_H5(name, s, soltype))
            amph5 = PHASECAL_H5(name, s, 'scalaramplitude')
            qa[f'spw{s}'] = _qa(qa_gains, amph5, along='time', window=5, minscale=0.05)
            norm[s] = normalise_amplitudes(amph5)
            for ms in mslist:  # per-integration phases: no decorrelation in the vector average
                _selfapply(ms, [amph5, PHASECAL_H5(name, s, 'phaseonly_int')], 'DATA', 'CORRECTED_DATA')
            flux[s] = _flux_density(mslist)
            terminal_print(f'Phase calibrator {name} spw{s}: bootstrapped flux density {flux[s]:.4f} Jy')
            plots.plot_timeseries(PHASECAL_H5(name, s, 'phaseonly'), 'phase000',
                                  f'plots/vlacal_phasecal_{_safe(name)}_spw{s}_phaseonly.png',
                                  f'Phase calibrator {name} spw{s} phaseonly')
            plots.plot_timeseries(amph5, 'amplitude000', f'plots/vlacal_phasecal_{_safe(name)}_spw{s}_scalaramplitude.png',
                                  f'Phase calibrator {name} spw{s} scalaramplitude')
        results[name] = dict(model=state['model'], qa=qa, normalisation=norm, flux_density_jy=flux,
                             flagged={ms: flag_fraction(ms) for ms in allms})
    return dict(phasecals=results, msB={n: {str(s): v for s, v in m.items()} for n, m in msB_all.items()})


def stage_leakage(info, args):
    """
    Leakage (D-terms) per spectral window with dqu.solve_leakage on the leakage calibrators, jointly:
    the phase calibrators (or --vlacal-leakcal fields) with enough parallactic angle coverage get their
    polarisation solved, unpolarised calibrators (3C84, OQ208, 3C147 at L-band) have it fixed at 0.
    The calibrator scans get the flux calibrator gains and a 30 s phaseonly solve; amplitude errors
    and phase errors common to R and L cancel in the solve, which takes Stokes I from the data.
    """
    phasecal_msB = (_summary('phasecal') or {}).get('msB', {})
    os.makedirs('leakage', exist_ok=True)
    results = {}
    for spw in info['spws']:
        s = spw['spw']
        mslist, names = [], []
        for name in info['leakcal']:
            prepared = phasecal_msB.get(name, {}).get(str(s))
            if prepared is None:  # not a phase calibrator: bandpass and a point source model
                prepared = []
                for ms in _scans(info, [name], s):
                    out = os.path.join('leakage', os.path.basename(ms).replace('.ms', '.B.ms'))
                    if not os.path.isdir(out):
                        _apply(ms, [BANDPASS_H5(s)], 'DATA', 'DATA', msout=out, updateweights=True)
                        set_constant_model_column(out, 'MODEL_DATA', 1.0)
                    prepared.append(out)
            for ms in prepared:
                snapshot_flags(ms, 'leakage', VLACAL_SNAPSHOT_STAGES)
                ph5 = f'h5_solutions/leakcal_phaseonly_{_tag(ms)}.h5'
                solve_ddecal(ms, ph5, 'phaseonly', 'DATA', solint=_slots(ms, 30.), nchan=0)
                reference(ph5, telescope=TELESCOPE)
                _selfapply(ms, [ph5], 'DATA', 'DATA_CAL')
                mslist.append(ms)
                names.append(name)
        res = solve_leakage(mslist, names, {n: info['leakcal'][n]['fitpol'] for n in info['leakcal']}, 'DATA_CAL',
                            LEAKAGE_H5(s), nchan=args['vlacal_leakage_nchan'])
        if spw['nchan'] > 1:
            res['qa'] = _qa(qa_leakage, LEAKAGE_H5(s))
            unwrap_phase(LEAKAGE_H5(s))
            plots.plot_spectrum(LEAKAGE_H5(s), 'plots/' + os.path.basename(LEAKAGE_H5(s)).replace('.h5', '.png'),
                                f'Leakage spw{s} (' + ', '.join(info['leakcal']) + ')')
        results[s] = res
    return dict(spws=results, calibrators=list(info['leakcal']))


def stage_polangle(info, args):
    """
    R-L phase (polarisation angle) per spectral window with xfsolve on the polarisation calibrator
    scans: polarised model (Perley-Butler 2017 Stokes I, Perley et al. 2026 polarisation) in the feed
    frame (parallactic angle), flux calibrator gains and leakage applied, phaseonly per integration
    against that model, then xfsolve per scan, combined over the scans.
    """
    results, scaninfo, warnings = {}, {}, []
    for spw in info['spws']:
        s = spw['spw']
        xlist = []
        for name in info['roles']['polcal']:
            for ms in _scans(info, [name], s):
                tag = _tag(ms)
                snapshot_flags(ms, 'polangle', VLACAL_SNAPSHOT_STAGES)
                setjy_dp3(ms)
                feed_frame_model(ms, None)
                _apply(ms, [BANDPASS_H5(s), LEAKAGE_H5(s)], 'DATA', 'DATA_PREAPPLY')
                phaseh5, xfh5 = (f'h5_solutions/polcal_{soltype}_{tag}.h5' for soltype in ('phaseonly', 'xfsolve'))
                solve_ddecal(ms, phaseh5, 'phaseonly', 'DATA_PREAPPLY', solint=1, nchan=0, modelcol='MODEL_FEED')
                reference(phaseh5, telescope=TELESCOPE)
                _selfapply(ms, [phaseh5], 'DATA_PREAPPLY', 'DATA_CAL')
                solve_xf(ms, xfh5, 'DATA_CAL', modelcol='MODEL_FEED', mode='model', nchan=1)
                xlist.append(xfh5)
                scaninfo[ms] = dict(field=name, spw=s, phaseonly=phaseh5, xfsolve=xfh5)
        theta = combine_xf(xlist, XFSOLVE_H5(s))
        qa = _qa(qa_crosshand, XFSOLVE_H5(s)) if spw['nchan'] >= 3 else {}
        unwrap_phase(XFSOLVE_H5(s))
        perscan = {}
        for xfh5 in xlist:
            single = xfh5.replace('.h5', '.single.h5')
            th = combine_xf([xfh5], single)
            os.remove(single)
            with np.errstate(invalid='ignore'):
                perscan[xfh5] = float(np.nanmedian(np.degrees(np.abs(np.angle(np.exp(1j * (th - theta)))))))
            if perscan[xfh5] > 5.:
                _warn(f'R-L phase of {xfh5} differs by {perscan[xfh5]:.1f} deg from the combined solution of spw{s}',
                      warnings)
        results[s] = dict(theta_deg=np.degrees(theta).tolist(), qa=qa, perscan_median_diff_deg=perscan)
        terminal_print(f'R-L phase spw{s}: {np.degrees(np.nanmedian(theta)):.1f} deg '
                       f'(position angle correction {np.degrees(np.nanmedian(theta)) / 2:.1f} deg)')
    return dict(spws=results, calibrators=info['roles']['polcal'], scans=scaninfo, warnings=warnings)


def stage_polcheck(info, args):
    """Image the calibrated polarisation calibrator (all spectral windows) and compare with the model."""
    scans = _summary('polangle')['scans']
    os.makedirs('polcheck', exist_ok=True)
    results = {}
    for name in info['roles']['polcal']:
        source = info['fields'][name]['polarisation_calibrator']
        calms = []
        for ms, sc in scans.items():
            if sc['field'] != name:
                continue
            s = sc['spw']
            out = os.path.join('polcheck', os.path.basename(ms).replace('.ms', '.cal.ms'))
            chain = [sc['phaseonly'], BANDPASS_H5(s), LEAKAGE_H5(s), XFSOLVE_H5(s)]
            polcheck.calibrated_scan(ms, chain, out, freqstep=1, nthreads=nthreads(), dysco=False)
            calms.append(out)
        nchunks = max(1, min(8, _nfreq(calms)))
        prefix = f'polcheck/{_safe(name)}'
        polcheck.image_iquv(calms, prefix, nchunks=nchunks, pixelscale=pixelscale(info, calms), imsize=512,
                            nthreads=nthreads())
        if nchunks == 1:  # WSClean names a single output channel without the channel index polcheck reads
            for f in glob.glob(prefix + '-[IQUV]-*.fits'):
                os.replace(f, f.replace(prefix + '-', prefix + '-0000-', 1))
        res = polcheck.measure(prefix, source, nchunks)
        res['warnings'] = polcheck.verdict(res, info['band'])
        res['thumbnails'] = polcheck.thumbnails(prefix, nchunks, 'plots/polcheck')
        polcheck.plot(res, f'plots/vlacal_polcheck_{_safe(name)}.png')
        for w in res['warnings']:
            terminal_print(f'WARNING polarisation check {name}: {w}')
        if not res['warnings']:
            terminal_print(f'Polarisation check {name}: pass')
        results[name] = res
        for ms in calms:
            shutil.rmtree(ms, ignore_errors=True)
    return results


def target_chain(info, name, spw):
    """h5parms applied to a target in a spectral window, outermost Jones term first."""
    phasecal = info['pairing'].get(name)
    chain = []
    if phasecal is not None and os.path.isfile(PHASECAL_H5(phasecal, spw, 'phaseonly')):
        chain += [PHASECAL_H5(phasecal, spw, 'scalaramplitude'), PHASECAL_H5(phasecal, spw, 'phaseonly')]
    chain.append(BANDPASS_H5(spw))
    if info['level'] in ('leakage', 'full'):
        chain.append(LEAKAGE_H5(spw))
    if info['level'] == 'full' and os.path.isfile(XFSOLVE_H5(spw)):
        chain.append(XFSOLVE_H5(spw))
    return chain


def stage_targets(info, args):
    """
    Apply the solutions to every target scan per spectral window (after quack, flagging and the gain
    curve as for the calibrators), correct the parallactic angle (sky frame), optionally average in time,
    flag (robust clipper on Stokes V and the cross-hands, AOFlagger for many channels), and concatenate the
    scans into one MS per target and spectral window.

    Weights: importvla writes WEIGHT = 1/sigma^2 from the nominal sensitivity (system temperatures) of
    the online system; they are scaled with the applied gain amplitudes (DP3 updateweights, as CASA
    applycal calwt=True).
    """
    os.makedirs('targets', exist_ok=True)
    outputs = []
    for name in info['roles']['targets']:
        for spw in info['spws']:
            s = spw['spw']
            final = f'targets/{_safe(name)}_spw{s}.ms'
            chain = target_chain(info, name, s)
            if os.path.isfile(final + '.done') and \
                    os.path.getmtime(final + '.done') > max(os.path.getmtime(h) for h in chain):
                outputs.append(final)
                continue
            scanms = []
            for scan in _field_scans(info, name, s):
                view = f'targets/{_safe(name)}.scan{scan:03d}.spw{s}.split.ms'
                out = f'targets/{_safe(name)}.scan{scan:03d}.spw{s}.cal.ms'
                split_scan_spw(info['ms'], info['fields'][name]['field_id'], scan, spw, view,
                               quack=args['vlacal_quack'], flag_mixed=args['vlacal_flag_mixed_baselines'],
                               gaincurve=args['vlacal_gaincurve'])
                acargs, nac = applycal_dp3_steps(view, chain, missingantennabehavior='flag', interpolation='linear',
                                                 updateweights=True)
                timestep = time_factor(view, args['vlacal_timeres'])
                steps = ['ac%d' % i for i in range(nac)] + ['pa'] + (['avg'] if timestep > 1 else [])
                if os.path.isdir(out):
                    shutil.rmtree(out)
                cmd = (f'DP3 numthreads={nthreads()} msin={view} msin.datacolumn=DATA msout={out} '
                       'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False '
                       + acargs + 'pa.type=PythonDPPP pa.python.module=parang pa.python.class=ParAng pa.mode=correct '
                       'pa.receptorangle=0 ' + (f'avg.type=averager avg.timestep={timestep} ' if timestep > 1 else '')
                       + 'steps=[' + ','.join(steps) + ']')
                terminal_print('Target DP3 run:', cmd)
                run(cmd, log=True)
                shutil.rmtree(view, ignore_errors=True)
                if not _fourpol(info):
                    refresh_fake_crosshands(out, 'DATA')
                clip_residuals(out, datacolumn='DATA', mode='target', nsigma=args['vlacal_target_clip'],
                               crosshands=_fourpol(info))
                if spw['nchan'] >= AOFLAGGER_MINCHAN:
                    aoflagger_column([out], aoflagger_strategy='default_StokesQUV.lua', column='DATA')
                scanms.append(out)
            if os.path.isdir(final):
                shutil.rmtree(final)
            with table(scanms, ack=False) as t:
                t.copy(final, deep=True)
            for ms in scanms:
                shutil.rmtree(ms, ignore_errors=True)
            with table(final, ack=False, readonly=False) as t:
                t.putkeyword('VLACAL_APPLIED', ('gaincurve,' if args['vlacal_gaincurve'] else '')
                             + ','.join(os.path.basename(h) for h in chain) + ',parang')
                t.putkeyword('VLACAL_FRAME', 'sky')
                if not _fourpol(info):  # RR/LL only: the cross-hands are fake (see facetselfcal FAKE_RLLR)
                    t.putcolkeyword('DATA', 'FAKE_RLLR', True)
            open(final + '.done', 'w').close()
            terminal_print(f'{final}: {100 * flag_fraction(final):.1f}% flagged')
            outputs.append(final)
    return dict(outputs=outputs, flagged={o: flag_fraction(o) for o in outputs},
                chains={f'{n} spw{spw["spw"]}': target_chain(info, n, spw['spw'])
                        for n in info['roles']['targets'] for spw in info['spws']})


def write_report_json(info):
    """
    Collect what the HTML report shows in misc/vlacal_report.json (same keys as meerkatcal_report.json,
    for the first spectral window, plus the vlacal-specific ones). Missing stages give null entries.
    """
    def existing(path):
        return path if path and os.path.isfile(path) else None

    spw0 = info['spws'][0]['spw']
    leak, pol, polchk, pcal, fcal, prep = (_summary(s) for s in ('leakage', 'polangle', 'polcheck', 'phasecal',
                                                                  'fluxcal', 'prepare'))
    flagging = {stage: (_summary(stage) or {}).get('flagged') for stage in ('fluxcal', 'targets')}
    if prep is not None:
        flagging['prepare'] = prep.get('flagged')
    leakage = None
    if leak is not None:
        block = (leak['spws'].get(str(spw0)) or {})
        leakage = dict(h5=existing(LEAKAGE_H5(spw0)), calibrators=leak['calibrators'], nchan=1,
                       gauge_c=(block.get('blocks') or [{}])[0].get('gauge_c'), qa=block.get('qa'),
                       solve={s: v.get('blocks') for s, v in leak['spws'].items()})
    xfsolve = None
    if pol is not None:
        xfsolve = dict(h5=existing(XFSOLVE_H5(spw0)), calibrators=pol['calibrators'], nchan=1,
                       qa=(pol['spws'].get(str(spw0)) or {}).get('qa'),
                       perscan_median_diff_deg={k: v for s in pol['spws'].values()
                                                for k, v in s['perscan_median_diff_deg'].items()},
                       perscan_unconstrained_mhz={}, warnings=pol.get('warnings', []),
                       perscan_h5=[sc['xfsolve'] for sc in pol['scans'].values()])
    report = dict(
        telescope='VLA', band=info['band'], level=info['level'], angle='celestial' if info['level'] == 'full' else None,
        refant=open('misc/refant.txt').read().strip() if os.path.isfile('misc/refant.txt') else None,
        roles=dict(bpcal=info['roles']['fluxcal'], phasecal=info['roles']['phasecal'],
                   polcal=info['roles']['polcal'], targets=info['roles']['targets'],
                   leakcal=list(info['leakcal'])),
        bandpass_h5=existing(BANDPASS_H5(spw0)) if info['spws'][0]['nchan'] > 1 else None,
        leakage=leakage, xfsolve=xfsolve, rm={}, polcheck=polchk, flagging=flagging,
        spws=info['spws'], pairing=info['pairing'], transition=info['transition'],
        gaincurve=(prep or {}).get('gaincurve'),
        phasecal_flux_density_jy={n: v['flux_density_jy'] for n, v in (pcal or {}).get('phasecals', {}).items()},
        fluxcal_amplitude_ratios=(fcal or {}).get('amplitude_ratios'))
    with open('misc/vlacal_report.json', 'w') as f:
        json.dump(report, f, indent=1, default=str)


# field role options of --vlacal: a restart with other values redoes the discovery
ROLE_OVERRIDES = ('vlacal_fluxcal', 'vlacal_phasecal', 'vlacal_polcal', 'vlacal_leakcal', 'vlacal_targets', 'vlacal_band')


def run_vlacal(mslist, args):
    """
    Run the historical VLA calibrator-based calibration (1GC) on a CASA importvla MS (all fields and
    spectral windows).

    Parameters
    ----------
    mslist : list of str
        The importvla Measurement Set (one MS).
    args : dict
        facetselfcal options.
    """
    if len(mslist) != 1:
        raise ValueError('--vlacal expects one importvla MS with all fields and spectral windows')
    ms = os.path.abspath(mslist[0].rstrip('/'))
    for d in ('misc', 'cal', 'h5_solutions', 'plots'):
        os.makedirs(d, exist_ok=True)
    stopafter = args['vlacal_stopafter']
    if stopafter is not None and stopafter not in STAGES:
        raise ValueError(f'--vlacal-stopafter should be one of {STAGES}')
    overrides = {k: args[k] for k in ROLE_OVERRIDES}
    if _done('discover'):
        info = json.load(open('misc/vlacal_discovery.json'))
        if info.get('overrides', overrides) != overrides:
            # a restart with other field roles: the targets may change (calibrated in the targets stage);
            # other calibrators would invalidate the finished calibrator stages
            new = discover(ms, args)
            changed = [r for r in new['roles'] if r != 'targets' and new['roles'][r] != info['roles'][r]]
            if changed:
                raise ValueError('vlacal: the calibrator roles (' + ', '.join(changed) + ') differ from those of the '
                                 'finished discover stage; start in a new directory to change calibrators')
            terminal_print('vlacal: targets now ' + ', '.join(new['roles']['targets']))
            info = new
            info['overrides'] = overrides
            with open('misc/vlacal_discovery.json', 'w') as fp:
                json.dump(info, fp, indent=1)
        if info.get('level_requested') != args['vlacal_level']:
            # a restart with another --vlacal-level: only the level and leakage calibrators change
            info['level'], info['leakcal'] = choose_level(info['roles'], info['fields'], info['geometry'],
                                                          info['spws'], args)
            info['level_requested'] = args['vlacal_level']
            terminal_print(f"vlacal: calibration level now {info['level']} (--vlacal-level {args['vlacal_level']})")
            with open('misc/vlacal_discovery.json', 'w') as fp:
                json.dump(info, fp, indent=1)
    else:
        info = discover(ms, args)
        info['overrides'] = overrides
        with open('misc/vlacal_discovery.json', 'w') as fp:
            json.dump(info, fp, indent=1)
        _finish('discover', dict(level=info['level'], roles=info['roles']))
    if stopafter == 'discover':
        return
    # DP3 is linked against MKL: the container's OMP_NUM_THREADS (the core count) would start that many
    # threads inside each DP3 thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        _run_stages(info, args, stopafter)
    finally:
        write_report_json(info)


def _run_stages(info, args, stopafter):
    """The stages after discovery, see run_vlacal."""
    if not _done('prepare'):
        _finish('prepare', stage_prepare(info, args))
    setup_refant(info, args)
    if stopafter == 'prepare':
        return
    if not _done('fluxcal'):
        _finish('fluxcal', stage_fluxcal(info, args))
    if stopafter == 'fluxcal':
        return
    if info['roles']['phasecal']:
        if not _done('phasecal'):
            _finish('phasecal', stage_phasecal(info, args))
    else:
        _warn('no phase calibrator: only the flux calibrator gains are transferred to the targets, the '
              'time-variable gains are left to the facetselfcal self-calibration')
    if stopafter == 'phasecal':
        return
    if info['level'] in ('leakage', 'full') and not _done('leakage'):
        _finish('leakage', stage_leakage(info, args))
    if stopafter == 'leakage':
        return
    if info['level'] == 'full' and not _done('polangle'):
        _finish('polangle', stage_polangle(info, args))
    if stopafter == 'polangle':
        return
    if info['level'] == 'full' and not _done('polcheck'):
        try:
            _finish('polcheck', stage_polcheck(info, args))
        except Exception as exc:  # noqa: BLE001  a failed check must not stop the calibration of the targets
            _warn(f'the polarisation calibrator check failed: {exc}')
    if stopafter == 'polcheck':
        return
    if not _done('targets'):
        _finish('targets', stage_targets(info, args))
    terminal_print('vlacal finished, calibrated target MSs (sky frame): '
                   + ' '.join(_summary('targets')['outputs']))
