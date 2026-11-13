"""Preparation of MeerKAT data for the calibrator-based calibration: split per field and scan, swap the feeds
(MeerKAT handedness, EVLA Memo 219), set RECEPTOR_ANGLE to 0, fix the UVW sign, and flag (known RFI
ranges, band edges, zeros, shadowing, AOFlagger)."""
import os
import shutil

import numpy as np
from casacore.tables import makecoldesc, maketabdesc, table, taql  # type: ignore

from submods.applycal import applycal_dp3_steps
from submods.flagging import (MEERKAT_RFIMASK_MAXBL, aoflagger_column, meerkat_rfi_ranges,
                              meerkat_static_rfimask)
from submods.ms_fixes import check_equidistant_times, fix_uvw
from submods.runtime_utils import run, terminal_print

# DP3 output storage of the meerkatcal calibrator MSs: Dysco for the data, the weights uncompressed.
# Dysco stores one weight for all correlations, which replaces the different XX/YY weights of MeerKAT
# data: that reweighting moved leakage solutions by up to 5e-3, the data compression alone by ~2e-6.
DYSCO_ARGS = 'msout.storagemanager=dysco msout.storagemanager.weightbitrate=0 '
# the target MSs: also the weights compressed, as facetselfcal writes its MSs (about half the size;
# for imaging the shared weight only costs a little S/N), unless --meerkatcal-target-exactweights
DYSCO_ARGS_TARGET = 'msout.storagemanager=dysco msout.storagemanager.weightbitrate=16 '

SUBMODPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def split_scan(ms, field_id, scan, outname, autocorr_only=False):
    """
    Copy one field and scan of a multi-field MS into a new MS, with a FIELD table of only that field
    (DP3 refuses a FIELD table with more than one row).

    A reference table (no data copy) cannot be used: its keywords are those of the input MS, so pointing
    its FIELD keyword to a one-row table changes the input MS.

    Parameters
    ----------
    ms : str
        Input Measurement Set.
    field_id : int
        FIELD_ID (row in the FIELD table).
    scan : int
        SCAN_NUMBER.
    outname : str
        Output MS.
    autocorr_only : bool, optional
        Copy only the autocorrelations (for the noise diode). Default False.
    """
    if os.path.isdir(outname):
        shutil.rmtree(outname)
    select = f'FIELD_ID=={field_id} && SCAN_NUMBER=={scan}' + (' && ANTENNA1==ANTENNA2' if autocorr_only else '')
    cmd = f"taql 'select from {ms} where {select} giving {outname} as plain'"
    terminal_print('Command:', cmd)
    run(cmd, taql=True)
    taql("delete from {} where rownr() != {}".format(outname + '/FIELD', field_id))
    taql("update {} set SOURCE_ID=0".format(outname + '/FIELD'))
    taql("update {} set FIELD_ID=0".format(outname))
    # one scan per MS, so DP3 never fills the gaps between scans with flagged time slots; missing dumps
    # inside a scan are filled (flagged, with DP3 UVWs of the opposite sign), which fix_uvw after every
    # DP3 run that writes a new MS corrects
    if not check_equidistant_times([outname], stop=False, return_result=True):
        terminal_print(f'WARNING: {outname} has missing time slots, DP3 inserts them as flagged data')


def preflag_parset(ms, band, edgefraction, include_hi, rfimask='sdp', prefix='pf'):
    """
    DP3 preflagger steps for the known RFI ranges of the band, the band edges, zero amplitudes and
    a static RFI mask on the short baselines.

    Parameters
    ----------
    ms : str
        Measurement Set (for the number of channels).
    band : str
        MeerKAT band, see meerkat_band.
    edgefraction : float
        Fraction of the channels flagged at each band edge.
    include_hi : bool
        Also flag Galactic HI (L-band).
    rfimask : str, optional
        Static RFI mask flagged on baselines up to MEERKAT_RFIMASK_MAXBL (not on autocorrelations):
        'sdp', 'caracal' or 'none', see meerkat_static_rfimask. Default 'sdp'.
    prefix : str, optional
        Step name prefix. Default 'pf'.

    Returns
    -------
    (str, list of str)
        DP3 arguments and step names.
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        freqs = t.getcol('CHAN_FREQ')[0] / 1e6
        widths = np.abs(t.getcol('CHAN_WIDTH')[0]) / 1e6
    nchan = len(freqs)
    fmin, fmax = float((freqs - widths / 2).min()), float((freqs + widths / 2).max())

    def in_band(rr):
        # a DP3 preflagger freqrange that matches no channel selects ALL channels (with blrange
        # that flags the whole band on those baselines), so only ranges that overlap the MS are used
        return [(lo, hi) for lo, hi in rr if hi > fmin and lo < fmax]

    nedge = int(round(edgefraction * nchan))
    args, steps = '', []
    ranges = in_band(meerkat_rfi_ranges(band, include_hi=include_hi))
    if ranges:
        args += f'{prefix}1.type=preflagger {prefix}1.freqrange=[' + ','.join(f'{lo}..{hi}MHz' for lo, hi in ranges) + '] '
        steps.append(prefix + '1')
    if nedge > 0:
        args += f'{prefix}2.type=preflagger {prefix}2.chan=[0..{nedge - 1},{nchan - nedge}..{nchan - 1}] '
        steps.append(prefix + '2')
    args += f'{prefix}3.type=preflagger {prefix}3.amplmin=1e-30 '  # exactly zero visibilities
    steps.append(prefix + '3')
    maskranges = in_band(meerkat_static_rfimask(rfimask, include_hi=include_hi))
    if maskranges:
        # the selections of one preflagger step are combined (AND): these frequencies on the
        # baselines from 1 m (no autocorrelations) to MEERKAT_RFIMASK_MAXBL
        args += (f'{prefix}4.type=preflagger {prefix}4.blrange=[1,{MEERKAT_RFIMASK_MAXBL}] {prefix}4.freqrange=['
                 + ','.join(f'{lo}..{hi}MHz' for lo, hi in maskranges) + '] ')
        steps.append(prefix + '4')
    return args, steps


def swap_parset(prefix='swap'):
    """DP3 Python step (polconv.py) that swaps the X and Y feeds."""
    return (f'{prefix}.type=PythonDPPP {prefix}.python.module=polconv {prefix}.python.class=PolConv '
            f'{prefix}.swapxy=1 '), [prefix]


def set_receptor_angle_zero(ms):
    """Set FEED RECEPTOR_ANGLE to 0 (after the feed swap the MS follows the IAU convention)."""
    with table(ms + '::FEED', ack=False, readonly=False) as t:
        t.putcol('RECEPTOR_ANGLE', np.zeros_like(t.getcol('RECEPTOR_ANGLE')))


def has_receptor_angle(ms):
    """Whether any FEED RECEPTOR_ANGLE is non-zero (i.e. the feeds were not swapped yet)."""
    with table(ms + '::FEED', ack=False) as t:
        return bool(np.any(t.getcol('RECEPTOR_ANGLE') != 0))


def prepare_calibrator_scan(ms, band, edgefraction=0.05, aoflagger_strategy=None, maxnchan=4096, rfimask='sdp',
                            gain_h5=None):
    """
    Prepare a calibrator scan MS in place: feed swap and preflagging (DP3), RECEPTOR_ANGLE 0, UVW
    sign, shadowing, AOFlagger on DATA (at the native resolution) and frequency averaging.

    The swap is only done if RECEPTOR_ANGLE is non-zero, and RECEPTOR_ANGLE is set to 0 right after,
    so a scan can never be swapped twice. A crash during the DP3 run leaves the MS partly swapped
    with RECEPTOR_ANGLE still non-zero: the caller must then split the scan again.

    Parameters
    ----------
    ms : str
        Calibrator scan MS (from split_scan).
    band : str
        MeerKAT band.
    edgefraction : float, optional
        Fraction of the band flagged at each edge. Default is 0.05.
    aoflagger_strategy : str or None, optional
        AOFlagger strategy for the raw data, None to skip (the default). AOFlagger on raw data of a bright
        calibrator flags the broadband dump-to-dump gain jumps (a few percent, larger than the noise),
        so by default the calibrators are flagged on the residuals after the first calibration
        (flag_residual) instead.
    maxnchan : int, optional
        Maximum number of channels after averaging, see average_frequency. Default 4096.
    rfimask : str, optional
        Static RFI mask for the short baselines, see preflag_parset. Default 'sdp'.
    gain_h5 : str or None, optional
        h5parm applied (data and weights) in the same DP3 run, the array-wide weight gain of
        --meerkatcal-weightgain (see weightgain.py). Default None.
    """
    pfargs, pfsteps = preflag_parset(ms, band, edgefraction, include_hi=True, rfimask=rfimask)
    steps = list(pfsteps)
    cmd = (f'DP3 msin={ms} msout=. msout.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM ' + pfargs)
    if gain_h5 is not None:
        acargs, nac = applycal_dp3_steps(ms, [gain_h5], updateweights=True)
        cmd += acargs
        steps = ['ac%d' % i for i in range(nac)] + steps
    swap = has_receptor_angle(ms)
    if swap:
        swapargs, swapsteps = swap_parset()
        cmd += swapargs
        steps = swapsteps + steps
    else:
        terminal_print(ms + ': RECEPTOR_ANGLE is 0, assuming the feeds were already swapped')
    cmd += 'steps=[' + ','.join(steps) + ']'
    terminal_print('Command:', cmd)
    run(cmd, log=True)
    if swap:
        set_receptor_angle_zero(ms)
    fix_uvw([ms])
    run('python ' + SUBMODPATH + '/shadow_flag.py --ms=' + ms)
    if aoflagger_strategy is not None:
        aoflagger_column([ms], aoflagger_strategy=aoflagger_strategy, column='DATA')
    average_frequency(ms, maxnchan)


def freq_factor(nchan, maxnchan):
    """
    Smallest integer averaging factor that gives at most maxnchan channels (1 for maxnchan 0). It need not
    divide nchan: read nchan_divisible(nchan, factor) channels (DP3 msin.nchan).

    The MeerKAT archive can remove the band-edge channels at download and average by a further factor,
    so the number of channels can be anything (3723 for a 4k L-band MS, 1861 after averaging by 2, a
    trimmed 32k MS), also a prime: a factor that divides nchan can then be far too large.
    """
    if not maxnchan or nchan <= maxnchan:
        return 1
    return int(np.ceil(nchan / maxnchan))


def nchan_divisible(nchan, freqstep):
    """
    Number of channels to read (DP3 msin.nchan) so that freqstep divides them: DP3 only averages whole
    channel groups, so the last nchan % freqstep (< freqstep) channels are left out. These are channels
    at the top band edge, which the band-edge pre-flagging flags anyway (e.g. 3723 channels of a MeerKAT
    L-band archive MS: 3722 read for freqstep 2).
    """
    return nchan - nchan % max(1, freqstep)


def time_factor(ms, timeres):
    """Integer time averaging factor closest to timeres (s) divided by the dump time, at least 1."""
    if not timeres:
        return 1
    with table(ms, ack=False) as t:
        interval = np.median(t.getcol('INTERVAL', 0, 10000))
    return max(1, int(round(timeres / interval)))


def average_frequency(ms, maxnchan):
    """
    Average an MS in place in frequency by the smallest integer factor that gives at most maxnchan
    channels (DP3 averager, flags and weights are handled by DP3), and store its data with Dysco
    (DYSCO_ARGS; also without averaging, if the data are not compressed yet).

    Parameters
    ----------
    ms : str
        Measurement Set.
    maxnchan : int
        Maximum number of channels, 0 or None to keep the MS as it is.

    Returns
    -------
    int
        The averaging factor (1 if the MS was not averaged).
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
    with table(ms, ack=False) as t:
        compressed = t.getdminfo('DATA')['TYPE'] == 'DyscoStMan'
    factor = freq_factor(nchan, maxnchan)
    if factor == 1 and compressed:
        return 1
    tmp = ms.rstrip('/') + '.avgtmp'
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    nread = nchan_divisible(nchan, factor)
    cmd = (f'DP3 msin={ms} msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM msout={tmp} '
           + (f'msin.nchan={nread} ' if nread != nchan else '')
           + f'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False {DYSCO_ARGS}'
           f'steps=[avg] avg.type=averager avg.freqstep={factor}')
    terminal_print('Command:', cmd)
    run(cmd, log=True)
    shutil.rmtree(ms)
    os.rename(tmp, ms)
    fix_uvw([ms])  # time slots inserted by DP3
    terminal_print(f'{ms}: averaged {nchan} -> {nread // factor} channels (Dysco)'
                   + (f', the top {nchan - nread} band-edge channels left out' if nread != nchan else ''))
    return factor


def flag_residual(ms, datacolumn='CORRECTED_DATA', modelcolumn='MODEL_DATA',
                  aoflagger_strategy='defaultMeerKAT_StokesQUV.lua'):
    """
    AOFlagger on the residual datacolumn - modelcolumn (written to RESIDUAL_DATA) of a calibrator.

    The default strategy flags on Stokes Q, U and V: the calibrator models leave the other sources in
    the wide UHF/L-band field out, and those (mostly unpolarised) sources dominate the Stokes I residual
    (on J1939-6342 at UHF the Stokes I strategy flagged 26% of the data, the QUV strategy 15%).

    Parameters
    ----------
    ms : str
        Measurement Set.
    datacolumn, modelcolumn : str, optional
        Calibrated data and model columns. Default CORRECTED_DATA and MODEL_DATA.
    aoflagger_strategy : str, optional
        AOFlagger strategy. Default 'defaultMeerKAT_StokesQUV.lua'.
    """
    with table(ms, ack=False, readonly=False) as t:
        if 'RESIDUAL_DATA' not in t.colnames():
            # its own (uncompressed) storage manager: a copy of the description of a Dysco DATA column
            # without dminfo makes casacore crash
            desc = makecoldesc('RESIDUAL_DATA', t.getcoldesc('DATA'))
            desc['desc'].pop('dataManagerType', None)
            desc['desc'].pop('dataManagerGroup', None)
            t.addcols(maketabdesc(desc), dminfo={'TYPE': 'StandardStMan', 'NAME': 'RESIDUAL_DATA_dm', 'SPEC': {}})
    taql(f'update {ms} set RESIDUAL_DATA = {datacolumn} - {modelcolumn}')
    aoflagger_column([ms], aoflagger_strategy=aoflagger_strategy, column='RESIDUAL_DATA')
    with table(ms, ack=False, readonly=False) as t:
        t.removecols('RESIDUAL_DATA')  # only needed for the flagging


# stages that keep a flag snapshot, in pipeline order
SNAPSHOT_STAGES = ('bandpass', 'leakage', 'phasecal', 'polangle')


def snapshot_flags(ms, stage, stages=SNAPSHOT_STAGES):
    """
    Keep the flags of an MS at the start of a stage in the column FLAG_<STAGE>, or restore them from it
    when the stage runs again. The applies and AOFlagger runs of a stage write FLAG in place, so without
    this a stage that is restarted (or that failed after flagging on bad solutions) would start from
    the flags of the earlier attempt. The snapshots of the later stages are removed, since they hold
    flags made after this stage.

    Parameters
    ----------
    ms : str
        Measurement Set.
    stage : str
        Stage name.
    stages : tuple of str, optional
        The stages with a snapshot, in pipeline order. Default SNAPSHOT_STAGES (meerkatcal).
    """
    col = 'FLAG_' + stage.upper()
    later = ['FLAG_' + s.upper() for s in stages[stages.index(stage) + 1:]]
    with table(ms, ack=False, readonly=False) as t:
        stale = [c for c in later if c in t.colnames()]
        if stale:
            t.removecols(stale)
        if col in t.colnames():
            terminal_print(f'{ms}: restoring the flags from the start of the {stage} stage ({col})')
            t.putcol('FLAG', t.getcol(col))
        else:
            t.addcols(makecoldesc(col, t.getcoldesc('FLAG')))
            t.putcol(col, t.getcol('FLAG'))


def flag_fraction(ms):
    """Fraction of flagged visibilities in an MS."""
    with table(ms, ack=False) as t:
        nrow = t.nrows()
        step = max(1, nrow // 2000)
        return float(np.mean(t.getcol('FLAG', rowincr=step)))
