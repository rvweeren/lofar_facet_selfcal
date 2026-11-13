"""Preparation of historical VLA data (CASA importvla MS) for --vlacal: split per field, scan and spectral
window, quack the start of the scans, flag zeros, shadowing and (EVLA-transition data) the VLA-EVLA
baselines, and apply the VLA gain curve. Also a robust residual clipper for data with few channels, where
AOFlagger has no frequency axis to work with."""
import os
import shutil

import numpy as np
from casacore.tables import makecoldesc, table, taql  # type: ignore

from submods.applycal import applycal_dp3_steps
from submods.meerkat_cal.solve import nthreads
from submods.ms_fixes import fix_twopol_ms, is_two_pol_ms
from submods.runtime_utils import run, terminal_print
from submods.vla_cal.gaincurve import write_gaincurve_h5

SUBMODPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# stages of --vlacal that keep a flag snapshot (meerkat_cal.prepare.snapshot_flags), in pipeline order
VLACAL_SNAPSHOT_STAGES = ('fluxcal', 'phasecal', 'leakage', 'polangle')


def _select_rows(ms, field_id, scan, ddid, outname):
    """Copy the rows of one field, scan and DATA_DESC_ID into a new MS with a one-row FIELD table."""
    if os.path.isdir(outname):
        shutil.rmtree(outname)
    cmd = (f"taql 'select from {ms} where FIELD_ID=={field_id} && SCAN_NUMBER=={scan} && "
           f"DATA_DESC_ID=={ddid} giving {outname} as plain'")
    terminal_print('Command:', cmd)
    run(cmd, taql=True)
    taql("delete from {} where rownr() != {}".format(outname + '/FIELD', field_id))
    taql("update {} set SOURCE_ID=0".format(outname + '/FIELD'))
    taql("update {} set FIELD_ID=0".format(outname))


def _fourpol_weights(ms):
    """
    After fix_twopol_ms (DATA, FLAG and WEIGHT_SPECTRUM to four correlations): WEIGHT and SIGMA to four
    correlations as well (RL as RR, LR as LL, as fix_twopol_ms does for WEIGHT_SPECTRUM; importvla writes
    no WEIGHT_SPECTRUM, and DP3 crashes on two-element WEIGHT rows with four correlations), and the
    two-correlation MODEL_DATA and CORRECTED_DATA of importvla removed.
    """
    with table(ms, ack=False, readonly=False) as t:
        for col in ('WEIGHT', 'SIGMA'):
            v = t.getcol(col)
            if v.shape[-1] == 2:  # direct (fixed-shape) storage: replace the column
                desc = t.getcoldesc(col)
                desc['shape'] = [4]
                desc['ndim'] = 1
                desc.pop('dataManagerGroup', None)
                desc.pop('dataManagerType', None)
                t.removecols(col)
                t.addcols(makecoldesc(col, desc))
                t.putcol(col, v[:, [0, 0, 1, 1]])
        stale = [c for c in ('MODEL_DATA', 'CORRECTED_DATA') if c in t.colnames()
                 and t.getcell(c, 0).shape[-1] != 4]
        if stale:
            t.removecols(stale)


def quack_slots(ms, quack):
    """Number of time slots in the first quack seconds of an MS (0 for quack 0)."""
    if not quack:
        return 0
    with table(ms, ack=False) as t:
        interval = float(np.median(t.getcol('INTERVAL', 0, 10000)))
    return int(np.ceil(quack / interval - 1e-6))


def split_scan_spw(ms, field_id, scan, spw, outname, quack=10., flag_mixed=True, gaincurve=True,
                   gaincurve_h5=None):
    """
    One field, scan and spectral window of a VLA MS as a new single-spw MS (DATA, WEIGHT_SPECTRUM),
    with the start of the scan quacked, zeros flagged, the VLA-EVLA baselines flagged (if there are
    EVLA antennas and flag_mixed), shadowing flagged, and the VLA gain curve applied (DP3 applycal with
    updateweights, as CASA applycal with gaintable gc and calwt=True).

    Parameters
    ----------
    ms : str
        The importvla MS (all fields and spectral windows).
    field_id, scan : int
        FIELD_ID and SCAN_NUMBER.
    spw : dict
        Spectral window from discover_spws (spw, ddid, ...).
    outname : str
        Output MS.
    quack : float, optional
        Seconds flagged at the start of the scan. Default 10.
    flag_mixed : bool, optional
        Flag the VLA-EVLA baselines. Default True.
    gaincurve : bool, optional
        Apply the VLA gain curve. Default True.
    gaincurve_h5 : str or None, optional
        h5parm for the gain curve (default <outname>.gaincurve.h5, removed afterwards).

    Returns
    -------
    dict
        Gain curve summary (or None).
    """
    tmp = outname.rstrip('/') + '.select'
    _select_rows(ms, field_id, scan, spw['ddid'], tmp)
    twopol = is_two_pol_ms(tmp)
    if twopol:  # DP3 needs four correlations: RL = LR = RR - LL (fix_twopol_ms.py), on the copy only
        fix_twopol_ms([tmp])
        _fourpol_weights(tmp)
    with table(tmp + '::ANTENNA', ack=False) as t:
        names = list(t.getcol('NAME'))
    args, steps = '', []
    nquack = quack_slots(tmp, quack)
    if nquack > 0:
        args += f'pfq.type=preflagger pfq.timeslot=[0..{nquack - 1}] '
        steps.append('pfq')
    args += 'pfz.type=preflagger pfz.amplmin=1e-30 '  # exactly zero visibilities
    steps.append('pfz')
    if flag_mixed and any(n.upper().startswith('EA') for n in names) and any(n.upper().startswith('VA') for n in names):
        args += 'pfm.type=preflagger pfm.baseline=VA*&EA* '
        steps.append('pfm')
    gcsummary = None
    if gaincurve:
        gch5 = gaincurve_h5 or outname.rstrip('/') + '.gaincurve.h5'
        # DP3 reads one spectral window (msin.band): the gain curve h5 gets that window's frequencies
        with table(tmp + '::SPECTRAL_WINDOW', ack=False) as t:
            freqs = t.getcell('CHAN_FREQ', spw['spw'])
        gcsummary = write_gaincurve_h5(tmp, gch5, freqs=freqs)
        acargs, nac = applycal_dp3_steps(tmp, [gch5], missingantennabehavior='flag', interpolation='linear',
                                         updateweights=True)
        args += acargs
        steps += ['ac%d' % i for i in range(nac)]
    if os.path.isdir(outname):
        shutil.rmtree(outname)
    cmd = (f'DP3 numthreads={nthreads()} msin={tmp} msin.band={spw["ddid"]} msin.datacolumn=DATA msout={outname} '
           'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False '
           + args + 'steps=[' + ','.join(steps) + ']')
    terminal_print('Split scan:', cmd)
    run(cmd, log=True)
    shutil.rmtree(tmp)
    if gaincurve and gaincurve_h5 is None:
        os.remove(gch5)
    if twopol:
        with table(outname, ack=False, readonly=False) as t:
            t.putcolkeyword('DATA', 'FAKE_RLLR', True)
    run('python ' + SUBMODPATH + '/shadow_flag.py --ms=' + outname)
    return gcsummary


def refresh_fake_crosshands(ms, column='DATA', chunk=100000):
    """
    For data with only RR and LL (fake cross-hands, fix_twopol_ms): set RL = LR = RR - LL of a column
    and their flags to those of RR or LL, so that the Stokes QUV flagging (AOFlagger) sees Stokes V
    only. After a calibration the fake cross-hands are RR - LL of the raw data corrected with the R-L
    gains, which is not noise.
    """
    with table(ms, ack=False, readonly=False) as t:
        for start in range(0, t.nrows(), chunk):
            n = min(chunk, t.nrows() - start)
            d = t.getcol(column, start, n)
            f = t.getcol('FLAG', start, n)
            d[..., 1] = d[..., 2] = d[..., 0] - d[..., 3]
            f[..., 1] = f[..., 2] = f[..., 0] | f[..., 3]
            t.putcol(column, d, start, n)
            t.putcol('FLAG', f, start, n)


def _hollow_median(x, half):
    """Running median of x over the 2*half neighbours of each sample (the sample itself excluded),
    edges padded with the edge value. scipy.ndimage.median_filter does not exclude the centre of a
    hollow 1-D footprint."""
    padded = np.pad(x, half, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(np.delete(windows, half, axis=1), axis=1)


def _detrend(q, ok, window):
    """
    q (row, chan, nq) of one baseline in time order minus its running median along time over the
    unflagged samples (ok, row x chan), per channel and quantity; flagged samples get the running
    median of the nearest unflagged ones. NaN where a channel has fewer than 10 unflagged samples.
    The median is over the window without the sample itself: with it, the sample is often its own
    median (deviation 0), which biases the robust sigma low (window 3 clipped 68% of 3C286).
    """
    dev = np.full(q.shape, np.nan, dtype=q.dtype)
    idx = np.arange(q.shape[0])
    half = max(1, int(window) // 2)
    for c in range(q.shape[1]):
        good = ok[:, c]
        if good.sum() < 10:
            continue
        for k in range(q.shape[2]):
            g = q[good, c, k]
            smooth = _hollow_median(g.real, half) + 1j * _hollow_median(g.imag, half)
            trend = (np.interp(idx, idx[good], smooth.real) + 1j * np.interp(idx, idx[good], smooth.imag))
            dev[:, c, k] = q[:, c, k] - trend
    return dev


def _robust_sigma(values):
    """Robust standard deviation (1.4826 MAD) of the real and imaginary parts of complex values."""
    parts = np.concatenate([values.real, values.imag])
    parts = parts[np.isfinite(parts)]
    if parts.size < 10:
        return np.nan
    return 1.4826 * np.median(np.abs(parts - np.median(parts)))


def clip_residuals(ms, datacolumn='CORRECTED_DATA', modelcolumn=None, nsigma=5., niter=2, mode='calibrator',
                   window=9, crosshands=True):
    """
    Flag outliers along time per baseline, robustly, for data with few channels.

    The quantities are Stokes V (RR - LL) and the cross-hands RL, LR (Q, U) of the residual: data - model
    for a calibrator (mode='calibrator'), the data for a target (mode='target'), with a running median
    along time subtracted per baseline and channel: a leakage or polarisation offset in the cross-hands
    does not flag, nor does its slow change over a long scan (the polarised term rotates with the
    parallactic angle: with a constant median per baseline the 5 sigma cut removed the ends of a 34 min
    3C286 scan, 16% of the data), nor a slow R-L gain drift in V. As for the
    AOFlagger Stokes QUV strategy, errors of the Stokes I model and gain drifts common to R and L cancel
    (on a bright calibrator they dominate the RR and LL residuals), while RFI, mostly polarised, remains.

    A sample (time slot, channel) is flagged in all correlations when any of the quantities is further
    than nsigma robust standard deviations from its baseline median.

    Parameters
    ----------
    ms : str
        Measurement Set.
    datacolumn : str, optional
        Calibrated data column. Default CORRECTED_DATA.
    modelcolumn : str or None, optional
        Model column (mode='calibrator'). Default None (no model subtracted).
    nsigma : float, optional
        Threshold. Default 5.
    niter : int, optional
        Iterations. Default 2.
    mode : str, optional
        'calibrator' or 'target'. Default 'calibrator'.
    window : int, optional
        Length (unflagged time slots) of the running median. Default 9.
    crosshands : bool, optional
        Use the cross-hands. False for data with only RR and LL (fake cross-hands from fix_twopol_ms):
        only Stokes V is used. Default True.

    Returns
    -------
    float
        Fraction of the visibilities flagged by this step.
    """
    with table(ms, ack=False, readonly=False) as t:
        data = t.getcol(datacolumn)
        flag = t.getcol('FLAG')
        a1, a2 = t.getcol('ANTENNA1'), t.getcol('ANTENNA2')
        model = t.getcol(modelcolumn) if (modelcolumn and mode == 'calibrator') else 0.
        res = data - model
        if crosshands:
            quantities = np.stack([res[..., 0] - res[..., 3], res[..., 1], res[..., 2]], axis=-1)
        else:
            quantities = (res[..., 0] - res[..., 3])[..., None]
        before = flag.copy()
        anyflag = np.any(flag, axis=-1)
        baseline = a1 * (max(a1.max(), a2.max()) + 1) + a2
        for _ in range(niter):
            bad = np.zeros(anyflag.shape, dtype=bool)
            for bl in np.unique(baseline):
                rows = baseline == bl
                if a1[rows][0] == a2[rows][0]:
                    continue  # autocorrelations
                q = quantities[rows]  # row (time order), chan, nq
                ok = ~anyflag[rows]
                dev = _detrend(q, ok, window)
                for k in range(q.shape[-1]):
                    sigma = _robust_sigma(dev[..., k][ok])
                    if not np.isfinite(sigma) or sigma == 0:
                        continue
                    bad[rows] |= (np.abs(dev[..., k]) > nsigma * sigma) & ok
            if not bad.any():
                break
            anyflag |= bad
        flag[anyflag] = True
        t.putcol('FLAG', flag)
    newfrac = float(np.mean(flag & ~before))
    terminal_print(f'{ms}: clipped {100 * newfrac:.2f}% of the visibilities ({mode}, {nsigma} sigma)')
    return newfrac
