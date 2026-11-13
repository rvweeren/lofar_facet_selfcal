"""Correction of a time-variable gain common to all antennas, channels and polarisations, measured per dump
from the archive weights (--meerkatcal-weightgain).

The archive MS (katdal mvftoms) has WEIGHT_SPECTRUM = N / (A_i A_j), with A_i, A_j the autocorrelations, so the
weights follow the raw system gain. When the gain of the whole array changes during the observation (a
problem of the digitiser or correlator gain settings, not a sky or antenna effect), the visibilities, the
noise and 1/sqrt(weight) all change by the same factor. A dump-by-dump change like that is not followed by the
calibrator solves (per-scan amplitudes interpolated to the targets) and leaves amplitude jumps and ramps in the
target data. This module measures the change from the weights and removes it from the visibilities:

  G(t) = sqrt(w_ref / w(t))     data / G, weights * G^2

with w(t) the median over the unflagged cross-correlations of a dump of WEIGHT_SPECTRUM / R(nu), R(nu) the
channel shape of the weights (XX, YY) measured from dumps spread over the whole observation, and w_ref the
median of w(t) over those dumps. Since R and w_ref are the same for every field, the calibrator and target
scans get the same normalisation and the flux scale still comes from the bandpass calibrator.

The weights also follow the system temperature (elevation, a bright source in the beam): G is proportional to
gain * Tsys, so the corrected data scale as S / Tsys. The absolute flux scale is kept (the bandpass amplitudes
absorb G at the bandpass calibrator times and any constant factor), and the normalised phase calibrator
amplitudes remove slow Tsys differences. What is left is the Tsys difference between target and phase calibrator
(elevation, the calibrator's own S/SEFD), typically 1-2%. It should only be used when the data have a known
array-wide gain problem.
"""
import json
import os

import numpy as np
from casacore.tables import table  # type: ignore
from losoto.h5parm import h5parm
from scipy.ndimage import median_filter

from submods.runtime_utils import terminal_print

REFERENCE_JSON = 'misc/meerkatcal_weightgain.json'
WARNING = ('--meerkatcal-weightgain: the visibilities are divided by a gain measured per dump from the archive '
           'weights, the same for all antennas, channels and polarisations. This should normally not be used: the '
           'weights also follow the system temperature (elevation, bright sources in the beam), which this '
           'correction then puts into the data. Only use it for data with a known array-wide time-variable '
           'gain problem (target amplitudes jumping or ramping by the same factor on all baselines).')
NSAMPLE = 24  # dumps used for the channel shape and the reference level
CHANSTEP = 4  # every CHANSTEP-th channel is used for the per-dump level


def _cross_unflagged(ws, fl, a1, a2):
    """XX and YY weights of the cross-correlations with unflagged, positive weights set to NaN elsewhere."""
    x = a1 != a2
    w = ws[x][:, :, [0, 3]].astype(float)
    w[fl[x][:, :, [0, 3]] | ~(w > 0)] = np.nan
    return w


def make_reference(ms, nsample=NSAMPLE):
    """
    Channel shape R(nu) of the weights (XX, YY, median 1 over the band) and reference level w_ref, from nsample
    dumps spread evenly over the observation (all fields).

    Parameters
    ----------
    ms : str or list of str
        The archive Measurement Set, or the already split MSs of the observation (the dumps are spread
        over all of them together).
    nsample : int, optional
        Number of dumps. Default NSAMPLE.

    Returns
    -------
    dict
        shape (nchan x 2 list, NaN for channels flagged in all sampled dumps), wref, nchan.
    """
    mslist = [ms] if isinstance(ms, str) else list(ms)
    dumps = []  # (time, MS index, first row) of every dump of every MS
    alltimes = []
    for k, m in enumerate(mslist):
        with table(m, ack=False) as t:
            times = t.getcol('TIME')
        alltimes.append(times)
        utimes, first = np.unique(times, return_index=True)
        dumps += [(u, k, f) for u, f in zip(utimes, first)]
    dumps.sort()
    pick = np.unique(np.linspace(0, len(dumps) - 1, nsample).round().astype(int))
    shapes, chunks = [], []
    for i in pick:
        utime, k, r0 = dumps[i]
        times = alltimes[k]
        n = int(np.argmax(times[r0:] != utime)) or len(times) - r0  # rows of a dump are contiguous
        with table(mslist[k], ack=False) as t:
            w = _cross_unflagged(t.getcol('WEIGHT_SPECTRUM', r0, n), t.getcol('FLAG', r0, n),
                                 t.getcol('ANTENNA1', r0, n), t.getcol('ANTENNA2', r0, n))
        if np.isfinite(w).sum() < 1000:
            continue
        with np.errstate(all='ignore'):
            m = np.nanmedian(w, axis=0)  # per channel and pol
            shapes.append(m / np.nanmedian(m, axis=0))
        chunks.append(w)
    if not shapes:
        raise RuntimeError(f'--meerkatcal-weightgain: no unflagged weights found in {ms}')
    with np.errstate(all='ignore'):
        shape = np.nanmedian(np.array(shapes), axis=0)
        wref = float(np.nanmedian([np.nanmedian(w[:, ::CHANSTEP] / shape[::CHANSTEP]) for w in chunks]))
    return dict(shape=shape.tolist(), wref=wref, nchan=int(shape.shape[0]))


def load_or_make_reference(ms):
    """The reference of make_reference, made once and kept in REFERENCE_JSON (also the per-scan results)."""
    if os.path.isfile(REFERENCE_JSON):
        return json.load(open(REFERENCE_JSON))
    terminal_print('--meerkatcal-weightgain: measuring the channel shape and reference level of the archive weights')
    ref = make_reference(ms)
    ref['scans'] = {}
    with open(REFERENCE_JSON, 'w') as f:
        json.dump(ref, f)
    return ref


def dump_levels(ms, ref, chanstep=CHANSTEP):
    """
    Per-dump weight level w(t) of a scan MS (native channels, archive weights): median over the unflagged
    cross-correlations, XX and YY, every chanstep-th channel, of WEIGHT_SPECTRUM / R(nu).

    Returns
    -------
    (np.ndarray, np.ndarray)
        Times and levels (NaN for dumps without enough unflagged data).
    """
    shape = np.array(ref['shape'], dtype=float)
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
    if nchan != ref['nchan']:
        raise ValueError(f'{ms} has {nchan} channels, the archive weight reference {ref["nchan"]}')
    shape = shape[::chanstep]
    with table(ms, ack=False) as t:
        times = np.unique(t.getcol('TIME'))
        level = np.full(len(times), np.nan)
        for it in t.iter('TIME', sort=False):
            i = np.searchsorted(times, it.getcell('TIME', 0))
            ws = it.getcolslice('WEIGHT_SPECTRUM', [0, 0], [nchan - 1, 3], [chanstep, 1])
            fl = it.getcolslice('FLAG', [0, 0], [nchan - 1, 3], [chanstep, 1])
            w = _cross_unflagged(ws, fl, it.getcol('ANTENNA1'), it.getcol('ANTENNA2')) / shape
            if np.isfinite(w).sum() >= 1000:
                level[i] = np.nanmedian(w)
    return times, level


def gains_from_levels(level, wref):
    """
    G(t) = sqrt(wref / w(t)) of a scan. The last dump of a scan is often a partial integration (lower weight,
    unchanged signal) and gets the level of the dump before it; dumps without a level are interpolated and the
    levels are median filtered over 3 dumps (single-dump outliers).
    """
    level = np.array(level, dtype=float)
    if len(level) >= 3:
        level[-1] = np.nan
    ok = np.isfinite(level)
    if not ok.any():
        return np.ones(len(level))
    idx = np.arange(len(level))
    level = np.interp(idx, idx[ok], level[ok])
    if len(level) >= 3:
        level = median_filter(level, size=3, mode='nearest')
    return np.sqrt(wref / level)


def write_h5(ms, times, gains, h5out):
    """amplitude000 sqrt(G) on every antenna, both polarisations (DP3 applycal divides baseline ij by G)."""
    with table(ms + '::ANTENNA', ack=False) as t:
        names, pos = list(t.getcol('NAME')), t.getcol('POSITION')
    with table(ms + '::FIELD', ack=False) as t:
        direction = t.getcol('PHASE_DIR')[0, 0]
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        freqs = t.getcol('CHAN_FREQ')[0]
    vals = np.broadcast_to(np.sqrt(gains)[:, None, None, None, None], (len(times), 2, len(names), 1, 2)).copy()
    if os.path.exists(h5out):
        os.remove(h5out)
    H = h5parm(h5out, readonly=False)
    ss = H.makeSolset('sol000')
    ss.obj.antenna.append(list(zip(names, pos)))
    ss.obj.source.append([('pointing', list(direction))])
    ss.makeSoltab('amplitude', 'amplitude000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                  axesVals=[times, np.array([freqs.min(), freqs.max()]), names, ['pointing'], ['XX', 'YY']],
                  vals=vals, weights=np.ones_like(vals))
    H.close()


def scan_h5(ms, archive_ms, h5out):
    """
    Measure G(t) of a split scan MS (native resolution, archive weights) and write it to h5out.
    The range of G is kept per scan in REFERENCE_JSON.

    Parameters
    ----------
    ms : str
        Scan MS (from split_scan, before any calibration or averaging).
    archive_ms : str or list of str
        The archive MS, or all already split input MSs (for the reference, made on the first call).
    h5out : str
        Output h5parm.

    Returns
    -------
    str
        h5out.
    """
    ref = load_or_make_reference(archive_ms)
    times, level = dump_levels(ms, ref)
    gains = gains_from_levels(level, ref['wref'])
    write_h5(ms, times, gains, h5out)
    ref.setdefault('scans', {})[os.path.basename(ms)] = dict(gmin=float(gains.min()), gmax=float(gains.max()),
                                                           gmedian=float(np.median(gains)), h5=h5out)
    with open(REFERENCE_JSON, 'w') as f:
        json.dump(ref, f)
    terminal_print(f'{ms}: weight gain G {gains.min():.3f}-{gains.max():.3f} (median {np.median(gains):.3f}), '
                   'divided out')
    return h5out
