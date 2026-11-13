"""VLA antenna gain curves (elevation dependence of the antenna gain) as an h5parm amplitude correction.

The curves are the CASA ones (casa-data nrao/VLA/GainCurves, exported by make_gaincurves.py to
facetselfcal/data/vla_gaincurves.json): the voltage gain of antenna i is g_i(za) = sum_n c_n za^n with za
the zenith angle in degrees, and CASA applycal divides a visibility by g_i g_j. Written as amplitude000
(pols XX, YY for the R and L receptors) and applied with DP3 applycal, this gives the same correction
(checked against CASA gencal(caltype='gc') + applycal on test data: agreement to 1.4e-4).
"""
import json
import os

import numpy as np
from casacore.measures import measures
import casacore.quanta as qa
from casacore.tables import table  # type: ignore
from losoto.h5parm import h5parm

from submods.runtime_utils import terminal_print

GAINCURVES_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               'facetselfcal', 'data', 'vla_gaincurves.json')


def antenna_number(name):
    """VLA antenna number from the name (VA01, EA27 -> 1, 27), None if the name has no number."""
    digits = ''.join(c for c in name if c.isdigit())
    return int(digits) if digits else None


def gaincurve_coefficients(freq, time, antenna_names, gaincurves=GAINCURVES_JSON):
    """
    Gain curve coefficients per antenna for a frequency and time, as CASA selects them: the rows of the
    band containing freq and valid at time, the antenna's own curve or else the default (antenna 0).

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    time : float
        Time (MJD seconds).
    antenna_names : list of str
        Antenna names (VA01, ...).
    gaincurves : str, optional
        The exported table (vla_gaincurves.json).

    Returns
    -------
    np.ndarray
        Coefficients (nant, 2 receptors, 4).
    """
    rows = [r for r in json.load(open(gaincurves))['rows']
            if r['bfreq'] <= freq <= r['efreq'] and r['btime'] <= time < r['etime']]
    if not rows:
        raise ValueError(f'No VLA gain curve for {freq / 1e9:.3f} GHz at MJD {time / 86400.:.1f}')
    per_ant = {r['antenna']: np.array(r['gain'], dtype=float) for r in rows}
    if 0 not in per_ant:
        raise ValueError(f'No default VLA gain curve for {freq / 1e9:.3f} GHz at MJD {time / 86400.:.1f}')
    return np.array([per_ant.get(antenna_number(n), per_ant[0]) for n in antenna_names])


def zenith_angles(times, direction, positions):
    """Zenith angle (deg) per time and antenna towards direction (RA, Dec in rad, J2000)."""
    dm = measures()
    field = dm.direction('J2000', qa.quantity(direction[0], 'rad'), qa.quantity(direction[1], 'rad'))
    za = np.zeros((len(times), len(positions)))
    frames = [dm.position('ITRF', *(qa.quantity(p, 'm') for p in pos)) for pos in positions]
    for it, time in enumerate(times):
        dm.do_frame(dm.epoch('UTC', qa.quantity(time, 's')))
        for ia, pos in enumerate(frames):
            dm.do_frame(pos)
            za[it, ia] = 90. - np.degrees(dm.measure(field, 'AZEL')['m1']['value'])
    return za


def write_gaincurve_h5(ms, h5out, freqs=None, step=30.):
    """
    Gain curve amplitudes for the field of a single-field MS (one per antenna, receptor and time sample
    every step seconds, interpolated linearly by DP3 applycal).

    Parameters
    ----------
    ms : str
        Measurement Set (one field).
    h5out : str
        Output h5parm (amplitude000, axes time, freq, ant, dir, pol XX, YY = R, L).
    freqs : array_like or None, optional
        Channel frequencies (Hz) of the h5parm. Default: those of the first spectral window of ms.
    step : float, optional
        Time sampling (s). Default 30.

    Returns
    -------
    dict
        Minimum and maximum gain correction per receptor (for the summary).
    """
    with table(ms, ack=False) as t:
        times = np.unique(t.getcol('TIME'))
    with table(ms + '::ANTENNA', ack=False) as t:
        names = list(t.getcol('NAME'))
        positions = t.getcol('POSITION')
    with table(ms + '::FIELD', ack=False) as t:
        direction = t.getcol('PHASE_DIR')[0, 0]
    if freqs is None:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            freqs = t.getcell('CHAN_FREQ', 0)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    # samples every step seconds plus the first and last time slot, so the interpolation covers the MS
    samples = np.unique(np.concatenate([np.arange(times[0], times[-1], step), [times[-1]]]))
    coeffs = gaincurve_coefficients(float(np.mean(freqs)), float(np.mean(times)), names)  # nant, 2, 4
    za = zenith_angles(samples, direction, positions)  # ntime, nant
    powers = za[:, :, None] ** np.arange(4)[None, None, :]  # ntime, nant, 4
    gain = np.einsum('tan,apn->tap', powers, coeffs)  # ntime, nant, 2
    vals = np.repeat(gain[:, None, :, None, :], len(freqs), axis=1)  # time, freq, ant, dir, pol
    if os.path.exists(h5out):
        os.remove(h5out)
    H = h5parm(h5out, readonly=False)
    ss = H.makeSolset('sol000')
    ss.obj.antenna.append(list(zip(names, positions)))
    ss.obj.source.append([('pointing', list(direction))])
    ss.makeSoltab('amplitude', 'amplitude000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                  axesVals=[samples, np.asarray(freqs, dtype=float), names, ['pointing'], ['XX', 'YY']],
                  vals=vals, weights=np.ones_like(vals))
    H.close()
    terminal_print(f'{h5out}: VLA gain curve, zenith angle {za.min():.0f}-{za.max():.0f} deg, gain '
                   f'{gain.min():.4f}-{gain.max():.4f}')
    return dict(za_deg=[float(za.min()), float(za.max())], gain=[float(gain.min()), float(gain.max())])
