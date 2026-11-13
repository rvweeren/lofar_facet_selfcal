"""Ionospheric Faraday rotation for the MeerKAT calibrator-based calibration: RM(t) towards a field from spinifex
(IONEX maps and a geomagnetic model) or from a user file (e.g. ALBUS), written to an h5parm with a
rotationmeasure000 soltab for DP3 applycal.

Sign convention: the polarisation angle is rotated by +RM lambda^2 (chi = chi_0 + RM lambda^2), which is
what DP3 applycal does with invert=False (corrupt) for a rotationmeasure soltab on IAU-frame linear data.
"""
import numpy as np
from casacore.tables import table, taql  # type: ignore
from losoto.h5parm import h5parm

from submods.runtime_utils import terminal_print

RM_TIMESTEP = 60.  # seconds between RM samples


def field_times(ms, field_id, timestep=RM_TIMESTEP):
    """
    Time grid (MJD seconds) covering the scans of a field, with samples every timestep and at the
    start and end of each scan.

    Parameters
    ----------
    ms : str
        Measurement Set.
    field_id : int
        FIELD_ID.
    timestep : float, optional
        Sampling (s). Default is RM_TIMESTEP.

    Returns
    -------
    np.ndarray
        Times (MJD seconds).
    """
    t = taql(f'select gmin(TIME) as t0, gmax(TIME) as t1 from {ms} where FIELD_ID=={field_id} '
             'groupby SCAN_NUMBER')
    times = []
    for t0, t1 in zip(t.getcol('t0'), t.getcol('t1')):
        times.extend(np.arange(t0, t1, timestep))
        times.append(t1)
    return np.unique(np.array(times))


def rm_spinifex(ms, direction, times, iono_model='ionex', **iono_kwargs):
    """
    Ionospheric RM towards a direction from the array centre with spinifex.

    Parameters
    ----------
    ms : str
        Measurement Set (for the antenna positions).
    direction : (float, float)
        RA, Dec (radians, J2000).
    times : np.ndarray
        Times (MJD seconds).
    iono_model : str, optional
        spinifex ionospheric model. Default 'ionex'.

    Returns
    -------
    (np.ndarray, np.ndarray)
        RM and its error (rad/m^2) at the times.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from spinifex.get_rm import get_rm_from_skycoord

    with table(ms + '::ANTENNA', ack=False) as t:
        pos = np.mean(t.getcol('POSITION'), axis=0)
    loc = EarthLocation.from_geocentric(*pos, unit=u.m)
    source = SkyCoord(direction[0] * u.rad, direction[1] * u.rad)
    rm = get_rm_from_skycoord(loc=loc, times=Time(times / 86400., format='mjd'), source=source,
                              iono_model_name=iono_model, **iono_kwargs)
    values = np.asarray(rm.rm, dtype=float).reshape(len(times), -1)
    errors = np.asarray(rm.rm_error, dtype=float).reshape(len(times), -1)
    # spinifex returns one value per time (or per time and pierce-point height for profile models)
    return values[:, 0], errors[:, 0]


def rm_from_file(rmfile, times):
    """
    RM(t) from a text file with two columns, time and RM (rad/m^2), interpolated to times. The time is
    in MJD seconds (as the MS TIME column) or, if all values are below 1e6, in MJD days. Lines
    starting with # are ignored.

    Parameters
    ----------
    rmfile : str
        File name.
    times : np.ndarray
        Times (MJD seconds).

    Returns
    -------
    np.ndarray
        RM (rad/m^2) at the times.
    """
    data = np.atleast_2d(np.loadtxt(rmfile, comments='#'))
    tfile, rmvals = data[:, 0], data[:, 1]
    if np.all(tfile < 1e6):
        tfile = tfile * 86400.
    order = np.argsort(tfile)
    tfile, rmvals = tfile[order], rmvals[order]
    if times.min() < tfile.min() - 600. or times.max() > tfile.max() + 600.:
        terminal_print(f'WARNING: {rmfile} does not cover the whole observation, the RM is extrapolated as constant')
    return np.interp(times, tfile, rmvals)


def write_rm_h5(h5out, ms, times, rm, direction, dirname='pointing'):
    """
    Write RM(t), the same for every antenna, to an h5parm (sol000/rotationmeasure000, axes time, ant, dir).

    Parameters
    ----------
    h5out : str
        Output h5parm (overwritten).
    ms : str
        Measurement Set (for the antenna names and positions).
    times : np.ndarray
        Times (MJD seconds).
    rm : np.ndarray
        RM (rad/m^2) per time.
    direction : (float, float)
        RA, Dec (radians).
    dirname : str, optional
        Direction name. Default 'pointing'.
    """
    import os
    if os.path.exists(h5out):
        os.remove(h5out)
    with table(ms + '::ANTENNA', ack=False) as t:
        names = t.getcol('NAME')
        pos = t.getcol('POSITION')
    H = h5parm(h5out, readonly=False)
    ss = H.makeSolset('sol000')
    ss.obj.antenna.append(list(zip(names, pos)))
    ss.obj.source.append([(dirname, list(direction))])
    vals = np.repeat(np.asarray(rm, dtype=float)[:, np.newaxis, np.newaxis], len(names), axis=1)
    ss.makeSoltab('rotationmeasure', 'rotationmeasure000', axesNames=['time', 'ant', 'dir'],
                  axesVals=[np.asarray(times, dtype=float), names, [dirname]], vals=vals,
                  weights=np.ones_like(vals))
    H.close()


def make_rm_h5(h5out, ms, field_id, direction, rmfile=None):
    """
    RM(t) h5parm towards a field, from rmfile if given, otherwise from spinifex.

    Parameters
    ----------
    h5out : str
        Output h5parm.
    ms : str or list of str
        Archive Measurement Set (for the scan times and antennas), or the MSs that contain the field
        (already split input; the times of all of them are used).
    field_id : int or list of int
        FIELD_ID of the field (per MS for a list of MSs).
    direction : (float, float)
        RA, Dec (radians).
    rmfile : str or None, optional
        RM(t) file, see rm_from_file.

    Returns
    -------
    dict
        Source, times and RM values (for the report).
    """
    if isinstance(ms, (list, tuple)):
        times = np.unique(np.concatenate([field_times(m, f) for m, f in zip(ms, field_id)]))
        ms, field_id = ms[0], field_id[0]
    else:
        times = field_times(ms, field_id)
    if rmfile is not None:
        rm, err, source = rm_from_file(rmfile, times), np.zeros(len(times)), rmfile
    else:
        rm, err = rm_spinifex(ms, direction, times)
        source = 'spinifex'
    write_rm_h5(h5out, ms, times, rm, direction)
    terminal_print(f'Ionospheric RM ({source}) towards field {field_id}: {np.min(rm):.3f} to {np.max(rm):.3f} rad/m^2 '
                   f'-> {h5out}')
    return dict(source=source, times=times.tolist(), rm=rm.tolist(), rm_error=err.tolist())
