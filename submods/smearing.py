"""Time and bandwidth smearing calculations, and the related MS helpers
(maximum uv-distance, resolution, distance to the pointing centre)."""
import re

import astropy.units as units
import numpy as np
import scipy.special
from astropy.coordinates import SkyCoord
from casacore.tables import table  # type: ignore

from facetselfcal.utils.parsers import parse_history
from submods.runtime_utils import get_logger, terminal_print


def isfloat(num):
    """
    Check if a value can be converted to a float.

    Parameters
    ----------
    num : any
        Value to test.

    Returns
    -------
    bool
        True if `num` can be converted to a float, False otherwise.
    """
    return isinstance(num, float) or (isinstance(num, str) and num.replace('.', '', 1).isdigit())


def get_time_preavg_factor_LTAdata(ms):
    """
    Get time pre-averaging factor (given by demixer.timestep)

    Parameters
    ----------
    ms : str
        Path to the Measurement Set whose history is inspected.

    Returns
    -------
    int
        Time pre-averaging factor recorded in the Measurement Set history.
    """
    parse_str = "demixer.timestep="
    parsed_history = parse_history(ms, parse_str)
    avg_num = re.findall(r'\d+', parsed_history.replace(parse_str, ''))[0]
    if avg_num.isdigit():
        factor = int(float(avg_num))
        if factor != 1:
            terminal_print("WARNING: " + ms + " time has been pre-averaged with factor " + str(
                factor) + ". This might cause time smearing effects.")
        return factor
    elif isfloat(avg_num):
        factor = float(avg_num)
        terminal_print("WARNING: parsed factor in " + ms + " is not a digit but a float")
        return factor
    else:
        terminal_print("WARNING: parsed factor in " + ms + " is not a float or digit")
        return None


def compute_distance_to_pointingcenter(msname, HBAorLBA='HBA', warn=False, returnval=False, dologging=True):
    """
    Compute distance to the pointing center. This is mainly useful for international baseline observation to check of the delay calibrator is not too far away.

    Parameters
    ----------
    msname : str
        path to the measurement set to check.
    HBAorLBA : str
        whether the data is HBA or LBA data. Can be 'HBA' or 'LBA'.
    warn : bool, optional
        Emit a warning when the phase center is farther than the telescope threshold.
    returnval : bool, optional
        Return the separation in degrees instead of only reporting it.
    dologging : bool, optional
        Write the measured separation to the get_logger().

    Returns
    -------
    float or None
        Separation from the pointing center in degrees when ``returnval`` is true;
        otherwise, None.
    """
    if HBAorLBA == 'HBA':
        warn_distance = 1.25
    if HBAorLBA == 'LBA':
        warn_distance = 3.0
    if HBAorLBA == 'other':
        warn_distance = 3.0

    field_table = table(msname + '::FIELD', ack=False)
    direction = field_table.getcol('PHASE_DIR').squeeze()
    ref_direction = field_table.getcol('REFERENCE_DIR').squeeze()
    field_table.close()
    c1 = SkyCoord(direction[0] * units.radian, direction[1] * units.radian, frame='icrs')
    c2 = SkyCoord(ref_direction[0] * units.radian, ref_direction[1] * units.radian, frame='icrs')
    seperation = c1.separation(c2).to(units.deg)
    terminal_print('Distance to pointing center', seperation)
    if dologging:
        get_logger().info('Distance to pointing center:' + str(seperation))
    if (seperation.value > warn_distance) and warn:
        terminal_print(
            'Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
        get_logger().warning(
            'Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
    if returnval:
        return seperation.value
    return


def get_resolution(ms):
    """
    Estimate angular resolution from the longest baseline and frequency.

    Parameters
    ----------
    ms : str
        Measurement Set path.

    Returns
    -------
    float
        Estimated resolution in arcseconds.
    """
    uvmax = get_uvwmax(ms)
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        freq = np.median(t.getcol('CHAN_FREQ'))
        terminal_print('Central freq [MHz]', freq / 1e6, 'Longest baselines [km]', uvmax / 1e3)
    res = 1.22 * 3600. * 180. * ((299792458. / freq) / uvmax) / np.pi
    return res


def get_uvwmax(ms):
    """
    Find the maximum squared sum of UVW coordinates.

    Parameters
    ----------
    ms : str
        path to a Measurement Set.

    Returns
    -------
    None
    """
    with table(ms, ack=False) as t:
        uvw = t.getcol('UVW')
        ssq = np.sqrt(np.sum(uvw ** 2, axis=1))
        terminal_print('UVW array shape:', uvw.shape)
    return np.max(ssq)


def bandwidthsmearing(chanw, freq, imsize, verbose=True, ms_name=None):
    """
    Calculate the fractional intensity loss due to bandwidth smearing.

    Parameters
    ----------
    chanw : float
        bandwidth.
    freq : float
        observing frequency.
    imsize : int
        image size in pixels.
    verbose : bool
        print information to the screen.
    ms_name : str or None, optional
        Measurement Set name to include in log and print messages.

    Returns
    -------
    R : float
        fractional intensity loss.
    """
    R = (chanw / freq) * (imsize / 6.)  # asume we have used 3 pixels per beam
    if verbose:
        ms_context = f' in {ms_name}' if ms_name is not None else ''
        message = f'R value for bandwidth smearing{ms_context} is: {R}'
        terminal_print(message)
        get_logger().info(message)
        if R > 1.:
            warning_context = f' for {ms_name}' if ms_name is not None else ''
            warning = (
                f'Warning, bandwidth smearing{warning_context} exceeds R=1 (R={R:.1f}). Try '
                'to increase your frequency resolution, or lower imsize, '
                'to reduce the R value below 1'
            )
            terminal_print(warning)
            get_logger().warning(warning)
    return R


def smearing_time(r, th, t):
    """
    Returns the left over intensity I/I0 after time smearing.

    Parameters
    ----------
    r : float or Astropy Quantity
        distance from the phase center in arcsec.
    th : float or Astropy Quantity
        angular resolution in arcsec.
    t : float
        averaging time in seconds.

    Returns
    -------
    float or astropy.units.Quantity
        Fraction of intensity remaining after time smearing.
    """
    r = r + 1e-9  # Add a tiny offset to prevent division by zero.

    I = 1 - 1.22e-9 * (r / th) ** 2 * t ** 2
    return I


def smearing_time_ms(msin, t):
    """
    Calculate time-smearing loss for a Measurement Set.

    Parameters
    ----------
    msin : str
        Measurement Set path.
    t : float
        Averaging time in seconds.

    Returns
    -------
    float
        Estimated time-smearing factor.
    """
    res = get_resolution(msin)
    r_dis = 3600. * compute_distance_to_pointingcenter(msin, returnval=True, dologging=False)
    return smearing_time(r_dis, res, t)


def smearing_time_ms_imsize(msin, imsize, pixelscale):
    """
    Calculate time-smearing loss for an image size and pixel scale.

    Parameters
    ----------
    msin : str
        Measurement Set path.
    imsize : int
        Image size in pixels.
    pixelscale : float
        Pixel scale in arcseconds.

    Returns
    -------
    float
        Estimated time-smearing factor.
    """
    with table(msin, readonly=True, ack=False) as t:
        time = np.unique(t.getcol('TIME'))
        tint = np.abs(time[1] - time[0])
    res = get_resolution(msin)
    r_dis = np.sqrt(2)*(imsize/2)*pixelscale # image half diagonal length in arcsec
    return smearing_time(r_dis, res, tint)


def smearing_bandwidth(r, th, nu, dnu):
    """
    Returns the left over intensity I/I0 after bandwidth smearing.

    Parameters
    ----------
    r : float or Astropy Quantity
        distance from the phase center in arcsec.
    th : float or Astropy Quantity
        angular resolution in arcsec.
    nu : float
        observing frequency.
    dnu : float
        averaging frequency.

    Returns
    -------
    float or astropy.units.Quantity
        Fraction of intensity remaining after bandwidth smearing.
    """
    r = r + 1e-9  # Add a tiny offset to prevent division by zero.
    I = (np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))) * ((th * nu) / (r * dnu)) * scipy.special.erf(
        np.sqrt(np.log(2)) * ((r * dnu) / (th * nu)))
    return I


def report_bandwidth_smearing(mslist, imsize):
    """Compute and report bandwidth smearing for each Measurement Set."""
    imsize = float(imsize)
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            chan_width = np.median(t.getcol('CHAN_WIDTH'))
            min_frequency = np.min(t.getcol('CHAN_FREQ')[0])
        bandwidthsmearing(chan_width, min_frequency, imsize, ms_name=ms)
