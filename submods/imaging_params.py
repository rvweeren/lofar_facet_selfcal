"""Imaging parameters: number of output channels, spectral polynomial order,
multiscale and uvmin updates, clean masks and thresholds, restoring beam,
and the high-dynamic-range detection."""
import os
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits
from casacore.tables import table  # type: ignore

from submods.facet_directions import MeerKAT_autodetect_highDR
from submods.fits_utils import flatten
from submods.image_utils import (findrms, get_image_dynamicrange,
                                 getlargestislandsize, makemask_extended)
from submods.ms_info import get_max_uv_lambda, get_w_rms_lambda
from submods.ms_operations import nested_mslistforimaging
from submods.runtime_utils import get_logger, run, terminal_print
from submods.telescopes import group_by_telescope, unique_telescopes


def round_up_to_even(number):
    """
    Round up to the nearest even number.

    Parameters
    ----------
    number : int or float
        Input number.

    Returns
    -------
    int
        Next even integer greater than or equal to `number`.
    """
    return int(np.ceil(number / 2.) * 2)


def set_channelsout(mslist, args, factor=1):
    """
    Determines the number of output channels (`channelsout`) for a list of measurement sets (MS) based on the telescope type and fractional bandwidth.

    Parameters
    ----------
    mslist : list of str
        List of paths to measurement sets (MS). The telescope is read from the MSs.
    args : dict
        Parsed facetselfcal options.
    factor : int or float, optional
        Multiplicative factor to adjust the number of output channels. Default is 1.

    Returns
    -------
    int
        The computed number of output channels, rounded up to the nearest even integer.

    Notes
    -----
    - For LOFAR and unknown telescopes, `channelsout` is calculated as `round_up_to_even(f_bw * 12 * factor)`.
    - For MeerKAT, `channelsout` is calculated as `round_up_to_even(f_bw * 13 * factor)`.
    - When the MSs come from several telescopes, the numbers for the MSs of each telescope are added.
    - The function assumes the existence of `get_fractional_bandwidth` and `round_up_to_even` helper functions.
    """
    groups = group_by_telescope(mslist)
    return sum(_channelsout_telescope(telescope, group, factor) for telescope, group in groups.items())


def _channelsout_telescope(telescope, mslist, factor=1):
    """
    Number of output channels for the MSs of one telescope (see set_channelsout).

    Parameters
    ----------
    telescope : str
        Telescope name.
    mslist : list of str
        The MSs of this telescope.
    factor : int or float, optional
        Multiplicative factor to adjust the number of output channels.

    Returns
    -------
    int
        Number of output channels.
    """
    # get median frequency of the first MS
    with table(mslist[0] + '::SPECTRAL_WINDOW', ack=False) as t:
        freq = np.median(t.getcol('REF_FREQUENCY'))

    f_bw = get_fractional_bandwidth(mslist)

    if telescope == 'LOFAR':
        channelsout = 6
        #channelsout = round_up_to_even(f_bw * 12 * factor)
    elif telescope == 'MeerKAT':
        if freq >= 1.7e9: # S-band:
            channelsout = 8 # use this for MeerKAT S-band
        else:
            channelsout = 12 # use this as default for MeerKAT L-band and UHF-band
    elif telescope == 'GMRT':
        channelsout = 8 # use this as default for GMRT
    elif telescope == 'ASKAP':
        channelsout = 8 # use this as default for ASKAP    
    else:
        channelsout = round_up_to_even(f_bw * 12 * factor)
    return channelsout


def update_fitspectralpol(args):
    """Update and return the FITS spectral polynomial order.

    ``args['fitspectralpol']`` is recomputed from ``channelsout`` when the
    ``update_fitspectralpol`` option is enabled.

    Parameters
    ----------
    args : dict
        Parsed facetselfcal options; ``channelsout``, ``fitspectralpol``,
        ``update_fitspectralpol`` are used and ``fitspectralpol`` is updated in
        place.

    Returns
    -------
    int or str
        Current spectral polynomial order, or its configured string value.
    """

    if args['update_fitspectralpol']:
        args['fitspectralpol'] = set_fitspectralpol(args['channelsout'])
    return args['fitspectralpol']


def mfs_imagename(selfcalcycle, mslist, args):
    """
    File name of the restored (MFS) image of a selfcal cycle.

    With --stack the image of the last stack is returned.

    Parameters
    ----------
    selfcalcycle : int
        Selfcal cycle of the image.
    mslist : list of str
        Measurement Sets of the run.
    args : dict
        Parsed facetselfcal options; ``channelsout``, ``imagename``,
        ``imager`` and ``stack`` are used.

    Returns
    -------
    str
        Image file name.
    """
    stackstr = ''
    if args['stack']:
        nstack = len(nested_mslistforimaging(mslist, stack=args['stack']))
        stackstr = '_stack' + str(nstack - 1).zfill(2)
    basename = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr
    if args['imager'] == 'DDFACET':
        return basename + '.app.restored.fits'
    imagename = basename + '-MFS-image.fits'
    # without channel images there is no MFS image; channelsout may also have been
    # changed after this image was made, so fall back on the file that exists
    if args['channelsout'] == 1 or not os.path.isfile(imagename):
        single = basename + '-image.fits'
        if args['channelsout'] == 1 or os.path.isfile(single):
            imagename = single
    return imagename


# minimum dynamic range (peak over clipped rms of the MFS image) for --auto-tapertukey,
# per telescope; telescopes not listed use AUTO_TAPERTUKEY_MIN_DR_DEFAULT. With several
# telescopes in the run the lowest threshold among them is used.
AUTO_TAPERTUKEY_MIN_DR = {'LOFAR': 50000., 'MeerKAT': 50000., 'GMRT': 50000.,
                          'VLA': 50000., 'EVLA': 50000., 'ASKAP': 50000., 'MWA': 50000.}
AUTO_TAPERTUKEY_MIN_DR_DEFAULT = 50000.
# fraction of --uvmaxim used as Tukey taper width, and the niter factor when it switches on
AUTO_TAPERTUKEY_WIDTH_FRACTION = 1. / 3.
AUTO_TAPERTUKEY_NITER_FACTOR = 2


def auto_tapertukey_blocker(args, user_set):
    """
    Reason why --auto-tapertukey cannot switch on for this run, or None.

    Parameters
    ----------
    args : dict
        Parsed facetselfcal options; ``imager``, ``robust`` and ``stack`` are used.
    user_set : bool
        True if --uvmaxim or --tapertukey was set by the user.

    Returns
    -------
    str or None
        Short reason, or None if the taper can be switched on.
    """
    if user_set:
        return '--uvmaxim or --tapertukey is set by hand'
    if args['stack']:
        return 'not available with --stack'
    if args['imager'] != 'WSCLEAN':
        return 'only available with WSClean'
    robust = args['robust']
    if isinstance(robust, str):
        if robust != 'uniform':
            return 'needs robust <= 0 (robust is ' + robust + ')'
    elif robust is not None and robust > 0.0:
        return 'needs robust <= 0 (robust is ' + str(robust) + ')'
    return None


def update_tapertukey(selfcalcycle, mslist, args, user_set):
    """
    Switch on an outer Tukey taper of the imaging weights when the dynamic range is high.

    Used with --auto-tapertukey. Once the dynamic range of the MFS image of
    ``selfcalcycle`` reaches the telescope threshold (AUTO_TAPERTUKEY_MIN_DR),
    ``uvmaxim`` is set to the longest uv distance in the data, ``tapertukey`` to
    a third of that, and ``niter`` is doubled. The taper then stays on for the rest
    of the run (``args['auto_tapertukey_active']``).

    Parameters
    ----------
    selfcalcycle : int
        Selfcal cycle of the image to check.
    mslist : list of str
        Measurement Sets of the run.
    args : dict
        Parsed facetselfcal options; ``uvmaxim``, ``tapertukey``, ``niter`` and
        ``auto_tapertukey_active`` are updated in place.
    user_set : bool
        True if --uvmaxim or --tapertukey was set by the user (the automatic
        taper is then never used).

    Returns
    -------
    bool
        True if the taper switched on in this call.
    """
    if not args['auto_tapertukey'] or args.get('auto_tapertukey_active', False):
        return False
    if auto_tapertukey_blocker(args, user_set) is not None:
        return False
    imagename = mfs_imagename(selfcalcycle, mslist, args)
    if not os.path.isfile(imagename):
        terminal_print('--auto-tapertukey: image ' + imagename + ' not found, taper not checked')
        return False
    dr = get_image_dynamicrange(imagename)
    min_dr = min(AUTO_TAPERTUKEY_MIN_DR.get(telescope, AUTO_TAPERTUKEY_MIN_DR_DEFAULT)
                 for telescope in unique_telescopes(mslist))
    if dr < min_dr:
        get_logger().info('--auto-tapertukey: DR %.0f below %.0f, no taper yet' % (dr, min_dr))
        return False

    uvmaxim = float(np.ceil(get_max_uv_lambda(mslist) / 100.) * 100.)  # round up, no data are cut
    tapertukey = float(np.round(uvmaxim * AUTO_TAPERTUKEY_WIDTH_FRACTION, 0))
    args['uvmaxim'] = uvmaxim
    args['tapertukey'] = tapertukey
    args['niter'] = int(args['niter'] * AUTO_TAPERTUKEY_NITER_FACTOR)
    args['auto_tapertukey_active'] = True
    msg = ('--auto-tapertukey: DR %.0f of %s is above %.0f, switching on the outer Tukey taper: '
           '--uvmaxim=%.0f --tapertukey=%.0f, --niter raised to %d'
           % (dr, imagename, min_dr, uvmaxim, tapertukey, args['niter']))
    terminal_print(msg)
    get_logger().info(msg)
    return True


# --auto-ddpsfgrid: allowed relative PSF error near the brightest source is (K/DR)^(1/3): PSF errors are
# corrected iteratively (CLEAN major cycles), so after ~3 corrections the error is eps^3
AUTO_DDPSFGRID_K = 5.
AUTO_DDPSFGRID_NMAX = 7


def update_ddpsfgrid(selfcalcycle, mslist, args, user_set, paralleldeconvolution_base):
    """
    Choose the WSClean -dd-psf-grid size N automatically (--auto-ddpsfgrid), after an imaging cycle.

    A point source at a distance dtheta from the centre of its PSF cell sees the uv coordinates
    shifted by w*dtheta (non-coplanar baselines), so within the main lobe (x ~ beam) its PSF differs
    by about eps = 2 pi w_rms dtheta beam. With the allowed error eps_tol = (K/DR)^(1/3), cells
    must be at most sqrt(2) * eps_tol / (2 pi w_rms beam) wide; N is the smallest odd number of
    such cells over the image (1 to AUTO_DDPSFGRID_NMAX). WSClean only uses direction-dependent
    PSFs with parallel deconvolution, so the sub-image size is set to at most imsize / N.

    Parameters
    ----------
    selfcalcycle : int
        Selfcal cycle of the image (its dynamic range is used).
    mslist : list of str
        Measurement Sets of the run.
    args : dict
        Parsed facetselfcal options; ``ddpsfgrid`` and ``paralleldeconvolution`` are updated.
    user_set : bool
        True if --ddpsfgrid was set by hand (then nothing is changed).
    paralleldeconvolution_base : int
        Parallel deconvolution size without the dd-psf-grid (restored when N = 1).

    Returns
    -------
    int or None
        The dd-psf-grid size (None for no grid).
    """
    if not args.get('auto_ddpsfgrid') or user_set or args['imager'] != 'WSCLEAN':
        return args['ddpsfgrid']
    imagename = mfs_imagename(selfcalcycle, mslist, args)
    if not os.path.isfile(imagename):
        return args['ddpsfgrid']
    dr = get_image_dynamicrange(imagename)
    header = fits.getheader(imagename)
    beam = np.radians(header['BMAJ'])
    if 'auto_ddpsfgrid_wrms' not in args:
        args['auto_ddpsfgrid_wrms'] = get_w_rms_lambda(mslist)
    w_rms = args['auto_ddpsfgrid_wrms']
    width = np.radians(args['imsize'] * args['pixelscale'] / 3600.)
    eps_tol = (AUTO_DDPSFGRID_K / dr) ** (1. / 3.)
    cell = np.sqrt(2.) * eps_tol / (2. * np.pi * max(w_rms, 1e-9) * beam)
    n = int(np.ceil(width / cell))
    n = n + 1 if n % 2 == 0 else n
    n = int(min(max(n, 1), AUTO_DDPSFGRID_NMAX))
    if n == 1:
        args['ddpsfgrid'] = None
        args['paralleldeconvolution'] = paralleldeconvolution_base
    else:
        args['ddpsfgrid'] = n
        maxsize = int(args['imsize'] // n)
        args['paralleldeconvolution'] = min(paralleldeconvolution_base, maxsize) if paralleldeconvolution_base > 0 else maxsize
    msg = ('--auto-ddpsfgrid: DR %.0f, w_rms %.0f lambda, beam %.1f arcsec, allowed PSF error %.3f -> cell %.2f deg '
           'for an image of %.2f deg: dd-psf-grid %s, parallel deconvolution %s'
           % (dr, w_rms, np.degrees(beam) * 3600, eps_tol, np.degrees(cell), np.degrees(width),
              '%dx%d' % (n, n) if n > 1 else 'off', args['paralleldeconvolution']))
    terminal_print(msg)
    get_logger().info(msg)
    return args['ddpsfgrid']


def update_channelsout(selfcalcycle, mslist, args):
    """
    Dynamically updates the 'channelsout' parameter in the global 'args' dictionary based on the image dynamic range and telescope type.

    Parameters
    ----------
    selfcalcycle : int
        The current self-calibration cycle number.
    mslist : list of str
        List of Measurement Set (MS) file paths.
    args : dict
        Parsed facetselfcal options; ``channelsout``, ``idg``, ``imagename``,
        ``imager``, ``stack``, ``update_channelsout`` are used
        and ``channelsout`` is updated in place.

    Returns
    -------
    int or float
        The updated value of 'channelsout' in the 'args' dictionary.

    Notes
    -----
    Behavior:
    - Checks if 'update_channelsout' is enabled in 'args'.
    - Determines the telescope(s) of the MSs in the list.
    - Constructs the image filename based on the imaging parameters in 'args'.
    - Computes the dynamic range of the image.
    - Adjusts 'channelsout' in 'args' according to the dynamic range thresholds and telescope type.
    - Returns the updated 'channelsout' value.
    """
    if args['update_channelsout']:
        dr = get_image_dynamicrange(mfs_imagename(selfcalcycle, mslist, args))

        # dynamic-range thresholds per telescope (the highest threshold reached sets the factor);
        # with several telescopes the largest factor of the telescopes in the run is used
        thresholds = {'LOFAR': [(1500, 2), (3000, 3), (6000, 4)],
                      'MeerKAT': [(30000, 1.5), (60000, 2), (90000, 3)]}
        factors = [factor for telescope in unique_telescopes(mslist)
                   for drlimit, factor in thresholds.get(telescope, []) if dr > drlimit]
        if len(factors) > 0:
            args['channelsout'] = set_channelsout(mslist, args, factor=max(factors))
    return args['channelsout']


def set_fitspectralpol(channelsout):
    """
    Determine the fitspectralpol value based on the number of output channels.

    Parameters
    ----------
    channelsout : int
        The number of output channels.

    Returns
    -------
    fitspectralpol : int
        The fitspectralpol value corresponding to the number of output channels.

    Raises
    ------
    Exception
    If the channelsout value is invalid.
    """
    if channelsout == 1:
        fitspectralpol = 1
    elif channelsout == 2:
        fitspectralpol = 2
    elif channelsout <= 6:
        fitspectralpol = 3
    elif channelsout <= 8:
        fitspectralpol = 5
    elif channelsout <= 10:
        fitspectralpol = 7
    elif channelsout > 10:
        fitspectralpol = 9
    else:
        terminal_print('channelsout', channelsout)
        raise Exception(f'channelsout has an invalid value: {channelsout}')
    return fitspectralpol


def get_fractional_bandwidth(mslist):
    """
    Compute fractional bandwidth of a list of MS.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.

    Returns
    -------
    float
        Fractional bandwidth.
    """
    freqaxis = []
    for ms in mslist:
        t = table(ms + '::SPECTRAL_WINDOW', readonly=True, ack=False)
        freq = t.getcol('CHAN_FREQ')[0]
        t.close()
        freqaxis.append(freq)
    freqaxis = np.hstack(freqaxis)
    f_bw = (np.max(freqaxis) - np.min(freqaxis)) / np.min(freqaxis)

    if f_bw == 0.0:  # single channel data
        t = table(mslist[0] + '::SPECTRAL_WINDOW', readonly=True, ack=False)  # just take the first and assume this is ok
        f_bw = t.getcol('TOTAL_BANDWIDTH')[0] / np.min(freqaxis)
        t.close()
    return f_bw


def calculate_restoringbeam(mslist, LBA):
    """
    Returns the restoring beam.

    Parameters
    ----------
    mslist : list
        currently unused.
    LBA : bool
        if data is LBA or not.

    Returns
    -------
    restoringbeam : float
        the restoring beam in arcsec.
    """
    if LBA:  # so we have LBA
        restoringbeam = 15.
    else:  # so we have HBA
        restoringbeam = 6.

    return restoringbeam


def write_RMsynthesis_weights(fitslist, outfile):
    """
    Write frequency weights for RM synthesis.

    Parameters
    ----------
    fitslist : list
        FITS image paths containing frequency metadata.
    outfile : str
        Output weights filename.

    Returns
    -------
    None
        Weights are written to disk.
    """
    rmslist = np.zeros(len(fitslist))

    for fits_id, fitsfile in enumerate(fitslist):
        hdu = flatten(fits.open(fitsfile, ignore_missing_end=True))
        rmslist[fits_id] = findrms(hdu.data)

    terminal_print('Image RMS values (scaled by 1e6):', rmslist * 1e6)
    rmslist = 1 / rmslist ** 2  # 1/variance
    rmslist = rmslist / np.max(rmslist)  # normalize to max 1

    with open(outfile, 'w') as f:
        for rms in rmslist:
            f.write(str(rms) + '\n')

    return


def getdeclinationms(ms):
    """
    Return approximate declination of the MS pointing center.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    float
        Approximate declination in degrees.
    """
    t = table(ms + '::FIELD', readonly=True, ack=False)
    direction = np.squeeze(t.getcol('PHASE_DIR'))
    t.close()
    return 360. * direction[1] / (2. * np.pi)


def declination_sensivity_factor(declination):
    """
    Compute sensitivity factor for LOFAR data, reduced by declination.

    Formula from G. Heald.

    Parameters
    ----------
    declination : float
        Declination in degrees.

    Returns
    -------
    float
        Sensitivity factor.
    """
    factor = 1. / (np.cos(2. * np.pi * (declination - 52.9) / 360.) ** 2)

    return factor


def is_binary(file_name):
    """
    Check if a file is binary or text-based.

    Parameters
    ----------
    file_name : str
        Path to the file to check.

    Returns
    -------
    bool
        True if the file is binary, False if it is text.
    """
    try:
        import magic
    except ImportError:
        return False  # Assume non-binary if magic is not available

    mime = magic.Magic(mime=True).from_file(file_name)
    return 'text' not in mime


def makemaskthresholdlist(maskthresholdlist, stop):
    """
    Create the sequence of imaging mask thresholds.

    Parameters
    ----------
    maskthresholdlist : list
        Initial threshold values.
    stop : float
        Minimum threshold at which to stop.

    Returns
    -------
    list
        Completed threshold sequence.
    """
    maskthresholdselfcalcycle = []
    for mm in range(stop):
        try:
            maskthresholdselfcalcycle.append(maskthresholdlist[mm])
        except:
            maskthresholdselfcalcycle.append(maskthresholdlist[-1])  # add last value
    return maskthresholdselfcalcycle


def niter_from_imsize(imsize, paralleldeconvolution=-1):
    """
    Calculate the number of deconvolution iterations based on image size.
    This function determines the appropriate number of iterations (niter) for
    deconvolution based on the image size and parallel deconvolution settings.
    Parameters
    ----------
    imsize : int or None
        The size of the image in pixels. Must be provided (not None).
    paralleldeconvolution : int, optional
        The number of parallel deconvolution processes. Default is -1.
        If <= 0, niter is calculated based on imsize.
        If > 0, niter is calculated as 40 * paralleldeconvolution.
    Returns
    -------
    int
        The number of iterations to use for deconvolution.
        - If paralleldeconvolution <= 0 and imsize < 1024: returns 15000
        - If paralleldeconvolution <= 0 and imsize >= 1024: returns 15000 * (imsize / 1024)
        - If paralleldeconvolution > 0: returns 40 * paralleldeconvolution
    Raises
    ------
    Exception
    If imsize is None.
    Examples
    --------
    >>> niter_from_imsize(512)
    15000
    >>> niter_from_imsize(2048)
    30000
    >>> niter_from_imsize(1024, paralleldeconvolution=10)
    400
    """

    if imsize is None:
        terminal_print('imsize not set')
        raise Exception('imsize not set')
    if paralleldeconvolution <= 0:
        if imsize < 1024:
            niter = 15000  # minimum  value
        else:
            niter = 15000 * int((float(imsize) / 1024.))
    else:
        niter = 40 * int(paralleldeconvolution)
    return niter


def multiscale_trigger(fitsmask, args):
    """
    Determines whether to enable multiscale cleaning based on the size of the largest island in a FITS mask.

    If the 'update_multiscale' argument is True and a FITS mask is provided, the function checks the size of the largest island
    (using `getlargestislandsize`). If this size exceeds 1000 pixels, multiscale cleaning is triggered by setting the `multiscale`
    flag to True. The function logs relevant information about the island size and the triggering of multiscale cleaning.

    Parameters
    ----------
    fitsmask : str or None
        Path to the optional FITS mask to analyze for island sizes.
    args : dict
        Parsed facetselfcal options; ``multiscale``, ``update_multiscale`` are
        used.

    Returns
    -------
    bool
        The value of the `multiscale` flag, possibly updated based on the FITS mask analysis.
    """
    # update multiscale cleaning setting if allowed/requested
    multiscale = args['multiscale']
    if args['update_multiscale'] and fitsmask is not None:
        terminal_print('Size largest island [pixels]:', getlargestislandsize(fitsmask))
        get_logger().info('Size largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
        if getlargestislandsize(fitsmask) > 1000:
            get_logger().info('Triggering multiscale clean')
            multiscale = True
    return multiscale


def update_uvmin(fitsmask, longbaseline, LBA, args, telescope=None):
    """
    Updates the 'uvmin' parameter in the global 'args' dictionary based on the properties of the provided FITS mask.

    If stacking is enabled in 'args', the function returns immediately without making changes.
    If 'longbaseline' is False, 'update_uvmin' is enabled in 'args', and a FITS mask is provided,
    the function checks the size of the largest island in the mask. If this size exceeds 1000 pixels,
    it sets 'uvmin' in 'args' to 750 (non-LBA), 350 (MWA) or 250 (LBA), indicating the presence of extended emission.
    Relevant information is printed and logged.

    Parameters
    ----------
    fitsmask : str or None
        FITS mask path to analyze for extended emission.
    longbaseline : bool
        Indicates whether long baselines are being used.
    LBA : bool
        Indicates whether the observation is in LBA mode.
    args : dict
        Parsed facetselfcal options; ``stack``, ``update_uvmin`` are used and
        ``uvmin`` is updated in place.
    telescope : str or None, optional
        Telescope name (None for a mixed-telescope run).

    Returns
    -------
    None
    """
    # update uvmin if allowed/requested
    if args['stack']:
        return
    if not longbaseline and args['update_uvmin'] and fitsmask is not None:
        if getlargestislandsize(fitsmask) > 1000:
            terminal_print('Size of largest island [pixels]:', getlargestislandsize(fitsmask))
            get_logger().info('Size of largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
            if telescope == 'MWA':
                terminal_print('Extended emission found, setting uvmin to 350 lambda')
                get_logger().info('Extended emission found, setting uvmin to 350 lambda')
                args['uvmin'] = 350
            elif not LBA:
                terminal_print('Extended emission found, setting uvmin to 750 klambda')
                get_logger().info('Extended emission found, setting uvmin to 750 klambda')
                args['uvmin'] = 750
            else:
                terminal_print('Extended emission found, setting uvmin to 250 klambda')
                get_logger().info('Extended emission found, setting uvmin to 250 klambda')
                args['uvmin'] = 250
    return


def update_fitsmask(fitsmask, maskthreshold_selfcalcycle, selfcalcycle, args, mslist, telescope, longbaseline):
    """
    Updates or generates FITS mask files for self-calibration imaging cycles.
    This function manages the creation and updating of FITS mask files used in radio interferometric imaging
    self-calibration cycles. It supports different imagers (WSCLEAN, DDFACET), optional stacking, and
    extended masking for specific telescopes (LOFAR, MeerKAT). The function can merge user-supplied DS9 region
    files and use the external toool breizorro for mask generation. It returns the updated mask
    filename, a list of mask filenames for each imaging set, and the last used image filename.

    Parameters
    ----------
    fitsmask : str or None
        Path to an existing FITS mask file, or None to generate a new mask.
    maskthreshold_selfcalcycle : dict
        Dictionary mapping selfcal cycle indices to mask threshold values.
    selfcalcycle : int
        Current self-calibration cycle index.
    args : dict
        Dictionary of imaging and calibration parameters, including imager type, stacking,
        mask options, image name, and more.
    mslist : list
        List of measurement sets to process.
    telescope : str
        Name of the telescope (e.g., 'LOFAR', 'MeerKAT').
    longbaseline : bool
        Whether long baselines are used (affects mask generation).

    Returns
    -------
    tuple
    fitsmask : str or None
        Path to the updated or generated FITS mask file, or None if not created.
    fitsmask_list : list
        List of FITS mask filenames (or None) for each imaging set.
    imagename : str
        The filename of the last processed image.
    """
    # MAKE MASK IF REQUESTED
    fitsmask_list = []
    for msim_id, mslistim in enumerate(nested_mslistforimaging(mslist, stack=args['stack'])):
        if args['stack']:
            stackstr = '_stack' + str(msim_id).zfill(2)
        else:
            stackstr = ''  # empty string

        # set imagename
        if args['imager'] == 'WSCLEAN':
            if args['idg']:
                imagename = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
            else:
                imagename = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
        if args['imager'] == 'DDFACET':
            imagename = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '.app.restored.fits'
        if args['channelsout'] == 1:  # strip MFS from name if no channels images present
            imagename = imagename.replace('-MFS', '').replace('-I', '')

        # check if we need/can do masking & mask
        if args['fitsmask'] is None:
            mask_mergelist = []
            if args['DS9cleanmaskregionfile'] is not None: 
                mask_mergelist.append(args['DS9cleanmaskregionfile'])
            if maskthreshold_selfcalcycle[selfcalcycle] > 0.0:
                if args['mask_extended'] and selfcalcycle >= args['mask_extended_start'] and \
                   args['imsize'] >= 1600 and (telescope == 'LOFAR' or telescope == 'MeerKAT') \
                   and not longbaseline:
                    if telescope == 'LOFAR':
                        makemask_extended(imagename, 'clean_masks/mask_extended.fits',kernel_size=39,rebin=8, threshold=7.5)
                    if telescope == 'MeerKAT':
                        makemask_extended(imagename,'clean_masks/mask_extended.fits',kernel_size=25,rebin=5, threshold=10.)   
                    mask_mergelist.append('clean_masks/mask_extended.fits')
                
                cmdm = 'breizorro --make-binary --fill-holes --threshold=' + str(maskthreshold_selfcalcycle[selfcalcycle]) + \
                       ' --restored-image=' + imagename + ' --boxsize=30 --outfile=' + 'clean_masks/' + os.path.basename(imagename) + '.mask.fits'
                if len(mask_mergelist) > 0:
                    cmdm += ' --merge=' + ','.join(map(str, mask_mergelist))

                if fitsmask is not None:
                    if os.path.isfile('clean_masks/' + os.path.basename(imagename) + '.mask.fits'):
                        Path('clean_masks/' + os.path.basename(imagename) + '.mask.fits').unlink(missing_ok=True)  # remove previous mask if it exists
                terminal_print('breizorro command:', cmdm)
                run(cmdm)
                
                # removed compressed version if it exists
                if os.path.isfile('clean_masks/' + os.path.basename(imagename) + '.mask.fits.gz'):
                    Path('clean_masks/' + os.path.basename(imagename) + '.mask.fits.gz').unlink(missing_ok=True)
                terminal_print('Now gzip mask ' + 'clean_masks/' + os.path.basename(imagename) + '.mask.fits')
                subprocess.run(['gzip', 'clean_masks/' + os.path.basename(imagename) + '.mask.fits'], check=True)
                fitsmask = 'clean_masks/' + os.path.basename(imagename) + '.mask.fits.gz'
                fitsmask_list.append(fitsmask)
            else:
                fitsmask = None  # no masking requested as args['maskthreshold'] less/equal 0
                fitsmask_list.append(fitsmask)
        else:
            fitsmask_list.append(fitsmask)
    return fitsmask, fitsmask_list, imagename


def match_fitsmask_to_image(fitsmask, imsize, pixelscale):
    """
    Make the clean mask of the previous cycle fit the image of this cycle.

    A restart can use another --imsize or --pixelscale than the run that
    made the mask. If the new image lies within the old mask, the mask is
    regridded (nearest pixel) onto the new grid. If the new image extends
    beyond the old mask, the mask is not used: sources beyond the old image
    edge are not in it, so this cycle relies on auto-masking.

    Parameters
    ----------
    fitsmask : str
        Clean mask (FITS) of the previous cycle.
    imsize : int
        Image size in pixels of this cycle.
    pixelscale : float
        Pixel size in arcsec of this cycle.

    Returns
    -------
    str or None
        The mask itself if it already fits, the regridded mask, or None.
    """
    with fits.open(fitsmask) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data
    ny_old, nx_old = data.shape[-2:]
    scale_old = abs(header['CDELT2']) * 3600.
    imsize = int(imsize)
    if nx_old == imsize and ny_old == imsize and np.isclose(scale_old, pixelscale, rtol=1e-6):
        return fitsmask

    # WSClean puts the phase centre at pixel N//2 (0-based), the same sky
    # position in both grids, so the old pixel follows linearly from the new one
    half_new = (imsize / 2.) * pixelscale
    if half_new > min(nx_old, ny_old) / 2. * scale_old + 0.5 * scale_old:
        msg = ('Previous clean mask ' + fitsmask + ' (%d x %d pixels of %.3g arcsec) does not cover the new image '
               '(%d pixels of %.3g arcsec): not used, this cycle relies on auto-masking'
               % (nx_old, ny_old, scale_old, imsize, pixelscale))
        terminal_print(msg)
        get_logger().warning(msg)
        return None

    idx = np.arange(imsize)
    ratio = pixelscale / scale_old
    x_old = np.rint((idx - imsize // 2) * ratio + nx_old // 2).astype(int)
    y_old = np.rint((idx - imsize // 2) * ratio + ny_old // 2).astype(int)
    newdata = data[..., y_old[:, None], x_old[None, :]]

    header['NAXIS1'] = imsize
    header['NAXIS2'] = imsize
    header['CRPIX1'] = imsize // 2 + 1
    header['CRPIX2'] = imsize // 2 + 1
    header['CDELT1'] = np.sign(header['CDELT1']) * pixelscale / 3600.
    header['CDELT2'] = np.sign(header['CDELT2']) * pixelscale / 3600.
    newmask = fitsmask.replace('.mask.fits.gz', '') + '.mask.imsize%d.fits.gz' % imsize
    fits.writeto(newmask, newdata, header, overwrite=True)
    msg = ('Previous clean mask ' + fitsmask + ' (%d x %d pixels of %.3g arcsec) regridded to the new image '
           '(%d pixels of %.3g arcsec): ' % (nx_old, ny_old, scale_old, imsize, pixelscale) + newmask)
    terminal_print(msg)
    get_logger().info(msg)
    return newmask


def set_fitsmask_restart(i, mslist, args):
    """
    Restore or prepare FITS mask state for a restart.

    Parameters
    ----------
    i : int
        Self-calibration cycle index.
    mslist : list
        Measurement Set paths.
    args : dict
        Parsed facetselfcal options; ``channelsout``, ``idg``, ``imagename``,
        ``imager``, ``imsize``, ``pixelscale``, ``stack`` are used.

    Returns
    -------
    tuple
    fitsmask : str or None
        Mask of the previous cycle for the last imaging set, or None if not found
        or if it does not cover the image of this cycle (see match_fitsmask_to_image).
    fitsmask_list : list
        Mask filenames (or None) for each imaging set.
    """
    fitsmask_list = []
    for msim_id, mslistim in enumerate(nested_mslistforimaging(mslist, stack=args['stack'])):
        if args['stack']:
            stackstr = '_stack' + str(msim_id).zfill(2)
        else:
            stackstr = ''  # empty string

        # same naming as used when the mask was made in update_fitsmask()
        if args['imager'] == 'DDFACET':
            imagename = os.path.basename(args['imagename']) + str(i - 1).zfill(3) + stackstr + '.app.restored.fits'
        else:
            imagename = os.path.basename(args['imagename']) + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits'
        if args['channelsout'] == 1:  # no MFS in name if no channel images present
            imagename = imagename.replace('-MFS', '').replace('-I', '')
        fitsmask = 'clean_masks/' + imagename + '.mask.fits.gz'

        if not os.path.isfile(fitsmask):
            # no mask from the previous cycle (e.g. maskthreshold <= 0), continue without one
            terminal_print('Cannot find: ' + fitsmask)
            fitsmask = None
        else:
            fitsmask = match_fitsmask_to_image(fitsmask, args['imsize'], args['pixelscale'])
        terminal_print('Appending fitsmask: ', fitsmask)
        fitsmask_list.append(fitsmask)
    return fitsmask, fitsmask_list


def autodetect_highDR(selfcalcycle, mslist, telescope, soltypecycles_list, solint_list, smoothnessconstraint_list,
                      args):
    """
    Automatically detect if high dynamic range (DR) settings are needed for MeerKAT data and update calibration cycles accordingly.

    Parameters
    ----------
    selfcalcycle : int
        The current self-calibration cycle number.
    mslist : list
        List of measurement set file paths.
    telescope : str
        Name of the telescope (e.g., 'MeerKAT').
    soltypecycles_list : list
        Nested list of solution type cycles per MS.
    solint_list : list
        Nested list of solution intervals per MS.
    smoothnessconstraint_list : list
        Nested list of smoothness constraints per solution type and MS.
    args : dict
        Parsed facetselfcal options; ``DDE``, ``auto``, ``automask_threshold``,
        ``imagename``, ``maskthreshold``, ``start``, ``stop`` are used and
        ``stop`` is updated in place.

    Returns
    -------
    tuple
        Updated solution-cycle, solution-interval, smoothness-constraint,
        automask-threshold, and mask-threshold lists.
    """
    
    # get frequency
    t = table(mslist[0] + '::SPECTRAL_WINDOW', ack=False)
    freq = np.median(t.getcol('CHAN_FREQ')[0]) # in Hz
    t.close()
    
    if args['auto'] and not args['DDE'] and telescope == 'MeerKAT':
        highDR, peak_flux = MeerKAT_autodetect_highDR(args['imagename'] + str(selfcalcycle).zfill(3) + '-MFS-image.fits')
        if highDR:
            # update soltypecycles_list (this is a nested list of ms)
            get_logger().info('High dynamic range MeerKAT data detected, updating self-calibration settings accordingly...')
            get_logger().info(f'Peak flux: {peak_flux} Jy/beam at frequency: {freq/1e6} MHz')
            terminal_print('High dynamic range MeerKAT data detected, updating self-calibration settings accordingly...')
            terminal_print(f'Peak flux: {peak_flux} Jy/beam at frequency: {freq/1e6} MHz')
             # update soltypecycles_list (this is a nested list of ms)
            for ms_id, ms in enumerate(mslist):
                soltypecycles_list[1][ms_id] = 2 # set soltypecycles_list[1][ms_id] to 2 for high DR MeerKAT data
                soltypecycles_list[2][ms_id] = 3 # set soltypecycles_list[2][ms_id] to 3 for high DR MeerKAT data
                soltypecycles_list[3][ms_id] = 999 # turn off as auto_flag_antennas is already performed 
                if freq > 2e9:  # S band
                    solint_list[0][ms_id] = '32sec'
                elif freq > 1e9:  # L band
                    solint_list[0][ms_id] = '16sec'
                elif freq <= 1e9:  # UHF band
                    solint_list[0][ms_id] = '8sec'

                if peak_flux > 0.75*((freq/1e9)**(-0.7)):
                    terminal_print('Very high dynamic range MeerKAT data detected, updating self-calibration settings accordingly...')
                    get_logger().info('Very high dynamic range MeerKAT data detected, updating self-calibration settings accordingly...')
                    solint_list[2][ms_id] = '32sec' # for scalarcomplexgain2, very short time intervals to deal with scintillations
                    smoothnessconstraint_list[1][ms_id] = 2.0 # for scalarcomplexgain1, for fine-scale bandpass corrections

                if args['start'] == 0 and args['stop'] <= 4:
                    terminal_print(f"High dynamic range: number of self-calibration cycles (--stop) increased from {args['stop']} to 6")
                    get_logger().info(f"High dynamic range: --stop increased from {args['stop']} to 6")
                    args['stop'] = 6 # do some extra selfcal cycles for high DR MeerKAT data
                #this means we can do restarting runs as well in case which stop is decided by the user
    # update maskthreshold_selfcalcycle and automaskthreshold_selfcalcycle in case of extra cycles
    maskthreshold_selfcalcycle = makemaskthresholdlist(args['maskthreshold'], args['stop'] + 1) # add extra in case of an extract step 
    automaskthreshold_selfcalcycle = makemaskthresholdlist(args['automask_threshold'], args['stop'] + 1) # add one extra in case of an extract step
    
    # ensure automaskthreshold_selfcalcycle is always smaller than maskthreshold_selfcalcycle
    for selfcalcycle_id, mval in enumerate(maskthreshold_selfcalcycle):
        if automaskthreshold_selfcalcycle[selfcalcycle_id] > mval:
            automaskthreshold_selfcalcycle[selfcalcycle_id] = maskthreshold_selfcalcycle[selfcalcycle_id]

    return soltypecycles_list, solint_list, smoothnessconstraint_list, automaskthreshold_selfcalcycle, maskthreshold_selfcalcycle
