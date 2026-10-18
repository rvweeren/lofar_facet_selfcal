"""Primary-beam correction of FITS images for MeerKAT, (E)VLA and uGMRT, and
writing primary-beam information to image headers."""
import glob

import astropy
import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from casacore.tables import table  # type: ignore

from submods.fits_utils import flatten
from submods.ms_fixes import get_telescope_from_ms
from submods.runtime_utils import get_logger, terminal_print


def write_primarybeam_info(cmd, imagebasename, telescope=None):
    """
    Updates the FITS header of images matching the given basename with primary beam correction information.

    This function searches for FITS files whose names start with `imagebasename` and end with '-pb.fits'.
    For each matching file, it updates the primary header's COMMENT field(s) to indicate whether a full
    primary beam correction has been applied, based on the provided command-line arguments.

    Parameters
    ----------
    cmd : str
        The command-line string used to run the imaging process. Determines which comments are added.
    imagebasename : str
        The base name of the image files to search for and update.
    telescope : str or None, optional
        Telescope name used for telescope-specific primary-beam metadata.

    Notes
    -----
    - If '-apply-facet-beam' or '-apply-primary-beam' is present in `cmd`, the header will indicate that
    the full primary beam correction has been applied.
    - If '-apply-facet-beam' is not present but '-apply-facet-solutions' is, the header will indicate that
    the primary beam correction has not been applied and manual correction may be necessary.
    """
    imagelist = glob.glob( imagebasename + '*image-pb.fits')
    for image in imagelist:
        get_logger().info('Updating FITS header primary beam correction info: ' + image)
        with fits.open(image, mode='update') as hdul:
            # Add a comment to the primary header
            if '-apply-facet-beam' in cmd:
                hdul[0].header['COMMENT'] = "Full primary beam correction applied. Image is science ready."
            if '-apply-facet-beam' not in cmd and '-apply-facet-solutions' in cmd:   
                hdul[0].header['COMMENT'] = "Full primary beam correction has not been applied." 
                if telescope is not None:
                    if telescope == 'MeerKAT' or telescope == 'GMRT' or telescope == 'VLA' or telescope == 'EVLA':
                        hdul[0].header['COMMENT'] = "You can use the <imagename>-MFS-image-manualpb.fits file."
                    else:
                        hdul[0].header['COMMENT'] = "Manually correct your image for the primary beam."        
                else:
                    hdul[0].header['COMMENT'] = "Manually correct your image for the primary beam."        
            if '-apply-primary-beam' in cmd:
                hdul[0].header['COMMENT'] = "Full primary beam correction applied. Image is science ready."


def MeerKAT_pbcor(fitsimage, outfile, freq=None, ms=None, pblimit=0.15): 
    """
    Apply a MeerKAT primary beam correction to a L-band FITS image.

    Parameters
    ----------
    fitsimage : str
        Path to the input FITS image that needs to be corrected.

    outfile : str
        Path where the primary-beam-corrected FITS image will be saved.

    freq : float, optional
        Observing frequency in GHz. If not provided, the function will attempt to
        read the frequency from the FITS header (CRVAL3 keyword, assumed to be in Hz).

    ms : str, optional
        Path to the Measurement Set (MS). If provided, the pointing center will be
        read from the 'REFERENCE_DIR' column of the MS. If not provided, the center 
        of the image will be assumed to be the pointing center.

    pblimit : float, optional
        Primary beam limit (in fraction of the peak) to apply the correction. Pixels with
        primary beam response below this limit will not be corrected. Default is 0.15 (zero means no limit).    

    Returns
    -------
    outfile : str
        Path to the corrected FITS image written to disk.

    Notes
    -----
    - This function applies a primary beam correction for MeerKAT L-band observations
    using the polynomial model published by T. Mauch et al. (2020, ApJ, 888, 61).
    - The correction is based on the distance from the pointing center and observing
    frequency, using a 10th-order polynomial model with coefficients:

    G1 = -0.3514e-3
    G2 =  0.5600e-7
    G3 = -0.0474e-10
    G4 =  0.00078e-13
    G5 =  0.00019e-16

    - For UHF and S-band frequencies, different coefficients are used as specified in the code.
    These were derived from fitting the katbeam model (average of 13 frequencies) with the given polynomial form.

    - The beam correction formula follows the AIPS PBCOR convention.

    References
    ----------
    - T. Mauch et al. 2020, "The 1.28 GHz MeerKAT DEEP2 Image," ApJ, 888, 61.
    - AIPS PBCOR: http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR

    Warnings
    --------
    - If the image center is not the actual pointing center and no MS is provided,
    the beam correction may be incorrectly applied. Ensure correct pointing 
    information is used when available.
    """


    if freq is None:
        hdul = fits.open(fitsimage)
        header = hdul[0].header
        freq = header['CRVAL3']/1e9
        terminal_print('Frequency found from the FITS image [GHz]:', freq)
        hdul.close()

    if (freq > 0.5) and (freq < 1.0):  # UHF-band
        G1 =-0.3409306414e-3
        G2 = 0.5002777932e-7
        G3 =-0.0406048877e-10
        G4 = 0.0017837276e-13
        G5 = -0.0000297565e-16
    elif (freq >= 1.0) and (freq < 1.7):  # L-band
        G1 =-0.3514e-3
        G2 = 0.5600e-7
        G3 =-0.0474e-10
        G4 = 0.00078e-13
        G5 = 0.00019e-16
    elif (freq >= 1.7) and (freq < 4.0):  # S-band
        G1 =-0.2829793167e-3
        G2 = 0.3462301721e-7
        G3 =-0.0237871692e-10
        G4 = 0.0009189657e-13
        G5 = -0.0000148312e-16
    else:
        raise ValueError('Frequency must be within a published MeerKAT band: 0.5-1.0, 1.0-1.7, or 1.7-4.0 GHz.')

    hdu = fits.open(fitsimage,  ignore_missing_end=True)
    hduflat = flatten(hdu)
    img = hduflat.data   
    terminal_print('IMAGE shape',img.shape)
    x, y = np.indices((img.shape))
  
    w = WCS(hduflat.header)
    if ms is not None:
        terminal_print('Taking pointing center from the ms')
        with table(ms + '::FIELD', readonly=True, ack=False) as t:
            ra_ref, dec_ref = t.getcol('REFERENCE_DIR').squeeze()
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')   
    else:
        center = w.pixel_to_world(img.shape[0]/2., img.shape[1]/2.)   
        terminal_print('Assume image center is the pointing center')
        terminal_print('Make sure this it correct, if not provide the ms to the function')
        terminal_print('CENTER image:',center)
       

    coordinates =  w.pixel_to_world(np.ravel(x),np.ravel(y))   
    sep = center.separation(coordinates)
    seprad = sep.arcmin # convert to arcmin
   
    separray = seprad.reshape(img.shape)
    X = freq*separray
    #separray = radius in arcminutes
    #freq = frequency in GHz.
    pb = 1. + G1*(X**2) + G2*(X**4) + G3*(X**6)  + G4*(X**8)  + G5*(X**10)

    hdu[0].data[0,0,:,:] = img/pb

    # Apply the primary beam limit by finding the first radial crossing.
    # This avoids holes if the polynomial becomes non-physical and rises again.
    if pblimit > 0.0:
        radial_distance = np.unique(np.sort(separray.ravel()))
        radial_pb = 1. + G1*((freq*radial_distance)**2) + G2*((freq*radial_distance)**4) + G3*((freq*radial_distance)**6) + G4*((freq*radial_distance)**8) + G5*((freq*radial_distance)**10)
        cutoff_idx = np.flatnonzero(radial_pb <= pblimit)

        if cutoff_idx.size > 0:
            cutoff_radius = radial_distance[cutoff_idx[0]]
            mask = separray >= cutoff_radius
            hdu[0].data[0,0,:,:][mask] = np.nan

    astropy.io.fits.writeto(outfile, hdu[0].data, hdu[0].header, overwrite=True)
    return outfile


# Classic VLA polynomial coefficients from AIPS PBCOR:
# F(x) = 1.0 + a * x / 10**3 + b * x**2 / 10**7 + c * x**3 / 10**10
# where x = (separation_arcmin * freq_GHz)**2
VLA_PB_COEFFS = {
    '4': (1.0, -0.897e-3, 2.71e-7, -0.242e-10),     # 0.0738 GHz
    'P': (1.0, -0.935e-3, 3.23e-7, -0.378e-10),     # 0.3275 GHz
    'L': (1.0, -1.343e-3, 6.579e-7, -1.186e-10),    # 1.465 GHz
    'C': (1.0, -1.372e-3, 6.940e-7, -1.309e-10),    # 4.885 GHz
    'X': (1.0, -1.306e-3, 6.253e-7, -1.100e-10),    # 8.435 GHz
    'U': (1.0, -1.305e-3, 6.155e-7, -1.030e-10),    # 14.965 GHz (Ku)
    'K': (1.0, -1.417e-3, 7.332e-7, -1.352e-10),    # 22.485 GHz
    'Q': (1.0, -1.321e-3, 6.185e-7, -0.983e-10),    # 43.315 GHz
}

# EVLA polynomial coefficients from EVLA Memo 195 (Perley 2016):
# P(r) = P0 + P2 * r^2 + P4 * r^4 + P6 * r^6 + P8 * r^8
# where r = separation_arcmin * freq_GHz, and frequency keys are in MHz
EVLA_PB_COEFFS = {
    # P-band (232 - 470 MHz)
    232: (1.0, -0.001137, 5.19e-07, -1.04e-10, 7.1e-15),
    246: (1.0, -0.00113, 5.04e-07, -1.02e-10, 7.7e-15),
    281: (1.0, -0.001106, 5.11e-07, -1.1e-10, 9.1e-15),
    296: (1.0, -0.001125, 5.27e-07, -1.14e-10, 9.6e-15),
    312: (1.0, -0.00103, 4.44e-07, -8.9e-11, 6.8e-15),
    328: (1.0, -0.00098, 4.25e-07, -8.7e-11, 6.9e-15),
    344: (1.0, -0.000974, 4.09e-07, -7.6e-11, 5.3e-15),
    357: (1.0, -0.000996, 4.23e-07, -7.9e-11, 5.1e-15),
    382: (1.0, -0.001002, 4.39e-07, -8.8e-11, 6.4e-15),
    392: (1.0, -0.001067, 5.13e-07, -1.12e-10, 9e-15),
    403: (1.0, -0.001057, 4.9e-07, -1.06e-10, 8.7e-15),
    421: (1.0, -0.001154, 5.85e-07, -1.33e-10, 1.08e-14),
    458: (1.0, -0.000993, 4.67e-07, -1.04e-10, 8.8e-15),
    470: (1.0, -0.00101, 4.85e-07, -1.07e-10, 8.6e-15),
    # L-band (1040 - 2000 MHz)
    1040: (1.0, -0.001529, 8.69e-07, -1.88e-10),
    1104: (1.0, -0.001486, 8.15e-07, -1.68e-10),
    1168: (1.0, -0.001439, 7.53e-07, -1.45e-10),
    1232: (1.0, -0.00145, 7.87e-07, -1.63e-10),
    1296: (1.0, -0.001428, 7.62e-07, -1.54e-10),
    1360: (1.0, -0.001449, 8.02e-07, -1.74e-10),
    1424: (1.0, -0.001462, 8.23e-07, -1.83e-10),
    1488: (1.0, -0.001455, 7.92e-07, -1.63e-10),
    1552: (1.0, -0.001435, 7.54e-07, -1.49e-10),
    1680: (1.0, -0.001443, 7.74e-07, -1.57e-10),
    1744: (1.0, -0.001462, 8.02e-07, -1.69e-10),
    1808: (1.0, -0.001488, 8.38e-07, -1.83e-10),
    1872: (1.0, -0.001486, 8.26e-07, -1.75e-10),
    1936: (1.0, -0.001459, 7.93e-07, -1.62e-10),
    2000: (1.0, -0.001508, 8.31e-07, -1.68e-10),
    # S-band (2052 - 3948 MHz)
    2052: (1.0, -0.001429, 7.52e-07, -1.47e-10),
    2180: (1.0, -0.001389, 7.06e-07, -1.33e-10),
    2436: (1.0, -0.001377, 6.9e-07, -1.27e-10),
    2564: (1.0, -0.001381, 6.92e-07, -1.26e-10),
    2692: (1.0, -0.001402, 7.23e-07, -1.4e-10),
    2820: (1.0, -0.001433, 7.62e-07, -1.54e-10),
    2948: (1.0, -0.001433, 7.46e-07, -1.42e-10),
    3052: (1.0, -0.001467, 8.05e-07, -1.7e-10),
    3180: (1.0, -0.001497, 8.38e-07, -1.8e-10),
    3308: (1.0, -0.001504, 8.37e-07, -1.77e-10),
    3436: (1.0, -0.001521, 8.63e-07, -1.88e-10),
    3564: (1.0, -0.001505, 8.37e-07, -1.75e-10),
    3692: (1.0, -0.001521, 8.51e-07, -1.79e-10),
    3820: (1.0, -0.001534, 8.57e-07, -1.77e-10),
    3948: (1.0, -0.001516, 8.3e-07, -1.66e-10),
    # C-band (4052 - 7948 MHz)
    4052: (1.0, -0.001406, 7.41e-07, -1.48e-10),
    4180: (1.0, -0.001385, 7.09e-07, -1.36e-10),
    4308: (1.0, -0.00138, 7.08e-07, -1.37e-10),
    4436: (1.0, -0.001362, 6.95e-07, -1.35e-10),
    4564: (1.0, -0.001365, 6.92e-07, -1.31e-10),
    4692: (1.0, -0.001339, 6.56e-07, -1.17e-10),
    4820: (1.0, -0.001371, 7.06e-07, -1.4e-10),
    4948: (1.0, -0.001358, 6.91e-07, -1.34e-10),
    5052: (1.0, -0.00136, 6.91e-07, -1.33e-10),
    5180: (1.0, -0.001353, 6.74e-07, -1.25e-10),
    5308: (1.0, -0.001359, 6.82e-07, -1.27e-10),
    5436: (1.0, -0.00138, 7.05e-07, -1.37e-10),
    5564: (1.0, -0.001376, 6.99e-07, -1.31e-10),
    5692: (1.0, -0.001405, 7.39e-07, -1.47e-10),
    5820: (1.0, -0.001394, 7.29e-07, -1.45e-10),
    5948: (1.0, -0.001428, 7.57e-07, -1.57e-10),
    6052: (1.0, -0.001445, 7.68e-07, -1.5e-10),
    6148: (1.0, -0.001422, 7.38e-07, -1.38e-10),
    6308: (1.0, -0.001463, 7.94e-07, -1.62e-10),
    6436: (1.0, -0.001478, 8.22e-07, -1.74e-10),
    6564: (1.0, -0.001473, 8e-07, -1.62e-10),
    6692: (1.0, -0.001455, 7.76e-07, -1.53e-10),
    6820: (1.0, -0.001487, 8.22e-07, -1.72e-10),
    6948: (1.0, -0.001472, 8.05e-07, -1.67e-10),
    7052: (1.0, -0.00147, 8.01e-07, -1.64e-10),
    7180: (1.0, -0.001503, 8.5e-07, -1.84e-10),
    7308: (1.0, -0.001482, 8.19e-07, -1.72e-10),
    7436: (1.0, -0.001498, 8.22e-07, -1.66e-10),
    7564: (1.0, -0.00149, 8.18e-07, -1.66e-10),
    7692: (1.0, -0.001481, 7.98e-07, -1.56e-10),
    7820: (1.0, -0.001474, 7.94e-07, -1.57e-10),
    7948: (1.0, -0.001448, 7.69e-07, -1.51e-10),
    # X-band (8052 - 11948 MHz)
    8052: (1.0, -0.001403, 7.21e-07, -1.37e-10),
    8180: (1.0, -0.001398, 7.1e-07, -1.32e-10),
    8308: (1.0, -0.001402, 7.16e-07, -1.35e-10),
    8436: (1.0, -0.0014, 7.12e-07, -1.32e-10),
    8564: (1.0, -0.001391, 6.95e-07, -1.25e-10),
    8692: (1.0, -0.001409, 7.34e-07, -1.49e-10),
    8820: (1.0, -0.00141, 7.36e-07, -1.45e-10),
    8948: (1.0, -0.00141, 7.34e-07, -1.43e-10),
    9052: (1.0, -0.001403, 7.2e-07, -1.36e-10),
    9180: (1.0, -0.001396, 7.09e-07, -1.31e-10),
    9308: (1.0, -0.001432, 7.68e-07, -1.55e-10),
    9436: (1.0, -0.001414, 7.43e-07, -1.47e-10),
    9564: (1.0, -0.001416, 7.45e-07, -1.47e-10),
    9692: (1.0, -0.001406, 7.26e-07, -1.39e-10),
    9820: (1.0, -0.001412, 7.36e-07, -1.43e-10),
    9948: (1.0, -0.001409, 7.29e-07, -1.39e-10),
    10052: (1.0, -0.001421, 7.46e-07, -1.45e-10),
    10180: (1.0, -0.001409, 7.25e-07, -1.36e-10),
    10308: (1.0, -0.001402, 7.13e-07, -1.31e-10),
    10436: (1.0, -0.001399, 7.09e-07, -1.29e-10),
    10564: (1.0, -0.001413, 7.37e-07, -1.43e-10),
    10692: (1.0, -0.001412, 7.34e-07, -1.41e-10),
    10820: (1.0, -0.001401, 7.12e-07, -1.31e-10),
    10948: (1.0, -0.001401, 7.12e-07, -1.31e-10),
    11052: (1.0, -0.001401, 7.12e-07, -1.31e-10),
    11180: (1.0, -0.001394, 6.99e-07, -1.24e-10),
    11308: (1.0, -0.001394, 7.01e-07, -1.26e-10),
    11436: (1.0, -0.001391, 6.94e-07, -1.22e-10),
    11564: (1.0, -0.001389, 6.92e-07, -1.22e-10),
    11692: (1.0, -0.001386, 6.8e-07, -1.15e-10),
    11820: (1.0, -0.001391, 6.88e-07, -1.19e-10),
    11948: (1.0, -0.001399, 6.97e-07, -1.22e-10),
    # Ku-band (12052 - 17948 MHz)
    12052: (1.0, -0.001399, 7.17e-07, -1.34e-10),
    12180: (1.0, -0.001392, 7.07e-07, -1.31e-10),
    12308: (1.0, -0.001393, 7.19e-07, -1.38e-10),
    12436: (1.0, -0.001393, 7.2e-07, -1.4e-10),
    12564: (1.0, -0.001395, 7.19e-07, -1.38e-10),
    12692: (1.0, -0.001397, 7.2e-07, -1.37e-10),
    12820: (1.0, -0.001388, 7.06e-07, -1.32e-10),
    12948: (1.0, -0.001397, 7.18e-07, -1.36e-10),
    13052: (1.0, -0.0014, 7.27e-07, -1.4e-10),
    13180: (1.0, -0.001406, 7.44e-07, -1.5e-10),
    13308: (1.0, -0.001403, 7.37e-07, -1.47e-10),
    13436: (1.0, -0.001392, 7.08e-07, -1.31e-10),
    13564: (1.0, -0.001384, 6.94e-07, -1.24e-10),
    13692: (1.0, -0.001382, 6.95e-07, -1.25e-10),
    13820: (1.0, -0.001376, 6.88e-07, -1.24e-10),
    13948: (1.0, -0.001384, 6.98e-07, -1.28e-10),
    14052: (1.0, -0.0014, 7.36e-07, -1.48e-10),
    14180: (1.0, -0.001397, 7.29e-07, -1.45e-10),
    14308: (1.0, -0.001399, 7.32e-07, -1.45e-10),
    14436: (1.0, -0.001396, 7.25e-07, -1.42e-10),
    14564: (1.0, -0.001393, 7.2e-07, -1.39e-10),
    14692: (1.0, -0.001384, 7.03e-07, -1.31e-10),
    14820: (1.0, -0.001388, 7.06e-07, -1.32e-10),
    14948: (1.0, -0.001393, 7.16e-07, -1.37e-10),
    15052: (1.0, -0.001402, 7.38e-07, -1.48e-10),
    15180: (1.0, -0.001407, 7.47e-07, -1.53e-10),
    15308: (1.0, -0.001406, 7.41e-07, -1.48e-10),
    15436: (1.0, -0.001399, 7.31e-07, -1.44e-10),
    15564: (1.0, -0.001397, 7.28e-07, -1.43e-10),
    15692: (1.0, -0.001401, 7.35e-07, -1.46e-10),
    15820: (1.0, -0.001402, 7.34e-07, -1.45e-10),
    15948: (1.0, -0.001399, 7.3e-07, -1.44e-10),
    16052: (1.0, -0.001419, 7.59e-07, -1.54e-10),
    16180: (1.0, -0.001419, 7.59e-07, -1.52e-10),
    16308: (1.0, -0.001412, 7.4e-07, -1.44e-10),
    16436: (1.0, -0.001407, 7.32e-07, -1.4e-10),
    16564: (1.0, -0.001408, 7.32e-07, -1.41e-10),
    16692: (1.0, -0.00141, 7.34e-07, -1.4e-10),
    16820: (1.0, -0.001407, 7.27e-07, -1.38e-10),
    16948: (1.0, -0.001423, 7.63e-07, -1.55e-10),
    17052: (1.0, -0.001437, 7.87e-07, -1.66e-10),
    17180: (1.0, -0.001438, 7.84e-07, -1.64e-10),
    17308: (1.0, -0.001445, 7.98e-07, -1.71e-10),
    17436: (1.0, -0.001452, 8.1e-07, -1.77e-10),
    17564: (1.0, -0.001458, 8.13e-07, -1.7e-10),
    17692: (1.0, -0.001456, 8.06e-07, -1.72e-10),
    17820: (1.0, -0.001453, 8e-07, -1.68e-10),
    17948: (1.0, -0.001452, 7.99e-07, -1.69e-10),
    # K-band (19052 - 25948 MHz)
    19052: (1.0, -0.001419, 7.56e-07, -1.53e-10),
    19180: (1.0, -0.001426, 7.7e-07, -1.59e-10),
    19308: (1.0, -0.001433, 7.82e-07, -1.64e-10),
    19436: (1.0, -0.001429, 7.73e-07, -1.6e-10),
    19564: (1.0, -0.001427, 7.7e-07, -1.59e-10),
    19692: (1.0, -0.001425, 7.65e-07, -1.56e-10),
    19820: (1.0, -0.00143, 7.76e-07, -1.62e-10),
    19948: (1.0, -0.001434, 7.81e-07, -1.63e-10),
    21052: (1.0, -0.001448, 8.05e-07, -1.73e-10),
    21180: (1.0, -0.001436, 7.84e-07, -1.63e-10),
    21308: (1.0, -0.001441, 7.94e-07, -1.68e-10),
    21436: (1.0, -0.001439, 7.89e-07, -1.66e-10),
    21564: (1.0, -0.001442, 7.96e-07, -1.69e-10),
    21692: (1.0, -0.001435, 7.81e-07, -1.61e-10),
    21820: (1.0, -0.001442, 7.92e-07, -1.66e-10),
    21948: (1.0, -0.001439, 7.82e-07, -1.61e-10),
    23052: (1.0, -0.001401, 7.21e-07, -1.37e-10),
    23180: (1.0, -0.001408, 7.31e-07, -1.41e-10),
    23308: (1.0, -0.001407, 7.28e-07, -1.39e-10),
    23436: (1.0, -0.001407, 7.31e-07, -1.41e-10),
    23564: (1.0, -0.001419, 7.47e-07, -1.47e-10),
    23692: (1.0, -0.001395, 7.1e-07, -1.33e-10),
    23820: (1.0, -0.001413, 7.36e-07, -1.42e-10),
    23948: (1.0, -0.001402, 7.21e-07, -1.36e-10),
    25052: (1.0, -0.001402, 7.17e-07, -1.31e-10),
    25180: (1.0, -0.001432, 7.73e-07, -1.58e-10),
    25308: (1.0, -0.001407, 7.22e-07, -1.33e-10),
    25436: (1.0, -0.001417, 7.43e-07, -1.45e-10),
    25564: (1.0, -0.001422, 7.52e-07, -1.48e-10),
    25692: (1.0, -0.001427, 7.59e-07, -1.52e-10),
    25820: (1.0, -0.001416, 7.42e-07, -1.44e-10),
    25948: (1.0, -0.001422, 7.46e-07, -1.45e-10),
    # Ka-band (28052 - 38048 MHz)
    28052: (1.0, -0.001444, 7.61e-07, -1.44e-10),
    28180: (1.0, -0.001439, 7.54e-07, -1.42e-10),
    28308: (1.0, -0.001457, 7.87e-07, -1.58e-10),
    28436: (1.0, -0.001457, 7.9e-07, -1.6e-10),
    28564: (1.0, -0.001455, 7.87e-07, -1.59e-10),
    28692: (1.0, -0.001458, 7.88e-07, -1.58e-10),
    28820: (1.0, -0.001453, 7.81e-07, -1.56e-10),
    28948: (1.0, -0.00146, 7.98e-07, -1.64e-10),
    31052: (1.0, -0.001415, 7.44e-07, -1.44e-10),
    31180: (1.0, -0.001408, 7.26e-07, -1.37e-10),
    31308: (1.0, -0.001413, 7.28e-07, -1.36e-10),
    31436: (1.0, -0.001394, 7.07e-07, -1.3e-10),
    31564: (1.0, -0.001404, 7.23e-07, -1.37e-10),
    31692: (1.0, -0.001427, 7.48e-07, -1.44e-10),
    31820: (1.0, -0.001418, 7.48e-07, -1.48e-10),
    31948: (1.0, -0.001413, 7.37e-07, -1.42e-10),
    34052: (1.0, -0.00142, 7.28e-07, -1.34e-10),
    34180: (1.0, -0.00146, 7.77e-07, -1.53e-10),
    34308: (1.0, -0.00142, 7.41e-07, -1.42e-10),
    34436: (1.0, -0.00142, 7.36e-07, -1.39e-10),
    34564: (1.0, -0.00146, 7.76e-07, -1.52e-10),
    34692: (1.0, -0.00142, 7.34e-07, -1.38e-10),
    34820: (1.0, -0.00142, 7.34e-07, -1.39e-10),
    34948: (1.0, -0.00145, 7.68e-07, -1.49e-10),
    37152: (1.0, -0.00142, 7.47e-07, -1.44e-10),
    37280: (1.0, -0.00141, 7.35e-07, -1.4e-10),
    37408: (1.0, -0.00145, 7.65e-07, -1.46e-10),
    37536: (1.0, -0.00141, 7.13e-07, -1.29e-10),
    37664: (1.0, -0.00141, 7.3e-07, -1.38e-10),
    37792: (1.0, -0.00145, 7.75e-07, -1.5e-10),
    37820: (1.0, -0.00145, 7.68e-07, -1.49e-10),
    38048: (1.0, -0.00141, 7.38e-07, -1.43e-10),
    # Q-band (41052 - 43948 MHz)
    41052: (1.0, -0.001453, 7.69e-07, -1.47e-10),
    41180: (1.0, -0.001479, 8.03e-07, -1.61e-10),
    41308: (1.0, -0.001475, 7.97e-07, -1.58e-10),
    41436: (1.0, -0.001451, 7.73e-07, -1.51e-10),
    41564: (1.0, -0.00145, 7.71e-07, -1.51e-10),
    41692: (1.0, -0.001465, 7.79e-07, -1.49e-10),
    41820: (1.0, -0.00146, 7.73e-07, -1.47e-10),
    41948: (1.0, -0.001434, 7.47e-07, -1.4e-10),
    43052: (1.0, -0.001428, 7.4e-07, -1.38e-10),
    43180: (1.0, -0.001418, 7.29e-07, -1.34e-10),
    43308: (1.0, -0.001433, 7.49e-07, -1.43e-10),
    43436: (1.0, -0.001438, 7.55e-07, -1.45e-10),
    43564: (1.0, -0.001419, 7.36e-07, -1.4e-10),
    43692: (1.0, -0.001397, 7.13e-07, -1.33e-10),
    43820: (1.0, -0.001423, 7.39e-07, -1.4e-10),
    43948: (1.0, -0.001452, 7.68e-07, -1.47e-10),
}

EVLA_BANDS = {
    'P': (200.0, 500.0),
    'L': (900.0, 2003.0),
    'S': (2003.0, 4001.0),
    'C': (4001.0, 8001.0),
    'X': (8001.0, 12001.0),
    'Ku': (12001.0, 18000.0),
    'K': (18000.0, 26500.0),
    'Ka': (26500.0, 40000.0),
    'Q': (40000.0, 50000.0),
}


def VLA_pbcor(fitsimage, outfile, freq=None, ms=None, pblimit=0.15, telescope=None):
    """
    Apply a (EVLA) VLA primary beam correction to a FITS image.
    Parameters
    ----------
    fitsimage : str
        Path to the input FITS image that needs to be corrected.

    outfile : str
        Path where the primary-beam-corrected FITS image will be saved.

    freq : float, optional
        Observing frequency in GHz. If not provided, the function reads CRVAL3
        from the FITS header, assuming it is in Hz.

    ms : str, optional
        Path to the Measurement Set. If provided, the pointing center is read
        from its FIELD/REFERENCE_DIR column. Otherwise, the image center is used as the pointing center.

    pblimit : float, optional
        Primary beam limit in fraction of the peak. Pixels outside the first radial crossing of this limit are set to NaN. Default is 0.15 (zero means no limit).

    telescope : str, optional
        Specify the telescope type: 'VLA' or 'EVLA'. Default is None, in which
        case it is read from ``ms``, or set to 'EVLA' if no ``ms`` is given.

    Returns
    -------
    outfile : str
        Path to the corrected FITS image written to disk.

    Notes
    -----
    The correction follows the AIPS PBCOR polynomial convention:
    see https://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR

    Note there is the VLA and the EVLA primary beam correction. 
    The VLA coffefficients are here https://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?PBCOR
    The EVLA coefficients can be found in see EVLA Memo 195, Jansky Very Large Array Primary Beam Characteristics by Rick Perley, revision dated June 2016 https://library.nrao.edu/public/memos/evla/EVLAM_195.pdf
    For the EVLA If the observing frequency is between two tabulated frequencies, then the beam is computed for each of the tabulated frequencies and then interpolated to the observing frequency.
    """

    # Classic VLA polynomial coefficients from AIPS PBCOR:
    # F(x) = 1.0 + a * x / 10**3 + b * x**2 / 10**7 + c * x**3 / 10**10
    # where x = (separation_arcmin * freq_GHz)**2
    VLA_PB_COEFFS = {
        '4': (1.0, -0.897e-3, 2.71e-7, -0.242e-10),     # 0.0738 GHz
        'P': (1.0, -0.935e-3, 3.23e-7, -0.378e-10),     # 0.3275 GHz
        'L': (1.0, -1.343e-3, 6.579e-7, -1.186e-10),    # 1.465 GHz
        'C': (1.0, -1.372e-3, 6.940e-7, -1.309e-10),    # 4.885 GHz
        'X': (1.0, -1.306e-3, 6.253e-7, -1.100e-10),    # 8.435 GHz
        'U': (1.0, -1.305e-3, 6.155e-7, -1.030e-10),    # 14.965 GHz (Ku)
        'K': (1.0, -1.417e-3, 7.332e-7, -1.352e-10),    # 22.485 GHz
        'Q': (1.0, -1.321e-3, 6.185e-7, -0.983e-10),    # 43.315 GHz
        }
    # EVLA polynomial coefficients from EVLA Memo 195 (Perley 2016):
    # P(r) = P0 + P2 * r^2 + P4 * r^4 + P6 * r^6 + P8 * r^8
    # where r = separation_arcmin * freq_GHz, and frequency keys are in MHz
    EVLA_PB_COEFFS = {
        # P-band (232 - 470 MHz)
        232: (1.0, -0.001137, 5.19e-07, -1.04e-10, 7.1e-15),
        246: (1.0, -0.00113, 5.04e-07, -1.02e-10, 7.7e-15),
        281: (1.0, -0.001106, 5.11e-07, -1.1e-10, 9.1e-15),
        296: (1.0, -0.001125, 5.27e-07, -1.14e-10, 9.6e-15),
        312: (1.0, -0.00103, 4.44e-07, -8.9e-11, 6.8e-15),
        328: (1.0, -0.00098, 4.25e-07, -8.7e-11, 6.9e-15),
        344: (1.0, -0.000974, 4.09e-07, -7.6e-11, 5.3e-15),
        357: (1.0, -0.000996, 4.23e-07, -7.9e-11, 5.1e-15),
        382: (1.0, -0.001002, 4.39e-07, -8.8e-11, 6.4e-15),
        392: (1.0, -0.001067, 5.13e-07, -1.12e-10, 9e-15),
        403: (1.0, -0.001057, 4.9e-07, -1.06e-10, 8.7e-15),
        421: (1.0, -0.001154, 5.85e-07, -1.33e-10, 1.08e-14),
        458: (1.0, -0.000993, 4.67e-07, -1.04e-10, 8.8e-15),
        470: (1.0, -0.00101, 4.85e-07, -1.07e-10, 8.6e-15),
        # L-band (1040 - 2000 MHz)
        1040: (1.0, -0.001529, 8.69e-07, -1.88e-10),
        1104: (1.0, -0.001486, 8.15e-07, -1.68e-10),
        1168: (1.0, -0.001439, 7.53e-07, -1.45e-10),
        1232: (1.0, -0.00145, 7.87e-07, -1.63e-10),
        1296: (1.0, -0.001428, 7.62e-07, -1.54e-10),
        1360: (1.0, -0.001449, 8.02e-07, -1.74e-10),
        1424: (1.0, -0.001462, 8.23e-07, -1.83e-10),
        1488: (1.0, -0.001455, 7.92e-07, -1.63e-10),
        1552: (1.0, -0.001435, 7.54e-07, -1.49e-10),
        1680: (1.0, -0.001443, 7.74e-07, -1.57e-10),
        1744: (1.0, -0.001462, 8.02e-07, -1.69e-10),
        1808: (1.0, -0.001488, 8.38e-07, -1.83e-10),
        1872: (1.0, -0.001486, 8.26e-07, -1.75e-10),
        1936: (1.0, -0.001459, 7.93e-07, -1.62e-10),
        2000: (1.0, -0.001508, 8.31e-07, -1.68e-10),
        # S-band (2052 - 3948 MHz)
        2052: (1.0, -0.001429, 7.52e-07, -1.47e-10),
        2180: (1.0, -0.001389, 7.06e-07, -1.33e-10),
        2436: (1.0, -0.001377, 6.9e-07, -1.27e-10),
        2564: (1.0, -0.001381, 6.92e-07, -1.26e-10),
        2692: (1.0, -0.001402, 7.23e-07, -1.4e-10),
        2820: (1.0, -0.001433, 7.62e-07, -1.54e-10),
        2948: (1.0, -0.001433, 7.46e-07, -1.42e-10),
        3052: (1.0, -0.001467, 8.05e-07, -1.7e-10),
        3180: (1.0, -0.001497, 8.38e-07, -1.8e-10),
        3308: (1.0, -0.001504, 8.37e-07, -1.77e-10),
        3436: (1.0, -0.001521, 8.63e-07, -1.88e-10),
        3564: (1.0, -0.001505, 8.37e-07, -1.75e-10),
        3692: (1.0, -0.001521, 8.51e-07, -1.79e-10),
        3820: (1.0, -0.001534, 8.57e-07, -1.77e-10),
        3948: (1.0, -0.001516, 8.3e-07, -1.66e-10),
        # C-band (4052 - 7948 MHz)
        4052: (1.0, -0.001406, 7.41e-07, -1.48e-10),
        4180: (1.0, -0.001385, 7.09e-07, -1.36e-10),
        4308: (1.0, -0.00138, 7.08e-07, -1.37e-10),
        4436: (1.0, -0.001362, 6.95e-07, -1.35e-10),
        4564: (1.0, -0.001365, 6.92e-07, -1.31e-10),
        4692: (1.0, -0.001339, 6.56e-07, -1.17e-10),
        4820: (1.0, -0.001371, 7.06e-07, -1.4e-10),
        4948: (1.0, -0.001358, 6.91e-07, -1.34e-10),
        5052: (1.0, -0.00136, 6.91e-07, -1.33e-10),
        5180: (1.0, -0.001353, 6.74e-07, -1.25e-10),
        5308: (1.0, -0.001359, 6.82e-07, -1.27e-10),
        5436: (1.0, -0.00138, 7.05e-07, -1.37e-10),
        5564: (1.0, -0.001376, 6.99e-07, -1.31e-10),
        5692: (1.0, -0.001405, 7.39e-07, -1.47e-10),
        5820: (1.0, -0.001394, 7.29e-07, -1.45e-10),
        5948: (1.0, -0.001428, 7.57e-07, -1.57e-10),
        6052: (1.0, -0.001445, 7.68e-07, -1.5e-10),
        6148: (1.0, -0.001422, 7.38e-07, -1.38e-10),
        6308: (1.0, -0.001463, 7.94e-07, -1.62e-10),
        6436: (1.0, -0.001478, 8.22e-07, -1.74e-10),
        6564: (1.0, -0.001473, 8e-07, -1.62e-10),
        6692: (1.0, -0.001455, 7.76e-07, -1.53e-10),
        6820: (1.0, -0.001487, 8.22e-07, -1.72e-10),
        6948: (1.0, -0.001472, 8.05e-07, -1.67e-10),
        7052: (1.0, -0.00147, 8.01e-07, -1.64e-10),
        7180: (1.0, -0.001503, 8.5e-07, -1.84e-10),
        7308: (1.0, -0.001482, 8.19e-07, -1.72e-10),
        7436: (1.0, -0.001498, 8.22e-07, -1.66e-10),
        7564: (1.0, -0.00149, 8.18e-07, -1.66e-10),
        7692: (1.0, -0.001481, 7.98e-07, -1.56e-10),
        7820: (1.0, -0.001474, 7.94e-07, -1.57e-10),
        7948: (1.0, -0.001448, 7.69e-07, -1.51e-10),
        # X-band (8052 - 11948 MHz)
        8052: (1.0, -0.001403, 7.21e-07, -1.37e-10),
        8180: (1.0, -0.001398, 7.1e-07, -1.32e-10),
        8308: (1.0, -0.001402, 7.16e-07, -1.35e-10),
        8436: (1.0, -0.0014, 7.12e-07, -1.32e-10),
        8564: (1.0, -0.001391, 6.95e-07, -1.25e-10),
        8692: (1.0, -0.001409, 7.34e-07, -1.49e-10),
        8820: (1.0, -0.00141, 7.36e-07, -1.45e-10),
        8948: (1.0, -0.00141, 7.34e-07, -1.43e-10),
        9052: (1.0, -0.001403, 7.2e-07, -1.36e-10),
        9180: (1.0, -0.001396, 7.09e-07, -1.31e-10),
        9308: (1.0, -0.001432, 7.68e-07, -1.55e-10),
        9436: (1.0, -0.001414, 7.43e-07, -1.47e-10),
        9564: (1.0, -0.001416, 7.45e-07, -1.47e-10),
        9692: (1.0, -0.001406, 7.26e-07, -1.39e-10),
        9820: (1.0, -0.001412, 7.36e-07, -1.43e-10),
        9948: (1.0, -0.001409, 7.29e-07, -1.39e-10),
        10052: (1.0, -0.001421, 7.46e-07, -1.45e-10),
        10180: (1.0, -0.001409, 7.25e-07, -1.36e-10),
        10308: (1.0, -0.001402, 7.13e-07, -1.31e-10),
        10436: (1.0, -0.001399, 7.09e-07, -1.29e-10),
        10564: (1.0, -0.001413, 7.37e-07, -1.43e-10),
        10692: (1.0, -0.001412, 7.34e-07, -1.41e-10),
        10820: (1.0, -0.001401, 7.12e-07, -1.31e-10),
        10948: (1.0, -0.001401, 7.12e-07, -1.31e-10),
        11052: (1.0, -0.001401, 7.12e-07, -1.31e-10),
        11180: (1.0, -0.001394, 6.99e-07, -1.24e-10),
        11308: (1.0, -0.001394, 7.01e-07, -1.26e-10),
        11436: (1.0, -0.001391, 6.94e-07, -1.22e-10),
        11564: (1.0, -0.001389, 6.92e-07, -1.22e-10),
        11692: (1.0, -0.001386, 6.8e-07, -1.15e-10),
        11820: (1.0, -0.001391, 6.88e-07, -1.19e-10),
        11948: (1.0, -0.001399, 6.97e-07, -1.22e-10),
        # Ku-band (12052 - 17948 MHz)
        12052: (1.0, -0.001399, 7.17e-07, -1.34e-10),
        12180: (1.0, -0.001392, 7.07e-07, -1.31e-10),
        12308: (1.0, -0.001393, 7.19e-07, -1.38e-10),
        12436: (1.0, -0.001393, 7.2e-07, -1.4e-10),
        12564: (1.0, -0.001395, 7.19e-07, -1.38e-10),
        12692: (1.0, -0.001397, 7.2e-07, -1.37e-10),
        12820: (1.0, -0.001388, 7.06e-07, -1.32e-10),
        12948: (1.0, -0.001397, 7.18e-07, -1.36e-10),
        13052: (1.0, -0.0014, 7.27e-07, -1.4e-10),
        13180: (1.0, -0.001406, 7.44e-07, -1.5e-10),
        13308: (1.0, -0.001403, 7.37e-07, -1.47e-10),
        13436: (1.0, -0.001392, 7.08e-07, -1.31e-10),
        13564: (1.0, -0.001384, 6.94e-07, -1.24e-10),
        13692: (1.0, -0.001382, 6.95e-07, -1.25e-10),
        13820: (1.0, -0.001376, 6.88e-07, -1.24e-10),
        13948: (1.0, -0.001384, 6.98e-07, -1.28e-10),
        14052: (1.0, -0.0014, 7.36e-07, -1.48e-10),
        14180: (1.0, -0.001397, 7.29e-07, -1.45e-10),
        14308: (1.0, -0.001399, 7.32e-07, -1.45e-10),
        14436: (1.0, -0.001396, 7.25e-07, -1.42e-10),
        14564: (1.0, -0.001393, 7.2e-07, -1.39e-10),
        14692: (1.0, -0.001384, 7.03e-07, -1.31e-10),
        14820: (1.0, -0.001388, 7.06e-07, -1.32e-10),
        14948: (1.0, -0.001393, 7.16e-07, -1.37e-10),
        15052: (1.0, -0.001402, 7.38e-07, -1.48e-10),
        15180: (1.0, -0.001407, 7.47e-07, -1.53e-10),
        15308: (1.0, -0.001406, 7.41e-07, -1.48e-10),
        15436: (1.0, -0.001399, 7.31e-07, -1.44e-10),
        15564: (1.0, -0.001397, 7.28e-07, -1.43e-10),
        15692: (1.0, -0.001401, 7.35e-07, -1.46e-10),
        15820: (1.0, -0.001402, 7.34e-07, -1.45e-10),
        15948: (1.0, -0.001399, 7.3e-07, -1.44e-10),
        16052: (1.0, -0.001419, 7.59e-07, -1.54e-10),
        16180: (1.0, -0.001419, 7.59e-07, -1.52e-10),
        16308: (1.0, -0.001412, 7.4e-07, -1.44e-10),
        16436: (1.0, -0.001407, 7.32e-07, -1.4e-10),
        16564: (1.0, -0.001408, 7.32e-07, -1.41e-10),
        16692: (1.0, -0.00141, 7.34e-07, -1.4e-10),
        16820: (1.0, -0.001407, 7.27e-07, -1.38e-10),
        16948: (1.0, -0.001423, 7.63e-07, -1.55e-10),
        17052: (1.0, -0.001437, 7.87e-07, -1.66e-10),
        17180: (1.0, -0.001438, 7.84e-07, -1.64e-10),
        17308: (1.0, -0.001445, 7.98e-07, -1.71e-10),
        17436: (1.0, -0.001452, 8.1e-07, -1.77e-10),
        17564: (1.0, -0.001458, 8.13e-07, -1.7e-10),
        17692: (1.0, -0.001456, 8.06e-07, -1.72e-10),
        17820: (1.0, -0.001453, 8e-07, -1.68e-10),
        17948: (1.0, -0.001452, 7.99e-07, -1.69e-10),
        # K-band (19052 - 25948 MHz)
        19052: (1.0, -0.001419, 7.56e-07, -1.53e-10),
        19180: (1.0, -0.001426, 7.7e-07, -1.59e-10),
        19308: (1.0, -0.001433, 7.82e-07, -1.64e-10),
        19436: (1.0, -0.001429, 7.73e-07, -1.6e-10),
        19564: (1.0, -0.001427, 7.7e-07, -1.59e-10),
        19692: (1.0, -0.001425, 7.65e-07, -1.56e-10),
        19820: (1.0, -0.00143, 7.76e-07, -1.62e-10),
        19948: (1.0, -0.001434, 7.81e-07, -1.63e-10),
        21052: (1.0, -0.001448, 8.05e-07, -1.73e-10),
        21180: (1.0, -0.001436, 7.84e-07, -1.63e-10),
        21308: (1.0, -0.001441, 7.94e-07, -1.68e-10),
        21436: (1.0, -0.001439, 7.89e-07, -1.66e-10),
        21564: (1.0, -0.001442, 7.96e-07, -1.69e-10),
        21692: (1.0, -0.001435, 7.81e-07, -1.61e-10),
        21820: (1.0, -0.001442, 7.92e-07, -1.66e-10),
        21948: (1.0, -0.001439, 7.82e-07, -1.61e-10),
        23052: (1.0, -0.001401, 7.21e-07, -1.37e-10),
        23180: (1.0, -0.001408, 7.31e-07, -1.41e-10),
        23308: (1.0, -0.001407, 7.28e-07, -1.39e-10),
        23436: (1.0, -0.001407, 7.31e-07, -1.41e-10),
        23564: (1.0, -0.001419, 7.47e-07, -1.47e-10),
        23692: (1.0, -0.001395, 7.1e-07, -1.33e-10),
        23820: (1.0, -0.001413, 7.36e-07, -1.42e-10),
        23948: (1.0, -0.001402, 7.21e-07, -1.36e-10),
        25052: (1.0, -0.001402, 7.17e-07, -1.31e-10),
        25180: (1.0, -0.001432, 7.73e-07, -1.58e-10),
        25308: (1.0, -0.001407, 7.22e-07, -1.33e-10),
        25436: (1.0, -0.001417, 7.43e-07, -1.45e-10),
        25564: (1.0, -0.001422, 7.52e-07, -1.48e-10),
        25692: (1.0, -0.001427, 7.59e-07, -1.52e-10),
        25820: (1.0, -0.001416, 7.42e-07, -1.44e-10),
        25948: (1.0, -0.001422, 7.46e-07, -1.45e-10),
        # Ka-band (28052 - 38048 MHz)
        28052: (1.0, -0.001444, 7.61e-07, -1.44e-10),
        28180: (1.0, -0.001439, 7.54e-07, -1.42e-10),
        28308: (1.0, -0.001457, 7.87e-07, -1.58e-10),
        28436: (1.0, -0.001457, 7.9e-07, -1.6e-10),
        28564: (1.0, -0.001455, 7.87e-07, -1.59e-10),
        28692: (1.0, -0.001458, 7.88e-07, -1.58e-10),
        28820: (1.0, -0.001453, 7.81e-07, -1.56e-10),
        28948: (1.0, -0.00146, 7.98e-07, -1.64e-10),
        31052: (1.0, -0.001415, 7.44e-07, -1.44e-10),
        31180: (1.0, -0.001408, 7.26e-07, -1.37e-10),
        31308: (1.0, -0.001413, 7.28e-07, -1.36e-10),
        31436: (1.0, -0.001394, 7.07e-07, -1.3e-10),
        31564: (1.0, -0.001404, 7.23e-07, -1.37e-10),
        31692: (1.0, -0.001427, 7.48e-07, -1.44e-10),
        31820: (1.0, -0.001418, 7.48e-07, -1.48e-10),
        31948: (1.0, -0.001413, 7.37e-07, -1.42e-10),
        34052: (1.0, -0.00142, 7.28e-07, -1.34e-10),
        34180: (1.0, -0.00146, 7.77e-07, -1.53e-10),
        34308: (1.0, -0.00142, 7.41e-07, -1.42e-10),
        34436: (1.0, -0.00142, 7.36e-07, -1.39e-10),
        34564: (1.0, -0.00146, 7.76e-07, -1.52e-10),
        34692: (1.0, -0.00142, 7.34e-07, -1.38e-10),
        34820: (1.0, -0.00142, 7.34e-07, -1.39e-10),
        34948: (1.0, -0.00145, 7.68e-07, -1.49e-10),
        37152: (1.0, -0.00142, 7.47e-07, -1.44e-10),
        37280: (1.0, -0.00141, 7.35e-07, -1.4e-10),
        37408: (1.0, -0.00145, 7.65e-07, -1.46e-10),
        37536: (1.0, -0.00141, 7.13e-07, -1.29e-10),
        37664: (1.0, -0.00141, 7.3e-07, -1.38e-10),
        37792: (1.0, -0.00145, 7.75e-07, -1.5e-10),
        37820: (1.0, -0.00145, 7.68e-07, -1.49e-10),
        38048: (1.0, -0.00141, 7.38e-07, -1.43e-10),
        # Q-band (41052 - 43948 MHz)
        41052: (1.0, -0.001453, 7.69e-07, -1.47e-10),
        41180: (1.0, -0.001479, 8.03e-07, -1.61e-10),
        41308: (1.0, -0.001475, 7.97e-07, -1.58e-10),
        41436: (1.0, -0.001451, 7.73e-07, -1.51e-10),
        41564: (1.0, -0.00145, 7.71e-07, -1.51e-10),
        41692: (1.0, -0.001465, 7.79e-07, -1.49e-10),
        41820: (1.0, -0.00146, 7.73e-07, -1.47e-10),
        41948: (1.0, -0.001434, 7.47e-07, -1.4e-10),
        43052: (1.0, -0.001428, 7.4e-07, -1.38e-10),
        43180: (1.0, -0.001418, 7.29e-07, -1.34e-10),
        43308: (1.0, -0.001433, 7.49e-07, -1.43e-10),
        43436: (1.0, -0.001438, 7.55e-07, -1.45e-10),
        43564: (1.0, -0.001419, 7.36e-07, -1.4e-10),
        43692: (1.0, -0.001397, 7.13e-07, -1.33e-10),
        43820: (1.0, -0.001423, 7.39e-07, -1.4e-10),
        43948: (1.0, -0.001452, 7.68e-07, -1.47e-10),
        }

    EVLA_BANDS = {
        'P': (200.0, 500.0),
        'L': (900.0, 2003.0),
        'S': (2003.0, 4001.0),
        'C': (4001.0, 8001.0),
        'X': (8001.0, 12001.0),
        'Ku': (12001.0, 18000.0),
        'K': (18000.0, 26500.0),
        'Ka': (26500.0, 40000.0),
        'Q': (40000.0, 50000.0),
        }

    if freq is None:
        with fits.open(fitsimage) as hdul:
            freq = hdul[0].header['CRVAL3'] / 1e9
        terminal_print('Frequency found from the FITS image [GHz]:', freq)

    hdu = fits.open(fitsimage, ignore_missing_end=True)
    hduflat = flatten(hdu)
    img = hduflat.data
    terminal_print('IMAGE shape', img.shape)
    ny, nx = img.shape
    y_idx, x_idx = np.indices(img.shape)

    w = WCS(hduflat.header)
    if ms is not None:
        terminal_print('Taking pointing center from the ms')
        with table(ms + '::FIELD', readonly=True, ack=False) as t:
            ref_dir = t.getcol('REFERENCE_DIR')
        ref_dir = np.squeeze(ref_dir)
        if ref_dir.ndim > 1:
            ref_dir = ref_dir[0]
        ra_ref, dec_ref = ref_dir[0], ref_dir[1]
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')
    else:
        center = w.pixel_to_world(nx / 2., ny / 2.)
        terminal_print('Assume image center is the pointing center')
        terminal_print('Make sure this is correct, if not provide the ms to the function')
        terminal_print('CENTER image:', center)

    coordinates = w.pixel_to_world(np.ravel(x_idx), np.ravel(y_idx))
    separation = center.separation(coordinates).arcmin.reshape(img.shape)

    def _eval_poly(r_arcmin, f_ghz, coeffs):
        """
        Evaluate a fitted radial-frequency polynomial.

        Parameters
        ----------
        r_arcmin : float
            Radius in arcminutes.
        f_ghz : float
            Frequency in GHz.
        coeffs : iterable
            Polynomial coefficients.

        Returns
        -------
        float
            Evaluated polynomial value.
        """
        x = (r_arcmin * f_ghz) ** 2
        pb = np.zeros_like(x, dtype=float)
        x_power = np.ones_like(x, dtype=float)
        for c in coeffs:
            pb += c * x_power
            x_power = x_power * x
        return pb

    if telescope is None:
        if ms is not None:
            telescope = get_telescope_from_ms(ms)
        else:
            telescope = 'EVLA'

    if str(telescope).upper() == 'VLA':
        # Classic VLA polynomial coefficients from AIPS PBCOR
        if freq < 0.15:
            coeffs = VLA_PB_COEFFS['4']
        elif freq < 0.8:
            coeffs = VLA_PB_COEFFS['P']
        elif freq < 2.5:
            coeffs = VLA_PB_COEFFS['L']
        elif freq < 6.5:
            coeffs = VLA_PB_COEFFS['C']
        elif freq < 11.5:
            coeffs = VLA_PB_COEFFS['X']
        elif freq < 18.0:
            coeffs = VLA_PB_COEFFS['U']
        elif freq < 30.0:
            coeffs = VLA_PB_COEFFS['K']
        else:
            coeffs = VLA_PB_COEFFS['Q']

        primary_beam = _eval_poly(separation, freq, coeffs)
        if pblimit > 0.0:
            radial_distance = np.unique(np.sort(separation.ravel()))
            radial_beam = _eval_poly(radial_distance, freq, coeffs)
    else:
        # EVLA primary beam with frequency interpolation per EVLA Memo 195 and AIPS PBCOR
        freq_mhz = freq * 1e3
        selected_band = None
        for b, (bmin, bmax) in EVLA_BANDS.items():
            if bmin <= freq_mhz <= bmax:
                selected_band = b
                break
        if selected_band is None:
            selected_band = min(EVLA_BANDS.keys(), key=lambda b: min(abs(freq_mhz - EVLA_BANDS[b][0]), abs(freq_mhz - EVLA_BANDS[b][1])))

        freqs_in_band = sorted([f for f in EVLA_PB_COEFFS if EVLA_BANDS[selected_band][0] <= f <= EVLA_BANDS[selected_band][1]])

        if freq_mhz <= freqs_in_band[0]:
            f1 = f2 = freqs_in_band[0]
            w_interp = 0.0
        elif freq_mhz >= freqs_in_band[-1]:
            f1 = f2 = freqs_in_band[-1]
            w_interp = 0.0
        else:
            f1 = freqs_in_band[0]
            f2 = freqs_in_band[-1]
            w_interp = 0.0
            for i in range(len(freqs_in_band) - 1):
                if freqs_in_band[i] <= freq_mhz <= freqs_in_band[i + 1]:
                    f1 = freqs_in_band[i]
                    f2 = freqs_in_band[i + 1]
                    w_interp = (freq_mhz - f1) / float(f2 - f1)
                    break

        if f1 == f2:
            primary_beam = _eval_poly(separation, f1 / 1e3, EVLA_PB_COEFFS[f1])
            if pblimit > 0.0:
                radial_distance = np.unique(np.sort(separation.ravel()))
                radial_beam = _eval_poly(radial_distance, f1 / 1e3, EVLA_PB_COEFFS[f1])
        else:
            pb1 = _eval_poly(separation, f1 / 1e3, EVLA_PB_COEFFS[f1])
            pb2 = _eval_poly(separation, f2 / 1e3, EVLA_PB_COEFFS[f2])
            primary_beam = (1.0 - w_interp) * pb1 + w_interp * pb2
            if pblimit > 0.0:
                radial_distance = np.unique(np.sort(separation.ravel()))
                rpb1 = _eval_poly(radial_distance, f1 / 1e3, EVLA_PB_COEFFS[f1])
                rpb2 = _eval_poly(radial_distance, f2 / 1e3, EVLA_PB_COEFFS[f2])
                radial_beam = (1.0 - w_interp) * rpb1 + w_interp * rpb2

    primary_beam = np.where(primary_beam <= 0.0, np.nan, primary_beam)

    if hdu[0].data.ndim == 4:
        hdu[0].data[0, 0, :, :] = img / primary_beam
    else:
        hdu[0].data[:, :] = img / primary_beam

    if pblimit > 0.0:
        cutoff_idx = np.flatnonzero(radial_beam <= pblimit)
        if cutoff_idx.size > 0:
            cutoff_radius = radial_distance[cutoff_idx[0]]
            mask = separation >= cutoff_radius
            if hdu[0].data.ndim == 4:
                hdu[0].data[0, 0, :, :][mask] = np.nan
            else:
                hdu[0].data[mask] = np.nan

    astropy.io.fits.writeto(outfile, hdu[0].data, hdu[0].header, overwrite=True)
    hdu.close()
    return outfile


def uGMRT_pbcor(fitsimage, outfile, freq=None, ms=None, pblimit=0.15):
    """
    Apply a (u)GMRT primary beam correction to a FITS image.
    Parameters
    ----------
    fitsimage : str
        Path to the input FITS image that needs to be corrected.

    outfile : str
        Path where the primary-beam-corrected FITS image will be saved.

    freq : float, optional
        Observing frequency in GHz. If not provided, the function reads CRVAL3
        from the FITS header, assuming it is in Hz.

    ms : str, optional
        Path to the Measurement Set. If provided, the pointing center is read
        from its FIELD/REFERENCE_DIR column. Otherwise, the image center is
        used as the pointing center.

    pblimit : float, optional
        Primary beam limit in fraction of the peak. Pixels outside the first
        radial crossing of this limit are set to NaN. Default is 0.15 (zero means no limit).

    Returns
    -------
    outfile : str
        Path to the corrected FITS image written to disk.

    Notes
    -----
    The correction follows the AIPS PBCOR polynomial convention:

    F(x) = 1 + a*x/10**3 + b*x**2/10**7 + c*x**3/10**10 + d*x**4/10**13

    where x is the squared distance from the pointing center in
    (arcmin * frequency_GHz)**2. The coefficients are from the GMRT primary
    beam fits shown in the GMRT calibration documentation. The fifth AIPS
    term, which would use PBPARM(7)/10**16, is not included because the table
    provides coefficients only through PBPARM(6).

    PBPARM are based on https://www.gmrt.ncra.tifr.res.in/gtac/sch/c51webfiles/gtac_51_status_doc.pdf
    """
    if freq is None:
        with fits.open(fitsimage) as hdul:
            freq = hdul[0].header['CRVAL3'] / 1e9
        terminal_print('Frequency found from the FITS image [GHz]:', freq)

    if 0.125 <= freq <= 0.250:
        coefficients = (-3.089, 39.314, -23.011, 5.037)
    elif 0.250 < freq <= 0.500:
        coefficients = (-3.129, 38.816, -21.608, 4.483)
    elif 0.550 <= freq <= 0.850:
        coefficients = (-3.263, 42.618, -25.580, 5.823)
    elif 1.050 <= freq <= 1.450:
        coefficients = (-2.614, 27.594, -13.268, 2.395)
    else:
        raise ValueError(
            'Frequency must be within a published (u)GMRT band: '
            '125-250, 250-500, 550-850, or 1050-1450 MHz.'
        )

    coefficient_a, coefficient_b, coefficient_c, coefficient_d = coefficients
    hdu = fits.open(fitsimage, ignore_missing_end=True)
    hduflat = flatten(hdu)
    img = hduflat.data
    terminal_print('IMAGE shape', img.shape)
    x, y = np.indices(img.shape)

    w = WCS(hduflat.header)
    if ms is not None:
        terminal_print('Taking pointing center from the ms')
        with table(ms + '::FIELD', readonly=True, ack=False) as t:
            ra_ref, dec_ref = t.getcol('REFERENCE_DIR').squeeze()
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')
    else:
        center = w.pixel_to_world(img.shape[0] / 2., img.shape[1] / 2.)
        terminal_print('Assume image center is the pointing center')
        terminal_print('Make sure this is correct, if not provide the ms to the function')
        terminal_print('CENTER image:', center)

    coordinates = w.pixel_to_world(np.ravel(x), np.ravel(y))
    separation = center.separation(coordinates).arcmin
    separation = separation.reshape(img.shape)
    radius_frequency = freq * separation
    beam_argument = radius_frequency ** 2
    primary_beam = (1. + coefficient_a * beam_argument / 10**3
                    + coefficient_b * beam_argument ** 2 / 10**7
                    + coefficient_c * beam_argument ** 3 / 10**10
                    + coefficient_d * beam_argument ** 4 / 10**13)

    hdu[0].data[0, 0, :, :] = img / primary_beam

    if pblimit > 0.0:
        radial_distance = np.unique(np.sort(separation.ravel()))
        radial_argument = (freq * radial_distance) ** 2
        radial_beam = (1. + coefficient_a * radial_argument / 10**3
                   + coefficient_b * radial_argument ** 2 / 10**7
                   + coefficient_c * radial_argument ** 3 / 10**10
                   + coefficient_d * radial_argument ** 4 / 10**13)
        cutoff_idx = np.flatnonzero(radial_beam <= pblimit)

        if cutoff_idx.size > 0:
            cutoff_radius = radial_distance[cutoff_idx[0]]
            hdu[0].data[0, 0, :, :][separation >= cutoff_radius] = np.nan

    astropy.io.fits.writeto(outfile, hdu[0].data, hdu[0].header, overwrite=True)
    hdu.close()
    return outfile
