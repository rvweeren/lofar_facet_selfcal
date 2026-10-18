"""Generic FITS image helpers."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def flatten(f):
    """
    Flatten the first FITS HDU to a two-dimensional image.

    Two-dimensional input is returned with its existing header and data.
    For higher-dimensional input, the first two axes and WCS metadata are
    retained while index zero is selected along every additional axis.
    Selected metadata such as beam size, observing facility, and frequency
    are copied to the new header.

    Parameters
    ----------
    f : astropy.io.fits.HDUList
        FITS HDU list whose primary HDU is
        flattened.

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        Two-dimensional primary HDU.
    """

    naxis = f[0].header['NAXIS']
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu
