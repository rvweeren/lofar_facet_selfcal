"""FITS image and WSClean model-image utilities: noise and dynamic range,
masks and island sizes, cleaning up and renaming model images."""
import concurrent.futures
import glob
import os
import shutil
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pyregion
import scipy.ndimage
import scipy.signal
from astropy.io import fits
from casacore.tables import table  # type: ignore

from submods.fits_utils import flatten
from submods.runtime_utils import run, terminal_print


def frequencies_from_models(model_basename):
    """
    This function takes a wsclean model basename string and returns the
    frequency list string wsclean should use for -channel-division-frequencies and the same as a np array

    Parameters
    ----------
    model_basename : str
        wsclean model basename to search for

    Returns
    -------
    freq_string : str
        String of frequency breaks to pass into wsclean
    freqs : np.array
        Array of frequencies needed for further checking
    """
    nonpblist = sorted(glob.glob(model_basename + '-????-model.fits'))
    pblist = sorted(glob.glob(model_basename + '-????-model-pb.fits'))
    # If pb and models
    if len(pblist) > 0:
        models = pblist
    else:
        models = nonpblist

    freqs = [] # Target freqs = Central_freq - (delta/2)
    for model in models:
        tmp_head = fits.getheader(model)
        central_freq = float(tmp_head['CRVAL3'])
        freq_width = float(tmp_head['CDELT3'])
        freqs.append(central_freq-(freq_width/2))

    freq_string = [str(x/1e6)+"e6" for x in freqs]

    freq_string = ",".join(freq_string)

    return freq_string, np.array(freqs)


def modify_freqs_from_ms(mslist, freqs, args):
    """
    This function takes a frequency array and trims it according to the frequencies available within an ms

    Parameters
    ----------
    mslist : [str]
        Paths to MSs to get frequency limits
    freqs : np.array
        Array containing frequency breaks for wsclean
    args : dict
        Parsed facetselfcal options; ``args['wscleanskymodel']`` is the model
        image basename that is renamed when frequencies are trimmed.

    Returns
    -------
    mod_freq_string : str
        String for wsclean with frequency cuts corrected by ms
    mod_freqs : np.array
        Array with modified frequencies matching string
    """
    ms_chan_freqs = []
    for ms in mslist:
        t = table(ms, readonly = True)
        t_chan_freqs = t.SPECTRAL_WINDOW.CHAN_FREQ[::][0] # This provides the frequencies of all channels in the MS
        ms_chan_freqs.append(t_chan_freqs)

    #Flatten Array
    ms_chan_freqs = np.concatenate(ms_chan_freqs, axis = None)
    # Fix max frequency first
    max_ms_freq = np.max(ms_chan_freqs)

    while max_ms_freq < freqs[-1]:
        freqs = freqs[:-1]

    # Fix min frequency - Requires model renaming
    min_ms_freq = np.min(ms_chan_freqs)
    rename_no = 0
    while min_ms_freq > freqs[1]:
        freqs = freqs[1:]
        rename_no += 1
    
    if rename_no > 0:
        rename_models(args['wscleanskymodel'], rename_no, model_prefix = "tmp_")

    mod_freq_string = [str(x/1e6)+"e6" for x in freqs]

    mod_freq_string = ",".join(mod_freq_string)

    return mod_freq_string, np.array(freqs)


def rename_models(model_basename, rename_no, model_prefix = "tmp_"):
    """
    Function renames args['wscleanskymodel'] based on the integer number that need to be renamed and a new prefix to append.
    Update argument parameter at the end

    Parameters
    ----------
    model_basename : str
        Base name shared by the model images to rename.
    rename_no : int
        Number of models that need to be renamed (2 -> Shift all basename models down 2)
    model_prefix : str
        Prefix to apppend to the new model names
    """

    nonpblist = sorted(glob.glob(model_basename + '-????-model.fits'))
    pblist = sorted(glob.glob(model_basename + '-????-model-pb.fits'))


    if len(nonpblist) > 0:
        # Remove not needed models 
        nonpblist = nonpblist[rename_no:]
        for model in nonpblist:
            split_number = model.split("-model.fits")[0]
            number = int(split_number[-4:])
            number_shift = str((number-2)).zfill(4)

            split_number = split_number[:-4] + number_shift 

            model_out = model_prefix + split_number + "-model.fits"
            shutil.copy(model, model_out)
    
    if len(pblist) > 0:
        # Remove not needed models
        pblist = pblist[rename_no:]
        for model in pblist:
            split_number = model.split("-model-pb.fits")[0]
            number = int(split_number[-4:])
            number_shift = str((number-2)).zfill(4)

            split_number = split_number[:-4] + number_shift 

            model_out = model_prefix + split_number + "-model-pb.fits"
            shutil.copy(model, model_out)


def clean_up_images(imagename, model=False):
    """
    Remeoves psf, residual, beam, and dirty channel images after a WSClean run to save disk space

    Parameters
    ----------
    imagename : str
        Image basename used in the WSClean run.
    model : bool, optional
        Also remove channel model images.
    """
    imagelist = sorted(glob.glob(imagename + '-????-*residual*.fits'))  
    imagelist += sorted(glob.glob(imagename + '-????-*dirty*.fits')) 
    imagelist += sorted(glob.glob(imagename + '-????-*psf*.fits')) 
    imagelist += sorted(glob.glob(imagename + '-????-*beam*.fits')) 
    if model:
        imagelist += sorted(glob.glob(imagename + '-????-*model*.fits'))
    for image in imagelist:
        Path(image).unlink(missing_ok=True)
    return


def get_image_size(fitsimage):
    """
    Find the dimensions of a FITS image.

    Parameters
    ----------
    fitsimage : str
        path to the FITS file.

    Returns
    -------
    imsize : tuple
        dimensions of the 2D image
    """
    hdulist = fits.open(fitsimage)
    shape = hdulist[0].data.squeeze().shape # squeeze out dimensions of 1
    hdulist.close()
    return shape


def get_image_dynamicrange(image):
    """
    Get dynamic range of an image (peak over rms)

    Parameters
    ----------
    image : str
        FITS image file name .

    Returns
    -------
    DR : float
        Dynamic range vale.
    """

    terminal_print('Compute image dynamic range (peak over rms): ', image)
    hdul = fits.open(image)
    image_rms = findrms(np.ndarray.flatten(hdul[0].data))
    DR = np.nanmax(np.ndarray.flatten(hdul[0].data)) / image_rms
    hdul.close()
    return DR


def max_area_of_island(grid):
    """
    Calculate the area of an island.

    Parameters
    ----------
    grid : ndarray
        input image.

    Returns
    -------
    None
    """
    rlen, clen = len(grid), len(grid[0])

    def neighbors(r, c):
        """
        Generate the neighbor coordinates of the given row and column that are within the bounds of the grid.

        Parameters
        ----------
        r : int
            row coordinate.
        c : int
            column coordinate.

        Yields
        ------
        tuple of int
            Row and column coordinates of a neighboring grid cell.
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (0 <= r + dr < rlen) and (0 <= c + dc < clen):
                yield r + dr, c + dc

    visited = [[False] * clen for _ in range(rlen)]

    def island_size(r, c):
        """
        Find the area of the land connected to the given coordinate.

        Return 0 if the coordinate is water or if it has already been explored in a previous call to island_size().

        Parameters
        ----------
        r : int
            row coordinate.
        c : int
            column coordinate.

        Returns
        -------
        int
            Number of land cells connected to the coordinate, or zero for water
            and previously visited cells.
        """
        if grid[r][c] == 0 or visited[r][c]:
            return 0
        area = 1
        stack = [(r, c)]
        visited[r][c] = True
        while stack:
            for r, c in neighbors(*stack.pop()):
                if grid[r][c] and not visited[r][c]:
                    stack.append((r, c))
                    visited[r][c] = True
                    area += 1
        return area

    return max(island_size(r, c) for r, c in product(range(rlen), range(clen)))


def getlargestislandsize(fitsmask):
    """
    Find the largest island in a given FITS mask.

    Parameters
    ----------
    fitsmask : str
        path to the FITS file.

    Returns
    -------
    max_area : float
        area of the largest island.
    """
    with fits.open(fitsmask) as hdulist:
        data = hdulist[0].data
        max_area = max_area_of_island(data[0, 0, :, :])
    return max_area


def makemask_extended(fitsimage, outputfitsfile, kernel_size=21, rebin=None, threshold=7.5):
    """
    Generate a binary mask from a FITS image by applying a median filter 
    to remove small-scale structures and thresholding to identify significant emission.

    This function is useful for creating masks of extended emission regions 
    in radio or astronomical images. It optionally downsamples the image before 
    median filtering (for performance on large images), then thresholds the 
    filtered image based on a user-defined sigma level to generate a binary mask. 
    The mask is written to a new FITS file.

    Parameters
    ----------
    fitsimage : str
        Path to the input FITS file containing the image data.
    outputfitsfile : str
        Path to the output FITS file where the binary mask will be saved.
    kernel_size : int, optional
        Size of the square kernel used in the 2D median filter (default is 21).
    rebin : int or None, optional
        Factor by which to downsample the image before filtering, to speed up processing.
        The image is upsampled back to the original size after filtering. If `None`, no rebinning is performed.
    threshold : float, optional
        Sigma threshold (in units of the image RMS) for generating the binary mask (default is 7.5).

    Notes
    -----
    - The mask is applied on a flattened version of the original data (assumes 4D shape [1, 1, y, x]).
    - The filtered and thresholded image is stored back into the FITS file in place.
    - Assumes the presence of a helper function `findrms` and a `flatten` function to reduce FITS HDUs.
    - Requires `scipy`, `skimage`, `numpy`, and `astropy.io.fits`.

    Returns
    -------
    None
        The result is saved directly to the specified output FITS file.
    """
    from skimage.transform import rescale
    
    hdulist = fits.open(fitsimage) 
    hduflat = flatten(hdulist)

    terminal_print(
        'Flattened image shape:',
        hduflat.data.shape,
        'Original FITS image shape:',
        hdulist[0].data.shape,
    )
    datashape = hduflat.data.shape

    # create median filtered image where emission on small scales is removed
    if rebin is not None:
        # needed for large images as scipy.signal.medfilt2d takes too long
        image_data = rescale(hduflat.data, 1./rebin, anti_aliasing=True) # downsample
        image_data = scipy.signal.medfilt2d(image_data, kernel_size=kernel_size)
        image_data = scipy.ndimage.zoom(image_data, float(hduflat.data.shape[0]/image_data.shape[0])) # upsample
        
        imagenoise = findrms(np.ndarray.flatten(image_data))
        terminal_print('Input image shape:', datashape, 'Filtered image shape:', image_data.shape)
        assert datashape == image_data.shape
    else:
        image_data = scipy.signal.medfilt2d(hduflat.data, kernel_size=kernel_size)
        imagenoise = findrms(np.ndarray.flatten(image_data))

    # do the masking
    image_data[image_data >= threshold*imagenoise] = 1.0    
    image_data[image_data < threshold*imagenoise] = 0.
    #hduflat.data = image_data
    hdulist[0].data[0,0,:,:] = image_data
    hdulist.writeto(outputfitsfile, overwrite=True)
    return


def gunzip_model_images(imagebasename, n_parallel=4):
    """
    Gunzips all model images with the given base name.
    Parameters
    ----------
    imagebasename : str
        The base filename for model images (e.g., 'myimage_001' if files are named  like 'myimage_001-0001-model-pb.fits').
    n_parallel : int
        Number of gunzip processes to run in parallel.
    """
    imagelist1 = glob.glob(imagebasename + '-????-model*.fits.gz') # channel maps
    imagelist2 = glob.glob(imagebasename + '-???-model*.fits.gz') # MFS maps
    imagelist = sorted(imagelist1) + sorted(imagelist2)

    def _gunzip(image):
        """
        Decompress one model image in place.

        Parameters
        ----------
        image : str
            Compressed image path.

        Returns
        -------
        None
            The image is decompressed on disk.
        """
        terminal_print('Now gunzip ' + image)
        subprocess.run(['gunzip', '-f', image], check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(_gunzip, image) for image in imagelist]
        for future in futures:
            future.result()
    return


def gzip_model_images(imagebasename, n_parallel=4):
    """
    Gzips all model images with the given base name.
    Parameters
    ----------
    imagebasename : str
        The base filename for model images (e.g., 'myimage_001' if files are named  like
        'myimage_001-0001-model-pb.fits').
    n_parallel : int
        Number of gzip processes to run in parallel.
    """

    imagelist1gz = glob.glob(imagebasename + '-????-model*.fits.gz') # gzip channel maps
    imagelist2gz = glob.glob(imagebasename + '-???-model*.fits.gz') # gzip MFS maps

    # delete any existing compressed files to avoid confusion
    for image in imagelist1gz + imagelist2gz:
        terminal_print('Removing ' + image + ' because it already exists')
        Path(image).unlink(missing_ok=True)

    imagelist1 = glob.glob(imagebasename + '-????-model*.fits') # channel maps
    imagelist2 = glob.glob(imagebasename + '-???-model*.fits') # MFS maps
    imagelist = sorted(imagelist1) + sorted(imagelist2)

    def _gzip(image):
        """
        Compress one model image in place.

        Parameters
        ----------
        image : str
            Image path to compress.

        Returns
        -------
        None
            The image is compressed on disk.
        """
        terminal_print('Now gzip ' + image)
        subprocess.run(['gzip', '-f', image], check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(_gzip, image) for image in imagelist]
        for future in futures:
            future.result()
    return


def fix_fpb_images(modelimagebasename):
    """
    Ensures that a set of "flat primary beam" (fpb) images exists for each corresponding
    "primary beam" (pb) image by copying pb images to fpb filenames if needed.

    Parameters
    ----------
    modelimagebasename : str
        The base filename for model images (e.g., 'myimage' if files are named like
        'myimage-0001-model-pb.fits').

        Behavior
        - Scans for all filenames matching '<basename>-????-model-pb.fits' and
        '<basename>-????-model-fpb.fits'.
        - If the number of pb and fpb images is equal, nothing is done.
        - If no fpb images exist, each pb image is copied to an fpb file by replacing
        the '-model-pb.fits' suffix with '-model-fpb.fits'.
        - If some but not all fpb images exist, an assertion error is raised.

    Notes
    -----
    This function uses `shutil.copy` to perform file copying and prints the copy commands.
    """
    pblist = glob.glob(modelimagebasename + '-????-model-pb.fits')
    fpblist = glob.glob(modelimagebasename + '-????-model-fpb.fits')
    
    if len(pblist) == len(fpblist): return # nothing is needed
    assert len(fpblist) == 0 # if we are here we should not have any fpb images
    for image in pblist:
       terminal_print('copying ' + image + ' ' + image.replace('-model-pb.fits','-model-fpb.fits'))
       shutil.copy(image, image.replace('-model-pb.fits','-model-fpb.fits'))


def mask_region_inv(infilename, ds9region, outfilename):
    """
    Applies an inverse mask to a FITS image using a DS9 region file, setting all pixels outside the specified region to zero.

    Parameters
    ----------
    infilename : str
        Path to the input FITS file.
    ds9region : str
        Path to the DS9 region file defining the region to keep.
    outfilename : str
        Path to the output FITS file where the masked image will be saved.

    Returns
    -------
    None
    """
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    map = hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == False)] = 0.0
    hdu.writeto(outfilename, overwrite=True)
    return


def mask_region(infilename, ds9region, outfilename):
    """
    Applies a mask to a FITS file based on a DS9 region file and writes the result to a new FITS file.

    Parameters
    ----------
    infilename : str
        Path to the input FITS file to be masked.
    ds9region : str
        Path to the DS9 region file specifying the mask region.
    outfilename : str
        Path to the output FITS file where the masked data will be saved.

    Returns
    -------
    None
        The function writes the masked FITS file to `outfilename` and does not return a value.

    Notes
    -----
    - The function sets the pixel values within the specified region to 0.0.
    - Assumes the FITS file and region file are compatible in terms of dimensions and WCS.
    - Requires the `pyregion`, `numpy`, and `astropy.io.fits` packages.
    """
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    map = hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == True)] = 0.0
    hdu.writeto(outfilename, overwrite=True)
    return


def removeneNaNfrommodel(imagenames):
    """
    Replace NaN and Inf pixel values in WSCLEAN model images with zeros.

    Parameters
    ----------
    imagenames : list of str
        List of model image paths.
    """

    for image_id, image in enumerate(imagenames):
        terminal_print('remove NaN/Inf values from model: ', image)
        with fits.open(image) as hdul:
            data = hdul[0].data
            data[np.where(~np.isfinite(data))] = 0.0
            hdul[0].data = data
            hdul.writeto(image, overwrite=True)
    return


def removenegativefrommodel(imagenames):
    """
    Replace negative pixel values in WSCLEAN model images with zeros.

    Parameters
    ----------
    imagenames : list of str
        List of model image paths.
    """

    perseus = False
    A1795 = False
    A1795imlist = sorted(
        glob.glob('/net/nieuwerijn/data2/rtimmerman/A1795_HBA/A1795/selfcal/selfcal_pix0.15_wide-????-model.fits'))

    for image_id, image in enumerate(imagenames):
        terminal_print('remove negatives from model: ', image)
        hdul = fits.open(image)
        data = hdul[0].data

        data[np.where(data < 0.0)] = 0.0
        hdul[0].data = data
        hdul.writeto(image, overwrite=True)
        hdul.close()

        if perseus:
            run('python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/editmodel.py {} /net/ouderijn/data2/rvweeren/PerseusHBA/inner_ring_j2000.reg /net/ouderijn/data2/rvweeren/PerseusHBA/outer_ring_j2000.reg'.format(
                image))
        # run('python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/editmodel.py ' + image)
        if A1795:
            cmdA1795 = 'python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/insert_highres.py '
            cmdA1795 += image + ' '
            cmdA1795 += A1795imlist[image_id] + ' '
            cmdA1795 += '/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/A1795core.reg '
            terminal_print('WSClean model-generation command:', cmdA1795)
            run(cmdA1795)

    return


def checkforzerocleancomponents(imagenames):
    """
    Check if something was cleaned, if not stop de script to avoid more obscure errors later

    Parameters
    ----------
    imagenames : list
        List of model images to check for clean components.

    Returns
    -------
    True if all model images are zero, False otherwise.
    """

    n_images = len(imagenames)
    n_zeros = 0
    for image_id, image in enumerate(imagenames):
        terminal_print("Check if there are non-zero pixels: ", image)
        hdul = fits.open(image)
        data = hdul[0].data
        if not np.any(data):  # this checks if all elements are 0.0
            terminal_print("Model image:", image, "contains only zeros.")
            n_zeros = n_zeros + 1
        hdul.close()
    if n_zeros == n_images:
        return True
    else:
        return False


def groupskymodel(skymodelin, facetfitsfile, skymodelout=None):
    """
    Group sky-model sources according to a facet image.

    Parameters
    ----------
    skymodelin : str
        Input sky-model path.
    facetfitsfile : str
        Facet FITS image path.
    skymodelout : str, optional
        Output sky-model path.

    Returns
    -------
    str
        Output sky-model path.
    """
    import lsmtool  # type: ignore
    terminal_print('Loading:', skymodelin)
    LSM = lsmtool.load(skymodelin)
    LSM.group(algorithm='facet', facet=facetfitsfile)
    if skymodelout is not None:
        LSM.write(skymodelout, clobber=True)
        return skymodelout
    else:
        LSM.write('fits_images/grouped_' + os.path.basename(skymodelin), clobber=True)
        return 'fits_images/grouped_' + os.path.basename(skymodelin)


def findrms(mIn, maskSup=1e-7):
    """
    Find the RMS of an array using the method from Cyril Tasse/kMS.

    Parameters
    ----------
    mIn : numpy.ndarray
        Input data array.
    maskSup : float, optional
        Clipping threshold for zero/masked values.

    Returns
    -------
    float
        Estimated RMS noise.
    """
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)
    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff: break
        rmsold = rms
    return rms


def getimsize(boxfile, cellsize=1.5, increasefactor=1.2, DDE=None):
    """
    Find image size needed to cover a DS9 boxfile region.

    Parameters
    ----------
    boxfile : str
        Path to the DS9 box file.
    cellsize : float
        Pixel cell size in arcseconds.
    increasefactor : float, optional
        Factor to expand the bounding box size.
    DDE : bool, optional
        Direction-dependent calibration mode flag.

    Returns
    -------
    int
        Computed image size in pixels.
    """
    r = pyregion.open(boxfile)

    xs = np.ceil((r[0].coord_list[2]) * increasefactor * 3600. / cellsize)
    ys = np.ceil((r[0].coord_list[3]) * increasefactor * 3600. / cellsize)

    imsize = np.ceil(max([xs, ys]))  # // Round up decimals to an integer
    if (imsize % 2 == 1):
        imsize = imsize + 1

    # if int(imsize) < 512:
    #    imsize = 512
    return int(imsize)
