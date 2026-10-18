"""Source catalogs and facet directions: calibration-error maps and catalogs,
compact-source fluxes, DDE facet directions files and ds9 regions."""
import ast
import glob
import os
import pickle
import shutil
import sys
from pathlib import Path

import astropy.units as units
import bdsf  # type: ignore
import matplotlib
import numpy as np
import pyregion
import scipy.ndimage
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from astropy.wcs import WCS

from submods.fits_utils import flatten
from submods.image_utils import findrms
from submods.plotting import plotimage_astropy
from submods.runtime_utils import get_logger, run, terminal_print

# submods directory, same value as submodpath in main.py
submodpath = os.path.dirname(os.path.abspath(__file__))


def create_homogenized_facetdirections(facetdirections, separateradius=5.0, args=None):
    """
    Create direction files with a common set of nearby unique directions.

    Parameters
    ----------
    facetdirections : list of str
        Input facet-direction file paths.
    separateradius : float, optional
        Matching radius in arcminutes.
    args : dict, optional
        Parsed facetselfcal options, passed on to :func:`parse_facetdirections`.

    Returns
    -------
    None
        Homogenized files are written to the current directory.
    """
    # check that facetdirections is a list of strings, otherwise this function cannot be used
    if isinstance(facetdirections, list) and all(isinstance(facetdirections, str) for facetdirections in facetdirections):
       terminal_print('Creating homogenized facetdirections file with all unique directions from the provided facetdirections files')
    else:
        terminal_print('The facetdirections argument is not a list of strings, so cannot create homogenized facetdirections file')    
  
    # we are going to that this selfcalcycle_start_list as a reference and use it later for all other facetdirection files that are made  
    selfcalcycle_start_list =  parse_facetdirections(facetdirections[0], 0, return_only_selfcalcycle_sel=True, args=args)

    # we want to create a new direction file that contains all unique directions from all facetdirection files
    # here unique means that the directions are not exactly the same, but they are within a certain radius of each other (e.g. 1 arcmin)
    # start with the first file as a reference, and then add directions from the other files that are not within the radius of the reference directions
    dirs, solintslist, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections[0], 1000, args=args)

    # we require that solintslist, smoothness, soltypelist_includedir are not None
    if solintslist is None or smoothness is None or soltypelist_includedir is None:
        terminal_print('We require a direction file that has at least these columns: RA DEC solints smoothness soltypelist_includedir')
        raise ValueError("One or more of the returned values from parse_facetdirections is None")

    # we do not yet care about the solintslist, smoothness, soltypelist_includedir, we worry about direction selection for the DDE solve first, and then we will worry about how to assign solints, smoothness, and soltypelist_includedir to the directions in the homogenized file
    for facetdirection in facetdirections[1:]:
        dirsnew, solintslistnew, smoothnessnew, soltypelist_includedirnew = parse_facetdirections(facetdirection, 1000, args=args)
        # dirsnew is an array of shape (Ndirs, 2) with RA and DEC in radians, we want to check for each direction in dirsnew if it is within a certain radius of any direction in dirs, if not we add it to dirs
       
        dirssky = SkyCoord(dirs[:, 0]*units.radian, dirs[:, 1]*units.radian, frame='icrs')
        dirsnewsky = SkyCoord(dirsnew[:, 0]*units.radian, dirsnew[:, 1]*units.radian, frame='icrs')
        idxc, idxcatalog, d2d, d3d = dirssky.search_around_sky(dirsnewsky, separateradius*units.arcmin)
        # only keep new sources that are not within the separateradius of any source in the current dirs variable
        dirsnewunique = dirsnew[~np.isin(np.arange(len(dirsnew)), idxcatalog)]
   
        #print('Found', len(dirsnewunique), 'new unique directions in facetdirection file', facetdirection)
        if len(dirsnewunique) > 0:
            dirs = np.vstack((dirs, dirsnewunique))
            # add entry to return_only_selfcalcycle_sel so is keeps the same length as dirs
            # return_only_selfcalcycle_sel is an simply 1D numpy array with length as dirs
            # fill the new entrie with the return_only_selfcalcycle_sel[0] value
            selfcalcycle_start_list = np.hstack((selfcalcycle_start_list, np.ones(len(dirsnewunique)) * selfcalcycle_start_list[0]))
            # convert return_only_selfcalcycle_sel to integers array
            selfcalcycle_start_list = selfcalcycle_start_list.astype(int)

    # total "unique" directions found
    terminal_print('Total unique directions found:', len(dirs))

    # Now write len(facetdirections) new direction files with the same directions, but with the solintslist, smoothness, soltypelist_includedir from the original files that were used to create the homogenized direction file. We use the selfcalcycle_start_list that we parsed from the first facetdirection file as a reference for all files.
    # If a direction has no corresponding direction in the original file (i.e. it was added from another file), we put all the soltype_includir entries at False, and put some standar defaults for smoothnes and solintslist    

    for facetdirection in facetdirections:
        dirsorig, solintslistorig, smoothnessorig, soltypelist_includedirorig = parse_facetdirections(facetdirection, 1000, args=args)
        solintslistnew = []
        smoothnessnew = []
        soltypelist_includedirnew = []
        for dir in dirs:
            # check if this dir is in the original dirsorig variable, if so we take the corresponding solintslist, smoothness, soltypelist_includedir from the original file, otherwise we put some default values
            # do this check via the angular separation between the dir and the dirsorig variable, if it is within the separateradius we say it is the same direction
            dirsky = SkyCoord(dir[0]*units.radian, dir[1]*units.radian, frame='icrs')
            dirorigssky = SkyCoord(dirsorig[:, 0]*units.radian, dirsorig[:, 1]*units.radian, frame='icrs')
            seperation = dirsky.separation(dirorigssky)
            # find the index of the original direction that is within the separateradius, if there are multiple we take the closest one, if there are none we put some default values
            idxcatalog = np.where(seperation < separateradius*units.arcmin)[0]
            if len(idxcatalog) > 0:
                if len(idxcatalog) > 1:
                    terminal_print('WARNING: Found multiple directions in the original facetdirection file that are within the separateradius of the direction in the homogenized file, this should not happen, check the original facetdirection file for duplicate directions')
                    raise Exception('Found multiple directions in the original facetdirection file that are within the separateradius of the direction in the homogenized file, this should not happen, check the original facetdirection file for duplicate directions')
                solintslistnew.append(solintslistorig[idxcatalog[0]])
                smoothnessnew.append(smoothnessorig[idxcatalog[0]])
                soltypelist_includedirnew.append(soltypelist_includedirorig[idxcatalog[0]])
            else:
                solintslistnew.append(solintslistorig[0])  # Append default value
                smoothnessnew.append(smoothnessorig[0])  # Append default value
                soltypelist_includedirnew.append([False]*len(soltypelist_includedirorig[0]))  # Append default value


        # now write the new direction file with the same name as the original file, but with _homogenized appended to the name, and with the new dirs, solintslistnew, smoothnessnew, soltypelist_includedirnew variables
        # write file in current working directory
        facetdirectionnew = os.path.basename( (facetdirection).rstrip('.txt') + '_homogenized.txt')
        terminal_print('Writing homogenized facetdirection file', facetdirectionnew, 'with', len(dirs), 'directions')
        # open file for writing

        # convert dirs back to RA and DEC in degrees for writing to file
        dirs_ra_dec = []
        for dir in dirs:
            dirsky = SkyCoord(dir[0]*units.radian, dir[1]*units.radian, frame='icrs')
            dirs_ra_dec.append((dirsky.ra.degree, dirsky.dec.degree))


        with open(facetdirectionnew, 'w') as f:
            # write header of the form "RA DEC solints smoothness soltypelist_includedir"
            f.write("#RA DEC start solints smoothness soltypelist_includedir\n")

            for source_counter, source in enumerate(dirs_ra_dec):
                terminal_print("start", selfcalcycle_start_list[source_counter])
                terminal_print('solints', solintslistnew[source_counter])
                terminal_print('smoothness', smoothnessnew[source_counter])
                terminal_print('soltypelist_includedir', soltypelist_includedirnew[source_counter])
                line = f"{source[0]:.6f} {source[1]:.6f} {selfcalcycle_start_list[source_counter]} " \
                f"[{','.join(str(s) for s in solintslistnew[source_counter])}] " \
                f"[{','.join(str(s) for s in smoothnessnew[source_counter])}] " \
                f"[{','.join(str(i) for i in soltypelist_includedirnew[source_counter])}] " \
                f"  # direction Dir{source_counter:02d}\n"
                f.write(line)


def calibration_error_map(fitsimage, outputfitsfile, kernelsize=31, rebin=None):
    """
    Generate a calibration error map for a radio interferometric image.

    This function estimates calibration artifacts around bright sources using 
    a morphological filtering technique described in Rudnick (2002, PASP, 114, 427).
    It computes the negative of the "Open map", which enhances regions of negative
    emission typically associated with calibration residuals.

    Parameters
    ----------
    fitsimage : str
        Path to the input FITS image file. The file must be a 2D or 4D FITS cube 
        with dimensions compatible with [1,1,NAXIS1,NAXIS2].

    outputfitsfile : str
        Path where the resulting calibration error map (as a FITS file) will be saved.

    kernelsize : int, optional
        Size of the filtering kernel (in pixels) used to compute the morphological
        opening (default is 31). This determines the spatial scale of the filtering.

    rebin : int or None, optional
        If specified, the output image will be rebinned by this factor to reduce 
        resolution and file size. If None, the original resolution is preserved.

    Notes
    -----
    - The algorithm applies a minimum filter followed by a maximum filter to the 
    input image data, effectively performing a morphological opening.
    - The resulting "Open map" is then inverted to emphasize negative features 
    (i.e., residual calibration errors).
    - If rebinning is applied, the header's WCS information is appropriately updated.

    References
    ----------
    Rudnick, L. (2002). "Diffuse radio emission in and around clusters." PASP, 114, 427.

    Returns
    -------
    None
        The output is written directly to a FITS file specified by `outputfitsfile`.
    """

    hdulist = fits.open(fitsimage) 
    header  = hdulist[0].header.copy()

    hduflat = flatten(hdulist)
        
    mins = scipy.ndimage.minimum_filter(hduflat.data, size=(kernelsize, kernelsize))
    openmap = -1.*scipy.ndimage.maximum_filter(mins, size=(kernelsize, kernelsize)) # -1* to make negative errors postive

    if rebin is None:
        hdulist[0].data[0,0,:,:] = openmap 
        hdulist.writeto(outputfitsfile, overwrite=True)
    else:
        from skimage.transform import rescale
        image_data = rescale(openmap, 1./rebin, anti_aliasing=True)
        header['BMAJ'] = header['BMAJ']*float(kernelsize)                                                 
        header['BMIN'] = header['BMIN']*float(kernelsize)                                                 
        header['BPA'] = 0.

        header['NAXIS1'] = openmap.shape[1]
        header['NAXIS2'] = openmap.shape[0]
        header['CRPIX1'] = header['CRPIX1']/float(rebin)                                                  
        header['CDELT1'] = header['CDELT1']*float(rebin) 
        header['CRPIX2'] = header['CRPIX2']/float(rebin)                                                  
        header['CDELT2'] = header['CDELT2']*float(rebin) 
        
        # add back extra 1,1 dimensions [1,1,NAXIS1,NAXIS2]
        image_data = image_data[np.newaxis, np.newaxis, :, :]
        hdu = fits.PrimaryHDU(image_data, header=header)
        hdu.writeto(outputfitsfile, overwrite=True)
    hdulist.close()
    return


def create_calibration_error_catalog(filename, outfile, thresh_pix=7.5, thresh_isl=7.5, interactive=False):
    """
    Creates a calibration error catalog from a given image file using the PyBDSF library.

    Parameters
    ----------
    filename : str
        Path to the input image file to be processed.
    outfile : str
        Path to the output file where the catalog will be saved in FITS format.
    thresh_pix : float, optional
        Pixel threshold for source detection. Default is 7.5.
    thresh_isl : float, optional
        Island threshold for source detection. Default is 7.5.
    interactive : bool, optional
        Open the PyBDSF fit view interactively after creating the catalog.

    Returns
    -------
    None
    """
    if os.path.isfile(outfile):
        Path(outfile).unlink(missing_ok=True)
    empty_catalog = False
    img = bdsf.process_image(filename,mean_map='const', rms_map=True, \
                             rms_box=(35,5), thresh_pix=thresh_pix,thresh_isl=thresh_isl,\
                             advanced_opts=True, minpix_isl=2)
    img.write_catalog(format='fits', outfile=outfile, catalog_type='srl', clobber=True)
    if not os.path.isfile(outfile):
        terminal_print('\033[93m' + 'No sources found, cannot write empty catalog' + '\033[0m')
        empty_catalog = True  
    if interactive:
        matplotlib.use('TkAgg')
        img.show_fit()
        matplotlib.use('Agg')
    del img
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    for f in glob.glob(os.path.dirname(filename) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    return empty_catalog


def get_number_of_sources_in_catalog(catalogfile):
    """
    Returns the number of sources in a given FITS catalog file.

    Parameters
    ----------
    catalogfile : str
        Path to the input catalog file in FITS format.

    Returns
    -------
    int
        Number of sources in the catalog.
    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    hdu_list.close()
    return len(catalog)


def update_calibration_error_catalog(catalogfile, outcatalogfile, distance=20., keep_N_brightest=20, previous_catalog=None, N_dir_max=45, interleave_sorting=True):
    """
    Updates a calibration error catalog by filtering, merging, and removing nearby sources.

    Parameters
    ----------
    catalogfile : str
        Path to the input catalog file in FITS format.
    outcatalogfile : str
        Path to the output catalog file in FITS format.
    distance : float, optional
        Minimum separation distance (in arcminutes) to consider sources as distinct. Default is 20 arcminutes.
    keep_N_brightest : int, optional
        Number of brightest sources to retain from the input catalog. Default is 20.
    previous_catalog : str, optional
        Path to a previous catalog file in FITS format to merge with the current catalog. Default is None.
    N_dir_max : int, optional
        Maximum number of directions (sources) to retain in the final catalog. Default is 45.
    interleave_sorting : bool, optional
        Alternate entries ordered by integrated flux and peak flux when sorting.

    Returns
    -------
    None
        The updated catalog is written to the specified output file.

    Notes
    -----
    - The function sorts the catalog by peak flux and retains the brightest sources.
    - If a previous catalog is provided, it merges the two catalogs, ensuring that entries from the previous catalog are prioritized.
    - Sources that are too close to each other (based on the `distance` parameter) are removed to avoid duplicates or closely separated directions.
    - The final catalog is limited to `N_dir_max` entries if it exceeds this limit.
    - The output catalog is saved in FITS format, overwriting any existing file at the specified path.
    """
    #from astropy.table import Table, vstack
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    #print(catalog.columns)
    hdu_list.close()
        
    # sort catalog on AFLUX flux and Peak_flux
    # first entry highest AFLUX
    # second entry highest Peak_flux
    # third entry second highest AFLUX
    # fourth entry second highest Peak_flux
    # and so on, alternating between AFLUX and Peak_flux

    if interleave_sorting:

         # Sort indices separately
        idx_aflux = catalog.argsort(keys='AFLUX', reverse=True)
        idx_peakflux  = catalog.argsort(keys='Peak_flux', reverse=True)

        # Interleave while avoiding duplicates
        seen = set()
        final_idx = []

        for i in range(max(len(idx_aflux), len(idx_peakflux))):
            if i < len(idx_aflux):
                if idx_aflux[i] not in seen:
                    final_idx.append(idx_aflux[i])
                    seen.add(idx_aflux[i])
            if i < len(idx_peakflux):
                if idx_peakflux[i] not in seen:
                    final_idx.append(idx_peakflux[i])
                    seen.add(idx_peakflux[i])

        # Reorder catalog
        catalog = catalog[final_idx]

    else:
        # sort catalog at Peak flux, brightest first
        idx = catalog.argsort(keys='Peak_flux', reverse=True)
        catalog = catalog[idx]

    terminal_print('Catalog entries before keeping only the ', keep_N_brightest, 'brightest sources. Input:', len(catalog))

    if len(catalog) > keep_N_brightest:
        catalog = catalog[0:keep_N_brightest]
    
    # merge with previous catalog
    if previous_catalog is not None:
         hdu_list_prev = fits.open(previous_catalog)
         catalog_prev = Table(hdu_list_prev[1].data)
         # combine catalogs
         # always put previous entries first in order
         catalog = vstack([catalog_prev, catalog]).copy()
         hdu_list_prev.close()
         
    # remove sources that are too nearby others, and duplicates from merging with previous catalog (if applicable)   
    new_catalog = catalog.copy()
    for source_id, source in enumerate(catalog[:-1]):
        c1 = SkyCoord(source['RA']*units.degree, source['DEC']*units.degree, frame='icrs')
        terminal_print('Finding sources that are less than', distance, 'arcmin close to SOURCE ID', source_id)
        for faintersource_id, faintersource in enumerate(catalog[source_id+1:]): # take only sources with a higher index, skip last one
            c2 = SkyCoord(faintersource['RA']*units.degree, faintersource['DEC']*units.degree, frame='icrs')
            if c1.separation(c2).to(units.arcmin).value < distance:
                terminal_print('Found close source with ID and peak flux', faintersource['Source_id'], faintersource['Peak_flux'], 'distance', c1.separation(c2).to(units.arcmin).value)
                removeidx = np.where((new_catalog['Peak_flux'] == faintersource['Peak_flux']) & (new_catalog['Source_id'] == faintersource['Source_id']))[0] # do a double comparson because the vstack from catalog_prev can merger sources with the same source_id
                assert len(removeidx) <= 1
                if len(removeidx) > 0:
                    new_catalog.remove_row(removeidx[0])
                 
    terminal_print('Catalog entries before / after removal of closely separated directions', len(catalog), len(new_catalog))
    
    # ensure catalog does not go over N_dir_max limit
    if len(new_catalog) > N_dir_max:
        new_catalog = new_catalog[0:N_dir_max]
        terminal_print('Kept only N_dir_max directions:', N_dir_max)
    new_catalog.write(outcatalogfile, format='fits', overwrite=True)


def add_peak_total_flux_to_catalog(catalogfile, fluxcatalogfile, match_radius=1.5):
    """
    Adds peak and total flux values from a flux catalog to an existing catalog based on positional matching.

    Parameters
    ----------
    catalogfile : str
        Path to the input catalog file in FITS format where peak flux values will be added.
    fluxcatalogfile : str
        Path to the flux catalog file in FITS format containing peak flux values.
    match_radius : float, optional
        Maximum separation distance (in arcminutes) for matching sources between the two catalogs. Default is 1.5 arcminutes.

    Returns
    -------
    None
        The updated catalog is saved back to the original `catalogfile`.

    Notes
    -----
    - The function reads both catalogs and checks for the presence of the 'AFLUX' or 'TFLUX' column in the input catalog.
    - If 'AFLUX' or 'TFLUX' is not present, they are added and initialized to zero.
    - For each source in the input catalog, the function searches for sources in the flux catalog within the specified `match_radius`.
    - If multiple matches are found, the source with the highest peak flux is used to update the 'AFLUX/TFLUX' value.
    - The updated catalog is written back to the original file, overwriting it.
    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)

    # add colunn 'AFLUX' if not present to catalog
    if 'AFLUX' not in catalog.colnames:
        # make columns with the same type as Peak_flux
        catalog['AFLUX'] = np.zeros(len(catalog), dtype=catalog['Peak_flux'].dtype)
        terminal_print('Added AFLUX column to catalog')
    hdu_list.close()
    
    # add column 'TFLUX' if not present to catalog
    if 'TFLUX' not in catalog.colnames:
        # make columns with the same type as Total_flux
        catalog['TFLUX'] = np.zeros(len(catalog), dtype=catalog['Total_flux'].dtype)
        terminal_print('Added TFLUX column to catalog')
    hdu_list.close()

    # set all catalog['AFLUX'] and 'TFLUX' to zero because there can be NaNs for new entries
    # this is caused by a vstack in update_calibration_error_catalog() with merges one catalog with and one without the 'AFLUX' column
    catalog['AFLUX'] = 0.0
    catalog['TFLUX'] = 0.0

    hdu_list_flux = fits.open(fluxcatalogfile)
    catalog_flux = Table(hdu_list_flux[1].data)
    hdu_list_flux.close()

    new_catalog = catalog.copy()
    
    if match_radius <= 0.0:
        # just copt over Peak_flux and Total_flux without matching
        new_catalog['AFLUX'] = catalog_flux['Peak_flux']
        new_catalog['TFLUX'] = catalog_flux['Total_flux']
        terminal_print('Match radius is zero or negative, just copied over Peak_flux and Total_flux without matching')
         # save updated catalog
        new_catalog.write(catalogfile, format='fits', overwrite=True)
        return
    for source_id, source in enumerate(catalog):
        # find sources in flux catalog within match_radius arcmin
        c1 = SkyCoord(source['RA']*units.degree, source['DEC']*units.degree, frame='icrs')
        for fluxsource_id, fluxsource in enumerate(catalog_flux):
            c2 = SkyCoord(fluxsource['RA']*units.degree, fluxsource['DEC']*units.degree, frame='icrs')
            # find sources closer than match_radius arcmin
            # from these sources, pick the one with the highest flux
            if c1.separation(c2).to(units.arcmin).value < match_radius:
                #print('Found flux source with ID', fluxsource['Source_id'], 'for error catalog source ID', source['Source_id'])
                #print('Peak flux error catalog / flux source:', catalog['Peak_flux'][source_id], fluxsource['Peak_flux'])
                # only update catalog if Peak_flux is higher than current value
                #print('match found', fluxsource['Peak_flux'],new_catalog['AFLUX'][source_id])
                if fluxsource['Peak_flux'] > new_catalog['AFLUX'][source_id]:
                    new_catalog['AFLUX'][source_id] = fluxsource['Peak_flux']
                    new_catalog['TFLUX'][source_id] = fluxsource['Total_flux']
                    #print('CC', new_catalog['AFLUX'][source_id],fluxsource['Peak_flux'])
        terminal_print('-- Set AFLUX to', new_catalog['AFLUX'][source_id], 'for source', source_id, 'Matching radius (arcmin)', match_radius ,'--')
        terminal_print('-- Set TFLUX to', new_catalog['TFLUX'][source_id], 'for source', source_id, 'Matching radius (arcmin)', match_radius ,'--')
                    
    # save updated catalog
    new_catalog.write(catalogfile, format='fits', overwrite=True)
    return


def filter_catalog_on_flux(catalogfile, freq, telescope, min_peakflux=0.02):
    """
    Remove catalog sources below the frequency-scaled flux threshold.

    Parameters
    ----------
    catalogfile : str
        FITS catalog path.
    freq : float
        Frequency in Hz used for threshold scaling.
    telescope : str
        Telescope name controlling whether filtering applies.
    min_peakflux : float, optional
        Reference minimum peak flux in Jy.

    Returns
    -------
    None
        The catalog is filtered in memory and subsequent facet files are
        updated by the caller.
    """
    # use AFLUX and TFLUX columns for filtering
    
    if telescope != 'LOFAR' and telescope != 'MeerKAT':
        return
    
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    hdu_list.close()
    new_catalog = catalog.copy()
    for source_id, source in enumerate(catalog):
        if source['AFLUX'] < min_peakflux*((freq/1.3e9)**(-0.7)) and \
           (source['AFLUX']*((source['TFLUX']/ source['AFLUX'])**(0.4))) <  min_peakflux*((freq/1.3e9)**(-0.7)):
            
            terminal_print('Removing source with ID', source['Source_id'], 'from catalog, AFLUX:', source['AFLUX'], 'below min_peakflux threshold:', min_peakflux*((freq/1.278e9)**(-0.7)))
            removeidx = np.where((new_catalog['Peak_flux'] == source['Peak_flux']) & (new_catalog['Source_id'] == source['Source_id']))[0] # do a double comparson because the vstack from catalog_prev can merger sources with the same source_id
            assert len(removeidx) <= 1
            if len(removeidx) > 0:
                new_catalog.remove_row(removeidx[0])
    
    terminal_print('Catalog entries before / after filtering on peak flux', len(catalog), len(new_catalog))
    
    # check that there are at least two sources left
    if len(new_catalog) < 2:
        # print in red text
        terminal_print('\033[91m' + 'Warning: less than two sources left in catalog after filtering on peak flux' + '\033[0m')
        terminal_print('Cannot proceed with less than two directions, run with user specified facetdirections file')
        sys.exit(1)
    
    new_catalog.write(catalogfile, format='fits', overwrite=True)
    return


def add_bright_source_to_catalog(catalogfile, fluxcatalogfile, freq):
    """
    Add the brightest source from the fluxcatalogfile to the catalogfile to ensure very bright sources are included as directions

    Parameters
    ----------
    catalogfile : str
        Path to the input catalog file in FITS format where a new source will be added.
    fluxcatalogfile : str
        Path to the flux catalog file in FITS format containing potential new sources.
    freq : float
        Observing frequency in hertz, used to set the bright-source flux threshold.

    Returns
    -------
    None
        The updated catalog is saved back to the original `catalogfile`.

    Notes
    -----
    - The function reads both catalogs and checks for the presence of very bright sources in the flux catalog.
    - All sources found that meet this criterion are added to the input catalog.
    - The updated catalog is written back to the original file, overwriting it.
    """
    # determine threshold based on frequency
    if freq > 1e9:
       threshold = 0.1*(freq/1e9)**(-0.7)
    elif freq > 300e6:
       threshold = 1.0*(freq/700e6)**(-0.7)
    elif freq > 100e6:
       threshold = 2.0*(freq/150e6)**(-0.7)
    elif freq > 30e6:      
       threshold = 5.0*(freq/50e6)**(-0.7)
    elif freq <= 30e6:
       threshold = 20.0*(freq/20e6)**(-0.7)   

    # sort flux catalog on Peak_flux, with the brightest first
    hdu_list_flux = fits.open(fluxcatalogfile)
    catalog_flux = Table(hdu_list_flux[1].data)
    idx = catalog_flux.argsort(keys='Peak_flux', reverse=True)
    catalog_flux = catalog_flux[idx]
    hdu_list_flux.close()

    # open catalog to which we will add sources
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    hdu_list.close()
    new_catalog = catalog.copy()
    for fluxsource_id, fluxsource in enumerate(catalog_flux):
        if fluxsource['Peak_flux'] > threshold:
            terminal_print('Adding bright source with ID', fluxsource['Source_id'], 'to error catalog, Peak_flux:', fluxsource['Peak_flux'])
            new_catalog.add_row(fluxsource)
    return


def add_source_to_catalog(catalogfile, fluxcatalogfile, distance=20.):
    """
    Add a source from the fluxcatalogfile to the catalogfile if no sources are found
    in the catalogfile. The new source should be at least distance arcmin away from all
    existing sources.

    Parameters
    ----------
    catalogfile : str
        Path to the input catalog file in FITS format where a new source will be added.
    fluxcatalogfile : str
        Path to the flux catalog file in FITS format containing potential new sources.
    distance : float, optional
        Minimum separation distance (in arcminutes) to consider the new source as distinct. Default is 20 arcminutes.

    Returns
    -------
    None
        The updated catalog is saved back to the original `catalogfile`.

    Notes
    -----
    - The function reads both catalogs and checks if the input catalog is empty.
    - It searches the flux catalog for a source that is at least `distance` arcminutes away from all existing sources.
    - The first source found that meets this criterion is added to the input catalog.
    - The updated catalog is written back to the original file, overwriting it.
    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    hdu_list.close()
    # create copy of catalog to which we will add a source
    new_catalog = catalog.copy()
    hdu_list_flux = fits.open(fluxcatalogfile)
    catalog_flux = Table(hdu_list_flux[1].data)
    # sort flux catalog on Peak_flux, with the brightest first
    idx = catalog_flux.argsort(keys='Peak_flux', reverse=True)
    catalog_flux = catalog_flux[idx]
    hdu_list_flux.close()
    for fluxsource_id, fluxsource in enumerate(catalog_flux):
        c2 = SkyCoord(fluxsource['RA']*units.degree, fluxsource['DEC']*units.degree, frame='icrs')
        too_close = False
        for source_id, source in enumerate(catalog):
            c1 = SkyCoord(source['RA']*units.degree, source['DEC']*units.degree, frame='icrs')
            if c1.separation(c2).to(units.arcmin).value < distance:
                too_close = True
                break
        if not too_close:
            terminal_print('Adding source with ID', fluxsource['Source_id'], 'to error catalog, it is at least', distance, 'arcmin away from all existing sources')
            new_catalog.add_row(fluxsource)
            # update the Peak_flux and to the new source setting it to a low value
            new_catalog['Peak_flux'][-1] = 1e-3 # 1 mJy, should not trigger bright source criteria
            break
    new_catalog.write(catalogfile, format='fits', overwrite=True)
    return


def getregionsize(regionfile):
    """
    Extract size of a DS9 region in degrees.

    Parameters
    ----------
    regionfile : str
        Path to the DS9 region file.

    Returns
    -------
    float
        Region size in degrees.
    """
    r = pyregion.open(regionfile)

    if len(r[:]) > 1:
        terminal_print('Only one region can be specified, your file contains', len(r[:]))
        raise Exception('Only one region can be specified, your file contains')

    if r[0].name not in ['box', 'circle', 'ellipse']:
        terminal_print('Only box, circle, or ellipse region supported', r[0].name)
        raise Exception('Only box, circle, or ellipse region supported')

    if r[0].name == 'box':
        boxsizex = r[0].coord_list[2]
        boxsizey = r[0].coord_list[3]

    if r[0].name == 'circle':
        boxradius = r[0].coord_list[2]
        boxsizex = boxradius * 2.
        boxsizey = boxradius * 2.

    if r[0].name == 'ellipse':
        boxradius1 = r[0].coord_list[2] # semi-major axis in the x-direction if angle=0
        boxradius2 = r[0].coord_list[3] # semi-minor axis in the y-direction if angle=0
        boxsizex = boxradius1 * 2.
        boxsizey = boxradius2 * 2.

    return max([boxsizex, boxsizey])


def split_facetdirections(facetregionfile):
    """
    Split a facet region file into individual region files for each facet.

    This function reads a region file containing multiple facet definitions and creates
    separate region files for each individual facet. Each output file is named
    'facet<N>.reg' where N is the facet index.

    Parameters
    ----------
    facetregionfile : str
        Path to the input region file containing multiple facet
        definitions that can be parsed by pyregion.

    Returns
    -------
    None
        The function writes output files to disk but does not return a value.

    Notes
    -----
    Side Effects:
    Creates multiple region files in the current working directory, one for each
    facet found in the input file. Files are named as 'facet0.reg', 'facet1.reg', etc.

    Examples
    --------
    >>> split_facetdirections('all_facets.reg')
    # Creates facet0.reg, facet1.reg, facet2.reg, etc.

    """
    # split of directory and filename to get the directory for the output files
    dirofinput = os.path.dirname(facetregionfile)
    if dirofinput == '':
        dirofinput = '.'  # if the input file is in the current directory, set dirofinput to '.'
    # write output files to the same directory as the input file
    r = pyregion.open(facetregionfile)
    for facet_id, facet in enumerate(r):
        r[facet_id:facet_id + 1].write(dirofinput + '/facet' + str(facet_id) + '.reg')
    return


def create_facet_directions(imagename, selfcalcycle, targetFlux=1.0, ms=None, imsize=None,
                            pixelscale=None, numClusters=0, weightBySize=False,
                            facetdirections=None, imsizemargin=100, restart=False,
                            via_h5=False, h5=None, args=None):
    """
    Create a facet region file based on an input image or file provided by the user.

    If there is an image, uses lsmtool tessellation algorithm to generate facet directions.
    This function also returns the solints and smoothness obtained from the facetdirections 
    file (if available). It is up to the calling function to utilize these values or not.

    Parameters
    ----------
    imagename : str
        Base name of the input image file or path to a skymodel file.
    selfcalcycle : int
        Current self-calibration cycle number. Processing only occurs when cycle is 0
        unless facetdirections is provided.
    targetFlux : float, optional
        Target flux in Jy for tessellation algorithm. Default is 1.0.
    ms : str, optional
        Path to the measurement set. Required for generating DS9 region files.
    imsize : int, optional
        Image size in pixels. Required for generating DS9 region files.
    pixelscale : float, optional
        Pixel scale in arcseconds. Required for generating DS9 region files.
    numClusters : int, optional
        Number of clusters for grouping algorithm. If 0, uses tessellation. Default is 0.
    weightBySize : bool, optional
        Whether to weight tessellation by source size. Default is False.
    facetdirections : str, optional
        Path to existing facet directions file (text format or pickle). If provided,
        skips image-based facet generation.
    imsizemargin : int, optional
        Margin to add to image size (not currently used in function body). Default is 100.
    restart : bool, optional
        If True, preserves existing ./facet_regions/facets.reg file when facetdirections is provided.
        Default is False.
    via_h5 : bool, optional
        If True, uses h5 file path for quick DS9 facet generation and returns early.
        Default is False.
    h5 : str, optional
        Path to h5 file when via_h5 is True.
    args : dict, optional
        Parsed facetselfcal options, passed on to :func:`parse_facetdirections`.

    Returns
    -------
    solints : int or None
        Solution intervals parsed from facetdirections file, or None if not available.
    smoothness : float or None
        Smoothness parameter parsed from facetdirections file, or None if not available.
    soltypelist_includedir : list or None
        Solution type list parsed from facetdirections file, or None if not available.
        None
        Returns None when via_h5 is True (early return).

    Notes
    -----
    - Generates './facet_regions/facetdirections.p' pickle file containing patch positions array.
    - Generates './facet_regions/facets.reg' DS9 region file when ms, imsize, and pixelscale are provided.
    - When selfcalcycle == 0 and no facetdirections file is provided, runs PyBDSF on
    the image and uses lsmtool for source grouping.
    - Patch positions are stored as [RA, Dec] in radians.
    """

    if via_h5: # this is quick call to ds9facetgenerator using an h5 file and a None return
        cmd = f'python {submodpath}/ds9facetgenerator.py '
        cmd += '--ms=' + ms + ' --h5=' + h5 + ' --DS9regionout=facet_regions/facets.reg '
        cmd += '--imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
        run(cmd)
        return

    solints = None  # initialize, if not filled then this is not used here and the settings are taken from facetselfcal argsparse
    smoothness = None  # initialize, if not filled then this is not used here and the settings are taken from facetselfcal argsparse
    soltypelist_includedir = None  # initialize
    if facetdirections is not None:
        if isinstance(facetdirections, str): # case 1 if facetselfcal is a string
            try:
                PatchPositions_array, solints, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, selfcalcycle, args=args)
            except:
                try:
                    f = open(facetdirections, 'rb')
                    PatchPositions_array = pickle.load(f)
                    f.close()
                except:
                    raise Exception('Trouble read file format:' + facetdirections)
        elif isinstance(facetdirections, list): # case 2 if facetselfcal is a list of strings (e.g. from glob)
            solints_out = [] # list of solints for each facetdirections file
            smoothness_out = [] # list of smoothness for each facetdirections file
            soltypelist_includedir_out = [] # list of soltypelist_includedir for each facetdirections file
            for facetdirections_id, facetdirections_file in enumerate(facetdirections): # this means we have N facetdirections files, with N the number of MS provided (=mslist)
                PatchPositions_array, solints_in, smoothness_in, soltypelist_includedir_in = parse_facetdirections(facetdirections_file, selfcalcycle, args=args)
                # append solints, smoothness, and soltypelist_includedir to lists (one entry per facetdirections file)
                # so the dimensions becomes N (=length of list soltypes), M (=number of facetdirections files) x L (number of dirctions in each facetdirections file)
                # note that parse_facetdirections return solints with dimensions L x N, smoothness with dimensions L X N, and soltypelist_includedir with dimensions L X N
                if solints_in is not None: # append in this for loop after we get into the second itteration of the for loop
                    terminal_print(
                        'Output solution-interval type:',
                        type(solints_out),
                        'Input solution-interval type:',
                        type(solints_in),
                    )
                    solints_out.append(solints_in)
                if smoothness_in is not None:
                    smoothness_out.append(smoothness_in)
                if soltypelist_includedir_in is not None:
                    soltypelist_includedir_out.append(soltypelist_includedir_in)   

            # first convert from nested list to array
            # this gives dimensions M (number of facetdirections) X L (number of directions in each facetdirections file) x N (number of soltypes)
            solints_out = np.array(solints_out)
            smoothness_out = np.array(smoothness_out)
            soltypelist_includedir_out = np.array(soltypelist_includedir_out)
            # now swap the axis axis0==>axis1, axis1==>axis2, and axis2==>axis0
            # first swap axis0 and axis2
            solints_out = np.swapaxes(solints_out, 0, 2) # (M, L, N) ==> (N, L, M)
            smoothness_out = np.swapaxes(smoothness_out, 0, 2) # (M, L, N) ==> (N, L, M)
            soltypelist_includedir_out = np.swapaxes(soltypelist_includedir_out, 0, 2) # (M, L, N) ==> (N, L, M)
            # now swap axis1 and axis2
            solints = np.swapaxes(solints_out, 1, 2) # (N, L, M) ==> (N, M, L)
            smoothness = np.swapaxes(smoothness_out, 1, 2) # (N, L, M) ==> (N, M, L)
            soltypelist_includedir = np.swapaxes(soltypelist_includedir_out, 1, 2) # (N, L, M) ==> (N, M, L)
        else:
            terminal_print('facetdirections should be a string or a list of strings, not {}'.format(type(facetdirections)))
            raise Exception('facetdirections should be a string or a list of strings, not {}'.format(type(facetdirections)))


        terminal_print('Facet patch positions:', PatchPositions_array)
        # write new facetdirections.p file
        if os.path.isfile('facet_regions/facetdirections.p'):
            Path('facet_regions/facetdirections.p').unlink(missing_ok=True)
        f = open('facet_regions/facetdirections.p', 'wb')
        pickle.dump(PatchPositions_array, f)
        f.close()

        # generate polygon composite regions file for WSClean imaging
        # in case of a restart this is not done, and the old facet_regions/facets.reg is kept
        # using the old facet_regions/facets.reg is important in case we change the number of facet directions, so the that first image after the restart is done with the old directions (and the h5file used was made using that)
        if ms is not None and imsize is not None and pixelscale is not None and not restart:
            cmd = f'python {submodpath}/ds9facetgenerator.py '
            cmd += '--ms=' + ms + ' --DS9regionout=facet_regions/facets.reg '
            cmd += '--h5=facet_regions/facetdirections.p --imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
            run(cmd)
        return solints, smoothness, soltypelist_includedir
    elif selfcalcycle == 0:
        # Only run this if selfcalcycle==0 [elif]
        # Try to load previous facet_regions/facetdirections.skymodel
        import lsmtool  # type: ignore
        if 'skymodel' not in imagename:
            img = bdsf.process_image(imagename + str(selfcalcycle).zfill(3) + '-MFS-image.fits', mean_map='zero',
                                     rms_map=True, rms_box=(160, 40))
            img.write_catalog(format='bbs', bbs_patches=None, outfile='facet_regions/facetdirections.skymodel', clobber=True)
            del img
            # move all *pybdsf.log files to a logs directory
            if not os.path.isdir('logs'):
                os.mkdir('logs')
            for f in glob.glob(os.path.dirname(imagename + str(selfcalcycle).zfill(3) + '-MFS-image.fits') + '/*pybdsf.log'):
                if os.path.isfile('logs/' + os.path.basename(f)):
                    os.remove('logs/' + os.path.basename(f))
                shutil.move(f, 'logs/')
        else:
            shutil.copy(imagename, 'facet_regions/facetdirections.skymodel')
        LSM = lsmtool.load('facet_regions/facetdirections.skymodel')

        if numClusters > 0:
            LSM.group(algorithm='cluster', numClusters=numClusters)
        else:
            LSM.group(algorithm='tessellate', targetFlux=str(targetFlux) + ' Jy', weightBySize=weightBySize)

        terminal_print('Number of directions', len(LSM.getPatchPositions()))
        PatchPositions = LSM.getPatchPositions()

        PatchPositions_array = np.zeros((len(LSM.getPatchPositions()), 2))

        for patch_id, patch in enumerate(PatchPositions.keys()):
            PatchPositions_array[patch_id, 0] = PatchPositions[patch][0].to(units.rad).value  # RA
            PatchPositions_array[patch_id, 1] = PatchPositions[patch][1].to(units.rad).value  # Dec
        # Run code below for if and elif
        if os.path.isfile('facet_regions/facetdirections.p'):
            Path('facet_regions/facetdirections.p').unlink(missing_ok=True)
        f = open('facet_regions/facetdirections.p', 'wb')
        pickle.dump(PatchPositions_array, f)
        f.close()

        # generate polygon composite regions file for WSClean imaging
        if ms is not None and imsize is not None and pixelscale is not None:
            cmd = f'python {submodpath}/ds9facetgenerator.py '
            cmd += '--ms=' + ms + ' --DS9regionout=facet_regions/facets.reg '
            cmd += '--h5=facet_regions/facetdirections.p --imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
            run(cmd)
        return solints, smoothness, soltypelist_includedir
    else:
        return solints, smoothness, soltypelist_includedir


def write_ds9_regions(ra_array, dec_array, filename="directions.reg", radius=120.0, color="red"):
    """
    Write DS9 region file entries for each RA, DEC pair.

    Parameters
    ----------
    ra_array : array-like
        Right Ascension values in degrees.
    dec_array : array-like
        Declination values in degrees.
    filename : str
        Output DS9 region file path.
    radius : str, optional
        Region radius specification (default is '100arcsec').
    color : str, optional
        Region color (default is 'white').
    """
    with open(filename, "w") as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1\n')
        f.write('fk5\n')
        for i, (ra, dec) in enumerate(zip(ra_array, dec_array)):
            f.write(f"circle({ra},{dec},{radius:.3f}\") # color={color} text={{Dir{i:02d}}}\n")


def parse_facetdirections(facetdirections, selfcalcycle, writeregioncircles=True, return_only_selfcalcycle_sel=False,
                          args=None):
    """
    Parse a facet directions file and return selected positions and optional parameters.
    This function reads a file containing facet directions (RA/DEC positions) and optional
    parameters for self-calibration. It filters directions based on the current selfcal cycle
    and returns arrays/lists suitable for further processing.
    Parameters
    ----------
    facetdirections : str
        Path to the facet directions file. The file should be in ASCII format with a 
        commented header line. Required columns are 'RA' and 'DEC'. Optional columns 
        include 'start', 'solints', 'soltypelist_includedir', and 'smoothness'.
    selfcalcycle : int
        Current self-calibration cycle number. Only directions with start <= selfcalcycle
        will be selected.
    writeregioncircles : bool, optional
        If True, write DS9 region file with facet center positions (default: True).
    return_only_selfcalcycle_sel : bool, optional
        If True, only return directions selected for the current selfcal cycle (default: False).
    args : dict, optional
        Parsed facetselfcal options. If given, the per-direction lists are
        validated against ``args['soltype_list']``. Default is None (no check).
    Returns
    -------
    PatchPositions_array : np.ndarray
        Array of shape (N, 2) containing RA and DEC positions in radians for selected
        directions.
    solints : list of list or None
        List of solution intervals for each selected direction, parsed from string 
        representation. None if 'solints' column not present in file.
    smoothness : list of list or None
        List of smoothness parameters for each selected direction, parsed from string
        representation. None if 'smoothness' column not present in file.
    start : np.ndarray 
        If return_only_selfcalcycle_sel is True, returns only the 'start' values for the selected directions as a numpy array. Otherwise, this is not returned.
    Raises
    ------
    ValueError
    If the number of entries in 'solints', 'smoothness', or 'soltypelist_includedir'
    does not match the length of args['soltype_list'].
    Notes
    -----
    - The function strips inline comments from the input file before parsing.
    - Full-line comments (starting with '#') are skipped except for the header.
    - If ``args`` is given, solints, smoothness and soltypelist_includedir are
      checked against the length of ``args['soltype_list']``.
    - Uses astropy.io.ascii for reading the table data.
    - Coordinates are converted from degrees to radians in the output array.
    """

    # Preprocess the file to strip inline comments
    clean_lines = []
    with open(facetdirections, 'r') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if i == 0:
                # Always keep the first line (commented header)
                clean_lines.append(stripped)
                continue
            if stripped.startswith('#'):
                continue  # skip other full-line comments
            if '#' in stripped:
                stripped = stripped.split('#')[0].strip()  # remove inline comments
            if stripped:
                clean_lines.append(stripped)

    # Read the cleaned data using astropy
    cleaned_data = '\n'.join(clean_lines)
    
    data = ascii.read(cleaned_data, format='commented_header', comment="\\s*#")
    ra, dec = data['RA'], data['DEC']


    try:
        start = data['start']
    except KeyError:
        start = np.zeros(len(ra))
    try:
        solints = data['solints']
    except KeyError:
        solints = None
    try:
        soltypelist_includedir = data['soltypelist_includedir']
    except KeyError:
        soltypelist_includedir = None
    try:
        smoothness = data['smoothness']
    except KeyError:
        smoothness = None

    if return_only_selfcalcycle_sel:
        return np.array(start)

    if writeregioncircles:
        write_ds9_regions(ra, dec, filename="facet_regions/facet_centers.reg", radius=120.0, color="red")
    # Only select ra/dec which are if they are in the selfcalcycle range
    a = np.where((start <= selfcalcycle))[0]
    rasel = ra[a]
    decsel = dec[a]
  
    if soltypelist_includedir is not None:
        soltypelist_includedir_sel_tmp = soltypelist_includedir[a]
        # create 2D array booleans
        soltypelist_includedir_sel = np.zeros((len(rasel), soltypelist_includedir_sel_tmp[0].count(',') + 1), dtype=bool)
        for dir_id in range(len(rasel)):
            # print(dir_id, soltypelist_includedir_sel_tmp[dir_id])
            soltypelist_includedir_sel[dir_id, :] = ast.literal_eval(soltypelist_includedir_sel_tmp[dir_id])
        soltypelist_includedir_sel = soltypelist_includedir_sel.tolist()

    PatchPositions_array = np.zeros((len(rasel), 2))
    PatchPositions_array[:, 0] = (rasel * units.deg).to(units.rad).value
    PatchPositions_array[:, 1] = (decsel * units.deg).to(units.rad).value

    # check for consistency of solints having the same number of entries as args['soltype_list']
    if solints is not None:
        for solint in solints:
            if args is not None and len(ast.literal_eval(solint)) != len(args['soltype_list']):
                terminal_print('Number of entries for solints in the direction file is', \
                      len(ast.literal_eval(solint)), 'but args["soltype_list"] has:', len(args['soltype_list']))
                raise ValueError('The number of solints in the directions file does not match the number of soltypes in args["soltype_list"]. '
                                 'Please check the directions file.')
    # check for consistency of smoothness having the same number of entries as args['soltype_list']
    if smoothness is not None:
        for sm in smoothness:
            if  args is not None and len(ast.literal_eval(sm)) != len(args['soltype_list']):
                terminal_print('Number of entries for smoothness in the direction file is', \
                      len(ast.literal_eval(sm)), 'but args["soltype_list"] has:', len(args['soltype_list']))
                raise ValueError('The number of smoothness in the directions file does not match the number of soltypes in args["soltype_list"]. '
                                 'Please check the directions file.')
    # check for consistency of soltypelist_includedir having the same number of entries as args['soltype_list']
    if soltypelist_includedir is not None:
        for soltypelist in soltypelist_includedir:
            if args is not None and len(ast.literal_eval(soltypelist)) != len(args['soltype_list']):
                terminal_print('Number of entries for soltypelist_includedir in the direction file is', \
                      len(ast.literal_eval(soltypelist)), 'but args["soltype_list"] has:', len(args['soltype_list']))
                raise ValueError('The number of soltypelist_includedir in the directions file does not match the number of soltypes in args["soltype_list"]. '
                                 'Please check the directions file.')            
    
    # Case for smoothness is not set in the direction file
    if solints is not None and smoothness is None:
        solintsel = solints[a]
        if soltypelist_includedir is not None:
            return PatchPositions_array, [ast.literal_eval(solint) for solint in solintsel], None, soltypelist_includedir_sel
        else:
            return PatchPositions_array, [ast.literal_eval(solint) for solint in solintsel], None, None
    if solints is None and smoothness is None:
        if soltypelist_includedir is not None:
            return PatchPositions_array, None, None, soltypelist_includedir_sel
        else:
            return PatchPositions_array, None, None, None

    # Case for smoothness is set in the direction file
    if solints is not None and smoothness is not None:
        solintsel = solints[a]
        smoothnesssel = smoothness[a]
        if soltypelist_includedir is not None:
            return PatchPositions_array, [ast.literal_eval(solint) for solint in solintsel], [ast.literal_eval(sm) for sm in smoothnesssel], soltypelist_includedir_sel
        else:
            return PatchPositions_array, [ast.literal_eval(solint) for solint in solintsel], [ast.literal_eval(sm) for sm in smoothnesssel], None
    if solints is None and smoothness is not None: 
        smoothnesssel = smoothness[a]
        if soltypelist_includedir is not None:
            return PatchPositions_array, None, [ast.literal_eval(sm) for sm in smoothnesssel], soltypelist_includedir_sel
        else:
            return PatchPositions_array, None, [ast.literal_eval(sm) for sm in smoothnesssel], None


def write_compactsource_flux(fitsimage, outputcatalog, interactive=False):
    """
    Processes a FITS image to detect compact sources and writes their flux information to an output catalog.

    Parameters
    ----------
    fitsimage : str
        Path to the input FITS image file.
    outputcatalog : str
        Path to the output catalog file where detected source fluxes will be saved.
    interactive : bool, optional
        Open the PyBDSF fit view interactively after processing the image.

    Returns
    -------
    None

    Notes
    -----
    - The function reads the FITS header to determine beam size and pixel size, then calculates appropriate box sizes for background RMS estimation.
    - Uses the PyBDSF (bdsf) library to process the image and extract source information.
    - The output catalog is written in FITS format and contains a source list with flux measurements.
    """
    with fits.open(fitsimage) as hdul:
        bmaj = hdul[0].header['BMAJ']
        bmin = hdul[0].header['BMIN']
        avgbeam = 3600. * 0.5 * (bmaj + bmin)
        pixsize = 3600. * (hdul[0].header['CDELT2'])
        # use small rmsbox to pick up small scale noise variations so that extened emission is not picked up
        rmsbox1 = int(7. * avgbeam / pixsize)
        rmsbox2 = int((rmsbox1 / 10.) + 1.) 
    img = bdsf.process_image(fitsimage, mean_map='zero', rms_map=True, rms_box=(rmsbox1, rmsbox2))
    img.write_catalog(format='fits', outfile=outputcatalog, catalog_type='srl', clobber=True)
    if interactive:
        matplotlib.use('TkAgg')
        img.show_fit()
        matplotlib.use('Agg')
    del img
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    
    for f in glob.glob(os.path.dirname(fitsimage) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    return


def determine_compactsource_flux(fitsimage):
    """
    Return total flux in compact sources in the FITS image.

    Parameters
    ----------
    fitsimage : str
        Path to the FITS image.

    Returns
    -------
    float
        Total compact source flux in Jy.
    """

    with fits.open(fitsimage) as hdul:
        bmaj = hdul[0].header['BMAJ']
        bmin = hdul[0].header['BMIN']
        avgbeam = 3600. * 0.5 * (bmaj + bmin)
        pixsize = 3600. * (hdul[0].header['CDELT2'])
        rmsbox1 = int(7. * avgbeam / pixsize)
        rmsbox2 = int((rmsbox1 / 10.) + 1.)

        img = bdsf.process_image(fitsimage, mean_map='zero', rms_map=True, rms_box=(rmsbox1, rmsbox2))
        total_flux_gaus = np.copy(img.total_flux_gaus)

    # trying to reset.....
    del img
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    for f in glob.glob(os.path.dirname(fitsimage) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    return total_flux_gaus


def merge_catalogs(cataloglist, outputcatalog):
    """
    Merge multiple FITS catalogs into a single catalog.

    Parameters
    ----------
    cataloglist : list of str
        List of paths to the input FITS catalog files.
    outputcatalog : str
        Path to the output merged catalog file.

    Returns
    -------
    None

    Notes
    -----
    - The function reads each input catalog, combines their data, and writes the merged data to the output catalog.
    - The output catalog is written in FITS format.
    """
    for catalog_id, catalogfile in enumerate(cataloglist):
        with fits.open(catalogfile) as hdul:
            if catalog_id == 0:
                merged_catalog = Table(hdul[1].data)
            else:
                catalog = Table(hdul[1].data)
                merged_catalog = vstack([merged_catalog, catalog])
    merged_catalog.write(outputcatalog, format='fits', overwrite=True)
    return


def MeerKAT_autodetect_highDR(fitsimage):
    """
    Determine if high dynamic range settings are needed for MeerKAT imaging.

    Evaluation is based on compact source fluxes within primary beam FWHM / 2.

    Parameters
    ----------
    fitsimage : str
        Path to the FITS image.

    Returns
    -------
    bool
        True if high dynamic range settings should be used.
    """
    outputfluxcatalog = fitsimage.replace('.fits', '_compactsource_fluxcatalog.DRcheck.fits')
    # get image frequency
    with fits.open(fitsimage, ignore_missing_end=True) as hdu:
        freq = hdu[0].header['CRVAL3']
    # determine FWHM of the MeerKAT primary beam at this frequency
    FWHM_PB = (57.5/60)*(1.5e9/freq) # in degrees
    # create compact source catalog from which we can get peak fluxes
    write_compactsource_flux(fitsimage, outputfluxcatalog)

    # determine if high dynamic range is needed based on the compact source fluxes within the primary beam FWHM
    hdu_list = fits.open(outputfluxcatalog)
    # sort this on peak flux
    catalog = Table(hdu_list[1].data)
    hdu_list.close()
    idx = catalog.argsort(keys='Peak_flux', reverse=True)
    catalog = catalog[idx]

    # check that the brightest source is above 0.05 Jy
    if (catalog['Peak_flux'][0]) < 0.05*((freq/1e9)**(-0.7)):
        terminal_print('=== Brightest source is below 0.05 Jy, no high DR settings needed ===')
        get_logger().info('Brightest source is below 0.05 Jy, no high DR settings needed')
        return False, catalog['Peak_flux'][0]  # no high DR settings needed
    terminal_print('HERE')
    # check that the brightest source is within the primary beam FWHM/4
    ra_center = fits.getheader(fitsimage)['CRVAL1']
    dec_center = fits.getheader(fitsimage)['CRVAL2']
    ra_source = catalog['RA'][0]
    dec_source = catalog['DEC'][0]
    # compute angular separation of the brightest source to the image center
    c1 = SkyCoord(ra=ra_center*units.degree, dec=dec_center*units.degree, frame='icrs')
    c2 = SkyCoord(ra=ra_source*units.degree, dec=dec_source*units.degree, frame='icrs')
    separation = c1.separation(c2).degree
    if separation < (FWHM_PB / 4.): # high dynamic range might be needed
        # now loop over all other sources and check that their peak flux is less than 30% of the brightest source
        brightest_flux = catalog['Peak_flux'][0]
        for source_id in range(1, len(catalog)):
            source_flux = catalog['Peak_flux'][source_id]
            if source_flux > (0.3 * brightest_flux): # so we have another bright source
                # check if this source is outside the FWHM_PB / 4.
                ra_source = catalog['RA'][source_id]
                dec_source = catalog['DEC'][source_id]
                c2 = SkyCoord(ra=ra_source*units.degree, dec=dec_source*units.degree, frame='icrs')
                separation = c1.separation(c2).degree
                if separation > (FWHM_PB / 4.): # we have found a bright source outside the FWHM/4      
                    terminal_print('=== Found multiple bright sources, cannot use high DR settings ===')
                    terminal_print('Brightest source flux:', brightest_flux, 'Jy, 2nd source flux:', source_flux, 'Jy')
                    terminal_print('Brightest source position:', ra_source, dec_source)
                    terminal_print('Other source position:', ra_source, dec_source)
                    get_logger().info('Found multiple bright sources, cannot use high DR settings')
                    return False, catalog['Peak_flux'][0] # we can stop here, no high DR settings can be used
    return True, catalog['Peak_flux'][0]  # if we end up here we can use high DR settings


def auto_determine_extractregion(fitsimage, min_extract_size=0.5, margin=800.):
    """
    Determine the extract region size based on the brightest artifact in the image.

    Parameters
    ----------
    fitsimage : str
        Path to the FITS image.
    min_extract_size : float
        Minimum extract region size in degrees.
    margin : float
        Margin in degrees.

    Returns
    -------
    float
        Determined box size in degrees.
    """
    if os.path.isdir('misc') == False:
        os.mkdir('misc')
    if os.path.isdir('plots') == False:
        os.mkdir('plots')

    thresh_pix = 4.5
    thresh_isl = 4.5  
    outputfluxcatalog = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_compactsource_fluxcatalog.fits')
    outputerrormap1 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_errormap1.fits')
    outputcatalog1 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_sources_catalog1.fits')
    outputerrormap2 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_errormap2.fits')
    outputcatalog2 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_sources_catalog2.fits')
    outputerrormap3 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_errormap3.fits')
    outputcatalog3 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_sources_catalog3.fits')
    outputerrormap4 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_errormap4.fits')
    outputcatalog4 = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_sources_catalog4.fits')

    with fits.open(fitsimage, ignore_missing_end=True) as hdu:
        freq = hdu[0].header['CRVAL3']  # in Hz
        pixsize = 3600. * (hdu[0].header['CDELT2']) # in arcsec
        match_radius = pixsize*31.*3./60. # arcmin

    # make the error map
    terminal_print('Making artifact map from:', fitsimage)
    terminal_print('Pixel size:', pixsize, 'arcsec')

    
    # compute the calibration error map, run 1 with small kernel to pick up small scale errors
    calibration_error_map(fitsimage,  outputerrormap1, kernelsize=31, rebin=31)
    # compute the calibration error map, run 2 with medium kernel to pick up medium scale errors
    calibration_error_map(fitsimage,  outputerrormap2, kernelsize=31*2, rebin=31*2)
    # compute the calibration error map, run 3 with large kernel to pick up large scale errors
    calibration_error_map(fitsimage,  outputerrormap3, kernelsize=31*4, rebin=31*4)
    # compute the calibration error map, run 4 with extra large kernel to pick up extra large scale errors
    calibration_error_map(fitsimage,  outputerrormap4, kernelsize=31*6, rebin=31*6)

    # create the artifact sources catalog
    empty_catalog1 = create_calibration_error_catalog(outputerrormap1, outputcatalog1, thresh_pix=thresh_pix, thresh_isl=thresh_isl)
    empty_catalog2 = create_calibration_error_catalog(outputerrormap2, outputcatalog2, thresh_pix=4.0, thresh_isl=4.0)
    empty_catalog3 = create_calibration_error_catalog(outputerrormap3, outputcatalog3, thresh_pix=3.5, thresh_isl=3.5)
    empty_catalog4 = create_calibration_error_catalog(outputerrormap4, outputcatalog4, thresh_pix=3.5, thresh_isl=3.5)

    # create list of catalogs to merge
    catalog_list = []
    if not empty_catalog1:
        catalog_list.append(outputcatalog1)
    if not empty_catalog2:
        catalog_list.append(outputcatalog2)
    if not empty_catalog3:
        catalog_list.append(outputcatalog3)
    if not empty_catalog4:
        catalog_list.append(outputcatalog4)     
    if len(catalog_list) > 1:
        terminal_print('Merging artifact source catalogs')
        merged_catalog = 'misc/' + os.path.basename(fitsimage).replace('.fits', '_artifact_sources_merged_catalog.fits')
        merge_catalogs(catalog_list, merged_catalog)
        # remove duplicate sources that are close to each other (within 3 arcmin), keep the brightest
        update_calibration_error_catalog(merged_catalog,merged_catalog, distance=3, keep_N_brightest=999, N_dir_max=999, interleave_sorting=False)
    
    # create compact source catalog from which we can get peak fluxes
    write_compactsource_flux(fitsimage, outputfluxcatalog)

    # check if just one/two sources were found in catalog in selfcalcycle 0
    if len(catalog_list) == 0:
        if freq < 1e9:  # UHF band
            extract_size = 3.0  # degrees
        elif freq >= 1e9 and freq < 2e9:  # L band
            extract_size = 1.5  # degrees
        elif freq >= 2e9:  # S band
            extract_size = 0.75  # degrees
        return extract_size

    # find peak fluxes from compact source catalog and add to filtered catalog (put in AFLUX column)
    add_peak_total_flux_to_catalog(merged_catalog, outputfluxcatalog, match_radius=match_radius)

    # === determine extract region based on the brighest artifact source ===
  
    # load the catalog
    with fits.open(merged_catalog) as hdu_list:
        catalog = Table(hdu_list[1].data)
    
    # sort catalog at Peak flux, brightest first
    idx = catalog.argsort(keys='Peak_flux', reverse=True)
    catalog = catalog[idx]

    # get the X Y positions of the artifact sources in fitsimage
    extract_size_start = 0.0
    hdu = flatten(fits.open(fitsimage, ignore_missing_end=True))
    w = WCS(hdu.header)
    x_center = hdu.header['NAXIS1'] / 2
    y_center = hdu.header['NAXIS2'] / 2
    
    # loop over sources
    for source in catalog:
        ra_source = source['RA']
        dec_source = source['DEC']
            
        x_source, y_source = w.wcs_world2pix(ra_source, dec_source, 0)
        x_dist = abs(x_source - x_center)
        y_dist = abs(y_source - y_center)
        #print('Artifact source pixel position:', x_source, y_source)
        #print('E-W/N-S artifact distance to image center (pixels):', x_dist, y_dist)
        max_distance = max([x_dist, y_dist])
        terminal_print('Max artifact distance (E-W/N-S) to image center (degrees):', max_distance * pixsize / 3600.)
        if 2 * (max_distance * pixsize) / 3600. > extract_size_start:  # in degrees
            extract_size_start = (2. * (max_distance * pixsize) / 3600.) + ((pixsize * margin) / 3600.)  # in degrees
            terminal_print('RA: {}, DEC: {}'.format(ra_source, dec_source))
    if extract_size_start < min_extract_size:
        extract_size_start = min_extract_size
    get_logger().info('Determined extract size of ' + str(extract_size_start) + ' degrees based on artifact sources')
    terminal_print('Determined extract size of ' + str(extract_size_start) + ' degrees based on artifact sources')
    
    # plot the results
    region_string = """
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1
    fk5
    box(137.2868914,-9.6412498,7200.000",7200.000",0)
    """
    with fits.open(fitsimage, ignore_missing_end=True) as hdu:
        r = pyregion.parse(region_string)
        r.write('misc/templatebox.reg')
        r = pyregion.open('misc/templatebox.reg')
        r[0].coord_list[0] = hdu[0].header['CRVAL1']  # units degr
        r[0].coord_list[1] = hdu[0].header['CRVAL2']  # units degr
        r[0].coord_list[2] = extract_size_start  # units degr
        r[0].coord_list[3] = extract_size_start  # units degr
        r.write('misc/templatebox.reg')
    
    # write ds9 region file from the artifact sources catalog: outputfluxcatalog
    with fits.open(outputcatalog1) as hdu_list:
        catalog_extract = Table(hdu_list[1].data)
    write_ds9_regions(catalog_extract['RA'], catalog_extract['DEC'], filename='misc/extractbox_artifact_sources.reg')
    
    # plot the locations of the artifact sources on top of the fits image to check if artifacts are correctly identified
    with fits.open(outputerrormap1) as hdulist:
        imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
        plotminmax = [-2.*imagenoise, 35.*imagenoise]

    plotimage_astropy(outputerrormap1, 'plots/auto_extractbox.png', mask=None, \
                      regionfile=['misc/extractbox_artifact_sources.reg', 'misc/templatebox.reg'], \
                      regioncolor='red', minmax=plotminmax, regionalpha=1.0)

    # remove template box region file
    Path('misc/templatebox.reg').unlink(missing_ok=True)
    return extract_size_start # in degrees
