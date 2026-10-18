"""Model data columns: predicting skymodels into the MS, MODEL_DATA handling for
direction-dependent solves, the model storage manager, and empty FITS images
for facet layouts."""
import multiprocessing
import pickle
import re
import subprocess

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from casacore.tables import table  # type: ignore

from submods.input_checks import (is_stokesdiagonal_modeltype_allowed,
                                  is_stokesi_modeltype_allowed)
from submods.ms_columns import remove_column_ms
from submods.ms_info import get_frequencies_from_ms
from submods.runtime_utils import get_logger, run, terminal_print
from submods.skymodels import grab_coord_MS


def create_empty_fitsimage(ms, imsize, pixelsize, outfile):
    """
    Create an empty FITS image with specified size and pixel scale.

    This function generates a FITS file filled with zeros, with the image center
    aligned to the phase center of the provided measurement set (MS). The FITS
    header is populated with appropriate WCS (World Coordinate System) information.

    Parameters
    ----------
    ms : str
        Path to the measurement set (MS) file. Used to determine the
        phase center coordinates.
    imsize : int
        Size of the image in pixels (assumes a square image).
    pixelsize : float
        Pixel size in arcseconds.
    outfile : str
        Path to the output FITS file.

    Returns
    -------
    None
        The function writes the FITS file to the specified output path.

    Notes
    -----
    - The WCS projection used is SIN (Sine projection).
    - The pixel scale is set in degrees, derived from the provided pixel size
    in arcseconds.
    """
    data = np.zeros((imsize, imsize))

    w = WCS(naxis=2)
    # what is the center pixel of the XY grid.
    w.wcs.crpix = [imsize / 2, imsize / 2]

    # coordinate of the pixel.
    w.wcs.crval = grab_coord_MS(ms)

    # the pixel scale
    w.wcs.cdelt = np.array([pixelsize / 3600., pixelsize / 3600.])

    # projection
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    # write the HDU object WITH THE HEADER
    header = w.to_header()
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(outfile, overwrite=True)
    return


def set_DDE_predict_skymodel_solve(wscleanskymodel):
    """
    Determines the tool to use for DDE (Direction Dependent Effects) prediction
    and solving based on the provided sky model.

    Parameters
    ----------
    wscleanskymodel : str
        The sky model string. If not None, WSCLEAN
        will be used; otherwise, DP3 will be used.

    Returns
    -------
    str
        'WSCLEAN' if a sky model is provided, otherwise 'DP3'.
    """
    if wscleanskymodel is not None:
        return 'WSCLEAN'
    else:
        return 'DP3'


def create_MODEL_DATA_PDIFF(inmslist, modelstoragemanager=None):
    """
    Creates the MODEL_DATA_PDIFF column.

    Parameters
    ----------
    inmslist : list
        list of input Measurement Sets.
    modelstoragemanager : str or None, optional
        Storage manager used for the generated model-data column.
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        nchan = len(get_frequencies_from_ms(ms))
        if modelstoragemanager is None:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF steps=[]')
            run("taql 'update " + ms + " set MODEL_DATA_PDIFF=array([0.5+0i,0+0i,0+0i,0.5+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
        elif modelstoragemanager == 'sisco_stokes_i':
            # create temporary column with the phase diff to avoid DP3 error std exception detected: Diagonal storage modes cannot store data for which the 2nd and 3rd correlation are non-zero
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF_TMP steps=[]') # create temporary column uncompressed
            run("taql 'update " + ms + " set MODEL_DATA_PDIFF_TMP=array([0.5+0i,0+0i,0+0i,0.5+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
            # now create the sisco compressed column based on the temporary column with the correct phase diff
            run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_PDIFF_TMP msout.datacolumn=MODEL_DATA_PDIFF msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i steps=[]')
            # remove temporary column
            remove_column_ms(ms, 'MODEL_DATA_PDIFF_TMP')
        elif modelstoragemanager == 'sisco_diagonal':
            # create temporary column with the phase diff to avoid DP3 error std exception detected: Diagonal storage modes cannot store data for which the 2nd and 3rd correlation are non-zero
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF_TMP steps=[]') # create temporary column uncompressed
            run("taql 'update " + ms + " set MODEL_DATA_PDIFF_TMP=array([0.5+0i,0+0i,0+0i,0.5+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
            # now create the sisco compressed column based on the temporary column with the correct phase diff
            run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_PDIFF_TMP msout.datacolumn=MODEL_DATA_PDIFF msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal steps=[]')
            # remove temporary column
            remove_column_ms(ms, 'MODEL_DATA_PDIFF_TMP')
        else:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF msout.storagemanager=' + modelstoragemanager + ' steps=[]')
            run("taql 'update " + ms + " set MODEL_DATA_PDIFF=array([0.5+0i,0+0i,0+0i,0.5+0i], [" + str(nchan) + ",4])'", log=True, taql=True)


def predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=False, sources=None, beamproximitylimit=240.0, modelstoragemanager=None):
    """
    Predict a sky model into a Measurement Set model column.

    Parameters
    ----------
    ms : str
        Measurement Set path.
    skymodel : str
        Sky-model path.
    modeldata : str, optional
        Destination model column.
    predictskywithbeam : bool, optional
        Apply the primary beam.
    sources : list, optional
        Source subset to predict.
    beamproximitylimit : float, optional
        Beam proximity limit.
    modelstoragemanager : object, optional
        Model storage manager.

    Returns
    -------
    None
        The model column is updated on disk.
    """
    cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msout=. '
    cmd += 'p.sourcedb=' + skymodel + ' steps=[p] p.type=predict msout.datacolumn=' + modeldata + ' '
    if sources is not None:
        cmd += 'p.sources=[' + str(sources) + '] '
    if predictskywithbeam:
        cmd += 'p.usebeammodel=True p.usechannelfreq=True p.beammode=array_factor '
        cmd += 'p.beamproximitylimit=' + str(beamproximitylimit) + ' '
    elif modelstoragemanager is not None:
        if modelstoragemanager == 'sisco_stokes_i':
            cmd += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i '
        elif modelstoragemanager == 'sisco_diagonal':
            cmd += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal '
        else:
            cmd += 'msout.storagemanager=' + modelstoragemanager + ' '
    terminal_print('Command:', cmd)
    run(cmd)


def updatemodelcols_includedir(modeldatacolumns, soltypelist_includedir, ms, dryrun=False, modelstoragemanager=None):
    """
    Update model columns for included calibration directions.

    Parameters
    ----------
    modeldatacolumns : list
        Model columns to update.
    soltypelist_includedir : list
        Solution types per direction.
    ms : str
        Measurement Set path.
    dryrun : bool, optional
        Only report the planned operation.
    modelstoragemanager : object, optional
        Model storage manager.

    Returns
    -------
    None
        Model columns are updated in place.
    """
    modeldatacolumns_solve = []
    modeldatacolumns_notselected = []
    id_kept = []
    id_removed = []

    f = open('facet_regions/facetdirections.p', 'rb')
    sourcedir = pickle.load(f)  # units are radian
    f.close()
    soltypelist_includedir_sel = np.array(soltypelist_includedir)  # convert to numpy array for easier indexing

    assert len(modeldatacolumns) == soltypelist_includedir_sel.shape[0]
    assert sourcedir.shape[0] == len(modeldatacolumns)
    assert soltypelist_includedir_sel.sum() > 0  # some element must be True

    if soltypelist_includedir_sel.sum() == len(modeldatacolumns):  # all are True, trivial case
        return modeldatacolumns, sourcedir, id_kept

    # step 1 remove columns from list that should not be solved
    for modelcolumn_id, modelcolumn in enumerate(modeldatacolumns):
        if soltypelist_includedir_sel[modelcolumn_id]:
            modeldatacolumns_solve.append(modelcolumn)
            id_kept.append(modelcolumn_id)
        else:
            modeldatacolumns_notselected.append(modelcolumn)
            id_removed.append(modelcolumn_id)

    modeldatacolumns_solve_newnames = modeldatacolumns_solve[:]
    terminal_print('soltypelist_includedir_sel for this pertubation', soltypelist_includedir_sel)
    # print(modeldatacolumns_solve)
    # print('Not selected', modeldatacolumns_notselected)
    # print(id_kept)
    # print(id_removed)
    terminal_print('Removed these directions coordinates')
    terminal_print('Removed source directions (RA/Dec):', sourcedir[id_removed][:])
    # print(sourcedir)
    # print(sourcedir[id_kept][:])
    # print(sourcedir.shape)
    # sourcedir_kept = sourcedir[id_kept][:]
    # sourcedir_removed = sourcedir[id_removed][:]
    # sys.exit()

    for removed_id in id_removed:
        c1 = SkyCoord(sourcedir[removed_id, 0] * units.radian, sourcedir[removed_id, 1] * units.radian,
                      frame='icrs')  # removed source
        distance = 1e9  # just a big number, larger than 180 degr
        for kept_id in id_kept:
            c2 = SkyCoord(sourcedir[kept_id, 0] * units.radian, sourcedir[kept_id, 1] * units.radian,
                          frame='icrs')  # kept source, looping over to find the closest
            angsep = c1.separation(c2).to(units.degree)
            # print(kept_id, angsep.value, '[degree]')
            if angsep.value < distance:
                distance = angsep.value
                closest_kept_modelcol = modeldatacolumns[kept_id]
        terminal_print('Removed', modeldatacolumns[removed_id], 'Closest kept is:', closest_kept_modelcol)

        modeldatacolumns_solve_newnames[modeldatacolumns_solve.index(closest_kept_modelcol)] = \
            modeldatacolumns_solve_newnames[modeldatacolumns_solve.index(closest_kept_modelcol)] + '+' + \
            modeldatacolumns[removed_id].split('MODEL_DATA_DD')[1]

    # print(modeldatacolumns_solve_newnames)
    # DP3 to create the missing columns
    for modelcol in modeldatacolumns_solve_newnames:
        t = table(ms, ack=False)
        colnames = t.colnames()
        t.close()
        if modelcol not in colnames:
            cmddppp = 'DP3 msin=' + ms + ' msin.datacolumn='
            cmddppp += modeldatacolumns[0] + ' msout=. steps=[] ' # modeldatacolumns[0] just use to first one as template
            cmddppp += 'msout.datacolumn=' + modelcol + ' '
            if modelstoragemanager is not None:
                if modelstoragemanager == 'sisco_stokes_i':
                    cmddppp += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i '
                elif modelstoragemanager == 'sisco_diagonal':
                    cmddppp += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal '
                else:
                    cmddppp += 'msout.storagemanager=' + modelstoragemanager + ' '
            terminal_print('DP3 command:', cmddppp)
            if not dryrun:
                run(cmddppp)

    # taql to fill the missing columns
    for modelcol in modeldatacolumns_solve_newnames:
        modeldatacolumns_taql_tmp = list(modelcol.split('+'))
        modeldatacolumns_taql = modeldatacolumns_taql_tmp[:]
        for mm_id, mm in enumerate(modeldatacolumns_taql_tmp):
            if mm_id > 0:
                modeldatacolumns_taql[mm_id] = 'MODEL_DATA_DD' + mm
        # print(modeldatacolumns_taql)
        colstr = '(' + '+'.join(map(str, modeldatacolumns_taql)) + ')'
        # print(colstr)

        if '+' in modelcol:  # we have a composite column
            taqlcmd = "taql" + " 'update " + ms + " set " + modelcol.replace("+", "\\+") + "=" + colstr + "'"
            terminal_print('TAQL command:', taqlcmd)
            if not dryrun:
                run(taqlcmd, taql=True)

    # create new column name MODEL_DATA_DDX+Y+Z with DP3
    # fill this column with taql

    # create modeldatacolumns_solve

    # return modeldatacolumns_solve

    return modeldatacolumns_solve_newnames, sourcedir[id_removed][:], id_kept


def remove_model_columns(mslist):
    """
    Removes columns related to model data from a list of Measurement Sets (MS).

    This function iterates over each MS in the provided list, identifies columns whose names match the pattern 'MODEL_DATA*',
    and removes them using the `remove_column_ms` function.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories.

    Returns
    -------
    None
    """
    terminal_print('Clean up MODEL_DATA type columns')
    for ms in mslist:
        t = table(ms)
        colnames = t.colnames()
        t.close()
        collist_del = [xdel for xdel in colnames if re.match('MODEL_DATA*', xdel)]
        for colname_remove in collist_del:
            remove_column_ms(ms, colname_remove)
    return


def set_modelstoragemanager(telescope, args):
    """
    Set the model storage manager for wsclean based on telescope capabilities and user arguments.

    This function determines which model storage manager to use for wsclean based on:
    1. User-specified preference in args['modelstoragemanager']
    2. Available options supported by the installed wsclean version
    3. Telescope-specific compatibility (for stokes_i compression)

    Parameters
    ----------
    telescope : str
        Telescope name used to check Stokes I model-type compatibility.
    args : dict
        Parsed facetselfcal options; ``modelstoragemanager`` is read (and may be
        updated in place), and ``args`` is passed to the model-type checks.

    Returns
    -------
    str or None
        The selected model storage manager ('stokes_i', 'sisco', or None).
        - 'stokes_i': Stokes I model compression (if supported and allowed for telescope)
        - 'sisco': SISCO model compression (fallback option)
        - None: No model compression (if not available or wsclean doesn't support the option)

    Raises
    ------
    None

    Notes
    -----
    - Checks wsclean capabilities by querying its help output
    - Falls back to 'sisco' compression if 'stokes_i' is unavailable but requested
    - Disables model storage manager if no compression method is available
    """

    if args['modelstoragemanager'] == 'None' or args['modelstoragemanager'] == 'none':
        args['modelstoragemanager'] = None

    wsclean_help = subprocess.check_output(['wsclean'], text=True)

    modelstoragemanager = None
    if args['modelstoragemanager'] == 'auto':
        if '-model-storage-manager' in wsclean_help:
            if is_stokesi_modeltype_allowed(args, telescope):
                if 'sisco-stokes-i' in wsclean_help:
                    terminal_print('auto: Using sisco stokes_i model compression')
                    modelstoragemanager = 'sisco_stokes_i'
                else:
                    terminal_print('auto: Using stokes_i model compression')
                    modelstoragemanager = 'stokes_i'
                    terminal_print('here')
            elif 'sisco' in wsclean_help:
                if is_stokesdiagonal_modeltype_allowed(args, telescope) and 'sisco-diagonal' in wsclean_help:
                    terminal_print('auto: Cannot use stokes_i model compression, using sisco_diagonal instead')
                    modelstoragemanager = 'sisco_diagonal'
                else:
                    terminal_print('auto: Cannot use stokes_i model compression, using sisco instead')
                    modelstoragemanager = 'sisco'   
            else:
                terminal_print('auto: No model compression possible, disabling model storage manager')
                modelstoragemanager = None  # we are here because wsclean does not support sisco compression    
        else:
            modelstoragemanager = None  # we are here because wsclean does not support the option -model-storage-manager            
    elif args['modelstoragemanager'] == 'stokes_i':
        if is_stokesi_modeltype_allowed(args, telescope):
            terminal_print('Using stokes_i model compression')
            modelstoragemanager = 'stokes_i'
        elif 'sisco' in wsclean_help:
            terminal_print('Cannot use stokes_i model compression, using sisco instead')
            modelstoragemanager = 'sisco'
        else:
            terminal_print('No model compression possible, disabling model storage manager')
            modelstoragemanager = None  # we are here because wsclean does not support sisco compression  
    elif args['modelstoragemanager'] == 'sisco':
        if 'sisco' in wsclean_help:
            terminal_print('Using sisco model compression')
            modelstoragemanager = 'sisco'
        else:
            terminal_print('No sisco model compression possible, disabling model storage manager')
            modelstoragemanager = None  # we are here because wsclean does not support sisco compression
    elif args['modelstoragemanager'] == 'sisco_stokes_i':
        if 'sisco-stokes-i' in wsclean_help and is_stokesi_modeltype_allowed(args, telescope):
            terminal_print('Using sisco stokes_i model compression')
            modelstoragemanager = 'sisco_stokes_i'
        else:
            terminal_print('No sisco_stokes_i model compression possible, disabling model storage manager')
            modelstoragemanager = None
    elif args['modelstoragemanager'] == 'sisco_diagonal':
        if 'sisco-diagonal' in wsclean_help and is_stokesdiagonal_modeltype_allowed(args, telescope):
            terminal_print('Using sisco_diagonal model compression')
            modelstoragemanager = 'sisco_diagonal'
        else:
            terminal_print('No sisco_diagonal model compression possible, disabling model storage manager')
            modelstoragemanager = None

    terminal_print('Storage manager:', modelstoragemanager)
    get_logger().info('Storage manager: ' + str(modelstoragemanager))
    return modelstoragemanager
