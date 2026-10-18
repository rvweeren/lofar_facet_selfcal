"""Build starting skymodels for calibration from external surveys (GSM, TGSS,
VLASS, LoTSS/FITS images) and the MeerKAT bandpass calibrator models."""
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

import astropy.units as units
import bdsf  # type: ignore
import numpy as np
import pyregion
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from casacore.tables import table  # type: ignore

from submods.runtime_utils import get_logger, run, terminal_print

# facetselfcal package directory and submods directory, same values as datapath
# and submodpath in main.py
datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal')
submodpath = os.path.dirname(os.path.abspath(__file__))


def grab_coord_MS(MS):
    """
    Read the coordinates of a field from one MS corresponding to the selection given in the parameters

    Parameters
    ----------
    MS : str
        Full name (with path) to one MS of the field

    Returns
    -------
    RA, Dec : "tuple"
        coordinates of the field (RA, Dec in deg , J2000)
    """

    # reading the coordinates ("position") from the MS
    # NB: they are given in rad,rad (J2000)
    [[[ra, dec]]] = table(MS + '::FIELD', readonly=True, ack=False).getcol('PHASE_DIR')

    # RA is stocked in the MS in [-pi;pi]
    # => shift for the negative angles before the conversion to deg (so that RA in [0;2pi])
    if ra < 0:
        ra = ra + 2 * np.pi

    # convert radians to degrees
    ra_deg = ra / np.pi * 180.
    dec_deg = dec / np.pi * 180.

    # and sending the coordinates in deg
    return (ra_deg, dec_deg)


def getGSM(ms_input, SkymodelPath='gsm.skymodel', Radius="5.", DoDownload="Force", Source="GSM", targetname="pointing",
           fluxlimit=None):
    """
    Download the skymodel for the target field

    Parameters
    ----------
    ms_input : str
        String from the list (map) of the target MSs
    SkymodelPath : str
        Full name (with path) to the skymodel; if YES is true, the skymodel will be downloaded here
    Radius : string with float (default = "5.")
        Radius for the TGSS/GSM cone search in degrees
    DoDownload : str ("Force" or "True" or "False")
        Download or not the TGSS skymodel or GSM.
        "Force": download skymodel from TGSS or GSM, delete existing skymodel if needed.
        "True" or "Yes": use existing skymodel file if it exists, download skymodel from
        TGSS or GSM if it does not.
        "False" or "No": Do not download skymodel, raise an exception if skymodel
        file does not exist.
    targetname : str
        Give the patch a certain name, default: "pointing"
    Source : str, optional
        Sky-model source to query, such as GSM or TGSS.
    fluxlimit : float or None, optional
        Minimum source flux density to include in the downloaded sky model.

    Returns
    -------
    str or int
        Sky-model path when downloaded, or zero when an existing model is reused.
    """
    import lsmtool  # type: ignore
    FileExists = os.path.isfile(SkymodelPath)
    if (not FileExists and os.path.exists(SkymodelPath)):
        raise ValueError("download_tgss_skymodel_target: Path: \"%s\" exists but is not a file!" % (SkymodelPath))
    download_flag = False
    # if not os.path.exists(os.path.dirname(SkymodelPath)):
    #    os.makedirs(os.path.dirname(SkymodelPath))
    if DoDownload.upper() == "FORCE":
        if FileExists:
            os.remove(SkymodelPath)
        download_flag = True
    elif DoDownload.upper() == "TRUE" or DoDownload.upper() == "YES":
        if FileExists:
            terminal_print("USING the exising skymodel in " + SkymodelPath)
            return (0)
        else:
            download_flag = True
    elif DoDownload.upper() == "FALSE" or DoDownload.upper() == "NO":
        if FileExists:
            terminal_print("USING the exising skymodel in " + SkymodelPath)
            return (0)
        else:
            raise ValueError(
                "download_tgss_skymodel_target: Path: \"%s\" does not exist and skymodel download is disabled!" % (
                    SkymodelPath))

    # If we got here, then we are supposed to download the skymodel.
    assert download_flag is True  # Jaja, belts and suspenders...
    terminal_print("DOWNLOADING skymodel for the target into " + SkymodelPath)

    # Reading a MS to find the coordinate (pyrap)
    [RATar, DECTar] = grab_coord_MS(ms_input)

    # Downloading the skymodel, skip after five tries
    errorcode = 1
    tries = 0
    while errorcode != 0 and tries < 5:
        if Source == 'TGSS':
            url = "http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv5.cgi?coord={},{}&radius={}&unit=deg&deconv=y".format(
                RATar, DECTar, Radius)
        elif Source == 'GSM':
            url = "https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord={},{}&radius={}&unit=deg&deconv=y".format(
                RATar, DECTar, Radius)
        errorcode = subprocess.run(["wget", "-O", SkymodelPath, url]).returncode
        time.sleep(5)
        tries += 1

    if not os.path.isfile(SkymodelPath):
        raise IOError(
            "download_tgss_skymodel_target: Path: \"%s\" does not exist after trying to download the skymodel." % (
                SkymodelPath))

    # Treat all sources as one group (direction)
    skymodel = lsmtool.load(SkymodelPath)
    if fluxlimit:
        skymodel.remove('I<' + str(fluxlimit))
    skymodel.group('single', root=targetname)
    skymodel.write(clobber=True)

    return SkymodelPath


def makeBBSmodelforFITS(filename, extrastrname=''):
    """
    Create a BBS sky model from a FITS image.

    Parameters
    ----------
    filename : str
        FITS image path.
    extrastrname : str, optional
        Additional output-name suffix.

    Returns
    -------
    str
        Generated model filename.
    """
    img = bdsf.process_image(filename,mean_map='zero', rms_map=True, rms_box = (100,10))
    img.write_catalog(format='bbs', bbs_patches='source', \
                      outfile='source' + extrastrname + '.skymodel'  , clobber=True)
    
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    for f in glob.glob(os.path.dirname(filename) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    return 'source' + extrastrname + '.skymodel'


def makeBBSmodelforVLASS(filename, extrastrname=''):
    """
    Create a BBS sky model from a VLASS image.

    Parameters
    ----------
    filename : str
        VLASS image path.
    extrastrname : str, optional
        Additional output-name suffix.

    Returns
    -------
    str
        Generated model filename.
    """
    img = bdsf.process_image(filename, mean_map='zero', rms_map=True, rms_box=(100, 10))
    # frequency=150e6, beam=(25./3600,25./3600,0.0) )
    img.write_catalog(format='bbs', bbs_patches='source', outfile='vlass' + extrastrname + '.skymodel', clobber=True)
    # bbsmodel = 'bla.skymodel'
    del img
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    for f in glob.glob(os.path.dirname(filename) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    return 'vlass' + extrastrname + '.skymodel'


def makeBBSmodelforTGSS(boxfile=None, fitsimage=None, pixelscale=None, imsize=None,
                        ms=None, extrastrname=''):
    """
    Creates a TGSS skymodel in DP3-readable format.

    Parameters
    ----------
    boxfile : str
        path to the DS9 region to create a model for.
    fitsimage : str
        name of the FITS image the model will be created from.
    pixelscale : float
        number of arcsec per pixel.
    imsize : int
        image size in pixels.
    ms : str
        if no box file is given, use this Measurement Set to determine the sky area to make a model of.
    extrastrname : str, optional
        Suffix appended to the generated sky-model filename.

    Returns
    -------
    tgss.skymodel
        name of the output skymodel (always tgss[#nr].skymodel).
    """
    tgsspixsize = 6.2
    if boxfile is None and imsize is None:
        terminal_print('Wrong input detected, boxfile or imsize needs to be set')
        raise Exception('Wrong input detected, boxfile or imsize needs to be set')
    if boxfile is not None:
        r = pyregion.open(boxfile)
        if len(r[:]) > 1:
            terminal_print('Composite region file, not allowed')
            raise Exception('Composite region file, not allowed')
        phasecenter = getregioncenter(boxfile)
        phasecenterc = phasecenter.replace('deg', '')
        xs = np.ceil((r[0].coord_list[2]) * 3600. / tgsspixsize)
        ys = np.ceil((r[0].coord_list[3]) * 3600. / tgsspixsize)
    else:
        t2 = table(ms + '::FIELD')
        phasedir = t2.getcol('PHASE_DIR').squeeze()
        t2.close()
        phasecenterc = ('{:12.8f}'.format(180. * np.mod(phasedir[0], 2. * np.pi) / np.pi) + ',' + '{:12.8f}'.format(
            180. * phasedir[1] / np.pi)).replace(' ', '')

        # phasecenterc = str() + ', ' + str()
        xs = np.ceil(imsize * pixelscale / tgsspixsize)
        ys = np.ceil(imsize * pixelscale / tgsspixsize)

    terminal_print('TGSS imsize:', xs)
    terminal_print('TGSS image center:', phasecenterc)
    get_logger().info('TGSS imsize:' + str(xs))
    get_logger().info('TGSS image center:' + str(phasecenterc))

    # sys.exit()

    if fitsimage is None:
        filename = SkyView.get_image_list(position=phasecenterc, survey='TGSS ADR1', pixels=int(xs), cache=False)
        terminal_print('TGSS image source:', filename)
        if os.path.isfile(filename[0].split('/')[-1]):
            Path(filename[0].split('/')[-1]).unlink(missing_ok=True)
        time.sleep(10)
        subprocess.run(['wget', filename[0]])
        filename = filename[0].split('/')[-1]
        terminal_print('TGSS image source:', filename)
    else:
        filename = fitsimage

    img = bdsf.process_image(filename, mean_map='zero', rms_map=True, rms_box=(100, 10), frequency=150e6,
                             beam=(25. / 3600, 25. / 3600, 0.0))
    img.write_catalog(format='bbs', bbs_patches='source', outfile='tgss' + extrastrname + '.skymodel', clobber=True)
    # bbsmodel = 'bla.skymodel'
    del img
    terminal_print('TGSS image source:', filename)
    # move all *pybdsf.log files to a logs directory
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    for f in glob.glob(os.path.dirname(filename) + '/*pybdsf.log'):
        if os.path.isfile('logs/' + os.path.basename(f)):
            os.remove('logs/' + os.path.basename(f))
        shutil.move(f, 'logs/')
    
    return 'tgss' + extrastrname + '.skymodel', filename


def getregioncenter(regionfile, standardbox=True):
    """
    Extract box center of a DS9 region.

    Parameters
    ----------
    regionfile : str
        path to the region file.
    standardbox : bool
        only allow square, non-rotated boxes.

    Returns
    -------
    regioncenter : str
        DP3 compatible string for phasecenter shifting.
    """
    r = pyregion.open(regionfile)

    if len(r[:]) > 1:
        terminal_print('Only one region can be specified, your file contains', len(r[:]))
        raise Exception('Only one region can be specified, your file contains')

    if r[0].name not in ['box', 'circle', 'ellipse']:
        terminal_print('Only box, circle, or ellipse region supported', r[0].name)
        raise Exception('Only box, circle, or ellipse region supported')

    if r[0].name == 'box':
        ra = r[0].coord_list[0]
        dec = r[0].coord_list[1]
        boxsizex = r[0].coord_list[2]
        boxsizey = r[0].coord_list[3]
        angle = r[0].coord_list[4]

    if r[0].name == 'circle':
        ra = r[0].coord_list[0]
        dec = r[0].coord_list[1]
        boxradius = r[0].coord_list[2]
        #boxsizex = boxradius * 2.
        #boxsizey = boxradius * 2.
        angle = 0.

    if r[0].name == 'ellipse':
        ra = r[0].coord_list[0]
        dec = r[0].coord_list[1]
        boxradius1 = r[0].coord_list[2] # semi-major axis in the x-direction if angle=0
        boxradius2 = r[0].coord_list[3] # semi-minor axis in the y-direction if angle=0
        #boxsizex = boxradiusx * 2.
        #boxsizey = boxradiusy * 2.
        angle = r[0].coord_list[4]

    if standardbox:
        if boxsizex != boxsizey:
            terminal_print('Only a square box region supported, you have these sizes:', boxsizex, boxsizey)
            raise Exception('Only a square box region supported')
        if np.abs(angle) > 1:
            terminal_print('Only normally oriented sqaure boxes are supported, your region is oriented under angle:', angle)
            raise Exception('Only normally oriented sqaure boxes are supported, your region is oriented under angle')

    regioncenter = ('{:12.8f}'.format(ra) + 'deg,' + '{:12.8f}'.format(dec) + 'deg').replace(' ', '')
    return regioncenter


def set_MeerKAT_bandpass_skymodel(ms):
    """
    Determines and sets the appropriate MeerKAT bandpass calibrator skymodel for an MS.

    The function inspects the frequency band of the MS and pointing direction to select a
    matching skymodel for J0408-6545 or J1939-6342 in UHF or L-band. S-band is not supported.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set (MS) file.

    Returns
    -------
    str
        Path to the selected skymodel file.

    Raises
    ------
    Exception
        If the frequency band is S-band or if no matching skymodel is found.
    """
    skymodpath = '/'.join(datapath.split('/')[0:-1])+'/facetselfcal/data'
    
    skymodel = None
    cJ0408_6545 = np.array([1.08358621, -1.1475981])
    cJ1939_6342 = np.array([5.1461782, -1.11199629])

    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        midfreq = np.mean(t.getcol('CHAN_FREQ')[0])

    UHF = False; Lband = False; Sband = False
    if (midfreq > 500e6) and (midfreq < 1.0e9):  # UHF-band
        UHF = True
    if (midfreq >= 1.0e9) and (midfreq < 1.7e9):  # L-band
        Lband = True
    if (midfreq >= 1.7e9) and (midfreq < 4.0e9):  # S-band
        Sband = True

    with table(ms + '::FIELD', ack=False) as t:
        adir = t.getcol('DELAY_DIR')[0][0][:]
    cdatta = SkyCoord(adir[0]*units.radian, adir[1]*units.radian, frame='icrs')

    if (cdatta.separation(SkyCoord(cJ0408_6545[0]*units.radian, cJ0408_6545[1]*units.radian, frame='icrs'))) < 0.05*units.deg:
        if Lband: skymodel = skymodpath + '/J0408-6545_L.skymodel'
        if UHF: skymodel = skymodpath + '/J0408-6545_UHF.skymodel'

    if (cdatta.separation(SkyCoord(cJ1939_6342[0]*units.radian, cJ1939_6342[1]*units.radian, frame='icrs'))) < 0.05*units.deg:
        if Lband: skymodel = skymodpath + '/J1939-6342_L.skymodel'
        if UHF: skymodel = skymodpath + '/J1939-6342_UHF.skymodel'

    if Sband:
        raise Exception('Cannot set skymodel for Sband')
    if skymodel is None:
        raise Exception('Could not find matching skymodel (options are J0408-6545 or J1939-6342)')
    terminal_print('skymodel is set to: ', skymodel)
    return skymodel


def mslist_return_stack(mslist, stack):
    """
    Returns a modified MS list based on the stack parameter.

    Parameters
    ----------
    mslist : list
        A list of measurement set (MS) file paths or identifiers.
    stack : bool
        If True, returns the entire mslist. If False, returns a list
        containing only the first element of mslist.

    Returns
    -------
    list
        The full mslist if stack is True, otherwise a single-element list
        containing the first item from mslist.

    Examples
    --------
    >>> mslist_return_stack(['ms1.ms', 'ms2.ms', 'ms3.ms'], True)
    ['ms1.ms', 'ms2.ms', 'ms3.ms']
    >>> mslist_return_stack(['ms1.ms', 'ms2.ms', 'ms3.ms'], False)
    ['ms1.ms']

    """
    return mslist if stack else [mslist[0]]


def set_skymodels_external_surveys(args, mslist):
    """
    Set up sky models from external surveys (TGSS, VLASS, GSM) or FITS images.
    This function generates sky models based on the specified external survey options
    (TGSS, VLASS, GSM, or arbitrary FITS images) for radio interferometry calibration.
    It processes measurement sets and creates appropriate sky models while validating
    input parameters to prevent conflicting options.
    Parameters
    ----------
    args : dict
        Dictionary of arguments containing:
        - startfromtgss : bool
        Flag to start from TGSS survey data
        - startfromvlass : bool
        Flag to start from VLASS survey data
        - startfromgsm : bool
        Flag to start from Global Sky Model
        - startfromimage : bool
        Flag to start from arbitrary FITS image
        - skymodel : str or None
        Path to sky model file (can be FITS or other format)
        - skymodelpointsource : str or None
        Path to point source sky model
        - start : int
        Starting cycle number
        - stack : bool
        Whether to stack multiple measurement sets
        - boxfile : str
        Path to box file for region definition
        - tgssfitsimage : str
        Path to TGSS FITS image
        - pixelscale : float
        Pixel scale in arcseconds
        - imsize : int
        Image size in pixels
    mslist : list of str
        List of measurement set paths to process
    Returns
    -------
    args : dict
        Updated arguments dictionary with skymodel field set to generated sky model(s)
    tgssfitsfile : str or None
        Path to TGSS FITS file if generated, None otherwise
    Raises
    ------
    Exception
    If conflicting options are provided (e.g., manual skymodel with startfrom* flags)
    If skymodel is not a FITS file when using --startfromimage
    If --startfromimage is not set when providing a FITS file as skymodel
    Notes
    -----
    - Only one external survey option should be used at a time
    - When using external surveys, manual skymodel provision is not allowed
    - If processing stacked measurement sets, returns list of skymodels
    - If processing single measurement set, returns single skymodel string
    """
    
    skymodel_list = []
    tgssfitsfile = None
    # --- TGSS ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromtgss'] and args['start'] == 0:
            if args['skymodel'] is None:
                tmpskymodel, tgssfitsfile = makeBBSmodelforTGSS(args['boxfile'],
                                                                fitsimage=args['tgssfitsimage'],
                                                                pixelscale=args['pixelscale'],
                                                                imsize=args['imsize'], ms=mstmp,
                                                                extrastrname=str(mstmp_id))
                skymodel_list.append(tmpskymodel)
            else:
                terminal_print('You cannot provide a skymodel/skymodelpointsource file manually while using --startfromtgss')
                raise Exception(
                    'You cannot provide a skymodel/skymodelpointsource manually while using --startfromtgss')

    # --- VLASS ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromvlass'] and args['start'] == 0:
            if args['skymodel'] is None and args['skymodelpointsource'] is None:
                run(f'python {submodpath}/vlass_search.py ' + mstmp)
                skymodel_list.append(makeBBSmodelforVLASS('vlass_poststamp.fits', extrastrname=str(mstmp_id)))
            else:
                terminal_print('You cannot provide a skymodel/skymodelpointsource manually while using --startfromvlass')
                raise Exception(
                    'You cannot provide a skymodel/skymodelpointsource manually while using --startfromvlass')

    # --- GSM ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromgsm'] and args['start'] == 0:
            if args['skymodel'] is None and args['skymodelpointsource'] is None:
                skymodel_list.append(getGSM(mstmp, SkymodelPath='gsm' + str(mstmp_id) + '.skymodel',
                                            Radius=str(args['pixelscale'] * args['imsize'] / 3600.)))
            else:
                terminal_print('You cannot provide a skymodel/skymodelpointsource manually while using --startfromgsm')
                raise Exception('You cannot provide a skymodel/skymodelpointsource manually while using --startfromgsm')

    # --- Arbitrary FITS image ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromimage'] and args['start'] == 0:
            if args['skymodel'].lower().endswith('.fits') and args['skymodelpointsource'] is None:
                skymodel_list.append(makeBBSmodelforFITS(args['skymodel'], extrastrname=str(mstmp_id)))
            elif args['skymodel'].lower().endswith('.fits') and (args['skymodelpointsource'] is not None):
                terminal_print('You cannot provide skymodelpointsource manually while using --startfromimage')
                raise Exception('You cannot provide skymodelpointsource manually while using --startfromimage')
            elif (not args['skymodel'].lower().endswith('.fits')) and args['skymodelpointsource'] is None:
                terminal_print('skymodel must be a FITS file and have the fits extension while using --startfromimage')
                raise Exception('skymodel must be a FITS file while using --startfromimage')
            else:
                terminal_print('Something unknown went wrong. Please check your input.')
                raise Exception('Something unknown went wrong. Please check your input.')
        elif args['skymodel'] is not None:
            if not (args['startfromimage']) and args['skymodel'].lower().endswith('.fits'):
                terminal_print('Option --startfromimage must be set if using a FITS image as skymodel.')
                raise Exception('Option --startfromimage must be set if using a FITS image as skymodel.')

    # note if skymodel_list is not set (len==0), args['skymodel'] keeps it value from argparse
    if len(skymodel_list) > 1:  # so startfromtgss or startfromvlass was done and --stack was true
        args['skymodel'] = skymodel_list
    if len(skymodel_list) == 1:  # so startfromtgss or startfromvlass was done
        args['skymodel'] = skymodel_list[0]  # make string again, not a list type
    terminal_print('Sky model input:', args['skymodel'])

    return args, tgssfitsfile
