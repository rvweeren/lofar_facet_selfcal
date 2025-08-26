#!/usr/bin/env python
# rotationmeasure updates
# https://ui.adsabs.harvard.edu/abs/2022ApJ...932..110K/abstract
#std exception detected: The TEC constraints do not yet support direction-dependent intervals
#python /net/rijn/data2/rvweeren/software/lofar_facet_selfcal/submods/MSChunker.py --timefraction=0.15 --mintime=1200 --mode=time L765157.ms.copy.subtracted
# run with less disk-space usage, remove all but merged h5, remove columns
# continue splitting functions in facetselfcal in separate modules
# auto update channels out and fitspectralpol for high dynamic range
# time, timefreq, freq med/avg steps (via losoto)
# BDA step DP3
# compression: blosc2
# useful? https://learning-python.com/thumbspage.html
# add html summary overview
# Stacking check that freq and time axes are identical
# scalarphasediff solve WEIGHT_SPECTRUM_PM should not be dysco compressed! Or not update weights there...
# BLsmooth cannot smooth more than bandwidth and time smearing allows, not checked now
# bug related to sources-pb.txt in facet imaging being empty if no -apply-beam is used
# fix RR-LL referencing for flaged solutions, check for possible superterp reference station
# put all fits images in images folder, all solutions in solutions folder? to reduce clutter
# phase detrending.
# log command into the FITS header
# BLsmooth constant smooth for gain solves
# use scalarphasediff sols stats for solints? test amplitude stats as well
# parallel solving with DP3, given that DP3 often does not use all cores?
# uvmin, uvmax, uvminim, uvmaxim per ms per soltype

# antidx = 0
# taql("select ANTENNA1,ANTENNA2,gntrue(FLAG)/(gntrue(FLAG)+gnfalse(FLAG)) as NFLAG from L656064_129_164MHz_uv_pre-cal.concat.ms WHERE (ANTENNA1=={:d} OR ANTENNA2=={:d}) AND ANTENNA1!=ANTENNA2".format(antidx, antidx)).getcol('NFLAG')
#read_MeerKAT_wscleanmodel_5spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J1939-6342_UHF.txt','J1939-6342_UHF.skymodel')
#read_MeerKAT_wscleanmodel_5spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J1939-6342_L.txt','J1939-6342_L.skymodel')
#read_MeerKAT_wscleanmodel_4spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J0408-6545_UHF.txt','J0408-6545_UHF.skymodel')
#read_MeerKAT_wscleanmodel_3spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J0408-6545_L.txt','J0408-6545_L.skymodel')


# example:
# python facetselfal.py -b box_18.reg --forwidefield --avgfreqstep=2 --avgtimestep=2 --smoothnessconstraint-list="[0.0,0.0,5.0]" --antennaconstraint-list="['core']" --solint-list=[1,20,120] --soltypecycles-list="[0,1,3]" --soltype-list="['tecandphase','tecandphase','scalarcomplexgain']" test.ms

# Standard library imports
import ast
import configparser
import fnmatch
import glob
import logging
import multiprocessing
import os
import os.path
import re
import subprocess
import sys

from itertools import product
from itertools import groupby

# Third party imports
import astropy
import astropy.stats
import astropy.units as units
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord
from astropy.time import Time
from astroquery.skyview import SkyView
from losoto import h5parm
import bdsf
from casacore.tables import taql, table, makecoldesc
import losoto
import losoto.lib_operations
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyregion
import math
import pickle
import tables
import scipy
import scipy.stats
import time

# Required for running with relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Modules
from arguments import option_parser
from submods.source_selection.selfcal_selection import get_images_solutions, main as quality_check
from submods.split_irregular_timeaxis import regularize_ms, split_ms
from submods.h5_helpers.reset_h5parm_structure import fix_h5
from submods.h5_merger import merge_h5
from submods.h5_helpers.split_h5 import split_multidir
from submods.h5_helpers.multidir_h5 import same_weights_multidir, is_multidir
from submods.h5_helpers.overwrite_table import copy_over_source_direction_h5
from submods.h5_helpers.modify_amplitude import get_median_amp, normamplitudes, normslope_withmatrix, normamplitudes_withmatrix
from submods.h5_helpers.modify_rotation import rotationmeasure_to_phase, fix_weights_rotationh5,  fix_rotationreference, fix_weights_rotationmeasureh5, fix_rotationmeasurereference
from submods.h5_helpers.modify_tec import fix_tecreference
from submods.h5_helpers.nan_values import remove_nans, removenans_fulljones
from submods.h5_helpers.update_sources import update_sourcedirname_h5_dde, update_sourcedir_h5_dde
from submods.h5_helpers.general_utils import make_utf8
from submods.h5_helpers.flagging import flaglowamps_fulljones, flag_bad_amps, flaglowamps, flaghighamps, flaghighamps_fulljones
from submods.source_selection.phasediff_output import GetSolint, generate_csv as generate_phasediff_csv
from submods.fair_log.config import add_config_to_h5, add_version_to_h5
from utils.parsers import parse_history, parse_source_id

# Set logger
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('selfcal.log')
formatter = logging.Formatter('%(levelname)s:%(asctime)s ---- %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

matplotlib.use('Agg')

# For NFS mounted disks
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def fix_time_axis_gmrt(mslist):
    """
    Fixes the TIME and INTERVAL columns in Measurement Sets (MS) from the GMRT telescope.
    For each MS in the provided list, this function:
    - Checks if the telescope is GMRT by reading the 'TELESCOPE_NAME' from the OBSERVATION table.
    - If so, computes the most common time interval between observations, ignoring large jumps.
    - Updates the INTERVAL column to have a uniform value equal to the computed interval.
    - Adjusts the TIME column so that all time stamps are aligned to this uniform interval.
    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Sets (MS) to process.
    Notes
    -----
    This function modifies the MS files in place.
    """
    for ms in mslist:
        with table(ms + '/OBSERVATION', ack=False) as t:
            telescope = t.getcol('TELESCOPE_NAME')[0]
        if telescope == 'GMRT': 
            with table(ms, ack=False, readonly=False) as t:
               times = taql('select distinct TIME from $t').getcol("TIME") 
               intervals = times[1 : ] - times[ : -1]
               intervals = intervals[intervals < 1.5 * np.min(intervals)] # Select only intervals that do not correspond with big jumps in time.
               intervalprecise = np.mean(intervals)
               
               # Update INTERVAL column.
               intervalsold    = t.getcol("INTERVAL")
               intervalsnew    = np.ones_like(intervalsold) * intervalprecise
               t.putcol("INTERVAL", intervalsnew)

               # Update TIME column.
               timesold        = t.getcol("TIME")
               timesnew        = timesold[0] + np.round((timesold - timesold[0]) / intervalprecise, 0) * intervalprecise
               t.putcol("TIME", timesnew)
               print('Time axis interval set to', intervalprecise)

def aoflagger_column(mslist, aoflagger_strategy=None, column='CORRECTED_DATA'):
    """
    Runs AOFlagger on a list of Measurement Sets (MS) using DP3, with optional custom flagging strategy and data column.
    This function checks if a custom AOFlagger strategy file is provided and exists. If the strategy file requests
    flagging of circular polarisation data (RR, RL, LR, LL) and the MS contains such data, it temporarily replaces
    these polarisations with linear equivalents (XX, XY, YX, YY) in a copy of the strategy file, as AOFlagger cannot
    flag circular polarisations in DP3. The function then runs DP3 with AOFlagger on each MS in the list.
    Args:
        mslist (list of str): List of Measurement Set paths to process.
        aoflagger_strategy (str, optional): Path or name of the AOFlagger strategy file to use. If not provided,
            the default strategy is used. If a name is given and not a file path, it is assumed to be in the
            'flagging_strategies' directory under 'datapath'.
        column (str, optional): Name of the data column in the MS to flag. Defaults to 'CORRECTED_DATA'.
    Warnings:
        - If the strategy file requests RR/RL/LR/LL flagging and the MS contains circular polarisation data,
          a warning is printed and the strategy file is temporarily modified to use XX/XY/YX/YY instead.
    Side Effects:
        - May create and remove temporary strategy files in the current working directory.
        - Executes system commands for file manipulation and running DP3.
    Prints:
        - Warnings about unsupported polarisation flagging.
        - The command being executed for AOFlagger.
    """

    if aoflagger_strategy is not None and not os.path.isfile(aoflagger_strategy):
        aoflagger_strategy = f'{datapath}/flagging_strategies/' + aoflagger_strategy

    # check if MS has circular polarisation data and RR/RL/LR/LL flagging is requested
    if aoflagger_strategy is not None:
        with table(mslist[0] + '/POLARIZATION', ack=False) as t:
            corr_type = t.getcol('CORR_TYPE')
            with open(aoflagger_strategy) as myfile:
                if ("LL" in myfile.read() or "RR" in myfile.read() or \
                    "LR" in myfile.read() or "RL" in myfile.read()) and np.array_equal(np.array([[5, 6, 7, 8]]), corr_type):
                    print("\033[33m" + "WARNING: AOFlagger cannot flag RR/RL/LR/LL data in DP3" + "\033[0m")
                    print("\033[33m" + "WARNING: Will temporarily replace RR/RL/LR/LL with XX/XY/YX/YY" + "\033[0m")
                    if os.path.isfile('tmp.' + os.path.basename(aoflagger_strategy)):
                        os.system('rm tmp.' + os.path.basename(aoflagger_strategy))
                    os.system('cp ' + aoflagger_strategy + ' tmp.' + os.path.basename(aoflagger_strategy))
                    # using linux sed command to replace the strings in the file
                    os.system('sed -i "s/RR/XX/g" tmp.' + os.path.basename(aoflagger_strategy))
                    os.system('sed -i "s/RL/XY/g" tmp.' + os.path.basename(aoflagger_strategy))
                    os.system('sed -i "s/LR/YX/g" tmp.' + os.path.basename(aoflagger_strategy)) 
                    os.system('sed -i "s/LL/YY/g" tmp.' + os.path.basename(aoflagger_strategy))
                    aoflagger_strategy = 'tmp.' + os.path.basename(aoflagger_strategy)
        
    # run DP3 with AOFlagger
    for ms in mslist:
        cmd = 'DP3 msin=' + ms + ' msin.datacolumn=' + column + ' msout=. '
        cmd += 'ao.type=aoflag '
        cmd += 'ao.keepstatistics=False '
        cmd += 'ao.memoryperc=50 '
        cmd += 'ao.overlapperc=10 '
        if aoflagger_strategy is not None:
            cmd += 'ao.strategy=' +  aoflagger_strategy + ' '         
        cmd += 'steps=[ao] '
        #try:
        print('Running AOFlagger on ' + ms + ' with strategy: ' + aoflagger_strategy)
        print(cmd)
        run(cmd)
        #except: # DP3 crashes when flagging on RR and LL for some reason, this is a workaround
        #    cmdao = 'aoflagger -column ' + column +' -strategy ' + aoflagger_strategy + ' ' + ms
        #    print('Running AOFlagger on ' + ms + ' with strategy: ' + aoflagger_strategy)
        #    run(cmdao)

def setjy_casa(ms):
    """
    Selects and applies the appropriate CASA setjy model image for a known calibrator present in the Measurement Set (MS).
    This function determines which standard calibrator (3C147, 3C286, 3C138, or 3C48) is present in the MS by comparing the field coordinates.
    It then selects the correct model image based on the observing frequency band (UHF, L, S, C, or X band) and runs the CASA setjy procedure
    using a helper script. If 3C286 is detected, it also sets the polarised model for this calibrator.
    Parameters
    ----------
    ms : str
        Path to the Measurement Set (MS) directory.
    Raises
    ------
    Exception
        If no known calibrator is found in the MS field coordinates.
    Notes
    -----
    - Assumes the MS contains only a single source (fieldid=0).
    - Requires access to model images named according to the calibrator and frequency band.
    - Relies on external scripts and functions: `casa_setjy.py` and `set_polarised_model_3C286`.
    """
    
    # find which calibrators is present in the MS (this cannot be a mulitsource MS), so fieldid is 0
    
    c_3C147 = np.array([85.650575, 49.852009])
    c_3C286 = np.array([202.784534, 30.509155])
    c_3C138 = np.array([80.291192, 16.639459])
    c_3C48  = np.array([24.422081, 33.159759])

    with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
        midfreq = np.mean(t.getcol('CHAN_FREQ')[0])

    UHF = False; Lband = False; Sband = False; Cband = False; Xband = False
    if midfreq < 1.0e9:  # UHF-band or lower
        UHF = True
    if (midfreq >= 1.0e9) and (midfreq < 1.7e9):  # L-band
        Lband = True
    if (midfreq >= 1.7e9) and (midfreq < 4.0e9):  # S-band
        Sband = True
    if (midfreq >= 4.0e9) and (midfreq < 8.0e9):  # C-band
        Cband = True
    if (midfreq >= 8.0e9) and (midfreq < 12.0e9): # X-band
        Xband = True

    with table(ms + '/FIELD', ack=False) as t:
        adir = t.getcol('DELAY_DIR')[0][0][:]
    cdatta = SkyCoord(adir[0]*units.radian, adir[1]*units.radian, frame='icrs')

    if (cdatta.separation(SkyCoord(c_3C147[0]*units.deg, c_3C147[1]*units.deg, frame='icrs'))) < 0.05*units.deg:
        if Lband or UHF: modelimage = '3C147_L.im'
        if Sband: modelimage = '3C147_S.im'
        if Cband: modelimage = '3C147_C.im' 
        if Xband: modelimage = '3C147_X.im'
    elif (cdatta.separation(SkyCoord(c_3C286[0]*units.deg, c_3C286[1]*units.deg, frame='icrs'))) < 0.05*units.deg:
        if Lband or UHF: modelimage = '3C286_L.im'
        if Sband: modelimage = '3C286_S.im'
        if Cband: modelimage = '3C286_C.im' 
        if Xband: modelimage = '3C286_X.im'     
    elif (cdatta.separation(SkyCoord(c_3C138[0]*units.deg, c_3C138[1]*units.deg, frame='icrs'))) < 0.05*units.deg:
        if Lband or UHF: modelimage = '3C147_L.im'
        if Sband: modelimage = '3C138_S.im'
        if Cband: modelimage = '3C138_C.im' 
        if Xband: modelimage = '3C138_X.im'
    elif (cdatta.separation(SkyCoord(c_3C48[0]*units.deg, c_3C48[1]*units.deg, frame='icrs'))) < 0.05*units.deg:
        if Lband or UHF: modelimage = '3C48_L.im'
        if Sband: modelimage = '3C48_S.im'
        if Cband: modelimage = '3C48_C.im' 
        if Xband: modelimage = '3C48_X.im'
    else:
        print('No calibrator found in MS that matches the coordinates of 3C147, 3C138, 3C286, or 3C48: cannot use CASA setjy')
        raise Exception('No calibrator found in MS that matches the coordinates of 3C147, 3C138, 3C286, or 3C48: cannot use CASA setjy')    
    print('Using model image for CASA setjy: ' + modelimage)
    cmdsetjy = f'python {submodpath}/casa_setjy.py '
    cmdsetjy += '--ms=' + ms + ' --fieldid=0 --modelimage=' + modelimage + ' '
 
    run(cmdsetjy)
    if (cdatta.separation(SkyCoord(c_3C286[0]*units.deg, c_3C286[1]*units.deg, frame='icrs'))) < 0.05*units.deg:
        print('Setting polarised model for 3C286')
        set_polarised_model_3C286(ms, chunksize=1000)
        

def fix_antenna_info_gmrt(mslist):
    """
    Removes specific antennas from a Measurement Set if the telescope is GMRT.

    Parameters:
        mslist (list): List of input Measurement Sets.

    Behavior:
        - Checks the 'TELESCOPE_NAME' in the OBSERVATION table of the Measurement Set.
        - If the telescope is 'GMRT', removes antennas 'C07', 'S05', and 'E01' by calling remove_antennas.

    Requires:
        - The 'table' context manager for reading Measurement Set tables.
        - The 'remove_antennas' function to perform the actual removal.
    """
    for ms in mslist:
        with table(ms + '/OBSERVATION', ack=False) as t:
            telescope = t.getcol('TELESCOPE_NAME')[0]
        if telescope == 'GMRT':    
            antennas_to_remove = ['C07', 'S05', 'E01'] # these antennas were never built, but are present in the ANTENNA table
            remove_antennas(ms, antennas_to_remove)

def remove_antennas(ms_path, antennas_to_remove):
    """
    Removes specified antennas from the POINTING and ANTENNA tables of a Measurement Set.
    Adapted from code by Emanuele De Rubeis.
    This function updates the POINTING and ANTENNA tables in the given Measurement Set (MS) by
    removing all rows corresponding to antennas whose names are listed in `antennas_to_remove`.
    It is useful for fixing inconsistencies where the ANTENNA table contains antennas not present
    in the POINTING table.

    Args:
        ms_path (str): Path to the Measurement Set directory.
        antennas_to_remove (list of str): List of antenna names to be removed from the tables.

    Raises:
        Exception: If there is an error accessing or modifying the Measurement Set tables.

    Side Effects:
        Modifies the POINTING and ANTENNA tables in-place by removing specified antennas.
        Prints information about removed rows or errors encountered.
    """
    
    with table(f"{ms_path}/POINTING", readonly=False) as pointing_table:
        rows_to_remove = [i for i, antenna in enumerate(pointing_table) if antenna['NAME'] in antennas_to_remove]
        print(f"Rows to remove from POINTING: {rows_to_remove}")
        if len(rows_to_remove) >0:
            pointing_table.removerows(rows_to_remove)
            print(f"Removed rows from POINTING: {rows_to_remove}")
    
    with table(f"{ms_path}/ANTENNA", readonly=False) as antenna_table:
        rows_to_remove = [i for i, antenna in enumerate(antenna_table) if antenna['NAME'] in antennas_to_remove]
        if len(rows_to_remove) > 0:
            antenna_table.removerows(rows_to_remove)
            print(f"Removed rows from ANTENNA: {rows_to_remove}")
 

def split_multidir_ms(ms):
    """
    Splits a multisource Measurement Set (MS) into separate single-source MS files.
    Parameters:
        ms (str): Path to the multisource Measurement Set to be split.
    Returns:
        list of str: List of paths to the newly created single-source Measurement Sets. 
                     If the input MS contains only a single source, returns a list containing the original MS path.
    Notes:
        - If the input MS is already a single-source MS, no splitting is performed.
        - Each output MS will contain data for only one source, with FIELD_ID and SOURCE_ID reset to 0.
        - Existing output MS directories will be removed before new ones are created.
    """
    with table(ms, readonly=True) as t:
        if len(np.unique(t.getcol('FIELD_ID'))) == 1:
            print(f"Measurement Set {ms} is already a single source MS, no splitting needed.")
            return [ms]  # No splitting needed, return original MS
        else:
            print(f"Splitting multisource Measurement Set {ms} into single source MS...")
    # get the source names
    with table(ms + '/FIELD', readonly=True) as t:
        source_names = t.getcol('NAME')
        field_ids = t.getcol('SOURCE_ID')

    
    mslist = []
    for field_id in field_ids:
                        
        outname = ms + '.' + source_names[field_id] 
        # Remove  MS if it exists
        if os.path.isdir(outname):
            os.system('rm -rf {}'.format(outname))
        cmd = "taql 'select from {} where FIELD_ID=={} giving {} as plain'".format(ms, field_id, outname)       
        print(cmd)
        run(cmd)

        taql("delete from {} where rownr() not in (select distinct FIELD_ID from {})".format(outname+'/FIELD', outname))

        # Set 'SOURCE_ID' to 0 in the FIELD subtable.
        taql("update {} set SOURCE_ID=0".format(outname+'/FIELD'))

        # Set 'FIELD_ID' to 0 in the main table.
        taql("update {} set FIELD_ID=0".format(outname))
        mslist.append(outname)
    print(f"Splitting completed. Created {len(mslist)} single source MS.")
    return mslist
 

def check_pointing_centers(mslist):
    """
    Checks whether the pointing centers of the provided measurement sets (MS) are aligned within a specified tolerance.

    Parameters:
        mslist (list of str): List of paths to measurement sets (MS) to check.

    Returns:
        bool: True if all MS pointing centers are aligned within 0.025 arcseconds, False otherwise.

    Warnings:
        Prints a warning message and logs a warning if any MS pointing center differs from the first MS by more than 0.025 arcseconds.

    Notes:
        - If only one MS is provided, the function returns True without performing any checks.
        - Requires the 'table', 'SkyCoord', and 'units' objects to be available in the scope.
    """
    
    if len(mslist) == 1 or args['stack']:
        return True  # Only one MS, no need to check alignment, for stacking no need to check alignment either
    with table(mslist[0] + '/FIELD', ack=False, readonly=True) as t:
        ra_ref, dec_ref = t.getcol('PHASE_DIR').squeeze() # reference direction in radians
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')
    
    align = True
    for ms in mslist[1:]:  
        with table(ms + '/FIELD', ack=False, readonly=True) as t:
            ra_ms, dec_ms = t.getcol('PHASE_DIR').squeeze()
        center_ms = SkyCoord(ra_ms * units.radian, dec_ms * units.radian, frame='icrs')
        if not center.separation(center_ms).deg < 0.025/3600:  # 0.025 arcsec tolerance
            print("\033[33m" + "WARNING: Pointing centers of MSs differ by " + str(center.separation(center_ms).deg) + " [deg] !\033[0m")
            print("\033[33m" + "Pointing center of " + ms + " is not aligned with the first MS!" + "\033[0m")
            print("\033[33m" + "Will use DP3 phaseshift to align them\033[0m")
            logger.warning('Pointing centers of MSs differ: ' + ms)
            align = False
    return align        

def write_processing_history(cmd, version, imagebasename):
    """
    Updates the FITS headers of images matching the given basename with processing history.

    This function writes the command used for processing and the facetselfcal version into the
    primary header of each FITS file whose name matches the provided image basename pattern.
    It also adds citation information as comments in the header.

    Args:
        cmd (str): The command used for processing, typically the command-line invocation.
        version (str): The version string of facetselfcal.
        imagebasename (str): The base name pattern to match FITS image files.

    Notes:
        - Only the portion of the command after 'facetselfcal.py' is recorded, if present.
        - FITS files are identified using glob with the pattern '{imagebasename}*image*.fits'.
        - The function modifies FITS files in place.
    """
    # strip everyting before the facetselcal.py string in cmd
    cmd = cmd.strip() 
    cmd = 'facetselfcal.py' + cmd.split('facetselfcal.py')[1] if 'facetselfcal.py' in cmd else cmd
    imagelist = glob.glob( imagebasename + '*image*.fits')
    for image in imagelist:
        print('Updating FITS header:', image)
        with fits.open(image, mode='update') as hdul:
            # Add the command used for processing to the primary header
            hdul[0].header['HISTORY'] = "facetselfcal version: " + version
            hdul[0].header['HISTORY'] = cmd
            hdul[0].header['COMMENT'] = "========================================================================"
            hdul[0].header['COMMENT'] = "       If you use facetselfcal for scientific work, please cite:        "
            hdul[0].header['COMMENT'] = "                 van Weeren et al. (2021, A&A, 651, 115)                "
            hdul[0].header['COMMENT'] = "========================================================================"

def write_primarybeam_info(cmd, imagebasename):
    """
    Updates the FITS header of images matching the given basename with primary beam correction information.

    This function searches for FITS files whose names start with `imagebasename` and end with '-pb.fits'.
    For each matching file, it updates the primary header's COMMENT field(s) to indicate whether a full
    primary beam correction has been applied, based on the provided command-line arguments.

    Parameters:
        cmd (str): The command-line string used to run the imaging process. Determines which comments are added.
        imagebasename (str): The base name of the image files to search for and update.

    Notes:
        - If '-apply-facet-beam' or '-apply-primary-beam' is present in `cmd`, the header will indicate that
          the full primary beam correction has been applied.
        - If '-apply-facet-beam' is not present but '-apply-facet-solutions' is, the header will indicate that
          the primary beam correction has not been applied and manual correction may be necessary.
    """
    imagelist = glob.glob( imagebasename + '*image-pb.fits')
    for image in imagelist:
        print('Updating FITS header:', image)
        with fits.open(image, mode='update') as hdul:
            # Add a comment to the primary header
            if '-apply-facet-beam' in cmd:
                hdul[0].header['COMMENT'] = "Full primary beam correction applied. Image is science ready."
            if '-apply-facet-beam' not in cmd and '-apply-facet-solutions' in cmd:   
                hdul[0].header['COMMENT'] = "Full primary beam correction has not been applied." 
                hdul[0].header['COMMENT'] = "Manually correct your image for the primary beam."
            if '-apply-primary-beam' in cmd:
                hdul[0].header['COMMENT'] = "Full primary beam correction applied. Image is science ready."

def check_applyfacetbeam_MeerKAT(mslist, imsize, pixsize, telescope, DDE):
    """
    Checks whether the image field of view (FoV) for MeerKAT data is too large to safely use the -apply-facet-beam option in WSClean, and enforces the --disable-primary-beam option if necessary.
    Parameters:
        mslist (list of str): List of measurement set (MS) file paths to check.
        imsize (float): Image size in pixels.
        pixsize (float): Pixel size in arcseconds.
        telescope (str): Name of the telescope. Function only applies checks if this is 'MeerKAT'.
        DDE (bool): If True, direction-dependent effects (DDE) calibration is enabled.
    Returns:
        None
    Side Effects:
        - If the image FoV is too large for any MS in mslist, sets args['disable_primary_beam'] = True.
        - Prints warnings to the console and logs a warning message.
        - Exits after the first MS that violates the safe FoV criterion.
    Notes:
        - The safe diameter is calculated based on the maximum frequency in the SPECTRAL_WINDOW table of each MS.
        - The function assumes the existence of a global 'args' dictionary and a 'logger' object.
        - The function also assumes the presence of 'compute_distance_to_pointingcenter' and 'table' utilities.
    """
    if telescope != 'MeerKAT' or not DDE: 
        return
    
    for ms in mslist:
        distance_pointing_center = compute_distance_to_pointingcenter(ms, HBAorLBA='other', warn=False, returnval=True, dologging=False)
        
        with table(ms+"::SPECTRAL_WINDOW", ack=False) as t:
            max_freq = t.getcol("CHAN_FREQ").max()
        safe_diameter = 60.*1.4*68.*(1.28e9/max_freq) # in arcsec
        if ((imsize*pixsize) + (distance_pointing_center*3600.) ) > safe_diameter:
            args['disable_primary_beam'] = True # will be set to False if one in mslist violates this criterium
            print("\033[33m" + "=== " + ms + " ===" + "\033[0m")
            print("\033[33m" + "Your image FoV is too large to use -apply-facet-beam in WSClean!" + "\033[0m")
            print("\033[33m" + "Code will run with the option --disable-primary-beam enforced" + "\033[0m")
            print("\033[33m" + "Imaged Fov [deg]: " + str(imsize*pixsize/3600) + "\033[0m") 
            print("\033[33m" + "Image center to telescope pointing center [deg]: " + str(distance_pointing_center) + "\033[0m")       
            print("\033[33m" + "Save Fov [deg]: " + str(safe_diameter/3600) + "\033[0m")
            logger.warning('Your image FoV is too large to use -apply-facet-beam in WSClean. The option --disable-primary-beam is automatically invoked: ' + ms)
        return    
        
def set_metadata_compression(mslist):
    """
    Sets the metadata compression flag based on the telescope name in the provided Measurement Set list.

    Parameters:
        mslist (list of str): List of paths to Measurement Sets. The function inspects the first set in the list.

    Side Effects:
        If the telescope name in the first Measurement Set is not 'LOFAR', sets the global 'args["metadata_compression"]' to False and prints a message.

    Notes:
        - Assumes that 'args' is a global variable accessible within the function's scope.
        - Requires the 'table' class/function to be imported and available.
    """
    t = table(mslist[0] + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()   
    if telescope != 'LOFAR':
        print('Not using LOFAR data, setting metadata compression to False')
        args['metadata_compression'] = False

def set_polarised_model_3C286(ms, chunksize=1000):
    """
    Calculates and inserts a polarised model for the 3C286 calibrator source into a Measurement Set (MS).
    This function performs the following steps:
        1. Obtains channel frequency data in GHz from the MS.
        2. Identifies the field ID corresponding to the 3C286 source (J1331+3030).
        3. Calculates the polarisation fraction and electric vector position angle (EVPA) for each frequency channel.
        4. Computes the Stokes IQUV values based on the initial Stokes I image, assuming Stokes V = 0.
        5. Converts the Stokes IQUV values to the XX, XY, YX, YY correlation model.
        6. Updates the MODEL_DATA column in the MS with the computed polarised model, processing the data in chunks.
    The polarisation model is computed as:
        Q = I * pfrac * cos(2 * EVPA)
        U = I * pfrac * sin(2 * EVPA)
        XY = U + iV (with V assumed to be 0)
        YX = U - iV (with V assumed to be 0)
    Args:
        ms (str): Path to the Measurement Set.
        chunksize (int, optional): Number of rows to process per chunk. Default is 1000.
    Returns:
        None
    Notes:
        - Assumes the presence of helper functions `calculate_evpa_3C286` and `calculate_pfrac_3C286`.
        - Requires the `tqdm`, `numpy`, and `casacore.tables` libraries.
        - Only updates the model for the 3C286 source (J1331+3030).
    """
    #obtain channel frequency data in GHz
    from tqdm import tqdm
    print('obtaining channel frequencies...')
    with table(ms + '/SPECTRAL_WINDOW', ack=False, readonly=True) as t:
        freqs = t.getcol('CHAN_FREQ') / 1e9     #freqs in GHz
        freqs = freqs.flatten()                 #freqs in 1D array
    
    # get polarization information to determine if linear or circular polarisation is used
    with table(ms + '/POLARIZATION', ack=False, readonly=True) as t:
        corr_type = t.getcol('CORR_TYPE') 

    #obtain field_id corresponding to J1331+3030 (3C286)
    with table(ms+'/FIELD', ack=False, readonly=True) as t:
        names = np.array(t.getcol('NAME'))
        try:
            source_id = np.array(t.getcol('SOURCE_ID'))[names == 'J1331+3030'][0]
        except:
            source_id = np.array(t.getcol('SOURCE_ID'))[names == '3C286'][0]
        print('source_id = ', source_id)

    #calculate polarisation characteristics
    print('calculating EVPA and polarisation fraction...')
    evpa  = calculate_evpa_3C286(freqs)
    pfrac = calculate_pfrac_3C286(freqs)

    #adjusting the existing model and input it back into the table
    with table(ms, ack=False, readonly=False) as t:
        with taql("select * from {} where FIELD_ID=={}".format(ms, source_id)) as tt:
            nrow = tt.nrows()   #the number of rows in the selected table

            #the number of chunks ( nrow//chunksize (+ 1 if chunksize not a
            #perfect divisor of nrow) )
            nchunk = nrow // chunksize + int(nrow % chunksize > 0)

            for ci in tqdm(range(nchunk), desc='looping over chunks'):
                cl = ci * chunksize     #the starting row of the chunk

                #for the last chunk the amount of rows is not necessarily
                #the chunksize. This corrects that
                crow = min(nrow - cl, chunksize)   #the length of the chunk

                #obtain the stokes I model (crow, nchan, ncor)
                model = tt.getcol('MODEL_DATA', startrow=cl, nrow=crow)

                #need to copy otherwise adjusted value will end up in I
                I = np.copy(model[:,:,0])

                #calculate the polarised model
                if np.array_equal(np.array([[9, 10, 11, 12]]), corr_type): # linear polarisation
                    model[:,:,0] = I + I*pfrac*np.cos(2*evpa) # XX
                    model[:,:,1] = I*pfrac*np.sin(2*evpa) # XY
                    model[:,:,2] = I*pfrac*np.sin(2*evpa) # YX
                    model[:,:,3] = I - I*pfrac*np.cos(2*evpa) # YY
                elif np.array_equal(np.array([[5, 6, 7, 8]]), corr_type): # circular polarisation
                    Q = I*pfrac*np.cos(2*evpa)
                    U = I*pfrac*np.sin(2*evpa)
                    model[:,:,0] = I # RR
                    model[:,:,1] =  (Q + (complex(0,1)*U)) # RL
                    model[:,:,2] =  (Q - (complex(0,1)*U)) # LR
                    model[:,:,3] = I # LL
                else:
                    raise ValueError("Unknown correlation type in POLARIZATION table. Expected linear or circular polarisation.")

                #inserting the polarised model
                tt.putcol('MODEL_DATA', model, startrow=cl, nrow=crow)
                
    print('successfully computed and inserted model')
            



def calculate_pfrac_3C286(nu):
    """
    Calculate the model polarization fraction (P) for 3C286 as a function of frequency.
    This function computes the polarization fraction of the radio source 3C286 based on a model
    with coefficients adopted from Hugo (2024), as implemented by Maarten Elion. The calculation
    is performed differently for frequencies above and below 1.1 GHz.
    Parameters
    ----------
    nu : np.ndarray
        Array of frequencies in GHz.
    Returns
    -------
    pfrac : np.ndarray
        Array of polarization fractions corresponding to the input frequencies.
    Notes
    -----
    - The model uses different coefficients for frequency ranges above and below 1.1 GHz.
    - The calculation is based on the wavelength corresponding to each frequency.
    """
    c = 2.99792e8   #speed of light [m/s]
    
    C0_high    = 0.080
    C2_high    = -0.053
    Clog2_high = -0.015

    C0_low     = 0.029
    C2_low     = -0.172
    Clog2_low  = -0.067

    #the wavelength in m
    wavelength = c / (nu*1e9) 

    #seperate the different frequency ranges
    mask_high = nu >= 1.1
    mask_low  = nu < 1.1

    #initialize solution array
    pfrac = np.zeros(nu.shape)

    #calculate polarization fraction higher range
    pfrac[mask_high] = C0_high + C2_high*wavelength[mask_high]**2 + \
                        Clog2_high*np.log10(wavelength[mask_high]**2)

    #calculate polarization fraction higher range
    pfrac[mask_low] = C0_low + C2_low*wavelength[mask_low]**2 + \
                        Clog2_low*np.log10(wavelength[mask_low]**2)

    return pfrac


def calculate_evpa_3C286(nu):
    """
    Calculate the model Electric Vector Polarization Angle (EVPA) for 3C286 in radians.
    This function computes the EVPA based on the observing frequency using a piecewise model
    with coefficients adopted from Hugo (2024), as implemented by Maarten Elion.
    Parameters
    ----------
    nu : np.ndarray
        Array of frequencies in GHz.
    Returns
    -------
    EVPA : np.ndarray
        Array of model EVPAs in radians, corresponding to the input frequencies.
    Notes
    -----
    - For frequencies >= 1.7 GHz, a quadratic model in wavelength squared is used.
    - For frequencies < 1.7 GHz, a more complex model involving log10 of frequency is used.
    - The output EVPA is in radians.
    """
    c = 2.99792e8   #speed of light [m/s]
    
    C0_high   = 32.64
    C2_high   = -85.37

    C0_low    = 29.53
    C2_low_nu = 4005.88
    C2_low    = -39.38

    #the wavelength in m
    wavelength = c / (nu*1e9) 

    #seperate the different frequency ranges
    mask_high = nu >= 1.7
    mask_low  = nu < 1.7

    #initialize solution array
    EVPA = np.zeros(nu.shape)
    
    #calculate the EVPA higher range
    EVPA[mask_high] = C0_high + C2_high*(wavelength[mask_high])**2

    #calculate the EVPA lower range
    EVPA[mask_low] = C0_low + wavelength[mask_low]**2 * (
                        C2_low_nu*(np.log10(nu[mask_low]))**3 + C2_low)

    #convert to radians
    EVPA *= np.pi/180

    return EVPA

def applycal_restart_di(mslist, selfcalcycle):
    """
    Apply merged selfcal solution files from a previous cycle in case of a restart for DI mode
    This makes CORRECTED_DATA from the previous selfcal cycle merged h5 solutions
    
    Parameters
    ----------
    mslist : list
        List of MS
    
    selfcalcycle : int
        selfcal cycle number
    
    Returns
    -------
    None

    """
    for ms in mslist:
        parmdbmergename = 'merged_selfcalcycle' + str(selfcalcycle-1).zfill(3) + '_' + os.path.basename(ms) + '.h5'
        applycal(ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])
    return


def MeerKAT_pbcor_Lband(fitsimage, outfile, freq=None, ms=None): 
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
    G1 =-0.3514e-3
    G2 = 0.5600e-7
    G3 =-0.0474e-10
    G4 = 0.00078e-13
    G5 = 0.00019e-16
 
    if freq is None:
        hdul = fits.open(fitsimage)
        header = hdul[0].header
        freq = header['CRVAL3']/1e9
        print('Frequency found from the FITS image [GHz]:', freq)
        hdul.close()
 
    hdu = fits.open(fitsimage,  ignore_missing_end=True)
    hduflat = flatten(hdu)
    img = hduflat.data   
    print('IMAGE shape',img.shape)
    x, y = np.indices((img.shape))
  
    w = WCS(hduflat.header)
    if ms is not None:
        print('Taking pointing center from the ms')
        with table(ms + '/FIELD', readonly=True, ack=False) as t:
            ra_ref, dec_ref = t.getcol('REFERENCE_DIR').squeeze()
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')   
    else:
        center = w.pixel_to_world(img.shape[0]/2., img.shape[1]/2.)   
        print('Assume image center is the pointing center')
        print('Make sure this it correct, if not provide the ms to the function')
        print('CENTER image:',center)
       

    coordinates =  w.pixel_to_world(np.ravel(x),np.ravel(y))   
    sep = center.separation(coordinates)
    seprad = sep.arcmin # convert to arcmin
   
    separray = seprad.reshape(img.shape)
    X = freq*separray
    #separray = radius in arcminutes
    #freq = frequency in GHz.
    pb = 1. + G1*(X**2) + G2*(X**4) + G3*(X**6)  + G4*(X**8)  + G5*(X**10)

    hdu[0].data[0,0,:,:] = img/pb

    astropy.io.fits.writeto(outfile, hdu[0].data, hdu[0].header, overwrite=True)
    return outfile


def frequencies_from_models(model_basename):
    """
    This function takes a wsclean model basename string and returns the 
    frequency list string wsclean should use for -channel-division-frequencies and the same as a np array
    
    Parameters:
    -----------
    model_basename: str
        wsclean model basename to search for

    Returns:
    --------
    freq_string: str
        String of frequency breaks to pass into wsclean
    freqs: np.array
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

def modify_freqs_from_ms(mslist, freqs):
    """
    This function takes a frequency array and trims it according to the frequencies available within an ms

    Parameters:
    -----------
    mslist: [str]
        Paths to MSs to get frequency limits
    freqs: np.array
        Array containing frequency breaks for wsclean

    Returns:
    --------
    mod_freq_string: str
        String for wsclean with frequency cuts corrected by ms
    mod_freqs: np.array
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


    Parameters:
    -----------
    rename_no: int
        Number of models that need to be renamed (2 -> Shift all basename models down 2)
    model_prefix: str
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

            command = f'cp {model} {model_out}'
            os.system(command)
    
    if len(pblist) > 0:
        # Remove not needed models
        pblist = pblist[rename_no:]
        for model in pblist:
            split_number = model.split("-model-pb.fits")[0]
            number = int(split_number[-4:])
            number_shift = str((number-2)).zfill(4)

            split_number = split_number[:-4] + number_shift 

            model_out = model_prefix + split_number + "-model-pb.fits"

            command = f'cp {model} {model_out}'
            os.system(command)

    # Still need to update args['wscleanskymodel'] to include prefix


def MeerKAT_antconstraint(antfile=None, ctype='all'):
    """
    Selects MeerKAT antenna names based on their distance from the array center.

    Parameters
    ----------
    antfile : str, optional
        Path to the CSV file containing MeerKAT antenna layout. If None, a default path is used.
    ctype : {'core', 'remote', 'all'}, optional
        Type of antennas to select:
            - 'core': Antennas within 1000 meters from the center.
            - 'remote': Antennas farther than 1000 meters from the center.
            - 'all': All antennas.

    Returns
    -------
    list of str
        List of selected antenna names.

    Raises
    ------
    SystemExit
        If `ctype` is not one of 'core', 'remote', or 'all'.

    Notes
    -----
    The CSV file is expected to have columns: 'Antenna', 'East', 'North', 'Up'.
    """
    if antfile is None:
        antfile = f'{datapath}/data/MeerKATlayout.csv'

    if ctype not in ['core', 'remote', 'all']:
        print('Wrong input detected, ctype needs to be in core,remote,or all')
        sys.exit()

    data = ascii.read(antfile, delimiter=';', header_start=0)
    distance = np.sqrt(data['East'] ** 2 + data['North'] ** 2 + data['Up'] ** 2)
    # print(distance
    idx_core = np.where(distance <= 1000.)
    idx_rs = np.where(distance > 1000.)
    if ctype == 'core':
        return data['Antenna'][idx_core].tolist()
    if ctype == 'remote':
        return data['Antenna'][idx_rs].tolist()
    if ctype == 'all':
        return data['Antenna'].tolist()


def round_up_to_even(number):
    """
    Round up to even number
    """
    return int(np.ceil(number / 2.) * 2)


def set_channelsout(mslist, factor=1):
    """
    Determines the number of output channels (`channelsout`) for a list of measurement sets (MS) based on the telescope type and fractional bandwidth.

    Parameters:
        mslist (list of str): List of paths to measurement sets (MS). The first MS in the list is used to determine the telescope type.
        factor (int or float, optional): Multiplicative factor to adjust the number of output channels. Default is 1.

    Returns:
        int: The computed number of output channels, rounded up to the nearest even integer.

    Notes:
        - For LOFAR and unknown telescopes, `channelsout` is calculated as `round_up_to_even(f_bw * 12 * factor)`.
        - For MeerKAT, `channelsout` is calculated as `round_up_to_even(f_bw * 13 * factor)`.
        - The function assumes the existence of `get_fractional_bandwidth` and `round_up_to_even` helper functions.
    """

    with table(mslist[0] + '/OBSERVATION', ack=False) as t:
        # Get the telescope name from the first MS in the list
        telescope = t.getcol('TELESCOPE_NAME')[0]
    
    f_bw = get_fractional_bandwidth(mslist)

    if telescope == 'LOFAR':
        channelsout = round_up_to_even(f_bw * 12 * factor)
    elif telescope == 'MeerKAT':
        channelsout = round_up_to_even(f_bw * 13 * factor) 
        # should result in channelsout=12 for L-band, with bandpass edges removed
    else:
        channelsout = round_up_to_even(f_bw * 12 * factor)
    return channelsout

def clean_up_images(imagename):
    """
    Remeoves psf, residual, beam, and dirty channel images after a WSClean run to save disk space
    
    Parameters:
    -----------
    ms : str
        The image basename used in the WSClean run
    """
    imagelist = sorted(glob.glob(imagename + '-????-*residual*.fits'))  
    imagelist += sorted(glob.glob(imagename + '-????-*dirty*.fits')) 
    imagelist += sorted(glob.glob(imagename + '-????-*psf*.fits')) 
    imagelist += sorted(glob.glob(imagename + '-????-*beam*.fits')) 
    for image in imagelist:
        os.system('rm -f ' + image)
    return

def flag_antenna_taql(ms, antennaname):
    """
    Flags all data in a Measurement Set (MS) corresponding to a specific antenna, identified by its name, using TaQL (Table Query Language).

    Parameters:
    -----------
    ms : str
        The path to the Measurement Set (MS) to be modified. This should include the full directory name of the MS.
    antennaname : str
        The name of the antenna to flag. This name should match exactly with the entry in the ANTENNA table of the MS.

    Functionality:
    --------------
    - Constructs a TaQL query to update the `FLAG` column in the MS's `MAIN` table.
    - Flags all rows where `ANTENNA1` or `ANTENNA2` corresponds to the antenna with the specified name.
    - Executes the constructed TaQL query using the `run` function (assumes `run` is defined elsewhere in your codebase to execute shell commands).

    Notes:
    ------
    - This function modifies the MS in-place; ensure you have a backup if needed.
    - The `run` function must be defined and capable of executing the constructed TaQL command in the appropriate environment.
    - Requires that the specified `antennaname` exists in the MS's ANTENNA table; otherwise, no rows will be flagged.

    Example:
    --------
    Suppose you have a Measurement Set `observation.ms` and want to flag an antenna named `DE601HBA`:

    ```python
    from facetselfcal import *
    flag_antenna_taql("observation.ms", "DE601HBA")
    ```

    This will flag all rows in `observation.ms` where either `ANTENNA1` or `ANTENNA2` corresponds to the antenna named `DE601`.

    Returns:
    --------
    None
    """
    cmd = "taql 'UPDATE " + ms + ' '
    cmd += "SET FLAG=true WHERE ANTENNA1 IN (SELECT ROWID() FROM " + ms
    cmd += "::ANTENNA WHERE NAME=\"" + antennaname + "\") OR ANTENNA2 IN "
    cmd += "(SELECT ROWID() FROM " + ms + "::ANTENNA WHERE NAME=\"" + antennaname + "\")' "
    print(cmd)
    run(cmd)
    return


def update_fitspectralpol():
    """
    Update fit spectral pol in arguments
    """

    if args['update_fitspectralpol']:
        args['fitspectralpol'] = set_fitspectralpol(args['channelsout'])
    return args['fitspectralpol']

def get_image_size(fitsimage):
    """ 
    Find the dimensions of a FITS image.
    Args:
        fitsimage (str): path to the FITS file.
    Returns:
        imsize (tuple): dimensions of the 2D image
    """
    hdulist = fits.open(fitsimage)
    shape = hdulist[0].data.squeeze().shape # squeeze out dimensions of 1
    hdulist.close()
    return shape
  
def fix_uvw(mslist):
    """
    The MeerKAT definition of UVW differs by a minus sign, but not always for some reason
    This leads to a mix of definitions inside a MS which causes problems when time averaging
    This function fixes that issue
    Parameters:
    mslist (str/list): Input Measurement Set(s) as a string or a list of strings.
    """
    mslist = [mslist] if isinstance(mslist, str) else mslist
    t = table(mslist[0] + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()
    if telescope != 'MeerKAT':
        return    

    for ms in mslist:
        cmd = "taql 'update " + ms + " set UVW=(-1.)*(mscal.UVWJ2000())'"
        print(cmd)
        run(cmd)
    return

def get_image_dynamicrange(image):
    """
    Get dynamic range of an image (peak over rms)

    Args:
        image (str): FITS image file name .
     
    Returns:
        DR (float): Dynamic range vale.
    """

    print('Compute image dynamic range (peak over rms): ', image)
    hdul = fits.open(image)
    image_rms = findrms(np.ndarray.flatten(hdul[0].data))
    DR = np.nanmax(np.ndarray.flatten(hdul[0].data)) / image_rms
    hdul.close()
    return DR


def is_stokesi_modeltype_allowed(args, telescope):
    """
    Determine if Stokes I compression is allowed for MODEL_DATA-type columns.

    Args:
        args (dict): Dictionary of arguments, including 'single_dual_speedup', 
                     'disable_primary_beam', and 'soltype_list'.
        telescope (str): The telescope name (e.g., 'LOFAR').

    Returns:
        bool: True if Stokes I compression is allowed, False otherwise.
    """
    if telescope == 'LOFAR': 
        if not args['single_dual_speedup']:
            if not args['disable_primary_beam']:
                return False # so in this case we want the keep the primary beam polarization information    
    
    notallowed_list = ['complexgain', 'amplitudeonly', 'phaseonly', 'fulljones', 'rotation', 'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude', 'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase', 'phaseonly_phmin', 'rotation_phmin', 'phaseonly_slope', 'scalarphasediff', 'scalarphasediffFR', 'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']
    for soltype in args['soltype_list']:
        if soltype in notallowed_list: return False
    return True




def update_channelsout(selfcalcycle, mslist):
    """
    Dynamically updates the 'channelsout' parameter in the global 'args' dictionary based on the image dynamic range and telescope type.

    Parameters:
        selfcalcycle (int): The current self-calibration cycle number.
        mslist (list of str): List of Measurement Set (MS) file paths.

    Returns:
        int or float: The updated value of 'channelsout' in the 'args' dictionary.

    Behavior:
        - Checks if 'update_channelsout' is enabled in 'args'.
        - Determines the telescope name from the first MS in the list.
        - Constructs the image filename based on the imaging parameters in 'args'.
        - Computes the dynamic range of the image.
        - Adjusts 'channelsout' in 'args' according to the dynamic range thresholds and telescope type.
        - Returns the updated 'channelsout' value.
    """
    if args['update_channelsout']:
        t = table(mslist[0] + '/OBSERVATION', ack=False)
        telescope = t.getcol('TELESCOPE_NAME')[0]
        t.close()
        # set stackstr
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
        dr = get_image_dynamicrange(imagename)

        if dr > 1500 and telescope == 'LOFAR':
            args['channelsout'] = set_channelsout(mslist, factor=2)
        if dr > 3000 and telescope == 'LOFAR':
            args['channelsout'] = set_channelsout(mslist, factor=3)
        if dr > 6000 and telescope == 'LOFAR':
            args['channelsout'] = set_channelsout(mslist, factor=4)
        if dr > 30000 and telescope == 'MeerKAT':
            args['channelsout'] = set_channelsout(mslist, factor=1.5)
        if dr > 60000 and telescope == 'MeerKAT':
            args['channelsout'] = set_channelsout(mslist, factor=2)
        if dr > 90000 and telescope == 'MeerKAT':
            args['channelsout'] = set_channelsout(mslist, factor=3)
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
    elif channelsout <= 12:
        fitspectralpol = 7
    elif channelsout > 12:
        fitspectralpol = 9
    else:
        print('channelsout', channelsout)
        raise Exception(f'channelsout has an invalid value: {channelsout}')
    return fitspectralpol


def get_fractional_bandwidth(mslist):
    """
    Compute fractional bandwidth of a list of MS
    input mslist: list of ms
    return fractional bandwidth
    """
    freqaxis = []
    for ms in mslist:
        t = table(ms + '/SPECTRAL_WINDOW', readonly=True, ack=False)
        freq = t.getcol('CHAN_FREQ')[0]
        t.close()
        freqaxis.append(freq)
    freqaxis = np.hstack(freqaxis)
    f_bw = (np.max(freqaxis) - np.min(freqaxis)) / np.min(freqaxis)

    if f_bw == 0.0:  # single channel data
        t = table(mslist[0] + '/SPECTRAL_WINDOW', readonly=True, ack=False)  # just take the first and assume this is ok
        f_bw = t.getcol('TOTAL_BANDWIDTH')[0] / np.min(freqaxis)
        t.close()
    return f_bw


def remove_column_ms(mslist, colname):
    """
    Remove a column from a Measurement Set or a list of Measurement Sets.

    Parameters:
    mslist (str/list): Input Measurement Set(s) as a string or a list of strings.
    colname (str): Column name to be removed.
    """
    mslist = [mslist] if isinstance(mslist, str) else mslist

    for ms in mslist:
        with table(ms, readonly=False, ack=False) as ts:
            ts.removecols([colname])


def merge_splitted_h5_ordered(modeldatacolumnsin, parmdb_out, clean_up=False):
    h5list_sols = []
    for colid, coln in enumerate(modeldatacolumnsin):
        h5list_sols.append('Dir' + str(colid).zfill(2) + '.h5')
    print('These are the h5 that need merging:', h5list_sols)
    if os.path.isfile(parmdb_out):
        os.system('rm -f ' + parmdb_out)

    f = open('facetdirections.p', 'rb')
    sourcedir = pickle.load(f)  # units are radian
    f.close()
    parmdb_merge_list = []

    for direction in sourcedir:
        print(direction)
        c1 = SkyCoord(direction[0] * units.radian, direction[1] * units.radian, frame='icrs')
        distance = 1e9
        # print(c1)
        for hsol in h5list_sols:
            H5 = tables.open_file(hsol, mode='r')
            dd = H5.root.sol000.source[:][0]
            H5.close()
            ra, dec = dd[1]
            c2 = SkyCoord(ra * units.radian, dec * units.radian, frame='icrs')
            angsep = c1.separation(c2).to(units.degree)
            # print(c2)

            # print(hsol, angsep.value, '[degree]')
            if angsep.value < distance:
                distance = angsep.value
                matchging_h5 = hsol
        parmdb_merge_list.append(matchging_h5)
        print('separation direction entry and h5 entry is:', distance, matchging_h5)
        assert abs(distance) < 0.00001  # there should always be a close to perfect match

    merge_h5(h5_out=parmdb_out, h5_tables=parmdb_merge_list, propagate_weights=True, convert_tec=False)

    if clean_up:
        for h5 in h5list_sols:
            os.system('rm -f ' + h5)
    return

def read_MeerKAT_wscleanmodel_5spix(filename, outfile):
    '''
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 5-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb
    Parameters:
    filename (str): input filename
    outfile (str): ouptput filename
    '''
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = data['col11'].astype(str)
    data['col11'] = np.char.add(data['col11'], ',')

    #add , after third spectral index element    
    data['col12'] = data['col12'].astype(str)
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add , after third spectral index element    
    data['col13'] = np.char.add(data['col13'], ',')
    
    #add , after LogarithmicSI element    
    data['col14'] = np.char.add(data['col14'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col15'] = data['col15'].astype(str)
    data['col15'] = np.char.add(data['col15'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # add formatting line
    formatstr = "'1iFormat = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation'"
    os.system("sed -i " + formatstr + " " + outfile)
    return    

def read_MeerKAT_wscleanmodel_4spix(filename, outfile):
    '''
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 4-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb
    Parameters:
    filename (str): input filename
    outfile (str): ouptput filename
    '''
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = data['col11'].astype(str)
    data['col11'] = np.char.add(data['col11'], ',')
    
    #add , after third spectral index element    
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add , after LogarithmicSI element    
    data['col13'] = np.char.add(data['col13'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col14'] = data['col14'].astype(str)
    data['col14'] = np.char.add(data['col14'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # add formatting line
    formatstr = "'1iFormat = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation'"
    os.system("sed -i " + formatstr + " " + outfile)
    return    



def read_MeerKAT_wscleanmodel_3spix(filename, outfile):
    '''
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 3-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb
    Parameters:
    filename (str): input filename
    outfile (str): ouptput filename
    '''
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = np.char.add(data['col11'], ',')


    #add , after LogarithmicSI element    
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col13'] = data['col13'].astype(str)
    data['col13'] = np.char.add(data['col13'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # add formatting line
    formatstr = "'1iFormat = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation'"
    os.system("sed -i " + formatstr + " " + outfile)
    return    

def copy_over_solutions_from_skipped_directions(modeldatacolumnsin, id_kept):
    """
   modeldatacolumnsin: all modeldatacolumns
   id_kept: indices of the modeldatacolumns kept in the solve id_kept
   """
    h5list_sols = []
    h5list_empty = []
    for colid, coln in enumerate(modeldatacolumnsin):
        if colid >= len(id_kept):
            h5list_empty.append('Dir' + str(colid).zfill(2) + '.h5')
        else:
            h5list_sols.append('Dir' + str(colid).zfill(2) + '.h5')
    print('These h5 have solutions:', h5list_sols)
    print('These h5 are empty:', h5list_empty)

    # fill the empty directions (those that were removed and not solve) with the closest valid solutions
    for h5 in h5list_empty:
        hempty = tables.open_file(h5, mode='a')
        direction = hempty.root.sol000.source[:][0]
        ra, dec = direction[1]
        c1 = SkyCoord(ra * units.radian, dec * units.radian, frame='icrs')
        # print(c1)

        distance = 1e9
        for h5sol in h5list_sols:
            hsol = tables.open_file(h5sol, mode='r')
            directionsol = hsol.root.sol000.source[:][0]
            rasol, decsol = directionsol[1]
            c2 = SkyCoord(rasol * units.radian, decsol * units.radian, frame='icrs')
            angsep = c1.separation(c2).to(units.degree)
            # print(h5, h5sol, angsep.value, '[degree]')
            if angsep.value < distance:
                distance = angsep.value
                matchging_h5 = h5sol
            hsol.close()
        print(h5 + ' needs solutions copied from ' + matchging_h5, distance)

        # copy over the values
        with tables.open_file(matchging_h5, mode='r') as hmatch:

            for sol_type in ['phase000', 'amplitude000', 'tec000', 'rotation000', 'rotationmeasure000']:
                try:
                    getattr(hempty.root.sol000, sol_type).val[:] = np.copy(getattr(hmatch.root.sol000, sol_type).val[:])
                    print(f'Copied over {sol_type}')
                except AttributeError:
                    pass

        hempty.flush()
        hempty.close()
    return


def filter_baseline_str_removestations(stationlist):
    """
    Generates a baseline filter string to exclude specific stations from processing.

    This function constructs a string that can be used to filter out baselines 
    involving specific stations in a radio interferometry dataset. The filter 
    string is formatted to exclude baselines that include any of the stations 
    provided in the input list.

    Args:
        stationlist (list of str): A list of station names to be excluded.

    Returns:
        str: A formatted baseline filter string that excludes the specified stations.
    """
    fbaseline = "'"
    for station_id, station in enumerate(stationlist):
        fbaseline = fbaseline + "!" + station + "&&*"
        if station_id + 1 < len(stationlist):
            fbaseline = fbaseline + ";"
    return fbaseline + "'"


def return_antennas_highflaggingpercentage(ms, percentage=85):
    """
    Identifies antennas with a high percentage of flagged data in a Measurement Set (MS).

    This function queries the provided Measurement Set (MS) to find antennas where the 
    percentage of flagged data exceeds the specified threshold. It uses the TaQL (Table Query 
    Language) to perform the query and returns a list of antenna names that meet the criteria.

    Args:
        ms (str): The path to the Measurement Set (MS) to be analyzed.
        percentage (float, optional): The flagging percentage threshold. Antennas with a 
            flagging percentage greater than this value will be returned. Default is 0.85 
            (85%).

    Returns:
        list: A list of antenna names (str) that have a flagging percentage above the specified 
        threshold.

    Example:
        >>> flagged_antennas = return_antennas_highflaggingpercentage("path/to/ms", 0.9)
        Finding stations with a flagging percentage above 90.0 ....
        Found: ['ANT1', 'ANT2']
    """
    print('Finding stations with a flagging percentage above ' + str(percentage) + ' ....')
    t = taql(""" SELECT antname, gsum(numflagged) AS numflagged, gsum(numvis) AS numvis,
       gsum(numflagged)/gsum(numvis) as percflagged FROM [[ SELECT mscal.ant1name() AS antname, ntrue(FLAG) AS numflagged, count(FLAG) AS numvis FROM """ + ms + """ ],[ SELECT mscal.ant2name() AS antname, ntrue(FLAG) AS numflagged, count(FLAG) AS numvis FROM """ + ms + """ ] ] GROUP BY antname HAVING percflagged > """ + str(
        percentage/100.) + """ """)

    flaggedants = [row["antname"] for row in t]
    print('Found:', flaggedants)
    t.close()
    return flaggedants


def create_empty_fitsimage(ms, imsize, pixelsize, outfile):
    """
    Create an empty FITS image with specified size and pixel scale.

    This function generates a FITS file filled with zeros, with the image center
    aligned to the phase center of the provided measurement set (MS). The FITS
    header is populated with appropriate WCS (World Coordinate System) information.

    Parameters:
        ms (str): Path to the measurement set (MS) file. Used to determine the
                  phase center coordinates.
        imsize (int): Size of the image in pixels (assumes a square image).
        pixelsize (float): Pixel size in arcseconds.
        outfile (str): Path to the output FITS file.

    Returns:
        None: The function writes the FITS file to the specified output path.

    Notes:
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

    Parameters:
        wscleanskymodel (str): The sky model string. If not None, WSCLEAN 
                                  will be used; otherwise, DP3 will be used.

    Returns:
        str: 'WSCLEAN' if a sky model is provided, otherwise 'DP3'.
    """
    if wscleanskymodel is not None:
        return 'WSCLEAN'
    else:
        return 'DP3'

def timebase(fov, ms, tau=0.995):
    """
    Find DP3 timebase value for BDA
    
    FoV: field of view in degrees
    ms: Measurement set
    tau: Peak flux loss factor

    Using formulas from Bridle & Schwab (1999)
    """

    with table(ms+"::SPECTRAL_WINDOW", ack=False) as t:
        f = t.getcol("CHAN_FREQ").max()
        
    with table(ms, ack=False) as t:
        time = np.unique(t.getcol('TIME'))
        dt = np.abs(time[1] - time[0])

    tb = 5.*10**14*np.sqrt(1-tau)/(f*fov*dt)
    
    return tb

def bda_mslist(mslist, pixsize, imsize, dryrun=False, metadata_compression=True):
    """
    BDA compress list of MS with DP3
    
    pixsize: pixel size in arcsec
    imsize: image size in pixels
    mslist: list of Measurement sets
    dryrun: create the actual MS (otherwise only bda_mslist is made)
    
    returns
    bda_mslist: list of BDA Measurement sets
    """
    bda_mslist = []
    for ms in mslist:
        cmd = 'DP3 msin=' + ms + ' steps=[bda] bda.type=bdaaverage msout=' + ms + '.bda '
        cmd += 'bda.maxinterval=300 '
        cmd += 'bda.timebase=' + str(timebase(imsize*pixsize/3600.,ms)) + ' '
        if not metadata_compression:
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '
        print(cmd)
        if not dryrun:
            if os.path.isdir(ms + '.bda'):
                os.system('rm -rf ' + ms + '.bda')
            run(cmd)
        bda_mslist.append(ms + '.bda')
    return bda_mslist

def getAntennas(ms):
    """
    Retrieve a list of antenna names from a Measurement Set (MS).

    Args:
        ms (str): The path to the Measurement Set (MS) directory.

    Returns:
        list: A list of antenna names as strings.

    Notes:
        This function accesses the 'ANTENNA' table within the provided
        Measurement Set to extract the antenna names.
    """
    t = table(ms + "/ANTENNA", readonly=True, ack=False)
    antennas = t.getcol('NAME')
    t.close()
    return antennas


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
    """
    import lsmtool
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
            print("USING the exising skymodel in " + SkymodelPath)
            return (0)
        else:
            download_flag = True
    elif DoDownload.upper() == "FALSE" or DoDownload.upper() == "NO":
        if FileExists:
            print("USING the exising skymodel in " + SkymodelPath)
            return (0)
        else:
            raise ValueError(
                "download_tgss_skymodel_target: Path: \"%s\" does not exist and skymodel download is disabled!" % (
                    SkymodelPath))

    # If we got here, then we are supposed to download the skymodel.
    assert download_flag is True  # Jaja, belts and suspenders...
    print("DOWNLOADING skymodel for the target into " + SkymodelPath)

    # Reading a MS to find the coordinate (pyrap)
    [RATar, DECTar] = grab_coord_MS(ms_input)

    # Downloading the skymodel, skip after five tries
    errorcode = 1
    tries = 0
    while errorcode != 0 and tries < 5:
        if Source == 'TGSS':
            errorcode = os.system(
                "wget -O " + SkymodelPath + " \'http://tgssadr.strw.leidenuniv.nl/cgi-bin/gsmv5.cgi?coord=" + str(
                    RATar) + "," + str(DECTar) + "&radius=" + str(Radius) + "&unit=deg&deconv=y\' ")
        elif Source == 'GSM':
            errorcode = os.system(
                "wget -O " + SkymodelPath + " \'https://lcs165.lofar.eu/cgi-bin/gsmv1.cgi?coord=" + str(
                    RATar) + "," + str(DECTar) + "&radius=" + str(Radius) + "&unit=deg&deconv=y\' ")
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


def concat_ms_wsclean_facetimaging(mslist, h5list=None, concatms=True):
    keyfunct = lambda x: ' '.join(sorted(getAntennas(x)))

    MSs_list = sorted(mslist, key=keyfunct)  # needs to be sorted

    groups = []
    for k, g in groupby(MSs_list, keyfunct):
        groups.append(list(g))
    print(f"Found {len(groups)} groups of datasets with same antennas.")

    for i, group in enumerate(groups, start=1):
        antennas = ', '.join(getAntennas(group[0]))
        print(f"WSClean MS group {i}: {group}")
        print(f"List of antennas: {antennas}")

    MSs_files_clean = []
    H5s_files_clean = []
    for g, group in enumerate(groups):
        if os.path.isdir(f'wsclean_concat_{g}.ms') and concatms:
            os.system(f'rm -rf wsclean_concat_{g}.ms')

        if h5list is not None:
            h5group = []
            for ms in group:
                h5group.append(time_match_mstoH5(h5list, ms))
            print('------------------------------')
            print('MS group and matched h5 group', group, h5group)
            print('------------------------------')
            if os.path.isfile(f'wsclean_concat_{g}.h5'):
                os.system(f'rm -rf wsclean_concat_{g}.h5')
            merge_h5(h5_out=f'wsclean_concat_{g}.h5', h5_tables=h5group, propagate_weights=True, time_concat=True)
            H5s_files_clean.append(f'wsclean_concat_{g}.h5')
        if concatms:
            print(f'taql select from {group} giving wsclean_concat_{g}.ms as plain')
            run(f'taql select from {group} giving wsclean_concat_{g}.ms as plain')
        MSs_files_clean.append(f'wsclean_concat_{g}.ms')

    # MSs_files_clean = ' '.join(MSs_files_clean)

    # print('Use the following ms files as input in wsclean:')
    # print(MSs_files_clean)

    return MSs_files_clean, H5s_files_clean


def check_for_BDPbug_longsolint(mslist, facetdirections):
    dirs, solints, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, 1000)

    if solints is None:
        return

    solint_reformat = np.array(solints)
    for ms in mslist:
        t = table(ms, readonly=True, ack=False)
        time = np.unique(t.getcol('TIME'))
        t.close()
        print('------------' + ms)
        ms_ntimes = len(time)
        # print(ms, ms_ntimes)
        for solintcycle_id, tmpval in enumerate(solint_reformat[0]):
            print(' --- ' + str('pertubation cycle=') + str(solintcycle_id) + '--- ')
            solints_cycle = solint_reformat[:, solintcycle_id]
            solints = [int(format_solint(x, ms)) for x in solints_cycle]
            print('Solint unmodified per direction', solints)
            solints = tweak_solints(solints,  ms_ntimes=ms_ntimes)
            print('Solint tweaked per direction   ', solints)

            lcm = math.lcm(*solints)
            divisors = [int(lcm / i) for i in solints]

        #solints = [int(format_solint(x, ms)) for x in solint]
        #solints = tweak_solints(solints, ms_ntimes=ms_ntimes)
        #lcm = math.lcm(*solints)
        #divisors = [int(lcm / i) for i in solints]
        #cmd += 'ddecal.solint=' + str(lcm) + ' '
        #cmd += 'ddecal.solutions_per_direction=' + "'" + str(divisors).replace(' ', '') + "' "

            print('Solint passed to DP3 would be:', lcm, ' --Number of timeslots in MS:', ms_ntimes)
            if lcm > int(1.5*ms_ntimes):
                print('Bad divisor for solutions_per_direction DDE solve. DP3 Solint > number of timeslots in the MS')
                sys.exit()
        print('------------')

    return


def selfcal_animatedgif(fitsstr, outname):
    """
    Generates an animated GIF from a FITS file using the DS9 visualization tool.

    This function constructs a command to run DS9 with specific parameters to 
    create an animated GIF from the provided FITS file. The GIF is saved to the 
    specified output file.

    Args:
        fitsstr (str): The path to the input FITS file.
        outname (str): The name of the output GIF file.

    Returns:
        None
    """
    limit_min = -250e-6
    limit_max = 2.5e-2
    cmd = 'ds9 ' + fitsstr + ' '
    cmd += '-single -view basic -frame first -geometry 800x800 -zoom to fit -sqrt -scale limits '
    cmd += str(limit_max) + ' ' + str(limit_max) + ' '
    cmd += '-cmap ch05m151008 -colorbar lock yes -frame lock wcs '
    cmd += '-lock scalelimits yes -movie frame gif 10 '
    cmd += outname + ' -quit'
    run(cmd)
    return


def find_closest_ddsol(h5, ms):
    """
    Find the closest direction in a multi-directional H5 file to the phase center of a Measurement Set (MS).

    Parameters
    ----------
    h5 : str
        Path to the H5 file containing directional solutions.
    ms : str
        Path to the Measurement Set (MS) whose phase center is used for comparison.

    Returns
    -------
    str
        The name of the closest direction in the H5 file to the phase center of the MS.

    Notes
    -----
    - This function uses the SkyCoord class from Astropy to calculate angular separations.
    - The closest direction is determined based on the smallest angular separation.
    """

    t2 = table(ms + '::FIELD', ack=False)
    phasedir = t2.getcol('PHASE_DIR').squeeze()
    t2.close()
    c1 = SkyCoord(phasedir[0] * units.radian, phasedir[1] * units.radian, frame='icrs')
    H5 = tables.open_file(h5)
    distance = 1e9  # just a big number
    for direction_id, direction in enumerate(H5.root.sol000.source[:]):
        ra, dec = direction[1]
        c2 = SkyCoord(ra * units.radian, dec * units.radian, frame='icrs')
        angsep = c1.separation(c2).to(units.degree)
        print(direction[0], angsep.value, '[degree]')
        if angsep.value < distance:
            distance = angsep.value
            dirname = direction[0]
    H5.close()
    return dirname


def set_beamcor(ms, beamcor_var):
    """
    Determines whether to apply beam correction for a measurement set (MS).

    Parameters:
    ms (str): The path to the measurement set (MS) file.
    beamcor_var (str): A string indicating whether to apply beam correction. 
                       Possible values are:
                       - 'no': Do not apply beam correction.
                       - 'yes': Apply beam correction.
                       - 'auto': Automatically determine based on observation data.

    Returns:
    bool: True if beam correction should be applied, False otherwise.

    Behavior:
    - If `beamcor_var` is 'no', beam correction is not applied.
    - If `beamcor_var` is 'yes', beam correction is applied.
    - If `beamcor_var` is 'auto', the function checks:
        - If the telescope is not LOFAR, beam correction is not applied.
        - If the telescope is LOFAR, it calculates the angular separation between 
          the phase center and the applied beam direction. If the separation is 
          less than 10 arcseconds, beam correction is not applied; otherwise, it is applied.

    Notes:
    - The function uses the `astropy.coordinates.SkyCoord` class to calculate angular separation.
    - Beam keywords are added to the MS if they are missing, using the `beam_keywords` function.
    - Logs information about the decision process and angular separation.
    """

    if beamcor_var == 'no':
        logger.info('Run DP3 applybeam: no')
        return False
    if beamcor_var == 'yes':
        logger.info('Run DP3 applybeam: yes')
        return True

    t = table(ms + '/OBSERVATION', ack=False)
    if t.getcol('TELESCOPE_NAME')[0] != 'LOFAR':
        t.close()
        logger.info('Run DP3 applybeam: no (because we are not using LOFAR observations)')
        return False
    t.close()

    # If we arrive here beamcor_var was set to auto and we are using a LOFAR observation
    # so we assume that beam was taken out in the field center only if user has set 'auto'
    # In case we have old prefactor data and no beam keywords available beam_keywords(ms) will add these
    tmpvar =  beam_keywords(ms) # just put output in tmp variable (not used)

    # now check if beam was taken out in the current phase center
    t = table(ms, readonly=True, ack=False)
    beamdir = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_DIR')

    t2 = table(ms + '::FIELD', ack=False)
    phasedir = t2.getcol('PHASE_DIR').squeeze()
    t.close()
    t2.close()

    c1 = SkyCoord(beamdir['m0']['value'] * units.radian, beamdir['m1']['value'] * units.radian, frame='icrs')
    c2 = SkyCoord(phasedir[0] * units.radian, phasedir[1] * units.radian, frame='icrs')
    # Calculate angular separation between phase center and direction in arcseconds
    angsep = c1.separation(c2).to(units.arcsec)

    # angular_separation is recent astropy functionality, do not use, instead use the older SkyCoord.seperation
    # angsep = 3600.*180.*astropy.coordinates.angular_separation(phasedir[0], phasedir[1], beamdir['m0']['value'], beamdir['m1']['value'])/np.pi

    print('Angular separation between phase center and applied beam direction is', angsep.value, '[arcsec]')
    logger.info(
        'Angular separation between phase center and applied beam direction is:' + str(angsep.value) + ' [arcsec]')

    # of less than 10 arcsec than do beam correction
    if angsep.value < 10.0:
        logger.info('Run DP3 applybeam: no')
        return False
    else:
        logger.info('Run DP3 applybeam: yes')
        return True


def isfloat(num):
    """Check if a value is a float."""
    return isinstance(num, float) or (isinstance(num, str) and num.replace('.', '', 1).isdigit())


def find_prime_factors(n):
    """
    Find the prime factors of a given integer.

    This function computes the prime factors of the input integer `n` 
    and returns them as a list. It first extracts all factors of 2, 
    then iterates through odd numbers to find other prime factors.

    Args:
        n (int): The integer to factorize. Must be greater than 0.

    Returns:
        list: A list of integers representing the prime factors of `n`.

    Example:
        >>> find_prime_factors(28)
        [2, 2, 7]
    """
    factorlist = []
    num = n
    while n % 2 == 0:
        factorlist.append(2)
        n = n / 2

    for i in range(3, int(num / 2) + 1, 2):
        while n % i == 0:
            factorlist.append(i)
            n = n / i
        if n == 1:
            break
    return factorlist


def tweak_solintsold(solints, solval=20):
    """
    Adjusts a list of solution intervals by rounding up values greater than a 
    specified threshold to the nearest even number.

    Parameters:
    solints (list of int): A list of solution intervals to be adjusted.
    solval (int, optional): The threshold value. Solution intervals greater 
        than this value will be rounded up to the nearest even number. 
        Defaults to 20.

    Returns:
    list of int: A list of adjusted solution intervals.
    """
    solints_return = []
    for sol in solints:
        soltmp = sol
        if soltmp > solval:
            soltmp += int(soltmp & 1)  # round up to even
        solints_return.append(soltmp)
    return solints_return


def tweak_solints(solints, solvalthresh=11, ms_ntimes=None):
    """
    Returns modified solints that can be factorized by 2 or 3 if input contains number >= solvalthresh
    """
    solints_return = []
    if np.max(solints) < solvalthresh:
        return solints
    
    # shorten solint to length of MS at most
    if ms_ntimes is not None:
        solints_copy = []
        for solint in solints:
            if solint > ms_ntimes:
                solints_copy.append(ms_ntimes)
            else:
                solints_copy.append(solint)
        solints = solints_copy
        
    possible_solints = listof2and3prime(startval=2, stopval=10000)
    if ms_ntimes is not None:
        possible_solints = remove_bad_endrounding(possible_solints, ms_ntimes)
    
    for sol in solints:
        solints_return.append(find_nearest(possible_solints, sol))

    return solints_return


def tweak_solints_single(solint, ms_ntimes, solvalthresh=11, ):
    """
    def tweak_solints_single(solint, ms_ntimes, solvalthresh=11):
        Adjusts the given solution interval (`solint`) to avoid having a small number 
        of leftover time slots near the end of the measurement set (ms).

        Parameters:
        -----------
        solint : int
            The initial solution interval to be adjusted.
        ms_ntimes : int
            The total number of time slots in the measurement set.
        solvalthresh : int, optional
            Threshold value for the solution interval. If `solint` is less than this 
            threshold, it is returned unchanged. Default is 11.

        Returns:
        --------
        int
            A modified solution interval that minimizes leftover time slots while 
            being as close as possible to the original `solint`.

    """
    if np.max(solint) < solvalthresh:
        return solint

    possible_solints = np.arange(1, 2 * solint)
    possible_solints = remove_bad_endrounding(possible_solints, ms_ntimes)

    return find_nearest(possible_solints, solint)


def remove_bad_endrounding(solints, ms_ntimes, ignorelessthan=11):
    """
    Filters a list of solution intervals (solints) to remove those that result in 
    significant rounding errors when dividing the total number of timeslots (ms_ntimes) 
    by the solution interval. Additionally, excludes solution intervals smaller than 
    a specified threshold.

    Args:
        solints (list of int): A list of possible solution intervals to evaluate.
        ms_ntimes (int): The total number of timeslots in the measurement set (MS).
        ignorelessthan (int, optional): The minimum solution interval to consider. 
            Solution intervals smaller than this value will be excluded. Defaults to 11.

    Returns:
        list of int: A filtered list of solution intervals that meet the criteria.
    """
    solints_out = []
    for solint in solints:
        if (float(ms_ntimes) / float(solint)) - (
                np.floor(float(ms_ntimes) / float(solint))) > 0.5 or solint < ignorelessthan:
            solints_out.append(solint)
    return solints_out


def listof2and3prime(startval=2, stopval=10000):
    """
    Generate a list of integers between `startval` and `stopval` (exclusive) 
    whose largest prime factor is either 2 or 3.

    Args:
        startval (int, optional): The starting value of the range (inclusive). Defaults to 2.
        stopval (int, optional): The ending value of the range (exclusive). Defaults to 10000.

    Returns:
        list: A list of integers satisfying the condition, including the initial value 1.
    """
    solint = [1]
    for i in np.arange(startval, stopval):
        factors = find_prime_factors(i)
        if len(factors) > 0:
            if factors[-1] == 2 or factors[-1] == 3:
                solint.append(i)
    return solint


def find_nearest(array, value):
    """
    Find the nearest value in an array to a given target value.

    Parameters:
    array (array-like): The input array to search. It will be converted to a NumPy array if not already one.
    value (float or int): The target value to find the closest match for in the array.

    Returns:
    float or int: The value from the array that is closest to the target value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_time_preavg_factor_LTAdata(ms):
    """
    Get time pre-averaging factor (given by demixer.timestep)
    :param ms: measurement set
    :return: averaging integer
    """
    parse_str = "demixer.timestep="
    parsed_history = parse_history(ms, parse_str)
    avg_num = re.findall(r'\d+', parsed_history.replace(parse_str, ''))[0]
    if avg_num.isdigit():
        factor = int(float(avg_num))
        if factor != 1:
            print("WARNING: " + ms + " time has been pre-averaged with factor " + str(
                factor) + ". This might cause time smearing effects.")
        return factor
    elif isfloat(avg_num):
        factor = float(avg_num)
        print("WARNING: parsed factor in " + ms + " is not a digit but a float")
        return factor
    else:
        print("WARNING: parsed factor in " + ms + " is not a float or digit")
        return None


def add_dummyms(msfiles):
    """
    Add dummy ms to create a regular freuqency grid when doing a concat with DPPP
    """
    if len(msfiles) == 1:
        return msfiles
    keyname = 'REF_FREQUENCY'
    freqaxis = []
    newmslist = []

    # Check for wrong REF_FREQUENCY which happens after a DPPP split in frequency
    for ms in msfiles:
        with table(ms + '/SPECTRAL_WINDOW', readonly=True) as t:
            freq = t.getcol('REF_FREQUENCY')[0]
        freqaxis.append(freq)
    freqaxis = np.sort(np.array(freqaxis))
    minfreqspacing = np.min(np.diff(freqaxis))
    if minfreqspacing == 0.0:
        keyname = 'CHAN_FREQ'

    freqaxis = []
    for ms in msfiles:
        with table(ms + '/SPECTRAL_WINDOW', readonly=True) as t:
            if keyname == 'CHAN_FREQ':
                freq = t.getcol(keyname)[0][0]
            else:
                freq = t.getcol(keyname)[0]
        freqaxis.append(freq)

    # put everything in order of increasing frequency
    freqaxis = np.array(freqaxis)
    idx = np.argsort(freqaxis)

    freqaxis = freqaxis[np.array(tuple(idx))]
    sortedmslist = list(msfiles[i] for i in idx)
    freqspacing = np.diff(freqaxis)
    minfreqspacing = np.min(np.diff(freqaxis))

    # insert dummies in the ms list if needed
    count = 0
    newmslist.append(sortedmslist[0])  # always start with the first ms the list
    for msnumber, ms in enumerate(sortedmslist[1::]):
        if int(round(freqspacing[msnumber] / minfreqspacing)) > 1:
            ndummy = int(round(freqspacing[msnumber] / minfreqspacing)) - 1

            for dummy in range(ndummy):
                newmslist.append('dummy' + str(count) + '.ms')
                print('Added dummy:', 'dummy' + str(count) + '.ms')
                count = count + 1
        newmslist.append(ms)

    print('Updated ms list with dummies inserted to create a regular frequency grid')
    print(newmslist)
    return newmslist


def number_of_unique_obsids(msfiles):
    """
    Basic function to get numbers of observations based on first part of ms name
    (assumes one uses "_" here)

     Args:
         msfiles (list): the list of ms
     Returns:
         reval (int): number of observations

    """
    obsids = []
    for ms in msfiles:
        obsids.append(os.path.basename(ms).split('_')[0])
        print('Using these observations ', np.unique(obsids))
    return len(np.unique(obsids))


def getobsmslist(msfiles, observationnumber):
    """
    Generate a list of measurement sets (MS) belonging to the same observation.

    This function groups measurement sets by their observation ID, which is 
    extracted from the filenames of the provided MS files. It then returns 
    the list of MS files corresponding to the specified observation number.

    Parameters:
    -----------
    msfiles : list of str
        A list of file paths to the measurement sets (MS).
    observationnumber : int
        The index of the observation to extract (0-based).

    Returns:
    --------
    list of str
        A list of file paths to the measurement sets belonging to the 
        specified observation.

    Notes:
    ------
    - The observation ID is assumed to be the first part of the filename, 
      separated by an underscore ('_').
    - The `observationnumber` parameter corresponds to the index of the 
      unique observation IDs in the order they appear in the input list.
    """
    obsids = []
    for ms in msfiles:
        obsids.append(os.path.basename(ms).split('_')[0])
    obsidsextract = np.unique(obsids)[observationnumber]
    mslist = []
    for ms in msfiles:
        if (os.path.basename(ms).split('_')[0]) == obsidsextract:
            mslist.append(ms)
    return mslist


def mscolexist(ms, colname):
    """
    Check if a column exists in a measurement set.

    This function verifies whether a specified column name exists in a given
    measurement set directory. It returns `True` if the column exists and `False`
    otherwise.

    Args:
        ms (str): The path to the measurement set directory.
        colname (str): The name of the column to check for existence.

    Returns:
        bool: `True` if the column exists in the measurement set, `False` otherwise.

    Raises:
        None: This function does not raise any exceptions, but it assumes that the
        `os` module and `table` class from `casacore.tables` are properly imported.
    """
    if os.path.isdir(ms):
        with table(ms, readonly=True, ack=False) as t:
            colnames = t.colnames()
            if colname in colnames:  # check if the column is in the list
                exist = True
            else:
                exist = False
    else:
        exist = False  # ms does not exist
    return exist


def concat_ms_from_same_obs(mslist, outnamebase, colname='DATA', dysco=True, metadata_compression=True):
    for observation in range(number_of_unique_obsids(mslist)):
        # insert dummies for completely missing blocks to create a regular freuqency grid for DPPP
        obs_mslist = getobsmslist(mslist, observation)
        obs_mslist = add_dummyms(obs_mslist)

        msoutconcat = outnamebase + '_' + str(observation) + '.ms'
        msfilesconcat = []

        # remove ms from the list where column DATA_SUB does not exist (to prevent NDPPP crash)
        for msnumber, ms in enumerate(obs_mslist):
            if os.path.isdir(ms):
                if mscolexist(ms, colname):
                    msfilesconcat.append(ms)
                else:
                    msfilesconcat.append('missing' + str(msnumber))
            else:
                msfilesconcat.append('missing' + str(msnumber))

            #  CONCAT
            cmd = 'DP3 msin="' + str(msfilesconcat) + '" msin.orderms=False '
            cmd += 'steps=[] '
            cmd += 'msin.datacolumn=%s msin.missingdata=True ' % colname
            cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
            if dysco:
                cmd += 'msout.storagemanager=dysco '
                cmd += 'msout.storagemanager.weightbitrate=16 '
            if not metadata_compression:
                cmd += 'msout.uvwcompression=False '
                cmd += 'msout.antennacompression=False '
                cmd += 'msout.scalarflags=False '
            cmd += 'msout=' + msoutconcat + ' '
            if os.path.isdir(msoutconcat):
                os.system('rm -rf ' + msoutconcat)
        run(cmd, log=False)
    return


def fix_equidistant_times(mslist, dryrun, dysco=True, metadata_compression=False):
    with table(mslist[0] + '/OBSERVATION', ack=False) as t:
        telescope = t.getcol('TELESCOPE_NAME')[0]
    mslist_return = []
    mslist_splitting_performed = []
    for ms in mslist:
        if telescope != 'LOFAR':
            if check_equidistant_times([ms], stop=False, return_result=True):
                print(ms + ' has a regular time axis')
                mslist_return.append(ms)
                mslist_splitting_performed.append(False)
            else:
                ms_path = regularize_ms(ms, overwrite=True, dryrun=dryrun)
                # Do the splitting
                mslist_return = mslist_return + split_ms(ms_path, overwrite=True,
                                                         prefix=os.path.basename(ms_path), return_mslist=True,
                                                         dryrun=dryrun, dysco=dysco, metadata_compression=metadata_compression)
                mslist_splitting_performed.append(True)
        else:
            mslist_return.append(ms)
            mslist_splitting_performed.append(False)
    return sorted(mslist_return), all(mslist_splitting_performed)


def check_equidistant_times(mslist, stop=True, return_result=False, tolerance=0.2):
    """ Check if times in mslist are equidistant"""

    for ms in mslist:
        with table(ms, ack=False) as t:
            times = np.unique(t.getcol('TIME'))
        if len(times) == 1:  # single timestep data
            return
        diff_times = np.abs(np.diff(times))[
                     :-1]  # take out the last one because this one can be special  due to rounding at the end....
        diff_times_medsub = np.abs(diff_times - np.median(diff_times))
        idx_deviating, = np.where(diff_times_medsub > tolerance * np.median(diff_times))  # 20% tolerance
        # print(len(idx_deviating))
        # print(idx_deviating)
        # sys.exit()
        if len(idx_deviating) > 0:
            print(diff_times)
            print('These time slots numbers are deviating', idx_deviating)
            print(diff_times[idx_deviating])
            print(ms,
                  'Time axis is not equidistant, this might cause DP3 errors and segmentation faults (check how your data was averaged')
            # raise Exception(ms +': Time axis is not equidistant')
            print(
                'Avoid averaging your data with CASA or CARACal, instead average with DP3, this usually solves the issue')

            # comment line out below if you are willing to take the risk
            if stop:
                print('If you want to take the risk comment out the sys.exit() in the Python code')
                sys.exit()
    if return_result:
        if len(idx_deviating) > 0:
            return False  # so irregular time axis
        else:
            return True  # so normal time axis
    return


def check_equidistant_freqs(mslist):
    """
    Check if the frequency channels in each Measurement Set (MS) in the provided list are equidistant.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories.

    Raises
    ------
    Exception
        If any MS in the list has frequency channels that are not equidistant within a tolerance of 1e-5 Hz.

    Notes
    -----
    - For single channel data, the function returns without performing any checks.
    - The function prints the unique frequency differences and the MS path if non-equidistant channels are detected.
    """
    for ms in mslist:
        with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
            chan_freqs = t.getcol('CHAN_FREQ')[0]
        if len(chan_freqs) == 1:  # single channel data
            return
        diff_freqs_unique = np.unique(np.diff(chan_freqs))
        if len(diff_freqs_unique) != 1:
            for dfreq in diff_freqs_unique[1:]:
                if np.abs(dfreq-diff_freqs_unique[0]) > 1e-5: # max abs diff tolerance is 1e-5 Hz 
                    print(np.abs(dfreq-diff_freqs_unique[0]))
                    print(diff_freqs_unique)
                    print(ms, 'Frequency channels are not equidistant, made a mistake in DP3 concat?')
                    raise Exception(ms + ': Freqeuency channels are no equidistant, made a mistake in DP3 concat?')
    return


def run(command, log=False):
    """ Execute a shell command through subprocess

    Args:
        command (str): the command to execute.
    Returns:
        retval (int): the return code of the executed process.
    """
    if log:
        print(command)
        logger.info(command)
    process = subprocess.run(command, shell=True,
                            stderr=subprocess.STDOUT, encoding="utf-8")
    retval = process.returncode
    #stdout = process.stdout
    stderr = process.stderr
    if retval!= 0:
        print("FAILED to run", command)
        print("return value is", retval)
        print("stderr is", stderr)
        #print("stdout is", stdout)
        raise Exception(command)
    return retval


def fix_bad_weightspectrum(mslist, clipvalue):
    """ Sets bad values in WEIGHT_SPECTRUM that affect imaging and subsequent self-calibration to 0.0.

    Args:
        mslist (list): a list of Measurement Sets to iterate over and fix outlier values of.
        clipvalue (float): value above which WEIGHT_SPECTRUM will be set to 0.
    Returns:
        None
    """
    for ms in mslist:
        print('Clipping WEIGHT_SPECTRUM manually', ms, clipvalue)
        with table(ms, readonly=False) as t:
            ws = t.getcol('WEIGHT_SPECTRUM')
            idx = np.where(ws > clipvalue)
            ws[idx] = 0.0
            t.putcol('WEIGHT_SPECTRUM', ws)
    return


def format_solint(solint, ms, return_ntimes=False):
    """ Format the solution interval for DP3 calls.

    Args:
        solint (int or str): input solution interval.
        ms (str): measurement set to extract the integration time from.
    Returns:
        solintout (str): processed solution interval.
    """
    if str(solint).isdigit():
        if return_ntimes:
            with table(ms, readonly=True, ack=False) as t:
                time = np.unique(t.getcol('TIME'))
            return str(solint), len(time)
        else:
            return str(solint)
    else:
        with table(ms, readonly=True, ack=False) as t:
            time = np.unique(t.getcol('TIME'))
            tint = np.abs(time[1] - time[0])
        if 's' in solint:
            solintout = int(np.rint(float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'm' in solint:
            solintout = int(np.rint(60. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'h' in solint:
            solintout = int(np.rint(3600. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if solintout < 1:
            solintout = 1
        if return_ntimes:
            return str(solintout), len(time)
        else:
            return str(solintout)


def format_nchan(nchan, ms):
    """ Format the solution interval for DP3 calls.

    Args:
        nchan (int or str): input solution interval along the frequency axis.
        ms (str): measurement set to extract the frequnecy resolution from.
    Returns:
        solintout (str): processed frequency solution interval.
    """
    if str(nchan).isdigit():
        return str(nchan)
    else:
        with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
            chanw = np.median(t.getcol('CHAN_WIDTH'))
        if 'Hz' in nchan:
            nchanout = int(np.rint(float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if 'kHz' in nchan:
            nchanout = int(np.rint(1e3 * float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if 'MHz' in nchan:
            nchanout = int(np.rint(1e6 * float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if nchanout < 1:
            nchanout = 1
        return str(nchanout)


def FFTdelayfinder(h5, refant):
    from scipy.fftpack import fft, fftfreq
    H = tables.open_file(h5)
    upsample_factor = 10

    # reference to refant
    refant_idx = np.where(H.root.sol000.phase000.ant[:] == refant)
    phase = H.root.sol000.phase000.val[:]
    phasen = phase - phase[:, :, refant_idx[0], :]

    phasecomplex = np.exp(phasen * 1j)
    freq = H.root.sol000.phase000.freq[:]
    timeaxis = H.root.sol000.phase000.time[:]
    timeaxis = timeaxis - np.min(timeaxis)

    delayaxis = fftfreq(upsample_factor * freq.size,
                        d=np.abs(freq[1] - freq[0]) / float(upsample_factor))

    for ant_id, ant in enumerate(H.root.sol000.phase000.ant[:]):
        delay = 0.0 * H.root.sol000.phase000.time[:]
        print('FFT delay finding for:', ant)
        for time_id, time in enumerate(H.root.sol000.phase000.time[:]):
            delay[time_id] = delayaxis[
                np.argmax(np.abs(fft(phasecomplex[time_id, :, ant_id, 0], n=upsample_factor * len(freq))))]
        plt.plot(timeaxis / 3600., delay * 1e9)
    plt.ylim(-2e-6 * 1e9, 2e-6 * 1e9)
    plt.ylabel('Delay [ns]')
    plt.xlabel('Time [hr]')
    # plt.title(ant)
    plt.show()
    H.close()
    return


def compute_distance_to_pointingcenter(msname, HBAorLBA='HBA', warn=False, returnval=False, dologging=True):
    """ Compute distance to the pointing center. This is mainly useful for international baseline observation to check of the delay calibrator is not too far away.

    Args:
        msname (str): path to the measurement set to check.
        HBAorLBA (str): whether the data is HBA or LBA data. Can be 'HBA' or 'LBA'.
    Returns:
        None
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
    print('Distance to pointing center', seperation)
    if dologging:
        logger.info('Distance to pointing center:' + str(seperation))
    if (seperation.value > warn_distance) and warn:
        print(
            'Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
        logger.warning(
            'Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
    if returnval:
        return seperation.value
    return


def remove_flagged_data_startend(mslist):
    """ Trim flagged data at the start and end of the observation.

    Args:
        mslist (list): list of measurement sets to iterate over.
    Returns:
        mslistout (list): list of measurement sets with flagged data trimmed.
    """

    taql = 'taql'
    mslistout = []

    for ms in mslist:
        with table(ms, readonly=True, ack=False) as t:
            alltimes = t.getcol('TIME')
            alltimes = np.unique(alltimes)

            newt = taql('select TIME from $t where FLAG[0,0]=False')
            time = newt.getcol('TIME')
            time = np.unique(time)

            print('There are', len(alltimes), 'times')
            print('There are', len(time), 'unique unflagged times')

            print('First unflagged time', np.min(time))
            print('Last unflagged time', np.max(time))

            goodstartid = np.where(alltimes == np.min(time))[0][0]
            goodendid = np.where(alltimes == np.max(time))[0][0] + 1

            print(goodstartid, goodendid)

        if (goodstartid != 0) or (goodendid != len(alltimes)):
            msout = ms + '.cut'
            if os.path.isdir(msout):
                os.system('rm -rf ' + msout)
                time.sleep(2)  # wait a bit to make sure the directory is removed

            cmd = taql + " ' select from " + ms + " where TIME in (select distinct TIME from " + ms
            cmd += " offset " + str(goodstartid)
            cmd += " limit " + str((goodendid - goodstartid)) + ") giving "
            cmd += msout + " as plain'"
            print(cmd)
            run(cmd)
            mslistout.append(msout)
        else:
            mslistout.append(ms)
    return mslistout


def force_close(h5):
    """ Close indivdual HDF5 file by force.

    Args:
        h5 (str): name of the h5parm to close.
    Returns:
        None
    """
    h5s = list(tables.file._open_files._handlers)
    for h in h5s:
        if h.filename == h5:
            logger.warning('force_close: Closed --> ' + h5 + '\n')
            print('Forced (!) closing', h5)
            h.close()
            return
    # sys.stderr.write(h5 + ' not found\n')
    return


def create_mergeparmdbname(mslist, selfcalcycle, autofrequencyaverage_calspeedup=False, skymodelsolve=False):
    """ Merges the h5parms for a given list of measurement sets and selfcal cycle.

    Args:
        mslist (list): list of measurement sets to iterate over.
        selfcalcycle (int): the selfcal cycle for which to merge h5parms.
        autofrequencyaverage_calspeedup (bool): add extra "avg" to h5parm name
        skymodelsolve (bool): add extra "sky" to the name (for solves against a skymodel)
    Returns:
        parmdblist (list): list of names of the merged h5parms.
    """
    
    if autofrequencyaverage_calspeedup: 
        tmpstr = '.avg.h5'
    else:
        tmpstr = '.h5'
    
    parmdblist = mslist[:]
    for ms_id, ms in enumerate(mslist):
        if skymodelsolve:
            parmdblist[ms_id] = 'merged_skyselfcalcycle' + str(selfcalcycle).zfill(3) + '_' + ms + tmpstr
        else:    
            parmdblist[ms_id] = 'merged_selfcalcycle' + str(selfcalcycle).zfill(3) + '_' + ms + tmpstr
    print('Created parmdblist', parmdblist)
    return parmdblist


def preapply(H5filelist, mslist, updateDATA=True, dysco=True):
    """ Pre-apply a given set of corrections to a list of measurement sets.

    Args:
        H5filelist (list): list of h5parms to apply.
        mslist (list): list of measurement set to apply corrections to.
        updateDATA (bool): overwrite DATA with CORRECTED_DATA after solutions have been applied.
        dysco (bool): dysco compress the CORRECTED_DATA column or not.
    Returns:
        None
    """
    for ms in mslist:
        parmdb = time_match_mstoH5(H5filelist, ms)
        applycal(ms, parmdb, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=dysco)
        if updateDATA:
            run("taql 'update " + ms + " set DATA=CORRECTED_DATA'")
    return


def preapply_bandpass(H5filelist, mslist, dysco=True, updateweights=True):
    """ Pre-apply a given set of corrections to a list of measurement sets.

    Args:
        H5filelist (list): list of h5parms to apply.
        mslist (list): list of measurement set to apply corrections to.
        dysco (bool): dysco compress the CORRECTED_DATA column or not.
        updateweights (bool): updateweights based on amplitudes in DP3
    Returns:
        None
    """
    for ms in mslist:
        parmdb = find_closest_H5time_toms(H5filelist, ms)
        # overwrite DATA here (!)
        applycal(ms, parmdb, msincol='DATA', msoutcol='DATA', dysco=dysco, updateweights=updateweights, missingantennabehavior='flag')
    return

def find_closest_H5time_toms(H5filelist, ms):
    """ Find the h5parms, from a given list, that falls closest to the time midpoint of the specified Measurement Set.

    Args:
        H5filelist (list): list of h5parms to apply.
        ms (str): Measurement Set to match h5parms to.
    Returns:
        H5filematch (str): h5parm that is closest in time to the measurement set.
    """
    with table(ms, ack=False) as t:
        timesms = np.sort(np.unique(t.getcol('TIME')))
        obs_length = np.max(timesms) - np.min(timesms)
        time_midpoint  = [(0.5*obs_length) + np.min(timesms)] # make list because closest_arrayvals needs to iterate over it
    H5filematch = None
    time_diff = float('inf')
    
    for H5file in H5filelist:
        with tables.open_file(H5file, mode='r') as H:
            times = None
            for sol_type in ['amplitude000', 'rotation000', 'phase000', 'tec000', 'rotationmeasure000']:
                try:
                    times = np.sort(getattr(H.root.sol000, sol_type).time[:])
                    break
                except AttributeError:
                    continue
                
            if times is not None:
                time_diff_tmp = abs(np.diff(closest_arrayvals(times,time_midpoint))[0])
                if time_diff_tmp < time_diff:
                    H5filematch = H5file
                    time_diff = time_diff_tmp

    if H5filematch is None or times is None:
        print('find_closest_H5time_toms: Cannot find matching H5file and ms')
        raise Exception('find_closest_H5time_toms: Cannot find matching H5file and ms')
    print(H5filematch, 'is closest in time to', ms,  ' diff to the midpoint of the ms is', time_diff, ' [s]')
    return H5filematch

def closest_arrayvals(a, b):
    result = None
    min_diff = float('inf')
    for x in a:
        for y in b:
            diff = abs(x - y)
            if diff < min_diff:
                min_diff = diff
                result = [x, y]
    return result

def time_match_mstoH5(H5filelist, ms):
    """ Find the h5parms, from a given list, that overlap in time with the specified Measurement Set.

    Args:
        H5filelist (list): list of h5parms to apply.
        ms (str): Measurement Set to match h5parms to.
    Returns:
        H5filematch (list): list of h5parms matching the measurement set.
    """
    with table(ms) as t:
        timesms = np.unique(t.getcol('TIME'))
    H5filematch = None

    for H5file in H5filelist:
        with tables.open_file(H5file, mode='r') as H:
            times = None
            for sol_type in ['amplitude000', 'rotation000', 'phase000', 'tec000', 'rotationmeasure000']:
                try:
                    times = getattr(H.root.sol000, sol_type).time[:]
                    break
                except AttributeError:
                    continue

            if times is not None and np.median(times) >= np.min(timesms) and np.median(times) <= np.max(timesms):
                print(H5file, 'overlaps in time with', ms)
                H5filematch = H5file

    if H5filematch is None:
        print('Cannot find matching H5file and ms')
        raise Exception('Cannot find matching H5file and ms')

    return H5filematch


def logbasicinfo(args, fitsmask, mslist, version, inputsysargs):
    """ Prints basic information to the screen.

    Args:
        args (iterable): list of input arguments.
        fitsmask (str): name of the user-provided FITS mask.
        mslist (list): list of input measurement sets.
    """
    logger.info(' '.join(map(str, inputsysargs)))

    logger.info('Version:                   ' + str(version))
    logger.info('Imsize:                    ' + str(args['imsize']))
    logger.info('Pixelscale [arcsec]:       ' + str(args['pixelscale']))
    logger.info('Niter:                     ' + str(args['niter']))
    logger.info('Uvmin:                     ' + str(args['uvmin']))
    logger.info('Multiscale:                ' + str(args['multiscale']))
    logger.info('Beam correction:           ' + str(args['beamcor']))
    logger.info('IDG:                       ' + str(args['idg']))
    logger.info('Widefield:                 ' + str(args['forwidefield']))
    logger.info('Flagslowamprms:            ' + str(args['flagslowamprms']))
    logger.info('flagslowphaserms:          ' + str(args['flagslowphaserms']))
    logger.info('Do linear:                 ' + str(args['dolinear']))
    logger.info('Do circular:               ' + str(args['docircular']))
    if args['boxfile'] is not None:
        logger.info('Bobxfile:                  ' + args['boxfile'])
    logger.info('Mslist:                    ' + ' '.join(map(str, mslist)))
    logger.info('User specified clean mask: ' + str(fitsmask))
    logger.info('Threshold for MakeMask:    ' + str(args['maskthreshold']))
    logger.info('Briggs robust:             ' + str(args['robust']))
        
    for ms in mslist:
        logger.info(' === ' + ms + ' ===')
        with table(mslist[0] + '/OBSERVATION', ack=False) as t:
            telescope = t.getcol('TELESCOPE_NAME')[0]
            logger.info('Telescope:                 ' + telescope)
        with table(ms, readonly=True, ack=False) as t:            
            time = np.unique(t.getcol('TIME'))
            logger.info('Integration time [s]:      {:.2f}'.format(np.abs(time[1] - time[0])))
            logger.info('Observation duration [hr]: {:.2f}'.format((np.max(time)-np.min(time))/3600.))
        with table(ms + '/SPECTRAL_WINDOW', readonly=True, ack=False) as t:
            chanw = np.median(t.getcol('CHAN_WIDTH'))
            freqs = t.getcol('CHAN_FREQ')[0]
            nfreq = len(t.getcol('CHAN_FREQ')[0])
            logger.info('Number of channels:        {:.2f}'.format(nfreq))
            logger.info('Bandwidth [MHz]:           {:.2f}'.format((np.max(freqs)-np.min(freqs))/1e6))
            logger.info('Start frequnecy [MHz]:     {:.2f}'.format(np.min(freqs)/1e6))      
            logger.info('End frequency [MHz]:       {:.2f}'.format(np.max(freqs)/1e6))
        logger.info('================')
    return


def max_area_of_island(grid):
    """ Calculate the area of an island.

    Args:
        grid (ndarray): input image.
    Returns:
        None
    """
    rlen, clen = len(grid), len(grid[0])

    def neighbors(r, c):
        """ Generate the neighbor coordinates of the given row and column that are within the bounds of the grid.

        Args:
            r (int): row coordinate.
            c (int): column coordinate.
        """
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (0 <= r + dr < rlen) and (0 <= c + dc < clen):
                yield r + dr, c + dc

    visited = [[False] * clen for _ in range(rlen)]

    def island_size(r, c):
        """ Find the area of the land connected to the given coordinate.

        Return 0 if the coordinate is water or if it has already been explored in a previous call to island_size().

        Args:
            r (int): row coordinate.
            c (int): column coordinate.
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
    """ Find the largest island in a given FITS mask.

    Args:
        fitsmask (str): path to the FITS file.
    Returns:
        max_area (float): area of the largest island.
    """
    with fits.open(fitsmask) as hdulist:
        data = hdulist[0].data
        max_area = max_area_of_island(data[0, 0, :, :])
    return max_area


def create_phase_slope(inmslist, incol='DATA', outcol='DATA_PHASE_SLOPE',
                       ampnorm=False, dysco=False, testscfactor=1., crosshandtozero=True):
    """ Creates a new column to solve for a phase slope from.

    Args:
        inmslist (list): list of input measurement sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        ampnorm (bool): If True, only takes phases from the input visibilities and sets their amplitude to 1.
        dysco (bool): dysco compress the output column.
    Returns:
        None
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)
            data = t.getcol(incol)
            dataslope = np.copy(data)
            for ff in range(data.shape[1] - 1):
                if ampnorm:
                    dataslope[:, ff, 0] = np.copy(
                        np.exp(1j * testscfactor * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                    dataslope[:, ff, 3] = np.copy(
                        np.exp(1j * testscfactor * (np.angle(data[:, ff, 3]) - np.angle(data[:, ff + 1, 3]))))
                    if crosshandtozero:
                        dataslope[:, ff, 1] = 0. * np.exp(1j * 0)
                        dataslope[:, ff, 2] = 0. * np.exp(1j * 0)
                else:
                    dataslope[:, ff, 0] = np.copy(np.abs(data[:, ff, 0]) * np.exp(
                        1j * testscfactor * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                    dataslope[:, ff, 3] = np.copy(np.abs(data[:, ff, 3]) * np.exp(
                        1j * testscfactor * (np.angle(data[:, ff, 3]) - np.angle(data[:, ff + 1, 3]))))
                    if crosshandtozero:
                        dataslope[:, ff, 1] = 0. * np.exp(1j * 0)
                        dataslope[:, ff, 2] = 0. * np.exp(1j * 0)

            # last freq set to second to last freq because difference reduces length of freq axis with one
            dataslope[:, -1, :] = np.copy(dataslope[:, -2, :])
            t.putcol(outcol, dataslope)

        del data, dataslope
    return


def stackwrapper(inmslist: list, msout_prefix: str = 'stack', column_to_normalise: str = 'DATA') -> None:
    """ Wraps the stack
    Arguments
    ---------
    inmslist : list
        List of input MSes to stack
    """
    if type(inmslist) is not list:
        raise TypeError('Incorrect input type for inmslist')
    print('Adding weight spectrum to stack')
    create_weight_spectrum(inmslist, 'WEIGHT_SPECTRUM_PM', updateweights=True,
                           updateweights_from_thiscolumn='MODEL_DATA')
    print('Attempting to normalise data to point source')
    normalize_data_bymodel(inmslist, outcol='DATA_NORM', incol=column_to_normalise,
                           modelcol='MODEL_DATA')
    print('Stacking datasets')

    start = time.time()
    msout_stacks, mss_timestacks = stackMS_taql(inmslist, outputms_prefix=msout_prefix, incol='DATA_NORM', outcol='DATA', weightref='WEIGHT_SPECTRUM_PM')
    now = time.time()
    print(f'Stacking took {now - start} seconds')

    assert len(msout_stacks) == len(mss_timestacks)

    return msout_stacks, mss_timestacks

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

    print(hduflat.data.shape, hdulist[0].data.shape)
    datashape = hduflat.data.shape

    # create median filtered image where emission on small scales is removed
    if rebin is not None:
        # needed for large images as scipy.signal.medfilt2d takes too long
        image_data = rescale(hduflat.data, 1./rebin, anti_aliasing=True) # downsample
        image_data = scipy.signal.medfilt2d(image_data, kernel_size=kernel_size)
        image_data = scipy.ndimage.zoom(image_data, float(hduflat.data.shape[0]/image_data.shape[0])) # upsample
        
        imagenoise = findrms(np.ndarray.flatten(image_data))
        print(datashape, image_data.shape)
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

def create_weight_spectrum_modelratio(inmslist, outweightcol, updateweights=False,
                                      originalmodel='MODEL_DATA', newmodel='MODEL_DATA_PHASE_SLOPE', backup=True):
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    stepsize = 1000000
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            weightref = 'WEIGHT_SPECTRUM'
            if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
                weightref = 'WEIGHT_SPECTRUM_SOLVE'  # for LoTSS-DR2 datasets
            if backup and ('WEIGHT_SPECTRUM_BACKUP' not in t.colnames()):
                #desc = t.getcoldesc(weightref)
                #desc['name'] = 'WEIGHT_SPECTRUM_BACKUP'
                #t.addcols(desc)
                addcol(t, weightref, 'WEIGHT_SPECTRUM_BACKUP', write_outcol=True) 
            if outweightcol not in t.colnames():
                print('Adding', outweightcol, 'to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = outweightcol
                #t.addcols(desc)
                addcol(t, weightref, outweightcol)
            # os.system('DP3 msin={ms} msin.datacolumn={weightref} msout=. msout.datacolumn={outweightcol} steps=[]')
            if t.nrows() < stepsize: stepsize = t.nrows()  # if less than stepsize rows, do not loop
            for row in range(0, t.nrows(), stepsize):
                print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                weight = t.getcol(weightref, startrow=row, nrow=stepsize, rowincr=1).astype(np.float64)
                if updateweights and originalmodel in t.colnames() and newmodel in t.colnames():
                    model_orig = t.getcol(originalmodel, startrow=row, nrow=stepsize, rowincr=1).astype(np.complex256)
                    model_new = t.getcol(newmodel, startrow=row, nrow=stepsize, rowincr=1).astype(np.complex256)

                    model_orig[:, :, 1] = model_orig[:, :, 0]  # make everything XX/RR
                    model_orig[:, :, 2] = model_orig[:, :, 0]  # make everything XX/RR
                    model_orig[:, :, 3] = model_orig[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 1] = model_new[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 2] = model_new[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 3] = model_new[:, :, 0]  # make everything XX/RR
                else:
                    model_orig = 1.
                    model_new = 1.
                print('Mean weights input', np.nanmean(weight))
                print('Mean weights change factor', np.nanmean((np.abs(model_orig)) ** 2))
                t.putcol(outweightcol, (weight * (np.abs(model_orig / model_new)) ** 2).astype(np.float64), startrow=row,
                         nrow=stepsize, rowincr=1)
                # print(weight.shape, model_orig.shape)
        print()
        del weight, model_orig, model_new


def addcol(t, incol, outcol, write_outcol=False, dysco=False):
    """ Add a new column to a MS. """
    if outcol not in t.colnames():
        logging.info('Adding column: ' + outcol)
        coldmi = t.getdminfo(incol)
        if dysco:
            coldmi['NAME'] = 'Dysco' + outcol
        else:
            coldmi['NAME'] = outcol
        try:
            t.addcols(makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
        except:
            coldmi['TYPE'] = "StandardStMan"  # DyscoStMan"
            t.addcols(makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
    if (outcol != incol) and write_outcol:
        # copy over the columns
        taql("UPDATE $t SET " + outcol + "=" + incol)

def create_weight_spectrum(inmslist, outweightcol, updateweights=False, updateweights_from_thiscolumn='MODEL_DATA',
                           backup=True):
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    stepsize = 1000000
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            weightref = 'WEIGHT_SPECTRUM'
            if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
                weightref = 'WEIGHT_SPECTRUM_SOLVE'  # for LoTSS-DR2 datasets
            if backup and ('WEIGHT_SPECTRUM_BACKUP' not in t.colnames()):
                print('Adding WEIGHT_SPECTRUM_BACKUP to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = 'WEIGHT_SPECTRUM_BACKUP'
                #t.addcols(desc)
                addcol(t, weightref, 'WEIGHT_SPECTRUM_BACKUP', write_outcol=True)

            if outweightcol not in t.colnames():
                print('Adding', outweightcol, 'to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = outweightcol
                #t.addcols(desc)
                addcol(t, weightref, outweightcol)

            # os.system('DP3 msin={ms} msin.datacolumn={weightref} msout=. msout.datacolumn={outweightcol} steps=[]')
            if t.nrows() < stepsize: stepsize = t.nrows() # if less than stepsize rows, do not loop
            for row in range(0, t.nrows(), stepsize):
                print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                weight = t.getcol(weightref, startrow=row, nrow=stepsize, rowincr=1).astype(np.float64)
                if updateweights and updateweights_from_thiscolumn in t.colnames():
                    model = t.getcol(updateweights_from_thiscolumn, startrow=row, nrow=stepsize, rowincr=1).astype(
                        np.complex256)
                    model[:, :, 1] = model[:, :, 0]  # make everything XX/RR
                    model[:, :, 2] = model[:, :, 0]  # make everything XX/RR
                    model[:, :, 3] = model[:, :, 0]  # make everything XX/RR
                else:
                    model = 1.
                print('Mean weights input', np.nanmean(weight))
                print('Mean weights change factor', np.nanmean((np.abs(model)) ** 2))
                t.putcol(outweightcol, (weight * (np.abs(model)) ** 2).astype(np.float64), startrow=row, nrow=stepsize,
                         rowincr=1)
                # print(weight.shape, model.shape)
        print()
        del weight, model


def create_weight_spectrum_taql(inmslist, outweightcol, updateweights=False,
                                updateweights_from_thiscolumn='MODEL_DATA'):
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        t = table(ms, readonly=False, ack=True)
        weightref = 'WEIGHT_SPECTRUM'
        if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
            weightref = 'WEIGHT_SPECTRUM_SOLVE'  # for LoTSS-DR2 datasets
        if outweightcol not in t.colnames():
            print('Adding', outweightcol, 'to', ms, 'based on', weightref)
            #desc = t.getcoldesc(weightref)
            #desc['name'] = outweightcol
            #t.addcols(desc)
            addcol(t, weightref, outweightcol)
        taql(f'UPDATE {ms} SET {updateweights_from_thiscolumn}[,1] = {updateweights_from_thiscolumn}[,0]')
        taql(f'UPDATE {ms} SET {updateweights_from_thiscolumn}[,2] = {updateweights_from_thiscolumn}[,0]')
        taql(f'UPDATE {ms} SET {updateweights_from_thiscolumn}[,3] = {updateweights_from_thiscolumn}[,0]')
        taql(f'UPDATE {ms} SET {outweightcol} = {weightref} * abs({updateweights_from_thiscolumn})**2')
        weightmean = taql('SELECT gmean(WEIGHT_SPECTRUM_PM) AS MEAN FROM ms1_nodysco_pointsource.ms').getcol('MEAN')
        change_factor = taql('SELECT gmean(abs(MODEL_DATA)**2) AS MEAN FROM ms1_nodysco_pointsource.ms').getcol(
            'MEAN')
        print('Mean weights input', weightmean)
        print('Mean weights change factor', change_factor)
        print()
        t.close()

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


def create_calibration_error_catalog(filename, outfile, thresh_pix=7.5,thresh_isl=7.5):
    """
    Creates a calibration error catalog from a given image file using the PyBDSF library.

    Parameters:
        filename (str): Path to the input image file to be processed.
        outfile (str): Path to the output file where the catalog will be saved in FITS format.
        thresh_pix (float, optional): Pixel threshold for source detection. Default is 7.5.
        thresh_isl (float, optional): Island threshold for source detection. Default is 7.5.

    Returns:
        None
    """
    img = bdsf.process_image(filename,mean_map='const', rms_map=True, \
                             rms_box=(35,5), thresh_pix=thresh_pix,thresh_isl=thresh_isl)
    img.write_catalog(format='fits', outfile=outfile, catalog_type='srl', clobber=True)
    del img
    return

def update_calibration_error_catalog(catalogfile, outcatalogfile, distance=20., keep_N_brightest=20, previous_catalog=None, N_dir_max=45):
    """
    Updates a calibration error catalog by filtering, merging, and removing nearby sources.
    Parameters:
    -----------
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
    Returns:
    --------
    None
        The updated catalog is written to the specified output file.
    Notes:
    ------
    - The function sorts the catalog by peak flux and retains the brightest sources.
    - If a previous catalog is provided, it merges the two catalogs, ensuring that entries from the previous catalog are prioritized.
    - Sources that are too close to each other (based on the `distance` parameter) are removed to avoid duplicates or closely separated directions.
    - The final catalog is limited to `N_dir_max` entries if it exceeds this limit.
    - The output catalog is saved in FITS format, overwriting any existing file at the specified path.
    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    print(catalog.columns)
    hdu_list.close()
    
    # sort catalog at Peak flux
    idx = catalog.argsort(keys='Peak_flux', reverse=True)
    catalog = catalog[idx]
    
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
        print('Trying to find sources too close to SOURCE ID', source_id)
        for faintersource_id, faintersource in enumerate(catalog[source_id+1:]): # take only sources with a higher index, skip last one
            c2 = SkyCoord(faintersource['RA']*units.degree, faintersource['DEC']*units.degree, frame='icrs')
            if c1.separation(c2).to(units.arcmin).value < distance:
                print('Found close source with ID', faintersource['Source_id'])
                removeidx = np.where((new_catalog['Peak_flux'] == faintersource['Peak_flux']) & (new_catalog['Source_id'] == faintersource['Source_id']))[0] # do a double comparson because the vstack from catalog_prev can merger sources with the same source_id
                assert len(removeidx) <= 1
                if len(removeidx) > 0:
                    new_catalog.remove_row(removeidx[0])
                 
    print('Catalog entries before / after removal of closely separated directions', len(catalog), len(new_catalog))
    
    # ensure catalog does not go over N_dir_max limit
    if len(new_catalog) > N_dir_max:
        new_catalog = new_catalog[0:N_dir_max]
        print('Kept only N_dir_max directions:', N_dir_max)
    new_catalog.write(outcatalogfile, format='fits', overwrite=True)


def write_facet_directions(catalogfile, facetdirections = 'directions.txt', ds9_region='directions.reg'):
    """
    Writes facet directions to a text file and generates a DS9 region file.
    This function processes a FITS catalog file to extract source information 
    (RA and DEC) and writes it to a specified text file in a format suitable 
    for self-calibration. It also generates a DS9 region file for visualization.
    Parameters:
    -----------
    catalogfile : str
        Path to the FITS catalog file containing source information.
    facetdirections : str, optional
        Name of the output text file containing facet directions. 
        Default is 'directions.txt'.
    ds9_region : str, optional
        Name of the output DS9 region file. Default is 'directions.reg'.
    Outputs:
    --------
    - A text file (`facetdirections`) containing RA, DEC, self-calibration 
        cycle, solution intervals, smoothness values, inclusion flags, and 
        direction labels for each source.
    - A DS9 region file (`ds9_region`) for visualizing the source positions.
    Notes:
    ------
    - The function uses hardcoded values for self-calibration cycles, solution 
        intervals, smoothness values, and inclusion flags.
    - The first `N_bright` sources are treated with different solution intervals 
        and smoothness values compared to the rest.
    - Beyond `N_normal` sources, a different set of inclusion flags is applied.
    - The function assumes the FITS catalog contains columns 'RA' and 'DEC'.
    Example:
    --------
    write_facet_directions('source_catalog.fits', 'output_directions.txt', 'output_regions.reg')
    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    #print(catalog.columns)
    #print(catalog['Peak_flux'])
    hdu_list.close()
    selfcalcycle = 0 # hard coded at zero is ok
    solints_bright = ["'32sec'", "'2min'", "'10min'","'192sec'"]
    solints = ["'32sec'", "'2min'", "'20min'","'192sec'"]
    smoothing_bright = [10,25,5,25]
    smoothing = [10,25,15,25]
    N_bright = 4 # first N_bright get less smoothness and solint
    N_normal = 45 # beyond this we have pertubative directions
    N_amps   = 35 # directions for scalarcomplexgain solve
    
    inclusion_flags = [True,True,True,True]
    inclusion_flags_no_amps = [True,True,False,True]
    inclusion_flags_pert = [False,False,False,True]
    
    write_ds9_regions(catalog['RA'], catalog['DEC'], filename=ds9_region) 
  
    with open(facetdirections,"w") as f:
        f.write('#RA DEC start solints smoothness soltypelist_includedir\n')
        for source_counter,source in enumerate(catalog):
            if source_counter < N_bright: 
                smoothnessvals = smoothing_bright
                solintvals    = solints_bright
            else:
                smoothnessvals = smoothing
                solintvals = solints
            if source_counter < N_amps:
                inclusion_flagsvals = inclusion_flags # include all solve types
            elif source_counter >= N_amps and source_counter < N_normal: 
                inclusion_flagsvals = inclusion_flags_no_amps # skip amp solving, but include first phases
            else:
                inclusion_flagsvals = inclusion_flags_pert # pertubative phases only
                
            line = f"{source['RA']} {source['DEC']} {selfcalcycle} " \
                   f"[{','.join(solintvals)}] " \
                   f"[{','.join(str(s) for s in smoothnessvals)}] " \
                   f"[{','.join(str(i) for i in inclusion_flagsvals)}] " \
                   f"# direction Dir{source_counter:02d}\n"
            f.write(line)


def auto_direction(selfcalcycle=0):
    """
    Automatically determines and processes calibration directions for self-calibration cycles.
    Parameters:
    -----------
    selfcalcycle : int, optional
        The self-calibration cycle number (default is 0). Determines the parameters for 
        artifact catalog creation and filtering.
    Returns:
    --------
    str
        The filename of the facet directions text file generated for the given self-calibration cycle.
    Description:
    ------------
    This function performs the following steps:
    1. Sets parameters (`keep_N_brightest`, `distance`, `N_dir_max`) based on the self-calibration cycle.
    2. Constructs filenames for input and output files, including error maps, catalogs, and plots.
    3. Generates an error map from the input image.
    4. Creates an artifact sources catalog from the error map.
    5. Updates the artifact catalog by filtering sources based on brightness, distance, and merging with 
       the previous catalog.
    6. Writes facet direction files for self-calibration and visualization.
    7. Plots the error map with overlaid artifact directions.
    Notes:
    ------
    - The function uses global `args` to retrieve input parameters such as `imagename`, `idg`, and `channelsout`.
    - The error map and artifact catalog are processed using external helper functions:
      `calibration_error_map`, `create_calibration_error_catalog`, `update_calibration_error_catalog`, 
      `write_facet_directions`, and `plotimage_astropy`.
    - The plot uses the noise level from the first error map (`imagename000-errormap.fits`) for consistent scaling.
    Dependencies:
    -------------
    - Requires the `astropy.io.fits` module for handling FITS files.
    - Assumes the presence of helper functions for error map generation, catalog creation, and plotting.
    """
    
    if selfcalcycle == 0:
        keep_N_brightest = 15
        distance = 20
        N_dir_max = 15
    if selfcalcycle == 1:
        keep_N_brightest = 20 
        distance = 20
        N_dir_max = 25
    if selfcalcycle == 2:
        keep_N_brightest = 30
        distance = 15
        N_dir_max = 35
    if selfcalcycle == 3:
        keep_N_brightest = 40   
        distance = 15
        N_dir_max = 45
    if selfcalcycle == 4:
        keep_N_brightest = 40   
        distance = 15
        N_dir_max = 45
    if selfcalcycle == 5:
        keep_N_brightest = 50   
        distance = 10
        N_dir_max = 60
    if selfcalcycle == 6:
        keep_N_brightest = 60   
        distance = 10
        N_dir_max = 70
    if selfcalcycle == 7:
        keep_N_brightest = 70   
        distance = 10
        N_dir_max = 80
    if selfcalcycle == 8:
        keep_N_brightest = 80   
        distance = 10
        N_dir_max = 90
    if selfcalcycle == 9:
        keep_N_brightest = 90   
        distance = 10
        N_dir_max = 100
    if selfcalcycle >= 10:
        keep_N_brightest = 100   
        distance = 10
        N_dir_max = 100

    
    # set image name input to compute error map
    if args['idg']:
        fitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + '-MFS-image.fits'
    else:
        fitsimage = args['imagename'] + str(selfcalcycle).zfill(3) +  '-MFS-image.fits'    
    if args['channelsout'] == 1:
        fitsimage = fitsimage.replace('-MFS', '').replace('-I', '')    
   
    # set input/output names
    outputerrormap =  args['imagename'] + str(selfcalcycle).zfill(3) + '-errormap.fits'
    outputcatalog  =   args['imagename'] + str(selfcalcycle).zfill(3) + '-errormap.srl.fits'
    outputcatalog_filtered =  args['imagename'] + str(selfcalcycle).zfill(3) + '-errormap.srl.filtered.fits'
    facetdirections = 'directions_' + str(selfcalcycle).zfill(3) + '.txt'
    directions_reg =  'directions_' + str(selfcalcycle).zfill(3) + '.reg'
    outplotname =  args['imagename'] + str(selfcalcycle).zfill(3) + '-errormap.png'
    
    if selfcalcycle > 0:
        previous_catalog =  args['imagename'] + str(selfcalcycle-1).zfill(3) + '-errormap.srl.filtered.fits'
        if not os.path.isfile(previous_catalog) or not os.path.isfile(args['imagename'] + str(0).zfill(3) + '-errormap.fits'):
            print('One of both of these files are missing:',previous_catalog, args['imagename'] + str(0).zfill(3) + '-errormap.fits')    
            raise Exception('Missing files')
    else:
        previous_catalog = None  
    
    # make the error map
    print('Making artifact map from:', fitsimage)
    calibration_error_map(fitsimage,  outputerrormap, kernelsize=31, rebin=31)
    
    # create the artifact sources catalog 
    create_calibration_error_catalog(outputerrormap, outputcatalog, thresh_pix=7.5,thresh_isl=7.5)
    
    # update the artifact catalog (keep only N-brightest, remove closely seperated sources, merge with previous catalog)
    update_calibration_error_catalog(outputcatalog,outputcatalog_filtered, distance=distance, keep_N_brightest=keep_N_brightest, previous_catalog=previous_catalog, N_dir_max=N_dir_max)
    
    # write facet direction file (for facetselfcal), also output directions.reg for visualization
    write_facet_directions(outputcatalog_filtered, facetdirections=facetdirections, ds9_region=directions_reg)

    # plot, # hardcode minmax to always usage image000 so it is easier to compare
    hdulist = fits.open(args['imagename'] + str(0).zfill(3) + '-errormap.fits') 
    imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
    plotminmax = [-2.*imagenoise, 35.*imagenoise]
    hdulist.close()
    plotimage_astropy(outputerrormap, outplotname, mask=None, regionfile=directions_reg, regioncolor='red', minmax=plotminmax, regionalpha=1.0)
    return facetdirections



def normalize_data_bymodel(inmslist, outcol='DATA_NORM', incol='DATA', modelcol='MODEL_DATA', stepsize=1000000):
    """
    Normalize visibility data by model data.

    Args:
        inmslist (str or list of str): List of input Measurement Set(s).
        outcol (str, optional): Name of the output column for normalized data. Default is 'DATA_NORM'.
        incol (str, optional): Name of the input column containing original data. Default is 'DATA'.
        modelcol (str, optional): Name of the model column. Default is 'MODEL_DATA'.
        stepsize (int, optional): Step size for processing rows. Default is 1000000.

    Returns:
        None

   """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                print('Adding', outcol, 'to', ms, 'based on', incol)
                desc = t.getcoldesc(incol)
                desc['name'] = outcol
                t.addcols(desc)
            for row in range(0, t.nrows(), stepsize):
                data = t.getcol(incol, startrow=row, nrow=stepsize, rowincr=1)
                if modelcol in t.colnames():
                    model = t.getcol(modelcol, startrow=row, nrow=stepsize, rowincr=1)
                    print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                    # print(np.max(abs(model)))
                    # print(np.min(abs(model)))
                    np.divide(data, model, out=data, where=np.abs(model) > 0)
                    t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)
                else:
                    t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)


def stackMS(inmslist, outputms='stack.MS', incol='DATA_NORM', outcol='DATA', weightref='WEIGHT_SPECTRUM_PM',
            outcol_weight='WEIGHT_SPECTRUM', stepsize=1000000):
    """ Henrik Feb 2025: This function is not used currently and does not support ulti-timestack MS. Can be removed?
    Stack a list of MSes.

    Arguments
    ---------
    inmslist : list

    """
    print(f'Using input column {incol}')
    print(f'Writing to {outputms}')
    if not isinstance(inmslist, list):
        os.system('cp -r {} {}'.format(inmslist, outputms))
        print("WARNING: Stacking was performed on only one MS, so not really a meaningful stack")
        return True
    if os.path.isdir(outputms):  # delete MS if it exists
        os.system('rm -rf ' + outputms)
    os.system('cp -r {} {}'.format(inmslist[0], outputms))
    taql('UPDATE stack.MS SET DATA=DATA_NORM*WEIGHT_SPECTRUM_PM')
    with table(outputms, readonly=False, ack=True) as t_main:
        for ms in inmslist[1:]:
            with table(ms, readonly=True, ack=True) as t:
                for row in range(0, t.nrows(), stepsize):
                    print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                    weight_main = t_main.getcol(outcol_weight, startrow=row, nrow=stepsize, rowincr=1)
                    visibi_main = t_main.getcol(outcol, startrow=row, nrow=stepsize, rowincr=1)

                    weight = t.getcol(weightref, startrow=row, nrow=stepsize, rowincr=1)
                    visibi = t.getcol(incol, startrow=row, nrow=stepsize, rowincr=1)

                    stacked_vis = visibi_main + (visibi * weight)

                    average_wgt = weight + weight_main
                    print(f'Writing stacked data to outputcolumn {outcol}')
                    t_main.putcol(outcol, stacked_vis, startrow=row, nrow=stepsize, rowincr=1)
                    print(f'Writing stacked weights to outputcolumn {outcol_weight}')
                    t_main.putcol(outcol_weight, average_wgt, startrow=row, nrow=stepsize, rowincr=1)
    # This is probably wrong / not needed.
    taql('UPDATE stack.MS SET DATA=DATA/WEIGHT_SPECTRUM')


def stackMS_taql(inmslist: list, outputms_prefix: str = 'stack', incol: str = 'DATA_NORM', outcol: str = 'DATA',
                 weightref: str = 'WEIGHT_SPECTRUM_PM', outcol_weight: str = 'WEIGHT_SPECTRUM'):
    """ Stack a list of MSes - per group with same time axis, one stacked MS is created.

    Arguments
    ---------
    inmslist : list
        List of input Measurement Sets to stack.
    outputms_prefix : str
        Name of the output MS.
    incol : str
        Column to stack from the individual MSes.
    outcol : str
        Name of the stacked data column in the output MS.
    weightref : str
        Name of the weight column to stack from the individual files.
    outcol_weight : str
        Name of the stacked weight column in the output MS.
    Returns
    ---------
    msout_stacked: list of stacked MS names
    mss_timestacks: list of input MS grouped in timestacks
    """

    # identify which MSs share the same time axis:
    starttimelist = [] # list of unique timestamps
    mss_timestacks = [] # list of MSs stacks
    for ms in inmslist:
        with table(ms, ack=False) as t:
            starttime = t.TIME[0]
            try: # check if timestamps already exist and if yes, add to this stack
                group = starttimelist.index(starttime)
                print('group', group)
                print(f'append {ms} to {starttime}: {mss_timestacks[group]}')
                mss_timestacks[group].append(ms)
            except ValueError: # add new list of MS for this timestamps if there is none already
                starttimelist.append(starttime)
                mss_timestacks.append([ms])
            print(f'new list {ms} to {starttime}')
    print(starttimelist)
    print(f'Found {len(starttimelist)} groups of MSs with same time axis.')
    print(f'Groups: {mss_timestacks}.')

    msout_stacked = []
    for timestack_id, inmslist_timestack in enumerate(mss_timestacks):
        outputms = f'{outputms_prefix}_t{timestack_id:02d}.MS'
        msout_stacked.append(outputms)
        print(f'Using input column {incol}')
        print(f'Writing to {outputms}')
        if not isinstance(inmslist_timestack, list):
            os.system('cp -r {} {}'.format(inmslist_timestack, outputms))
            print("WARNING: Stacking was performed on only one MS, so not really a meaningful stack")
            return True
        if os.path.isdir(outputms):  # delete MS if it exists
            os.system('rm -rf ' + outputms)
        os.system('cp -r {} {}'.format(inmslist_timestack[0], outputms))

        TAQLSTR = f'UPDATE {outputms} SET DATA = ('
        sum_clause = ' + '.join(
            [f'ms{idx:02d}.DATA_NORM * ms{idx:02d}.WEIGHT_SPECTRUM_PM' for idx in range(1, len(inmslist_timestack) + 1)])
        sum_weight_clause = ' + '.join([f'ms{idx:02d}.WEIGHT_SPECTRUM_PM' for idx in range(1, len(inmslist_timestack) + 1)])
        from_clause = ', '.join([f'{ms} AS ms{idx:02d}' for idx, ms in enumerate(inmslist_timestack, start=1)])

        taql_query = f'{TAQLSTR} {sum_clause}) / ({sum_weight_clause}) FROM {from_clause}'

        print('Stacking DATA')
        print(taql_query)
        taql(taql_query)

        print('Stacking WEIGHT_SPECTRUM')
        print(f'UPDATE {outputms} SET {outcol_weight} = ({sum_weight_clause}) FROM {from_clause}')
        taql(f'UPDATE {outputms} SET {outcol_weight} = ({sum_weight_clause}) FROM {from_clause}')

    return msout_stacked, mss_timestacks


def create_phasediff_column(inmslist, incol='DATA', outcol='DATA_CIRCULAR_PHASEDIFF', dysco=True, stepsize=1000000):
    """ Creates a new column for the phase difference solve.

    Args:
        inmslist (list): list of input Measurement Sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        dysco (bool): dysco compress the output column.
        stepsize (int): step size for row looping in casacore tables getcol/putcol
    """

    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)

            for row in range(0, t.nrows(), stepsize):
                print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                data = t.getcol(incol, startrow=row, nrow=stepsize, rowincr=1)
                phasediff = np.copy(np.angle(data[:, :, 0]) - np.angle(data[:, :, 3]))  # RR - LL
                data[:, :, 0] = 0.5 * np.exp(
                    1j * phasediff)  # because I = RR+LL/2 (this is tricky because we work with phase diff)
                data[:, :, 3] = data[:, :, 0]
                t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)
                del data
                del phasediff

    if False:
        # data = t.getcol(incol)
        # t.putcol(outcol, data)
        # t.close()

        time.sleep(2)
        cmd = "taql 'update " + ms + " set "
        cmd += outcol + "[,0]=0.5*EXP(1.0i*(PHASE(" + incol + "[,0])-PHASE(" + incol + "[,3])))'"
        # cmd += outcol + "[,3]=" + outcol + "[,0],"
        # cmd += outcol + "[,1]=0+0i,"
        # cmd += outcol + "[,2]=0+0i'"
        print(cmd)
        run(cmd)
        cmd = "taql 'update " + ms + " set "
        # cmd += outcol + "[,0]=0.5*EXP(1.0i*(PHASE(" + incol + "[,0])-PHASE(" + incol + "[,3]))),"
        cmd += outcol + "[,3]=" + outcol + "[,0]'"
        # cmd += outcol + "[,1]=0+0i,"
        # cmd += outcol + "[,2]=0+0i'"
        print(cmd)
        run(cmd)
    return


def create_phase_column(inmslist, incol='DATA', outcol='DATA_PHASEONLY', dysco=True):
    """ Creates a new column containging visibilities with their original phase, but unity amplitude.

    Args:
        inmslist (list): list of input Measurement Sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        dysco (bool): dysco compress the output column.
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)
            data = t.getcol(incol)
            data[:, :, 0] = np.copy(np.exp(1j * np.angle(data[:, :, 0])))  # because I = xx+yy/2
            data[:, :, 3] = np.copy(np.exp(1j * np.angle(data[:, :, 3])))  # because I = xx+yy/2
            t.putcol(outcol, data)
        del data
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
    --------
    - Scans for all filenames matching '<basename>-????-model-pb.fits' and
      '<basename>-????-model-fpb.fits'.
    - If the number of pb and fpb images is equal, nothing is done.
    - If no fpb images exist, each pb image is copied to an fpb file by replacing
      the '-model-pb.fits' suffix with '-model-fpb.fits'.
    - If some but not all fpb images exist, an assertion error is raised.

    Notes
    -----
    This function uses `os.system` to perform file copying and prints the copy commands.
    """
    pblist = glob.glob(modelimagebasename + '-????-model-pb.fits')
    fpblist = glob.glob(modelimagebasename + '-????-model-fpb.fits')
    
    if len(pblist) == len(fpblist): return # nothing is needed
    assert len(fpblist) == 0 # if we are here we should not have any fpb images
    for image in pblist:
       print('cp ' + image + ' ' + image.replace('-model-pb.fits','-model-fpb.fits'))
       os.system('cp ' + image + ' ' + image.replace('-model-pb.fits','-model-fpb.fits'))


def create_MODEL_DATA_PDIFF(inmslist, modelstoragemanager=None):
    """ Creates the MODEL_DATA_PDIFF column.

    Args:
      inmslist (list): list of input Measurement Sets.
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        if modelstoragemanager is None:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF steps=[]')
        else:
             run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF msout.storagemanager=' + modelstoragemanager + ' steps=[]')
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,0]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,3]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,1]=(0+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,2]=(0+0i)'")


def fulljonesparmdb(h5):
    """ Checks if a given h5parm has a fulljones solution table as sol000.

    Args:
        h5 (str): path to the h5parm.
    Returns:
        fulljones (bool): whether the sol000 contains fulljones solutions.
    """
    H = tables.open_file(h5)
    try:
        pol_p = H.root.sol000.phase000.pol[:]
        pol_a = H.root.sol000.amplitude000.pol[:]
        if len(pol_p) == 4 and len(pol_a) == 4:
            fulljones = True
        else:
            fulljones = False
    except:
        fulljones = False
    H.close()
    return fulljones


def reset_gains_noncore(h5parm, keepanntennastr='CS'):
    """ Resets the gain of non-CS stations to unity amplitude and zero phase.

    Args:
        h5parm (str): path to the H5parm to reset gains of.
        keepantennastr (str): string containing antennas to keep.
    Returns:
      None
    """
    fulljones = fulljonesparmdb(h5parm)  # True/False
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure = check_soltabs(h5parm)

    with tables.open_file(h5parm) as H:

        if hasphase:
            phase = H.root.sol000.phase000.val[:]
            antennas = H.root.sol000.phase000.ant[:]
            axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
        if hasamps:
            amp = H.root.sol000.amplitude000.val[:]
            antennas = H.root.sol000.amplitude000.ant[:]
            axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
        if hastec:
            tec = H.root.sol000.tec000.val[:]
            antennas = H.root.sol000.tec000.ant[:]
            axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
        if hasrotation:
            rotation = H.root.sol000.rotation000.val[:]
            antennas = H.root.sol000.rotation000.ant[:]
            axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
        if hasrotationmeasure:
            faradayrotation = H.root.sol000.rotationmeasure000.val[:]
            antennas = H.root.sol000.rotationmeasure000.ant[:]
            axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')

        for antennaid, antenna in enumerate(antennas):
            if antenna[0:2] != keepanntennastr:
                if hasphase:
                    antennaxis = axisn.index('ant')
                    axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                    print('Resetting phase', antenna, 'Axis entry number', axisn.index('ant'))
                    # print(phase[:,:,antennaid,...])
                    if antennaxis == 0:
                        phase[antennaid, ...] = 0.0
                    if antennaxis == 1:
                        phase[:, antennaid, ...] = 0.0
                    if antennaxis == 2:
                        phase[:, :, antennaid, ...] = 0.0
                    if antennaxis == 3:
                        phase[:, :, :, antennaid, ...] = 0.0
                    if antennaxis == 4:
                        phase[:, :, :, :, antennaid, ...] = 0.0
                    # print(phase[:,:,antennaid,...])
                if hasamps:
                    antennaxis = axisn.index('ant')
                    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                    print('Resetting amplitude', antenna, 'Axis entry number', axisn.index('ant'))
                    if antennaxis == 0:
                        amp[antennaid, ...] = 1.0
                    if antennaxis == 1:
                        amp[:, antennaid, ...] = 1.0
                    if antennaxis == 2:
                        amp[:, :, antennaid, ...] = 1.0
                    if antennaxis == 3:
                        amp[:, :, :, antennaid, ...] = 1.0
                    if antennaxis == 4:
                        amp[:, :, :, :, antennaid, ...] = 1.0
                    if fulljones:
                        amp[..., 1] = 0.0  # XY, assume pol is last axis
                        amp[..., 2] = 0.0  # YX, assume pol is last axis

                if hastec:
                    antennaxis = axisn.index('ant')
                    axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
                    print('Resetting TEC', antenna, 'Axis entry number', axisn.index('ant'))
                    if antennaxis == 0:
                        tec[antennaid, ...] = 0.0
                    if antennaxis == 1:
                        tec[:, antennaid, ...] = 0.0
                    if antennaxis == 2:
                        tec[:, :, antennaid, ...] = 0.0
                    if antennaxis == 3:
                        tec[:, :, :, antennaid, ...] = 0.0
                    if antennaxis == 4:
                        tec[:, :, :, :, antennaid, ...] = 0.0

                if hasrotation:
                    antennaxis = axisn.index('ant')
                    axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
                    print('Resetting rotation', antenna, 'Axis entry number', axisn.index('ant'))
                    if antennaxis == 0:
                        rotation[antennaid, ...] = 0.0
                    if antennaxis == 1:
                        rotation[:, antennaid, ...] = 0.0
                    if antennaxis == 2:
                        rotation[:, :, antennaid, ...] = 0.0
                    if antennaxis == 3:
                        rotation[:, :, :, antennaid, ...] = 0.0
                    if antennaxis == 4:
                        rotation[:, :, :, :, antennaid, ...] = 0.0
                if hasrotationmeasure:
                    antennaxis = axisn.index('ant')
                    axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
                    print('Resetting faradayrotation', antenna, 'Axis entry number', axisn.index('ant'))
                    if antennaxis == 0:
                        faradayrotation[antennaid, ...] = 0.0
                    if antennaxis == 1:
                        faradayrotation[:, antennaid, ...] = 0.0
                    if antennaxis == 2:
                        faradayrotation[:, :, antennaid, ...] = 0.0
                    if antennaxis == 3:
                        faradayrotation[:, :, :, antennaid, ...] = 0.0
                    if antennaxis == 4:
                        faradayrotation[:, :, :, :, antennaid, ...] = 0.0
                        # fill values back in
        if hasphase:
            H.root.sol000.phase000.val[:] = np.copy(phase)
        if hasamps:
            H.root.sol000.amplitude000.val[:] = np.copy(amp)
        if hastec:
            H.root.sol000.tec000.val[:] = np.copy(tec)
        if hasrotation:
            H.root.sol000.rotation000.val[:] = np.copy(rotation)
        if hasrotationmeasure:
            H.root.sol000.rotationmeasure000.val[:] = np.copy(faradayrotation)

    return


# reset_gains_noncore('merged_selfcalcycle11_testquick260.ms.avg.h5')
# sys.exit()


def phaseup(msinlist, datacolumn='DATA', superstation='core', start=0, dysco=True, metadata_compression=True):
    """ Phase up stations into a superstation.

    Args:
        msinlist (list): list of input Measurement Sets to iterate over.
        datacolumn (str): the input data column to phase up data from.
        superstation (str): stations to phase up. Can be 'core' or 'superterp'.
        start (int): selfcal cylce that is being started from. Phaseup will only occur if start == 0.
        dysco (bool): dysco compress the output dataset.
    Returns:
        msoutlist (list): list of output Measurement Sets.
    """
    msoutlist = []
    for ms in msinlist:
        msout = ms + '.phaseup'
        msoutlist.append(msout)

        cmd = "DP3 msin=" + ms + " steps=[add,filter] "
        cmd += "msout=" + msout + " msin.datacolumn=" + datacolumn + " "
        cmd += "filter.type=filter filter.remove=True "
        # Do not set to true: DP3's UVW compression does not work with the StationAdder (yet).
        cmd += "msout.uvwcompression=False "
        if dysco:
            cmd += "msout.storagemanager=dysco "
            cmd += 'msout.storagemanager.weightbitrate=16 '
        cmd += "add.type=stationadder "
        if superstation == 'core':
            cmd += "add.stations={ST001:'CS*'} filter.baseline='!CS*&&*' "
        if superstation == 'superterp':
            cmd += "add.stations={ST001:'CS00[2-7]*'} filter.baseline='!CS00[2-7]*&&*' "
        if not metadata_compression:
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '

        if start == 0:  # only phaseup if start selfcal from cycle 0, so skip for a restart
            if os.path.isdir(msout):
                os.system('rm -rf ' + msout)
                time.sleep(2)  # wait for the directory to be removed
            print(cmd)
            run(cmd)
    return msoutlist


def findfreqavg(ms, imsize, bwsmearlimit=1.0):
    """ Find the frequency averaging factor for a Measurement Set given a bandwidth smearing constraint.

    Args:
        ms (str): path to the Measurement Set.
        imsize (float): size of the image in pixels.
        bwsmearlimit (float): the fractional acceptable bandwidth smearing.
    Returns:
        avgfactor (int): the frequency averaging factor for the Measurement Set.
    """
    with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
        bwsmear = bandwidthsmearing(np.median(t.getcol('CHAN_WIDTH')), np.min(t.getcol('CHAN_FREQ')[0]), float(imsize), verbose=False)
        nfreq = len(t.getcol('CHAN_FREQ')[0])
    avgfactor = 0

    for count in range(2, 21):  # try average values between 2 to 20
        if bwsmear < (bwsmearlimit / float(count)):  # factor X avg
            if nfreq % count == 0:
                avgfactor = count
    return avgfactor


def compute_markersize(H5file):
    """ Computes matplotlib markersize for an H5parm.

    Args:
        H5file (str): path to an H5parm.
    Returns:
        markersize (int): marker size.
    """
    ntimes = ntimesH5(H5file)
    markersize = 2
    if ntimes < 450:
        markersize = 4
    if ntimes < 100:
        markersize = 10
    if ntimes < 50:
        markersize = 15
    
    if ntimes == 1:
        markersize = 2
        nfreqs = number_freqchan_h5(H5file)
        if nfreqs < 450:
            markersize = 4
        if nfreqs < 100:
            markersize = 10
        if nfreqs < 50:
            markersize = 15
    
    return markersize


def ntimesH5(H5file):
    """ Returns the number of timeslots in an H5parm.

    Args:
        H5file (str): path to H5parm.
    Returns:
        times (int): length of the time axis.
    """

    sol_types = ['amplitude000', 'phase000', 'tec000', 'rotationmeasure000', 'rotation000']

    with tables.open_file(H5file, mode='r') as H:
        for sol_type in sol_types:
            try:
                return len(getattr(H.root.sol000, sol_type).time[:])
            except AttributeError:
                continue

        print('No amplitude000, phase000, tec000, rotationmeasure000, or rotation000 solutions found')
        raise Exception('No amplitude000, phase000, tec000, rotationmeasure000, or rotation000 solutions found')


def create_backup_flag_col(ms, flagcolname='FLAG_BACKUP'):
    """ Creates a backup of the FLAG column.

    Args:
        ms (str): path to the Measurement Set.
        flagcolname (str): name of the output column.
    Returns:
        None
    """
    cname = 'FLAG'
    flags = []
    with table(ms, readonly=False, ack=True) as t:
        if flagcolname not in t.colnames():
            flags = t.getcol('FLAG')
            print('Adding flagging column', flagcolname, 'to', ms)
            desc = t.getcoldesc(cname)
            newdesc = makecoldesc(flagcolname, desc)
            newdmi = t.getdminfo(cname)
            newdmi['NAME'] = flagcolname
            t.addcols(newdesc, newdmi)
            t.putcol(flagcolname, flags)
    del flags
    return


def check_phaseup_station(ms):
    """ Check if the Measurement Set contains a superstation.

    Args:
        ms (str): path to the Measurement Set.
    Returns:
        None
    """
    with table(ms + '/ANTENNA', ack=False) as t:
        antennasms = list(t.getcol('NAME'))
    substr = 'ST'  # to check if a a superstation is present, assume this 'ST' string, usually ST001
    hassuperstation = any(substr in mystring for mystring in antennasms)
    print('Contains superstation?', hassuperstation)
    return hassuperstation


def checklongbaseline(ms):
    """ Check if the Measurement Set contains international stations.

    Args:
        ms (str): path to the Measurement Set.
    Returns:
        None
    """
    with table(ms + '/ANTENNA', ack=False) as t:
        antennasms = list(t.getcol('NAME'))
    substr = 'DE'  # to check if a German station is present, if yes assume this is long baseline data
    haslongbaselines = any(substr in mystring for mystring in antennasms)
    print('Contains long baselines?', haslongbaselines)
    return haslongbaselines


def average(mslist, freqstep, timestep=None, start=0, msinnchan=None, msinstartchan=0.,
            phaseshiftbox=None, msinntimes=None, makecopy=False,
            makesubtract=False, delaycal=False, freqresolution='195.3125kHz',
            dysco=True, cmakephasediffstat=False, dataincolumn='DATA',
            removeinternational=False, removemostlyflaggedstations=False, 
            useaoflagger=False, useaoflaggerbeforeavg=True, aoflagger_strategy=None,
            metadata_compression=True):
    """ Average and/or phase-shift a list of Measurement Sets.

    Args:
        mslist (list): list of Measurement Sets to iterate over.
        freqstep (int): the number of frequency slots to average.
        timestep (int): the number of time slots to average.
        start (int): selfcal cycle that is being started from.
        msinnchan (int): number of channels to take from the input Measurement Set.
        msinstartchan (int): start chanel for msinnchan
        phaseshiftbox (str): path to a DS9 region file to phaseshift or "align" 
                            "align" means phaseshift to pointing center of the first MS 
        msinntimes (int): number of timeslots to take from the input Measurement Set.
        makecopy (bool): appends '.copy' when making a copy of a Measurement Set.
        dysco (bool): Dysco compress the output Measurement Set.
    Returns:
        outmslist (list): list of output Measurement Sets.
    """
    # sanity check
    if len(mslist) != len(freqstep):
        print('Hmm, made a mistake with freqstep?')
        raise Exception('len(mslist) != len(freqstep)')

    outmslist = []
    for ms_id, ms in enumerate(mslist):
        if (int(''.join([i for i in str(freqstep[ms_id]) if i.isdigit()])) > 0) or (timestep is not None) or (
                msinnchan is not None) or \
                (phaseshiftbox is not None) or (msinntimes is not None) \
                or removeinternational or removemostlyflaggedstations:  # if this is True then average

            # set this first, change name if needed below
            msout = ms + '.avg'
            if makecopy:
                msout = ms + '.copy'
            if makesubtract:
                msout = ms + '.subtracted'
            if cmakephasediffstat:
                msout = ms + '.avgphasediffstat'

            msout = os.path.basename(msout)
            cmd = 'DP3 msin=' + ms + ' av.type=averager '
            
            if not metadata_compression and msout != '.':
                cmd += 'msout.uvwcompression=False '
                cmd += 'msout.antennacompression=False '
                cmd += 'msout.scalarflags=False '
            
            if check_phaseup_station(ms):
                cmd += 'msout.uvwcompression=False '
                # cmd += 'msout.antennacompression=False '
            cmd += 'msout=' + msout + ' msin.weightcolumn=WEIGHT_SPECTRUM '
            cmd += 'msin.datacolumn=' + dataincolumn + ' '
            if dysco:
                cmd += 'msout.storagemanager=dysco '
                cmd += 'msout.storagemanager.weightbitrate=16 '
            if phaseshiftbox is not None and not (phaseshiftbox == 'align' and ms_id ==0): # avoid phaseshift on first MS if 'align' is used
                if removeinternational:
                    cmd += ' steps=[f,shift,av] '
                    cmd += " f.type=filter f.baseline='[CR]S*&' f.remove=True "
                else:
                    cmd += ' steps=[shift,av] '
                cmd += ' shift.type=phaseshifter '
                if phaseshiftbox == 'align':
                    with table(mslist[0] + '/FIELD', ack=False) as t:
                        ra_ref, dec_ref = t.getcol('PHASE_DIR').squeeze() # get the reference direction of the first MS in radians
                    print(f'Aligning phase center to {ra_ref}, {dec_ref} (in radians) of first MS {mslist[0]}')
                    # shift to the first MS's phase center
                    cmd += ' shift.phasecenter=\\[' + str(ra_ref) + ',' + str(dec_ref) + '\\] '
                else:
                    cmd += ' shift.phasecenter=\\[' + getregionboxcenter(phaseshiftbox) + '\\] '
            else:
                if removeinternational:
                    cmd += ' steps=[f,av] '
                    cmd += " f.type=filter f.baseline='[CR]S*&' f.remove=True "
                else:
                    cmd += ' steps=[av] '
            if removemostlyflaggedstations:
                flagstationlist = return_antennas_highflaggingpercentage(ms, percentage=args['removemostlyflaggedstations_percentage'])
                if len(flagstationlist) > 0:
                    baselinestr = filter_baseline_str_removestations(flagstationlist)
                    cmd += 'fs.type=filter fs.baseline=' + baselinestr + ' fs.remove=True '
                    cmd = cmd.replace('steps=[', 'steps=[fs,')

                    # freqavg
            if freqstep[ms_id] is not None:
                if str(freqstep[ms_id]).isdigit():
                    cmd += 'av.freqstep=' + str(freqstep[ms_id]) + ' '
                else:
                    freqstepstr = ''.join([i for i in freqstep[ms_id] if not i.isalpha()])
                    freqstepstrnot = ''.join([i for i in freqstep[ms_id] if i.isalpha()])
                    if freqstepstrnot != 'Hz' and freqstepstrnot != 'kHz' and freqstepstrnot != 'MHz':
                        print('For frequency averaging only units of (k/M)Hz are allowed, used:', freqstepstrnot)
                        raise Exception('For frequency averaging only units of " (k/M)Hz" are allowed')
                    cmd += 'av.freqresolution=' + str(freqstep[ms_id]) + ' '

                    # timeavg
            if timestep is not None:
                if str(timestep).isdigit():
                    cmd += 'av.timestep=' + str(int(timestep)) + ' '
                else:
                    timestepstr = ''.join([i for i in timestep if not i.isalpha()])
                    timestepstrnot = ''.join([i for i in timestep if i.isalpha()])
                    if timestepstrnot != 's' and timestepstrnot != 'sec':
                        print('For time averaging only units of s(ec) are allowed, used:', timestepstrnot)
                        raise Exception('For time averaging only units of "s(ec)" are allowed')
                    cmd += 'av.timeresolution=' + str(timestepstr) + ' '

            if msinnchan is not None:
                cmd += 'msin.nchan=' + str(msinnchan) + ' '
                cmd += 'msin.startchan=' + str(msinstartchan) + ' '
            if msinntimes is not None:
                cmd += 'msin.ntimes=' + str(msinntimes) + ' '
            if useaoflagger:
                cmd += 'ao.type=aoflag '
                cmd += 'ao.keepstatistics=False '
                cmd += 'ao.memoryperc=50 '
                cmd += 'ao.overlapperc=10 '
                if aoflagger_strategy is not None:
                    if os.path.isfile(aoflagger_strategy): # try full location first
                        cmd += 'ao.strategy=' +  aoflagger_strategy + ' '
                    else: # try strategy in flagging_strategies
                        cmd += 'ao.strategy=' + f'{datapath}/flagging_strategies/' + aoflagger_strategy + ' '
                if useaoflaggerbeforeavg:
                   cmd = cmd.replace("steps=[", "steps=[ao,") # insert s first step
                else:
                   cmd = cmd.replace("av] ", "av,ao] ") # insert as last step (av was previous last step)
            if start == 0:
                print('Average with default WEIGHT_SPECTRUM:', cmd)
                if os.path.isdir(msout):
                    os.system('rm -rf ' + msout)
                    time.sleep(2)  # wait for the directory to be removed
                run(cmd, log=True)

            msouttmp = ms + '.avgtmp'
            msouttmp = os.path.basename(msouttmp)
            cmd = 'DP3 msin=' + ms + ' av.type=averager '
            
            if not metadata_compression and msout != '.':
                cmd += 'msout.uvwcompression=False '
                cmd += 'msout.antennacompression=False '
                cmd += 'msout.scalarflags=False '
            
            
            if check_phaseup_station(ms):
                cmd += 'msout.uvwcompression=False '
            if removeinternational:
                cmd += ' steps=[f,av] '
                cmd += " f.type=filter f.baseline='[CR]S*&' f.remove=True "
            else:
                cmd += ' steps=[av] '
            if dysco:
                cmd += ' msout.storagemanager=dysco '
                cmd += 'msout.storagemanager.weightbitrate=16 '
            cmd += 'msout=' + msouttmp + ' msin.weightcolumn=WEIGHT_SPECTRUM_SOLVE '
            if removemostlyflaggedstations:
                # step below can be skipped since flagstationlist was already made/defined
                # flagstationlist = return_antennas_highflaggingpercentage(ms)
                if len(flagstationlist) > 0:
                    baselinestr = filter_baseline_str_removestations(flagstationlist)
                    cmd += 'fs.type=filter fs.baseline=' + baselinestr + ' fs.remove=True '
                    cmd = cmd.replace('steps=[', 'steps=[fs,')

            # freqavg
            if freqstep[ms_id] is not None:
                if str(freqstep[ms_id]).isdigit():
                    cmd += 'av.freqstep=' + str(freqstep[ms_id]) + ' '
                else:
                    freqstepstr = ''.join([i for i in freqstep[ms_id] if not i.isalpha()])
                    freqstepstrnot = ''.join([i for i in freqstep[ms_id] if i.isalpha()])
                    if freqstepstrnot != 'Hz' and freqstepstrnot != 'kHz' and freqstepstrnot != 'MHz':
                        print('For frequency averaging only units of (k/M)Hz are allowed, used:', freqstepstrnot)
                        raise Exception('For frequency averaging only units of " (k/M)Hz" are allowed')
                    cmd += 'av.freqresolution=' + str(freqstep[ms_id]) + ' '

                    # timeavg
            if timestep is not None:
                if str(timestep).isdigit():
                    cmd += 'av.timestep=' + str(int(timestep)) + ' '
                else:
                    timestepstr = ''.join([i for i in timestep if not i.isalpha()])
                    timestepstrnot = ''.join([i for i in timestep if i.isalpha()])
                    if timestepstrnot != 's' and timestepstrnot != 'sec':
                        print('For time averaging only units of s(ec) are allowed, used:', timestepstrnot)
                        raise Exception('For time averaging only units of "s(ec)" are allowed')
                    cmd += 'av.timeresolution=' + str(timestepstr) + ' '
            if msinnchan is not None:
                cmd += 'msin.nchan=' + str(msinnchan) + ' '
                cmd += 'msin.startchan=' + str(msinstartchan) + ' '
            if msinntimes is not None:
                cmd += 'msin.ntimes=' + str(msinntimes) + ' '

            if start == 0:
                with table(ms) as t:
                    if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():  # check if present otherwise this is not needed
                        print('Average with default WEIGHT_SPECTRUM_SOLVE:', cmd)
                        if os.path.isdir(msouttmp):
                            os.system('rm -rf ' + msouttmp)
                            time.sleep(2)  # wait for the directory to be removed
                        run(cmd)

                        # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                        with table(msout, readonly=False) as t2:
                            print('Adding WEIGHT_SPECTRUM_SOLVE')
                            #desc = t2.getcoldesc('WEIGHT_SPECTRUM')
                            #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
                            #t2.addcols(desc)
                            addcol(t2, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

                            with table(msouttmp, readonly=True) as t3:
                                imweights = t3.getcol('WEIGHT_SPECTRUM')
                                t2.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

                        # clean up
                        os.system('rm -rf ' + msouttmp)

            outmslist.append(msout)
        else:
            outmslist.append(ms)  # so no averaging happened
        
   
    if start > 0 and makecopy:  # only fix UVW if this is not the first selfcal cycle
        for ms in outmslist:
            if not os.path.isdir(ms):
                print('No MS found:', ms)
                print('Maybe you made a mistake and you did not set --start=0?')
                raise Exception('MS found and averaging did not produce it because start > 0')
                
    # fix MeerKAT UVW coordinates (needs to be done each time we average with DP3)
    fix_uvw(outmslist) 
   
    return outmslist


def uvmaxflag(msin, uvmax):
    """
    Flags visibilities in a Measurement Set (MS) with UV distances greater than a specified maximum.

    Parameters:
        msin (str): Path to the input Measurement Set.
        uvmax (float): Maximum allowed UV distance (in wavelengths). Visibilities with UV distances greater than this value will be flagged.

    Returns:
        None

    Side Effects:
        Executes a DP3 command to flag data in the input MS based on the specified UV distance threshold.
    """
    cmd = 'DP3 msin=' + msin + ' msout=. steps=[f] f.type=uvwflag f.uvlambdamax=' + str(uvmax)
    print(cmd)
    run(cmd)
    return


def tecandphaseplotter(h5, ms, outplotname='plot.png'):
    """ Make TEC and phase plots.

    Args:
        h5 (str): path to the H5parm to plot.
        ms (str): path to th ecorresponding Measurement Set.
        outplotname (str): name of the output plot.
    Returns:
        None
    """
    if not os.path.isdir('plotlosoto%s' % os.path.basename(
            ms)):  # needed because if this is the first plot this directory does not yet exist
        os.system('mkdir plotlosoto%s' % os.path.basename(ms))
    cmd = f'python {submodpath}/plot_tecandphase.py  '
    cmd += '--H5file=' + h5 + ' --outfile=plotlosoto%s/%s_nolosoto.png' % (os.path.basename(ms), outplotname)
    print(cmd)
    run(cmd)
    return


def runaoflagger(mslist, strategy=None):
    """ Run aoglagger on a Measurement Set.

    Args:
        mslist (list): list of Measurement Sets to iterate over.
    Returns:
        None
    """
    for ms in mslist:
        if strategy is not None:
            if os.path.isfile(strategy): # try full location first
                cmd = 'aoflagger -strategy ' + strategy + ' ' + ms
            else: # try strategy in flagging_strategies
                cmd = 'aoflagger -strategy ' + f'{datapath}/flagging_strategies/' + strategy + ' ' + ms
        else:
            cmd = 'aoflagger ' + ms
        print(cmd)
        run(cmd)
    return


def build_applycal_dde_cmd(inparmdblist):
    if not isinstance(inparmdblist, list):
        inparmdblist = [inparmdblist]
    cmd = ''  # empy string to start with
    count = 0
    for parmdb in inparmdblist:
        with tables.open_file(parmdb) as H:
            soltabs = list(H.root.sol000._v_children.keys())
        if 'phase000' in soltabs:
            cmd += 'ddecal.applycal.ac' + str(count) + '.parmdb=' + parmdb + ' '
            cmd += 'ddecal.applycal.ac' + str(count) + '.correction=phase000 '
            count = count + 1
        if 'amplitude000' in soltabs:
            cmd += 'ddecal.applycal.ac' + str(count) + '.parmdb=' + parmdb + ' '
            cmd += 'ddecal.applycal.ac' + str(count) + '.correction=amplitude000 '
            count = count + 1

        H.close()

    if count < 1:
        print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        raise Exception('Something went wrong, cannot build the applycal command. H5 file is valid?')

    cmd += 'ddecal.applycal.steps=['
    for i in range(count):
        cmd += 'ac' + str(i)
        if i < count - 1:  # to avoid last comma in the steps list
            cmd += ','
    cmd += ']'

    return cmd


def corrupt_modelcolumns(ms, h5parm, modeldatacolumns, modelstoragemanager=None):
    """ Ccorrupt a list of model data columns with H5parm solutions

    Args:
        ms (str): path to a Measurement Set to apply solutions to.
        h5parm (list/str): H5parms to apply.
        modeldatacolumns (list): Model data columns list, there should be more than one, also these columns should alread exist. Note that input column will be overwritten.
    Returns:
        None
    """

    special_DIL = False
    with tables.open_file(h5parm, mode='r') as H:
        dirnames = None
        for sol_type in ['phase000', 'amplitude000', 'tec000', 'rotation000','rotationmeasure000']:
            try:
                dirnames = getattr(H.root.sol000, sol_type).dir[:]
                break  # Stop if dirnames is found
            except tables.NoSuchNodeError:
                continue  # Move to the next soltype if the current one is missing

    dirnames = dirnames.tolist()
    if 'DIL' in dirnames[0].decode("utf-8"):
        special_DIL = True

    for m_id, modelcolumn in enumerate(modeldatacolumns):
        if special_DIL:
            applycal(ms, h5parm, msincol=modelcolumn, msoutcol=modelcolumn,
                     dysco=False, invert=False, direction=dirnames[m_id].decode("utf-8"), modelstoragemanager=modelstoragemanager)
        else:
            applycal(ms, h5parm, msincol=modelcolumn, msoutcol=modelcolumn,
                     dysco=False, invert=False, direction=modelcolumn, modelstoragemanager=modelstoragemanager)
    return


def applycal(ms, inparmdblist, msincol='DATA', msoutcol='CORRECTED_DATA',
             msout='.', dysco=True, modeldatacolumns=[], invert=True, direction=None,
             find_closestdir=False, updateweights=False, modelstoragemanager=None, 
             missingantennabehavior='error', metadata_compression=True, timeslotsperparmupdate=200,
             auto_update_timeslotsperparmupdate=False):
    """ Apply an H5parm to a Measurement Set.

    Args:
        ms (str): path to a Measurement Set to apply solutions to.
        inparmdblist (list): list of H5parms to apply.
        msincol (str): input column to apply solutions to.
        msoutcol (str): output column to store corrected data in.
        msout (str): name of the output Measurement Set.
        dysco (bool): Dysco compress the output Measurement Set.
        modeldatacolumns (list): Model data columns list, if len(modeldatacolumns) > 1 we have a DDE solve
        invert (bool): invert the applycal (=corrupt)
        direction (str): Name of the direction in a multi-dir h5 for the applycal (find_closestdir needs to be False in this case)
        find_closestdir (bool): find closest direction (to phasedir MS) in multi-dir h5 file to apply
        updateweights (bool): Update WEIGHT_SPECTRUM in DP3
        missingantennabehavior (str): for DP3, must be error or flag
    Returns:
        None
    """
    # get number of frequency channels in MS and update timeslotsperparmupdate if needed
    # for large MSs with many frequency channels it smees we need to reduce the timeslotsperparmupdate to avoid segmentation faults
    # For example GMRT data with nchan=2688 and 3535 results in a segmentation faults with the DP3 default timeslotsperparmupdate=200
    if auto_update_timeslotsperparmupdate:
        with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
            nfreq = len(t.getcol('CHAN_FREQ')[0])
            if nfreq > 500:  # if more than 500 channels reduce timeslotsperparmupdate
                timeslotsperparmupdate = 100
            if nfreq > 1000: 
                timeslotsperparmupdate = 20
            if nfreq > 2000:
                timeslotsperparmupdate = 10
            if nfreq > 4000:
                timeslotsperparmupdate = 5

    if find_closestdir and direction is not None:
        print('Wrong input, you cannot use find_closestdir and set a direction')
        raise Exception('Wrong input, you cannot use find_closestdir and set a direction')

    if len(modeldatacolumns) > 1:
        return
        # to allow both a list or a single file (string)
    if not isinstance(inparmdblist, list):
        inparmdblist = [inparmdblist]

    cmd = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(), 8])) + ' msin=' + ms
    cmd += ' msout=' + msout + ' '
    
    if check_phaseup_station(ms): 
        if msout != '.': cmd += 'msout.uvwcompression=False ' # only invoke when writing new MS
    cmd += 'msin.datacolumn=' + msincol + ' '
    if msout == '.':
        cmd += 'msout.datacolumn=' + msoutcol + ' '
    if not metadata_compression and msout != '.':
        cmd += 'msout.uvwcompression=False '
        cmd += 'msout.antennacompression=False '
        cmd += 'msout.scalarflags=False '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    if modelstoragemanager is not None and not dysco:
        cmd += 'msout.storagemanager=' + modelstoragemanager + ' '
    count = 0
    for parmdb in inparmdblist:
        if find_closestdir:
            direction = make_utf8(find_closest_ddsol(parmdb, ms))
            print('Applying direction:', direction)
        if fulljonesparmdb(parmdb):
            cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
            cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
            cmd += 'ac' + str(count) + '.type=applycal '
            cmd += 'ac' + str(count) + '.correction=fulljones '
            cmd += 'ac' + str(count) + '.soltab=[amplitude000,phase000] '
            cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
            if not invert:
                cmd += 'ac' + str(count) + '.invert=False '
            if direction is not None:
                if direction.startswith(
                        'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                    cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                else:
                    cmd += 'ac' + str(count) + '.direction=' + direction + ' '
            count = count + 1
        else:
            with tables.open_file(parmdb) as H:
                soltabs = list(H.root.sol000._v_children.keys())

            if not invert:  # so corrupt, rotation comes first in a rotation+diagonal apply
                if 'rotation000' in soltabs or 'rotationmeasure000' in soltabs:
                    # note that rotation comes before amplitude&phase for a corrupt (important if the solve was a rotation+diagonal one)
                    cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                    cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                    cmd += 'ac' + str(count) + '.type=applycal '
                    cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                    if 'rotation000' in soltabs: cmd += 'ac' + str(count) + '.correction=rotation000 '
                    if 'rotationmeasure000' in soltabs: cmd += 'ac' + str(count) + '.correction=rotationmeasure000 '
                    cmd += 'ac' + str(count) + '.invert=False '
                    if direction is not None:
                        if direction.startswith(
                                'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                            cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                        else:
                            cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                    count = count + 1

            if 'phase000' in soltabs:
                cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                cmd += 'ac' + str(count) + '.correction=phase000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '
                if direction is not None:
                    if direction.startswith(
                            'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                        cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                        cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                count = count + 1

            if 'amplitude000' in soltabs:
                cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                cmd += 'ac' + str(count) + '.correction=amplitude000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '
                if updateweights:
                    cmd += 'ac' + str(count) + '.updateweights=True '
                if direction is not None:
                    if direction.startswith(
                            'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                        cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                        cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                count = count + 1

            if 'tec000' in soltabs:
                cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                cmd += 'ac' + str(count) + '.correction=tec000 '
                if not invert:
                    cmd += 'ac' + str(count) + '.invert=False '
                if direction is not None:
                    if direction.startswith(
                            'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                        cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                    else:
                        cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                count = count + 1

            if invert:  # so applycal, rotation comes last in a rotation+diagonal apply
                if 'rotation000' in soltabs or 'rotationmeasure000' in soltabs:
                    cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                    cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                    cmd += 'ac' + str(count) + '.type=applycal '
                    cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                    if 'rotation000' in soltabs: cmd += 'ac' + str(count) + '.correction=rotation000 '
                    if 'rotationmeasure000' in soltabs: cmd += 'ac' + str(count) + '.correction=rotationmeasure000 '
                    cmd += 'ac' + str(
                        count) + '.invert=True '  # by default True but set here as a reminder because order matters for rotation+diagonal in this DP3 step depending on invert=True/False
                    if direction is not None:
                        if direction.startswith(
                                'MODEL_DATA'):  # because then the direction name in the h5 contains bracket strings
                            cmd += 'ac' + str(count) + '.direction=[' + direction + '] '
                        else:
                            cmd += 'ac' + str(count) + '.direction=' + direction + ' '
                    count = count + 1

            H.close()

    if count < 1:
        print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        raise Exception('Something went wrong, cannot build the applycal command. H5 file is valid?')
    # build the steps command
    cmd += 'steps=['
    for i in range(count):
        cmd += 'ac' + str(i)
        if i < count - 1:  # to avoid last comma in the steps list
            cmd += ','
    cmd += ']'

    print('DP3 applycal:', cmd)
    run(cmd, log=True)
    return


def inputchecker(args, mslist):
    """ Check input validity.
    Args:
        args (dict): argparse inputs.
        mslist (str list): list of ms
    """
    # if args['BLsmooth']:
    #    if True in args['BLsmooth_list']:
    #        print('--BLsmooth cannot be used together with --BLsmooth-list')
    #        raise Exception('--BLsmooth cannot be used together with --BLsmooth-list')

    # set telescope
    with table(mslist[0] + '/OBSERVATION', ack=False) as t:
        telescope = t.getcol('TELESCOPE_NAME')[0]

    if 0 in args['useaoflagger_correcteddata_selfcalcycle_list']:
        print('--useaoflagger-correcteddata-selfcalcycle-list cannot contain 0')
        raise Exception('--useaoflagger-correcteddata-selfcalcycle-list cannot contain 0') 

    if args['useaoflagger_correcteddata'] and args['DDE']:
        print('--useaoflagger-correcteddata cannot be used together with --DDE')
        raise Exception('--useaoflagger-correcteddata cannot be used together with --DDE')

    if args['aoflagger_strategy'] is not None:
        if not os.path.isfile(args['aoflagger_strategy']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy']):
                print('Flagging strategy file not found:', args['aoflagger_strategy'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy'])
    
    if args['aoflagger_strategy_correcteddata'] is not None:
        if not os.path.isfile(args['aoflagger_strategy_correcteddata']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy_correcteddata']):
                print('Flagging strategy file not found:', args['aoflagger_strategy_correcteddata'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy_correcteddata'])  

    if args['aoflagger_strategy_afterbandpassapply'] is not None:
        if not os.path.isfile(args['aoflagger_strategy_afterbandpassapply']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy_afterbandpassapply']):
                print('Flagging strategy file not found:', args['aoflagger_strategy_afterbandpassapply'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy_afterbandpassapply'])  

    if args['skymodelsetjy'] and args['skymodel'] is not None:
        print('--skymodelsetjy cannot be used together with --skymodel')
        raise Exception('--skymodelsetjy cannot be used together with --skymodel')
    
    if args['skymodelsetjy'] and args['skymodelpointsource'] is not None:
        print('--skymodelsetjy cannot be used together with --skymodelpointsource')
        raise Exception('--skymodelsetjy cannot be used together with --skymodelpointsource')
    
    if args['skymodelsetjy'] and args['wscleanskymodel'] is not None:
        print('--skymodelsetjy cannot be used together with --wscleanskymodel')
        raise Exception('--skymodelsetjy cannot be used together with --wscleanskymodel')

    if args['auto_directions'] and not args['DDE']:
       print('--auto_directions can only be used in combination with --DDE')
       raise Exception('--auto_directions can only be used in combination with --DDE')

    if args['auto_directions'] and telescope != 'LOFAR':
       print('--auto_directions can only be used with LOFAR observations for now')
       raise Exception('--auto_directions can only be used with LOFAR observations for now')

    if args['DP3_BDA_imaging'] and not args['DDE']:
        print('--DP3-BDA-imaging can only be used in combination with --DDE')
        raise Exception('--DP3-BDA-imaging can only be used in combination with --DDE')

    if args['DP3_BDA_imaging'] and args['groupms_h5facetspeedup']:
        print('--DP3-BDA-imaging cannot be used together with --groupms-h5facetspeedup')
        raise Exception('--DP3-BDA-imaging cannot be used together with --groupms-h5facetspeedup')

    if True in args['BLsmooth_list']:
        if len(args['soltypecycles_list']) != len(args['BLsmooth_list']):
            print('--BLsmooth-list length does not match the length of --soltype-list')
            raise Exception('--BLsmooth-list length does not match the length of --soltype-list')

    if args['modelstoragemanager'] != 'stokes_i' and args['modelstoragemanager'] is not None:
         print(args['modelstoragemanager'])
         print('Wrong input for --modelstoragemanager, needs to be "stokes_i" or None')
         raise Exception('Wrong input for --modelstoragemanager, needs to be stokes_i or None')

    if args['bandpass']:
        if args['stack'] or args['DDE'] or args['stopafterskysolve'] or args['stopafterpreapply']:
            print('--bandpass cannot be used with --stack, --DDE, --stopafterskysolve, or --stopafterpreapply')
            raise Exception('--bandpass cannot be used with --stack or --DDE')
        if args['skymodel'] is None and args['skymodelpointsource'] is None \
            and args['wscleanskymodel'] is None and not args['skymodelsetjy']:
            print('skymodel, skymodelpointsource, skymodelsetjy, or wscleanskymodel needs to be set')
            raise Exception('skymodel, skymodelpointsource, or wscleanskymodel needs to be set')

    for tmp in args['BLsmooth_list']:
        # print(args['BLsmooth_list'])
        if not (isinstance(tmp, bool)):
            print(args['BLsmooth_list'])
            print('--BLsmooth-list is not a list of booleans')
            raise Exception('--BLsmooth-list is not a list of booleans')

    if args['stack']:  # avoid options that cannot be used when --stack is set
        if args['DDE']:
            print('--dde cannot be used with --stack')
            raise Exception('--dde cannot be used with --stack')
        if args['compute_phasediffstat']:
            print('--compute-phasediffstat cannot be used with --stack')
            raise Exception('--compute-phasediffstat cannot be used with --stack')
        if args['fitsmask'] is not None:
            print('--fitsmask cannot be used with --stack')
            raise Exception('--fitsmask cannot be used with --stack')
        if args['update_uvmin']:
            print('--update-uvmin cannot be used with --stack')
            raise Exception('--update-uvmin cannot be used with --stack')
        if args['update_multiscale']:
            print('--update-multiscale cannot be used with --stack')
            raise Exception('--update-multiscale cannot be used with --stack')
        if args['remove_outside_center']:
            print('--remove-outside-center cannot be used with --stack')
            raise Exception('--remove-outside-center cannot be used with --stack')
        if args['auto']:
            print('--auto cannot be used with --stack')
            raise Exception('--auto cannot be used with --stack')
        if args['tgssfitsimage'] is not None:
            print('--tgssfitsimage cannot be used with --stack')
            raise Exception('--tgssfitsimage cannot be used with --stack')
        if args['QualityBasedWeights']:
            print('--QualityBasedWeights cannot be used with --stack')
            raise Exception('--QualityBasedWeights cannot be used with --stack')

    if not args['stack']:
        if type(args['skymodel']) is list:
            print('Skymodel cannot be a list if --stack is not set')
            raise Exception('Skymodel cannot be a list if --stack is not set')

    if args['DDE'] and args['preapplyH5_list'][0] is not None:
        print('--DDE and --preapplyH5-list cannot be used together')
        raise Exception('--DDE and --preapplyH5_list cannot be used together')

    if args['DDE']:
        for ms in mslist:
            with table(ms, readonly=True, ack=False) as t:
                if 'CORRECTED_DATA' in t.colnames():  # not allowed for DDE runs (because solving from DATA and imaging from DATA with an h5)
                    print(ms, 'contains a CORRECTED_DATA column, this is not allowed when using --DDE')
                    raise Exception('CORRECTED_DATA should not be present when using option --DDE')

    if args['DDE'] and not args['forwidefield']:
        print('--forwidefield needs to be set in DDE mode')
        raise Exception('--forwidefield needs to be set in DDE mode')

    if args['DDE'] and args['idg']:
        print('Option --idg cannot be used with option --DDE')
        raise Exception('Option --idg cannot be used with option --DDE')

    if not args['stack']:
        if type(args['skymodelpointsource']) is list:
            print('skymodelpointsource cannot be a list if --stack is not set')
            raise Exception('skymodelpointsource cannot be a list if --stack is not set')

    if not args['stack']:
        if type(args['wscleanskymodel']) is list:
            print('wscleanskymodel cannot be a list if --stack is not set')
            raise Exception('wscleanskymodel cannot be a list if --stack is not set')

            # sanity check for resetdir_list input
    if type(args['resetdir_list']) is not list:
        print('--resetdir-list needs to be of type list')
        raise Exception('--resetdir-list needs to be of type list')
    for resetdir in args['resetdir_list']:  # check if it contains None or a integer list-type
        if resetdir is not None:
            if type(resetdir) is not list:
                print('--resetdir-list needs to None list-items, or contain a list of directions_id')
                raise Exception('--resetdir-list needs to None list-items, or contain a list of directions_id')
            else:
                for dir_id in resetdir:
                    if type(dir_id) is not int:
                        print('--resetdir-list, direction IDs provided need to be integers')
                        raise Exception('--resetdir-list, direction IDs provided need to be integers')
                    if dir_id < 0:  # if we get here we have an integer
                        print('--resetdir-list, direction IDs provided need to be integers >= 0')
                        raise Exception('--resetdir-list, direction IDs provided need to be integers >= 0')
                    data = ascii.read(args['facetdirections'])
                    if dir_id + 1 > len(data):
                        print(
                            '--direction IDs provided for reset is too high for the number of directions provided by ' +
                            args['facetdirections'])
                        raise Exception(
                            '--direction IDs provided for reset is too high for the number of directions provided by ' +
                            args['facetdirections'])

    if args['groupms_h5facetspeedup']:
        if not args['DDE']:
            print('--groupms-h5facetspeedup can only be used with --DDE')
            raise Exception('--groupms-h5facetspeedup can only be used with --DDE')
    if args['DDE_predict'] == 'DP3':
        if args['fitspectralpol'] < 1:
            print(
                '--fitspectralpol needs to be turned on, otherwise no skymodel is produced by WSClean and we cannot predict these components with DP3. Put --DDE-predict=WSCLEAN or fitspectralpol>0')
            raise Exception('--Invalid combination of --fitspectralpol and --DDE-predict')
        if type(args['fitspectralpol']) is not str:
            if args['fitspectralpol'] < 1:
                print(
                    '--fitspectralpol needs to be turned on, otherwise no skymodel is produced by WSClean and we cannot predict these components with DP3. Put --DDE-predict=WSCLEAN or fitspectralpol>0')
                raise Exception('--Invalid combination of --fitspectralpol and --DDE-predict')

    if args['uvmin'] is not None and type(args['uvmin']) is not list:
        if args['uvmin'] < 0.0:
            print('--uvmin needs to be positive')
            raise Exception('--uvmin needs to be positive')
    if args['uvminim'] is not None and type(args['uvminim']) is not list:
        if args['uvminim'] < 0.0:
            print('--uvminim needs to be positive')
            raise Exception('--uvminim needs to be positive')
    if args['uvmaxim'] is not None and args['uvminim'] is not None and type(args['uvmaxim']) is not list and type(
            args['uvminim']) is not list:
        if args['uvmaxim'] <= args['uvminim']:
            print('--uvmaxim needs to be larger than --uvminim')
            raise Exception('--uvmaxim needs to be larger than --uvminim')
    if args['uvmax'] is not None and args['uvmin'] is not None and type(args['uvmax']) is not list and type(
            args['uvmin']) is not list:
        if args['uvmax'] <= args['uvmin']:
            print('--uvmax needs to be larger than --uvmin')
            raise Exception('--uvmaxim needs to be larger than --uvmin')
            # print(args['uvmax'], args['uvmin'], args['uvminim'],args['uvmaxim'])

    if 'fulljones' in args['soltype_list'] and args['doflagging'] and not args['forwidefield']:
        print('--doflagging is True, cannot be combined with fulljones solve, set it to False or use --forwidefield')
        raise Exception('--doflagging is True, cannot be combined with fulljones solve')

    if args['iontimefactor'] <= 0.0:
        print('BLsmooth iontimefactor needs to be positive')
        raise Exception('BLsmooth iontimefactor needs to be positive')
    if args['iontimefactor'] > 10.0:
        print('BLsmooth iontimefactor is way too high')
        raise Exception('BLsmooth iontimefactor is way too high')

    if args['ionfreqfactor'] <= 0.0:
        print('BLsmooth tecfactor needs to be positive')
        raise Exception('BLsmooth tecfactor needs to be positive')
    if args['ionfreqfactor'] > 10000.0:
        print('BLsmooth tecfactor is way too high')
        raise Exception('BLsmooth tecfactor is way too high')

    if args['phaseshiftbox'] is not None:
        if not os.path.isfile(args['phaseshiftbox']):
            print('Cannot find:', args['phaseshiftbox'])
            raise Exception('Cannot find:' + args['phaseshiftbox'])

    if args['beamcor'] not in ['auto', 'yes', 'no']:
        print('beamcor is not auto, yes, or no')
        raise Exception('Invalid input, beamcor is not auto, yes, or no')

    if args['beamcor'] != 'auto' and telescope != 'LOFAR':
        print('beamcor is a LOFAR specific option, keep this at "auto"')
        raise Exception('beamcor is a LOFAR specific option, keep this at "auto"')

    if args['DDE_predict'] not in ['DP3', 'WSCLEAN']:
        print('DDE-predict is not DP3 or WSCLEAN')
        raise Exception('DDE-predict is not DP3 or WSCLEAN')

    for nrtmp in args['normamps_list']:
        if nrtmp not in ['normamps_per_ant', 'normslope', 'normamps', 'normslope+normamps',
                         'normslope+normamps_per_ant'] and nrtmp is not None:
            print(
                'Invalid input: --normamps_list can only contain "normamps", "normslope", "normamps_per_ant", "normslope+normamps", "normslope+normamps_per_ant" or None')
            raise Exception(
                'Invalid input: --normamps_list can only contain "normamps", "normslope", "normamps_per_ant", "normslope+normamps", "normslope+normamps_per_ant" or None')

    for antennaconstraint in args['antennaconstraint_list']:
        if antennaconstraint not in ['superterp', 'coreandfirstremotes', 'core', 'remote',
                                     'all', 'international', 'alldutch', 'core-remote',
                                     'coreandallbutmostdistantremotes', 'alldutchbutnoST001',
                                     'distantremote', 'alldutchandclosegerman'] \
                and antennaconstraint is not None:
            print(
                'Invalid input, antennaconstraint can only be core, superterp, coreandfirstremotes, remote, alldutch, international, alldutchandclosegerman, or all')
            raise Exception(
                'Invalid input, antennaconstraint can only be core, superterp, coreandfirstremotes, remote, alldutch, international, alldutchandclosegerman, or all')

    for resetsols in args['resetsols_list']:
        if resetsols not in ['superterp', 'coreandfirstremotes', 'core', 'remote',
                             'all', 'international', 'alldutch', 'core-remote', 'coreandallbutmostdistantremotes',
                             'alldutchbutnoST001', 'distantremote', 'alldutchandclosegerman'] \
                and resetsols is not None:
            print(
                'Invalid input, resetsols can only be core, superterp, coreandfirstremotes, remote, alldutch, international, distantremote, alldutchandclosegerman, or all')
            raise Exception(
                'Invalid input, resetsols can only be core, superterp, coreandfirstremotes, remote, alldutch, international, distantremote, alldutchandclosegerman, or all')

    # if args['DDE']:
    #   for soltype in args['soltype_list']:
    #    if soltype in ['scalarphasediff', 'scalarphasediffFR']:
    #        print('Invalid soltype input in combination with DDE type solve')
    #        raise Exception('Invalid soltype input in combination with DDE type solve')

    for soltype in args['soltype_list']:
        if soltype not in ['complexgain', 'scalarcomplexgain', 'scalaramplitude',
                           'amplitudeonly', 'phaseonly', 'fulljones', 'rotation',
                           'rotation+diagonal', 'rotation+diagonalphase',
                           'rotation+diagonalamplitude', 'rotation+scalar',
                           'rotation+scalaramplitude', 'rotation+scalarphase', 'tec',
                           'tecandphase', 'scalarphase',
                           'scalarphasediff', 'scalarphasediffFR', 'phaseonly_phmin',
                           'rotation_phmin', 'tec_phmin',
                           'tecandphase_phmin', 'scalarphase_phmin', 'scalarphase_slope',
                           'phaseonly_slope', 'faradayrotation', 'faradayrotation+diagonal',
                           'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude',
                           'faradayrotation+scalar', 'faradayrotation+scalaramplitude',
                           'faradayrotation+scalarphase']:
            print('Invalid soltype input')
            raise Exception('Invalid soltype input')

    # check that there is only on scalarphasediff solve and it the first entry
    if 'scalarphasediff' in args['soltype_list'] or 'scalarphasediffFR' in args['soltype_list']:
        if (args['soltype_list'][0] != 'scalarphasediff') and \
                (args['soltype_list'][0] != 'scalarphasediffFR'):
            print('scalarphasediff/scalarphasediffFR need to be to first solves in the list')
            raise Exception('scalarphasediff/scalarphasediffFR need to be to first solves in the list')
        sccount = 0
        for soltype in args['soltype_list']:
            if soltype == 'scalarphasediff' or soltype == 'scalarphasediffFR':
                sccount = sccount + 1
        if sccount > 1:
            print('only one scalarphasediff/scalarphasediffFR solve allowed')
            raise Exception('only one scalarphasediff/scalarphasediffFR solve allowed')

    if args['facetdirections'] is not None:
        if not os.path.isfile(args['facetdirections']):
            print('--facetdirections file does not exist')
            raise Exception('--facetdirections file does not exist')
        check_for_BDPbug_longsolint(mslist, args['facetdirections'])

    if args['DDE']:
        if 'fulljones' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonal' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonalamplitude' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonalphase' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalar' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalarphase' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalaramplitude' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonal' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonalamplitude' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonalphase' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalar' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalarphase' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalaramplitude' in args['soltype_list']:
            print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')


        if args['wscleanskymodel'] is not None and args['facetdirections'] is None:
            print('If --DDE and --wscleanskymodel are set provide a direction file via --facetdirections')
            raise Exception('DDE with a wscleanskymodel requires a user-specified facetdirections')
        if args['wscleanskymodel'] is not None and args['Nfacets'] > 0:
            print('If --DDE and --wscleanskymodel are set you cannot use Nfacets')
            raise Exception('If --DDE and --wscleanskymodel are set you cannot use Nfacets')

    for ms in mslist:
        if not check_phaseup_station(ms):
            for soltype_id, soltype in enumerate(args['soltype_list']):
                if soltype in ['scalarphasediff', 'scalarphasediff']:
                    if args['antennaconstraint_list'][soltype_id] not in ['superterp', 'coreandfirstremotes', 'core',
                                                                          'remote', 'distantremote',
                                                                          'all', 'international', 'alldutch',
                                                                          'core-remote',
                                                                          'coreandallbutmostdistantremotes',
                                                                          'alldutchbutnoST001',
                                                                          'alldutchandclosegerman'] and args[
                        'phaseupstations'] is None:
                        print(
                            'scalarphasediff/scalarphasediff type solves require a antennaconstraint, for example "core", or phased-up data')
                        raise Exception(
                            'scalarphasediff/scalarphasediff type solves require a antennaconstraint, or phased-up data')

    if args['boxfile'] is not None:
        if not (os.path.isfile(args['boxfile'])):
            print('Cannot find boxfile, file does not exist')
            raise Exception('Cannot find boxfile, file does not exist')

    if args['fitsmask'] is not None and args['fitsmask'] != 'nofitsmask':
        if not (os.path.isfile(args['fitsmask'])):
            print('Cannot find fitsmask, file does not exist')
            raise Exception('Cannot find fitsmask, file does not exist')

    if args['fitsmask'] is not None and args['fitsmask_start'] is not None:
        if not (os.path.isfile(args['fitsmask'])):
            print('Cannot set fitsmask and fitsmask-start at the same time')
            raise Exception('Cannot set fitsmask and fitsmask-start at the same time')

    if args['DS9cleanmaskregionfile'] is not None: 
        if not (os.path.isfile(args['DS9cleanmaskregionfile'])):
            print('Cannot find DS9cleanmaskregionfile, file does not exist')
            raise Exception('Cannot find DS9cleanmaskregionfile, file does not exist')


    if args['skymodel'] is not None:
        if type(args['skymodel']) is str:
            # print(type(args['skymodel']), args['skymodel'][0])
            if not (os.path.isfile(args['skymodel'])) and not (os.path.isdir(args['skymodel'])):
                print('Cannot find skymodel, file does not exist', args['skymodel'])
                raise Exception('Cannot find skymodel, file does not exist')
        if type(args['skymodel']) is list:
            for skym in args['skymodel']:
                if not os.path.isfile(skym) and not os.path.isdir(skym):
                    print('Cannot find skymodel, file does not exist', skym)
                    raise Exception('Cannot find skymodel, file does not exist')

    if args['docircular'] and args['dolinear']:
        print('Conflicting input, docircular and dolinear used')
        raise Exception('Conflicting input, docircular and dolinear used')

    if which('DP3') is None:
        print('Cannot find DP3, forgot to source lofarinit.[c]sh?')
        raise Exception('Cannot find DP3, forgot to source lofarinit.[c]sh?')

    if which('wsclean') is None:
        print('Cannot find WSclean, forgot to source lofarinit.[c]sh?')
        raise Exception('Cannot find WSClean, forgot to source lofarinit.[c]sh?')

    if which('MakeMask.py') is None and which('breizorro') is None:
        print('Cannot find MakeMask.py or breizorro, forgot to install it?')
        raise Exception('Cannot find MakeMask.py or breizorro, forgot to install it?')

    if which('taql') is None:
        print('Cannot find taql, forgot to install it?')
        raise Exception('Cannot find taql, forgot to install it?')

    # Check boxfile and imsize settings
    if args['boxfile'] is None and args['imsize'] is None:
        if not checklongbaseline(sorted(args['ms'])[0]):
            print('Incomplete input detected, either boxfile or imsize is required')
            raise Exception('Incomplete input detected, either boxfile or imsize is required')

    if args['boxfile'] is not None and args['imsize'] is not None:
        print('Wrong input detected, both boxfile and imsize are set')
        raise Exception('Wrong input detected, both boxfile and imsize are set')

    if args['imager'] not in ['DDFACET', 'WSCLEAN']:
        print('Wrong input detected for option --imager, should be DDFACET or WSCLEAN')
        raise Exception('Wrong input detected for option --imager, should be DDFACET or WSCLEAN')

    if args['phaseupstations'] is not None:
        if args['phaseupstations'] not in ['core', 'superterp']:
            print('Wrong input detected for option --phaseupstations, should be core or superterp')
            raise Exception('Wrong input detected for option --phaseupstations, should be core or superterp')
    if args['phaseupstations'] is not None:
        if args['phaseupstations'] in args['antennaconstraint_list']:
            print(
                'Wrong input detected for option --antennaconstraint-list, --phaseupstations is set and phased-up stations are not available anymore for --antennaconstraint-list')
            raise Exception(
                'Wrong input detected for option --antennaconstraint-list, --phaseupstations is set and phased-up stations are not available anymore for --antennaconstraint-list')

    if args['soltypecycles_list'][0] != 0:
        print('Wrong input detected for option --soltypecycles-list should always start with 0')
        raise Exception('Wrong input detected for option --soltypecycles-list should always start with 0')

    if len(args['soltypecycles_list']) != len(args['soltype_list']):
        print('Wrong input detected, length soltypecycles-list does not match that of soltype-list')
        raise Exception('Wrong input detected, length soltypecycles-list does not match that of soltype-list')

    for soltype_id, soltype in enumerate(args['soltype_list']):
        wronginput = False
        if soltype in ['tecandphase', 'tec', 'tec_phmin', 'tecandphase_phmin']:
            try:  # in smoothnessconstraint_list is not filled by the user
                if args['smoothnessconstraint_list'][soltype_id] > 0.0:
                    print('smoothnessconstraint should be 0.0 for a tec-like solve')
                    wronginput = True
            except:
                pass
            if wronginput:
                raise Exception('smoothnessconstraint should be 0.0 for a tec-like solve')

    for smoothnessconstraint in args['smoothnessconstraint_list']:
        if smoothnessconstraint < 0.0:
            print('Smoothnessconstraint must be equal or larger than 0.0')
            raise Exception('Smoothnessconstraint must be equal or larger than 0.0')
    for smoothnessreffrequency in args['smoothnessreffrequency_list']:
        if smoothnessreffrequency < 0.0:
            print('Smoothnessreffrequency must be equal or larger than 0.0')
            raise Exception('Smoothnessreffrequency must be equal or larger than 0.0')

    if (args['skymodel'] is not None) and (args['skymodelpointsource']) is not None:
        print('Wrong input, you cannot use a separate skymodel file and then also set skymodelpointsource')
        raise Exception('Wrong input, you cannot use a separate skymodel file and then also set skymodelpointsource')
    if (args['skymodelpointsource'] is not None and type(args['skymodelpointsource']) is not list):
        if args['skymodelpointsource'] <= 0.0:
            print('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
            raise Exception('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
    if type(args['skymodelpointsource']) is list:
        for skymp in args['skymodelpointsource']:
            if float(skymp) <= 0.0:
                print('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
                raise Exception('Wrong input, flux density provided for skymodelpointsource is <= 0.0')

    if (args['msinstartchan'] < 0):
        print('Wrong input for msinstartchan, must be larger than zero')
        raise Exception('Wrong input for msinstartchan, must be larger than zero')

    if (args['msinnchan'] is not None):
        if (args['msinnchan'] <= 0):
            print('Wrong input for msinnchan, must be larger than zero')
            raise Exception('Wrong input for msinnchan, must be larger than zero')
    if (args['msinntimes'] is not None):
        if (args['msinntimes'] <= 1):
            print('Wrong input for msinntimes, must be larger than 1')
            raise Exception('Wrong input for msinntimes, must be larger than 1')

    if (args['skymodelpointsource'] is not None) and (args['predictskywithbeam']):
        print('Combination of skymodelpointsource and predictskywithbeam not supported')
        print('Provide a skymodel file to predict the sky with the beam')
        raise Exception('Combination of skymodelpointsource and predictskywithbeam not supported')

    if (args['wscleanskymodel'] is not None) and (args['skymodelpointsource']) is not None:
        print('Wrong input, you cannot use a wscleanskymodel and then also set skymodelpointsource')
        raise Exception('Wrong input, you cannot use a wscleanskymodel and then also set skymodelpointsource')

    if (args['wscleanskymodel'] is not None) and (args['skymodel']) is not None:
        print('Wrong input, you cannot use a wscleanskymodel and then also set skymodel')
        raise Exception('Wrong input, you cannot use a wscleanskymodel and then also set skymodel')

    if (args['wscleanskymodel'] is not None) and (args['predictskywithbeam']):
        print('Combination of wscleanskymodel and predictskywithbeam not supported')
        print('Provide a skymodel component file to predict the sky with the beam')
        raise Exception('Combination of wscleanskymodel and predictskywithbeam not supported')

    if (args['wscleanskymodel'] is not None) and (args['imager'] == 'DDFACET'):
        print('Combination of wscleanskymodel and DDFACET as an imager is not supported')
        raise Exception('Combination of wscleanskymodel and DDFACET as an imager is not supported')
    if (args['wscleanskymodel'] is not None):
        if len(glob.glob(args['wscleanskymodel'] + '-????-model.fits')) < 2:
            print('Not enough WSClean channel model images found')
            print(glob.glob(args['wscleanskymodel'] + '-????-model.fits'))
            raise Exception('Not enough WSClean channel model images found')
        # if len(glob.glob(args['wscleanskymodel'] + '-????-model.fits')) != args['channelsout']:
        #    print('Number of model images provided needs to match channelsout')
        #    raise Exception('Number of model images provided needs to match channelsout')
        if (args['wscleanskymodel'].find('/') != -1):
            print('wscleanskymodel contains a slash, not allowed, needs to be in pwd')
            raise Exception('wscleanskymodel contains a slash, not allowed, needs to be in pwd')
        if (args['wscleanskymodel'].find('..') != -1):
            print('wscleanskymodel contains .., not allowed, needs to be in pwd')
            raise Exception('wscleanskymodel contains .., not allowed, needs to be in pwd')
    return


def get_resolution(ms):
    uvmax = get_uvwmax(ms)
    with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
        freq = np.median(t.getcol('CHAN_FREQ'))
        print('Central freq [MHz]', freq / 1e6, 'Longest baselines [km]', uvmax / 1e3)
    res = 1.22 * 3600. * 180. * ((299792458. / freq) / uvmax) / np.pi
    return res


def get_uvwmax(ms):
    """ Find the maximum squared sum of UVW coordinates.

    Args:
        ms (str): path to a Measurement Set.
    Returns:
        None
    """
    with table(ms, ack=False) as t:
        uvw = t.getcol('UVW')
        ssq = np.sqrt(np.sum(uvw ** 2, axis=1))
        print(uvw.shape)
    return np.max(ssq)


def makeBBSmodelforFITS(filename, extrastrname=''):
    img = bdsf.process_image(filename,mean_map='zero', rms_map=True, rms_box = (100,10))
    img.write_catalog(format='bbs', bbs_patches='source', \
                      outfile='source' + extrastrname + '.skymodel'  , clobber=True)
    return 'source' + extrastrname + '.skymodel'


def makeBBSmodelforVLASS(filename, extrastrname=''):
    img = bdsf.process_image(filename, mean_map='zero', rms_map=True, rms_box=(100, 10))
    # frequency=150e6, beam=(25./3600,25./3600,0.0) )
    img.write_catalog(format='bbs', bbs_patches='source', outfile='vlass' + extrastrname + '.skymodel', clobber=True)
    # bbsmodel = 'bla.skymodel'
    del img
    return 'vlass' + extrastrname + '.skymodel'


def makeBBSmodelforTGSS(boxfile=None, fitsimage=None, pixelscale=None, imsize=None,
                        ms=None, extrastrname=''):
    """ Creates a TGSS skymodel in DP3-readable format.

    Args:
        boxfile (str): path to the DS9 region to create a model for.
        fitsimage (str): name of the FITS image the model will be created from.
        pixelscale (float): number of arcsec per pixel.
        imsize (int): image size in pixels.
        ms (str): if no box file is given, use this Measurement Set to determine the sky area to make a model of.
    Returns:
        tgss.skymodel: name of the output skymodel (always tgss[#nr].skymodel).
    """
    tgsspixsize = 6.2
    if boxfile is None and imsize is None:
        print('Wring input detected, boxfile or imsize needs to be set')
        raise Exception('Wring input detected, boxfile or imsize needs to be set')
    if boxfile is not None:
        r = pyregion.open(boxfile)
        if len(r[:]) > 1:
            print('Composite region file, not allowed')
            raise Exception('Composite region file, not allowed')
        phasecenter = getregionboxcenter(boxfile)
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

    print('TGSS imsize:', xs)
    print('TGSS image center:', phasecenterc)
    logger.info('TGSS imsize:' + str(xs))
    logger.info('TGSS image center:' + str(phasecenterc))

    # sys.exit()

    if fitsimage is None:
        filename = SkyView.get_image_list(position=phasecenterc, survey='TGSS ADR1', pixels=int(xs), cache=False)
        print(filename)
        if os.path.isfile(filename[0].split('/')[-1]):
            os.system('rm -f ' + filename[0].split('/')[-1])
        time.sleep(10)
        os.system('wget ' + filename[0])
        filename = filename[0].split('/')[-1]
        print(filename)
    else:
        filename = fitsimage

    img = bdsf.process_image(filename, mean_map='zero', rms_map=True, rms_box=(100, 10), frequency=150e6,
                             beam=(25. / 3600, 25. / 3600, 0.0))
    img.write_catalog(format='bbs', bbs_patches='source', outfile='tgss' + extrastrname + '.skymodel', clobber=True)
    # bbsmodel = 'bla.skymodel'
    del img
    print(filename)
    return 'tgss' + extrastrname + '.skymodel', filename


def getregionboxcenter(regionfile, standardbox=True):
    """ Extract box center of a DS9 box region.

    Args:
        regionfile (str): path to the region file.
        standardbox (bool): only allow square, non-rotated boxes.
    Returns:
        regioncenter (str): DP3 compatible string for phasecenter shifting.
    """
    r = pyregion.open(regionfile)

    if len(r[:]) > 1:
        print('Only one region can be specified, your file contains', len(r[:]))
        raise Exception('Only one region can be specified, your file contains')

    if r[0].name != 'box':
        print('Only box region supported')
        raise Exception('Only box region supported')

    ra = r[0].coord_list[0]
    dec = r[0].coord_list[1]
    boxsizex = r[0].coord_list[2]
    boxsizey = r[0].coord_list[3]
    angle = r[0].coord_list[4]

    if standardbox:
        if boxsizex != boxsizey:
            print('Only a square box region supported, you have these sizes:', boxsizex, boxsizey)
            raise Exception('Only a square box region supported')
        if np.abs(angle) > 1:
            print('Only normally oriented sqaure boxes are supported, your region is oriented under angle:', angle)
            raise Exception('Only normally oriented sqaure boxes are supported, your region is oriented under angle')

    regioncenter = ('{:12.8f}'.format(ra) + 'deg,' + '{:12.8f}'.format(dec) + 'deg').replace(' ', '')
    return regioncenter


def smearing_bandwidth(r, th, nu, dnu):
    """ Returns the left over intensity I/I0 after bandwidth smearing.
    Args:
        r (float or Astropy Quantity): distance from the phase center in arcsec.
        th (float or Astropy Quantity): angular resolution in arcsec.
        nu (float): observing frequency.
        dnu (float): averaging frequency.
    """
    r = r + 1e-9  # Add a tiny offset to prevent division by zero.
    I = (np.sqrt(np.pi) / (2 * np.sqrt(np.log(2)))) * ((th * nu) / (r * dnu)) * scipy.special.erf(
        np.sqrt(np.log(2)) * ((r * dnu) / (th * nu)))
    return I


def bandwidthsmearing(chanw, freq, imsize, verbose=True):
    """ Calculate the fractional intensity loss due to bandwidth smearing.

    Args:
        chanw (float): bandwidth.
        freq (float): observing frequency.
        imsize (int): image size in pixels.
        verbose (bool): print information to the screen.
    Returns:
        R (float): fractional intensity loss.
    """
    R = (chanw / freq) * (imsize / 6.)  # asume we have used 3 pixels per beam
    if verbose:
        print('R value for bandwidth smearing is:', R)
        logger.info('R value for bandwidth smearing is: ' + str(R))
        if R > 1.:
            print('Warning, try to increase your frequency resolution, or lower imsize, to reduce the R value below 1')
            logger.warning(
                'Warning, try to increase your frequency resolution, or lower imsize, to reduce the R value below 1')
    return R


def smearing_time(r, th, t):
    """ Returns the left over intensity I/I0 after time smearing.
    Args:
        r (float or Astropy Quantity): distance from the phase center in arcsec.
        th (float or Astropy Quantity): angular resolution in arcsec.
        t (float): averaging time in seconds.
    """
    r = r + 1e-9  # Add a tiny offset to prevent division by zero.

    I = 1 - 1.22e-9 * (r / th) ** 2 * t ** 2
    return I


def smearing_time_ms(msin, t):
    res = get_resolution(msin)
    r_dis = 3600. * compute_distance_to_pointingcenter(msin, returnval=True, dologging=False)
    return smearing_time(r_dis, res, t)

def smearing_time_ms_imsize(msin, imsize, pixelscale):
    with table(msin, readonly=True, ack=False) as t:
        time = np.unique(t.getcol('TIME'))
        tint = np.abs(time[1] - time[0])
    res = get_resolution(msin)
    r_dis = 3600.*np.sqrt(2)*imsize*pixelscale # image diagonal length in arcsec
    return smearing_time(r_dis, res, tint)

def flag_smeared_data(msin):
    Ismear = smearing_time_ms(msin, get_time_preavg_factor_LTAdata(msin))
    if Ismear < 0.5:
        print('Smeared', Ismear)
        # uvmaxflag(msin, uvmax)

    # uvmax = get_uvwmax(ms)
    with table(msin + '/SPECTRAL_WINDOW', ack=False) as t:
        freq = np.median(t.getcol('CHAN_FREQ'))
    # print('Central freq [MHz]', freq/1e6, 'Longest baselines [km]', uvmax/1e3)

    t = get_time_preavg_factor_LTAdata(msin)
    r_dis = 3600. * compute_distance_to_pointingcenter(msin, returnval=True, dologging=False)

    flagval = None
    for uvrange in list(np.arange(100e3, 5000e3, 10e3)):
        res = 1.22 * 3600. * 180. * ((299792458. / freq) / uvrange) / np.pi
        Ismear = smearing_time(r_dis, res, t)
        if Ismear > 0.968:
            flagval = uvrange
            # print(uvrange,Ismear)
    print('Data above', flagval / 1e3, 'klambda is affected by time smearing')
    if flagval / 1e3 < 650:
        print('Flagging data above uvmin value of [klambda]', msin, flagval / 1e3)
        uvmaxflag(msin, flagval)
    return


def number_freqchan_h5(h5parmin):
    """
    Get the number of frequencies in an H5 solution file.

    Args:
        h5parmin (str): Path to the input H5parm file.

    Returns:
        int: Number of frequency channels in the H5 file.
    """
    freq = []
    solution_types = ['phase000', 'amplitude000', 'rotation000', 'tec000', 'rotationmeasure000']

    with tables.open_file(h5parmin) as H:
        for sol_type in solution_types:
            try:
                freq = getattr(H.root.sol000, sol_type).freq[:]
                break  # Stop once we successfully retrieve the frequency data
            except tables.NoSuchNodeError:
                continue  # Try the next solution type if current one is missing

    print('Number of frequency channels in this solutions file is:', len(freq))
    return len(freq)


def calculate_restoringbeam(mslist, LBA):
    """ Returns the restoring beam.

    Args:
        mslist (list): currently unused.
        LBA (bool): if data is LBA or not.
    Returns:
        restoringbeam (float): the restoring beam in arcsec.
    """
    if LBA:  # so we have LBA
        restoringbeam = 15.
    else:  # so we have HBA
        restoringbeam = 6.

    return restoringbeam


def print_title(version):
    print(r"""
               _______    ___       ______  _______ .___________.
              |   ____|  /   \     /      ||   ____||           |
              |  |__    /  ^  \   |  ,----'|  |__   `---|  |----`
              |   __|  /  /_\  \  |  |     |   __|      |  |     
              |  |    /  _____  \ |  `----.|  |____     |  |     
              |__|   /__/     \__\ \______||_______|    |__|     

         _______. _______  __       _______   ______      ___       __      
        /       ||   ____||  |     |   ____| /      |    /   \     |  |     
       |   (----`|  |__   |  |     |  |__   |  ,----'   /  ^  \    |  |     
        \   \    |   __|  |  |     |   __|  |  |       /  /_\  \   |  |     
    .----)   |   |  |____ |  `----.|  |     |  `----. /  _____  \  |  `----.
    |_______/    |_______||_______||__|      \______|/__/     \__\ |_______|


                      Reinout van Weeren (2021, A&A, 651, 115)

                              Starting.........
          """)

    print('\n\nVERSION: ' + version + '\n\n')
    logger.info('VERSION: ' + version)
    return

def makemslist(mslist):
    """ Create the input list for e.g. ddf-pipeline.

    Args:
        mslist (list): list of input Measurement Sets
    Returns:
        None
    """
    os.system('rm -rf mslist.txt')
    f = open('mslist.txt', 'w')
    for ms in mslist:
        f.write(str(ms) + '\n')
    f.close()
    return


def antennaconstraintstr(ctype, antennasms, HBAorLBA, useforresetsols=False, telescope='LOFAR'):
    """ Formats an anntena constraint string in a DP3-suitable format.

    Args:
        ctype (str): constraint type. Can be superterp, core, coreandfirstremotes, remote, alldutch, all, international, core-remote, coreandallbutmostdistantremotes, alldutchandclosegerman or alldutchbutnoST001.
        antennasms (list): antennas present in the Measurement Set.
        HBAorLBA (str): indicate HBA or LBA data. Can be HBA or LBA.
        useforresetsols (bool): whether it will be used with reset solution. Removes antennas that are not in antennasms.
        telescope (str): telescope name, used to check if MeerKAT data is used
    Returns:
        antstr (str): antenna constraint string for DP3.
    """
    antennasms = list(antennasms)
    # print(antennasms)
    if ctype != 'superterp' and ctype != 'core' and ctype != 'coreandfirstremotes' and \
            ctype != 'remote' and ctype != 'alldutch' and ctype != 'all' and \
            ctype != 'international' and ctype != 'core-remote' and ctype != 'coreandallbutmostdistantremotes' and \
            ctype != 'alldutchandclosegerman' and \
            ctype != 'alldutchbutnoST001' and ctype != 'distantremote' and ctype != 'mediumremote' and \
            ctype != 'closeremote' and ctype != 'corebutsuperterp' and ctype != 'closeinternational' and \
            ctype != 'distantinternational' and ctype != 'superstation':

        print('Invalid input, ctype can only be "superterp" or "core"')
        raise Exception('Invalid input, ctype can only be "superterp" or "core"')
    if HBAorLBA == 'LBA':
        if ctype == 'superterp':
            antstr = ['CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA', 'ST001']
        if ctype == 'core':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'ST001']
        if ctype == 'corebutsuperterp':
            antstr = ['CS001LBA', 'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 
                      'CS028LBA', 'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 
                      'CS301LBA', 'CS302LBA', 'CS401LBA', 'CS501LBA']            
        if ctype == 'coreandfirstremotes':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS106LBA', 'ST001']
        if ctype == 'coreandallbutmostdistantremotes':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS106LBA', 'RS307LBA', 'RS406LBA', 'RS407LBA', 'ST001']
        if ctype == 'remote':
            antstr = ['RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA', 'RS310LBA', 'RS406LBA', 'RS407LBA',
                      'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA', 'RS409LBA', 'RS508LBA', 'RS509LBA']
        if ctype == 'distantremote':
            antstr = ['RS310LBA', 'RS407LBA', 'RS208LBA', 'RS210LBA', 'RS409LBA', 'RS508LBA', 'RS509LBA']
        if ctype == 'mediumremote':
            antstr = ['RS406LBA', 'RS307LBA']
        if ctype == 'closeremote':
            antstr = ['RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA', 'RS106LBA']
        if ctype == 'alldutch':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS310LBA', 'RS406LBA', 'RS407LBA', 'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA',
                      'RS409LBA', 'RS508LBA', 'RS509LBA', 'ST001']
        if ctype == 'alldutchbutnoST001':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS310LBA', 'RS406LBA', 'RS407LBA', 'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA',
                      'RS409LBA', 'RS508LBA', 'RS509LBA']

        if ctype == 'all':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS310LBA', 'RS406LBA', 'RS407LBA', 'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA',
                      'RS409LBA', 'RS508LBA', 'RS509LBA',
                      'DE601LBA', 'DE602LBA', 'DE603LBA', 'DE604LBA', 'DE605LBA', 'DE609LBA', 'FR606LBA',
                      'SE607LBA', 'UK608LBA', 'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA', 'ST001']
        if ctype == 'international':
            antstr = ['DE601LBA', 'DE602LBA', 'DE603LBA', 'DE604LBA', 'DE605LBA', 'DE609LBA', 'FR606LBA',
                      'SE607LBA', 'UK608LBA', 'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA']
        if ctype == 'closeinternational':
            antstr = ['DE601LBA', 'DE602LBA']
        if ctype == 'distantinternational':
            antstr = ['DE603LBA', 'DE604LBA', 'DE605LBA', 'DE609LBA', 'FR606LBA', 'SE607LBA', 'UK608LBA',
                      'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA']          
        if ctype == 'core-remote':
            antstr1 = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                       'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                       'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                       'CS302LBA', 'CS401LBA', 'CS501LBA', 'ST001']
            antstr2 = ['RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA', 'RS310LBA', 'RS406LBA', 'RS407LBA',
                       'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA', 'RS409LBA', 'RS508LBA', 'RS509LBA']
        if ctype == 'alldutchandclosegerman':
            antstr = ['CS001LBA', 'CS002LBA', 'CS003LBA', 'CS004LBA', 'CS005LBA', 'CS006LBA', 'CS007LBA',
                      'CS011LBA', 'CS013LBA', 'CS017LBA', 'CS021LBA', 'CS024LBA', 'CS026LBA', 'CS028LBA',
                      'CS030LBA', 'CS031LBA', 'CS032LBA', 'CS101LBA', 'CS103LBA', 'CS201LBA', 'CS301LBA',
                      'CS302LBA', 'CS401LBA', 'CS501LBA', 'RS503LBA', 'RS305LBA', 'RS205LBA', 'RS306LBA',
                      'RS310LBA', 'RS406LBA', 'RS407LBA', 'RS106LBA', 'RS307LBA', 'RS208LBA', 'RS210LBA',
                      'RS409LBA', 'RS508LBA', 'RS509LBA', 'ST001', 'DE601LBA', 'DE605LBA']

    if HBAorLBA == 'HBA':
        if ctype == 'superterp':
            antstr = ['CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1', 'ST001']
        if ctype == 'remote':
            antstr = ['RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                      'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA']
        if ctype == 'distantremote':
            antstr = ['RS310HBA', 'RS407HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA']
        if ctype == 'mediumremote':
            antstr = ['RS406HBA', 'RS307HBA']
        if ctype == 'closeremote':
            antstr = ['RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS106HBA']            
        if ctype == 'core':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1', 'ST001']
        if ctype == 'corebutsuperterp':
            antstr = ['CS001HBA0', 'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 
                      'CS028HBA0', 'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 
                      'CS301HBA0', 'CS302HBA0', 'CS401HBA0', 'CS501HBA0', 
                      'CS001HBA1', 'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 
                      'CS028HBA1', 'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 
                      'CS301HBA1', 'CS302HBA1', 'CS401HBA1', 'CS501HBA1']  
        if ctype == 'coreandfirstremotes':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1', 'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA',
                      'RS106HBA', 'ST001']
        if ctype == 'coreandallbutmostdistantremotes':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1', 'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA',
                      'RS106HBA', 'RS307HBA', 'RS406HBA', 'RS407HBA', 'ST001']
        if ctype == 'alldutch':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1',
                      'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                      'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA', 'ST001']
        if ctype == 'alldutchandclosegerman':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1',
                      'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                      'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA', 'ST001',
                      'DE601HBA', 'DE605HBA']

        if ctype == 'alldutchbutnoST001':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1',
                      'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                      'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA']
        if ctype == 'all':
            antstr = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                      'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                      'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                      'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                      'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                      'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                      'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                      'CS302HBA1', 'CS401HBA1', 'CS501HBA1',
                      'RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                      'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA',
                      'DE601HBA', 'DE602HBA', 'DE603HBA', 'DE604HBA', 'DE605HBA', 'DE609HBA', 'FR606HBA',
                      'SE607HBA', 'UK608HBA', 'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA', 'ST001']
        if ctype == 'international':
            antstr = ['DE601HBA', 'DE602HBA', 'DE603HBA', 'DE604HBA', 'DE605HBA', 'DE609HBA', 'FR606HBA',
                      'SE607HBA', 'UK608HBA', 'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA']
        if ctype == 'closeinternational':
            antstr = ['DE601HBA', 'DE602HBA']
        if ctype == 'distantinternational':
            antstr = ['DE603HBA', 'DE604HBA', 'DE605HBA', 'DE609HBA', 'FR606HBA', 'SE607HBA', 'UK608HBA',
                      'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA']        
        if ctype == 'core-remote':
            antstr1 = ['CS001HBA0', 'CS002HBA0', 'CS003HBA0', 'CS004HBA0', 'CS005HBA0', 'CS006HBA0', 'CS007HBA0',
                       'CS011HBA0', 'CS013HBA0', 'CS017HBA0', 'CS021HBA0', 'CS024HBA0', 'CS026HBA0', 'CS028HBA0',
                       'CS030HBA0', 'CS031HBA0', 'CS032HBA0', 'CS101HBA0', 'CS103HBA0', 'CS201HBA0', 'CS301HBA0',
                       'CS302HBA0', 'CS401HBA0', 'CS501HBA0',
                       'CS001HBA1', 'CS002HBA1', 'CS003HBA1', 'CS004HBA1', 'CS005HBA1', 'CS006HBA1', 'CS007HBA1',
                       'CS011HBA1', 'CS013HBA1', 'CS017HBA1', 'CS021HBA1', 'CS024HBA1', 'CS026HBA1', 'CS028HBA1',
                       'CS030HBA1', 'CS031HBA1', 'CS032HBA1', 'CS101HBA1', 'CS103HBA1', 'CS201HBA1', 'CS301HBA1',
                       'CS302HBA1', 'CS401HBA1', 'CS501HBA1', 'ST001']
            antstr2 = ['RS503HBA', 'RS305HBA', 'RS205HBA', 'RS306HBA', 'RS310HBA', 'RS406HBA', 'RS407HBA',
                       'RS106HBA', 'RS307HBA', 'RS208HBA', 'RS210HBA', 'RS409HBA', 'RS508HBA', 'RS509HBA']

    if telescope == 'MeerKAT':
        if ctype == 'core':
            antstr = MeerKAT_antconstraint(ctype='core')
        if ctype == 'remote':
            antstr = MeerKAT_antconstraint(ctype='remote')
        if ctype == 'all':
            antstr = MeerKAT_antconstraint(ctype='all')

    if telescope == 'GMRT':
        if ctype == 'core':
            antstr = ['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14']    
        if ctype == 'remote':
            antstr = ['E02','E03','E04','E05','E06','S01','S02','S03','S04','S06','W01','W02','W03','W04','W05','W06'] 
        if ctype == 'all':
            antstr = ['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14'] + \
                     ['E02','E03','E04','E05','E06','S01','S02','S03','S04','S06','W01','W02','W03','W04','W05','W06']

    if telescope == 'ASKAP':
        if ctype == 'core':
            antstr = ['ak01','ak02','ak03','ak04','ak05','ak06','ak07','ak08','ak09','ak10','ak11','ak12',\
                      'ak13','ak14','ak15','ak16','ak17','ak18','ak19','ak20','ak21','ak22','ak23','ak25',\
                      'ak26','ak29']    
        if ctype == 'remote':
            antstr = ['ak24','ak27','ak28','ak30','ak31','ak32','ak33','ak34','ak35','ak36'] 
        if ctype == 'all':
            antstr = ['ak01','ak02','ak03','ak04','ak05','ak06','ak07','ak08','ak09','ak10','ak11','ak12',\
                      'ak13','ak14','ak15','ak16','ak17','ak18','ak19','ak20','ak21','ak22','ak23','ak24',\
                      'ak25','ak26','ak27','ak28','ak29','ak30','ak31','ak32','ak33','ak34','ak35','ak36'] 

    if ctype == 'superstation':
        antstr = ['ST001']
    if useforresetsols:
        antstrtmp = list(antstr)
        for ant in antstr:
            if ant not in antennasms:
                antstrtmp.remove(ant)
        return antstrtmp

    if ctype != 'core-remote':
        antstrtmp = list(
            antstr)  # important to use list here, otherwise it's not a copy(!!) and antstrtmp refers to antstr
        for ant in antstr:
            if ant not in antennasms:
                antstrtmp.remove(ant)
        antstr = ','.join(map(str, antstrtmp))
        antstr = '[[' + antstr + ']]'
    else:
        antstrtmp1 = list(antstr1)
        for ant in antstr1:
            if ant not in antennasms:
                antstrtmp1.remove(ant)
        antstr1 = ','.join(map(str, antstrtmp1))

        antstrtmp2 = list(antstr2)
        for ant in antstr2:
            if ant not in antennasms:
                antstrtmp2.remove(ant)
        antstr2 = ','.join(map(str, antstrtmp2))

        antstr = '[[' + antstr1 + '],[' + antstr2 + ']]'

    return antstr


def makephasediffh5(phaseh5, refant):
    """
    Adjusts the phase solutions in a HDF5 file by referencing them to a specified reference antenna.

    This function opens the given HDF5 file containing phase calibration solutions, computes the phase
    difference with respect to the specified reference antenna, and updates the phase solutions accordingly.
    It is intended for use with scalarphase/phaseonly solutions and does not support tecandphase solutions
    where the frequency axis is missing.

    Parameters
    ----------
    phaseh5 : str
        Path to the HDF5 file containing phase calibration solutions.
    refant : str
        Name of the reference antenna to which phases will be referenced.

    Notes
    -----
    - The function modifies the HDF5 file in place.
    - Only the XX polarization is updated; the YY polarization is set to zero.
    - The function assumes the structure of the HDF5 file matches the expected LOFAR calibration format.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5pol = tables.open_file(phaseh5, mode='a')

    phase_pol = H5pol.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    phase_pol_tmp = np.copy(phase_pol)
    # antenna   = H5pol.root.sol000.phase000.ant[:]
    print('Shape to make phase diff array', phase_pol.shape)
    print('Using refant:', refant)
    logger.info('Refant for XX/YY or RR/LL phase-referencing' + refant)

    # Reference phases so that we correct the phase difference with respect to a reference station
    refant_idx = np.where(H5pol.root.sol000.phase000.ant[:].astype(str) == refant)
    phase_pol_tmp_ref = phase_pol_tmp - phase_pol_tmp[:, :, refant_idx[0], :, :]

    phase_pol[:, :, :, :, 0] = phase_pol_tmp_ref[:, :, :, :, 0]  # XX
    phase_pol[:, :, :, :, -1] = 0.0 * phase_pol_tmp_ref[:, :, :, :, 0]  # YY

    H5pol.root.sol000.phase000.val[:] = phase_pol
    H5pol.flush()
    H5pol.close()
    return


def makephaseCDFh5(phaseh5, backup=True, testscfactor=1.):
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    if backup:
        if os.path.isfile(phaseh5 + '.psbackup'):
            os.system('rm -f ' + phaseh5 + '.psbackup')
        os.system('cp ' + phaseh5 + ' ' + phaseh5 + '.psbackup')

    H5 = tables.open_file(phaseh5, mode='a')

    phaseCDF = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    print('Shape to make phase CDF array', phaseCDF.shape)
    nfreq = len(H5.root.sol000.phase000.freq[:])
    for ff in range(nfreq - 1):
        # reverse order so phase increase towards lower frequnecies
        phaseCDF[:, nfreq - ff - 2, ...] = np.copy(
            phaseCDF[:, nfreq - ff - 2, ...] + (testscfactor * phaseCDF[:, nfreq - ff - 1, ...]))

    print(phaseCDF.shape)
    H5.root.sol000.phase000.val[:] = phaseCDF
    H5.flush()
    H5.close()
    return


def makephaseCDFh5_h5merger(phaseh5, ms, modeldatacolumns, backup=True, testscfactor=1.):
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    # if soltypein == 'scalarphase_slope':
    #   single_pol_merge = True
    # if soltypein == 'phaseonly_slope':
    #   single_pol = False
    # if soltypein not in ['scalarphase_slope','phaseonly_slope']:
    #   print('Wrong input in makephaseCDFh5_h5merger')
    #   sys.exit()

    if len(modeldatacolumns) > 1:
        merge_all_in_one = False
    else:
        merge_all_in_one = True

    if backup:
        if os.path.isfile(phaseh5 + '.psbackup'):
            os.system('rm -f ' + phaseh5 + '.psbackup')
        os.system('cp ' + phaseh5 + ' ' + phaseh5 + '.psbackup')
    # going to overwrite phaseh5, first move to new file
    if os.path.isfile(phaseh5 + '.in'):
        os.system('rm -f ' + phaseh5 + '.in')
    os.system('mv ' + phaseh5 + ' ' + phaseh5 + '.in')

    merge_h5(h5_out=phaseh5, h5_tables=phaseh5 + '.in', ms_files=ms, merge_all_in_one=merge_all_in_one,
             propagate_weights=True)
    H5 = tables.open_file(phaseh5, mode='a')

    phaseCDF = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    phaseCDF_tmp = np.copy(phaseCDF)
    print('Shape to make phase CDF array', phaseCDF.shape)
    nfreq = len(H5.root.sol000.phase000.freq[:])
    for ff in range(nfreq - 1):
        # reverse order so phase increase towards lower frequnecies
        phaseCDF[:, nfreq - ff - 2, ...] = np.copy(
            phaseCDF[:, nfreq - ff - 2, ...] + (testscfactor * phaseCDF[:, nfreq - ff - 1, ...]))

    print(phaseCDF.shape)
    H5.root.sol000.phase000.val[:] = phaseCDF
    H5.flush()
    H5.close()
    return


def copyoverscalarphase(scalarh5, phasexxyyh5):
    """
    Copies scalar phase solutions from a HDF5 file (scalarh5) to the XX and YY polarization axes
    of another HDF5 file (phasexxyyh5).

    This function is intended for use with scalar phase or phase-only solutions, and does not work
    for tecandphase solutions where the frequency axis is missing for phase000.

    Args:
        scalarh5 (str): Path to the input HDF5 file containing scalar phase solutions.
        phasexxyyh5 (str): Path to the HDF5 file with XX and YY polarization axes to be updated.

    Notes:
        - The function assumes the input files follow the structure produced by LOFAR calibration software.
        - The phase values from the scalar file are copied to both the XX (pol=0) and YY (pol=-1) axes
          for each antenna and direction.
        - The function modifies the phasexxyyh5 file in place.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5 = tables.open_file(scalarh5, mode='r')
    H5pol = tables.open_file(phasexxyyh5, mode='a')

    phase = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir
    phase_pol = H5pol.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    antenna = H5.root.sol000.phase000.ant[:]
    print('Shapes for pol copy', phase.shape, phase_pol.shape)

    for ant in range(len(antenna)):
        phase_pol[:, :, ant, :, 0] = phase[:, :, ant, :]  # XX
        phase_pol[:, :, ant, :, -1] = phase[:, :, ant, :]  # YY

    H5pol.root.sol000.phase000.val[:] = phase_pol[:, :, :]
    H5pol.flush()

    H5.close()
    H5pol.close()
    return


def copyovergain(gaininh5, gainouth5, soltype):
    """
    Copies gain solutions (amplitude and phase) from an input HDF5 file to an output HDF5 file,
    handling both scalar and polarized solutions depending on the solution type and the presence
    of polarization axes.

    Parameters
    ----------
    gaininh5 : str
        Path to the input HDF5 file containing gain solutions.
    gainouth5 : str
        Path to the output HDF5 file where gain solutions will be copied.
    soltype : str
        Type of solution to copy. Determines whether to copy both amplitude and phase
        ('full'), or only amplitude ('scalaramplitude' or 'amplitudeonly').

    Notes
    -----
    - If the solution table contains a polarization axis, the amplitude and phase arrays
      are copied directly.
    - If there is no polarization axis, the function expands the amplitude and phase arrays
      to fill the polarization dimension in the output file (typically for XX and YY).
    - Phase values are set to zero if only amplitude solutions are requested.
    - The function uses PyTables and h5parm for HDF5 file access and manipulation.

    Returns
    -------
    None
    """
    H5in = tables.open_file(gaininh5, mode='r')
    H5out = tables.open_file(gainouth5, mode='a')
    antenna = H5in.root.sol000.amplitude000.ant[:]

    h5 = h5parm.h5parm(gaininh5)
    ss = h5.getSolset('sol000')
    st = ss.getSoltab('amplitude000')
    axesnames = st.getAxesNames()
    h5.close()

    if 'pol' in axesnames:

        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            phase = H5in.root.sol000.phase000.val[:]
            H5out.root.sol000.phase000.val[:] = phase
        else:
            H5out.root.sol000.phase000.val[:] = 0.0

        amplitude = H5in.root.sol000.amplitude000.val[:]
        print('Shapes for gain copy with polarizations', amplitude.shape)
        H5out.root.sol000.amplitude000.val[:] = amplitude

    else:
        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            phase = H5in.root.sol000.phase000.val[:]
            phase_pol = H5out.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol

        amplitude = H5in.root.sol000.amplitude000.val[:]
        amplitude_pol = H5out.root.sol000.amplitude000.val[:]  # time, freq, ant, dir, pol
        print('Shapes for gain copy 1 pol', amplitude.shape)

        for ant in range(len(antenna)):
            if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
                phase_pol[:, :, ant, :, 0] = phase[:, :, ant, :]  # XX
                phase_pol[:, :, ant, :, -1] = phase[:, :, ant, :]  # YY
            amplitude_pol[:, :, ant, :, 0] = amplitude[:, :, ant, :]  # XX
            amplitude_pol[:, :, ant, :, -1] = amplitude[:, :, ant, :]  # YY
        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            H5out.root.sol000.phase000.val[:] = phase_pol[:, :, :]
        else:
            H5out.root.sol000.phase000.val[:] = 0.0
        H5out.root.sol000.amplitude000.val[:] = amplitude_pol[:, :, :]

    H5out.flush()
    H5in.close()
    H5out.close()
    return


def set_weights_h5_to_one(h5parm):
    """
    Set weights for the solutions that have valid numbers to 1.0
    This is useful for bandpass solutions because the losoto time median preserves the time depedent flagging otherwise
    Args:
      h5parm: h5parm file
    """
    with tables.open_file(h5parm) as H:
        soltabs = list(H.root.sol000._v_children.keys())
    H = tables.open_file(h5parm, mode='a')
    
    if 'phase000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.phase000.val[:])
         # update weights to 1.0
         H.root.sol000.phase000.weight[goodvals] = 1.0

    if 'amplitude000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.amplitude000.val[:])
         # update weights to 1.0
         H.root.sol000.amplitude000.weight[goodvals] = 1.0

    if 'tec000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.tec000.val[:])
         # update weights to 1.0
         H.root.sol000.tec000.weight[goodvals] = 1.0

    if 'rotation000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.rotation000.val[:])
         # update weights to 1.0
         H.root.sol000.rotation000.weight[goodvals] = 1.0

    if 'rotationmeasure000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.rotationmeasure000.val[:])
         # update weights to 1.0
         H.root.sol000.rotationmeasure000.weight[goodvals] = 1.0

    H.flush()
    H.close()
    return
    
    
    
def fix_phasereference(h5parm, refant):
    """ Phase reference values with respect to a reference station
    Args:
      h5parm: h5parm file
      refant: reference antenna
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')

    phase = H.root.sol000.phase000.val[:]
    refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant)  # to deal with byte strings
    print(refant_idx, refant)
    antennaxis = axisn.index('ant')

    print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        phasen = phase - phase[refant_idx[0], ...]
    if antennaxis == 1:
        phasen = phase - phase[:, refant_idx[0], ...]
    if antennaxis == 2:
        phasen = phase - phase[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        phasen = phase - phase[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        phasen = phase - phase[:, :, :, :, refant_idx[0], ...]
    phase = np.copy(phasen)

    # fill values back in
    H.root.sol000.phase000.val[:] = np.copy(phase)

    H.flush()
    H.close()
    return


def resetsolsforstations(h5parm, stationlist, refant=None):
    """ Reset solutions for stations

    Args:
      h5parm: h5parm file
      stationlist: station name list
      refant: reference antenna
    """
    print(h5parm, stationlist)
    fulljones = fulljonesparmdb(h5parm)  # True/False
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure = check_soltabs(h5parm)

    H = tables.open_file(h5parm, 'r+')

    if hasamps:
        antennas = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')

    elif hasphase:
        antennas = H.root.sol000.phase000.ant[:]
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')

    elif hastec:
        antennas = H.root.sol000.tec000.ant[:]
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')

    elif hasrotation:
        antennas = H.root.sol000.rotation000.ant[:]
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')

    elif hasrotationmeasure:
        antennas = H.root.sol000.rotationmeasure000.ant[:]
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')


    # in case refant is None but h5 still has phase
    # this can happen with a scalaramplitude and soltypelist_includedir is used
    # in this case we have pertubative direction
    # in this case h5_merger has already been run which created a phase000 entry
    if refant is None and hasphase:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create rotation000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotation:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create rotationmeasure000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotationmeasure:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create tec000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hastec:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    if hasamps:
        amp = H.root.sol000.amplitude000.val[:]
    if hasphase:  # also phasereference
        phase = H.root.sol000.phase000.val[:]
        refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            phasen = phase - phase[refant_idx[0], ...]
        if antennaxis == 1:
            phasen = phase - phase[:, refant_idx[0], ...]
        if antennaxis == 2:
            phasen = phase - phase[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            phasen = phase - phase[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            phasen = phase - phase[:, :, :, :, refant_idx[0], ...]
        phase = np.copy(phasen)

    if hastec:
        tec = H.root.sol000.tec000.val[:]
        refant_idx = np.where(H.root.sol000.tec000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
        print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            tecn = tec - tec[refant_idx[0], ...]
        if antennaxis == 1:
            tecn = tec - tec[:, refant_idx[0], ...]
        if antennaxis == 2:
            tecn = tec - tec[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            tecn = tec - tec[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            tecn = tec - tec[:, :, :, :, refant_idx[0], ...]
        tec = np.copy(tecn)

    if hasrotation:
        rotation = H.root.sol000.rotation000.val[:]
        refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
        print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            rotationn = rotation - rotation[refant_idx[0], ...]
        if antennaxis == 1:
            rotationn = rotation - rotation[:, refant_idx[0], ...]
        if antennaxis == 2:
            rotationn = rotation - rotation[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            rotationn = rotation - rotation[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            rotationn = rotation - rotation[:, :, :, :, refant_idx[0], ...]
        rotation = np.copy(rotationn)

    if hasrotationmeasure:
        faradayrotation = H.root.sol000.rotationmeasure000.val[:]
        refant_idx = np.where(H.root.sol000.rotationmeasure000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
        print('Referencing faradayrotation to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            faradayrotationn = faradayrotation - faradayrotation[refant_idx[0], ...]
        if antennaxis == 1:
            faradayrotationn = faradayrotation - faradayrotation[:, refant_idx[0], ...]
        if antennaxis == 2:
            faradayrotationn = faradayrotation - faradayrotation[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            faradayrotationn = faradayrotation - faradayrotation[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            faradayrotationn = faradayrotation - faradayrotation[:, :, :, :, refant_idx[0], ...]
        faradayrotation = np.copy(faradayrotationn)


    for antennaid, antenna in enumerate(antennas.astype(str)):  # to deal with byte formatted array
        # if not isinstance(antenna, str):
        #  antenna_str = antenna.decode() # to deal with byte formatted antenna names
        # else:
        #  antenna_str = antenna # already str type

        print(antenna, hasphase, hasamps, hastec, hasrotation)
        if antenna in stationlist:  # in this case reset value to 0.0 (or 1.0)
            if hasphase:
                antennaxis = axisn.index('ant')
                axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                print('Resetting phase', antenna, 'Axis entry number', axisn.index('ant'))
                # print(phase[:,:,antennaid,...])
                if antennaxis == 0:
                    phase[antennaid, ...] = 0.0
                if antennaxis == 1:
                    phase[:, antennaid, ...] = 0.0
                if antennaxis == 2:
                    phase[:, :, antennaid, ...] = 0.0
                if antennaxis == 3:
                    phase[:, :, :, antennaid, ...] = 0.0
                if antennaxis == 4:
                    phase[:, :, :, :, antennaid, ...] = 0.0
                # print(phase[:,:,antennaid,...])
            if hasamps:
                antennaxis = axisn.index('ant')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                print('Resetting amplitude', antenna, 'Axis entry number', axisn.index('ant'))
                if antennaxis == 0:
                    amp[antennaid, ...] = 1.0
                if antennaxis == 1:
                    amp[:, antennaid, ...] = 1.0
                if antennaxis == 2:
                    amp[:, :, antennaid, ...] = 1.0
                if antennaxis == 3:
                    amp[:, :, :, antennaid, ...] = 1.0
                if antennaxis == 4:
                    amp[:, :, :, :, antennaid, ...] = 1.0
                if fulljones:
                    print('pol entry axis:', axisn.index('pol'))
                    if len(axisn) != axisn.index('pol') + 1:
                        print('Pol-axis not the last enrty, cannot handle this')
                        sys.exit()
                    # hardcoded, assumes pol-axis is last
                    if antennaxis == 0:
                        amp[antennaid, ..., 1] = 0.
                        amp[antennaid, ..., 2] = 0.
                    if antennaxis == 1:
                        amp[:, antennaid, ..., 1] = 0.
                        amp[:, antennaid, ..., 2] = 0.
                    if antennaxis == 2:
                        amp[:, :, antennaid, ..., 1] = 0.
                        amp[:, :, antennaid, ..., 2] = 0.
                    if antennaxis == 3:
                        amp[:, :, :, antennaid, ..., 1] = 0.
                        amp[:, :, :, antennaid, ..., 2] = 0.
                    if antennaxis == 4:
                        amp[:, :, :, :, antennaid, ..., 1] = 0.
                        amp[:, :, :, :, antennaid, ..., 2] = 0.

                    # k = axisn.index('pol')
                    # amp[tuple(slice(None) if j != k else antennaid for j in range(arr.ndim))]
                    # amp[...,1] = 0.0 # XY, assumpe pol is last axis
                    # amp[...,2] = 0.0 # YX, assume pol is last axis

            if hastec:
                antennaxis = axisn.index('ant')
                axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
                print('Resetting TEC', antenna, 'Axis entry number', axisn.index('ant'))
                if antennaxis == 0:
                    tec[antennaid, ...] = 0.0
                if antennaxis == 1:
                    tec[:, antennaid, ...] = 0.0
                if antennaxis == 2:
                    tec[:, :, antennaid, ...] = 0.0
                if antennaxis == 3:
                    tec[:, :, :, antennaid, ...] = 0.0
                if antennaxis == 4:
                    tec[:, :, :, :, antennaid, ...] = 0.0
            if hasrotation:
                antennaxis = axisn.index('ant')
                axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
                print('Resetting rotation', antenna, 'Axis entry number', axisn.index('ant'))
                if antennaxis == 0:
                    rotation[antennaid, ...] = 0.0
                if antennaxis == 1:
                    rotation[:, antennaid, ...] = 0.0
                if antennaxis == 2:
                    rotation[:, :, antennaid, ...] = 0.0
                if antennaxis == 3:
                    rotation[:, :, :, antennaid, ...] = 0.0
                if antennaxis == 4:
                    rotation[:, :, :, :, antennaid, ...] = 0.0
            if hasrotationmeasure:
                antennaxis = axisn.index('ant')
                axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
                print('Resetting faradayrotation', antenna, 'Axis entry number', axisn.index('ant'))
                if antennaxis == 0:
                    faradayrotation[antennaid, ...] = 0.0
                if antennaxis == 1:
                    faradayrotation[:, antennaid, ...] = 0.0
                if antennaxis == 2:
                    faradayrotation[:, :, antennaid, ...] = 0.0
                if antennaxis == 3:
                    faradayrotation[:, :, :, antennaid, ...] = 0.0
                if antennaxis == 4:
                    faradayrotation[:, :, :, :, antennaid, ...] = 0.0

    # fill values back in
    if hasphase:
        H.root.sol000.phase000.val[:] = np.copy(phase)
    if hasamps:
        H.root.sol000.amplitude000.val[:] = np.copy(amp)
    if hastec:
        H.root.sol000.tec000.val[:] = np.copy(tec)
    if hasrotation:
        H.root.sol000.rotation000.val[:] = np.copy(rotation)
    if hasrotationmeasure:
        H.root.sol000.rotationmeasure000.val[:] = np.copy(faradayrotation)

    H.flush()
    H.close()
    return


def check_soltabs(h5parm):
    """
    Check the presence of various solution types in an h5 file.
    Returns if phase, amplitude, tec, and rotation are in h5.
    """
    hasphase = hasamps = hasrotation = hastec = hasrotationmeasure = False

    with tables.open_file(h5parm) as H:
        soltabs = list(H.root.sol000._v_children.keys())
    for s in soltabs:
        if 'amplitude' in s:
            hasamps = True
        elif 'phase' in s:
            hasphase = True
        elif 'tec' in s:
            hastec = True
        elif 'rotation' in s:
            hasrotation = True
        elif 'rotationmeasure' in s:
            hasrotationmeasure = True

    return hasphase, hasamps, hasrotation, hastec, hasrotationmeasure


def resetsolsfordir(h5parm, dirlist, refant=None):
    """ Reset solutions for directions (DDE solves only)

    Args:
      h5parm: h5parm file
      dirlist: list of direction_id to reset
      refant: reference antenna
    """
    print(h5parm, dirlist)
    fulljones = fulljonesparmdb(h5parm)
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure = check_soltabs(h5parm)

    # in case refant is None but h5 still has phase
    # this can happen with a scalaramplitude and soltypelist_includedir is used
    # in this case we have pertubative direction
    # in this case h5_merger has already been run which created a phase000 entry
    if refant is None and hasphase:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create rotation000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotation:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create hasrotationmeasure000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotationmeasure:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    # should not be needed as h5_merger does not create tec000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hastec:
        refant = findrefant_core(h5parm)
        force_close(h5parm)

    H = tables.open_file(h5parm, mode='r+')

    if hasamps:
        directions = H.root.sol000.amplitude000.dir[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')

    elif hasphase:
        directions = H.root.sol000.phase000.dir[:]
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')

    elif hastec:
        directions = H.root.sol000.tec000.dir[:]
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')

    elif hasrotation:
        directions = H.root.sol000.rotation000.dir[:]
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')

    elif hasrotationmeasure:
        directions = H.root.sol000.rotationmeasure000.dir[:]
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')

    if hasamps:
        amp = H.root.sol000.amplitude000.val[:]
    if hasphase:  # also phasereference
        phase = H.root.sol000.phase000.val[:]
        refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
        print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            phasen = phase - phase[refant_idx[0], ...]
        if antennaxis == 1:
            phasen = phase - phase[:, refant_idx[0], ...]
        if antennaxis == 2:
            phasen = phase - phase[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            phasen = phase - phase[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            phasen = phase - phase[:, :, :, :, refant_idx[0], ...]
        phase = np.copy(phasen)

    if hastec:
        tec = H.root.sol000.tec000.val[:]
        refant_idx = np.where(H.root.sol000.tec000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
        print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            tecn = tec - tec[refant_idx[0], ...]
        if antennaxis == 1:
            tecn = tec - tec[:, refant_idx[0], ...]
        if antennaxis == 2:
            tecn = tec - tec[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            tecn = tec - tec[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            tecn = tec - tec[:, :, :, :, refant_idx[0], ...]
        tec = np.copy(tecn)

    if hasrotation:
        rotation = H.root.sol000.rotation000.val[:]
        refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
        print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            rotationn = rotation - rotation[refant_idx[0], ...]
        if antennaxis == 1:
            rotationn = rotation - rotation[:, refant_idx[0], ...]
        if antennaxis == 2:
            rotationn = rotation - rotation[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            rotationn = rotation - rotation[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            rotationn = rotation - rotation[:, :, :, :, refant_idx[0], ...]
        rotation = np.copy(rotationn)

    if hasrotationmeasure:
        faradayrotation = H.root.sol000.rotationmeasure000.val[:]
        refant_idx = np.where(H.root.sol000.rotationmeasure000.ant[:].astype(str) == refant)  # to deal with byte strings
        print(refant_idx, refant)
        antennaxis = axisn.index('ant')
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
        print('Referencing faradayrotation to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            faradayrotationn = faradayrotation - faradayrotation[refant_idx[0], ...]
        if antennaxis == 1:
            faradayrotationn = faradayrotation - faradayrotation[:, refant_idx[0], ...]
        if antennaxis == 2:
            faradayrotationn = faradayrotation - faradayrotation[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            faradayrotationn = faradayrotation - faradayrotation[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            faradayrotationn = faradayrotation - faradayrotation[:, :, :, :, refant_idx[0], ...]
        faradayrotation = np.copy(faradayrotationn)

    for directionid, direction in enumerate(directions.astype(str)):  # to deal with byte formatted array
        # if not isinstance(antenna, str):
        #  antenna_str = antenna.decode() # to deal with byte formatted antenna names
        # else:
        #  antenna_str = antenna # already str type

        print(directionid, direction, hasphase, hasamps, hastec, hasrotation)
        if directionid in dirlist:  # in this case reset value to 0.0 (or 1.0)
            if hasphase:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                print('Resetting phase direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                # print(phase[:,:,directionid,...])
                if diraxis == 0:
                    phase[directionid, ...] = 0.0
                if diraxis == 1:
                    phase[:, directionid, ...] = 0.0
                if diraxis == 2:
                    phase[:, :, directionid, ...] = 0.0
                if diraxis == 3:
                    phase[:, :, :, directionid, ...] = 0.0
                if diraxis == 4:
                    phase[:, :, :, :, directionid, ...] = 0.0
                # print(phase[:,:,directionid,...])
            if hasamps:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                print('Resetting amplitude direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                if diraxis == 0:
                    amp[directionid, ...] = 1.0
                if diraxis == 1:
                    amp[:, directionid, ...] = 1.0
                if diraxis == 2:
                    amp[:, :, directionid, ...] = 1.0
                if diraxis == 3:
                    amp[:, :, :, directionid, ...] = 1.0
                if diraxis == 4:
                    amp[:, :, :, :, directionid, ...] = 1.0
                if fulljones:
                    print('pol entry axis:', axisn.index('pol'))
                    if len(axisn) != axisn.index('pol') + 1:
                        print('Pol-axis not the last enrty, cannot handle this')
                        sys.exit()
                    # hardcoded, assumes pol-axis is last
                    if diraxis == 0:
                        amp[directionid, ..., 1] = 0.
                        amp[directionid, ..., 2] = 0.
                    if diraxis == 1:
                        amp[:, directionid, ..., 1] = 0.
                        amp[:, directionid, ..., 2] = 0.
                    if diraxis == 2:
                        amp[:, :, directionid, ..., 1] = 0.
                        amp[:, :, directionid, ..., 2] = 0.
                    if diraxis == 3:
                        amp[:, :, :, directionid, ..., 1] = 0.
                        amp[:, :, :, directionid, ..., 2] = 0.
                    if diraxis == 4:
                        amp[:, :, :, :, directionid, ..., 1] = 0.
                        amp[:, :, :, :, directionid, ..., 2] = 0.

                    # amp[...,1] = 0.0 # XY, assumpe pol is last axis
                    # amp[...,2] = 0.0 # YX, assume pol is last axis

            if hastec:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
                print('Resetting TEC direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                if diraxis == 0:
                    tec[directionid, ...] = 0.0
                if diraxis == 1:
                    tec[:, directionid, ...] = 0.0
                if diraxis == 2:
                    tec[:, :, directionid, ...] = 0.0
                if diraxis == 3:
                    tec[:, :, :, directionid, ...] = 0.0
                if diraxis == 4:
                    tec[:, :, :, :, directionid, ...] = 0.0
            if hasrotation:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
                print('Resetting rotation direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                if diraxis == 0:
                    rotation[directionid, ...] = 0.0
                if diraxis == 1:
                    rotation[:, directionid, ...] = 0.0
                if diraxis == 2:
                    rotation[:, :, directionid, ...] = 0.0
                if diraxis == 3:
                    rotation[:, :, :, directionid, ...] = 0.0
                if diraxis == 4:
                    rotation[:, :, :, :, directionid, ...] = 0.0

            if hasrotationmeasure:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
                print('Resetting faradayrotation direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                if diraxis == 0:
                    faradayrotation[directionid, ...] = 0.0
                if diraxis == 1:
                    faradayrotation[:, directionid, ...] = 0.0
                if diraxis == 2:
                    faradayrotation[:, :, directionid, ...] = 0.0
                if diraxis == 3:
                    faradayrotation[:, :, :, directionid, ...] = 0.0
                if diraxis == 4:
                    faradayrotation[:, :, :, :, directionid, ...] = 0.0

    # fill values back in
    if hasphase:
        H.root.sol000.phase000.val[:] = np.copy(phase)
    if hasamps:
        H.root.sol000.amplitude000.val[:] = np.copy(amp)
    if hastec:
        H.root.sol000.tec000.val[:] = np.copy(tec)
    if hasrotation:
        H.root.sol000.rotation000.val[:] = np.copy(rotation)
    if hasrotationmeasure:
        H.root.sol000.rotationmeasure000.val[:] = np.copy(faradayrotation)

    H.flush()
    H.close()
    return


def radec_to_xyz(ra, dec, time):
    """ Convert ra and dec coordinates to ITRS coordinates for LOFAR observations.

    Args:
        ra (astropy Quantity): right ascension
        dec (astropy Quantity): declination
        time (float): MJD time in seconds
    Returns:
        pointing_xyz (ndarray): NumPy array containing the X, Y and Z coordinates
    """
    obstime = Time(time / 3600 / 24, scale='utc', format='mjd')
    loc_LOFAR = EarthLocation(lon=0.11990128407256424, lat=0.9203091252660295, height=6364618.852935438 * units.m)

    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_altaz = dir_pointing.transform_to(AltAz(obstime=obstime, location=loc_LOFAR))
    dir_pointing_xyz = dir_pointing_altaz.transform_to(ITRS)

    pointing_xyz = np.asarray([dir_pointing_xyz.x, dir_pointing_xyz.y, dir_pointing_xyz.z])
    return pointing_xyz


def losotolofarbeam(parmdb, soltabname, ms, inverse=False, useElementResponse=True, useArrayFactor=True,
                    useChanFreq=True, beamlib='stationresponse'):
    """
    Do the beam correction via this imported losoto operation

    Args:
        parmdb (str): path to the h5parm corrections will be stored in.
        soltabname (str): name of the soltab corrections will be stored in.
        ms (str): path to the MS (used to determine the stations present).
        inverse (bool): calculate the inverse beam correction (i.e. undo the beam).
        useElementResponse (bool): correct for the "element beam" (distance to the tile beam centre).
        useArrayFactor (bool): correct for the "array factor" (sensitivity loss as function of distance to the pointing centre).
        useChanFreq (bool): calculate a beam correction for every channel.
        beamlib (str): beam calculation mode. Can be 'stationresponse' to use the LOFARBeam library (deprecated) or everybeam to use the EveryBeam library.
    """

    H5 = h5parm.h5parm(parmdb, readonly=False)
    soltab = H5.getSolset('sol000').getSoltab(soltabname)
    times = soltab.getAxisValues('time')
    numants = taql('select gcount(*) as numants from ' + ms + '::ANTENNA').getcol('numants')[0]
    H5ants = len(soltab.getAxisValues('ant'))
    if numants != H5ants:
        H5.close()
        raise ValueError('Number of antennas in Measurement Set does not match number of antennas in H5parm.')

    if (beamlib.lower() == 'stationresponse') or (beamlib.lower() == 'lofarbeam'):
        from lofar.stationresponse import stationresponse
        sr = stationresponse(ms, inverse, useElementResponse, useArrayFactor, useChanFreq)

        for vals, coord, selection in soltab.getValuesIter(returnAxes=['ant', 'time', 'pol', 'freq'], weight=False):
            vals = losoto.lib_operations.reorderAxes(vals, soltab.getAxesNames(), ['ant', 'time', 'freq', 'pol'])

            for stationnum in range(numants):
                logger.debug('stationresponse working on station number %i' % stationnum)
                for itime, time in enumerate(times):
                    beam = sr.evaluateStation(time=time, station=stationnum)
                    # Reshape from [nfreq, 2, 2] to [nfreq, 4]
                    beam = beam.reshape(beam.shape[0], 4)

                    if soltab.getAxisLen('pol') == 2:
                        beam = beam[:, [0, 3]]  # get only XX and YY

                    if soltab.getType() == 'amplitude':
                        vals[stationnum, itime, :, :] = np.abs(beam)
                    elif soltab.getType() == 'phase':
                        vals[stationnum, itime, :, :] = np.angle(beam)
                    else:
                        logger.error('Beam prediction works only for amplitude/phase solution tables.')
                        return 1

            vals = losoto.lib_operations.reorderAxes(vals, ['ant', 'time', 'freq', 'pol'],
                                                     [ax for ax in soltab.getAxesNames() if
                                                      ax in ['ant', 'time', 'freq', 'pol']])
            soltab.setValues(vals, selection)
    elif beamlib.lower() == 'everybeam':
        from tqdm import tqdm
        from joblib import Parallel, delayed, parallel_backend
        import dill as pickle
        import psutil
        freqs = soltab.getAxisValues('freq')

        # Obtain direction to calculate beam for.
        dirs = taql('SELECT REFERENCE_DIR,PHASE_DIR FROM {ms:s}::FIELD'.format(ms=ms))
        ra_ref, dec_ref = dirs.getcol('REFERENCE_DIR').squeeze()
        ra, dec = dirs.getcol('PHASE_DIR').squeeze()
        reference_xyz = list(zip(*radec_to_xyz(ra_ref * units.rad, dec_ref * units.rad, times)))
        phase_xyz = list(zip(*radec_to_xyz(ra * units.rad, dec * units.rad, times)))

        for vals, coord, selection in soltab.getValuesIter(returnAxes=['ant', 'time', 'pol', 'freq'], weight=False):
            vals = losoto.lib_operations.reorderAxes(vals, soltab.getAxesNames(), ['ant', 'time', 'freq', 'pol'])
            stationloop = tqdm(range(numants))
            stationloop.set_description('Stations processed: ')
            for stationnum in range(numants):
                stationloop.update()
                logger.debug('everybeam working on station number %i' % stationnum)
                # Parallelise over channels to speed things along.
                with parallel_backend('loky', n_jobs=len(psutil.Process().cpu_affinity())):
                    results = Parallel()(delayed(process_channel_everybeam)(f, stationnum=stationnum,
                                                                            useElementResponse=useElementResponse,
                                                                            useArrayFactor=useArrayFactor,
                                                                            useChanFreq=useChanFreq, ms=ms, freqs=freqs,
                                                                            times=times, ra=ra, dec=dec, ra_ref=ra_ref,
                                                                            dec_ref=dec_ref,
                                                                            reference_xyz=reference_xyz,
                                                                            phase_xyz=phase_xyz) for f in
                                         range(len(freqs)))
                    for freqslot in results:
                        ifreq, beam = freqslot
                        if soltab.getAxisLen('pol') == 2:
                            beam = beam.reshape((beam.shape[0], 4))[:, [0, 3]]  # get only XX and YY
                        if soltab.getType() == 'amplitude':
                            vals[stationnum, :, ifreq, :] = np.abs(beam)
                        elif soltab.getType() == 'phase':
                            vals[stationnum, :, ifreq, :] = np.angle(beam)
                        else:
                            logger.error('Beam prediction works only for amplitude/phase solution tables.')
                            return 1
            vals = losoto.lib_operations.reorderAxes(vals, ['ant', 'time', 'freq', 'pol'],
                                                     [ax for ax in soltab.getAxesNames() if
                                                      ax in ['ant', 'time', 'freq', 'pol']])
            soltab.setValues(vals, selection)

    else:
        H5.close()
        raise ValueError('Unsupported beam library specified')

    H5.close()
    return


def process_channel_everybeam(ifreq, stationnum, useElementResponse, useArrayFactor, useChanFreq, ms, freqs, times, ra,
                              dec, ra_ref, dec_ref, reference_xyz, phase_xyz):
    import everybeam
    if useElementResponse and useArrayFactor:
        # print('Full (element+array_factor) beam correction requested. Using use_differential_beam=False.')
        obs = everybeam.load_telescope(ms, use_differential_beam=False, use_channel_frequency=useChanFreq)
    elif not useElementResponse and useArrayFactor:
        # print('Array factor beam correction requested. Using use_differential_beam=True.')
        obs = everybeam.load_telescope(ms, use_differential_beam=True, use_channel_frequency=useChanFreq)
    elif useElementResponse and not useArrayFactor:
        # print('Element beam correction requested.')
        # Not sure how to do this with EveryBeam.
        raise NotImplementedError('Element beam only correction is not implemented in facetselfcal.')

    # print(f'Processing channel {ifreq}')
    freq = freqs[ifreq]
    timeslices = np.empty((len(times), 2, 2), dtype=np.complex128)
    for itime, time in enumerate(times):
        # timeloop.update()
        if not useElementResponse and useArrayFactor:
            # Array-factor-only correction.
            beam = obs.array_factor(times[itime], stationnum, freq, phase_xyz[itime], reference_xyz[itime])
        else:
            beam = obs.station_response(time=time, station_idx=stationnum, freq=freq, ra=ra, dec=dec)
        # beam = beam.reshape(4)
        timeslices[itime] = beam
    return ifreq, timeslices


# losotolofarbeam('P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive_templatejones.h5', 'amplitude000', 'P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive', inverse=False, useElementResponse=False, useArrayFactor=True, useChanFreq=True)

def set_MeerKAT_bandpass_skymodel(ms):
    """
    Determines and sets the appropriate MeerKAT bandpass calibrator skymodel for a given Measurement Set (MS) file.
    The function inspects the frequency band of the MS and the field pointing direction to select a matching skymodel
    for one of the supported calibrators (J0408-6545 or J1939-6342) in either UHF or L-band. S-band is not supported.
    Raises an exception if no matching skymodel is found or if the band is S-band.
        ms (str): Path to the Measurement Set (MS) file.
    Returns:
        str: Path to the selected skymodel file.
    Raises:
        Exception: If the frequency band is S-band or if no matching skymodel is found.
    """
    skymodpath = '/'.join(datapath.split('/')[0:-1])+'/facetselfcal/data'
    
    skymodel = None
    cJ0408_6545 = np.array([1.08358621, -1.1475981])
    cJ1939_6342 = np.array([5.1461782, -1.11199629])

    with table(ms + '/SPECTRAL_WINDOW', ack=False) as t:
        midfreq = np.mean(t.getcol('CHAN_FREQ')[0])

    UHF = False; Lband = False; Sband = False
    if (midfreq > 500e6) and (midfreq < 1.0e9):  # UHF-band
        UHF = True
    if (midfreq >= 1.0e9) and (midfreq < 1.7e9):  # L-band
        Lband = True
    if (midfreq >= 1.7e9) and (midfreq < 4.0e9):  # S-band
        Sband = True

    with table(ms + '/FIELD', ack=False) as t:
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
    print('skymodel is set to: ', skymodel)
    return skymodel

def cleanup(mslist):
    """ Clean up directory

    Args:
        mslist: list with MS files
    """
    for ms in mslist:
        os.system('rm -rf ' + ms)

    os.system('rm -f *first-residual.fits')
    os.system('rm -f *psf.fits')
    os.system('rm -f *-00*-*.fits')
    os.system('rm -f *dirty.fits')
    os.system('rm -f solintimage*model.fits')
    os.system('rm -f solintimage*residual.fits')
    os.system('rm -f *pybdsf.log')
    return


def flagms_startend(ms, tecsolsfile, tecsolint):
    """

    Args:
        ms: measurement set
        tecsolsfile: solution file with TEC
        tecsolint:
        example of taql command: taql ' select from test.ms where TIME in (select distinct TIME from test.ms offset 0 limit 1798) giving test.ms.cut as plain'
    """

    taql = 'taql'

    msout = ms + '.cut'

    H5 = h5parm.h5parm(tecsolsfile)
    tec = H5.getSolset('sol000').getSoltab('tec000').getValues()
    tecvals = tec[0]

    axis_names = H5.getSolset('sol000').getSoltab('tec000').getAxesNames()
    time_ind = axis_names.index('time')
    ant_ind = axis_names.index('ant')

    # ['time', 'ant', 'dir', 'freq']
    reftec = tecvals[:, 0, 0, 0]

    # print np.shape( tecvals[:,:,0,0]), np.shape( reftec[:,None]),
    tecvals = tecvals[:, :, 0, 0] - reftec[:, None]  # reference to zero

    times = tec[1]['time']

    # print tecvals[:,0]

    goodtimesvec = []

    for timeid, time in enumerate(times):
        tecvals[timeid, :]

        # print timeid, np.count_nonzero( tecvals[timeid,:])
        goodtimesvec.append(np.count_nonzero(tecvals[timeid, :]))

    goodstartid = np.argmax(np.array(goodtimesvec) > 0)
    goodendid = len(goodtimesvec) - np.argmax(np.array(goodtimesvec[::-1]) > 0)

    print('First good solutionslot,', goodstartid, ' out of', len(goodtimesvec))
    print('Last good solutionslot,', goodendid, ' out of', len(goodtimesvec))
    H5.close()

    if (goodstartid != 0) or (goodendid != len(goodtimesvec)):  # only do if needed to save some time
        cmd = taql + " ' select from " + ms + " where TIME in (select distinct TIME from " + ms
        cmd += " offset " + str(goodstartid * int(tecsolint))
        cmd += " limit " + str((goodendid - goodstartid) * int(tecsolint)) + ") giving "
        cmd += msout + " as plain'"

        print(cmd)
        run(cmd)

        os.system('rm -rf ' + ms)
        time.sleep(2)  # give some time to remove the MS
        os.system('mv ' + msout + ' ' + ms)
    return


# flagms_startend('P215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archive','phaseonlyP215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archivesolsgrid_9.h5', 2)


def removestartendms(ms, starttime=None, endtime=None, dysco=True, metadata_compression=True):
    """
    Removes the start and/or end times from a Measurement Set (MS) and processes it using DP3.

    This function creates a new MS with the specified time range removed and optionally applies
    DYSCO compression. It also creates a WEIGHT_SPECTRUM_SOLVE column based on the processed data.

    Args:
        ms (str): The path to the input Measurement Set (MS).
        starttime (str, optional): The start time to cut from the MS in a format recognized by DP3.
                                   If None, no start time is specified. Defaults to None.
        endtime (str, optional): The end time to cut from the MS in a format recognized by DP3.
                                 If None, no end time is specified. Defaults to None.
        dysco (bool, optional): Whether to use DYSCO compression for the output MS. Defaults to True.

    Returns:
        None

    Side Effects:
        - Creates a new MS with the '.cut' suffix.
        - Temporarily creates a '.cuttmp' MS, which is removed after processing.
        - Adds a new column 'WEIGHT_SPECTRUM_SOLVE' to the '.cut' MS.
        - Prints the DP3 commands executed.
        - Removes any pre-existing '.cut' or '.cuttmp' directories.

    Notes:
        - The function uses the DP3 tool for processing the MS.
        - The `check_phaseup_station` function is used to determine if UVW compression should be disabled.
        - The `run` function is used to execute the DP3 commands.
        - The `table` function from the casacore library is used to manipulate the MS columns.
    """
    # chdeck if output is already there and remove
    if os.path.isdir(ms + '.cut'):
        os.system('rm -rf ' + ms + '.cut')
    if os.path.isdir(ms + '.cuttmp'):
        os.system('rm -rf ' + ms + '.cuttmp')
    time.sleep(2)  # give some time to remove the MS

    cmd = 'DP3 msin=' + ms + ' ' + 'msout=' + ms + '.cut '
    if check_phaseup_station(ms): cmd += 'msout.uvwcompression=False '
    
    if not metadata_compression:
        cmd += 'msout.uvwcompression=False '
        cmd += 'msout.antennacompression=False '
        cmd += 'msout.scalarflags=False '
    
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM steps=[] '
    if starttime is not None:
        cmd += 'msin.starttime=' + starttime + ' '
    if endtime is not None:
        cmd += 'msin.endtime=' + endtime + ' '
    print(cmd)
    run(cmd)

    cmd = 'DP3 msin=' + ms + ' ' + 'msout=' + ms + '.cuttmp '
    if check_phaseup_station(ms): cmd += 'msout.uvwcompression=False '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM_SOLVE steps=[] '
    if starttime is not None:
        cmd += 'msin.starttime=' + starttime + ' '
    if endtime is not None:
        cmd += 'msin.endtime=' + endtime + ' '
    print(cmd)
    run(cmd)

    # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
    print('Adding WEIGHT_SPECTRUM_SOLVE')
    with table(ms + '.cut', readonly=False) as t:
        #desc = t.getcoldesc('WEIGHT_SPECTRUM')
        #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
        #t.addcols(desc)
        addcol(t, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

        with table(ms + '.cuttmp', readonly=True) as t2:
            imweights = t2.getcol('WEIGHT_SPECTRUM')
        t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

    # clean up
    os.system('rm -rf ' + ms + '.cuttmp')

    return

# removestartendms('P219+50_PSZ2G084.10+58.72.dysco.sub.shift.avg.weights.ms.archive',endtime='16-Apr-2015/02:14:47.0')
# removestartendms('P223+50_PSZ2G084.10+58.72.dysco.sub.shift.avg.weights.ms.archive',starttime='24-Feb-2015/22:16:00.0')
# removestartendms('P223+52_PSZ2G088.98+55.07.dysco.sub.shift.avg.weights.ms.archive',starttime='19-Feb-2015/22:40:00.0')
# removestartendms('P223+55_PSZ2G096.14+56.24.dysco.sub.shift.avg.weights.ms.archive',starttime='31-Mar-2015/20:11:00.0')
# removestartendms('P227+53_PSZ2G088.98+55.07.dysco.sub.shift.avg.weights.ms.archive',starttime='19-Feb-2015/22:40:00.0')

def which(file_name):
    for path in os.environ["PATH"].split(os.pathsep):
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return None


def archive(mslist, outtarname, regionfile, fitsmask, imagename, dysco=True, mergedh5_i=None, 
            facetregionfile=None, metadata_compression=True): 
    """
    Archives calibrated measurement sets and related files into a tarball.

    This function processes a list of measurement sets (MS), applies calibration and optional compression,
    and then archives the resulting files along with specified auxiliary files into a compressed tarball.

    Args:
        mslist (list of str): List of measurement set filenames to process and archive.
        outtarname (str): Name of the output tarball file.
        regionfile (str or None): Path to a region file to include in the archive, if it exists.
        fitsmask (str or None): Path to a FITS mask file to include in the archive, if it exists.
        imagename (str): Name of the image file to include in the archive.
        dysco (bool, optional): Whether to use Dysco compression for output measurement sets. Defaults to True.
        mergedh5_i (list of str or None, optional): List of merged HDF5 files to include in the archive, if provided.
        facetregionfile (str or None, optional): Path to a facet region file to include in the archive, if it exists.
        metadata_compression (bool, optional): Whether to enable metadata compression. Defaults to True.

    Returns:
        None

    Side Effects:
        - Creates calibrated copies of the input measurement sets with optional compression.
        - Removes any existing output tarball with the same name before creating a new one.
        - Archives specified files into a compressed tarball.
        - Removes temporary calibrated measurement sets after archiving.
        - Logs the creation of the tarball.

    Raises:
        None explicitly, but may raise exceptions if external commands fail.
    """
    path = '/disks/ftphome/pub/vanweeren'
    for ms in mslist:
        msout = ms + '.calibrated'
        if os.path.isdir(msout):
            os.system('rm -rf ' + msout)
            time.sleep(2)  # give it some time to remove
        cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msout=' + msout + ' '
        if check_phaseup_station(ms): cmd += 'msout.uvwcompression=False '
        cmd += 'msin.datacolumn=CORRECTED_DATA steps=[] '
        if dysco:
            cmd += 'msout.storagemanager=dysco '
            cmd += 'msout.storagemanager.weightbitrate=16 '
        if not metadata_compression:
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '
        run(cmd)

    msliststring = ' '.join(map(str, glob.glob('*.calibrated')))
    cmd = 'tar -zcf ' + outtarname + ' ' + msliststring + ' selfcal.log ' + imagename + ' '

    if fitsmask is not None:  # add fitsmask to tar if it exists
        if os.path.isfile(fitsmask):
            cmd += fitsmask + ' '

    if regionfile is not None:  # add box regionfile to tar if it exists
        if os.path.isfile(regionfile):
            cmd += regionfile + ' '

    if mergedh5_i is not None:
        mergedh5_i_string = ' '.join(map(str, mergedh5_i))
        cmd += mergedh5_i_string + ' '

    if facetregionfile is not None:  # add facet region file to tar if it exists
        if os.path.isfile(facetregionfile):
            cmd += facetregionfile + ' '

    if os.path.isfile(outtarname):
        os.system('rm -f ' + outtarname)
    logger.info('Creating archived calibrated tarball: ' + outtarname)
    run(cmd)

    for ms in mslist:
        msout = ms + '.calibrated'
        os.system('rm -rf ' + msout)
    return


def setinitial_solint(mslist, options):
    """
    take user input solutions,nchan,smoothnessconstraint,antennaconstraint and expand them to all ms
    these list can then be updated later with values from auto_determinesolints for example
    """

    nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
        smoothnessspectralexponent_list, smoothnessrefdistance_list, antennaconstraint_list, resetsols_list, \
        resetdir_list, normamps_list, soltypecycles_list, uvmin_list, uvmax_list, uvminim_list, uvmaxim_list, \
        solve_msinnchan_list, solve_msinstartchan_list, antenna_averaging_factors_list, \
        antenna_smoothness_factors_list = \
        ([] for _ in range(20))

    # make here uvminim_list and uvmaxim_list, because the have just the length of mslist
    for ms_id, ms in enumerate(mslist):
        try:
            uvminim = options.uvminim[ms_id]
        except:
            uvminim = options.uvminim  # apparently we just have a float and not a list
        uvminim_list.append(uvminim)
    for ms_id, ms in enumerate(mslist):
        try:
            uvmaxim = options.uvmaxim[ms_id]
        except:
            uvmaxim = options.uvmaxim  # apparently we just have a float and not a list
        uvmaxim_list.append(uvmaxim)

    for soltype_id, soltype in enumerate(options.soltype_list):
        nchan_ms = []  # list with len(mslist)
        solint_ms = []  # list with len(mslist)
        antennaconstraint_list_ms = []  # list with len(mslist)
        resetsols_list_ms = []  # list with len(mslist)
        resetdir_list_ms = []  # list with len(mslist)
        normamps_list_ms = []  # list with len(mslist)
        smoothnessconstraint_list_ms = []  # list with len(mslist)
        smoothnessreffrequency_list_ms = []  # list with len(mslist)
        smoothnessspectralexponent_list_ms = []  # list with len(mslist)
        smoothnessrefdistance_list_ms = []  # list with len(mslist)
        BLsmooth_list_ms = []  # list with len(mslist)
        soltypecycles_list_ms = []  # list with len(mslist)
        uvmin_list_ms = []  # list with len(mslist)
        uvmax_list_ms = []  # list with len(mslist)
        # uvminim_list_ms = []  # list with len(mslist)
        # uvmaxim_list_ms = []  # list with len(mslist)
        solve_msinnchan_list_ms = []  # list with len(mslist)
        solve_msinstartchan_list_ms = []  # list with len(mslist)
        antenna_averaging_factors_list_ms = []  # list with len(mslist)
        antenna_smoothness_factors_list_ms = []  # list with len(mslist)

        for ms in mslist:
            # use try statement in case the user did not provide all the info for certain soltypes

            # solint
            try:
                solint = options.solint_list[soltype_id]
            except:
                solint = 1

            # nchan
            try:
                nchan = options.nchan_list[soltype_id]
            except:
                nchan = 1  # if nothing is set use 1

            # smoothnessconstraint
            try:
                smoothnessconstraint = options.smoothnessconstraint_list[soltype_id]
            except:
                smoothnessconstraint = 0.0

            # BLsmooth
            try:
                BLsmooth = options.BLsmooth_list[soltype_id]
            except:
                BLsmooth = False

            # smoothnessreffrequency
            try:
                smoothnessreffrequency = options.smoothnessreffrequency_list[soltype_id]
            except:
                smoothnessreffrequency = 0.0

            # smoothnessspectralexponent
            try:
                smoothnessspectralexponent = options.smoothnessspectralexponent_list[soltype_id]
            except:
                smoothnessspectralexponent = -1.0

            # smoothnessrefdistance
            try:
                smoothnessrefdistance = options.smoothnessrefdistance_list[soltype_id]
            except:
                smoothnessrefdistance = 0.0

            # antennaconstraint
            try:
                antennaconstraint = options.antennaconstraint_list[soltype_id]
            except:
                antennaconstraint = None

            # resetsols
            try:
                resetsols = options.resetsols_list[soltype_id]
            except:
                resetsols = None

            # resetdir
            try:
                resetdir = options.resetdir_list[soltype_id]
            except:
                resetdir = None

            # normamps
            try:
                normamps = options.normamps_list[soltype_id]
            except:
                normamps = 'normamps'

            # uvmin
            try:
                uvmin = options.uvmin[soltype_id]
            except:
                uvmin = None

            # uvmax
            try:
                uvmax = options.uvmax[soltype_id]
            except:
                uvmax = None
            
            # solve_msinnchan
            try:
                solve_msinnchan = options.solve_msinnchan_list[soltype_id]
            except:
                solve_msinnchan = 'all'
            
            # solve_msinstartchan
            try:
                solve_msinstartchan = options.solve_msinstartchan_list[soltype_id]
            except:
                solve_msinstartchan = 0

            # antenna_averaging_factors
            try:
                antenna_averaging_factors = options.antenna_averaging_factors_list[soltype_id]
            except:
                antenna_averaging_factors = None    
            
            # antenna_smoothness_factors
            try:
                antenna_smoothness_factors = options.antenna_smoothness_factors_list[soltype_id]
            except:
                antenna_smoothness_factors = None

            # uvminim
            # try:
            #  uvminim = options.uvminim[soltype_id]
            # except:
            #  uvminim = 80.

            # uvmaxim
            # try:
            #  uvmaxim = options.uvmaxim[soltype_id]
            # except:
            #  uvmaxim = None

            # soltypecycles
            soltypecycles = options.soltypecycles_list[soltype_id]

            # force nchan 1 for tec(andphase) solve and in case smoothnessconstraint is invoked
            # if soltype == 'tec' or  soltype == 'tecandphase' or smoothnessconstraint > 0.0:
            if soltype == 'tec' or soltype == 'tecandphase':
                nchan = 1

            nchan_ms.append(nchan)
            solint_ms.append(solint)
            smoothnessconstraint_list_ms.append(smoothnessconstraint)
            BLsmooth_list_ms.append(BLsmooth)
            smoothnessreffrequency_list_ms.append(smoothnessreffrequency)
            smoothnessspectralexponent_list_ms.append(smoothnessspectralexponent)
            smoothnessrefdistance_list_ms.append(smoothnessrefdistance)
            antennaconstraint_list_ms.append(antennaconstraint)
            resetsols_list_ms.append(resetsols)
            resetdir_list_ms.append(resetdir)
            normamps_list_ms.append(normamps)

            soltypecycles_list_ms.append(soltypecycles)
            uvmin_list_ms.append(uvmin)
            uvmax_list_ms.append(uvmax)
            # uvminim_list_ms.append(uvminim)
            # uvmaxim_list_ms.append(uvmaxim)
            solve_msinnchan_list_ms.append(solve_msinnchan)
            solve_msinstartchan_list_ms.append(solve_msinstartchan)
            antenna_averaging_factors_list_ms.append(antenna_averaging_factors)
            antenna_smoothness_factors_list_ms.append(antenna_smoothness_factors)

        nchan_list.append(nchan_ms)  # list of lists
        solint_list.append(solint_ms)  # list of lists
        antennaconstraint_list.append(antennaconstraint_list_ms)  # list of lists
        resetsols_list.append(resetsols_list_ms)  # list of lists
        resetdir_list.append(resetdir_list_ms)  # list of lists
        normamps_list.append(normamps_list_ms)  # list of lists
        BLsmooth_list.append(BLsmooth_list_ms)  # list of lists
        smoothnessconstraint_list.append(smoothnessconstraint_list_ms)  # list of lists
        smoothnessreffrequency_list.append(smoothnessreffrequency_list_ms)  # list of lists
        smoothnessspectralexponent_list.append(smoothnessspectralexponent_list_ms)  # list of lists
        smoothnessrefdistance_list.append(smoothnessrefdistance_list_ms)
        uvmin_list.append(uvmin_list_ms)  # list of lists
        uvmax_list.append(uvmax_list_ms)  # list of lists
        # uvminim_list.append(uvminim_list_ms) # list of lists
        # uvmaxim_list.append(uvmaxim_list_ms)      # list of lists
        solve_msinnchan_list.append(solve_msinnchan_list_ms)  # list of lists
        solve_msinstartchan_list.append(solve_msinstartchan_list_ms)  # list of lists
        antenna_averaging_factors_list.append(antenna_averaging_factors_list_ms)  # list of lists
        antenna_smoothness_factors_list.append(antenna_smoothness_factors_list_ms)  # list of lists

        soltypecycles_list.append(soltypecycles_list_ms)

    print('soltype:', options.soltype_list, mslist)
    print('nchan:', nchan_list)
    print('solint:', solint_list)
    print('BLsmooth:', BLsmooth_list)
    print('smoothnessconstraint:', smoothnessconstraint_list)
    print('smoothnessreffrequency:', smoothnessreffrequency_list)
    print('smoothnessspectralexponent:', smoothnessspectralexponent_list)
    print('smoothnessrefdistance:', smoothnessrefdistance_list)
    print('antennaconstraint:', antennaconstraint_list)
    print('resetsols:', resetsols_list)
    print('resetdir:', resetdir_list)
    print('normamps:', normamps_list)
    print('soltypecycles:', soltypecycles_list)
    print('uvmin:', uvmin_list)
    print('uvmax:', uvmax_list)
    print('uvminim:', uvminim_list)
    print('uvmaxim:', uvmaxim_list)
    print('solve_msinnchan:', solve_msinnchan_list)
    print('solve_msinstartchan:', solve_msinstartchan_list)
    print('antenna_averaging_factors:', antenna_averaging_factors_list)
    print('antenna_smoothness_factors:', antenna_smoothness_factors_list)

    logger.info('soltype: ' + str(options.soltype_list) + ' ' + str(mslist))
    logger.info('nchan: ' + str(options.nchan_list))
    logger.info('solint: ' + str(options.solint_list))
    logger.info('BLsmooth: ' + str(options.BLsmooth_list))
    logger.info('smoothnessconstraint: ' + str(options.smoothnessconstraint_list))
    logger.info('smoothnessreffrequency: ' + str(options.smoothnessreffrequency_list))
    logger.info('smoothnessspectralexponent: ' + str(options.smoothnessspectralexponent_list))
    logger.info('smoothnessrefdistance: ' + str(options.smoothnessrefdistance_list))
    logger.info('antennaconstraint: ' + str(options.antennaconstraint_list))
    logger.info('resetsols: ' + str(options.resetsols_list))
    logger.info('resetdir: ' + str(options.resetdir_list))
    logger.info('normamps: ' + str(options.normamps_list))
    logger.info('soltypecycles: ' + str(options.soltypecycles_list))
    logger.info('uvmin: ' + str(options.uvmin))
    logger.info('uvmax: ' + str(options.uvmax))
    logger.info('uvminim: ' + str(options.uvminim))
    logger.info('uvmaxim: ' + str(options.uvmaxim))
    logger.info('solve_msinnchan: ' + str(options.solve_msinnchan_list))
    logger.info('solve_msinstartchan: ' + str(options.solve_msinstartchan_list))
    logger.info('antenna_averaging_factors: ' + str(options.antenna_averaging_factors_list))
    logger.info('antenna_smoothness_factors: ' + str(options.antenna_smoothness_factors_list))

    return nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, \
           smoothnessreffrequency_list, smoothnessspectralexponent_list, \
           smoothnessrefdistance_list, antennaconstraint_list, resetsols_list, \
           resetdir_list, soltypecycles_list, uvmin_list, uvmax_list, uvminim_list, \
           uvmaxim_list, normamps_list, solve_msinnchan_list, \
           solve_msinstartchan_list, antenna_averaging_factors_list, \
           antenna_smoothness_factors_list


def getms_amp_stats(ms, datacolumn='DATA', uvcutfraction=0.666, robustsigma=True):
    uvdismod = get_uvwmax(ms) * uvcutfraction
    t = taql(
        'SELECT ' + datacolumn + ',UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    flags = t.getcol('FLAG')
    data = t.getcol(datacolumn)
    data = np.ma.masked_array(data, flags)
    t.close()

    amps_rr = np.abs(data[:, :, 0])
    amps_ll = np.abs(data[:, :, 3])

    # remove zeros from LL
    idx = np.where(amps_ll != 0.0)
    amps_ll = amps_ll[idx]
    amps_rr = amps_rr[idx]

    # remove zeros from RR
    idx = np.where(amps_rr != 0.0)
    amps_ll = amps_ll[idx]
    amps_rr = amps_rr[idx]

    amplogratio = np.log10(amps_rr / amps_ll)  # we assume Stokes V = 0, so RR = LL
    if robustsigma:
        logampnoise = astropy.stats.sigma_clipping.sigma_clipped_stats(amplogratio)[2]
    else:
        logampnoise = np.std(amplogratio)
    print(ms, logampnoise, np.mean(amplogratio))
    return logampnoise


def getms_phase_stats(ms, datacolumn='DATA', uvcutfraction=0.666):
    uvdismod = get_uvwmax(ms) * uvcutfraction
    t = taql(
        'SELECT ' + datacolumn + ',UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    flags = t.getcol('FLAG')
    data = t.getcol(datacolumn)
    data = np.ma.masked_array(data, flags)
    t.close()

    phase_rr = np.angle(data[:, :, 0])
    phase_ll = np.angle(data[:, :, 3])

    # remove zeros from LL (flagged data)
    idx = np.where(phase_ll != 0.0)
    phase_ll = phase_ll[idx]
    phase_rr = phase_rr[idx]

    # remove zeros from RR (flagged data)
    idx = np.where(phase_rr != 0.0)
    phase_ll = phase_ll[idx]
    phase_rr = phase_rr[idx]

    phasediff = np.mod(phase_rr - phase_ll, 2. * np.pi)
    phasenoise = scipy.stats.circstd(phasediff, nan_policy='omit')

    print(ms, phasenoise, scipy.stats.circmean(phasediff, nan_policy='omit'))
    return phasenoise


def getmsmodelinfo(ms, modelcolumn, fastrms=False, uvcutfraction=0.333):
    t = table(ms + '/SPECTRAL_WINDOW', ack=False)
    chanw = np.median(t.getcol('CHAN_WIDTH'))
    freq = np.median(t.getcol('CHAN_FREQ'))
    nfreq = len(t.getcol('CHAN_FREQ')[0])
    t.close()
    uvdismod = get_uvwmax(ms) * uvcutfraction  # take range [uvcutfraction*uvmax - 1.0uvmax]

    HBA_upfreqsel = 0.75  # select only freqcencies above 75% of the available bandwidth
    freqct = 1000e6
    # the idea is that for HBA if you are far out in the beam the noise gets up much more at the higher freqs and the model flux goes down due to the spectral index. In this way we get more conservative solints

    t = taql(
        'SELECT ' + modelcolumn + ',DATA,UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    model = np.abs(t.getcol(modelcolumn))
    flags = t.getcol('FLAG')
    data = t.getcol('DATA')
    print('Compute visibility noise of the dataset with robust sigma clipping', ms)
    logger.info('Compute visibility noise of the dataset with robust sigma clipping: ' + ms)
    if fastrms:  # take only every fifth element of the array to speed up the computation
        if freq > freqct:  # HBA
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(
                data[0:data.shape[0]:5, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3],
                mask=flags[0:data.shape[0]:5, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3])[2]  # use XY and YX
        else:
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[0:data.shape[0]:5, :, 1:3],
                                                                     mask=flags[0:data.shape[0]:5, :, 1:3])[
                2]  # use XY and YX
    else:
        if freq > freqct:  # HBA
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(
                data[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3],
                mask=flags[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3])[2]  # use XY and YX
        else:
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[:, :, 1:3],
                                                                     mask=flags[:, :, 1:3])[2]  # use XY and YX

    model = np.ma.masked_array(model, flags)
    if freq > freqct:  # HBA:
        flux = np.ma.mean((model[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 0] + model[:, int(np.floor(
            float(nfreq) * HBA_upfreqsel)):-1,
                                                                                         3]) * 0.5)  # average XX and YY (ignore XY and YX, they are zero, or nan, in other words this is Stokes I)
    else:
        flux = np.ma.mean(
            (model[:, :, 0] + model[:, :, 3]) * 0.5)  # average XX and YY (ignore XY and YX, they are zero, or nan)
    time = np.unique(t.getcol('TIME'))
    tint = np.abs(time[1] - time[0])
    print('Integration time visibilities', tint)
    logger.info('Integration time visibilities [s]: ' + str(tint))
    t.close()

    del data, flags, model
    print('Noise visibilities:', noise, 'Jy')
    print('Flux in model:', flux, 'Jy')
    print('UV-selection to compute model flux:', str(uvdismod / 1e3), 'km')
    logger.info('Noise visibilities: ' + str(noise) + 'Jy')
    logger.info('Flux in model: ' + str(flux) + 'Jy')
    logger.info('UV-selection to compute model flux: ' + str(uvdismod / 1e3) + 'km')

    return noise, flux, tint, chanw


def return_soltype_index(soltype_list, soltype, occurence=1, onetectypeoccurence=False):
    """
    Returns the index of a specified solution type in a list of solution types.

    Parameters:
        soltype_list (list of str): A list of solution types.
        soltype (str): The solution type to search for.
        occurence (int, optional): The occurrence of the solution type to find. Defaults to 1.
        onetectypeoccurence (bool, optional): If True, treats 'tecandphase' and 'tec' as equivalent. Defaults to False.

    Returns:
        int or None: The index of the specified solution type in the list, or None if not found.
    """
    if onetectypeoccurence:
        if soltype == 'tecandphase' or soltype == 'tec':
            soltype_list = [sol.replace('tecandphase', 'tec') for sol in soltype_list]
            soltype = 'tec'

    sol_index = None
    count = 0
    for sol_id, sol in enumerate(soltype_list):
        if sol == soltype:
            sol_index = sol_id
            count = count + 1
            if occurence == count:
                return sol_index
            else:
                sol_index = None
    return sol_index


def auto_determinesolints(mslist, soltype_list, longbaseline, LBA,
                          innchan_list=None, insolint_list=None,
                          uvdismod=None, modelcolumn='MODEL_DATA', redo=False,
                          inBLsmooth_list=None,
                          insmoothnessconstraint_list=None, insmoothnessreffrequency_list=None,
                          insmoothnessspectralexponent_list=None,
                          insmoothnessrefdistance_list=None,
                          inantennaconstraint_list=None, inresetsols_list=None, inresetdir_list=None,
                          innormamps_list=None,
                          insoltypecycles_list=None, tecfactorsolint=1.0, gainfactorsolint=1.0,
                          gainfactorsmoothness=1.0, phasefactorsolint=1.0, delaycal=False):
    """
    determine the solution time and frequency intervals based on the amount of compact source flux and noise
    """
    # 1 find the first tec/tecandphase in the soltype_list
    # set the solints there based on this code
    # update antennaconstraint if needed

    # find the first (scalar)complexgain
    # compute solints

    # based on scalarcomplexgain determine what to do
    # proceed normally, as is now
    # do an 'all' constrained solve
    # no complexgain solves at all, how to do this? update soltype_list here, or set

    # soltype_list == 'tec'

    for ms_id, ms in enumerate(mslist):
        if longbaseline:
            noise, flux, tint, chanw = getmsmodelinfo(ms, modelcolumn, uvcutfraction=0.075)
        else:
            noise, flux, tint, chanw = getmsmodelinfo(ms, modelcolumn)
        for soltype_id, soltype in enumerate(soltype_list):

            ######## TEC or TECANDPHASE ######
            ######## for first occurence of tec(andphase) #######
            if soltype in ['tec', 'tecandphase'] and \
                    ((soltype_id == return_soltype_index(soltype_list, 'tec', occurence=1, onetectypeoccurence=True)) or \
                     (soltype_id == return_soltype_index(soltype_list, 'tecandphase', occurence=1,
                                                         onetectypeoccurence=True))):

                if LBA:
                    if longbaseline:
                        solint_sf = 3.0e-3 * tecfactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 4.0e-2 * tecfactorsolint  # 0.5e-3 # for tecandphase and coreconstraint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.5e-2 * tecfactorsolint  # for tecandphase, no coreconstraint
                    else:  # for -- HBA dutch --
                        solint_sf = 4.0e-2 * tecfactorsolint  # for tecandphase, no coreconstraint

                if soltype == 'tec':
                    solint_sf = solint_sf / np.sqrt(2.)  # tec and coreconstraint

                # trigger antennaconstraint_phase core if solint > tint
                if not longbaseline and (tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3) > tint):
                    print(tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                    solint_sf = solint_sf / 30.
                    print('Trigger_antennaconstraint core:', soltype, ms)
                    logger.info('Trigger_antennaconstraint core: ' + soltype + ' ' + ms)
                    inantennaconstraint_list[soltype_id][ms_id] = 'core'
                    # do another pertubation, a slow solve of the core stations
                    # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
                    if (tint * solint_sf * ((noise / flux) ** 2) * (
                            chanw / 390.625e3) < 720.0):  # less than 12 min now, also doing
                        inantennaconstraint_list[soltype_id + 1][ms_id] = 'remote'  # or copy over input ??
                        insoltypecycles_list[soltype_id + 1][ms_id] = insoltypecycles_list[soltype_id][
                            ms_id]  # do + 1 here??
                        insolint_list[soltype_id + 1][ms_id] = int(
                            np.rint(10. * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)))
                        if insolint_list[soltype_id + 1][ms_id] < 1:
                            insolint_list[soltype_id + 1][ms_id] = 1
                    else:
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None  # or copy over input
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                # round to nearest integer
                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                # frequency scaling is need because if we avearge in freqeuncy the solint should not change for a tec(andphase) solve

                if (longbaseline) and (not LBA) and (soltype == 'tec') \
                        and (soltype_list[1] == 'tecandphase'):
                    if solint < 0.5 and (solint * tint < 16.):  # so less then 16 sec
                        print('Longbaselines bright source detected: changing from tec to tecandphase solve')
                        insoltypecycles_list[soltype_id][ms_id] = 999
                        insoltypecycles_list[1][ms_id] = 0

                if solint < 1:
                    solint = 1
                if (float(solint) * tint / 3600.) > 0.5:  # so check if larger than 30 min
                    print('Warning, it seems there is not enough flux density on the longer baselines for solving')
                    logger.warning(
                        'Warning, it seems there is not enough flux density on the longer baselines for solving')
                    solint = np.rint(0.5 * 3600. / tint)  # max is 30 min

                print(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3), 'Using tec(andphase) solint:', solint)
                logger.info(str(solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) + '-- Using tec(andphase) solint:' + str(solint))
                print('Using tec(andphase) solint [s]:', float(solint) * tint)
                logger.info('Using tec(andphase) solint [s]: ' + str(float(solint) * tint))

                insolint_list[soltype_id][ms_id] = int(solint)
                innchan_list[soltype_id][ms_id] = 1

            ######## SCALARPHASE or PHASEONLY ######
            ######## for first occurence of tec(andphase) #######
            if soltype in ['scalarphase', 'phaseonly'] and \
                    (insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
                    ((soltype_id == return_soltype_index(soltype_list, 'scalarphase', occurence=1,
                                                         onetectypeoccurence=True)) or
                     (soltype_id == return_soltype_index(soltype_list, 'phaseonly', occurence=1,
                                                         onetectypeoccurence=True))):

                if LBA:
                    if longbaseline:
                        solint_sf = 3.0e-3 * phasefactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 4.0e-2 * phasefactorsolint  # 0.5e-3 # for tecandphase and coreconstraint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.5e-2 * phasefactorsolint  # for tecandphase, no coreconstraint
                    else:  # for -- HBA dutch --
                        solint_sf = 4.0e-2 * phasefactorsolint  # for tecandphase, no coreconstraint

                if soltype == 'scalarphase':
                    solint_sf = solint_sf / np.sqrt(2.)  # decrease solint if scalarphase

                # trigger antennaconstraint_phase core if solint > tint
                # needs checking, this might be wrong, this assumes we use [scalarphase/phaseonly,scalarphase/phaseonly, (scalar)complexgain] so 3 steps.....
                if not longbaseline and (tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3) > tint):
                    print(tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                    solint_sf = solint_sf / 30.
                    print('Trigger_antennaconstraint core:', soltype, ms)
                    logger.info('Trigger_antennaconstraint core: ' + soltype + ' ' + ms)
                    inantennaconstraint_list[soltype_id][ms_id] = 'core'
                    # do another pertubation, a slow solve of the core stations
                    # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
                    if (tint * solint_sf * ((noise / flux) ** 2) * (
                            chanw / 390.625e3) < 720.0):  # less than 12 min now, also doing
                        inantennaconstraint_list[soltype_id + 1][ms_id] = 'remote'  # or copy over input ??
                        insoltypecycles_list[soltype_id + 1][ms_id] = insoltypecycles_list[soltype_id][
                            ms_id]  # do + 1 here??
                        insolint_list[soltype_id + 1][ms_id] = int(
                            np.rint(10. * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)))
                        if insolint_list[soltype_id + 1][ms_id] < 1:
                            insolint_list[soltype_id + 1][ms_id] = 1
                    else:
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None  # or copy over input

                # round to nearest integer
                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                # frequency scaling is needed because if we avearge in freqeuncy the solint should not change for a (scalar)phase solve with smoothnessconstraint

                if solint < 1:
                    solint = 1
                if (float(solint) * tint / 3600.) > 0.5:  # so check if larger than 30 min
                    print('Warning, it seems there is not enough flux density on the longer baselines for solving')
                    logger.warning(
                        'Warning, it seems there is not enough flux density on the longer baselines for solving')
                    solint = np.rint(0.5 * 3600. / tint)  # max is 30 min

                print(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3), 'Using (scalar)phase solint:', solint)
                logger.info(str(solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) + '-- Using (scalar)phase solint:' + str(solint))
                print('Using (scalar)phase solint [s]:', float(solint) * tint)
                logger.info('Using (scalar)phase solint [s]: ' + str(float(solint) * tint))

                insolint_list[soltype_id][ms_id] = int(solint)
                innchan_list[soltype_id][ms_id] = 1  # because we use smoothnessconstraint

            ######## COMPLEXGAIN or SCALARCOMPLEXGAIN or AMPLITUDEONLY or SCALARAMPLITUDE ######
            # requires smoothnessconstraint
            # for first occurence of (scalar)complexgain
            print(insmoothnessconstraint_list, soltype_id, ms_id)
            if soltype in ['complexgain', 'scalarcomplexgain'] and (
                    insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
                    ((soltype_id == return_soltype_index(soltype_list, 'complexgain', occurence=1)) or
                     (soltype_id == return_soltype_index(soltype_list, 'scalarcomplexgain', occurence=1))):

                if longbaseline:
                    thr_disable_gain = 24.  # 32. #  72.
                else:
                    thr_disable_gain = 64.  # 32. #  72.

                thr_SM15Mhz = 1.5
                thr_gain_trigger_allantenna = 32.  # 16. # 8.

                tgain_max = 4.  # do not allow ap solves that are more than 4 hrs
                tgain_min = 0.3333  # check if less than 20 min, min solint is 20 min

                innchan_list[soltype_id][ms_id] = 1

                if LBA:
                    if longbaseline:
                        solint_sf = 0.4 * gainfactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 10.0 * gainfactorsolint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.8 * gainfactorsolint  #
                    else:  # for -- HBA dutch --
                        solint_sf = 0.8 * gainfactorsolint  #

                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                print(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3), 'Computes gain solint:', solint, ' ')
                logger.info(
                    str(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)) + ' Computes gain solint: ' + str(
                        solint))
                print('Computes gain solint [hr]:', float(solint) * tint / 3600.)
                logger.info('Computes gain solint [hr]: ' + str(float(solint) * tint / 3600.))

                # do not allow very short ap solves
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) < tgain_min:  # check if less than tgain_min (20 min)
                    solint = np.rint(tgain_min * 3600. / tint)  # minimum tgain_min is 20 min
                    print('Setting gain solint to 20 min (the min value allowed):', float(solint) * tint / 3600.)
                    logger.info(
                        'Setting gain solint to 20 min (the min value allowed): ' + str(float(solint) * tint / 3600.))

                # do not allow ap solves that are more than tgain_max (4) hrs
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) > tgain_max:  # so check if larger than 4 hrs
                    print('Warning, it seems there is not enough flux density for gain solving')
                    logger.warning('Warning, it seems there is not enough flux density for gain solving')
                    solint = np.rint(tgain_max * 3600. / tint)  # max is tgain_max (4) hrs

                # trigger 15 MHz smoothnessconstraint
                # print('TEST:', ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.))
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) < thr_SM15Mhz:  # so check if smaller than 2 hr
                    insmoothnessconstraint_list[soltype_id][ms_id] = 5.0*gainfactorsmoothness
                else:
                    print('Increasing smoothnessconstraint to 15 MHz')
                    logger.info('Increasing smoothnessconstraint to 15 MHz')
                    insmoothnessconstraint_list[soltype_id][ms_id] = 15.0*gainfactorsmoothness

                # trigger nchan=0 solve because not enough S/N
                if not longbaseline and (((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) > thr_gain_trigger_allantenna):
                    inantennaconstraint_list[soltype_id][ms_id] = 'all'
                    solint = np.rint(
                        2.0 * 3600. / tint)  # 2 hrs nchan=0 solve (do not do bandpass because slope can diverge)
                    innchan_list[soltype_id][
                        ms_id] = 0  # no frequency dependence, smoothnessconstraint will be turned of in runDPPPbase
                    print('Triggering antennaconstraint all:', soltype, ms)
                    logger.info('Triggering antennaconstraint all: ' + soltype + ' ' + ms)
                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None

                # completely disable slow solve if the solints get too long, target is too faint
                if (((solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)) * tint / 3600.) > thr_disable_gain):
                    insoltypecycles_list[soltype_id][ms_id] = 999
                    print('Disabling solve:', soltype, ms)
                    logger.info('Disabling solve: ' + soltype + ' ' + ms)
                else:
                    insoltypecycles_list[soltype_id][
                        ms_id] = 3  # set to user input value? problem because not retained now (for example value can have been set to 999 above)

                insolint_list[soltype_id][ms_id] = int(solint)


    print('soltype:', soltype_list, mslist)
    print('nchan:', innchan_list)
    print('solint:', insolint_list)
    print('BLsmooth:', inBLsmooth_list)
    print('smoothnessconstraint:', insmoothnessconstraint_list)
    print('smoothnessreffrequency:', insmoothnessreffrequency_list)
    print('smoothnessspectralexponent_list:', insmoothnessspectralexponent_list)
    print('smoothnessrefdistance_list:', insmoothnessrefdistance_list)
    print('antennaconstraint:', inantennaconstraint_list)
    print('resetsols:', inresetsols_list)
    print('resetdir:', inresetdir_list)
    print('normamps:', innormamps_list)
    print('soltypecycles:', insoltypecycles_list)

    logger.info('soltype: ' + str(soltype_list) + ' ' + str(mslist))
    logger.info('nchan: ' + str(innchan_list))
    logger.info('solint: ' + str(insolint_list))
    logger.info('BLsmooth: ' + str(inBLsmooth_list))
    logger.info('smoothnessconstraint: ' + str(insmoothnessconstraint_list))
    logger.info('smoothnessreffrequency: ' + str(insmoothnessreffrequency_list))
    logger.info('smoothnessspectralexponent: ' + str(insmoothnessspectralexponent_list))
    logger.info('smoothnessrefdistance: ' + str(insmoothnessrefdistance_list))
    logger.info('antennaconstraint: ' + str(inantennaconstraint_list))
    logger.info('resetsols: ' + str(inresetsols_list))
    logger.info('resetdir: ' + str(inresetdir_list))
    logger.info('soltypecycles: ' + str(insoltypecycles_list))

    return innchan_list, insolint_list, inBLsmooth_list, insmoothnessconstraint_list, insmoothnessreffrequency_list, insmoothnessspectralexponent_list, insmoothnessrefdistance_list, inantennaconstraint_list, inresetsols_list, inresetdir_list, insoltypecycles_list, innormamps_list


def create_beamcortemplate(ms):
    """
    create a DPPP gain H5 template solutution file that can be filled with losoto
    """
    H5name = ms + '_templatejones.h5'

    cmd = "DP3 numthreads=" + str(
        np.min([multiprocessing.cpu_count(), 24])) + " msin=" + ms + " msin.datacolumn=DATA msout=. "
    # cmd += 'msin.modelcolumn=DATA '
    cmd += "steps=[ddecal] ddecal.type=ddecal "
    cmd += "ddecal.maxiter=1 ddecal.nchan=1 "
    cmd += "ddecal.modeldatacolumns='[DATA]' "
    cmd += "ddecal.mode=complexgain ddecal.h5parm=" + H5name + " "
    cmd += "ddecal.solint=10 ddecal.solveralgorithm=directioniterative "
    cmd += "ddecal.datause=dual"  # extra speedup
    # cmd += "ddecal.usedualvisibilities=True" # extra speedup
    print(cmd)
    run(cmd)

    return H5name


def create_losoto_beamcorparset(ms, refant='CS003HBA0'):
    """
    Create a losoto parset to fill the beam correction values'.
    """
    parset = 'losotobeam.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n\n\n')

    f.write('[plotphase]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-0.5,0.5]\n')
    f.write('prefix = plotlosoto%s/phases_beam\n' % os.path.basename(ms))
    f.write('refAnt = %s\n\n\n' % refant)

    f.write('[plotamp]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/amplitude000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [0.2,1]\n')
    f.write('prefix = plotlosoto%s/amplitudes_beam\n' % os.path.basename(ms))

    f.close()
    return parset


def create_losoto_tecandphaseparset(ms, refant='CS003HBA0', outplotname='fasttecandphase', markersize=2):
    parset = 'losoto_plotfasttecandphase.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = []\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plottecandphase]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('soltabToAdd = tec000\n')
    f.write('figSize=[120,20]\n')
    f.write('markerSize=%s\n' % int(markersize))
    f.write('prefix = plotlosoto%s/fasttecandphase\n' % os.path.basename(ms))
    f.write('refAnt = %s\n' % refant)

    f.close()
    return parset


def create_losoto_tecparset(ms, refant='CS003HBA0', outplotname='fasttec', markersize=2):
    parset = 'losoto_plotfasttec.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = []\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plottec]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/tec000]\n')
    f.write('axesInPlot = [time]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-0.2,0.2]\n')
    f.write('figSize=[120,20]\n')
    f.write('markerSize=%s\n' % int(markersize))
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname))
    f.write('refAnt = %s\n' % refant)

    f.close()
    return parset


def create_losoto_rotationparset(ms, refant='CS003HBA0', onechannel=False, outplotname='rotation', markersize=2):
    parset = 'losoto_plotrotation.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plotrotation]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/rotation000]\n')
    f.write('markerSize=%s\n' % int(markersize))
    if onechannel:
        f.write('axesInPlot = [time]\n')
    else:
        f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-1.57,1.57]\n')  # rotation needs to be plotted from -pi/2 to pi/2
    f.write('figSize=[120,20]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname))
    f.write('refAnt = %s\n' % refant)
    f.close()
    return parset


def create_losoto_fastphaseparset(ms, refant='CS003HBA0', onechannel=False, onepol=False, outplotname='fastphase', onetime=False, markersize=2):
    parset = 'losoto_plotfastphase.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plotphase]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    if onechannel:
        f.write('markerSize=%s\n' % int(markersize))
        f.write('axesInPlot = [time]\n')
        if not onepol:
            f.write('axisInCol = pol\n')
    if onetime:
        f.write('markerSize=%s\n' % int(markersize))
        f.write('axesInPlot = [freq]\n')
        if not onepol:
            f.write('axisInCol = pol\n')
    if not onechannel and not onetime:
        f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('figSize=[120,20]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname))
    f.write('refAnt = %s\n' % refant)

    if not onepol:
        f.write('[plotphasediff]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/phase000]\n')
        if onechannel:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [time]\n')
        if onetime:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [freq]\n')    
        if not onechannel and not onetime:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        f.write('minmax = [-3.14,3.14]\n')
        f.write('figSize=[120,20]\n')
        f.write('prefix = plotlosoto%s/%spoldiff\n' % (ms, outplotname))
        f.write('refAnt = %s\n' % refant)
        f.write('axisDiff=pol\n')

    f.close()
    return parset


def create_losoto_flag_apgridparset(ms, flagging=True, maxrms=7.0, maxrmsphase=7.0, includesphase=True,
                                    refant='CS003HBA0', onechannel=False, medamp=2.5, flagphases=True,
                                    onepol=False, outplotname='slowamp', fulljones=False, onetime=False, markersize=2):
    parset = 'losoto_flag_apgrid.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    # f.write('pol = []\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plotamp]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/amplitude000]\n')
    if onechannel:
        f.write('markerSize=%s\n' % int(markersize))
        f.write('axesInPlot = [time]\n')
        if not onepol:
            f.write('axisInCol = pol\n')
    if onetime:
        f.write('axesInPlot = [freq]\n')
        f.write('markerSize=%s\n' % int(markersize))
        if not onepol:
            f.write('axisInCol = pol\n')
    if not onechannel and not onetime:
        f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    # if longbaseline:
    #  f.write('minmax = [0,2.5]\n')
    # else:
    f.write('minmax = [%s,%s]\n' % (str(medamp / 4.0), str(medamp * 2.5)))
    # f.write('minmax = [0,2.5]\n')
    f.write('prefix = plotlosoto%s/%samp\n\n\n' % (ms, outplotname))

    if fulljones:
        f.write('[plotampXYYX]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/amplitude000]\n')
        f.write('pol = [XY, YX]\n')
        if onechannel:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [time]\n')
        if onetime:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [freq]\n')   
        if not onetime and not onechannel:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        f.write('minmax = [%s,%s]\n' % (str(0.0), str(0.5)))
        f.write('prefix = plotlosoto%s/%sampXYYX\n\n\n' % (ms, outplotname))

    if includesphase:
        f.write('[plotphase]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/phase000]\n')
        if onechannel:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [time]\n')
            if not onepol:
                f.write('axisInCol = pol\n')
        if onetime:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [freq]\n')
            if not onepol:
                f.write('axisInCol = pol\n')
        if not onetime and not onechannel:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        f.write('minmax = [-3.14,3.14]\n')
        f.write('prefix = plotlosoto%s/%sphase\n' % (ms, outplotname))
        f.write('refAnt = %s\n\n\n' % refant)

        if not onepol and not fulljones:
            f.write('[plotphasediff]\n')
            f.write('operation = PLOT\n')
            f.write('soltab = [sol000/phase000]\n')
            if onechannel:
                f.write('markerSize=%s\n' % int(markersize))
                f.write('axesInPlot = [time]\n')
            if onetime:
                f.write('markerSize=%s\n' % int(markersize))
                f.write('axesInPlot = [freq]\n')  
            if not onetime and not onechannel:
                f.write('axesInPlot = [time,freq]\n')
            f.write('axisInTable = ant\n')
            f.write('minmax = [-3.14,3.14]\n')
            f.write('figSize=[120,20]\n')
            f.write('prefix = plotlosoto%s/%spoldiff\n' % (ms, outplotname))
            f.write('refAnt = %s\n' % refant)
            f.write('axisDiff=pol\n\n\n')

    if flagging:
        f.write('[flagamp]\n')
        f.write('soltab = [sol000/amplitude000]\n')
        f.write('operation = FLAG\n')
        if onechannel:
            f.write('axesToFlag = [time]\n')
        if onetime:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesToFlag = [freq]\n')
        if not onetime and not onechannel:
            f.write('axesToFlag = [time,freq]\n')
        f.write('mode = smooth\n')
        f.write('maxCycles = 3\n')
        f.write('windowNoise = 7\n')
        f.write('maxRms = %s\n' % str(maxrms))
        if onechannel:
            f.write('order  = [5]\n\n\n')
        if onetime:
            f.write('order  = [5]\n\n\n')
        if not onetime and not onechannel:
            f.write('order  = [5,5]\n\n\n')

        if includesphase and flagphases:
            f.write('[flagphase]\n')
            f.write('soltab = [sol000/phase000]\n')
            f.write('operation = FLAG\n')
            if onechannel:
                f.write('axesToFlag = [time]\n')
            if onetime:
                f.write('axesToFlag = [freq]\n')    
            if not onetime and not onechannel:
                f.write('axesToFlag = [time,freq]\n')
            f.write('mode = smooth\n')
            f.write('maxCycles = 3\n')
            f.write('windowNoise = 7\n')
            f.write('maxRms = %s\n' % str(maxrmsphase))
            if onechannel:
                f.write('order  = [5]\n\n\n')
            if onetime:
                f.write('order  = [5]\n\n\n')
            if not onetime and not onechannel:
                f.write('order  = [5,5]\n\n\n')

        f.write('[plotampafter]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/amplitude000]\n')
        if onechannel:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [time]\n')
            if not onepol:
                f.write('axisInCol = pol\n')
        if onetime:
            f.write('markerSize=%s\n' % int(markersize))
            f.write('axesInPlot = [freq]\n')
            if not onepol:
                f.write('axisInCol = pol\n')
        if not onetime and not onechannel:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        # f.write('minmax = [0,2.5]\n')
        f.write('minmax = [%s,%s]\n' % (str(medamp / 4.0), str(medamp * 2.5)))
        f.write('prefix = plotlosoto%s/%sampfl\n\n\n' % (ms, outplotname))

        if includesphase and flagphases:
            f.write('[plotphase_after]\n')
            f.write('operation = PLOT\n')
            f.write('soltab = [sol000/phase000]\n')
            if onechannel:
                f.write('markerSize=%s\n' % int(markersize))
                f.write('axesInPlot = [time]\n')
                if not onepol:
                    f.write('axisInCol = pol\n')
            if onetime:
                f.write('markerSize=%s\n' % int(markersize))
                f.write('axesInPlot = [freq]\n')
                if not onepol:
                    f.write('axisInCol = pol\n')
            if not onetime and not onechannel:
                f.write('axesInPlot = [time,freq]\n')
            f.write('axisInTable = ant\n')
            f.write('minmax = [-3.14,3.14]\n')
            f.write('prefix = plotlosoto%s/%sphasefl\n' % (ms, outplotname))
            f.write('refAnt = %s\n' % refant)

    f.close()
    return parset


def create_losoto_bandpassparset(intype):
    '''
    Create losoto parset than takes median along the time axis
    Can be used to create a bandpass
    Parameters:
    intype (str): set "phase" or "amplitude" or amplitude and phase ("a&p") smoothing, input should be one of these strings
    '''
    assert intype == 'phase' or intype == 'amplitude' or intype == 'a&p'
    parset = 'losoto_bandpass.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    if intype == 'amplitude' or intype == 'a&p':
        f.write('[bandpassamp]\n')
        f.write('operation = SMOOTH\n')
        f.write('soltab = [sol000/amplitude000]\n')
        f.write('axesToSmooth = [time] # axes to smooth\n')
        f.write('mode = median\n')
        f.write('replace = False\n')
        f.write('log = True\n')

    if intype == 'phase' or intype == 'a&p':
        f.write('[bandpassphase]\n')
        f.write('operation = SMOOTH\n')
        f.write('soltab = [sol000/phase000]\n')
        f.write('axesToSmooth = [time] # axes to smooth\n')
        f.write('mode = median\n')
        f.write('replace = False\n')
        f.write('log = False\n')

    f.close()
    return parset    
    
    
    

def create_losoto_mediumsmoothparset(ms, boxsize, longbaseline, includesphase=True, refant='CS003HBA0',
                                     onechannel=False, outplotname='runningmedian'):
    parset = 'losoto_mediansmooth.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('pol = []\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    if includesphase:
        f.write('[smoothphase]\n')
        f.write('soltab = [sol000/phase000]\n')
        f.write('operation= SMOOTH\n')
        if onechannel:
            f.write('axesToSmooth = [time]\n')
            f.write('size = [%s]\n' % (boxsize))
        else:
            f.write('axesToSmooth = [freq,time]\n')
            f.write('size = [%s,%s]\n' % (boxsize, boxsize))
        f.write('mode = runningmedian\n\n\n')

    f.write('[smoothamp]\n')
    f.write('soltab = [sol000/amplitude000]\n')
    f.write('operation= SMOOTH\n')
    if onechannel:
        f.write('axesToSmooth = [time]\n')
        f.write('size = [%s]\n' % (boxsize))
    else:
        f.write('axesToSmooth = [freq,time]\n')
        f.write('size = [%s,%s]\n' % (boxsize, boxsize))
    f.write('mode = runningmedian\n\n\n')

    f.write('[plotamp_after]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/amplitude000]\n')
    if onechannel:
        f.write('axesInPlot = [time]\n')
    else:
        f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    if longbaseline:
        f.write('minmax = [0,2.5]\n')
    else:
        f.write('minmax = [0,2.5]\n')
    f.write('prefix = plotlosoto%s/amps_smoothed\n\n\n' % os.path.basename(ms))

    if includesphase:
        f.write('[plotphase_after]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/phase000]\n')
        if onechannel:
            f.write('axesInPlot = [time]\n')
        else:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        f.write('minmax = [-3.14,3.14]\n')
        f.write('prefix = plotlosoto%s/phases_smoothed\n\n\n' % os.path.basename(ms))
        f.write('refAnt = %s\n' % refant)

        f.write('[plotphase_after1rad]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/phase000]\n')
        if onechannel:
            f.write('axesInPlot = [time]\n')
        else:
            f.write('axesInPlot = [time,freq]\n')
        f.write('axisInTable = ant\n')
        f.write('minmax = [-1,1]\n')
        f.write('prefix = plotlosoto%s/phases_smoothed1rad\n' % os.path.basename(ms))
        f.write('refAnt = %s\n' % refant)

    f.close()
    return parset


def check_phaseup(H5name):
    """
    Check if the antenna 'ST001' is present in any solution type within the H5 file.
    """
    with tables.open_file(H5name, mode='r') as H5:
        ants = []
        for sol_type in ['phase000', 'amplitude000', 'rotation000', 'tec000', 'rotationmeasure000']:
            try:
                ants = getattr(H5.root.sol000, sol_type).ant[:]
                break  # Stop if antennas are found in the current solution type
            except tables.NoSuchNodeError:
                continue  # Try the next solution type if the current one is missing

    return 'ST001' in ants


def fixbeam_ST001(H5name):
    H5 = h5parm.h5parm(H5name, readonly=False)

    ants = H5.getSolset('sol000').getAnt().keys()
    antsrs = fnmatch.filter(ants, 'RS*')
    ST001 = False

    if 'ST001' in ants:
        ST001 = True
        amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()
        ampvals = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
        phasevals = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]

        idx = np.where(amps[1]['ant'] == 'ST001')[0][0]
        idxrs = np.where(amps[1]['ant'] == antsrs[0])[0][0]
        # idx106 = np.where(amps[1]['ant'] == 'RS106HBA')[0][0]
        # idx305 = np.where(amps[1]['ant'] == 'RS305HBA')[0][0]
        # idx508 = np.where(amps[1]['ant'] == 'RS508HBA')[0][0]
        # idx406 = np.where(amps[1]['ant'] == 'RS406HBA')[0][0]
        # idxnonecheck = np.where(amps[1]['ant'] == 'blaHBA')[0][0]

        # ampvals[:,:, idx, 0,:] = 1.0  # set amplitude to 1.
        # phasevals[:,:, idx, 0,:] = 0.0 # set phase to 0.

        ampvals[:, :, idx, 0, :] = ampvals[:, :, idxrs, 0, :]
        phasevals[:, :, idx, 0, :] = 0.0

        H5.getSolset('sol000').getSoltab('amplitude000').setValues(ampvals)
        H5.getSolset('sol000').getSoltab('phase000').setValues(phasevals)

    H5.close()

    return ST001


def split_facetdirections(facetregionfile):
    """
    split composite facet region file into individual polygon region files
    """
    r = pyregion.open(facetregionfile)
    for facet_id, facet in enumerate(r):
        r[facet_id:facet_id + 1].write('facet' + str(facet_id) + '.reg')
    return


def create_facet_directions(imagename, selfcalcycle, targetFlux=1.0, ms=None, imsize=None,
                            pixelscale=None, numClusters=0, weightBySize=False,
                            facetdirections=None, imsizemargin=100, restart=False,
                            via_h5=False, h5=None):
    """
    Create a facet region file based on an input image or file provided by the user
    if there is an image use lsmtool tessellation algorithm

    This function also returns the solints and smoothness obtained out of the facetdirections file (if avail). It is up to
    the function that calls this to do something with it or not.
    """
    if via_h5: # this is quick call to ds9facetgenerator using an h5 file and a None return
        cmd = f'python {submodpath}/ds9facetgenerator.py '
        cmd += '--ms=' + ms + ' --h5=' + h5 + ' '
        cmd += '--imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
        run(cmd)
        return

    solints = None  # initialize, if not filled then this is not used here and the settings are taken from facetselfcal argsparse
    smoothness = None  # initialize, if not filled then this is not used here and the settings are taken from facetselfcal argsparse
    soltypelist_includedir = None  # initialize
    if facetdirections is not None:
        try:
            PatchPositions_array, solints, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, selfcalcycle)
        except:
            try:
                f = open(facetdirections, 'rb')
                PatchPositions_array = pickle.load(f)
                f.close()
            except:
                raise Exception('Trouble read file format:' + facetdirections)
        print(PatchPositions_array)

        # write new facetdirections.p file
        if os.path.isfile('facetdirections.p'):
            os.system('rm -f facetdirections.p')
        f = open('facetdirections.p', 'wb')
        pickle.dump(PatchPositions_array, f)
        f.close()

        # generate polygon composite regions file for WSClean imaging
        # in case of a restart this is not done, and the old facets.reg is kept
        # using the old facets.reg is important in case we change the number of facet directions, so the that first image after the restart is done with the old directions (and the h5file used was made using that)
        if ms is not None and imsize is not None and pixelscale is not None and not restart:
            cmd = f'python {submodpath}/ds9facetgenerator.py '
            cmd += '--ms=' + ms + ' '
            cmd += '--h5=facetdirections.p --imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
            run(cmd)
        return solints, smoothness, soltypelist_includedir
    elif selfcalcycle == 0:
        # Only run this if selfcalcycle==0 [elif]
        # Try to load previous facetdirections.skymodel
        import lsmtool
        if 'skymodel' not in imagename:
            img = bdsf.process_image(imagename + str(selfcalcycle).zfill(3) + '-MFS-image.fits', mean_map='zero',
                                     rms_map=True, rms_box=(160, 40))
            img.write_catalog(format='bbs', bbs_patches=None, outfile='facetdirections.skymodel', clobber=True)
            del img
        else:
            os.system('cp -r {} facetdirections.skymodel'.format(imagename))
        LSM = lsmtool.load('facetdirections.skymodel')

        if numClusters > 0:
            LSM.group(algorithm='cluster', numClusters=numClusters)
        else:
            LSM.group(algorithm='tessellate', targetFlux=str(targetFlux) + ' Jy', weightBySize=weightBySize)

        print('Number of directions', len(LSM.getPatchPositions()))
        PatchPositions = LSM.getPatchPositions()

        PatchPositions_array = np.zeros((len(LSM.getPatchPositions()), 2))

        for patch_id, patch in enumerate(PatchPositions.keys()):
            PatchPositions_array[patch_id, 0] = PatchPositions[patch][0].to(units.rad).value  # RA
            PatchPositions_array[patch_id, 1] = PatchPositions[patch][1].to(units.rad).value  # Dec
        # Run code below for if and elif
        if os.path.isfile('facetdirections.p'):
            os.system('rm -f facetdirections.p')
        f = open('facetdirections.p', 'wb')
        pickle.dump(PatchPositions_array, f)
        f.close()

        # generate polygon composite regions file for WSClean imaging
        if ms is not None and imsize is not None and pixelscale is not None:
            cmd = f'python {submodpath}/ds9facetgenerator.py '
            cmd += '--ms=' + ms + ' '
            cmd += '--h5=facetdirections.p --imsize=' + str(imsize + int(imsize*0.15)) + ' --pixelscale=' + str(pixelscale)
            run(cmd)
        return solints, smoothness, soltypelist_includedir
    else:
        return solints, smoothness, soltypelist_includedir


def write_ds9_regions(ra_array, dec_array, filename="directions.reg", radius=120.0, color="red"):
    """
    Writes DS9 region file entries for each RA, DEC pair.

    Parameters:
    - ra_array (np.ndarray): 1D array of Right Ascension values (in degrees)
    - dec_array (np.ndarray): 1D array of Declination values (in degrees)
    - filename (str): Output text file name
    - radius (float): Radius of the circle in arcseconds
    - color (str): Color to use in DS9 region entries
    """
    with open(filename, "w") as f:
        f.write('# Region file format: DS9 version 4.1\n')
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1\n')
        f.write('fk5\n')
        for i, (ra, dec) in enumerate(zip(ra_array, dec_array)):
            f.write(f"circle({ra},{dec},{radius:.3f}\") # color={color} text={{Dir{i:02d}}}\n")

def parse_facetdirections(facetdirections, selfcalcycle, writeregioncircles=True):
    """
       parse the facetdirections.txt file and return a list of facet directions
       for the given selfcalcycle. In the future, this function should also return a
       list of solints, smoothness, nchans and other things
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

    if writeregioncircles:
        write_ds9_regions(ra, dec, filename="facet_centers.reg", radius=120.0, color="red")
    # Only select ra/dec which are if they are in the selfcalcycle range
    a = np.where((start <= selfcalcycle))[0]
    rasel = ra[a]
    decsel = dec[a]

    if soltypelist_includedir is not None and 'args' in globals():
        soltypelist_includedir_sel_tmp = soltypelist_includedir[a]

        # create 2D array booleans
        soltypelist_includedir_sel = np.zeros((len(rasel), len(args['soltype_list'])), dtype=bool)
        for dir_id in range(len(rasel)):
            # print(dir_id, soltypelist_includedir_sel_tmp[dir_id])
            soltypelist_includedir_sel[dir_id, :] = ast.literal_eval(soltypelist_includedir_sel_tmp[dir_id])

    PatchPositions_array = np.zeros((len(rasel), 2))
    PatchPositions_array[:, 0] = (rasel * units.deg).to(units.rad).value
    PatchPositions_array[:, 1] = (decsel * units.deg).to(units.rad).value

    # check for consistency of solints having the same number of entries as args['soltype_list']
    if solints is not None:
        for solint in solints:
            if len(ast.literal_eval(solint)) != len(args['soltype_list']):
                print('Number of entries for solints in the direction file is', \
                      len(ast.literal_eval(solint)), 'but args["soltype_list"] has:', len(args['soltype_list']))
                raise ValueError('The number of solints in the directions file does not match the number of soltypes in args["soltype_list"]. '
                                 'Please check the directions file.')
    # check for consistency of smoothness having the same number of entries as args['soltype_list']
    if smoothness is not None:
        for sm in smoothness:
            if len(ast.literal_eval(sm)) != len(args['soltype_list']):
                print('Number of entries for smoothness in the direction file is', \
                      len(ast.literal_eval(sm)), 'but args["soltype_list"] has:', len(args['soltype_list']))
                raise ValueError('The number of smoothness in the directions file does not match the number of soltypes in args["soltype_list"]. '
                                 'Please check the directions file.')
    # check for consistency of soltypelist_includedir having the same number of entries as args['soltype_list']
    if soltypelist_includedir is not None:
        for soltypelist in soltypelist_includedir:
            if len(ast.literal_eval(soltypelist)) != len(args['soltype_list']):
                print('Number of entries for soltypelist_includedir in the direction file is', \
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



def prepare_DDE(imagebasename, selfcalcycle, mslist,
                DDE_predict='DP3', restart=False, disable_IDG_DDE_predict=True,
                telescope='LOFAR', skyview=None, wscleanskymodel=None,
                skymodel=None):
    if telescope == 'LOFAR' and not disable_IDG_DDE_predict:
        idg = True  # predict WSCLEAN with beam using IDG (wsclean facet mode with h5 is not efficient here)
    else:
        idg = False

   
    solints, smoothness, soltypelist_includedir = create_facet_directions(imagebasename, selfcalcycle,
                                                              targetFlux=args['targetFlux'], ms=mslist[0], imsize=args['imsize'],
                                                              pixelscale=args['pixelscale'], numClusters=args['Nfacets'],
                                                              facetdirections=args['facetdirections'], restart=restart)
    
    # --- start CREATE facets.fits -----
    # remove previous facets.fits if needed
    if os.path.isfile('facets.fits'):
        os.system('rm -f facets.fits')
    if skyview == None:
        if not restart and wscleanskymodel is None and skymodel is None:
            os.system('cp ' + imagebasename + str(selfcalcycle).zfill(3) + '-MFS-image.fits' + ' facets.fits')
        if not restart and wscleanskymodel is not None:
            # os.system('cp ' + glob.glob(wscleanskymodel + '-????-*model*.fits')[0] + ' facets.fits')
            create_empty_fitsimage(mslist[0], int(args['imsize']), float(args['pixelscale']), 'facets.fits')
        if not restart and skymodel is not None:
            create_empty_fitsimage(mslist[0], int(args['imsize']), float(args['pixelscale']), 'facets.fits')
    else:
        os.system('cp ' + skyview + ' facets.fits')
    
    if restart:  # in that case we also have a previous image avaialble
        os.system('cp ' + imagebasename + str(selfcalcycle - 1).zfill(3) + '-MFS-image.fits' + ' facets.fits')

    # --- end CREATE facets.fits -----

    # FILL in facets.fits with values, every facets get a constant value, for lsmtool
    hdu = fits.open('facets.fits')
    hduflat = flatten(hdu)
    region = pyregion.open('facets.reg')
    
    for facet_id, facet in enumerate(region):
        region[facet_id:facet_id + 1].write('facet' + str(facet_id) + '.reg')  # split facet from region file
        r = pyregion.open('facet' + str(facet_id) + '.reg')
        print('Filling facets.fits with:', 'facet' + str(facet_id) + '.reg')
        manualmask = r.get_mask(hdu=hduflat)
        if len(hdu[0].data.shape) == 4:
            hdu[0].data[0][0][np.where(manualmask == True)] = facet_id
        else:
            hdu[0].data[np.where(manualmask == True)] = facet_id
    hdu.writeto('facets.fits', overwrite=True)

    if restart:
        # restart with DDE_predict=DP3 because then only the variable modeldatacolumns is made
        # So the wsclean predict step is skipped in makeimage but variable modeldatacolumns is created
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facets.reg',
                                     DDE_predict='DP3',
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        # selfcalcycle-1 because makeimage has not yet produced an image at this point
        if args['fitspectralpol'] > 0 and DDE_predict == 'DP3':
            dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle - 1).zfill(3) + '-sources.txt', 'facets.fits')
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off
    elif skyview is not None:
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facets.reg',
                                     DDE_predict=DDE_predict,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        if args['fitspectralpol'] > 0:
            dde_skymodel = groupskymodel(imagebasename, 'facets.fits')  # imagebasename
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off

    elif wscleanskymodel is not None:  # DDE wscleanskymodel solve at the start

        nonpblist = glob.glob(wscleanskymodel + '-????-model.fits')
        pblist = glob.glob(wscleanskymodel + '-????-model-pb.fits')

        if len(pblist) > 0:
            channelsout_forpredict = len(pblist)
        else:
            channelsout_forpredict = len(nonpblist)

        modeldatacolumns = makeimage(mslist, wscleanskymodel,
                                     args['pixelscale'], args['imsize'], channelsout_forpredict, predict=True,
                                     onlypredict=True, facetregionfile='facets.reg',
                                     DDE_predict='WSCLEAN', idg=idg,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        # assume there is no model for DDE wscleanskymodel solve at the start
        # since we are making image000 afterwards anyway setting a dummy now is ok
        dde_skymodel = 'dummy.skymodel'
        print(modeldatacolumns)
    else:
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facets.reg',
                                     DDE_predict=DDE_predict, idg=idg,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        if args['fitspectralpol'] > 0 and DDE_predict == 'DP3':
            dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle).zfill(3) + '-sources.txt', 'facets.fits')
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off
    # check if -pb version of source list exists
    # needed because image000 does not have a pb version as no facet imaging is used, however, if IDG is used it does exist and hence the following does also handfle that
    if telescope == 'LOFAR' and wscleanskymodel is None:  # not for MeerKAT because WSClean still has a bug if no primary beam is used, for now assume we do not use a primary beam for MeerKAT
        if os.path.isfile(imagebasename + str(selfcalcycle).zfill(3) + '-sources-pb.txt'):
            if args['fitspectralpol'] > 0:
                dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle).zfill(3) + '-sources-pb.txt',
                                             'facets.fits')
            else:
                dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off

    return modeldatacolumns, dde_skymodel, solints, smoothness, soltypelist_includedir


def is_scalar_array_for_wsclean(h5list):
    is_scalar = True  # Start with True to catch cases where the last polarization dimension is missing

    for h5 in h5list:
        with tables.open_file(h5, mode='r') as H5:
            for sol_type in ['phase000', 'amplitude000']:
                try:
                    val = getattr(H5.root.sol000, sol_type).val[:]  # time, freq, ant, dir, pol
                    if not np.array_equal(val[..., 0], val[..., -1]):
                        is_scalar = False
                except AttributeError:
                    pass

    return is_scalar


# this version corrupts the MODEL_DATA column
def calibrateandapplycal(mslist, selfcalcycle, solint_list, nchan_list,
                         soltypecycles_list, smoothnessconstraint_list,
                         smoothnessreffrequency_list, smoothnessspectralexponent_list,
                         smoothnessrefdistance_list,
                         antennaconstraint_list, resetsols_list, resetdir_list, normamps_list,
                         BLsmooth_list,
                         solve_msinnchan_list, solve_msinstartchan_list,
                         antenna_averaging_factors_list, antenna_smoothness_factors_list,
                         normamps=False, normamps_per_ms=False, skymodel=None,
                         predictskywithbeam=False,
                         longbaseline=False,
                         skymodelsource=None,
                         skymodelpointsource=None, wscleanskymodel=None, skymodelsetjy=False,
                         mslist_beforephaseup=None,
                         modeldatacolumns=[], dde_skymodel=None,
                         DDE_predict='WSCLEAN', telescope='LOFAR',
                         mslist_beforeremoveinternational=None, soltypelist_includedir=None):
    ## --- start STACK code ---
    if args['stack']:
        # create MODEL_DATA because in case it does not exist (needed in case user gives external model(s))
        # only for first (=0) selfcalcycle cycle and if user provides a model
        if ((skymodel is not None) or (skymodelpointsource is not None)
            or (wscleanskymodel is not None)) and selfcalcycle == 0:
            for ms_id, ms in enumerate(mslist):  # do the predicts (only used for stacking)
                print('Doing sky predict for stacking...')
                if skymodel is not None and type(skymodel) is str:
                    predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam,
                               sources=skymodelsource, modelstoragemanager=args['modelstoragemanager'])
                if skymodel is not None and type(skymodel) is list:
                    predictsky(ms, skymodel[ms_id], modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam,
                               sources=skymodelsource, modelstoragemanager=args['modelstoragemanager'])
                if wscleanskymodel is not None and type(wscleanskymodel) is str and not args['phasediff_only']:
                    makeimage([ms], wscleanskymodel, 1., 1.,
                              len(glob.glob(wscleanskymodel + '-????-model.fits')),
                              0, 0.0, onlypredict=True, idg=False,
                              fulljones_h5_facetbeam=not args['single_dual_speedup'])
                if wscleanskymodel is not None and type(wscleanskymodel) is list and not args['phasediff_only']:
                    makeimage([ms], wscleanskymodel[ms_id], 1., 1.,
                              len(glob.glob(wscleanskymodel[ms_id] + '-????-model.fits')),
                              0, 0.0, onlypredict=True, idg=False,
                              fulljones_h5_facetbeam=not args['single_dual_speedup'])

                if skymodelpointsource is not None and type(skymodelpointsource) is float:
                    # create MODEL_DATA (no dysco!)
                    if args['modelstoragemanager'] is None:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
                    else:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + args['modelstoragemanager'] + ' steps=[]', log=True)                        
                    # do the predict with taql
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource) + "+0i)'",
                        log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource) + "+0i)'",
                        log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'", log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'", log=True)

                if skymodelpointsource is not None and type(skymodelpointsource) is list:
                    # create MODEL_DATA (no dysco!)
                    if args['modelstoragemanager'] is None:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
                    else:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + args['modelstoragemanager'] + ' steps=[]', log=True)
                    # do the predict with taql
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource[ms_id]) + "+0i)'",
                        log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource[ms_id]) + "+0i)'",
                        log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'", log=True)
                    run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'", log=True)

        # do the stack and normalization
        # mslist_stacked = stacked MS (one per common time axis / timestack); mss_timestacks = list of MSs that were used to create each stack
        mslist_stacked, mss_timestacks = stackwrapper(mslist, msout_prefix='stack', column_to_normalise='DATA')
        mslist_orig = mslist[:]  # so we can use it for the applycal
        mslist = mslist_stacked

        # set MODEL_DATA to point source in stacks
        for ms in mslist:
            t = table(ms, ack=False)
            if 'MODEL_DATA' not in t.colnames():
                t.close()
                run(f'DP3 msin={ms} msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
            else:
                t.close()
            print(f'Predict point source for  {ms}')
            # do the predict with taql
            run(f"taql 'update {ms} set MODEL_DATA[,0]=(1.0+ +0i)'", log=True)
            run(f"taql 'update {ms} set MODEL_DATA[,3]=(1.0+ +0i)'", log=True)
            run(f"taql 'update {ms} set MODEL_DATA[,1]=(0+0i)'", log=True)
            run(f"taql 'update {ms} set MODEL_DATA[,2]=(0+0i)'", log=True)

        # set all these to None to avoid skymodel predicts in runDPPPbase()
        skymodelpointsource = None
        wscleanskymodel = None
        skymodel = None
        skymodelsetjy = False
    ## --- end STACK code ---

    if len(modeldatacolumns) > 1:
        merge_all_in_one = False
    else:
        merge_all_in_one = True

    soltypecycles_list_array = np.array(soltypecycles_list)  # needed to slice (slicing does not work in nested l
    pertubation = []  # len(mslist)
    for _ in mslist:
        pertubation.append(False)

    parmdbmergelist = [[] for _ in
                       range(len(mslist))]  # [[],[],[],[]] nested list length mslist used for Jurjen's h5_merge
    # LOOP OVER THE ENTIRE SOLTYPE LIST (so includes pertubations via a pre-applycal)
    for soltypenumber, soltype in enumerate(args['soltype_list']):
        # SOLVE LOOP OVER MS
        parmdbmslist = []
        for msnumber, ms in enumerate(mslist):
            # check we are above far enough in the selfcal to solve for the extra pertubation
            if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]:
                print('selfcalcycle, soltypenumber', selfcalcycle, soltypenumber)
                if (soltypenumber < len(args['soltype_list']) - 1):

                    print(selfcalcycle, soltypecycles_list[soltypenumber + 1][msnumber])
                    print('_______________________')
                    print(soltypecycles_list_array, soltypenumber, len(soltypecycles_list_array))
                    print('Array soltypecycles_list ahead',
                          soltypecycles_list_array[soltypenumber + 1:len(soltypecycles_list_array[:, 0]), msnumber])
                    # if (selfcalcycle >= soltypecycles_list[soltypenumber+1][msnumber]): # this looks one soltype ahead...hmmm, not good
                    if selfcalcycle >= np.min(
                            soltypecycles_list_array[soltypenumber + 1:len(soltypecycles_list_array[:, 0]),
                            msnumber]):  # this looks all soltype ahead
                        pertubation[msnumber] = True
                    else:
                        pertubation[msnumber] = False
                else:
                    pertubation[msnumber] = False

                if ((skymodel is not None) or (skymodelpointsource is not None) or (
                        wscleanskymodel is not None) or (args['skymodelsetjy'])):
                    parmdb = soltype + str(soltypenumber) + '_skyselfcalcycle' + str(selfcalcycle).zfill(
                        3) + '_' + os.path.basename(ms) + '.h5'
                else:
                    parmdb = soltype + str(soltypenumber) + '_selfcalcycle' + str(selfcalcycle).zfill(
                        3) + '_' + os.path.basename(ms) + '.h5'

                # set create_modeldata to False it was already prediceted before
                create_modeldata = True
                if (soltypenumber >= 1) or (selfcalcycle > 0 and not args['keepusingstartingskymodel']): 
                    create_modeldata = False

                runDPPPbase(ms, solint_list[soltypenumber][msnumber], nchan_list[soltypenumber][msnumber], parmdb,
                            soltype,
                            uvmin=args['uvmin'],
                            SMconstraint=smoothnessconstraint_list[soltypenumber][msnumber],
                            SMconstraintreffreq=smoothnessreffrequency_list[soltypenumber][msnumber],
                            SMconstraintspectralexponent=smoothnessspectralexponent_list[soltypenumber][msnumber],
                            SMconstraintrefdistance=smoothnessrefdistance_list[soltypenumber][msnumber],
                            antennaconstraint=antennaconstraint_list[soltypenumber][msnumber],
                            resetsols=resetsols_list[soltypenumber][msnumber],
                            resetsols_list=resetsols_list,
                            resetdir=resetdir_list[soltypenumber][msnumber],
                            restoreflags=args['restoreflags'], flagging=args['doflagging'], skymodel=skymodel,
                            flagslowphases=args['doflagslowphases'], flagslowamprms=args['flagslowamprms'],
                            flagslowphaserms=args['flagslowphaserms'],
                            predictskywithbeam=predictskywithbeam, BLsmooth=BLsmooth_list[soltypenumber][msnumber],
                            skymodelsource=skymodelsource,
                            skymodelpointsource=skymodelpointsource, wscleanskymodel=wscleanskymodel,
                            iontimefactor=args['iontimefactor'], ionfreqfactor=args['ionfreqfactor'], blscalefactor=args['blscalefactor'], 
                            dejumpFR=args['dejumpFR'], uvminscalarphasediff=args['uvminscalarphasediff'],
                            create_modeldata=create_modeldata,
                            selfcalcycle=selfcalcycle, dysco=args['dysco'], blsmooth_chunking_size=args['blsmooth_chunking_size'],
                            soltypenumber=soltypenumber, ampresetvalfactor=args['ampresetvalfactor'],
                            flag_ampresetvalfactor=args['flag_ampresetvalfactor'],
                            clipsolutions=args['clipsolutions'], clipsolhigh=args['clipsolhigh'],
                            clipsollow=args['clipsollow'], uvmax=args['uvmax'], modeldatacolumns=modeldatacolumns,
                            preapplyH5_dde=parmdbmergelist[msnumber], dde_skymodel=dde_skymodel,
                            DDE_predict=DDE_predict, ncpu_max=args['ncpu_max_DP3solve'], soltype_list=args['soltype_list'],
                            DP3_dual_single=args['single_dual_speedup'], soltypelist_includedir=soltypelist_includedir,
                            normamps=normamps, modelstoragemanager=args['modelstoragemanager'], skymodelsetjy=skymodelsetjy,
                            solve_msinnchan=solve_msinnchan_list[soltypenumber][msnumber],
                            solve_msinstartchan=solve_msinstartchan_list[soltypenumber][msnumber],
                            antenna_averaging_factors=antenna_averaging_factors_list[soltypenumber][msnumber],
                            antenna_smoothness_factors=antenna_smoothness_factors_list[soltypenumber][msnumber])
                parmdbmslist.append(parmdb)
                parmdbmergelist[msnumber].append(parmdb)  # for h5_merge

        # NORMALIZE amplitudes
        if normamps and (soltype in ['complexgain', 'scalarcomplexgain', 'rotation+diagonal',
                                     'rotation+diagonalamplitude', 'rotation+scalar',
                                     'rotation+scalaramplitude', 'amplitudeonly', 'scalaramplitude',
                                     'faradayrotation+diagonal', 'faradayrotation+diagonalamplitude',
                                     'faradayrotation+scalar', 'faradayrotation+scalaramplitude']) and len(
            parmdbmslist) > 0:
            print('Doing global amplitude-type normalization')

            # [soltypenumber][msnumber] msnumber=0 is ok because we  do all ms at once
            if normamps_list[soltypenumber][0] == 'normamps':  #
                print('Performing global amplitude normalization')
                normamplitudes(parmdbmslist,
                               norm_per_ms=normamps_per_ms)  # list of h5 for different ms, all same soltype

            if normamps_list[soltypenumber][0] == 'normamps_per_ant':
                print('Performing global amplitude normalization per antenna')
                normamplitudes_withmatrix(parmdbmslist)

            if normamps_list[soltypenumber][0] == 'normslope':
                print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)

            if normamps_list[soltypenumber][0] == 'normslope+normamps':
                print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)  # first do the slope
                normamplitudes(parmdbmslist, norm_per_ms=normamps_per_ms)

            if normamps_list[soltypenumber][0] == 'normslope+normamps_per_ant':
                print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)  # first do the slope
                normamplitudes_withmatrix(parmdbmslist)

        if args['phasediff_only']:
            continue

        # APPLYCAL or PRE-APPLYCAL or CORRUPT
        count = 0
        for msnumber, ms in enumerate(mslist):
            if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]:  # Check cycle condition
                print(pertubation[msnumber], parmdbmslist[count], msnumber, count)

                if pertubation[msnumber]:  # Indicates another solve follows after this
                    if soltypenumber == 0:
                        if len(modeldatacolumns) > 1:
                            if DDE_predict == 'WSCLEAN':
                                corrupt_modelcolumns(ms, parmdbmslist[count], modeldatacolumns, modelstoragemanager=args['modelstoragemanager'])  # For DDE
                        else:
                            corrupt_modelcolumns(ms, parmdbmslist[count], ['MODEL_DATA'], modelstoragemanager=args['modelstoragemanager'])  # Saves disk space
                    else:
                        if len(modeldatacolumns) > 1:
                            if DDE_predict == 'WSCLEAN':
                                corrupt_modelcolumns(ms, parmdbmslist[count], modeldatacolumns, modelstoragemanager=args['modelstoragemanager'])  # For DDE
                        else:
                            corrupt_modelcolumns(ms, parmdbmslist[count], ['MODEL_DATA'], modelstoragemanager=args['modelstoragemanager'])  # Saves disk space
                else:  # This is the last solve; no other perturbation
                    # Reverse solution tables ([::-1]) to switch from corrupt to correct; essential as fulljones and diagonal solutions do not commute
                    applycal(ms, parmdbmergelist[msnumber][::-1], msincol='DATA', msoutcol='CORRECTED_DATA',
                             dysco=args['dysco'], modeldatacolumns=modeldatacolumns)  # Saves disk space

                count += 1  # Extra counter because parmdbmslist can be shorter than mslist as soltypecycles_list goes per ms

    if args['phasediff_only']:
        return []

    wsclean_h5list = []

    for msnumber, ms in enumerate(mslist):
        if ((skymodel is not None) or (skymodelpointsource is not None) or (
                wscleanskymodel is not None) or (args['skymodelsetjy'])):
            parmdbmergename = 'merged_skyselfcalcycle' + str(selfcalcycle).zfill(3) + '_' + os.path.basename(
                ms) + '.h5'
            parmdbmergename_pc = 'merged_skyselfcalcycle' + str(selfcalcycle).zfill(
                3) + '_linearfulljones_' + os.path.basename(ms) + '.h5'
        else:
            parmdbmergename = 'merged_selfcalcycle' + str(selfcalcycle).zfill(3) + '_' + os.path.basename(ms) + '.h5'
            parmdbmergename_pc = 'merged_selfcalcycle' + str(selfcalcycle).zfill(
                3) + '_linearfulljones_' + os.path.basename(ms) + '.h5'
        if os.path.isfile(parmdbmergename):
            os.system('rm -f ' + parmdbmergename)
        if os.path.isfile(parmdbmergename_pc):
            os.system('rm -f ' + parmdbmergename_pc)
        wsclean_h5list.append(parmdbmergename)

        # add extra from preapplyH5_list
        if args['preapplyH5_list'][0] is not None:
            preapplyh5parm = time_match_mstoH5(args['preapplyH5_list'], ms)
            # replace the source direction coordinates so that the merge goes correctly
            copy_over_source_direction_h5(parmdbmergelist[msnumber][0], preapplyh5parm)
            parmdbmergelist[msnumber].append(preapplyh5parm)

        if is_scalar_array_for_wsclean(parmdbmergelist[msnumber]):
            single_pol_merge = True
        else:
            single_pol_merge = False

        # reset h5parm structure
        if not merge_all_in_one:  # only for a DDE solve
            parmdbmergelist[msnumber] = fix_h5(parmdbmergelist[msnumber])

        print(parmdbmergename, parmdbmergelist[msnumber], ms)
        if args['reduce_h5size'] and ('tec' not in args['soltype_list']) and ('tecandphase' not in args['soltype_list']):
            merge_h5(h5_out=parmdbmergename, h5_tables=parmdbmergelist[msnumber][::-1],
                     merge_all_in_one=merge_all_in_one,
                     propagate_weights=True, single_pol=single_pol_merge)
        else:
            merge_h5(h5_out=parmdbmergename, h5_tables=parmdbmergelist[msnumber][::-1], ms_files=ms,
                     convert_tec=True, merge_all_in_one=merge_all_in_one,
                     propagate_weights=True, single_pol=single_pol_merge)
        # add CS stations back for superstation
        if mslist_beforephaseup is not None:
            print('mslist_beforephaseup: ' + mslist_beforephaseup[msnumber])
            if is_scalar_array_for_wsclean([parmdbmergename]):
                single_pol_merge = True
            else:
                single_pol_merge = False
            merge_h5(h5_out=parmdbmergename.replace("selfcalcycle",
                                                    "addCS_selfcalcycle"), h5_tables=parmdbmergename,
                     ms_files=mslist_beforephaseup[msnumber], convert_tec=True,
                     merge_all_in_one=merge_all_in_one, single_pol=single_pol_merge,
                     propagate_weights=True, add_cs=True)

        # make LINEAR solutions from CIRCULAR (never do a single_pol merge here!)
        if ('scalarphasediff' in args['soltype_list']) or ('scalarphasediffFR' in args['soltype_list']) or args['docircular']:
            merge_h5(h5_out=parmdbmergename_pc, h5_tables=parmdbmergename, circ2lin=True,
                     propagate_weights=True)
            # add CS stations back for superstation
            if mslist_beforephaseup is not None:
                merge_h5(h5_out=parmdbmergename_pc.replace("selfcalcycle",
                                                           "addCS_selfcalcycle"),
                         h5_tables=parmdbmergename_pc,
                         ms_files=mslist_beforephaseup[msnumber], convert_tec=True,
                         merge_all_in_one=merge_all_in_one,
                         propagate_weights=True, add_cs=True)

        if False:
            # testing only to check if merged H5 file is correct and makes a good image
            applycal(ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])

        # plot merged solution file
        print('single_pol_merge',single_pol_merge)
        losotoparset = create_losoto_flag_apgridparset(ms, flagging=False,
                                                       medamp=get_median_amp(parmdbmergename),
                                                       outplotname=
                                                       parmdbmergename.split('_' + os.path.basename(ms) + '.h5')[0],
                                                       refant=findrefant_core(parmdbmergename),
                                                       fulljones=fulljonesparmdb(parmdbmergename),
                                                       onepol=single_pol_merge)
        run('losoto ' + parmdbmergename + ' ' + losotoparset)
        force_close(parmdbmergename)

        ## --- start STACK code ---
        if args['stack']:
            # for each stacked MS (one per common time axis), apply the solutions to each direction MS
            for orig_ms in mss_timestacks[msnumber]:
                applycal(orig_ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])
        ## --- end STACK code ---

    if args['QualityBasedWeights'] and selfcalcycle >= args['QualityBasedWeights_start']:
        for ms in mslist:
            run('python3 NeReVar.py --filename=' + ms + \
                ' --dt=' + str(args['QualityBasedWeights_dtime']) + ' --dnu=' + str(args['QualityBasedWeights_dfreq']) + \
                ' --DiagDir=plotlosoto' + ms + '/NeReVar/ --basename=_selfcalcycle' + str(selfcalcycle).zfill(
                3) + ' --modelcol=MODEL_DATA')


    if len(modeldatacolumns) > 0:
        np.save('wsclean_h5list' + str(selfcalcycle).zfill(3) + '.npy', wsclean_h5list)
        return wsclean_h5list
    else:
        return []


def predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=False, sources=None, beamproximitylimit=240.0, modelstoragemanager=None):
    cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msout=. '
    cmd += 'p.sourcedb=' + skymodel + ' steps=[p] p.type=predict msout.datacolumn=' + modeldata + ' '
    if sources is not None:
        cmd += 'p.sources=[' + str(sources) + '] '
    if predictskywithbeam:
        cmd += 'p.usebeammodel=True p.usechannelfreq=True p.beammode=array_factor '
        cmd += 'p.beamproximitylimit=' + str(beamproximitylimit) + ' '
    elif modelstoragemanager is not None:
        cmd += 'msout.storagemanager=' + modelstoragemanager + ' '
    print(cmd)
    run(cmd)


def runDPPPbase(ms, solint, nchan, parmdb, soltype, uvmin=1.,
                SMconstraint=0.0, SMconstraintreffreq=0.0,
                SMconstraintspectralexponent=-1.0, SMconstraintrefdistance=0.0, antennaconstraint=None,
                resetsols=None, resetsols_list=[None], resetdir=None,
                resetdir_list=[None], restoreflags=False,
                maxiter=100, tolerance=1e-4, flagging=False, skymodel=None, flagslowphases=True,
                flagslowamprms=7.0, flagslowphaserms=7.0, incol='DATA',
                predictskywithbeam=False, BLsmooth=False, skymodelsource=None,
                skymodelpointsource=None, wscleanskymodel=None, iontimefactor=0.01, ionfreqfactor=1.0,
                blscalefactor=1.0, dejumpFR=False, uvminscalarphasediff=0, selfcalcycle=0, dysco=True,
                blsmooth_chunking_size=8, soltypenumber=0, create_modeldata=True,
                clipsolutions=False, clipsolhigh=1.5, clipsollow=0.667,
                ampresetvalfactor=10., flag_ampresetvalfactor=False, uvmax=None,
                modeldatacolumns=[], solveralgorithm='directioniterative', solveralgorithm_dde='directioniterative',
                preapplyH5_dde=[],
                dde_skymodel=None, DDE_predict='WSCLEAN', beamproximitylimit=240.,
                ncpu_max=24, bdaaverager=False, DP3_dual_single=True, soltype_list=None, soltypelist_includedir=None,
                normamps=True, modelstoragemanager=None, pixelscale=None, imsize=None, skymodelsetjy=False,
                solve_msinnchan='all', solve_msinstartchan=0,
                antenna_averaging_factors=None, antenna_smoothness_factors=None):
    soltypein = soltype  # save the input soltype is as soltype could be modified (for example by scalarphasediff)

    with table(ms + '/OBSERVATION', ack=False) as t:
        telescope = t.getcol('TELESCOPE_NAME')[0]

    modeldata = 'MODEL_DATA'  # the default, update if needed for scalarphasediff and phmin solves
    if BLsmooth:
        # Open the measurement set and check column names
        with table(ms, ack=False) as t:
            colnames = t.colnames()

        # If 'SMOOTHED_DATA' column doesn't exist, run the BLsmooth command
        if 'SMOOTHED_DATA' not in colnames:
            blsmooth_command = (
                f"python {submodpath}/BLsmooth.py -n 8 -c {blsmooth_chunking_size} -i {incol} "
                f"-o SMOOTHED_DATA -f {iontimefactor} -s {blscalefactor} "
                f"-u {ionfreqfactor} {ms}"
            )
            print()
            run(blsmooth_command)

        incol = 'SMOOTHED_DATA'

    if skymodelsetjy and create_modeldata and len(modeldatacolumns) == 0:
        setjy_casa(ms)
    
     
    if skymodel is not None and create_modeldata and len(modeldatacolumns) == 0:
        predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam, sources=skymodelsource, modelstoragemanager=modelstoragemanager)

    # if wscleanskymodel is not None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR' and create_modeldata:
    if wscleanskymodel is not None and create_modeldata and len(modeldatacolumns) == 0:
        makeimage([ms], wscleanskymodel, 1., 1., len(glob.glob(wscleanskymodel + '-????-model.fits')),
                  0, 0.0, onlypredict=True, idg=False)

    # if skymodelpointsource is not None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR' and create_modeldata:
    if skymodelpointsource is not None and create_modeldata and len(modeldatacolumns) == 0:
        # create MODEL_DATA (no dysco!)
        if modelstoragemanager is None:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]')
        else:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + modelstoragemanager + ' steps=[]')
        # do the predict with taql
        run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource) + "+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource) + "+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'")

    if soltype == 'scalarphasediff' or soltype == 'scalarphasediffFR':
        # PM means point source model adjusted weights
        create_weight_spectrum(ms, 'WEIGHT_SPECTRUM_PM', updateweights_from_thiscolumn='MODEL_DATA',
                               updateweights=False)  # always do to re-initialize WEIGHT_SPECTRUM_PM (because stack.MS is re-created each selfcalcycle, also MODEL_DATA changes
        # for now use updateweights=False, seems to give better results for scalarphasediff type solves
        # updateweights means WEIGHT_SPECTRUM_PM is updated based on MODEL_DATA**2, however so far this does not seem to give better results
        # update July 2024, tested on Perseus LBA with updateweights True and False, again difference is small

        # check if colnames are there each time because of stack.MS
        t = table(ms, ack=False)
        colnames = t.colnames()
        t.close()  # needs a close here because below were are writing columns potentially
        if 'DATA_CIRCULAR_PHASEDIFF' not in colnames:
            create_phasediff_column(ms, incol=incol, dysco=dysco)
        if 'MODEL_DATA_PDIFF' not in colnames:
            create_MODEL_DATA_PDIFF(ms, modelstoragemanager)  # make a point source
        soltype = 'phaseonly'  # do this type of solve, maybe scalarphase is fine? 'scalarphase' #
        incol = 'DATA_CIRCULAR_PHASEDIFF'
        modeldata = 'MODEL_DATA_PDIFF'

    if soltype in ['phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin', 'scalarphase_phmin']:
        create_phase_column(ms, incol=incol, outcol='DATA_PHASEONLY', dysco=dysco)
        create_phase_column(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASEONLY', dysco=False)
        soltype = soltype.split('_phmin')[0]
        incol = 'DATA_PHASEONLY'
        modeldata = 'MODEL_DATA_PHASEONLY'

    if soltype in ['phaseonly_slope', 'scalarphase_slope']:
        create_phase_slope(ms, incol=incol, outcol='DATA_PHASE_SLOPE', ampnorm=True, dysco=dysco)
        create_phase_slope(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASE_SLOPE', ampnorm=True, dysco=dysco)
        soltype = soltype.split('_slope')[0]
        incol = 'DATA_PHASE_SLOPE'
        modeldata = 'MODEL_DATA_PHASE_SLOPE'
        # udpate weights according to weights * (MODEL_DATA/MODEL_DATA_PHASE_SLOPE)**2
        create_weight_spectrum_modelratio(ms, 'WEIGHT_SPECTRUM_PM',
                                          updateweights=True, originalmodel='MODEL_DATA',
                                          newmodel='MODEL_DATA_PHASE_SLOPE', backup=True)

    if soltype in ['phaseonly', 'complexgain', 'fulljones', 'rotation+diagonal', 'amplitudeonly',
                   'rotation+diagonalamplitude',
                   'rotation+diagonalphase', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude']:  # for 1D plotting
        onepol = False
    if soltype in ['scalarphase', 'tecandphase', 'tec', 'scalaramplitude',
                   'scalarcomplexgain', 'rotation', 'rotation+scalar',
                   'rotation+scalarphase', 'rotation+scalaramplitude', 'faradayrotation', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
        onepol = True

    if restoreflags:
        cmdtaql = "'update " + ms + " set FLAG=FLAG_BACKUP'"
        print("Restore flagging column: " + "taql " + cmdtaql)
        run("taql " + cmdtaql)

    t = table(ms + '/SPECTRAL_WINDOW', ack=False)
    freq = np.median(t.getcol('CHAN_FREQ')[0])
    t.close()

    t = table(ms + '/ANTENNA', ack=False)
    antennasms = t.getcol('NAME')
    t.close()

    t = table(ms, readonly=True, ack=False)
    ms_ntimes = len(np.unique(t.getcol('TIME')))
    t.close()

    if telescope == 'LOFAR':
        if freq > 100e6:
            HBAorLBA = 'HBA'
        else:
            HBAorLBA = 'LBA'
        print('This is', HBAorLBA, 'data')
    else:
        HBAorLBA = 'other'
    print('This ms contains', antennasms)

    # determine if phases needs to be included, important if slowgains do not contain phase solutions
    includesphase = True
    if soltype == 'scalaramplitude' or soltype == 'amplitudeonly' \
            or soltype == 'rotation+diagonalphase' or soltype == 'rotation+scalarphase' \
            or soltype == 'faradayrotation+diagonalphase' or soltype == 'faradayrotation+scalarphase':
        includesphase = False

    # figure out which weight_spectrum column to use
    if soltypein == 'scalarphasediff' or soltypein == 'scalarphasediffFR' or \
            soltypein == 'phaseonly_slope' or soltypein == 'scalarphase_slope':
        weight_spectrum = 'WEIGHT_SPECTRUM_PM'
    else:
        # check for WEIGHT_SPECTRUM_SOLVE from DR2 products
        t = table(ms, ack=False)
        if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
            weight_spectrum = 'WEIGHT_SPECTRUM_SOLVE'
        else:
            weight_spectrum = 'WEIGHT_SPECTRUM'
        t.close()

        # check for previous old parmdb and remove them
    if os.path.isfile(parmdb):
        print('H5 file exists  ', parmdb)
        os.system('rm -f ' + parmdb)

    cmd = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(), ncpu_max])) + \
          ' msin=' + ms + ' msout=. '
    
    if solve_msinnchan != 'all':
        SMconstraint = 0.0  # set SMconstraint to 0 so that DP3 does not solve for frequency dependence
        nchan = 0  # set nchan to 0 so that DP3 does not solve for frequency dependence

    if soltype == 'rotation+diagonal':
        cmd += 'ddecal.rotationdiagonalmode=diagonal '
    if soltype == 'rotation+diagonalamplitude':
        cmd += 'ddecal.rotationdiagonalmode=diagonalamplitude '
    if soltype == 'rotation+diagonalphase':
        cmd += 'ddecal.rotationdiagonalmode=diagonalphase '
    if soltype == 'rotation+scalar':  # =scalarcomplexgain
        cmd += 'ddecal.rotationdiagonalmode=scalar '
    if soltype == 'rotation+scalaramplitude':
        cmd += 'ddecal.rotationdiagonalmode=scalaramplitude '
    if soltype == 'rotation+scalarphase':
        cmd += 'ddecal.rotationdiagonalmode=scalarphase '

    if soltype == 'faradayrotation+diagonal':
        cmd += 'ddecal.faradaydiagonalmode=diagonal '
    if soltype == 'faradayrotation+diagonalamplitude':
        cmd += 'ddecal.faradaydiagonalmode=diagonalamplitude '
    if soltype == 'faradayrotation+diagonalphase':
        cmd += 'ddecal.faradaydiagonalmode=diagonalphase '
    if soltype == 'faradayrotation+scalar':
        cmd += 'ddecal.faradaydiagonalmode=scalar '
    if soltype == 'faradayrotation+scalaramplitude':
        cmd += 'ddecal.faradaydiagonalmode=scalaramplitude '
    if soltype == 'faradayrotation+scalarphase':
        cmd += 'ddecal.faradaydiagonalmode=scalarphase '


    # deal with the special cases because DP3 only knows soltype rotation+diagonal (it uses rotationdiagonalmode)
    if soltype in ['rotation+diagonalamplitude', 'rotation+diagonalphase', \
                   'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase']:
        cmd += 'ddecal.mode=rotation+diagonal '
        # cmd += 'ddecal.rotationreference=True '
    elif soltype in ['faradayrotation+diagonalamplitude', 'faradayrotation+diagonalphase', \
                   'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
        cmd += 'ddecal.mode=faradayrotation '  # 
    else:
        cmd += 'ddecal.mode=' + soltype + ' '

    # note there are two things to check before we can use ddecal.datause=single/dual
    # 1 current solve needs to be of the correct type
    # 2 previous solves should not violate the assumptions of the current single/dual solve
    if soltype_list is not None:
        print(f"soltype_list slice: {soltype_list[0:soltypenumber]}")
        print(soltype, soltype_list)
        if DP3_dual_single:
            if soltype in ['complexgain', 'amplitudeonly', 'phaseonly'] \
                    and 'fulljones' not in soltype_list[0:soltypenumber] \
                    and 'rotation' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonal' not in soltype_list[0:soltypenumber]:
                cmd += 'ddecal.datause=dual '
            if soltype in ['scalarcomplexgain', 'scalaramplitude', \
                           'tec', 'tecandphase', 'scalarphase'] \
                    and 'fulljones' not in soltype_list[0:soltypenumber] \
                    and 'rotation' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'complexgain' not in soltype_list[0:soltypenumber] \
                    and 'amplitudeonly' not in soltype_list[0:soltypenumber] \
                    and 'phaseonly' not in soltype_list[0:soltypenumber]: \
                    cmd += 'ddecal.datause=single '

    cmd += 'msin.weightcolumn=' + weight_spectrum + ' '
    if bdaaverager and pixelscale is not None and imsize is not None:
        cmd += 'steps=[bda,ddecal] ddecal.type=ddecal bda.type=bdaaverager '
    else:
        cmd += 'steps=[ddecal] ddecal.type=ddecal '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '

    modeldatacolumns_solve = []  # empty, will be filled below if applicable
    dir_id_kept = []  # empty, will be filled below if applicable
    if len(modeldatacolumns) > 0:
        if DDE_predict == 'DP3' and soltypelist_includedir is not None:
            print('DDE_predict with soltypelist_includedir is not supported')
            raise Exception('DDE_predict with soltypelist_includedir is not supported')

        if soltypelist_includedir is not None:
            modeldatacolumns_solve, sourcedir_removed, dir_id_kept = updatemodelcols_includedir(modeldatacolumns,
                                                                                                soltypenumber,
                                                                                                soltypelist_includedir,
                                                                                                ms, modelstoragemanager=modelstoragemanager)

        if DDE_predict == 'DP3':
            cmd += 'ddecal.sourcedb=' + dde_skymodel + ' '
            if telescope == 'LOFAR':  # predict with array factor for LOFAR data
                cmd += 'ddecal.usebeammodel=True '
                cmd += 'ddecal.usechannelfreq=True ddecal.beammode=array_factor '
                cmd += 'ddecal.beamproximitylimit=' + str(beamproximitylimit) + ' '
        else:
            if len(modeldatacolumns_solve) > 0:  # >0 (and not > 1 because we can have 1 direction left)
                cmd += "ddecal.modeldatacolumns='[" + ','.join(map(str, modeldatacolumns_solve)) + "]' "
            else:
                cmd += "ddecal.modeldatacolumns='[" + ','.join(map(str, modeldatacolumns)) + "]' "
        if len(modeldatacolumns) > 1:  # so we are doing a dde solve
            cmd += 'ddecal.solveralgorithm=' + solveralgorithm_dde + ' '
        else:  # in case the list still has length 1
            cmd += 'ddecal.solveralgorithm=' + solveralgorithm + ' '
    else:
        cmd += "ddecal.modeldatacolumns='[" + modeldata + "]' "
        cmd += 'ddecal.solveralgorithm=' + solveralgorithm + ' '

    cmd += 'ddecal.maxiter=' + str(int(maxiter)) + ' ddecal.propagatesolutions=True '
    # Do list comprehension if solint is a list
    if type(solint) == list:
        if len(dir_id_kept) > 0:
            print(solint)
            print(dir_id_kept)
            solint = [solint[i] for i in dir_id_kept]  # overwrite solint, selecting on the directions kept
        solints = [int(format_solint(x, ms)) for x in solint]
        solints = tweak_solints(solints, ms_ntimes=ms_ntimes)
        lcm = math.lcm(*solints)
        divisors = [int(lcm / i) for i in solints]
        cmd += 'ddecal.solint=' + str(lcm) + ' '
        cmd += 'ddecal.solutions_per_direction=' + "'" + str(divisors).replace(' ', '') + "' "
    else:
        solint_integer = format_solint(solint, ms)  # create the integer number for DP3
        cmd += 'ddecal.solint=' + str(tweak_solints_single(int(solint_integer), ms_ntimes)) + ' '
    cmd += 'ddecal.nchan=' + format_nchan(nchan, ms) + ' '
    cmd += 'ddecal.h5parm=' + parmdb + ' '


    # format antenna_averaging_factors
    if antenna_averaging_factors is not None:
        antenna_averaging_factors_splitstr = antenna_averaging_factors.split(',')
        groupstr_all = []
        antenna_averaging_factors_new = [] 
        for antgroup in antenna_averaging_factors_splitstr:
            groupstr = antennaconstraintstr(antgroup.split(':')[0], antennasms, HBAorLBA, telescope=telescope, useforresetsols=True)
            groupstr_all = groupstr_all + groupstr
            antenna_averaging_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + antgroup.split(':')[1])
                
        if len(groupstr_all) != len(set(groupstr_all)):
            print('There are duplicate antennas in antenna_averaging_factors, please check your input')
            raise Exception('There are duplicate antennas in antenna_averaging_factors, please check your input')
        groupstr_complement = list(set(groupstr_all) ^ set (antennasms))  # get the complement of the antenna group
        if len(groupstr_complement) > 0: 
            antenna_averaging_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:1')  # add the complement antennas with average factor 1
            
        cmd += 'ddecal.antenna_averaging_factors=' + '[' + ','.join(antenna_averaging_factors_new) + '] '

    if bdaaverager  and pixelscale is not None and imsize is not None:
        cmd += 'bda.frequencybase= ' + 'bda.minchannels=' + format_nchan(nchan, ms) + ' '
        if type(solint) == list:
            cmd += 'bda.timebase= ' + 'bda.maxinterval=' + int(lcm / np.max(divisors)) + ' '
        else:
            cmd += 'bda.timebase= ' + 'bda.maxinterval=' + format_solint(solint, ms) + ' '

    # preapply H5 from previous pertubation for DDE solves with DP3
    if (len(modeldatacolumns) > 1) and (len(preapplyH5_dde) > 0):
        if DDE_predict == 'DP3':
            cmd += build_applycal_dde_cmd(preapplyH5_dde) + ' '
        else:
            cmd += 'msin.datacolumn=DATA '  # to prevent solving out of CORRECTED_PREAPPLY$N
    else:
        cmd += 'msin.datacolumn=' + incol + ' '

    # SET UVMIN
    if soltypein == 'scalarphasediff' or soltypein == 'scalarphasediffFR':
        if uvminscalarphasediff is not None:
            cmd += 'ddecal.uvlambdamin=' + str(uvminscalarphasediff) + ' '
        else:
            if uvmin != 0:
                cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '
    else:
        if uvmin != 0:
            cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '
        if uvmax is not None:  # no need to see uvlambdamax for scalarphasediff solves since there we always solve against a point source
            cmd += 'ddecal.uvlambdamax=' + str(uvmax) + ' '

    if antennaconstraint is not None:
        cmd += 'ddecal.antennaconstraint=' + antennaconstraintstr(antennaconstraint, antennasms, HBAorLBA,
                                                                  telescope=telescope) + ' '

    # format antenna_smoothness_factors
    if antenna_smoothness_factors is not None:
        antenna_smoothness_factors_splitstr = antenna_smoothness_factors.split(',')
        groupstr_all = []
        smoothness_factors = []
        for antgroup in antenna_smoothness_factors_splitstr:
            smoothness_factors.append(antgroup.split(':')[1])
        print('smoothness_factors', smoothness_factors)
        smoothness_factors_float = list(map(float, smoothness_factors))
        antenna_smoothness_factors_new = [] 
        for antgroup in antenna_smoothness_factors_splitstr:

            groupstr = antennaconstraintstr(antgroup.split(':')[0], antennasms, HBAorLBA, telescope=telescope, useforresetsols=True)
            groupstr_all = groupstr_all + groupstr
            if np.max(smoothness_factors_float) <= 1.0:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + antgroup.split(':')[1])
            else:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + \
                                                      str(float(antgroup.split(':')[1])/np.max(smoothness_factors_float)))
                  
        if len(groupstr_all) != len(set(groupstr_all)):
            print('There are duplicate antennas in antenna_smoothness_factors, please check your input')
            raise Exception('There are duplicate antennas in antenna_smoothness_factors, please check your input')
        groupstr_complement = list(set(groupstr_all) ^ set (antennasms))  # get the complement of the antenna group
        if len(groupstr_complement) > 0: 
            if np.max(smoothness_factors_float) <= 1.0:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:1.0')  # add the complement antennas with factor 1.0
            else:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:'
                                                      + str(1.0/np.max(smoothness_factors_float)))  # add the complement antennas with factor 1.0/maximum smooth
            
        if np.max(smoothness_factors_float) > 1.0: # handle the case where the smoothness factors are larger than 1.0
            if type(SMconstraint) == list:
                SMconstraint = [sf*np.max(smoothness_factors_float) for sf in SMconstraint]  # increase SMconstraint with the maximum smoothness factor
            else:
                SMconstraint = np.max(smoothness_factors_float) * SMconstraint  # increase SM

        cmd += 'ddecal.antenna_smoothness_factors=' + '[' + ','.join(antenna_smoothness_factors_new) + '] '


    if np.max(SMconstraint) > 0.0 and nchan != 0:
        if type(SMconstraint) == list:
            if len(dir_id_kept) > 0:
                print(SMconstraint)
                print(dir_id_kept)
                SMconstraint = [SMconstraint[i] for i in dir_id_kept]  # overwrite SMconstraint, selecting on the directions kept
            smoothness_dd_factors = [ ddsf/np.max(SMconstraint) for ddsf in SMconstraint]  
            cmd += 'ddecal.smoothness_dd_factors=' + "'" + str(smoothness_dd_factors).replace(' ', '') + "' "
        cmd += 'ddecal.smoothnessconstraint=' + str(np.max(SMconstraint) * 1e6) + ' '
        cmd += 'ddecal.smoothnessreffrequency=' + str(SMconstraintreffreq * 1e6) + ' '
        cmd += 'ddecal.smoothnessspectralexponent=' + str(SMconstraintspectralexponent) + ' '
        cmd += 'ddecal.smoothnessrefdistance=' + str(SMconstraintrefdistance * 1e3) + ' '  # input units in km


    if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation',
                   'rotation+scalarphase', 'rotation+diagonalphase', 'faradayrotation+scalarphase', 'faradayrotation+diagonalphase', 'faradayrotation']:
        cmd += 'ddecal.tolerance=' + str(tolerance) + ' '
        if soltype in ['tecandphase', 'tec']:
            cmd += 'ddecal.approximatetec=True '
            cmd += 'ddecal.stepsize=0.2 '
            cmd += 'ddecal.maxapproxiter=45 '
            cmd += 'ddecal.approxtolerance=6e-3 '
    if soltype in ['complexgain', 'scalarcomplexgain', 'scalaramplitude', 'amplitudeonly',
                   'rotation+diagonal', 'fulljones', 'rotation+scalar',
                   'rotation+diagonalamplitude', 'rotation+scalaramplitude', 'faradayrotation+diagonal', 'faradayrotation+scalar', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalaramplitude']:
        cmd += 'ddecal.tolerance=' + str(tolerance) + ' '  # for now the same as phase soltypes
    # cmd += 'ddecal.detectstalling=False '


    # DETERMINE IF WE CAN USE SOLUTIONS FROM PREVIOUS SELFCAL CYCLE
    if args['startfrominitialsolutions']:
        print('Checking if a solution file from a previous selfcalcycle is present...')
        # update_sourcedir_h5_dde() will cause issues? 
        initialsolutions_exist = False
        initialsolutions_directions_equal = False
        current_ndir = len(modeldatacolumns)
        if (len(modeldatacolumns_solve) > 0) and (len(modeldatacolumns) != len(modeldatacolumns_solve)):
            # means that we require parmdb + .backup 
            # number directions being solved for is different as soltypelist_includedir functionality is used    
            current_ndir = len(modeldatacolumns_solve)
            parmdbtmp = parmdb + '.backup'
        else:
            parmdbtmp = parmdb
            
        if selfcalcycle == 0:  # do deal with the "skyselfcalcycle if it exists"
            previous_parmdb = parmdbtmp.replace('selfcalcycle'+str(0).zfill(3),'skyselfcalcycle'+str(0).zfill(3))
            if os.path.isfile(previous_parmdb): initialsolutions_exist = True
                    
        else:
            previous_parmdb = parmdbtmp.replace('selfcalcycle'+str(selfcalcycle).zfill(3),                                  'selfcalcycle'+str(selfcalcycle-1).zfill(3))
            if os.path.isfile(previous_parmdb): initialsolutions_exist = True
                
        if initialsolutions_exist: # we autatically guarantee that the soltype is the same because we check for the parmdb name which contains the soltype in there 
        # check n_dir equals current number of dirctions to solve
            print('Found initial solution file:',previous_parmdb)
            with tables.open_file(previous_parmdb) as Hprev:
                for soltab in Hprev.root.sol000._v_groups.keys():
                    previous_ndir = len(Hprev.root.sol000._f_get_child(soltab).dir[:])
                print('Number of directions in ' + previous_parmdb + ':',previous_ndir)     
            if previous_ndir <= 1 and current_ndir <=1: #this is a DI solve
                initialsolutions_directions_equal = True
            else:
               if len(previous_ndir) == len(current_ndir): 
                   initialsolutions_directions_equal = True # this is a DD solve
                   # number of directions did not change from previous selfcalcycle     

        if initialsolutions_exist and initialsolutions_directions_equal: 
            # if we get here it means we can can use ddecal.initialsolutions
            cmd += 'ddecal.initialsolutions.h5parm=' + previous_parmdb + ' '
            if soltypein == 'fulljones': 
                # just for extra safety 
                # cmd += 'ddecal.initialsolutions.gaintype=fulljones ' 
                # for all soletypes DP3 should be able to figure it out in principle by itself 
                cmd += 'initialsolutions.soltab=[amplitude000,phase000] '
            # set ddecal.initialsolutions.soltab
            else:
                with tables.open_file(previous_parmdb) as Hprev:
                   soltabs = list(Hprev.root.sol000._v_groups.keys())
                   if 'error000' in soltabs: soltabs.remove('error000')
                   
                cmd += "ddecal.initialsolutions.soltab='[" + ','.join(map(str, soltabs)) + "]' "

    ms_tmp = None # set to None so that we can check if it is created later
    if solve_msinnchan != 'all':
        # if we are solving for aa select channel range, we need to create a temporary MS with the average function
        
        # create a list of the same columns as in the original MS 
        columns_to_create = []
        if incol != 'DATA':
            columns_to_create.append(incol)
        if len(modeldatacolumns) == 0:
            columns_to_create.append(modeldata)    
        else:
            if len(modeldatacolumns_solve) > 0:
                for mdc in modeldatacolumns_solve:
                    columns_to_create.append(mdc)
            else:
                for mdc in modeldatacolumns:
                    columns_to_create.append(mdc)
        columns_to_create.append('DATA') # to create DATA because XY and YX were set to zero above
        
        ms_tmp = create_splitted_ms(ms, columns_to_create, solve_msinnchan=solve_msinnchan, 
                                    solve_msinstartchan=solve_msinstartchan, dysco=dysco, 
                                    modelstoragemanager=modelstoragemanager, incol=incol, 
                                    metadata_compression=args['metadata_compression'])

        # now replace the MS in the cmd command string with the new temporary MS
        cmd = cmd.replace('msin=' + ms, 'msin=' + ms_tmp)
        
    
    # RUN THE SOLVE WITH DP3
    print('DP3 solve:', cmd)
    logger.info('DP3 solve: ' + cmd)
    run(cmd)

    if ms_tmp is not None:
        # remove the temporary MS
        print('Removing temporary MS:', ms_tmp)
        os.system('rm -rf ' + ms_tmp)

    if selfcalcycle == 0 and (soltypein == "scalarphasediffFR" or soltypein == "scalarphasediff") and not args['phasediff_only']:
        os.system("cp -r " + parmdb + " " + parmdb + ".scbackup")

    if (len(modeldatacolumns_solve) > 0) and (len(modeldatacolumns) != len(modeldatacolumns_solve)):
        # fix coordinates otherwise h5merge will merge all directions into one when add_directions is done (as all coordinates are the same up to this point)
        update_sourcedir_h5_dde(parmdb, 'facetdirections.p', dir_id_kept=dir_id_kept)

        # we need to add back the extra direction into the h5 file
        outparmdb = 'adddirback' + parmdb
        if os.path.isfile(outparmdb):
            os.system('rm -f ' + outparmdb)
        merge_h5(h5_out=outparmdb, h5_tables=parmdb, add_directions=sourcedir_removed.tolist(), propagate_weights=False, convert_tec=False)

        # now we split them all into separate h5 per direction so we can reorder and fill them
        print('Splitting directions into separate h5')
        split_multidir(outparmdb)

        # fill the added emtpy directions with the closest ones that were solved for
        print('Copy over solutions from skipped directions')
        copy_over_solutions_from_skipped_directions(modeldatacolumns, dir_id_kept)

        # create backup of parmdb and remove orginal and cleanup
        os.system('cp -f ' + parmdb + ' ' + parmdb + '.backup')
        os.system('rm -f ' + parmdb)
        os.system('rm -f ' + outparmdb)

        # merge h5 files in order of the directions in facetdirections.p and recreate parmdb
        # clean up previously splitted directions inside this function
        print('Merge h5 files in correct order and recreate parmdb')
        merge_splitted_h5_ordered(modeldatacolumns, parmdb, clean_up=True)

        # fix direction names
        update_sourcedirname_h5_dde(parmdb, modeldatacolumns)
        # sys.exit()

    if len(modeldatacolumns) > 1:  # and DDE_predict == 'WSCLEAN':
        update_sourcedir_h5_dde(parmdb, 'facetdirections.p')

    if has0coordinates(parmdb):
        logger.warning('Direction coordinates are zero in: ' + parmdb)

    # Rotation checking
    if soltype in ['rotation', 'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase']:
        remove_nans(parmdb, 'rotation000')
        fix_weights_rotationh5(parmdb)
        refant = findrefant_core(parmdb)
        force_close(parmdb)
        fix_rotationreference(parmdb, refant)

    # Faraday rotation checking
    if soltype in ['faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude',
                   'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
        remove_nans(parmdb, 'rotationmeasure000')
        fix_weights_rotationmeasureh5(parmdb)
        refant = findrefant_core(parmdb)
        force_close(parmdb)
        fix_rotationmeasurereference(parmdb, refant)

    # tec checking
    if soltype in ['tec', 'tecandphase']:
        remove_nans(parmdb, 'tec000')
        refant = findrefant_core(parmdb)
        fix_tecreference(parmdb, refant)
        force_close(parmdb)

    # phase checking
    if soltype in ['rotation+diagonal', 'rotation+diagonalphase',
                   'rotation+scalar', 'rotation+scalarphase',
                   'scalarcomplexgain', 'complexgain', 'scalarphase',
                   'phaseonly', 'faradayrotation+diagonal', 'faradayrotation+scalar', 'faradayrotation+diagonalphase', 'faradayrotation+scalarphase']:
        remove_nans(parmdb, 'phase000')
        refant = findrefant_core(parmdb)
        fix_phasereference(parmdb, refant)
        force_close(parmdb)

    if int(maxiter) == 1:  # this is a template solve only
        print('Template solve, not going to make plots or do solution flagging')
        return

    outplotname = parmdb.split('_' + os.path.basename(ms) + '.h5')[0]

    if incol == 'DATA_CIRCULAR_PHASEDIFF':
        print('Manually updating H5 to get the phase difference correct')
        refant = findrefant_core(parmdb)  # phase matrix plot
        force_close(parmdb)
        makephasediffh5(parmdb, refant)
    if incol == 'DATA_CIRCULAR_PHASEDIFF' and soltypein == 'scalarphasediffFR':
        print('Fiting for Faraday Rotation with losoto on the phase differences')
        # work with copies H5 because losoto changes the format splitting off the length 1 direction axis creating issues
        # with H5merge (also add additional solution talbes which we do not want)
        os.system('cp -f ' + parmdb + ' ' + 'FRcopy' + parmdb)
        losoto_parsetFR = create_losoto_FRparset(ms, refant=findrefant_core(parmdb), outplotname=outplotname,
                                                 dejump=dejumpFR)
        run('losoto ' + 'FRcopy' + parmdb + ' ' + losoto_parsetFR)
        rotationmeasure_to_phase('FRcopy' + parmdb, parmdb, dejump=dejumpFR)
        run('losoto ' + parmdb + ' ' + create_losoto_FRparsetplotfit(ms, refant=findrefant_core(parmdb),
                                                                     outplotname=outplotname))
        force_close(parmdb)

    if incol == 'DATA_PHASE_SLOPE':
        print('Manually updating H5 to get the cumulative phase')
        # makephaseCDFh5(parmdb)
        makephaseCDFh5_h5merger(parmdb, ms, modeldatacolumns)

    if resetsols is not None:
        if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'fulljones',
                       'complexgain', 'scalarcomplexgain', 'rotation+diagonal',
                       'rotation+diagonalamplitude', 'rotation+diagonalphase',
                       'rotation+scalar', 'rotation+scalarphase', 'rotation+scalaramplitude', 'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
            refant = findrefant_core(parmdb)
            force_close(parmdb)
        else:
            refant = None
        resetsolsforstations(parmdb, antennaconstraintstr(resetsols, antennasms, HBAorLBA, useforresetsols=True,
                                                          telescope=telescope), refant=refant)

    if resetdir is not None:
        if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'fulljones',
                       'complexgain', 'scalarcomplexgain', 'rotation+diagonal',
                       'rotation+diagonalamplitude', 'rotation+diagonalphase',
                       'rotation+scalar', 'rotation+scalarphase', 'rotation+scalaramplitude', 'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
            refant = findrefant_core(parmdb)
            force_close(parmdb)
        else:
            refant = None

        resetsolsfordir(parmdb, resetdir, refant=refant)

    if number_freqchan_h5(parmdb) > 1:
        onechannel = False
    else:
        onechannel = True

    # Check for bad values (amplitudes/fulljones)
    if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude',
                   'fulljones', 'rotation+diagonal', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude', 'faradayrotation+diagonal', 'faradayrotation+scalaramplitude','faradayrotation+diagonalamplitude','faradayrotation+scalar']:
        if resetdir is not None or resetsols is not None:
            flag_bad_amps(parmdb, setweightsphases=includesphase, flagamp1=False,
                          flagampxyzero=False)  # otherwise it flags the solutions which where reset
        else:
            flag_bad_amps(parmdb, setweightsphases=includesphase)
        if soltype == 'fulljones':
            removenans_fulljones(parmdb)
        else:
            remove_nans(parmdb, 'amplitude000')
        medamp = get_median_amp(parmdb)

        if soltype != 'amplitudeonly' and soltype != 'scalaramplitude' \
                and soltype != 'rotation+diagonalamplitude' \
                and soltype != 'rotation+scalaramplitude' \
                and soltype != 'faradayrotation+scalaramplitude' \
                and soltype != 'faradayrotation+diagonalamplitude' :
            remove_nans(parmdb, 'phase000')

        if soltype == 'fulljones':
            # if normamps: #and all(rsl is None for rsl in resetsols_list) and all(rdl is None for rdl in resetdir_list):
            # otherwise you get too much setting to 1 due to large amp deviations, in particular fullones on raw data which has very high correlator amps (with different ILT vals), also resets in that case cause issues (resets are ok if the amplitudes are close to 1). Hence using the normamps test seems the most logical choice
            flaglowamps_fulljones(parmdb, lowampval=medamp / ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                                  setweightsphases=includesphase)
            flaghighamps_fulljones(parmdb, highampval=medamp * ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                                   setweightsphases=includesphase)
        else:
            # if normamps: #and all(rsl is None for rsl in resetsols_list) and all(rdl is None for rdl in resetdir_list):
            # otherwise you get too much setting to 1 due to large amp deviations, in particular fullones on raw data which has very high correlator amps (with different ILT vals), also resets in that case cause issues (resets are ok if the amplitudes are close to 1).  Hence using the normamps test seems the most logical choice
            flaglowamps(parmdb, lowampval=medamp / ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor), setweightsphases=includesphase)
            flaghighamps(parmdb, highampval=medamp * ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                         setweightsphases=includesphase)

        if soltype == 'fulljones' and clipsolutions:
            print('Fulljones and solution clipping not supported')
            raise Exception('Fulljones and clipsolutions not implemtened')
        if clipsolutions:
            flaglowamps(parmdb, lowampval=clipsollow, flagging=True, setweightsphases=True)
            flaghighamps(parmdb, highampval=clipsolhigh, flagging=True, setweightsphases=True)

    # ---------------------------------
    # ---------------------------------
    # makes plots and do LOSOTO flagging
    if soltype in ['rotation', 'rotation+diagonal', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude',
                   'rotation+scalarphase', 'rotation+diagonalphase']:

        losotoparset_rotation = create_losoto_rotationparset(ms, onechannel=onechannel, outplotname=outplotname + 'ROT',
                                                             refant=findrefant_core(parmdb))  # phase matrix plot
        force_close(parmdb)
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_rotation
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)

        if soltype in ['rotation+scalarphase', 'rotation+diagonalphase', 'faradayrotation+scalarphase','faradayrotation+diagonalphase']:
            losotoparset_phase = create_losoto_fastphaseparset(ms, onechannel=onechannel, onepol=onepol,
                                                               outplotname=outplotname,
                                                               refant=findrefant_core(parmdb),onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb))  # phase matrix plot
            cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_phase
            force_close(parmdb)
            print(cmdlosoto)
            logger.info(cmdlosoto)
            run(cmdlosoto)

    if soltype in ['phaseonly', 'scalarphase'] and not args['phasediff_only']:
        losotoparset_phase = create_losoto_fastphaseparset(ms, onechannel=onechannel, onepol=onepol,
                                                           outplotname=outplotname,
                                                           refant=findrefant_core(parmdb), onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb))  # phase matrix plot
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_phase
        force_close(parmdb)
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)

    if soltype in ['tecandphase', 'tec']:
        tecandphaseplotter(parmdb, ms,
                           outplotname=outplotname)  # use own plotter because losoto cannot add tec and phase

    if soltype in ['tec']:
        losotoparset_tec = create_losoto_tecparset(ms, outplotname=outplotname,
                                                   refant=findrefant_core(parmdb),
                                                   markersize=compute_markersize(parmdb))
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_tec
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)
        force_close(parmdb)

    if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude',
                   'fulljones', 'rotation+diagonal', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude', 'faradayrotation+diagonal', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude']:
        print('Do flagging?:', flagging)
        if flagging and not onechannel and ntimesH5(parmdb) > 1 :
            if soltype == 'fulljones':
                print('Fulljones and flagging not implemtened')
                raise Exception('Fulljones and flagging not implemtened')
            else:
                losotoparset = create_losoto_flag_apgridparset(ms, flagging=True, maxrms=flagslowamprms,
                                                               maxrmsphase=flagslowphaserms,
                                                               includesphase=includesphase, onechannel=onechannel,
                                                               medamp=medamp, flagphases=flagslowphases, onepol=onepol,
                                                               outplotname=outplotname, refant=findrefant_core(parmdb), onetime=ntimesH5(parmdb)==1, markersize=compute_markersize(parmdb))
                force_close(parmdb)
        else:
            losotoparset = create_losoto_flag_apgridparset(ms, flagging=False, includesphase=includesphase,
                                                           onechannel=onechannel, medamp=medamp, onepol=onepol,
                                                           outplotname=outplotname,
                                                           refant=findrefant_core(parmdb),
                                                           fulljones=fulljonesparmdb(parmdb),onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb))
            force_close(parmdb)

        # MAKE losoto command
        if flagging:
            os.system('cp -f ' + parmdb + ' ' + parmdb + '.backup')
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)
    
    if is_multidir(parmdb):
        same_weights_multidir(parmdb)
    
    if len(tables.file._open_files.filenames) >= 1:  # for debugging
        print('End runDPPPbase, some HDF5 files are not closed:', tables.file._open_files.filenames)
        force_close(parmdb)
    return


def create_splitted_ms(ms, columns_to_create, solve_msinnchan, solve_msinstartchan,
                      dysco=True, modelstoragemanager=None, incol='DATA', 
                      metadata_compression=True, ncpu_max=8):
    """
    Create a temporary Measurement Set (MS) with selected frequency channels and specified columns.

    Parameters:
        ms (str): Path to the input Measurement Set.
        columns_to_create (list of str): List of column names to create and copy into the new MS.
        solve_msinnchan (int): Number of frequency channels to include in the split MS.
        solve_msinstartchan (int): Starting channel index for the split.
        dysco (bool, optional): Whether to use DYSCO compression for the output MS. Defaults to True.
        modelstoragemanager (str or None, optional): Storage manager to use for model columns. Defaults to None.
        incol (str, optional): Name of the input data column. Defaults to 'DATA'.
        metadata_compression (bool, optional): Whether to enable metadata compression. Defaults to True.
        ncpu_max (int, optional): Maximum number of CPU threads to use. Defaults to 8.

    Returns:
        str: Path to the newly created temporary Measurement Set.
    """
    # create a new temporary MS with the average function
    ms_tmp = average([ms], freqstep=[1], makecopy=True, msinnchan=solve_msinnchan, msinstartchan=solve_msinstartchan,
                        dysco=dysco, metadata_compression=metadata_compression)[0]
    
    # set DATA XY and YX to zero to avoid error from the Stokes-I storageManager
    run("taql" + " 'update " + ms_tmp + " set DATA[,1]=(0+0i)'")
    run("taql" + " 'update " + ms_tmp + " set DATA[,2]=(0+0i)'")
    
    for col in columns_to_create:                  
        cmdcol = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(), ncpu_max])) + \
                    ' msin=' + ms_tmp + ' msout=. '
    
        cmdcol += 'steps=[] msout.datacolumn=' + col + ' '
        if col == incol: # in this case we can use the dysco storage manager
            if dysco:
                cmdcol += 'msout.storagemanager=dysco '
                cmdcol += 'msout.storagemanager.weightbitrate=16 '
        elif modelstoragemanager is not None:           
            cmdcol += 'msout.storagemanager=' + modelstoragemanager + ' '
        print(cmdcol)
        run(cmdcol)
        # now copy over the column to the temporary MS
        print('=== Copying column ' + col + ' to temporary MS \n\n')
        tout = table(ms_tmp, ack=False, readonly=False)
        tin  = table(ms, ack=False, readonly=True)
        
        stepsize = 1000000
        if tin.nrows() < stepsize: stepsize = tin.nrows()
        for row in range(0, tin.nrows(), stepsize):
            datain = tin.getcol(col, startrow=row, nrow=stepsize, rowincr=1)
            tout.putcol(col, datain[:,solve_msinstartchan:solve_msinstartchan + solve_msinnchan,:], startrow=row, nrow=stepsize, rowincr=1)
        tout.close()
        tin.close()    

    return ms_tmp

def mask_region_inv(infilename, ds9region, outfilename):
    """
    Applies an inverse mask to a FITS image using a DS9 region file, setting all pixels outside the specified region to zero.

    Parameters:
        infilename (str): Path to the input FITS file.
        ds9region (str): Path to the DS9 region file defining the region to keep.
        outfilename (str): Path to the output FITS file where the masked image will be saved.

    Returns:
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


def remove_outside_box(mslist, imagebasename, pixsize, imsize,
                       channelsout, single_dual_speedup=True,
                       outcol='SUBTRACTED_DATA', dysco=True, userbox=None,
                       idg=False, h5list=[], facetregionfile=None,
                       disable_primary_beam=False, ddcor=True, modelstoragemanager=None, parallelgridding=1,
                       metadata_compression=True):
    """
    Removes emission outside a specified box region from measurement sets (MS) by predicting and subtracting the model
    within the defined region. Optionally applies direction-dependent calibration corrections and manages output columns.
    Parameters
    ----------
    mslist : list of str
        List of measurement set file paths to process.
    imagebasename : str
        Basename for the image FITS file (expects '-MFS-image.fits' suffix).
    pixsize : float
        Pixel size for imaging (arcseconds or degrees, depending on context).
    imsize : int
        Image size (number of pixels per side).
    channelsout : int
        Number of output channels for imaging.
    single_dual_speedup : bool, optional
        If True, enables speedup for single/dual polarization (default: True).
    outcol : str, optional
        Name of the output data column to store subtracted data (default: 'SUBTRACTED_DATA').
    dysco : bool, optional
        If True, uses DYSCO storage manager for output (default: True).
    userbox : float, str, or None, optional
        User-specified box size in degrees, a region file, 'keepall', or None to use default (default: None).
    idg : bool, optional
        If True, uses IDG for imaging (default: False).
    h5list : list of str, optional
        List of H5 calibration tables for DDE calibration (default: []).
    facetregionfile : str or None, optional
        Path to facet region file for DDE calibration (default: None).
    disable_primary_beam : bool, optional
        If True, disables primary beam correction during prediction (default: False).
    ddcor : bool, optional
        If True, applies direction-dependent corrections after subtraction (default: True).
    modelstoragemanager : str or None, optional
        Storage manager for model data (default: None).
    parallelgridding : int, optional
        Number of parallel gridding threads (default: 1).
    metadata_compression : bool, optional
        If True, enables metadata compression for output (default: True).
    Returns
    -------
    None
    Notes
    -----
    - If `userbox` is a float, it is interpreted as the box size in degrees.
    - If `userbox` is a region file or 'keepall', it controls the region to keep or disables subtraction, respectively.
    - The function creates a region file for the box if needed, predicts the model, subtracts it from the data, and averages the result.
    - If DDE calibration tables are provided (`h5list`), direction-dependent corrections are applied.
    - Temporary columns and files may be created and removed to manage disk space.
    """
    # get imageheader to check frequency
    hdul = fits.open(imagebasename + '-MFS-image.fits')
    header = hdul[0].header

    if len(h5list) != 0:
        datacolumn = 'DATA'  # for DDE
    else:
        datacolumn = 'CORRECTED_DATA'

    if (header['CRVAL3'] < 500e6):  # means we have LOFAR?, just a random value here
        boxsize = 1.5  # degr
    if (header['CRVAL3'] > 500e6) and (header['CRVAL3'] < 1.0e9):  # UHF-band
        boxsize = 3.0  # degr
    if (header['CRVAL3'] >= 1.0e9) and (header['CRVAL3'] < 1.7e9):  # L-band
        boxsize = 2.0  # degr
    if (header['CRVAL3'] >= 1.7e9) and (header['CRVAL3'] < 4.0e9):  # S-band
        boxsize = 1.5  # degr
        
    try:
        boxsize = float(userbox) # userbox is a number
        # overwrite boxsize value here with user specified number
        userbox = 'templatebox.reg' # set this as we are going to use the template region file                 
    except:
        pass # userbox is not a number    

    # create square box file templatebox.reg
    region_string = """
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1
    fk5
    box(137.2868914,-9.6412498,7200.000",7200.000",0)
    """
    r = pyregion.parse(region_string)
    r.write('templatebox.reg')
    r = pyregion.open('templatebox.reg')
    r[0].coord_list[0] = header['CRVAL1']  # units degr
    r[0].coord_list[1] = header['CRVAL2']  # units degr
    r[0].coord_list[2] = boxsize  # units degr
    r[0].coord_list[3] = boxsize  # units degr
    r.write('templatebox.reg')

    # predict the model non-DDE case
    if len(h5list) == 0:
        if userbox is None:
            makeimage(mslist, imagebasename, pixsize, imsize,
                      channelsout, onlypredict=True, squarebox='templatebox.reg',
                      idg=idg, disable_primarybeam_predict=disable_primary_beam,
                      fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
            phaseshiftbox = 'templatebox.reg'
        else:
            if userbox != 'keepall':
                makeimage(mslist, imagebasename, pixsize, imsize,
                          channelsout, onlypredict=True, squarebox=userbox,
                          idg=idg, disable_primarybeam_predict=disable_primary_beam,
                          fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
                phaseshiftbox = userbox
            else:
                phaseshiftbox = None  # so option keepall was set by the user
    else:  # so this we are in DDE mode as h5list is not empty
        if userbox is None:
            makeimage(mslist, imagebasename, pixsize, imsize,
                      channelsout, onlypredict=True, squarebox='templatebox.reg',
                      idg=idg, h5list=h5list, facetregionfile=facetregionfile,
                      disable_primarybeam_predict=disable_primary_beam,
                      fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
            phaseshiftbox = 'templatebox.reg'
        else:
            if userbox != 'keepall':
                makeimage(mslist, imagebasename, pixsize, imsize,
                          channelsout, onlypredict=True, squarebox=userbox,
                          idg=idg, h5list=h5list, facetregionfile=facetregionfile,
                          disable_primarybeam_predict=disable_primary_beam,
                          fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
                phaseshiftbox = userbox
            else:
                phaseshiftbox = None  # so option keepall was set by the user
    # write new data column (if keepall was not set) where rest of the field outside the box is removed
    if phaseshiftbox is not None:
        stepsize = 100000
        
        for ms in mslist:
            # check if outcol exists, if not create it with DP3
            t = table(ms)
            colnames = t.colnames()
            t.close()
            if outcol not in colnames:
                cmd = 'DP3 msin=' + ms + ' msout=. steps=[] msout.datacolumn=' + outcol + ' '
                cmd += 'msin.datacolumn=' + datacolumn + ' '
                if dysco:
                    cmd += 'msout.storagemanager=dysco '
                    cmd += 'msout.storagemanager.weightbitrate=16 '
                print(cmd)
                run(cmd)
            t = table(ms, readonly=False)
            if t.nrows() < stepsize: stepsize = t.nrows()
            for row in range(0, t.nrows(), stepsize):
                print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                data = t.getcol(datacolumn, startrow=row, nrow=stepsize, rowincr=1)
                model = t.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
                t.putcol(outcol, data - model, startrow=row, nrow=stepsize, rowincr=1)
            t.close()
        average(mslist, freqstep=[1] * len(mslist), timestep=1,
                phaseshiftbox=phaseshiftbox, dysco=dysco, makesubtract=True,
                dataincolumn=outcol, metadata_compression=metadata_compression)
        remove_column_ms(mslist, outcol) # remove SUBTRACTED_DATA to free up space
    else:  # so have have "keepall", no subtract, just a copy
        average(mslist, freqstep=[1] * len(mslist), timestep=1,
                phaseshiftbox=phaseshiftbox, dysco=dysco, makesubtract=True,
                dataincolumn=datacolumn, metadata_compression=metadata_compression)
    
    # applycal of closest direction (in multidir h5)
    if len(h5list) != 0 and ddcor and userbox != 'keepall':
        for ms_id, ms in enumerate(mslist):
            ms = os.path.basename(ms)
            if os.path.isdir(ms + '.subtracted_ddcor'):
                os.system('rm -rf ' + ms + '.subtracted_ddcor')
                time.sleep(2)  # wait for the directory to be removed
            applycal(ms + '.subtracted',h5list[ms_id], find_closestdir=True, 
                     msout=ms + '.subtracted_ddcor', dysco=dysco, metadata_compression=metadata_compression)
            with table(ms + '.subtracted') as t:
                if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():  # check if WEIGHT_SPECTRUM_SOLVE is present otherwise this is not needed
                    print('Going to copy over WEIGHT_SPECTRUM_SOLVE')
                    # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                    with table(ms + '.subtracted_ddcor', readonly=False) as t2:
                        print('Adding WEIGHT_SPECTRUM_SOLVE')
                        #desc = t2.getcoldesc('WEIGHT_SPECTRUM')
                        #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
                        #t2.addcols(desc)
                        addcol(t2, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')
                        imweights = t.getcol('WEIGHT_SPECTRUM_SOLVE')
                        t2.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)
            # remove uncorrected file to save disk space
            os.system('rm -rf ' + ms + '.subtracted')

    return


def makeimage(mslist, imageout, pixsize, imsize, channelsout, niter=100000, robust=-0.5,
              uvtaper=None, multiscale=False, predict=True, onlypredict=False, fitsmask=None,
              idg=False, uvminim=80, fitspectralpol=3,
              restoringbeam=15, automask=2.5,
              removenegativecc=True, usewgridder=True, paralleldeconvolution=0,
              parallelgridding=1,
              fullpol=False, selfcalcycle=None,
              uvmaxim=None, h5list=[], facetregionfile=None, squarebox=None,
              DDE_predict='WSCLEAN', DDEimaging=False,
              wgridderaccuracy=1e-4, nosmallinversion=False,
              stack=False, disable_primarybeam_predict=False, disable_primarybeam_image=False,
              facet_beam_update_time=120,
              singlefacetpredictspeedup=True, forceimagingwithfacets=True,
              fulljones_h5_facetbeam=False, sharedfacetreads=False):
    """
    forceimagingwithfacets (bool): force imaging with facetregionfile (facets.reg) even if len(h5list)==0, in this way we can still get a primary beam correction per facet and this image can be use for a DDE predict with the same type of beam correction (this is useful for making image000 when there are no DDE h5 corrections yet and we do not want to use IDG)
    """
    
    # since the addition of the -model-storage-manager the option -model-column can be used without hitting a DP3 error 
    #Related to this was that WSClean up to now uses the same storage manager (read: file) for multiple model columns. While this is 'legal', it turned out that Dp3's DDECal with those model columns as input didn't work with this (due to complicated multi-threading issues).
    if '-model-storage-manager' in subprocess.check_output(['wsclean'], text=True):
        predict_inmodelcol = True
    else:
        predict_inmodelcol = False

    if '-shared-facet-reads' in subprocess.check_output(['wsclean'], text=True):
        sharedfacetreads = True
    sharedfacetreads = False # for now force it to False
    # it seems to slow the imaging down, instead of speed it up
    
    if args['modelstoragemanager'] == 'stokes_i':
        modelstoragemanagerwsclean = 'stokes-i' # because WSclean uses a different name than DP3
                
    fitspectrallogpol = False  # for testing Perseus
    msliststring = ' '.join(map(str, mslist))
    if idg:
        parallelgridding = 1
    t = table(mslist[0] + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()

    #if telescope != 'LOFAR' and not onlypredict and facetregionfile is not None:
    #    nosmallinversion = True
    if telescope != 'LOFAR':
        nosmallinversion = True  

    #  --- DI predict only without facets ---
    # for example to subtract region of sky for the --remove-outside-center option
    if onlypredict and facetregionfile is None:
        if predict:
            if squarebox is not None:
                for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
                    print(model, 'box_' + model)
                    mask_region(model, squarebox, 'box_' + model)

            cmd = 'wsclean -predict '
            # if not usewgridder and not idg:
            #  cmd += '-padding 1.8 '
            if channelsout > 1:

                if args['fix_model_frequencies']: # Implementing model manipulation
                    freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                    mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs) # Adjust models to fit incoming ms
                    #Overwrite relevant args if needed 
                    if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                        args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                        imageout = args['wscleanskymodel']
                    channelsout = len(mod_freqs)
                    cmd += '-channel-division-frequencies ' + mod_freq_string + ' '
                
                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if not disable_primarybeam_predict:
                    if telescope == 'LOFAR':
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(wgridderaccuracy) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '
            
            if args['modelstoragemanager'] is not None:
                cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

            if squarebox is None:
                cmd += '-name ' + imageout + ' ' + msliststring
            else:
                cmd += '-name ' + 'box_' + imageout + ' ' + msliststring
            print('PREDICT STEP: ', cmd)
            run(cmd)
        return
    #  --- end predict only ---

    #  --- DDE-NO-FACET-LOOP CORRUPT-predict, so everything ends up in MODEL_DATA and is corrupted
    # for example for --remove-outside-center option with --DDE
    if onlypredict and facetregionfile is not None and predict and squarebox is not None:
        # do the masking
        for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
            print(squarebox, model, 'box_' + model)
            mask_region(model, squarebox, 'box_' + model)

            # predict with wsclean
        cmd = 'wsclean -predict '
        # if not usewgridder and not idg:
        #  cmd += '-padding 1.8 '
        if channelsout > 1:
            cmd += '-channels-out ' + str(channelsout) + ' '
            if args['gapchanneldivision']:
                cmd += '-gap-channel-division '
        if idg:
            cmd += '-gridder idg -idg-mode cpu '
            if not disable_primarybeam_predict:
                if telescope == 'LOFAR':
                    cmd += '-grid-with-beam -use-differential-lofar-beam '
                    cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
            # cmd += '-pol iquv '
            cmd += '-pol i '
            cmd += '-padding 1.8 '
        else:
            if usewgridder:
                cmd += '-gridder wgridder '
                cmd += '-wgridder-accuracy ' + str(wgridderaccuracy) + ' '
                if nosmallinversion:
                    cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
            if parallelgridding > 1:
                cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

        cmd += '-facet-regions ' + facetregionfile + ' '
        cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list)) + ' amplitude000,phase000 '
        if sharedfacetreads: cmd += '-shared-facet-reads '

        if not fulljones_h5_facetbeam:
            if not is_scalar_array_for_wsclean(h5list):
                cmd += '-diagonal-visibilities '  # different XX and YY solutions
            else:
                cmd += '-scalar-visibilities '  # scalar solutions

        if telescope == 'LOFAR' or telescope == 'MeerKAT':
            if not disable_primarybeam_predict:
                cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                if telescope == 'LOFAR': cmd += '-use-differential-lofar-beam '

        if args['modelstoragemanager'] is not None:
            cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

        cmd += '-name box_' + imageout + ' ' + msliststring
        if DDE_predict == 'WSCLEAN':
            print('DDE PREDICT STEP: ', cmd)
            run(cmd)
        return
        #  --- NO-FACET-LOOP end DDE CORRUPT-predict only ---

    #  --- DDE-FACET-LOOP predict, so all facets are predicted  ---
    #  --- each facet gets its own model column ---
    if onlypredict and facetregionfile is not None and predict:
        # step 1 open facetregionfile
        modeldatacolumns_list = []
        r = pyregion.open(facetregionfile)
        for facet_id, facet in enumerate(r):
            r[facet_id:facet_id + 1].write('facet' + str(facet_id) + '.reg')  # split facet from region file

            # step 2 mask outside of region file
            if not singlefacetpredictspeedup:  # not needed, because WSClean will do the facet cutting
                for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
                    modelout = 'facet_' + model
                    if DDE_predict == 'WSCLEAN':
                        print(model, modelout)
                        mask_region_inv(model, 'facet' + str(facet_id) + '.reg', modelout)

            # step 3 predict with wsclean
            cmd = 'wsclean -predict '
            if predict_inmodelcol:  # directly predict the right column name
                cmd += '-model-column MODEL_DATA_DD' + str(facet_id) + ' '
            if args['modelstoragemanager'] is not None:
                cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '
            # if not usewgridder and not idg:
            #  cmd += '-padding 1.8 '
            if channelsout > 1:
                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if not disable_primarybeam_predict:
                    if telescope == 'LOFAR':
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(wgridderaccuracy) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

            # NEW CODE FOR SPEEDUP
            if singlefacetpredictspeedup:
                cmd += '-facet-regions ' + 'facet' + str(facet_id) + '.reg' + ' '
                if telescope == 'LOFAR' or telescope == 'MeerKAT':
                    if not disable_primarybeam_predict:
                        # check if -model-fpb.fits is there for image000 (in case image000 was made without facets)
                        if selfcalcycle == 0:
                            fix_fpb_images(imageout)    
                        cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                        if telescope == 'LOFAR': cmd += '-use-differential-lofar-beam '
                        if not fulljones_h5_facetbeam:
                            # cmd += '-diagonal-visibilities ' # different XX and YY solutions
                            cmd += '-scalar-visibilities '  # scalar solutions

                cmd += '-name ' + imageout + ' ' + msliststring
            else:
                cmd += '-name facet_' + imageout + ' ' + msliststring

            if DDE_predict == 'WSCLEAN':
                print('DDE PREDICT STEP: ', cmd)
                run(cmd)

            # step 4 copy over to MODEL_DATA_DDX
            for ms in mslist:
                cmddppp = 'DP3 msin=' + ms + ' msin.datacolumn=MODEL_DATA msout=. steps=[] '
                cmddppp += 'msout.datacolumn=MODEL_DATA_DD' + str(facet_id)
                if DDE_predict == 'WSCLEAN' and not predict_inmodelcol:
                    run(cmddppp)
            modeldatacolumns_list.append('MODEL_DATA_DD' + str(facet_id))

        return modeldatacolumns_list
    #  --- end DDE predict only ---

    # ----- MAIN IMAGING PART ------
    os.system('rm -f ' + imageout + '-*.fits')
    imcol = 'CORRECTED_DATA'
    # check for all ms in mslist
    # situation can be that only some ms have CORRECTED_DATA based on what happens in the beamcor
    # so some ms can have a beam correction and others not
    # for example because of different directions where the beam was applied
    for ms in mslist:
        t = table(ms, readonly=True, ack=False)
        colnames = t.colnames()
        if 'CORRECTED_DATA' not in colnames:  # for first imaging run
            imcol = 'DATA'
        t.close()

    baselineav = str(1.5e3 * 60000. * 2. * np.pi * 1.5 / (24. * 60. * 60 * float(imsize)))

    if args['imager'] == 'WSCLEAN':
        cmd = 'wsclean '
        cmd += '-no-update-model-required '
        cmd += '-minuv-l ' + str(uvminim) + ' '
        if uvmaxim is not None:
            cmd += '-maxuv-l ' + str(uvmaxim) + ' '
        cmd += '-size ' + str(int(imsize)) + ' ' + str(int(imsize)) + ' -reorder '
        if type(robust) is not str:
            cmd += '-weight briggs ' + str(robust)
        else:
            cmd += '-weight ' + str(robust)
        # cmd += ' -clean-border 1 ' # not needed anymore for WSCleand
        cmd += ' -parallel-reordering 4 '
        # -weighting-rank-filter 3 -fit-beam
        cmd += '-mgain ' + str(args['mgain']) + ' -data-column ' + imcol + ' '
        # if not usewgridder and not idg:
        #  cmd += '-padding 1.4 '
        if channelsout > 1:
            if args['fix_model_frequencies']: # Implementing model manipulation
                freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs) # Adjust models to fit incoming ms
                #Overwrite relevant args
                if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                    args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                    imageout = args['wscleanskymodel']
                channelsout = len(mod_freqs)
                cmd += '-channel-division-frequencies ' + mod_freq_string + ' '
            cmd += ' -join-channels -channels-out ' + str(channelsout) + ' '
            if args['gapchanneldivision']:
                cmd += '-gap-channel-division '
        if paralleldeconvolution > 0:
            cmd += '-parallel-deconvolution ' + str(paralleldeconvolution) + ' '
        if parallelgridding > 1:
            cmd += '-parallel-gridding ' + str(parallelgridding) + ' '
        if args['deconvolutionchannels'] > 0 and channelsout > 1:
            cmd += '-deconvolution-channels ' + str(args['deconvolutionchannels']) + ' '
        if automask > 0.5:
            cmd += '-auto-mask ' + str(automask) + ' -auto-threshold 0.5 '  # to avoid automask 0
        if args['localrmswindow'] > 0:
            cmd += '-local-rms-window ' + str(args['localrmswindow']) + ' '

        if args['ddpsfgrid'] is not None:
            cmd += '-dd-psf-grid ' + str(args['ddpsfgrid']) + ' ' + str(args['ddpsfgrid']) + ' '

        if multiscale:
            # cmd += '-multiscale '+' -multiscale-scales 0,4,8,16,32,64 -multiscale-scale-bias 0.6 '
            # cmd += '-multiscale '+' -multiscale-scales 0,6,12,16,24,32,42,64,72,128,180,256,380,512,650 '
            cmd += '-multiscale '
            cmd += '-multiscale-scale-bias ' + str(args['multiscalescalebias']) + ' '
            if args['multiscalemaxscales'] == 0:
                cmd += '-multiscale-max-scales ' + str(int(np.rint(np.log2(float(imsize)) - 3))) + ' '
            else:  # use value set by user
                cmd += '-multiscale-max-scales ' + str(int(args['multiscalemaxscales'])) + ' '
        if fitsmask is not None and fitsmask != 'nofitsmask':
            if os.path.isfile(fitsmask):
                shape = get_image_size(fitsmask)
                if (shape[0] == int(imsize)) and (shape[1] == int(imsize)): # to allow for a restart with different imsize
                    cmd += '-fits-mask ' + fitsmask + ' '
            else:
                print('fitsmask: ', fitsmask, 'does not exist')
                raise Exception('fitsmask does not exist')
        if uvtaper is not None:
            cmd += '-taper-gaussian ' + uvtaper + ' '
        if args['taperinnertukey'] is not None:
            cmd += '-taper-inner-tukey ' + str(args['taperinnertukey']) + ' '

        if (fitspectralpol > 0) and not (fullpol):
            cmd += '-save-source-list '

        if (fitspectralpol > 0) and (channelsout > 1) and not (fullpol):
            if fitspectrallogpol:
                cmd += '-fit-spectral-log-pol ' + str(fitspectralpol) + ' '
            else:
                cmd += '-fit-spectral-pol ' + str(fitspectralpol) + ' '
        if idg:
            cmd += '-gridder idg -idg-mode cpu '
            if not disable_primarybeam_image:
                if telescope == 'LOFAR':
                    cmd += '-grid-with-beam -use-differential-lofar-beam '
                    cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
            # cmd += '-pol iquv -link-polarizations i '
            cmd += '-pol i '
            cmd += '-padding 1.4 '
        else:
            if fullpol:
                cmd += '-pol iquv -join-polarizations '
            else:
                cmd += '-pol i '
            if len(h5list) == 0 and not args['groupms_h5facetspeedup']:  # only use baseline-averaging if there are no h5 facet-solutions
                # if not forceimagingwithfacets:
                if facetregionfile is None: # do not allow baseline-averaging with facets
                    # facet imaging works with  baseline-averaging but is very slow/inefficient there is no speedup, rather an order of magnitude slow down
                    cmd += '-baseline-averaging ' + baselineav + ' '
            if usewgridder:
                cmd += '-gridder wgridder '
                cmd += '-wgridder-accuracy ' + str(wgridderaccuracy) + ' '
                if nosmallinversion:
                    cmd += '-no-min-grid-resolution '  # '-no-small-inversion '

        if len(h5list) > 0:
            cmd += '-facet-regions ' + facetregionfile + ' '
            if sharedfacetreads: cmd += '-shared-facet-reads '
            if args['groupms_h5facetspeedup'] and len(mslist) > 1:
                mslist_concat, h5list_concat = concat_ms_wsclean_facetimaging(mslist, h5list=h5list, concatms=False)
                cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list_concat)) + ' '
                cmd += ' amplitude000,phase000 '
            else:
                cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list)) + ' amplitude000,phase000 '

            if not fulljones_h5_facetbeam:
                if not is_scalar_array_for_wsclean(h5list):
                    cmd += '-diagonal-visibilities '  # different XX and YY solutions
                else:
                    cmd += '-scalar-visibilities '  # scalar solutions

            if telescope == 'LOFAR' or telescope == 'MeerKAT':
                if not disable_primarybeam_image:
                    cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                    if telescope == 'LOFAR': cmd += '-use-differential-lofar-beam '
        elif forceimagingwithfacets and facetregionfile is not None:  # so h5list is zero, but we still want facet imaging
            if args['groupms_h5facetspeedup'] and len(mslist) > 1:
                mslist_concat, h5list_concat_tmp = concat_ms_wsclean_facetimaging(mslist, concatms=False)
            cmd += '-facet-regions ' + facetregionfile + ' '
            if sharedfacetreads: cmd += '-shared-facet-reads '
            if (telescope == 'LOFAR' or telescope == 'MeerKAT') and not disable_primarybeam_image:
                cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                if telescope == 'LOFAR': cmd += '-use-differential-lofar-beam '
                if not fulljones_h5_facetbeam:
                    # cmd += '-diagonal-visibilities ' # different XX and YY solutions
                    cmd += '-scalar-visibilities '  # scalar solutions
        else:
            if telescope == 'LOFAR' and not check_phaseup_station(mslist[0]) and not idg:
                if not disable_primarybeam_image:
                    cmd += '-apply-primary-beam -use-differential-lofar-beam '
                    cmd += '-facet-beam-update ' + str(facet_beam_update_time) + ' '

        cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec '
        if args['groupms_h5facetspeedup'] and len(mslist) > 1 and facetregionfile is not None:
            msliststring_concat = ' '.join(map(str, mslist_concat))
            print('WSCLEAN: ', cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)
            logger.info(cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)
            run(cmd + ' -nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)

        else:
            print('WSCLEAN: ', cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)
            logger.info(cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)
            run(cmd + ' -nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)

        clean_up_images(imageout)
        
        # write info about how the primary beam correction was done to the FITS header and processing history
        write_processing_history(' '.join(map(str, sys.argv)), facetselfcal_version, imageout)
        write_primarybeam_info(cmd, imageout)
        
        # REMOVE nagetive model components, these are artifacts (only for Stokes I)
        if removenegativecc:
            if idg:
                removenegativefrommodel(sorted(glob.glob(imageout + '-????-*model*.fits')))  # only Stokes I
            else:
                removenegativefrommodel(sorted(glob.glob(imageout + '-????-*model*.fits')))

        # Remove NaNs from array (can happen if certain channels from channels-out are flagged, or from -apply-facet-beam far into the sidelobes)
        if idg:
            removeneNaNfrommodel(glob.glob(imageout + '-????-*model*.fits'))  # only Stokes I
        else:
            removeneNaNfrommodel(glob.glob(imageout + '-????-*model*.fits'))

        # Check is anything was cleaned. If not, stop the selfcal to avoid obscure errors later
        if channelsout > 1:
            model_allzero = checkforzerocleancomponents(glob.glob(imageout + '-????-*model*.fits'))  # only Stokes I
        else:
            model_allzero = checkforzerocleancomponents(glob.glob(imageout + '-*model*.fits'))
        if model_allzero:
            logger.error("All channel maps models were zero: Stopping the selfcal")
            print("All channel maps models were zero: Stopping the selfcal")
            sys.exit(1)

        if predict and len(h5list) == 0 and not DDEimaging:
            # We check for DDEimaging to avoid a predict for image000 in a --DDE run
            # because at that moment there is no h5list yet, avoiding an unnecessary DI-type predict
            cmd = 'wsclean -predict '  # -size '
            # cmd += str(int(imsize)) + ' ' + str(int(imsize)) +  ' -predict '
            # if not usewgridder and not idg:
            #     cmd += '-padding 1.8 '
            if channelsout > 1:
                if args['fix_model_frequencies']: # Implementing model manipulation
                    freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                    mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs) # Adjust models to fit incoming ms
                    #Overwrite relevant args
                    if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                        args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                        imageout = args['wscleanskymodel']
                    channelsout = len(mod_freqs)
                    cmd += '-channel-division-frequencies ' + mod_freq_string + ' '

                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if telescope == 'LOFAR':
                    if not disable_primarybeam_predict:
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(wgridderaccuracy) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

                # Needs multi-col predict
                # if h5 is not None:
                #     cmd += '-facet-regions ' + facetregionfile + ' '
                #     cmd += '-apply-facet-solutions ' + h5 + ' amplitude000,phase000 '
                #     if telescope == 'LOFAR':
                #         cmd += '-apply-facet-beam -facet-beam-update 600 -use-differential-lofar-beam '
                #         cmd += '-diagonal-solutions '
                if args['modelstoragemanager'] is not None:
                    cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

                cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec ' + msliststring
                print('PREDICT STEP: ', cmd)
                run(cmd)

        if args['imager'] == 'DDFACET':
            makemslist(mslist)
            # restoringbeam = '15'
            cmd = 'DDF.py --Data-MS=mslist.txt --Deconv-PeakFactor=0.001 --Data-ColName=' + imcol + ' ' + \
                  '--Parallel-NCPU=32 --Output-Mode=Clean --Deconv-CycleFactor=0 ' + \
                  '--Deconv-MaxMinorIter=' + str(niter) + ' --Deconv-MaxMajorIter=5 ' + \
                  '--Deconv-Mode=SSD --Weight-Robust=' + str(robust) + ' --Image-NPix=' + str(int(imsize)) + ' ' + \
                  '--CF-wmax=50000 --CF-Nw=100 --Beam-Model=None --Beam-LOFARBeamMode=A --Beam-NBand=1 ' + \
                  '--Output-Also=onNeds --Image-Cell=' + str(pixsize) + ' --Facets-NFacets=1 --Freq-NDegridBand=1 ' + \
                  '--Deconv-RMSFactor=3.0 --Deconv-FluxThreshold=0.0 --Data-Sort=1 --Cache-Dir=. --Freq-NBand=2 ' + \
                  '--GAClean-MinSizeInit=10 --Facets-DiamMax=1.5 --Facets-DiamMin=0.1 ' + \
                  '--Cache-Dirty=auto --Weight-ColName=WEIGHT_SPECTRUM --Output-Name=' + imageout + ' ' + \
                  '--Comp-BDAMode=1 --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --Cache-Reset=1 ' + \
                  '--RIME-ForwardMode=BDA-degrid --Predict-ColName=MODEL_DATA --Selection-UVRange=[0.1,2000.] ' + \
                  '--Output-RestoringBeam=' + str(restoringbeam) + ' --Mask-SigTh=5.0 '
            if fitsmask is not None and fitsmask != 'nofitsmask':
                cmd += '--Mask-External=' + fitsmask + ' --Mask-Auto=0 '
            else:
                cmd += '--Mask-Auto=1 '

            print(cmd)
            run(cmd)


def removeneNaNfrommodel(imagenames):
    """
    replace NaN/inf pixels values in WSCLEAN model images with zeros
    """

    for image_id, image in enumerate(imagenames):
        print('remove NaN/Inf values from model: ', image)
        with fits.open(image) as hdul:
            data = hdul[0].data
            data[np.where(~np.isfinite(data))] = 0.0
            hdul[0].data = data
            hdul.writeto(image, overwrite=True)
    return


def removenegativefrommodel(imagenames):
    """
    replace negative pixel values in WSCLEAN model images with zeros
    """

    perseus = False
    A1795 = False
    A1795imlist = sorted(
        glob.glob('/net/nieuwerijn/data2/rtimmerman/A1795_HBA/A1795/selfcal/selfcal_pix0.15_wide-????-model.fits'))

    for image_id, image in enumerate(imagenames):
        print('remove negatives from model: ', image)
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
            print(cmdA1795)
            run(cmdA1795)

    return


def checkforzerocleancomponents(imagenames):
    """ Check if something was cleaned, if not stop de script to avoid more obscure errors later

    Args:
        imagenames (list): List of model images to check for clean components.

    Returns:
        True if all model images are zero, False otherwise.
    """

    n_images = len(imagenames)
    n_zeros = 0
    for image_id, image in enumerate(imagenames):
        print("Check if there are non-zero pixels: ", image)
        hdul = fits.open(image)
        data = hdul[0].data
        if not np.any(data):  # this checks if all elements are 0.0
            print("Model image:", image, "contains only zeros.")
            n_zeros = n_zeros + 1
        hdul.close()
    if n_zeros == n_images:
        return True
    else:
        return False


def updatemodelcols_includedir(modeldatacolumns, soltypenumber, soltypelist_includedir, ms, dryrun=False, modelstoragemanager=None):
    modeldatacolumns_solve = []
    modeldatacolumns_notselected = []
    id_kept = []
    id_removed = []

    f = open('facetdirections.p', 'rb')
    sourcedir = pickle.load(f)  # units are radian
    f.close()
    assert sourcedir.shape[0] == len(modeldatacolumns)
    assert soltypenumber < soltypelist_includedir.shape[1]
    assert len(modeldatacolumns) == soltypelist_includedir.shape[0]

    soltypelist_includedir_sel = soltypelist_includedir[:, soltypenumber]  # select the correct soltype pertubation
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
    print('soltypelist_includedir_sel for this pertubation', soltypelist_includedir_sel)
    # print(modeldatacolumns_solve)
    # print('Not selected', modeldatacolumns_notselected)
    # print(id_kept)
    # print(id_removed)
    print('Removed these directions coordinates')
    print(sourcedir[id_removed][:])
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
        print('Removed', modeldatacolumns[removed_id], 'Closest kept is:', closest_kept_modelcol)

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
                cmddppp += 'msout.storagemanager=' + modelstoragemanager + ' '
            print(cmddppp)
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
            print(taqlcmd)
            if not dryrun:
                run(taqlcmd)

    # create new column name MODEL_DATA_DDX+Y+Z with DP3
    # fill this column with taql

    # create modeldatacolumns_solve

    # return modeldatacolumns_solve

    return modeldatacolumns_solve_newnames, sourcedir[id_removed][:], id_kept


def groupskymodel(skymodelin, facetfitsfile, skymodelout=None):
    import lsmtool
    print('Loading:', skymodelin)
    LSM = lsmtool.load(skymodelin)
    LSM.group(algorithm='facet', facet=facetfitsfile)
    if skymodelout is not None:
        LSM.write(skymodelout, clobber=True)
        return skymodelout
    else:
        LSM.write('grouped_' + skymodelin, clobber=True)
        return 'grouped_' + skymodelin


def findrms(mIn, maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
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


def _add_astropy_beam(fitsname):
    """ Add beam from astropy

    Args:
        fitsname: name of fits file
    Returns:
        ellipse
    """

    head = fits.getheader(fitsname)
    bmaj = head['BMAJ']
    bmin = head['BMIN']
    bpa = head['BPA']
    cdelt = head['CDELT2']
    bmajpix = bmaj / cdelt
    bminpix = bmin / cdelt
    ellipse = matplotlib.patches.Ellipse((20, 20), bmajpix, bminpix, bpa)
    return ellipse


def plotimage_astropy(fitsimagename, outplotname, mask=None, regionfile=None, \
                      cmap='bone', regioncolor='yellow', minmax=None, regionalpha=0.6):

    # image noise info
    hdulist = fits.open(fitsimagename)
    imagenoiseinfo = findrms(np.ndarray.flatten(hdulist[0].data))
    
    try:
        if minmax is None:
            logger.info(fitsimagename + ' Max image: ' + str(np.max(np.ndarray.flatten(hdulist[0].data))))
            logger.info(fitsimagename + ' Min image: ' + str(np.min(np.ndarray.flatten(hdulist[0].data))))
            logger.info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
    except:
        pass # so we can also use the function without a logger open

    hdulist = flatten(fits.open(fitsimagename))

    data = fits.getdata(fitsimagename)
    head = fits.getheader(fitsimagename)
    f = plt.figure()
    ax = f.add_subplot(111, projection=WCS(hdulist.header) ) #, slices=('x', 'y', 0, 0))
    if minmax is None:
        img = ax.imshow(data[0, 0, :, :], cmap=cmap, vmax=16 * imagenoiseinfo, vmin=-6 * imagenoiseinfo)
        ax.set_title(fitsimagename + ' (noise = {} mJy/beam)'.format(round(imagenoiseinfo * 1e3, 3)),fontsize=6)
    else:
        img = ax.imshow(data[0, 0, :, :], cmap=cmap, vmax=minmax[1], vmin=minmax[0])
        #ax.set_title(fitsimagename, fontsize=6)
        ax.set_title(fitsimagename + ' (noise = {} mJy/beam)'.format(round(imagenoiseinfo * 1e3, 3)),fontsize=6)
    
    ax.grid(True)
    ax.coords[0].set_axislabel_position('b') # for some reason this needs to be hardcoded, otherwise RA gets on top axis
    ax.coords[0].set_ticks_position('bt')
    ax.coords[0].set_ticklabel_position('b') # for some reason this needs to be hardcoded, otherwise RA gets on top axis
    
    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    try:
        from astropy.visualization.wcsaxes import add_beam, add_scalebar
        add_beam(ax, header=hdulist.header,  frame=True) 
    except Exception as e:
        print(f"Cannot plot beam on image, failed with error: {e}. Skipping.")

    cbar = plt.colorbar(img)
    cbar.set_label('Flux (mJy beam$^{-1}$)')
    #ax.add_artist(_add_astropy_beam(fitsimagename))
 
    try: 
        if regionfile is not None:
            import regions
            ds9regions = regions.Regions.read(regionfile, format='ds9')
            for ds9region in ds9regions:
                reg = ds9region.to_pixel(WCS(hdulist.header))
                reg.plot(ax=ax, color=regioncolor, alpha=regionalpha)
    except Exception as e:
        print(f"Cannot overplot facets, failed with error: {e}. Skipping.")
        
    if mask is not None:
        maskdata = fits.getdata(mask)[0, 0, :, :]
        ax.contour(maskdata, colors='red', levels=[0.1 * imagenoiseinfo], filled=False, alpha=0.6, linewidths=1)

    if os.path.isfile(outplotname + '.png'):
        os.system('rm -f ' + outplotname + '.png')
    plt.savefig(outplotname, dpi=450, format='png')
    plt.close()
    return


def plotimage(selfcalcycle, stackstr='', mask=None, regionfile=None):
    """
    Tries to plot the image using astropy first, and falls back to aplpy if astropy fails.
    Parameters:
    selfcalcycle (int): selfcal cycle number so we can pick up the correct image
    stackstr (str): basename string in case we are stacking (otherwise it is an empty string)
    mask (str): fits clean mask image (will be overplot with red contours)
    regionfile (str): DS9 facet region file for --DDE mode, facet layout will be shown in yellow
    """
    if args['imager'] == 'WSCLEAN':
        if args['idg']:
            plotpngimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '.png'
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '-MFS-image.fits'
        else:
            plotpngimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '.png'
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '-MFS-image.fits'
        if args['imager'] == 'DDFACET':
            plotpngimage = args['imagename'] + str(selfcalcycle) + '.png'
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '.app.restored.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '.app.restored.fits'

        if args['channelsout'] == 1:
            plotpngimage = plotpngimage.replace('-MFS', '').replace('-I', '')
            plotfitsimage = plotfitsimage.replace('-MFS', '').replace('-I', '')    
            plotfitsimage000 = plotfitsimage000.replace('-MFS', '').replace('-I', '')

    # find noise of image000, so we can directly compare images between selfcal cycles by having a hardcoded range
    hdulist = fits.open(plotfitsimage000) 
    imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
    plotminmax = [-3.*imagenoise, 10.*imagenoise]
    hdulist.close()

    try:
        plotimage_astropy(plotfitsimage, plotpngimage, mask, regionfile=regionfile, minmax=plotminmax)
    except Exception as e:
        print(f"Astropy plotting failed with error: {e}. Switching to aplpy.")
        plotimage_aplpy(plotfitsimage, plotpngimage, mask, plotfitsimage)


def plotimage_aplpy(fitsimagename, outplotname, mask=None, rmsnoiseimage=None):
    import aplpy
    # image noise for plotting
    if rmsnoiseimage is None:
        hdulist = fits.open(fitsimagename)
    else:
        hdulist = fits.open(rmsnoiseimage)
    imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
    hdulist.close()

    # image noise info
    hdulist = fits.open(fitsimagename)
    imagenoiseinfo = findrms(np.ndarray.flatten(hdulist[0].data))
    logger.info(fitsimagename + ' Max image: ' + str(np.max(np.ndarray.flatten(hdulist[0].data))))
    logger.info(fitsimagename + ' Min image: ' + str(np.min(np.ndarray.flatten(hdulist[0].data))))
    hdulist.close()

    f = aplpy.FITSFigure(fitsimagename, slices=[0, 0])
    f.show_colorscale(vmax=16 * imagenoise, vmin=-6 * imagenoise, cmap='bone')
    f.set_title(fitsimagename + ' (noise = {} mJy/beam)'.format(round(imagenoiseinfo * 1e3, 3)))
    try:  # to work around an aplpy error
        f.add_beam()
        f.beam.set_frame(True)
        f.beam.set_color('white')
        f.beam.set_edgecolor('black')
        f.beam.set_linewidth(1.)
    except:
        pass

    f.add_grid()
    f.grid.set_color('white')
    f.grid.set_alpha(0.5)
    f.grid.set_linewidth(0.2)
    f.add_colorbar()
    f.colorbar.set_axis_label_text('Flux (mJy beam$^{-1}$)')
    if mask is not None:
        try:
            f.show_contour(mask, colors='red', levels=[0.1 * imagenoise], filled=False, smooth=1, alpha=0.6,
                           linewidths=1)
        except:
            pass
    if os.path.isfile(outplotname + '.png'):
        os.system('rm -f ' + outplotname + '.png')
    f.save(outplotname, dpi=120, format='png')
    logger.info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
    return


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

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


def beamcor_and_lin2circ(ms, msout='.', dysco=True, beam=True, lin2circ=False,
                         circ2lin=False, losotobeamlib='stationresponse', 
                         update_poltable=True, idg=False, metadata_compression=True):
    """
    correct a ms for the beam in the phase center (array_factor only)
    """

    # check if there are applybeam corrections in the header
    # should be there unless a very old DP3 version has been used
    usedppp = beam_keywords(ms)

    losoto = 'losoto'
    # taql = 'taql'
    H5name = create_beamcortemplate(ms)
    phasedup = check_phaseup(H5name)  # in case no beamcor is done we still need this

    if (lin2circ or circ2lin):
        tp = table(ms + '/POLARIZATION', ack=False)
        polinfo = tp.getcol('CORR_TYPE')
        if lin2circ:  # so in this case input must be linear
            if not np.array_equal(np.array([[9, 10, 11, 12]]), polinfo):
                print(polinfo)
                raise Exception('Input data is not linear, cannot convert to circular')
        if circ2lin:  # so in this case input must be circular
            if not np.array_equal(np.array([[5, 6, 7, 8]]), polinfo):
                print(polinfo)
                raise Exception('Input data is not circular, cannot convert to linear')
        tp.close()

    if beam:
        tp = table(ms + '/POLARIZATION', ack=False)
        polinfo = tp.getcol('CORR_TYPE')
        tp.close()
        if np.array_equal(np.array([[5, 6, 7, 8]]), polinfo):  # so we have circular data
            raise Exception('Cannot do DP3 beam correction on input data that is circular')

    if lin2circ and circ2lin:
        print('Wrong input in function, both lin2circ and circ2lin are True')
        raise Exception('Wrong input in function, both lin2circ and circ2lin are True')

    if beam and not args['phasediff_only']:
        losotolofarbeam(H5name, 'phase000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True,
                        beamlib=losotobeamlib)
        losotolofarbeam(H5name, 'amplitude000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True,
                        beamlib=losotobeamlib)

        phasedup = fixbeam_ST001(H5name)
        parset = create_losoto_beamcorparset(ms, refant=findrefant_core(H5name))
        force_close(H5name)

        # print('Phase up dataset, cannot use DPPP beam, do manual correction')
        cmdlosoto = losoto + ' ' + H5name + ' ' + parset
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)

    if usedppp and not phasedup:
        cmddppp = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=DATA '
        cmddppp += 'msout=' + msout + ' '
        cmddppp += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        
        if not metadata_compression and msout != '.':
            cmddppp += 'msout.uvwcompression=False '
            cmddppp += 'msout.antennacompression=False '
            cmddppp += 'msout.scalarflags=False '
        
        if check_phaseup_station(ms):
            if msout != '.': cmddppp += 'msout.uvwcompression=False ' # only when writing new MS:
        if msout == '.':
            cmddppp += 'msout.datacolumn=CORRECTED_DATA '
        if (lin2circ or circ2lin) and beam:
            cmddppp += 'steps=[beam,pystep] '
            if idg:
                cmddppp += 'beam.type=applybeam beam.updateweights=False '  # weights
            else:
                cmddppp += 'beam.type=applybeam beam.updateweights=True '  # weights
            cmddppp += 'beam.direction=[] '  # correction for the current phase center
            # cmddppp += 'beam.beammode= ' default is full, will undo element as well(!)

            cmddppp += 'pystep.python.module=polconv '
            cmddppp += 'pystep.python.class=PolConv '
            cmddppp += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmddppp += 'pystep.lin2circ=1 '
            if circ2lin:
                cmddppp += 'pystep.circ2lin=1 '

        if beam and not (lin2circ or circ2lin):
            cmddppp += 'steps=[beam] '
            if idg:
                cmddppp += 'beam.type=applybeam beam.updateweights=False '  # weights
            else:
                cmddppp += 'beam.type=applybeam beam.updateweights=True '  # weights
            cmddppp += 'beam.direction=[] '  # correction for the current phase center
            # cmddppp += 'beam.beammode= ' default is full, will undo element as well(!)

        if (lin2circ or circ2lin) and not beam:
            cmddppp += 'steps=[pystep] '
            cmddppp += 'pystep.python.module=polconv '
            cmddppp += 'pystep.python.class=PolConv '
            cmddppp += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmddppp += 'pystep.lin2circ=1 '
            if circ2lin:
                cmddppp += 'pystep.circ2lin=1 '

        if dysco:
            cmddppp += 'msout.storagemanager=dysco '
            cmddppp += 'msout.storagemanager.weightbitrate=16 '

        print('DP3 applybeam/polconv:', cmddppp)
        run(cmddppp)
        if msout == '.':
            # run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
            run("DP3 msin=" + ms + " msout=. msin.datacolumn=CORRECTED_DATA msout.datacolumn=DATA steps=[]", log=True)
            remove_column_ms(ms, 'CORRECTED_DATA')  # remove the column so it does not get used in the next step
    else:
        cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=DATA '
        cmd += 'msout=' + msout + ' '
        if check_phaseup_station(ms): 
            if msout != '.': cmd += 'msout.uvwcompression=False ' # only when writing new MS
        cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        if msout == '.':
            cmd += 'msout.datacolumn=CORRECTED_DATA '
        
        if not metadata_compression and msout != '.':
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '

        if (lin2circ or circ2lin) and beam:
            cmd += 'steps=[ac1,ac2,pystep] '
            cmd += 'pystep.python.module=polconv '
            cmd += 'pystep.python.class=PolConv '
            cmd += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmd += 'pystep.lin2circ=1 '
            if circ2lin:
                cmd += 'pystep.circ2lin=1 '

            cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
            cmd += 'ac1.type=applycal ac2.type=applycal '
            cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 '
            if idg:
                cmd += 'ac2.updateweights=False '
            else:
                cmd += 'ac2.updateweights=True '
        if beam and not (lin2circ or circ2lin):
            cmd += 'steps=[ac1,ac2] '
            cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
            cmd += 'ac1.type=applycal ac2.type=applycal '
            cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 '
            if idg:
                cmd += 'ac2.updateweights=False '
            else:
                cmd += 'ac2.updateweights=True '
        if (lin2circ or circ2lin) and not beam:
            cmd += 'steps=[pystep] '
            cmd += 'pystep.python.module=polconv '
            cmd += 'pystep.python.class=PolConv '
            cmd += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmd += 'pystep.lin2circ=1 '
            if circ2lin:
                cmd += 'pystep.circ2lin=1 '

        if dysco:
            cmd += 'msout.storagemanager=dysco '
            cmd += 'msout.storagemanager.weightbitrate=16 '
        print('DP3 applycal/polconv:', cmd)
        run(cmd, log=True)
        if msout == '.':
            # run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
            cmdcopycolumn = "DP3 msin=" + ms + " msout=. msin.datacolumn=CORRECTED_DATA msout.datacolumn=DATA steps=[]"
            if dysco:
                cmdcopycolumn += ' msout.storagemanager=dysco '
                cmdcopycolumn += ' msout.storagemanager.weightbitrate=16 '
            run(cmdcopycolumn, log=True) # use DP3 and not taql so column beam keyword are copied over
            remove_column_ms(ms, 'CORRECTED_DATA')  # remove the column so it does not get used in the next step
               
    # update ms POLTABLE
    if (lin2circ or circ2lin) and update_poltable:
        tp = table(ms + '/POLARIZATION', readonly=False, ack=True)
        if lin2circ:
            tp.putcol('CORR_TYPE', np.array([[5, 6, 7, 8]], dtype=np.int32))  # FROM LIN-->CIRC
        if circ2lin:
            tp.putcol('CORR_TYPE', np.array([[9, 10, 11, 12]], dtype=np.int32))  # FROM CIRC-->LIN
        tp.close()

    return


def beam_keywords(ms, add_beamkeywords=True):
    """
    Check for beam application keywords in a measurement set (ms).
    If add_beamkeywords True then add keywords in case they are missing
    """
    
    t = table(ms + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()
    
    applybeam_info = False
    with table(ms, readonly=True, ack=False) as t:
        try:
            beammode = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_MODE')
            applybeam_info = True
            print('DP3 applybeam was used')
        except:
            applybeam_info = False
            print('No applybeam beam keywords were found. Possibly an old DP3 version was used in prefactor.')
            print('Adding keywords manually assuming the beam was taken out in the pointing center')
            logger.warning('No applybeam beam keywords were found. Possibly an old DP3 version was used in prefactor.')
            logger.warning('Adding keywords manually assuming the beam was taken out in the pointing center')
   
    if not applybeam_info and add_beamkeywords and telescope == 'LOFAR':
            with table(ms + '/FIELD', readonly=True, ack=False) as t:
                ref_direction = t.getcol('REFERENCE_DIR').squeeze()
            cmddppp = 'DP3 msin=' + ms + ' msout=. steps=[sb] sb.type=setbeam sb.beammode=default '  
            cmddppp += 'sb.direction=['+  str(ref_direction[0]) +',' +  str(ref_direction[1]) + ']'
            print(cmddppp)
            run(cmddppp)
            applybeam_info = True

    return applybeam_info


def beamcormodel(ms, dysco=True):
    """
    create MODEL_DATA_BEAMCOR where we store beam corrupted model data
    """
    H5name = ms + '_templatejones.h5'

    cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=MODEL_DATA msout=. '
    cmd += 'msout.datacolumn=MODEL_DATA_BEAMCOR steps=[ac1,ac2] '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
    cmd += 'ac1.type=applycal ac2.type=applycal '
    cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 ac2.updateweights=False '
    cmd += 'ac1.invert=False ac2.invert=False '  # Here we corrupt with the beam !
    print('DP3 applycal:', cmd)
    run(cmd, log=True)

    return


def write_RMsynthesis_weights(fitslist, outfile):
    rmslist = np.zeros(len(fitslist))

    for fits_id, fitsfile in enumerate(fitslist):
        hdu = flatten(fits.open(fitsfile, ignore_missing_end=True))
        rmslist[fits_id] = findrms(hdu.data)

    print(rmslist * 1e6)
    rmslist = 1 / rmslist ** 2  # 1/variance
    rmslist = rmslist / np.max(rmslist)  # normalize to max 1

    with open(outfile, 'w') as f:
        for rms in rmslist:
            f.write(str(rms) + '\n')

    return


def findamplitudenoise(parmdb):
    """
      find the 'amplitude noise' in a parmdb, return non-clipped rms value
      """
    with h5parm.h5parm(parmdb, readonly=True) as H5:
        amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
        weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    idx = np.where(weights != 0.0)

    amps = amps[idx]
    amps = amps[np.isfinite(amps)]
    amps = np.log10(np.ndarray.flatten(amps))

    noise = findrms(amps)

    logger.info('Noise and clipped noise' + str(parmdb) + ' ' + str(np.std(amps)) + ' ' + str(noise))

    # do not return clipped noise, we are intersted in finding data with high outliers
    return np.std(amps)


def getimsize(boxfile, cellsize=1.5, increasefactor=1.2, DDE=None):
    """
   find imsize need to image a DS9 boxfile region
   """
    r = pyregion.open(boxfile)

    xs = np.ceil((r[0].coord_list[2]) * increasefactor * 3600. / cellsize)
    ys = np.ceil((r[0].coord_list[3]) * increasefactor * 3600. / cellsize)

    imsize = np.ceil(xs)  # // Round up decimals to an integer
    if (imsize % 2 == 1):
        imsize = imsize + 1

    # if int(imsize) < 512:
    #    imsize = 512
    return int(imsize)


def smoothsols(parmdb, ms, longbaseline, includesphase=True):
    losoto = 'losoto'

    cmdlosoto = losoto + ' ' + parmdb + ' '
    noise = findamplitudenoise(parmdb)
    smooth = False
    if noise >= 0.1:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '9', longbaseline, includesphase=includesphase)
        smooth = True
    if noise < 0.1 and noise >= 0.08:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '7', longbaseline, includesphase=includesphase)
        smooth = True
    if noise < 0.08 and noise >= 0.07:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '5', longbaseline, includesphase=includesphase)
        smooth = True
    if noise < 0.07 and noise >= 0.04:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '3', longbaseline, includesphase=includesphase)
        smooth = True
    if smooth:
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)
    return


def change_refant(parmdb, soltab):
    """
    Changes the reference antenna, if needed, for phase
    """
    with h5parm.h5parm(parmdb, readonly=False) as H5:
        phases = H5.getSolset('sol000').getSoltab(soltab).getValues()[0]
        weights = H5.getSolset('sol000').getSoltab(soltab).getValues(weight=True)[0]
        axesnames = H5.getSolset('sol000').getSoltab(soltab).getAxesNames()
        print('axesname', axesnames)
        # print 'SHAPE', np.shape(weights)#, np.size(weights[:,:,0,:,:])

        antennas = list(H5.getSolset('sol000').getSoltab(soltab).getValues()[1]['ant'])
        # print antennas

        if 'pol' in axesnames:
            idx0 = np.where((weights[:, :, 0, :, :] == 0.0))[0]
            idxnan = np.where((~np.isfinite(phases[:, :, 0, :, :])))[0]

            refant = ' '
            tmpvar = float(np.size(weights[:, :, 0, :, :]))
        else:
            idx0 = np.where((weights[:, 0, :, :] == 0.0))[0]
            idxnan = np.where((~np.isfinite(phases[:, 0, :, :])))[0]

            refant = ' '
            tmpvar = float(np.size(weights[:, 0, :, :]))

        if ((float(len(idx0)) / tmpvar) > 0.5) or ((float(len(idxnan)) / tmpvar) > 0.5):
            logger.info('Trying to changing reference anntena')

            for antennaid, antenna in enumerate(antennas[1::]):
                print(antenna)
                if 'pol' in axesnames:
                    idx0 = np.where((weights[:, :, antennaid + 1, :, :] == 0.0))[0]
                    idxnan = np.where((~np.isfinite(phases[:, :, antennaid + 1, :, :])))[0]
                    tmpvar = float(np.size(weights[:, :, antennaid + 1, :, :]))
                else:
                    idx0 = np.where((weights[:, antennaid + 1, :, :] == 0.0))[0]
                    idxnan = np.where((~np.isfinite(phases[:, antennaid + 1, :, :])))[0]
                    tmpvar = float(np.size(weights[:, antennaid + 1, :, :]))

                print(idx0, idxnan, ((float(len(idx0)) / tmpvar)))
                if ((float(len(idx0)) / tmpvar) < 0.5) and ((float(len(idxnan)) / tmpvar) < 0.5):
                    logger.info('Found new reference anntena,' + str(antenna))
                    refant = antenna
                    break

        if refant != ' ':
            for antennaid, antenna in enumerate(antennas):
                if 'pol' in axesnames:
                    phases[:, :, antennaid, :, :] = phases[:, :, antennaid, :, :] - phases[:, :, antennas.index(refant),
                                                                                    :,
                                                                                    :]
                else:
                    # phases[:,antennaid,:,:] = phases[:,antennaid,:,:] - phases[:,antennas.index(refant),:,:]
                    phases[:, :, antennaid, :] = phases[:, :, antennaid, :] - phases[:, :, antennas.index(refant), :]
            H5.getSolset('sol000').getSoltab(soltab).setValues(phases)
    return


def calculate_solintnchan(compactflux):
    if compactflux >= 3.5:
        nchan = 5.
        solint_phase = 1.

    if compactflux <= 3.5:
        nchan = 5.
        solint_phase = 1.

    if compactflux <= 1.0:
        nchan = 10.
        solint_phase = 2

    if compactflux <= 0.75:
        nchan = 15.
        solint_phase = 3.

    # solint_ap = 100. / np.sqrt(compactflux)
    solint_ap = 120. / (compactflux ** (1. / 3.))  # do third power-scaling
    # print solint_ap
    if solint_ap < 60.:
        solint_ap = 60.  # shortest solint_ap allowed
    if solint_ap > 180.:
        solint_ap = 180.  # longest solint_ap allowed

    if compactflux <= 0.4:
        nchan = 15.
        solint_ap = 180.

    return int(nchan), int(solint_phase), int(solint_ap)


def determine_compactsource_flux(fitsimage):
    """
    return total flux in compect sources in the fitsimage
    input: a fits image
    output: flux density in Jy
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

    return total_flux_gaus


def getdeclinationms(ms):
    """
    return approximate declination of pointing center of the ms
    input: a ms
    output: declination in degrees
    """
    t = table(ms + '/FIELD', readonly=True, ack=False)
    direction = np.squeeze(t.getcol('PHASE_DIR'))
    t.close()
    return 360. * direction[1] / (2. * np.pi)


# print getdeclinationms('1E216.dysco.sub.shift.avg.weights.set0.ms')

def declination_sensivity_factor(declination):
    """
    compute sensitivy factor lofar data, reduced by delclination, eq. from G. Heald.
    input declination is units of degrees
    """
    factor = 1. / (np.cos(2. * np.pi * (declination - 52.9) / 360.) ** 2)

    return factor


def is_binary(file_name):
    """
    Check if a file is binary or text-based.

    Args:
        file_name (str): Path to the file to check.

    Returns:
        bool: True if the file is binary, False if it is text.
    """
    try:
        import magic
    except ImportError:
        return False  # Assume non-binary if magic is not available

    mime = magic.Magic(mime=True).from_file(file_name)
    return 'text' not in mime


def has0coordinates(h5):
    """
    Check if the coordinates in the directions are 0, avoids being hit by this rare DP3 bug
    """
    h5 = tables.open_file(h5)
    for c in h5.root.sol000.source[:]:
        x, y = c[1]
        if x == 0. and y == 0.:
            h5.close()
            return True
    h5.close()
    return False


def findrefant_core(H5file):
    """
    Basically like the other one, but now it actually uses losoto
    """
    H = h5parm.h5parm(H5file)
    solset = H.getSolset('sol000')
    soltabs = solset.getSoltabNames()
    for st in soltabs:
        # Find a reasonable soltab to use
        if 'phase000' in st or 'rotation000' in st or 'tec000' in st or 'rotationmeasure000' in st:
            break
    soltab = solset.getSoltab(st)

    # Find core stations
    ants = soltab.getValues()[1]['ant']
    if 'ST001' in ants:
        H.close()
        return 'ST001'
    cs_indices = np.where(['CS' in ant for ant in ants])[0]

    #  MeerKAT
    if 'm013' in ants:
        H.close()
        return 'm013'
    if 'm012' in ants:
        H.close()
        return 'm012'
    if 'm011' in ants:
        H.close()
        return 'm011'
    if 'm010' in ants:
        H.close()
        return 'm010'
    if 'm009' in ants:
        H.close()
        return 'm009'
    if 'm002' in ants:
        H.close()
        return 'm002'
    if 'm001' in ants:
        H.close()
        return 'm001'

    #  GMRT
    if 'C00' in ants:
        H.close()
        return 'C00'
    if 'C01' in ants:
        H.close()
        return 'C01'
    if 'C02' in ants:
        H.close()
        return 'C02'
    if 'C03' in ants:
        H.close()
        return 'C03'
    if 'C04' in ants:
        H.close()
        return 'C04'
    if 'C05' in ants:
        H.close()
        return 'C05'
    if 'C06' in ants:
        H.close()
        return 'C06'
    if 'C08' in ants:
        H.close()
        return 'C08'
    if 'C09' in ants:
        H.close()
        return 'C09'
    if 'C10' in ants:
        H.close()
        return 'C10'
    if 'C11' in ants:
        H.close()
        return 'C11'    
    if 'C12' in ants:
        H.close()
        return 'C12'   
    if 'C13' in ants:
        H.close()
        return 'C13'       
    if 'C14' in ants:
        H.close()
        return 'C14'   

    #  ASKAP
    if 'ak01' in ants:
        H.close()
        return 'ak01'
    if 'ak02' in ants:
        H.close()
        return 'ak02'
    if 'ak03' in ants:
        H.close()
        return 'ak03'
    if 'ak04' in ants:
        H.close()
        return 'ak04'
    if 'ak05' in ants:
        H.close()
        return 'ak05'
    if 'ak06' in ants:
        H.close()
        return 'ak06'
    if 'ak07' in ants:
        H.close()
        return 'ak07'
    if 'ak08' in ants:
        H.close()
        return 'ak08'
    if 'ak09' in ants:
        H.close()
        return 'ak09'
    if 'ak10' in ants:
        H.close()
        return 'ak10'
    if 'ak11' in ants:
        H.close()
        return 'ak11'    
    if 'ak12' in ants:
        H.close()
        return 'ak12'   
    if 'ak13' in ants:
        H.close()
        return 'ak13'       
    if 'ak14' in ants:
        H.close()
        return 'ak14' 

    if len(cs_indices) == 0:  # in case there are no CS stations try with RS
        cs_indices = np.where(['RS' in ant for ant in ants])[0]

    # Find the antennas and which dimension that corresponds to
    ant_index = np.where(np.array(soltab.getAxesNames()) == 'ant')[0][0]

    # Find the antenna with the least flagged datapoint
    weightsum = []
    ndims = soltab.getValues()[0].ndim
    for cs in cs_indices:
        slc = [slice(None)] * ndims
        slc[ant_index] = cs
        weightsum.append(np.nansum(soltab.getValues(weight=True)[0][tuple(slc)]))
    maxant = np.argmax(weightsum)
    H.close()
    # force_close(H5file) this does not work for some reasons, because it is inside a function that's called in a function call ass argument?
    return ants[maxant]


def create_losoto_FRparsetplotfit(ms, refant='CS001LBA', outplotname='FR'):
    """
    Create a losoto parset to fit Faraday Rotation on the phase difference'.
    """
    parset = 'losotoFR_plotresult.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('[plotFRresult]\n')
    f.write('pol = [XX,YY]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname + 'phases_fitFR'))
    f.write('refAnt = %s\n\n\n' % refant)
    f.close()
    return parset


def create_losoto_FRparset(ms, refant='CS001LBA', freqminfitFR=20e6, outplotname='FR', onlyplotFRfit=False,
                           dejump=False):
    """
    Create a losoto parset to fit Faraday Rotation on the phase difference'.
    """
    parset = 'losotoFR.parset'
    os.system('rm -f ' + parset)
    f = open(parset, 'w')

    f.write('[duplicate]\n')
    f.write('operation = DUPLICATE\n')
    f.write('soltab = sol000/phase000\n')
    f.write('soltabOut = phaseOrig000\n\n\n')

    # should not be needed as already done but just in case....
    f.write('[reset]\n')
    f.write('operation = RESET\n')
    f.write('soltab = sol000/phase000\n')
    f.write('pol = YY\n')
    f.write('dataVal = 0.0\n\n\n')

    f.write('[plotphase]\n')
    f.write('pol = [XX,YY]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname + 'phases_beforeFR'))
    f.write('refAnt = %s\n\n\n' % refant)

    f.write('[faraday]\n')
    f.write('operation = FARADAY\n')
    f.write('soltab = sol000/phase000\n')
    f.write('refAnt = %s\n' % refant)
    f.write('maxResidual = 2\n')
    f.write('freq.minmaxstep = [%s,1e9]\n' % str(freqminfitFR))
    f.write('soltabOut = rotationmeasure000\n\n\n')

    f.write('[plotFR]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = sol000/rotationmeasure000\n')
    f.write('axesInPlot = [time]\n')
    f.write('axisInTable = ant\n')
    f.write('prefix = plotlosoto%s/%s\n\n\n' % (ms, outplotname + 'FR'))

    if dejump:
        f.write('[frdejump]\n')
        f.write('operation = FRJUMP\n')
        f.write('soltab = sol000/rotationmeasure000\n')
        f.write('soltabOut = rotationmeasure001\n')
        f.write('clipping = [%s,1e9]\n\n\n' % str(freqminfitFR))

        f.write('[plotFR_dejump]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = sol000/rotationmeasure001\n')
        f.write('axesInPlot = [time]\n')
        f.write('axisInTable = ant\n')
        f.write('prefix = plotlosoto%s/%s\n\n\n' % (ms, outplotname + 'FRdejumped'))

    f.write('[residuals]\n')
    f.write('operation = RESIDUALS\n')
    f.write('soltab = sol000/phase000\n')
    if dejump:
        f.write('soltabsToSub = rotationmeasure001\n\n\n')
    else:
        f.write('soltabsToSub = rotationmeasure000\n\n\n')

    f.write('[plotRES]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = sol000/phase000\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('AxisInTable = ant\n')
    f.write('AxisDiff = pol\n')
    f.write('plotFlag = True\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms, outplotname + 'residualphases_afterFR'))
    f.write('refAnt = %s\n' % refant)
    f.write('minmax = [-3.14,3.14]\n\n\n')

    f.close()
    return parset


# to remove H5/h5 and other files out of a wildcard selection if needed
def removenonms(mslist):
    """ Remove files that are not MS (ending on wrong extension)

    Args:
        mslist: measurement set list

    Returns:
        New list
    """
    newmslist = []
    for ms in mslist:
        if ms.lower().endswith(('.h5', '.png', '.parset', '.fits', '.backup', '.obj', '.log', '.reg', '.gz', '.tar',
                                '.tmp', '.ddfcache')) or \
                ms.lower().startswith(('plotlosoto', 'solintimage')):
            print('WARNING, removing ', ms, 'not a ms-type? Removed it!')
        else:
            newmslist.append(ms)
    return newmslist



def check_valid_ms(mslist):
    """
    Validates a list of Measurement Set (MS) directories.
    This function performs the following checks on each MS in the provided list:
    1. Verifies that each MS path exists and is a directory.
    2. Ensures that no MS path starts with a '.' character (to avoid relative or hidden paths).
    3. Checks that each MS contains more than 20 unique time steps (sufficient data for self-calibration).
    4. Ensures there are no duplicate MS entries in the list.
    Raises:
        Exception: If any MS directory does not exist.
        Exception: If any MS path starts with a '.' character.
        Exception: If any MS contains 20 or fewer unique time steps.
        Exception: If there are duplicate MS entries in the list.
    Args:
        mslist (list of str): List of paths to Measurement Set directories to validate.
    Returns:
        None
    """
    for ms in mslist:
        if not os.path.isdir(ms):
            print(ms, ' does not exist')
            raise Exception('ms does not exist')
        if ms.startswith("."):
            print(ms, ' This ms starts with a "." character, this is not allowed')
            raise Exception('Invalid ms name, do not use relative paths')

    for ms in mslist:
        t = table(ms, ack=False)
        times = np.unique(t.getcol('TIME'))

        if len(times) <= 20:
            print('---------------------------------------------------------------------------')
            print('ERROR, ', ms, 'not enough timesteps in ms/too short observation')
            print('---------------------------------------------------------------------------')
            raise Exception(
                'You are providing an MS with less than 21 timeslots, that is not enough to self-calibrate on')
        t.close()
    
    if any(mslist.count(x) > 1 for x in mslist):
        print('There are duplicates in the mslist, please remove them')
        raise Exception('There are duplicates in the mslist, please remove them')
    return


def makemaskthresholdlist(maskthresholdlist, stop):
    maskthresholdselfcalcycle = []
    for mm in range(stop):
        try:
            maskthresholdselfcalcycle.append(maskthresholdlist[mm])
        except:
            maskthresholdselfcalcycle.append(maskthresholdlist[-1])  # add last value
    return maskthresholdselfcalcycle


def niter_from_imsize(imsize):
    if imsize is None:
        print('imsize not set')
        raise Exception('imsize not set')
    if imsize < 1024:
        niter = 15000  # minimum  value
    else:
        niter = 15000 * int((float(imsize) / 1024.))

    return niter


def basicsetup(mslist):
    longbaseline = checklongbaseline(mslist[0])
    if args['removeinternational']:
        print('Forcing longbaseline to False as --removeinternational has been specified')
        longbaseline = False
        # Determine HBA or LBA
    t = table(mslist[0] + '/SPECTRAL_WINDOW', ack=False)
    freq = np.median(t.getcol('CHAN_FREQ')[0])
    t.close()
    # set telescope
    t = table(mslist[0] + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()

    if telescope == 'LOFAR':
        if freq < 100e6:
            LBA = True
            HBAorLBA = 'HBA'
        else:
            LBA = False
            HBAorLBA = 'LBA'
    else:
        HBAorLBA = 'other'
        LBA = False
        HBA = False

    # set some default values if not provided
    if args['uvmin'] is None:
        if longbaseline:
            args['uvmin'] = 20000.
        else:
            if LBA:
                if freq >= 40e6:
                    args['uvmin'] = 80.
                if freq < 40e6:
                    args['uvmin'] = 60.
            else:
                args['uvmin'] = 350.

    if type(args['uvminim']) is not list:
        if args['uvminim'] is None:
            if telescope == 'LOFAR':
                args['uvminim'] = 80.
            else:
                args['uvminim'] = 10.  # MeerKAt for example

    if args['pixelscale'] is None and telescope != 'MeerKAT':
        if LBA:
            if longbaseline:
                args['pixelscale'] = 0.08
            else:
                args['pixelscale'] = np.rint(3.0 * 54e6 / freq)
        else:
            if longbaseline:
                args['pixelscale'] = 0.04
            else:
                args['pixelscale'] = 1.5
    elif args['pixelscale'] is None and telescope == 'MeerKAT':
        if freq < 1e9:  # UHF-band
            args['pixelscale'] = pixelscale = 1.8
        elif freq < 2e9:  # L-band
            args['pixelscale'] = pixelscale = 1.
        elif freq < 4e9:  # S-band
            args['pixelscale'] = pixelscale = 0.5
    elif args['pixelscale'] is None and telescope == 'ASKAP':
        if freq < 1e9:  # UHF-band
            args['pixelscale'] = pixelscale = 2.0
        elif freq < 1.4e9:  # L-band-low
            args['pixelscale'] = pixelscale = 1.5
        elif freq < 2.0e9:  # L-band-high
            args['pixelscale'] = pixelscale = 1.0

    if (args['delaycal'] or args['auto']) and longbaseline and not LBA:
        if args['imsize'] is None:
            args['imsize'] = 2048

    if args['boxfile'] is not None:
        if args['DDE']:
            args['imsize'] = getimsize(args['boxfile'], args['pixelscale'], increasefactor=1.025)
        else:
            args['imsize'] = getimsize(args['boxfile'], args['pixelscale'])
    if args['niter'] is None:
        args['niter'] = niter_from_imsize(args['imsize'])

    if args['auto'] and not longbaseline:
        args['update_uvmin'] = True
        args['usemodeldataforsolints'] = True
        args['forwidefield'] = True
        args['autofrequencyaverage'] = True
        if LBA:
            args['BLsmooth_list'] = [True] * len(args['soltype_list'])
        else:
            args['update_multiscale'] = True  # HBA only
            if args['autofrequencyaverage_calspeedup']:
                args['soltypecycles_list'] = [0, 999, 2]
                args['stop'] = 8

    if args['paralleldeconvolution'] == 0: # means determine automatically
        if args['imsize'] > 1600 and telescope == 'MeerKAT':
            args['paralleldeconvolution'] = 1200
        elif args['imsize'] > 1600: 
            args['paralleldeconvolution'] =  np.min([2600, int(args['imsize'] / 2)])
            
    if args['auto'] and longbaseline and not args['delaycal']:
        args['update_uvmin'] = False
        args['usemodeldataforsolints'] = True
        args['forwidefield'] = True
        args['autofrequencyaverage'] = True
        args['update_multiscale'] = True

        args['soltypecycles_list'] = [0, 3]
        args['soltype_list'] = [args['targetcalILT'], 'scalarcomplexgain']
        if args['targetcalILT'] == 'tec' or args['targetcalILT'] == 'tecandphase':
            args['smoothnessconstraint_list'] = [0.0, 5.0]
        else:
            args['smoothnessconstraint_list'] = [10.0, 5.0]
            args['smoothnessreffrequency_list'] = [120.0, 0.0]
            args['smoothnessspectralexponent_list'] = [-1.0, -1.0]
            args['smoothnessrefdistance_list'] = [0.0, 0.0]
        args['uvmin'] = 20000

        if LBA:
            args['BLsmooth_list'] = [True] * len(args['soltype_list'])

    if args['delaycal'] and LBA:
        print('Option automated delaycal can only be used for HBA')
        raise Exception('Option automated delaycal can only be used for HBA')
    if args['delaycal'] and not longbaseline:
        print('Option automated delaycal can only be used for longbaseline data')
        raise Exception('Option automated delaycal can only be used for longbaseline data')

    if args['delaycal'] and longbaseline and not LBA:
        args['update_uvmin'] = False
        # args['usemodeldataforsolints'] = True # NEEDS SPECIAL SETTINGS to be implemented
        args['solint_list'] = "['5min','32sec','1hr']"
        args['forwidefield'] = True
        args['autofrequencyaverage'] = True
        args['update_multiscale'] = True

        args['soltypecycles_list'] = [0, 0, 3]
        args['soltype_list'] = ['scalarphasediff', 'scalarphase', 'scalarcomplexgain']
        args['smoothnessconstraint_list'] = [8.0, 2.0, 15.0]
        args['smoothnessreffrequency_list'] = [120., 144., 0.0]
        args['smoothnessspectralexponent_list'] = [-2.0, -1.0, -1.0]
        args['smoothnessrefdistance_list'] = [0.0, 0.0, 0.0]
        args['antennaconstraint_list'] = ['alldutch', None, None]
        args['nchan_list'] = [1, 1, 1]
        args['uvmin'] = 40000
        args['stop'] = 8
        args['maskthreshold'] = [5]
        args['docircular'] = True

    # reset tecandphase -> tec for LBA
    if LBA and args['usemodeldataforsolints']:
        args['soltype_list'][1] = 'tec'
        # args['soltype_list'][0] = 'tec'

        if freq < 30e6:
            # args['soltype_list'] = ['tecandphase','tec']    # no scalarcomplexgain in the list, do not use "tec" that gives rings around sources for some reason
            args['soltype_list'] = ['tecandphase', 'tecandphase']  # no scalarcomplexgain in the list

    if args['forwidefield']:
        args['doflagging'] = False
        args['clipsolutions'] = False

    automask = 2.5
    if args['maskthreshold'][-1] < automask:
        automask = args['maskthreshold'][-1]  # in case we use a low value for maskthreshold, like Herc A

    args['imagename'] = args['imagename'] + '_'
    if args['fitsmask'] is not None:
        fitsmask = args['fitsmask']
    else:
        if args['fitsmask_start'] is not None and args['start'] == 0:
            fitsmask = args['fitsmask_start']
        else:    
            fitsmask = None

    if args['boxfile'] is not None:
        outtarname = (args['boxfile'].split('/')[-1]).split('.reg')[0] + '.tar.gz'
    else:
        outtarname = 'calibrateddata' + '.tar.gz'

    maskthreshold_selfcalcycle = makemaskthresholdlist(args['maskthreshold'], args['stop'])

    # set telescope
    t = table(mslist[0] + '/OBSERVATION', ack=False)
    telescope = t.getcol('TELESCOPE_NAME')[0]
    t.close()
    # idgin = args['idg'] # store here as we update args['idg'] at some point to made image000 for selfcalcycle 0 in when --DDE is enabled

    if type(args['channelsout']) is str:
        if args['channelsout'] == 'auto':
            args['channelsout'] = set_channelsout(mslist)
        else:
            raise Exception("channelsout needs to be an integer or 'auto'")
    else:
        if args['channelsout'] < 1:
            print('channelsout', args['channelsout'])
            raise Exception("channelsout needs to be a positive integer")

    if type(args['fitspectralpol']) is str:
        if args['fitspectralpol'] == 'auto':
            args['fitspectralpol'] = set_fitspectralpol(args['channelsout'])
        else:
            raise Exception("channelsout needs to be an integer or 'auto'")

    if args['parallelgridding'] == 0: # means the user asks to set it out automatically
        if args['imsize'] > 20000:
            args['parallelgridding'] = 1
        if args['imsize'] < 18000:
            args['parallelgridding'] = 2
        if args['imsize'] < 12500:
            args['parallelgridding'] = 4
        if args['imsize'] < 6000:
            args['parallelgridding'] = 6

    return longbaseline, LBA, HBAorLBA, freq, automask, fitsmask, \
        maskthreshold_selfcalcycle, outtarname, telescope


def compute_phasediffstat(mslist, args, nchan='1953.125kHz', solint='10min'):
    """
    Compute phasediff statistics using circular standard deviations on the solutions of a scalarphasediff solve for
    international stations.

    The procedure and rational is described in Section 3.3 of de Jong et al. (2024)

    :param mslist: list of measurement sets
    :param args: input arguments
    :param nchan: n channels
    :param solint: solution interval
    """

    mslist_input = mslist[:]  # make a copy

    # Verify if we are in circular pol basis and do beam correction
    for ms in mslist:
        beamcor_and_lin2circ(ms, dysco=args['dysco'],
                             beam=set_beamcor(ms, args['beamcor']),
                             lin2circ=True,
                             losotobeamlib=args['losotobeamcor_beamlib'],
                             metadata_compression=args['metadata_compression'])

    # Phaseup if needed
    if args['phaseupstations']:
        mslist = phaseup(mslist, datacolumn='DATA', superstation=args['phaseupstations'],
                         dysco=args['dysco'], metadata_compression=args['metadata_compression'])

    # Solve and get best solution interval
    for ms_id, ms in enumerate(mslist):
        scorelist = []
        parmdb = 'scalarphasediffstat' + '_' + os.path.basename(ms) + '.h5'
        runDPPPbase(ms, str(solint) + 'min', nchan, parmdb, 'scalarphasediff', uvminscalarphasediff=0.0,
                    dysco=args['dysco'], modelstoragemanager=args['modelstoragemanager'])

        # Reference solution interval
        ref_solint = solint

        print(scorelist)

        # Set optimal std score
        optimal_score = 1.75

        if type(ref_solint) == str:
            if 'min' in ref_solint:
                ref_solint = float(re.findall(r'-?\d+', ref_solint)[0])
            elif ref_solint[-1] == 's' or ref_solint[-1] == 'sec':
                ref_solint = float(re.findall(r'-?\d+', ref_solint)[0]) // 60
            else:
                sys.exit("ERROR: ref_solint needs to be a float with solution interval in minutes "
                         "or string ending on min (minutes) or s/sec (seconds)")

        S = GetSolint(parmdb, optimal_score=optimal_score, ref_solint=ref_solint)

        # Write to STAT SCORE to original MS DATA-col header mslist_input
        with table(mslist_input[ms_id], readonly=False) as t:
            t.putcolkeyword('DATA', 'SCALARPHASEDIFF_STAT', S.cstd)

        if args['phasediff_only']:
            generate_phasediff_csv(glob.glob("scalarphasediffstat*.h5"))
        else:
            S.plot_C("T=" + str(round(S.best_solint, 2)) + " min", ms + '_phasediffscore.png')

    return


def multiscale_trigger(fitsmask):
    """
    Determines whether to enable multiscale cleaning based on the size of the largest island in a FITS mask.

    If the 'update_multiscale' argument is True and a FITS mask is provided, the function checks the size of the largest island
    (using `getlargestislandsize`). If this size exceeds 1000 pixels, multiscale cleaning is triggered by setting the `multiscale`
    flag to True. The function logs relevant information about the island size and the triggering of multiscale cleaning.

    Args:
        fitsmask: The FITS mask array to analyze for island sizes.

    Returns:
        bool: The value of the `multiscale` flag, possibly updated based on the FITS mask analysis.
    """
    # update multiscale cleaning setting if allowed/requested
    multiscale = args['multiscale']
    if args['update_multiscale'] and fitsmask is not None:
        print('Size largest island [pixels]:', getlargestislandsize(fitsmask))
        logger.info('Size largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
        if getlargestislandsize(fitsmask) > 1000:
            logger.info('Triggering multiscale clean')
            multiscale = True
    return multiscale


def update_uvmin(fitsmask, longbaseline, LBA):
    """
    Updates the 'uvmin' parameter in the global 'args' dictionary based on the properties of the provided FITS mask.

    If stacking is enabled in 'args', the function returns immediately without making changes.
    If 'longbaseline' is False, 'update_uvmin' is enabled in 'args', and a FITS mask is provided,
    the function checks the size of the largest island in the mask. If this size exceeds 1000 pixels,
    it sets 'uvmin' in 'args' to 750 (for non-LBA) or 250 (for LBA), indicating the presence of extended emission.
    Relevant information is printed and logged.

    Parameters:
        fitsmask: The FITS mask to analyze for extended emission.
        longbaseline (bool): Indicates whether long baselines are being used.
        LBA (bool): Indicates whether the observation is in LBA mode.

    Returns:
        None
    """
    # update uvmin if allowed/requested
    if args['stack']:
        return
    if not longbaseline and args['update_uvmin'] and fitsmask is not None:
        if getlargestislandsize(fitsmask) > 1000:
            print('Size of largest island [pixels]:', getlargestislandsize(fitsmask))
            logger.info('Size of largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
            if not LBA:
                print('Extended emission found, setting uvmin to 750 klambda')
                logger.info('Extended emission found, setting uvmin to 750 klambda')
                args['uvmin'] = 750
            else:
                print('Extended emission found, setting uvmin to 250 klambda')
                logger.info('Extended emission found, setting uvmin to 250 klambda')
                args['uvmin'] = 250
    return


def update_fitsmask(fitsmask, maskthreshold_selfcalcycle, selfcalcycle, args, mslist, telescope, longbaseline):
    """
    Updates or generates FITS mask files for self-calibration imaging cycles.
    This function manages the creation and updating of FITS mask files used in radio interferometric imaging
    self-calibration cycles. It supports different imagers (WSCLEAN, DDFACET), optional stacking, and
    extended masking for specific telescopes (LOFAR, MeerKAT). The function can merge user-supplied DS9 region
    files and uses external tools (breizorro, MakeMask.py) for mask generation. It returns the updated mask
    filename, a list of mask filenames for each imaging set, and the last used image filename.
    Args:
        fitsmask (str or None): Path to an existing FITS mask file, or None to generate a new mask.
        maskthreshold_selfcalcycle (dict): Dictionary mapping selfcal cycle indices to mask threshold values.
        selfcalcycle (int): Current self-calibration cycle index.
        args (dict): Dictionary of imaging and calibration parameters, including imager type, stacking, 
            mask options, image name, and more.
        mslist (list): List of measurement sets to process.
        telescope (str): Name of the telescope (e.g., 'LOFAR', 'MeerKAT').
        longbaseline (bool): Whether long baselines are used (affects mask generation).
    Returns:
        tuple:
            fitsmask (str or None): Path to the updated or generated FITS mask file, or None if not created.
            fitsmask_list (list): List of FITS mask filenames (or None) for each imaging set.
            imagename (str): The filename of the last processed image.
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
                        makemask_extended(imagename,'mask_extended.fits',kernel_size=39,rebin=8, threshold=7.5)
                    if telescope == 'MeerKAT':
                        makemask_extended(imagename,'mask_extended.fits',kernel_size=25,rebin=5, threshold=10.)   
                    mask_mergelist.append('mask_extended.fits')
                
                if which('breizorro') is not None:
                    cmdm = 'breizorro --make-binary --fill-holes --threshold=' + str(maskthreshold_selfcalcycle[selfcalcycle]) + \
                       ' --restored-image=' + imagename + ' --boxsize=30 --outfile=' + imagename + '.mask.fits'
                    if len(mask_mergelist) > 0:
                        cmdm += ' --merge=' + ','.join(map(str, mask_mergelist))
                else:
                    cmdm = 'MakeMask.py --Th=' + str(maskthreshold_selfcalcycle[selfcalcycle]) + \
                       ' --RestoredIm=' + imagename
                if fitsmask is not None:
                    if os.path.isfile(imagename + '.mask.fits'):
                        os.system('rm -f ' + imagename + '.mask.fits')
                print(cmdm)
                run(cmdm)
                fitsmask = imagename + '.mask.fits'
                fitsmask_list.append(fitsmask)
            else:
                fitsmask = None  # no masking requested as args['maskthreshold'] less/equal 0
                fitsmask_list.append(fitsmask)
        else:
            fitsmask_list.append(fitsmask)
    return fitsmask, fitsmask_list, imagename



def remove_model_columns(mslist):
    """
    Removes columns related to model data from a list of Measurement Sets (MS).

    This function iterates over each MS in the provided list, identifies columns whose names match the pattern 'MODEL_DATA*',
    and removes them using the `remove_column_ms` function.

    Args:
        mslist (list of str): List of paths to Measurement Set directories.

    Returns:
        None
    """
    print('Clean up MODEL_DATA type columns')
    for ms in mslist:
        t = table(ms)
        colnames = t.colnames()
        t.close()
        collist_del = [xdel for xdel in colnames if re.match('MODEL_DATA*', xdel)]
        for colname_remove in collist_del:
            remove_column_ms(ms, colname_remove)
    return


def set_fitsmask_restart(i, mslist):
    fitsmask_list = []
    for msim_id, mslistim in enumerate(nested_mslistforimaging(mslist, stack=args['stack'])):
        if args['stack']:
            stackstr = '_stack' + str(msim_id).zfill(2)
        else:
            stackstr = ''  # empty string

        if args['idg']:
            if os.path.isfile(args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits'):
                fitsmask = args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits'
            else:
                print('Cannot find: ' + args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits')
        else:
            if args['imager'] == 'WSCLEAN':
                if os.path.isfile(args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits'):
                    fitsmask = args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits'
                else:
                    print('Cannot find: ' + args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits')  
            if args['imager'] == 'DDFACET':
                if os.path.isfile(args['imagename'] + str(i - 1).zfill(3) + stackstr + '.app.restored.fits'):
                    fitsmask = args['imagename'] + str(i - 1).zfill(3) + stackstr + '.app.restored.fits.mask.fits'
                else:
                    print('Cannot find: ' + args['imagename'] + str(i - 1).zfill(3) + stackstr + '.app.restored.fits.mask.fits')  
        if args['channelsout'] == 1:
            if args['imager'] == 'WSCLEAN':
                fitsmask = args['imagename'] + str(i - 1).zfill(3) + stackstr + '-MFS-image.fits.mask.fits'
            if args['imager'] == 'DDFACET':
                fitsmask = args['imagename'] + str(i - 1).zfill(3) + stackstr + '.app.restored.fits.mask.fits'
            fitsmask = fitsmask.replace('-MFS', '').replace('-I', '')
        print('Appending fitsmask: ', fitsmask) 
        fitsmask_list.append(fitsmask)
    return fitsmask, fitsmask_list


def create_Ateam_seperation_plots(mslist, start=0):
    """
    Create Ateam and Sun, Moon, Jupiter seperation plots
    input: mslist (list), list of MS
    """
    if start != 0:
        return
    for ms in mslist:
        outputname = 'Ateam_' + ms + '.png'
        try:
            run(f'check_Ateam_separation.py --outputimage=' + outputname + ' ' + ms)
        except Exception:
            print(f"check_Ateam_separation.py does not exist")
    return

def nested_mslistforimaging(mslist, stack=False):
    if not stack:
        return [mslist]  # has format [[ms1.ms,ms2.ms,....]]
    else:
        # sort MSs in groups with same phase center
        for msid, ms in enumerate(mslist):
            with table(f'{ms}::FIELD') as tf:
                phasecenter = tf[0]['PHASE_DIR'][0]
                if msid == 0:
                    mslistreturn = [[ms]]
                    phasecenterlist = [phasecenter]
                else:
                    image_id = None
                    #iterate over already existing imaging groups
                    for iid, image_phasecenter in enumerate(phasecenterlist):
                        if np.linalg.norm(image_phasecenter-phasecenter) < 4.84e-8: # phase center distance less than 0.01 arcsec
                            image_id = iid
                    # if ms phase center already has an image list, add to that.
                    if image_id is not None:
                        mslistreturn[image_id].append(ms)
                    else:
                        phasecenterlist.append(phasecenter)
                        mslistreturn.append([ms])
        print(f'Found {len(mslistreturn)} imaging groups: ', mslistreturn)
        return mslistreturn  # has format [[ms1.ms,..],[ms2.ms,..],[...]]

def flag_autocorr(mslist):
    """
    Flag autocorrelations in MS
    input: list of MS
    """
    for ms in mslist:
       cmd = 'DP3 msin=' + ms + ' msout=. steps=[pr] '
       cmd += 'pr.type=preflagger pr.corrtype=auto'
       logger.info(cmd)
       run(cmd)
    return   


def mslist_return_stack(mslist, stack):
    """Returns the full mslist if 'stack' is True; otherwise, returns a list containing only the first element."""
    return mslist if stack else [mslist[0]]


def set_skymodels_external_surveys(args, mslist):
    # Make starting skymodel from TGSS, VLASS, or LOFAR/LINC GSM if requested
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
                print('You cannot provide a skymodel/skymodelpointsource file manually while using --startfromtgss')
                raise Exception(
                    'You cannot provide a skymodel/skymodelpointsource manually while using --startfromtgss')

    # --- VLASS ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromvlass'] and args['start'] == 0:
            if args['skymodel'] is None and args['skymodelpointsource'] is None:
                run(f'python {submodpath}/vlass_search.py ' + mstmp)
                skymodel_list.append(makeBBSmodelforVLASS('vlass_poststamp.fits', extrastrname=str(mstmp_id)))
            else:
                print('You cannot provide a skymodel/skymodelpointsource manually while using --startfromvlass')
                raise Exception(
                    'You cannot provide a skymodel/skymodelpointsource manually while using --startfromvlass')

    # --- GSM ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromgsm'] and args['start'] == 0:
            if args['skymodel'] is None and args['skymodelpointsource'] is None:
                skymodel_list.append(getGSM(mstmp, SkymodelPath='gsm' + str(mstmp_id) + '.skymodel',
                                            Radius=str(args['pixelscale'] * args['imsize'] / 3600.)))
            else:
                print('You cannot provide a skymodel/skymodelpointsource manually while using --startfromgsm')
                raise Exception('You cannot provide a skymodel/skymodelpointsource manually while using --startfromgsm')

    # --- Arbitrary FITS image ---
    for mstmp_id, mstmp in enumerate(mslist_return_stack(mslist, args['stack'])):
        if args['startfromimage'] and args['start'] == 0:
            if args['skymodel'].endswith('.fits') and args['skymodelpointsource'] is None:
                skymodel_list.append(makeBBSmodelforFITS(args['skymodel'], extrastrname=str(mstmp_id)))
            elif args['skymodel'].endswith('.fits') and (args['skymodelpointsource'] is not None):
                print('You cannot provide skymodelpointsource manually while using --startfromimage')
                raise Exception('You cannot provide skymodelpointsource manually while using --startfromimage')
            elif (not args['skymodel'].endswith('.fits')) and args['skymodelpointsource'] is None:
                print('skymodel must be a FITS file and have the fits extension while using --startfromimage')
                raise Exception('skymodel must be a FITS file while using --startfromimage')
            else:
                print('Something unknown went wrong. Please check your input.')
                raise Exception('Something unknown went wrong. Please check your input.')
        elif args['skymodel'] is not None:
            if not (args['startfromimage']) and args['skymodel'].lower().endswith('.fits'):
                print('Option --startfromimage must be set if using a FITS image as skymodel.')
                raise Exception('Option --startfromimage must be set if using a FITS image as skymodel.')

    # note if skymodel_list is not set (len==0), args['skymodel'] keeps it value from argparse
    if len(skymodel_list) > 1:  # so startfromtgss or startfromvlass was done and --stack was true
        args['skymodel'] = skymodel_list
    if len(skymodel_list) == 1:  # so startfromtgss or startfromvlass was done
        args['skymodel'] = skymodel_list[0]  # make string again, not a list type
    print(args['skymodel'])

    return args, tgssfitsfile


def early_stopping(station: str = 'international', cycle: int = None):
    """
    Determine early-stopping based on Neural network image validation and solutions and image-based metrics.

    :param station: international stations or dutch stations
    :param cycle: cycle number
    :param nn_model_cache: neural network model cache

    Returns: stop == True, continue == False
    """

    # Early stopping
    if cycle == args['start']:
        global nn_model, predict_nn
        try:
            # NN score
            from submods.source_selection.image_score import get_nn_model, predict_nn
            nn_model = get_nn_model(cache=args['nn_model_cache'])
        except (ImportError, SystemExit):
            logger.info(
                "WARNING: Issues with downloading/getting Neural Network model.. Skipping and continue without."
                "\nMost likely due to issues with accessing cortExchange or no internet access.")
            nn_model = None

    # Start only after cycle 3
    if cycle <= 3:
        return False

    # Get already obtained selfcal images and merged solutions
    images, mergedh5 = get_images_solutions()
    if len(images) == 0:
        logger.info("WARNING: Issues with finding images for early-stopping. Skipping and continue without...")
        return False
    qualitymetrics = quality_check(mergedh5, images, station)

    if nn_model is not None:
        predict_score = predict_nn(images[cycle], nn_model)
        logger.info(f"Neural network score: {predict_score}")
    else:
        predict_score = 1.0

    # Open selfcal quality CSV
    df = pd.read_csv(f"./selfcal_quality_plots/selfcal_performance_{qualitymetrics[0]}.csv")

    # Get image statistics
    minmax_ratio = df['min/max'][cycle] / df['min/max'][0]
    if minmax_ratio != minmax_ratio:  # check for nan in final cycle
        minmax_ratio = df['min/max'][cycle] / df['min/max'][0]
        rms_ratio = df['rms'][cycle] / df['rms'][0]
    else:
        rms_ratio = df['rms'][cycle] / df['rms'][0]

    iltj_id = parse_source_id(mergedh5[cycle])+"_"

    # Selection criteria (good image and stable solutions, if predict==1.0, no neural network is used)
    if (predict_score < 0.5 and df['phase'][cycle] < 0.1 and rms_ratio < 1.0 and minmax_ratio < 0.85) or \
        (predict_score < 0.5 and df['phase'][cycle] < 0.2 and rms_ratio < 0.9 and minmax_ratio < 0.5) or \
        (predict_score < 0.5 and df['phase'][cycle] < 0.3 and rms_ratio < 0.95 and minmax_ratio < 0.3) or \
        (predict_score < 0.15 and df['phase'][cycle] < 0.5 and rms_ratio < 1.0 and minmax_ratio < 1.0) or \
        (predict_score < 0.3 and df['phase'][cycle] < 0.05) or \
        (df['phase'][cycle] < 0.003) or \
        (df['phase'][cycle] < 0.1 and rms_ratio < 0.5 and minmax_ratio < 0.1 and predict_score == 1.0) or \
        (df['phase'][cycle] < 0.05 and minmax_ratio < 0.1 and rms_ratio < 0.5 and predict_score == 1.0):
        logger.info(f"Early-stopping at cycle {cycle}, because selfcal converged")
        logger.info(f"Best image: Cycle {max(df['min/max'].argmin(), df['rms'].argmin())}")
        logger.info(f"Best solutions: Cycle {df['phase'].argmin()}")
        logger.info(f'{mergedh5[cycle]} --> best_{iltj_id}solutions.h5')
        os.system(f'cp {mergedh5[cycle]} best_{iltj_id}solutions.h5')
        os.system(f'cp {images[cycle]} best_{images[cycle].split("/")[-1]}')
        return True
    elif (df['rms'][cycle-1] < df['rms'][cycle] and df['min/max'][cycle-1] < df['min/max'][cycle]
          and df['rms'][cycle-2] < df['rms'][cycle] and df['min/max'][cycle-2] < df['min/max'][cycle]
            and df['rms'][cycle-3] < df['rms'][cycle] and df['min/max'][cycle-3] < df['min/max'][cycle]) or \
            (minmax_ratio > 1.0 and rms_ratio > 1.0) or \
            (df['phase'][cycle-1] < df['phase'][cycle] and df['phase'][cycle-2] < df['phase'][cycle]
             and df['phase'][cycle-3] < df['phase'][cycle]) or \
            (1.0 > predict_score > 0.85 and cycle > 10):
        logger.info(f"Early-stopping at cycle {cycle}, because selfcal starts to diverge...")
        logger.info(f"Best image: Cycle {max(df['min/max'].argmin(), df['rms'].argmin())}")
        logger.info(f"Best solutions: Cycle {df['phase'].argmin()}")
        logger.info(f'{mergedh5[cycle]} --> best_{iltj_id}solutions.h5')
        os.system(f'cp {mergedh5[cycle]} best_{iltj_id}solutions.h5')
        os.system(f'cp {images[cycle]} best_{images[cycle].split("/")[-1]}')
        return True
    else:
        logger.info(f"No early-stopping at cycle {cycle}")
        logger.info(f"Best image: Cycle {max(df['min/max'].argmin(), df['rms'].argmin())}")
        logger.info(f"Best solutions: Cycle {df['phase'].argmin()}")
        if cycle == args['stop'] - 1:
            logger.info(f'{mergedh5[cycle]} --> best_{iltj_id}solutions.h5')
            os.system(f'cp {mergedh5[cycle]} best_{iltj_id}solutions.h5')
            os.system(f'cp {images[cycle]} best_{images[cycle].split("/")[-1]}')

    return False

###############################
############## MAIN ###########
###############################

def main():

    options = option_parser()

    # Temporary warning
    if options.helperscriptspathh5merge is not None or options.helperscriptspath is not None:
        print("WARNING: --helperscriptspath and --helperscriptspathh5merge are not being used anymore")

    # If a config file exists, then read the information. Priotise specified config over default.
    if os.path.isfile(options.configpath):
        config = options.configpath
    else:
        config = "facetselfcal_config.txt"

    if os.path.isfile(config):
        print("A config file (%s) exists, using it. This contains:" % config)
        parser = configparser.ConfigParser()
        # Preserve upper case in options
        parser.optionxform = str
        with open(config) as f:
            parser.read_string("[DEFAULT]\n" + f.read())
        for k, v in parser["DEFAULT"].items():
            print(f"{k} = {v}")
            if k not in vars(options).keys():
                raise KeyError(
                    "Encountered invalid option {:s} in config file {:s}.".format(
                        k, os.path.abspath(config)
                    )
                )
            if os.path.exists(os.path.expanduser(v)):
                setattr(options, k, v)
            else:
                setattr(options, k, ast.literal_eval(v))
    global args
    args = vars(options)

    with open("full_config.txt", "w") as file:
        for key, value in args.items():
            file.write(f"{key} = {value}\n")

    if args['stack']:
        args['dysco'] = False  # no dysco compression allowed as multiple various steps violate the assumptions that need to be valid for proper dysco compression
        args['noarchive'] = True
    if args['skymodelsetjy']:
        args['dysco'] = False  # no dysco compression allowed as CASA does not work with dysco compression

    global submodpath, datapath, facetselfcal_version
    datapath = os.path.dirname(os.path.abspath(__file__))
    submodpath = '/'.join(datapath.split('/')[0:-1])+'/submods'
    os.system(f'cp {submodpath}/polconv.py .')

    facetselfcal_version = '15.6.0'
    print_title(facetselfcal_version)

    # copy h5s locally
    for h5parm_id, h5parmdb in enumerate(args['preapplyH5_list']):
        if h5parmdb is not None:
            os.system('cp ' + h5parmdb + ' .')  # make them local because source direction will be updated for merging
            args['preapplyH5_list'][h5parm_id] = h5parmdb.split('/')[-1]  # update input list to local location

    # deal with glob-like string input for preapplybandpassH5_list
    if  len(args['preapplybandpassH5_list']) == 1 and \
        args['preapplybandpassH5_list'][0] is not None and \
        ('*' in args['preapplybandpassH5_list'][0] or '?' in args['preapplybandpassH5_list'][0]): 
             args['preapplybandpassH5_list'] = glob.glob(args['preapplybandpassH5_list'][0])
             assert len(args['preapplybandpassH5_list']) >= 1 # assert that something is found
             print('Found these bandpass solutions', args['preapplybandpassH5_list'])
    # copy bandpass h5s locally
    for h5parm_id, h5parmdb in enumerate(args['preapplybandpassH5_list']):
        if h5parmdb is not None:
            os.system('cp ' + h5parmdb + ' .')  # make them local because source direction will be updated for merging
            args['preapplybandpassH5_list'][h5parm_id] = h5parmdb.split('/')[-1]  # update input list to local location

    # reorder lists based on sorted(args['ms'])
    if type(args['skymodel']) is list:
        args['skymodel'] = [x for _, x in sorted(zip(args['ms'], args['skymodel']))]
    if type(args['wscleanskymodel']) is list:
        args['wscleanskymodel'] = [x for _, x in sorted(zip(args['ms'], args['wscleanskymodel']))]
    if type(args['skymodelpointsource']) is list:
        args['skymodelpointsource'] = [x for _, x in sorted(zip(args['ms'], args['skymodelpointsource']))]
    mslist = sorted(args['ms'])

    # remove non-ms that ended up in mslist
    # mslist = removenonms(mslist)

    # turn of metadata_compression for non-LOFAR data if needed
    set_metadata_compression(mslist)

    # remove trailing slashes
    mslist_tmp = []
    for ms in mslist:
        mslist_tmp.append(ms.rstrip('/'))
    mslist = mslist_tmp[:]  # .copy()

    # remove ms which are too short (to catch Elais-N1 case of 600s of data)
    if not args['testing']:
        check_valid_ms(mslist)
        # check if ms channels are equidistant in freuqency (to confirm DP3 concat was used properly)
        check_equidistant_freqs(mslist)
        # do some input checking
        inputchecker(args, mslist)

    # TEST ONLY REMOVE
    if False:
        modeldatacolumnsin = ['MODEL_DATA_DD0', 'MODEL_DATA_DD1', 'MODEL_DATA_DD2', 'MODEL_DATA_DD3', 'MODEL_DATA_DD4']
        soltypenumber = 0
        dirs, solints, smoothness, soltypelist_includedir = parse_facetdirections(args['facetdirections'], 0)
        modeldatacolumns, sourcedir_removed, id_kept = updatemodelcols_includedir(modeldatacolumnsin, soltypenumber,
                                                                                  soltypelist_includedir, mslist[0],
                                                                                  dryrun=True)
        # print(len(sourcedir_removed))
        # for ddir in sourcedir_removed:
        sourcedir_removed = sourcedir_removed.tolist()
        print(sourcedir_removed[0])
        print(modeldatacolumns)

        copy_over_solutions_from_skipped_directions(modeldatacolumnsin, id_kept)
        merge_splitted_h5_ordered(modeldatacolumnsin, 'test.h5', clean_up=False)
        sys.exit()

    # fix irregular time axes or avoid bloated MS, do this first before any other step
    # in case of multiple scans on the calibrator this avoids DP3 adding of lot flagged data in between
    if args['bandpass']:
        # args['skipbackup'] will set based on whether all MS were splitted
        # in case all MS were splitted a backup is not needed anymore
        args['flag_ampresetvalfactor'] = True  # set to True to flag bad bandpass solutions which values are too high or low
        if args['skymodel'] is None and not args['skymodelsetjy']: 
            args['skymodel'] = set_MeerKAT_bandpass_skymodel(mslist[0]) # try to set skymodel automatically    
        mslist, args['skipbackup'] = fix_equidistant_times(mslist, args['start'] != 0, 
                                                           dysco=args['dysco'], metadata_compression=args['metadata_compression'])

    if args['timesplitbefore']:
        mslist, args['skipbackup'] = fix_equidistant_times(mslist, args['start'] != 0, 
                                                           dysco=args['dysco'], metadata_compression=args['metadata_compression'])

    # cut ms if there are flagged times at the start or end of the ms
    if args['remove_flagged_from_startend']:
        mslist = sorted(remove_flagged_data_startend(mslist))

    if not args['skipbackup']:  # work on copy of input data as a backup
        print('Creating a copy of the data and work on that....')
        mslist = average(mslist, freqstep=[1] * len(mslist), timestep=1, start=args['start'], makecopy=True,
                         dysco=args['dysco'], useaoflagger=(args['useaoflagger'] and args['useaoflaggerbeforeavg']), 
                         aoflagger_strategy=args['aoflagger_strategy'], metadata_compression=args['metadata_compression'],
                         msinnchan=args['msinnchan'], msinstartchan=args['msinstartchan'], msinntimes=args['msinntimes'],
                         removeinternational=args['removeinternational'], phaseshiftbox=args['phaseshiftbox'],
                         removemostlyflaggedstations=args['removemostlyflaggedstations'])
        args['msinntimes'] = None # since we use it above set msinntimes to to None now
        args['msinnchan'] = None  # since we use it above set msinnchan to None now
        args['msinstartchan'] = 0  # since we use it above set msinstartchan to 0 now
        args['removeinternational'] = False  # since we use it above set removeinternational to False now
        args['removemostlyflaggedstations'] = False  # since we use it above set removemostlyflaggedstations to False now
        args['phaseshiftbox'] = None  # since we use it above set phaseshiftbox to None now
        if args['useaoflagger'] and args['useaoflaggerbeforeavg']:
            args['useaoflagger'] = False # turn off now because we used it above    

    # take out bad WEIGHT_SPECTRUM values if weightspectrum_clipvalue is set
    if args['weightspectrum_clipvalue'] is not None:
        fix_bad_weightspectrum(mslist, clipvalue=args['weightspectrum_clipvalue'])

    # extra flagging if requested
    #if args['start'] == 0 and args['useaoflagger'] and args['useaoflaggerbeforeavg']:
    #    runaoflagger(mslist, strategy=args['aoflagger_strategy'])

    # create Ateam plots
    if not args['phasediff_only']:
        create_Ateam_seperation_plots(mslist, start=args['start'])

    # fix UVW coordinates (for time averaging with MeerKAT data)
    if args['start'] == 0: fix_uvw(mslist)

    # fix antenna info for GMRT data (nothing happens for other telescopes)
    if args['start'] == 0: fix_antenna_info_gmrt(mslist)

    # fix irregular time axes if needed (do this after flagging)
    # in case --bandpass was set this step should not do anything because
    # the MS were already splitted before in case of gaps (so they will be regular now)
    # the [0] at the end is to avoid the extra output returned by fix_equidistant_times
    mslist = fix_equidistant_times(mslist, args['start'] != 0, \
                                   dysco=args['dysco'], metadata_compression=args['metadata_compression'])[0]

    # fix TIME axis for GMRT data (nothing happens for other telescopes)
    if args['start'] == 0: fix_time_axis_gmrt(mslist)

    # reset weights if requested
    if args['resetweights']:
        for ms in mslist:
            cmd = "'update " + ms + " set WEIGHT_SPECTRUM=WEIGHT_SPECTRUM_SOLVE'"
            run("taql " + cmd)

    # SETUP VARIOUS PARAMETERS
    longbaseline, LBA, HBAorLBA, freq, automask, fitsmask, maskthreshold_selfcalcycle, \
        outtarname, telescope = basicsetup(mslist)

    # set model storagemanager
    if args['modelstoragemanager'] == 'stokes_i' and '-model-storage-manager' in subprocess.check_output(['wsclean'], text=True):
        if is_stokesi_modeltype_allowed(args, telescope):
            print('Using stokes_i model compression')
        else:
            print('Cannot use stokes_i model compression')
            args['modelstoragemanager'] = None
    else:
        args['modelstoragemanager'] = None  # we are here because wsclean does not support -model-storage-manager   

    print(args['modelstoragemanager'])

    # check if we could average more
    avgfreqstep = []  # vector of len(mslist) with average values, 0 means no averaging
    for ms in mslist:
        if args['avgfreqstep'] is None and args['autofrequencyaverage'] and not LBA \
                and not args['autofrequencyaverage_calspeedup']:  # autoaverage
            avgfreqstep.append(findfreqavg(ms, float(args['imsize'])))
        else:
            if args['avgfreqstep'] is not None: 
                avgfreqstep.append(args['avgfreqstep'])  # take over handpicked average value
            else:
                if args['useaoflagger']: # so we also trigger if flagging is requested
                    avgfreqstep.append(1) 
                else:
                    avgfreqstep.append(0)  # put to zero, zero means no average

    # COMPUTE PHASE-DIFF statistic
    if args['compute_phasediffstat']:
        if longbaseline and len(args['soltype_list'])<2: # of more than one soltype list, phasediff stat is generated later on in the script
            compute_phasediffstat(mslist, args)
            if args['phasediff_only']:
                return
        else:
            logger.info("--compute-phasediffstat requested but no long-baselines in dataset.")

    # set once here, preserve original mslist in case --removeinternational was set
    if args['removeinternational'] is not None:
        # used for h5_merge add_ms_stations option
        mslist_beforeremoveinternational = mslist[:]  # copy by slicing otherwise list refers to original
    else:
        mslist_beforeremoveinternational = None

    if not check_pointing_centers(mslist): args['phaseshiftbox'] = 'align' # set phaseshiftbox

    # AVERAGE if requested/possible
    mslist = average(mslist, freqstep=avgfreqstep, timestep=args['avgtimestep'],
                     start=args['start'], msinnchan=args['msinnchan'], msinstartchan=args['msinstartchan'],
                     phaseshiftbox=args['phaseshiftbox'], msinntimes=args['msinntimes'],
                     dysco=args['dysco'], removeinternational=args['removeinternational'],
                     removemostlyflaggedstations=args['removemostlyflaggedstations'], useaoflagger=args['useaoflagger'], aoflagger_strategy=args['aoflagger_strategy'], useaoflaggerbeforeavg=args['useaoflaggerbeforeavg'],
                     metadata_compression=args['metadata_compression'])

    for ms in mslist:
        compute_distance_to_pointingcenter(ms, HBAorLBA=HBAorLBA, warn=longbaseline, returnval=False)

    # extra flagging if requested
    #if args['start'] == 0 and args['useaoflagger'] and not args['useaoflaggerbeforeavg']:
    #    runaoflagger(mslist, strategy=args['aoflagger_strategy'])

    # compute bandwidth smearing
    with table(mslist[0] + '/SPECTRAL_WINDOW', ack=False) as t:
        bwsmear = bandwidthsmearing(np.median(t.getcol('CHAN_WIDTH')), np.min(t.getcol('CHAN_FREQ')[0]),
                                    float(args['imsize']))
 
    # backup flagging column for option --restoreflags if needed
    if args['restoreflags']:
        for ms in mslist:
            create_backup_flag_col(ms)

    # LOG INPUT SETTINGS
    logbasicinfo(args, fitsmask, mslist, facetselfcal_version, sys.argv)

    # Make starting skymodel from TGSS or VLASS survey if requested

    args, tgssfitsfile = set_skymodels_external_surveys(args, mslist)

    nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
        smoothnessspectralexponent_list, smoothnessrefdistance_list, \
        antennaconstraint_list, resetsols_list, resetdir_list, soltypecycles_list, \
        uvmin_list, uvmax_list, uvminim_list, uvmaxim_list, normamps_list, solve_msinnchan_list, \
        solve_msinstartchan_list, antenna_averaging_factors_list, antenna_smoothness_factors_list = \
        setinitial_solint(mslist, options)

 
    # Get restoring beam for DDFACET in case it is needed
    restoringbeam = calculate_restoringbeam(mslist, LBA)

    # set once here, will be updated in the loop below if phaseup is requested
    if args['phaseupstations'] is not None:
        # used for h5_merge add_CS option
        mslist_beforephaseup = mslist[:]  # note copy by slicing otherwise list refers to original
    else:
        mslist_beforephaseup = None

    # extra smearing flagging
    if args['flagtimesmeared'] and args['start'] == 0:
        for ms in mslist:
            flag_smeared_data(ms)
    
    # flag autocorrelations for MeerKAT (these are not flagged by the SDP pipeline)
    if telescope == 'MeerKAT' and args['start'] == 0:
        flag_autocorr(mslist)
  
    wsclean_h5list = []
    facetregionfile = None
    soltypelist_includedir = None
    modeldatacolumns = []
    if args['stack']:
        fitsmask_list = [None] * len(mslist)
    else:
        fitsmask_list = [
            fitsmask]  # *len(mslist) last part not needed because of the enumerate(nested_mslistforimaging(mslist, stack=args['stack']))

    if args['groupms_h5facetspeedup'] and args['start'] == 0 and len(mslist) > 1:
        concat_ms_wsclean_facetimaging(mslist)

    # create facets.reg so we have it avaialble for image000
    # so that we can use WSClean facet mode, but without having h5 DDE solutions
    if args['facetdirections'] is not None and args['start'] == 0:
        create_facet_directions(None, 0, ms=mslist[0], imsize=args['imsize'],
                                pixelscale=args['pixelscale'], facetdirections=args['facetdirections'])
        facetregionfile = 'facets.reg'  # so when making image000 we can use it without having h5 DDE solutions

    if args['start'] > 0 and  args['stop'] == args['start'] and args['remove_outside_center']:
        print('Only doing an imaging and extract step')
        args['stop'] = args['stop'] + 1 # increase stop so we go into the selfcal loop
        remove_outside_center_only = True
    else:
        remove_outside_center_only = False

    # check if we can use -apply-facet-beam or disable_primary_beam needs to be set
    check_applyfacetbeam_MeerKAT(mslist, args['imsize'], args['pixelscale'], telescope, args['DDE'])

    # ----- START SELFCAL LOOP -----
    for i in range(args['start'], args['stop']):

        # update removenegativefrommodel setting, for high dynamic range it is better to keep negative clean components (based on a very clear 3C84 test case)
        if args['autoupdate_removenegativefrommodel'] and i > 1 and not args['DDE']:
            args['removenegativefrommodel'] = False
        if args['autoupdate_removenegativefrommodel'] and args[
            'DDE']:  # never remove negative clean components for a DDE solve
            args['removenegativefrommodel'] = False
        if args['autoupdate_removenegativefrommodel'] and telescope == 'MeerKAT':
            # never remove negative clean components for MeerKAT as the image quality is already very good
            args['removenegativefrommodel'] = False

        # AUTOMATICALLY PICKUP PREVIOUS MASK (in case of a restart) and trigger multiscale
        if (i > 0) and (args['fitsmask'] is None):
            fitsmask, fitsmask_list = set_fitsmask_restart(i, mslist)
            # update to multiscale cleaning if large island is present
            args['multiscale'] = multiscale_trigger(fitsmask)
            # update uvmin if allowed/requested
            update_uvmin(fitsmask, longbaseline, LBA)
            # update channelsout
            args['channelsout'] = update_channelsout(i - 1, mslist)
            # update fitspectralpol
            args['fitspectralpol'] = update_fitspectralpol()

        # BEAM CORRECTION AND/OR CONVERT TO CIRCULAR/LINEAR CORRELATIONS
        for ms in mslist:
            if ((args['docircular'] or args['dolinear']) or (set_beamcor(ms, args['beamcor']))) and (i == 0):
                beamcor_and_lin2circ(ms, dysco=args['dysco'],
                                     beam=set_beamcor(ms, args['beamcor']),
                                     lin2circ=args['docircular'],
                                     circ2lin=args['dolinear'],
                                     losotobeamlib=args['losotobeamcor_beamlib'], idg=args['idg'],
                                     metadata_compression=args['metadata_compression'])

        # TMP AVERAGE TO SPEED UP CALIBRATION
        if args['autofrequencyaverage_calspeedup'] and i == 0:
            avgfreqstep = []
            mslist_backup = mslist[:]  # make a backup list, note copy by slicing otherwise list refers to original
            for ms in mslist:
                avgfreqstep.append(findfreqavg(ms, float(args['imsize']), bwsmearlimit=3.5))
            mslist = average(mslist, freqstep=avgfreqstep, timestep=4, 
                             dysco=args['dysco'], metadata_compression=args['metadata_compression'])
        if args['autofrequencyaverage_calspeedup'] and i == args['stop'] - 3:
            mslist = mslist_backup[:]  # reset back, note copy by slicing otherwise list refers to original
            preapply(create_mergeparmdbname(mslist, i - 1, autofrequencyaverage_calspeedup=True), mslist, updateDATA=False,
                     dysco=args['dysco'])  # do not overwrite DATA column

        # PHASE-UP if requested
        if args['phaseupstations'] is not None:
            if (i == 0) or (i == args['start']):
                mslist = phaseup(mslist, datacolumn='DATA', superstation=args['phaseupstations'],
                                 start=i, dysco=args['dysco'], metadata_compression=args['metadata_compression'])

        # PRE-APPLY BANDPASS-TYPE SOLUTIONS
        if (args['preapplybandpassH5_list'][0]) is not None and i == 0:
            preapply_bandpass(args['preapplybandpassH5_list'], mslist, dysco=args['dysco'], 
                              updateweights=args['preapplybandpassH5_updateweights'])
            if args['useaoflagger_afterbandpassapply']:
                aoflagger_column(mslist, aoflagger_strategy=args['aoflagger_strategy_afterbandpassapply'], column='DATA')

        # PRE-APPLY SOLUTIONS (from a nearby direction for example)
        if (args['preapplyH5_list'][0]) is not None and i == 0:
            preapply(args['preapplyH5_list'], mslist, dysco=args['dysco'])

        if args['stopafterpreapply']:
            print('Stopping as requested via --stopafterpreapply')
            return

        # REMOVE EXISTING MODEL COLUMNS IN CASE OF a RESTART:
        # this can be important in case the model column storage manager has changed
        if args['start'] != 0: remove_model_columns(mslist)
        
        # CALIBRATE AGAINST THE INITAL SKYMODEL (selfcalcycle 0) IF REQUESTED
        if (args['skymodel'] is not None or args['skymodelpointsource'] is not None
            or args['wscleanskymodel'] is not None or args['skymodelsetjy']) and (i == 0):
            # Function that
            # add patches for DDE predict
            # also do prepare_DDE
            if args['DDE']:
                modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, soltypelist_includedir = (
                    prepare_DDE(args['skymodel'], i, mslist,
                                DDE_predict='DP3', restart=False, skyview=tgssfitsfile,
                                wscleanskymodel=args['wscleanskymodel'], skymodel=args['skymodel']))

                if candidate_solints is not None:
                    candidate_solints = np.swapaxes(np.array([candidate_solints] * len(mslist)), 1, 0).T.tolist()
                    solint_list = candidate_solints
                if candidate_smoothness is not None:
                    candidate_smoothness = np.swapaxes(np.array([candidate_smoothness] * len(mslist)), 1, 0).T.tolist()
                    smoothnessconstraint_list = candidate_smoothness
            else:
                dde_skymodel = None
            wsclean_h5list = calibrateandapplycal(mslist, i, solint_list, nchan_list, 
                                                  soltypecycles_list, smoothnessconstraint_list,
                                                  smoothnessreffrequency_list,
                                                  smoothnessspectralexponent_list, smoothnessrefdistance_list,
                                                  antennaconstraint_list, resetsols_list, resetdir_list,
                                                  normamps_list, BLsmooth_list, solve_msinnchan_list, solve_msinstartchan_list,
                                                  antenna_averaging_factors_list, antenna_smoothness_factors_list,
                                                  normamps=args['normampsskymodel'],
                                                  skymodel=args['skymodel'],
                                                  predictskywithbeam=args['predictskywithbeam'],
                                                  longbaseline=longbaseline,
                                                  skymodelsource=args['skymodelsource'],
                                                  skymodelpointsource=args['skymodelpointsource'],
                                                  wscleanskymodel=args['wscleanskymodel'], skymodelsetjy=args['skymodelsetjy'],
                                                  mslist_beforephaseup=mslist_beforephaseup,
                                                  telescope=telescope,
                                                  modeldatacolumns=modeldatacolumns, dde_skymodel=dde_skymodel,
                                                  DDE_predict=set_DDE_predict_skymodel_solve(args['wscleanskymodel']),
                                                  mslist_beforeremoveinternational=mslist_beforeremoveinternational,
                                                  soltypelist_includedir=soltypelist_includedir)

        # Generate phasediff stat CSV here if more than 2
        if args['compute_phasediffstat'] and len(args['soltype_list'])>=2:
            if args['solint_list'][0]=='10min':
                generate_phasediff_csv(glob.glob("scalarphasediff*.h5"))
            else:
                print("WARNING: Cannot generate phasediff CSV because solution interval for scalarphasediff is not 10min")
        if args['phasediff_only']:
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return

        # SET MULTISCALE
        if args['multiscale'] and i >= args['multiscale_start']:
            multiscale = True
        else:
            multiscale = False

        # RESTART FOR A DDE RUN, set modeldatacolumns and dde_skymodel
        if args['DDE'] and args['start'] != 0 and i == args['start']:
            if args['auto_directions']: args['facetdirections'] = 'directions_' + str(i-1).zfill(3) + '.txt'
           
            modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, soltypelist_includedir = (
                prepare_DDE(args['imagename'], i, mslist,
                            DDE_predict=args['DDE_predict'], restart=True, telescope=telescope))
            wsclean_h5list = list(np.load('wsclean_h5list' + str(i-1).zfill(3) + '.npy'))
            # re-create facets.reg here
            # in a restart the number of directions from the previous facets.reg might not match that in the h5
            create_facet_directions(args['imagename'], i, ms=mslist[0], imsize=args['imsize'],
                            pixelscale= args['pixelscale'], via_h5=True, h5=wsclean_h5list[0])

        # RESTART FOR A DI RUN, set CORRECTED_DATA from previous selfcal cycle
        if not args['DDE'] and  args['start'] != 0 and i == args['start']: 
            applycal_restart_di(mslist, i)

        #  --- start imaging part ---
        for msim_id, mslistim in enumerate(nested_mslistforimaging(mslist, stack=args['stack'])):
            if args['stack']:
                stackstr = '_stack' + str(msim_id).zfill(2)
            else:
                stackstr = ''  # empty string
            if len(modeldatacolumns) > 1:
                facetregionfile = 'facets.reg'
            # else:
            # if args['DDE'] and i == 0: # we are making image000 without having DDE solutions yet
            # if telescope == 'LOFAR' and not args['disable_IDG_DDE_predict']: # so image000 has model-pb
            #    args['idg'] = True
            makeimage(mslistim, args['imagename'] + str(i).zfill(3) + stackstr,
                      args['pixelscale'], args['imsize'],
                      args['channelsout'], args['niter'], args['robust'],
                      multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[msim_id],
                      uvminim=args['uvminim'], predict=not args['stopafterskysolve'],
                      fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                      restoringbeam=restoringbeam, automask=automask,
                      removenegativecc=args['removenegativefrommodel'],
                      paralleldeconvolution=args['paralleldeconvolution'],
                      parallelgridding=args['parallelgridding'],
                      h5list=wsclean_h5list,
                      facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                      stack=args['stack'],
                      disable_primarybeam_image=args['disable_primary_beam'],
                      disable_primarybeam_predict=args['disable_primary_beam'],
                      fulljones_h5_facetbeam=not args['single_dual_speedup'])
            
            if remove_outside_center_only:
                remove_outside_box(mslist, args['imagename'] + str(i).zfill(3), args['pixelscale'],
                    args['imsize'], args['channelsout'], single_dual_speedup=args['single_dual_speedup'],
                    dysco=args['dysco'], userbox=args['remove_outside_center_box'], idg=args['idg'],
                    h5list=wsclean_h5list, facetregionfile=facetregionfile,
                    disable_primary_beam=args['disable_primary_beam'], 
                    modelstoragemanager=args['modelstoragemanager'], parallelgridding=args['parallelgridding'],
                    metadata_compression=args['metadata_compression'])
                if not args['keepmodelcolumns']: remove_model_columns(mslist)
                return
            
            # args['idg'] = idgin # set back
            if args['makeimage_ILTlowres_HBA']:
                if args['phaseupstations'] is None:
                    briggslowres = -1.5
                else:
                    briggslowres = -0.5
                makeimage(mslistim, args['imagename'] + '1.2arcsectaper' + str(i).zfill(3) + stackstr,
                          args['pixelscale'], args['imsize'],
                          args['channelsout'], args['niter'], briggslowres, uvtaper='1.2arcsec',
                          multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[msim_id],
                          uvminim=args['uvminim'], uvmaxim=args['uvmaxim'], fitspectralpol=args['fitspectralpol'],
                          automask=automask, removenegativecc=False,
                          predict=False,
                          paralleldeconvolution=args['paralleldeconvolution'],
                          parallelgridding=args['parallelgridding'],
                          h5list=wsclean_h5list, stack=args['stack'],
                          disable_primarybeam_image=args['disable_primary_beam'],
                          disable_primarybeam_predict=args['disable_primary_beam'],
                          fulljones_h5_facetbeam=not args['single_dual_speedup'])
            if args['makeimage_fullpol']:
                makeimage(mslistim, args['imagename'] + 'fullpol' + str(i).zfill(3) + stackstr,
                          args['pixelscale'], args['imsize'],
                          args['channelsout'], args['niter'], args['robust'],
                          multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[msim_id],
                          uvminim=args['uvminim'], uvmaxim=args['uvmaxim'], fitspectralpol=0,
                          automask=automask, removenegativecc=False, predict=False,
                          paralleldeconvolution=args['paralleldeconvolution'],
                          parallelgridding=args['parallelgridding'],
                          fullpol=True,
                          facetregionfile=facetregionfile,
                          stack=args['stack'],
                          disable_primarybeam_image=args['disable_primary_beam'],
                          disable_primarybeam_predict=args['disable_primary_beam'],
                          fulljones_h5_facetbeam=not args['single_dual_speedup'])

            # PLOT IMAGE
            plotimage(i, stackstr, mask=fitsmask_list[msim_id], regionfile='facets.reg' if (args['DDE'] and facetregionfile is not None) else None)
        #  --- end imaging part ---

        modeldatacolumns = []
        if args['DDE']:
            if args['auto_directions']: args['facetdirections'] = auto_direction(i)
            modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, soltypelist_includedir = (
                prepare_DDE(args['imagename'], i, mslist,
                            DDE_predict=args['DDE_predict'], telescope=telescope))

            if candidate_solints is not None:
                candidate_solints = np.swapaxes(np.array([candidate_solints] * len(mslist)), 1, 0).T.tolist()
                solint_list = candidate_solints
            if candidate_smoothness is not None:
                candidate_smoothness = np.swapaxes(np.array([candidate_smoothness] * len(mslist)), 1, 0).T.tolist()
                smoothnessconstraint_list = candidate_smoothness
        else:
            dde_skymodel = None

        if args['stopafterskysolve']:
            print('Stopping as requested via --stopafterskysolve')
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return
        
        if args['bandpass'] and args['bandpass_stop'] == 0: 
            print('Stopping as requested via --bandpass and compute bandpass')
            for parmdb in create_mergeparmdbname(mslist, i, skymodelsolve=True):
                run('losoto ' + parmdb + ' ' + create_losoto_bandpassparset('a&p'))
                set_weights_h5_to_one(parmdb)
                if os.path.isfile(parmdb.replace('merged_', 'bandpass_')):
                    # remove existing bandpass file
                    os.system('rm -f ' + parmdb.replace('merged_', 'bandpass_'))
                os.system('mv ' + parmdb + ' ' + parmdb.replace('merged_', 'bandpass_'))    
                if not args['keepmodelcolumns']: remove_model_columns(mslist)
                return
        
        # run aoflagger on the corrected data culumn if requested  
        if i in args['useaoflagger_correcteddata_selfcalcycle_list'] and args['useaoflagger_correcteddata']:
            aoflagger_column(mslist, aoflagger_strategy=args['aoflagger_strategy_correcteddata'], column='CORRECTED_DATA')
                    
        # REDETERMINE SOLINTS IF REQUESTED
        if (i >= 0) and (args['usemodeldataforsolints']):
            print('Recomputing solints .... ')
            nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
                smoothnessspectralexponent_list, smoothnessrefdistance_list, \
                antennaconstraint_list, resetsols_list, resetdir_list, \
                soltypecycles_list, normamps_list = \
                auto_determinesolints(mslist, args['soltype_list'],
                                      longbaseline, LBA,
                                      innchan_list=nchan_list, insolint_list=solint_list,
                                      insmoothnessconstraint_list=smoothnessconstraint_list,
                                      insmoothnessreffrequency_list=smoothnessreffrequency_list,
                                      insmoothnessspectralexponent_list=smoothnessspectralexponent_list,
                                      insmoothnessrefdistance_list=smoothnessrefdistance_list,
                                      inantennaconstraint_list=antennaconstraint_list,
                                      inresetsols_list=resetsols_list,
                                      inresetdir_list=resetdir_list,
                                      innormamps_list=normamps_list,
                                      inBLsmooth_list=BLsmooth_list,
                                      insoltypecycles_list=soltypecycles_list, redo=True,
                                      tecfactorsolint=args['tecfactorsolint'],
                                      gainfactorsolint=args['gainfactorsolint'],
                                      gainfactorsmoothness=args['gainfactorsmoothness'],
                                      phasefactorsolint=args['phasefactorsolint'],
                                      delaycal=args['delaycal'])

        # CALIBRATE AND APPLYCAL
        wsclean_h5list = calibrateandapplycal(mslist, i, solint_list, nchan_list, 
                                              soltypecycles_list,
                                              smoothnessconstraint_list, smoothnessreffrequency_list,
                                              smoothnessspectralexponent_list, smoothnessrefdistance_list,
                                              antennaconstraint_list, resetsols_list, resetdir_list,
                                              normamps_list, BLsmooth_list,
                                              solve_msinnchan_list, solve_msinstartchan_list,
                                              antenna_averaging_factors_list, antenna_smoothness_factors_list,
                                              normamps=args['normampsskymodel'] if args['keepusingstartingskymodel'] else args['normamps'],
                                              skymodel=args['skymodel'] if args['keepusingstartingskymodel'] else None,
                                              skymodelpointsource=args['skymodelpointsource'] if args['keepusingstartingskymodel'] else None,
                                              wscleanskymodel=args['wscleanskymodel'] if args['keepusingstartingskymodel'] else None, 
                                              skymodelsetjy=args['skymodelsetjy'] if args['keepusingstartingskymodel'] else False,
                                              longbaseline=longbaseline,
                                              predictskywithbeam=args['predictskywithbeam'], skymodelsource=args['skymodelsource'],                                          
                                              mslist_beforephaseup=mslist_beforephaseup,
                                              modeldatacolumns=modeldatacolumns, dde_skymodel=dde_skymodel,
                                              DDE_predict=args['DDE_predict'],
                                              mslist_beforeremoveinternational=mslist_beforeremoveinternational,
                                              soltypelist_includedir=soltypelist_includedir)


        if args['bandpass'] and i >=args['bandpass_stop']: 
            print('Stopping as requested via --bandpass and compute bandpass')
            for parmdb in create_mergeparmdbname(mslist, i, skymodelsolve=True):
                run('losoto ' + parmdb + ' ' + create_losoto_bandpassparset('a&p'))
                set_weights_h5_to_one(parmdb)
                if os.path.isfile(parmdb.replace('merged_', 'bandpass_')):
                    # remove existing bandpass file
                    os.system('rm -f ' + parmdb.replace('merged_', 'bandpass_'))
                os.system('mv ' + parmdb + ' ' + parmdb.replace('merged_', 'bandpass_'))
                if not args['keepmodelcolumns']: remove_model_columns(mslist)
                return

        # update uvmin if allowed/requested
        update_uvmin(fitsmask, longbaseline, LBA)

        # update fitsmake if allowed/requested
        fitsmask, fitsmask_list, imagename = update_fitsmask(fitsmask, maskthreshold_selfcalcycle, i, args, mslist, telescope, longbaseline)

        # update to multiscale cleaning if large island is present
        args['multiscale'] = multiscale_trigger(fitsmask)

        # update channelsout
        args['channelsout'] = update_channelsout(i, mslist)

        # update fitspectralpol
        args['fitspectralpol'] = update_fitspectralpol()

        # Get additional diagnostics and/or early-stopping --> in particular useful for calibrator selection and automation
        if args['early_stopping'] and len(mslist)>1:
            logger.info("WARNING: --early-stopping not yet developed for multiple input MeasurementSets.\nSkipping early-stopping evaluation.")
        elif args['early_stopping'] and early_stopping(station='international' if longbaseline else 'alldutch', cycle=i):
            break

    # Write config file to merged h5parms
    h5s = glob.glob('best_*solutions.h5') if args['early_stopping'] else glob.glob("merged_*.h5")
    for h5 in h5s:
        # Write the user-specified configuration file to h5parm and otherwise all input parameters if config file not specified
        add_config_to_h5(h5, args['configpath']) if args['configpath'] is not None else add_config_to_h5(h5, 'full_config.txt')
        add_version_to_h5(h5, facetselfcal_version)

    # remove sources outside central region after selfcal (to prepare for DDE solves)
    if args['remove_outside_center']:
        # make image after calibration so the calibration and images match
        # normally we would finish with calibration and not have the subsequent image, make this i+1 image here
        makeimage(mslistim, args['imagename'] + str(i + 1).zfill(3),
                  args['pixelscale'], args['imsize'],
                  args['channelsout'], args['niter'], args['robust'],
                  multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask,
                  uvminim=args['uvminim'], predict=False,
                  fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                  restoringbeam=restoringbeam, automask=automask,
                  removenegativecc=args['removenegativefrommodel'],
                  paralleldeconvolution=args['paralleldeconvolution'],
                  parallelgridding=args['parallelgridding'],
                  h5list=wsclean_h5list,
                  facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                  disable_primarybeam_image=args['disable_primary_beam'],
                  disable_primarybeam_predict=args['disable_primary_beam'],
                  fulljones_h5_facetbeam=not args['single_dual_speedup'])

        remove_outside_box(mslist, args['imagename'] + str(i + 1).zfill(3), args['pixelscale'],
                           args['imsize'], args['channelsout'], single_dual_speedup=args['single_dual_speedup'],
                           dysco=args['dysco'],
                           userbox=args['remove_outside_center_box'], idg=args['idg'],
                           h5list=wsclean_h5list, facetregionfile=facetregionfile,
                           disable_primary_beam=args['disable_primary_beam'], modelstoragemanager=args['modelstoragemanager'], parallelgridding=args['parallelgridding'], metadata_compression=args['metadata_compression'])

    # REMOVE MODEL_DATA type columns after selfcal
    if not args['keepmodelcolumns']: remove_model_columns(mslist)
    
    # ARCHIVE DATA AFTER SELFCAL if requested
    if not longbaseline and not args['noarchive']:
        if not LBA:
            if args['DDE']:
                mergedh5_i = glob.glob('merged_selfcalcycle' + str(i).zfill(3) + '*.h5')
                archive(mslist, outtarname, args['boxfile'], fitsmask, imagename, dysco=args['dysco'],
                        mergedh5_i=mergedh5_i, facetregionfile=facetregionfile, metadata_compression=args['metadata_compression'])
            else:
                archive(mslist, outtarname, args['boxfile'], fitsmask, imagename, dysco=args['dysco'], metadata_compression=args['metadata_compression'])
            cleanup(mslist)


if __name__ == "__main__":
    main()
