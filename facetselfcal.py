#!/usr/bin/env python

# normamps full jones, deal with solnorm on crosshands only? currently normaps not used for fulljones

# restart with updated soltype_list?
# ('selfcalcycle, soltypenumber', 10, 2)
# Traceback (most recent call last):
#   File "/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/facetselfcal.py", line 4955, in <module>
#     main()
#   File "/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/facetselfcal.py", line 4895, in main
#     docircular=args['docircular'])
#   File "/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/facetselfcal.py", line 3648, in calibrateandapplycal
#     print(selfcalcycle,soltypecycles_list[soltypenumber+1][msnumber])
# IndexError: list index out of range

# implement idea of phase detrending.
# decrease niter if multiscale is triggered, smart move?
# h5 linear to circular solution conversion
# do not predict sky second time in pertubation solve?
# to do: log command into the FITS header
# BLsmooth not for gain solves opttion
# BLsmooth constant smooth for gain solves
# only trigger HBA upper band selection for sources outside the FWHM?
# if noise goes up stop selfcal
# for phaseup option add back core stations in solution file via https://github.com/lmorabit/lofar-vlbi/blob/master/bin/gains_toCS_h5parm.py
# make Ateam plot


# example:
# python facetselfal.py -b box_18.reg --forwidefield --usewgridder --avgfreqstep=2 --avgtimestep=2 --smoothnessconstraint-list="[0.0,0.0,5.0]" --antennaconstraint-list="['core']" --solint-list=[1,20,120] --soltypecycles-list="[0,1,3]" --soltypelist="['tecandphase','tecandphase','scalarcomplexgain']" test.ms

# Standard library imports
import argparse
import ast
import fnmatch
import glob
import logging
import multiprocessing
import os
import os.path
import pickle
import re
import subprocess
import sys
import time

from itertools import product

# Third party imports
import astropy
import astropy.stats
import astropy.units as units
import bdsf
import casacore.tables as pt
import losoto
import losoto.lib_operations
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyregion
import tables

from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.wcs import WCS
from losoto import h5parm

logger = logging.getLogger(__name__)
logging.basicConfig(filename='selfcal.log',
                    format='%(levelname)s:%(asctime)s ---- %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger.setLevel(logging.DEBUG)
matplotlib.use('Agg')
# For NFS mounted disks
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# from astropy.utils.data import clear_download_cache
# clear_download_cache()


# this function does not work, for some reason cannot modify the source table
# def copy_over_sourcedirection_h5(h5ref, h5):
   # Href = tables.open_file(h5ref, mode='r')
   # ssdir = np.copy(Href.root.sol000.source[0]['dir'])
   # Href.close()
   # H = tables.open_file(h5, mode='a')
   # print(ssdir, H.root.sol000.source[0]['dir'])
   # H.root.sol000.source[0]['dir'] = np.copy(ssdir)
   # H.flush()
   # print(ssdir, H.root.sol000.source[0]['dir'])
   # H.close()
   # return


def run(command):
    ''' Execute a shell command through subprocess

    Args:
        command (str): the command to execute.
    Returns:
        reval (int): the returncode of the command.
    '''
    retval = subprocess.call(command, shell=True)
    if retval != 0:
        print('FAILED to run '+command+': return value is '+str(retval))
        raise Exception(command)
    return retval


def fix_bad_weightspectrum(mslist, clipvalue):
    ''' Sets bad values in WEIGHT_SPECTRUM that affect imaging and subsequent self-calibration to 0.0.

    Args:
        mslist (list): a list of Measurement Sets to iterate over and fix outlier values of.
        clipvalue (float): value above which WEIGHT_SPECTRUM will be set to 0.
    Returns:
        None
    '''
    for ms in mslist:
        print('Clipping WEIGHT_SPECTRUM manually', ms, clipvalue)
        t = pt.table(ms, readonly=False)
        ws = t.getcol('WEIGHT_SPECTRUM')
        idx = np.where(ws > clipvalue)
        ws[idx] = 0.0
        t.putcol('WEIGHT_SPECTRUM', ws)
        t.close()
    return


def format_solint(solint, ms):
    ''' Format the solution interval for DP3 calls.

    Args:
        solint (int or str): input solution interval.
        ms (str): measurement set to extract the integration time from.
    Returns:
        solintout (str): processed solution interval.
    '''
    if str(solint).isdigit():
        return str(solint)
    else:
        t = pt.table(ms, readonly=True, ack=False)
        time = np.unique(t.getcol('TIME'))
        tint = np.abs(time[1] - time[0])
        t.close()
        if 's' in solint:
            solintout = int(np.rint(float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'm' in solint:
            solintout = int(np.rint(60. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'h' in solint:
            solintout = int(np.rint(3600. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if solintout < 1:
            solintout = 1
        return str(solintout)


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

   delayaxis = fftfreq(upsample_factor*freq.size,
                       d=np.abs(freq[1]-freq[0])/float(upsample_factor))

   for ant_id, ant in enumerate(H.root.sol000.phase000.ant[:]):
      delay = 0.0*H.root.sol000.phase000.time[:]
      print('FFT delay finding for:', ant)
      for time_id, time in enumerate(H.root.sol000.phase000.time[:]):
         delay[time_id] = delayaxis[np.argmax(
             np.abs(fft(phasecomplex[time_id, :, ant_id, 0], n=upsample_factor*len(freq))))]
      plt.plot(timeaxis/3600., delay*1e9)
   plt.ylim(-2e-6*1e9, 2e-6*1e9)
   plt.ylabel('Delay [ns]')
   plt.xlabel('Time [hr]')
   # plt.title(ant)
   plt.show()
   H.close()
   return


def check_strlist_or_intlist(argin):
    ''' Check if the argument is a list of integers or a list of strings with correct formatting.

    Args:
        argin (str): input string to check.
    Returns:
        arg (list): properly formatted list extracted from the output.
    '''

    # check if input is a list and make proper list format
    arg = ast.literal_eval(argin)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))

    # check for integer list
    if all([isinstance(item, int) for item in arg]):
        if np.min(arg) < 1:
            raise argparse.ArgumentTypeError("solint_list cannot contain values smaller than 1")
        else:
            return arg
    # so not an integer list, so now check for string list
    if all([isinstance(item, str) for item in arg]):
        # check if string contains numbers
        for item2 in arg:
            # print(item2)
            if not any([ch.isdigit() for ch in item2]):
                raise argparse.ArgumentTypeError("solint_list needs to contain some number characters, not only units")
            # check in the number in there is smaller than 1
            # print(re.findall(r'[+-]?\d+(?:\.\d+)?',item2)[0])
            if float(re.findall(r'[+-]?\d+(?:\.\d+)?', item2)[0]) <= 0.0:
                raise argparse.ArgumentTypeError("numbers in solint_list cannot be smaller than zero")
            # check if string contains proper time formatting
            if ('hr' in item2) or ('min' in item2) or ('sec' in item2) or ('h' in item2) or ('m' in item2) or ('s' in item2) or ('hour' in item2) or ('minute' in item2) or ('second' in item2):
                pass
            else:
                raise argparse.ArgumentTypeError("solint_list needs to have proper time formatting (h(r), m(in), s(ec))")
        return arg
    else:
        raise argparse.ArgumentTypeError("solint_list must be a list of positive integers or a list of properly formatted strings")


def compute_distance_to_pointingcenter(msname, HBAorLBA='HBA'):
    ''' Compute distance to the pointing center. This is mainly useful for international baseline observation to check of the delay calibrator is not too far away.

    Args:
        msname (str): path to the measurement set to check.
        HBAorLBA (str): whether the data is HBA or LBA data. Can be 'HBA' or 'LBA'.
    Returns:
        None
    '''
    if HBAorLBA == 'HBA':
        warn_distance = 1.25
    if HBAorLBA == 'LBA':
        warn_distance = 3.0

    field_table = pt.table(msname + '::FIELD')
    direction = field_table.getcol('PHASE_DIR').squeeze()
    ref_direction = field_table.getcol('REFERENCE_DIR').squeeze()
    field_table.close()
    c1 = SkyCoord(direction[0] * units.radian, direction[1] * units.radian, frame='icrs')
    c2 = SkyCoord(ref_direction[0] * units.radian, ref_direction[1] * units.radian, frame='icrs')
    seperation = c1.separation(c2).to(units.deg)
    print('Distance to pointing center', seperation)
    logger.info('Distance to pointing center:' + str(seperation))
    if seperation.value > warn_distance:
        print('Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
        logger.warning('Warning: you are trying to selfcal a source far from the pointing, this is probably going to produce bad results')
    return


def remove_flagged_data_startend(mslist):
    ''' Trim flagged data at the start and end of the observation.

    Args:
        mslist (list): list of measurement sets to iterate over.
    Returns:
        mslistout (list): list of measurement sets with flagged data trimmed.
    '''

    taql = 'taql'
    mslistout = []

    for ms in mslist:
        t = pt.table(ms, readonly=True, ack=False)
        alltimes = t.getcol('TIME')
        alltimes = np.unique(alltimes)

        newt = pt.taql('select TIME from $t where FLAG[0,0]=False')
        time = newt.getcol('TIME')
        time = np.unique(time)

        print('There are', len(alltimes), 'times')
        print('There are', len(time), 'unique unflagged times')

        print('First unflagged time', np.min(time))
        print('Last unflagged time', np.max(time))

        goodstartid = np.where(alltimes == np.min(time))[0][0]
        goodendid = np.where(alltimes == np.max(time))[0][0] + 1

        print(goodstartid, goodendid)
        t.close()

        if (goodstartid != 0) or (goodendid != len(alltimes)):
            msout = ms + '.cut'
            if os.path.isdir(msout):
                os.system('rm -rf ' + msout)

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


def check_code_is_uptodate():
    ''' Check if the code is at its latest version.
    
    From https://jckantor.github.io/cbe61622/A.02-Downloading_Python_source_files_from_github.html
    '''
    import filecmp
    url = "https://raw.githubusercontent.com/jurjen93/lofar_helpers/master/h5_merger.py"

    try:
        result = subprocess.run(["wget", "--no-cache", "--backups=1", url, "--output-document=tmpfile"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(result.stderr.decode("utf-8"))
    except:  # for Python versions 3.5 and lower
        os.system("wget --no-cache --backups=1 " + url + " --output-document=tmpfile")

    if not filecmp.cmp('h5_merger.py', 'tmpfile'):
        print('Warning, you are using an old version of h5_merger.py')
        print('Download the latest version from https://github.com/jurjen93/lofar_helpers')
        logger.warning('Using an old h5_merger.py version, download the latest one from https://github.com/jurjen93/lofar_helpers')
        time.sleep(1)

    with open('h5_merger.py') as f:
        if 'propagate_flags' not in f.read():
            print("Update h5_merger, this version misses the propagate_flags option")
            sys.exit()
    return


def force_close(h5):
    ''' Close indivdual HDF5 file by force.

    Args:
        h5 (str): name of the h5parm to close.
    Returns:
        None
    '''
    h5s = list(tables.file._open_files._handlers)
    for h in h5s:
        if h.filename == h5:
            logger.warning('force_close: Closed --> ' + h5 + '\n')
            print('Forced (!) closing', h5)
            h.close()
            return
    # sys.stderr.write(h5 + ' not found\n')
    return


def create_mergeparmdbname(mslist, selfcalcycle):
    ''' Merges the h5parms for a given list of measurement sets and selfcal cycle.

    Args:
        mslist (list): list of measurement sets to iterate over.
        selfcalcycle (int): the selfcal cycle for which to merge h5parms.
    Returns:
        parmdblist (list): list of names of the merged h5parms.
    '''
    parmdblist = mslist[:]
    for ms_id, ms in enumerate(mslist):
        parmdblist[ms_id] = 'merged_selfcalcyle' + str(selfcalcycle).zfill(3) + '_' + ms + '.avg.h5'
    print('Created parmdblist', parmdblist)
    return parmdblist


def preapplydelay(H5filelist, mslist, applydelaytype, dysco=True):
    ''' Pre-apply a given list of h5parms to a measurement set, specifically intended for post-delay calibration sources.

    Args:
        H5filelist (list): list of h5parms to apply.
        mslist (list): list of measurement set to apply corrections to.
        applydelaytype (str): 'linear' or 'circular' to indicate the polarisation type of the solutions.
        dysco (bool): dysco compress the circular data column or not.
    Returns:
        None
    '''
    for ms in mslist:
        parmdb = time_match_mstoH5(H5filelist, ms)
        # from LINEAR to CIRCULAR
        if applydelaytype == 'circular':
            scriptn = 'python lin2circ.py'
            cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --outcol=DATA_CIRC'
            if not dysco:
                cmdlin2circ += ' --nodysco'
            run(cmdlin2circ)
            # APPLY solutions
            applycal(ms, parmdb, msincol='DATA_CIRC', msoutcol='CORRECTED_DATA', dysco=dysco)
        else:
            applycal(ms, parmdb, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=dysco)
        # from CIRCULAR to LINEAR
        if applydelaytype == 'circular':
            cmdlin2circ = scriptn + ' -i ' + ms + ' --column=CORRECTED_DATA --lincol=DATA --back'
            if not dysco:
                cmdlin2circ += ' --nodysco'
            run(cmdlin2circ)
        else:
            run("taql 'update " + ms + " set DATA=CORRECTED_DATA'")
    return


def preapply(H5filelist, mslist, updateDATA=True, dysco=True):
    ''' Pre-apply a given set of corrections to a list of measurement sets.

    Args:
        H5filelist (list): list of h5parms to apply.
        mslist (list): list of measurement set to apply corrections to.
        updateDATA (bool): overwrite DATA with CORRECTED_DATA after solutions have been applied.
        dysco (bool): dysco compress the CORRECTED_DATA column or not.
    Returns:
        None
    '''
    for ms in mslist:
        parmdb = time_match_mstoH5(H5filelist, ms)
        applycal(ms, parmdb, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=dysco)
        if updateDATA:
            run("taql 'update " + ms + " set DATA=CORRECTED_DATA'")
    return


def time_match_mstoH5(H5filelist, ms):
    ''' Find the h5parms, from a given list, that overlap in time with the specified Measurement Set.

    Args:
        H5filelist (list): list of h5parms to apply.
        ms (str): Measurement Set to match h5parms to.
    Returns:
        H5filematch (list): list of h5parms matching the measurement set.
    '''
    t = pt.table(ms)
    timesms = np.unique(t.getcol('TIME'))
    t.close()
    H5filematch = None

    for H5file in H5filelist:
        H = tables.open_file(H5file, mode='r')    
        try:
            times = H.root.sol000.amplitude000.time[:]
        except:
            pass
        try:
            times = H.root.sol000.rotation000.time[:]
        except:
            pass 
        try:
            times = H.root.sol000.phase000.time[:]
        except:
            pass      
        try:
            times = H.root.sol000.tec000.time[:]
        except:
            pass
        if np.median(times) >= np.min(timesms) and np.median(times) <= np.max(timesms):
            print(H5file, 'overlaps in time with', ms)
            H5filematch = H5file
        H.close()

    if H5filematch is None:
        print('Cannot find matching H5file and ms')
        sys.exit()

    return H5filematch


def logbasicinfo(args, fitsmask, mslist, version, inputsysargs):
    ''' Prints basic information to the screen.

    Args:
        args (iterable): list of input arguments.
        fitsmask (str): name of the user-provided FITS mask.
        mslist (list): list of input measurement sets.
    '''
    logger.info(' '.join(map(str,inputsysargs)))

    logger.info('Version:                   ' + str(version))
    logger.info('Imsize:                    ' + str(args['imsize']))
    logger.info('Pixelscale:                ' + str(args['pixelscale']))
    logger.info('Niter:                     ' + str(args['niter']))
    logger.info('Uvmin:                     ' + str(args['uvmin']  ))
    logger.info('Multiscale:                ' + str(args['multiscale']))
    logger.info('No beam correction:        ' + str(args['no_beamcor']))
    logger.info('IDG:                       ' + str(args['idg']))
    logger.info('Widefield:                 ' + str(args['forwidefield']))
    logger.info('Flagslowamprms:            ' + str(args['flagslowamprms']))
    logger.info('flagslowphaserms:          ' + str(args['flagslowphaserms']))
    logger.info('Do linear:                 ' + str(args['dolinear']))
    logger.info('Do circular:               ' + str(args['docircular']))
    if args['boxfile'] is not None:
        logger.info('Bobxfile:                  ' + args['boxfile'])
    logger.info('Mslist:                    ' + ' '.join(map(str,mslist)))
    logger.info('User specified clean mask: ' + str(fitsmask))
    logger.info('Threshold for MakeMask:    ' + str(args['maskthreshold']))
    logger.info('Briggs robust:             ' + str(args['robust']))
    return


def max_area_of_island(grid):
    ''' Calculate the area of an island.

    Args:
        grid (ndarray): input image.
    Returns:
        None
    '''
    rlen, clen = len(grid), len(grid[0])

    def neighbors(r, c):
        ''' Generate the neighbor coordinates of the given row and column that are within the bounds of the grid.

        Args:
            r (int): row coordinate.
            c (int): column coordinate.
        '''
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (0 <= r + dr < rlen) and (0 <= c + dc < clen):
                yield r + dr, c + dc

    visited = [[False] * clen for _ in range(rlen)]

    def island_size(r, c):
        ''' Find the area of the land connected to the given coordinate.

        Return 0 if the coordinate is water or if it has already been explored in a previous call to island_size().

        Args:
            r (int): row coordinate.
            c (int): column coordinate.
        '''
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
    ''' Find the largest island in a given FITS mask.
  
    Args:
        fitsmask (str): path to the FITS file.
    Returns:
        max_area (float): area of the largest island.
    '''
    hdulist = fits.open(fitsmask)
    data = hdulist[0].data
    max_area = max_area_of_island(data[0, 0, :, :])
    hdulist.close()
    return max_area


def create_phase_slope(inmslist, incol='DATA', outcol='DATA_PHASE_SLOPE', ampnorm=False, dysco=True):
    ''' Creates a new column to solve for a phase slope from.

    Args:
        inmslist (list): list of input measurement sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        ampnorm (bool): If True, only takes phases from the input visibilities and sets their amplitude to 1.
        dysco (bool): dysco compress the output column.
    Returns:
        None
    '''
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        t = pt.table(ms, readonly=False, ack=True)
        if outcol not in t.colnames():
            print('Adding', outcol, 'to', ms)
            desc = t.getcoldesc(incol)
            newdesc = pt.makecoldesc(outcol, desc)
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
                dataslope[:, ff, 0] = np.copy(np.exp(1j * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                dataslope[:, ff, 3] = np.copy(np.exp(1j * (np.angle(data[:, ff, 3]) -np.angle(data[:, ff + 1, 3]))))
            else:
                dataslope[:, ff, 0] = np.copy(np.abs(data[:, ff, 0]) * np.exp(1j * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                dataslope[:, ff, 3] = np.copy(np.abs(data[:, ff, 3]) * np.exp(1j * (np.angle(data[:, ff, 3]) - np.angle(data[:, ff + 1, 3]))))

        # last freq set to second to last freq because difference reduces length of freq axis with one
        dataslope[:, -1, :] = np.copy(dataslope[:, -2, :])
        t.putcol(outcol, dataslope)
        t.close()
        # print( np.nanmedian(np.abs(data)))
        # print( np.nanmedian(np.abs(dataslope)))
        del data, dataslope
    return


def create_phasediff_column(inmslist, incol='DATA', outcol='DATA_CIRCULAR_PHASEDIFF', dysco=True):
    ''' Creates a new column for the phase difference solve.

    Args:
        inmslist (list): list of input Measurement Sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        dysco (bool): dysco compress the output column.
    '''
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        t = pt.table(ms, readonly=False, ack=True)
        if outcol not in t.colnames():
            print('Adding', outcol, 'to', ms)
            desc = t.getcoldesc(incol)
            newdesc = pt.makecoldesc(outcol, desc)
            newdmi = t.getdminfo(incol)
            if dysco:
                newdmi['NAME'] = 'Dysco' + outcol
            else:
                newdmi['NAME'] = outcol
            t.addcols(newdesc, newdmi)


        data = t.getcol(incol)
        phasediff =  np.copy(np.angle(data[:, :, 0]) - np.angle(data[:, :, 3]))  #RR - LL
        data[:, :, 0] = 0.5 * np.exp(1j * phasediff)  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        data[:, :, 3] = data[:, :, 0]
        t.putcol(outcol, data)
        t.close()
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
    ''' Creates a new column containging visibilities with their original phase, but unity amplitude.

    Args:
        inmslist (list): list of input Measurement Sets.
        incol (str): name of the input column to copy (meta)data from.
        outcol (str): name of the output column that will be created.
        dysco (bool): dysco compress the output column.
    '''
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        t = pt.table(ms, readonly=False, ack=True)
        if outcol not in t.colnames():
            print('Adding', outcol, 'to', ms)
            desc = t.getcoldesc(incol)
            newdesc = pt.makecoldesc(outcol, desc)
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
        t.close()
        del data
    return


def create_MODEL_DATA_PDIFF(inmslist):
    ''' Creates the MODEL_DATA_PDIFF column.

    Args:
      inmslist (list): list of input Measurement Sets.
    '''
    if not isinstance(inmslist, list):
        inmslist = [inmslist] 
    for ms in inmslist:
        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_PDIFF steps=[]')
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,0]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,3]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,1]=(0+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,2]=(0+0i)'")


def fulljonesparmdb(h5):
    ''' Checks if a given h5parm has a fulljones solution table as sol000.

    Args:
        h5 (str): path to the h5parm.
    Returns:
        fulljones (bool): whether the sol000 contains fulljones solutions.
    '''
    H=tables.open_file(h5) 
    try:
        phase = H.root.sol000.phase000.val[:]
        amplitude = H.root.sol000.amplitude000.val[:]
        if phase.shape[-1] == 4 and amplitude.shape[-1] == 4:
            fulljones = True
        else:
            fulljones = False
    except:
        fulljones = False
    H.close()
    return fulljones


def reset_gains_noncore(h5parm, keepanntennastr='CS'):
    ''' Resets the gain of non-CS stations to unity amplitude and zero phase.
    
    Args:
        h5parm (str): path to the H5parm to reset gains of.
        keepantennastr (str): string containing antennas to keep.
    Returns:
      None
    '''
    fulljones = fulljonesparmdb(h5parm)  # True/False
    hasphase = True
    hasamps  = True
    hasrotatation = True
    hastec = True

    H=tables.open_file(h5parm, mode='a')
    # Figure out if we have phase and/or amplitude solutions.
    try:
        antennas = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
    except: 
        hasamps = False
    try:
        antennas = H.root.sol000.phase000.ant[:]
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
    except:
        hasphase = False
    try:
        antennas = H.root.sol000.tec000.ant[:]
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
    except:
        hastec = False
    try:
        antennas = H.root.sol000.rotation000.ant[:]
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
    except:
        hasrotatation = False     

    if hasphase:
        phase = H.root.sol000.phase000.val[:]
    if hasamps:  
        amp = H.root.sol000.amplitude000.val[:]
    if hastec:
        tec = H.root.sol000.tec000.val[:]
    if hasrotatation:
        rotation = H.root.sol000.rotation000.val[:]

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
                tec[antennaid,...] = 0.0
              if antennaxis == 1:
                tec[:,antennaid,...] = 0.0
              if antennaxis == 2:
                tec[:,:,antennaid,...] = 0.0
              if antennaxis == 3:
                tec[:,:,:,antennaid,...] = 0.0  
              if antennaxis == 4:
                tec[:,:,:,:,antennaid,...] = 0.0 
                        
            if hasrotatation:
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
    
    # fill values back in
    if hasphase:
        H.root.sol000.phase000.val[:] = np.copy(phase)
    if hasamps:  
        H.root.sol000.amplitude000.val[:] = np.copy(amp)
    if hastec:
        H.root.sol000.tec000.val[:] = np.copy(tec) 
    if hasrotatation:
        H.root.sol000.rotation000.val[:] = np.copy(rotatation)       

    H.flush()
    H.close()
    return

# reset_gains_noncore('merged_selfcalcyle11_testquick260.ms.avg.h5')
# sys.exit()

def phaseup(msinlist,datacolumn='DATA',superstation='core', start=0, dysco=True):
    ''' Phase up stations into a superstation.

    Args:
        msinlist (list): list of input Measurement Sets to iterate over.
        datacolumn (str): the input data column to phase up data from.
        superstation (str): stations to phase up. Can be 'core' or 'superterp'.
        start (int): selfcal cylce that is being started from. Phaseup will only occur if start == 0.
        dysco (bool): dysco compress the output dataset.
    Returns:
        msoutlist (list): list of output Measurement Sets.
    '''
    msoutlist = []
    for ms in msinlist:
      msout=ms + '.phaseup'
      msoutlist.append(msout)

      cmd = "DP3 msin=" + ms + " steps=[add,filter] msout.writefullresflag=False "
      cmd += "msout=" + msout + " msin.datacolumn=" + datacolumn + " "
      cmd += "filter.type=filter filter.remove=True "
      if dysco:
        cmd += "msout.storagemanager=dysco "
      cmd += "add.type=stationadder "
      if superstation == 'core':
        cmd += "add.stations={ST001:'CS*'} filter.baseline='!CS*&&*' "
      if superstation == 'superterp':
        cmd += "add.stations={ST001:'CS00[2-7]*'} filter.baseline='!CS00[2-7]*&&*' "  

      if start == 0: # only phaseup if start selfcal from cycle 0, so skip for a restart
          if os.path.isdir(msout):
              os.system('rm -rf ' + msout)
          print(cmd)
          run(cmd)
    return msoutlist

def findfreqavg(ms, imsize, bwsmearlimit=1.0):
    
  t = pt.table(ms + '/SPECTRAL_WINDOW',ack=False)
  bwsmear = bandwidthsmearing(np.median(t.getcol('CHAN_WIDTH')), \
            np.min(t.getcol('CHAN_FREQ')[0]), np.float(imsize), verbose=False)
  nfreq = len(t.getcol('CHAN_FREQ')[0])
  t.close()
  avgfactor = 0
  
  for count in range(2,21): # try average values between 2 to 20
     if bwsmear  < (bwsmearlimit/np.float(count)): # factor X avg
        if nfreq % count == 0:
           avgfactor = count
  return avgfactor

def compute_markersize(H5file):
    ntimes = ntimesH5(H5file)
    markersize = 2
    if ntimes < 450:
      markersize = 4
    if ntimes < 100:
      markersize = 10
    if ntimes < 50:
      markersize = 15      
    return markersize

def ntimesH5(H5file):
   # function to return number of timeslots in H5 solution
   H=tables.open_file(H5file, mode='r')
   try:
     times= H.root.sol000.amplitude000.time[:]
   except: # apparently no slow amps available
     try:
       times= H.root.sol000.phase000.time[:]
     except:
       try:  
         times= H.root.sol000.tec000.time[:]    
       except:  
         try:
           times= H.root.sol000.rotationmeasure000.time[:]    
         except:
           try:
             times= H.root.sol000.rotation000.time[:]
           except:    
             print('No amplitude000,phase000, tec000, rotation000, or rotationmeasure000 solutions found')  
             sys.exit()
   H.close()
   return len(times)

def create_backup_flag_col(ms, flagcolname='FLAG_BACKUP'):
    cname = 'FLAG'
    flags = []
    t = pt.table(ms, readonly=False, ack=True)
    if flagcolname not in t.colnames():
      flags = t.getcol('FLAG')  
      print('Adding flagging column',flagcolname,'to',ms)            
      desc = t.getcoldesc(cname)
      newdesc = pt.makecoldesc(flagcolname, desc)
      newdmi = t.getdminfo(cname)
      newdmi['NAME'] = flagcolname
      t.addcols(newdesc, newdmi)  
      t.putcol(flagcolname, flags)     

    t.close()
    del flags
    return    
    

def checklongbaseline(ms):
    t   = pt.table(ms + '/ANTENNA',ack=False)
    antennasms = list(t.getcol('NAME'))
    t.close()
    substr = 'DE' # to check if a German station is present, if yes assume this is long baseline data
    haslongbaselines =  any(substr in mystring for mystring in antennasms)
    print('Contains long baselines?', haslongbaselines)
    return haslongbaselines

def average(mslist, freqstep, timestep=None, start=0, msinnchan=None, phaseshiftbox=None, msinntimes=None, makecopy=False, delaycal=False, timeresolution='32', freqresolution='195.3125kHz', dysco=True):
    # sanity check
    if len(mslist) != len(freqstep):
      print('Hmm, made a mistake with freqstep?')
      sys.exit()
    
    outmslist = []
    for ms_id, ms in enumerate(mslist):
      if (freqstep[ms_id] > 0) or (timestep != None) or (msinnchan != None) or \
          (phaseshiftbox != None) or (msinntimes != None): # if this is True then average
        if makecopy:
          msout = ms + '.copy'
        else:
          msout = ms + '.avg'  
        cmd = 'DP3 msin=' + ms + ' av.type=averager '
        cmd += 'msout='+ msout + ' msin.weightcolumn=WEIGHT_SPECTRUM msout.writefullresflag=False '
        if dysco:
          cmd += 'msout.storagemanager=dysco '    
        if phaseshiftbox != None:
          cmd += ' steps=[shift,av] '
          cmd += ' shift.type=phaseshifter '
          cmd += ' shift.phasecenter=\['+getregionboxcenter(phaseshiftbox)+'\] '
        else:    
          cmd +=' steps=[av] ' 
        
        if freqstep[ms_id] != None:
          cmd +='av.freqstep=' + str(freqstep[ms_id]) + ' '
        if timestep != None:  
          cmd +='av.timestep=' + str(timestep) + ' '
        if msinnchan != None:
           cmd +='msin.nchan=' + str(msinnchan) + ' ' 
        if msinntimes != None:
           cmd +='msin.ntimes=' + str(msinntimes) + ' ' 
        if start == 0:
          print('Average with default WEIGHT_SPECTRUM:', cmd)
          if os.path.isdir(msout):
            os.system('rm -rf ' + msout)
          run(cmd)

        msouttmp = ms + '.avgtmp'  
        cmd = 'DP3 msin=' + ms + ' steps=[av] av.type=averager '
        if dysco:
            cmd+= ' msout.storagemanager=dysco '
        cmd+= 'msout='+ msouttmp + ' msin.weightcolumn=WEIGHT_SPECTRUM_SOLVE msout.writefullresflag=False '
        if freqstep[ms_id] != None:
          cmd+='av.freqstep=' + str(freqstep[ms_id]) + ' '
        if timestep != None:  
          cmd+='av.timestep=' + str(timestep) + ' '
        if msinnchan != None:
           cmd+='msin.nchan=' + str(msinnchan) + ' '
        if msinntimes != None:
           cmd +='msin.ntimes=' + str(msinntimes) + ' ' 
           
        if start == 0:
          t = pt.table(ms)
          if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames(): # check if present otherwise this is not needed
            t.close()   
            print('Average with default WEIGHT_SPECTRUM_SOLVE:', cmd)
            if os.path.isdir(msouttmp):
              os.system('rm -rf ' + msouttmp)
            run(cmd)
          
            # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
            t  = pt.table(msout, readonly=False)
            print('Adding WEIGHT_SPECTRUM_SOLVE')
            desc = t.getcoldesc('WEIGHT_SPECTRUM')
            desc['name']='WEIGHT_SPECTRUM_SOLVE'
            t.addcols(desc)

            t2 = pt.table(msouttmp, readonly=True)
            imweights = t2.getcol('WEIGHT_SPECTRUM')
            t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

            # Fill WEIGHT_SPECTRUM with WEIGHT_SPECTRUM from second ms
            t2.close()
            t.close() 

            # clean up
            os.system('rm -rf ' + msouttmp)
          
          
        outmslist.append(msout)
      else:
        outmslist.append(ms)  # so no averaging happened
    
    return outmslist


def tecandphaseplotter(h5, ms, outplotname='plot.png'):
    if not os.path.isdir('plotlosoto%s'  % ms): # needed because if this is the first plot this directory does not yet exist
      os.system('mkdir plotlosoto%s'  % ms)
    cmd = 'python plot_tecandphase.py  '
    cmd += '--H5file=' + h5 + ' --outfile=plotlosoto%s/%s_nolosoto.png' % (ms,outplotname)
    print(cmd)
    run(cmd)
    return

def runaoflagger(mslist):
    for ms in mslist:
       cmd = 'aoflagger ' + ms
       run(cmd)
    return


def applycal(ms, inparmdblist, msincol='DATA',msoutcol='CORRECTED_DATA', msout='.', dysco=True):

    # to allow both a list or a single file (string)
    if not isinstance(inparmdblist,list):
     inparmdblist = [inparmdblist]    
    
    cmd = 'DP3 numthreads='+ str(multiprocessing.cpu_count()) + ' msin=' + ms
    cmd += ' msout=' + msout + ' '
    cmd += 'msin.datacolumn=' + msincol + ' '
    if msout == '.':
      cmd += 'msout.datacolumn=' + msoutcol + ' '
    if dysco:
      cmd += 'msout.storagemanager=dysco '
    count = 0
    for parmdb in inparmdblist:
      if fulljonesparmdb(parmdb):
        cmd += 'ac' + str(count) +'.parmdb='+parmdb + ' '
        cmd += 'ac' + str(count) +'.type=applycal '  
        cmd += 'ac' + str(count) +'.correction=fulljones '
        cmd += 'ac' + str(count) +'.soltab=[amplitude000,phase000] '  
        count = count + 1
      else:  
        H=tables.open_file(parmdb) 
        try:
          phase = H.root.sol000.phase000.val[:]
          cmd += 'ac' + str(count) +'.parmdb='+parmdb + ' '
          cmd += 'ac' + str(count) +'.type=applycal '  
          cmd += 'ac' + str(count) +'.correction=phase000 '
          count = count + 1
        except:
          pass  
        
        try:
          phase = H.root.sol000.tec000.val[:]
          cmd += 'ac' + str(count) +'.parmdb='+parmdb + ' '
          cmd += 'ac' + str(count) +'.type=applycal '  
          cmd += 'ac' + str(count) +'.correction=tec000 '
          count = count + 1        
        except:
          pass  
        
        try:
          phase = H.root.sol000.rotation000.val[:]
          cmd += 'ac' + str(count) +'.parmdb='+parmdb + ' '
          cmd += 'ac' + str(count) +'.type=applycal '  
          cmd += 'ac' + str(count) +'.correction=rotation000 '
          count = count + 1        
        except:
          pass  
        
        try:
          phase = H.root.sol000.amplitude000.val[:]
          cmd += 'ac' + str(count) +'.parmdb='+parmdb + ' '
          cmd += 'ac' + str(count) +'.type=applycal '  
          cmd += 'ac' + str(count) +'.correction=amplitude000 '
          count = count + 1        
        except:
          pass  
      
        H.close()
    
    if count < 1:
        print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        sys.exit(1)
    # build the steps command    
    cmd += 'steps=['
    for i in range(count):
      cmd += 'ac'+ str(i)
      if i < count-1: # to avoid last comma in the steps list
        cmd += ','
    cmd += ']'

    print('DP3 applycal:', cmd)
    run(cmd) 
    return


def inputchecker(args):

  for ms_id, ms in enumerate(args['ms']):
    if ms.find('/') != -1:
      print('All ms need to be local, no "/" are allowed in ms name')
      sys.exit(1)

  if args['iontimefactor'] <= 0.0:
    print('BLsmooth iontimefactor needs to be positive')
    sys.exit(1)
  if args['iontimefactor'] > 10.0:
    print('BLsmooth iontimefactor is way too high')
    sys.exit(1)

  if args['ionfreqfactor'] <= 0.0:
    print('BLsmooth tecfactor needs to be positive')
    sys.exit(1)
  if args['ionfreqfactor'] > 10000.0:
    print('BLsmooth tecfactor is way too high')
    sys.exit(1)

  if not os.path.isfile('lib_multiproc.py'):
    print('Cannot find lib_multiproc.py, file does not exist, use --helperscriptspath')
    sys.exit(1)
  if not os.path.isfile('h5_merger.py'):
    print('Cannot find h5_merger.py, file does not exist, use --helperscriptspath or --helperscriptspathh5merge')
    sys.exit(1)
  if not os.path.isfile('plot_tecandphase.py'):
    print('Cannot find plot_tecandphase.py, file does not exist, use --helperscriptspath')
    sys.exit(1)
  if not os.path.isfile('lin2circ.py'):
    print('Cannot find lin2circ.py, file does not exist, use --helperscriptspath')
    sys.exit(1)    
  if not os.path.isfile('BLsmooth.py'):
    print('Cannot find BLsmooth.py, file does not exist, use --helperscriptspath')
    sys.exit(1)
  if not os.path.isfile('polconv.py'):
    print('Cannot find polconv.py, file does not exist, use --helperscriptspath')
    sys.exit(1)    
  

  if args['phaseshiftbox'] != None:
    if not os.path.isfile(args['phaseshiftbox']):
      print('Cannot find:',args['phaseshiftbox'])
      sys.exit(1)
  
  if not args['no_beamcor'] and args['idg']:
    print('beamcor=True and IDG=True is not possible')
    sys.exit(1)
  
  for antennaconstraint in args['antennaconstraint_list']:
    if antennaconstraint not in ['superterp', 'coreandfirstremotes','core', 'remote',\
                                 'all', 'international', 'alldutch', 'core-remote','coreandallbutmostdistantremotes','alldutchbutnoST001'] \
                         and antennaconstraint != None:
      print('Invalid input, antennaconstraint can only be core, superterp, coreandfirstremotes, remote, alldutch, international, or all')
      sys.exit(1)

  for resetsols in args['resetsols_list']:
    if resetsols not in ['superterp', 'coreandfirstremotes','core', 'remote',\
                                 'all', 'international', 'alldutch', 'core-remote','coreandallbutmostdistantremotes', 'alldutchbutnoST001'] \
                         and resetsols != None:
      print('Invalid input, resetsols can only be core, superterp, coreandfirstremotes, remote, alldutch, international, or all')
      sys.exit(1)

  for soltype in args['soltype_list']:
    if soltype not in ['complexgain','scalarcomplexgain','scalaramplitude','amplitudeonly', 'phaseonly',\
                       'fulljones', 'rotation', 'rotation+diagonal','tec','tecandphase','scalarphase',\
                       'scalarphasediff','scalarphasediffFR' , 'phaseonly_phmin', 'rotation_phmin', 'tec_phmin',\
                       'tecandphase_phmin','scalarphase_phmin','scalarphase_slope','phaseonly_slope']:
      print('Invalid soltype input')
      sys.exit(1)    

  if args['boxfile'] != None:
    if not (os.path.isfile(args['boxfile'])):
      print('Cannot find boxfile, file does not exist')
      sys.exit(1)
      
  if args['fitsmask'] != None:
    if not (os.path.isfile(args['fitsmask'])):
      print('Cannot find fitsmask, file does not exist')
      sys.exit(1)      

  if args['skymodel'] != None:
    if not (os.path.isfile(args['skymodel'])) and not (os.path.isdir(args['skymodel'])):
      print('Cannot find skymodel, file does not exist')
      sys.exit(1)

  if args['docircular'] and args['dolinear']:
      print('Conflicting input, docircular and dolinear used')
      sys.exit(1)

  if which('DP3') == None:
    print('Cannot find DP3, forgot to source lofarinit.[c]sh?')
    sys.exit(1)

  # Check boxfile and imsize settings
  if args['boxfile'] == None and args['imsize'] == None:
    if not checklongbaseline(sorted(args['ms'])[0]):      
      print('Incomplete input detected, either boxfile or imsize is required')
      sys.exit(1)
    
  if args['boxfile'] != None and args['imsize'] != None:
    print('Wrong input detected, both boxfile and imsize are set')
    sys.exit(1)

  if args['imager'] not in ['DDFACET', 'WSCLEAN']: 
    print('Wrong input detected for option --imager, should be DDFACET or WSCLEAN')
    sys.exit(1)  

  if args['phaseupstations'] != None:
    if args['phaseupstations'] not in ['core', 'superterp']:    
      print('Wrong input detected for option --phaseupstations, should be core or superterp')
      sys.exit(1)  

  if args['soltypecycles_list'][0] != 0:
     print('Wrong input detected for option --soltypecycles-list should always start with 0') 
     sys.exit(1)

  if len(args['soltypecycles_list']) != len(args['soltype_list']): 
     print('Wrong input detected, length soltypecycles-list does not match that of soltype-list') 
     sys.exit(1)
 
  for soltype_id, soltype in enumerate(args['soltype_list']):
    wronginput = False
    if soltype in ['tecandphase', 'tec', 'tec_phmin', 'tecandphase_phmin']:    
      try: # in smoothnessconstraint_list is not filled by the user
        if args['smoothnessconstraint_list'][soltype_id] > 0.0:
          print('smoothnessconstraint should be 0.0 for a tec-like solve')
          wronginput = True
      except:
        pass    
      if wronginput:
       sys.exit(1)    
  
  for smoothnessconstraint in args['smoothnessconstraint_list']:
    if smoothnessconstraint < 0.0:
      print('Smoothnessconstraint must be equal or larger than 0.0')
      sys.exit(1)
  for smoothnessreffrequency in args['smoothnessreffrequency_list']:
    if smoothnessreffrequency < 0.0:
      print('Smoothnessreffrequency must be equal or larger than 0.0')
      sys.exit(1)
  
  if (args['skymodel'] != None) and (args['skymodelpointsource']) !=None:
    print('Wrong input, you cannot use a separate skymodel file and then also set skymodelpointsource')
    sys.exit(1)
  if (args['skymodelpointsource'] != None):
    if (args['skymodelpointsource'] <= 0.0):
      print('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
      sys.exit(1)
  if (args['msinnchan'] != None):
    if (args['msinnchan'] <= 0):
      print('Wrong input for msinnchan, must be larger than zero')
      sys.exit(1)
  if (args['msinntimes'] != None):
    if (args['msinntimes'] <= 1):
      print('Wrong input for msinntimes, must be larger than 1')
      sys.exit(1)


  if (args['skymodelpointsource'] != None) and (args['predictskywithbeam']):
    print('Combination of skymodelpointsource and predictskywithbeam not supported')
    print('Provide a skymodel file to predict the sky with the beam')
    sys.exit(1)

  if (args['wscleanskymodel'] != None) and (args['skymodelpointsource']) !=None:
    print('Wrong input, you cannot use a wscleanskymodel and then also set skymodelpointsource')
    sys.exit(1)

  if (args['wscleanskymodel'] != None) and (args['skymodel']) !=None:
    print('Wrong input, you cannot use a wscleanskymodel and then also set skymodel')
    sys.exit(1)

  if (args['wscleanskymodel'] != None) and (args['predictskywithbeam']):
    print('Combination of wscleanskymodel and predictskywithbeam not supported')
    print('Provide a skymodel component file to predict the sky with the beam')
    sys.exit(1)

  if (args['wscleanskymodel'] != None) and (args['imager'] == 'DDFACET'):
    print('Combination of wscleanskymodel and DDFACET as an imager is not supported')
    sys.exit(1)
  if (args['wscleanskymodel'] != None): 
    if len(glob.glob(args['wscleanskymodel'] + '-????-model.fits')) < 2:
      print('Not enough WSClean channel model images found')
      print(glob.glob(args['wscleanskymodel'] + '-????-model.fits'))
      sys.exit(1)
    if (args['wscleanskymodel'].find('/') != -1):
      print('wscleanskymodel contains a slash, not allowed, needs to be in pwd') 
      sys.exit(1)
    if (args['wscleanskymodel'].find('..') != -1):
      print('wscleanskymodel contains .., not allowed, needs to be in pwd')      
      sys.exit(1)  
  return
  

def get_uvwmax(ms):
    t = pt.table(ms)
    uvw = t.getcol('UVW')
    ssq = np.sqrt(np.sum(uvw**2, axis=1))
    print(uvw.shape)
    t.close()
    return np.max(ssq)    

def makeBBSmodelforTGSS(boxfile=None, fitsimage=None, pixelscale=None, imsize=None, ms=None):

    tgsspixsize = 6.2    
    if boxfile == None and imsize == None:
        print('Wring input detected, boxfile or imsize needs to be set')
        sys.exit()
    if boxfile != None:
       r = pyregion.open(boxfile)
       if len(r[:]) > 1:
          print('Composite region file, not allowed') 
          sys.exit()
       phasecenter = getregionboxcenter(boxfile)
       phasecenterc = phasecenter.replace('deg','')
       xs = np.ceil((r[0].coord_list[2])*3600./tgsspixsize)
       ys = np.ceil((r[0].coord_list[3])*3600./tgsspixsize)
    else:
       t2 = pt.table(ms + '::FIELD')
       phasedir = t2.getcol('PHASE_DIR').squeeze()
       t2.close()
       phasecenterc =  ('{:12.8f}'.format(180.*np.mod(phasedir[0], 2.*np.pi)/np.pi) + ',' + '{:12.8f}'.format(180.*phasedir[1]/np.pi)).replace(' ','')
       
       # phasecenterc = str() + ', ' + str()
       xs = np.ceil(imsize*pixelscale/tgsspixsize)
       ys = np.ceil(imsize*pixelscale/tgsspixsize)
    
    print('TGSS imsize:', xs)
    print('TGSS image center:', phasecenterc)
    logger.info('TGSS imsize:' + str(xs))
    logger.info('TGSS image center:' + str(phasecenterc))
    
    # sys.exit()
 
    if fitsimage == None:
        filename = SkyView.get_image_list(position=phasecenterc,survey='TGSS ADR1', pixels=np.int(xs), cache=False)
        print(filename)
        if os.path.isfile(filename[0].split('/')[-1]):
          os.system('rm -f ' + filename[0].split('/')[-1])
        time.sleep(10)
        os.system('wget ' + filename[0])
        filename = filename[0].split('/')[-1]
        print(filename)
    else:
        filename = fitsimage  
    
    img = bdsf.process_image(filename,mean_map='zero', rms_map=True, rms_box = (100,10), \
                             frequency=150e6, beam=(25./3600,25./3600,0.0) )
    img.write_catalog(format='bbs', bbs_patches='single', outfile='tgss.skymodel', clobber=True)
    # bbsmodel = 'bla.skymodel'
    del img
    return 'tgss.skymodel'

def getregionboxcenter(regionfile, standardbox=True):
    """
    Extract box center of a DS9 box region. 
    Input is regionfile Return DP3 compatible string for phasecenter shifting
    """
    r = pyregion.open(regionfile)
    
    if len(r[:]) > 1:
      print('Only one region can be specified, your file contains', len(r[:]))
      sys.exit() 
    
    if r[0].name != 'box':
      print('Only box region supported')
      sys.exit()
    
    ra  = r[0].coord_list[0]
    dec = r[0].coord_list[1]
    boxsizex = r[0].coord_list[2]
    boxsizey = r[0].coord_list[3]
    angle = r[0].coord_list[4]
    
    if standardbox:
      if boxsizex != boxsizey:
        print('Only a square box region supported, you have these sizes:', boxsizex, boxsizey)
        sys.exit()
      if np.abs(angle) > 1:
        print('Only normally oriented sqaure boxes are supported, your region is oriented under angle:', angle)
        sys.exit()   
    
    regioncenter =  ('{:12.8f}'.format(ra) + 'deg,' + '{:12.8f}'.format(dec) + 'deg').replace(' ', '')
    return regioncenter



def bandwidthsmearing(chanw, freq, imsize, verbose=True):

  R =  (chanw/freq)*(imsize/6.) # asume we have used 3 pixels per beam
  if verbose:
    print('R value for bandwidth smearing is:', R)
    logger.info('R value for bandwidth smearing is: ' + str(R))
    if R > 1.:
      print('Warning, try to increase your frequency resolution, or lower imsize, to reduce the R value below 1')
      logger.warning('Warning, try to increase your frequency resolution, or lower imsize, to reduce the R value below 1')
  
  return R

def number_freqchan_h5(h5parmin):
    '''
    Function to get the number of freqcencies in H5 solution file
    Input: H5 file
    Return: Number of freqcencies in the H5 file
    '''
    H=tables.open_file(h5parmin)
    
    try:
       freq = H.root.sol000.phase000.freq[:]
       # print('You solutions do not contain phase values')
    except:    
       pass
    
    try:
        freq = H.root.sol000.amplitude000.freq[:] # apparently we only have amplitudes
    except:
        pass
    
    try:
        freq = H.root.sol000.rotation000.freq[:] # apparently we only have rotatioon
    except:
        pass

    try:
        freq = H.root.sol000.tec000.freq[:] # apparently we only have rotatioon
    except:
        pass

    H.close()
    print('Number of frequency channels in this solutions file is:', len(freq))
    return len(freq)


def calculate_restoringbeam(mslist, LBA):
    
    if LBA: # so we have LBA
      restoringbeam = 15.
    else : # so we have HBA
      restoringbeam = 6.  
    
    return restoringbeam



def print_title(version):
    print("""
              __        ______    _______    ___      .______      
             |  |      /  __  \  |   ____|  /   \     |   _  \     
             |  |     |  |  |  | |  |__    /  ^  \    |  |_)  |    
             |  |     |  |  |  | |   __|  /  /_\  \   |      /     
             |  `----.|  `--'  | |  |    /  _____  \  |  |\  \----.
             |_______| \______/  |__|   /__/     \__\ | _| `._____|
                                                                   
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
                                                                              
    
                      Reinout van Weeren (2021, A&A, in press)

                              Starting.........
          """)

    print('\n\nVERSION: ' + version + '\n\n')
    logger.info('VERSION: ' + version)
    return

def makemslist(mslist):
    os.system('rm -rf mslist.txt')
    f=open('mslist.txt', 'w')
    for ms in mslist:
       f.write(str(ms)+'\n')
    f.close()
    return

def antennaconstraintstr(ctype, antennasms, HBAorLBA, useforresetsols=False):
    antennasms = list(antennasms)
    # print(antennasms)
    if ctype != 'superterp' and ctype != 'core' and ctype != 'coreandfirstremotes' and \
       ctype != 'remote' and ctype != 'alldutch' and ctype != 'all' and \
       ctype != 'international' and ctype != 'core-remote' and ctype != 'coreandallbutmostdistantremotes' and \
       ctype != 'alldutchbutnoST001' :
        print('Invalid input, ctype can only be "superterp" or "core"')
        sys.exit(1)
    if HBAorLBA == 'LBA':  
      if ctype == 'superterp':  
        antstr=['CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA','ST001']
      if ctype == 'core':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','ST001']
      if ctype == 'coreandfirstremotes':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','RS503LBA','RS305LBA','RS205LBA','RS306LBA', \
                'RS106LBA','ST001']
      if ctype == 'coreandallbutmostdistantremotes':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','RS503LBA','RS305LBA','RS205LBA','RS306LBA', \
                'RS106LBA','RS307LBA','RS406LBA','RS407LBA','ST001']
      if ctype == 'remote':
        antstr=['RS503LBA','RS305LBA','RS205LBA','RS306LBA', 'RS310LBA','RS406LBA','RS407LBA',\
                'RS106LBA','RS307LBA','RS208LBA','RS210LBA', 'RS409LBA','RS508LBA','RS509LBA']
      if ctype == 'alldutch':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','RS503LBA','RS305LBA','RS205LBA','RS306LBA', \
                'RS310LBA','RS406LBA','RS407LBA','RS106LBA','RS307LBA','RS208LBA','RS210LBA', \
                'RS409LBA','RS508LBA','RS509LBA', 'ST001']
      if ctype == 'alldutchbutnoST001':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','RS503LBA','RS305LBA','RS205LBA','RS306LBA', \
                'RS310LBA','RS406LBA','RS407LBA','RS106LBA','RS307LBA','RS208LBA','RS210LBA', \
                'RS409LBA','RS508LBA','RS509LBA']

      if ctype == 'all':
        antstr=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','RS503LBA','RS305LBA','RS205LBA','RS306LBA', \
                'RS310LBA','RS406LBA','RS407LBA','RS106LBA','RS307LBA','RS208LBA','RS210LBA', \
                'RS409LBA','RS508LBA','RS509LBA', \
                'DE601LBA','DE602LBA','DE603LBA','DE604LBA', 'DE605LBA','DE609LBA','FR606LBA', \
                'SE607LBA','UK608LBA','PL610LBA','PL611LBA', 'PL612LBA','IE613LBA','LV614LBA','ST001']          
      if ctype == 'international':
        antstr=['DE601LBA','DE602LBA','DE603LBA','DE604LBA', 'DE605LBA','DE609LBA','FR606LBA', \
                'SE607LBA','UK608LBA','PL610LBA','PL611LBA', 'PL612LBA','IE613LBA','LV614LBA']    
      if ctype == 'core-remote':
        antstr1=['CS001LBA','CS002LBA','CS003LBA','CS004LBA','CS005LBA','CS006LBA','CS007LBA', \
                'CS011LBA','CS013LBA','CS017LBA','CS021LBA','CS024LBA','CS026LBA','CS028LBA', \
                'CS030LBA','CS031LBA','CS032LBA','CS101LBA','CS103LBA','CS201LBA','CS301LBA', \
                'CS302LBA','CS401LBA','CS501LBA','ST001']
        antstr2=['RS503LBA','RS305LBA','RS205LBA','RS306LBA', 'RS310LBA','RS406LBA','RS407LBA',\
                'RS106LBA','RS307LBA','RS208LBA','RS210LBA', 'RS409LBA','RS508LBA','RS509LBA']
          

    if HBAorLBA == 'HBA':    
      if ctype == 'superterp': 
         antstr=['CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                 'CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1','ST001']
      if ctype == 'remote':
        antstr=['RS503HBA','RS305HBA','RS205HBA','RS306HBA', 'RS310HBA','RS406HBA','RS407HBA', \
                'RS106HBA','RS307HBA','RS208HBA','RS210HBA', 'RS409HBA','RS508HBA','RS509HBA']
      if ctype == 'core':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1','ST001']
      if ctype == 'coreandfirstremotes':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1','RS503HBA' ,'RS305HBA' ,'RS205HBA' ,'RS306HBA',  \
                'RS106HBA','ST001']
      if ctype == 'coreandallbutmostdistantremotes':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1','RS503HBA' ,'RS305HBA' ,'RS205HBA' ,'RS306HBA',  \
                'RS106HBA','RS307HBA','RS406HBA','RS407HBA','ST001']
      if ctype == 'alldutch':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1', \
                'RS503HBA','RS305HBA','RS205HBA','RS306HBA', 'RS310HBA','RS406HBA','RS407HBA', \
                'RS106HBA','RS307HBA','RS208HBA','RS210HBA', 'RS409HBA','RS508HBA','RS509HBA','ST001']
      if ctype == 'alldutchbutnoST001':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1', \
                'RS503HBA','RS305HBA','RS205HBA','RS306HBA', 'RS310HBA','RS406HBA','RS407HBA', \
                'RS106HBA','RS307HBA','RS208HBA','RS210HBA', 'RS409HBA','RS508HBA','RS509HBA']
      if ctype == 'all':
        antstr=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1', \
                'RS503HBA','RS305HBA','RS205HBA','RS306HBA', 'RS310HBA','RS406HBA','RS407HBA', \
                'RS106HBA','RS307HBA','RS208HBA','RS210HBA', 'RS409HBA','RS508HBA','RS509HBA', \
                'DE601HBA','DE602HBA','DE603HBA','DE604HBA', 'DE605HBA','DE609HBA','FR606HBA', \
                'SE607HBA','UK608HBA','PL610HBA','PL611HBA', 'PL612HBA','IE613HBA','LV614HBA','ST001']
      if ctype == 'international':
        antstr=['DE601HBA','DE602HBA','DE603HBA','DE604HBA', 'DE605HBA','DE609HBA','FR606HBA', \
                'SE607HBA','UK608HBA','PL610HBA','PL611HBA', 'PL612HBA','IE613HBA','LV614HBA']        
      if ctype == 'core-remote':
        antstr1=['CS001HBA0','CS002HBA0','CS003HBA0','CS004HBA0','CS005HBA0','CS006HBA0','CS007HBA0', \
                'CS011HBA0','CS013HBA0','CS017HBA0','CS021HBA0','CS024HBA0','CS026HBA0','CS028HBA0', \
                'CS030HBA0','CS031HBA0','CS032HBA0','CS101HBA0','CS103HBA0','CS201HBA0','CS301HBA0', \
                'CS302HBA0','CS401HBA0','CS501HBA0',\
                'CS001HBA1','CS002HBA1','CS003HBA1','CS004HBA1','CS005HBA1','CS006HBA1','CS007HBA1', \
                'CS011HBA1','CS013HBA1','CS017HBA1','CS021HBA1','CS024HBA1','CS026HBA1','CS028HBA1', \
                'CS030HBA1','CS031HBA1','CS032HBA1','CS101HBA1','CS103HBA1','CS201HBA1','CS301HBA1', \
                'CS302HBA1','CS401HBA1','CS501HBA1','ST001']
        antstr2=['RS503HBA','RS305HBA','RS205HBA','RS306HBA', 'RS310HBA','RS406HBA','RS407HBA', \
                'RS106HBA','RS307HBA','RS208HBA','RS210HBA', 'RS409HBA','RS508HBA','RS509HBA']

    if useforresetsols:
        antstrtmp = list(antstr) 
        for ant in antstr:
            if ant not in antennasms:
                antstrtmp.remove(ant) 
        return antstrtmp        

    if ctype != 'core-remote':
        antstrtmp = list(antstr) # important to use list here, otherwise it's not a copy(!!) and antstrtmp refers to antstr
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
        
        antstr =  '[[' + antstr1 + '],[' + antstr2 + ']]'

    return antstr    


def makephasediffh5(phaseh5): 
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5pol = tables.open_file(phaseh5,mode='a')

    phase_pol = H5pol.root.sol000.phase000.val[:] # time, freq, ant, dir, pol
    phase_pol_tmp = np.copy(phase_pol)
    # antenna   = H5pol.root.sol000.phase000.ant[:]
    print('Shape to make phase diff array', phase_pol.shape)
    
    # for ant in range(len(antenna)):
    phase_pol[:, :, :, :,0]  = phase_pol_tmp[:, :, :, :,0] # XX
    phase_pol[:, :, :, :,-1] = 0.0*phase_pol_tmp[:, :, :, :,0] # YY


    H5pol.root.sol000.phase000.val[:] = phase_pol
    H5pol.flush()
    H5pol.close()
    return

def makephaseCDFh5(phaseh5): 
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5 = tables.open_file(phaseh5,mode='a')

    phaseCDF = H5.root.sol000.phase000.val[:] # time, freq, ant, dir, pol
    phaseCDF_tmp = np.copy(phaseCDF)
    print('Shape to make phase CDF array', phaseCDF.shape)
    nfreq = len(H5.root.sol000.phase000.freq[:])
    for ff in range(nfreq-1):
      # reverse order so phase increase towards lower frequnecies
      phaseCDF[:,nfreq-ff-2, ...]  = np.copy(phaseCDF[:,nfreq-ff-2, ...] + phaseCDF[:, nfreq-ff-1, ...])
    
    print(phaseCDF.shape)
    H5.root.sol000.phase000.val[:] = phaseCDF
    H5.flush()
    H5.close()
    return


def copyoverscalarphase(scalarh5, phasexxyyh5): 
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5    = tables.open_file(scalarh5, mode='r')
    H5pol = tables.open_file(phasexxyyh5,mode='a')

    phase     = H5.root.sol000.phase000.val[:] # time, freq, ant, dir
    phase_pol = H5pol.root.sol000.phase000.val[:] # time, freq, ant, dir, pol
    antenna   = H5.root.sol000.phase000.ant[:]
    print('Shapes for pol copy', phase.shape, phase_pol.shape)
    
    for ant in range(len(antenna)):
      phase_pol[:, :, ant, :,0] = phase[:, :, ant, :] # XX
      phase_pol[:, :, ant, :,-1] = phase[:, :, ant, :] # YY

    H5pol.root.sol000.phase000.val[:] = phase_pol[:,:,:]
    H5pol.flush()


    H5.close()
    H5pol.close()
    return

def copyovergain(gaininh5,gainouth5, soltype):
    H5in    = tables.open_file(gaininh5, mode='r')
    H5out   = tables.open_file(gainouth5,mode='a')
    antenna   = H5in.root.sol000.amplitude000.ant[:]
   
    h5 = h5parm.h5parm(gaininh5)
    ss = h5.getSolset('sol000')
    st = ss.getSoltab('amplitude000')    
    axesnames = st.getAxesNames()
    h5.close()
   
    if 'pol' in axesnames:

        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            phase     = H5in.root.sol000.phase000.val[:] 
            H5out.root.sol000.phase000.val[:] = phase
        else:    
            H5out.root.sol000.phase000.val[:] = 0.0
        
        amplitude = H5in.root.sol000.amplitude000.val[:] 
        print('Shapes for gain copy with polarizations', amplitude.shape)
        H5out.root.sol000.amplitude000.val[:] = amplitude
        
    else:
        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            phase     = H5in.root.sol000.phase000.val[:]
            phase_pol = H5out.root.sol000.phase000.val[:] # time, freq, ant, dir, pol
           
        amplitude = H5in.root.sol000.amplitude000.val[:] 
        amplitude_pol   = H5out.root.sol000.amplitude000.val[:] # time, freq, ant, dir, pol
        print('Shapes for gain copy 1 pol', amplitude.shape)
        
        for ant in range(len(antenna)):
            if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
                phase_pol[:, :, ant, :,0] = phase[:, :, ant, :] # XX
                phase_pol[:, :, ant, :,-1] = phase[:, :, ant, :] # YY    
            amplitude_pol[:, :, ant, :,0] = amplitude[:, :, ant, :] # XX
            amplitude_pol[:, :, ant, :,-1] = amplitude[:, :, ant, :] # YY   
        if soltype != 'scalaramplitude' and soltype != 'amplitudeonly':
            H5out.root.sol000.phase000.val[:] = phase_pol[:,:,:]
        else:
            H5out.root.sol000.phase000.val[:] = 0.0
        H5out.root.sol000.amplitude000.val[:] = amplitude_pol[:,:,:]
    
    H5out.flush()
    H5in.close()
    H5out.close()
    return    

def resetgains(parmdb):
   H5 = tables.open_file(parmdb, mode='a')
   H5.root.sol000.phase000.val[:] = 0.0
   H5.root.sol000.amplitude000.val[:] = 1.0
   H5.flush()
   H5.close()
   return

def resetsolsforstations(h5parm, stationlist, refant=None):
   print(h5parm, stationlist)
   fulljones = fulljonesparmdb(h5parm) # True/False
   hasphase = True
   hasamps  = True
   hasrotatation = True
   hastec = True
   
   H=tables.open_file(h5parm, mode='a')
   
   # figure of we have phase and/or amplitude solutions
   try:
     antennas = H.root.sol000.amplitude000.ant[:]
     axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
   except: 
      hasamps = False
   try:
     antennas = H.root.sol000.phase000.ant[:]
     axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
   except:
     hasphase = False
   try:
     antennas = H.root.sol000.tec000.ant[:]
     axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
   except:
     hastec = False
   try:
     antennas = H.root.sol000.rotation000.ant[:]
     axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
   except:
     hasrotatation = False   

   if hasamps:  
     amp = H.root.sol000.amplitude000.val[:]
   if hasphase: # also phasereference
     phase = H.root.sol000.phase000.val[:]
     refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant) # to deal with byte strings
     print(refant_idx, refant)
     antennaxis = axisn.index('ant')  
     axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
     print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
     if antennaxis == 0:
        phasen = phase - phase[refant_idx[0],...]
     if antennaxis == 1:
        phasen = phase - phase[:,refant_idx[0],...]
     if antennaxis == 2:
        phasen = phase - phase[:,:,refant_idx[0],...]
     if antennaxis == 3:
        phasen = phase - phase[:,:,:,refant_idx[0],...]
     if antennaxis == 4:
        phasen = phase - phase[:,:,:,:,refant_idx[0],...]     
     phase = np.copy(phasen)
     
     
   if hastec:
     tec = H.root.sol000.tec000.val[:]  
     refant_idx = np.where(H.root.sol000.tec000.ant[:].astype(str) == refant) # to deal with byte strings
     print(refant_idx, refant)
     antennaxis = axisn.index('ant')  
     axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
     print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
     if antennaxis == 0:
        tecn = tec - tec[refant_idx[0],...]
     if antennaxis == 1:
        tecn = tec - tec[:,refant_idx[0],...]
     if antennaxis == 2:
        tecn = tec - tec[:,:,refant_idx[0],...]
     if antennaxis == 3:
        tecn = tec - tec[:,:,:,refant_idx[0],...]
     if antennaxis == 4:
        tecn = tec - tec[:,:,:,:,refant_idx[0],...]     
     tec = np.copy(tecn)

   if hasrotatation:
     rotation = H.root.sol000.rotation000.val[:]       
     refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant) # to deal with byte strings
     print(refant_idx, refant)
     antennaxis = axisn.index('ant')  
     axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
     print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
     if antennaxis == 0:
        rotationn = rotation - rotation[refant_idx[0],...]
     if antennaxis == 1:
        rotationn = rotation - rotation[:,refant_idx[0],...]
     if antennaxis == 2:
        rotationn = rotation - rotation[:,:,refant_idx[0],...]
     if antennaxis == 3:
        rotationn = rotation - rotation[:,:,:,refant_idx[0],...]
     if antennaxis == 4:
        rotationn = rotation - rotation[:,:,:,:,refant_idx[0],...]     
     rotation = np.copy(rotationn)
     
     
   for antennaid,antenna in enumerate(antennas.astype(str)): # to deal with byte formatted array
     # if not isinstance(antenna, str):
     #  antenna_str = antenna.decode() # to deal with byte formatted antenna names
     # else:
     #  antenna_str = antenna # already str type 
       
     print(antenna, hasphase, hasamps, hastec, hasrotatation)     
     if antenna in stationlist: # in this case reset value to 0.0 (or 1.0)
       if hasphase:
         antennaxis = axisn.index('ant')  
         axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
         print('Resetting phase', antenna, 'Axis entry number', axisn.index('ant'))
         # print(phase[:,:,antennaid,...])
         if antennaxis == 0:
           phase[antennaid,...] = 0.0
         if antennaxis == 1:
           phase[:,antennaid,...] = 0.0
         if antennaxis == 2:
           phase[:,:,antennaid,...] = 0.0
         if antennaxis == 3:
           phase[:,:,:,antennaid,...] = 0.0  
         if antennaxis == 4:
           phase[:,:,:,:,antennaid,...] = 0.0
         # print(phase[:,:,antennaid,...])  
       if hasamps:
         antennaxis = axisn.index('ant')  
         axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
         print('Resetting amplitude', antenna, 'Axis entry number', axisn.index('ant'))
         if antennaxis == 0:
           amp[antennaid,...] = 1.0
         if antennaxis == 1:
           amp[:,antennaid,...] = 1.0
         if antennaxis == 2:
           amp[:,:,antennaid,...] = 1.0
         if antennaxis == 3:
           amp[:,:,:,antennaid,...] = 1.0  
         if antennaxis == 4:
           amp[:,:,:,:,antennaid,...] = 1.0
         if fulljones:  
           amp[...,1] = 0.0 # XY, assumpe pol is last axis
           amp[...,2] = 0.0 # YX, assume pol is last axis
           
       if hastec:
         antennaxis = axisn.index('ant')  
         axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
         print('Resetting TEC', antenna, 'Axis entry number', axisn.index('ant'))
         if antennaxis == 0:
           tec[antennaid,...] = 0.0
         if antennaxis == 1:
           tec[:,antennaid,...] = 0.0
         if antennaxis == 2:
           tec[:,:,antennaid,...] = 0.0
         if antennaxis == 3:
           tec[:,:,:,antennaid,...] = 0.0  
         if antennaxis == 4:
           tec[:,:,:,:,antennaid,...] = 0.0                         
       if hasrotatation:
         antennaxis = axisn.index('ant')  
         axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
         print('Resetting rotation', antenna, 'Axis entry number', axisn.index('ant'))
         if antennaxis == 0:
           rotation[antennaid,...] = 0.0
         if antennaxis == 1:
           rotation[:,antennaid,...] = 0.0
         if antennaxis == 2:
           rotation[:,:,antennaid,...] = 0.0
         if antennaxis == 3:
           rotation[:,:,:,antennaid,...] = 0.0  
         if antennaxis == 4:
           rotation[:,:,:,:,antennaid,...] = 0.0     
   # fill values back in
   if hasphase:
     H.root.sol000.phase000.val[:] = np.copy(phase)
   if hasamps:  
     H.root.sol000.amplitude000.val[:] = np.copy(amp)
   if hastec:
     H.root.sol000.tec000.val[:] = np.copy(tec) 
   if hasrotatation:
     H.root.sol000.rotation000.val[:] = np.copy(rotatation)       
     
   H.flush()
   H.close()


   return


def removenans(parmdb, soltab):
   H5 = h5parm.h5parm(parmdb, readonly=False)
   vals =H5.getSolset('sol000').getSoltab(soltab).getValues()[0]
   weights = H5.getSolset('sol000').getSoltab(soltab).getValues(weight=True)[0]   
   
   idxnan  = np.where((~np.isfinite(vals))) 
   
   # print(idxnan)
   # print('Found some NaNs', vals[idxnan])
   print('Found some NaNs, flagging them....')

   if H5.getSolset('sol000').getSoltab(soltab).getType() == 'phase':
       vals[idxnan] = 0.0
   if H5.getSolset('sol000').getSoltab(soltab).getType() == 'amplitude':
       vals[idxnan] = 1.0            
   if H5.getSolset('sol000').getSoltab(soltab).getType() == 'rotation':
       vals[idxnan] = 0.0
   
   weights[idxnan] = 0.0

   H5.getSolset('sol000').getSoltab(soltab).setValues(weights,weight=True)
   H5.getSolset('sol000').getSoltab(soltab).setValues(vals)
   H5.close()
   return


def losotolofarbeam(parmdb, soltabname, ms, inverse=False, useElementResponse=True, useArrayFactor=True, useChanFreq=True, beamlib='stationresponse'):
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

    numants = pt.taql('select gcount(*) as numants from '+ms+'::ANTENNA').getcol('numants')[0]
    H5ants = len(soltab.getAxisValues('ant'))
    if numants != H5ants:
        H5.close()
        raise ValueError('Number of antennas in Measurement Set does not match number of antennas in H5parm.')

    if (beamlib.lower() == 'stationresponse') or (beamlib.lower() == 'lofarbeam'):
        from lofar.stationresponse import stationresponse
        sr = stationresponse(ms, inverse, useElementResponse, useArrayFactor, useChanFreq)

        for vals, coord, selection in soltab.getValuesIter(returnAxes=['ant','time','pol','freq'], weight=False):
            vals = losoto.lib_operations.reorderAxes( vals, soltab.getAxesNames(), ['ant','time','freq','pol'] )

            for stationnum in range(numants):
                logger.debug('Working on station number %i' % stationnum)
                for itime, time in enumerate(times):
                    beam = sr.evaluateStation(time=time, station=stationnum)
                    # Reshape from [nfreq, 2, 2] to [nfreq, 4]
                    beam = beam.reshape(beam.shape[0], 4)

                    if soltab.getAxisLen('pol') == 2:
                        beam = beam[:,[0,3]] # get only XX and YY
                       
                    if soltab.getType() == 'amplitude':
                        vals[stationnum, itime, :, :] = np.abs(beam)
                    elif soltab.getType() == 'phase':
                        vals[stationnum, itime, :, :] = np.angle(beam)
                    else:
                        logger.error('Beam prediction works only for amplitude/phase solution tables.')
                        return 1

            vals = losoto.lib_operations.reorderAxes( vals, ['ant','time','freq','pol'], [ax for ax in soltab.getAxesNames() if ax in ['ant','time','freq','pol']] )
            soltab.setValues(vals, selection)
    elif beamlib.lower() == 'everybeam':
        import everybeam

        freqs = soltab.getAxisValues('freq')
        
        if useElementResponse and useArrayFactor:
            print('Full (element+array_factor) beam correction requested. Using use_differential_beam=False.')
            obs = everybeam.load_telescope(ms, use_differential_beam=False, use_channel_frequency=useChanFreq)
        elif not useElementResponse and useArrayFactor:
            print('Array factor beam correction requested. Using use_differential_beam=True.')
            obs = everybeam.load_telescope(ms, use_differential_beam=True, use_channel_frequency=useChanFreq)
        elif useElementResponse and not useArrayFactor:
            print('Element beam correction requested.')
            # Not sure how to do this with EveryBeam.
            raise NotImplementedError('Element beam correction is not implemented in facetselfcal.')

        # Obtain direction to calculate beam for.
        phasedir = pt.taql('SELECT PHASE_DIR FROM {ms:s}::FIELD'.format(ms=ms))
        ra, dec = phasedir.getcol('PHASE_DIR').squeeze()

        for vals, coord, selection in soltab.getValuesIter(returnAxes=['ant','time','pol','freq'], weight=False):
            vals = losoto.lib_operations.reorderAxes( vals, soltab.getAxesNames(), ['ant','time','freq','pol'] )

            for stationnum in range(numants):
                logger.debug('Working on station number %i' % stationnum)
                for ifreq, freq in enumerate(freqs):
                    for itime, time in enumerate(times):
                        beam = obs.station_response(time=time, station_idx=stationnum, freq=freq, ra=ra, dec=dec)
                        beam = beam.reshape(4)

                        if soltab.getAxisLen('pol') == 2:
                            beam = beam[[0,3]] # get only XX and YY
                           
                        if soltab.getType() == 'amplitude':
                            vals[stationnum, itime, ifreq, :] = np.abs(beam)
                        elif soltab.getType() == 'phase':
                            vals[stationnum, itime, ifreq, :] = np.angle(beam)
                        else:
                            logger.error('Beam prediction works only for amplitude/phase solution tables.')
                            return 1

            vals = losoto.lib_operations.reorderAxes( vals, ['ant','time','freq','pol'], [ax for ax in soltab.getAxesNames() if ax in ['ant','time','freq','pol']] )
            soltab.setValues(vals, selection)

    else:
        H5.close()
        raise ValueError('Unsupported beam library specified')
    
    H5.close()
    return
 

# losotolofarbeam('P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive_templatejones.h5', 'amplitude000', 'P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive', inverse=False, useElementResponse=False, useArrayFactor=True, useChanFreq=True)


def cleanup(mslist):
    
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

    taql = 'taql'
       
    msout = ms + '.cut'
    
    H5 =h5parm.h5parm(tecsolsfile)
    tec = H5.getSolset('sol000').getSoltab('tec000').getValues() 
    tecvals = tec[0]
    
    axis_names = H5.getSolset('sol000').getSoltab('tec000').getAxesNames()
    time_ind = axis_names.index('time')
    ant_ind = axis_names.index('ant')
    
    # ['time', 'ant', 'dir', 'freq']
    reftec = tecvals[:,0,0,0] 
    
    # print np.shape( tecvals[:,:,0,0]), np.shape( reftec[:,None]), 
    tecvals = tecvals[:,:,0,0] - reftec[:,None] # reference to zero
    
    times   = tec[1]['time']
    
    # print tecvals[:,0]
    
    goodtimesvec = []
    
    for timeid, time in enumerate(times):
    
      tecvals[timeid,:]
      
      # print timeid, np.count_nonzero( tecvals[timeid,:])
      goodtimesvec.append(np.count_nonzero( tecvals[timeid,:]))



    goodstartid = np.argmax (np.array(goodtimesvec) > 0)
    goodendid   = len(goodtimesvec) - np.argmax (np.array(goodtimesvec[::-1]) > 0)
    
    print('First good solutionslot,', goodstartid, ' out of', len(goodtimesvec))
    print('Last good solutionslot,', goodendid, ' out of', len(goodtimesvec))    
    H5.close()
    
    if (goodstartid != 0) or (goodendid != len(goodtimesvec)): # only do if needed to save some time
    
        cmd = taql + " ' select from " + ms + " where TIME in (select distinct TIME from " + ms 
        cmd+= " offset " + str(goodstartid*np.int(tecsolint)) 
        cmd+= " limit " + str((goodendid-goodstartid)*np.int(tecsolint)) +") giving " 
        cmd+= msout + " as plain'"
        
        print(cmd)
        run(cmd)
        
        os.system('rm -rf ' + ms)
        os.system('mv ' + msout + ' ' + ms)
    return


# flagms_startend('P215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archive','phaseonlyP215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archivesolsgrid_9.h5', 2)
# sys.exit()




def removestartendms(ms, starttime=None, endtime=None, dysco=True):

    # chdeck if output is already there and remove    
    if os.path.isdir(ms + '.cut'):
          os.system('rm -rf ' + ms + '.cut')  
    if os.path.isdir(ms + '.cuttmp'):
          os.system('rm -rf ' + ms + '.cuttmp')  

        
    cmd = 'DP3 msin=' + ms + ' ' + 'msout=' + ms + '.cut '
    if dysco:
      cmd+= 'msout.storagemanager=dysco '    
    cmd+=  'msin.weightcolumn=WEIGHT_SPECTRUM steps=[] msout.writefullresflag=False ' 
    if starttime is not None:
      cmd+= 'msin.starttime=' + starttime + ' '
    if endtime is not None:  
      cmd+= 'msin.endtime=' + endtime   + ' '   
    print(cmd)  
    run(cmd)
    
    cmd = 'DP3 msin=' + ms + ' ' + 'msout=' + ms + '.cuttmp '
    if dysco:
      cmd+= 'msout.storagemanager=dysco '
    cmd+= 'msin.weightcolumn=WEIGHT_SPECTRUM_SOLVE steps=[] msout.writefullresflag=False '  
    if starttime is not None:
      cmd+= 'msin.starttime=' + starttime + ' '
    if endtime is not None:  
      cmd+= 'msin.endtime=' + endtime   + ' '
    print(cmd)
    run(cmd)    


    # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
    t  = pt.table(ms + '.cut' , readonly=False)

    print('Adding WEIGHT_SPECTRUM_SOLVE')
    desc = t.getcoldesc('WEIGHT_SPECTRUM')
    desc['name']='WEIGHT_SPECTRUM_SOLVE'
    t.addcols(desc)

    t2 = pt.table(ms + '.cuttmp' , readonly=True)
    imweights = t2.getcol('WEIGHT_SPECTRUM')
    t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

    # Fill WEIGHT_SPECTRUM with WEIGHT_SPECTRUM from second ms
    t2.close()
    t.close() 

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

def _add_astropy_beam(fitsname):
  head = fits.getheader(fitsname)
  bmaj = head['BMAJ']
  bmin = head['BMIN']
  bpa  = head['BPA']
  cdelt = head['CDELT2']
  bmajpix = bmaj/cdelt
  bminpix = bmin/cdelt
  ellipse = matplotlib.patches.Ellipse((20,20), bmajpix,bminpix,bpa)
  return ellipse

def plotimage_astropy(fitsimagename, outplotname, mask=None, rmsnoiseimage=None):
  # image noise for plotting
  if rmsnoiseimage == None:
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

  data = fits.getdata(fitsimagename)
  head = fits.getheader(fitsimagename)
  f = plt.figure()
  ax = f.add_subplot(111,projection=WCS(head),slices=('x','y',0,0))
  img = ax.imshow(data[0,0,:,:],cmap='bone',vmax=16*imagenoise, vmin=-6*imagenoise)
  ax.set_title(fitsimagename+' (noise = {} mJy/beam)'.format(round(imagenoiseinfo*1e3, 3)))
  ax.grid(True)
  ax.set_xlabel('Right Ascension (J2000)')
  ax.set_ylabel('Declination (J2000)') 
  cbar = plt.colorbar(img)
  cbar.set_label('Flux (Jy beam$^{-1}$')
  ax.add_artist(_add_astropy_beam(fitsimagename))

  if mask is not None:
    maskdata = fits.getdata(mask)[0,0,:,:]
    ax.contour(maskdata, colors='red', levels=[0.1*imagenoise],filled=False, smooth=1, alpha=0.6, linewidths=1)

  if os.path.isfile(outplotname + '.png'):
      os.system('rm -f ' + outplotname + '.png')
  plt.savefig(outplotname, dpi=450, format='png')
  logger.info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
  return

def plotimage_aplpy(fitsimagename, outplotname, mask=None, rmsnoiseimage=None):
  import aplpy
  # image noise for plotting
  if rmsnoiseimage == None:
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
  f.show_colorscale(vmax=16*imagenoise, vmin=-6*imagenoise, cmap='bone')
  f.set_title(fitsimagename+' (noise = {} mJy/beam)'.format(round(imagenoiseinfo*1e3, 3)))
  try: # to work around an aplpy error
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
  f.colorbar.set_axis_label_text('Flux (Jy beam$^{-1}$)')
  if mask is not None:
    try:  
      f.show_contour(mask, colors='red', levels=[0.1*imagenoise], filled=False, smooth=1, alpha=0.6, linewidths=1)
    except:
      pass  
  if os.path.isfile(outplotname + '.png'):
      os.system('rm -f ' + outplotname + '.png')
  f.save(outplotname, dpi=120, format='png')
  logger.info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
  return


def plotimage(fitsimagename,outplotname,mask=None,rmsnoiseimage=None):
  # This code tries astropy first, switches to aplpy afterwards.
  try:
      plotimage_astropy(fitsimagename,outplotname,mask,rmsnoiseimage)
  except:
      plotimage_aplpy(fitsimagename,outplotname,mask,rmsnoiseimage)



def archive(mslist, outtarname, regionfile, fitsmask, imagename, dysco=True):
  path = '/disks/ftphome/pub/vanweeren'
  for ms in mslist:
    msout = ms + '.calibrated'
    if os.path.isdir(msout):
      os.system('rm -rf ' + msout)
    cmd  ='DP3 numthreads='+ str(multiprocessing.cpu_count()) +' msin=' + ms + ' msout=' + msout + ' '
    cmd +='msin.datacolumn=CORRECTED_DATA msout.writefullresflag=False steps=[] '
    if dysco:
      cmd += 'msout.storagemanager=dysco '    
    run(cmd)
 

  msliststring = ' '.join(map(str, glob.glob('*.calibrated') ))
  cmd = 'tar -zcf ' + outtarname + ' ' + msliststring + ' selfcal.log ' +  imagename + ' '
     
  if fitsmask != None:  # add fitsmask to tar if it exists
    if os.path.isfile(fitsmask):
      cmd +=  fitsmask + ' '

  if regionfile != None:  # add box regionfile to tar if it exists
    if os.path.isfile(regionfile):
      cmd +=  regionfile + ' '
     
 
  if os.path.isfile(outtarname):
      os.system('rm -f ' + outtarname)
  logger.info('Creating archived calibrated tarball: ' + outtarname)    
  run(cmd)
  
  for ms in mslist:
    msout = ms + '.calibrated'   
    os.system('rm -rf ' + msout)
  return



def setinitial_solint(mslist, longbaseline, LBA, options):
   """
   take user input solutions,nchan,smoothnessconstraint,antennaconstraint and expand them to all ms
   these list can then be updated later with values from auto_determinesolints for example
   """

   if os.path.isfile('nchan.p') and os.path.isfile('solint.p'):
    
      f = open('nchan.p', 'rb') 
      nchan_list = pickle.load(f)        
      f.close()   
  
      f = open('solint.p', 'rb') 
      solint_list = pickle.load(f)        
      f.close()   

      f = open('antennaconstraint.p', 'rb') 
      antennaconstraint_list = pickle.load(f)        
      f.close()   

      f = open('resetsols.p', 'rb') 
      resetsols_list = pickle.load(f)        
      f.close()   

      f = open('smoothnessconstraint.p', 'rb') 
      smoothnessconstraint_list = pickle.load(f)        
      f.close()
      
      f = open('smoothnessreffrequency.p', 'rb') 
      smoothnessreffrequency_list = pickle.load(f)        
      f.close()
      
      f = open('smoothnessspectralexponent.p', 'rb')
      smoothnessspectralexponent_list = pickle.load(f)
      f.close()

      f = open('smoothnessrefdistance.p', 'rb')
      smoothnessrefdistance_list = pickle.load(f)
      f.close()

      f = open('soltypecycles.p', 'rb') 
      soltypecycles_list = pickle.load(f)        
      f.close()   
      
  
   else:
      nchan_list  = [] # list with len(options.soltype_list)
      solint_list = [] # list with len(options.soltype_list)
      smoothnessconstraint_list = [] # nested list with len(options.soltype_list), inner list is for ms
      smoothnessreffrequency_list = [] # nested list with len(options.soltype_list), inner list is for ms
      smoothnessspectralexponent_list = [] # nest list with len(options.soltype_list), inner list is for ms
      smoothnessrefdistance_list = [] #  # nest list with len(options.soltype_list), inner list is for ms
      antennaconstraint_list = [] # nested list with len(options.soltype_list), inner list is for ms
      resetsols_list = [] # nested list with len(options.soltype_list), inner list is for ms
      soltypecycles_list = []  # nested list with len(options.soltype_list), inner list is for ms

      for soltype_id, soltype in enumerate(options.soltype_list):
        nchan_ms   = [] # list with len(mslist)
        solint_ms  = [] # list with len(mslist)
        antennaconstraint_list_ms   = [] # list with len(mslist)
        resetsols_list_ms = [] # list with len(mslist)
        smoothnessconstraint_list_ms  = [] # list with len(mslist)
        smoothnessreffrequency_list_ms  = [] # list with len(mslist)
        smoothnessspectralexponent_list_ms = [] # list with len(mslist)
        smoothnessrefdistance_list_ms = [] # list with len(mslist)
        soltypecycles_list_ms = [] # list with len(mslist)

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
            nchan = 10     

          # smoothnessconstraint 
          try:
            smoothnessconstraint = options.smoothnessconstraint_list[soltype_id]
          except:
            smoothnessconstraint = 0.0

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

          # soltypecycles
          soltypecycles = options.soltypecycles_list[soltype_id]

          # force nchan 1 for tec(andphase) solve and in case smoothnessconstraint is invoked
          if soltype == 'tec' or  soltype == 'tecandphase' or smoothnessconstraint > 0.0:
            nchan  = 1

 
          nchan_ms.append(nchan)
          solint_ms.append(solint)
          smoothnessconstraint_list_ms.append(smoothnessconstraint)
          smoothnessreffrequency_list_ms.append(smoothnessreffrequency)
          smoothnessspectralexponent_list_ms.append(smoothnessspectralexponent)
          smoothnessrefdistance_list_ms.append(smoothnessrefdistance)
          antennaconstraint_list_ms.append(antennaconstraint)
          resetsols_list_ms.append(resetsols)
          soltypecycles_list_ms.append(soltypecycles)

        
        nchan_list.append(nchan_ms)   # list of lists
        solint_list.append(solint_ms) # list of lists
        antennaconstraint_list.append(antennaconstraint_list_ms)   # list of lists
        resetsols_list.append(resetsols_list_ms) # list of lists
        smoothnessconstraint_list.append(smoothnessconstraint_list_ms) # list of lists
        smoothnessreffrequency_list.append(smoothnessreffrequency_list_ms) # list of lists
        smoothnessspectralexponent_list.append(smoothnessspectralexponent_list_ms) # list of lists
        smoothnessrefdistance_list.append(smoothnessrefdistance_list_ms)
        
        soltypecycles_list.append(soltypecycles_list_ms)

      f = open('nchan.p', 'wb') 
      pickle.dump(nchan_list,f)        
      f.close()   
  
      f = open('solint.p', 'wb') 
      pickle.dump(solint_list,f)        
      f.close()  
      
      f = open('smoothnessconstraint.p', 'wb') 
      pickle.dump(smoothnessconstraint_list,f)        
      f.close()  

      f = open('smoothnessreffrequency.p', 'wb') 
      pickle.dump(smoothnessreffrequency_list,f)        
      f.close()  

      f = open('smoothnessspectralexponent.p', 'wb') 
      pickle.dump(smoothnessspectralexponent_list,f)        
      f.close()  

      f = open('smoothnessrefdistance.p', 'wb') 
      pickle.dump(smoothnessrefdistance_list,f)        
      f.close()  
      
      f = open('antennaconstraint.p', 'wb') 
      pickle.dump(antennaconstraint_list,f)        
      f.close()        

      f = open('resetsols.p', 'wb') 
      pickle.dump(resetsols_list,f)        
      f.close()  

      f = open('soltypecycles.p', 'wb') 
      pickle.dump(soltypecycles_list,f)        
      f.close()   
      
      
   print('soltype:',options.soltype_list, mslist)   
   print('nchan:',nchan_list)
   print('solint:',solint_list)
   print('smoothnessconstraint:',smoothnessconstraint_list)
   print('smoothnessreffrequency:',smoothnessreffrequency_list)
   print('smoothnessspectralexponent:',smoothnessspectralexponent_list)
   print('smoothnessrefdistance:',smoothnessrefdistance_list)
   print('antennaconstraint:',antennaconstraint_list)
   print('resetsols:',resetsols_list)
   print('soltypecycles:',soltypecycles_list)

   logger.info('soltype: '+ str(options.soltype_list) + ' ' + str(mslist))   
   logger.info('nchan: ' + str(options.nchan_list))
   logger.info('solint: ' + str(options.solint_list))
   logger.info('smoothnessconstraint: ' + str(options.smoothnessconstraint_list))
   logger.info('smoothnessreffrequency: ' + str(options.smoothnessreffrequency_list))
   logger.info('smoothnessspectralexponent: ' + str(options.smoothnessspectralexponent_list))
   logger.info('smoothnessrefdistance: ' + str(options.smoothnessrefdistance_list))
   logger.info('antennaconstraint: ' + str(options.antennaconstraint_list))
   logger.info('resetsols: ' + str(options.resetsols_list))
   logger.info('soltypecycles: ' + str(options.soltypecycles_list))   
   
   return nchan_list, solint_list, smoothnessconstraint_list, smoothnessreffrequency_list, smoothnessspectralexponent_list, smoothnessrefdistance_list, antennaconstraint_list, resetsols_list, soltypecycles_list

def getmsmodelinfo(ms, modelcolumn, fastrms=False, uvcutfraction=0.333):
   t = pt.table(ms + '/SPECTRAL_WINDOW')
   chanw = np.median(t.getcol('CHAN_WIDTH'))
   freq = np.median(t.getcol('CHAN_FREQ'))
   nfreq = len(t.getcol('CHAN_FREQ')[0])
   t.close()
   uvdismod = get_uvwmax(ms)*uvcutfraction # take range [uvcutfraction*uvmax - 1.0uvmax]
   
   HBA_upfreqsel = 0.75 # select only freqcencies above 75% of the available bandwidth
   freqct = 1000e6
   # the idea is that for HBA if you are far out in the beam the noise gets up much more at the higher freqs and the model flux goes down due to the spectral index. In this way we get more conservative solints
          
          
   t = pt.taql('SELECT ' + modelcolumn + ',DATA,UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > '+ str(uvdismod) )
   model = np.abs(t.getcol(modelcolumn))
   flags = t.getcol('FLAG')
   data  = t.getcol('DATA')
   print('Compute visibility noise of the dataset with robust sigma clipping', ms)
   logger.info('Compute visibility noise of the dataset with robust sigma clipping: ' + ms)
   if fastrms:    # take only every fifth element of the array to speed up the computation
     if freq > freqct: # HBA
        noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[0:data.shape[0]:5,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,1:3],\
        mask=flags[0:data.shape[0]:5,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,1:3])[2] # use XY and YX
     else:   
        noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[0:data.shape[0]:5,:,1:3],\
        mask=flags[0:data.shape[0]:5,:,1:3])[2] # use XY and YX
   else:
     if freq > freqct: # HBA
        noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[:,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,1:3],\
        mask=flags[:,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,1:3])[2] # use XY and YX
     else:
        noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[:,:,1:3],\
        mask=flags[:,:,1:3])[2] # use XY and YX         
   
   model = np.ma.masked_array(model, flags)
   if freq > freqct: # HBA:
      flux  = np.ma.mean((model[:,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,0] + model[:,np.int(np.floor(np.float(nfreq)*HBA_upfreqsel)):-1,3])*0.5) # average XX and YY (ignore XY and YX, they are zero, or nan, in other words this is Stokes I)
   else:
      flux  = np.ma.mean((model[:,:,0] + model[:,:,3])*0.5) # average XX and YY (ignore XY and YX, they are zero, or nan)
   time  = np.unique(t.getcol('TIME'))
   tint  = np.abs(time[1]-time[0])
   print('Integration time visibilities', tint)
   logger.info('Integration time visibilities: ' + str(tint))
   t.close()

   del data, flags, model
   print('Noise visibilities:', noise, 'Jy')
   print('Flux in model:', flux, 'Jy')
   print('UV-selection to compute model flux:', str(uvdismod/1e3), 'km')
   logger.info('Noise visibilities: ' + str(noise) + 'Jy')
   logger.info('Flux in model: ' + str(flux) + 'Jy')
   logger.info('UV-selection to compute model flux: ' + str(uvdismod/1e3) + 'km')

   
   return noise, flux, tint, chanw

def return_soltype_index(soltype_list, soltype, occurence=1, onetectypeoccurence=False):
   
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

def auto_determinesolints(mslist, soltype_list, longbaseline, LBA,\
                          innchan_list=None, insolint_list=None,\
                          uvdismod=None, modelcolumn='MODEL_DATA', redo=False,\
                          insmoothnessconstraint_list=None, insmoothnessreffrequency_list=None, \
                          insmoothnessspectralexponent_list=None,\
                          insmoothnessrefdistance_list=None,\
                          inantennaconstraint_list=None, inresetsols_list=None, \
                          insoltypecycles_list=None, tecfactorsolint=1.0, gainfactorsolint=1.0,\
                          phasefactorsolint=1.0, delaycal=False):
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
              (soltype_id == return_soltype_index(soltype_list, 'tecandphase', occurence=1, onetectypeoccurence=True))) :
          
             if LBA: 
               if longbaseline:
                 solint_sf = 3.0e-3*tecfactorsolint # untested
               else: #for -- LBA dutch --
                 solint_sf = 4.0e-2*tecfactorsolint #0.5e-3 # for tecandphase and coreconstraint
          
             else: # for -- HBA --
               if longbaseline:
                 solint_sf = 0.5e-2*tecfactorsolint # for tecandphase, no coreconstraint          
               else: #for -- HBA dutch --
                 solint_sf = 4.0e-2*tecfactorsolint # for tecandphase, no coreconstraint          
          
             if soltype == 'tec':
               solint_sf = solint_sf/np.sqrt(2.) # tec and coreconstraint
 
 
             # trigger antennaconstraint_phase core if solint > tint 
             if not longbaseline and (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) > tint):
               print(tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3))
               solint_sf = solint_sf/30. 
               print('Trigger_antennaconstraint core:', soltype, ms)
               logger.info('Trigger_antennaconstraint core: '+ soltype + ' ' + ms)
               inantennaconstraint_list[soltype_id][ms_id] = 'core'
               # do another pertubation, a slow solve of the core stations
               # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
               if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 720.0): # less than 12 min now, also doing
                 inantennaconstraint_list[soltype_id+1][ms_id] = 'remote' # or copy over input ??
                 insoltypecycles_list[soltype_id+1][ms_id] = insoltypecycles_list[soltype_id][ms_id] # do + 1 here??
                 insolint_list[soltype_id+1][ms_id] = np.int(np.rint(10.*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) ))
                 if insolint_list[soltype_id+1][ms_id] < 1:
                   insolint_list[soltype_id+1][ms_id] = 1    
               else:
                 insoltypecycles_list[soltype_id+1][ms_id] = 999                 
             
             else:  
               inantennaconstraint_list[soltype_id][ms_id] = None # or copy over input            
             
             # round to nearest integer  
             solint = np.rint(solint_sf* ((noise/flux)**2) * (chanw/390.625e3) )
             # frequency scaling is need because if we avearge in freqeuncy the solint should not change for a tec(andphase) solve
             
             if (longbaseline) and (not LBA) and (soltype == 'tec') \
                and (soltype_list[1] == 'tecandphase'):
                if solint < 0.5 and (solint*tint < 16.): # so less then 16 sec   
                   print('Longbaselines bright source detected: changing from tec to tecandphase solve')
                   insoltypecycles_list[soltype_id][ms_id] = 999
                   insoltypecycles_list[1][ms_id] = 0  
             
             if solint < 1:
                solint = 1        
             if (np.float(solint)*tint/3600.) > 0.5: # so check if larger than 30 min
               print('Warning, it seems there is not enough flux density on the longer baselines for solving')
               logger.warning('Warning, it seems there is not enough flux density on the longer baselines for solving')
               solint = np.rint(0.5*3600./tint) # max is 30 min 

             print(solint_sf*((noise/flux)**2)*(chanw/390.625e3), 'Using tec(andphase) solint:', solint)
             logger.info(str(solint_sf*((noise/flux)**2)*(chanw/390.625e3)) + '-- Using tec(andphase) solint:' + str(solint))
             print('Using tec(andphase) solint [s]:', np.float(solint)*tint)
             logger.info('Using tec(andphase) solint [s]: ' + str(np.float(solint)*tint))
          
             insolint_list[soltype_id][ms_id] = np.int(solint)
             innchan_list[soltype_id][ms_id] = 1
             
          ######## SCALARPHASE or PHASEONLY ######
          ######## for first occurence of tec(andphase) #######
          if soltype in ['scalarphase', 'phaseonly'] and \
              (insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
              ((soltype_id == return_soltype_index(soltype_list, 'scalarphase', occurence=1, onetectypeoccurence=True)) or \
              (soltype_id == return_soltype_index(soltype_list, 'phaseonly', occurence=1, onetectypeoccurence=True))) :
          
             if LBA: 
               if longbaseline:
                 solint_sf = 3.0e-3*phasefactorsolint # untested
               else: #for -- LBA dutch --
                 solint_sf = 4.0e-2*phasefactorsolint #0.5e-3 # for tecandphase and coreconstraint
          
             else: # for -- HBA --
               if longbaseline:
                 solint_sf = 0.5e-2*phasefactorsolint # for tecandphase, no coreconstraint          
               else: #for -- HBA dutch --
                 solint_sf = 4.0e-2*phasefactorsolint # for tecandphase, no coreconstraint          
          
             if soltype == 'scalarphase':
               solint_sf = solint_sf/np.sqrt(2.) # decrease solint if scalarphase
 
 
             # trigger antennaconstraint_phase core if solint > tint
             # needs checking, this might be wrong, this assumes we use [scalarphase/phaseonly,scalarphase/phaseonly, (scalar)complexgain] so 3 steps.....
             if not longbaseline and (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) > tint):
               print(tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3))
               solint_sf = solint_sf/30. 
               print('Trigger_antennaconstraint core:', soltype, ms)
               logger.info('Trigger_antennaconstraint core: '+ soltype + ' ' + ms)
               inantennaconstraint_list[soltype_id][ms_id] = 'core'
               # do another pertubation, a slow solve of the core stations
               # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
               if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 720.0): # less than 12 min now, also doing
                 inantennaconstraint_list[soltype_id+1][ms_id] = 'remote' # or copy over input ??
                 insoltypecycles_list[soltype_id+1][ms_id] = insoltypecycles_list[soltype_id][ms_id] # do + 1 here??
                 insolint_list[soltype_id+1][ms_id] = np.int(np.rint(10.*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) ))
                 if insolint_list[soltype_id+1][ms_id] < 1:
                   insolint_list[soltype_id+1][ms_id] = 1    
               else:
                 insoltypecycles_list[soltype_id+1][ms_id] = 999                 
             
             else:  
               inantennaconstraint_list[soltype_id][ms_id] = None # or copy over input            
             
             # round to nearest integer  
             solint = np.rint(solint_sf* ((noise/flux)**2) * (chanw/390.625e3) )
             # frequency scaling is needed because if we avearge in freqeuncy the solint should not change for a (scalar)phase solve with smoothnessconstraint
             
             # if (longbaseline) and (not LBA) and (soltype == 'tec') \
             #   and (soltype_list[1] == 'tecandphase'):
             #   if solint < 0.5 and (solint*tint < 16.): # so less then 16 sec   
             #      print('Longbaselines bright source detected: changing from tec to tecandphase solve')
             #      insoltypecycles_list[soltype_id][ms_id] = 999
             #      insoltypecycles_list[1][ms_id] = 0  
             
             if solint < 1:
                solint = 1        
             if (np.float(solint)*tint/3600.) > 0.5: # so check if larger than 30 min
               print('Warning, it seems there is not enough flux density on the longer baselines for solving')
               logger.warning('Warning, it seems there is not enough flux density on the longer baselines for solving')
               solint = np.rint(0.5*3600./tint) # max is 30 min 

             print(solint_sf*((noise/flux)**2)*(chanw/390.625e3), 'Using (scalar)phase solint:', solint)
             logger.info(str(solint_sf*((noise/flux)**2)*(chanw/390.625e3)) + '-- Using (scalar)phase solint:' + str(solint))
             print('Using (scalar)phase solint [s]:', np.float(solint)*tint)
             logger.info('Using (scalar)phase solint [s]: ' + str(np.float(solint)*tint))
          
             insolint_list[soltype_id][ms_id] = np.int(solint)
             innchan_list[soltype_id][ms_id] = 1 # because we use smoothnessconstraint
             



          ######## COMPLEXGAIN or SCALARCOMPLEXGAIN or AMPLITUDEONLY or SCALARAMPLITUDE ######
          # requires smoothnessconstraint
          # for first occurence of (scalar)complexgain 
          if soltype in ['complexgain', 'scalarcomplexgain'] and (insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
              ((soltype_id == return_soltype_index(soltype_list, 'complexgain', occurence=1)) or \
              (soltype_id == return_soltype_index(soltype_list, 'scalarcomplexgain', occurence=1))):

             thr_disable_gain = 64. # 32. #  72.
             thr_SM15Mhz = 4.
             thr_gain_trigger_allantenna =  32. # 16. # 8.
             
             tgain_max = 4. # do not allow ap solves that are more than 4 hrs
             tgain_min = 0.3333  # check if less than 20 min, min solint is 20 min
             
             innchan_list[soltype_id][ms_id] = 1
 
             if LBA: 
               if longbaseline:
                 solint_sf = 0.4*gainfactorsolint # untested
               else: #for -- LBA dutch --
                 solint_sf = 10.0*gainfactorsolint 
          
             else: # for -- HBA --
               if longbaseline:
                 solint_sf = 0.8*gainfactorsolint #   
               else: #for -- HBA dutch --
                 solint_sf = 0.8*gainfactorsolint #      

             solint = np.rint(solint_sf*((noise/flux)**2)*(chanw/390.625e3)) 
             print(solint_sf*((noise/flux)**2)*(chanw/390.625e3), 'Computes gain solint:', solint, ' ')
             logger.info(str(solint_sf*((noise/flux)**2)*(chanw/390.625e3)) + ' Computes gain solint: ' + str(solint))
             print('Computes gain solint [hr]:', np.float(solint)*tint/3600.)
             logger.info('Computes gain solint [hr]: ' + str(np.float(solint)*tint/3600.))

             # do not allow very short ap solves
             if ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.) < tgain_min: #  check if less than tgain_min (20 min)
               solint = np.rint(tgain_min*3600./tint) # minimum tgain_min is 20 min 
               print('Setting gain solint to 20 min (the min value allowed):', np.float(solint)*tint/3600.)
               logger.info('Setting gain solint to 20 min (the min value allowed): ' + str(np.float(solint)*tint/3600.))

             # do not allow ap solves that are more than tgain_max (4) hrs
             if ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.) > tgain_max: # so check if larger than 30 min
               print('Warning, it seems there is not enough flux density for gain solving')
               logger.warning('Warning, it seems there is not enough flux density for gain solving')
               solint = np.rint(tgain_max*3600./tint) # max is tgain_max (4) hrs  

             # trigger 15 MHz smoothnessconstraint 
             # print('TEST:', ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.))
             if ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.) < thr_SM15Mhz: # so check if larger than 30 min
               insmoothnessconstraint_list[soltype_id][ms_id] = 5.0
             else:
               print('Increasing smoothnessconstraint to 15 MHz')
               logger.info('Increasing smoothnessconstraint to 15 MHz')
               insmoothnessconstraint_list[soltype_id][ms_id] = 15.0

             # trigger nchan=0 solve because not enough S/N
             if not longbaseline and (((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.) > thr_gain_trigger_allantenna):
               inantennaconstraint_list[soltype_id][ms_id] = 'all'
               solint = np.rint(2.0*3600./tint) # 2 hrs nchan=0 solve (do not do bandpass because slope can diverge)
               innchan_list[soltype_id][ms_id] = 0 # no frequency dependence, smoothnessconstraint will be turned of in runDPPPbase
               print('Triggering antennaconstraint all:', soltype, ms)
               logger.info('Triggering antennaconstraint all: ' + soltype + ' ' + ms)
             else:  
               inantennaconstraint_list[soltype_id][ms_id] = None
          
             # completely disable slow solve if the solints get too long, target is too faint
             if (((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.) > thr_disable_gain):
               insoltypecycles_list[soltype_id][ms_id] = 999
               print('Disabling solve:', soltype, ms)
               logger.info('Disabling solve: ' + soltype + ' '+ ms)
             else:  
               insoltypecycles_list[soltype_id][ms_id] = 3 # set to user input value? problem because not retained now
  
             insolint_list[soltype_id][ms_id] = np.int(solint)
  
             # --------------- NCHAN ------------- NOT BEING USED, keep code in case we need it later

             if insmoothnessconstraint_list[soltype_id][ms_id] == 0.0 and innchan_list[soltype_id][ms_id] != 0: # DOES NOT GET HERE BECAUSE smoothnessconstraint > 0 test above

                if LBA: 
                  if longbaseline:
                    print('Not supported')
                    sys.exit()
                  else: #for -- LBA dutch, untested --
                    nchan_sf = 0.75 # for tecandphase and coreconstraint
          
                else: # for -- HBA --
                  if longbaseline:
                    nchan_sf = 0.0075 #   
                  else: #for -- HBA dutch --
                    nchan_sf = 0.75 #          
          
                nchan = np.rint(nchan_sf*(noise/flux)**2)

                # do not allow very low nchan solves
                if (np.float(nchan)*chanw/1e6) < 2.0: #  check if less than 2 MHz
                  nchan = np.rint(2.0*1e6/chanw) # 2 MHz
               
                # do not allow nchan solves that are more than 15 MHz
                if (np.float(nchan)*chanw/1e6) > 15.0: 
                  print('Warning, it seems there is not enough flux density on the longer baselines for solving')
                  nchan = np.rint(15*1e6/chanw) # 15 MHz  
 
                print(nchan_sf*(noise/flux)**2, 'Using gain nchan:', nchan)
                print('Using gain nchan [MHz]:', np.float(nchan)*chanw/1e6) 

                innchan_list[soltype_id][ms_id] = np.int(nchan)


   f = open('nchan.p', 'wb') 
   pickle.dump(innchan_list,f)        
   f.close()   
  
   f = open('solint.p', 'wb') 
   pickle.dump(insolint_list,f)        
   f.close()   

   f = open('smoothnessconstraint.p', 'wb') 
   pickle.dump(insmoothnessconstraint_list,f)        
   f.close()         

   f = open('smoothnessreffrequency.p', 'wb') 
   pickle.dump(insmoothnessreffrequency_list,f)        
   f.close()   

   f = open('smoothnessspectralexponent.p', 'wb') 
   pickle.dump(insmoothnessspectralexponent_list,f)        
   f.close()   

   f = open('smoothnessrefdistance.p', 'wb') 
   pickle.dump(insmoothnessrefdistance_list,f)        
   f.close()   
  
   f = open('antennaconstraint.p', 'wb') 
   pickle.dump(inantennaconstraint_list,f)        
   f.close()   

   f = open('resetsols.p', 'wb') 
   pickle.dump(inresetsols_list,f)        
   f.close()   

   f = open('soltypecycles.p', 'wb') 
   pickle.dump(insoltypecycles_list,f)        
   f.close()         

   print('soltype:',soltype_list, mslist)   
   print('nchan:',innchan_list)
   print('solint:',insolint_list)
   print('smoothnessconstraint:',insmoothnessconstraint_list)
   print('smoothnessreffrequency:',insmoothnessreffrequency_list)
   print('smoothnessspectralexponent_list:',insmoothnessspectralexponent_list)
   print('smoothnessrefdistance_list:',insmoothnessrefdistance_list)
   print('antennaconstraint:',inantennaconstraint_list)
   print('resetsols:',inresetsols_list)
   print('soltypecycles:',insoltypecycles_list)

   logger.info('soltype: '+ str(soltype_list) + ' ' + str(mslist))   
   logger.info('nchan: ' + str(innchan_list))
   logger.info('solint: ' + str(insolint_list))
   logger.info('smoothnessconstraint: ' + str(insmoothnessconstraint_list))
   logger.info('smoothnessreffrequency: ' + str(insmoothnessreffrequency_list))
   logger.info('smoothnessspectralexponent: ' + str(insmoothnessspectralexponent_list))
   logger.info('smoothnessrefdistance: ' + str(insmoothnessrefdistance_list))
   logger.info('antennaconstraint: ' + str(inantennaconstraint_list))
   logger.info('resetsols: ' + str(inresetsols_list))
   logger.info('soltypecycles: ' + str(insoltypecycles_list))

      
   return innchan_list, insolint_list, insmoothnessconstraint_list, insmoothnessreffrequency_list, insmoothnessspectralexponent_list, insmoothnessrefdistance_list, inantennaconstraint_list, inresetsols_list, insoltypecycles_list



def create_beamcortemplate(ms):
  """
  create a DPPP gain H5 template solutution file that can be filled with losoto
  """
  H5name = ms + '_templatejones.h5'   

  cmd = 'DP3 numthreads='+str(multiprocessing.cpu_count())+ ' msin=' + ms + ' msin.datacolumn=DATA msout=. '
  cmd += 'msin.modelcolumn=DATA '
  cmd += 'steps=[ddecal] ddecal.type=ddecal '
  cmd += 'ddecal.maxiter=1 ddecal.usemodelcolumn=True ddecal.nchan=1 '
  cmd += 'ddecal.mode=complexgain ddecal.h5parm=' + H5name  + ' '
  cmd += 'ddecal.solint=10'

  print(cmd)
  run(cmd)

  return H5name

def create_losoto_beamcorparset(ms, refant='CS003HBA0'):
    """
    Create a losoto parset to fill the beam correction values'.
    """
    parset = 'losotobeam.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n\n\n')

    f.write('[plotphase]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-0.5,0.5]\n')
    f.write('prefix = plotlosoto%s/phases_beam\n' % ms)
    f.write('refAnt = %s\n\n\n' % refant)

    f.write('[plotamp]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/amplitude000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [0.2,1]\n')
    f.write('prefix = plotlosoto%s/amplitudes_beam\n' %ms)

    f.close()
    return parset

def create_losoto_tecandphaseparset(ms, refant='CS003HBA0', outplotname='fasttecandphase', markersize=2):
    parset = 'losoto_plotfasttecandphase.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')
  
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
    f.write('markerSize=%s\n' % np.int(markersize))
    f.write('prefix = plotlosoto%s/fasttecandphase\n' % ms)
    f.write('refAnt = %s\n' % refant)
  
    f.close()
    return parset

def create_losoto_tecparset(ms, refant='CS003HBA0', outplotname='fasttec', markersize=2):
    parset = 'losoto_plotfasttec.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')
  
    f.write('pol = []\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plottec]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/tec000]\n')
    f.write('axesInPlot = [time]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-0.2,0.2]\n')
    f.write('figSize=[120,20]\n')
    f.write('markerSize=%s\n' % np.int(markersize))
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname))
    f.write('refAnt = %s\n' % refant)
  
    f.close()
    return parset



def create_losoto_rotationparset(ms, refant='CS003HBA0', onechannel=False, \
                                 outplotname='rotatation', markersize=2):
    parset = 'losoto_plotrotation.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plotrotation]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/rotation000]\n')
    f.write('markerSize=%s\n' % np.int(markersize))
    if onechannel:
      f.write('axesInPlot = [time]\n')      
    else:
      f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('figSize=[120,20]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname))
    f.write('refAnt = %s\n' % refant)
    f.close()
    return parset


def create_losoto_fastphaseparset(ms, refant='CS003HBA0', onechannel=False, onepol=False, outplotname='fastphase'):
    parset = 'losoto_plotfastphase.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

    f.write('pol = [XX,YY]\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    f.write('[plotphase]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    if onechannel:
      f.write('axesInPlot = [time]\n')
      if not onepol:
        f.write('axisInCol = pol\n')
      
    else:
      f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('figSize=[120,20]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname))
    f.write('refAnt = %s\n' % refant)

    if not onepol:
      f.write('[plotphasediff]\n')
      f.write('operation = PLOT\n')
      f.write('soltab = [sol000/phase000]\n')
      if onechannel:
        f.write('axesInPlot = [time]\n')
      else:
        f.write('axesInPlot = [time,freq]\n')
      f.write('axisInTable = ant\n')
      f.write('minmax = [-3.14,3.14]\n')
      f.write('figSize=[120,20]\n')
      f.write('prefix = plotlosoto%s/%spoldiff\n' % (ms,outplotname))
      f.write('refAnt = %s\n' % refant)  
      f.write('axisDiff=pol\n')
 
        
    f.close()
    return parset


def create_losoto_flag_apgridparset(ms, flagging=True, maxrms=7.0, maxrmsphase=7.0, includesphase=True, \
                                    refant='CS003HBA0', onechannel=False, medamp=2.5, flagphases=True, \
                                    onepol=False, outplotname='slowamp', fulljones=False):

    parset= 'losoto_flag_apgrid.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

    # f.write('pol = []\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')
   
    f.write('[plotamp]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/amplitude000]\n')
    if onechannel:
      f.write('axesInPlot = [time]\n')
      if not onepol:
        f.write('axisInCol = pol\n')
    else:
      f.write('axesInPlot = [time,freq]\n')   
    f.write('axisInTable = ant\n')
    # if longbaseline:
    #  f.write('minmax = [0,2.5]\n')        
    # else:    
    f.write('minmax = [%s,%s]\n' % (str(medamp/4.0), str(medamp*2.5)))
    # f.write('minmax = [0,2.5]\n')
    f.write('prefix = plotlosoto%s/%samp\n\n\n' % (ms,outplotname))

    if fulljones:
       f.write('[plotampXYYX]\n')
       f.write('operation = PLOT\n')
       f.write('soltab = [sol000/amplitude000]\n')
       f.write('pol = [XY, YX]\n')
       if onechannel:
         f.write('axesInPlot = [time]\n')
       else:
         f.write('axesInPlot = [time,freq]\n')   
       f.write('axisInTable = ant\n')
       f.write('minmax = [%s,%s]\n' % (str(0.0), str(0.5)))
       f.write('prefix = plotlosoto%s/%sampXYYX\n\n\n' % (ms,outplotname))
       


    if includesphase:   
        f.write('[plotphase]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/phase000]\n')
        if onechannel:
          f.write('axesInPlot = [time]\n')
          if not onepol:
            f.write('axisInCol = pol\n')
        else:
           f.write('axesInPlot = [time,freq]\n')    
        f.write('axisInTable = ant\n')
        f.write('minmax = [-3.14,3.14]\n')
        f.write('prefix = plotlosoto%s/%sphase\n' % (ms,outplotname))
        f.write('refAnt = %s\n\n\n' % refant)

    if flagging:
        f.write('[flagamp]\n')
        f.write('soltab = [sol000/amplitude000]\n')
        f.write('operation = FLAG\n')
        if onechannel:
          f.write('axesToFlag = [time]\n')
        else:
          f.write('axesToFlag = [time,freq]\n')
        f.write('mode = smooth\n')
        f.write('maxCycles = 3\n')
        f.write('windowNoise = 7\n')
        f.write('maxRms = %s\n' % str(maxrms))
        if onechannel:
          f.write('order  = [5]\n\n\n')
        else:
          f.write('order  = [5,5]\n\n\n')  
    
        if includesphase and flagphases:
            f.write('[flagphase]\n')
            f.write('soltab = [sol000/phase000]\n')
            f.write('operation = FLAG\n')
            if onechannel:
              f.write('axesToFlag = [time]\n')
            else:
              f.write('axesToFlag = [time,freq]\n')
            f.write('mode = smooth\n')
            f.write('maxCycles = 3\n')
            f.write('windowNoise = 7\n')
            f.write('maxRms = %s\n' % str(maxrmsphase))
            if onechannel:
              f.write('order  = [5]\n\n\n')
            else:
              f.write('order  = [5,5]\n\n\n')  

        f.write('[plotampafter]\n')
        f.write('operation = PLOT\n')
        f.write('soltab = [sol000/amplitude000]\n')
        if onechannel:
          f.write('axesInPlot = [time]\n')
          if not onepol:
            f.write('axisInCol = pol\n')
        else:
          f.write('axesInPlot = [time,freq]\n')   
        f.write('axisInTable = ant\n')
        # f.write('minmax = [0,2.5]\n')
        f.write('minmax = [%s,%s]\n' % (str(medamp/4.0), str(medamp*2.5)))
        f.write('prefix = plotlosoto%s/%sampfl\n\n\n' % (ms,outplotname))

        if includesphase and flagphases:
            f.write('[plotphase_after]\n')
            f.write('operation = PLOT\n')
            f.write('soltab = [sol000/phase000]\n')
            if onechannel:
              f.write('axesInPlot = [time]\n')
              if not onepol:
                f.write('axisInCol = pol\n')
            else:
              f.write('axesInPlot = [time,freq]\n')   
            f.write('axisInTable = ant\n')
            f.write('minmax = [-3.14,3.14]\n')
            f.write('prefix = plotlosoto%s/%sphasefl\n' % (ms,outplotname))
            f.write('refAnt = %s\n' % refant)
  
  
    f.close()
    return parset

def create_losoto_mediumsmoothparset(ms, boxsize, longbaseline, includesphase=True, refant='CS003HBA0',\
                                     onechannel=False, outplotname='runningmedian'):
    parset= 'losoto_mediansmooth.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

    f.write('pol = []\n')
    f.write('soltab = [sol000/*]\n')
    f.write('Ncpu = 0\n\n\n')

    if includesphase:
        f.write('[smoothphase]\n')
        f.write('soltab = [sol000/phase000]\n')
        f.write('operation= SMOOTH\n')
        if onechannel:
          f.write('axesToSmooth = [time]\n')
          f.write('size = [%s]\n' % (boxsize, boxsize))
        else:
          f.write('axesToSmooth = [freq,time]\n')  
          f.write('size = [%s,%s]\n' % (boxsize, boxsize))
        f.write('mode = runningmedian\n\n\n')

    f.write('[smoothamp]\n')
    f.write('soltab = [sol000/amplitude000]\n')
    f.write('operation= SMOOTH\n')
    if onechannel:
      f.write('axesToSmooth = [time]\n')
      f.write('size = [%s]\n' % (boxsize, boxsize))
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
    f.write('prefix = plotlosoto%s/amps_smoothed\n\n\n' % ms)

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
        f.write('prefix = plotlosoto%s/phases_smoothed\n\n\n' % ms)
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
        f.write('prefix = plotlosoto%s/phases_smoothed1rad\n' % ms)
        f.write('refAnt = %s\n' % refant)

    f.close()
    return parset

def check_phaseup(H5name):
    H5 =  tables.open_file(H5name, mode='r')
    try:
      ants   = H5.root.sol000.phase000.ant[:]
    except:
      pass

    try:
      ants   = H5.root.sol000.amplitude000.ant[:]
    except:
      pass
  
    try:
      ants   = H5.root.sol000.rotation000.ant[:]
    except:
      pass
  
    try:
      ants   = H5.root.sol000.tec000.ant[:]
    except:
      pass
  
    # H5 = h5parm.h5parm(H5name, readonly=False)
    # ants = H5.getSolset('sol000').getAnt().keys()
    H5.close()
    if 'ST001' in ants:
        return True
    else:
        return False

def fixbeam_ST001(H5name):
    
   H5 = h5parm.h5parm(H5name, readonly=False)
   
   ants = H5.getSolset('sol000').getAnt().keys()
   antsrs = fnmatch.filter(ants,'RS*')
   ST001 = False
   
   if 'ST001' in ants:
     ST001 = True
     amps    = H5.getSolset('sol000').getSoltab('amplitude000').getValues()
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

     ampvals[:,:, idx, 0,:] = ampvals[:,:, idxrs, 0,:]
     phasevals[:,:, idx, 0,:] = 0.0 
     
     H5.getSolset('sol000').getSoltab('amplitude000').setValues(ampvals)
     H5.getSolset('sol000').getSoltab('phase000').setValues(phasevals)
   
   H5.close()
    
   return ST001

def circular(ms, linear=False, dysco=True):
    """
    convert to circular correlations
    """
    taql = 'taql'
    scriptn = 'python lin2circ.py'
    if linear:
      cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --lincol=CORRECTED_DATA --back' 
    else:
      cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --outcol=CORRECTED_DATA' 
    if not dysco:
      cmdlin2circ += ' --nodysco'
    print(cmdlin2circ)
    run(cmdlin2circ)
    run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
    return


def beamcor_and_lin2circ(ms, dysco=True, beam=True, lin2circ=False, circ2lin=False, losotobeamlib='stationresponse'):
    """
    correct a ms for the beam in the phase center (array_factor only)
    """
    
    # check if there are applybeam corrections in the header
    # should be there unless a very old DP3 version has been used
    usedppp = beamkeywords(ms)

    losoto = 'losoto'
    taql = 'taql'
    H5name = create_beamcortemplate(ms)

    if lin2circ and circ2lin:
       print('Wrong input in function, both lin2circ and circ2lin are True')
       sys.exit()

    losotolofarbeam(H5name, 'phase000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True, beamlib=losotobeamlib)
    losotolofarbeam(H5name, 'amplitude000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True, beamlib=losotobeamlib)


    phasedup = fixbeam_ST001(H5name)
    parset = create_losoto_beamcorparset(ms, refant=findrefant_core(H5name))
    force_close(H5name)

    if usedppp and not phasedup :
        cmddppp = 'DP3 numthreads='+str(multiprocessing.cpu_count())+ ' msin=' + ms + ' msin.datacolumn=DATA msout=. '
        cmddppp += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        cmddppp += 'msout.datacolumn=CORRECTED_DATA '
        if (lin2circ or circ2lin) and beam:
          cmddppp += 'steps=[beam,pystep] '
          cmddppp += 'beam.type=applybeam beam.updateweights=True ' # weights
          cmddppp += 'beam.direction=[] ' # correction for the current phase center
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
          cmddppp += 'beam.type=applybeam beam.updateweights=True ' # weights
          cmddppp += 'beam.direction=[] ' # correction for the current phase center
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
       
        print('DP3 applybeam/polconv:', cmddppp)
        run(cmddppp)
        run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
    else:
        # print('Phase up dataset, cannot use DPPP beam, do manual correction')
        cmdlosoto = losoto + ' ' + H5name + ' ' + parset
        print(cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)
    
        cmd = 'DP3 numthreads='+str(multiprocessing.cpu_count())+ ' msin=' + ms + ' msin.datacolumn=DATA msout=. '
        cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        cmd += 'msout.datacolumn=CORRECTED_DATA '

        if (lin2circ or circ2lin) and beam:
          cmd += 'steps=[ac1,ac2,pystep] '
          cmd += 'pystep.python.module=polconv '
          cmd += 'pystep.python.class=PolConv '
          cmd += 'pystep.type=PythonDPPP '
          if lin2circ:
             cmd += 'pystep.lin2circ=1 '
          if circ2lin: 
             cmd += 'pystep.circ2lin=1 '  
          
          cmd += 'ac1.parmdb='+H5name + ' ac2.parmdb='+H5name + ' '
          cmd += 'ac1.type=applycal ac2.type=applycal '
          cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 ac2.updateweights=True '   
        if beam and not (lin2circ or circ2lin):
          cmd += 'steps=[ac1,ac2] '
          cmd += 'ac1.parmdb='+H5name + ' ac2.parmdb='+H5name + ' '
          cmd += 'ac1.type=applycal ac2.type=applycal '
          cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 ac2.updateweights=True '           
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
        print('DP3 applycal/polconv:', cmd)
        run(cmd)
        run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
 
        # Add beam correction keyword here.
        # This code only applies the array factor and assumes the element beam was corrected already.
        # Valid values are Element, ArrayFactor or Full.
        if False:
           try:
              t = pt.table(ms, readonly=False)
              t.putcolkeywords('DATA', {'LOFAR_APPLIED_BEAM_MODE': 'Full'})
              t2 = pt.table(ms + '::FIELD')
              phasedir = t2.getcol('PHASE_DIR').squeeze()
              t2.close()
              beamdir = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_DIR')
              # Right ascension in radians is set in m0
              # Declination in radians is set in m1
              beamdir['m0']['value'] = phasedir[0]
              beamdir['m1']['value'] = phasedir[1]
              t.putcolkeywords('DATA', {'LOFAR_APPLIED_BEAM_DIR': beamdir})
              t.close()
           except:
              print('Warning could not update LOFAR BEAM keywords in ms, it seems this data was preprocessed with a very old DP3 version')  
    return


def beamkeywords(ms):
    t = pt.table(ms, readonly=True)
    applybeam_info = False
    try:
       beammode = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_MODE')
       applybeam_info = True
       print('DP3 applybeam was used')
    except:
       print('No applybeam beam keywords were found, very old DP3 version was used in prefactor?')
    t.close()
    return applybeam_info

def beamcormodel(ms, dysco=True):
    """
    create MODEL_DATA_BEAMCOR where we store beam corrupted model data
    """   
    H5name = ms + '_templatejones.h5'   
    
    cmd = 'DP3 numthreads='+str(multiprocessing.cpu_count())+' msin=' + ms + ' msin.datacolumn=MODEL_DATA msout=. '
    cmd += 'msout.datacolumn=MODEL_DATA_BEAMCOR steps=[ac1,ac2] '
    if dysco:
      cmd +=  'msout.storagemanager=dysco '    
    cmd += 'ac1.parmdb='+H5name + ' ac2.parmdb='+H5name + ' '
    cmd += 'ac1.type=applycal ac2.type=applycal '
    cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 ac2.updateweights=False '
    cmd += 'ac1.invert=False ac2.invert=False ' # Here we corrupt with the beam !
    print('DP3 applycal:', cmd)
    run(cmd)
   
    return

def findrms(mIn,maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
    """
    m=mIn[np.abs(mIn)>maskSup]
    rmsold=np.std(m)
    diff=1e-1
    cut=3.
    bins=np.arange(np.min(m),np.max(m),(np.max(m)-np.min(m))/30.)
    med=np.median(m)
    for i in range(10):
        ind=np.where(np.abs(m-med)<rmsold*cut)[0]
        rms=np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: break
        rmsold=rms
    return rms


def findamplitudenoise(parmdb):
      """
      find the 'amplitude noise' in a parmdb, return non-clipped rms value
      """
      H5 = h5parm.h5parm(parmdb, readonly=True) 
      amps =H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
      weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
      H5.close()

      idx = np.where(weights != 0.0)
      
      amps = amps[idx]
      amps = amps[np.isfinite(amps)]
      amps = np.log10(np.ndarray.flatten(amps))
      
      
      noise = findrms(amps)
      
      logger.info('Noise and clipped noise' + str(parmdb) + ' ' + str(np.std(amps)) + ' ' + str(noise))

      # do not return clipped noise, we are intersted in finding data with high outliers
      return np.std(amps)


def getimsize(boxfile, cellsize=1.5, increasefactor=1.2):
   """
   find imsize need to image a DS9 boxfile region
   """
   r = pyregion.open(boxfile)
   
   xs = np.ceil((r[0].coord_list[2])*increasefactor*3600./cellsize)
   ys = np.ceil((r[0].coord_list[3])*increasefactor*3600./cellsize)

   imsize = np.ceil(xs) # // Round up decimals to an integer
   if(imsize % 2 == 1): 
       imsize = imsize + 1
   
   # if np.int(imsize) < 512:
   #    imsize = 512
   return np.int(imsize)


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
    '''
    Changes the reference antenna, if needed, for phase
    '''
    H5     = h5parm.h5parm(parmdb, readonly=False) 
    phases = H5.getSolset('sol000').getSoltab(soltab).getValues()[0]
    weights= H5.getSolset('sol000').getSoltab(soltab).getValues(weight=True)[0]
    axesnames = H5.getSolset('sol000').getSoltab(soltab).getAxesNames() 
    print('axesname', axesnames)
    # print 'SHAPE', np.shape(weights)#, np.size(weights[:,:,0,:,:])

    
    antennas = list(H5.getSolset('sol000').getSoltab(soltab).getValues()[1]['ant'])
    # print antennas
    
    if 'pol' in axesnames:
      idx0    = np.where((weights[:,:,0,:,:] == 0.0))[0]
      idxnan  = np.where((~np.isfinite(phases[:,:,0,:,:])))[0]
    
      refant = ' '
      tmpvar = np.float(np.size(weights[:,:,0,:,:]))
    else:
      idx0    = np.where((weights[:,0,:,:] == 0.0))[0]
      idxnan  = np.where((~np.isfinite(phases[:,0,:,:])))[0]
    
      refant = ' '
      tmpvar = np.float(np.size(weights[:,0,:,:]))   
    
    if ((np.float(len(idx0))/tmpvar) > 0.5) or ((np.float(len(idxnan))/tmpvar) > 0.5):
      logger.info('Trying to changing reference anntena')
    

      for antennaid,antenna in enumerate(antennas[1::]):
            print(antenna)
            if 'pol' in axesnames:
              idx0    = np.where((weights[:,:,antennaid+1,:,:] == 0.0))[0]
              idxnan  = np.where((~np.isfinite(phases[:,:,antennaid+1,:,:])))[0]
              tmpvar = np.float(np.size(weights[:,:,antennaid+1,:,:]))
            else:
              idx0    = np.where((weights[:,antennaid+1,:,:] == 0.0))[0]
              idxnan  = np.where((~np.isfinite(phases[:,antennaid+1,:,:])))[0]
              tmpvar = np.float(np.size(weights[:,antennaid+1,:,:]))
            
            print(idx0, idxnan, ((np.float(len(idx0))/tmpvar)))
            if  ((np.float(len(idx0))/tmpvar) < 0.5) and ((np.float(len(idxnan))/tmpvar) < 0.5):
              logger.info('Found new reference anntena,' + str(antenna))
              refant = antenna
              break
    
    
    if refant != ' ':
        for antennaid,antenna in enumerate(antennas):
            if 'pol' in axesnames:
              phases[:,:,antennaid,:,:] = phases[:,:,antennaid,:,:] - phases[:,:,antennas.index(refant),:,:]
            else:
              # phases[:,antennaid,:,:] = phases[:,antennaid,:,:] - phases[:,antennas.index(refant),:,:]
              phases[:,:,antennaid,:] = phases[:,:,antennaid,:] - phases[:,:,antennas.index(refant),:]   
        H5.getSolset('sol000').getSoltab(soltab).setValues(phases)     

    H5.close()
    return


def calculate_solintnchan(compactflux):
    
    if compactflux >= 3.5:
        nchan = 5.
        solint_phase = 1.
        
    if compactflux <= 3.5:
        nchan = 5.
        solint_phase = 1.
  
    if compactflux <= 1.0:
        nchan= 10.
        solint_phase = 2
 
    if compactflux <= 0.75:
        nchan= 15.
        solint_phase = 3.

 
    # solint_ap = 100. / np.sqrt(compactflux)
    solint_ap = 120. /(compactflux**(1./3.)) # do third power-scaling
    # print solint_ap
    if solint_ap < 60.:
        solint_ap = 60.  # shortest solint_ap allowed
    if solint_ap > 180.:
        solint_ap = 180.  # longest solint_ap allowed
 
    if compactflux <= 0.4:
        nchan= 15.
        solint_ap = 180.
 
    return np.int(nchan), np.int(solint_phase), np.int(solint_ap)




def determine_compactsource_flux(fitsimage):
    
    hdul = fits.open(fitsimage)
    bmaj = hdul[0].header['BMAJ']
    bmin = hdul[0].header['BMIN']
    avgbeam = 3600.*0.5*(bmaj + bmin)
    pixsize = 3600.*(hdul[0].header['CDELT2'])
    rmsbox1 = np.int(7.*avgbeam/pixsize)
    rmsbox2 = np.int((rmsbox1/10.) + 1.)
    
    img = bdsf.process_image(fitsimage,mean_map='zero', rms_map=True, rms_box = (rmsbox1,rmsbox2))
    total_flux_gaus = np.copy(img.total_flux_gaus)
    hdul.close()
    # trying to reset.....
    del img
    
    return total_flux_gaus


def getdeclinationms(ms):
    '''
    return approximate declination of pointing center of the ms
    input: a ms
    output: declination in degrees
    '''
    t = pt.table(ms +'/FIELD', readonly=True)
    direction = np.squeeze ( t.getcol('PHASE_DIR') )
    t.close()
    return 360.*direction[1]/(2.*np.pi)

# print getdeclinationms('1E216.dysco.sub.shift.avg.weights.set0.ms')
# sys.exit()

def declination_sensivity_factor(declination):
    '''
    compute sensitivy factor lofar data, reduced by delclination, eq. from G. Heald.
    input declination is units of degrees
    '''
    factor = 1./(np.cos(2.*np.pi*(declination - 52.9)/360.)**2)

    return factor



def flaglowamps(parmdb, lowampval=0.1, flagging=True, setweightsphases=True):
    '''
    flag bad amplitudes in H5 parmdb, those with values < lowampval
    assume pol-axis is present (can handle length, 1 (scalar), 2 (diagonal), or 4 (fulljones))
    '''
    H5 = h5parm.h5parm(parmdb, readonly=False) 
    amps =H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    amps_xx = amps[...,0]
    amps_yy = amps[...,-1] # so this also works for pol axis length 1
    weights_xx = weights[...,0]
    weights_yy =weights[...,-1]
    idx_xx = np.where(amps_xx < lowampval)
    idx_yy = np.where(amps_yy < lowampval)
    
    if flagging: # no flagging
      weights_xx[idx_xx] = 0.0
      weights_yy[idx_yy] = 0.0
      print('Settting some weights to zero in flaglowamps')
    
    amps_xx[idx_xx] = 1.0
    amps_yy[idx_yy] = 1.0

    weights[...,0] = weights_xx
    weights[...,-1] = weights_yy
    amps[...,0] = amps_xx
    amps[...,-1] = amps_yy

    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights,weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        phases_xx = phases[...,0]
        phases_yy = phases[...,-1]
        weights_p_xx = weights_p[...,0]
        weights_p_yy = weights_p[...,-1]
        
        if flagging: # no flagging
            weights_p_xx[idx_xx] = 0.0
            weights_p_yy[idx_yy] = 0.0
            phases_xx[idx_xx] = 0.0
            phases_yy[idx_yy] = 0.0
            
            weights_p[...,0] = weights_p_xx
            weights_p[...,-1] = weights_p_yy
            phases[...,0] = phases_xx
            phases[...,-1] = phases_yy
        
            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p,weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)
    H5.close()
    return

def flaghighamps(parmdb, highampval=10.,flagging=True, setweightsphases=True):
    '''
    flag bad amplitudes in H5 parmdb, those with values > highampval
    '''
    H5 = h5parm.h5parm(parmdb, readonly=False) 
    amps =H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    idx = np.where(amps > highampval)
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
    
    
    if flagging: 
      weights[idx] = 0.0
      print('Settting some weights to zero in flaghighamps')
    amps[idx] = 1.0
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights,weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases =H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        if flagging: 
            weights_p[idx] = 0.0
            phases[idx] = 0.0
            # print(idx)
            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p,weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)
    
    # H5.getSolset('sol000').getSoltab('phase000').flush()
    # H5.getSolset('sol000').getSoltab('amplitude000').flush()
    H5.close()
    return

def flagbadamps(parmdb, setweightsphases=True):
    '''
    flag bad amplitudes in H5 parmdb, those with amplitude==1.0
    '''
    H5 = h5parm.h5parm(parmdb, readonly=False) 
    amps =H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    idx = np.where(amps <= 0.0)
    amps[idx] = 1.0
    
    idx = np.where(amps == 1.0)
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
    
    weights[idx] = 0.0
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights,weight=True)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases =H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        weights_p[idx] = 0.0
        phases[idx] = 0.0

        H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p,weight=True)
        H5.getSolset('sol000').getSoltab('phase000').setValues(phases)
    
    H5.close()
    return


def medianamp(h5):
    # assume pol-axis is present (can handle length, 1 (scalar), 2 (diagonal), or 4 (fulljones))
    H=tables.open_file(h5) 
    amplitude = H.root.sol000.amplitude000.val[:]
    weights   = H.root.sol000.amplitude000.weight[:]
    if amplitude.shape[-1] == 4:
      fulljones = True
    else:
      fulljones = False
    H.close()
    
    print('Amplitude and Weights shape:', weights.shape, amplitude.shape)
    amps_xx = amplitude[...,0]
    amps_yy = amplitude[...,-1] # so this also works for pol axis length 1
    weights_xx = weights[...,0]
    weights_yy = weights[...,-1]

    idx_xx = np.where(weights_xx != 0.0)    
    idx_yy = np.where(weights_yy != 0.0)
    
    medamps = 0.5*(10**(np.nanmedian(np.log10(amps_xx[idx_xx]))) + 10**(np.nanmedian(np.log10(amps_yy[idx_yy]))))
    print('Median  Stokes I amplitude of ', h5, ':', medamps)
    
    if fulljones:
       amps_xy = amplitude[...,1]
       amps_yx = amplitude[...,2] 
       weights_xy = weights[...,1]
       weights_yx = weights[...,2]
       idx_xy = np.where(weights_xy != 0.0)    
       idx_yx = np.where(weights_yx != 0.0)
       medamps_cross = 0.5*(10**(np.nanmedian(np.log10(amps_xy[idx_xy]))) + 10**(np.nanmedian(np.log10(amps_yx[idx_yx]))))
       print('Median amplitude of XY+YX ', h5, ':', medamps_cross)

    logger.info('Median Stokes I amplitude of ' + h5 + ': ' + str(medamps))
    return medamps

def normamplitudes(parmdb):
    '''
    normalize amplitude solutions to one
    '''
    
    if len(parmdb) == 1:
      H5 = h5parm.h5parm(parmdb[0], readonly=False) 
      amps =H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
      weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
      idx = np.where(weights != 0.0)
    
      amps = np.log10(amps)
      logger.info('Mean amplitudes before normalization: ' + str(10**(np.nanmean(amps[idx]))))
      amps = amps - (np.nanmean(amps[idx]))
      logger.info('Mean amplitudes after normalization: ' + str(10**(np.nanmean(amps[idx]))))
      amps = 10**(amps)

      H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps) 
      H5.close()

    else:
      # amps = []  
      for i, parmdbi in enumerate(parmdb):
          H5 = h5parm.h5parm(parmdbi, readonly=True) 
          ampsi = np.copy(H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0])
          weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
          idx = np.where(weights != 0.0)
          logger.info(parmdbi + '  Normfactor: '+ str(10**(np.nanmean(np.log10(ampsi[idx])))))
          if i == 0:
            amps = np.ndarray.flatten(ampsi[idx])
          else:
            amps = np.concatenate((amps, np.ndarray.flatten(ampsi[idx])),axis=0)

          # print np.shape(amps), parmdbi
          H5.close()
      normmin = (np.nanmean(np.log10(amps))) 
      logger.info('Global normfactor: ' + str(10**normmin))
      # now write the new H5 files
      for parmdbi in parmdb:  
         H5   = h5parm.h5parm(parmdbi, readonly=False) 
         ampsi = np.copy(H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0])
         ampsi = (np.log10(ampsi)) - normmin
         ampsi = 10**ampsi
         H5.getSolset('sol000').getSoltab('amplitude000').setValues(ampsi) 
         H5.close()
    return


def checkforzerocleancomponents(imagenames):
    '''
    check if something was cleaned, if not stop de script to avoid more obscure errors later
    '''
    for image_id, image in enumerate(imagenames):
        print('Check if there are non-zero pixels: ', image)
        hdul = fits.open(image)
        data = hdul[0].data
        if not np.any(data): # this checks if all elements are 0.0
          print('Model image:', image, 'contains only zeros. Stopping the selfcal')
          hdul.close()
          logger.error('Model image: ' + image + ' contains only zeros. Stopping the selfcal')
          sys.exit()
        hdul.close()
    return


def removenegativefrommodel(imagenames):
    '''
    replace negative pixel values in WSCLEAN model images with zeros
    '''
    perseus = False
    A1795   = False
    A1795imlist = sorted(glob.glob('/net/nieuwerijn/data2/rtimmerman/A1795_HBA/A1795/selfcal/selfcal_pix0.15_wide-????-model.fits'))
    
    for image_id, image in enumerate(imagenames):
        print('remove negatives from model: ', image)
        hdul = fits.open(image)
        data = hdul[0].data
        
        data[np.where(data < 0.0)] = 0.0
        hdul[0].data = data
        hdul.writeto(image, overwrite=True)
        hdul.close()
    
        if perseus:
          run('python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/editmodel.py {} /net/ouderijn/data2/rvweeren/PerseusHBA/inner_ring_j2000.reg /net/ouderijn/data2/rvweeren/PerseusHBA/outer_ring_j2000.reg'.format(image))
         # run('python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/editmodel.py ' + image)
        if A1795: 
          cmdA1795 = 'python /net/rijn/data2/rvweeren/LoTSS_ClusterCAL/insert_highres.py '
          cmdA1795 +=  image + ' '
          cmdA1795 +=  A1795imlist[image_id] + ' '
          cmdA1795 += '/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/A1795core.reg '
          print(cmdA1795)
          run(cmdA1795)
    
    return

def makeimage(mslist, imageout, pixsize, imsize, channelsout, niter, robust, \
              uvtaper=None, multiscale=False, predict=True, onlypredict=False, fitsmask=None, \
              idg=False, deepmultiscale=False, uvminim=80, fitspectralpol=True, \
              fitspectralpolorder=3, imager='WSCLEAN', restoringbeam=15, automask=2.5, \
              removenegativecc=True, usewgridder=False, paralleldeconvolution=0, \
              deconvolutionchannels=0, parallelgridding=1, multiscalescalebias=0.8, \
              fullpol=False, taperinnertukey=None):
    fitspectrallogpol = False # for testing Perseus
    msliststring = ' '.join(map(str, mslist))
    
    #  --- predict only when starting from external model images ---
    if onlypredict:
      if predict:
        cmd = 'wsclean -padding 1.8 -predict ' 
        if channelsout > 1:
          cmd += '-channels-out ' + str(channelsout) + ' '   
        if idg:
          cmd += '-use-idg -grid-with-beam -use-differential-lofar-beam -idg-mode cpu '
          cmd += '-beam-aterm-update 800 '
          cmd += '-pol iquv '
        else:
          if usewgridder:
            cmd +='-use-wgridder '  
          if parallelgridding > 1:
            cmd += '-parallel-gridding ' + str(parallelgridding) + ' ' 
        cmd += '-name ' + imageout + ' ' + msliststring
        print('PREDICT STEP: ', cmd)
        run(cmd)    
      return    
    #  --- end predict only ---
    
    os.system('rm -f ' + imageout + '-*.fits')
    imcol = 'CORRECTED_DATA'
    t = pt.table(mslist[0],readonly=True) # just test for first ms in mslist
    colnames =t.colnames()
    if 'CORRECTED_DATA' not in colnames: # for first imaging run
      imcol = 'DATA'
    t.close()
    # baselineav = str (1.5e3*60000.*2.*np.pi *np.float(pixsize)/(24.*60.*60*np.float(imsize)) )
    baselineav = str (1.5e3*60000.*2.*np.pi *1.5/(24.*60.*60*np.float(imsize)) )
   
    if imager == 'WSCLEAN':
      cmd = 'wsclean '
      cmd += '-no-update-model-required -minuv-l ' + str(uvminim) + ' '
      cmd += '-size ' + str(np.int(imsize)) + ' ' + str(np.int(imsize)) + ' -reorder '
      cmd += '-weight briggs ' + str(robust) + ' -clean-border 1 -parallel-reordering 4 '
      # -weighting-rank-filter 3 -fit-beam
      cmd += '-mgain 0.8 -data-column ' + imcol + ' -padding 1.4 '
      if channelsout > 1:
        cmd += ' -join-channels -channels-out ' + str(channelsout) + ' '
      if paralleldeconvolution > 0:
        cmd += '-parallel-deconvolution ' +  str(paralleldeconvolution) + ' '
      if parallelgridding > 1:
        cmd += '-parallel-gridding ' + str(parallelgridding) + ' '  
      if deconvolutionchannels > 0 and channelsout > 1:
        cmd += '-deconvolution-channels ' +  str(deconvolutionchannels) + ' '
      if automask > 0.5:
        cmd += '-auto-mask '+ str(automask)  + ' -auto-threshold 0.5 ' # to avoid automask 0

      
      if multiscale:
         # cmd += '-multiscale '+' -multiscale-scales 0,4,8,16,32,64 -multiscale-scale-bias 0.6 '
         # cmd += '-multiscale '+' -multiscale-scales 0,6,12,16,24,32,42,64,72,128,180,256,380,512,650 '
         cmd += '-multiscale '
         cmd += '-multiscale-scale-bias ' + str(multiscalescalebias) + ' '
         cmd += '-multiscale-max-scales ' + str(np.int(np.rint(np.log2(np.float(imsize)) -3))) + ' '
      if fitsmask != None:
        if os.path.isfile(fitsmask): 
          cmd += '-fits-mask '+ fitsmask + ' '
        else:
          print('fitsmask: ', fitsmask, 'does not exist')
          sys.exit(1)
      if uvtaper != None:
         cmd += '-taper-gaussian ' + uvtaper + ' '
      if taperinnertukey !=None:
         cmd += '-taper-inner-tukey ' + str(taperinnertukey) + ' '
      if idg:
        cmd += '-use-idg -grid-with-beam -use-differential-lofar-beam -idg-mode cpu '
        cmd += '-beam-aterm-update 800 '
        cmd += '-pol iquv -link-polarizations i '
      else:
        if fitspectralpol and channelsout > 1:
           if fitspectrallogpol: 
             cmd += '-fit-spectral-log-pol ' + str(fitspectralpolorder) + ' '   
           else:
             cmd += '-fit-spectral-pol ' + str(fitspectralpolorder) + ' '        
        if fullpol:
          cmd += '-pol iquv -join-polarizations '  
        else:
          cmd += '-pol i '
        cmdbtmp = '-baseline-averaging ' + baselineav + ' '
        cmd += '-baseline-averaging ' + baselineav + ' '
        if usewgridder:
          cmd +='-use-wgridder '  
          # cmd +='-wgridder-accuracy 1e-4 '
    
      cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec ' 
      print('WSCLEAN: ', cmd + '-nmiter 12 -niter ' + str(niter) + ' ' + msliststring)
      logger.info(cmd + ' -niter ' + str(niter) + ' ' + msliststring)
      run(cmd + '-nmiter 12 -niter ' + str(niter) + ' ' + msliststring)        
        

      if deepmultiscale:
        
        # predict first to fill MODEL_DATA so we can continue with clean
        cmdp = 'wsclean -size ' 
        cmdp += str(np.int(imsize)) + ' ' + str(np.int(imsize)) +  ' -padding 1.8 -predict ' 
        if channelsout > 1:
           cmdp += ' -channels-out ' + str(channelsout) + ' '
        if idg:
          cmdp += '-use-idg -grid-with-beam -use-differential-lofar-beam -idg-mode cpu '
          cmdp += '-beam-aterm-update 800 '
          cmdp += '-pol iquv '
        else:
          if usewgridder:    
            cmd +='-use-wgridder '  
            # cmd +='-wgridder-accuracy 1e-4 '
          
        cmdp += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec ' + msliststring
        print('PREDICT STEP for continue: ', cmdp)
        run(cmdp)
       
        # NOW continue cleaning
        if not multiscale: # if multiscale is true then this is already set above
          # cmd += '-multiscale '+' -multiscale-scales 0,4,8,16,32,64 '
          cmd += '-multiscale '
          cmd += '-multiscale-scale-bias ' + str(multiscalescalebias) + ' '
          cmd += '-multiscale-max-scales ' + str(np.int(np.rint(np.log2(np.float(imsize)) -3))) + ' '
        cmd += '-niter ' + str(np.int(niter/5)) + ' -continue ' + msliststring

        # Remove baselinedepedent averaging because of -continue from MODEL_DATA  
        if not idg:
          cmd = cmd.replace(cmdbtmp,'')
        print('WSCLEAN continue: ', cmd)
        run(cmd)

      # REMOVE nagetive model components, these are artifacts (only for Stokes I)
      if removenegativecc:
        if idg:
            removenegativefrommodel(sorted(glob.glob(imageout +'-????-I-model*.fits')))  # only Stokes I
        else:    
            removenegativefrommodel(sorted(glob.glob(imageout + '-????-model.fits')))

      # Check is anything was cleaned. If not, stop the selfcal to avoid obscure errors later 
      if idg:
        checkforzerocleancomponents(glob.glob(imageout +'-????-I-model*.fits'))  # only Stokes I
      else:    
        checkforzerocleancomponents(glob.glob(imageout + '-????-model.fits'))

      if predict:
        cmd = 'wsclean -size ' 
        cmd += str(np.int(imsize)) + ' ' + str(np.int(imsize)) +  ' -padding 1.8 -predict ' 
        if channelsout > 1:
          cmd += ' -channels-out ' + str(channelsout) + ' '  
        if idg:
          cmd += '-use-idg -grid-with-beam -use-differential-lofar-beam -idg-mode cpu '
          cmd += '-beam-aterm-update 800 '
          cmd += '-pol iquv '
        else:
          if usewgridder:
            cmd +='-use-wgridder '  
            # cmd +='-wgridder-accuracy 1e-4 '    
          if parallelgridding > 1:
            cmd += '-parallel-gridding ' + str(parallelgridding) + ' ' 

      
        cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec ' + msliststring
        print('PREDICT STEP: ', cmd)
        run(cmd)
        
        
    if imager == 'DDFACET':
        makemslist(mslist)
        # restoringbeam = '15'
        cmd = 'DDF.py --Data-MS=mslist.txt --Deconv-PeakFactor=0.001 --Data-ColName=' + imcol + ' ' + \
              '--Parallel-NCPU=32 --Output-Mode=Clean --Deconv-CycleFactor=0 ' + \
              '--Deconv-MaxMinorIter=' + str(niter) + ' --Deconv-MaxMajorIter=5 ' + \
              '--Deconv-Mode=SSD --Weight-Robust=' + str(robust) + ' --Image-NPix=' + str(np.int(imsize)) + ' ' + \
              '--CF-wmax=50000 --CF-Nw=100 --Beam-Model=None --Beam-LOFARBeamMode=A --Beam-NBand=1 ' + \
              '--Output-Also=onNeds --Image-Cell=' + str(pixsize) + ' --Facets-NFacets=1 --Freq-NDegridBand=1 ' + \
              '--Deconv-RMSFactor=3.0 --Deconv-FluxThreshold=0.0 --Data-Sort=1 --Cache-Dir=. --Freq-NBand=2 ' + \
              '--GAClean-MinSizeInit=10 --Facets-DiamMax=1.5 --Facets-DiamMin=0.1 ' + \
              '--Cache-Dirty=auto --Weight-ColName=WEIGHT_SPECTRUM --Output-Name=' + imageout + ' ' + \
              '--Comp-BDAMode=1 --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --Cache-Reset=1 ' + \
              '--RIME-ForwardMode=BDA-degrid --Predict-ColName=MODEL_DATA --Selection-UVRange=[0.1,2000.] ' + \
              '--Output-RestoringBeam=' + str(restoringbeam) + ' --Mask-SigTh=5.0 '
        if fitsmask != None:
           cmd += '--Mask-External=' + fitsmask + ' --Mask-Auto=0 ' 
        else:
           cmd += '--Mask-Auto=1 '
        
        print(cmd)
        run(cmd)


def calibrateandapplycal(mslist, selfcalcycle, args, solint_list, nchan_list, \
              soltype_list, soltypecycles_list, smoothnessconstraint_list, \
              smoothnessreffrequency_list, smoothnessspectralexponent_list, smoothnessrefdistance_list, \
              antennaconstraint_list, resetsols_list, uvmin=0, normamps=False, skymodel=None, \
              predictskywithbeam=False, restoreflags=False, flagging=False, \
              longbaseline=False, BLsmooth=False, flagslowphases=True, \
              flagslowamprms=7.0, flagslowphaserms=7.0, skymodelsource=None, \
              skymodelpointsource=None, wscleanskymodel=None, iontimefactor=0.01, \
              ionfreqfactor=1.0, blscalefactor=1.0, dejumpFR=False, uvminscalarphasediff=0, \
              docircular=False, mslist_beforephaseup=None, dysco=True, blsmooth_chunking_size=8):

   soltypecycles_list_array = np.array(soltypecycles_list) # needed to slice (slicing does not work in nested l
   incol = [] # len(mslist)
   pertubation = [] # len(mslist)
   for ms in mslist:
     incol.append('DATA') # start here, will be updated at applycal step for next solve if needed
     pertubation.append(False) 
   
   parmdbmergelist =  [[] for x in range(len(mslist))]   #  [[],[],[],[]] nested list length mslist used for Jurjen's h5_merge
   # LOOP OVER THE ENTIRE SOLTYPE LIST (so includes pertubations via a pre-applycal)
   for soltypenumber, soltype in enumerate(soltype_list):
     # SOLVE LOOP OVER MS
     parmdbmslist = []
     for msnumber, ms in enumerate(mslist):
       # check we are above far enough in the selfcal to solve for the extra pertubation
       if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]: 
         print('selfcalcycle, soltypenumber',selfcalcycle, soltypenumber)
         if (soltypenumber < len(soltype_list)-1):
             
           print(selfcalcycle,soltypecycles_list[soltypenumber+1][msnumber])
           print('Array soltypecycles_list ahead',soltypecycles_list_array[soltypenumber+1:len(soltypecycles_list_array[:,0]),msnumber])
           # if (selfcalcycle >= soltypecycles_list[soltypenumber+1][msnumber]): # this looks one soltpype ahead...hmmm, not good 
           if selfcalcycle >= np.min(soltypecycles_list_array[soltypenumber+1:len(soltypecycles_list_array[:,0]),msnumber]): # this looks all soltype ahead   
             pertubation[msnumber] = True
           else:
             pertubation[msnumber] = False   
         else:
           pertubation[msnumber] = False     
             
         if skymodel != None and selfcalcycle == 0:  
           parmdb = soltype + str(soltypenumber) + '_skyselfcalcyle' + str(selfcalcycle).zfill(3) + '_' + ms + '.h5'
         else:
           parmdb = soltype + str(soltypenumber) + '_selfcalcyle' + str(selfcalcycle).zfill(3) + '_' + ms + '.h5'
          
         runDPPPbase(ms, solint_list[soltypenumber][msnumber], nchan_list[soltypenumber][msnumber], parmdb, soltype, \
                     longbaseline=longbaseline, uvmin=uvmin, \
                     SMconstraint=smoothnessconstraint_list[soltypenumber][msnumber], \
                     SMconstraintreffreq=smoothnessreffrequency_list[soltypenumber][msnumber],\
                     SMconstraintspectralexponent=smoothnessspectralexponent_list[soltypenumber][msnumber],\
                     SMconstraintrefdistance=smoothnessrefdistance_list[soltypenumber][msnumber],\
                     antennaconstraint=antennaconstraint_list[soltypenumber][msnumber], \
                     resetsols=resetsols_list[soltypenumber][msnumber], \
                     restoreflags=restoreflags, maxiter=100, flagging=flagging, skymodel=skymodel, \
                     flagslowphases=flagslowphases, flagslowamprms=flagslowamprms, \
                     flagslowphaserms=flagslowphaserms, incol=incol[msnumber], \
                     predictskywithbeam=predictskywithbeam, BLsmooth=BLsmooth, skymodelsource=skymodelsource, \
                     skymodelpointsource=skymodelpointsource, wscleanskymodel=wscleanskymodel,\
                     iontimefactor=iontimefactor, ionfreqfactor=ionfreqfactor, blscalefactor=blscalefactor, dejumpFR=dejumpFR, uvminscalarphasediff=uvminscalarphasediff,\
                     selfcalcycle=selfcalcycle, dysco=dysco, blsmooth_chunking_size=blsmooth_chunking_size)

         parmdbmslist.append(parmdb)
         parmdbmergelist[msnumber].append(parmdb) # for h5_merge
       
     # NORMALIZE amplitudes
     if normamps and (soltype in ['complexgain','scalarcomplexgain','rotation+diagonal',\
                                  'amplitudeonly','scalaramplitude']) and len(parmdbmslist) > 0:
       print('Doing global gain normalization')
       normamplitudes(parmdbmslist) # list of h5 for different ms, all same soltype

     # APPLYCAL or PRE-APPLYCAL
     count = 0
     for msnumber, ms in enumerate(mslist):
       if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]: #
         print(pertubation[msnumber], parmdbmslist[count], msnumber, count)
         if pertubation[msnumber]: # so another solve follows after this
           if soltypenumber == 0:  
             applycal(ms, parmdbmslist[count], msincol='DATA',msoutcol='CORRECTED_PREAPPLY' + str(soltypenumber), dysco=dysco)
           else: 
             applycal(ms, parmdbmslist[count], msincol=incol[msnumber], msoutcol='CORRECTED_PREAPPLY' + str(soltypenumber), dysco=dysco) # msincol gets incol from previous solve 
           incol[msnumber] = 'CORRECTED_PREAPPLY' + str(soltypenumber) # SET NEW incol for next solve
         else: # so this is the last solve, no other pertubation
           if soltypenumber == 0:  
             applycal(ms, parmdbmslist[count], msincol='DATA',msoutcol='CORRECTED_DATA', dysco=dysco)
           else:
             applycal(ms, parmdbmslist[count], msincol=incol[msnumber], msoutcol='CORRECTED_DATA', dysco=dysco) # msincol gets incol from previous solve
         count = count + 1 # extra counter because parmdbmslist can have less length than mslist as soltypecycles_list goes per ms
   

   # merge all solutions
   print(parmdbmergelist)
   # try:
   if True:
     # import h5_merger
     for msnumber, ms in enumerate(mslist):
       if skymodel != None and selfcalcycle == 0: 
         parmdbmergename = 'merged_skyselfcalcyle' + str(selfcalcycle).zfill(3) + '_' + ms + '.h5'
         parmdbmergename_pc = 'merged_skyselfcalcyle' + str(selfcalcycle).zfill(3) + '_linearfulljones_' + ms + '.h5'
       else:
         parmdbmergename = 'merged_selfcalcyle' + str(selfcalcycle).zfill(3) + '_' + ms + '.h5'
         parmdbmergename_pc = 'merged_selfcalcyle' + str(selfcalcycle).zfill(3) + '_linearfulljones_' + ms + '.h5' 
       if os.path.isfile(parmdbmergename):
         os.system('rm -f ' + parmdbmergename)
       if os.path.isfile(parmdbmergename_pc):
         os.system('rm -f ' + parmdbmergename_pc)
       
       # add extra from preapplyH5_list
       if args['preapplyH5_list'][0] != None:
         preapplyh5parm = time_match_mstoH5(args['preapplyH5_list'], ms)  
         # replace the source direction coordinates so that the merge goes correctly
         # copy_over_sourcedirection_h5(parmdbmergelist[msnumber][0], preapplyh5parm)
         parmdbmergelist[msnumber].append(preapplyh5parm)
       
       print(parmdbmergename,parmdbmergelist[msnumber],ms)
       h5_merger.merge_h5(h5_out=parmdbmergename,h5_tables=parmdbmergelist[msnumber],ms_files=ms,\
                          convert_tec=True, merge_all_in_one=True, propagate_flags=True)
       # add CS stations back for superstation
       if mslist_beforephaseup != None:
         print('mslist_beforephaseup: ' + mslist_beforephaseup[msnumber])
         h5_merger.merge_h5(h5_out=parmdbmergename.replace("selfcalcyle",\
                            "addCS_selfcalcyle"),h5_tables=parmdbmergename, \
                            ms_files=mslist_beforephaseup[msnumber], convert_tec=True, merge_all_in_one=True, \
                            propagate_flags=True, add_cs=True)  
       
       # make LINEAR solutions from CIRCULAR 
       if ('scalarphasediff' in soltype_list) or ('scalarphasediffFR' in soltype_list) or docircular:
         h5_merger.merge_h5(h5_out=parmdbmergename_pc, h5_tables=parmdbmergename, circ2lin=True)
         # add CS stations back for superstation
         if mslist_beforephaseup != None:
           h5_merger.merge_h5(h5_out=parmdbmergename_pc.replace("selfcalcyle",\
                              "addCS_selfcalcyle"),h5_tables=parmdbmergename_pc, \
                              ms_files=mslist_beforephaseup[msnumber], convert_tec=True, merge_all_in_one=True, \
                              propagate_flags=True, add_cs=True)         

       
       if False:
         # testing only to check if merged H5 file is correct and makes a good image
         applycal(ms, parmdbmergename, msincol='DATA',msoutcol='CORRECTED_DATA', dysco=dysco)
       
       # plot merged solution file
       losotoparset = create_losoto_flag_apgridparset(ms, flagging=False, \
                            medamp=medianamp(parmdbmergename), \
                            outplotname=parmdbmergename.split('_' + ms + '.h5')[0], \
                            refant=findrefant_core(parmdbmergename))  
       run('losoto ' + parmdbmergename + ' ' + losotoparset)
       force_close(parmdbmergename)
   return 


def is_binary(file_name):
    ''' Check if a file contains text (and thus is a skymodel file, for example).
    Example from https://stackoverflow.com/questions/2472221/how-to-check-if-a-file-contains-plain-text
    Args:
        file_name (str): path to the file to determine the binary nature of.
    Returns:
        result (bool): returns whether the file is binary (True) or not (False).
    '''
    import magic
    f = magic.Magic(mime=True)
    mime = f.from_file(file_name)
    if 'text' in mime:
        return False
    else:
        return True


def predictsky_wscleanfits(ms, imagebasename, usewgridder=True):
    '''
    Predict the sky from model channels fits images (from a previous run, so frequencies need to overlap)
    '''
    channelsout = len(glob.glob(imagebasename + '-????-model.fits'))
    cmd = 'wsclean -channels-out '+ str(channelsout)+ ' -padding 1.8 -pol i ' 
    if usewgridder:
       cmd +='-use-wgridder '   
    cmd+= '-name ' + imagebasename + ' -predict ' + ms
    print(cmd)
    run(cmd)
    time.sleep(1)
    return
    

def predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=False, sources=None):
   
   if False:
   # if is_binary(skymodel):
      sourcedb = skymodel 
   else:
      # make sourcedb
      sourcedb = skymodel + 'sourcedb'
      if os.path.isfile(sourcedb):
         os.system('rm -rf ' + sourcedb)
      cmdmsdb = "makesourcedb in=" + skymodel + " "
      cmdmsdb += "out=" + sourcedb + " outtype='blob' format='<' append=False"
      print(cmdmsdb)
      run(cmdmsdb)         
   
   cmd = 'DP3 numthreads='+str(multiprocessing.cpu_count())+ ' msin=' + ms + ' msout=. ' 
   cmd += 'p.sourcedb=' + sourcedb + ' steps=[p] p.type=predict msout.datacolumn=' + modeldata + ' '
   if sources != None:
      cmd += 'p.sources=[' + str(sources) + '] '    
   if predictskywithbeam:
      cmd += 'p.usebeammodel=True p.usechannelfreq=True p.beammode=array_factor ' 
   print(cmd)
   run(cmd)
   return    

def runDPPPbase(ms, solint, nchan, parmdb, soltype, longbaseline=False, uvmin=0, \
                SMconstraint=0.0, SMconstraintreffreq=0.0, \
                SMconstraintspectralexponent=-1.0, SMconstraintrefdistance=0.0, antennaconstraint=None, \
                resetsols=None, restoreflags=False, solveralgorithm='directionsolve', \
                maxiter=100, flagging=False, skymodel=None, flagslowphases=True, \
                flagslowamprms=7.0, flagslowphaserms=7.0, incol='DATA', \
                predictskywithbeam=False, BLsmooth=False, skymodelsource=None, \
                skymodelpointsource=None, wscleanskymodel=None, iontimefactor=0.01, ionfreqfactor=1.0,\
                blscalefactor=1.0, dejumpFR=False, uvminscalarphasediff=0,selfcalcycle=0, dysco=True, blsmooth_chunking_size=8):
    
    soltypein = soltype # save the input soltype is as soltype could be modified (for example by scalarphasediff)
    
    
    modeldata = 'MODEL_DATA' # the default, update if needed for scalarphasediff and phmin solves
    if BLsmooth:
      print('python BLsmooth.py -n 8 -c '+ str(blsmooth_chunking_size) + ' -i '+ incol + ' -o SMOOTHED_DATA -f ' + str(iontimefactor) + \
                ' -s ' + str(blscalefactor) + ' -u ' + str(ionfreqfactor) + ' ' + ms)
      run('python BLsmooth.py -n 8 -c '+ str(blsmooth_chunking_size) + ' -i '+ incol + ' -o SMOOTHED_DATA -f ' + str(iontimefactor) + \
                ' -s ' + str(blscalefactor) + ' -u ' + str(ionfreqfactor) + ' ' + ms)
      incol = 'SMOOTHED_DATA'    

    if soltype == 'scalarphasediff' or soltype == 'scalarphasediffFR':
      create_phasediff_column(ms, incol=incol, dysco=dysco)
      soltype = 'phaseonly' # do this type of solve, maybe scalarphase is fine? 'scalarphase' #
      incol='DATA_CIRCULAR_PHASEDIFF'
      skymodel = None # solve out of MODEL_DATA complex(1,0)
      create_MODEL_DATA_PDIFF(ms)
      modeldata = 'MODEL_DATA_PDIFF'

    if skymodel !=None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR':
        predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam, sources=skymodelsource)

    if wscleanskymodel !=None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR':
        makeimage([ms], wscleanskymodel, 1., 1., len(glob.glob(wscleanskymodel + '-????-model.fits')), 0, 0.0, \
               onlypredict=True, idg=False, usewgridder=True)


    if skymodelpointsource !=None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR':
        # create MODEL_DATA (no dysco!)
        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]')
        # do the predict with taql
        run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource)+ "+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource)+ "+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'")
        run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'")
        

    if soltype in ['phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin','scalarphase_phmin']:
      create_phase_column(ms, incol=incol, outcol='DATA_PHASEONLY', dysco=dysco)
      create_phase_column(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASEONLY', dysco=dysco)
      soltype = soltype.split('_phmin')[0]
      incol = 'DATA_PHASEONLY'
      modeldata = 'MODEL_DATA_PHASEONLY'

    if soltype in ['phaseonly_slope', 'scalarphase_slope']:
      create_phase_slope(ms, incol=incol, outcol='DATA_PHASE_SLOPE', ampnorm=False, dysco=dysco)
      create_phase_slope(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASE_SLOPE', ampnorm=False, dysco=dysco)
      soltype = soltype.split('_slope')[0]
      incol = 'DATA_PHASE_SLOPE'
      modeldata = 'MODEL_DATA_PHASE_SLOPE'      

    if soltype in ['fulljones']:
      print('Setting XY and YX to 0+0i')  
      run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'")
      run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'")   
     
    if soltype in ['phaseonly','complexgain','fulljones','rotation+diagonal','amplitudeonly']: # for 1D plotting
      onepol = False
    if soltype in ['scalarphase','tecandphase','tec','scalaramplitude','scalarcomplexgain','rotation']:
      onepol = True

    if restoreflags:
      cmdtaql = "'update " + ms + " set FLAG=FLAG_BACKUP'"
      print("Restore flagging column: " + "taql " + cmdtaql)
      run("taql " + cmdtaql)  


    t    = pt.table(ms + '/SPECTRAL_WINDOW',ack=False)
    freq = np.median(t.getcol('CHAN_FREQ')[0])
    t.close()
    
    t          = pt.table(ms + '/ANTENNA',ack=False)
    antennasms = t.getcol('NAME')
    t.close()
    if freq > 100e6:
      HBAorLBA = 'HBA'
    else:
      HBAorLBA = 'LBA'   
    print('This is', HBAorLBA, 'data')
    print('This ms contains', antennasms)

    # determine if phases needs to be included, important if slowgains do not contain phase solutions    
    includesphase = True
    if soltype == 'scalaramplitude' or soltype == 'amplitudeonly':
      includesphase = False

    # figure out which weight_spectrum column to use
    t = pt.table(ms)
    if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
       weight_spectrum =  'WEIGHT_SPECTRUM_SOLVE'
    else:
       weight_spectrum =  'WEIGHT_SPECTRUM'
    t.close()   
    
    # check for previous old parmdb and remove them   
    if os.path.isfile(parmdb):
      print('H5 file exists  ', parmdb)
      os.system('rm -f ' + parmdb)

    cmd = 'DP3 numthreads='+str(multiprocessing.cpu_count())+ ' msin=' + ms + ' msin.datacolumn=' + incol + ' '
    cmd += 'msout=. ddecal.mode=' + soltype + ' '
    cmd += 'msin.weightcolumn='+weight_spectrum + ' '
    cmd += 'steps=[ddecal] ddecal.type=ddecal '
    if dysco:
      cmd += 'msout.storagemanager=dysco '    
    cmd += 'ddecal.solveralgorithm=' + solveralgorithm + ' '
    cmd += 'ddecal.maxiter='+str(np.int(maxiter)) + ' ddecal.propagatesolutions=True '
    cmd += 'ddecal.usemodelcolumn=True '
    cmd += 'msin.modelcolumn=' + modeldata + ' '  
    cmd += 'ddecal.solint=' + format_solint(solint, ms) + ' '
    cmd += 'ddecal.nchan=' + str(nchan) + ' '
    cmd += 'ddecal.h5parm=' + parmdb + ' '
    
    # SET UVMIN
    if soltypein == 'scalarphasediff' or soltypein == 'scalarphasediffFR':
       if uvminscalarphasediff != None:
         cmd += 'ddecal.uvlambdamin=' + str(uvminscalarphasediff) + ' '
       else:  
         if uvmin != 0:
           cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '     
    else:
       if uvmin != 0:
         cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '        

    if antennaconstraint != None:
        cmd += 'ddecal.antennaconstraint=' + antennaconstraintstr(antennaconstraint, antennasms, HBAorLBA) + ' '
    if SMconstraint > 0.0 and nchan != 0:
        cmd += 'ddecal.smoothnessconstraint=' + str(SMconstraint*1e6) + ' ' 
        cmd += 'ddecal.smoothnessreffrequency=' + str(SMconstraintreffreq*1e6) + ' '
        cmd += 'ddecal.smoothnessspectralexponent=' + str(SMconstraintspectralexponent) + ' '
        cmd += 'ddecal.smoothnessrefdistance=' + str(SMconstraintrefdistance*1e3) + ' ' # input units in km
        
    if soltype in ['phaseonly','scalarphase','tecandphase','tec','rotation']:
       cmd += 'ddecal.tolerance=1.e-4 '
       if soltype in ['tecandphase','tec']:             
          cmd += 'ddecal.approximatetec=True '
          cmd += 'ddecal.stepsize=0.2 '
          cmd += 'ddecal.maxapproxiter=45 '
          cmd += 'ddecal.approxtolerance=6e-3 '
    if soltype in ['complexgain','scalarcomplexgain','scalaramplitude','amplitudeonly','rotation+diagonal','fulljones']:   
       cmd += 'ddecal.tolerance=1.e-4 ' # for now the same as phase soltypes
 

    print('DP3 solve:', cmd)
    logger.info('DP3 solve: ' + cmd)    
    if selfcalcycle > 0 and (soltypein=="scalarphasediffFR" or soltypein=="scalarphasediff"):
        h5_tocopy = glob.glob("*_"+ms+".h5.scbackup")[0] # What if your ms nums somehow share a common base??
        print("COPYING PREVIOUS SCALARPHASEDIFF SOLUTION")
        os.system('cp -r ' + h5_tocopy + ' ' + parmdb)
    else:
        run(cmd)
    if selfcalcycle==0 and (soltypein=="scalarphasediffFR" or soltypein=="scalarphasediff"):
        os.system("cp -r " + parmdb + " " + parmdb + ".scbackup")

    
    if has0coordinates(parmdb):
       logger.warning('Direction coordinates are zero in: ' + parmdb)
    
    if np.int(maxiter) == 1: # this is a template solve only
      print('Template solve, not going to make plots or do solution flagging')
      return    

    outplotname = parmdb.split('_' + ms + '.h5')[0]

    if incol == 'DATA_CIRCULAR_PHASEDIFF':
      print('Manually updating H5 to get the phase difference correct')
      makephasediffh5(parmdb)
    if incol == 'DATA_CIRCULAR_PHASEDIFF' and soltypein == 'scalarphasediffFR':  
      print('Fiting for Faraday Rotation with losoto on the phase differences')
      # work with copies H5 because losoto changes the format splitting off the length 1 direction axis creating issues with H5merge (also add additional solution talbes which we do not want)
      os.system('cp -f ' + parmdb + ' ' + 'FRcopy' + parmdb) 
      losoto_parsetFR = create_losoto_FRparset(ms, refant=findrefant_core(parmdb), outplotname=outplotname,dejump=dejumpFR)
      run('losoto ' + 'FRcopy' + parmdb + ' ' + losoto_parsetFR)
      rotationmeasure_to_phase('FRcopy' + parmdb, parmdb, dejump=dejumpFR)
      run('losoto ' + parmdb + ' ' + create_losoto_FRparsetplotfit(ms, refant=findrefant_core(parmdb), outplotname=outplotname))
      force_close(parmdb)

      
    if incol == 'DATA_PHASE_SLOPE':
      print('Manually updating H5 to get the cumulative phase')
      makephaseCDFh5(parmdb)

    if resetsols != None:
      if soltype in ['phaseonly','scalarphase','tecandphase','tec','rotation','fulljones','complexgain','scalarcomplexgain']:
         refant=findrefant_core(parmdb)
         force_close(parmdb)
      else:
         refant = None 
      resetsolsforstations(parmdb, antennaconstraintstr(resetsols, antennasms, HBAorLBA, useforresetsols=True), refant=refant)

    if number_freqchan_h5(parmdb) > 1:
      onechannel = False
    else:
      onechannel = True  

    # Roation checking
    if soltype in ['rotation','rotation+diagonal']:
      removenans(parmdb, 'rotation000')

    # Check for bad values  
    if soltype in ['scalarcomplexgain','complexgain','amplitudeonly','scalaramplitude','fulljones','rotation+diagonal']:
      flagbadamps(parmdb, setweightsphases=includesphase)
      removenans(parmdb, 'amplitude000')
      medamp = medianamp(parmdb) # fu
      
      if soltype != 'amplitudeonly' and soltype != 'scalaramplitude':
         try:
           change_refant(parmdb,'phase000')
         except:
           pass
         removenans(parmdb, 'phase000')
      flaglowamps(parmdb, lowampval=medamp*0.1, flagging=flagging, setweightsphases=includesphase)
      flaghighamps(parmdb, highampval=medamp*10., flagging=flagging, setweightsphases=includesphase)
      
    # makes plots and do LOSOTO flagging      
    if soltype in ['rotation','rotation+diagonal']:

      losotoparset_rotation = create_losoto_rotationparset(ms, onechannel=onechannel, outplotname=outplotname, refant=findrefant_core(parmdb)) # phase matrix plot
      force_close(parmdb)
      cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_rotation
      print(cmdlosoto)
      logger.info(cmdlosoto)
      run(cmdlosoto)

    # print(findrefant_core(parmdb))
    # print(onechannel)
    # if len(tables.file._open_files.filenames) >= 1: # for debugging      
    #  print('Location 1 Some HDF5 files are not closed:', tables.file._open_files.filenames)
    #  sys.exit()        
    
    if soltype in ['phaseonly','scalarphase']:
      losotoparset_phase = create_losoto_fastphaseparset(ms, onechannel=onechannel, onepol=onepol, outplotname=outplotname, refant=findrefant_core(parmdb)) # phase matrix plot
      cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_phase
      force_close(parmdb)
      print(cmdlosoto)
      logger.info(cmdlosoto)
      run(cmdlosoto)
      # if len(tables.file._open_files.filenames) >= 1: # for debugging
      #  print('Location 1.5 Some HDF5 files are not closed:', tables.file._open_files.filenames)
      #  sys.exit()
      #


    if soltype in ['tecandphase', 'tec']:
       tecandphaseplotter(parmdb, ms, outplotname=outplotname) # use own plotter because losoto cannot add tec and phase
        
    if soltype in ['tec']:
       losotoparset_tec = create_losoto_tecparset(ms, outplotname=outplotname,\
                             refant=findrefant_core(parmdb), markersize=compute_markersize(parmdb))
       cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_tec
       print(cmdlosoto)
       logger.info(cmdlosoto)
       run(cmdlosoto)
       force_close(parmdb)

      
    if soltype in ['scalarcomplexgain','complexgain','amplitudeonly','scalaramplitude', \
                   'fulljones','rotation+diagonal'] and (ntimesH5(parmdb) > 1): # plotting/flagging fails if only 1 timeslot
       print('Do flagging?:', flagging)
       if flagging and not onechannel:
          if soltype == 'fulljones':
            print('Fulljones and flagging not implemtened')
            sys.exit()
          else:    
            losotoparset = create_losoto_flag_apgridparset(ms, flagging=True, maxrms=flagslowamprms, \
                                                           maxrmsphase=flagslowphaserms, \
                                                           includesphase=includesphase, onechannel=onechannel, \
                                                           medamp=medamp, flagphases=flagslowphases, onepol=onepol,\
                                                           outplotname=outplotname, refant=findrefant_core(parmdb))
            force_close(parmdb)
       else:
          losotoparset = create_losoto_flag_apgridparset(ms, flagging=False, includesphase=includesphase, \
                         onechannel=onechannel, medamp=medamp, onepol=onepol, outplotname=outplotname,\
                         refant=findrefant_core(parmdb), fulljones=fulljonesparmdb(parmdb)) 
          force_close(parmdb)

       # MAKE losoto command    
       if flagging:
         os.system('cp -f ' + parmdb + ' ' + parmdb + '.backup')
       cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset
       print(cmdlosoto)
       logger.info(cmdlosoto)
       run(cmdlosoto)
    if len(tables.file._open_files.filenames) >= 1: # for debugging
      print('End runDPPPbase, some HDF5 files are not closed:', tables.file._open_files.filenames)
      force_close(parmdb)
    return 

def rotationmeasure_to_phase(H5filein, H5fileout, dejump=False): 
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5in = tables.open_file(H5filein,mode='r')
    H5out = tables.open_file(H5fileout,mode='a')
    c = 2.99792458e8
    
    if dejump:
       rotationmeasure = H5in.root.sol000.rotationmeasure001.val[:]
    else:
       rotationmeasure = H5in.root.sol000.rotationmeasure000.val[:]    
    phase = H5in.root.sol000.phase000.val[:] # time, freq, ant, dir, pol
    freq  =  H5in.root.sol000.phase000.freq[:]
    wav = c/freq

    print('FR step: Shape rotationmeasure000/1', rotationmeasure.shape)
    print('FR step: Shape phase000', phase.shape)
    
    for antenna_id,antennatmp in enumerate(H5in.root.sol000.phase000.ant[:]):
       for freq_id, freqtmp in enumerate(freq):
          phase[:, freq_id, antenna_id, 0, 0]  = \
              2.*rotationmeasure[antenna_id,:]*(wav[freq_id])**2  # notice the factor of 2 because of RR-LL
    # note slice over time-axis
    phase[..., -1] = 0.0 # assume last axis is pol-axis, set YY to zero because losoto residual computation changes this froom 0.0 (it divides the poll diff over the XX and YY phases and does not put it all in XX)
   
    H5out.root.sol000.phase000.val[:] = phase
    H5out.flush()
    H5out.close()
    H5in.close()
    return

def has0coordinates(h5):
    h5 = tables.open_file(h5)
    for c in h5.root.sol000.source[:]:
        x, y = c[1]
        if x==0. and y==0.:
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
        if 'phase000' in st or 'rotation000' in st or 'tec000' in st:
            break
    soltab = solset.getSoltab(st)

    # Find core stations
    ants = soltab.getValues()[1]['ant']
    if 'ST001' in ants:
        return 'ST001'
    cs_indices = np.where(['CS' in ant for ant in ants])[0]

    if len(cs_indices) == 0: # in case there are no CS stations try with RS
       cs_indices = np.where(['RS' in ant for ant in ants])[0]

    # Find the antennas and which dimension that corresponds to
    ant_index = np.where(np.array(soltab.getAxesNames())=='ant')[0][0]

    # Find the antenna with the least flagged datapoint
    weightsum = []
    ndims = soltab.getValues()[0].ndim
    for cs in cs_indices:
        slc = [slice(None)]*ndims
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
    f=open(parset, 'w')

    f.write('[plotFRresult]\n')
    f.write('pol = [XX,YY]\n')
    f.write('operation = PLOT\n')
    f.write('soltab = [sol000/phase000]\n')
    f.write('axesInPlot = [time,freq]\n')
    f.write('axisInTable = ant\n')
    f.write('minmax = [-3.14,3.14]\n')
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname + 'phases_fitFR'))
    f.write('refAnt = %s\n\n\n' % refant)
    f.close()
    return parset

def create_losoto_FRparset(ms, refant='CS001LBA', freqminfitFR=20e6, outplotname='FR', onlyplotFRfit=False, dejump=False):
    """
    Create a losoto parset to fit Faraday Rotation on the phase difference'.
    """
    parset = 'losotoFR.parset'
    os.system('rm -f ' + parset)
    f=open(parset, 'w')

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
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname + 'phases_beforeFR'))
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
    f.write('prefix = plotlosoto%s/%s\n\n\n' % (ms,outplotname + 'FR'))

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
       f.write('prefix = plotlosoto%s/%s\n\n\n' % (ms,outplotname + 'FRdejumped'))



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
    f.write('prefix = plotlosoto%s/%s\n' % (ms,outplotname + 'residualphases_afterFR'))
    f.write('refAnt = %s\n' % refant)
    f.write('minmax = [-3.14,3.14]\n\n\n')

    f.close()
    return parset


# to remove H5/h5 and other files out of a wildcard selection if needed
def removenonms(mslist):
  newmslist = []  
  for ms in mslist:      
   if ms.lower().endswith(('.h5', '.png', '.parset', '.fits', '.backup', '.obj', '.log', '.p', '.reg', '.gz', '.tar', '.tmp', '.ddfcache')) or \
      ms.lower().startswith(('plotlosoto','solintimage')):
     print('WARNING, removing ', ms, 'not a ms-type? Removed it!') 
   else:
     newmslist.append(ms)  
  return newmslist     

# check is there are enough timesteps in the ms
# for example this will remove an observations of length 600s
# in that case a lot of assumptions break, for example amplitude flagging in losoto
def select_valid_ms(mslist):
  newmslist = []  
  
  for ms in mslist:
    if not os.path.isdir(ms):
      print(ms, ' does not exist')
      sys.exit(1)
  
  for ms in mslist:  
    t = pt.table(ms, ack=False)
    times = np.unique(t.getcol('TIME'))
    
    if len(times) > 30:
      newmslist.append(ms)
    else:
      print('---------------------------------------------------------------------------')  
      print('WARNING, removing ', ms, 'not enough timesteps in ms/too short observation')  
      print('---------------------------------------------------------------------------')   
    t.close()
    
    
  return newmslist    

def arg_as_list(s):                                                            
    v = ast.literal_eval(s)                                                    
    if type(v) is not list:                                                    
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v  

def makemaskthresholdlist(maskthresholdlist, stop):
   maskthresholdselfcalcycle = []
   for mm in range(stop):
      try: 
        maskthresholdselfcalcycle.append(maskthresholdlist[mm])
      except:
        maskthresholdselfcalcycle.append(maskthresholdlist[-1]) # add last value
   return maskthresholdselfcalcycle

def niter_from_imsize(imsize):
   if imsize == None:
     print('imsize not set')
     sys.exit()
   if imsize < 1024:
     niter = 15000 # minimum  value
   else:
     niter = 15000*np.int((np.float(imsize)/1024.)) 

   return niter

def basicsetup(mslist, args):
   longbaseline =  checklongbaseline(mslist[0])

   # Determine HBA or LBA
   t    = pt.table(mslist[0] + '/SPECTRAL_WINDOW',ack=False)
   freq = np.median(t.getcol('CHAN_FREQ')[0])
   t.close()

   if freq < 100e6:
     LBA = True
     HBAorLBA = 'HBA'
   else: 
     LBA = False      
     HBAorLBA = 'LBA'

   # set some default values if not provided
   if args['uvmin'] == None:
     if LBA:
         if freq >= 40e6:
           args['uvmin'] = 80.
         if freq < 40e6:
           args['uvmin'] = 60.  
     else:
         args['uvmin'] = 350.

   if args['pixelscale'] == None:  
     if LBA:
       if longbaseline:
         args['pixelscale'] = 0.08 
       else:
         args['pixelscale'] = np.rint(3.0*54e6/freq)  
     else:
       if longbaseline:
         args['pixelscale'] = 0.04
       else:
         args['pixelscale'] = 1.5

   if (args['delaycal'] or args['auto']) and  longbaseline and not LBA:
     if args['imsize'] == None:
       args['imsize'] = 2048   

   if args['boxfile'] != None:
     args['imsize']   = getimsize(args['boxfile'], args['pixelscale'])
   if args['niter'] == None:
     args['niter'] = niter_from_imsize(args['imsize'])  

   if args['auto'] and not longbaseline:
     args['update_uvmin'] = True
     args['usemodeldataforsolints'] = True
     args['forwidefield'] = True
     args['autofrequencyaverage'] = True
     if LBA:
       args['BLsmooth'] = True
     else:
       args['update_multiscale'] = True # HBA only  
       if args['autofrequencyaverage_calspeedup']:
           args['soltypecycles_list'] = [0,999,2]
           args['stop'] = 8

   if args['auto'] and  longbaseline and not args['delaycal']:
     args['update_uvmin'] = False
     args['usemodeldataforsolints'] = True
     args['forwidefield'] = True
     args['autofrequencyaverage'] = True
     args['update_multiscale'] = True
     
     args['soltypecycles-list'] = [0,3]
     args['soltype_list'] = [args['targetcalILT'],'scalarcomplexgain']
     if args['targetcalILT'] == 'tec' or args['targetcalILT'] == 'tecandphase':
        args['smoothnessconstraint_list'] = [0.0, 5.0]
     else:
        args['smoothnessconstraint_list'] = [10.0, 5.0] 
        args['smoothnessreffrequency_list'] = [120.0, 0.0] 
        args['smoothnessspectralexponent_list'] = [-1.0, -1.0]
        args['smoothnessrefdistance_list'] = [0.0,0.0]
     args['uvmin'] =  20000
     if LBA:
       args['BLsmooth'] = True

   if args['delaycal'] and LBA:
       print('Option automated delaycal can only be used for HBA')
       sys.exit() 
   if args['delaycal'] and not longbaseline:
       print('Option automated delaycal can only be used for longbaseline data')
       sys.exit()

   if args['delaycal'] and longbaseline and not LBA:
     args['update_uvmin'] = False
     # args['usemodeldataforsolints'] = True # NEEDS SPECIAL SETTINGS to be implemented
     args['solint_list']="['5min','32sec','1hr']" 
     args['forwidefield'] = True
     args['autofrequencyaverage'] = True
     args['update_multiscale'] = True
     
     args['soltypecycles_list'] = [0,0,3]
     args['soltype_list'] = ['scalarphasediff','scalarphase','scalarcomplexgain']
     args['smoothnessconstraint_list'] = [8.0,2.0,15.0]
     args['smoothnessreffrequency_list'] = [120.,144.,0.0]
     args['smoothnessspectralexponent_list'] = [-2.0,-1.0,-1.0]
     args['smoothnessrefdistance_list'] = [0.0,0.0,0.0]
     args['antennaconstraint_list'] = ['alldutch',None,None]
     args['nchan_list'] = [1,1,1]    
     args['uvmin'] =  40000 
     args['stop'] = 8
     args['maskthreshold'] = [5]
     args['docircular'] = True


   # reset tecandphase -> tec for LBA 
   if LBA and args['usemodeldataforsolints']:
     args['soltype_list'][1] = 'tec'  
     # args['soltype_list'][0] = 'tec'  
     if freq < 30e6:
       # args['soltype_list'] = ['tecandphase','tec']    # no scalarcomplexgain in the list, do not use "tec" that gives rings around sources for some reason
       args['soltype_list'] = ['tecandphase','tecandphase']    # no scalarcomplexgain in the list

   if args['forwidefield']:
      args['doflagging'] = False

   automask = 2.5
   if args['maskthreshold'][-1] < automask:
     automask = args['maskthreshold'][-1] # in case we use a low value for maskthreshold, like Herc A    

   args['imagename']  = args['imagename'] + '_'
   if args['fitsmask'] != None:
     fitsmask = args['fitsmask']
   else:
     fitsmask = None

   if args['boxfile'] != None:
     outtarname = (args['boxfile'].split('/')[-1]).split('.reg')[0] + '.tar.gz'
   else:
     outtarname = 'calibrateddata' + '.tar.gz' 

   maskthreshold_selfcalcycle = makemaskthresholdlist(args['maskthreshold'], args['stop'])
    
   return longbaseline, LBA, HBAorLBA, freq, automask, fitsmask, \
          maskthreshold_selfcalcycle, outtarname, args

###############################
############## MAIN ###########
###############################

def main():
   
   # flagms_startend('P217+57_object.dysco.sub.shift.avg.weights.ms.archive0','tecandphase0_selfcalcyle1_P217+57_object.dysco.sub.shift.avg.weights.ms.archive0.h5',1)
   # sys.exit()
   
   parser = argparse.ArgumentParser(description='Self-Calibrate a facet from a LOFAR observation')
   parser.add_argument('-b','--boxfile', help='boxfile', type=str)
   parser.add_argument('--imsize', help='image size, required if boxfile is not used', type=int)
   parser.add_argument('--pixelscale','--pixelsize', help='pixels size in arcsec, default=3.0/1.5 (LBA/HBA)', type=float)
   parser.add_argument('-i','--imagename', help='imagename, default=image', default='image', type=str)
   parser.add_argument('--fitsmask', help='fitsmask for deconvolution (needs to match image size), if not provided use automasking', type=str)
   parser.add_argument('-n', '--niter', help='niter, default=compute automatically', default=None, type=int)
   parser.add_argument('--robust', help='Briggs robust paramter, default=-0.5', default=-0.5, type=float)
   parser.add_argument('--channelsout', help='channelsout, default=6', default=6, type=int)
   parser.add_argument('--multiscale', help='use multiscale deconvolution, not recommended/unstable', action='store_true')
   parser.add_argument('--multiscale-start', help='start multiscale deconvolution at this selfcal cycle (default=1)', default=1, type=int)
   parser.add_argument('--multiscalescalebias', help='multiscalescale bias scale paramter (see WSClean documentation), default=0.8', default=0.8, type=float)
   parser.add_argument('--deepmultiscale', help='do extra multiscale deconvolution on the residual', action='store_true')
   parser.add_argument('--uvminim', help='inner uv-cut for imaging in lambda, default=80', default=80., type=float)
   parser.add_argument('--usewgridder', help='use wgridder in WSClean, mainly useful for very large images (True/False, default=True)', type=ast.literal_eval, default=True)

   parser.add_argument('--phaseupstations', help='phase up to a superstation (core or superterp, default None)', default=None, type=str)
   parser.add_argument('--phaseshiftbox', help='shift phasecenter to center of this DS9 region box file', default=None, type=str)
   parser.add_argument('--paralleldeconvolution', help='parallel-deconvolution size for wsclean, default=0 (means no parallel deconvolution, suggested value in about 2000, only use for very large images)', default=0, type=int)
   parser.add_argument('--parallelgridding', help='parallel-gridding for wsclean, default=1 (means no parallel gridding)', default=1, type=int)
   parser.add_argument('--deconvolutionchannels', help='deconvolution-channels value for wsclean, default=0 (means deconvolution-channels equals channels-out)', default=0, type=int)


   parser.add_argument('--idg', help='use the Image Domain gridder', action='store_true')
   parser.add_argument('--maskthreshold', help='Maskthresholds used from image1 onwards made by MakeMask.py, default= default=[5.0,4.5,4.0,4.0,3.5,3.5,3.5,...]', default=[5.0,4.5,4.5,4.5,4.0], type=arg_as_list)
   parser.add_argument('--imager', help='Imager to use WSClean or DDFACET, default WSCLEAN', default='WSCLEAN', type=str)
   parser.add_argument('--fitspectralpol', help='use fit-spectral-pol in WSClean (True/False, default=True)', type=ast.literal_eval, default=True)
   parser.add_argument('--fitspectralpolorder', help='fit-spectral-pol order for WSClean, default=3', default=3, type=int)
   parser.add_argument('--taperinnertukey', help='Value for taper-inner-tukey in WSClean, useful to supress negative bowls when using --uvminim (1.5-4.0 time uvminim might give good results, default=None)', default=None, type=float)
 
   parser.add_argument('--removenegativefrommodel', help='remove negative clean components in model predict (True/False, default=True (turned off by default at selfcalcycle 2, see option autoupdate-removenegativefrommodel)', type=ast.literal_eval, default=True)
   parser.add_argument('--autoupdate-removenegativefrommodel', help='Turn off removing negative clean components at selfcalcycle 2 (for high dynamic range imaging it is better to keep all clean components)', type=ast.literal_eval, default=True)
   
   parser.add_argument('--autofrequencyaverage', help='Try frequency averaging if it does not result in bandwidth smearing',  action='store_true')
   parser.add_argument('--autofrequencyaverage-calspeedup', help='Try extra averaging during some selfcalcycles to speed up calibration', action='store_true')
   
   parser.add_argument('--avgfreqstep', help='Extra DP3 frequnecy averaging to speed up a solve, this is done before any other correction, could be useful for long baseline infield calibrators', type=int, default=None)
   parser.add_argument('--avgtimestep', help='Extra DP3 time averaging to speed up a solve, this is done before any other correction, could be useful for long baseline infield calibrators', type=int, default=None)
   parser.add_argument('--msinnchan', help='Before averarging, only take this number input channels', type=int, default=None)
   parser.add_argument('--msinntimes', help='DP3 msin.ntimes setting, mainly for testing purposes', type=int, default=None)
   parser.add_argument('--weightspectrum-clipvalue', help='Extra option to take out bad WEIGHT_SPECTRUM values above the provided number, use with care and first check manually and set the appropriate value (default None, so nothing happens)', type=float, default=None)
   
   # calibration options
   parser.add_argument('-u', '--uvmin', help='inner uv-cut for calibration in lambda, default=80/350 (LBA/HBA)', type=float)
   parser.add_argument('--uvminscalarphasediff', help='inner uv-cut for scalarphasediff calibration in lambda, default it takes the value from --uvmin', type=float, default=None)
   
   parser.add_argument("--update-uvmin", help='Update uvmin automatically for the Dutch array', action='store_true')
   parser.add_argument("--update-multiscale", help='Switch to multiscale automatically if large island of emission are present', action='store_true')
   parser.add_argument("--soltype-list", type=arg_as_list, default=['tecandphase','tecandphase','scalarcomplexgain'],help="List of complexgain,scalarcomplexgain,scalaramplitude,amplitudeonly,phaseonly,fulljones,rotation,rotation+diagonal,tec,tecandphase,scalarphase,scalarphasediff,scalarphasediffFR,phaseonly_phmin,rotation_phmin,tec_phmin,tecandphase_phmin,scalarphase_phmin,scalarphase_slope,phaseonly_slope")
   parser.add_argument("--solint-list", type=check_strlist_or_intlist, default=[1,1,120],help="List of values")   
   parser.add_argument("--nchan-list", type=arg_as_list, default=[1,1,10],help="List of values")
   parser.add_argument("--smoothnessconstraint-list", type=arg_as_list, default=[0.,0.,5.],help="List of values")
   parser.add_argument("--smoothnessreffrequency-list", type=arg_as_list, default=[0.,0.,0.],help="An optional reference frequency (in MHz) for the smoothness constraint. When unequal to 0, the size of the smoothing kernel will vary over frequency by a factor of smoothnessreffrequency*(frequnecy**smoothnessspectralexponent)")
   parser.add_argument("--smoothnessspectralexponent-list", type=arg_as_list, default=[-1.,-1.,-1.],help="If smoothnessreffrequency is not equal to zero then this paramter determines the freqeuency scaling law, default=-1 (1/nu), for scalarphasediff -2 might be useful")
   parser.add_argument("--smoothnessrefdistance-list", type=arg_as_list, default=[0.,0.,0.],help="If smoothnessrefdistance is not equal to zero then this paramter determines the freqeuencysmoothness reference distance in units of km, with the smoothness scaling with distance, see DP3 documentation")   
   parser.add_argument("--antennaconstraint-list", type=arg_as_list, default=[None,None,None],help="List of values")
   parser.add_argument("--resetsols-list", type=arg_as_list, default=[None,None,None],help="Values of these stations will be rest to 0.0 (phases), or 1.0 (amplitudes), default None, possible settings are the same as for antennaconstraint-list (alldutch, core, etc)")
   parser.add_argument("--soltypecycles-list", type=arg_as_list, default=[0,999,3],help="List of values, first entry is required to be 0")
   parser.add_argument("--BLsmooth", help='Employ BLsmooth for low S/N data', action='store_true')
   parser.add_argument("--iontimefactor", help='BLsmooth ionfactor (default=0.01, larger is more smoothing, see BLsmooth documentation)', type=float, default=0.01)
   parser.add_argument("--ionfreqfactor", help='BLsmooth tecfactor (default=1, larger is more smoothing, see BLsmooth documentation)', type=float, default=1.0)
   parser.add_argument("--blscalefactor", help='BLsmooth blscalefactor (default=1.0, see BLsmooth documentation)', type=float, default=1.0)
   parser.add_argument('--dejumpFR', help='Dejump Faraday solutions when using scalarphasediffFR', action='store_true')
   
   parser.add_argument('--usemodeldataforsolints', help='Determine solints from MODEL_DATA', action='store_true')
   parser.add_argument('--tecfactorsolint', help='Experts only', type=float, default=1.0)
   parser.add_argument('--gainfactorsolint', help='Experts only', type=float, default=1.0)
   parser.add_argument('--phasefactorsolint', help='Experts only', type=float, default=1.0)
   parser.add_argument("--preapplyH5-list", type=arg_as_list, default=[None],help="List of H5 files, one per ms")
   # parser.add_argument("--applydelaycalH5-list", type=arg_as_list, default=[None],help="List of H5 files from the delay calibrator, one per ms")
   # parser.add_argument("--applydelaytype", type=str, default='circular', help="Options: circular or linear. If --docircular was used for finding the delay solutions use circular (the default)")

   # general options
   parser.add_argument('--skymodel', help='skymodel for first selfcalcycle', type=str)
   parser.add_argument('--skymodelsource', help='source name (string) in skymodel, default=None (means the skymodel only contains one source/patch', type=str)
   parser.add_argument('--skymodelpointsource', help='If set, start from a point source in the phase center with the flux density given by this parameter, default=None (None means do not use this option)', type=float, default=None)
      # general options
   parser.add_argument('--wscleanskymodel', help='WSclean basename for model images (for a WSClean predict)', type=str, default=None)
   parser.add_argument('--predictskywithbeam', help='predict the skymodel with the beam array factor', action='store_true')
   parser.add_argument('--startfromtgss', help='Start from TGSS skymodel for positions (boxfile required)', action='store_true')
   parser.add_argument('--tgssfitsimage', help='Start TGSS fits image for model (if not provided use SkyView', type=str)
   parser.add_argument('--no-beamcor', help='Do not correct the visilbities for the array factor', action='store_true')
   parser.add_argument('--losotobeamcor-beamlib', help="Beam library to use when not using DP3 for the beam correction. Can be 'stationreponse', 'lofarbeam' (identical and deprecated) or 'everybeam'", type=str, default='stationresponse')
   # parser.add_argument('--use-dpppbeamcor', help='Use DP3 for beam correction, requires recent DP3 version and no phased-up stations', action='store_true')
   parser.add_argument('--docircular', help='Convert linear to circular correlations', action='store_true')
   parser.add_argument('--dolinear', help='Convert circular to linear correlations', action='store_true')
   parser.add_argument('--forwidefield', help='Keep solutions such that they can be used for widefield imaging/screens', action='store_true')
   parser.add_argument('--doflagging', help='Flag on complexgain solutions (True/False, default=True)', type=ast.literal_eval, default=True)
   parser.add_argument('--dysco', help='Use Dysco compression (True/False, default=True)', type=ast.literal_eval, default=True)
   parser.add_argument('--restoreflags', help='Restore flagging column after each selfcal cycle, only relevant if --doflagging=True', action='store_true')
   parser.add_argument('--remove-flagged-from-startend', help='Remove flagged time slots at the start and end of an observations. Do not use if you want to combine DD solutions later for widefield imaging', action='store_true')
   parser.add_argument('--flagslowamprms', help='RMS outlier value to flag on slow amplitudes (default=7.0)', default=7.0, type=float)
   parser.add_argument('--flagslowphaserms', help='RMS outlier value to flag on slow phases (default=7.0)', default=7.0, type=float)
   parser.add_argument('--doflagslowphases', help='If solution flagging is done also flag outliers phases in the slow phase solutions (True/False, default=True)', type=ast.literal_eval, default=True)
   parser.add_argument('--useaoflagger', help='Run AOflagger on input data', action='store_true')
   parser.add_argument('--useaoflaggerbeforeavg', help='Flag with AOflagger before (True) or after averaging (False), default=True', type=ast.literal_eval, default=True)
   parser.add_argument('--normamps', help='Normalize global amplitudes to 1.0 (True/False, default=True, turned off if fulljones is used)', type=ast.literal_eval, default=True)
   parser.add_argument('--normampsskymodel', help='Normalize global amplitudes to 1.0 when solving against an external skymodel (True/False, default=False, turned off if fulljones is used)', type=ast.literal_eval, default=False)
   parser.add_argument('--resetweights', help='If you want to ignore weight_spectrum_solve', action='store_true')
   parser.add_argument('--start', help='Start selfcal cycle at this iteration, default=0', default=0, type=int)
   parser.add_argument('--stop', help='Stop selfcal cycle at this iteration, default=10', default=10, type=int)
   parser.add_argument('--stopafterskysolve', help='Stop calibration after solving against external skymodel', action='store_true')
   parser.add_argument('--noarchive', help='Do not archive the data', action='store_true')
   parser.add_argument('--skipbackup', help='Leave the original ms intact and work and always work on a DP3 copied dataset (not yet implemented)', action='store_true')
   parser.add_argument('--helperscriptspath', help='location were additional helper scripts are located', default='/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/', type=str)
   parser.add_argument('--helperscriptspathh5merge', help='location were  helper scripts h5merge is located (default is None which means the same as helperscriptspath', default=None, type=str)
   parser.add_argument('--auto', help='Trigger fully automated processing (HBA only for now)', action='store_true')
   parser.add_argument('--delaycal', help='Trigger settings suitable for ILT delay calibration, HBA-ILT only - still under construction', action='store_true')
   parser.add_argument('--targetcalILT', help='Type of automated target calibration for HBA international baseline data when --auto is used. Options are: tec, tecandphase, scalarphase, type (default=tec)', default='tec', type=str)
   parser.add_argument('--makeimage-ILTlowres-HBA', help='Make 1.2 arcsec tapered image as quality check of ILT 1 arcsec imaging', action='store_true')
   parser.add_argument('--makeimage-fullpol', help='Under development, make Stokes IQUV version for quality checking', action='store_true')
   parser.add_argument('--blsmooth_chunking_size', type=int, help='Chunking size for blsmooth. Standard is 8, larger values are slower but save on memory. Lower values are faster',default=8)
  
   parser.add_argument('ms', nargs='+', help='msfile(s)')

   options = parser.parse_args()

   # if a config file exists, then read the information
   if os.path.isfile('facetselfcal_config.txt'):
      print( 'A config file exists, using it. This contains:' )
      with open('facetselfcal_config.txt','r') as f:
         lines = f.readlines()
      for line in lines:
         print( line )
         # first get the value
         lineval = line.split('=')[1].lstrip().rstrip('\n')
         try:
            lineval = float( lineval )
            if int(lineval) - lineval == 0:
               lineval = int(lineval)
         except:
            if '[' in lineval:
                lineval = arg_as_list(lineval)
         # this updates the vaue if it exists, or creates a new one if it doesn't
         setattr( options, line.split('=')[0].rstrip(), lineval )


   args = vars(options)

   version = '5.1.0'
   print_title(version)

   os.system('cp ' + args['helperscriptspath'] + '/lib_multiproc.py .')
   if args['helperscriptspathh5merge'] != None:
     os.system('cp ' + args['helperscriptspathh5merge'] + '/h5_merger.py .')
     sys.path.append(os.path.abspath(args['helperscriptspathh5merge']))
   else:
     os.system('cp ' + args['helperscriptspath'] + '/h5_merger.py .')
   
   global h5_merger
   import h5_merger
   os.system('cp ' + args['helperscriptspath'] + '/plot_tecandphase.py .')
   os.system('cp ' + args['helperscriptspath'] + '/lin2circ.py .')
   os.system('cp ' + args['helperscriptspath'] + '/BLsmooth.py .')
   os.system('cp ' + args['helperscriptspath'] + '/polconv.py .')

   inputchecker(args)
   check_code_is_uptodate()


   for h5parm_id, h5parmdb in enumerate(args['preapplyH5_list']):
     if h5parmdb != None:
       os.system('cp ' + h5parmdb +  ' .') # make them local because source direction will ne updated for merging    
       args['preapplyH5_list'][h5parm_id] = h5parmdb.split('/')[-1] # update input list to local location

   mslist = sorted(args['ms'])

   # remove non-ms that ended up in mslist
   mslist = removenonms(mslist)

   # remove ms which are too short (to catch Elais-N1 case of 600s of data)
   mslist = sorted(select_valid_ms(mslist))
   
   # cut ms if there are flagged times at the start or end of the ms
   if args['remove_flagged_from_startend']:
      mslist = sorted(remove_flagged_data_startend(mslist))
      
   if not args['skipbackup']: # work on copy of input data as a backup
      print('Creating a copy of the data and work on that....')
      mslist = average(mslist, freqstep= [0]*len(mslist), timestep=1, start=args['start'], \
                       makecopy=True, dysco=args['dysco'])

   # take out bad WEIGHT_SPECTRUM values if weightspectrum_clipvalue is set
   if args['weightspectrum_clipvalue'] != None:
      fix_bad_weightspectrum(mslist, clipvalue=args['weightspectrum_clipvalue'])
      
   # extra flagging if requested
   if args['start'] == 0 and args['useaoflagger'] and args['useaoflaggerbeforeavg']:  
     runaoflagger(mslist) 

   # reset weights if requested
   if args['resetweights']:
     for ms in mslist:
       cmd = "'update " + ms + " set WEIGHT_SPECTRUM=WEIGHT_SPECTRUM_SOLVE'"
       run("taql " + cmd)

   # SETUP VARIOUS PARAMETERS
   longbaseline, LBA, HBAorLBA, freq, automask, fitsmask, maskthreshold_selfcalcycle, \
       outtarname, args = basicsetup(mslist, args)
   

   # PRE-APPLY SOLUTIONS (from a nearby direction for example)
   # if (args['applydelaycalH5_list'][0]) != None and  args['start'] == 0:
   #      preapplydelay(args['applydelaycalH5_list'], mslist, args['applydelaytype'], dyso=args['dysco'])

   # check if we could average more
   avgfreqstep = []  # vector of len(mslist) with average values, 0 means no averaging
   for ms in mslist:
      if args['avgfreqstep'] == None and args['autofrequencyaverage'] and not LBA \
        and not args['autofrequencyaverage_calspeedup']: # autoaverage
        avgfreqstep.append(findfreqavg(ms,np.float(args['imsize'])))
      else:
        if args['avgfreqstep'] != None:
           avgfreqstep.append(args['avgfreqstep']) # take over handpicked average value
        else:
           avgfreqstep.append(0) # put to zero, zero means no average



   # AVERAGE if requested/possible
   mslist = average(mslist, freqstep=avgfreqstep, timestep=args['avgtimestep'], \
                    start=args['start'], msinnchan=args['msinnchan'],\
                    phaseshiftbox=args['phaseshiftbox'], msinntimes=args['msinntimes'],\
                    dysco=args['dysco'])

   if longbaseline:
     compute_distance_to_pointingcenter(mslist[0], HBAorLBA=HBAorLBA)

   # extra flagging if requested
   if args['start'] == 0 and args['useaoflagger'] and not args['useaoflaggerbeforeavg']:
     runaoflagger(mslist) 


   t    = pt.table(mslist[0] + '/SPECTRAL_WINDOW',ack=False)
   bwsmear = bandwidthsmearing(np.median(t.getcol('CHAN_WIDTH')), np.min(t.getcol('CHAN_FREQ')[0]), np.float(args['imsize']))
   t.close()


   # backup flagging column for option --restoreflags if needed
   if args['restoreflags']:
     for ms in mslist:
       create_backup_flag_col(ms)
    
   # LOG INPUT SETTINGS
   logbasicinfo(args, fitsmask, mslist, version, sys.argv)


   if args['startfromtgss'] and args['start'] == 0:
     if args['skymodel'] == None:
       args['skymodel'] = makeBBSmodelforTGSS(args['boxfile'],fitsimage = args['tgssfitsimage'], \
                                              pixelscale=args['pixelscale'], imsize=args['imsize'], ms=mslist[0])
     else:
       # print('You need to provide a boxfile to use --startfromtgss')
       print('You cannot provide a skymodel file manually while using --startfromtgss')
       sys.exit(1)


   if args['start'] == 0:
     os.system('rm -f nchan.p solint.p smoothnessconstraint.p smoothnessreffrequency.p smoothnessspectralexponent.p smoothnessrefdistance.p antennaconstraint.p resetsols.p soltypecycles.p') 



   nchan_list,solint_list,smoothnessconstraint_list, smoothnessreffrequency_list, \
   smoothnessspectralexponent_list, smoothnessrefdistance_list, \
   antennaconstraint_list, resetsols_list, soltypecycles_list = \
                                              setinitial_solint(mslist, longbaseline, LBA, options)


   # Get restoring beam for DDFACET in case it is needed
   restoringbeam = calculate_restoringbeam(mslist, LBA)

   # set once here, will be updated in the loop below if phaseup is requested
   if args['phaseupstations'] != None:
      # used for h5_merge add_CS option 
      mslist_beforephaseup = mslist[:]  # note copy by slicing otherwise list refers to original
   else:
      mslist_beforephaseup = None

   # ----- START SELFCAL LOOP -----
   for i in range(args['start'],args['stop']):

     # update removenegativefrommodel setting, for high dynamic range it is better to keep negative clean components (based on very clear 3C84 test case)
     if args['autoupdate_removenegativefrommodel'] and i > 1:
        args['removenegativefrommodel'] = False

     # AUTOMATICALLY PICKUP PREVIOUS MASK (in case of a restart)
     if (i > 0) and (args['fitsmask'] == None):
       if args['idg']:  
         if os.path.isfile(args['imagename'] + str(i-1).zfill(3) + '-MFS-I-image.fits.mask.fits'):
             fitsmask = args['imagename'] + str(i-1).zfill(3) + '-MFS-I-image.fits.mask.fits'
       else:
         if args['imager'] == 'WSCLEAN':
           if os.path.isfile(args['imagename'] + str(i-1).zfill(3) + '-MFS-image.fits.mask.fits'):
               fitsmask = args['imagename'] + str(i-1).zfill(3) + '-MFS-image.fits.mask.fits'
         if args['imager'] == 'DDFACET':
           if os.path.isfile(args['imagename'] + str(i-1).zfill(3) + '.app.restored.fits'):
               fitsmask = args['imagename'] + str(i-1).zfill(3) + '.app.restored.fits.mask.fits'

       
     # BEAM CORRECTION
     # if not args['no_beamcor'] and i == 0:
     #    for ms in mslist:
     #      beamcor(ms, usedppp=args['use_dpppbeamcor'], dysco=args['dysco'])

     # CONVERT TO CIRCULAR/LINEAR CORRELATIONS      
     # if (args['docircular'] or args['dolinear']) and i == 0:
     #    for ms in mslist:
     #      circular(ms, linear=args['dolinear'], dysco=args['dysco'])

     # BEAM CORRECTION AND/OR CONVERT TO CIRCULAR/LINEAR CORRELATIONS 
     if ((args['docircular'] or args['dolinear']) or (not args['no_beamcor'])) and (i == 0):
       for ms in mslist:
         beamcor_and_lin2circ(ms, dysco=args['dysco'], \
                              beam=(not args['no_beamcor']), \
                              lin2circ=args['docircular'], \
                              circ2lin=args['dolinear'], \
                              losotobeamlib=args['losotobeamcor_beamlib'])


     # PRE-APPLY SOLUTIONS (from a nearby direction for example)
     # if (args['preapplyH5_list'][0]) != None and i == 0:
     #    preapply(args['preapplyH5_list'], mslist, dysco=args['dysco'])

     # TMP AVERAGE TO SPEED UP CALIBRATION
     if args['autofrequencyaverage_calspeedup'] and i == 0:
         avgfreqstep = []
         mslist_backup = mslist[:] # make a backup list, note copy by slicing otherwise list refers to original
         for ms in mslist:
            avgfreqstep.append(findfreqavg(ms,np.float(args['imsize']),bwsmearlimit=3.5))
         mslist = average(mslist, freqstep=avgfreqstep, timestep=4, dysco=args['dysco'])
     if args['autofrequencyaverage_calspeedup'] and i == args['stop'] - 3:
         mslist = mslist_backup[:]  # reset back, note copy by slicing otherwise list refers to original 
         preapply(create_mergeparmdbname(mslist, i-1), mslist, updateDATA=False, dysco=args['dysco']) # do not overwrite DATA column

     # PHASE-UP if requested
     if args['phaseupstations'] != None:
         if (i == 0) or (i == args['start']):
             mslist = phaseup(mslist,datacolumn='DATA',superstation=args['phaseupstations'], \
                              start=i, dysco=args['dysco'])
     # PRE-APPLY SOLUTIONS (from a nearby direction for example)
     if (args['preapplyH5_list'][0]) != None and i == 0:
         preapply(args['preapplyH5_list'], mslist, dysco=args['dysco'])


     # CALIBRATE AGAINST SKYMODEL
     if (args['skymodel'] != None or args['skymodelpointsource'] != None \
         or args['wscleanskymodel'] != None) and (i ==0):
        calibrateandapplycal(mslist, i, args, solint_list, nchan_list, args['soltype_list'], \
                             soltypecycles_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
                             smoothnessspectralexponent_list, smoothnessrefdistance_list, \
                             antennaconstraint_list, resetsols_list, uvmin=args['uvmin'], normamps=args['normampsskymodel'], \
                             skymodel=args['skymodel'], \
                             predictskywithbeam=args['predictskywithbeam'], \
                             restoreflags=args['restoreflags'], flagging=args['doflagging'], \
                             longbaseline=longbaseline, BLsmooth=args['BLsmooth'], \
                             flagslowphases=args['doflagslowphases'], \
                             flagslowamprms=args['flagslowamprms'], flagslowphaserms=args['flagslowphaserms'],\
                             skymodelsource=args['skymodelsource'], skymodelpointsource=args['skymodelpointsource'],\
                             wscleanskymodel=args['wscleanskymodel'], iontimefactor=args['iontimefactor'], \
                             ionfreqfactor=args['ionfreqfactor'], \
                             blscalefactor=args['blscalefactor'], dejumpFR=args['dejumpFR'],\
                             uvminscalarphasediff=args['uvminscalarphasediff'], \
                             docircular=args['docircular'], mslist_beforephaseup=mslist_beforephaseup, dysco=args['dysco'],\
                             blsmooth_chunking_size=args['blsmooth_chunking_size']) 


  
     # TRIGGER MULTISCALE
     if args['multiscale'] and i >= args['multiscale_start']:
       multiscale = True
     else:
       multiscale = False  

     # MAKE IMAGE
     makeimage(mslist, args['imagename'] + str(i).zfill(3), args['pixelscale'], args['imsize'], \
               args['channelsout'], args['niter'], args['robust'], \
               multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask, \
               deepmultiscale=args['deepmultiscale'], uvminim=args['uvminim'], \
               fitspectralpol=args['fitspectralpol'], \
               imager=args['imager'], restoringbeam=restoringbeam, automask=automask, \
               removenegativecc=args['removenegativefrommodel'], fitspectralpolorder=args['fitspectralpolorder'], \
               usewgridder=args['usewgridder'], paralleldeconvolution=args['paralleldeconvolution'],\
               deconvolutionchannels=args['deconvolutionchannels'], \
               parallelgridding=args['parallelgridding'], multiscalescalebias=args['multiscalescalebias'],\
               taperinnertukey=args['taperinnertukey'])
     if args['makeimage_ILTlowres_HBA']:
       makeimage(mslist, args['imagename'] +'1.2arcsectaper' + str(i).zfill(3), \
               args['pixelscale'], args['imsize'], \
               args['channelsout'], args['niter'], -0.2, uvtaper='1.2arcsec', \
               multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask, \
               uvminim=args['uvminim'], fitspectralpol=args['fitspectralpol'], \
               automask=automask, removenegativecc=False, \
               fitspectralpolorder=args['fitspectralpolorder'], predict=False, \
               usewgridder=args['usewgridder'], paralleldeconvolution=args['paralleldeconvolution'],\
               deconvolutionchannels=args['deconvolutionchannels'], \
               parallelgridding=args['parallelgridding'], multiscalescalebias=args['multiscalescalebias'],\
               taperinnertukey=args['taperinnertukey'])   
     if args['makeimage_fullpol']:
       makeimage(mslist, args['imagename'] +'fullpol' + str(i).zfill(3), \
               args['pixelscale'], args['imsize'], \
               args['channelsout'], args['niter'], args['robust'], \
               multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask, \
               uvminim=args['uvminim'], fitspectralpol=False, \
               automask=automask, removenegativecc=False, predict=False, \
               usewgridder=args['usewgridder'], paralleldeconvolution=args['paralleldeconvolution'],\
               deconvolutionchannels=args['deconvolutionchannels'], \
               parallelgridding=args['parallelgridding'],\
               multiscalescalebias=args['multiscalescalebias'], fullpol=True,\
               taperinnertukey=args['taperinnertukey']) 


  
     # MAKE FIGURE WITH APLPY
     if args['imager'] == 'WSCLEAN':
       if args['idg']:
         plotimage(args['imagename'] + str(i).zfill(3) +'-MFS-I-image.fits',args['imagename'] + str(i).zfill(3) + '.png' , \
                   mask=fitsmask, rmsnoiseimage=args['imagename'] + str(0).zfill(3) +'-MFS-I-image.fits')
       else:
         plotimage(args['imagename'] + str(i).zfill(3) +'-MFS-image.fits',args['imagename'] + str(i) + '.png' , \
                   mask=fitsmask, rmsnoiseimage=args['imagename'] + str(0).zfill(3) +'-MFS-image.fits')
     if args['imager'] == 'DDFACET':
       plotimage(args['imagename'] + str(i).zfill(3) +'.app.restored.fits',args['imagename'] + str(i) + '.png' , \
                   mask=fitsmask, rmsnoiseimage=args['imagename'] + str(0).zfill(3) +'.app.restored.fits')

     if args['stopafterskysolve']:
       print('Stopping as requested via --stopafterskysolve')
       sys.exit(0)
     
     # REDETERMINE SOLINTS IF REQUESTED
     if (i >= 0) and (args['usemodeldataforsolints']):
       print('Recomputing solints .... ')
       nchan_list,solint_list,smoothnessconstraint_list,smoothnessreffrequency_list,\
                              smoothnessspectralexponent_list, smoothnessrefdistance_list, \
                              antennaconstraint_list, resetsols_list, \
                              soltypecycles_list  = \
                              auto_determinesolints(mslist, args['soltype_list'], \
                              longbaseline, LBA, \
                              innchan_list=nchan_list, insolint_list=solint_list, \
                              insmoothnessconstraint_list=smoothnessconstraint_list, \
                              insmoothnessreffrequency_list=smoothnessreffrequency_list,\
                              insmoothnessspectralexponent_list=smoothnessspectralexponent_list,\
                              insmoothnessrefdistance_list=smoothnessrefdistance_list,\
                              inantennaconstraint_list=antennaconstraint_list, \
                              inresetsols_list=resetsols_list, \
                              insoltypecycles_list=soltypecycles_list, redo=True, \
                              tecfactorsolint=args['tecfactorsolint'], \
                              gainfactorsolint=args['gainfactorsolint'], \
                              phasefactorsolint=args['phasefactorsolint'], delaycal=args['delaycal'])  

     # CALIBRATE AND APPLYCAL
     calibrateandapplycal(mslist, i, args, solint_list, nchan_list, args['soltype_list'], soltypecycles_list,\
                           smoothnessconstraint_list, smoothnessreffrequency_list,\
                           smoothnessspectralexponent_list, smoothnessrefdistance_list,\
                           antennaconstraint_list, resetsols_list, uvmin=args['uvmin'], \
                           normamps=args['normamps'], restoreflags=args['restoreflags'], \
                           flagging=args['doflagging'], longbaseline=longbaseline, \
                           BLsmooth=args['BLsmooth'], flagslowphases=args['doflagslowphases'], \
                           flagslowamprms=args['flagslowamprms'], flagslowphaserms=args['flagslowphaserms'],\
                           iontimefactor=args['iontimefactor'], ionfreqfactor=args['ionfreqfactor'], blscalefactor=args['blscalefactor'],\
                           dejumpFR=args['dejumpFR'], uvminscalarphasediff=args['uvminscalarphasediff'],\
                           docircular=args['docircular'], mslist_beforephaseup=mslist_beforephaseup, dysco=args['dysco'],\
                           blsmooth_chunking_size=args['blsmooth_chunking_size'])


 
     # MAKE MASK AND UPDATE UVMIN IF REQUESTED
     if args['fitsmask'] == None:
       if args['imager'] == 'WSCLEAN':   
         if args['idg']:  
           imagename  = args['imagename'] + str(i).zfill(3) + '-MFS-I-image.fits'
         else:
           imagename  = args['imagename'] + str(i).zfill(3) + '-MFS-image.fits'
       if args['imager'] == 'DDFACET':
         imagename  = args['imagename'] + str(i).zfill(3) +'.app.restored.fits'

       if maskthreshold_selfcalcycle[i] > 0.0:    
         cmdm  = 'MakeMask.py --Th='+ str(maskthreshold_selfcalcycle[i]) + ' --RestoredIm=' + imagename
         if fitsmask != None:
           if os.path.isfile(imagename + '.mask.fits'):
             os.system('rm -f ' + imagename + '.mask.fits')
         run(cmdm)
         fitsmask = imagename + '.mask.fits'
      
         # update uvmin if allowed/requested
         if not longbaseline and args['update_uvmin']:
           if getlargestislandsize(fitsmask) > 1000:
             print('Size is largest island [pixels]:', getlargestislandsize(fitsmask))
             logger.info('Size is largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
             if not LBA:
               print('Extended emission found, setting uvmin to 750 klambda')
               logger.info('Extended emission found, setting uvmin to 750 klambda')
               args['uvmin'] = 750
             else:
               print('Extended emission found, setting uvmin to 250 klambda')
               logger.info('Extended emission found, setting uvmin to 250 klambda')
               args['uvmin'] = 250   
         # update to multiscale cleaning if large island is present
         if args['update_multiscale']:       
           print('Size is largest island [pixels]:', getlargestislandsize(fitsmask))
           logger.info('Size is largest island [pixels]:' + str(getlargestislandsize(fitsmask)))
           if getlargestislandsize(fitsmask) > 1000:
             logger.info('Triggering multiscale clean')
             args['multiscale'] = True 
       
       else:
         fitsmask = None # no masking requested as args['maskthreshold'] less/equal 0
          
     # CUT FLAGGED DATA FROM MS AT START&END to win some compute time if possible
     # if TEC and not args['forwidefield']: # does not work for phaseonly sols
     #  if (i == 0) or (i == args['phasecycles']) or (i == args['phasecycles'] + 1) or (i == args['phasecycles'] + 2) \
     #    or (i == args['phasecycles'] + 3) or (i == args['phasecycles'] + 4):
     #     for msnumber, ms in enumerate(mslist): 
     #         flagms_startend(ms, 'phaseonly' + ms + parmdb + str(i) + '.h5', np.int(solint_phase[msnumber]))
  
   # ARCHIVE DATA AFTER SELFCAL if requested 
   if not longbaseline and not args['noarchive'] :
     if not LBA:   
      archive(mslist, outtarname, args['boxfile'], fitsmask, imagename, dysco=args['dysco'])    
      cleanup(mslist)

if __name__ == "__main__":
   main()
