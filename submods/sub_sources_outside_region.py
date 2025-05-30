#!/usr/bin/env python
import casacore.tables as pt
import os, sys
import ast
import numpy as np
import argparse
import pyregion
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import glob
import subprocess
import os.path
import psutil
from scipy.interpolate import InterpolatedUnivariateSpline


def checklongbaseline(ms):
    t = pt.table(ms + '/ANTENNA', ack=False)
    antennasms = list(t.getcol('NAME'))
    t.close()
    substr = 'DE'  # to check if a German station is present, if yes assume this is long baseline data
    haslongbaselines = any(substr in mystring for mystring in antennasms)
    print('Contains long baselines?', haslongbaselines)
    return haslongbaselines


def get_physical_cpus():
    return psutil.cpu_count(logical=False)


def getcpus():
    nodefile = os.getenv('PBS_NODEFILE')
    slurmcpus = os.getenv('SLURM_JOB_CPUS_PER_NODE')
    if nodefile and os.path.isfile(nodefile):
        lines = len(open(nodefile).readlines())
        return lines
    elif slurmcpus:
        return int(slurmcpus)
    else:
        return get_physical_cpus()


class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def die(s):
    print(bcolors.FAIL + s + bcolors.ENDC)
    # if database and use_database():
    #    update_status(None,'Failed')
    raise Exception(s)


def report(s):
    print(bcolors.OKGREEN + s + bcolors.ENDC)


def warn(s):
    print(bcolors.OKBLUE + s + bcolors.ENDC)


def run(command):
    retval = subprocess.call(command, shell=True)
    if retval != 0:
        print('FAILED to run ' + command + ': return value is ' + str(retval))
        raise Exception(command)
    return retval


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def gethistorykey(image, searchkey):
    hdul = fits.open(image)
    his = hdul[0].header['HISTORY']
    for line in his:
        if str(searchkey) in line:
            # print searchkey, line
            keyval = line.split('=')[1]
            keyval = str(keyval.strip())

    hdul.close()
    if keyval is None:
        print('Could not determine', searchkey, 'from HISTORY of the fits header')
        sys.exit(1)
    return str(keyval)


def getimsize(image):
    imsizeddf = None
    robustddf = None
    cellddf = None
    hdul = fits.open(image)
    his = hdul[0].header['HISTORY']
    for line in his:
        if 'Image-NPix' in line:
            imsizeddf = np.int_(line.split('=')[1])
        elif 'Image-Cell' in line:
            cellddf = np.float_(line.split('=')[1])
        elif 'Weight-Robust' in line:
            robustddf = np.float_(line.split('=')[1])

    if imsizeddf is None:
        print('Could not determine the image size, should have been 20000(?) or 6000(?)')
        sys.exit(1)
    if cellddf is None:
        print('Could not determine the image cell size')
        sys.exit(1)
    if robustddf is None:
        print('Could not determine the image robust')
        sys.exit(1)

    hdul.close()
    return imsizeddf, robustddf, cellddf


def getobsmslist(msfiles, observationnumber):
    obsids = []
    for ms in msfiles:
        obsids.append(ms.split('_')[0])
    obsidsextract = np.unique(obsids)[observationnumber]
    mslist = []
    for ms in msfiles:
        if (ms.split('_')[0]) == obsidsextract:
            mslist.append(ms)
    return mslist


def number_of_unique_obsids(msfiles):
    obsids = []
    for ms in msfiles:
        obsids.append(ms.split('_')[0])
    print('Using these observations ', np.unique(obsids))
    return len(np.unique(obsids))


def get_solutions_timerange(sols):
    t = np.load(sols)['BeamTimes']
    return np.min(t), np.max(t)


def fixsymlinks(ddsols):
    # Code from Tim for fixing symbolic links for DDS3_
    # dds3smoothed = glob.glob('SOLSDIR/*/*killMS.DDS3_full_smoothed*npz')
    dds3 = glob.glob('SOLSDIR/*/killMS.' + ddsols + '.sols.npz')
    for i in range(0, len(dds3)):
        symsolname = dds3[i].split('killMS.' + ddsols + '.sols.npz')[0] + 'killMS.' + ddsols + '_smoothed.sols.npz'
        solname = dds3[i]

        start_time, t1 = get_solutions_timerange(solname)
        print(start_time)
        # Rounding different on different computers which is a pain.
        # divide by 10000 to to get rid of last number because rounding can go wrong
        print('%s_%s*_smoothed.npz' % (ddsols, int(start_time) // 10000))
        if len(glob.glob('%s_%s*_smoothed.npz' % (ddsols, int(start_time) // 10000))) != 1:
            raise Exception('Non-unique matching of solution files')
        start_time = glob.glob('%s_%s*_smoothed.npz' % (ddsols, int(start_time) // 10000))[0].split('_')[2]

        if os.path.islink(symsolname):
            print('Symlink ' + symsolname + ' already exists, recreating')
            os.unlink(symsolname)
            os.symlink(os.path.relpath('../../%s_%s_smoothed.npz' % (ddsols, start_time)), symsolname)
        else:
            print('Symlink ' + symsolname + ' does not yet exist, creating')
            os.symlink(os.path.relpath('../../%s_%s_smoothed.npz' % (ddsols, start_time)), symsolname)

    return


def add_dummyms(msfiles):
    '''
    Add dummy ms to create a regular freuqency grid when doing a concat with DPPP
    '''
    if len(msfiles) == 1:
        return msfiles
    keyname = 'REF_FREQUENCY'
    freqaxis = []
    newmslist = []

    # Check for wrong REF_FREQUENCY which happens after a DPPP split in frequency
    for ms in msfiles:
        t = pt.table(ms + '/SPECTRAL_WINDOW', readonly=True)
        freq = t.getcol('REF_FREQUENCY')[0]
        t.close()
        freqaxis.append(freq)
    freqaxis = np.sort(np.array(freqaxis))
    minfreqspacing = np.min(np.diff(freqaxis))
    if minfreqspacing == 0.0:
        keyname = 'CHAN_FREQ'

    freqaxis = []
    for ms in msfiles:
        t = pt.table(ms + '/SPECTRAL_WINDOW', readonly=True)
        if keyname == 'CHAN_FREQ':
            freq = t.getcol(keyname)[0][0]
        else:
            freq = t.getcol(keyname)[0]
        t.close()
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


def columnchecker(mslist, colname):
    for ms in mslist:
        t = pt.table(ms, ack=False)
        if colname not in t.colnames():
            print(colname, ' not present in ', ms)
            sys.exit()
        t.close()


def filechecker(clustercat, dicomask, indico, h5sols, HMPmodelfits, uselowres):
    '''
    Check if files are present to avoid errors to avoid crashes
    '''
    if HMPmodelfits is None:
        if not os.path.isfile(indico):
            raise IOError(indico + ' does not exist')
        if not os.path.isfile(dicomask):
            raise IOError(dicomask + ' does not exist')
    else:
        if not os.path.isfile(HMPmodelfits):
            raise IOError(HMPmodelfits + ' does not exist')

    if not os.path.isfile(clustercat):
        raise IOError(clustercat + ' does not exist')

    if h5sols is None:
        if not os.path.isdir('SOLSDIR'):
            raise IOError('SOLSDIR directory does not exist')

        solsfiletmp = glob.glob('DDS3_full*smoothed.npz')
        if len(solsfiletmp) < 1:
            raise IOError('Cannot find the DDS3_full*smoothed.npz file(s)')

        solsfiletmp = glob.glob('DDS3_full_slow*.npz')
        if len(solsfiletmp) < 1:
            raise IOError('Cannot find the DDS3_full_slow*.npz file(s)')
    else:
        if not os.path.isfile(h5sols):
            raise IOError(h5sols + ' does not exist')
    return


def striparchivename():
    mslist = glob.glob('L*_SB*.ms.archive')
    for ms in mslist:
        outname = ms.rstrip('.archive')
        if os.path.exists(outname):
            if os.path.islink(outname):
                print('Link to', outname, 'already exists')
                continue
            else:
                raise RuntimeError(ms + ' and ' + outname + ' both exist in the directory!')
        cmd = 'ln -s ' + ms + ' ' + outname
        print(cmd)
        os.system(cmd)

    return


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
            t.addcols(pt.makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
        except:
            coldmi['TYPE'] = "StandardStMan"  # DyscoStMan"
            t.addcols(pt.makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
    if (outcol != incol) and write_outcol:
        # copy over the columns
        taql("UPDATE $t SET " + outcol + "=" + incol)

def addextraweights(msfiles):
    '''
    Adds the column WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT from IMAGING_WEIGHT from DR2
    Input msfiles (list of ms)
    '''

    for ms in msfiles:
        ts = pt.table(ms, readonly=False)
        colnames = ts.colnames()
        if 'WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT' not in colnames:
            #desc = ts.getcoldesc('WEIGHT_SPECTRUM')
            #desc['name'] = 'WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT'
            #ts.addcols(desc)
            addcol(ts, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT')
            ts.close()  # to write results

        else:
            print('WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT already exists')
            ts.close()

        ts = pt.table(ms, readonly=False)

        if 'IMAGING_WEIGHT' in colnames:
            iw = ts.getcol('IMAGING_WEIGHT')
            ws_tmp = ts.getcol('WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT')
            n, nfreq, npol = np.shape(ws_tmp)

            for i in range(npol):
                print('Copying over correlation ', i, ms)
                ws_tmp[:, :, i] = iw
                ts.putcol('WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT', ws_tmp)
        else:
            print('IMAGING_WEIGHT column is not present in:', ms)
        ts.close()

    return


def mask_region_cube(infilename, ds9region, outfilename):
    hdu = fits.open(infilename)
    # hduflat = flatten(hdu)
    # map=hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hdu[0], shape=(hdu[0].header['NAXIS1'], hdu[0].header['NAXIS2']))
    # rmsval = np.mean(hdu[0].data[0][0][np.where(manualmask == True)])
    for i in range(hdu[0].header['NAXIS4']):
        hdu[0].data[i][0][np.where(manualmask == True)] = 0.0
    hdu.writeto(outfilename, overwrite=True)
    return


def mask_region(infilename, ds9region, outfilename):
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    # map=hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    # rmsval = np.mean(hdu[0].data[0][0][np.where(manualmask == True)])
    hdu[0].data[0][0][np.where(manualmask == True)] = 0.0
    hdu.writeto(outfilename, overwrite=True)


def mask_except_region(infilename, ds9region, outfilename):
    hdu = fits.open(infilename)
    hduflat = flatten(hdu)
    # map=hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == False)] = 0.0
    hdu.writeto(outfilename, overwrite=True)

    return


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis = f[0].header['NAXIS']
    # if naxis<2:
    #     raise RadioError('Can\'t make map from this')
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


def removecolumn(msfile, colname):
    t = pt.table(msfile, readonly=False)
    colnames = t.colnames()
    if colname in colnames:
        print('Removing ', colname, 'from ', msfile)
        t.removecols(colname)
    t.close()
    return


def getregionboxcenter(regionfile, standardbox=True):
    """
    Extract box center of a DS9 box region. 
    Input is regionfile Return NDPPP compatible string for phasecenter shifting
    """
    r = pyregion.open(regionfile)

    if len(r[:]) > 1:
        print('Only one region can be specified, your file contains', len(r[:]))
        sys.exit()

    if r[0].name != 'box':
        print('Only box region supported')
        sys.exit()

    ra = r[0].coord_list[0]
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

    regioncenter = ('{:12.8f}'.format(ra) + 'deg,' + '{:12.8f}'.format(dec) + 'deg').replace(' ', '')
    return regioncenter


def mscolexist(ms, colname):
    """ Check if a colname exists in the measurement set ms, returns either True or False """
    if os.path.isdir(ms):
        t = pt.table(ms, readonly=True)
        colnames = t.colnames()
        if colname in colnames:  # check if the column is in the list
            exist = True
        else:
            exist = False
        t.close()
    else:
        exist = False  # ms does not exist
    return exist


def adjustboxrotationlocalnorth(boxregionfile, fitsimage):
    r = pyregion.open(boxregionfile)

    hdu = fits.open(fitsimage)
    hduflat = flatten(hdu)

    CRVAL1 = hduflat.header['CRVAL1']  # RA
    # CRVAL2 = hduflat.header['CRVAL2'] # Dec 

    hdu.close()

    if len(r[:]) > 1:
        print('Only one region can be specified, your file contains', len(r[:]))
        sys.exit()

    if r[0].name != 'box':
        print('Only box region supported')
        sys.exit()

    ra = r[0].coord_list[0]
    # dec = r[0].coord_list[1]
    # boxsizex = r[0].coord_list[2]
    # boxsizey = r[0].coord_list[3]
    # angle = r[0].coord_list[4]

    r[0].coord_list[4] = CRVAL1 - ra  # rotate box
    print('Angle adjusted box', CRVAL1 - ra)

    if os.path.isfile('adjustedbox.reg'):
        os.system('rm -rf adjustedbox.reg')

    r.write("adjustedbox.reg")

    # if True:
    #  return    
    #
    # if standardbox:
    #  if boxsizex != boxsizey:
    #    print('Only a square box region supported, you have these sizes:', boxsizex, boxsizey)
    #    sys.exit()
    #  if np.abs(angle) > 1:
    #    print('Only normally oriented sqaure boxes are supported, your region is oriented under angle:', angle)
    #    sys.exit()   

    # if os.path.isfile('poly.reg'):
    #  os.system('rm -rf poly.reg')    
    # f=open('poly.reg', 'w')
    # f.write('# Region file format: DS9 version 4.1\n')
    # f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
    # f.write('fk5\n')
    # f.write('polygon(%s)\n' % vertices)
    # f.close()
    # regioncenter =  ('{:12.8f}'.format(ra) + 'deg,' + '{:12.8f}'.format(dec) + 'deg').replace(' ', '')

    return 'adjustedbox.reg'


def ddfbootstrapcorrection(mslist, incol, outcol, dysco=True):
    for ms in mslist:
        # step 1, create a new column
        t = pt.table(ms, readonly=False, ack=True)
        colnames = t.colnames()
        if outcol not in colnames:
            print(incol, outcol)
            desc = t.getcoldesc(incol)
            newdesc = pt.makecoldesc(outcol, desc)
            newdmi = t.getdminfo(incol)
            if dysco:
                newdmi['NAME'] = 'Dysco' + outcol
            else:
                newdmi['NAME'] = outcol
            t.addcols(newdesc, newdmi)
        # t.close()
        # step 2, get the correction factors
        filenamecross = os.path.basename(ms).split('_')[0] + 'crossmatch-results-2.npy'
        if not os.path.isfile(filenamecross):
            print('Error: cannot find', filenamecross)
            sys.exit()
        scale = np.load(filenamecross)[:, 0]
        filenamefreqs = os.path.basename(ms).split('_')[0] + 'frequencies.txt'
        if not os.path.isfile(filenamefreqs):
            print('Error: cannot find', filenamefreqs)
            sys.exit()
        freqs, _, _, mask = np.genfromtxt(filenamefreqs, converters={0: float, 1: str, 2: str, 3: eval}, unpack=True)
        freqs = freqs[mask]
        print(freqs)

        # InterpolatedUS gives us linear interpolation between points and extrapolation outside it
        spl = InterpolatedUnivariateSpline(freqs, scale, k=1)

        # get the freqs from the ms to apply to
        tfreq = pt.table(ms + '/SPECTRAL_WINDOW', readonly=True, ack=False)
        frq = tfreq[0]['REF_FREQUENCY']
        factor = spl(frq)
        print(frq, factor)
        tfreq.close()
        # t=pt.table(ms,readonly=False)
        print('Scaling Bootstrap factors', ms, 'using', filenamecross, filenamefreqs)
        for row in range(0, t.nrows(), 3000000):
            print('Reading', incol)
            d = t.getcol(incol, startrow=row, nrow=3000000, rowincr=1)
            print('Writing', outcol)
            t.putcol(outcol, d * factor, startrow=row, nrow=3000000, rowincr=1)
        t.close()

    return outcol


def main():
    parser = argparse.ArgumentParser(
        description='Keep soures inside box region, subtract everything else and create new ms')
    parser.add_argument('-b', '--boxfile',
                        help='boxfile, required argument, use --boxfile=fullfield if you want to subtract the entire field',
                        required=True, type=str)
    parser.add_argument('-m', '--mslist', help='DR2 mslist file, default=big-mslist.txt', default='big-mslist.txt',
                        type=str)
    parser.add_argument('-c', '--column', help='Input column for the ms, default=DATA', default='DATA',
                        type=str)  # DATA_DI_CORRECTED
    parser.add_argument('-f', '--freqavg', help='channel averaging, default=4', default=4, type=int)
    parser.add_argument('-t', '--timeavg', help='timesample averaging, default=2', default=2, type=int)

    # if getcpuworks:
    parser.add_argument('-n', '--ncpu', help='number of cpu to use, default=%i' % getcpus(), default=getcpus(),
                        type=int)
    # else:
    # parser.add_argument('-n','--ncpu', help='number of cpu to use, default=%i' % os.cpu_count(), default= os.cpu_count(), type=int)

    parser.add_argument('-p', '--prefixname', help='prefixname for output ms, default=object', default='object',
                        type=str)
    parser.add_argument('--adjustboxrotation',
                        help='Adjust box rotation for the local north to ensure consistency between different pointings (True/False, default=True)',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--nodysco', help='Do not dysco compress output', action='store_false')
    parser.add_argument('--split', help='Do not concat but keep 10 SB blocks', action='store_true')
    parser.add_argument('--aoflaggerbefore', help='Do an extra round of AOflagger on input data', action='store_true')
    parser.add_argument('--aoflaggerafter', help='Do an extra round of AOflagger on averaged output data',
                        action='store_true')
    parser.add_argument('--maxamplitude', help='flag amplitudes above this number, default=1e6', default=1.e6,
                        type=float)
    # parser.add_argument('--takeoutbeam', help='Correct for the beam on the phase-shifted target data', action='store_true')
    parser.add_argument('--uselowres',
                        help='Use the high resolution mode for subtraction, otherwise use the low resolution',
                        action='store_true')
    parser.add_argument('--noconcat', help='Stop after making the DATA_SUB column', action='store_true')
    parser.add_argument('--nophaseshift', help='Do not phaseshift', action='store_true')
    parser.add_argument('--keeplongbaselines',
                        help='Use a Selection-UVRangeKm=[0.100000,5000.000000] instead of the DR2 default',
                        action='store_true')
    parser.add_argument('--h5sols', help='HDF5 solution file, default=None', type=str)
    parser.add_argument('--h5solstring', help='HDF5 solution string, default=sol000/phase000+amplitude000',
                        default='sol000/phase000+amplitude000', type=str)
    parser.add_argument('--DDESolutions-DDSols', help='optional harcoded DDF solution string', type=str)
    parser.add_argument('--clustercat', help='Cluster/nodes npy file, default=image_dirin_SSD_m.npy.ClusterCat.npy',
                        default='image_dirin_SSD_m.npy.ClusterCat.npy', type=str)
    parser.add_argument('--chunkhours', help='Data-ChunkHours for DDF.py, default=9.0', default=9.0, type=float)
    parser.add_argument('--dicomask', help='Mask for filtering the Dico model, default=None (automatically determined)',
                        type=str)
    parser.add_argument('--indico', help='Input Dico model, default=None (automatically determined)', type=str)
    parser.add_argument('--nofixsym', help='Do not attempt to fix killms symlinks -- assume ok', action='store_true')
    parser.add_argument('--nopredict',
                        help='Do not do predict step (for use if repeating last step in case of failure)',
                        action='store_true')
    parser.add_argument('--onlyuseweightspectrum',
                        help='Only use WEIGHT_SPECTRUM, ignore IMAGING_WEIGHT (if internation station are present this is automatically set to True',
                        action='store_true')
    parser.add_argument('--HMPmodelfits', help='if provided, use this HMP model fits  for predict')
    parser.add_argument('--nosubtract',
                        help='Do not do subtract step (for use if repeating last step in case of failure)',
                        action='store_true')
    parser.add_argument('--ddfbootstrapcorrection',
                        help='Apply the bootstrap factor corrections from the ddf-pipeline. Used for ILT data depending on ddf-pipeline version used. Two files are needed for this: "obsid"_crossmatch-results-2.npy and "obsid"frequencies.txt.  This corection should not be used for LoTSS-type data at 6 arcsec resolution as it is already applied to the data products.',
                        action='store_true')
    parser.add_argument('--overwriteoutput', help='Overwrite concat ms if it exists', action='store_true')
    parser.add_argument('--useHMP', help='Use HMP', action='store_true')

    args = vars(parser.parse_args())

    if args['mslist'] == 'big-mslist.txt':
        striparchivename()

    uselowres = args['uselowres']
    if not uselowres:
        fullmask = 'image_full_ampphase_di_m.NS.mask01.fits'
        indico = 'image_full_ampphase_di_m.NS.DicoModel'
        outdico = 'image_full_ampphase_di_m_SUB.NS.DicoModel'
    else:
        fullmask = 'image_full_low_m.mask01.fits'
        indico = 'image_full_low_m.DicoModel'
        outdico = 'image_full_low_m_SUB.DicoModel'

    if args['dicomask'] is not None:
        fullmask = args['dicomask']
    if args['indico'] is not None:
        indico = args['indico']
    if args['HMPmodelfits'] is not None:
        fullmask = args['HMPmodelfits']

    if not os.path.isfile(args['mslist']):
        # # try to make it
        # from make_mslists import make_list
        # success = make_list(workdir=os.getcwd())
        # if not os.path.isfile(args['mslist']):
        raise IOError('File', args['mslist'], 'does not exist and could not be created')

    boxfile = args['boxfile']
    ncpu = args['ncpu']
    timestepavg = args['timeavg']
    freqstepavg = args['freqavg']
    obsid = args['prefixname']

    dopredict = not (args['nopredict'])  # True
    dosubtract = not (args['nosubtract'])  # True
    doconcat = True
    dokmscal = False
    dophaseshift = True
    doflagafter = args['aoflaggerafter']
    amplmax = args['maxamplitude']  # flag data with amplitues above this number
    takeoutbeam = False  # not supported by NDPPP #args['takeoutbeam'], Update 2022: DP3 supports this now but not yet implemented in this script
    holesfixed = True

    aoflagger = args['aoflaggerbefore']
    dysco = args['nodysco']
    split = args['split']  # ouput seperate ms for DDF pipeline
    clustercat = args['clustercat']  # 'image_dirin_SSD_m.npy.ClusterCat.npy'

    if args['keeplongbaselines']:
        uvsel = "[0.0100000,5000.000000]"  # lower limit is 10 meters!
    else:
        uvsel = "[0.0100000,1000.000000]"  # lower limit is 10 meters!

    # print doflagafter, takeoutbeam, aoflagger, dysco, split
    filechecker(clustercat, fullmask, indico, args['h5sols'], args['HMPmodelfits'], args['uselowres'])
    if args['h5sols'] is None:
        if args['DDESolutions_DDSols']:
            ddsols = args['DDESolutions_DDSols'].split('_smoothed')[0]
        else:
            ddsols = 'DDS3_full'
        print(ddsols)

        if not (args['nofixsym']):
            fixsymlinks(ddsols)
        solsfile = glob.glob(ddsols + '*smoothed.npz')
        if len(solsfile) < 1:
            print('Cannot find the correct solution file')
            sys.exit()

    msfiles = ascii.read(args['mslist'], data_start=0)
    msfiles = list(msfiles[:][msfiles.colnames[0]])  # convert to normal list of strings

    if checklongbaseline(msfiles[0]):
        args['onlyuseweightspectrum'] = True

    # bootstrap correction
    if args['ddfbootstrapcorrection']:
        args['column'] = ddfbootstrapcorrection(msfiles, args['column'], args['column'] + '_SCALED',
                                                dysco=dysco)  # update the colname here because we need to proceed from that

    t = pt.table(msfiles[0] + '/OBSERVATION')
    fieldname = t.getcol('LOFAR_TARGET')['array'][0]
    t.close()

    msoutconcat = fieldname + '_' + obsid + '.dysco.sub.shift.avg.weights.ms.archive'

    print(boxfile, args['nophaseshift'])
    if boxfile != 'fullfield' and not args['nophaseshift']:
        composite = False
        r = pyregion.open(boxfile)
        if len(r[:]) > 1:
            composite = True
        else:
            print(boxfile)
            phasecenter = '[' + getregionboxcenter(boxfile, standardbox=True) + ']'
            print(phasecenter)
            if args['adjustboxrotation']:
                print('Correcting boxfile for the local north')
                boxfile = adjustboxrotationlocalnorth(boxfile, fullmask)  # use adjustedbox.reg


    else:
        dophaseshift = False
        composite = False

    # do not phase shift because the user asks specifically
    if args['nophaseshift']:
        dophaseshift = False

    colname = 'DATA_SUB'

    outmask = 'cutoutmask.fits'  # just a name, can be anything

    for observation in range(number_of_unique_obsids(msfiles)):
        if os.path.isdir(msoutconcat + str(observation)):
            print('Warning MS exists:', msoutconcat + str(observation))
            if args['overwriteoutput']:
                os.system('rm -rf ' + msoutconcat + str(observation))
            else:
                sys.exit()
        if os.path.isdir(msoutconcat + str(observation) + '.tmpweight'):
            print('Warning MS exists:', msoutconcat + str(observation) + '.tmpweight')
            if args['overwriteoutput']:
                os.system('rm -rf ' + msoutconcat + str(observation) + '.tmpweight')
            else:
                sys.exit()

    columnchecker(msfiles, args['column'])

    imagenpix, robust, imagecell = getimsize(fullmask)

    if dopredict:
        os.system('rm -f ' + outdico)  # clean up
        os.system('rm -f ' + outmask)

        if boxfile != 'fullfield':
            if args['HMPmodelfits'] is not None:
                mask_region_cube(args['HMPmodelfits'], boxfile, outmask)
            else:
                mask_region(fullmask, boxfile, outmask)
        else:
            outmask = fullmask
            if args['HMPmodelfits'] is not None:
                print('This combination of fullfield and HMPmodelfits doet not work')
                sys.exit()

        if args['HMPmodelfits'] is None:
            run("MaskDicoModel.py --MaskName=%s --InDicoModel=%s --OutDicoModel=%s" % (outmask, indico, outdico))

        if holesfixed:
            print('Starting DDF for prediction')
            if args['h5sols'] is not None:
                if not args['useHMP']:
                    run("DDF.py --Output-Name=image_dd_SUB --Data-ChunkHours=" + str(
                        args['chunkhours']) + " --Data-MS=" + args[
                            'mslist'] + " --Deconv-PeakFactor 0.001000 --Data-ColName " + args[
                            'column'] + " --Parallel-NCPU=" + str(
                        ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=1 --Deconv-Mode SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust " + str(
                        robust) + " --Image-NPix=" + str(
                        imagenpix) + " --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell " + str(
                        imagecell) + " --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam 6.000000 --Freq-NBand=2 --RIME-DecorrMode=FT --SSDClean-SSDSolvePars [S,Alpha] --SSDClean-BICFactor 0 --Mask-Auto=1 --Mask-SigTh=5.00 --Mask-External=" + outmask + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[" +
                        args['h5sols'] + ":" + args[
                            'h5solstring'] + "] --Predict-InitDicoModel=" + outdico + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_SUB' --Misc-IgnoreDeprecationMarking=1")
                else:
                    run("DDF.py --Output-Name=image_dd_SUB --Data-ChunkHours=" + str(
                        args['chunkhours']) + " --Data-MS=" + args['mslist'] + " --Data-ColName " + args[
                            'column'] + " --Parallel-NCPU=" + str(
                        ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=" + gethistorykey(outmask,
                                                                                                           'Beam-CenterNorm') + " --Beam-LOFARBeamMode=" + gethistorykey(
                        outmask, 'Beam-LOFARBeamMode') + " --Deconv-Mode HMP --Beam-Model=" + gethistorykey(outmask,
                                                                                                            'Beam-Model') + " --Weight-Robust " + str(
                        robust) + " --Image-NPix=" + str(
                        imagenpix) + " --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell " + str(
                        imagecell) + " --Facets-NFacets=" + gethistorykey(outmask,
                                                                          'Facets-NFacets') + " --Freq-NDegridBand=" + gethistorykey(
                        outmask, 'Freq-NDegridBand') + " --Beam-NBand=" + gethistorykey(outmask,
                                                                                        'Beam-NBand') + " --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam=" + gethistorykey(
                        outmask, 'Output-RestoringBeam') + " --Freq-NBand=" + gethistorykey(outmask,
                                                                                            'Freq-NBand') + " --RIME-DecorrMode=FT --Mask-Auto=1 --Mask-SigTh=5.00 " + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[" +
                        args['h5sols'] + ":" + args[
                            'h5solstring'] + "] --Predict-InitDicoModel=" + outdico + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_SUB' --Misc-IgnoreDeprecationMarking=1")


            else:
                if not args['useHMP']:
                    if args['DDESolutions_DDSols']:
                        ddsolstr = args['DDESolutions_DDSols']
                    else:
                        ddsolstr = "DDS3_full_smoothed,DDS3_full_slow"

                    run("DDF.py --Output-Name=image_full_ampphase_di_m.NS_SUB --Data-ChunkHours=" + str(
                        args['chunkhours']) + " --Data-MS=" + args[
                            'mslist'] + " --Deconv-PeakFactor 0.001000 --Data-ColName " + args[
                            'column'] + " --Parallel-NCPU=" + str(
                        ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=1 --Deconv-Mode SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust " + str(
                        robust) + " --Image-NPix=" + str(
                        imagenpix) + " --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell " + str(
                        imagecell) + " --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam 6.000000 --Freq-NBand=2 --RIME-DecorrMode=FT --SSDClean-SSDSolvePars [S,Alpha] --SSDClean-BICFactor 0 --Mask-Auto=1 --Mask-SigTh=5.00 --Mask-External=" + outmask + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[" + ddsolstr + "] --Predict-InitDicoModel=" + outdico + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_SUB' --DDESolutions-SolsDir=SOLSDIR --Misc-IgnoreDeprecationMarking=1")
                else:
                    if args['DDESolutions_DDSols']:
                        ddsolstr = args['DDESolutions_DDSols']
                    else:
                        ddsolstr = "DDS3_full_smoothed,DDS3_full_slow"

                    run("DDF.py --Output-Name=image_dd_SUB --Data-ChunkHours=" + str(
                        args['chunkhours']) + " --Data-MS=" + args['mslist'] + " --Data-ColName " + args[
                            'column'] + " --Parallel-NCPU=" + str(
                        ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=" + gethistorykey(outmask,
                                                                                                           'Beam-CenterNorm') + " --Beam-LOFARBeamMode=" + gethistorykey(
                        outmask, 'Beam-LOFARBeamMode') + " --Deconv-Mode HMP --Beam-Model=" + gethistorykey(outmask,
                                                                                                            'Beam-Model') + " --Weight-Robust " + str(
                        robust) + " --Image-NPix=" + str(
                        imagenpix) + " --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell " + str(
                        imagecell) + " --Facets-NFacets=" + gethistorykey(outmask,
                                                                          'Facets-NFacets') + " --Freq-NDegridBand=" + gethistorykey(
                        outmask, 'Freq-NDegridBand') + " --Beam-NBand=" + gethistorykey(outmask,
                                                                                        'Beam-NBand') + " --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam=" + gethistorykey(
                        outmask, 'Output-RestoringBeam') + " --Freq-NBand=" + gethistorykey(outmask,
                                                                                            'Freq-NBand') + " --RIME-DecorrMode=FT --Mask-Auto=1 --Mask-SigTh=5.00 " + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=[" + ddsolstr + "] --Predict-InitDicoModel=" + outdico + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_SUB' --Misc-IgnoreDeprecationMarking=1")


        else:
            print('Starting DDF for prediction')
            if args['DDESolutions_DDSols']:
                ddsolstr = args['DDESolutions_DDSols']
            else:
                ddsolstr = "DDS3_full_smoothed"

            run("DDF.py --Output-Name=image_full_ampphase_di_m.NS_SUB --Data-MS=" + args[
                'mslist'] + " --Deconv-PeakFactor 0.001000 --Data-ColName " + args[
                    'column'] + " --Parallel-NCPU=" + str(
                ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=1 --Deconv-Mode SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust " + str(
                robust) + " --Image-NPix=" + str(
                imagenpix) + " --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell " + str(
                imagecell) + " --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam 6.000000 --Freq-NBand=2 --RIME-DecorrMode=FT --SSDClean-SSDSolvePars [S,Alpha] --SSDClean-BICFactor 0 --Mask-Auto=1 --Mask-SigTh=5.00 --Mask-External=" + outmask + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=" + ddsolstr + " --DDESolutions-SolsDir=SOLSDIR --Predict-InitDicoModel=" + outdico + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_SUB' --Misc-IgnoreDeprecationMarking=1")

    # clear up ddfcache files to save disk space
    os.system('rm -rf *.ddfcache')

    # Subtract the columns
    if dosubtract:
        for ms in msfiles:
            t = pt.table(ms, readonly=False, ack=True)
            colnames = t.colnames()

            if ('PREDICT_SUB' in colnames) and (args['column'] in colnames):
                if colname not in colnames:
                    # Append new column containing all sources
                    desc = t.getcoldesc(args['column'])
                    newdesc = pt.makecoldesc(colname, desc)
                    newdmi = t.getdminfo(args['column'])
                    newdmi['NAME'] = 'Dysco' + colname
                    t.addcols(newdesc, newdmi)

                for row in range(0, t.nrows(), 3000000):
                    print('Reading', 'PREDICT_SUB')
                    f = t.getcol('PREDICT_SUB', startrow=row, nrow=3000000, rowincr=1)
                    print('Reading', args['column'])
                    d = t.getcol(args['column'], startrow=row, nrow=3000000, rowincr=1)

                    print('Writing %s' % colname)
                    t.putcol(colname, d - f, startrow=row, nrow=3000000, rowincr=1)
            else:
                print('Warning, ', ms, ' does not contain PREDICT_SUB and/or ' + args['column'] + ', skipping.....')

            t.close()
        if not args['onlyuseweightspectrum']:
            addextraweights(msfiles)
        # if not args['noconcat']:
        # if not args['keeplongbaselines']:
        # addextraweights(msfiles)

    if composite:
        print('Since you are using a composite DS9 region file I am not going to phaseshift')
        dophaseshift = False
        # sys.exit()

    if args['noconcat']:
        print('You requested the noconcat option, phaseshift and average only')
        for ms in msfiles:

            msout1 = obsid + '_' + ms + '.sub.shift.avg.ms'

            cmd = 'DPPP msin="' + str(ms) + '" msout.writefullresflag=False '
            if dophaseshift:
                cmd += 'steps=[phaseshift,average] '
                cmd += 'phaseshift.type=phaseshift phaseshift.phasecenter=' + phasecenter + ' '
            else:
                cmd += 'steps=[average] '
            cmd += 'average.timestep=' + str(timestepavg) + ' average.freqstep=' + str(freqstepavg) + ' '
            if args['onlyuseweightspectrum']:
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
            else:
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT '
            cmd += 'msout.storagemanager=dysco '
            cmd += 'msout=' + msout1 + ' '
            cmd += 'msin.datacolumn=%s ' % colname
            print(cmd)
            run(cmd)

            if not args['onlyuseweightspectrum']:  # run DP3 again to copy over other weights
                msout2 = obsid + '_' + ms + '.sub.shift.avg.ms.tmpweight'

                cmd = 'DPPP msin="' + str(ms) + '" msout.writefullresflag=False '
                if dophaseshift:
                    cmd += 'steps=[phaseshift,average] '
                    cmd += 'phaseshift.type=phaseshift phaseshift.phasecenter=' + phasecenter + ' '
                else:
                    cmd += 'steps=[average] '
                cmd += 'average.timestep=' + str(timestepavg) + ' average.freqstep=' + str(freqstepavg) + ' '
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM msout.storagemanager=dysco '
                cmd += 'msout=' + msout2 + ' '
                cmd += 'msin.datacolumn=%s ' % colname
                print(cmd)
                run(cmd)
                # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                t = pt.table(msout1, readonly=False)
                print('Adding WEIGHT_SPECTRUM_SOLVE')
                #desc = t.getcoldesc('WEIGHT_SPECTRUM')
                #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
                #t.addcols(desc)
                addcol(t, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

                t2 = pt.table(msout2, readonly=True)
                imweights = t2.getcol('WEIGHT_SPECTRUM')
                t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)
                # Fill WEIGHT_SPECTRUM with WEIGHT_SPECTRUM from second ms
                t2.close()
                t.close()

                # clean up
                os.system('rm -rf ' + msout2)
                print(' ')
                print(' ')
                print('Ouput column WEIGHT_SPECTRUM used for imaging (contains IMAGING_WEIGHT from DR2)')
                print('Ouput column WEIGHT_SPECTRUM_SOLVE used for calibration (contains WEIGHT_SPECTRUM from DR2)')

        sys.exit()  # done, no more steps needed

    if dokmscal:
        outmask_target = 'inregionmask.fits'
        outdico_target = 'image_full_ampphase_di_m_TAR.NS.DicoModel'

        mask_except_region(fullmask, boxfile, outmask_target)

        run("MaskDicoModel.py --MaskName=%s --InDicoModel=%s --OutDicoModel=%s" % (
        outmask_target, indico, outdico_target))

        run("DDF.py --Output-Name=image_full_ampphase_di_m.NS_TAR --Data-MS=" + args[
            'mslist'] + " --Deconv-PeakFactor 0.001000 --Data-ColName " + args['column'] + " --Parallel-NCPU=" + str(
            ncpu) + " --Facets-CatNodes=" + clustercat + " --Beam-CenterNorm=1 --Deconv-Mode SSD --Beam-Model=LOFAR --Beam-LOFARBeamMode=A --Weight-Robust -0.500000 --Image-NPix=20000 --CF-wmax 50000 --CF-Nw 100 --Output-Also onNeds --Image-Cell 1.500000 --Facets-NFacets=11 --SSDClean-NEnlargeData 0 --Freq-NDegridBand 1 --Beam-NBand 1 --Facets-DiamMax 1.5 --Facets-DiamMin 0.1 --Deconv-RMSFactor=3.000000 --SSDClean-ConvFFTSwitch 10000 --Data-Sort 1 --Cache-Dir=. --Log-Memory 1 --Cache-Weight=reset --Output-Mode=Predict --Output-RestoringBeam 6.000000 --Freq-NBand=2 --RIME-DecorrMode=FT --SSDClean-SSDSolvePars [S,Alpha] --SSDClean-BICFactor 0 --Mask-Auto=1 --Mask-SigTh=5.00 --Mask-External=" + outmask + " --DDESolutions-GlobalNorm=None --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --DDESolutions-DDSols=" + solsfile + " --Predict-InitDicoModel=" + outdico_target + " --Selection-UVRangeKm=" + uvsel + " --GAClean-MinSizeInit=10 --Cache-Reset 1 --Beam-Smooth=1 --Predict-ColName='PREDICT_TAR' --Misc-IgnoreDeprecationMarking=1")

        for ms in msfiles:
            # run('kMS.py --MSName %s --SolverType KAFCA --PolMode Scalar --BaseImageName Predict_DDT --NIterKF 6 --CovQ 0.100000 --LambdaKF=0.500000 --NCPU 32 --OutSolsName DIT --PowerSmooth=0.000000 --InCol DATA_SUB --Weighting Natural --UVMinMax=0.100000,1000.000000 --SolsDir=SOLSDIR --SolverType CohJones --PolMode Scalar --SkyModelCol PREDICT_TAR --OutCol DATA_SUB_CORRECTED --ApplyToDir 0 --dt 1.0 --NChanSols 1'%(ms))

            run('kMS.py --MSName %s --SolverType KAFCA --PolMode Scalar --NIterKF 6 --CovQ 0.100000 --LambdaKF=0.500000 --NCPU 32 --OutSolsName DIT --PowerSmooth=0.000000 --InCol DATA_SUB --Weighting Natural --UVMinMax=0.100000,1000.000000 --SolsDir=SOLSDIR --SolverType CohJones --SkyModelCol PREDICT_TAR --OutCol DATA_SUB_CORRECTED --dt 1.0 --NChanSols 1' % (
                ms))

        colname = "DATA_SUB_CORRECTED"

    # can manually update mslist for other selection
    # msfiles   = ascii.read('big-mslist.txt',data_start=0)
    # msfiles   = list(msfiles[:][msfiles.colnames[0]]) # convert to normal list of strings

    for observation in range(number_of_unique_obsids(msfiles)):

        # insert dummies for completely missing blocks to create a regular freuqency grid for DPPP
        obs_mslist = getobsmslist(msfiles, observation)

        obs_mslist = add_dummyms(obs_mslist)

        currentmsoutconcat = msoutconcat + str(observation)
        if doconcat:
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

            # FIRST CONCAT WITH WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT
            cmd = 'DPPP msin="' + str(msfilesconcat) + '" msin.orderms=False '
            if aoflagger:
                cmd += 'aoflagger.type=aoflagger '
                if takeoutbeam:
                    cmd += 'steps=[phaseshift,aoflagger,applybeam,average] '
                    cmd += 'applybeam.type=applybeam applybeam.usechannelfreq=True '
                    cmd += 'applybeam.beammode=array_factor '  # do no update weights from beam because in this case we just want IMAGING_WEIGHT
                else:
                    cmd += 'steps=[phaseshift,aoflagger,average] '
            else:
                if takeoutbeam:
                    cmd += 'steps=[phaseshift,applybeam,average] '
                    cmd += 'applybeam.type=applybeam applybeam.usechannelfreq=True '
                    cmd += 'applybeam.beammode=array_factor '  # do no update weights from beam because in this case we just want IMAGING_WEIGHT
                else:
                    cmd += 'steps=[phaseshift,average] '

            if not dophaseshift:
                cmd = cmd.replace('phaseshift,', '')

            cmd += 'msin.datacolumn=%s msin.missingdata=True ' % colname
            if dysco:
                cmd += 'msout.storagemanager=dysco '
            if args['onlyuseweightspectrum']:
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
            else:
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM_FROM_IMAGING_WEIGHT '
            cmd += 'msout=' + currentmsoutconcat + ' '
            if dophaseshift:
                cmd += 'phaseshift.type=phaseshift phaseshift.phasecenter=' + phasecenter + ' '

            cmd += 'average.type=averager '
            cmd += 'average.timestep=' + str(timestepavg) + ' average.freqstep=' + str(freqstepavg) + ' '

            print(cmd)
            run(cmd)

            if not args['onlyuseweightspectrum']:
                # SECOND CONCAT WITH WEIGHT_SPECTRUM
                cmd = 'DPPP msin="' + str(msfilesconcat) + '" msin.orderms=False '
                if aoflagger:
                    cmd += 'aoflagger.type=aoflagger '
                    if takeoutbeam:
                        cmd += 'steps=[phaseshift,aoflagger,applybeam,average] '
                        cmd += 'applybeam.type=applybeam applybeam.usechannelfreq=True '
                        cmd += 'applybeam.beammode=array_factor applybeam.updateweights=True '
                    else:
                        cmd += 'steps=[phaseshift,aoflagger,average] '
                else:
                    if takeoutbeam:
                        cmd += 'steps=[phaseshift,applybeam,average] '
                        cmd += 'applybeam.type=applybeam applybeam.usechannelfreq=True '
                        cmd += 'applybeam.beammode=array_factor applybeam.updateweights=True '
                    else:
                        cmd += 'steps=[phaseshift,average] '
                cmd += 'msin.datacolumn=%s msin.missingdata=True ' % colname
                if dysco:
                    cmd += 'msout.storagemanager=dysco '
                if not dophaseshift:
                    cmd = cmd.replace('phaseshift,', '')

                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
                cmd += 'msout=' + currentmsoutconcat + '.tmpweight '
                if dophaseshift:
                    cmd += 'phaseshift.type=phaseshift phaseshift.phasecenter=' + phasecenter + ' '
                cmd += 'average.type=averager '
                cmd += 'average.timestep=' + str(timestepavg) + ' average.freqstep=' + str(freqstepavg) + ' '

                print(cmd)
                run(cmd)

                # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                t = pt.table(currentmsoutconcat, readonly=False)

                print('Adding WEIGHT_SPECTRUM_SOLVE')
                #desc = t.getcoldesc('WEIGHT_SPECTRUM')
                #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
                #t.addcols(desc)
                addcol(t, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

                t2 = pt.table(currentmsoutconcat + '.tmpweight', readonly=True)
                imweights = t2.getcol('WEIGHT_SPECTRUM')
                t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

                # Fill WEIGHT_SPECTRUM with WEIGHT_SPECTRUM from second ms
                t2.close()
                t.close()

                # clean up
                os.system('rm -rf ' + currentmsoutconcat + '.tmpweight')

                print(' ')
                print(' ')
                print('Ouput column WEIGHT_SPECTRUM used for imaging (contains IMAGING_WEIGHT from DR2)')
                print('Ouput column WEIGHT_SPECTRUM_SOLVE used for calibration (contains WEIGHT_SPECTRUM from DR2)')

            # cmd += 'msin.starttime=12May2015/19:23:22.0 msin.endtime=13May2015/01:43:00.0 '

        if doflagafter:
            cmd = 'DPPP msin=' + currentmsoutconcat + ' msout=. msin.datacolumn=DATA '
            cmd += 'steps=[aoflagger,preflag] aoflagger.type=aoflagger preflag.type=preflagger '
            cmd += 'preflag.amplmax=' + str(amplmax) + ' '
            run(cmd)

        if split:

            nchanperblock = np.int_(20 / freqstepavg)
            t = pt.table(currentmsoutconcat + '/SPECTRAL_WINDOW', readonly=True)
            nchan = t.getcol('NUM_CHAN')[0]
            t.close()

            for chan in range(0, nchan, nchanperblock):
                msout = obsid + '_chan' + str(chan) + '-' + str(chan + nchanperblock - 1) + '.ms'

                cmd = 'DPPP msin=' + currentmsoutconcat + ' msout=' + msout + ' msin.datacolumn=DATA '
                if dysco:
                    cmd += 'msout.storagemanager=dysco '
                cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM_SOLVE '
                cmd += 'steps=[] ' + 'msin.startchan=' + str(chan) + ' '
                cmd += 'msin.nchan=' + str(nchanperblock) + ' ' + 'msout=' + msout + ' '
                print(cmd)
                run(cmd)


if __name__ == "__main__":
    main()
