#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tables
from losoto import h5parm
import sys

def findrefant_core(H5file, telescope='LOFAR'):
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
    
    if telescope == 'LOFAR':
        # try core stations first
        cs_indices = np.where(['CS' in ant for ant in ants])[0]
        if len(cs_indices) == 0:  # in case there are no CS stations try with RS
            cs_indices = np.where(['RS' in ant for ant in ants])[0]

    if telescope == 'MeerKAT':
        possible_refants = possible_refants = ['m013', 'm012', 'm011', 'm010', 'm009', 'm002', 'm001','m000','m008','m007','m006','m005','m004','m003','m016','m017','m018','m019']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]
    
    if telescope == 'GMRT':
        possible_refants = ['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if telescope == 'ASKAP':
        possible_refants = ['ak01','ak02','ak03','ak04','ak05','ak06','ak07','ak08','ak09','ak10','ak11','ak12','ak13','ak14']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if telescope == 'MWA':
        possible_refants = ["tile012", "tile013", "tile014", "tile015", "tile017", "tile024", "tile025", "tile026", "tile027", "tile032",
                            "tile033", "tile034", "tile035", "tile036", "tile037", "tile038", "tile041", "tile042", "tile043", "tile044",
                            "tile045", "tile046", "tile047", "tile048", "tile063", "tile064", "tile065", "tile066", "tile067", "tile068",
                            "tile083", "tile084", "tile094", "tile095"]
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if len(cs_indices) == 0:
        # print in red
        print('\033[91mWarning: no reference stations found, using all antennas to find refant\033[0m')
        cs_indices = np.arange(len(ants))
   
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
    return ants[maxant]

def phaseplot(phase):
    phase = np.mod(phase + np.pi, 2. * np.pi) - np.pi
    return phase

def delay2phase(delay, freq):
    phase = 2. * np.pi * delay * freq
    return phase

def TEC2phase(TEC, freq):
    phase = -1. * (TEC * 8.44797245e9 / freq)
    return phase

def is_multidir(h5: str = None):
    """
    Check if h5 has multiple directions

    Args:
        h5: h5parm

    Returns: boolean

    """

    with tables.open_file(h5) as H:
        for soltab in H.root.sol000._v_groups.keys():
            dirs = H.root.sol000._f_get_child(soltab).dir[:]
            if len(dirs) > 1:
                return True
    return False


parser = argparse.ArgumentParser(description='Plot DPPP tecandphase solutions by combining the TEC and phase offset')
# imaging options
parser.add_argument('-h5', '--H5file', help='H5 solution file, must contain tecandphase solutions', type=str,
                    required=True)
parser.add_argument('-o', '--outfile', help='Output figure name (png format)', type=str, required=True)
parser.add_argument('-p', '--plotpoints', help='Plot points instead of lines', action='store_true')
parser.add_argument('--telescope', help='Telescope name', type=str, required=True)

# parser.add_argument('-f','--freq', help='Frequnecy at which the phase corrections are plotted', type=float, required=True)
args = vars(parser.parse_args())

H = tables.open_file(args['H5file'], mode='r')
soltabs = list(H.root.sol000._v_children.keys())
print('Found the following soltabs in your H5 file:', soltabs)

try:
    delay = H.root.sol000.delay000.val[:]
    antennas = H.root.sol000.delay000.ant[:].tolist()
    times = H.root.sol000.delay000.time[:]
    freq = H.root.sol000.delay000.freq[:][0]
    nodelay = False
except:
    print('Your solutions contain do NOT contain delay values')
    nodelay = True

try:
    tec = H.root.sol000.tec000.val[:]
    antennas = H.root.sol000.tec000.ant[:].tolist()
    times = H.root.sol000.tec000.time[:]
    freq = H.root.sol000.tec000.freq[:][0]
    notec = False
except:
    print('Your solutions contain do NOT contain TEC values')
    notec = True

try:
    phase = H.root.sol000.phase000.val[:]
    antennas = H.root.sol000.phase000.ant[:].tolist()
    times = H.root.sol000.phase000.time[:]
    freq = H.root.sol000.phase000.freq[:][0]
    containsphase = True
    if notec:
        freq = H.root.sol000.phase000.freq[:]
except:
    print('Your solutions contain do NOT contain phase values')
    containsphase = False
    pass



H.close()


print('Plotting at a frequency of:', freq / 1e6, 'MHz')

times = (times - np.min(times)) / 3600.  # in hrs since obs start


ysizefig = float(len(antennas))
antennas = [x.decode('utf-8') for x in antennas]  # convert to proper string list
refant = antennas.index(findrefant_core((args['H5file']), telescope=args['telescope']))
print('--- Using reference antenna:', antennas[refant])

if containsphase:
    if notec:
        freqidx = int(len(freq) / 2)
        refphase = phase[:, freqidx, refant, 0, 0]
    else:
        if nodelay:
            refphase = phase[:, refant, 0, 0] + TEC2phase(tec[:, refant, 0, 0], freq)
        else:
            refphase = phase[:, refant, 0, 0] + TEC2phase(tec[:, refant, 0, 0], freq) + delay2phase(delay[:, refant, 0, 0], freq)    
else:
    if nodelay:
        refphase = TEC2phase(tec[:, refant, 0, 0], freq)
    else:
        refphase = TEC2phase(tec[:, refant, 0, 0], freq) + delay2phase(delay[:, refant, 0, 0], freq)


with tables.open_file(args['H5file']) as H:
    dirs = H.root.sol000.tec000.dir[:]
    N_dir = len(dirs) 

outplotname = args['outfile']
for direction_id, direction in enumerate(dirs):
    if N_dir > 1:
        outplotname =  args['outfile'].replace('.png', '_Dir' + str(direction_id).zfill(2) + '.png')
        print('Creating:', outplotname)

    fig, ax = plt.subplots(nrows=len(antennas), ncols=1, figsize=(9, 1.5 * ysizefig), squeeze=True, sharex='col')
    figcount = 0

    for antenna_id, antenna in enumerate(antennas):

        #print(figcount)
        if containsphase:

            if notec:
                phasep = phaseplot(phase[:, freqidx, antenna_id, direction_id, 0] - refphase)
            elif nodelay:
                phasep = phaseplot(phase[:, antenna_id, direction_id, 0] + TEC2phase(tec[:, antenna_id, direction_id, 0], freq) - refphase)
            else:
                phasep = phaseplot(phase[:, antenna_id, direction_id, 0] + TEC2phase(tec[:, antenna_id, direction_id, 0], freq) + delay2phase(delay[:, antenna_id, direction_id, 0], freq) - refphase)
        else:
            if nodelay:
                phasep = phaseplot(TEC2phase(tec[:, antenna_id, direction_id, 0], freq) - refphase)
            else:
                phasep = phaseplot(TEC2phase(tec[:, antenna_id, direction_id, 0], freq) + delay2phase(delay[:, antenna_id, direction_id, 0], freq) - refphase)

        if args['plotpoints']:
            ax[figcount].plot(times, phasep, '.')
        else:
            ax[figcount].plot(times, phasep)

        ax[figcount].set_ylabel('phase [rad]')
        if figcount == len(antennas) - 1:
            ax[figcount].set_xlabel('time [hr]')
        ax[figcount].set_title(antenna, position=(0.5, 0.75))
        ax[figcount].set_ylim(-np.pi, np.pi)
        figcount += 1

    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=1.0)
    plt.savefig(outplotname)
    plt.close()
