#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tables
from losoto import h5parm

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
        H.close()
        return 'ST001'
    cs_indices = np.where(['CS' in ant for ant in ants])[0]

    # temporary MeerKAT fix
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
    return ants[maxant]


def phaseplot(phase):
    phase = np.mod(phase + np.pi, 2. * np.pi) - np.pi
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

# parser.add_argument('-f','--freq', help='Frequnecy at which the phase corrections are plotted', type=float, required=True)
args = vars(parser.parse_args())

H = tables.open_file(args['H5file'], mode='r')

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
refant = antennas.index(findrefant_core((args['H5file'])))

if containsphase:
    if notec:
        freqidx = int(len(freq) / 2)
        refphase = phase[:, freqidx, refant, 0, 0]
    else:
        refphase = phase[:, refant, 0, 0] + TEC2phase(tec[:, refant, 0, 0], freq)
else:
    refphase = TEC2phase(tec[:, refant, 0, 0], freq)

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
            else:
                phasep = phaseplot(phase[:, antenna_id, direction_id, 0] + TEC2phase(tec[:, antenna_id, direction_id, 0], freq) - refphase)
        else:
            phasep = phaseplot(TEC2phase(tec[:, antenna_id, direction_id, 0], freq) - refphase)

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
