"""Reset h5parm solutions to unity for selected stations or directions."""
import sys

import numpy as np
import tables

from ..runtime_utils import terminal_print
from .h5_info import (amplitude_leakage_paramdb, check_soltabs, findrefant_core,
                      force_close, fulljonesparmdb)


def resetsolsforstations(h5parm, stationlist, refant=None, telescope='LOFAR'):
    """
    Reset solutions for stations

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file to modify.
    stationlist : list of str or str
        Station names to reset, or ``'all'`` to reset every station.
    refant : str or None, optional
        Reference antenna used when resetting phase-referenced solutions.
    telescope : str, optional
        Telescope name used when selecting a reference antenna.
    """
    terminal_print('H5Parm file:', h5parm, 'Station list:', stationlist)
    if isinstance(stationlist, str) and stationlist.lower() == 'all':
        resetall = True
    else:
        resetall = False
    fulljones = fulljonesparmdb(h5parm)  # True/False
    amplitudeleakage = amplitude_leakage_paramdb(h5parm)  # True/False
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure, hasdelay = check_soltabs(h5parm)

    # in case refant is None but h5 still has phase
    # this can happen with a scalaramplitude and soltypelist_includedir is used
    # in this case we have pertubative direction
    # in this case h5_merger has already been run which created a phase000 entry
    if refant is None and hasphase:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

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

    elif hasdelay:
        antennas = H.root.sol000.delay000.ant[:]
        axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')    

    elif hasrotation:
        antennas = H.root.sol000.rotation000.ant[:]
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')

    elif hasrotationmeasure:
        antennas = H.root.sol000.rotationmeasure000.ant[:]
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')

    # Expand the special ``all`` value to the station names used by the
    # solution tables, so the existing per-station reset logic handles every
    # station.
    if resetall:
        stationlist = antennas.astype(str).tolist()

    # should not be needed as h5_merger does not create rotation000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotation:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create rotationmeasure000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotationmeasure:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create tec000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hastec:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create delay000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasdelay:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)    

    if hasamps:
        amp = H.root.sol000.amplitude000.val[:]
    if hasphase:  # also phasereference
        phase = H.root.sol000.phase000.val[:]
        refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant)  # to deal with byte strings
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
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
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
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

    if hasdelay:
        delay = H.root.sol000.delay000.val[:]
        refant_idx = np.where(H.root.sol000.delay000.ant[:].astype(str) == refant)  # to deal with byte strings
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing delay to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            delayn = delay - delay[refant_idx[0], ...]
        if antennaxis == 1:
            delayn = delay - delay[:, refant_idx[0], ...]
        if antennaxis == 2:
            delayn = delay - delay[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            delayn = delay - delay[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            delayn = delay - delay[:, :, :, :, refant_idx[0], ...]
        delay = np.copy(delayn)

    if hasrotation:
        rotation = H.root.sol000.rotation000.val[:]
        refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant)  # to deal with byte strings
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
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
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing faradayrotation to ', refant, 'Axis entry number', axisn.index('ant'))
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

        terminal_print(
            'Antenna name:',
            antenna,
            'Has phase solutions:',
            hasphase,
            'Has amplitude solutions:',
            hasamps,
            'Has TEC solutions:',
            hastec,
            'Has rotation solutions:',
            hasrotation,
            'Has delay solutions:',
            hasdelay,
        )
        if antenna in stationlist:  # in this case reset value to 0.0 (or 1.0)
            if hasphase:
                axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting phase', antenna, 'Axis entry number', axisn.index('ant'))
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
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting amplitude', antenna, 'Axis entry number', axisn.index('ant'))
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
                if fulljones or amplitudeleakage:
                    terminal_print('pol entry axis:', axisn.index('pol'))
                    if len(axisn) != axisn.index('pol') + 1:
                        terminal_print('Pol-axis not the last enrty, cannot handle this')
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
                axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting TEC', antenna, 'Axis entry number', axisn.index('ant'))
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
            if hasdelay:
                axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting delay', antenna, 'Axis entry number', axisn.index('ant'))
                if antennaxis == 0:
                    delay[antennaid, ...] = 0.0
                if antennaxis == 1:
                    delay[:, antennaid, ...] = 0.0
                if antennaxis == 2:
                    delay[:, :, antennaid, ...] = 0.0
                if antennaxis == 3:
                    delay[:, :, :, antennaid, ...] = 0.0
                if antennaxis == 4:
                    delay[:, :, :, :, antennaid, ...] = 0.0
            if hasrotation:
                axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting rotation', antenna, 'Axis entry number', axisn.index('ant'))
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
                axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
                antennaxis = axisn.index('ant')
                terminal_print('Resetting faradayrotation', antenna, 'Axis entry number', axisn.index('ant'))
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
    if hasdelay:
        H.root.sol000.delay000.val[:] = np.copy(delay)    
    if hasrotation:
        H.root.sol000.rotation000.val[:] = np.copy(rotation)
    if hasrotationmeasure:
        H.root.sol000.rotationmeasure000.val[:] = np.copy(faradayrotation)

    H.flush()
    H.close()
    return


def resetsolsfordir(h5parm, dirlist, refant=None, telescope='LOFAR'):
    """
    Reset solutions for directions (DDE solves only)

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file to modify.
    dirlist : list of str
        Direction names whose solutions should be reset.
    refant : str or None, optional
        Reference antenna used when resetting phase-referenced solutions.
    telescope : str, optional
        Telescope name used when selecting a reference antenna.
    """
    terminal_print('H5Parm file:', h5parm, 'Direction list:', dirlist)
    fulljones = fulljonesparmdb(h5parm)
    amplitudeleakage = amplitude_leakage_paramdb(h5parm)
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure, hasdelay = check_soltabs(h5parm)

    # in case refant is None but h5 still has phase
    # this can happen with a scalaramplitude and soltypelist_includedir is used
    # in this case we have pertubative direction
    # in this case h5_merger has already been run which created a phase000 entry
    if refant is None and hasphase:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create rotation000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotation:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create hasrotationmeasure000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasrotationmeasure:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create tec000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hastec:
        refant = findrefant_core(h5parm, telescope=telescope)
        force_close(h5parm)

    # should not be needed as h5_merger does not create delay000
    # keep this code in case of future h5_merger updates so we are safe
    if refant is None and hasdelay:
        refant = findrefant_core(h5parm, telescope=telescope)
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

    elif hasdelay:
        directions = H.root.sol000.delay000.dir[:]
        axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')

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
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
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
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
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

    if hasdelay:
        delay = H.root.sol000.delay000.val[:]
        refant_idx = np.where(H.root.sol000.delay000.ant[:].astype(str) == refant)  # to deal with byte strings
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing delay to ', refant, 'Axis entry number', axisn.index('ant'))
        if antennaxis == 0:
            delayn = delay - delay[refant_idx[0], ...]
        if antennaxis == 1:
            delayn = delay - delay[:, refant_idx[0], ...]
        if antennaxis == 2:
            delayn = delay - delay[:, :, refant_idx[0], ...]
        if antennaxis == 3:
            delayn = delay - delay[:, :, :, refant_idx[0], ...]
        if antennaxis == 4:
            delayn = delay - delay[:, :, :, :, refant_idx[0], ...]
        delay = np.copy(delayn)

    if hasrotation:
        rotation = H.root.sol000.rotation000.val[:]
        refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant)  # to deal with byte strings
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
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
        terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
        axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')
        antennaxis = axisn.index('ant')
        terminal_print('Referencing faradayrotation to ', refant, 'Axis entry number', axisn.index('ant'))
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

        terminal_print(
            'Direction index:',
            directionid,
            'Direction coordinates (RA/Dec, rad):',
            direction,
            'Has phase solutions:',
            hasphase,
            'Has amplitude solutions:',
            hasamps,
            'Has TEC solutions:',
            hastec,
            'Has rotation solutions:',
            hasrotation,
        )
        if directionid in dirlist:  # in this case reset value to 0.0 (or 1.0)
            if hasphase:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                terminal_print('Resetting phase direction ID', directionid, 'Axis entry number', axisn.index('dir'))
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
                terminal_print('Resetting amplitude direction ID', directionid, 'Axis entry number', axisn.index('dir'))
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
                if fulljones or amplitudeleakage:
                    terminal_print('pol entry axis:', axisn.index('pol'))
                    if len(axisn) != axisn.index('pol') + 1:
                        terminal_print('Pol-axis not the last enrty, cannot handle this')
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
                terminal_print('Resetting TEC direction ID', directionid, 'Axis entry number', axisn.index('dir'))
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
            if hasdelay:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')
                terminal_print('Resetting delay direction ID', directionid, 'Axis entry number', axisn.index('dir'))
                if diraxis == 0:
                    delay[directionid, ...] = 0.0
                if diraxis == 1:
                    delay[:, directionid, ...] = 0.0
                if diraxis == 2:
                    delay[:, :, directionid, ...] = 0.0
                if diraxis == 3:
                    delay[:, :, :, directionid, ...] = 0.0
                if diraxis == 4:
                    delay[:, :, :, :, directionid, ...] = 0.0
            if hasrotation:
                diraxis = axisn.index('dir')
                axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')
                terminal_print('Resetting rotation direction ID', directionid, 'Axis entry number', axisn.index('dir'))
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
                terminal_print('Resetting faradayrotation direction ID', directionid, 'Axis entry number', axisn.index('dir'))
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
    if hasdelay:
        H.root.sol000.delay000.val[:] = np.copy(delay)    
    if hasrotation:
        H.root.sol000.rotation000.val[:] = np.copy(rotation)
    if hasrotationmeasure:
        H.root.sol000.rotationmeasure000.val[:] = np.copy(faradayrotation)

    H.flush()
    H.close()
    return
