"""Apply calibration solutions (h5parm) to Measurement Sets with DP3, and match
solution times to MS times."""
import multiprocessing
import os
import shutil
import subprocess

import astropy.units as units
import numpy as np
import tables
from astropy.coordinates import SkyCoord
from casacore.tables import table  # type: ignore

from submods.h5_helpers.general_utils import make_utf8
from submods.h5_helpers.h5_info import amplitude_leakage_paramdb, fulljonesparmdb
from submods.h5_helpers.reset_solutions import resetsolsforstations
from submods.ms_columns import remove_column_ms
from submods.ms_fixes import fix_uvw
from submods.ms_info import check_phaseup_station
from submods.runtime_utils import _track_resource_phase, run, terminal_print


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
        terminal_print(
            'Right ascension (rad):',
            direction[0],
            'Angular separation (deg):',
            angsep.value,
            '[degree]',
        )
        if angsep.value < distance:
            distance = angsep.value
            dirname = direction[0]
    H5.close()
    return dirname


def create_mergeparmdbname(mslist, selfcalcycle, skymodelsolve=False):
    """
    Merges the h5parms for a given list of measurement sets and selfcal cycle.

    Parameters
    ----------
    mslist : list
        list of measurement sets to iterate over.
    selfcalcycle : int
        the selfcal cycle for which to merge h5parms.
    skymodelsolve : bool
        add extra "sky" to the name (for solves against a skymodel)

    Returns
    -------
    parmdblist : list
        list of names of the merged h5parms.
    """

    tmpstr = '.h5'
    
    parmdblist = mslist[:]
    for ms_id, ms in enumerate(mslist):
        if skymodelsolve:
            parmdblist[ms_id] = 'h5_solutions/merged_skyselfcalcycle' + str(selfcalcycle).zfill(3) + '_' + ms + tmpstr
        else:    
            parmdblist[ms_id] = 'h5_solutions/merged_selfcalcycle' + str(selfcalcycle).zfill(3) + '_' + ms + tmpstr
    terminal_print('Created parmdblist', parmdblist)
    return parmdblist


def preapply(H5filelist, mslist, updateDATA=True, dysco=True):
    """
    Pre-apply a given set of corrections to a list of measurement sets.

    Parameters
    ----------
    H5filelist : list
        list of h5parms to apply.
    mslist : list
        list of measurement set to apply corrections to.
    updateDATA : bool
        overwrite DATA with CORRECTED_DATA after solutions have been applied.
    dysco : bool
        dysco compress the CORRECTED_DATA column or not.

    Returns
    -------
    None
    """
    for ms in mslist:
        parmdb = time_match_mstoH5(H5filelist, ms)
        applycal(ms, parmdb, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=dysco)
        if updateDATA:
            run("taql 'update " + ms + " set DATA=CORRECTED_DATA'", taql=True)
    return


def preapply_bandpass(H5filelist, mslist, dysco=True, updateweights=True):
    """
    Pre-apply a given set of corrections to a list of measurement sets.

    Parameters
    ----------
    H5filelist : list
        list of h5parms to apply.
    mslist : list
        list of measurement set to apply corrections to.
    dysco : bool
        dysco compress the CORRECTED_DATA column or not.
    updateweights : bool
        updateweights based on amplitudes in DP3

    Returns
    -------
    None
    """
    for ms in mslist:
        parmdb = find_closest_H5time_toms(H5filelist, ms)
        # overwrite DATA here (!)
        applycal(ms, parmdb, msincol='DATA', msoutcol='DATA', dysco=dysco, updateweights=updateweights, missingantennabehavior='flag')
    return


def find_closest_H5time_toms(H5filelist, ms):
    """
    Find the h5parms, from a given list, that falls closest to the time midpoint of the specified Measurement Set.

    Parameters
    ----------
    H5filelist : list
        list of h5parms to apply.
    ms : str
        Measurement Set to match h5parms to.

    Returns
    -------
    H5filematch : str
        h5parm that is closest in time to the measurement set.
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
            for sol_type in ['amplitude000', 'rotation000', 'phase000', 'tec000', 'rotationmeasure000', 'delay000']:
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
        terminal_print('find_closest_H5time_toms: Cannot find matching H5file and ms')
        raise Exception('find_closest_H5time_toms: Cannot find matching H5file and ms')
    terminal_print(
        'Closest solution H5 file:',
        H5filematch,
        'is closest in time to',
        'Measurement Set:',
        ms,
        ' diff to the midpoint of the ms is',
        'Time difference from MS midpoint (s):',
        time_diff,
        ' [s]',
    )
    return H5filematch


def closest_arrayvals(a, b):
    """
    Find the closest pair of values from two iterables.

    Parameters
    ----------
    a : iterable
        First set of values.
    b : iterable
        Second set of values.

    Returns
    -------
    list or None
        Closest pair as ``[value_from_a, value_from_b]``.
    """
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
    """
    Find the h5parms, from a given list, that overlap in time with the specified Measurement Set.

    Parameters
    ----------
    H5filelist : list
        list of h5parms to apply.
    ms : str
        Measurement Set to match h5parms to.

    Returns
    -------
    H5filematch : list
        list of h5parms matching the measurement set.
    """
    with table(ms) as t:
        timesms = np.unique(t.getcol('TIME'))
    H5filematch = None

    for H5file in H5filelist:
        with tables.open_file(H5file, mode='r') as H:
            times = None
            for sol_type in ['amplitude000', 'rotation000', 'phase000', 'tec000', 'rotationmeasure000', 'delay000']:
                try:
                    times = getattr(H.root.sol000, sol_type).time[:]
                    break
                except AttributeError:
                    continue

            if times is not None and np.median(times) >= np.min(timesms) and np.median(times) <= np.max(timesms):
                terminal_print('Solution H5 file:', H5file, 'overlaps in time with', 'Measurement Set:', ms)
                H5filematch = H5file

    if H5filematch is None:
        terminal_print('Cannot find matching H5file and ms')
        raise Exception('Cannot find matching H5file and ms')

    return H5filematch


def build_applycal_dde_cmd(inparmdblist):
    """
    Build the DP3 apply-calibration command for DDE solutions.

    Parameters
    ----------
    inparmdblist : list
        H5 solution files or solution specifications.

    Returns
    -------
    str
        Constructed DP3 command fragment.
    """
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
        terminal_print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        raise Exception('Something went wrong, cannot build the applycal command. H5 file is valid?')

    cmd += 'ddecal.applycal.steps=['
    for i in range(count):
        cmd += 'ac' + str(i)
        if i < count - 1:  # to avoid last comma in the steps list
            cmd += ','
    cmd += ']'

    return cmd


def corrupt_modelcolumns(ms, h5parm, modeldatacolumns, modelstoragemanager=None):
    """
    Ccorrupt a list of model data columns with H5parm solutions

    Parameters
    ----------
    ms : str
        path to a Measurement Set to apply solutions to.
    h5parm : list/str
        H5parms to apply.
    modeldatacolumns : list
        Model data columns list, there should be more than one, also these columns should alread exist. Note that input column will be overwritten.
    modelstoragemanager : str or None, optional
        Storage manager used when applying solutions to the model columns.

    Returns
    -------
    None
    """

    special_DIL = False
    with tables.open_file(h5parm, mode='r') as H:
        dirnames = None
        for sol_type in ['phase000', 'amplitude000', 'tec000', 'rotation000','rotationmeasure000', 'delay000']:
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


@_track_resource_phase("applycal")
def applycal(ms, inparmdblist, msincol='DATA', msoutcol='CORRECTED_DATA',
             msout='.', dysco=True, modeldatacolumns=[], invert=True, direction=None,
             find_closestdir=False, updateweights=False, modelstoragemanager=None, 
             missingantennabehavior='error', metadata_compression=True, timeslotsperparmupdate=200,
             auto_update_timeslotsperparmupdate=False):
    """
    Apply an H5parm to a Measurement Set.

    Parameters
    ----------
    ms : str
        path to a Measurement Set to apply solutions to.
    inparmdblist : list
        list of H5parms to apply.
    msincol : str
        input column to apply solutions to.
    msoutcol : str
        output column to store corrected data in.
    msout : str
        name of the output Measurement Set.
    dysco : bool
        Dysco compress the output Measurement Set.
    modeldatacolumns : list
        Model data columns list, if len(modeldatacolumns) > 1 we have a DDE solve
    invert : bool
        invert the applycal (if invert is False then = corrupt)
    direction : str
        Name of the direction in a multi-dir h5 for the applycal (find_closestdir needs to be False in this case)
    find_closestdir : bool
        find closest direction (to phasedir MS) in multi-dir h5 file to apply
    updateweights : bool
        Update WEIGHT_SPECTRUM in DP3
    missingantennabehavior : str
        for DP3, must be error or flag
    modelstoragemanager : str or None, optional
        Storage manager used for the output visibility column.
    metadata_compression : bool, optional
        Enable Measurement Set metadata compression when writing a new output.
    timeslotsperparmupdate : int, optional
        Number of time slots processed per DP3 parameter update.
    auto_update_timeslotsperparmupdate : bool, optional
        Reduce the parameter-update batch size automatically for wide-band data.

    Returns
    -------
    None
    """
    
    sisco_modelstoragemanager_modes = ['sisco_stokes_i', 'sisco_diagonal', 'sisco']

    if 'sisco-diagonal' in subprocess.check_output(['wsclean'], text=True):
        fix_sisco_samecol_issue = False  # if sisco-diagonal is present we have a recent DP3/casacore version where writing and reading from the same column is supported, so we can disable the fix for the sisco same column issue (writing to a temporary column and then copying back) which is time consuming
    else:
        fix_sisco_samecol_issue = True
    
    # check if we have a fixed shape column MS, otherwise we need to use use fix_sisco_samecol_issue as well
    # note from May 2026 onwards WSClean will write fixed shape model data columns when DATA also has a fixed shape, before this change WSClean wrote variable shape model data columns (which causes bad performance when writing sisco compressed columns into itself)
    with table(ms, ack=False) as t:
        coldesc = t.coldesc(msincol) # type dictionary
        # check if the the key coldesc['desc']['shape'] exist
        if 'shape' not in coldesc['desc']:
            fix_sisco_samecol_issue = True

    # get number of frequency channels in MS and update timeslotsperparmupdate if needed
    # for large MSs with many frequency channels it smees we need to reduce the timeslotsperparmupdate to avoid segmentation faults
    # For example GMRT data with nchan=2688 and 3535 results in a segmentation faults with the DP3 default timeslotsperparmupdate=200
    if auto_update_timeslotsperparmupdate:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
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
        terminal_print('Wrong input, you cannot use find_closestdir and set a direction')
        raise Exception('Wrong input, you cannot use find_closestdir and set a direction')

    if len(modeldatacolumns) > 1:
        return
        # to allow both a list or a single file (string)
    if not isinstance(inparmdblist, list):
        inparmdblist = [inparmdblist]

    # sisco compression does not support reading and writing to the same column
    msoutcol_orig = msoutcol # save original name
    if msincol == msoutcol and msout == '.' and modelstoragemanager in sisco_modelstoragemanager_modes and fix_sisco_samecol_issue:
        msoutcol = msoutcol + '_SISCO_TEMP'

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
        if modelstoragemanager == 'sisco_stokes_i':
            cmd += 'msout.storagemanager=sisco '
            cmd += 'msout.storagemanager.sisco_mode=stokes_i '
        elif modelstoragemanager == 'sisco_diagonal':
            cmd += 'msout.storagemanager=sisco '
            cmd += 'msout.storagemanager.sisco_mode=diagonal '
        else: 
            cmd += 'msout.storagemanager=' + modelstoragemanager + ' '
    count = 0
    for parmdb in inparmdblist:
        if find_closestdir:
            direction = make_utf8(find_closest_ddsol(parmdb, ms))
            terminal_print('Applying direction:', direction)
        if fulljonesparmdb(parmdb) or amplitude_leakage_paramdb(parmdb):
            cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
            cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
            cmd += 'ac' + str(count) + '.type=applycal '
            cmd += 'ac' + str(count) + '.correction=fulljones '
            if amplitude_leakage_paramdb(parmdb):
                cmd += 'ac' + str(count) + '.soltab=[amplitude000] '
            else:
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

            if 'delay000' in soltabs:
                cmd += 'ac' + str(count) + '.missingantennabehavior=' + missingantennabehavior + ' '
                cmd += 'ac' + str(count) + '.parmdb=' + parmdb + ' '
                cmd += 'ac' + str(count) + '.type=applycal '
                cmd += 'ac' + str(count) + '.timeslotsperparmupdate=' + str(timeslotsperparmupdate) + ' '
                cmd += 'ac' + str(count) + '.correction=delay000 '
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
        terminal_print('Something went wrong, cannot build the applycal command. H5 file is valid?')
        raise Exception('Something went wrong, cannot build the applycal command. H5 file is valid?')
    # build the steps command
    cmd += 'steps=['
    for i in range(count):
        cmd += 'ac' + str(i)
        if i < count - 1:  # to avoid last comma in the steps list
            cmd += ','
    cmd += ']'

    terminal_print('DP3 applycal:', cmd)
    run(cmd, log=True)
    if msout != '.':
        fix_uvw([msout])
    
    if msincol == msoutcol_orig and msout == '.' and modelstoragemanager in sisco_modelstoragemanager_modes and fix_sisco_samecol_issue:
        # copy back from temporary column to original column with taql
        taql_cmd = f"taql 'UPDATE {ms} SET {msoutcol_orig} = {msoutcol}'"
        terminal_print('Copying back from temporary column to original column with taql:', taql_cmd)
        run(taql_cmd, log=True, taql=True)
        # remove temporary column
        remove_column_ms(ms, msoutcol)
    return


def h5flags2ms(h5parm, ms, dysco=True):
    """
    Copy flags from h5parm (which means weight=0) to a ms
    It this this by first creating a copy of the h5parm file, then resetting all the solution values to 1.0 or 0.0
    The appycal() is used to so that that flags make it to the MS
    Since this is essentially a copy, since the solutions are all 1.0 or 0.0 we do this on the DATA column
    which means DATA does not change, apart from the flags being copied over

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file containing solution flags.
    ms : str
        Path to the Measurement Set that receives the flags.
    dysco : bool, optional
        Request Dysco compression when applying the flags. Defaults to True.
    """
    # create a copy of the h5parm file, then reset all the solution values to 1.0 or 0.0
    h5parmcopy = h5parm + '.copy'
    # now copy the file with
    if os.path.isfile(h5parmcopy):
        os.remove(h5parmcopy)
    shutil.copy(h5parm, h5parmcopy)

    try:
        shutil.copy2(h5parm, h5parmcopy)
        resetsolsforstations(h5parmcopy, stationlist='all', refant=None)
        applycal(ms, h5parmcopy, msincol='DATA', msoutcol='DATA', dysco=dysco)
    finally:
        if os.path.exists(h5parmcopy):
            os.remove(h5parmcopy)
