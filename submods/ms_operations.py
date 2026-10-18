"""Measurement Set operations: splitting, concatenating, averaging, phasing up,
trimming, archiving, and MS lists for imaging."""
import glob
import json
import multiprocessing
import os
import shutil
import time
from itertools import groupby
from pathlib import Path

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from casacore.tables import table, taql  # type: ignore

from submods.antenna_constraints import filter_baseline_str_removestations
from submods.applycal import applycal, time_match_mstoH5
from submods.flagging import return_antennas_highflaggingpercentage
from submods.h5_merger import merge_h5
from submods.ms_columns import _get_stack_groups, addcol, mscolexist
from submods.ms_fixes import fix_uvw, get_telescope_from_ms
from submods.ms_info import (check_phaseup_station, get_chan_freqs, getAntennas,
                             number_of_unique_obsids)
from submods.runtime_utils import get_logger, run, terminal_print
from submods.skymodels import getregioncenter
from submods.smearing import bandwidthsmearing

# facetselfcal package directory (holds flagging_strategies/), same as datapath in main.py
datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal')


def split_columns(ms, outms, column='CORRECTED_DATA'):
    """
    Split a column from a Measurement Set (MS), mimicking CASA split.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set to split.
    outms : str
        Path to the output Measurement Set.
    column : str
        Name of the column to split. Default is 'CORRECTED_DATA'.

    Returns
    -------
    str
        Path to the newly created Measurement Set containing only the specified column.

    Notes
    -----
    - The output MS will contain only the specified column, with other data columns removed.
    - Existing output MS directories will be removed before new ones are created.
    """ 
    # Remove  MS if it exists
    if os.path.isdir(outms):
        shutil.rmtree(outms, ignore_errors=True)
    with table(ms, ack=False) as t:
        colnames = t.colnames()
    if column not in colnames:
        raise ValueError('Column {} not found in {}'.format(column, ms))
    # only select columns that are present, TaQL fails on missing ones
    # (e.g. FLAG_CATEGORY or WEIGHT_SPECTRUM are not always present)
    wanted = ['FLAG_CATEGORY', 'WEIGHT', 'SIGMA', 'ARRAY_ID', 'DATA_DESC_ID', 'EXPOSURE',
              'FEED1', 'FEED2', 'FIELD_ID', 'FLAG_ROW', 'INTERVAL', 'OBSERVATION_ID', 'PROCESSOR_ID',
              'SCAN_NUMBER', 'STATE_ID', 'TIME', 'TIME_CENTROID', 'UVW', 'ANTENNA1', 'ANTENNA2', 'FLAG',
              'WEIGHT_SPECTRUM']
    missing = [col for col in wanted if col not in colnames]
    if missing:
        terminal_print('Columns not present in {}, skipping: {}'.format(ms, ','.join(missing)))
    selected = [col for col in wanted if col in colnames] + [column]
    cmd = "taql 'select {} from {} giving {} as plain'".format(','.join(selected), ms, outms)
    terminal_print('Command:', cmd)
    run(cmd, taql=True)
    fix_uvw([outms])
    return


def split_multidir_ms(ms, field_names=None, dryrun=False, compressed=False, compress_target_only=True, fix_uvw_coordinates=True):
    """
    Splits a multisource Measurement Set (MS) into separate single-source MS files.

    Parameters
    ----------
    ms : str, or list of str
        Path to the multisource Measurement Set to be split.
    field_names : list of str, optional
        List of source names corresponding to source names for each FIELD_ID.
    dryrun : bool, optional
        If True, the function will not exceute the TaQl commands. Default is False.
    compressed : bool, optional
        If True, the output MS will be compressed. Default is False.
    compress_target_only : bool, optional
        If True, only the target MS will be compressed. The target source MS is assumed to be the one with the most time integration. Default is True.
    fix_uvw_coordinates : bool, optional
        If True, the UVW coordinates will be fixed. Default is True.

    Returns
    -------
    list of str
        List of paths to the newly created single-source Measurement Sets.
        If the input MS contains only a single source, returns a list containing the original MS path.

    Notes
    -----
    - If the input MS is already a single-source MS, no splitting is performed.
    - Each output MS will contain data for only one source, with FIELD_ID and SOURCE_ID reset to 0.
    - Existing output MS directories will be removed before new ones are created.
    """
    # in case ms is a list, we loop over the list and call this function for each ms in the list, and then we return a list of lists of ms, we flatten this list of lists to a single list of ms
    
    # check that field_names is of type list, if not raise an error
    if field_names is not None and not isinstance(field_names, list):
        raise ValueError("field_names should be a list of strings, but got type {}".format(type(field_names)))
    
    if isinstance(ms, str):
        ms = [ms]
    mslistout = []

    # check that all ms in the list exist, if not raise an error (MS are directories)
    for msin in ms:
        if not os.path.isdir(msin):
            raise ValueError("Measurement Set {} does not exist".format(msin))


    # find the source in the MS which has the most time integration, we assume this is the target source, and we will compress this source if compress_target_only is True

        
    for msin in ms:
        # load source names and field ids from the FIELD subtable, 
        with table(msin + '::FIELD', readonly=True, ack=False) as t:
            source_names = t.getcol('NAME')
        with table(msin, readonly=True, ack=False) as t:
            field_ids = t.getcol('FIELD_ID')
            unique_field_ids = np.unique(field_ids)

        # compute the total time integration for each source, and find the source with the most time integration
        time_integrations = []
        with table(msin, readonly=True, ack=False) as t:
            for field_id in unique_field_ids:
                time_integrations.append(np.sum(t.getcol('INTERVAL')[field_ids == field_id]))
        target_field_id = unique_field_ids[np.argmax(time_integrations)]
        if compress_target_only and compressed:
            terminal_print('Integration samples for each source:', time_integrations)
            # print the source name of the target source that will be compressed
            terminal_print(source_names[target_field_id] + ' will be compressed because it has the most integration time.')

        # set metadata_compression to True for LOFAR data, and False for other telescopes
        if get_telescope_from_ms(msin) == 'LOFAR':
            metadata_compression = True
        else:
            metadata_compression = False   

        with table(msin, readonly=True, ack=False) as t:
            if len(np.unique(t.getcol('FIELD_ID'))) == 1:
                terminal_print(f"Measurement Set {msin} is already a single source MS, no splitting needed.")
                mslistout.append(msin)
            else:
                terminal_print(f"Splitting multisource Measurement Set {msin} into single source MS...")
        # get the source names
        with table(msin + '::FIELD', readonly=True, ack=False) as t:
            source_names = t.getcol('NAME')
            field_ids = t.getcol('SOURCE_ID')

        # check that the field_names provided are in the source_names, if not raise an error
        if field_names is not None:
            for field_name in field_names:
                if field_name not in source_names:
                    raise ValueError(f"Field name '{field_name}' provided in field_names is not found in the source names of the MS. Available source names: {source_names}")
                    
        # if field_names is not None, only split those sources
        if field_names is not None:
            field_ids = [i for i, name in enumerate(source_names) if name in field_names]
            terminal_print('Splitting the following sources from the MS:', field_names)
        else:
            terminal_print('Splitting all sources from the MS:', source_names) 


        for field_id in field_ids:
            outname = os.path.basename(msin) + '.' + source_names[field_id]
            # Remove  MS if it exists
            if os.path.isdir(outname) and not dryrun:
                shutil.rmtree(outname, ignore_errors=True)
         
            if not dryrun:
                cmd = "taql 'select from {} where FIELD_ID=={} giving {} as plain'".format(msin, field_id, outname)
                terminal_print('Command:', cmd)
                run(cmd, taql=True)

                taql("delete from {} where rownr() not in (select distinct FIELD_ID from {})".format(outname+'/FIELD', outname))

                # Set 'SOURCE_ID' to 0 in the FIELD subtable.
                taql("update {} set SOURCE_ID=0".format(outname+'/FIELD'))

                # Set 'FIELD_ID' to 0 in the main table.
                taql("update {} set FIELD_ID=0".format(outname))

                # compress the output with DP3 if compressed is True
            if compressed:
                cmddp3 = 'DP3 msin=' + outname + ' steps=[] msout.storagemanager=dysco '
                cmddp3 += 'msout=' + outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id]) + ' '
                if not metadata_compression:
                    cmddp3 += 'msout.uvwcompression=False '
                    cmddp3 += 'msout.antennacompression=False '
                    #cmddp3 += 'msout.scalarflags=False '

                if not compress_target_only: # always compress if compress_target_only is False
                    terminal_print('Compressing the output MS with DP3 and dysco...')
                    # remove MS if it exists
                    if os.path.isdir(outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id])) and not dryrun:
                        shutil.rmtree(outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id]), ignore_errors=True)
                    terminal_print('DP3 command:', cmddp3)
                    if not dryrun:
                        run(cmddp3)
                        # remove uncompressed MS
                        shutil.rmtree(outname, ignore_errors=True)
                        if fix_uvw_coordinates:
                            fix_uvw([outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id])])
                    outname = outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id]) # so the append works at the end of the loop
                elif compress_target_only and field_id == target_field_id: # only compress if this is the target source
                    terminal_print('Compressing the target source MS with DP3 and dysco...')
                    # remove MS if it exists
                    if os.path.isdir(outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id])) and not dryrun:
                        shutil.rmtree(outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id]), ignore_errors=True)
                    terminal_print('DP3 command:', cmddp3)
                    if not dryrun:
                        run(cmddp3)        
                        # remove uncompressed MS
                        shutil.rmtree(outname, ignore_errors=True)
                        if fix_uvw_coordinates:
                            fix_uvw([outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id])])
                    outname = outname.replace('.' + source_names[field_id], '.dysco.' + source_names[field_id]) # so the append works at the end of the loop

            mslistout.append(outname)
    terminal_print(f"Splitting completed. Created {len(mslistout)} single source MS.")
    return mslistout


def applycal_restart_di(mslist, selfcalcycle, args):
    """
    Apply merged selfcal solution files from a previous cycle in case of a restart for DI mode
    This makes CORRECTED_DATA from the previous selfcal cycle merged h5 solutions

    Parameters
    ----------
    mslist : list
        List of MS

    selfcalcycle : int
        selfcal cycle number
    args : dict
        Parsed facetselfcal options; ``stack`` and ``dysco`` are used.

    Returns
    -------
    None
    """
    previous_cycle = selfcalcycle - 1
    if args['stack']:
        for stacked_ms, ms_group in _get_stack_groups(mslist):
            parmdbmergename = (
                'h5_solutions/merged_selfcalcycle'
                + str(previous_cycle).zfill(3) + '_'
                + os.path.basename(stacked_ms) + '.h5'
            )
            if not os.path.isfile(parmdbmergename):
                raise FileNotFoundError(
                    f'Cannot restart DI self-calibration cycle {selfcalcycle}: '
                    f'missing merged H5 for previous cycle {previous_cycle}, '
                    f'stacked MS {stacked_ms}: {parmdbmergename}'
                )
            for ms in ms_group:
                applycal(ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])
    else:
        for ms in mslist:
            parmdbmergename = 'h5_solutions/merged_selfcalcycle' + str(previous_cycle).zfill(3) + '_' + os.path.basename(ms) + '.h5'
            applycal(ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])
    return


def timebase(fov, ms, tau=0.995):
    """
    Find DP3 timebase value for BDA.

    Uses formulas from Bridle & Schwab (1999).

    Parameters
    ----------
    fov : float
        Field of view in degrees.
    ms : str
        Measurement Set path.
    tau : float
        Peak flux loss factor.

    Returns
    -------
    float
        DP3 timebase value for BDA.
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
    BDA compress list of MS with DP3.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    pixsize : float
        Pixel size in arcsec.
    imsize : int
        Image size in pixels.
    dryrun : bool, optional
        Whether to create the actual MS (otherwise only bda_mslist is made).
    metadata_compression : bool, optional
        Whether to use metadata compression.

    Returns
    -------
    list of str
        List of BDA Measurement Sets.
    """
    bda_mslist = []
    for ms in mslist:
        if not dryrun:
            # fix uwv definition if needed (for MeerKAT data) in case of standalone usage of this function
            fix_uvw([ms])
        cmd = 'DP3 msin=' + ms + ' steps=[bda] bda.type=bdaaverage msout=' + ms + '.bda '
        cmd += 'bda.maxinterval=300 '
        cmd += 'bda.timebase=' + str(timebase(imsize*pixsize/3600.,ms)) + ' '
        if not metadata_compression:
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '
        terminal_print('Command:', cmd)
        if not dryrun:
            if os.path.isdir(ms + '.bda'):
                shutil.rmtree(ms + '.bda')
            run(cmd)
        bda_mslist.append(ms + '.bda')
    fix_uvw(bda_mslist)
    return bda_mslist


def create_pointing_list(mslist):
    """
    Create a list of pointing centers from a list of Measurement Sets (MS).

    Rounds pointing centers to 1 arcsecond precision to group MSs with
    the same pointing center for concatenation.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Set paths.

    Returns
    -------
    list of str
        List of unique pointing centers formatted as 'RA,Dec'.
    """
    pointing_list = []
    terminal_print('Taking pointing center from the ms')
    for ms in mslist:
        with table(ms + '::FIELD', readonly=True, ack=False) as t:
            ra_ref, dec_ref = t.getcol('REFERENCE_DIR').squeeze()
            center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')   
            # Round to 1arcsec precision
            center_ra = round(center.ra.deg * 3600) / 3600
            center_dec = round(center.dec.deg * 3600) / 3600
            center = SkyCoord(center_ra * units.deg, center_dec * units.deg, frame='icrs')
            pointing_list.append(center)

    return pointing_list


def concat_ms_wsclean_facetimaging(mslist, h5list=None, concatms=True):
    """
    Group compatible Measurement Sets and prepare WSClean inputs.

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths to group.
    h5list : list of str or None, optional
        H5 solutions to match and
        merge for each group.
    concatms : bool, optional
        Concatenate each group into an output MS. If False, existing
        wsclean_concat_*.ms are reused and only missing ones are created.

    Returns
    -------
    tuple
        Lists of concatenated Measurement Set and H5 file paths.
    """
    plist = create_pointing_list(mslist)
    pointing_by_ms = dict(zip(mslist, plist))
    chan_freqs_by_ms = {ms: get_chan_freqs(ms) for ms in mslist}
    antennas_by_ms = {ms: tuple(sorted(getAntennas(ms))) for ms in mslist}

    def keyfunct(ms):
        """
        Return the grouping key for a Measurement Set.

        Parameters
        ----------
        ms : str
            Path to the Measurement Set.

        Returns
        -------
        str
            Grouping key string.
        """
        center = pointing_by_ms[ms]
        return (center.ra.deg, center.dec.deg, antennas_by_ms[ms], chan_freqs_by_ms[ms])

    MSs_list = sorted(mslist, key=keyfunct)  # needs to be sorted

    groups = []
    for k, g in groupby(MSs_list, keyfunct):
        groups.append(list(g))
    terminal_print(f"Found {len(groups)} groups of datasets with the same pointing center, antennas, and channel frequencies.")

    for i, group in enumerate(groups, start=1):
        antennas = ', '.join(antennas_by_ms[group[0]])
        terminal_print(f"WSClean MS group {i}: {group}")
        terminal_print(f"List of antennas: {antennas}")
        if len(chan_freqs_by_ms[group[0]]) > 0:
            terminal_print(f"Frequency range: {chan_freqs_by_ms[group[0]][0]:.3e} - {chan_freqs_by_ms[group[0]][-1]:.3e} Hz ({len(chan_freqs_by_ms[group[0]])} channels)")

    MSs_files_clean = []
    H5s_files_clean = []
    for g, group in enumerate(groups):
        if os.path.isdir(f'wsclean_concat_{g}.ms') and concatms:
            shutil.rmtree(f'wsclean_concat_{g}.ms', ignore_errors=True)

        if h5list is not None:
            h5group = []
            for ms in group:
                h5group.append(time_match_mstoH5(h5list, ms))
            terminal_print('------------------------------')
            terminal_print('MS group and matched h5 group', group, h5group)
            terminal_print('------------------------------')
            if os.path.isfile(f'wsclean_concat_{g}.h5'):
                os.remove(f'wsclean_concat_{g}.h5')
            merge_h5(h5_out=f'wsclean_concat_{g}.h5', h5_tables=h5group, propagate_weights=True, time_concat=True)
            H5s_files_clean.append(f'wsclean_concat_{g}.h5')
        # with concatms=False reuse the existing concatenated MS, but create it if it is missing
        # (e.g. on a restart with --start > 0 in a directory without wsclean_concat_*.ms)
        if concatms or not os.path.isdir(f'wsclean_concat_{g}.ms'):
            terminal_print(f'taql select from {group} giving wsclean_concat_{g}.ms as plain')
            run(f'taql select from {group} giving wsclean_concat_{g}.ms as plain', taql=True)
            fix_uvw(f'wsclean_concat_{g}.ms')
        MSs_files_clean.append(f'wsclean_concat_{g}.ms')

    # MSs_files_clean = ' '.join(MSs_files_clean)

    # print('Use the following ms files as input in wsclean:')
    terminal_print('Measurement Sets used for imaging:', MSs_files_clean)
    return MSs_files_clean, H5s_files_clean


def add_dummyms(msfiles):
    """
    Add dummy measurement sets to create a regular frequency grid when concatenating with DPPP.
    This function analyzes a list of measurement set files and inserts dummy MS entries
    where there are gaps in the frequency spacing. This ensures a regular frequency grid
    when the measurement sets are concatenated using DPPP.
    Parameters
    ----------
    msfiles : list of str
        List of paths to measurement set files to be processed.
    Returns
    -------
    list of str
        Updated list of measurement set files with dummy MS entries ('dummyX.ms') 
        inserted where needed to maintain regular frequency spacing. If only one MS
        is provided, returns the original list unchanged.
    Notes
    -----
    - The function first checks the REF_FREQUENCY values in each MS. If these are
    identical (which can happen after DPPP split in frequency), it falls back
    to using CHAN_FREQ values.
    - The input MS files are automatically sorted by increasing frequency.
    - Dummy MS filenames follow the pattern 'dummy0.ms', 'dummy1.ms', etc.
    - The function prints information about added dummy MS files and the final list.
    Examples
    --------
    >>> msfiles = ['observation1.ms', 'observation3.ms']
    >>> result = add_dummyms(msfiles)
    Added dummy: dummy0.ms
    Updated ms list with dummies inserted to create a regular frequency grid
    ['observation1.ms', 'dummy0.ms', 'observation3.ms']
    """
    
    if len(msfiles) == 1:
        return msfiles
    keyname = 'REF_FREQUENCY'
    freqaxis = []
    newmslist = []

    # Check for wrong REF_FREQUENCY which happens after a DPPP split in frequency
    for ms in msfiles:
        with table(ms + '::SPECTRAL_WINDOW', readonly=True) as t:
            freq = t.getcol('REF_FREQUENCY')[0]
        freqaxis.append(freq)
    freqaxis = np.sort(np.array(freqaxis))
    minfreqspacing = np.min(np.diff(freqaxis))
    if minfreqspacing == 0.0:
        keyname = 'CHAN_FREQ'

    freqaxis = []
    for ms in msfiles:
        with table(ms + '::SPECTRAL_WINDOW', readonly=True) as t:
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
    positive_freqspacing = freqspacing[freqspacing > 0.0]

    # Duplicate frequencies do not represent a gap. Ignore their zero spacing
    # when determining the regular grid step to avoid division by zero.
    if len(positive_freqspacing) == 0:
        terminal_print('No positive frequency spacing found; no dummy MS files needed')
        return sortedmslist
    minfreqspacing = np.min(positive_freqspacing)

    # insert dummies in the ms list if needed
    count = 0
    newmslist.append(sortedmslist[0])  # always start with the first ms the list
    for msnumber, ms in enumerate(sortedmslist[1::]):
        if int(round(freqspacing[msnumber] / minfreqspacing)) > 1:
            ndummy = int(round(freqspacing[msnumber] / minfreqspacing)) - 1

            for dummy in range(ndummy):
                newmslist.append('dummy' + str(count) + '.ms')
                terminal_print('Added dummy:', 'dummy' + str(count) + '.ms')
                count = count + 1
        newmslist.append(ms)

    terminal_print('Updated ms list with dummies inserted to create a regular frequency grid')
    terminal_print('Measurement Set list after regularization:', newmslist)
    return newmslist


def getobsmslist(msfiles, observationnumber):
    """
    Generate a list of measurement sets (MS) belonging to the same observation.

    This function groups measurement sets by their observation ID, which is
    extracted from the filenames of the provided MS files. It then returns
    the list of MS files corresponding to the specified observation number.

    Parameters
    ----------
    msfiles : list of str
        A list of file paths to the measurement sets (MS).
    observationnumber : int
        The index of the observation to extract (0-based).

    Returns
    -------
    list of str
        A list of file paths to the measurement sets belonging to the
        specified observation.

    Notes
    -----
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


def concat_ms_from_same_obs(mslist, outnamebase, colname='DATA', dysco=True, metadata_compression=True):
    """
    Concatenate measurement sets from the same observation into regular frequency grids.

    This function groups measurement sets by observation ID and concatenates them into
    single output measurement sets with regular frequency grids. Missing frequency blocks
    are filled with dummy measurement sets to maintain grid regularity. It is important the the input mslist in in the asceding order of frequency. 
    The function uses DP3 (DPPP) for the concatenation process.

    Parameters
    ----------
    mslist : list of str
        List of input measurement set paths to be concatenated.
    outnamebase : str
        Base name for output concatenated measurement sets. The observation number
        will be appended (e.g., 'outnamebase_0.ms', 'outnamebase_1.ms').
    colname : str, optional
        Name of the data column to concatenate (default: 'DATA').
    dysco : bool, optional
        If True, use Dysco storage manager for compression (default: True).
    metadata_compression : bool, optional
        If True, enable compression for UVW coordinates, antenna data, and flags
        (default: True).

    Returns
    -------
    None
        The function writes concatenated measurement sets to disk but returns nothing.

    Notes
    -----
    - The function uses DP3 (DPPP) for the concatenation process.
    - Missing measurement sets or those without the specified column are replaced
    with dummy entries labeled 'missing<number>'.
    - Existing output measurement sets are removed before creation.
    - The function assumes `number_of_unique_obsids`, `getobsmslist`, `add_dummyms`,
    `mscolexist`, and `run` are defined elsewhere in the codebase.
    """

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
                shutil.rmtree(msoutconcat, ignore_errors=True)
        run(cmd, log=False)
        fix_uvw([msoutconcat])
    return


def phaseup(msinlist, datacolumn='DATA', superstation='core', start=0, dysco=True, metadata_compression=True):
    """
    Phase up stations into a superstation.

    Parameters
    ----------
    msinlist : list
        list of input Measurement Sets to iterate over.
    datacolumn : str
        the input data column to phase up data from.
    superstation : str
        stations to phase up. Can be 'core' or 'superterp'.
    start : int
        selfcal cylce that is being started from. Phaseup will only occur if start == 0.
    dysco : bool
        dysco compress the output dataset.
    metadata_compression : bool, optional
        Enable Measurement Set metadata compression in the output.

    Returns
    -------
    msoutlist : list
        list of output Measurement Sets.
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
                shutil.rmtree(msout, ignore_errors=True)
                time.sleep(2)  # wait for the directory to be removed
            terminal_print('Command:', cmd)
            run(cmd)
    return msoutlist


def findfreqavg(ms, imsize, bwsmearlimit=1.0, msinnchan=None):
    """
    Find the frequency averaging factor for a Measurement Set given a bandwidth smearing constraint.

    Parameters
    ----------
    ms : str
        path to the Measurement Set.
    imsize : float
        size of the image in pixels.
    bwsmearlimit : float
        the fractional acceptable bandwidth smearing.
    msinnchan : int or None, optional
        Override the number of input channels used to calculate the averaging factor.

    Returns
    -------
    avgfactor : int
        the frequency averaging factor for the Measurement Set.
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        bwsmear = bandwidthsmearing(np.median(t.getcol('CHAN_WIDTH')), np.min(t.getcol('CHAN_FREQ')[0]), float(imsize), verbose=False)
        nfreq = len(t.getcol('CHAN_FREQ')[0])
    
    if msinnchan is not None:
        nfreq = msinnchan
    
    avgfactor = 0
    for count in range(2, 21):  # try average values between 2 to 20
        if bwsmear < (bwsmearlimit / float(count)):  # factor X avg
            if nfreq % count == 0:
                avgfactor = count
    return avgfactor


def average(mslist, freqstep, timestep=None, start=0, msinnchan=None, msinstartchan=0.,
            phaseshiftbox=None, msinntimes=None, msinstarttimeslot=None, makecopy=False,
            make_extract=False, delaycal=False, freqresolution='195.3125kHz',
            dysco=True, cmakephasediffstat=False, dataincolumn='DATA',
            removeinternational=False, removemostlyflaggedstations=False, 
            aoflagger=False, aoflaggerbeforeavg=True, aoflagger_strategy=None,
            metadata_compression=True, flag_antennas=None, args=None):
    """
    Averages Measurement Sets (MS) in frequency and/or time using DP3, with options for filtering, flagging, and phase shifting.

    Parameters
    ----------
    mslist : list of str
        List of input Measurement Set filenames.
    freqstep : list
        List of frequency averaging steps or resolutions for each MS. Can be integer (number of channels) or string with units (e.g., '195.3125kHz').
    timestep : int or str, optional
        Time averaging step. Integer (number of time slots) or string with units (e.g., '10s'). Default is None (no time averaging).
    start : int, optional
        Start index for self-calibration cycle. Default is 0.
    msinnchan : int, optional
        Number of input channels to use from each MS. Default is None (use all).
    msinstartchan : int or float, optional
        Starting channel index for input MS. Default is 0.
    phaseshiftbox : str or None, optional
        Region or reference for phase shifting. If 'align', aligns to the first MS's phase center. Default is None.
    msinntimes : int, optional
        Number of input time slots to use from each MS. Default is None (use all).
    msinstarttimeslot : int, optional
        Starting time slot index for input MS. Default is None.
    makecopy : bool, optional
        If True, output MS will have '.copy' suffix. Default is False.
    make_extract : bool, optional
        If True, output MS will have '.extracted' suffix. Default is False.
    delaycal : bool, optional
        If True, perform delay calibration. Default is False.
    freqresolution : str, optional
        Frequency resolution for averaging (e.g., '195.3125kHz'). Default is '195.3125kHz'.
    dysco : bool, optional
        If True, use Dysco storage manager for output MS. Default is True.
    cmakephasediffstat : bool, optional
        If True, output MS will have '.avgphasediffstat' suffix. Default is False.
    dataincolumn : str, optional
        Name of the data column to use in input MS. Default is 'DATA'.
    removeinternational : bool, optional
        If True, remove international stations from MS. Default is False.
    removemostlyflaggedstations : bool, optional
        If True, remove stations with high flagging percentage. Default is False.
    aoflagger : bool, optional
        If True, apply AOFlagger for RFI flagging. Default is False.
    aoflaggerbeforeavg : bool, optional
        If True, run AOFlagger before averaging. Default is True.
    aoflagger_strategy : str or None, optional
        AOFlagger strategy file or name. Default is None.
    metadata_compression : bool, optional
        If True, enable metadata compression in output MS. Default is True.
    flag_antennas : list or None, optional
        List of antennas to flag/remove. Default is None.
    args : dict, optional
        Parsed facetselfcal options. Required when ``removemostlyflaggedstations``
        or ``flag_antennas`` is set (``removemostlyflaggedstations_percentage``
        is used).

    Returns
    -------
    list of str
        List of output Measurement Set filenames (averaged or original, depending on options).

    Raises
    ------
    Exception
        If input parameters are inconsistent or unsupported units are provided.
    """    
    # sanity check
    if len(mslist) != len(freqstep):
        terminal_print('Hmm, made a mistake with freqstep?')
        raise Exception('len(mslist) != len(freqstep)')

    # prevent GMRT data from being in time because of UVW coordinates issue
    # DP3 fills in UVW coordinates incorrectly for GMRT data when time gaps are present
    # if timestep is not None and args['telescope'] == 'GMRT':
    #    if timestep > 1:
    #        print('Time averaging cannot be used for GMRT data due to UVW issues')
    #        raise Exception('Time averaging cannot be used for GMRT data due to UVW issues')

    outmslist = []
    for ms_id, ms in enumerate(mslist):
        if (int(''.join([i for i in str(freqstep[ms_id]) if i.isdigit()])) > 0) or (timestep is not None) or (
                msinnchan is not None) or \
                (phaseshiftbox is not None) or (msinntimes is not None) or (msinstarttimeslot is not None) \
                or removeinternational or removemostlyflaggedstations or aoflagger:  # if this is True then average
            
            # set this first, change name if needed below
            msout = ms + '.avg'
            if makecopy:
                msout = ms + '.copy'
            if make_extract:
                msout = ms + '.extracted'
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
                    with table(mslist[0] + '::FIELD', ack=False) as t:
                        ra_ref, dec_ref = t.getcol('PHASE_DIR').squeeze() # get the reference direction of the first MS in radians
                    terminal_print(f'Aligning phase center to {ra_ref}, {dec_ref} (in radians) of first MS {mslist[0]}')
                    # shift to the first MS's phase center
                    cmd += ' shift.phasecenter=\\[' + str(ra_ref) + ',' + str(dec_ref) + '\\] '
                else:
                    cmd += ' shift.phasecenter=\\[' + getregioncenter(phaseshiftbox, standardbox=False) + '\\] '
            else:
                if removeinternational:
                    cmd += ' steps=[f,av] '
                    cmd += " f.type=filter f.baseline='[CR]S*&' f.remove=True "
                else:
                    cmd += ' steps=[av] '
            if removemostlyflaggedstations or flag_antennas is not None:
                flagstationlist = return_antennas_highflaggingpercentage(ms, percentage=args['removemostlyflaggedstations_percentage'])
                flagstationlist = flagstationlist + flag_antennas if flag_antennas is not None else flagstationlist
                # remove duplicates
                flagstationlist = list(set(flagstationlist))
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
                        terminal_print('For frequency averaging only units of (k/M)Hz are allowed, used:', freqstepstrnot)
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
                        terminal_print('For time averaging only units of s(ec) are allowed, used:', timestepstrnot)
                        raise Exception('For time averaging only units of "s(ec)" are allowed')
                    cmd += 'av.timeresolution=' + str(timestepstr) + ' '

            if msinnchan is not None:
                cmd += 'msin.nchan=' + str(msinnchan) + ' '
                cmd += 'msin.startchan=' + str(msinstartchan) + ' '
            if msinntimes is not None:
                cmd += 'msin.ntimes=' + str(msinntimes) + ' '
            if msinstarttimeslot is not None:
                cmd += 'msin.starttimeslot=' + str(msinstarttimeslot) + ' '
            if aoflagger:
                cmd += 'ao.type=aoflag '
                cmd += 'ao.keepstatistics=False '
                cmd += 'ao.memoryperc=50 '
                cmd += 'ao.overlapperc=10 '
                if aoflagger_strategy is not None:
                    if os.path.isfile(aoflagger_strategy): # try full location first
                        cmd += 'ao.strategy=' +  aoflagger_strategy + ' '
                    else: # try strategy in flagging_strategies
                        cmd += 'ao.strategy=' + f'{datapath}/flagging_strategies/' + aoflagger_strategy + ' '
                if aoflaggerbeforeavg:
                   cmd = cmd.replace("steps=[", "steps=[ao,") # insert s first step
                else:
                   cmd = cmd.replace("av] ", "av,ao] ") # insert as last step (av was previous last step)
            if start == 0:
                terminal_print('Average with default WEIGHT_SPECTRUM:', cmd)
                if os.path.isdir(msout):
                    shutil.rmtree(msout, ignore_errors=True)
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
                        terminal_print('For frequency averaging only units of (k/M)Hz are allowed, used:', freqstepstrnot)
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
                        terminal_print('For time averaging only units of s(ec) are allowed, used:', timestepstrnot)
                        raise Exception('For time averaging only units of "s(ec)" are allowed')
                    cmd += 'av.timeresolution=' + str(timestepstr) + ' '
            if msinnchan is not None:
                cmd += 'msin.nchan=' + str(msinnchan) + ' '
                cmd += 'msin.startchan=' + str(msinstartchan) + ' '
            if msinntimes is not None:
                cmd += 'msin.ntimes=' + str(msinntimes) + ' '
            if msinstarttimeslot is not None:
                cmd += 'msin.starttimeslot=' + str(msinstarttimeslot) + ' '

            if start == 0:
                with table(ms) as t:
                    if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():  # check if present otherwise this is not needed
                        terminal_print('Average with default WEIGHT_SPECTRUM_SOLVE:', cmd)
                        if os.path.isdir(msouttmp):
                            shutil.rmtree(msouttmp, ignore_errors=True)
                            time.sleep(2)  # wait for the directory to be removed
                        run(cmd)

                        # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                        with table(msout, readonly=False) as t2:
                            terminal_print('Adding WEIGHT_SPECTRUM_SOLVE')
                            #desc = t2.getcoldesc('WEIGHT_SPECTRUM')
                            #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
                            #t2.addcols(desc)
                            addcol(t2, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

                            with table(msouttmp, readonly=True) as t3:
                                imweights = t3.getcol('WEIGHT_SPECTRUM')
                                t2.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

                        # clean up
                        shutil.rmtree(msouttmp, ignore_errors=True)

            outmslist.append(msout)
        else:
            outmslist.append(ms)  # so no averaging happened
        
   
    if start > 0 and makecopy:  # only fix UVW if this is not the first selfcal cycle
        for ms in outmslist:
            if not os.path.isdir(ms):
                terminal_print('No MS found:', ms)
                terminal_print('Maybe you made a mistake and you did not set --start=0?')
                raise Exception('MS found and averaging did not produce it because start > 0')
                
    # fix MeerKAT UVW coordinates (needs to be done each time we average with DP3)
    fix_uvw(outmslist) 

    return outmslist


def makemslist(mslist):
    """
    Create the input list for e.g. ddf-pipeline.

    Parameters
    ----------
    mslist : list
        list of input Measurement Sets

    Returns
    -------
    None
    """
    Path('mslist.txt').unlink(missing_ok=True)
    f = open('mslist.txt', 'w')
    for ms in mslist:
        f.write(str(ms) + '\n')
    f.close()
    return


def cleanup(mslist):
    """
    Clean up directory

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths and other run products to remove.
    """
    for ms in mslist:
        shutil.rmtree(ms, ignore_errors=True)

    Path('*first-residual.fits').unlink(missing_ok=True)
    Path('*psf.fits').unlink(missing_ok=True)
    Path('*-00*-*.fits').unlink(missing_ok=True)
    Path('*dirty.fits').unlink(missing_ok=True)
    Path('solintimage*model.fits').unlink(missing_ok=True)
    Path('solintimage*residual.fits').unlink(missing_ok=True)
    Path('*pybdsf.log').unlink(missing_ok=True)
    return


def removestartendms(ms, starttime=None, endtime=None, dysco=True, metadata_compression=True):
    """
    Removes the start and/or end times from a Measurement Set (MS) and processes it using DP3.

    This function creates a new MS with the specified time range removed and optionally applies
    DYSCO compression. It also creates a WEIGHT_SPECTRUM_SOLVE column based on the processed data.

    Parameters
    ----------
    ms : str
        The path to the input Measurement Set (MS).
    starttime : str, optional
        The start time to cut from the MS in a format recognized by DP3.
        If None, no start time is specified. Defaults to None.
    endtime : str, optional
        The end time to cut from the MS in a format recognized by DP3.
        If None, no end time is specified. Defaults to None.
    dysco : bool, optional
        Whether to use DYSCO compression for the output MS. Defaults to True.
    metadata_compression : bool, optional
        Enable Measurement Set metadata compression in the output.

    Returns
    -------
    None

    Notes
    -----
    Side Effects:
    - Creates a new MS with the '.cut' suffix.
    - Temporarily creates a '.cuttmp' MS, which is removed after processing.
    - Adds a new column 'WEIGHT_SPECTRUM_SOLVE' to the '.cut' MS.
    - Prints the DP3 commands executed.
    - Removes any pre-existing '.cut' or '.cuttmp' directories.
    - The function uses the DP3 tool for processing the MS.
    - The `check_phaseup_station` function is used to determine if UVW compression should be disabled.
    - The `run` function is used to execute the DP3 commands.
    - The `table` function from the casacore library is used to manipulate the MS columns.
    """
    # chdeck if output is already there and remove
    if os.path.isdir(ms + '.cut'):
        shutil.rmtree(ms + '.cut', ignore_errors=True)
    if os.path.isdir(ms + '.cuttmp'):
        shutil.rmtree(ms + '.cuttmp', ignore_errors=True)
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
    terminal_print('Command:', cmd)
    run(cmd)
    fix_uvw([ms + '.cut'])

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
    terminal_print('Command:', cmd)
    run(cmd)
    fix_uvw([ms + '.cuttmp'])

    # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
    terminal_print('Adding WEIGHT_SPECTRUM_SOLVE')
    with table(ms + '.cut', readonly=False) as t:
        #desc = t.getcoldesc('WEIGHT_SPECTRUM')
        #desc['name'] = 'WEIGHT_SPECTRUM_SOLVE'
        #t.addcols(desc)
        addcol(t, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')

        with table(ms + '.cuttmp', readonly=True) as t2:
            imweights = t2.getcol('WEIGHT_SPECTRUM')
        t.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)

    # clean up
    shutil.rmtree(ms + '.cuttmp', ignore_errors=True)

    return


def archive(mslist, outtarname, regionfile, fitsmask, imagename, args, dysco=True, mergedh5_i=None, 
            facetregionfile=None, metadata_compression=True): 
    """
    Archives calibrated measurement sets and related files into a tarball.

    This function processes a list of measurement sets (MS), applies calibration and optional compression,
    and then archives the resulting files along with specified auxiliary files into a compressed tarball.

    Parameters
    ----------
    mslist : list of str
        List of measurement set filenames to process and archive.
    outtarname : str
        Name of the output tarball file.
    regionfile : str or None
        Path to a region file to include in the archive, if it exists.
    fitsmask : str or None
        Path to a FITS mask file to include in the archive, if it exists.
    imagename : str
        Name of the image file to include in the archive.
    args : dict
        Parsed facetselfcal options; ``DDE`` is used.
    dysco : bool, optional
        Whether to use Dysco compression for output measurement sets. Defaults to True.
    mergedh5_i : list of str or None, optional
        List of merged HDF5 files to include in the archive, if provided.
    facetregionfile : str or None, optional
        Path to a facet region file to include in the archive, if it exists.
    metadata_compression : bool, optional
        Whether to enable metadata compression. Defaults to True.

    Returns
    -------
    None

    Raises
    ------
    None explicitly, but may raise exceptions if external commands fail.

    Notes
    -----
    Side Effects:
    - Creates calibrated copies of the input measurement sets with optional compression.
    - Removes any existing output tarball with the same name before creating a new one.
    - Archives specified files into a compressed tarball.
    - Removes temporary calibrated measurement sets after archiving.
    - Logs the creation of the tarball.
    """
    path = '/disks/ftphome/pub/vanweeren'
    for ms in mslist:
        msout = ms + '.calibrated'
        if os.path.isdir(msout):
            shutil.rmtree(msout, ignore_errors=True)
            time.sleep(2)  # give it some time to remove
        cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msout=' + msout + ' steps=[] '
        if check_phaseup_station(ms): cmd += 'msout.uvwcompression=False '
        if not args['DDE']: # this works for DDE solves because then the default is DATA
            cmd += 'msin.datacolumn=CORRECTED_DATA '
        if dysco:
            cmd += 'msout.storagemanager=dysco '
            cmd += 'msout.storagemanager.weightbitrate=16 '
        if not metadata_compression:
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '
        run(cmd)
        # fix MeerKAT UVW coordinates (needs to be done each time we create a new MS)
        fix_uvw([msout]) 
   

    imagename_pb = imagename.replace('.fits', '-pb.fits')
    
    msliststring = ' '.join(map(str, glob.glob('*.calibrated')))
    cmd = 'tar -zcf ' + outtarname + ' ' + msliststring + ' logs/selfcal.log ' + imagename + ' '

    if os.path.isfile(imagename_pb):
        cmd += imagename_pb + ' '

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
        Path(outtarname).unlink(missing_ok=True)
    get_logger().info('Creating archived calibrated tarball: ' + outtarname)
    run(cmd)

    for ms in mslist:
        msout = ms + '.calibrated'
        shutil.rmtree(msout, ignore_errors=True)
        time.sleep(2)  # give it some time to remove
    return


def create_splitted_ms(ms, columns_to_create, solve_msinnchan, solve_msinstartchan,
                      dysco=True, modelstoragemanager=None, incol='DATA', 
                      metadata_compression=True, ncpu_max=8):
    """
    Create a temporary Measurement Set (MS) with selected frequency channels and specified columns.

    Parameters
    ----------
    ms : str
        Path to the input Measurement Set.
    columns_to_create : list of str
        List of column names to create and copy into the new MS.
    solve_msinnchan : int
        Number of frequency channels to include in the split MS.
    solve_msinstartchan : int
        Starting channel index for the split.
    dysco : bool, optional
        Whether to use DYSCO compression for the output MS. Defaults to True.
    modelstoragemanager : str or None, optional
        Storage manager to use for model columns. Defaults to None.
    incol : str, optional
        Name of the input data column. Defaults to 'DATA'.
    metadata_compression : bool, optional
        Whether to enable metadata compression. Defaults to True.
    ncpu_max : int, optional
        Maximum number of CPU threads to use. Defaults to 8.

    Returns
    -------
    str
        Path to the newly created temporary Measurement Set.
    """
    # create a new temporary MS with the average function
    ms_tmp = average([ms], freqstep=[1], makecopy=True, msinnchan=solve_msinnchan, msinstartchan=solve_msinstartchan,
                        dysco=dysco, metadata_compression=metadata_compression)[0]
    
    # set DATA XY and YX to zero to avoid error from the Stokes-I storageManager
    run("taql" + " 'update " + ms_tmp + " set DATA[,1]=(0+0i)'", taql=True)
    run("taql" + " 'update " + ms_tmp + " set DATA[,2]=(0+0i)'", taql=True)
    
    for col in columns_to_create:                  
        cmdcol = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(), ncpu_max])) + \
                    ' msin=' + ms_tmp + ' msout=. '
    
        cmdcol += 'steps=[] msout.datacolumn=' + col + ' '
        if col == incol: # in this case we can use the dysco storage manager
            if dysco:
                cmdcol += 'msout.storagemanager=dysco '
                cmdcol += 'msout.storagemanager.weightbitrate=16 '
        elif modelstoragemanager is not None: 
            if modelstoragemanager == 'sisco_stokes_i':
                cmdcol += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i '
            elif modelstoragemanager == 'sisco_diagonal':
                cmdcol += 'msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal '
            else:          
                cmdcol += 'msout.storagemanager=' + modelstoragemanager + ' '
        terminal_print('DP3 column command:', cmdcol)
        run(cmdcol)
        # now copy over the column to the temporary MS
        terminal_print('=== Copying column ' + col + ' to temporary MS \n\n')
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


def get_startchan_nchan(freqs, startfreq, endfreq):
    """
    Try to automatically set arg['msinstartchan'] and arg['msinnchan']

    Parameters
    ----------
    freqs : ndarray
        Channel frequencies in hertz.
    startfreq : float
        Requested lower frequency in hertz.
    endfreq : float
        Requested upper frequency in hertz.

    Returns
    -------
    tuple of int
        Starting channel index and number of channels in the selected range.
    """
    msinstartchan = (np.abs(freqs - startfreq)).argmin()
    highestchannel = (np.abs(freqs - endfreq)).argmin()
    msinnchan = highestchannel - msinstartchan + 1
    # make sure args['msinnchan'] can be divided by 4 and 3, and if not adjust it downwards
    while msinnchan % 4 != 0 or msinnchan % 3 != 0:
        msinnchan -= 1
    return msinstartchan, msinnchan


def nested_mslistforimaging(mslist, stack=False, announce=True):
    """
    Create nested MS groups used by imaging workflows.

    Parameters
    ----------
    mslist : list
        Measurement Set paths.
    stack : bool, optional
        Group inputs for stacked imaging.
    announce : bool, optional
        Print the stacked imaging groups.

    Returns
    -------
    list
        Nested Measurement Set groups.
    """
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
        if announce:
            terminal_print(f'Found {len(mslistreturn)} imaging groups: ', mslistreturn)
        return mslistreturn  # has format [[ms1.ms,..],[ms2.ms,..],[...]]


def _write_stack_imaging_group_manifest(mslist):
    groups = nested_mslistforimaging(mslist, stack=True, announce=False)
    manifest_path = Path("misc") / "stack_imaging_groups.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "groups": [
            {
                "stack_id": f"{stack_id:02d}",
                "measurement_sets": [str(ms) for ms in group],
            }
            for stack_id, group in enumerate(groups)
        ],
    }
    temporary_path = manifest_path.with_name(
        ".{}-tmp".format(manifest_path.name)
    )
    temporary_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
