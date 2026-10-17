"""(E)VLA helpers: array configuration, EVLA IF pairs, and splitting an MS
into its spectral windows."""
import os
import shutil

import numpy as np
from casacore.tables import table  # type: ignore

from submods.ms_fixes import get_telescope_from_ms, remove_syspower
from submods.runtime_utils import run, terminal_print


def get_vla_maxconfiguration(mslist):
    """
    Extract the VLA array configuration (A, B, C, or D) from a list of Measurement Sets.

    If multiple configurations are found, returns the one with the longest baselines.
    This determines the pixel scale for imaging.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Set paths.

    Returns
    -------
    str
        The VLA configuration with the longest baselines ('A', 'B', 'C', or 'D').
    """
    # check that the input is a list
    assert isinstance(mslist, list), "Input must be a list of Measurement Sets"
    config_list = []
    for ms in mslist:
        config_list.append(get_vla_configuration(ms))
    if 'A' in config_list:
        return 'A'
    elif 'B' in config_list:
        return 'B'
    elif 'C' in config_list:
        return 'C'
    else:
        return 'D'


def get_vla_configuration(ms):
    """
    Extract the VLA array configuration (A, B, C, or D) from a Measurement Set.

    Calculates the maximum baseline length from the ANTENNA sub-table.

    Parameters
    ----------
    ms : str
        The path to the main Measurement Set directory (e.g., 'my_data.ms').

    Returns
    -------
    str
        'A', 'B', 'C', or 'D' based on the physical antenna distribution.

    Raises
    ------
    FileNotFoundError
    If the ANTENNA sub-table cannot be found at the expected path.
    """
    antenna_table_path = os.path.join(ms, 'ANTENNA')
    
    if not os.path.exists(antenna_table_path):
        raise FileNotFoundError(f"Could not find the ANTENNA sub-table at: {antenna_table_path}")
    
    # Open the antenna sub-table and extract positions
    with table(antenna_table_path, readonly=True, ack=False) as t:
        positions = t.getcol('POSITION')  # Shape: (num_antennas, 3) or (3, num_antennas)
    
    # Handle both shapes just in case casacore format varies by version
    if positions.shape[1] == 3 and positions.shape[0] != 3:
        # If shape is (N, 3), convert to (3, N) for the loop
        positions = positions.T
        
    num_ants = positions.shape[1]
    max_baseline_km = 0.0
    
    # Compute the maximum physical distance between any pair of antennas
    for i in range(num_ants):
        for j in range(i + 1, num_ants):
            dist_km = np.linalg.norm(positions[:, i] - positions[:, j]) / 1000.0
            if dist_km > max_baseline_km:
                max_baseline_km = dist_km
                
    # Classify based on official NRAO maximum baseline bounds
    if max_baseline_km > 15.0:
        return 'A'
    elif max_baseline_km > 4.0:
        return 'B'
    elif max_baseline_km > 1.5:
        return 'C'
    else:
        return 'D'


def get_EVLA_IF_pair(ms):
    """
    Return the EVLA IF pair encoded in a single-SPW measurement set.

    EVLA has two independent data streams, represented by the A/C and B/D
    IF pairs. The pair is read from the first spectral-window ``NAME`` and is
    returned as ``"AC"`` or ``"BD"`` so measurement sets can be sorted by
    IF pair.

    Parameters
    ----------
    ms : str
        Path to the measurement set.

    Returns
    -------
    str
        The detected IF pair, either ``"AC"`` or ``"BD"``.

    Raises
    ------
    ValueError
    If the spectral-window name does not identify an A/C or B/D pair.
    """
    # check telescope is EVLA
    if get_telescope_from_ms(ms) != 'EVLA':
        terminal_print('IF pair detection is only supported for EVLA measurement sets, but the telescope for MS ' + ms + ' is ' + get_telescope_from_ms(ms))
        raise ValueError('IF pair detection is only supported for EVLA measurement sets.')

    with table(ms + '::SPECTRAL_WINDOW', readonly=True, ack=False) as t:
        name = t.getcol('NAME')[0]
    # format is 'EVLA_X#B2D2#53'
    # strip off the part between the # symbols
    name = name.split('#')[1]

    # if characetrs A and C are in then return 'AC'
    if 'A' in name and 'C' in name:
        return 'AC'
    elif 'B' in name and 'D' in name:
        return 'BD'
    else:
        # there is some problem
        raise ValueError('Could not determine IF pair for EVLA MS ' + ms + ', NAME=' + name)


def split_ms_spws(ms, dysco=True):
    """
    Split a measurement set into one output measurement set per SPW.

    The input measurement set is inspected to determine its number of spectral
    windows. Each resulting file is named ``<input>_spwNNN.ms`` (or, for an
    input ending in ``.ms``, ``<input base>_spwNNN.ms``), with the SPW index
    zero-padded to three digits.

    Parameters
    ----------
    ms : str
        Path to the input measurement set.
    dysco : bool, optional
        Whether to use the dysco storage manager for the output measurement sets.    

    Returns
    -------
    None
        The split measurement sets are written to disk.
    """

    # get SPWids from the MS
    with table(ms + '::DATA_DESCRIPTION', readonly=True) as t:
        spw_ids = t.getcol('SPECTRAL_WINDOW_ID')

    # check that there are more than one unique SPW, otherwise do nothing
    if len(np.unique(spw_ids)) <= 1:
        terminal_print('Only one unique SPW found in ' + ms + ', no splitting needed.') 
        return

    # get the unique DATA_DESC_IDs actually present in the main data rows
    with table(ms, readonly=True) as t:
        # Using np.unique isolates only the IDs that contain actual rows of data
        active_ddids = np.unique(t.getcol("DATA_DESC_ID"))

    # map those DDIDs back to their actual Spectral Window IDs
    with table(ms + '::DATA_DESCRIPTION', readonly=True) as t:
        # Fetch the mapping array where row index = DDID, value = SPW_ID
        spw_mapping = t.getcol("SPECTRAL_WINDOW_ID")
    
        # Filter the map using our active DDIDs
        active_spws = [spw_mapping[ddid] for ddid in active_ddids]

    terminal_print(f"Spectral Windows with actual data: {sorted(list(set(active_spws)))}")

    # remove syspower column from the MS, if it exists, to avoid DP3 slowdowns
    remove_syspower(ms)

    for spw in sorted(list(set(active_spws))):    
        # use zfill to make sure the spw number is 3 digits
        # put the ms in the current working directory, not in the same directory as the input ms, to avoid permission issues
        ms_basename = os.path.basename(ms)
        if ms_basename.endswith('.ms') or ms_basename.endswith('.MS'):
            output_ms = os.path.join(os.getcwd(), ms_basename[:-3] + '_spw' + str(spw).zfill(3) + '.ms')
        else:
            output_ms = os.path.join(os.getcwd(), ms_basename + '_spw' + str(spw).zfill(3) + '.ms')

        #with table(ms, readonly=True) as orig_ms:
        #    # Select the rows matching your desired SPW
        #    query = """
        #        SELECT FROM $orig_ms
        #        WHERE DATA_DESC_ID IN [
        #            SELECT FROM ::DATA_DESCRIPTION
        #            WHERE SPECTRAL_WINDOW_ID == $spw
        #            GIVING [ROWID()]
        #        ]
        #    """
        #    selected_rows = taql(query)

        #    # Use .copy() with deep/valuecopy flags to force a new physical MS on disk
        #    # delete the output MS if it already exists
        if os.path.exists(output_ms):
            if os.path.isdir(output_ms):
                shutil.rmtree(output_ms)
            else:
                os.remove(output_ms)
        #    selected_rows.copy(output_ms, deep=True, valuecopy=True)
        #    selected_rows.close()
        cmddp3 = f"DP3 msin={ms} msout={output_ms} steps=[] msin.band={spw} "
        cmddp3 += "msout.uvwcompression=False msout.antennacompression=False "
        if dysco:
            cmddp3 += f"msout.storagemanager=dysco msout.storagemanager.weightbitrate=16 "
        terminal_print('DP3 command:', cmddp3)
        run(cmddp3)


def collect_all_frequencies(mslist):
    """
    Collect all unique frequencies from a list of Measurement Sets.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Set paths.

    Returns
    -------
    numpy.ndarray
        Sorted array of unique channel frequencies in Hz.
    """
    frequencies = set()
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', readonly=True) as t:
            freqs = t.getcol('CHAN_FREQ')[0]# Get the first channel frequency for each SPW    
            terminal_print(f'Frequencies in {ms}: {freqs}')
            frequencies.update(freqs)
    return np.sort(list(frequencies))
