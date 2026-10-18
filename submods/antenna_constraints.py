"""Antenna constraints for calibration: station groupings per telescope
(LOFAR, MeerKAT, ...) and baseline selection strings."""
import os
import sys

import numpy as np
from astropy.io import ascii

from submods.runtime_utils import terminal_print

# facetselfcal package directory (holds data/), same as datapath in main.py
datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal')


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
        terminal_print('Wrong input detected, ctype needs to be in core,remote,or all')
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


def filter_baseline_str_removestations(stationlist):
    """
    Generates a baseline filter string to exclude specific stations from processing.

    This function constructs a string that can be used to filter out baselines
    involving specific stations in a radio interferometry dataset. The filter
    string is formatted to exclude baselines that include any of the stations
    provided in the input list.

    Parameters
    ----------
    stationlist : list of str
        A list of station names to be excluded.

    Returns
    -------
    str
        A formatted baseline filter string that excludes the specified stations.
    """
    fbaseline = "'"
    for station_id, station in enumerate(stationlist):
        fbaseline = fbaseline + "!" + station + "&&*"
        if station_id + 1 < len(stationlist):
            fbaseline = fbaseline + ";"
    return fbaseline + "'"


def max_in_str(s):
    """
    Extracts the maximum positive integer value from a comma-separated string of key:value pairs.
    Each entry in the input string should be in the format 'key:value', where 'value' is expected to be a positive integer.
    If any value is not a positive integer, a ValueError is raised.
    If no valid positive integer values are found, a ValueError is raised.

    Parameters
    ----------
    s : str
        A comma-separated string of key:value pairs (e.g., "a:3,b:7,c:2").

    Returns
    -------
    int
        The maximum positive integer value found among the values.

    Raises
    ------
    ValueError
        If any entry is malformed, contains a non-positive integer value, or if no valid values are found.
    """

    values = []
    for x in s.split(','):
        try:
            key, val = x.split(':', 1)
            if val.isdigit() and int(val) > 0:   # only allow positive integers
                values.append(int(val))
            else:
                raise ValueError(f"Invalid value (must be positive int): {val}")
        except ValueError as e:
            raise ValueError(f"Invalid entry '{x}': {e}")
    if not values:
        raise ValueError("No valid positive integer values found")
    return max(values)


def antennaconstraintstr(ctype, antennasms, HBAorLBA, useforresetsols=False, telescope='LOFAR'):
    """
    Formats an anntena constraint string in a DP3-suitable format.

    Parameters
    ----------
    ctype : str
        constraint type. Can be superterp, core, coreandfirstremotes, remote, alldutch, all, international, core-remote, coreandallbutmostdistantremotes, alldutchandclosegerman or alldutchbutnoST001.
    antennasms : list
        antennas present in the Measurement Set.
    HBAorLBA : str
        indicate HBA or LBA data. Can be HBA or LBA.
    useforresetsols : bool
        whether it will be used with reset solution. Removes antennas that are not in antennasms.
    telescope : str
        telescope name, used to check if MeerKAT data is used

    Returns
    -------
    antstr : str
        antenna constraint string for DP3.
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

        terminal_print('Invalid input, ctype can only be "superterp" or "core"')
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
                      'SE607LBA', 'UK608LBA', 'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA', 'BG616LBA', 'ST001']
        if ctype == 'international':
            antstr = ['DE601LBA', 'DE602LBA', 'DE603LBA', 'DE604LBA', 'DE605LBA', 'DE609LBA', 'FR606LBA',
                      'SE607LBA', 'UK608LBA', 'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA', 'BG616LBA']
        if ctype == 'closeinternational':
            antstr = ['DE601LBA', 'DE602LBA']
        if ctype == 'distantinternational':
            antstr = ['DE603LBA', 'DE604LBA', 'DE605LBA', 'DE609LBA', 'FR606LBA', 'SE607LBA', 'UK608LBA',
                      'PL610LBA', 'PL611LBA', 'PL612LBA', 'IE613LBA', 'LV614LBA', 'BG616LBA']          
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
                      'SE607HBA', 'UK608HBA', 'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA', 'BG616HBA', 'ST001']
        if ctype == 'international':
            antstr = ['DE601HBA', 'DE602HBA', 'DE603HBA', 'DE604HBA', 'DE605HBA', 'DE609HBA', 'FR606HBA',
                      'SE607HBA', 'UK608HBA', 'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA', 'BG616HBA']
        if ctype == 'closeinternational':
            antstr = ['DE601HBA', 'DE602HBA']
        if ctype == 'distantinternational':
            antstr = ['DE603HBA', 'DE604HBA', 'DE605HBA', 'DE609HBA', 'FR606HBA', 'SE607HBA', 'UK608HBA',
                      'PL610HBA', 'PL611HBA', 'PL612HBA', 'IE613HBA', 'LV614HBA', 'BG616HBA']        
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
