"""Read-only queries on Measurement Sets: antennas, channel frequencies,
observation IDs, phased-up stations, long baselines and the reference antenna."""
import os

import numpy as np
from casacore.tables import table  # type: ignore

from submods.h5_helpers.h5_info import refant_candidates
from submods.runtime_utils import terminal_print


def getAntennas(ms):
    """
    Retrieve a list of antenna names from a Measurement Set (MS).

    Parameters
    ----------
    ms : str
        The path to the Measurement Set (MS) directory.

    Returns
    -------
    list
        A list of antenna names as strings.

    Notes
    -----
    This function accesses the 'ANTENNA' table within the provided
    Measurement Set to extract the antenna names.
    """
    t = table(ms + "::ANTENNA", readonly=True, ack=False)
    antennas = t.getcol('NAME')
    t.close()
    return antennas


def get_chan_freqs(ms):
    """
    Retrieve channel frequencies from the SPECTRAL_WINDOW table of a Measurement Set (MS).

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    tuple
        A tuple of channel frequencies rounded to 0.1 Hz.
    """
    with table(ms + '::SPECTRAL_WINDOW', readonly=True, ack=False) as t:
        chan_freqs = t.getcol('CHAN_FREQ')
    if isinstance(chan_freqs, np.ndarray) and chan_freqs.dtype == object:
        freq_list = np.concatenate([np.array(f).ravel() for f in chan_freqs])
    else:
        freq_list = np.array(chan_freqs).ravel()
    return tuple(np.round(freq_list, 1).tolist())


def number_of_unique_obsids(msfiles):
    """
    Basic function to get numbers of observations based on first part of ms name
    (assumes one uses "_" here)

    Parameters
    ----------
    msfiles : list
        the list of ms

    Returns
    -------
    reval : int
        number of observations
    """
    obsids = []
    for ms in msfiles:
        obsids.append(os.path.basename(ms).split('_')[0])
        terminal_print('Using these observations ', np.unique(obsids))
    return len(np.unique(obsids))


def get_frequencies_from_ms(mslist):
    """
    Retrieve the channel frequencies of the measurement sets (MS) in the provided list.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is an array of channel frequencies (in Hz)
        for the corresponding MS in the input list.

    Raises
    ------
    Exception
        If the channel frequencies cannot be found in any of the MS.
    """
    # if input is string, make it a list
    if isinstance(mslist, str):
        mslist = [mslist]

    t = table(mslist[0] + '::SPECTRAL_WINDOW', ack=False)
    freqs = t.getcol('CHAN_FREQ')[0]  # in Hz
    t.close()
    return freqs


def check_phaseup_station(ms):
    """
    Check if the Measurement Set contains a superstation.

    Parameters
    ----------
    ms : str
        path to the Measurement Set.

    Returns
    -------
    None
    """
    with table(ms + '::ANTENNA', ack=False) as t:
        antennasms = list(t.getcol('NAME'))
    substr = 'ST'  # to check if a a superstation is present, assume this 'ST' string, usually ST001
    hassuperstation = any(substr in mystring for mystring in antennasms)
    terminal_print(f'Measurement Set: {ms}, contains superstation: {hassuperstation}')
    return hassuperstation


def is_linear_ms(ms):
    """
    Check if the Measurement Set has linear correlations (XX, XY, YX, YY).

    Parameters
    ----------
    ms : str
        path to the Measurement Set.

    Returns
    -------
    bool
        True if CORR_TYPE is [9, 10, 11, 12].
    """
    with table(ms + '::POLARIZATION', ack=False) as t:
        corr_type = t.getcol('CORR_TYPE')
    return np.array_equal(corr_type, [[9, 10, 11, 12]])


def checklongbaseline(ms):
    """
    Check if the Measurement Set contains international stations.

    Parameters
    ----------
    ms : str
        path to the Measurement Set.

    Returns
    -------
    None
    """
    with table(ms + '::ANTENNA', ack=False) as t:
        antennasms = list(t.getcol('NAME'))
    substr = 'DE'  # to check if a German station is present, if yes assume this is long baseline data
    haslongbaselines = any(substr in mystring for mystring in antennasms)
    terminal_print('Contains long baselines?', haslongbaselines)
    return haslongbaselines


def choose_refant_from_ms(mslist, telescope, maxrows=20000, maxchan=64):
    """
    Choose one reference antenna for a whole run: the preferred antenna (see refant_candidates)
    that is present in all Measurement Sets and has the most unflagged data.

    The flags are sampled (at most about maxrows rows and maxchan channels per MS) to keep this fast
    for large data sets. ST001 is chosen if all Measurement Sets contain it, as findrefant_core does.

    Parameters
    ----------
    mslist : list of str
        Paths to the Measurement Sets.
    telescope : str
        Telescope name as in the MS OBSERVATION table.
    maxrows : int, optional
        Approximate number of rows to read per MS.
    maxchan : int, optional
        Approximate number of channels to read per MS.

    Returns
    -------
    str
        Name of the reference antenna.
    """
    unflagged = {}  # antenna name -> summed unflagged fraction of its baselines
    common = None
    for ms in mslist:
        with table(ms + '::ANTENNA', ack=False) as t:
            names = list(t.getcol('NAME'))
        with table(ms, ack=False) as t:
            rowincr = max(1, t.nrows() // maxrows)
            nchan, ncorr = t.getcell('FLAG', 0).shape
            flags = t.getcolslice('FLAG', blc=[0, 0], trc=[nchan - 1, ncorr - 1],
                                  inc=[max(1, nchan // maxchan), 1], rowincr=rowincr)
            ant1 = t.getcol('ANTENNA1', rowincr=rowincr)
            ant2 = t.getcol('ANTENNA2', rowincr=rowincr)
        good = 1. - flags.mean(axis=(1, 2))
        per_ant = np.bincount(ant1, good, len(names)) + np.bincount(ant2, good, len(names))
        for name, value in zip(names, per_ant):
            unflagged[name] = unflagged.get(name, 0.) + value
        common = set(names) if common is None else common & set(names)

    ants = [name for name in unflagged if name in common]
    if 'ST001' in ants:
        return 'ST001'
    candidates = refant_candidates(ants, telescope=telescope)
    return ants[candidates[np.argmax([unflagged[ants[i]] for i in candidates])]]
