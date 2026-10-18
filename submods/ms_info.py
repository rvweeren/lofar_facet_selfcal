"""Read-only queries on Measurement Sets: antennas, channel frequencies,
observation IDs, phased-up stations and long baselines."""
import os

import numpy as np
from casacore.tables import table  # type: ignore

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
