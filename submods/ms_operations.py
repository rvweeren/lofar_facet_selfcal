"""Measurement Set operations and queries (phased-up stations, long baselines)."""
from casacore.tables import table  # type: ignore

from submods.runtime_utils import terminal_print


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
