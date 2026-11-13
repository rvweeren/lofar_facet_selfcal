"""Telescope identification for runs that can combine Measurement Sets from different telescopes.

facetselfcal used to take the telescope of the first MS for the whole run. These helpers give the
telescope of each MS (read from its OBSERVATION table), so that per-MS steps follow the telescope of
that MS and joint steps (imaging with WSClean) can decide on the set of telescopes in the run.
"""

import re
from collections import Counter

import numpy as np
from casacore.tables import table  # type: ignore

# telescopes with a primary beam model in EveryBeam that WSClean can apply (-apply-primary-beam,
# -apply-facet-beam). Historical VLA data (TELESCOPE_NAME 'VLA', CASA importvla) are not supported by
# EveryBeam ("telescope type VLA is not implemented") and are corrected afterwards with VLA_pbcor.
EVERYBEAM_TELESCOPES = frozenset({'LOFAR', 'MeerKAT', 'GMRT', 'EVLA', 'MWA'})

# telescopes for which facetselfcal can do a primary beam correction of the image itself
# (manual pbcor), mapped to the name of the correction function in submods/primary_beam.py.
# VLA and EVLA share one correction (VLA_pbcor picks the coefficients per MS).
PBCOR_FUNCTION_NAME = {'MeerKAT': 'MeerKAT_pbcor', 'GMRT': 'uGMRT_pbcor', 'VLA': 'VLA_pbcor',
                       'EVLA': 'VLA_pbcor'}


def _as_list(mslist):
    """Return ``mslist`` as a list (a single MS path becomes a one-element list)."""
    if isinstance(mslist, (str, bytes)):
        return [mslist]
    return list(mslist)


def telescope_of(ms):
    """
    Telescope name of a Measurement Set.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    str
        TELESCOPE_NAME from the OBSERVATION table (for example 'LOFAR', 'MeerKAT', 'GMRT', 'EVLA').
    """
    with table(str(ms) + '::OBSERVATION', ack=False) as t:
        return str(t.getcol('TELESCOPE_NAME')[0])


def telescopes_of(mslist):
    """
    Telescope name of every Measurement Set in a list.

    Parameters
    ----------
    mslist : str or list of str
        Measurement Set path(s).

    Returns
    -------
    list of str
        One telescope name per MS, in the order of ``mslist``.
    """
    return [telescope_of(ms) for ms in _as_list(mslist)]


def unique_telescopes(mslist):
    """
    Set of telescopes of a list of Measurement Sets.

    Parameters
    ----------
    mslist : str or list of str
        Measurement Set path(s).

    Returns
    -------
    frozenset of str
        The telescope names present in ``mslist``.
    """
    return frozenset(telescopes_of(mslist))


def is_mixed(mslist):
    """
    Whether a list of Measurement Sets comes from more than one telescope.

    Parameters
    ----------
    mslist : str or list of str
        Measurement Set path(s).

    Returns
    -------
    bool
        True if the MSs have more than one TELESCOPE_NAME.
    """
    return len(unique_telescopes(mslist)) > 1


def single_telescope(mslist):
    """
    The telescope of a list of Measurement Sets, if they all come from one telescope.

    For the functions with telescope-specific presets that take one telescope name.

    Parameters
    ----------
    mslist : str or list of str
        Measurement Set path(s).

    Returns
    -------
    str or None
        The telescope name, or None if the MSs come from more than one telescope (or the list is empty).
    """
    telescopes = unique_telescopes(mslist)
    return next(iter(telescopes)) if len(telescopes) == 1 else None


def mslist_of_telescope(mslist, telescope):
    """
    Measurement Sets of one telescope.

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths.
    telescope : str or collection of str
        Telescope name, or several names (for example ``('VLA', 'EVLA')``).

    Returns
    -------
    list of str
        The MSs in ``mslist`` whose telescope is ``telescope``, in the input order.
    """
    names = {telescope} if isinstance(telescope, str) else set(telescope)
    return [ms for ms in _as_list(mslist) if telescope_of(ms) in names]


def group_by_telescope(mslist):
    """
    Group Measurement Sets by telescope.

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths.

    Returns
    -------
    dict
        Telescope name -> list of MS paths (input order kept within each group). Groups are ordered
        by the first occurrence of the telescope in ``mslist``.
    """
    groups = {}
    for ms in _as_list(mslist):
        groups.setdefault(telescope_of(ms), []).append(ms)
    return groups


def pbcor_function_name(telescopes):
    """
    Name of the manual primary beam correction for a set of telescopes.

    Parameters
    ----------
    telescopes : collection of str
        Telescope names of the MSs in a joint image.

    Returns
    -------
    str or None
        The name of the function in submods/primary_beam.py when all telescopes share one manual
        correction (for example VLA + EVLA), otherwise None (no manual correction possible).
    """
    names = {PBCOR_FUNCTION_NAME.get(tel) for tel in telescopes}
    if len(names) == 1 and None not in names:
        return names.pop()
    return None


def describe_telescopes(telescopes):
    """
    Human-readable list of telescope names, e.g. 'LOFAR + MeerKAT'.

    Parameters
    ----------
    telescopes : str or collection of str
        Telescope name(s).

    Returns
    -------
    str
        The sorted unique names joined with ' + '.
    """
    if isinstance(telescopes, str):
        return telescopes
    return ' + '.join(sorted(set(telescopes)))


# antenna-name patterns used to recognise the telescope of an H5 file (which stores no telescope name)
_ANTENNA_PATTERNS = (
    ('LOFAR', re.compile(r'^(CS|RS|ST)\d{3}|^[A-Z]{2}\d{3}(HBA|LBA)|^(DE|FR|SE|UK|PL|IE|LV|IT|BG)6\d\d')),
    ('MeerKAT', re.compile(r'^m\d{3}$')),
    ('GMRT', re.compile(r'^[CESW]\d{2}$')),
    ('ASKAP', re.compile(r'^ak\d{2}$')),
    ('EVLA', re.compile(r'^ea\d{2}$')),
    ('VLA', re.compile(r'^(VA|EA)\d{2}$')),
    ('MWA', re.compile(r'^(Tile|Hex)')),
)


def telescope_from_antennas(antennas):
    """
    Guess the telescope from antenna names (for H5 files, which do not store a telescope name).

    Parameters
    ----------
    antennas : list of str or bytes
        Antenna names.

    Returns
    -------
    str or None
        The telescope whose naming pattern matches most antennas, or None if no pattern matches.
    """
    votes = Counter()
    for ant in antennas:
        name = ant.decode() if isinstance(ant, (bytes, np.bytes_)) else str(ant)
        for telescope, pattern in _ANTENNA_PATTERNS:
            if pattern.search(name):
                votes[telescope] += 1
                break
    if not votes:
        return None
    return votes.most_common(1)[0][0]
