"""Read-only queries on h5parm files (solution types, axes, reference antenna)
and a helper to force-close open HDF5 handles."""
import numpy as np
import tables
from losoto import h5parm  # type: ignore

from ..runtime_utils import get_logger, terminal_print


def force_close(h5):
    """
    Close indivdual HDF5 file by force.

    Parameters
    ----------
    h5 : str
        name of the h5parm to close.

    Returns
    -------
    None
    """
    h5s = list(tables.file._open_files._handlers)
    for h in h5s:
        if h.filename == h5:
            get_logger().warning('force_close: Closed --> ' + h5 + '\n')
            terminal_print('Forced (!) closing', h5)
            h.close()
            return
    # sys.stderr.write(h5 + ' not found\n')
    return


def amplitude_leakage_paramdb(h5):
    """
    Checks if a given h5parm has amplitude leakage solutions in sol000.

    Parameters
    ----------
    h5 : str
        path to the h5parm.

    Returns
    -------
    amplitudeleakage : bool
        whether the sol000 contains amplitude leakage solutions.
    """
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure, hasdelay = check_soltabs(h5)
    if hasphase:
        return False # if we have phase solutions, we cannot we do not have amplitude only leakage solutions

    H = tables.open_file(h5)
    try:
        pol_a = H.root.sol000.amplitude000.pol[:]
        if len(pol_a) == 4:
            amplitudeleakage = True
        else:
            amplitudeleakage = False
    except:
        amplitudeleakage = False
    H.close()
    return amplitudeleakage


def fulljonesparmdb(h5):
    """
    Checks if a given h5parm has a fulljones solution table as sol000.

    Parameters
    ----------
    h5 : str
        path to the h5parm.

    Returns
    -------
    fulljones : bool
        whether the sol000 contains fulljones solutions.
    """
    H = tables.open_file(h5)
    try:
        pol_p = H.root.sol000.phase000.pol[:]
        pol_a = H.root.sol000.amplitude000.pol[:]
        if len(pol_p) == 4 and len(pol_a) == 4:
            fulljones = True
        else:
            fulljones = False
    except:
        fulljones = False
    H.close()
    return fulljones


def ntimesH5(H5file):
    """
    Returns the number of timeslots in an H5parm.

    Parameters
    ----------
    H5file : str
        path to H5parm.

    Returns
    -------
    times : int
        length of the time axis.
    """

    sol_types = ['amplitude000', 'phase000', 'tec000', 'rotationmeasure000', 'rotation000', 'delay000']

    with tables.open_file(H5file, mode='r') as H:
        for sol_type in sol_types:
            try:
                return len(getattr(H.root.sol000, sol_type).time[:])
            except AttributeError:
                continue

        terminal_print('No amplitude000, phase000, tec000, rotationmeasure000, rotation000, or delay000 solutions found')
        raise Exception('No amplitude000, phase000, tec000, rotationmeasure000, rotation000, or delay000 solutions found')


def number_freqchan_h5(h5parmin):
    """
    Get the number of frequencies in an H5 solution file.

    Parameters
    ----------
    h5parmin : str
        Path to the input H5parm file.

    Returns
    -------
    int
        Number of frequency channels in the H5 file.
    """
    freq = []
    solution_types = ['phase000', 'amplitude000', 'rotation000', 'tec000', 'rotationmeasure000', 'delay000']

    with tables.open_file(h5parmin) as H:
        for sol_type in solution_types:
            try:
                freq = getattr(H.root.sol000, sol_type).freq[:]
                break  # Stop once we successfully retrieve the frequency data
            except tables.NoSuchNodeError:
                continue  # Try the next solution type if current one is missing

    terminal_print('Number of frequency channels in this solutions file is:', len(freq))
    return len(freq)


def check_soltabs(h5parm):
    """
    Check the presence of various solution types in an H5 file.

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm solution file.

    Returns
    -------
    tuple of bool
        Flags indicating whether phase, amplitude, tec, and rotation soltabs exist.
    """
    hasphase = hasamps = hasrotation = hastec = hasrotationmeasure = hasdelay = False

    with tables.open_file(h5parm) as H:
        soltabs = list(H.root.sol000._v_children.keys())
    for s in soltabs:
        if 'amplitude' in s:
            hasamps = True
        elif 'phase' in s:
            hasphase = True
        elif 'tec' in s:
            hastec = True
        elif 'rotationmeasure' in s:
            hasrotationmeasure = True
        elif 'rotation' in s:
            hasrotation = True
        elif 'delay' in s:
            hasdelay = True    

    return hasphase, hasamps, hasrotation, hastec, hasrotationmeasure, hasdelay


def check_phaseup(H5name):
    """
    Check if the H5parm file contains phased-up stations.

    This function opens an H5parm file and checks if 'ST001' (indicating a phased-up
    superstation) is present in the antenna list of any available solution table.

    Parameters
    ----------
    H5name : str
        Path to the H5parm file to be checked.

    Returns
    -------
    bool
        True if 'ST001' is found in the antenna list, indicating phased-up stations
        are present. False otherwise.

    Notes
    -----
    The function searches through multiple solution types in the following order:
    'phase000', 'amplitude000', 'rotation000', 'tec000', 'rotationmeasure000', 'delay000'.
    It stops at the first solution type where antennas are successfully found.
    """

    with tables.open_file(H5name, mode='r') as H5:
        ants = []
        for sol_type in ['phase000', 'amplitude000', 'rotation000', 'tec000', 'rotationmeasure000', 'delay000']:
            try:
                ants = getattr(H5.root.sol000, sol_type).ant[:]
                break  # Stop if antennas are found in the current solution type
            except tables.NoSuchNodeError:
                continue  # Try the next solution type if the current one is missing

    return 'ST001' in ants


def is_scalar_array_for_wsclean(h5list):
    """
    Determine if the solutions in H5parm files are scalar (polarization-independent) for WSClean.

    This function checks whether the calibration solutions stored in H5parm files can be
    treated as scalar values, meaning they are either missing a polarization axis or have
    identical values across all polarizations.

    Parameters
    ----------
    h5list : list of str
        List of paths to H5parm files containing calibration solutions.

    Returns
    -------
    bool
        True if solutions are scalar (no polarization axis or identical across polarizations),
        False if solutions differ between polarizations.

    Notes
    -----
    The function performs two checks:
    1. First checks if any solutions lack a polarization axis entirely
    2. For solutions with a polarization axis, compares values between first and last
    polarization indices to determine if they are identical

    The function examines both 'phase000' and 'amplitude000' solution tables within
    the 'sol000' solution set. The polarization axis is assumed to be the last dimension
    of the solution arrays (shape: time, freq, ant, dir, pol).
    """

    is_scalar = True  # Start with True
    pol_axis = False # Start with False

    # check for arrays that do not have a pol-axis
    for h5 in h5list:
        with tables.open_file(h5) as H:
            # try amplitude first
            try:
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                if 'pol' in axisn:
                    pol_axis = True
            except AttributeError:
                pass
            # then try phase
            try:
                axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')
                if 'pol' in axisn:
                    pol_axis = True
            except AttributeError:
                pass

    if not pol_axis:
        terminal_print('Detected solutions without a polarization axis')
        return is_scalar  # True

    # for arrays that do have a pol-axis (assume pol is last axis)
    for h5 in h5list:
        with tables.open_file(h5, mode='r') as H5:
            for sol_type in ['phase000', 'amplitude000']:
                try:
                    val = getattr(H5.root.sol000, sol_type).val[:]  # time, freq, ant, dir, pol
                    if not np.array_equal(val[..., 0], val[..., -1]):
                        is_scalar = False
                except AttributeError:
                    pass

    return is_scalar


def has0coordinates(h5):
    """
    Check whether any direction has zero coordinates.

    This detects a rare DP3 issue in which a direction is written with both
    coordinates set to zero.

    Parameters
    ----------
    h5 : str or pathlib.Path
        Path to the H5 calibration file.

    Returns
    -------
    bool
        ``True`` if any direction in ``sol000/source`` has both
        coordinates equal to zero, otherwise ``False``.
    """
    h5 = tables.open_file(h5)
    for c in h5.root.sol000.source[:]:
        x, y = c[1]
        if x == 0. and y == 0.:
            h5.close()
            return True
    h5.close()
    return False


_fixed_refant = None  # set with set_refant(), then findrefant_core returns it whenever possible


def set_refant(refant):
    """
    Fix the reference antenna returned by findrefant_core for the rest of the run.

    Parameters
    ----------
    refant : str or None
        Antenna name, or None to go back to choosing the reference antenna per H5 file.
    """
    global _fixed_refant
    _fixed_refant = refant


def refant_candidates(ants, telescope='LOFAR'):
    """
    Indices of the preferred reference antennas of a telescope among the given antenna names.

    For LOFAR, the core stations are preferred, with remote stations used as a fallback.
    Other supported telescopes use their predefined preferred-antenna lists. If none of the
    names match, all antennas are candidates (with a warning).

    Parameters
    ----------
    ants : list of str
        Antenna names.
    telescope : str, optional
        Telescope name as in the MS OBSERVATION table. Supported values include ``'LOFAR'``,
        ``'MeerKAT'``, ``'GMRT'``, ``'ASKAP'``, ``'EVLA'``/``'VLA'``, and ``'MWA'``.

    Returns
    -------
    numpy.ndarray
        Indices into ``ants``.
    """
    cs_indices = []
    if telescope == 'LOFAR':
        # try core stations first
        cs_indices = np.where(['CS' in ant for ant in ants])[0]
        if len(cs_indices) == 0:  # in case there are no CS stations try with RS
            cs_indices = np.where(['RS' in ant for ant in ants])[0]

    if telescope == 'MeerKAT':
        possible_refants = possible_refants = ['m013', 'm012', 'm011', 'm010', 'm009', 'm002', 'm001','m000','m008','m007','m006','m005','m004','m003','m016','m017','m018','m019']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]
    
    if telescope == 'GMRT':
        possible_refants = ['C00','C01','C02','C03','C04','C05','C06','C08','C09','C10','C11','C12','C13','C14']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if telescope == 'ASKAP':
        possible_refants = ['ak01','ak02','ak03','ak04','ak05','ak06','ak07','ak08','ak09','ak10','ak11','ak12','ak13','ak14']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if telescope in ['EVLA', 'VLA']:
        possible_refants = ['ea02','ea03','ea04','ea05','ea06','ea07','ea08','ea09','ea10','ea11','ea12','ea13','ea14', 'ea15','ea16','ea17','ea18','ea19','ea20','ea21','ea22','ea23','ea24','ea25','ea26','ea27','ea28']
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]

    if telescope == 'MWA':
        possible_refants = [
            "Tile012", "Tile013", "Tile014", "Tile015", "Tile017", "Tile024", "Tile025", "Tile026", "Tile027",
            "Tile032", "Tile033", "Tile034", "Tile035", "Tile036", "Tile037", "Tile038",
            "Tile041", "Tile042", "Tile043", "Tile044", "Tile045", "Tile046", "Tile047", "Tile048",
            "Tile051", "Tile052", "Tile053", "Tile054", "Tile055", "Tile056", "Tile057", "Tile058",
            "Tile063", "Tile064", "Tile065", "Tile066", "Tile067", "Tile068",
            "Tile071", "Tile072", "Tile073", "Tile074", "Tile075", "Tile076", "Tile077", "Tile078",
            "Tile083", "Tile084", "Tile094", "Tile095",
            "Tile101", "Tile102", "Tile103", "Tile104", "Tile105", "Tile106", "Tile107", "Tile108",
            "Tile111", "Tile112", "Tile113", "Tile114", "Tile115", "Tile116", "Tile117", "Tile118",
            "Tile121", "Tile122", "Tile123", "Tile124", "Tile125", "Tile126", "Tile127", "Tile128",
            "Tile131", "Tile132", "Tile133", "Tile134", "Tile135", "Tile136", "Tile137", "Tile138",
            "Tile141", "Tile142", "Tile143", "Tile144", "Tile145", "Tile146", "Tile147", "Tile148",
            "Tile151", "Tile152", "Tile153", "Tile154", "Tile155", "Tile156", "Tile157", "Tile158",
            "Tile161", "Tile162", "Tile163", "Tile164", "Tile165", "Tile166", "Tile167", "Tile168",
        ]
        cs_indices = np.where([ant in possible_refants for ant in ants])[0]
        

    if len(cs_indices) == 0:
        # print in red
        terminal_print(
            f'\033[91mWarning: no {telescope} reference stations found: none of the antenna names in the MS matched any expected {telescope} names; using all antennas to find refant\033[0m'
        )
        get_logger().warning(
            'No %s reference stations found: none of the antenna names in the MS matched any expected %s names; using all antennas to find refant.',
            telescope,
            telescope,
        )
        cs_indices = np.arange(len(ants))
    return np.asarray(cs_indices)


def findrefant_core(H5file, telescope='LOFAR'):
    """
    Select a reference antenna from the core or preferred antennas.

    If a reference antenna was fixed with set_refant (facetselfcal does this at the start of a run,
    see --refant), that antenna is returned, unless it is not in the H5 file or all its solutions
    are flagged; then the choice below is used with a warning.
    Otherwise the function opens the ``sol000`` solution set, selects a suitable
    solution table, and chooses the candidate antenna (see refant_candidates) with the greatest sum
    of unflagged weights.

    Parameters
    ----------
    H5file : str or pathlib.Path
        Path to the H5 calibration file.
    telescope : str, optional
        Telescope whose preferred antennas should
        be used. Supported values include ``'LOFAR'``, ``'MeerKAT'``,
        ``'GMRT'``, ``'ASKAP'``, ``'EVLA'``/``'VLA'``, and ``'MWA'``.

    Returns
    -------
    str
        Name of the selected reference antenna.
    """
    H = h5parm.h5parm(H5file)
    solset = H.getSolset('sol000')
    soltabs = solset.getSoltabNames()
    for st in soltabs:
        # Find a reasonable soltab to use
        if 'phase000' in st or 'rotation000' in st or 'tec000' in st or 'rotationmeasure000' in st or 'delay000' in st:
            break
    soltab = solset.getSoltab(st)
    ants = list(soltab.getValues()[1]['ant'])

    # Find the antennas and which dimension that corresponds to
    ant_index = np.where(np.array(soltab.getAxesNames()) == 'ant')[0][0]
    weights = soltab.getValues(weight=True)[0]

    def weightsum(idx):
        return np.nansum(np.take(weights, idx, axis=ant_index))

    if _fixed_refant is not None:
        if _fixed_refant in ants and weightsum(ants.index(_fixed_refant)) > 0:
            H.close()
            return _fixed_refant
        terminal_print(f'\033[91mWarning: fixed reference antenna {_fixed_refant} is not in {H5file} '
                       'or all its solutions are flagged, choosing another one for this H5 file\033[0m')
        get_logger().warning('Fixed reference antenna %s not usable for %s, choosing another one.',
                             _fixed_refant, H5file)

    if 'ST001' in ants:
        H.close()
        return 'ST001'

    # Find the antenna with the least flagged datapoint
    cs_indices = refant_candidates(ants, telescope=telescope)
    maxant = np.argmax([weightsum(cs) for cs in cs_indices])
    H.close()
    return ants[cs_indices[maxant]]
