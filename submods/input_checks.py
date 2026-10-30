"""Input checks: command-line option validation (inputchecker), Measurement Set
validity, antenna-factor and memory checks, primary-beam FoV and pointing
checks, and model-type compatibility."""
import glob
import math
import os
import sys

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from casacore.tables import table  # type: ignore

from submods.antenna_constraints import available_antennaconstraints, max_in_str
from submods.facet_directions import parse_facetdirections
from submods.ms_fixes import get_telescope_from_ms
from submods.ms_info import check_phaseup_station, checklongbaseline
from submods.runtime_utils import get_logger, terminal_print, which
from submods.smearing import compute_distance_to_pointingcenter
from submods.solints import format_solint, remove_bad_endrounding, tweak_solints

# facetselfcal package directory (holds flagging_strategies/ and data/), same as datapath in main.py
datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal')


def is_ms_regularized(ms_path: str) -> bool:
    """
    Checks if a Measurement Set has a perfectly regularized time-baseline grid structure.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory.

    Returns
    -------
    bool
        True if the MS is perfectly regularized, False otherwise.
    """
    try:
        # Open the main table; read-only mode is safer and faster
        with table(ms_path, readonly=True, ack=False) as t:
            # If the table is entirely empty, it cannot be considered regularized
            if t.nrows() == 0:
                return False
                
            # Extract only the TIME column to minimize memory usage
            times = t.getcol("TIME")
            
        # Count how many baseline rows exist for each unique timestamp
        _, counts = np.unique(times, return_counts=True)
        
        # If the grid is perfectly regular, every timestamp must have the exact same count.
        # This means the unique list of counts will have a length of exactly 1.
        return len(np.unique(counts)) == 1

    except Exception as e:
        terminal_print(f"Error reading Measurement Set at {ms_path}: {e}")
        return False


def check_pointing_centers(mslist, args):
    """
    Checks whether the pointing centers of the provided measurement sets (MS) are aligned within a specified tolerance.

    Parameters
    ----------
    mslist : list of str
        List of paths to measurement sets (MS) to check.
    args : dict
        Parsed facetselfcal options; ``stack`` is used.

    Returns
    -------
    bool
        True if all MS pointing centers are aligned within 0.025 arcseconds, False otherwise.

    Warnings
    --------
    Prints a warning message and logs a warning if any MS pointing center differs from the first MS by more than 0.025 arcseconds.

    Notes
    -----
    - If only one MS is provided, the function returns True without performing any checks.
    - Requires the 'table', 'SkyCoord', and 'units' objects to be available in the scope.
    """
    
    if len(mslist) == 1 or args['stack']:
        return True  # Only one MS, no need to check alignment, for stacking no need to check alignment either
    with table(mslist[0] + '::FIELD', ack=False, readonly=True) as t:
        ra_ref, dec_ref = t.getcol('PHASE_DIR').squeeze() # reference direction in radians
        center = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')
    
    align = True
    for ms in mslist[1:]:  
        with table(ms + '::FIELD', ack=False, readonly=True) as t:
            ra_ms, dec_ms = t.getcol('PHASE_DIR').squeeze()
        center_ms = SkyCoord(ra_ms * units.radian, dec_ms * units.radian, frame='icrs')
        if not center.separation(center_ms).deg < 0.025/3600:  # 0.025 arcsec tolerance
            terminal_print("\033[33m" + "WARNING: Pointing centers of MSs differ by " + str(center.separation(center_ms).deg) + " [deg] !\033[0m")
            terminal_print("\033[33m" + "Pointing center of " + ms + " is not aligned with the first MS!" + "\033[0m")
            terminal_print("\033[33m" + "Will use DP3 phaseshift to align them\033[0m")
            get_logger().warning('Pointing centers of MSs differ: ' + ms)
            align = False
    return align


def check_applyfacetbeam(mslist, imsize, pixsize, telescope, args, enlarge_safe_FoV_diameter=1.0):
    """
    Checks whether the image field of view (FoV) for MeerKAT/GMRT data is too large to safely use the -apply-facet-beam/-apply-primary-beam option in WSClean, and enforces the --disable-primary-beam option if necessary.

    Parameters
    ----------
    mslist : list of str
        List of measurement set (MS) file paths to check.
    imsize : float
        Image size in pixels.
    pixsize : float
        Pixel size in arcseconds.
    telescope : str
        Name of the telescope. Function only applies checks if this is 'MeerKAT' or 'GMRT'.
    args : dict
        Parsed facetselfcal options; ``disable_primary_beam`` is set to True in
        place if the field of view is too large.
    enlarge_safe_FoV_diameter : float, optional
        Factor to enlarge the safe FoV diameter. Default is 1.0 (no enlargement).

    Returns
    -------
    None

    Notes
    -----
    Side Effects:
    - If the image FoV is too large for any MS in mslist, sets args['disable_primary_beam'] = True.
    - Prints warnings to the console and logs a warning message.
    - Exits after the first MS that violates the safe FoV criterion.
    - The safe diameter is calculated based on the maximum frequency across all MSs in mslist.
    - The function also assumes the presence of 'compute_distance_to_pointingcenter' and 'table' utilities.
    """
    if telescope not in ['GMRT', 'MeerKAT', 'VLA', 'EVLA', 'MWA']:
        return

    # Calculate safe diameter for each MS based on its frequency characteristics
    safe_diameters = []
    for ms in mslist:
        with table(ms + "::SPECTRAL_WINDOW", ack=False) as t:
            chan_freqs = t.getcol("CHAN_FREQ")
            max_freq = chan_freqs.max()
            freq = np.median(chan_freqs)

        if telescope == 'MeerKAT':
            safe_diam = 60. * 68. * (1.28e9 / max_freq)  # in arcsec

        elif telescope == 'MWA':  # FWHM is 25 degr at 150 MHz
            safe_diam = 60. * 25. * 60. * (150e6 / max_freq)

        elif telescope == 'VLA' or telescope == 'EVLA':
            safe_diam = 60. * 30. * (1.4e9 / max_freq)  # in arcsec

        elif telescope == 'GMRT':
            if 0.125e9 <= freq <= 0.250e9:
                safe_diam = 60. * 120. * (187.5e6 / max_freq)  # in arcsec
            elif 0.250e9 < freq <= 0.500e9:
                safe_diam = 60. * 75. * (375e6 / max_freq)  # in arcsec
            elif 0.550e9 <= freq <= 0.850e9:
                safe_diam = 60. * 38. * (700e6 / max_freq)  # in arcsec
            elif 1.050e9 <= freq <= 1.450e9:
                safe_diam = 60. * 23. * (1230e6 / max_freq)  # in arcsec
            else:
                raise ValueError("Frequency {} GHz is outside the supported GMRT frequency ranges for primary beam checks.".format(freq / 1e9))

        # enlarge_safe_FoV_diameter: factor to enlarge the safe FoV diameter. Default is 1.0 (no enlargement).
        # use with care as it can use unstable behaviour when the beam goes through a null which can cause numerical issues in the primary beam correction.
        safe_diam *= enlarge_safe_FoV_diameter  # Enlarge the safe FoV diameter by the specified factor
        safe_diameters.append(safe_diam)

    # Use the most restrictive (lowest) safe diameter across the entire mslist
    safe_diameter = min(safe_diameters)

    for ms in mslist:
        distance_pointing_center = compute_distance_to_pointingcenter(ms, HBAorLBA='other', warn=False, returnval=True, dologging=False)

        if ((imsize * pixsize) + (distance_pointing_center * 3600.)) > safe_diameter:
            args['disable_primary_beam'] = True  # set to True if one in mslist violates this criterion
            terminal_print("\033[33m" + "=== " + ms + " ===" + "\033[0m")
            terminal_print("\033[33m" + "Your image FoV is too large to use -apply-facet-beam/-apply-primary-beam in WSClean!" + "\033[0m")
            terminal_print("\033[33m" + "Code will run with the option --disable-primary-beam enforced" + "\033[0m")
            terminal_print("\033[33m" + "Imaged FoV [deg]: " + str(imsize * pixsize / 3600.) + "\033[0m")
            terminal_print("\033[33m" + "Image center to telescope pointing center [deg]: " + str(distance_pointing_center) + "\033[0m")
            terminal_print("\033[33m" + "Safe FoV [deg]: " + str(safe_diameter / 3600.) + "\033[0m")
            get_logger().warning('Your image FoV is too large to use -apply-facet-beam/-apply-primary-beam in WSClean. The option --disable-primary-beam is automatically invoked: ' + ms)
            break
    return


def is_stokesdiagonal_modeltype_allowed(args, telescope):
    """
    Determine if Diagonal sisco compression is allowed for MODEL_DATA-type columns.

    Parameters
    ----------
    args : dict
        Dictionary of arguments, including 'single_dual_speedup',
        'disable_primary_beam', and 'soltype_list'.
    telescope : str
        The telescope name (e.g., 'LOFAR').

    Returns
    -------
    bool
        True if Diagonal sisco compression is allowed, False otherwise.
    """
    if telescope == 'LOFAR': 
        if not args['single_dual_speedup']:
            if not args['disable_primary_beam']:
                return False # so in this case we want to keep the primary beam polarization information    
    
    notallowed_list = ['fulljones', 'rotation', \
                       'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude', \
                       'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase', \
                       'rotation_phmin', 'faradayrotation', 'faradayrotation+diagonal', \
                       'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', \
                       'faradayrotation+scalar', 'faradayrotation+scalaramplitude', \
                       'faradayrotation+scalarphase', 'leakage', 'leakageamplitude']
    if not args['docircular']:  # a phase-difference solve on linear data is applied as a full-Jones matrix
        notallowed_list += ['scalarphasediff', 'scalarphasediffFR', 'treemfphasediff']
    for soltype in args['soltype_list']:
        if soltype in notallowed_list: return False
    return True


def is_stokesi_modeltype_allowed(args, telescope):
    """
    Determine if Stokes I compression is allowed for MODEL_DATA-type columns.

    Parameters
    ----------
    args : dict
        Dictionary of arguments, including 'single_dual_speedup',
        'disable_primary_beam', and 'soltype_list'.
    telescope : str
        The telescope name (e.g., 'LOFAR').

    Returns
    -------
    bool
        True if Stokes I compression is allowed, False otherwise.
    """
    if telescope == 'LOFAR': 
        if not args['single_dual_speedup']:
            if not args['disable_primary_beam']:
                return False # so in this case we want to keep the primary beam polarization information    
    
    notallowed_list = ['complexgain', 'amplitudeonly', 'phaseonly', 'fulljones', 'rotation', \
                       'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude', \
                       'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase', \
                       'phaseonly_phmin', 'rotation_phmin', 'phaseonly_slope', 'scalarphasediff', \
                       'scalarphasediffFR', 'treemfphasediff', 'faradayrotation', 'faradayrotation+diagonal', \
                       'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', \
                       'faradayrotation+scalar', 'faradayrotation+scalaramplitude', \
                       'faradayrotation+scalarphase', 'leakage', 'leakageamplitude']
    for soltype in args['soltype_list']:
        if soltype in notallowed_list: return False
    return True


def check_antenna_factors(antenna_averaging_factors_list, antenna_smoothness_factors_list, mslist, facetdirections,
                          args):
    """
    Validates antenna averaging and smoothness factors for a list of measurement sets (MS) and facet directions.
    This function performs the following checks:
    1. Ensures that all antenna smoothness factors are within the allowed range (> 0 and <= 1.0).
    2. For each measurement set and solution interval cycle, checks that the maximum antenna averaging factor does not exceed the maximum allowed divisor.
    3. If the maximum antenna averaging factor exceeds the allowed value, attempts to upscale the least common multiple (LCM) of solution intervals, provided it does not exceed a hard limit (4096).
    4. If any check fails, prints a warning and exits the program.

    Parameters
    ----------
    antenna_averaging_factors_list : list
        Nested list containing antenna averaging factors for each solution interval cycle and measurement set.
    antenna_smoothness_factors_list : list
        Nested list containing antenna smoothness factors for each perturbation and measurement set.
    mslist : list
        List of measurement set file paths.
    facetdirections : str or None
        String specifying facet directions and related parameters, or None if not used.
    args : dict
        Parsed facetselfcal options; ``DDE`` is used, and ``args`` is passed on
        to :func:`parse_facetdirections` for validation.

    Raises
    ------
    SystemExit
        If any antenna smoothness factor is not in the allowed range, or if any antenna averaging factor exceeds the allowed value and cannot be upscaled within limits.
    """
    facetdirections_list = facetdirections if isinstance(facetdirections, list) else [facetdirections]
    for fd in facetdirections_list:
        len_pert = np.array(antenna_smoothness_factors_list).shape[0]
        for ms_id, ms in enumerate(mslist):
            for asf_id in range(len_pert):
                antenna_smoothness_factors = antenna_smoothness_factors_list[asf_id][ms_id]
                if antenna_smoothness_factors is not None and args["DDE"]:
                    max_smoothness_factor = max(float(x.split(':')[1]) for x in antenna_smoothness_factors.split(','))
                    if max_smoothness_factor > 1.0 or max_smoothness_factor <=0.0:
                        terminal_print('WARNING: The maximum antenna smoothness factor', max_smoothness_factor, 'should be > 0 and <= 1.0 for a DDE solve')
                        terminal_print('This is not allowed')
                        sys.exit(1)
        
        if fd is not None:
            dirs, solintslist, smoothness, soltypelist_includedir = parse_facetdirections(fd, 1000, args=args)    
            if solintslist is None:
                return

            solint_reformat = np.array(solintslist)  
            for ms_id, ms in enumerate(mslist):
                terminal_print('=== Checking antenna averaging factors for MS:', ms, '===')
                with table(ms, readonly=True, ack=False) as t:
                    ms_ntimes = len(np.unique(t.getcol('TIME')))
                for solintcycle_id, tmpval in enumerate(solint_reformat[0]): 
                    antenna_averaging_factors = antenna_averaging_factors_list[solintcycle_id][ms_id]
                    solints_cycle = solint_reformat[:, solintcycle_id]
                    solints = [int(format_solint(x, ms)) for x in solints_cycle]        
                    solints = tweak_solints(solints, ms_ntimes=ms_ntimes)
                    lcm = math.lcm(*solints)
                    divisors = [int(lcm / i) for i in solints]
                    terminal_print('pertubation cycle =', solintcycle_id)
                    terminal_print('divisors and lcm:', divisors, lcm)
                    if antenna_averaging_factors is not None:
                        max_avg_factor = max_in_str(antenna_averaging_factors)
                        if max_avg_factor > max(divisors):
                            upscale_factor = int(np.ceil(max_avg_factor / max(divisors)))
                            if lcm * upscale_factor < 4096: # avoid too large solints
                                lcm = lcm * upscale_factor
                                divisors = [int(lcm / i) for i in solints]
                                terminal_print('Updated divisors and lcm:', divisors, lcm)
                            else:     
                                terminal_print('WARNING: The maximum antenna averaging factor', max_avg_factor, 'is larger than the maximum allowed value of', max(divisors))
                                terminal_print('This is not allowed by DP3')
                                sys.exit(1) 
    return


def test_antenna_averaging_factors(antenna_averaging_factors_list, mslist, facetdirections, args=None):
    """
    Inspect antenna averaging factors for compatible solution intervals.

    Parameters
    ----------
    antenna_averaging_factors_list : list
        Requested antenna factors.
    mslist : list of str
        Measurement Set paths to inspect.
    facetdirections : str
        Facet-direction file path.
    args : dict, optional
        Parsed facetselfcal options, passed on to :func:`parse_facetdirections`
        for validation. Default is None.

    Returns
    -------
    None
        Diagnostic information is printed.
    """
    dirs, solintslist, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, 1000, args=args)    
    solint_reformat = np.array(solintslist)
    for ms_id, ms in enumerate(mslist):
        with table(ms, readonly=True, ack=False) as t:
            ms_ntimes = len(np.unique(t.getcol('TIME')))
        
        for solintcycle_id, tmpval in enumerate(solint_reformat[0]): 
            antenna_averaging_factors = antenna_averaging_factors_list[solintcycle_id][ms_id]
            terminal_print(' --- ' + str('pertubation cycle=') + str(solintcycle_id) + '--- ')
            solints_cycle = solint_reformat[:, solintcycle_id]
            solints = [int(format_solint(x, ms)) for x in solints_cycle]        
            solints = tweak_solints(solints, ms_ntimes=ms_ntimes)
            lcm = math.lcm(*solints)
            terminal_print('LCM of solints', lcm)
            divisors = [int(lcm / i) for i in solints]
            terminal_print('Divisors of solints', divisors)
            terminal_print(
                'Antenna averaging factors:',
                antenna_averaging_factors,
                'Solution intervals:',
                solints,
            )
            if antenna_averaging_factors is not None:
                terminal_print('Check and update antenna averaging factors if needed')
                all_possible_avg_factors = range(1,max(solints)*lcm)
                allowed_avg_factors = []
                allowed_avg_factors_ondir = []
                for dir_id, solint in enumerate(np.array(solints)*lcm):
                    allowed_avg_factors_ondir = [i for i in all_possible_avg_factors if solint % i == 0]
                    allowed_avg_factors.append(allowed_avg_factors_ondir)
                
        
                #print('List of allowed averaging factors per direction in combination with lcm', allowed_avg_factors)
                allowed_avg_factors = set(allowed_avg_factors[0]).intersection(*allowed_avg_factors)
                allowed_avg_factors = sorted(list(allowed_avg_factors))
                if len(allowed_avg_factors) == 0:
                    terminal_print('No common averaging factors found for all directions. This must be a bug because a value of 1 should always be possible')
                    sys.exit()
                
                # remove averaging factors that are larger than largest divisor
                #print('List of allowed averaging factors for all directions in combination with lcm', allowed_avg_factors)
                allowed_avg_factors = [x for x in allowed_avg_factors if x <= max(divisors)]
                terminal_print('List of allowed averaging factors for all directions in combination with lcm', allowed_avg_factors)
                
                # find nearest allowed averaging factor to the one request by user
                str_aaf = antenna_averaging_factors.split(',')
                for aaf in str_aaf:
                  user_aaf = int(aaf.split(':')[1])
                  # find closest value in allowed_avg_factors
                  closest_aaf = min(allowed_avg_factors, key=lambda x: abs(x - user_aaf))
                  if closest_aaf != user_aaf:
                      terminal_print('Requested antenna averaging factor', user_aaf, 'not possible. Using closest possible value', closest_aaf)
                  else:
                      terminal_print('Requested antenna averaging factor', user_aaf, 'is possible.')
                
                
                if False:
                    allowed_avg_factors_tmp = sorted(list(set(np.array(allowed_avg_factors)/lcm)))
                    #print('List of allowed averaging factors divided by lcm', allowed_avg_factors_tmp)
                    
                    terminal_print('Allowed antenna averaging factors', allowed_avg_factors)

                    terminal_print('Solution intervals:', solints)
                    bad_avg_factors = [] # list of bad averaging factors due to rounding issues
                    for sol in solints:
                        terminal_print('Doing sol', sol)
                        for avg_factor_id, avg_factor in enumerate(allowed_avg_factors_tmp):
                            possible_solints = remove_bad_endrounding([sol/avg_factor], ms_ntimes)
                            #print('Possible solints for avg factor', sol/avg_factor, possible_solints)
                            if len(possible_solints) == 0:
                                terminal_print('Averaging factor', avg_factor, 'not possible for solint', sol)
                                bad_avg_factors.append(avg_factor)
                        
                    bad_avg_factors = list(np.array(bad_avg_factors)*lcm)
                    # make integer list and remove duplicates
                    bad_avg_factors = sorted(list(set([int(x) for x in bad_avg_factors])))
                    terminal_print('Bad averaging factors due to rounding issues', bad_avg_factors)
                    terminal_print('Original antenna averaging factors', allowed_avg_factors)
                    allowed_avg_factors = [x for x in allowed_avg_factors if x not in bad_avg_factors]
                    terminal_print('New list of allowed antenna averaging factors', allowed_avg_factors)
                    if len(allowed_avg_factors) == 0:
                        terminal_print('No allowed averaging factors left')
                        sys.exit()
                    sys.exit()          
    #    print('Temorary exit for testing')
    #sys.exit()
    return


def check_for_highmem_longsolint(mslist, facetdirections, args=None):
    """
    Check whether DDE solution intervals may require excessive memory.

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths to inspect.
    facetdirections : str
        Facet-direction file path.
    args : dict, optional
        Parsed facetselfcal options, passed on to :func:`parse_facetdirections`
        for validation. Default is None.

    Returns
    -------
    None
        Exits the process if a solution interval is too large.
    """
    dirs, solints, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, 1000, args=args)

    if solints is None:
        return

    solint_reformat = np.array(solints)
    for ms in mslist:
        t = table(ms, readonly=True, ack=False)
        time = np.unique(t.getcol('TIME'))
        t.close()
        terminal_print('------------' + ms)
        ms_ntimes = len(time)
        # print(ms, ms_ntimes)
        for solintcycle_id, tmpval in enumerate(solint_reformat[0]):
            terminal_print(' --- ' + str('pertubation cycle=') + str(solintcycle_id) + '--- ')
            solints_cycle = solint_reformat[:, solintcycle_id]
            solints = [int(format_solint(x, ms)) for x in solints_cycle]
            terminal_print('Solint unmodified per direction', solints)
            solints = tweak_solints(solints,  ms_ntimes=ms_ntimes)
            terminal_print('Solint tweaked per direction   ', solints)

            lcm = math.lcm(*solints)
            divisors = [int(lcm / i) for i in solints]
            
            terminal_print('Solint passed to DP3 is:', lcm, ' --Number of timeslots in MS:', ms_ntimes)
            if lcm > int(10.*ms_ntimes):
                terminal_print('Bad divisor for solutions_per_direction DDE solve. DP3 Solint > number of timeslots in the MS')
                sys.exit()
        terminal_print('------------')

    return


def check_equidistant_freqs(mslist):
    """
    Check if the frequency channels in each Measurement Set (MS) in the provided list are equidistant.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories.

    Raises
    ------
    Exception
    If any MS in the list has frequency channels that are not equidistant within a tolerance of 1e-5 Hz.

    Notes
    -----
    - For single channel data, the function returns without performing any checks.
    - The function prints the unique frequency differences and the MS path if non-equidistant channels are detected.
    """
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            chan_freqs = t.getcol('CHAN_FREQ')[0]
        if len(chan_freqs) == 1:  # single channel data
            return
        diff_freqs_unique = np.unique(np.diff(chan_freqs))
        if len(diff_freqs_unique) != 1:
            for dfreq in diff_freqs_unique[1:]:
                if np.abs(dfreq-diff_freqs_unique[0]) > 1e-5: # max abs diff tolerance is 1e-5 Hz 
                    terminal_print('Channel-spacing deviation (Hz):', np.abs(dfreq-diff_freqs_unique[0]))
                    terminal_print('Unique channel spacings (Hz):', diff_freqs_unique)
                    terminal_print(
                        'Measurement Set:',
                        ms,
                        'Frequency channels are not equidistant, made a mistake in DP3 concat?',
                    )
                    raise Exception(ms + ': Freqeuency channels are no equidistant, made a mistake in DP3 concat?')
    return


def inputchecker(args, mslist):
    """
    Check input validity.

    Parameters
    ----------
    args : dict
        argparse inputs.
    mslist : str list
        list of ms
    """

    if args['remove_outside_center_box'] is not None:
         # carry out checks to see if args['remove_outside_center_box'] is correctly set by user
         # check if the box can be converted to a float number
         # check if it is a strong 'keepall' or 'auto'
         # if it is a string but not 'keepall' or 'auto' it must be a file name and then we need to check of the file exists
        try:
            float(args['remove_outside_center_box'])
        except ValueError:
            if args['remove_outside_center_box'] != 'keepall' and args['remove_outside_center_box'] != 'auto':
                if not os.path.exists(args['remove_outside_center_box']):
                    terminal_print(f"File {args['remove_outside_center_box']} does not exist.")
                    raise Exception(f"File {args['remove_outside_center_box']} does not exist.")     

    # if args['smoothnessrefdistance_list'] contains a number higher than 0.0
    if any(x > 0.0 for x in args['smoothnessrefdistance_list']):
        # so we have a non-zero value in the list, in that case check that the list has the same length as args['soltype_list']
        if len(args['smoothnessrefdistance_list']) != len(args['soltype_list']):
            terminal_print('--smoothnessrefdistance-list and --soltype-list must have the same length')
            raise Exception('--smoothnessrefdistance-list and --soltype-list must have the same length')

    # check that the number of CPUs is >= 1 for WSClean 
    if args['ncpu_max_WSClean'] is not None:
        if args['ncpu_max_WSClean'] < 1:
            terminal_print('--ncpu-max-WSClean must be >= 1')
            raise Exception('--ncpu-max-WSClean must be >= 1')

    # check that the number of CPUs is >= 1 for DP3
    if args['ncpu_max_DP3solve'] is not None:
        if args['ncpu_max_DP3solve'] < 1:
            terminal_print('--ncpu-max-DP3solve must be >= 1')
            raise Exception('--ncpu-max-DP3solve must be >= 1')

    # check that the file exists 
    if isinstance(args['imsize'], str):
        if not os.path.exists(args['imsize']):
            terminal_print(f"File {args['imsize']} does not exist.")
            raise Exception(f"File {args['imsize']} does not exist.")

    # check that the MS has only one FIELD_ID if split_fieldname is not set, otherwise the splitting will be done on the fieldname column
    if args['split_fieldname'] is None:
        for ms in mslist:
            with table(ms, readonly=True) as t:
                if len(np.unique(t.getcol('FIELD_ID'))) != 1:
                    terminal_print(f"Measurement Set {ms} is already not a single source MS, it contains multiple FIELD_IDs. Please split the MS into single source MSs before running facetselfcal.")
                    raise Exception(f"Measurement Set {ms} is already not a single source MS, it contains multiple FIELD_IDs. Please split the MS into single source MSs before running facetselfcal.")
 
    if args['telescope'] == 'Meerkat' or args['telescope'] == 'GMRT':
        # prevent users from doing a bandpass correction for a DDE solve
        if (args['preapplybandpassH5_list'][0]) is not None and args['DDE']:
            terminal_print('Pre-applying bandpass correction is not allowed for DDE solve')
            raise Exception('Pre-applying bandpass correction is not allowed for DDE solve')
        
    #if args['telescope'] == 'GMRT':
    #    # do not allow any time averaging for GMRT data
    #    # this is because of issues with the UVW coordinates for time gaps that are filled with DP3
    #    # the UVW coordinates in this filled gaps are not correct and lead to image artifacts 
    #    if args['avgtimestep'] is not None:
    #        if args['avgtimestep'] > 1:
    #            print('Time averaging cannot be used for GMRT data due to UVW issues')
    #            raise Exception('Time averaging cannot be used for GMRT data due to UVW issues')
    #    if args['remove_outside_center_avgtimestep'] > 1:
    #        print('Time averaging cannot be used for GMRT data due to UVW issues')
    #        raise Exception('Time averaging cannot be used for GMRT data due to UVW issues')        

    assert args['start'] >= 0, '--start must be >= 0'
    if args['stop'] is not None:
        assert args['stop'] >= 1, '--stop must be >= 1'

    if args['start'] > 0 and args['stop'] is None:
        terminal_print('--start>0 cannot be used without specifying --stop')
        raise Exception('--start>0 cannot be used without specifying --stop')

    if args['stop'] is not None and args['stop'] < args['start']:
        terminal_print('--stop must be greater or equal to --start')
        raise Exception('--stop must be greater or equal to --start')

    if not args['remove_outside_center'] and not args['createresidualdatacolumn']:
        # do not allow start to be the same as stop when --remove-outside-center is not set
        if args['stop'] is not None:
            if args['start'] == args['stop']:
                terminal_print('--start cannot be equal to --stop when --remove-outside-center/createresidualdatacolumn is not set')
                raise Exception('--start cannot be equal to --stop when --remove-outside-center/createresidualdatacolumn is not set')

    if 0 in args['aoflagger_correcteddata_selfcalcycle_list']:
        terminal_print('--aoflagger-correcteddata-selfcalcycle-list cannot contain 0')
        raise Exception('--aoflagger-correcteddata-selfcalcycle-list cannot contain 0') 

    if 0 in args['aoflagger_residualdata_selfcalcycle_list']:
        terminal_print('--aoflagger-residualdata-selfcalcycle-list cannot contain 0')
        raise Exception('--aoflagger-residualdata-selfcalcycle-list cannot contain 0') 

    if type(args['channelsout']) is str:
        if args['channelsout'] != 'auto':
            raise Exception("channelsout needs to be an integer or 'auto'")
    else:
        if args['channelsout'] < 1:
            terminal_print('channelsout', args['channelsout'])
            raise Exception("channelsout needs to be a positive integer")

    if type(args['channelsout']) is not str and type(args['fitspectralpol']) is not str:
        if args['fitspectralpol'] >= args['channelsout']:
            terminal_print('--fitspectralpol must be less than --channelsout')
            raise Exception('--fitspectralpol must be less than --channelsout')

    if args['aoflagger_correcteddata'] and args['DDE']:
        terminal_print('--aoflagger-correcteddata cannot be used together with --DDE')
        raise Exception('--aoflagger-correcteddata cannot be used together with --DDE')

    if args['aoflagger_strategy'] is not None:
        if not os.path.isfile(args['aoflagger_strategy']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy']):
                terminal_print('Flagging strategy file not found:', args['aoflagger_strategy'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy'])
    
    if args['aoflagger_strategy_correcteddata'] is not None:
        if not os.path.isfile(args['aoflagger_strategy_correcteddata']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy_correcteddata']):
                terminal_print('Flagging strategy file not found:', args['aoflagger_strategy_correcteddata'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy_correcteddata'])  

    if args['aoflagger_strategy_residualdata'] is not None:
        if not os.path.isfile(args['aoflagger_strategy_residualdata']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy_residualdata']):
                terminal_print('Flagging strategy file not found:', args['aoflagger_strategy_residualdata'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy_residualdata'])  
    
    if args['aoflagger_strategy_afterbandpassapply'] is not None:
        if not os.path.isfile(args['aoflagger_strategy_afterbandpassapply']): # try full location first
            if not os.path.isfile(f'{datapath}/flagging_strategies/' + args['aoflagger_strategy_afterbandpassapply']):
                terminal_print('Flagging strategy file not found:', args['aoflagger_strategy_afterbandpassapply'])
                raise Exception('Flagging strategy file not found:', args['aoflagger_strategy_afterbandpassapply'])  

    if args['skymodelsetjy'] and args['skymodel'] is not None:
        terminal_print('--skymodelsetjy cannot be used together with --skymodel')
        raise Exception('--skymodelsetjy cannot be used together with --skymodel')
    
    if args['skymodelsetjy'] and args['skymodelpointsource'] is not None:
        terminal_print('--skymodelsetjy cannot be used together with --skymodelpointsource')
        raise Exception('--skymodelsetjy cannot be used together with --skymodelpointsource')
    
    if args['skymodelsetjy'] and args['wscleanskymodel'] is not None:
        terminal_print('--skymodelsetjy cannot be used together with --wscleanskymodel')
        raise Exception('--skymodelsetjy cannot be used together with --wscleanskymodel')

    if args['auto_directions'] and not args['DDE']:
       terminal_print('--auto_directions can only be used in combination with --DDE')
       raise Exception('--auto_directions can only be used in combination with --DDE')

    if args['auto_directions'] and args['telescope'] not in ['LOFA','MeeKAT']:
       terminal_print('--auto_directions can only be used with LOFAR and MeerKAT observations for now')
       raise Exception('--auto_directions can only be used with LOFAR  and MeerKAT observations for now')

    if args['DP3_BDA_imaging'] and not args['DDE']:
        terminal_print('--DP3-BDA-imaging can only be used in combination with --DDE')
        raise Exception('--DP3-BDA-imaging can only be used in combination with --DDE')

    if args['DP3_BDA_imaging'] and args['groupms_h5facetspeedup']:
        terminal_print('--DP3-BDA-imaging cannot be used together with --groupms-h5facetspeedup')
        raise Exception('--DP3-BDA-imaging cannot be used together with --groupms-h5facetspeedup')

    if True in args['BLsmooth_list']:
        if len(args['soltypecycles_list']) != len(args['BLsmooth_list']):
            terminal_print('--BLsmooth-list length does not match the length of --soltype-list')
            raise Exception('--BLsmooth-list length does not match the length of --soltype-list')

    if args['modelstoragemanager'] not in ['stokes_i', 'sisco', 'auto', 'sisco_stokes_i', 'sisco_diagonal', 'None', 'none', None]:
         terminal_print('Model storage manager:', args['modelstoragemanager'])
         terminal_print('Wrong input for --modelstoragemanager, needs to be "stokes_i", "sisco" , "auto", "sisco_stokes_i", "sisco_diagonal", or "None"')
         raise Exception('Wrong input for --modelstoragemanager, needs to be stokes_i, sisco, auto, sisco_stokes_i, sisco_diagonal, or None')

    if args['bandpass']:
        if args['stack'] or args['DDE'] or args['stopafterskysolve'] or args['stopafterpreapply']:
            terminal_print('--bandpass cannot be used with --stack, --DDE, --stopafterskysolve, or --stopafterpreapply')
            raise Exception('--bandpass cannot be used with --stack or --DDE')
        if args['skymodel'] is None and args['skymodelpointsource'] is None \
            and args['wscleanskymodel'] is None and not args['skymodelsetjy'] \
            and args['telescope'] != 'MeerKAT':
            terminal_print('skymodel, skymodelpointsource, skymodelsetjy, or wscleanskymodel needs to be set')
            raise Exception('skymodel, skymodelpointsource, or wscleanskymodel needs to be set')

    for tmp in args['BLsmooth_list']:
        # print(args['BLsmooth_list'])
        if not (isinstance(tmp, bool)):
            terminal_print('Baseline smoothing settings:', args['BLsmooth_list'])
            terminal_print('--BLsmooth-list is not a list of booleans')
            raise Exception('--BLsmooth-list is not a list of booleans')

    if args['stack']:  # avoid options that cannot be used when --stack is set
        if args['DDE']:
            terminal_print('--dde cannot be used with --stack')
            raise Exception('--dde cannot be used with --stack')
        if args['compute_phasediffstat']:
            terminal_print('--compute-phasediffstat cannot be used with --stack')
            raise Exception('--compute-phasediffstat cannot be used with --stack')
        if args['fitsmask'] is not None:
            terminal_print('--fitsmask cannot be used with --stack')
            raise Exception('--fitsmask cannot be used with --stack')
        if args['update_uvmin']:
            terminal_print('--update-uvmin cannot be used with --stack')
            raise Exception('--update-uvmin cannot be used with --stack')
        if args['update_multiscale']:
            terminal_print('--update-multiscale cannot be used with --stack')
            raise Exception('--update-multiscale cannot be used with --stack')
        if args['remove_outside_center']:
            terminal_print('--remove-outside-center cannot be used with --stack')
            raise Exception('--remove-outside-center cannot be used with --stack')
        if args['auto']:
            terminal_print('--auto cannot be used with --stack')
            raise Exception('--auto cannot be used with --stack')
        if args['tgssfitsimage'] is not None:
            terminal_print('--tgssfitsimage cannot be used with --stack')
            raise Exception('--tgssfitsimage cannot be used with --stack')
        if args['QualityBasedWeights']:
            terminal_print('--QualityBasedWeights cannot be used with --stack')
            raise Exception('--QualityBasedWeights cannot be used with --stack')

    if not args['stack']:
        if type(args['skymodel']) is list:
            terminal_print('Skymodel cannot be a list if --stack is not set')
            raise Exception('Skymodel cannot be a list if --stack is not set')

    if args['DDE'] and args['preapplyH5_list'][0] is not None:
        terminal_print('--DDE and --preapplyH5-list cannot be used together')
        raise Exception('--DDE and --preapplyH5_list cannot be used together')

    if args['DDE']:
        for ms in mslist:
            with table(ms, readonly=True, ack=False) as t:
                if 'CORRECTED_DATA' in t.colnames():  # not allowed for DDE runs (because solving from DATA and imaging from DATA with an h5)
                    terminal_print(
                        'Measurement Set:',
                        ms,
                        'contains a CORRECTED_DATA column, this is not allowed when using --DDE',
                    )
                    raise Exception('CORRECTED_DATA should not be present when using option --DDE')

    if args['DDE'] and args['idg']:
        terminal_print('Option --idg cannot be used with option --DDE')
        raise Exception('Option --idg cannot be used with option --DDE')

    if not args['stack']:
        if type(args['skymodelpointsource']) is list:
            terminal_print('skymodelpointsource cannot be a list if --stack is not set')
            raise Exception('skymodelpointsource cannot be a list if --stack is not set')

    if not args['stack']:
        if type(args['wscleanskymodel']) is list:
            terminal_print('wscleanskymodel cannot be a list if --stack is not set')
            raise Exception('wscleanskymodel cannot be a list if --stack is not set')

            # sanity check for resetdir_list input
    if type(args['resetdir_list']) is not list:
        terminal_print('--resetdir-list needs to be of type list')
        raise Exception('--resetdir-list needs to be of type list')
    for resetdir in args['resetdir_list']:  # check if it contains None or a integer list-type
        if resetdir is not None:
            if type(resetdir) is not list:
                terminal_print('--resetdir-list needs to None list-items, or contain a list of directions_id')
                raise Exception('--resetdir-list needs to None list-items, or contain a list of directions_id')
            else:
                for dir_id in resetdir:
                    if type(dir_id) is not int:
                        terminal_print('--resetdir-list, direction IDs provided need to be integers')
                        raise Exception('--resetdir-list, direction IDs provided need to be integers')
                    if dir_id < 0:  # if we get here we have an integer
                        terminal_print('--resetdir-list, direction IDs provided need to be integers >= 0')
                        raise Exception('--resetdir-list, direction IDs provided need to be integers >= 0')
                    data = ascii.read(args['facetdirections'])
                    if dir_id + 1 > len(data):
                        terminal_print(
                            '--direction IDs provided for reset is too high for the number of directions provided by ' +
                            args['facetdirections'])
                        raise Exception(
                            '--direction IDs provided for reset is too high for the number of directions provided by ' +
                            args['facetdirections'])

    if args['groupms_h5facetspeedup']:
        if not args['DDE']:
            terminal_print('--groupms-h5facetspeedup can only be used with --DDE')
            raise Exception('--groupms-h5facetspeedup can only be used with --DDE')
    if args['DDE_predict'] == 'DP3':
        if args['fitspectralpol'] < 1:
            terminal_print(
                '--fitspectralpol needs to be turned on, otherwise no skymodel is produced by WSClean and we cannot predict these components with DP3. Put --DDE-predict=WSCLEAN or fitspectralpol>0')
            raise Exception('--Invalid combination of --fitspectralpol and --DDE-predict')
        if type(args['fitspectralpol']) is not str:
            if args['fitspectralpol'] < 1:
                terminal_print(
                    '--fitspectralpol needs to be turned on, otherwise no skymodel is produced by WSClean and we cannot predict these components with DP3. Put --DDE-predict=WSCLEAN or fitspectralpol>0')
                raise Exception('--Invalid combination of --fitspectralpol and --DDE-predict')

    if args['uvmin'] is not None and type(args['uvmin']) is not list:
        if args['uvmin'] < 0.0:
            terminal_print('--uvmin needs to be positive')
            raise Exception('--uvmin needs to be positive')
    if args['uvminim'] is not None and type(args['uvminim']) is not list:
        if args['uvminim'] <= 0.0:
            terminal_print('--uvminim needs to be positive and non-zero')
            raise Exception('--uvminim needs to be positive')
    if args['uvmaxim'] is not None and args['uvminim'] is not None and type(args['uvmaxim']) is not list and type(
            args['uvminim']) is not list:
        if args['uvmaxim'] <= args['uvminim']:
            terminal_print('--uvmaxim needs to be larger than --uvminim')
            raise Exception('--uvmaxim needs to be larger than --uvminim')
    if args['uvmax'] is not None and args['uvmin'] is not None and type(args['uvmax']) is not list and type(
            args['uvmin']) is not list:
        if args['uvmax'] <= args['uvmin']:
            terminal_print('--uvmax needs to be larger than --uvmin')
            raise Exception('--uvmaxim needs to be larger than --uvmin')
            # print(args['uvmax'], args['uvmin'], args['uvminim'],args['uvmaxim'])

    if ('fulljones' in args['soltype_list'] or 'leakage' in args['soltype_list'] or 'leakageamplitude' in args['soltype_list']) and args['doflagging'] and not args['forwidefield']:
        terminal_print('--doflagging is True, cannot be combined with fulljones solve, set it to False or use --forwidefield')
        raise Exception('--doflagging is True, cannot be combined with fulljones solve')

    if args['iontimefactor'] <= 0.0:
        terminal_print('BLsmooth iontimefactor needs to be positive')
        raise Exception('BLsmooth iontimefactor needs to be positive')
    if args['iontimefactor'] > 10.0:
        terminal_print('BLsmooth iontimefactor is way too high')
        raise Exception('BLsmooth iontimefactor is way too high')

    if args['ionfreqfactor'] <= 0.0:
        terminal_print('BLsmooth tecfactor needs to be positive')
        raise Exception('BLsmooth tecfactor needs to be positive')
    if args['ionfreqfactor'] > 10000.0:
        terminal_print('BLsmooth tecfactor is way too high')
        raise Exception('BLsmooth tecfactor is way too high')

    if args['phaseshiftbox'] is not None:
        if not os.path.isfile(args['phaseshiftbox']):
            terminal_print('Cannot find:', args['phaseshiftbox'])
            raise Exception('Cannot find:' + args['phaseshiftbox'])

    if args['beamcor'] not in ['auto', 'yes', 'no']:
        terminal_print('beamcor is not auto, yes, or no')
        raise Exception('Invalid input, beamcor is not auto, yes, or no')

    if args['beamcor'] != 'auto' and args['telescope'] != 'LOFAR':
        terminal_print('beamcor is a LOFAR specific option, keep this at "auto"')
        raise Exception('beamcor is a LOFAR specific option, keep this at "auto"')

    if args['DDE_predict'] not in ['DP3', 'WSCLEAN']:
        terminal_print('DDE-predict is not DP3 or WSCLEAN')
        raise Exception('DDE-predict is not DP3 or WSCLEAN')

    for nrtmp in args['normamps_list']:
        if nrtmp not in ['normamps_per_ant', 'normslope', 'normamps', 'normslope+normamps',
                         'normslope+normamps_per_ant'] and nrtmp is not None:
            terminal_print(
                'Invalid input: --normamps_list can only contain "normamps", "normslope", "normamps_per_ant", "normslope+normamps", "normslope+normamps_per_ant" or None')
            raise Exception(
                'Invalid input: --normamps_list can only contain "normamps", "normslope", "normamps_per_ant", "normslope+normamps", "normslope+normamps_per_ant" or None')

    for antennaconstraint in args['antennaconstraint_list']:
        if antennaconstraint not in ['superterp', 'coreandfirstremotes', 'core', 'remote',
                                     'all', 'international', 'alldutch', 'core-remote',
                                     'coreandallbutmostdistantremotes', 'alldutchbutnoST001',
                                     'distantremote', 'alldutchandclosegerman', 'corebutsuperterp'] \
                and antennaconstraint is not None:
            terminal_print(
                'Invalid input, antennaconstraint can only be core, superterp, corebutsuperterp, coreandfirstremotes, remote, alldutch, international, alldutchandclosegerman, or all')
            raise Exception(
                'Invalid input, antennaconstraint can only be core, superterp, corebutsuperterp, coreandfirstremotes, remote, alldutch, international, alldutchandclosegerman, or all')

    for resetsols in args['resetsols_list']:
        if resetsols not in ['superterp', 'coreandfirstremotes', 'core', 'remote', 'corebutsuperterp',
                             'all', 'international', 'alldutch', 'core-remote', 'coreandallbutmostdistantremotes',
                             'alldutchbutnoST001', 'distantremote', 'alldutchandclosegerman'] \
                and resetsols is not None:
            terminal_print(
                'Invalid input, resetsols can only be core, superterp, corebutsuperterp, coreandfirstremotes, remote, alldutch, international, distantremote, alldutchandclosegerman, or all')
            raise Exception(
                'Invalid input, resetsols can only be core, superterp, corebutsuperterp, coreandfirstremotes, remote, alldutch, international, distantremote, alldutchandclosegerman, or all')

    # most antenna constraint types are LOFAR station groups, check what exists for this telescope
    available = available_antennaconstraints(args['telescope'])
    for ctype in args['antennaconstraint_list'] + args['resetsols_list']:
        if ctype is not None and ctype not in available:
            msg = ("antennaconstraint/resetsols '" + ctype + "' is not defined for telescope " +
                   args['telescope'] + ', available: ' + ', '.join(available))
            terminal_print(msg)
            raise Exception(msg)

    # the reference antenna must be in every MS (ST001 is created later with --phaseupstations)
    if args['refant'] is not None and not (args['refant'] == 'ST001' and args['phaseupstations'] is not None):
        for ms in mslist:
            with table(ms + '::ANTENNA', ack=False) as t:
                if args['refant'] not in t.getcol('NAME'):
                    msg = '--refant ' + args['refant'] + ' is not an antenna of ' + ms
                    terminal_print(msg)
                    raise Exception(msg)

    for soltype in args['soltype_list']:
        if soltype not in ['complexgain', 'scalarcomplexgain', 'scalaramplitude',
                           'amplitudeonly', 'phaseonly', 'fulljones', 'rotation',
                           'rotation+diagonal', 'rotation+diagonalphase',
                           'rotation+diagonalamplitude', 'rotation+scalar',
                           'rotation+scalaramplitude', 'rotation+scalarphase', 'tec',
                           'tecandphase', 'scalarphase',
                           'scalarphasediff', 'scalarphasediffFR', 'phaseonly_phmin',
                           'rotation_phmin', 'tec_phmin',
                           'tecandphase_phmin', 'scalarphase_phmin', 'scalarphase_slope',
                           'phaseonly_slope', 'faradayrotation', 'faradayrotation+diagonal',
                           'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude',
                           'faradayrotation+scalar', 'faradayrotation+scalaramplitude',
                           'faradayrotation+scalarphase', 'leakage', 'leakageamplitude',
                           'tec+phase', 'tec+delay', 'tec+phase+delay', 'treemf',
                           'treemfphasediff'] and soltype is not None:
            terminal_print('Invalid soltype input')
            raise Exception('Invalid soltype input')

    # check that there is only on scalarphasediff solve and it the first entry
    phasediff_types = ['scalarphasediff', 'scalarphasediffFR', 'treemfphasediff']
    if any(soltype in phasediff_types for soltype in args['soltype_list']):
        if args['soltype_list'][0] not in phasediff_types:
            terminal_print('scalarphasediff/scalarphasediffFR/treemfphasediff need to be to first solves in the list')
            raise Exception('scalarphasediff/scalarphasediffFR/treemfphasediff need to be to first solves in the list')
        sccount = 0
        for soltype in args['soltype_list']:
            if soltype in phasediff_types:
                sccount = sccount + 1
        if sccount > 1:
            terminal_print('only one scalarphasediff/scalarphasediffFR/treemfphasediff solve allowed')
            raise Exception('only one scalarphasediff/scalarphasediffFR/treemfphasediff solve allowed')

    # check if args['facetdirections'] is a string and if the file exists
    if args['facetdirections'] is not None:
        if type(args['facetdirections']) is str:
            if not os.path.isfile(args['facetdirections']):
                terminal_print('--facetdirections file does not exist')
                raise Exception('--facetdirections file does not exist')
        # check if args['facetdirections'] is a list and if each item is a file that exists
        elif type(args['facetdirections']) is list:
            # do not allow list of length 1, because then the user should just provide a string input instead of a list
            if len(args['facetdirections']) == 1:
                terminal_print('If only one facetdirections file is provided, it should be provided as a string, not as a list of strings')
                raise Exception('If only one facetdirections file is provided, it should be provided as a string, not as a list of strings')
            for facetdirections in args['facetdirections']:
                if not os.path.isfile(facetdirections):
                    terminal_print('--facetdirections file does not exist:', facetdirections)
                    raise Exception('--facetdirections file does not exist:' + facetdirections)
            # check that the facetdirections files have the same number of directions and that the soltypelist_includedir_ref have the same dimensions for each file, otherwise we cannot use the same reference list for all files in the DDE solve
            dirs_ref, solints_ref, smoothness_ref, soltypelist_includedir_ref = parse_facetdirections(args['facetdirections'][0], 1000, args=args)
            
            if solints_ref is not None:
                    solints_ref = np.swapaxes(np.array([solints_ref] * len(mslist)), 1, 0).T.tolist()
            if smoothness_ref is not None:
                    smoothness_ref = np.swapaxes(np.array([smoothness_ref] * len(mslist)), 1, 0).T.tolist()
            if soltypelist_includedir_ref is not None:
                soltypelist_includedir_ref = np.swapaxes(np.array([soltypelist_includedir_ref] * len(mslist)), 1, 0).T.tolist()
            for facetdirections in args['facetdirections'][1:]:
                dirs, solints, smoothness, soltypelist_includedir = parse_facetdirections(facetdirections, 1000, args=args)
                if solints is not None:
                    solints = np.swapaxes(np.array([solints] * len(mslist)), 1, 0).T.tolist()
                if smoothness is not None:
                    smoothness = np.swapaxes(np.array([smoothness] * len(mslist)), 1, 0).T.tolist()
                if soltypelist_includedir is not None:
                    soltypelist_includedir = np.swapaxes(np.array([soltypelist_includedir] * len(mslist)), 1, 0).T.tolist()

                if len(dirs) != len(dirs_ref):
                    terminal_print('All facetdirections files need to have the same number of directions')
                    raise Exception('All facetdirections files need to have the same number of directions')
                if solints_ref is not None:
                    # check that the shapes match for all files
                    if solints is None:
                        terminal_print('All facetdirections files need to have solints if one of them has solints')
                        raise Exception('All facetdirections files need to have solints if one of them has solints')
                    if np.array(solints).shape != np.array(solints_ref).shape:
                        terminal_print('All facetdirections files need to have the same solints shape')
                        raise Exception('All facetdirections files need to have the same solints shape')
                if smoothness_ref is not None:
                    # check that the shapes match for all files
                    if smoothness is None:
                        terminal_print('All facetdirections files need to have smoothness if one of them has smoothness')
                        raise Exception('All facetdirections files need to have smoothness if one of them has smoothness')
                    if np.array(smoothness).shape != np.array(smoothness_ref).shape:
                        terminal_print('All facetdirections files need to have the same smoothness shape')
                        raise Exception('All facetdirections files need to have the same smoothness shape')
                if soltypelist_includedir_ref is not None:
                    # check that the shapes match for all files
                    if soltypelist_includedir is None:
                        terminal_print('All facetdirections files need to have soltypelist_includedir if one of them has soltypelist_includedir')
                        raise Exception('All facetdirections files need to have soltypelist_includedir if one of them has soltypelist_includedir')
                # check that shapes of dirs match for all files
                if np.array(dirs).shape != np.array(dirs_ref).shape:
                    terminal_print('All facetdirections files need to have the same dirs shape')
                    raise Exception('All facetdirections files need to have the same dirs shape')
                # check that the dirs themselves match for all files
                if not np.array_equal(dirs, dirs_ref):
                    terminal_print('All facetdirections files need to have the same directions')
                    raise Exception('All facetdirections files need to have the same directions')             
            # check that the length of this list equals the length of mslist
            assert len(args['facetdirections']) == len(mslist), 'If --facetdirections is a list, its length needs to match the length of the mslist'
            
            # check that the start values for the selfcalcycle_sel are the same for all files, otherwise we cannot use the same reference list for all files in the DDE solve
            startvals_ref = parse_facetdirections(args['facetdirections'][0], 0, return_only_selfcalcycle_sel=True, args=args)
            for facetdirections in args['facetdirections'][1:]:
                startvals = parse_facetdirections(facetdirections, 0, return_only_selfcalcycle_sel=True, args=args)
                # check that arrays are the same
                if not np.array_equal(startvals, startvals_ref):
                    terminal_print('All facetdirections files need to have the same selfcalcycle_sel start values if they are provided')
                    raise Exception('All facetdirections files need to have the same selfcalcycle_sel start values if they are provided')

        else: # raise errror if args['facetdirections'] is not a string or a list
            terminal_print('--facetdirections needs to be a string or a list of strings')
            raise Exception('--facetdirections needs to be a string or a list of strings')            

    if args['DDE']:
        if 'fulljones' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonal' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonalamplitude' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+diagonalphase' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalar' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalarphase' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'rotation+scalaramplitude' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonal' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonalamplitude' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+diagonalphase' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalar' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalarphase' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'faradayrotation+scalaramplitude' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'leakage' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')
        if 'leakageamplitude' in args['soltype_list']:
            terminal_print('Invalid soltype input in combination with --DDE')
            raise Exception('Invalid soltype input in combination with --DDE')    


        if args['wscleanskymodel'] is not None and args['facetdirections'] is None:
            terminal_print('If --DDE and --wscleanskymodel are set provide a direction file via --facetdirections')
            raise Exception('DDE with a wscleanskymodel requires a user-specified facetdirections')
        if args['wscleanskymodel'] is not None and args['Nfacets'] > 0:
            terminal_print('If --DDE and --wscleanskymodel are set you cannot use Nfacets')
            raise Exception('If --DDE and --wscleanskymodel are set you cannot use Nfacets')

    for ms in mslist:
        if not check_phaseup_station(ms):
            for soltype_id, soltype in enumerate(args['soltype_list']):
                if soltype in ['scalarphasediff', 'scalarphasediff', 'treemfphasediff']:
                    if args['antennaconstraint_list'][soltype_id] not in ['superterp', 'coreandfirstremotes', 'core',
                                                                          'remote', 'distantremote',
                                                                          'all', 'international', 'alldutch',
                                                                          'core-remote', 'corebutsuperterp',
                                                                          'coreandallbutmostdistantremotes',
                                                                          'alldutchbutnoST001',
                                                                          'alldutchandclosegerman'] and args[
                        'phaseupstations'] is None:
                        terminal_print(
                            'scalarphasediff/scalarphasediff type solves require a antennaconstraint, for example "core", or phased-up data')
                        raise Exception(
                            'scalarphasediff/scalarphasediff type solves require a antennaconstraint, or phased-up data')

    if args['boxfile'] is not None:
        if not (os.path.isfile(args['boxfile'])):
            terminal_print('Cannot find boxfile, file does not exist')
            raise Exception('Cannot find boxfile, file does not exist')

    if args['fitsmask'] is not None and args['fitsmask'] != 'nofitsmask':
        if not (os.path.isfile(args['fitsmask'])):
            terminal_print('Cannot find fitsmask, file does not exist')
            raise Exception('Cannot find fitsmask, file does not exist')

    if args['fitsmask'] is not None and args['fitsmask_start'] is not None:
        if not (os.path.isfile(args['fitsmask'])):
            terminal_print('Cannot set fitsmask and fitsmask-start at the same time')
            raise Exception('Cannot set fitsmask and fitsmask-start at the same time')

    if args['DS9cleanmaskregionfile'] is not None: 
        if not (os.path.isfile(args['DS9cleanmaskregionfile'])):
            terminal_print('Cannot find DS9cleanmaskregionfile, file does not exist')
            raise Exception('Cannot find DS9cleanmaskregionfile, file does not exist')

    if args['DS9cleanmaskregionfile_exclude'] is not None: 
        if not (os.path.isfile(args['DS9cleanmaskregionfile_exclude'])):
            terminal_print('Cannot find DS9cleanmaskregionfile_exclude, file does not exist')
            raise Exception('Cannot find DS9cleanmaskregionfile_exclude, file does not exist')

    if args['skymodel'] is not None:
        if type(args['skymodel']) is str:
            # print(type(args['skymodel']), args['skymodel'][0])
            if not (os.path.isfile(args['skymodel'])) and not (os.path.isdir(args['skymodel'])):
                terminal_print('Cannot find skymodel, file does not exist', args['skymodel'])
                raise Exception('Cannot find skymodel, file does not exist')
        if type(args['skymodel']) is list:
            for skym in args['skymodel']:
                if not os.path.isfile(skym) and not os.path.isdir(skym):
                    terminal_print('Cannot find skymodel, file does not exist', skym)
                    raise Exception('Cannot find skymodel, file does not exist')

    if args['docircular'] and args['dolinear']:
        terminal_print('Conflicting input, docircular and dolinear used')
        raise Exception('Conflicting input, docircular and dolinear used')

    if which('DP3') is None:
        terminal_print('Cannot find DP3, forgot to source lofarinit.[c]sh?')
        raise Exception('Cannot find DP3, forgot to source lofarinit.[c]sh?')

    if which('wsclean') is None:
        terminal_print('Cannot find WSclean, forgot to source lofarinit.[c]sh?')
        raise Exception('Cannot find WSClean, forgot to source lofarinit.[c]sh?')

    if which('breizorro') is None:
        terminal_print('Cannot find breizorro, forgot to install it?')
        raise Exception('Cannot find breizorro, forgot to install it?')

    if which('taql') is None:
        terminal_print('Cannot find taql, forgot to install it?')
        raise Exception('Cannot find taql, forgot to install it?')

    # Check boxfile and imsize settings
    if args['boxfile'] is None and args['imsize'] is None:
        if not checklongbaseline(sorted(args['ms'])[0]) and args['telescope'] == 'LOFAR':
            terminal_print('Incomplete input detected, either boxfile or imsize is required')
            raise Exception('Incomplete input detected, either boxfile or imsize is required')
        elif args['telescope'] == 'MeerKAT':
            if not args['auto'] or args['DDE']:  # auto will set imsize for MeerKAT
                terminal_print('Incomplete input detected, either boxfile or imsize is required')
                raise Exception('Incomplete input detected, either boxfile or imsize is required')
        else:
            terminal_print('Incomplete input detected, either boxfile or imsize is required')
            raise Exception('Incomplete input detected, either boxfile or imsize is required')       

    if args['boxfile'] is not None and args['imsize'] is not None:
        terminal_print('Wrong input detected, both boxfile and imsize are set')
        raise Exception('Wrong input detected, both boxfile and imsize are set')

    if args['imager'] not in ['DDFACET', 'WSCLEAN']:
        terminal_print('Wrong input detected for option --imager, should be DDFACET or WSCLEAN')
        raise Exception('Wrong input detected for option --imager, should be DDFACET or WSCLEAN')

    if args['phaseupstations'] is not None:
        if args['phaseupstations'] not in ['core', 'superterp']:
            terminal_print('Wrong input detected for option --phaseupstations, should be core or superterp')
            raise Exception('Wrong input detected for option --phaseupstations, should be core or superterp')
    if args['phaseupstations'] is not None:
        if args['phaseupstations'] in args['antennaconstraint_list']:
            terminal_print(
                'Wrong input detected for option --antennaconstraint-list, --phaseupstations is set and phased-up stations are not available anymore for --antennaconstraint-list')
            raise Exception(
                'Wrong input detected for option --antennaconstraint-list, --phaseupstations is set and phased-up stations are not available anymore for --antennaconstraint-list')

    if args['soltypecycles_list'][0] != 0:
        terminal_print('Wrong input detected for option --soltypecycles-list should always start with 0')
        raise Exception('Wrong input detected for option --soltypecycles-list should always start with 0')

    if len(args['soltypecycles_list']) != len(args['soltype_list']):
        terminal_print('Wrong input detected, length soltypecycles-list does not match that of soltype-list')
        raise Exception('Wrong input detected, length soltypecycles-list does not match that of soltype-list')

    # DP3 applies the smoothness constraint before the tec/delay or Faraday-rotation fit, so it smooths
    # over phase wraps the fit itself handles; the fitted frequency model already makes the solutions smooth
    for soltype_id, soltype in enumerate(args['soltype_list']):
        wronginput = False
        if soltype in ['tecandphase', 'tec', 'tec_phmin', 'tecandphase_phmin', 'tec+phase', 'tec+delay', 'tec+phase+delay', 'treemf', 'delay',
                       'treemfphasediff',
                       'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase',
                       'faradayrotation+diagonalamplitude', 'faradayrotation+scalar',
                       'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
            try:  # in smoothnessconstraint_list is not filled by the user
                if args['smoothnessconstraint_list'][soltype_id] > 0.0:
                    terminal_print('smoothnessconstraint should be 0.0 for a tec/delay/faradayrotation-like solve')
                    wronginput = True
            except:
                pass
            if wronginput:
                raise Exception('smoothnessconstraint should be 0.0 for a tec/delay/faradayrotation-like solve')

    for smoothnessconstraint in args['smoothnessconstraint_list']:
        if smoothnessconstraint < 0.0:
            terminal_print('Smoothnessconstraint must be equal or larger than 0.0')
            raise Exception('Smoothnessconstraint must be equal or larger than 0.0')
    for smoothnessreffrequency in args['smoothnessreffrequency_list']:
        if smoothnessreffrequency < 0.0:
            terminal_print('Smoothnessreffrequency must be equal or larger than 0.0')
            raise Exception('Smoothnessreffrequency must be equal or larger than 0.0')

    if (args['skymodel'] is not None) and (args['skymodelpointsource']) is not None:
        terminal_print('Wrong input, you cannot use a separate skymodel file and then also set skymodelpointsource')
        raise Exception('Wrong input, you cannot use a separate skymodel file and then also set skymodelpointsource')
    if (args['skymodelpointsource'] is not None and type(args['skymodelpointsource']) is not list):
        if args['skymodelpointsource'] <= 0.0:
            terminal_print('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
            raise Exception('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
    if type(args['skymodelpointsource']) is list:
        for skymp in args['skymodelpointsource']:
            if float(skymp) <= 0.0:
                terminal_print('Wrong input, flux density provided for skymodelpointsource is <= 0.0')
                raise Exception('Wrong input, flux density provided for skymodelpointsource is <= 0.0')

    if (args['msinstartchan'] < 0):
        terminal_print('Wrong input for msinstartchan, must be larger than zero')
        raise Exception('Wrong input for msinstartchan, must be larger than zero')

    if (args['msinnchan'] is not None):
        if (args['msinnchan'] <= 0):
            terminal_print('Wrong input for msinnchan, must be larger than zero')
            raise Exception('Wrong input for msinnchan, must be larger than zero')
    if (args['msinntimes'] is not None):
        if (args['msinntimes'] <= 1):
            terminal_print('Wrong input for msinntimes, must be larger than 1')
            raise Exception('Wrong input for msinntimes, must be larger than 1')

    if (args['skymodelpointsource'] is not None) and (args['predictskywithbeam']):
        terminal_print('Combination of skymodelpointsource and predictskywithbeam not supported')
        terminal_print('Provide a skymodel file to predict the sky with the beam')
        raise Exception('Combination of skymodelpointsource and predictskywithbeam not supported')

    if (args['wscleanskymodel'] is not None) and (args['skymodelpointsource']) is not None:
        terminal_print('Wrong input, you cannot use a wscleanskymodel and then also set skymodelpointsource')
        raise Exception('Wrong input, you cannot use a wscleanskymodel and then also set skymodelpointsource')

    if (args['wscleanskymodel'] is not None) and (args['skymodel']) is not None:
        terminal_print('Wrong input, you cannot use a wscleanskymodel and then also set skymodel')
        raise Exception('Wrong input, you cannot use a wscleanskymodel and then also set skymodel')

    if (args['wscleanskymodel'] is not None) and (args['predictskywithbeam']):
        terminal_print('Combination of wscleanskymodel and predictskywithbeam not supported')
        terminal_print('Provide a skymodel component file to predict the sky with the beam')
        raise Exception('Combination of wscleanskymodel and predictskywithbeam not supported')

    if (args['wscleanskymodel'] is not None) and (args['imager'] == 'DDFACET'):
        terminal_print('Combination of wscleanskymodel and DDFACET as an imager is not supported')
        raise Exception('Combination of wscleanskymodel and DDFACET as an imager is not supported')
    if (args['wscleanskymodel'] is not None):
        if len(glob.glob(args['wscleanskymodel'] + '-????-model.fits')) < 2 and len(glob.glob(args['wscleanskymodel'] + '-????-model.fits.gz')) < 2:
            terminal_print('Not enough WSClean channel model images found')
            terminal_print(
                'Matching WSClean model images:',
                glob.glob(args['wscleanskymodel'] + '-????-model.fits*'),
            )
            raise Exception('Not enough WSClean channel model images found')
        if (args['wscleanskymodel'].find('/') != -1):
            terminal_print('wscleanskymodel contains a slash, not allowed, needs to be in pwd')
            raise Exception('wscleanskymodel contains a slash, not allowed, needs to be in pwd')
        if (args['wscleanskymodel'].find('..') != -1):
            terminal_print('wscleanskymodel contains .., not allowed, needs to be in pwd')
            raise Exception('wscleanskymodel contains .., not allowed, needs to be in pwd')
    return


def check_if_ms_exists(mslist):
    """
    Check if all measurement sets in the provided list exist.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories to check.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If any MS directory does not exist.
    """
    for ms in mslist:
        if not os.path.isdir(ms):
            # print in red color
            terminal_print("\033[91m" + ms + "\033[0m", "\033[91m does not exist\033[0m")
            raise Exception('ms does not exist')
    return


def removenonms(mslist):
    """
    Remove files that are not MS (ending on wrong extension)

    Parameters
    ----------
    mslist : list of str
        Candidate paths from which non-Measurement-Set files are removed.

    Returns
    -------
    New list
    """
    newmslist = []
    for ms in mslist:
        if ms.lower().endswith(('.h5', '.png', '.parset', '.fits', '.backup', '.obj', '.log', '.reg', '.gz', '.tar',
                                '.tmp', '.ddfcache')) or \
                ms.lower().startswith(('solution_plots_', 'solintimage')):
            terminal_print('WARNING, removing ', ms, 'not a ms-type? Removed it!')
        else:
            newmslist.append(ms)
    return newmslist


def check_valid_ms(mslist):
    """
    Validates a list of Measurement Set (MS) directories.
    This function performs the following checks on each MS in the provided list:
    1. Verifies that each MS path exists and is a directory.
    2. Ensures that no MS path starts with a '.' character (to avoid relative or hidden paths).
    3. Checks that each MS contains more than 20 unique time steps (sufficient data for self-calibration).
    4. Ensures there are no duplicate MS entries in the list.
    5. Checks that there is only one spectral window (SPW) in each MS.
    6. Validates that each MS has a perfectly regularized time-baseline grid structure.
    7. Checks that the MS is a directory and not some other file type.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories to validate.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If any MS directory does not exist.
    Exception
        If any MS path starts with a '.' character.
    Exception
        If any MS contains 20 or fewer unique time steps.
    Exception
        If there are duplicate MS entries in the list.
    Exception
        If any MS contains more than one spectral window (SPW).
    Exception
        If any MS is not perfectly regularized.
    Exception
        If any MS path is not a directory.
    """
    for ms in mslist:
        if not os.path.isdir(ms):
            terminal_print('Measurement Set:', ms, ' does not exist')
            raise Exception('ms does not exist')
        if ms.startswith("."):
            terminal_print('Measurement Set:', ms, ' This ms starts with a "." character, this is not allowed')
            raise Exception('Invalid ms name, do not use relative paths')

    # check we have a directory and not some other file type 
    for ms in mslist:
        if not os.path.isdir(ms):
            terminal_print('Measurement Set:', ms, ' is not a directory, this is not allowed')
            raise Exception('Invalid ms name, do not use relative paths')

    # check that each MS contains only one spectral window (SPW)
    for ms in mslist:
        with table(ms + '::SPECTRAL_WINDOW', readonly=True) as t:
            n_spws = t.nrows()
        if n_spws != 1:
            terminal_print('Measurement Set:', ms, ' contains more than one spectral window (SPW)')
            raise Exception('Each MS must contain exactly one spectral window (SPW)')

    # check if MS is regularized
    for ms in mslist:
        if not is_ms_regularized(ms):
            terminal_print('Measurement Set:', ms, ' is not perfectly regularized')
            raise Exception('Each MS must have a perfectly regularized time-baseline grid structure')

    # check that each MS contains more than 20 unique time steps
    for ms in mslist:
        t = table(ms, ack=False)
        times = np.unique(t.getcol('TIME'))
        telescope = get_telescope_from_ms(mslist[0])
        if (len(times) <= 20) and not (telescope == 'MWA'): # MWA has many very short scans
            terminal_print('---------------------------------------------------------------------------')
            terminal_print('ERROR, ', ms, 'not enough timesteps in ms/too short observation')
            terminal_print('---------------------------------------------------------------------------')
            raise Exception(
                'You are providing an MS with less than 21 timeslots, that is not enough to self-calibrate on')
        t.close()

    # check for duplicates in the mslist
    if any(mslist.count(x) > 1 for x in mslist):
        terminal_print('There are duplicates in the mslist, please remove them')
        raise Exception('There are duplicates in the mslist, please remove them')
    return
