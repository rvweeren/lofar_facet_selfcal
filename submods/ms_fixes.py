"""Functions that repair or sanitise Measurement Sets (weights, UVW, time axis,
antenna tables, GMRT/VLA specific issues) before self-calibration."""

import gc
import logging
import os
import re
import shutil
import sys

import numpy as np
from casacore.tables import table, taql  # type: ignore

from submods.runtime_utils import run, terminal_print
from submods.split_irregular_timeaxis import regularize_ms, split_ms


def remove_syspower(mslist):
    """
    Remove the SYSPOWER column from a measurement set, if it exists.

    This is necessary because the SYSPOWER column, if it is large, makes DP3 slow.
    Only remove the SYSPOWER column if it exists, otherwise do nothing for EVLA and VLA.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Set paths.
    """
    if isinstance(mslist, str):        
        mslist = [mslist]
    for ms in mslist:
        if get_telescope_from_ms(ms) in ['EVLA', 'VLA']:
            with table(ms, readonly=False, ack=False) as t:
                if 'SYSPOWER' in t.keywordnames():
                    terminal_print('Removing SYSPOWER column from ' + ms)
                    t.removekeyword('SYSPOWER')
                   
            if os.path.isdir(ms + '/SYSPOWER'):
                shutil.rmtree(ms + '/SYSPOWER')


def fix_GMRT_weights(mslist, args):
    """
    Copy parallel-hand weights into the cross-hand GMRT weight columns.

    Parameters
    ----------
    mslist : list of str
        Measurement Set paths to update.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is used.

    Returns
    -------
    None
        Measurement Sets are modified in place.
    """
    if args['telescope'] != 'GMRT':
        return
    
    for ms in mslist:
        terminal_print('Fixing RL and LR weights in MS', ms)
        run("taql" + " 'update " + ms + " set WEIGHT_SPECTRUM[,1]=WEIGHT_SPECTRUM[,0]'", taql=True) # set RL weights to RR weights
        run("taql" + " 'update " + ms + " set WEIGHT_SPECTRUM[,2]=WEIGHT_SPECTRUM[,3]'", taql=True) # set LR weights to LL weights
    return


def updateCrosshand(ms, variance, chunk_size=10000):
    """
    Fill the XY/RL and YX/LR cross-hand polarizations with complex Gaussian random noise
    based on the provided variance.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    variance : float or tuple/list of (float, float)
        The noise variance in Jy^2. If a tuple (var_RR, var_LL) is provided,
        the geometric mean sqrt(var_RR * var_LL) is used for the cross-hands.
    chunk_size : int, default=1000
        Number of rows to process at once.
    """
    logging.info("- Adaptation of cross-hand polarisation visibility data -")
    logging.info("Filling crosshand polarizations with random noise in MS %s", ms)
    with table(ms, readonly=False, ack=False) as t:
        if 'DATA' not in t.colnames():
            terminal_print("DATA column not found, skipping.")
            raise Exception("DATA column not found in the measurement set.")

        # Determine cross-hand noise variance and standard deviation per component (Re and Im)
        if isinstance(variance, (tuple, list, np.ndarray)):
            var_cross = float(np.sqrt(variance[0] * variance[1]))
        else:
            var_cross = float(variance)

        # For a complex visibility V = Re + 1j*Im with total Var(V) = var_cross,
        # Var(Re) = Var(Im) = var_cross / 2.
        sigma_1d = np.sqrt(max(var_cross, 0.0) / 2.0)
        logging.info(f"Using crosshand variance = {var_cross:.6e} Jy^2 (1D sigma = {sigma_1d:.6e} Jy)")

        # Check shape using a single-row sample
        sample = t.getcol("DATA", startrow=0, nrow=1)
        if sample.shape[-1] != 4:
            logging.warning("Expected 4 pols in DATA, found %d. Skipping.", sample.shape[-1])
            return

        rng = np.random.default_rng()
        total_rows = t.nrows()
        for start_row in range(0, total_rows, chunk_size):
            nrow = min(chunk_size, total_rows - start_row)
            chunk = t.getcol("DATA", startrow=start_row, nrow=nrow)
            nchan = chunk.shape[1]
            
            # Generate independent complex Gaussian noise for RL (idx 1) and LR (idx 2)
            noise_rl = (rng.normal(0.0, sigma_1d, size=(nrow, nchan)) + 
                        1j * rng.normal(0.0, sigma_1d, size=(nrow, nchan))).astype(chunk.dtype)
            noise_lr = (rng.normal(0.0, sigma_1d, size=(nrow, nchan)) + 
                        1j * rng.normal(0.0, sigma_1d, size=(nrow, nchan))).astype(chunk.dtype)

            chunk[:, :, 1] = noise_rl
            chunk[:, :, 2] = noise_lr

            t.putcol("DATA", chunk, startrow=start_row, nrow=nrow)
            del chunk, noise_rl, noise_lr
            terminal_print(f"Processed rows {start_row} to {start_row + nrow} of {total_rows} for cross-hand noise injection.")
            gc.collect()


def parse_bad_freq_ranges(bad_freq_ranges):
    """
    Parse bad frequency range specifications into a list of (f_min, f_max) tuples in Hz.

    Parameters
    ----------
    bad_freq_ranges : str, list, or tuple
        Bad frequency range specifications. Supported inputs:
        - string in DP3/preflagger format (e.g., "99MHz..128MHz,167MHz..188MHz")
        - list/tuple of strings: ["99MHz..128MHz", "167MHz..188MHz"]
        - list/tuple of frequency tuples/lists: [(99e6, 128e6), ...]

    Returns
    -------
    list of tuple
        List of (f_min, f_max) tuples in Hz.
    """
    if not bad_freq_ranges:
        return []

    def _to_hz(val):
        """
        Convert a frequency value to Hz.

        Parameters
        ----------
        val : float or str
            Frequency value, optionally with a unit suffix.

        Returns
        -------
        float
            Frequency in Hz.
        """
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        s_lower = s.lower()
        if s_lower.endswith("ghz"):
            return float(s[:-3].strip()) * 1e9
        elif s_lower.endswith("mhz"):
            return float(s[:-3].strip()) * 1e6
        elif s_lower.endswith("khz"):
            return float(s[:-3].strip()) * 1e3
        elif s_lower.endswith("hz"):
            return float(s[:-2].strip())
        else:
            return float(s)

    parsed_ranges = []
    if isinstance(bad_freq_ranges, str):
        clean_str = bad_freq_ranges.strip().lstrip("[").rstrip("]")
        tokens = [t.strip() for t in re.split(r"[,;\s]+", clean_str) if t.strip()]
    elif isinstance(bad_freq_ranges, (list, tuple)):
        tokens = bad_freq_ranges
    else:
        return []

    for item in tokens:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            f1, f2 = _to_hz(item[0]), _to_hz(item[1])
            parsed_ranges.append((min(f1, f2), max(f1, f2)))
        elif isinstance(item, str) and ".." in item:
            parts = item.split("..")
            if len(parts) == 2:
                f1, f2 = _to_hz(parts[0]), _to_hz(parts[1])
                parsed_ranges.append((min(f1, f2), max(f1, f2)))

    return parsed_ranges


def getVarianceRRLL(ms, data_column="DATA", num_baselines=6, candidate_pool_size=25,
                    top_quantile=0.85, max_flag_fraction=0.50, min_unflagged_samples=500,
                    bad_freq_ranges="99MHz..128MHz,167MHz..188MHz,243MHz..301MHz,347MHz..349.7MHz,355MHz..380MHz,390MHz..391MHz,750MHz..850MHz",
                    random_seed=42):
    """
    Determine the robust noise variance of the RR and LL visibility columns using the longest baselines.

    This method estimates the thermal noise variance per visibility sample for the RR and LL
    (or XX and YY) correlations. It selects the longest physical baselines in the array where
    extended sky emission is heavily resolved out. To be robust against RFI and systematic
    imperfections:
    1. Known bad frequency ranges (e.g. persistent RFI bands) are masked and excluded from
    the calculation.
    2. Consecutive frequency-channel differences (ΔV = V_{chan+1} - V_{chan}) are used to
    subtract out residual continuum sky flux, bandpass ripples, and fringe patterns.
    3. The Median Absolute Deviation (MAD) is computed separately on the real and imaginary parts
    to eliminate transient RFI spikes without being skewed by outliers.
    4. Multiple diverse antenna pairs are randomly sampled across the top baseline-length quantile
    to avoid bias if the antennas on the absolute longest baseline are malfunctioning or noisy.
    5. Baselines with excessive flagging or near-zero variance (dead antennas) are rejected.
    6. A cross-baseline consensus filter (MAD-based) discards outlier baseline variances.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    data_column : str, default="DATA"
        Name of the visibility data column in the Measurement Set to analyze
        (e.g., "DATA", "CORRECTED_DATA", "RESIDUAL_DATA").
    num_baselines : int, default=6
        Target number of healthy, independent long baselines to evaluate and include
        in the consensus variance calculation.
    candidate_pool_size : int, default=25
        Maximum number of candidate long baselines to draw from the top-length pool.
        Larger values provide more candidates for antenna diversity and backup if
        several baselines are flagged or dead.
    top_quantile : float, default=0.85
        Quantile cutoff (between 0.0 and 1.0) defining the threshold for "long baselines"
        based on 3D physical antenna separation in the ANTENNA subtable.
        Default 0.85 selects baselines in the top 15% longest baseline lengths.
    max_flag_fraction : float, default=0.50
        Maximum allowed fraction of flagged samples (0.0 to 1.0) on a candidate baseline.
        Baselines exceeding this flag threshold are skipped.
    min_unflagged_samples : int, default=500
        Minimum number of valid, unflagged adjacent-channel difference pairs required
        on a baseline. Baselines with fewer unflagged samples are discarded.
    bad_freq_ranges : str, list of str, list of tuples, or None, default="99MHz..128MHz,..."
        Frequency ranges with bad data/known RFI to ignore during variance calculation.
        Can be given as a comma/space-separated string (e.g. "99MHz..128MHz,167MHz..188MHz"),
        a bracketed string ("[99MHz..128MHz, ...]"), or a list of range strings / (fmin, fmax) tuples.
    random_seed : int or None, default=42
        Random seed used to shuffle candidate long baselines to ensure reproducible,
        diverse baseline selection across runs.

    Returns
    -------
    tuple of (float, float)
        A 2-tuple (var_RR, var_LL) containing the estimated thermal noise variance in Jy^2
        for the RR and LL polarizations (or XX and YY if linear).
    """
    logging.info("- Estimating robust visibility variance for RR and LL from long baselines -")
    
    rng = np.random.default_rng(random_seed)

    # 1. Identify RR and LL (or XX and YY) polarization indices
    with table(ms + "::POLARIZATION", readonly=True, ack=False) as tpol:
        corr_type = tpol.getcol("CORR_TYPE")[0]
    corr_map = {corr: idx for idx, corr in enumerate(corr_type)}
    
    if 5 in corr_map and 8 in corr_map:
        idx_rr, idx_ll = corr_map[5], corr_map[8]
    elif 9 in corr_map and 12 in corr_map:
        idx_rr, idx_ll = corr_map[9], corr_map[12]
    elif len(corr_type) == 2:
        idx_rr, idx_ll = 0, 1
    elif len(corr_type) >= 4:
        idx_rr, idx_ll = 0, 3
    else:
        idx_rr, idx_ll = 0, min(3, len(corr_type) - 1)

    pol_indices = {"RR": idx_rr, "LL": idx_ll}

    # 2. Identify and mask bad frequency channels
    parsed_bad_ranges = parse_bad_freq_ranges(bad_freq_ranges)
    with table(ms + "::SPECTRAL_WINDOW", readonly=True, ack=False) as tspect:
        chan_freqs = tspect.getcol("CHAN_FREQ")[0]
    bad_freq_mask = np.zeros(len(chan_freqs), dtype=bool)

    if parsed_bad_ranges:
        for f_min, f_max in parsed_bad_ranges:
            bad_freq_mask |= (chan_freqs >= f_min) & (chan_freqs <= f_max)
        n_bad = np.sum(bad_freq_mask)
        if n_bad > 0:
            logging.info(f"Ignoring {n_bad}/{len(chan_freqs)} frequency channels matching bad frequency ranges.")

    # 3. Compute physical baseline lengths from ANTENNA subtable
    with table(ms + "::ANTENNA", readonly=True, ack=False) as t_ant:
        ant_pos = t_ant.getcol("POSITION")
        ant_names = t_ant.getcol("NAME") if "NAME" in t_ant.colnames() else None
        num_ants = len(ant_pos)

    baseline_lengths = {}
    for a1 in range(num_ants):
        for a2 in range(a1 + 1, num_ants):
            dist = np.linalg.norm(ant_pos[a1] - ant_pos[a2])
            baseline_lengths[(a1, a2)] = dist

    sorted_bls = sorted(baseline_lengths.items(), key=lambda x: x[1], reverse=True)
    all_lengths = np.array([bl[1] for bl in sorted_bls])

    # Filter candidates from upper quantile of baseline lengths
    cutoff_dist = np.quantile(all_lengths, top_quantile)
    long_candidates = [pair for pair, dist in sorted_bls if dist >= cutoff_dist]
    if not long_candidates:
        long_candidates = [pair for pair, dist in sorted_bls[:candidate_pool_size]]

    # Shuffle candidates to sample various long baselines
    rng.shuffle(long_candidates)
    pool_candidates = long_candidates[:candidate_pool_size]

    # Ensure antenna diversity (avoid over-representing a single bad antenna)
    ant_usage_count = {i: 0 for i in range(num_ants)}
    selected_pairs = []
    for a1, a2 in pool_candidates:
        if ant_usage_count[a1] < 2 and ant_usage_count[a2] < 2:
            selected_pairs.append((a1, a2))
            ant_usage_count[a1] += 1
            ant_usage_count[a2] += 1
        if len(selected_pairs) >= candidate_pool_size:
            break

    for pair in pool_candidates:
        if pair not in selected_pairs:
            selected_pairs.append(pair)

    # 4. Robust variance estimator using channel differencing + MAD
    def robust_variance_from_visibilities(vis, flags):
        """
        Estimate visibility variance while excluding flagged samples.

        Parameters
        ----------
        vis : numpy.ndarray
            Visibility values.
        flags : numpy.ndarray
            Boolean flag mask.

        Returns
        -------
        float
            Robust variance estimate.
        """
        flags_eff = flags | bad_freq_mask[np.newaxis, :]
        diff = vis[:, 1:] - vis[:, :-1]
        valid_diff = (~flags_eff[:, 1:]) & (~flags_eff[:, :-1])
        diff_valid = diff[valid_diff]
        n_samples = diff_valid.size

        if n_samples < min_unflagged_samples:
            return np.nan, np.nan, n_samples

        diff_re = diff_valid.real
        diff_im = diff_valid.imag

        mad_re = np.median(np.abs(diff_re - np.median(diff_re)))
        mad_im = np.median(np.abs(diff_im - np.median(diff_im)))

        sigma_re = (1.4826 * mad_re) / np.sqrt(2.0)
        sigma_im = (1.4826 * mad_im) / np.sqrt(2.0)

        var_total = (sigma_re ** 2) + (sigma_im ** 2)
        std_total = np.sqrt(var_total)
        return float(var_total), float(std_total), n_samples

    # 5. Measure variance across selected baselines
    with table(ms, readonly=True, ack=False) as t_main:
        colnames = t_main.colnames()
        has_flag_row = "FLAG_ROW" in colnames
        if data_column not in colnames:
            raise KeyError(f"Column '{data_column}' not found in MS {ms}")

        baseline_results = []
        for a1, a2 in selected_pairs:
            taql_query = f"ANTENNA1 == {a1} && ANTENNA2 == {a2}"
            with t_main.query(taql_query) as sub_t:
                n_rows = sub_t.nrows()
                if n_rows == 0:
                    continue

                vis_data = sub_t.getcol(data_column)
                flags = sub_t.getcol("FLAG")

                if has_flag_row:
                    flag_rows = sub_t.getcol("FLAG_ROW")
                    flags[flag_rows, :, :] = True

                bl_length = baseline_lengths.get((min(a1, a2), max(a1, a2)), 0.0)
                a1_label = ant_names[a1] if ant_names is not None else str(a1)
                a2_label = ant_names[a2] if ant_names is not None else str(a2)

                res_entry = {
                    "baseline": (a1, a2),
                    "antennas": (a1_label, a2_label),
                    "length_m": bl_length,
                }

                skip_baseline = False
                for pol_name, pol_idx in pol_indices.items():
                    vis_pol = vis_data[:, :, pol_idx]
                    flag_pol = flags[:, :, pol_idx]

                    flags_eff_pol = flag_pol | bad_freq_mask[np.newaxis, :]
                    flag_frac = float(np.mean(flags_eff_pol))
                    res_entry[f"flag_frac_{pol_name}"] = flag_frac

                    if flag_frac > max_flag_fraction:
                        skip_baseline = True
                        break

                    var_val, std_val, n_diffs = robust_variance_from_visibilities(vis_pol, flag_pol)

                    # Discard dead antenna or numerical failure
                    if np.isnan(var_val) or var_val <= 1e-18:
                        skip_baseline = True
                        break

                    res_entry[f"var_{pol_name}"] = var_val
                    res_entry[f"std_{pol_name}"] = std_val

                if not skip_baseline:
                    baseline_results.append(res_entry)

                if len(baseline_results) >= num_baselines * 2:
                    break

    if len(baseline_results) == 0:
        logging.warning("No unflagged healthy long baselines found. Falling back to default variance 1.0.")
        return 1.0, 1.0

    # 6. Outlier rejection across baselines
    consensus_vars = {}
    for pol_name in ["RR", "LL"]:
        vars_all = np.array([b[f"var_{pol_name}"] for b in baseline_results])
        med_var = np.median(vars_all)
        mad_var = np.median(np.abs(vars_all - med_var))

        cutoff = max(3.0 * (1.4826 * mad_var), 0.5 * med_var)
        good_mask = np.abs(vars_all - med_var) <= cutoff

        if np.sum(good_mask) == 0:
            good_mask = np.ones(len(vars_all), dtype=bool)

        consensus_vars[pol_name] = float(np.median(vars_all[good_mask]))

    var_rr = consensus_vars["RR"]
    var_ll = consensus_vars["LL"]

    logging.info(f"Consensus visibility noise variance: RR = {var_rr:.6e} Jy^2 (std = {np.sqrt(var_rr):.6e} Jy), "
                    f"LL = {var_ll:.6e} Jy^2 (std = {np.sqrt(var_ll):.6e} Jy)")
    logging.info(f"Estimated from {len(baseline_results)} healthy long baselines.")
    for b in baseline_results[:5]:
        logging.debug(f"  BL {b['antennas'][0]}-{b['antennas'][1]} ({b['length_m']:.1f} m): "
                        f"RR var={b.get('var_RR', np.nan):.3e}, LL var={b.get('var_LL', np.nan):.3e}, "
                        f"flag_RR={b.get('flag_frac_RR', 0.0):.1%}")

    return var_rr, var_ll


def fix_time_axis_gmrt(mslist, args):
    """
    Fixes the TIME and INTERVAL columns in Measurement Sets (MS) from the GMRT telescope.
    For each MS in the provided list, this function:
    - Checks if the telescope is GMRT by reading the 'TELESCOPE_NAME' from the OBSERVATION table.
    - If so, computes the most common time interval between observations, ignoring large jumps.
    - Updates the INTERVAL column to have a uniform value equal to the computed interval.
    - Adjusts the TIME column so that all time stamps are aligned to this uniform interval.
    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Sets (MS) to process.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is used.
    Notes
    -----
    This function modifies the MS files in place.
    """
    for ms in mslist:
        if args['telescope'] == 'GMRT': 
            with table(ms, ack=False, readonly=False) as t:
               times = taql('select distinct TIME from $t').getcol("TIME") 
               intervals = times[1 : ] - times[ : -1]
               intervals = intervals[intervals < 1.5 * np.min(intervals)] # Select only intervals that do not correspond with big jumps in time.
               intervalprecise = np.mean(intervals)
               
               # Update INTERVAL column.
               intervalsold    = t.getcol("INTERVAL")
               intervalsnew    = np.ones_like(intervalsold) * intervalprecise
               t.putcol("INTERVAL", intervalsnew)

               # Update TIME column.
               timesold        = t.getcol("TIME")
               timesnew        = timesold[0] + np.round((timesold - timesold[0]) / intervalprecise, 0) * intervalprecise
               t.putcol("TIME", timesnew)
               terminal_print('Time axis interval set to', intervalprecise)


def fix_antenna_info_gmrt(mslist, args):
    """
    Removes specific antennas from a Measurement Set if the telescope is GMRT.

    Parameters
    ----------
    mslist : list
        List of input Measurement Sets.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is used.

    Notes
    -----
    Behavior:
    - Checks the 'TELESCOPE_NAME' in the OBSERVATION table of the Measurement Set.
    - If the telescope is 'GMRT', removes antennas 'C07', 'S05', and 'E01' by calling remove_antennas.
    Requires:
    - The 'table' context manager for reading Measurement Set tables.
    - The 'remove_antennas' function to perform the actual removal.
    """
    for ms in mslist:
        if args['telescope'] == 'GMRT':    
            antennas_to_remove = ['C07', 'S05', 'E01'] # these antennas were never built, but are present in the ANTENNA table
            remove_antennas(ms, antennas_to_remove)


def remove_antennas(ms_path, antennas_to_remove):
    """
    Removes specified antennas from the POINTING and ANTENNA tables of a Measurement Set.
    Adapted from code by Emanuele De Rubeis.
    This function updates the POINTING and ANTENNA tables in the given Measurement Set (MS) by
    removing all rows corresponding to antennas whose names are listed in `antennas_to_remove`.
    It is useful for fixing inconsistencies where the ANTENNA table contains antennas not present
    in the POINTING table.

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory.
    antennas_to_remove : list of str
        List of antenna names to be removed from the tables.

    Raises
    ------
    Exception
        If there is an error accessing or modifying the Measurement Set tables.

    Notes
    -----
    Side Effects:
    Modifies the POINTING and ANTENNA tables in-place by removing specified antennas.
    Prints information about removed rows or errors encountered.
    """
    
    with table(f"{ms_path}::POINTING", readonly=False) as pointing_table:
        rows_to_remove = [i for i, antenna in enumerate(pointing_table) if antenna['NAME'] in antennas_to_remove]
        terminal_print(f"Rows to remove from POINTING: {rows_to_remove}")
        if len(rows_to_remove) >0:
            pointing_table.removerows(rows_to_remove)
            terminal_print(f"Removed rows from POINTING: {rows_to_remove}")
    
    with table(f"{ms_path}::ANTENNA", readonly=False) as antenna_table:
        rows_to_remove = [i for i, antenna in enumerate(antenna_table) if antenna['NAME'] in antennas_to_remove]
        if len(rows_to_remove) > 0:
            antenna_table.removerows(rows_to_remove)
            terminal_print(f"Removed rows from ANTENNA: {rows_to_remove}")


def fix_twopol_ms(mslist):
    """
    Fixes 2-polarisation Measurement Sets (MS) by converting them to 4-polarisation format.
    This function checks each Measurement Set in the provided list to determine if it is a 2-polarisation MS.
    If so, it runs a helper script to create fake cross-hand correlations, effectively converting
    the MS to a 4-polarisation format.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Sets (MS) to process.

    Notes
    -----
    - The function assumes the existence of a helper script 'fix_twopol_ms.py' located in the 'submodpath' directory.
    """
    # for standalone running
    datapathc = os.path.dirname(os.path.abspath(__file__))
    submodpathc = '/'.join(datapathc.split('/')[0:-1])+'/submods'
    
    # check if mslist is of type list, if not convert it to a list
    if not isinstance(mslist, list):
        mslist = [mslist]
    for ms in mslist:
        if is_two_pol_ms(ms):
            terminal_print('Fixing 2-pol MS:', ms)
            terminal_print('Convert to 4-pol by creating fake cross-hand correlations')
            cmd = f'python {submodpathc}/fix_twopol_ms.py -c 10000 {ms}'
            terminal_print('Command:', cmd)
            run(cmd)


def fix_variable_reference_pointing(mslist):
    """
    Convert variable FIELD.PHASE_DIR references to fixed references when all rows agree.

    The direction values are left unchanged. Measurement Sets with mixed reference
    frames are rejected because their values must be converted before the metadata
    can safely be made fixed.
    """
    updates = []
    for ms in mslist:
        field_path = os.fspath(ms) + '::FIELD'
        with table(field_path, ack=False, readonly=True) as field_table:
            columns = field_table.colnames()
            if 'PHASE_DIR' not in columns:
                raise ValueError(f'{ms}: FIELD table has no PHASE_DIR column')
            if field_table.nrows() == 0:
                raise ValueError(f'{ms}: FIELD table has no rows')

            measure_info = field_table.getcolkeywords('PHASE_DIR').get('MEASINFO')
            if not isinstance(measure_info, dict) or measure_info.get('type') != 'direction':
                raise ValueError(f'{ms}: FIELD.PHASE_DIR has invalid MEASINFO metadata')

            ref_column = measure_info.get('VarRefCol')
            if not ref_column:
                if measure_info.get('Ref'):
                    continue
                raise ValueError(
                    f'{ms}: FIELD.PHASE_DIR has neither a fixed Ref nor a VarRefCol'
                )
            if ref_column not in columns:
                raise ValueError(
                    f'{ms}: FIELD.PHASE_DIR references missing column {ref_column}'
                )

            reference_values = np.asarray(field_table.getcol(ref_column)).reshape(-1)
            if reference_values.size != field_table.nrows():
                raise ValueError(
                    f'{ms}: {ref_column} has {reference_values.size} values for '
                    f'{field_table.nrows()} FIELD rows'
                )

            ref_codes = np.asarray(measure_info.get('TabRefCodes', [])).reshape(-1).tolist()
            ref_types = np.asarray(measure_info.get('TabRefTypes', [])).reshape(-1).tolist()
            if not ref_types or len(ref_codes) != len(ref_types):
                raise ValueError(
                    f'{ms}: FIELD.PHASE_DIR has incomplete variable-reference metadata'
                )

            ref_type_by_code = {}
            for ref_code, ref_type in zip(ref_codes, ref_types):
                ref_type_by_code.setdefault(ref_code, str(ref_type).strip().upper())
            known_ref_types = set(ref_type_by_code.values())

            row_ref_types = []
            for ref_value in reference_values.tolist():
                if isinstance(ref_value, str):
                    ref_name = ref_value.strip().upper()
                    if ref_name in known_ref_types:
                        row_ref_types.append(ref_name)
                        continue
                    try:
                        ref_value = int(ref_name)
                    except ValueError as error:
                        raise ValueError(
                            f'{ms}: FIELD.PHASE_DIR has unknown reference {ref_value!r}'
                        ) from error

                try:
                    ref_type = ref_type_by_code[ref_value]
                except KeyError as error:
                    raise ValueError(
                        f'{ms}: FIELD.PHASE_DIR has unknown reference code {ref_value!r}'
                    ) from error
                row_ref_types.append(ref_type)

            unique_ref_types = set(row_ref_types)
            if len(unique_ref_types) != 1:
                raise ValueError(
                    f'{ms}: FIELD.PHASE_DIR uses multiple reference frames '
                    f'({sorted(unique_ref_types)}); convert the directions to one '
                    'frame before fixing the metadata'
                )

            updates.append((ms, field_path, ref_column, row_ref_types[0]))

    for ms, field_path, ref_column, ref_type in updates:
        with table(field_path, ack=False, readonly=False) as field_table:
            measure_info = field_table.getcolkeywords('PHASE_DIR').get('MEASINFO')
            if not isinstance(measure_info, dict) or measure_info.get('VarRefCol') != ref_column:
                raise RuntimeError(
                    f'{ms}: FIELD.PHASE_DIR metadata changed during validation'
                )
            field_table.putcolkeyword(
                'PHASE_DIR', 'MEASINFO', {'type': 'direction', 'Ref': ref_type}
            )
        terminal_print(
            f'Fixed variable FIELD.PHASE_DIR reference in {ms} to {ref_type}'
        )


def is_two_pol_ms(ms):
    """
    Determines if a Measurement Set (MS) contains only two polarisation correlations

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    bool
        True if the MS contains only two polarisation correlations, False otherwise.

    Notes
    -----
    - The function reads the 'CORR_TYPE' column from the POLARIZATION table of the MS.
    - It checks if the correlation types correspond to two polarisation correlations.
    """
    with table(ms + '::POLARIZATION', ack=False, readonly=True) as t:
        corr_type = t.getcol('CORR_TYPE')
        if np.size(corr_type) == 2:
            return True
        else:
            return False


def set_metadata_compression(mslist, args):
    """
    Sets the metadata compression flag based on the telescope name in the provided Measurement Set list.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Sets. The function inspects the first set in the list.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is read and
        ``args['metadata_compression']`` may be set to False.

    Notes
    -----
    Side Effects:
    If the telescope name in the first Measurement Set is not 'LOFAR', sets 'args["metadata_compression"]' to False and prints a message.
    - Requires the 'table' class/function to be imported and available.
    """
 
    if args['telescope'] != 'LOFAR':
        terminal_print('Not using LOFAR data, setting metadata compression to False')
        args['metadata_compression'] = False


def fix_uvw(mslist):
    """
    The MeerKAT definition of UVW differs by a minus sign, but not always for some reason
    This leads to a mix of definitions inside a MS which causes problems when time averaging
    This function fixes that issue

    Parameters
    ----------
    mslist : str/list
        Input Measurement Set(s) as a string or a list of strings.
    """
    mslist = [mslist] if isinstance(mslist, str) else mslist
    if get_telescope_from_ms(mslist[0]) != 'MeerKAT':
        return    

    for ms in mslist:
        cmd = "taql 'update " + ms + " set UVW=(-1.)*(mscal.UVWJ2000())'"
        terminal_print('Command:', cmd)
        run(cmd, taql=True)
    return


def fix_equidistant_times(mslist, dryrun, args, dysco=True, metadata_compression=False):
    """
    Ensures that the measurement sets (MS) in the provided list have equidistant time axes.
    For non-LOFAR telescopes, checks if the time axis is regular; if not, regularizes and splits the MS as needed.
    For LOFAR telescopes, no changes are made.

    Parameters
    ----------
    mslist : list of str
        List of paths to measurement sets (MS).
    dryrun : bool
        If True, performs a dry run without making changes.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is used.
    dysco : bool, optional
        If True, enables DYSCO compression during splitting. Defaults to True.
    metadata_compression : bool, optional
        If True, enables metadata compression during splitting. Defaults to False.

    Returns
    -------
    tuple
        - sorted_mslist (list of str): Sorted list of processed MS paths.
        - all_splitting_performed (bool): True if splitting was performed for all MS, False otherwise.
    """

    mslist_return = []
    mslist_splitting_performed = []
    for ms in mslist:
        if args['telescope'] != 'LOFAR':
            if check_equidistant_times([ms], stop=False, return_result=True):
                terminal_print(ms + ' has a regular time axis')
                mslist_return.append(ms)
                mslist_splitting_performed.append(False)
            else:
                ms_path = regularize_ms(ms, overwrite=True, dryrun=dryrun)
                # Do the splitting
                mslist_return = mslist_return + split_ms(ms_path, overwrite=True,
                                                         prefix=os.path.basename(ms_path), return_mslist=True,
                                                         dryrun=dryrun, dysco=dysco, metadata_compression=metadata_compression)
                mslist_splitting_performed.append(True)
        else:
            mslist_return.append(ms)
            mslist_splitting_performed.append(False)
    return sorted(mslist_return), all(mslist_splitting_performed)


def check_equidistant_times(mslist, stop=True, return_result=False, tolerance=0.2):
    """
    Checks whether the time axis in the given Measurement Set (MS) files is equidistant.

    For each MS file in `mslist`, the function examines the 'TIME' column and determines if the time intervals
    between consecutive entries are consistent within a specified tolerance. If any time slots deviate beyond
    the tolerance, a warning is printed and, by default, the program exits unless `stop` is set to False.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set files to check.
    stop : bool, optional
        If True (default), the function will exit the program if non-equidistant time slots are found.
        If False, the function will continue execution after printing warnings.
    return_result : bool, optional
        If True, the function returns a boolean indicating whether all MS files have equidistant time axes.
        If False (default), the function does not return anything.
    tolerance : float, optional
        Relative tolerance (default 0.2) for detecting deviations from the median time interval.

    Returns
    -------
    bool or None
        If `return_result` is True, returns True if all MS files have equidistant time axes, False otherwise.
        If `return_result` is False, returns None.

    Notes
    -----
    - If non-equidistant time slots are detected, the function prints detailed information about the deviations.
    - It is recommended to average data with DP3 rather than CASA or CARACal to avoid time axis irregularities.
    - The function may exit the program if irregularities are found and `stop` is True.
    """

    for ms in mslist:
        with table(ms, ack=False) as t:
            times = np.unique(t.getcol('TIME'))
        if len(times) == 1:  # single timestep data
            return
        diff_times = np.abs(np.diff(times))[
                     :-1]  # take out the last one because this one can be special  due to rounding at the end....
        diff_times_medsub = np.abs(diff_times - np.median(diff_times))
        idx_deviating, = np.where(diff_times_medsub > tolerance * np.median(diff_times))  # 20% tolerance
        # print(len(idx_deviating))
        # print(idx_deviating)
        # sys.exit()
        if len(idx_deviating) > 0:
            terminal_print('Time interval differences (s):', diff_times)
            terminal_print('These time slots numbers are deviating', idx_deviating)
            terminal_print('Deviating time intervals (s):', diff_times[idx_deviating])
            terminal_print(
                'Measurement Set:',
                ms,
                'Time axis is not equidistant, this might cause DP3 errors and segmentation faults (check how your data was averaged',
            )
            # raise Exception(ms +': Time axis is not equidistant')
            terminal_print(
                'Avoid averaging your data with CASA or CARACal, instead average with DP3.')

            # comment line out below if you are willing to take the risk
            if stop:
                terminal_print('If you want to take the risk comment out the sys.exit() in the Python code')
                sys.exit()
    if return_result:
        if len(idx_deviating) > 0:
            return False  # so irregular time axis
        else:
            return True  # so normal time axis
    return


def fix_zero_weight_spectrum(mslist):
    """
    Some old MeerKAT Measurement Sets have a WEIGHT_SPECTRUM column which only contains 0.0 values which
    affect imaging and subsequent self-calibration, resulting in all data being flagged. This function sets these
    values to 1.0.

    Parameters
    ----------
    mslist : list
        a list of Measurement Sets to iterate over and fix outlier values of.

    Returns
    -------
    None
    """

    # check if mslist is a string (single ms) and convert to list if needed
    if isinstance(mslist, str):
        mslist = [mslist]

    for ms in mslist:
        fix_ws = False
        with table(ms, readonly=True, ack=False) as t:
            ws = t.getcol('WEIGHT_SPECTRUM')
            if np.nanmax(ws) == 0.0:
                fix_ws = True
        if fix_ws:
            # print in orange color
            terminal_print('\033[93m' + 'Warning: WEIGHT_SPECTRUM has only 0.0 values in ' + ms + '\033[0m')
            terminal_print('Fixing WEIGHT_SPECTRUM manually', ms)
            cmdtaql = 'taql "update %s set WEIGHT_SPECTRUM=1.0"' % ms
            #print('Running command:', cmdtaql)
            run(cmdtaql, log=True, taql=True)


def fix_bad_weightspectrum(mslist, clipvalue, use_taql=True):
    """
    Sets bad values in WEIGHT_SPECTRUM that affect imaging and subsequent self-calibration to 0.0.

    Parameters
    ----------
    mslist : list
        a list of Measurement Sets to iterate over and fix outlier values of.
    clipvalue : float
        value above which WEIGHT_SPECTRUM will be set to 0.
    use_taql : bool, optional
        If True, uses TAQL to update the WEIGHT_SPECTRUM column. If False, uses casacore tables. Defaults to True.

    Returns
    -------
    None
    """
    
    # check that clipvalue is larger than 0
    if clipvalue <= 0.0:
        raise ValueError('clipvalue must be larger than 0.0')

    if isinstance(mslist, str):
        mslist = [mslist]

    for ms in mslist:
        terminal_print('Clipping WEIGHT_SPECTRUM manually', ms, clipvalue)
        if  use_taql:
            cmdtaql = f'taql "UPDATE {ms} SET WEIGHT_SPECTRUM[WEIGHT_SPECTRUM > {clipvalue}] = 0.0"'
            terminal_print('Running command:', cmdtaql)
            run(cmdtaql, log=True, taql=True)
        else:
            with table(ms, readonly=False) as t:
                ws = t.getcol('WEIGHT_SPECTRUM')
                idx = np.where(ws > clipvalue)
                ws[idx] = 0.0
                t.putcol('WEIGHT_SPECTRUM', ws)
    return


def remove_flagged_data_startend(mslist):
    """
    Trim flagged data at the start and end of the observation.

    Parameters
    ----------
    mslist : list
        list of measurement sets to iterate over.

    Returns
    -------
    mslistout : list
        list of measurement sets with flagged data trimmed.
    """

    taql = 'taql'
    mslistout = []

    for ms in mslist:
        with table(ms, readonly=True, ack=False) as t:
            alltimes = t.getcol('TIME')
            alltimes = np.unique(alltimes)

            newt = taql('select TIME from $t where FLAG[0,0]=False')
            time = newt.getcol('TIME')
            time = np.unique(time)

            terminal_print('There are', len(alltimes), 'times')
            terminal_print('There are', len(time), 'unique unflagged times')

            terminal_print('First unflagged time', np.min(time))
            terminal_print('Last unflagged time', np.max(time))

            goodstartid = np.where(alltimes == np.min(time))[0][0]
            goodendid = np.where(alltimes == np.max(time))[0][0] + 1

            terminal_print('First unflagged time index:', goodstartid, 'Last unflagged time index:', goodendid)

        if (goodstartid != 0) or (goodendid != len(alltimes)):
            msout = ms + '.cut'
            if os.path.isdir(msout):
                shutil.rmtree(msout, ignore_errors=True)
                time.sleep(2)  # wait a bit to make sure the directory is removed

            cmd = taql + " ' select from " + ms + " where TIME in (select distinct TIME from " + ms
            cmd += " offset " + str(goodstartid)
            cmd += " limit " + str((goodendid - goodstartid)) + ") giving "
            cmd += msout + " as plain'"
            terminal_print('Command:', cmd)
            run(cmd, taql=True)
            mslistout.append(msout)
        else:
            mslistout.append(ms)
    fix_uvw(mslistout)
    return mslistout


def get_telescope_from_ms(mslist):
    """
    Determine the telescope name from the first Measurement Set (MS) in the provided list.
    This function reads the 'TELESCOPE_NAME' from the 'OBSERVATION' table of the first MS
    in the given list and returns it as a string.

    Parameters
    ----------
    mslist : list of str
        List of paths to Measurement Set directories.

    Returns
    -------
    str
        The name of the telescope associated with the first MS.

    Raises
    ------
    Exception
        If the 'TELESCOPE_NAME' cannot be found in the MS.
    """
    # if input is string, make it a list
    if isinstance(mslist, str):
        mslist = [mslist]

    with table(mslist[0] + '::OBSERVATION', ack=False) as t:
        telescope = t.getcol('TELESCOPE_NAME')[0]
    return telescope
