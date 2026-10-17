#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

"""
Reinout van Weeren,
based on work by Martijn Oei and Francesco de Gasperin

Adapt the MS format of uGMRT data to one usable by LOFAR software.
Memory-optimized version to handle large datasets without overloading memory.
"""

import os, sys, logging, time, re
import numpy as np
from casacore import tables # type: ignore
import gc


class MS(object):

    def __init__(self, ms_file, chunk_size=1000):
        logging.info("Starting work on MS at '" + ms_file + "'...")

        self.ms_file = ms_file
        self.chunk_size = chunk_size
        self.t       = tables.table(ms_file,                      readonly = False, ack = False)
        self.tpol    = tables.table(ms_file + "/POLARIZATION",    readonly = False, ack = False)
        self.tspect  = tables.table(ms_file + "/SPECTRAL_WINDOW", readonly = False, ack = False)


    def close(self):
        '''
        Close tables opened in '__init__'.
        '''
        logging.info("Closing tables...")

        self.t.close()
        self.tpol.close()
        self.tspect.close()


    def columnExists(self, columnName):
        '''
        Check whether a column with name 'columnName' exists.
        '''
        columnNames = self.t.colnames()

        return (columnName in columnNames)


    def twoPolms(self):
        '''
        Check whether the MS has two polarizations (XX and YY or RR and LL) only.
        '''
        corr_type = self.tpol.getcol("CORR_TYPE")

        if np.size(corr_type) == 2:
            logging.info("MS has two polarizations.")
            return True
        logging.info("MS has {} polarizations.".format(np.size(corr_type)))
        return False


    def removeColumns(self):
        '''
        Remove columns that are never used by the LOFAR software, and are thus a waste of disk space (e.g. "SIGMA_SPECTRUM"),
        or that are generated (in the right shape) later in the pipeline (e.g. "MODEL_DATA").
        Note: removal of columns can give errors when executing LOFAR command 'msoverview'.
        '''
        logging.info("- Removal of unnecessary data columns -")

        for columnName in ["SIGMA_SPECTRUM", "MODEL_DATA", "CORRECTED_DATA"]: # This list could possibly be expanded.
            if (self.columnExists(columnName)):
                print("Removing column '{}'...".format(columnName))
                self.t.removecols(columnName)


    def updatePolarisation(self, twopol):
        '''
        Make sure that the MS contains 4 polarisations.
        "CORR_TYPE"    column description comment: 'The polarization type for each correlation product, as a Stokes enum.'
        "CORR_PRODUCT" column description comment: 'Indices describing receptors of feed going into correlation'
        Add the FAKE_RLLR keyword to indicate that fake RL/LR polarizations will be added.
        '''
        logging.info("- Adaptation of polarisation metadata -")

        correlationTypesNew      = np.array([[5, 6, 7, 8]])
        correlationProductsNew   = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]]])
        numberOfCorrelationsNew  = 4

        self.tpol.putcol("CORR_TYPE",    correlationTypesNew)
        self.tpol.putcol("CORR_PRODUCT", correlationProductsNew)
        self.tpol.putcol("NUM_CORR",     numberOfCorrelationsNew)

        #Add the FAKE_RLLR keyword to indicate RL/LR polarizations.   
        logging.info("- Adding FAKE_RLLR keyword -")
        if twopol:
            self.t.putcolkeyword('DATA', 'FAKE_RLLR', True)
        else:
            self.t.putcolkeyword('DATA', 'FAKE_RLLR', False)


    def updateFreqMetadata(self):
        '''
        Flip the frequency channel order to match the LOFAR convention of ascending frequencies.
        We first check whether the flip is necessary at all, to make the code robust to running this program twice on the same MS.
        '''
        frequencies = self.tspect.getcol("CHAN_FREQ")

        if (frequencies[0, 0] > frequencies[0, -1]):
            logging.info("- Adaptation of frequency metadata -")
            self.tspect.putcol("CHAN_FREQ", np.fliplr(frequencies))
            return True
        else:
            logging.info("Frequency order already correct.")
            return False


    def updateFieldMetadata(self):
        '''
        MSs that were originally multi-field, have been split up in single-field MSs.
        Adapt the field metadata accordingly.
        '''
        logging.info("- Adaptation of field information -")

        pathMS      = self.ms_file
        pathMSField = self.ms_file + "/FIELD"

        # Remove metadata of other fields in the FIELD subtable.
        tables.taql("delete from $pathMSField where rownr() not in (select distinct FIELD_ID from $pathMS)")

        # Set 'SOURCE_ID' to 0 in the FIELD subtable.
        tables.taql("update $pathMSField set SOURCE_ID=0")

        # Set 'FIELD_ID' to 0 in the main table.
        tables.taql("update $pathMS set FIELD_ID=0")

    @staticmethod
    def _parse_bad_freq_ranges(bad_freq_ranges):
        '''
        Parse bad frequency range specifications into a list of (f_min, f_max) tuples in Hz.

        Supported inputs:
          - string in DP3/preflagger format:
            "99MHz..128MHz,167MHz..188MHz,243MHz..301MHz,347MHz..349.7MHz,355MHz..380MHz,390MHz..391MHz,750MHz..850MHz"
            or "[99MHz..128MHz, ...]"
          - list/tuple of strings: ["99MHz..128MHz", "167MHz..188MHz"]
          - list/tuple of frequency tuples/lists: [(99e6, 128e6), ...]
        '''
        if not bad_freq_ranges:
            return []

        def _to_hz(val):
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

    def getVarianceRRLL(self, data_column="DATA", num_baselines=6, candidate_pool_size=25,
                        top_quantile=0.85, max_flag_fraction=0.50, min_unflagged_samples=500,
                        bad_freq_ranges="99MHz..128MHz,167MHz..188MHz,243MHz..301MHz,347MHz..349.7MHz,355MHz..380MHz,390MHz..391MHz,750MHz..850MHz",
                        random_seed=42):
        '''
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
        '''
        logging.info("- Estimating robust visibility variance for RR and LL from long baselines -")
        
        rng = np.random.default_rng(random_seed)

        # 1. Identify RR and LL (or XX and YY) polarization indices
        corr_type = self.tpol.getcol("CORR_TYPE")[0]
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
        parsed_bad_ranges = self._parse_bad_freq_ranges(bad_freq_ranges)
        chan_freqs = self.tspect.getcol("CHAN_FREQ")[0]
        bad_freq_mask = np.zeros(len(chan_freqs), dtype=bool)

        if parsed_bad_ranges:
            for f_min, f_max in parsed_bad_ranges:
                bad_freq_mask |= (chan_freqs >= f_min) & (chan_freqs <= f_max)
            n_bad = np.sum(bad_freq_mask)
            if n_bad > 0:
                logging.info(f"Ignoring {n_bad}/{len(chan_freqs)} frequency channels matching bad frequency ranges.")

        # 3. Compute physical baseline lengths from ANTENNA subtable
        with tables.table(self.ms_file + "/ANTENNA", readonly=True, ack=False) as t_ant:
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
        has_flag_row = self.columnExists("FLAG_ROW")
        if not self.columnExists(data_column):
            raise KeyError(f"Column '{data_column}' not found in MS {self.ms_file}")

        baseline_results = []
        for a1, a2 in selected_pairs:
            taql_query = f"ANTENNA1 == {a1} && ANTENNA2 == {a2}"
            with self.t.query(taql_query) as sub_t:
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

    def updateCrosshand(self, variance):
        '''
        Fill the XY/RL and YX/LR cross-hand polarizations with complex Gaussian random noise
        based on the provided variance.

        Parameters
        ----------
        variance : float or tuple/list of (float, float)
            The noise variance in Jy^2. If a tuple (var_RR, var_LL) is provided,
            the geometric mean sqrt(var_RR * var_LL) is used for the cross-hands.
        '''
        logging.info("- Adaptation of cross-hand polarisation visibility data -")
        logging.info("Filling crosshand polarizations with random noise in MS %s", self.ms_file)
        if not self.columnExists("DATA"):
            logging.warning("DATA column not found, skipping.")
            return

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
        sample = self.t.getcol("DATA", startrow=0, nrow=1)
        if sample.shape[-1] != 4:
            logging.warning("Expected 4 pols in DATA, found %d. Skipping.", sample.shape[-1])
            return

        rng = np.random.default_rng()
        total_rows = self.t.nrows()
        for start_row in range(0, total_rows, self.chunk_size):
            nrow = min(self.chunk_size, total_rows - start_row)
            chunk = self.t.getcol("DATA", startrow=start_row, nrow=nrow)
            nchan = chunk.shape[1]
            
            # Generate independent complex Gaussian noise for RL (idx 1) and LR (idx 2)
            noise_rl = (rng.normal(0.0, sigma_1d, size=(nrow, nchan)) + 
                        1j * rng.normal(0.0, sigma_1d, size=(nrow, nchan))).astype(chunk.dtype)
            noise_lr = (rng.normal(0.0, sigma_1d, size=(nrow, nchan)) + 
                        1j * rng.normal(0.0, sigma_1d, size=(nrow, nchan))).astype(chunk.dtype)

            chunk[:, :, 1] = noise_rl
            chunk[:, :, 2] = noise_lr

            self.t.putcol("DATA", chunk, startrow=start_row, nrow=nrow)
            del chunk, noise_rl, noise_lr
            print(f"Processed rows {start_row} to {start_row + nrow} of {total_rows} for cross-hand noise injection.")
            gc.collect()


    def updateIntervals(self):
        '''
        Update the INTERVAL and TIME columns,
        so that differences between time stamps are always integer multiples of a fixed constant.
        '''
        logging.info("- Adaptation of time intervals -")

        # Calculate the new interval.
        pathMS          = self.ms_file
        times           = (tables.taql("select distinct TIME from $pathMS")).getcol("TIME")
        intervals       = times[1 : ] - times[ : -1]
        intervals       = intervals[intervals < 1.5 * np.min(intervals)] # Select only intervals that do not correspond with big jumps in time.
        intervalPrecise = np.mean(intervals)

        # Update INTERVAL column.
        intervalsOld    = self.t.getcol("INTERVAL")
        intervalsNew    = np.ones_like(intervalsOld) * intervalPrecise
        self.t.putcol("INTERVAL", intervalsNew)

        # Update TIME column.
        timesOld        = self.t.getcol("TIME")
        timesNew        = timesOld[0] + np.round((timesOld - timesOld[0]) / intervalPrecise, 0) * intervalPrecise
        self.t.putcol("TIME", timesNew)

        logging.info("Interval set to %f" % intervalPrecise)
        logging.debug("Time intervals (old; possibly unequal):")
        logging.debug(np.unique(timesOld)[1 : ] - np.unique(timesOld)[ : -1])
        logging.debug("Time intervals (new; should be equal):")
        logging.debug(np.unique(timesNew)[1 : ] - np.unique(timesNew)[ : -1])


    def _get_data_shape_and_info(self, column_name):
        '''
        Get shape information and metadata for a column without loading all data.
        '''
        # Get a small sample to determine shape
        sample_data = self.t.getcol(column_name, startrow=0, nrow=1)
        total_rows = self.t.nrows()
        
        keywordNames = self.t.colkeywordnames(column_name)
        columnDescription = self.t.getcoldesc(column_name)
        dataManagerInfo = self.t.getdminfo(column_name)
        
        return sample_data.shape, total_rows, keywordNames, columnDescription, dataManagerInfo


    def _process_data_chunk(self, data_chunk, updateFreq, expand_polarizations=True):
        '''
        Process a chunk of data (flip frequencies and expand polarizations if needed).
        '''
        if updateFreq:
            data_chunk = data_chunk[:, ::-1, :]
        
        if expand_polarizations and data_chunk.shape[2] == 2:
            # Expand from 2 to 4 polarizations
            if data_chunk.dtype == np.complex128 or data_chunk.dtype == np.complex64:
                # For visibility data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX -> XX
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY -> YY
                # XY and YX remain zero
            elif data_chunk.dtype == np.bool_ or data_chunk.dtype == bool:
                # For flag data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX flags
                new_chunk[:, :, 1] = data_chunk[:, :, 0]  # XY flags (copy from XX)
                new_chunk[:, :, 2] = data_chunk[:, :, 0]  # YX flags (copy from XX)
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY flags
            else:
                # For weight data
                new_chunk = np.zeros((data_chunk.shape[0], data_chunk.shape[1], 4), dtype=data_chunk.dtype)
                new_chunk[:, :, 0] = data_chunk[:, :, 0]  # XX weights
                new_chunk[:, :, 3] = data_chunk[:, :, 1]  # YY weights
                # set XY and YX weights (if zero DP3 will flag all visibilities when averaging)
                new_chunk[:, :, 1] = data_chunk[:, :, 0]  # XY weights
                new_chunk[:, :, 2] = data_chunk[:, :, 1]  # YX weights
            
            return new_chunk
        
        return data_chunk


    def _create_new_column(self, column_name, sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix):
        '''
        Create a new column with updated data manager info.
        '''
        # Update data manager info
        dataManagerInfo["NAME"] = f"Tiled{column_name.title()}{manager_suffix}"
        
        # Calculate appropriate tile shapes based on data dimensions
        if len(sample_shape) == 3:  # (time, freq, pol)
            n_pol = 4 if sample_shape[2] == 2 else sample_shape[2]
            n_freq = sample_shape[1]
            
            # Optimize tile shape for chunked processing
            tile_time = min(self.chunk_size, total_rows)
            tile_freq = min(40, n_freq)
            
            dataManagerInfo["SPEC"]["DEFAULTTILESHAPE"] = np.array([n_pol, tile_freq, tile_time], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["TileShape"] = np.array([n_pol, tile_freq, tile_time], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["CubeShape"] = np.array([n_pol, n_freq, total_rows], dtype=np.int32)
            dataManagerInfo["SPEC"]["HYPERCUBES"]["*1"]["CellShape"] = np.array([n_pol, n_freq], dtype=np.int32)
        
        # Remove old column if it exists
        if self.columnExists(column_name):
            self.t.removecols(column_name)
        
        # Add new column
        self.t.addcols(tables.makecoldesc(column_name, columnDescription), dataManagerInfo)


    def updateColumns(self, updateFreq):
        '''
        Update DATA, FLAG and WEIGHT_SPECTRUM columns using chunked processing to avoid memory overload.
        '''
        logging.info("- Change existing (or create alternative) columns for data, flags and weights -")
        
        # Process each column separately to minimize memory usage
        for column_name in ["DATA", "FLAG", "WEIGHT_SPECTRUM"]:
            if not self.columnExists(column_name):
                logging.warning(f"Column {column_name} does not exist, skipping...")
                continue
                
            logging.info(f"Processing column '{column_name}'...")
            
            # Get metadata without loading all data
            sample_shape, total_rows, keywordNames, columnDescription, dataManagerInfo = self._get_data_shape_and_info(column_name)
            
            # Determine manager suffix
            manager_suffix = {"DATA": "Martijn", "FLAG": "Martijn", "WEIGHT_SPECTRUM": "Martijn"}.get(column_name, "Martijn")
            
            # Create temporary column to store processed data
            temp_column_name = f"{column_name}_TEMP"
            
            # Update column description for new polarization structure if needed
            if sample_shape[2] == 2:
                new_sample_shape = (sample_shape[0], sample_shape[1], 4)
            else:
                new_sample_shape = sample_shape
            
            # Create new column
            self._create_new_column(temp_column_name, new_sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix)
            
            # Process data in chunks
            logging.info(f"Processing {total_rows} rows in chunks of {self.chunk_size}...")
            
            for start_row in range(0, total_rows, self.chunk_size):
                end_row = min(start_row + self.chunk_size, total_rows)
                current_chunk_size = end_row - start_row
                
                if start_row % (self.chunk_size * 10) == 0:  # Progress update every 10 chunks
                    logging.info(f"Processing rows {start_row} to {end_row-1} ({100*end_row/total_rows:.1f}%)")
                
                # Load chunk
                data_chunk = self.t.getcol(column_name, startrow=start_row, nrow=current_chunk_size)
                
                # Process chunk
                processed_chunk = self._process_data_chunk(data_chunk, updateFreq, expand_polarizations=True)
                
                # Write processed chunk to temporary column
                self.t.putcol(temp_column_name, processed_chunk, startrow=start_row, nrow=current_chunk_size)
                
                # Force garbage collection to free memory
                del data_chunk, processed_chunk
                gc.collect()
            
            # Remove original column and rename temporary column
            logging.info(f"Finalizing column '{column_name}'...")
            self.t.removecols(column_name)
            
            # Rename temporary column to original name
            # Note: Direct renaming is not supported in casacore, so we need to recreate
            self._create_new_column(column_name, new_sample_shape, total_rows, columnDescription, dataManagerInfo, manager_suffix)
            
            # Copy data from temp column to final column in chunks
            for start_row in range(0, total_rows, self.chunk_size):
                end_row = min(start_row + self.chunk_size, total_rows)
                current_chunk_size = end_row - start_row
                
                temp_data = self.t.getcol(temp_column_name, startrow=start_row, nrow=current_chunk_size)
                self.t.putcol(column_name, temp_data, startrow=start_row, nrow=current_chunk_size)
                
                del temp_data
                gc.collect()
            
            # Remove temporary column
            self.t.removecols(temp_column_name)
            
            logging.info(f"Column '{column_name}' processing complete.")
            
            # Force garbage collection between columns
            gc.collect()


def readArguments():
    import argparse
    parser = argparse.ArgumentParser("Adapt uGMRT MS to LOFAR MS format.")
    parser.add_argument("-v", "--verbose", help="Be verbose. Default is False", required=False, action="store_true")
    parser.add_argument("-c", "--chunk-size", help="Number of rows to process at once (default: 1000)", 
                       type=int, default=1000, required=False)
    parser.add_argument("ms_files", help="MeasurementSet name(s).", type=str, nargs="+")
    args = parser.parse_args()
    return vars(args)


if (__name__ == "__main__"):
    start_time = time.time()

    args = readArguments()
    verbose = args["verbose"]
    chunk_size = args["chunk_size"]
    ms_files = args["ms_files"]

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f'Reading MSs with chunk size: {chunk_size}...')
    MSs = []
    for ms_file in ms_files:
        MSs.append(MS(ms_file, chunk_size=chunk_size))

    for MS in MSs:
        twopol = MS.twoPolms()
        MS.removeColumns()
        MS.updatePolarisation(twopol)
        updateFreq = MS.updateFreqMetadata()
        MS.updateFieldMetadata()
        MS.updateIntervals()
        MS.updateColumns(updateFreq)
        if twopol:
            variance = MS.getVarianceRRLL()
            MS.updateCrosshand(variance)
        MS.close()

    logging.debug('Running time %.0f s' % (time.time() - start_time))
    logging.info('Done.')
