#!/usr/bin/env python3
"""
shadow_flag.py

Detect shadowed antennas in a Measurement Set by recomputing UVW coordinates
from antenna positions, phase centre and timestamp with casacore, then flag the
affected baselines and time ranges using DP3's PreFlagger step.

This replicates the behaviour of CASA's flagdata(mode='shadow') but uses the
Python casacore library for detection and DP3 for the actual flagging.

Usage (inside the flocs7.0.5.sif container):
    python3 shadow_flag.py --ms <MS> [--tolerance <meters>] [--output-ms <out_MS>]
                           [--container <sif>] [--ncpu <N>] [--dry-run]

Shadowing algorithm
-------------------
For each row in the MAIN table (one row = one baseline at one time):
  - projected_separation = sqrt(U^2 + V^2)   (metres, in the UV plane)
  - If projected_separation < (diameter_ant1 + diameter_ant2) / 2 - tolerance:
      * W > 0  →  antenna2 is further from the source  →  antenna2 is shadowed
      * W < 0  →  antenna1 is further from the source  →  antenna1 is shadowed
      * W == 0 →  both antennas are in the same plane; flag both
Consecutive time slots where the same antenna is shadowed are merged into a
single time range before being handed to DP3.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

try:
    import casacore.measures as cm
    import casacore.tables as ct
    from casacore.quanta import quantity
except ImportError:
    sys.exit("ERROR: casacore Python library not found. "
             "Run this script inside the provided Singularity container.")


# ---------------------------------------------------------------------------
# Helper: MJD seconds → DP3 / casacore MVTime string  (DD-Mon-YYYY/HH:MM:SS.sss)
# ---------------------------------------------------------------------------

_MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _mjd_sec_to_dp3(mjd_sec: float) -> str:
    """Return a casacore MVTime string suitable for DP3 abstime."""
    unix_ts = (mjd_sec / 86400.0 - 40587.0) * 86400.0
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    sec_frac = dt.second + dt.microsecond / 1e6
    return (f"{dt.day:02d}-{_MONTH_ABBR[dt.month - 1]}-{dt.year:04d}"
            f"/{dt.hour:02d}:{dt.minute:02d}:{sec_frac:06.3f}")


def _ranges_total_hours(ranges) -> float:
    """Return total duration of [t0, t1] ranges in hours."""
    return sum(max(0.0, t1 - t0) for t0, t1 in ranges) / 3600.0


def _get_measure_ref(table_obj, column_name: str, default: str) -> str:
    """Return the measure reference code stored in the column keywords."""
    try:
        measinfo = table_obj.getcolkeyword(column_name, 'MEASINFO')
    except Exception:
        return default
    return str(measinfo.get('Ref', default))


def _phase_dir_cell_to_radec(phase_dir_cell) -> tuple[float, float]:
    """Extract (ra, dec) in radians from a FIELD/PHASE_DIR cell."""
    values = np.asarray(phase_dir_cell, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError(
            f"Unexpected PHASE_DIR cell shape: {np.shape(phase_dir_cell)}"
        )
    return float(values[0]), float(values[1])


def _compute_antenna_uvw(ant_positions: np.ndarray, time_mjd_sec: float,
                         time_ref: str, phase_dir_radec: tuple[float, float],
                         phase_dir_ref: str) -> np.ndarray:
    """Compute per-antenna UVW coordinates in metres for one time/field."""
    measures = cm.measures()
    ref_pos = np.asarray(ant_positions[0], dtype=float)

    measures.do_frame(
        measures.position(
            'itrf',
            quantity(float(ref_pos[0]), 'm'),
            quantity(float(ref_pos[1]), 'm'),
            quantity(float(ref_pos[2]), 'm'),
        )
    )
    measures.do_frame(measures.epoch(time_ref, quantity(float(time_mjd_sec), 's')))
    measures.do_frame(
        measures.direction(
            phase_dir_ref,
            quantity(float(phase_dir_radec[0]), 'rad'),
            quantity(float(phase_dir_radec[1]), 'rad'),
        )
    )

    ant_uvw = np.empty((len(ant_positions), 3), dtype=float)
    for ant_idx, ant_pos in enumerate(np.asarray(ant_positions, dtype=float)):
        delta = ant_pos - ref_pos
        baseline = measures.baseline(
            'itrf',
            quantity(float(delta[0]), 'm'),
            quantity(float(delta[1]), 'm'),
            quantity(float(delta[2]), 'm'),
        )
        uvw = measures.to_uvw(baseline)
        ant_uvw[ant_idx, :] = np.asarray(uvw['xyz'].get_value('m'), dtype=float)

    return ant_uvw


# ---------------------------------------------------------------------------
# Shadow detection
# ---------------------------------------------------------------------------

def detect_shadowed_antennas(ms_path: str, tolerance: float = 0.0,
                              verbose: bool = False):
    """
    Read the MS and return a dict mapping antenna_name → sorted list of
    (t_start_mjd_sec, t_end_mjd_sec) intervals during which that antenna
    is shadowed.

    Parameters
    ----------
    ms_path   : path to the Measurement Set
    tolerance : allowed shadow overlap in metres (default 0.0)
    verbose   : print progress information

    Returns
    -------
    dict  antenna_name (str) → list of [t_start, t_end] pairs (MJD seconds)
    """

    # -- ANTENNA subtable --------------------------------------------------
    ant_tab = ct.table(os.path.join(ms_path, 'ANTENNA'),
                       readonly=True, ack=False)
    ant_names = list(ant_tab.getcol('NAME'))
    ant_diameters = ant_tab.getcol('DISH_DIAMETER')  # metres
    ant_positions = ant_tab.getcol('POSITION')       # ITRF positions in metres
    ant_tab.close()

    if verbose:
        print(f"[shadow] {len(ant_names)} antennas read from ANTENNA table")
        for i, (n, d) in enumerate(zip(ant_names, ant_diameters)):
            print(f"  [{i:3d}] {n:20s}  diameter={d:.1f} m")

    # -- FIELD subtable: read phase centres --------------------------------
    field_tab = ct.table(os.path.join(ms_path, 'FIELD'),
                         readonly=True, ack=False)
    phase_dir_ref = _get_measure_ref(field_tab, 'PHASE_DIR', 'J2000')
    phase_dir_by_field = {
        rownr: _phase_dir_cell_to_radec(field_tab.getcell('PHASE_DIR', rownr))
        for rownr in range(field_tab.nrows())
    }
    field_tab.close()

    # -- MAIN table: read time and baseline metadata -----------------------
    main_tab = ct.table(ms_path, readonly=True, ack=False)

    if verbose:
        print(f"[shadow] Reading {main_tab.nrows():,} rows from MAIN table …")

    time_col = 'TIME_CENTROID' if 'TIME_CENTROID' in main_tab.colnames() else 'TIME'
    time_ref = _get_measure_ref(main_tab, time_col, 'UTC')
    times = main_tab.getcol(time_col)        # MJD seconds, shape (nrow,)
    ant1 = main_tab.getcol('ANTENNA1')       # shape (nrow,)
    ant2 = main_tab.getcol('ANTENNA2')       # shape (nrow,)
    field_ids = main_tab.getcol('FIELD_ID')  # shape (nrow,)
    interval = float(main_tab.getcol('INTERVAL')[0])  # integration time (seconds)
    main_tab.close()
    time_ant_set: set = set()
    n_shadow = 0

    sort_idx = np.lexsort((times, field_ids))
    sorted_times = times[sort_idx]
    sorted_fields = field_ids[sort_idx]

    start = 0
    while start < len(sort_idx):
        field_id = int(sorted_fields[start])
        time_val = float(sorted_times[start])
        end = start + 1
        while end < len(sort_idx):
            if sorted_fields[end] != field_id or sorted_times[end] != time_val:
                break
            end += 1

        row_idx = sort_idx[start:end]
        row_ant1 = ant1[row_idx]
        row_ant2 = ant2[row_idx]
        cross_mask = row_ant1 != row_ant2

        if np.any(cross_mask):
            ant_uvw = _compute_antenna_uvw(
                ant_positions=ant_positions,
                time_mjd_sec=time_val,
                time_ref=time_ref,
                phase_dir_radec=phase_dir_by_field[field_id],
                phase_dir_ref=phase_dir_ref,
            )

            baseline_uvw = ant_uvw[row_ant2] - ant_uvw[row_ant1]
            proj_dist = np.sqrt(baseline_uvw[:, 0] ** 2 + baseline_uvw[:, 1] ** 2)
            w = baseline_uvw[:, 2]

            r1 = ant_diameters[row_ant1] / 2.0
            r2 = ant_diameters[row_ant2] / 2.0
            threshold = r1 + r2 - tolerance
            shadowed_mask = cross_mask & (proj_dist < threshold)

            if np.any(shadowed_mask):
                shadow_ant1 = row_ant1[shadowed_mask]
                shadow_ant2 = row_ant2[shadowed_mask]
                shadow_w = w[shadowed_mask]
                n_shadow += int(shadowed_mask.sum())

                for ant1_idx, ant2_idx, w_val in zip(shadow_ant1,
                                                     shadow_ant2,
                                                     shadow_w):
                    if w_val < 0:
                        time_ant_set.add((time_val, int(ant1_idx)))
                    elif w_val > 0:
                        time_ant_set.add((time_val, int(ant2_idx)))
                    else:
                        time_ant_set.add((time_val, int(ant1_idx)))
                        time_ant_set.add((time_val, int(ant2_idx)))

        start = end

    if verbose:
        print(f"[shadow] {n_shadow:,} rows with potential shadowing "
              f"(before per-antenna grouping)")

    if not time_ant_set:
        if verbose:
            print("[shadow] No shadowing detected.")
        return {}

    # -- Group by antenna, collect sorted unique times --------------------
    ant_times: dict = defaultdict(list)
    for t, ant_idx in time_ant_set:
        ant_times[ant_idx].append(t)

    for ant_idx in ant_times:
        ant_times[ant_idx].sort()

    # -- Determine integration time (half-width to build time ranges) -----
    # Use the median difference between consecutive unique time stamps;
    # fall back to the INTERVAL column value read from the MAIN table.
    unique_times = np.unique(times)
    if len(unique_times) > 1:
        diffs = np.diff(unique_times)
        int_time = float(np.median(diffs))
    else:
        int_time = interval  # fallback: integration time from INTERVAL column

    half_dt = int_time / 2.0

    # -- Merge consecutive time steps into contiguous ranges --------------
    # Two time steps are "consecutive" if their difference ≤ 1.5 × int_time
    merge_gap = 1.5 * int_time

    result: dict = {}
    for ant_idx, sorted_times in ant_times.items():
        name = ant_names[ant_idx]
        ranges = []
        seg_start = sorted_times[0]
        seg_end = sorted_times[0]
        for t in sorted_times[1:]:
            if t - seg_end <= merge_gap:
                seg_end = t
            else:
                ranges.append((seg_start - half_dt, seg_end + half_dt))
                seg_start = t
                seg_end = t
        ranges.append((seg_start - half_dt, seg_end + half_dt))
        result[name] = ranges

    if verbose:
        total_ranges = sum(len(v) for v in result.values())
        print(f"[shadow] Shadowed antennas: {list(result.keys())}")
        print(f"[shadow] Total time ranges to flag: {total_ranges}")

    return result


# ---------------------------------------------------------------------------
# DP3 parset generation and execution
# ---------------------------------------------------------------------------

def build_dp3_parset(ms_path: str, output_ms: str,
                     shadowed: dict, ncpu: int = 0) -> str:
    """
    Build a DP3 parset string that flags all shadowed baselines/time-ranges.

    Each shadowed antenna gets its own PreFlagger step that uses named keyword
    sets combined with `expr = ant and time` so that only baselines involving
    that antenna *during the shadowed intervals* are flagged.

    Parameters
    ----------
    ms_path   : input MS path
    output_ms : output MS path (can be the same as ms_path for in-place update)
    shadowed  : dict  antenna_name → list of (t_start, t_end) MJD-second pairs
    ncpu      : number of DP3 threads (0 = DP3 default)

    Returns
    -------
    parset string
    """
    step_names = []
    step_defs = []

    for i, (ant_name, ranges) in enumerate(sorted(shadowed.items())):
        # Guard against names that would corrupt the parset syntax
        if any(c in ant_name for c in (']', '[', '\n', '\r')):
            raise ValueError(f"Antenna name contains unsafe characters: {ant_name!r}")

        step_id = f"shadow{i:04d}"
        step_names.append(step_id)

        # Build the abstime list: "t1_start..t1_end, t2_start..t2_end, ..."
        time_strings = ", ".join(
            f"{_mjd_sec_to_dp3(t0)}..{_mjd_sec_to_dp3(t1)}"
            for t0, t1 in ranges
        )

        # Escape square brackets inside the baseline spec:
        # DP3 baseline format [[ant_name]] means "any baseline with ant_name"
        step_defs.append(
            f"# Antenna: {ant_name}  ({len(ranges)} time range(s))\n"
            f"{step_id}.type        = preflagger\n"
            f"{step_id}.ant.baseline = [[{ant_name}]]\n"
            f"{step_id}.time.abstime = [{time_strings}]\n"
            f"{step_id}.expr        = ant and time\n"
        )

    steps_list = "[" + ", ".join(step_names) + "]"

    # Use "msout = ." for in-place flag updates (more efficient: only FLAG column
    # is updated; no data copy).  For a new output MS, write the full path.
    if output_ms == ms_path:
        msout_line = "msout                   = ."
    else:
        msout_line = f"msout                   = {output_ms}"

    parset_lines = [
        f"msin                    = {ms_path}",
        f"msin.datacolumn         = DATA",
        msout_line,
        f"steps                   = {steps_list}",
        "",
    ]
    if ncpu > 0:
        parset_lines.insert(3, f"numthreads              = {ncpu}")

    parset_lines += step_defs

    return "\n".join(parset_lines) + "\n"


def run_dp3(parset_str: str, container: str = "", verbose: bool = False,
            dry_run: bool = False) -> int:
    """
    Write the parset to a temp file and execute DP3 (optionally via Singularity).

    Returns the DP3 exit code (0 = success).
    """
    # Write the parset to a temp file in the shared working directory if possible,
    # otherwise fall back to /tmp (which may be machine-local).
    _shared_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
    parset_dir = _shared_tmp if os.path.isdir(_shared_tmp) else '/tmp'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.parset',
                                     prefix='shadow_flag_', delete=False,
                                     dir=parset_dir) as fh:
        fh.write(parset_str)
        parset_path = fh.name

    if verbose or dry_run:
        print(f"\n[DP3] Parset written to: {parset_path}")
        print("--- parset contents ---")
        print(parset_str)
        print("-----------------------")

    if dry_run:
        print("[DP3] Dry-run mode: DP3 will NOT be executed.")
        try:
            os.unlink(parset_path)
        except OSError:
            pass
        return 0

    if container:
        cmd = ["singularity", "exec", container, "DP3", parset_path]
    else:
        cmd = ["DP3", parset_path]

    if verbose:
        print(f"[DP3] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    try:
        os.unlink(parset_path)
    except OSError:
        pass

    return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Flag shadowed antennas using casacore UVW detection + DP3 PreFlagger.\n"
            "Equivalent to CASA flagdata(mode='shadow') but without CASA."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--ms', required=True,
                   help='Input Measurement Set path')
    p.add_argument('--tolerance', type=float, default=0.0,
                   help='Allowed shadow overlap in metres (default: 0.0)')
    p.add_argument('--output-ms', default='',
                   help='Output MS path. Defaults to in-place update of --ms.')
    p.add_argument('--container', default='',
                   help='Singularity .sif image to invoke DP3 through. '
                        'If empty, DP3 must be on $PATH.')
    p.add_argument('--ncpu', type=int, default=0,
                   help='Number of DP3 threads (0 = DP3 default).')
    p.add_argument('--dry-run', action='store_true',
                   help='Detect shadowing and print the DP3 parset, '
                        'but do not execute DP3.')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Verbose output.')
    return p.parse_args()


def main():
    args = parse_args()

    ms_path = os.path.abspath(args.ms)
    if not os.path.isdir(ms_path):
        sys.exit(f"ERROR: MS not found: {ms_path}")

    output_ms = os.path.abspath(args.output_ms) if args.output_ms else ms_path

    # -- Detect shadowing --------------------------------------------------
    print(f"[shadow] Detecting shadowing in: {ms_path}")
    print(f"[shadow] Tolerance: {args.tolerance} m")

    shadowed = detect_shadowed_antennas(ms_path, tolerance=args.tolerance,
                                        verbose=args.verbose)

    if not shadowed:
        print("[shadow] No shadowing found. Nothing to flag.")
        return

    print(f"\n[shadow] Shadowed antenna summary:")
    for ant_name, ranges in sorted(shadowed.items()):
        total_hours = _ranges_total_hours(ranges)
        print(f"  {ant_name}: {len(ranges)} time range(s), "
              f"total shadowed={total_hours:.3f} hr")
        if args.verbose:
            for t0, t1 in ranges:
                print(f"      {_mjd_sec_to_dp3(t0)}  →  {_mjd_sec_to_dp3(t1)}")

    # -- Build DP3 parset --------------------------------------------------
    parset = build_dp3_parset(ms_path, output_ms, shadowed, ncpu=args.ncpu)

    # -- Run DP3 -----------------------------------------------------------
    rc = run_dp3(parset, container=args.container,
                 verbose=args.verbose, dry_run=args.dry_run)

    if rc != 0:
        sys.exit(f"ERROR: DP3 exited with code {rc}")

    if not args.dry_run:
        print(f"\n[shadow] Flagging complete. Output: {output_ms}")


if __name__ == '__main__':
    main()
