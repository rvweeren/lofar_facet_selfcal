#!/usr/bin/env python3
"""Merge bandpass HDF5 solutions by antenna.

The script reads one or more input HDF5 files, combines the phase000 and
amplitude000 solutions along the antenna axis, and writes a merged HDF5 file.
It supports either an average or a weighted median combination. The per-file
weight is taken from the number of entries along the time axis.

Flags are taken from the dataset named weight and values with flag 0 are ignored.
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import tables


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def expand_inputs(patterns: List[str]) -> List[str]:
    expanded = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(pattern)
    return expanded


def _collapse_to_ant_freq(values: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 5:
        if arr.shape[1] != freq_len or arr.shape[2] != ant_len:
            raise ValueError(f"Unexpected value shape {arr.shape} for freq/antenna axes")
        arr = arr[0, :, :, 0, :]  # take the first time slice and first polarization slice
        arr = np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 4:
        if arr.shape[1] != freq_len or arr.shape[2] != ant_len:
            raise ValueError(f"Unexpected value shape {arr.shape} for freq/antenna axes")
        arr = arr[0, :, :, :]  # take the first time slice
        arr = np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 3:
        if arr.shape[0] != freq_len or arr.shape[1] != ant_len:
            raise ValueError(f"Unexpected value shape {arr.shape} for freq/antenna axes")
        arr = np.expand_dims(arr, axis=-1)
    elif arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    else:
        raise ValueError(f"Unsupported data shape {arr.shape}")

    if arr.shape[-1] != 2:
        raise ValueError(f"Expected a trailing real/imaginary axis of length 2, got {arr.shape}")

    return arr[..., 0] + 1j * arr[..., 1]


def _collapse_flags(flags: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    arr = np.asarray(flags)
    if arr.ndim == 5:
        if arr.shape[1] != freq_len or arr.shape[2] != ant_len:
            raise ValueError(f"Unexpected flag shape {arr.shape} for freq/antenna axes")
        arr = arr[0, :, :, 0, 0]
        arr = np.transpose(arr, (1, 0))
    elif arr.ndim == 4:
        if arr.shape[1] != freq_len or arr.shape[2] != ant_len:
            raise ValueError(f"Unexpected flag shape {arr.shape} for freq/antenna axes")
        arr = arr[0, :, :, 0]
        arr = np.transpose(arr, (1, 0))
    elif arr.ndim == 3:
        if arr.shape[0] != freq_len or arr.shape[1] != ant_len:
            raise ValueError(f"Unexpected flag shape {arr.shape} for freq/antenna axes")
        arr = np.transpose(arr, (1, 0))
    elif arr.ndim == 2:
        arr = arr
    else:
        raise ValueError(f"Unsupported flag shape {arr.shape}")

    return arr.astype(float)


def _collapse_phase(values: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 5 or arr.shape[1] != freq_len or arr.shape[2] != ant_len:
        raise ValueError(f"Unexpected phase value shape {arr.shape}")
    return np.transpose(arr[0, :, :, 0, :], (1, 0, 2))


def _collapse_phase_flags(flags: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    arr = np.asarray(flags)
    if arr.ndim != 5 or arr.shape[1] != freq_len or arr.shape[2] != ant_len:
        raise ValueError(f"Unexpected phase flag shape {arr.shape}")
    return np.transpose(arr[0, :, :, 0, :], (1, 0, 2)).astype(float)


def _collapse_scalar_solution(values: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 5 or arr.shape[1] != freq_len or arr.shape[2] != ant_len:
        raise ValueError(f"Unexpected scalar solution shape {arr.shape}")
    return np.transpose(arr[0, :, :, 0, :], (1, 0, 2))


def _collapse_scalar_flags(flags: np.ndarray, freq_len: int, ant_len: int) -> np.ndarray:
    return _collapse_scalar_solution(flags, freq_len, ant_len).astype(float)


def load_solution(path: str, solution_name: str) -> dict:
    with tables.open_file(path, mode="r") as handle:
        group = handle.root.sol000[solution_name]
        data = np.asarray(group.val[:])
        freq = np.asarray(group.freq[:])
        time = np.asarray(group.time[:])
        antennas = [decode_value(item) for item in group.ant[:]]
        flags = np.asarray(group.weight[:])
        pol_len = len(group.pol[:])

    freq_len = len(freq)
    ant_len = len(antennas)
    if solution_name == "phase000":
        values = _collapse_phase(data, freq_len, ant_len)
        flag_values = _collapse_phase_flags(flags, freq_len, ant_len)
    else:
        values = _collapse_scalar_solution(data, freq_len, ant_len)
        flag_values = _collapse_scalar_flags(flags, freq_len, ant_len)

    expected_shape = (ant_len, freq_len, pol_len)
    if values.shape != expected_shape:
        raise ValueError(
            f"Resolved value shape {values.shape} does not match expected layout {expected_shape}"
        )
    if flag_values.shape != expected_shape:
        raise ValueError(
            f"Resolved flag shape {flag_values.shape} does not match expected layout {expected_shape}"
        )

    return {
        "path": path,
        "data": values,
        "freq": freq,
        "time": time,
        "antennas": antennas,
        "flags": flag_values,
        "weight": float(len(time)),
    }


def build_antenna_order(solutions: List[dict]) -> List[str]:
    positions = {}
    first_seen = {}

    for file_index, solution in enumerate(solutions):
        for antenna_index, antenna in enumerate(solution["antennas"]):
            if antenna not in first_seen:
                first_seen[antenna] = file_index
            positions.setdefault(antenna, []).append(antenna_index)

    return [
        antenna
        for antenna, _ in sorted(
            positions.items(),
            key=lambda item: (
                float(np.median(item[1])),
                first_seen.get(item[0], 0),
                item[0],
            ),
        )
    ]


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _phase_to_complex(angle: float) -> complex:
    return np.exp(1j * _wrap_angle(angle))


def _pick_reference_antenna(solutions: List[dict], reference_antenna: Optional[str]) -> Optional[str]:
    common_antennas = set(solutions[0]["antennas"])
    for solution in solutions[1:]:
        common_antennas &= set(solution["antennas"])

    if not common_antennas:
        return None

    if reference_antenna is not None and reference_antenna in common_antennas:
        return reference_antenna

    for antenna in solutions[0]["antennas"]:
        if antenna in common_antennas:
            return antenna
    return None


def _relative_amplitude_value(value: complex, reference_value: complex) -> float:
    return np.log(np.maximum(np.abs(value), 1e-12)) - np.log(np.maximum(np.abs(reference_value), 1e-12))


def _phase_reference(value: complex, reference_value: complex) -> float:
    """Return the signed phase difference relative to the reference antenna.

    If the reference phasor is effectively zero, treat its phase angle as 0 so we
    do not collapse the relative phase to 0 through zero-product arithmetic.
    """
    if np.isclose(np.abs(reference_value), 0.0):
        return _wrap_angle(np.angle(value))
    return _wrap_angle(np.angle(value) - np.angle(reference_value))


def _relative_phase_value(value: complex, reference_value: complex) -> float:
    return _phase_reference(value, reference_value)


def _weighted_average(values: List[float], weights: List[float]) -> float:
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        raise ValueError("Cannot average with non-positive total weight")
    return float(np.sum(values_arr * weights_arr) / total_weight)


def _weighted_median(values: List[float], weights: List[float]) -> float:
    values_arr = np.asarray(values, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values_arr)
    sorted_values = values_arr[order]
    sorted_weights = weights_arr[order]
    cumulative = np.cumsum(sorted_weights)
    midpoint = cumulative[-1] / 2.0
    index = np.searchsorted(cumulative, midpoint, side="left")
    return float(sorted_values[index])


def _weighted_circular_mean(angles: List[float], weights: List[float]) -> float:
    angles_arr = np.asarray(angles, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    vector_sum = np.sum(weights_arr * np.exp(1j * angles_arr))
    if np.isclose(vector_sum, 0.0):
        return 0.0
    return float(np.angle(vector_sum))


def merge_solution(
    solutions: List[dict],
    method: str,
    reference_antenna: Optional[str] = None,
    is_amplitude: bool = False,
    flag_policy: str = "any",
    copy_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if flag_policy not in {"any", "all"}:
        raise ValueError("Flag policy must be 'any' or 'all'")

    if is_amplitude:
        antenna_order = build_antenna_order(solutions)
        n_ant = len(antenna_order)
        freq_axis = solutions[0]["freq"]
        n_freq = len(freq_axis)
        n_pol = solutions[0]["data"].shape[2]
        merged_values = np.zeros((n_ant, n_freq, n_pol), dtype=np.float64)
        merged_flags = np.zeros((n_ant, n_freq, n_pol), dtype=np.int8)
        first_solution = solutions[0]

        antenna_presence = {
            antenna: sum(1 for solution in solutions if antenna in solution["antennas"]) for antenna in antenna_order
        }

        for antenna_idx, antenna in enumerate(antenna_order):
            for freq_idx in range(n_freq):
                for pol_idx in range(n_pol):
                    if copy_first and antenna in first_solution["antennas"]:
                        first_idx = first_solution["antennas"].index(antenna)
                        first_flag = float(first_solution["flags"][first_idx, freq_idx, pol_idx])
                        first_value = first_solution["data"][first_idx, freq_idx, pol_idx]
                        if first_flag > 0.0 and np.isfinite(first_value) and first_value > 0.0:
                            merged_values[antenna_idx, freq_idx, pol_idx] = first_value
                            merged_flags[antenna_idx, freq_idx, pol_idx] = 1
                        continue

                    values = []
                    weights = []
                    all_present_valid = True
                    for solution in solutions:
                        if antenna not in solution["antennas"]:
                            continue
                        local_idx = solution["antennas"].index(antenna)
                        if float(solution["flags"][local_idx, freq_idx, pol_idx]) == 0.0:
                            all_present_valid = False
                            continue
                        value = solution["data"][local_idx, freq_idx, pol_idx]
                        if not np.isfinite(value) or value <= 0.0:
                            continue
                        values.append(float(np.log(value)))
                        weights.append(solution["weight"])

                    if not values or (flag_policy == "all" and not all_present_valid):
                        continue
                    if method == "average":
                        merged_log = _weighted_average(values, weights)
                    elif method == "median":
                        merged_log = _weighted_median(values, weights)
                    else:
                        raise ValueError("Method must be 'average' or 'median' for amplitude")
                    merged_values[antenna_idx, freq_idx, pol_idx] = np.exp(merged_log)
                    merged_flags[antenna_idx, freq_idx, pol_idx] = 1

        return merged_values, merged_flags, solutions[0]["freq"], np.array(antenna_order, dtype="S")

    reference = _pick_reference_antenna(solutions, reference_antenna)
    if reference is None:
        raise ValueError("No common reference antenna found across input files")

    antenna_order = build_antenna_order(solutions)
    n_ant = len(antenna_order)
    freq_axis = solutions[0]["freq"]
    n_freq = len(freq_axis)
    n_pol = solutions[0]["data"].shape[2]

    referenced_solutions = []
    for solution in solutions:
        if reference not in solution["antennas"]:
            referenced_solutions.append(solution)
            continue

        ref_idx = solution["antennas"].index(reference)
        referenced = dict(solution)
        referenced_data = np.full_like(solution["data"], np.nan, dtype=np.float64)
        ref_value = solution["data"][ref_idx, :, :]
        for ant_idx, _ in enumerate(solution["antennas"]):
            if ant_idx == ref_idx:
                referenced_data[ant_idx, :, :] = 0.0
                continue
            value = solution["data"][ant_idx, :, :]
            valid = (
                (solution["flags"][ant_idx, :, :] > 0.0)
                & (solution["flags"][ref_idx, :, :] > 0.0)
                & np.isfinite(value)
                & np.isfinite(ref_value)
            )
            # Match fix_phasereference: subtract the reference phase by
            # broadcasting over every retained frequency and polarization.
            relative = value - ref_value
            referenced_data[ant_idx, :, :] = np.where(valid, relative, np.nan)
        referenced["data"] = referenced_data
        referenced_solutions.append(referenced)

    solutions = referenced_solutions
    merged_values = np.zeros((n_ant, n_freq, n_pol), dtype=np.float64)
    merged_flags = np.zeros((n_ant, n_freq, n_pol), dtype=np.int8)
    antenna_presence = {
        antenna: sum(1 for solution in solutions if antenna in solution["antennas"]) for antenna in antenna_order
    }
    first_solution = solutions[0]

    for antenna_idx, antenna in enumerate(antenna_order):
        for freq_idx in range(n_freq):
            for pol_idx in range(n_pol):
                if antenna == reference:
                    merged_values[antenna_idx, freq_idx, pol_idx] = 0.0
                    reference_valid = all(
                        float(solution["flags"][solution["antennas"].index(reference), freq_idx, pol_idx])
                        > 0.0
                        for solution in solutions
                        if reference in solution["antennas"]
                    )
                    merged_flags[antenna_idx, freq_idx, pol_idx] = int(
                        flag_policy == "any" or reference_valid
                    )
                    continue

                if copy_first and antenna in first_solution["antennas"]:
                    first_idx = first_solution["antennas"].index(antenna)
                    first_flag = float(first_solution["flags"][first_idx, freq_idx, pol_idx])
                    first_value = first_solution["data"][first_idx, freq_idx, pol_idx]
                    if first_flag > 0.0 and np.isfinite(first_value):
                        merged_values[antenna_idx, freq_idx, pol_idx] = first_value
                        merged_flags[antenna_idx, freq_idx, pol_idx] = 1
                    continue

                if antenna_presence.get(antenna, 0) == 1:
                    unique_solution = next(solution for solution in solutions if antenna in solution["antennas"])
                    unique_idx = unique_solution["antennas"].index(antenna)
                    if float(unique_solution["flags"][unique_idx, freq_idx, pol_idx]) == 0.0:
                        continue
                    value = unique_solution["data"][unique_idx, freq_idx, pol_idx]
                    if not np.isfinite(value):
                        continue
                    merged_values[antenna_idx, freq_idx, pol_idx] = value
                    merged_flags[antenna_idx, freq_idx, pol_idx] = 1
                    continue

                values = []
                weights = []
                all_present_valid = True
                for solution in solutions:
                    if antenna not in solution["antennas"]:
                        continue
                    local_idx = solution["antennas"].index(antenna)
                    if float(solution["flags"][local_idx, freq_idx, pol_idx]) == 0.0:
                        all_present_valid = False
                        continue
                    value = solution["data"][local_idx, freq_idx, pol_idx]
                    if not np.isfinite(value):
                        continue
                    values.append(value)
                    weights.append(solution["weight"])

                if not values or (flag_policy == "all" and not all_present_valid):
                    continue

                merged_values[antenna_idx, freq_idx, pol_idx] = _weighted_circular_mean(values, weights)
                merged_flags[antenna_idx, freq_idx, pol_idx] = 1

    return merged_values, merged_flags, freq_axis, np.array(antenna_order, dtype="S")


def _expand_to_h5parm(values: np.ndarray, flags: np.ndarray, freq_axis: np.ndarray, time_axis: np.ndarray, antennas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_time = len(time_axis)
    n_freq = len(freq_axis)
    n_ant = len(antennas)
    expanded_values = np.zeros((n_time, n_freq, n_ant, 1, 2), dtype=np.float64)
    expanded_flags = np.zeros((n_time, n_freq, n_ant, 1, 2), dtype=np.float64)

    for t_idx in range(n_time):
        for f_idx in range(n_freq):
            for a_idx in range(n_ant):
                value = values[a_idx, f_idx]
                expanded_values[t_idx, f_idx, a_idx, 0, 0] = np.real(value)
                expanded_values[t_idx, f_idx, a_idx, 0, 1] = np.imag(value)
                expanded_flags[t_idx, f_idx, a_idx, 0, :] = 1.0 if flags[a_idx, f_idx] else 0.0

    return expanded_values, expanded_flags


def _expand_phase_to_h5parm(values: np.ndarray, flags: np.ndarray, freq_axis: np.ndarray, time_axis: np.ndarray, antennas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_time = len(time_axis)
    n_freq = len(freq_axis)
    n_ant = len(antennas)
    n_pol = values.shape[2]
    expanded_values = np.zeros((n_time, n_freq, n_ant, 1, n_pol), dtype=np.float64)
    expanded_flags = np.zeros((n_time, n_freq, n_ant, 1, n_pol), dtype=np.float64)
    expanded_values[:, :, :, 0, :] = np.transpose(values, (1, 0, 2))[None, ...]
    expanded_flags[:, :, :, 0, :] = np.transpose(flags, (1, 0, 2))[None, ...]
    return expanded_values, expanded_flags


def _expand_scalar_to_h5parm(values: np.ndarray, flags: np.ndarray, freq_axis: np.ndarray, time_axis: np.ndarray, antennas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_time = len(time_axis)
    n_freq = len(freq_axis)
    n_ant = len(antennas)
    n_pol = values.shape[2]
    expanded_values = np.zeros((n_time, n_freq, n_ant, 1, n_pol), dtype=np.float64)
    expanded_flags = np.zeros((n_time, n_freq, n_ant, 1, n_pol), dtype=np.float64)
    expanded_values[:, :, :, 0, :] = np.transpose(values, (1, 0, 2))[None, ...]
    expanded_flags[:, :, :, 0, :] = np.transpose(flags, (1, 0, 2))[None, ...]
    return expanded_values, expanded_flags


def _copy_attrs(source_node, target_node) -> None:
    for attr_name in source_node._v_attrs._f_list():
        target_node._v_attrs[attr_name] = source_node._v_attrs[attr_name]


def _copy_dataset_attrs(source_node, target_node) -> None:
    for attr_name in source_node._v_attrs._f_list():
        target_node._v_attrs[attr_name] = source_node._v_attrs[attr_name]


def _load_template(path: str) -> dict:
    with tables.open_file(path, mode="r") as handle:
        sol_group = handle.root.sol000
        phase_group = sol_group.phase000
        amplitude_group = sol_group.amplitude000

        template = {
            "root_attrs": {name: handle.root._v_attrs[name] for name in handle.root._v_attrs._f_list()},
            "sol_attrs": {name: sol_group._v_attrs[name] for name in sol_group._v_attrs._f_list()},
            "phase_group_attrs": {name: phase_group._v_attrs[name] for name in phase_group._v_attrs._f_list()},
            "amplitude_group_attrs": {name: amplitude_group._v_attrs[name] for name in amplitude_group._v_attrs._f_list()},
            "phase_array_attrs": {},
            "amplitude_array_attrs": {},
            "source_rows": sol_group.source[:] if hasattr(sol_group, "source") else None,
            "source_descr": sol_group.source.description if hasattr(sol_group, "source") else None,
            "antenna_rows": sol_group.antenna[:] if hasattr(sol_group, "antenna") else None,
            "antenna_descr": sol_group.antenna.description if hasattr(sol_group, "antenna") else None,
            "phase_dir_axis": np.asarray(phase_group.dir[:]) if hasattr(phase_group, "dir") else np.array([], dtype="S"),
            "phase_pol_axis": np.asarray(phase_group.pol[:]) if hasattr(phase_group, "pol") else np.array([], dtype="S"),
            "amplitude_dir_axis": np.asarray(amplitude_group.dir[:]) if hasattr(amplitude_group, "dir") else np.array([], dtype="S"),
            "amplitude_pol_axis": np.asarray(amplitude_group.pol[:]) if hasattr(amplitude_group, "pol") else np.array([], dtype="S"),
        }

        for name in ["val", "freq", "time", "ant", "dir", "pol", "weight"]:
            if hasattr(phase_group, name):
                template["phase_array_attrs"][name] = {
                    attr_name: phase_group._f_get_child(name)._v_attrs[attr_name]
                    for attr_name in phase_group._f_get_child(name)._v_attrs._f_list()
                }
            if hasattr(amplitude_group, name):
                template["amplitude_array_attrs"][name] = {
                    attr_name: amplitude_group._f_get_child(name)._v_attrs[attr_name]
                    for attr_name in amplitude_group._f_get_child(name)._v_attrs._f_list()
                }

        return template


def _merge_table_rows(rows_list: List[np.ndarray], order: List[str], description) -> np.ndarray:
    if not rows_list:
        return np.array([], dtype=description.dtype if hasattr(description, "dtype") else np.dtype([("name", "S16"), ("position", "f4", (3,))]))

    merged = []
    seen = set()
    for rows in rows_list:
        if rows is None:
            continue
        for row in rows:
            name = row[0].decode("utf-8") if isinstance(row[0], bytes) else str(row[0])
            if name in seen:
                continue
            seen.add(name)
            merged.append(row)

    ordered = []
    for name in order:
        for row in merged:
            row_name = row[0].decode("utf-8") if isinstance(row[0], bytes) else str(row[0])
            if row_name == name:
                ordered.append(row)
                break
    for row in merged:
        row_name = row[0].decode("utf-8") if isinstance(row[0], bytes) else str(row[0])
        if row_name not in {item[0].decode("utf-8") if isinstance(item[0], bytes) else str(item[0]) for item in ordered}:
            ordered.append(row)

    if not ordered:
        return np.array([], dtype=rows_list[0].dtype)
    return np.array(ordered, dtype=rows_list[0].dtype)


def write_output(output_path: str, phase: np.ndarray, phase_flags: np.ndarray, amplitude: np.ndarray, amplitude_flags: np.ndarray, freq_axis: np.ndarray, time_axis: np.ndarray, antennas: np.ndarray, template: dict) -> None:
    with tables.open_file(output_path, mode="w") as handle:
        for attr_name, attr_value in template["root_attrs"].items():
            handle.root._v_attrs[attr_name] = attr_value

        sol_group = handle.create_group("/", "sol000")
        for attr_name, attr_value in template["sol_attrs"].items():
            sol_group._v_attrs[attr_name] = attr_value

        phase_group = handle.create_group("/sol000", "phase000", title="phase")
        phase_group._v_attrs["parmdb_type"] = "phase"
        phase_group._v_attrs["type"] = "phase"
        phase_group._v_attrs["soltab_type"] = "phase"
        for attr_name, attr_value in template["phase_group_attrs"].items():
            if attr_name not in {"parmdb_type", "type", "soltab_type"}:
                phase_group._v_attrs[attr_name] = attr_value
        amplitude_group = handle.create_group("/sol000", "amplitude000", title="amplitude")
        amplitude_group._v_attrs["parmdb_type"] = "amplitude"
        amplitude_group._v_attrs["type"] = "amplitude"
        amplitude_group._v_attrs["soltab_type"] = "amplitude"
        for attr_name, attr_value in template["amplitude_group_attrs"].items():
            if attr_name not in {"parmdb_type", "type", "soltab_type"}:
                amplitude_group._v_attrs[attr_name] = attr_value

        phase_dir_axis = template.get("phase_dir_axis", np.array([], dtype="S"))
        phase_pol_axis = template.get("phase_pol_axis", np.array([], dtype="S"))
        amplitude_dir_axis = template.get("amplitude_dir_axis", np.array([], dtype="S"))
        amplitude_pol_axis = template.get("amplitude_pol_axis", np.array([], dtype="S"))

        phase_vals, phase_weight = _expand_phase_to_h5parm(phase, phase_flags, freq_axis, time_axis, antennas)
        amplitude_vals, amplitude_weight = _expand_scalar_to_h5parm(amplitude, amplitude_flags, freq_axis, time_axis, antennas)

        for name, values in [("val", phase_vals), ("freq", freq_axis), ("time", time_axis), ("ant", np.array(antennas, dtype="S")), ("dir", phase_dir_axis), ("pol", phase_pol_axis), ("weight", phase_weight)]:
            node = handle.create_array(phase_group, name, values)
            if name in template["phase_array_attrs"]:
                for attr_name, attr_value in template["phase_array_attrs"][name].items():
                    node._v_attrs[attr_name] = attr_value

        for name, values in [("val", amplitude_vals), ("freq", freq_axis), ("time", time_axis), ("ant", np.array(antennas, dtype="S")), ("dir", amplitude_dir_axis), ("pol", amplitude_pol_axis), ("weight", amplitude_weight)]:
            node = handle.create_array(amplitude_group, name, values)
            if name in template["amplitude_array_attrs"]:
                for attr_name, attr_value in template["amplitude_array_attrs"][name].items():
                    node._v_attrs[attr_name] = attr_value

        if template["antenna_descr"] is not None:
            antenna_rows = _merge_table_rows([template["antenna_rows"]], [decode_value(item) for item in antennas], template["antenna_descr"])
            antenna_table = handle.create_table(sol_group, "antenna", template["antenna_descr"], expectedrows=len(antenna_rows))
            for row in antenna_rows:
                antenna_table.row["name"] = row[0]
                antenna_table.row["position"] = row[1]
                antenna_table.row.append()
            antenna_table.flush()

        if template["source_descr"] is not None:
            source_rows = template["source_rows"]
            if source_rows is None:
                source_rows = np.array([], dtype=template["source_descr"].dtype if hasattr(template["source_descr"], "dtype") else np.dtype([("name", "S128"), ("dir", "f4", (2,))]))
            source_table = handle.create_table(sol_group, "source", template["source_descr"], expectedrows=len(source_rows))
            for row in source_rows:
                source_table.row["name"] = row[0]
                source_table.row["dir"] = row[1]
                source_table.row.append()
            source_table.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LOFAR H5Parm bandpass solutions from multiple input files. "
            "The phase000 and amplitude000 soltabs are combined over the union "
            "of antennas while keeping polarizations independent. Input files "
            "are weighted by their number of time slots; phases are referenced "
            "to a common antenna, and amplitudes are combined independently. "
            "Flags are handled per antenna, frequency, and polarization. The "
            "output contains two time slots at the global minimum and maximum "
            "input times."
        )
    )
    parser.add_argument("--method", choices=["average", "median"], default="average", help="Combination method")
    parser.add_argument(
        "--reference-antenna",
        default=None,
        help=(
            "Reference antenna common to all inputs. If omitted, the first "
            "antenna in the first input that is present in every input is "
            "selected automatically."
        ),
    )
    parser.add_argument(
        "--flag-policy",
        choices=["any", "all"],
        default="any",
        help=(
            "Flag handling per antenna/frequency/polarization: 'any' keeps the "
            "output valid when at least one input containing that antenna is valid; "
            "'all' flags it when any such input is flagged. Example: if antA is "
            "valid in input1 but flagged in input2, antA is valid with 'any' and "
            "flagged with 'all'. An input missing antA is ignored."
        ),
    )
    parser.add_argument(
        "--phase-from-first",
        action="store_true",
        help="Copy phase solutions from the first input; merge phases only for antennas missing there",
    )
    parser.add_argument(
        "--amplitude-from-first",
        action="store_true",
        help="Copy amplitude solutions from the first input; merge amplitudes only for antennas missing there",
    )
    parser.add_argument("output", help="Output HDF5 file")
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files or glob patterns")
    args = parser.parse_args()

    if not args.output:
        parser.error("An output path is required")

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        parser.error("No input files were found")

    # Re-load all solutions for both datasets.
    phase_solutions = [load_solution(path, "phase000") for path in input_paths]
    amplitude_solutions = [load_solution(path, "amplitude000") for path in input_paths]

    reference_antenna = args.reference_antenna
    merged_phase, phase_flags, freq_axis, antennas = merge_solution(
        phase_solutions,
        args.method,
        reference_antenna=reference_antenna,
        is_amplitude=False,
        flag_policy=args.flag_policy,
        copy_first=args.phase_from_first,
    )
    merged_amplitude, amplitude_flags, _, _ = merge_solution(
        amplitude_solutions,
        args.method,
        reference_antenna=reference_antenna,
        is_amplitude=True,
        flag_policy=args.flag_policy,
        copy_first=args.amplitude_from_first,
    )

    all_solution_times = np.concatenate(
        [solution["time"] for solution in phase_solutions + amplitude_solutions]
    )
    representative_time = np.array(
        [np.min(all_solution_times), np.max(all_solution_times)], dtype=np.float64
    )

    template = _load_template(input_paths[0])

    write_output(
        args.output,
        merged_phase,
        phase_flags,
        merged_amplitude,
        amplitude_flags,
        freq_axis,
        representative_time,
        antennas,
        template,
    )

    print(f"Wrote merged output to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
