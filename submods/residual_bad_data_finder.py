#!/usr/bin/env python3
"""Plot flag- and weight-aware residual power for every antenna in an MS.

Example:
    python submods/residual_bad_data_finder.py observation.ms
    python submods/residual_bad_data_finder.py observation.ms --time-bin-seconds 60 \\
        --channels-per-bin 4 --output residual_power.png

The run saves the time-frequency image at the requested output path, plus
frequency-aggregated ``_time`` and time-aggregated ``_freq`` line-plot views.
The Measurement Set is opened read-only. By default, the plotted score is the
mean WEIGHT_SPECTRUM * |residual|^2 per valid complex visibility; --ignore-weights
instead computes unweighted mean |residual|^2. Both scores are diagnostics, not
calibrated chi-squared values or automatic flagging decisions. Only parallel-hand
correlations (XX/YY or RR/LL) are included; cross-hands are ignored. The current
plot requires a single DATA_DESC_ID / spectral window in the MS. Progress bars
show Measurement Set processing and plot rendering progress.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator
from casacore.tables import table as casacore_table
from tqdm import tqdm


PARALLEL_HAND_CORRELATIONS = {
    5: "RR",
    8: "LL",
    9: "XX",
    12: "YY",
}


@dataclass
class ResidualDiagnostic:
    scores: np.ndarray
    sample_counts: np.ndarray
    usable_cells: np.ndarray
    relative_scores: np.ndarray
    antenna_names: np.ndarray
    time_edges_minutes: np.ndarray
    frequency_edges_mhz: np.ndarray
    total_cross_samples: int
    flagged_cross_samples: int
    valid_cross_samples: int
    weight_column: Optional[str]
    correlations: tuple[str, ...]
    time_samples_per_bin: int
    integration_width_seconds: float

    @property
    def score_label(self) -> str:
        if self.weight_column is None:
            return "Mean(|residual|²) per valid complex sample"
        return (
            f"Mean({self.weight_column} × |residual|²) "
            "per valid complex sample"
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an antenna-by-antenna time/frequency plot of residual power. "
            "Only XX/YY or RR/LL parallel hands are used, and the Measurement "
            "Set is read-only. Currently supports a single DATA_DESC_ID / "
            "spectral window."
        ),
        epilog=(
            "Example: python submods/residual_bad_data_finder.py observation.ms "
            "--time-bin-seconds 60 --channels-per-bin 4"
        ),
    )
    parser.add_argument("ms", help="Measurement Set containing residual visibilities")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Time-frequency PNG path; companion line plots use _time and _freq "
            "suffixes (default: <MS-name>.residual_power.png)"
        ),
    )
    parser.add_argument(
        "--data-column",
        default="RESIDUAL_DATA",
        help="Residual visibility column (default: RESIDUAL_DATA)",
    )
    parser.add_argument(
        "--weight-column",
        choices=("auto", "WEIGHT_SPECTRUM", "WEIGHT"),
        default="auto",
        help=(
            "Visibility weights; auto prefers WEIGHT_SPECTRUM and falls back "
            "to WEIGHT (default: auto)"
        ),
    )
    parser.add_argument(
        "--ignore-weights",
        action="store_true",
        help=(
            "Ignore WEIGHT/WEIGHT_SPECTRUM and compute unweighted residual power; "
            "FLAG and FLAG_ROW are still applied"
        ),
    )
    parser.add_argument(
        "--time-bin-seconds",
        type=positive_float,
        default=None,
        help=(
            "Average into the nearest whole number of integration samples "
            "(minimum 1); default keeps native integrations"
        ),
    )
    parser.add_argument(
        "--channels-per-bin",
        type=positive_int,
        default=1,
        help="Number of adjacent frequency channels per plotted bin (default: 1)",
    )
    parser.add_argument(
        "--min-samples",
        type=positive_int,
        default=15,
        help="Minimum valid complex samples for a plotted cell (default: 15)",
    )
    parser.add_argument(
        "--min-sample-fraction",
        type=float,
        default=0.2,
        help=(
            "Also require this fraction of the median antenna sample count in "
            "each time/frequency cell (0 to 1; default: 0.2)"
        ),
    )
    parser.add_argument(
        "--columns",
        type=positive_int,
        default=None,
        help="Number of antenna panels per row (default: automatic, up to 10)",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap (default: magma)",
    )
    parser.add_argument(
        "--scale",
        choices=("log", "sqrt", "linear"),
        default="log",
        help=(
            "Value scaling for the heatmap colors and 1D plot y-axes "
            "(default: log)"
        ),
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=2.0,
        help="Lower shared color-scale percentile (default: 2)",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=98.0,
        help="Upper shared color-scale percentile (default: 98)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=positive_int,
        default=50000,
        help="Rows read from the MS at a time (default: 50000)",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=140,
        help="PNG resolution (default: 140)",
    )
    parser.add_argument(
        "--top-antennas",
        type=positive_int,
        default=10,
        help="Number of highest relative-score antennas to print (default: 10)",
    )
    parser.add_argument(
        "--high-ratio",
        type=positive_float,
        default=2.0,
        help=(
            "Ratio counted as high in the textual summary; descriptive only "
            "(default: 2)"
        ),
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        action="store_true",
        dest="overwrite",
        help="Overwrite existing plots (the default; retained for compatibility)",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Fail if any output plot already exists",
    )
    parser.set_defaults(overwrite=True)
    return parser


def axis_edges_from_centers(
    centers: np.ndarray, fallback_width: float
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        raise ValueError("Cannot make an axis for an empty coordinate array")
    if centers.size == 1:
        half_width = max(abs(fallback_width), 1.0) / 2.0
        return np.array([centers[0] - half_width, centers[0] + half_width])
    differences = np.diff(centers)
    if np.any(differences <= 0):
        raise ValueError("Axis centers must be strictly increasing")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - differences[0] / 2.0
    edges[-1] = centers[-1] + differences[-1] / 2.0
    return edges


def build_time_grid(
    times: np.ndarray,
    time_bin_seconds: Optional[float],
    integration_width: Optional[float],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if times.size == 0:
        raise ValueError("TIME contains no samples")
    if time_bin_seconds is not None and (
        not math.isfinite(time_bin_seconds) or time_bin_seconds <= 0
    ):
        raise ValueError("time_bin_seconds must be a positive finite number")
    centers, sample_ids = np.unique(times, return_inverse=True)
    if integration_width is None or not math.isfinite(integration_width) or integration_width <= 0:
        positive_differences = np.diff(centers)
        positive_differences = positive_differences[positive_differences > 0]
        integration_width = (
            float(np.median(positive_differences))
            if positive_differences.size
            else 1.0
        )

    if time_bin_seconds is None:
        samples_per_bin = 1
    else:
        samples_per_bin = max(
            1,
            int(math.floor(time_bin_seconds / integration_width + 0.5)),
        )

    sample_edges = np.empty(centers.size + 1, dtype=np.float64)
    if centers.size == 1:
        sample_edges[0] = centers[0] - integration_width / 2.0
        sample_edges[1] = centers[0] + integration_width / 2.0
    else:
        sample_edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
        sample_edges[0] = centers[0] - integration_width / 2.0
        sample_edges[-1] = centers[-1] + integration_width / 2.0

    time_ids = sample_ids // samples_per_bin
    edge_indices = np.arange(0, centers.size + 1, samples_per_bin)
    if edge_indices[-1] != centers.size:
        edge_indices = np.append(edge_indices, centers.size)
    edges = sample_edges[edge_indices]
    return (
        time_ids.astype(np.int64, copy=False),
        (edges - edges[0]) / 60.0,
        samples_per_bin,
        float(integration_width),
    )


def select_weight_column(columns: set[str], requested: str) -> str:
    if requested == "auto":
        if "WEIGHT_SPECTRUM" in columns:
            return "WEIGHT_SPECTRUM"
        if "WEIGHT" in columns:
            return "WEIGHT"
        raise ValueError("MS has neither WEIGHT_SPECTRUM nor WEIGHT")
    if requested not in columns:
        raise ValueError(f"Requested weight column {requested!r} is not present in the MS")
    return requested


def read_weight_chunk(
    ms: object,
    weight_column: str,
    start: int,
    row_count: int,
    channel_count: int,
    polarization_count: int,
) -> np.ndarray:
    weights = ms.getcol(weight_column, startrow=start, nrow=row_count)
    if weight_column == "WEIGHT" and weights.ndim == 2:
        weights = np.broadcast_to(weights[:, None, :], (row_count, channel_count, polarization_count))
    if weights.shape != (row_count, channel_count, polarization_count):
        raise ValueError(
            f"{weight_column} shape {weights.shape} does not match residual data "
            f"shape {(row_count, channel_count, polarization_count)}"
        )
    return weights


def calculate_diagnostic(
    ms_path: Path,
    data_column: str,
    requested_weight_column: str,
    time_bin_seconds: Optional[float],
    channels_per_bin: int,
    min_samples: int,
    min_sample_fraction: float,
    chunk_rows: int,
    ignore_weights: bool = False,
) -> ResidualDiagnostic:
    ms = casacore_table(str(ms_path), readonly=True, ack=False)
    try:
        columns = set(ms.colnames())
        required_columns = {
            data_column,
            "FLAG",
            "ANTENNA1",
            "ANTENNA2",
            "TIME",
            "DATA_DESC_ID",
        }
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(f"Missing required MS columns: {', '.join(missing)}")

        weight_column = (
            None
            if ignore_weights
            else select_weight_column(columns, requested_weight_column)
        )
        nrows = ms.nrows()
        if nrows == 0:
            raise ValueError("Measurement Set contains no rows")

        times = ms.getcol("TIME").astype(np.float64, copy=False)
        if not np.all(np.isfinite(times)):
            raise ValueError("TIME contains non-finite values")
        time_ids, time_edges_minutes, time_samples_per_bin, integration_width_seconds = build_time_grid(
            times,
            time_bin_seconds,
            float(np.median(ms.getcol("INTERVAL")))
            if "INTERVAL" in columns
            else None,
        )
        ntime = len(time_edges_minutes) - 1

        ant1 = ms.getcol("ANTENNA1").astype(np.int64, copy=False)
        ant2 = ms.getcol("ANTENNA2").astype(np.int64, copy=False)
        with casacore_table(
            f"{ms_path}/ANTENNA", readonly=True, ack=False
        ) as antenna_table:
            antenna_names = antenna_table.getcol("NAME")
        nants = len(antenna_names)
        if np.any((ant1 < 0) | (ant1 >= nants) | (ant2 < 0) | (ant2 >= nants)):
            raise ValueError("ANTENNA1/ANTENNA2 contains an index outside the ANTENNA table")

        ddids = np.unique(ms.getcol("DATA_DESC_ID"))
        if ddids.size != 1:
            raise ValueError(
                "This plot currently requires a single DATA_DESC_ID / spectral window; "
                f"found {ddids.tolist()}"
            )
        with casacore_table(
            f"{ms_path}/DATA_DESCRIPTION", readonly=True, ack=False
        ) as description_table:
            description_id = int(ddids[0])
            spectral_window_id = int(
                description_table.getcell("SPECTRAL_WINDOW_ID", description_id)
            )
            polarization_id = int(
                description_table.getcell("POLARIZATION_ID", description_id)
            )
        with casacore_table(
            f"{ms_path}/POLARIZATION", readonly=True, ack=False
        ) as polarization_table:
            correlation_types = np.asarray(
                polarization_table.getcell("CORR_TYPE", polarization_id),
                dtype=np.int64,
            )
        parallel_hand_indices = np.flatnonzero(
            np.isin(correlation_types, tuple(PARALLEL_HAND_CORRELATIONS))
        )
        if parallel_hand_indices.size == 0:
            raise ValueError(
                "POLARIZATION metadata has no XX, YY, RR, or LL parallel hands "
                f"(CORR_TYPE={correlation_types.tolist()})"
            )
        correlations = tuple(
            PARALLEL_HAND_CORRELATIONS[int(correlation_types[index])]
            for index in parallel_hand_indices
        )
        with casacore_table(
            f"{ms_path}/SPECTRAL_WINDOW", readonly=True, ack=False
        ) as spectral_window_table:
            channel_frequencies = np.asarray(
                spectral_window_table.getcell("CHAN_FREQ", spectral_window_id),
                dtype=np.float64,
            )
            if "CHAN_WIDTH" in spectral_window_table.colnames():
                channel_widths = np.asarray(
                    spectral_window_table.getcell("CHAN_WIDTH", spectral_window_id),
                    dtype=np.float64,
                )
            else:
                channel_widths = np.ones_like(channel_frequencies)

        first_data = ms.getcol(data_column, startrow=0, nrow=1)
        if first_data.ndim != 3:
            raise ValueError(f"{data_column} must have row, channel, and polarization axes")
        channel_count, polarization_count = first_data.shape[1:]
        if correlation_types.size != polarization_count:
            raise ValueError(
                f"POLARIZATION metadata lists {correlation_types.size} correlations "
                f"but {data_column} contains {polarization_count}"
            )
        if channel_count != channel_frequencies.size:
            raise ValueError(
                f"{data_column} has {channel_count} channels but its spectral window "
                f"defines {channel_frequencies.size}"
            )
        del first_data

        channel_order = np.argsort(channel_frequencies)
        channel_groups = [
            channel_order[start : start + channels_per_bin]
            for start in range(0, channel_count, channels_per_bin)
        ]
        frequency_centers_hz = np.array(
            [np.mean(channel_frequencies[group]) for group in channel_groups],
            dtype=np.float64,
        )
        frequency_widths_hz = np.array(
            [np.sum(np.abs(channel_widths[group])) for group in channel_groups],
            dtype=np.float64,
        )
        frequency_edges_mhz = axis_edges_from_centers(
            frequency_centers_hz / 1e6,
            float(np.sum(frequency_widths_hz)) / 1e6,
        )

        nfrequency = len(channel_groups)
        power_sums = np.zeros((ntime, nfrequency, nants), dtype=np.float64)
        sample_counts = np.zeros((ntime, nfrequency, nants), dtype=np.int64)
        total_cross_samples = 0
        flagged_cross_samples = 0
        valid_cross_samples = 0
        total_valid_power = 0.0
        row_flags_all = (
            ms.getcol("FLAG_ROW").astype(bool, copy=False)
            if "FLAG_ROW" in columns
            else np.zeros(nrows, dtype=bool)
        )

        for start in tqdm(
            range(0, nrows, chunk_rows),
            desc="Processing Measurement Set rows",
            unit="chunk",
        ):
            row_count = min(chunk_rows, nrows - start)
            row_slice = slice(start, start + row_count)
            residual = ms.getcol(data_column, startrow=start, nrow=row_count)
            flags = ms.getcol("FLAG", startrow=start, nrow=row_count).astype(
                bool, copy=False
            )
            weights = (
                read_weight_chunk(
                    ms,
                    weight_column,
                    start,
                    row_count,
                    channel_count,
                    polarization_count,
                )
                if weight_column is not None
                else None
            )
            row_ant1 = ant1[row_slice]
            row_ant2 = ant2[row_slice]
            cross_baseline = row_ant1 != row_ant2
            row_usable = cross_baseline & ~row_flags_all[row_slice]
            chunk_time_ids = time_ids[row_slice]

            if flags.shape != residual.shape:
                raise ValueError(
                    f"FLAG shape {flags.shape} does not match "
                    f"{data_column} shape {residual.shape}"
                )
            total_cross_samples += (
                int(np.count_nonzero(cross_baseline))
                * channel_count
                * parallel_hand_indices.size
            )
            selected_flags = flags[:, :, parallel_hand_indices]
            selected_residual = residual[:, :, parallel_hand_indices]
            selected_weights = (
                weights[:, :, parallel_hand_indices]
                if weights is not None
                else None
            )
            flagged_cross_samples += int(
                np.count_nonzero(
                    (selected_flags | row_flags_all[row_slice, None, None])
                    & cross_baseline[:, None, None]
                )
            )

            valid = (
                ~selected_flags
                & row_usable[:, None, None]
                & np.isfinite(selected_residual.real)
                & np.isfinite(selected_residual.imag)
            )
            if selected_weights is not None:
                valid &= np.isfinite(selected_weights) & (selected_weights > 0)
            valid_cross_samples += int(np.count_nonzero(valid))

            with np.errstate(over="ignore", invalid="ignore"):
                residual_power = (
                    selected_residual.real.astype(np.float64) ** 2
                    + selected_residual.imag.astype(np.float64) ** 2
                )
                weighted_power = np.where(
                    valid,
                    residual_power
                    if selected_weights is None
                    else selected_weights * residual_power,
                    0.0,
                )
            total_valid_power += float(
                weighted_power[cross_baseline].sum(dtype=np.float64)
            )

            for frequency_bin, channel_indices in enumerate(channel_groups):
                row_power = weighted_power[:, channel_indices, :].sum(
                    axis=(1, 2), dtype=np.float64
                )
                row_sample_count = valid[:, channel_indices, :].sum(
                    axis=(1, 2), dtype=np.int64
                )
                for row_antennas in (row_ant1, row_ant2):
                    keys = (
                        chunk_time_ids[cross_baseline] * nants
                        + row_antennas[cross_baseline]
                    )
                    power_sums[:, frequency_bin, :] += np.bincount(
                        keys,
                        weights=row_power[cross_baseline],
                        minlength=ntime * nants,
                    ).reshape(ntime, nants)
                    sample_counts[:, frequency_bin, :] += np.bincount(
                        keys,
                        weights=row_sample_count[cross_baseline],
                        minlength=ntime * nants,
                    ).astype(np.int64).reshape(ntime, nants)

        if int(sample_counts.sum()) != 2 * valid_cross_samples:
            raise AssertionError("Antenna sample counts failed the endpoint consistency check")
        if not np.isclose(
            float(power_sums.sum()),
            2.0 * total_valid_power,
            rtol=1e-10,
            atol=1e-6,
        ):
            raise AssertionError("Antenna power sums failed the endpoint consistency check")

    finally:
        ms.close()

    scores = np.full_like(power_sums, np.nan)
    np.divide(power_sums, sample_counts, out=scores, where=sample_counts > 0)
    usable_cells = np.zeros_like(sample_counts, dtype=bool)
    relative_scores = np.full_like(scores, np.nan)

    for time_index in tqdm(
        range(ntime),
        desc="Calculating residual scores",
        unit="time bin",
    ):
        for frequency_index in range(nfrequency):
            counts = sample_counts[time_index, frequency_index]
            positive_counts = counts[counts > 0]
            if positive_counts.size == 0:
                continue
            count_threshold = max(
                min_samples,
                min_sample_fraction * float(np.median(positive_counts)),
            )
            usable = (
                (counts >= count_threshold)
                & np.isfinite(scores[time_index, frequency_index])
                & (scores[time_index, frequency_index] > 0)
            )
            usable_cells[time_index, frequency_index] = usable
            if not np.any(usable):
                continue
            bin_median = float(np.median(scores[time_index, frequency_index, usable]))
            relative_scores[time_index, frequency_index, usable] = (
                scores[time_index, frequency_index, usable] / bin_median
            )

    return ResidualDiagnostic(
        scores=scores,
        sample_counts=sample_counts,
        usable_cells=usable_cells,
        relative_scores=relative_scores,
        antenna_names=antenna_names,
        time_edges_minutes=time_edges_minutes,
        frequency_edges_mhz=frequency_edges_mhz,
        total_cross_samples=total_cross_samples,
        flagged_cross_samples=flagged_cross_samples,
        valid_cross_samples=valid_cross_samples,
        weight_column=weight_column,
        correlations=correlations,
        time_samples_per_bin=time_samples_per_bin,
        integration_width_seconds=integration_width_seconds,
    )


def print_candidate_summary(
    diagnostic: ResidualDiagnostic, high_ratio: float, top_antennas: int
) -> None:
    summaries = []
    for antenna_id, name in enumerate(
        tqdm(
            diagnostic.antenna_names,
            desc="Ranking antennas",
            unit="antenna",
        )
    ):
        usable = diagnostic.usable_cells[:, :, antenna_id]
        ratios = diagnostic.relative_scores[:, :, antenna_id][usable]
        if ratios.size == 0:
            continue
        counts = diagnostic.sample_counts[:, :, antenna_id][usable]
        summaries.append(
            (
                float(np.median(ratios)),
                float(np.percentile(ratios, 90)),
                int(np.count_nonzero(ratios > high_ratio)),
                int(ratios.size),
                float(np.median(counts)),
                int(antenna_id),
                str(name),
            )
        )
    summaries.sort(reverse=True)

    print("Highest median within-bin residual-power ratios (screening only):")
    print("rank antenna       median   p90   bins_above_ratio   median_samples")
    for rank, summary in enumerate(summaries[:top_antennas], start=1):
        median_ratio, p90_ratio, high_bins, bin_count, median_samples, antenna_id, name = summary
        print(
            f"{rank:>4} {name:<12} {median_ratio:>7.3f} {p90_ratio:>6.3f} "
            f"{high_bins:>5}/{bin_count:<5} {median_samples:>14.0f}"
        )


def resolve_plot_columns(antenna_count: int, requested_columns: Optional[int]) -> int:
    if requested_columns is not None:
        return requested_columns
    return min(10, max(1, math.ceil(math.sqrt(antenna_count))))


def color_normalization(scale: str, vmin: float, vmax: float) -> Normalize:
    if scale == "log":
        return LogNorm(vmin=vmin, vmax=vmax)
    if scale == "sqrt":
        return PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    if scale == "linear":
        return Normalize(vmin=vmin, vmax=vmax)
    raise ValueError(f"Unsupported plot scale: {scale}")


def create_plot(
    diagnostic: ResidualDiagnostic,
    output_path: Path,
    ms_name: str,
    colormap_name: str,
    scale: str,
    columns: Optional[int],
    dpi: int,
    vmin_percentile: float,
    vmax_percentile: float,
) -> tuple[float, float]:
    masked_scores = np.ma.masked_where(
        ~diagnostic.usable_cells | ~np.isfinite(diagnostic.scores),
        diagnostic.scores,
    )
    values = masked_scores.compressed()
    if values.size == 0:
        raise ValueError("No antenna/time/frequency cells have enough valid samples to plot")

    vmin, vmax = np.percentile(values, [vmin_percentile, vmax_percentile])
    if vmin <= 0:
        vmin = float(np.min(values[values > 0]))
    if vmax <= vmin:
        center = max(float(np.median(values)), np.finfo(np.float64).tiny)
        vmin, vmax = center / math.sqrt(10.0), center * math.sqrt(10.0)

    antenna_count = len(diagnostic.antenna_names)
    columns = resolve_plot_columns(antenna_count, columns)
    rows = math.ceil(antenna_count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 2.1 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad("#d9d9d9")
    norm = color_normalization(scale, float(vmin), float(vmax))
    mesh = None

    for antenna_id, ax in enumerate(
        tqdm(
            axes.flat,
            total=axes.size,
            desc="Drawing time-frequency panels",
            unit="panel",
        )
    ):
        if antenna_id >= antenna_count:
            ax.set_visible(False)
            continue
        ax.set_facecolor("#d9d9d9")
        mesh = ax.pcolormesh(
            diagnostic.time_edges_minutes,
            diagnostic.frequency_edges_mhz,
            masked_scores[:, :, antenna_id].T,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.set_title(
            str(diagnostic.antenna_names[antenna_id]),
            fontsize=8,
            loc="left",
            pad=2,
        )
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="both", labelsize=7, length=2, pad=1)

    fig.suptitle(
        "Flag- and weight-aware parallel-hand residual power",
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.965,
        f"{ms_name} | Correlations: {', '.join(diagnostic.correlations)}",
        ha="center",
        fontsize=9,
    )
    fig.supxlabel("Elapsed time (min)", fontsize=9, y=0.075)
    fig.supylabel("Frequency (MHz)", fontsize=9, x=0.015)
    fig.subplots_adjust(
        left=0.06,
        right=0.89,
        bottom=0.14,
        top=0.925,
        wspace=0.10,
        hspace=0.20,
    )
    colorbar_ax = fig.add_axes([0.92, 0.19, 0.014, 0.65])
    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.set_label(
        diagnostic.score_label,
        fontsize=8,
        labelpad=7,
    )
    colorbar.ax.tick_params(labelsize=7)
    flag_fraction = (
        diagnostic.flagged_cross_samples / diagnostic.total_cross_samples
        if diagnostic.total_cross_samples
        else 0.0
    )
    fig.text(
        0.5,
        0.015,
        f"Shared {scale} scale; gray = too few valid samples. "
        "Each cross-baseline contributes to both endpoint antennas. "
        f"FLAG/FLAG_ROW masked {flag_fraction:.1%} of selected parallel-hand samples.",
        ha="center",
        fontsize=7,
    )

    try:
        fig.savefig(output_path, dpi=dpi, format="png", facecolor="white")
    finally:
        plt.close(fig)
    return float(vmin), float(vmax)


def create_aggregated_line_plot(
    diagnostic: ResidualDiagnostic,
    output_path: Path,
    ms_name: str,
    columns: Optional[int],
    dpi: int,
    x_axis: str,
    scale: str,
) -> tuple[float, float]:
    eligible = (
        diagnostic.usable_cells
        & np.isfinite(diagnostic.scores)
        & (diagnostic.scores > 0)
        & (diagnostic.sample_counts > 0)
    )
    selected_counts = np.where(eligible, diagnostic.sample_counts, 0)
    selected_scores = np.where(eligible, diagnostic.scores, 0.0)

    if x_axis == "time":
        reduction_axis = 1
        centers = (
            diagnostic.time_edges_minutes[:-1]
            + diagnostic.time_edges_minutes[1:]
        ) / 2
        x_label = "Elapsed time (min)"
        aggregation_label = "frequency-aggregated"
    elif x_axis == "frequency":
        reduction_axis = 0
        centers = (
            diagnostic.frequency_edges_mhz[:-1]
            + diagnostic.frequency_edges_mhz[1:]
        ) / 2
        x_label = "Frequency (MHz)"
        aggregation_label = "time-aggregated"
    else:
        raise ValueError(f"Unsupported line-plot x axis: {x_axis}")

    weighted_power = np.sum(
        selected_scores * selected_counts,
        axis=reduction_axis,
        dtype=np.float64,
    )
    sample_counts = np.sum(
        selected_counts,
        axis=reduction_axis,
        dtype=np.int64,
    )
    scores = np.full(weighted_power.shape, np.nan, dtype=np.float64)
    np.divide(weighted_power, sample_counts, out=scores, where=sample_counts > 0)
    plot_values = scores[np.isfinite(scores) & (scores > 0)]
    if plot_values.size == 0:
        raise ValueError(f"No usable values for the {x_axis} line plot")

    y_min = float(np.min(plot_values))
    y_max = float(np.max(plot_values))
    if y_max <= y_min:
        y_min /= math.sqrt(10.0)
        y_max *= math.sqrt(10.0)
    else:
        y_min /= 1.05
        y_max *= 1.05

    antenna_count = len(diagnostic.antenna_names)
    columns = resolve_plot_columns(antenna_count, columns)
    rows = math.ceil(antenna_count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 2.1 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for antenna_id, ax in enumerate(
        tqdm(
            axes.flat,
            total=axes.size,
            desc=f"Drawing {x_axis} panels",
            unit="panel",
        )
    ):
        if antenna_id >= antenna_count:
            ax.set_visible(False)
            continue
        antenna_scores = scores[:, antenna_id]
        if np.any(np.isfinite(antenna_scores)):
            ax.plot(
                centers,
                antenna_scores,
                color="tab:blue",
                linewidth=0.8,
            )
        else:
            ax.set_facecolor("#eeeeee")
            ax.text(
                0.5,
                0.5,
                "No usable bins",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#666666",
            )
        ax.set_title(
            str(diagnostic.antenna_names[antenna_id]),
            fontsize=8,
            loc="left",
            pad=2,
        )
        if scale == "sqrt":
            ax.set_yscale("function", functions=(np.sqrt, np.square))
        elif scale in ("log", "linear"):
            ax.set_yscale(scale)
        else:
            raise ValueError(f"Unsupported plot scale: {scale}")
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        ax.grid(True, which="major", alpha=0.2, linewidth=0.4)

    fig.suptitle(
        f"Parallel-hand residual power vs {x_label.lower()} "
        f"({aggregation_label})",
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.965,
        f"{ms_name} | Correlations: {', '.join(diagnostic.correlations)}",
        ha="center",
        fontsize=9,
    )
    fig.supxlabel(x_label, fontsize=9, y=0.075)
    fig.supylabel(
        diagnostic.score_label,
        fontsize=9,
        x=0.015,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        bottom=0.14,
        top=0.925,
        wspace=0.10,
        hspace=0.20,
    )
    fig.text(
        0.5,
        0.015,
        f"Common {scale} y-scale; usable cells pooled by valid sample count.",
        ha="center",
        fontsize=7,
    )

    try:
        fig.savefig(output_path, dpi=dpi, format="png", facecolor="white")
    finally:
        plt.close(fig)
    return y_min, y_max


def append_output_suffix(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_{suffix}{output_path.suffix}"
    ).resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.min_sample_fraction <= 1:
        parser.error("--min-sample-fraction must be between 0 and 1")
    if not (
        0 <= args.vmin_percentile < args.vmax_percentile <= 100
    ):
        parser.error("color percentiles must satisfy 0 <= vmin < vmax <= 100")

    ms_path = Path(args.ms).expanduser().resolve()
    if not ms_path.exists():
        parser.error(f"Measurement Set does not exist: {ms_path}")
    if not ms_path.is_dir():
        parser.error(f"Input path is not a Measurement Set directory: {ms_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = Path.cwd() / f"{ms_path.name}.residual_power.png"
    output_paths = {
        "time-frequency": output_path,
        "frequency": append_output_suffix(output_path, "freq"),
        "time": append_output_suffix(output_path, "time"),
    }
    for plot_name, plot_path in output_paths.items():
        try:
            plot_path.relative_to(ms_path)
        except ValueError:
            pass
        else:
            parser.error(
                f"{plot_name} PNG output must be outside the Measurement Set directory"
            )
        if plot_path.exists():
            if not plot_path.is_file():
                parser.error(f"Output path is not a file: {plot_path}")
            if not args.overwrite:
                parser.error(
                    f"Output already exists: {plot_path}; remove --no-overwrite "
                    "to replace it"
                )

    try:
        diagnostic = calculate_diagnostic(
            ms_path=ms_path,
            data_column=args.data_column,
            requested_weight_column=args.weight_column,
            time_bin_seconds=args.time_bin_seconds,
            channels_per_bin=args.channels_per_bin,
            min_samples=args.min_samples,
            min_sample_fraction=args.min_sample_fraction,
            chunk_rows=args.chunk_rows,
            ignore_weights=args.ignore_weights,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vmin, vmax = create_plot(
            diagnostic=diagnostic,
            output_path=output_paths["time-frequency"],
            ms_name=ms_path.name,
            colormap_name=args.cmap,
            scale=args.scale,
            columns=args.columns,
            dpi=args.dpi,
            vmin_percentile=args.vmin_percentile,
            vmax_percentile=args.vmax_percentile,
        )
        frequency_limits = create_aggregated_line_plot(
            diagnostic=diagnostic,
            output_path=output_paths["frequency"],
            ms_name=ms_path.name,
            columns=args.columns,
            dpi=args.dpi,
            x_axis="frequency",
            scale=args.scale,
        )
        time_limits = create_aggregated_line_plot(
            diagnostic=diagnostic,
            output_path=output_paths["time"],
            ms_name=ms_path.name,
            columns=args.columns,
            dpi=args.dpi,
            x_axis="time",
            scale=args.scale,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    flag_fraction = (
        diagnostic.flagged_cross_samples / diagnostic.total_cross_samples
        if diagnostic.total_cross_samples
        else 0.0
    )
    valid_fraction = (
        diagnostic.valid_cross_samples / diagnostic.total_cross_samples
        if diagnostic.total_cross_samples
        else 0.0
    )
    validity_description = (
        "valid unflagged residual samples"
        if diagnostic.weight_column is None
        else "valid data and weights"
    )
    print(f"Measurement Set: {ms_path}")
    print(f"Correlations included: {', '.join(diagnostic.correlations)}")
    if diagnostic.weight_column is None:
        print("Weights: ignored (unweighted residual power)")
    else:
        print(f"Weight column: {diagnostic.weight_column}")
    if args.time_bin_seconds is None:
        print(
            "Time bin: native integrations "
            f"(typical interval {diagnostic.integration_width_seconds:.3g} s)"
        )
    else:
        effective_bin_seconds = (
            diagnostic.time_samples_per_bin * diagnostic.integration_width_seconds
        )
        print(
            f"Time bin: requested {args.time_bin_seconds:g} s; "
            f"using {diagnostic.time_samples_per_bin} integration sample(s) "
            f"(approximately {effective_bin_seconds:.3g} s)"
        )
    print(f"Saved time-frequency plot: {output_paths['time-frequency']}")
    print(f"Saved frequency plot: {output_paths['frequency']}")
    print(
        f"Shared frequency-plot {args.scale} y range: "
        f"{frequency_limits[0]:.4g} to {frequency_limits[1]:.4g}"
    )
    print(f"Saved time plot: {output_paths['time']}")
    print(
        f"Shared time-plot {args.scale} y range: "
        f"{time_limits[0]:.4g} to {time_limits[1]:.4g}"
    )
    print(
        f"FLAG/FLAG_ROW masked {flag_fraction:.1%}; "
        f"{valid_fraction:.1%} of selected parallel-hand cross-correlation "
        f"samples were {validity_description}"
    )
    print(f"Shared {args.scale} color range: {vmin:.4g} to {vmax:.4g}")
    print_candidate_summary(diagnostic, args.high_ratio, args.top_antennas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())