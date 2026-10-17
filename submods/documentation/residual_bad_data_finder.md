# `residual_bad_data_finder.py`: find antennas, times and frequencies with excess residual power

`submods/residual_bad_data_finder.py` helps to answer the question: *which antenna, and when and at which frequencies, has bad data left in my residual visibilities?*

It reads one Measurement Set (MS) with a **residual data column** (data minus the model, for example `RESIDUAL_DATA`) and makes, for every antenna, a time-frequency map of the residual power, plus two line plots: residual power against time and residual power against frequency. It also prints a ranked list of antennas whose residual power is high compared with the other antennas.

The script is a **screening tool**. It opens the MS read-only, never flags or changes anything, and its numbers are not calibrated chi-squared values. Use it to decide where to look, and then decide yourself what to flag. It is not specific to one telescope: it works for any MS with XX/YY or RR/LL correlations (uGMRT, MeerKAT, LOFAR, VLA, ...).

Contents: [1. Quick start](#1-quick-start) · [2. What the script computes](#2-what-the-script-computes) · [3. Output](#3-output) · [4. How to read the plots](#4-how-to-read-the-plots) · [5. Options](#5-command-line-options) · [6. Implementation details](#6-implementation-details) · [7. Run time and memory](#7-run-time-and-memory) · [8. Limitations and troubleshooting](#8-limitations-and-troubleshooting)

---

## 1. Quick start

```bash
python submods/residual_bad_data_finder.py obs.ms
```

This uses the `RESIDUAL_DATA` column, weights from `WEIGHT_SPECTRUM` (or `WEIGHT`), the native time and frequency resolution, and writes three PNG files to the current directory:

- `obs.ms.residual_power.png`: per-antenna time-frequency maps,
- `obs.ms.residual_power_time.png`: per-antenna residual power against time (averaged over frequency),
- `obs.ms.residual_power_freq.png`: per-antenna residual power against frequency (averaged over time).

For a large MS, average in time and frequency to keep the plots readable and the memory use low:

```bash
python submods/residual_bad_data_finder.py obs.ms \
    --time-bin-seconds 60 --channels-per-bin 4 \
    --output plots/residual_power.png
```

The python environment needs `numpy`, `matplotlib`, `python-casacore` and `tqdm`.

Useful variants:

```bash
# another residual column, for example after subtracting a model yourself
... --data-column CORRECTED_DATA

# ignore the weights (useful if the weights are known to be wrong)
... --ignore-weights

# linear colour/y scale instead of log, and 8 antenna panels per row
... --scale linear --columns 8

# do not overwrite existing plots
... --no-overwrite
```

---

## 2. What the script computes

### The score per antenna, time bin and frequency bin

For every cross-correlation (autocorrelations are skipped) and every **parallel-hand** correlation (XX and YY, or RR and LL; cross-hands are ignored), the script takes the residual visibility `r` and its weight `w` and computes

```
w |r|^2        (default)
|r|^2          (with --ignore-weights)
```

A sample is used only if it is not flagged (`FLAG` and `FLAG_ROW`), if the real and imaginary parts are finite, and (when weights are used) if the weight is finite and positive.

The samples are summed per **antenna**, **time bin** and **frequency bin**. Every baseline contributes to **both** of its antennas. The score of a cell is

```
score = sum(w |r|^2) / number of valid complex samples
```

so it is the mean weighted residual power per valid sample. If the weights are inverse noise variances and the residuals are pure noise, the score is about 2 (one for the real and one for the imaginary part). In practice the weights are rarely calibrated exactly, so **compare scores between antennas and bins, not with an absolute value**.

### Time and frequency bins

- **Time.** Without `--time-bin-seconds` each unique `TIME` value (integration) is its own bin. With `--time-bin-seconds` the integrations are grouped in sets of `N` consecutive integrations, with `N` the requested bin divided by the integration time (the median `INTERVAL`), rounded to the nearest whole number (at least 1). The time axis of the plots is elapsed time in minutes since the start of the first integration.
- **Frequency.** The channels are sorted by frequency and grouped in sets of `--channels-per-bin` adjacent channels (the last group can be smaller).

### Which cells are plotted

A cell with only a few valid samples gives a noisy score. A cell is used (**usable**) only if its number of valid samples is at least

```
max(--min-samples, --min-sample-fraction x median sample count of the antennas in that time/frequency bin)
```

with defaults 15 and 0.2. The second condition removes antennas that are mostly flagged in that bin. Cells that are not usable are shown in gray and are left out of the colour scale, the line plots and the ranking.

### The relative score and the antenna ranking

For the ranking printed at the end, every usable cell is divided by the **median score of all usable antennas in the same time/frequency bin**. This relative score is 1 for a typical antenna. It removes effects that are common to all antennas (for example RFI in a channel, or a time range with a higher sky contribution), so the ranking highlights antennas that are worse than the rest of the array at the same time and frequency.

For each antenna the script reports:

| column | meaning |
|---|---|
| `median` | median relative score over all usable cells of the antenna (the ranking key) |
| `p90` | 90th percentile of the relative score: high if the antenna is bad in part of the data |
| `bins_above_ratio` | number of usable cells with a relative score above `--high-ratio` (default 2), out of all usable cells of the antenna |
| `median_samples` | median number of valid samples per usable cell |

The `--top-antennas` (default 10) antennas with the highest median are printed. The thresholds are descriptive only: nothing is flagged.

Note that the **plots show the absolute score**, not the relative score. A problem that affects all antennas (for example RFI) is therefore visible in the plots but not in the ranking.

---

## 3. Output

### Files

| file | content |
|---|---|
| `<output>` (default `<MS-name>.residual_power.png`) | one time-frequency panel per antenna, with a shared colour scale |
| `<output-stem>_time.png` | one panel per antenna: score against time, averaged over frequency |
| `<output-stem>_freq.png` | one panel per antenna: score against frequency, averaged over time |

The default output goes to the current directory. The plots must be written outside the MS directory. Existing plots are overwritten unless `--no-overwrite` is given.

### The time-frequency plot

One panel per antenna (in `ANTENNA` table order), time (minutes) on the x axis, frequency (MHz) on the y axis, and the score as colour. All panels share the same colour scale, set from the `--vmin-percentile` and `--vmax-percentile` (default 2 and 98) of all usable cells, with `--scale` log (default), sqrt or linear. Gray cells have too few valid samples (flagged, or not enough data). The footer gives the percentage of the selected parallel-hand samples that were flagged by `FLAG`/`FLAG_ROW`.

### The line plots

The usable cells of each antenna are combined over frequency (the `_time` plot) or over time (the `_freq` plot), with each cell weighted by its number of valid samples. So the value is again the mean weighted residual power per valid sample. All panels share the same y range (the minimum and maximum over all antennas, with a 5% margin) and the same `--scale`. An antenna without usable cells shows "No usable bins".

### Terminal output

At the end the script prints:

- the MS, the correlations used and the weight column (or that weights were ignored),
- the time bin used (the number of integrations and the resulting bin width in seconds),
- the paths of the three plots and their colour/y ranges,
- the flagged fraction and the fraction of valid samples,
- the antenna ranking (section 2).

Progress bars show the processing of the MS and the drawing of the plots.

---

## 4. How to read the plots

Because all panels share the same scale, look for panels and patterns that stand out from the others.

| What you see | Likely cause | What to do |
|---|---|---|
| One panel brighter than the rest over the whole time-frequency plane; high `median` in the ranking | A bad antenna (bad calibration solutions, wrong weights, hardware problem) | Check its calibration solutions; flag or reweight the antenna |
| A bright patch in one panel only | The antenna is bad during part of the time or part of the band | Flag that time/frequency range for the antenna; the ranking shows a high `p90` and many `bins_above_ratio` with a normal `median` |
| A horizontal stripe in all panels | RFI or a bad channel range, common to all antennas | Flag the channels (for all antennas) |
| A vertical stripe in all panels | A bad time range for the whole array (RFI burst, ionospheric event, poor calibration in that interval) | Flag or recalibrate that time range |
| A vertical stripe in one panel | The antenna is bad for a short time (for example a slew, a tracking problem or a dropout) | Flag that time range for the antenna |
| Several antennas higher than the rest without a clear pattern | Antennas with many short baselines (for example in the core of the array) pick up more unmodelled extended emission | Usually not bad data; compare with the line plots and check the model |
| A step in the `_time` plot | Something changed at that time (for example a calibration interval or a scan boundary) | Look at the calibration solutions around that time |
| A spike or a rising edge in the `_freq` plot | RFI, or a band edge with low sensitivity | Flag the channels if all antennas show it |

A single bad antenna also raises the score of the other antennas a little, because each of its baselines is counted for both endpoints. With `N` antennas the effect on each other antenna is about `1/(N-1)` of the excess of the bad antenna, so it is small for arrays of more than a few antennas.

---

## 5. Command-line options

Run `python submods/residual_bad_data_finder.py -h` for the full list with defaults.

**Input and output**

| option | default | meaning |
|---|---|---|
| `ms` | | Measurement Set with the residual column |
| `-o`, `--output` | `<MS-name>.residual_power.png` in the current directory | path of the time-frequency plot; the line plots get `_time` and `_freq` added to the name |
| `--data-column` | `RESIDUAL_DATA` | residual visibility column |
| `--weight-column` | `auto` | `WEIGHT_SPECTRUM`, `WEIGHT`, or `auto` (`WEIGHT_SPECTRUM` if present, otherwise `WEIGHT`) |
| `--ignore-weights` | off | use unweighted `|r|^2`; flags are still applied |
| `--overwrite` / `--no-overwrite` | overwrite | whether existing plots may be replaced |

**Binning and cell selection**

| option | default | meaning |
|---|---|---|
| `--time-bin-seconds` | native | time bin, rounded to a whole number of integrations |
| `--channels-per-bin` | 1 | number of adjacent channels per frequency bin |
| `--min-samples` | 15 | minimum number of valid complex samples in a usable cell |
| `--min-sample-fraction` | 0.2 | also require this fraction (0 to 1) of the median antenna sample count in that bin |

**Plotting**

| option | default | meaning |
|---|---|---|
| `--scale` | `log` | `log`, `sqrt` or `linear`, for the colours and the y axes of the line plots |
| `--vmin-percentile`, `--vmax-percentile` | 2, 98 | percentiles of the usable scores that set the shared colour range |
| `--cmap` | `magma` | matplotlib colour map |
| `--columns` | automatic (up to 10) | number of antenna panels per row |
| `--dpi` | 140 | resolution of the PNG files |

**Summary and performance**

| option | default | meaning |
|---|---|---|
| `--top-antennas` | 10 | number of antennas printed in the ranking |
| `--high-ratio` | 2.0 | relative score counted as high in `bins_above_ratio` |
| `--chunk-rows` | 50000 | number of MS rows read at a time |

---

## 6. Implementation details

The script is a single file. Function names are given in parentheses.

**Reading the MS (`calculate_diagnostic`).** The MS is opened read-only. The full `TIME`, `ANTENNA1`, `ANTENNA2` and `FLAG_ROW` columns are read first; the data, flags and weights are then read in chunks of `--chunk-rows` rows. The `DATA_DESCRIPTION`, `POLARIZATION`, `SPECTRAL_WINDOW` and `ANTENNA` subtables give the correlation types, channel frequencies and antenna names. The parallel hands are recognised from `CORR_TYPE` (5 = RR, 8 = LL, 9 = XX, 12 = YY). When `WEIGHT` is used (one weight per row and correlation) it is broadcast over the channels (`read_weight_chunk`).

**Binning.** `build_time_grid` makes the time bins from the unique `TIME` values; the bin edges lie halfway between neighbouring integrations. `axis_edges_from_centers` makes the frequency bin edges.

**Accumulation.** For every chunk and frequency bin the weighted power and the number of valid samples per row are summed over the channels of the bin and the parallel hands, and then added to the time bin and to **both** antennas of the baseline with `np.bincount`. Two arrays of shape `(time bins, frequency bins, antennas)` hold the power sums (float64) and sample counts (int64).

**Consistency checks.** After the pass the script checks that the total sample count equals twice the number of valid samples, and that the total power equals twice the total valid power (every sample is counted once for each endpoint antenna). A failure raises an `AssertionError`; this should not happen and points to a bug.

**Scores and ranking.** The scores, the usable-cell mask and the relative scores are computed per time/frequency bin (section 2). `print_candidate_summary` makes the ranking.

**Plots.** `create_plot` draws the time-frequency panels with `pcolormesh` (rasterised) and a shared normalisation (`color_normalization`); `create_aggregated_line_plot` makes the `_time` and `_freq` plots. `resolve_plot_columns` chooses the panel layout (about the square root of the number of antennas, at most 10 per row). Matplotlib runs with the non-interactive `Agg` backend, so no display is needed.

**Input checks (`main`).** The MS must exist and be a directory, the plots must be outside the MS, `--min-sample-fraction` must be between 0 and 1, and the colour percentiles must satisfy `0 <= vmin < vmax <= 100`. The MS must contain the data column, `FLAG`, `ANTENNA1`, `ANTENNA2`, `TIME` and `DATA_DESC_ID`, and at least one parallel-hand correlation.

---

## 7. Run time and memory

- **Run time** is dominated by reading the data column, flags and weights. In addition, the accumulation loops over the frequency bins for every chunk, so many frequency bins (for example thousands of native channels) make the run slower. `--channels-per-bin` reduces this.
- **Memory:** the accumulation arrays use 16 bytes per cell, `time bins x frequency bins x antennas`. For example 3000 integrations x 2048 channels x 30 antennas needs about 3 GB for these arrays alone, and the scores and relative scores need more of the same size. For long observations with many channels use `--time-bin-seconds` and `--channels-per-bin`. A chunk of data, flags and weights is also in memory (lower `--chunk-rows` if needed).
- The figure size grows with the number of antennas (about 2.5 x 2.1 inch per panel). For arrays with many stations (for example LOFAR) the plots become large; use `--columns` and `--dpi` to adjust.

---

## 8. Limitations and troubleshooting

**Limitations**

- Only MS files with a **single `DATA_DESC_ID` / spectral window** are supported. Split or combine spectral windows first.
- Only the parallel hands are used; cross-hand problems (leakage, XY phase) are not seen.
- The score depends on the scale of the weights. Compare antennas and bins with each other, and use `--ignore-weights` if the weights are unreliable.
- The residual contains more than bad data: unmodelled sky emission (mostly on short baselines), direction-dependent calibration errors and noise. Antennas with many short baselines can score higher without having bad data.
- The time axis uses integration index, not wall-clock time: a gap in the observation (for example between scans) is not shown as an empty column, but the cells next to it are stretched to the middle of the gap. With `--time-bin-seconds` a bin can include integrations on both sides of a gap.
- The ranking thresholds (`--high-ratio`, the 90th percentile) are heuristics. Always look at the plots.

**Troubleshooting**

| message / symptom | likely cause / remedy |
|---|---|
| `Missing required MS columns: RESIDUAL_DATA` | The MS has no residual column. Make one (for example data minus model), or give another column with `--data-column`. |
| `MS has neither WEIGHT_SPECTRUM nor WEIGHT` | Use `--ignore-weights`. |
| `This plot currently requires a single DATA_DESC_ID / spectral window` | Split the MS per spectral window (or combine them), and run the script per MS. |
| `POLARIZATION metadata has no XX, YY, RR, or LL parallel hands` | The MS has only cross-hands or Stokes parameters; not supported. |
| `No antenna/time/frequency cells have enough valid samples to plot` | Almost everything is flagged, or the bins are too small. Increase `--time-bin-seconds` / `--channels-per-bin`, or lower `--min-samples` / `--min-sample-fraction`. |
| Many gray cells | As above: the cells have too few valid samples. |
| The colour scale is dominated by a few very bright cells | Lower `--vmax-percentile`, or use `--scale log` (the default). |
| `Output already exists` | Remove `--no-overwrite`, or give another `--output`. |
| The run uses too much memory | Use `--time-bin-seconds` and `--channels-per-bin` (section 7), and/or lower `--chunk-rows`. |
