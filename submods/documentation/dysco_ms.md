# `dysco_ms.py`: Dysco compression for any Measurement Set

`submods/dysco_ms.py` is a single-file Python tool that compresses and decompresses radio-interferometric
Measurement Sets (MS) with the [Dysco](https://arxiv.org/abs/1609.02019) lossy compression
storage manager. It works on single- and multi-field, single- and multi-SPW MSs with 1, 2 or 4
correlations: LOFAR, MeerKAT, (E)VLA, uGMRT, and others. It can also choose the Dysco bitrate
automatically, based on the noise and signal in your visibilities.

```bash
python submods/dysco_ms.py compress my.ms                        # -> my.ms.dysco
python submods/dysco_ms.py compress my.ms --databitrate auto     # bitrate chosen from the data
python submods/dysco_ms.py decompress my.ms.dysco                # -> my.ms.dysco.nodysco
```

---

## Contents

1. [Why this tool?](#1-why-this-tool)
2. [Requirements and installation](#2-requirements-and-installation)
3. [Quick start](#3-quick-start)
4. [Sub-commands and options](#4-sub-commands-and-options)
5. [Automatic bitrate selection](#5-automatic-bitrate-selection)
6. [What happens to your data](#6-what-happens-to-your-data)
7. [Compatibility with other software](#7-compatibility-with-other-software)
8. [Design and implementation](#8-design-and-implementation)
9. [Storage-manager pitfalls the tool guards against](#9-storage-manager-pitfalls-the-tool-guards-against)
10. [Test results](#10-test-results)
11. [Limitations and troubleshooting](#11-limitations-and-troubleshooting)
12. [References](#12-references)

---

## 1. Why this tool?

Visibility data take up a lot of disk space. Uncompressed, every complex visibility costs 8 bytes
in each of DATA, CORRECTED_DATA and MODEL_DATA, and WEIGHT_SPECTRUM adds 4 bytes per value. Dysco
(Offringa 2016) quantises visibilities to a few bits after a smart normalisation. At its
default 10 bits per float it shrinks the DATA column about 3.2×, while the error it adds is
typically only a few percent of the thermal noise. That makes the extra noise negligible in images.

The usual way to Dysco-compress an MS is DP3
(`msout.storagemanager=dysco`). That works well for LOFAR-like data, but:

* **DP3 handles only one SPW and one field at a time.** Multi-SPW (E)VLA data and multi-field
  MeerKAT/uGMRT/VLA data cannot be compressed in one go.
* **DP3 rewrites the MS.** It writes its own set of columns, so extra columns, subtables or keywords
  in your MS may not survive in the form you expect.
* **Choosing the bitrate is guesswork.** Ten bits is a safe default, but
  noise-dominated data could often use fewer bits, while very bright sources or data with a large
  dynamic range may need more.

`dysco_ms.py` addresses these points:

* It changes **only the storage managers** of the selected columns. Every other column, keyword and
  subtable is copied unchanged, and the row order is kept unless you ask otherwise.
* It works on **any MS layout**: multi-SPW, multi-field, and 1, 2 or 4 correlations. If SPWs differ in
  shape it can split the MS per shape.
* It exposes **all DP3 Dysco settings**, with DP3's defaults, plus DP3's lossless metadata
  compression options (scalar flags, antenna and UVW compression).
* It can choose the bitrate automatically (`--databitrate auto --weightbitrate auto`) by trial
  compression of samples of your data, against a noise-increase criterion you set.
* It **guards against storage-manager pitfalls**. Several of these would otherwise crash or
  silently corrupt data; see [Section 9](#9-storage-manager-pitfalls-the-tool-guards-against).
* It **verifies** every output against the input. Your input is replaced (`--replace`) only
  after verification passes.

---

## 2. Requirements and installation

* **Python ≥ 3.9** and **numpy**.
* **python-casacore** linked against **casacore ≥ 3.7**, which has Dysco built in (tested with
  casacore 3.8.1 and python-casacore 3.6.1). The extra storage managers `StokesIStMan`,
  `AntennaPairStMan` and `UvwStMan` are also part of these casacore versions.

The tool is part of facetselfcal and needs nothing beyond these. It is a single file that does not
import anything from the rest of the package, so `submods/dysco_ms.py` can also be copied and run
on its own. When facetselfcal is installed with pip, the tool is also available as the command
`dysco_ms` (for example `dysco_ms compress my.ms`). With the LOFAR/FLoCs Singularity container
(which has everything):

```bash
singularity exec -B /net flocs.sif python submods/dysco_ms.py compress my.ms
```

Run it on a machine whose CPU matches the container's build flags. Containers built with
`-march=cascadelake`, for example, warn on older CPUs and may crash with "illegal instruction".

---

## 3. Quick start

```bash
# Compress with DP3 defaults (DATA, CORRECTED_DATA, WEIGHT_SPECTRUM at 10/12 bits)
python submods/dysco_ms.py compress my.ms                        # -> my.ms.dysco

# Choose the output name yourself
python submods/dysco_ms.py compress my.ms --msout my_dysco.ms    # -> my_dysco.ms

# Let the tool choose the bitrates (prints a per-field/SPW report)
python submods/dysco_ms.py compress my.ms --databitrate auto --weightbitrate auto   # -> my.ms.dysco

# Only look at what the data need, without writing anything
python submods/dysco_ms.py analyse my.ms                         # no output MS

# Base the automatic choice on the target field only (ignore bright calibrators)
python submods/dysco_ms.py compress my.ms --databitrate auto --auto-fields MyTarget # -> my.ms.dysco

# Replace the input MS by the compressed one (only after verification passes)
python submods/dysco_ms.py compress my.ms --replace              # -> my.ms (now compressed)

# Multi-SPW MS whose SPWs have different numbers of channels: one output MS per shape
python submods/dysco_ms.py compress evla.ms --split-by-shape     # -> evla.ms.shape0_<nchan>ch<ncorr>pol.dysco,
                                                          #    evla.ms.shape1_<nchan>ch<ncorr>pol.dysco, ...

# Undo everything: all compressed columns back to standard storage managers
python submods/dysco_ms.py decompress my.ms.dysco                # -> my.ms.dysco.nodysco
python submods/dysco_ms.py decompress my.ms.dysco --msout my_uncompressed.ms        # -> my_uncompressed.ms

# Inspect storage managers, Dysco settings and per-column sizes
python submods/dysco_ms.py info my.ms --layout                   # no output MS
```

Output names in short:

* `compress` writes `<input>.dysco` and `decompress` writes `<input>.nodysco`, unless you give
  `--msout`.
* The input is never modified, except with `--replace`. That option first writes `<input>.dysco`
  (or `--msout`), verifies it, then moves it to the input's name and deletes the original.
* An existing output is not overwritten unless you add `--overwrite`.
* With `--split-by-shape`, the shape tag `.shape<i>_<nchan>ch<ncorr>pol` is inserted before the
  last extension of the output name. For example, `--msout evla_dysco.ms` gives
  `evla_dysco.shape0_64ch4pol.ms`, `evla_dysco.shape1_128ch4pol.ms`, ...
* The last lines of the log print the output path (`Done: <output>`).

Typical output ends with a summary like:

```
[  54.5s]   Dysco data: worst sampled typical eps/sigma = 0.1002 (noise increase 0.501%), ...
[  54.5s] Size: input 13.6 GB -> output 7.3 GB (53.8%)
[  54.5s] Verification passed
[  54.5s] Settings used: databitrate=6 (auto), weightbitrate=12 (auto), distribution=TruncatedGaussian,
          disttruncation=2.5, normalization=AF, scalarflags=True, antennacompression=True, uvwcompression=False
```

---

## 4. Sub-commands and options

### 4.1 `compress MS`

Writes a new MS (default name `<MS>.dysco`) in which the selected columns are Dysco-compressed.

**Dysco settings.** Names and defaults follow DP3's `msout.storagemanager.*`.

| Option | Default | Meaning |
|---|---|---|
| `--databitrate N\|auto` | `10` | Bits per float for visibility columns: 2, 3, 4, 6, 8, 10, 12 or 16. `0` = do not compress visibilities. `auto`: see Section 5. |
| `--weightbitrate N\|auto` | `12` | Bits per value for WEIGHT_SPECTRUM (same allowed values). `0` = do not compress weights. |
| `--distribution` | `TruncatedGaussian` | `Uniform`, `Gaussian`, `TruncatedGaussian` or `StudentT` (`StudentsT` is accepted too). |
| `--disttruncation` | `2.5` | Truncation level of the TruncatedGaussian distribution. |
| `--normalization` | `AF` | `AF`, `RF` or `Row`. AF is switched to RF automatically if AF is unsafe for the MS layout (Section 9). If you set AF explicitly, the tool refuses instead. |
| `--studenttnu` | `0` | ν for the StudentT distribution. |
| `--columns` | `DATA,CORRECTED_DATA,WEIGHT_SPECTRUM` | Columns to compress. Any complex column can be listed (e.g. MODEL_DATA). Of the float columns, only WEIGHT_SPECTRUM is supported. Missing columns are skipped. |

**Lossless metadata compression.** These are the DP3 `msout.*` options of the same names.

| Option | Default | Meaning |
|---|---|---|
| `--scalarflags / --no-scalarflags` | on | Store one flag per channel for all 4 correlations (`StokesIStMan`). Only used with 4 correlations and identical flags in every correlation (see Section 6). |
| `--antennacompression / --no-…` | off | Store ANTENNA1/ANTENNA2 as one repeating baseline pattern (`AntennaPairStMan`). |
| `--uvwcompression / --no-…` | off | Store UVW per antenna instead of per baseline (`UvwStMan`). |

Each of these is switched off automatically, with a note, when the MS does not meet its
requirements.

**Data handling.**

| Option | Default | Meaning |
|---|---|---|
| `--merge-corr-flags` | off | With scalar flags: if flags differ between correlations, OR them over the correlations instead of switching scalar flags off. |
| `--keep-flagged-values` | off | Do not set flagged visibilities to NaN or fully flagged weights to 0 before compressing. |
| `--reorder` | off | Write rows ordered by TIME, DATA_DESC_ID, ANTENNA1, ANTENNA2. Only needed when rows are interleaved so badly that Dysco blocks would be tiny (the tool tells you). |
| `--no-pad` | off | Refuse instead of appending flagged padding rows (Section 6.3). |
| `--split-by-shape` | off | If SPWs have different (nchan, ncorr), write one MS per shape: `<msout-root>.shape<i>_<nchan>ch<ncorr>pol<ext>`. |

**Automatic bitrate.** See Section 5.

| Option | Default | Meaning |
|---|---|---|
| `--noise-tolerance` | `0.01` | Allowed fractional increase of the noise (1%). |
| `--weight-tolerance` | `0.005` | Allowed rms relative error of the weights. |
| `--auto-fields` | all | Comma-separated field names or ids that determine the bitrate. |
| `--auto-group-quantile` | `1.0` | Fraction of field/SPW groups that must meet the tolerance. `1.0` = all; `0.95` ignores the worst 5%. |
| `--auto-windows` | `3` | Sample windows per field/SPW group. |
| `--auto-sample-mb` | `2000` | Maximum amount of data read for the analysis. |
| `--tmpdir` | next to output | Where trial compressions are written (deleted afterwards). |

**Output and safety.**

| Option | Default | Meaning |
|---|---|---|
| `--msout` | `<MS>.dysco` | Output MS. Must differ from the input. |
| `--overwrite` | off | Overwrite an existing `--msout`. |
| `--replace` | off | After successful verification, replace the input MS by the output. |
| `--verify sample\|full\|none` | `sample` | Compare output and input on sampled rows, on all rows, or not at all. |
| `--uvw-tolerance` | `1e-4` m | Maximum UVW reconstruction error allowed for `--uvwcompression`. |
| `--chunk-mb` | `1024` | Approximate memory per column chunk while copying. |
| `--force` | off | Ignore the free-disk-space check. |
| `--dry-run` | off | Inspect and plan only; also runs the bitrate analysis if `auto` is requested. |

### 4.2 `decompress MS`

Writes `<MS>.nodysco` (or `--msout`) in which every column on `DyscoStMan`, `StokesIStMan`,
`UvwStMan` or `AntennaPairStMan` is moved back to standard storage managers:

* visibilities, weights and flags go to `TiledColumnStMan` (about 1 MB tiles, as DP3 uses);
* UVW goes to `TiledColumnStMan`;
* ANTENNA1/2 go to `StandardStMan`.

Padding rows added by `compress` are removed. Takes `--msout`, `--replace`, `--overwrite`,
`--verify`, `--chunk-mb`, `--force` and `--dry-run`.

Decompressing does not restore the precision lost by Dysco. It only makes the MS readable by
software without Dysco support, and faster to access.

### 4.3 `analyse MS`

Runs the automatic bitrate analysis (Section 5) and prints the report without writing an MS.
`--all-bits` evaluates every bit count instead of stopping at the first acceptable one.

### 4.4 `info MS`

Prints rows, antennas, fields, SPWs, correlations, the size on disk, and every storage manager
with its columns, on-disk size and Dysco settings. `--layout` also reads the row metadata and
reports:

* the Dysco block size;
* the number of padding rows that would be needed;
* the shape groups;
* the baseline pattern period.

### 4.5 Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Verification failed (output kept, input untouched) or nothing to analyse. |
| `2` | Refused: an option or layout problem, explained in the `ERROR:` line. |

---

## 5. Automatic bitrate selection

### 5.1 Criterion

Dysco adds quantisation error ε to every visibility. The thermal noise σ then effectively grows
to √(σ² + ε²). With `--noise-tolerance t`, the tool picks the **smallest bitrate** for which

    √(1 + (ε/σ)²) − 1 ≤ t        i.e. ε/σ ≤ √((1+t)² − 1)

For the default t = 1% this means ε/σ ≤ 0.142. The condition must hold in every
(column, field, SPW) group, or in the fraction set by `--auto-group-quantile`. Only one
databitrate is used for all compressed visibility columns, as in DP3.

For the weights, the smallest bitrate is chosen whose rms relative weight error (compared with
what Dysco can store at best, see Section 6.1) is at most `--weight-tolerance`.

### 5.2 Method

1. **Sampling.** For every (FIELD_ID, DATA_DESC_ID) group, `--auto-windows` windows of two Dysco
   blocks are taken, spread over the observation and aligned to the block boundaries Dysco will
   use. The total is capped by `--auto-sample-mb`.
2. **Noise.** σ is estimated per window as 1.4826 × MAD / √2 of the differences between adjacent
   channels and between adjacent time steps of the same baseline. The smaller of the two is used.
   Differencing removes smooth signal, and the MAD (median absolute deviation) ignores outliers.
   The rms of what remains above the noise is reported as S/N per visibility.
3. **Trial compression.** The samples are written to a small temporary table with Dysco at each
   bit count, using your distribution/normalization settings and the same value conditioning as
   the real run. The table is built so that Dysco uses exactly the block layout and antenna count
   of the real output: the real first block comes first, and fake TIME values reproduce the block
   size. The data are then read back.
4. **Error.** ε is the *typical* error: 1.4826 × MAD of the error per real component. A robust
   measure is needed because strong unflagged outliers (RFI, band edges) are clipped by Dysco and
   would dominate an rms. The rms error and the fraction of visibilities with error > 10σ are
   reported as well. If that fraction exceeds 10⁻⁴, the tool advises flagging RFI first.
5. **Choice.** Per group, the median over its windows is used. Bit counts are tried in increasing
   order until the criterion holds.

### 5.3 The report

The bitrate is printed to the terminal in three places:

* the analysis table (one row per column/field/SPW, with σ, S/N and ε/σ per bit count, plus the
  bits that group needs);
* an `AUTO: databitrate = N bits (...)` line;
* the final `Settings used: databitrate=N (auto), ...` line.

```
column           field           spw      sigma S/N|polloss       2b       3b       4b       6b       8b      10b  need
DATA             Sausage          42  1.409e-02          62  65.6830  21.0963   8.3821   1.7303   0.4048   0.0997  10
DATA             Sausage           7  1.740e-02        6.99   7.7515   2.5019   0.9918   0.2026   0.0481   0.0117  8
...
Recommended databitrate  : 10
[    9.1s] AUTO: databitrate = 10 bits (lowest bitrate with <= 1% noise increase in every field/SPW)
```

`need` is `n/a` for groups without unflagged data. It is `>N` for groups that were not
evaluated beyond N bits (the search stops at the first acceptable bit count).

### 5.4 What to expect

Dysco's error scales with the data amplitude within a block, not with the noise. So the needed
bitrate depends on the **signal-to-noise per visibility**:

* noise-dominated target fields typically need **6–8 bits**;
* bright calibrators, very bright targets (e.g. Hydra A), RFI-affected SPWs and uncalibrated or
  unflagged data need **10–12 bits**.

The strict default (every group must pass) therefore often lands at 10 bits, DP3's default.
To gain more on the science data:

* `--auto-fields <target>` ignores calibrator fields. The calibrator data are then stored at the
  same, lower bitrate and get more extra noise; check the verification output.
* `--auto-group-quantile 0.95` ignores the worst 5% of field/SPW groups, e.g. a few RFI-dominated
  SPWs.

The DATA column scales simply as 10 / bits relative to the default: 1.25× smaller at 8 bits,
1.67× at 6 bits.

---

## 6. What happens to your data

### 6.1 Value conditioning (compressed columns only)

* **Flagged visibilities are set to NaN** before compression, as DP3 does. Dysco excludes NaNs from
  its normalisation, so flagged RFI no longer inflates the quantisation step of good data. This is
  **irreversible**: unflagging later gives NaN. Use `--keep-flagged-values` to keep the values.
* **Weights of channels in which all correlations are flagged are set to 0.** DP3 zeroes the weight
  of every flagged correlation. But Dysco stores only one weight per channel (the minimum over
  the correlations), so zeroing one flagged correlation would zero the weight of the unflagged
  ones too. The tool therefore zeroes a weight only when all correlations are flagged.
* **Negative or non-finite weights are set to 0** (and counted), because Dysco cannot store them.
* **Dysco weights are the minimum over correlations.** WEIGHT_SPECTRUM is stored as one value per
  channel and returned for all correlations. The analysis reports this loss as `polloss`.

### 6.2 Flags with `--scalarflags`

`StokesIStMan` stores one flag per channel. It only accepts flags that are identical in all four
correlations. Before writing, the tool checks a sample of the flags:

* If the flags differ between correlations, scalar flags are **switched off** and the flags are
  stored exactly. This is common for CASA-flagged EVLA data and for MeerKAT (0.2–0.7% of the
  channels in our tests).
* With `--merge-corr-flags`, flags are instead **OR-ed over the correlations**. A visibility is
  flagged if any correlation is flagged. The number of extra flags is reported.
* If differences show up later during writing (missed by the sample), they are OR-ed and reported.

With 1 or 2 correlations (e.g. uGMRT RR/LL), scalar flags are always switched off.

### 6.3 Padding rows

Dysco stores data in blocks whose size is fixed by the first change of TIME, FIELD_ID or
DATA_DESC_ID in the MS. If the number of rows is not a multiple of that block size, casacore
cannot read the last block back (Section 9). The tool then appends up to one block of
**fully flagged copies of the last row**:

* FLAG and FLAG_ROW are true;
* WEIGHT and WEIGHT_SPECTRUM are 0;
* visibilities are NaN.

Their number is stored in the main-table keyword `DYSCO_PADDING_ROWS`. `decompress` removes them
again, and `compress` ignores them when re-compressing. DP3 does something similar when it fills
missing baselines with flagged data.

Padding is typically needed for (E)VLA data, where the number of baselines per time step varies. Our EVLA
test had 150 padding rows on 1.1 million. Use `--no-pad` to refuse instead.

### 6.4 Columns that are not selected

* Columns not listed in `--columns` keep their storage manager and values.
* Columns that are **already Dysco-compressed** but not selected (e.g. MODEL_DATA in some LOFAR
  MSs) are rewritten with their existing Dysco settings. Re-encoding adds a little quantisation
  noise, and the tool warns about it.
* Re-compressing an already compressed column also adds a second round of quantisation noise.

---

## 7. Compatibility with other software

| Software | Dysco columns | Scalar flags / antenna / UVW compression |
|---|---|---|
| DP3, WSClean (built against casacore ≥ 3.7, e.g. in recent LOFAR containers) | ✔ | ✔ |
| CASA 6.7 (casatools) | ✔ (tested: `listobs`, `flagdata`) | ✘ — "Shared library stokesistman not found" |
| Older casacore / python-casacore without Dysco | ✘ | ✘ |

`--scalarflags` is on by default, matching DP3. **If the MS must stay readable by CASA**, use
`--no-scalarflags` and leave antenna/UVW compression off, or run `decompress` before using CASA.
The tool prints a note whenever these storage managers end up in the output.

---

## 8. Design and implementation

### 8.1 Why python-casacore and not DP3

Casacore ≥ 3.7 contains Dysco and the other compressing storage managers, so they can be used
directly through python-casacore by declaring them in the data-manager info of a column. This
works for any MS layout, preserves everything else in the MS, and needs no DP3 parset. The
semantics of DP3's MSWriter are reproduced:

* the Dysco SPEC record `{dataBitCount, weightBitCount, distribution, normalization,
  distributionTruncation, studentTNu}`;
* a separate Dysco instance per column;
* flagged data set to NaN;
* `StokesIStMan` for FLAG, `UvwStMan` for UVW and `AntennaPairStMan` for ANTENNA1/2.

### 8.2 Processing steps

1. **Inspect.** The tool reads the subtables (SPW channels, polarizations, data descriptions,
   fields, antennas), the column descriptions and storage managers, and the row metadata TIME,
   FIELD_ID, DATA_DESC_ID, ANTENNA1 and ANTENNA2. For 100 M rows the metadata take about 2.4 GB.
   With `--reorder` or `--split-by-shape` the rows are taken from a TaQL reference table.
2. **Plan and check the layout.** From the metadata alone the tool works out:
   * Dysco's block size, the number of blocks and the padding needed;
   * whether AF normalization is safe;
   * one shape per compressed column;
   * the flag pre-scan for scalar flags;
   * the baseline pattern for antenna compression;
   * for UVW compression, an exact simulation of UvwStMan's derivation of per-antenna UVWs and the
     resulting reconstruction error.

   Options that cannot be honoured are switched off with a printed reason.
3. **Automatic bitrate**, if requested (Section 5).
4. **Disk-space check.** An estimate of the output size is compared with the free space.
5. **Build the output skeleton.**
   * Deep-copy the input without rows. Columns on special managers are temporarily mapped to
     `StandardStMan`, because an empty Dysco column cannot be copied.
   * Copy the subtables again with their rows, using a fast file-level table copy (casacore's
     no-row copy empties subtables as well).
   * Remove the target columns and add them back, one storage manager per call, with:
     * fixed-shape descriptions (Dysco only supports fixed-shape, "direct" columns);
     * the new managers;
     * managers involving ANTENNA1/2 first, because Dysco and UvwStMan look these columns up when
       they are created.
6. **Copy the rows** in chunks (`--chunk-mb`, a whole number of Dysco blocks), always in ascending
   row order and per chunk in this column order:
   1. ANTENNA1/2, written row by row until the AntennaPairStMan pattern is known;
   2. TIME, FIELD_ID, DATA_DESC_ID;
   3. all other columns;
   4. UVW and FLAG;
   5. the Dysco columns.

   Dysco and UvwStMan read the antenna, time, field and SPW of a row when that row is written, so
   these must already be present. Columns whose cells are all undefined (e.g. FLAG_CATEGORY) are
   skipped. Columns with partly undefined or varying-shape cells are copied cell by cell.
7. **Pad** the last Dysco block if needed (Section 6.3).
8. **Verify.**
   * Row count, column set, keywords and the row counts of all subtables.
   * Exact equality of every losslessly stored column (NaN-aware, accounting for merged flags).
   * The same finite/NaN pattern in the Dysco columns, and the measured ε/σ and weight error.
   * A full check of UVW (within `--uvw-tolerance`) and ANTENNA1/2 when those are compressed.
   * That the padding rows are flagged.

   `--verify sample` (default) checks up to 64 windows (two per field/SPW group) plus the last block;
   `--verify full` checks every row.
9. **Replace** (only with `--replace` and a passed verification). The input is renamed, the
   output moved in its place, and the original deleted.

If anything fails while writing, the partial output is closed and removed. The input is never
modified, except by a verified `--replace`.

### 8.3 Performance

Copying runs at roughly 50–150 MB/s of input on the test machines, limited by I/O. Dysco encoding
is multi-threaded inside casacore. A 65 GB MeerKAT MS read over NFS took 14 minutes; the bitrate
analysis adds about a minute. Subtables are copied at file level, so large subtables such as an
EVLA SYSPOWER table (2.6 GB) cost seconds, not minutes.

---

## 9. Storage-manager pitfalls the tool guards against

These behaviours of casacore's storage managers were found by reading the source and confirmed
by experiment with casacore 3.8.1. Several of them **crash or silently corrupt memory**. Any
script that writes these managers should guard against them.

| Storage manager | Pitfall | What `dysco_ms.py` does |
|---|---|---|
| DyscoStMan | Only fixed-shape columns: all rows must have the same (nchan, ncorr). | Checks the shapes per DATA_DESC_ID; refuses, or splits with `--split-by-shape`. |
| DyscoStMan | The block size is set once, from the first change of TIME/FIELD_ID/DATA_DESC_ID; later blocks are simply `row // blocksize` (they may straddle time steps). | Simulates the block size. Refuses tiny blocks (rows interleaved per baseline) and suggests `--reorder`. |
| DyscoStMan | **Decoding a block reads ANTENNA1/2 for a full block**, so if nrow is not a multiple of the block size the last block cannot be read ("access to non-existing row"). | Appends flagged padding rows (Section 6.3). |
| DyscoStMan | With **AF normalization**, per-antenna arrays are sized from the first block. A higher antenna index later on **corrupts the heap** (`malloc(): corrupted top size`). | Switches to RF normalization (or refuses if AF was requested explicitly). |
| DyscoStMan | An MS consisting of a single TIME/FIELD/DDID block cannot be stored (abort on close). | Skips Dysco with a warning. |
| DyscoStMan | Only bit counts 2, 3, 4, 6, 8, 10, 12, 16 work. Others terminate the process from a worker thread. | Validates bitrates before writing. |
| DyscoStMan | Writes `StudentsT` in its spec but only reads `StudentT`. | Accepts both and writes `StudentT`. |
| DyscoStMan | Weights: one value per channel (the minimum over correlations); negative or NaN weights give garbage. | Zeroes bad weights; zeroes flagged weights only when all correlations are flagged. |
| casacore copy | A deep copy without rows fails for Dysco columns and also empties all subtables. | Maps special managers to StandardStMan for the skeleton copy, then copies the subtables with rows. |
| python-casacore | `addcols` accepts only one data manager per call. | Adds the columns one manager at a time, antenna managers first. |
| StokesIStMan | Hard-codes 4 correlations: with 1 or 2 it reads and writes beyond its buffers (**silent corruption**). It throws if the correlations are not flagged identically. | Used only with 4 correlations; checks the flags first (Section 6.2). |
| AntennaPairStMan | Needs one strictly repeating baseline pattern for the whole MS. ANTENNA1 and ANTENNA2 must be written interleaved until the pattern is complete. | Checks periodicity; writes the first pattern row by row. |
| UvwStMan | Derives one UVW per antenna from the baselines in order; throws "Inconsistent UVW value" if UVWs are not exact antenna differences. Needs nrow to be a whole number of blocks; autocorrelations read back as 0. | Simulates the derivation and checks the reconstruction error on all rows before enabling it; disables it when padding is needed; verifies UVW afterwards. |

**Where non-antenna-consistent UVWs come from.** In practice we found two sources:

* the DP3 StationAdder superstation ST001, whose UVWs are a flag-weighted average of its stations'
  baselines;
* some uGMRT data sets.

DP3's own averaging keeps UVWs consistent. Note that DP3 6.6 enables `msout.uvwcompression` by
default and aborts (`std::terminate`) when writing such MSs. Pass `msout.uvwcompression=false`
to DP3 for them; facetselfcal already does.

---

## 10. Test results

Subsets of real observations, compressed with default settings. Sizes are of the whole MS,
including uncompressed columns.

| Data set | Layout | Notes | Size |
|---|---|---|---|
| LOFAR LBA 3C380 | 1 field, 1 SPW, 4 pol | auto: 6 bits; scalar flags + antenna compression; UVW compression refused (ST001) | 13.6 → 7.3 GB (auto) |
| EVLA, 1 field | 63 SPWs × 64 ch | scalar flags off (per-correlation flags differ); antenna + UVW compression lossless | 2.5 GB → 0.90 GB |
| EVLA, 3 fields | 65 SPWs × 64 ch | 150 padding rows; antenna/UVW compression refused (varying baselines) | 9.0 → 6.1 GB (MODEL_DATA uncompressed) |
| MeerKAT, 3 fields | 1 SPW × 1861 ch | UVW compression lossless (max error 9×10⁻¹³ m) | 5.0 → 2.2 GB |
| MeerKAT, full MS | 575k rows | auto: 10 bits (Hydra A is very bright) | 64.9 → 28.5 GB in 14 min |
| uGMRT, 4 fields | 1 SPW × 2048 ch, RR/LL | scalar flags off (2 pol); unflagged RFI reported | 2.9 GB → 0.89 GB |
| uGMRT, 1 field | 1 SPW × 292 ch | auto: 8 bits | 101 → 21 MB (auto) |
| Mixed shapes | 2 SPWs: 292 and 146 ch | `--split-by-shape` → 2 MSs | ✔ |

All compressions and the corresponding decompressions passed verification; decompressions
reproduced every losslessly stored column exactly. A WSClean dirty image made from the 6-bit
LOFAR data differs from the uncompressed image by 3% of the image noise rms. DP3 reads the
compressed outputs, including scalar flags and antenna compression.

**DATA column alone, relative to the default (10 bits):**

| Data set | Uncompressed | Default | auto (strict) | auto (relaxed¹) |
|---|---|---|---|---|
| uGMRT, 1 field | 69 MB | 21.6 MB | 8 b: 1.25× smaller | 8 b: 1.25× |
| EVLA, 1 field | 2594 MB | 817 MB | 10 b: 1.00× | 8 b: 1.25× |
| EVLA, 3 fields | 2267 MB | 713 MB | 10 b: 1.00× | 10 b: 1.00× |
| MeerKAT, 3 fields | 2637 MB | 824 MB | 10 b: 1.00× | 10 b: 1.00× |
| uGMRT, 4 fields | 2017 MB | 617 MB | 12 b: 0.83× (larger) | 8 b: 1.25× |

¹ `--auto-group-quantile 0.95` for single-field data, `--auto-fields <target>` for multi-field data.

---

## 11. Limitations and troubleshooting

* **"different shapes … Use --split-by-shape".** Dysco cannot store SPWs with different numbers of
  channels or correlations in one column. Split per shape, or select SPWs beforehand.
* **"Dysco block size would be only N rows".** The rows are interleaved (e.g. sorted by baseline,
  then SPW). Use `--reorder`. Reading through the sorted reference table is slower.
* **Auto chooses 12–16 bits.** Usually bright calibrators, very bright sources, unflagged RFI or
  uncalibrated data. Check the per-group table. Consider `--auto-fields`,
  `--auto-group-quantile`, or flagging RFI first.
* **"errors > 10 sigma".** Strong unflagged outliers are clipped by Dysco. Flag RFI (e.g.
  AOFlagger) before compressing.
* **Noise estimate.** σ comes from channel and time differences. If adjacent channels are
  correlated (e.g. Hanning-smoothed data), σ is underestimated, which leads to more bits than
  strictly needed (a conservative choice).
* **One bitrate per column.** Dysco has one setting per column, so all fields and SPWs share it.
* **Memory.** The row metadata (about 24 bytes per row) are kept in memory, plus one chunk per
  column (`--chunk-mb`).
* **CASA cannot read the output.** You used scalar flags (the default) or antenna/UVW compression;
  see Section 7.
* **Interrupted run.** The partial output is removed automatically. If the process was killed
  hard, delete the output MS by hand; the input is untouched.

---

## 12. References

* A. R. Offringa (2016), *Compression of interferometric radio-astronomical data*,
  A&A 595, A99 — [arXiv:1609.02019](https://arxiv.org/abs/1609.02019). The Dysco paper.
* Dysco source: [github.com/aroffringa/dysco](https://github.com/aroffringa/dysco); now part of
  [casacore](https://github.com/casacore/casacore) (`tables/Dysco`, `tables/AlternateMans`).
* DP3 output step documentation (Dysco and metadata-compression options):
  [dp3.readthedocs.io/en/latest/steps/Output.html](https://dp3.readthedocs.io/en/latest/steps/Output.html).
* [python-casacore](https://github.com/casacore/python-casacore).
