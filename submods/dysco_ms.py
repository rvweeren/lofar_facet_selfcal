#!/usr/bin/env python3
"""
dysco_ms.py -- Dysco-compress or decompress any Measurement Set with python-casacore.

Works for single- and multi-field, single- and multi-SPW Measurement Sets with 1, 2 or 4
correlations (LOFAR, MeerKAT, (E)VLA, uGMRT, ...). Unlike DP3 it does not regrid or rewrite
the MS: all columns, keywords and subtables are kept, only the storage managers of the
selected columns are changed.

Sub-commands
  compress    Dysco-compress DATA/CORRECTED_DATA/WEIGHT_SPECTRUM (+ optional scalar flags,
              UVW and antenna compression) into a new MS.
  decompress  Move all compressed columns back to standard casacore storage managers.
  analyse     Measure noise/signal in the visibilities and recommend Dysco bitrates.
  info        Print the layout and storage managers of an MS.

Run inside the flocs container, e.g.
  singularity exec -B /net flocs.sif python submods/dysco_ms.py compress my.ms --databitrate auto

Documentation (options, automatic bitrate selection, storage-manager pitfalls, test results):
submods/documentation/dysco_ms.md

Notes on the casacore storage managers (casacore >= 3.7, built-in Dysco):
  * DyscoStMan only supports fixed-shape columns, so all rows of a compressed column must
    have the same (nchan, ncorr). MSs with SPWs of different shapes can be split per shape
    with --split-by-shape.
  * Dysco derives its block size from the first change of TIME/FIELD_ID/DATA_DESC_ID and
    reads ANTENNA1/2 of a whole block when decoding. If the number of rows is not a multiple
    of the block size the last block cannot be read back, so the output is padded with fully
    flagged copies of the last row. The number of padding rows is stored in the main-table
    keyword DYSCO_PADDING_ROWS and the rows are removed again by 'decompress'.
  * With AF normalization Dysco sizes its per-antenna arrays from the first block. If a
    higher antenna index appears later this corrupts memory, so RF normalization is used
    instead in that case.
  * StokesIStMan (scalar flags) only works for 4 correlations with identical flags per
    correlation; UvwStMan and AntennaPairStMan need a strictly repeating baseline pattern.
    These options are switched off automatically (with a warning) when not applicable.
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import casacore.tables as ct

DYSCO_BITS = (2, 3, 4, 6, 8, 10, 12, 16)
DEFAULT_COLUMNS = ('DATA', 'CORRECTED_DATA', 'WEIGHT_SPECTRUM')
DISTRIBUTIONS = {'uniform': 'Uniform', 'gaussian': 'Gaussian',
                 'truncatedgaussian': 'TruncatedGaussian', 'studentt': 'StudentT',
                 'studentst': 'StudentT'}
NORMALIZATIONS = {'af': 'AF', 'rf': 'RF', 'row': 'Row'}
SPECIAL_STMANS = ('DyscoStMan', 'StokesIStMan', 'UvwStMan', 'AntennaPairStMan')
PAD_KEYWORD = 'DYSCO_PADDING_ROWS'
META_COLUMNS = ('ANTENNA1', 'ANTENNA2', 'TIME', 'FIELD_ID', 'DATA_DESC_ID')
BYTES_PER_TYPE = {'complex': 8, 'dcomplex': 16, 'float': 4, 'double': 8, 'int': 4,
                  'uint': 4, 'short': 2, 'boolean': 0.125, 'uchar': 1, 'int64': 8}
_t_start = time.time()


def log(msg):
    print(f'[{time.time() - _t_start:8.1f}s] {msg}', flush=True)


def warn(msg):
    log(f'WARNING: {msg}')


class DyscoError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------
# Inspection
# ----------------------------------------------------------------------------------------

def column_managers(t):
    """Map column name -> dminfo entry (TYPE, NAME, SPEC, ...)."""
    result = {}
    for entry in t.getdminfo().values():
        for col in entry['COLUMNS']:
            result[col] = entry
    return result


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fmt_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024.0


class MSInfo:
    """Static description of an MS plus its row metadata (in the order rows will be written)."""

    def __init__(self, path, reorder=False, ddids=None, read_meta=True):
        self.path = os.path.abspath(path.rstrip('/'))
        self.table = ct.table(self.path, ack=False)
        self.npad_in = int(self.table.getkeyword(PAD_KEYWORD)) \
            if PAD_KEYWORD in self.table.keywordnames() else 0
        self.nrow_total = self.table.nrows()
        self.colnames = self.table.colnames()
        self.managers = column_managers(self.table)
        self.coldesc = {c: self.table.getcoldesc(c) for c in self.colnames}
        self.coltype = {c: self.table.coldatatype(c) for c in self.colnames}

        spw = ct.table(self.path + '/SPECTRAL_WINDOW', ack=False)
        self.spw_nchan = spw.getcol('NUM_CHAN')
        spw.close()
        pol = ct.table(self.path + '/POLARIZATION', ack=False)
        self.pol_ncorr = pol.getcol('NUM_CORR')
        pol.close()
        dd = ct.table(self.path + '/DATA_DESCRIPTION', ack=False)
        self.dd_spw = dd.getcol('SPECTRAL_WINDOW_ID')
        self.dd_pol = dd.getcol('POLARIZATION_ID')
        dd.close()
        fld = ct.table(self.path + '/FIELD', ack=False)
        self.field_names = list(fld.getcol('NAME')) if fld.nrows() else []
        fld.close()
        ant = ct.table(self.path + '/ANTENNA', ack=False)
        self.nant = ant.nrows()
        ant.close()

        # The source table: input without padding rows, optionally re-ordered / selected.
        src = self.table
        nreal = self.nrow_total - self.npad_in
        if self.npad_in:
            src = src.selectrows(np.arange(nreal))
        where = ''
        if ddids is not None:
            where = 'WHERE DATA_DESC_ID IN [%s]' % ','.join(str(d) for d in ddids)
        if where or reorder:
            order = 'ORDERBY TIME, DATA_DESC_ID, ANTENNA1, ANTENNA2' if reorder else ''
            src = ct.taql(f'SELECT FROM $1 {where} {order}', tables=[src])
        self.src = src
        self.nrow = src.nrows()
        if read_meta:
            self.read_meta()

    def read_meta(self, chunk=10_000_000):
        n = self.nrow
        self.time = np.empty(n, np.float64)
        self.field = np.empty(n, np.int32)
        self.ddid = np.empty(n, np.int32)
        self.ant1 = np.empty(n, np.int32)
        self.ant2 = np.empty(n, np.int32)
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            self.time[s:s + m] = self.src.getcol('TIME', s, m)
            self.field[s:s + m] = self.src.getcol('FIELD_ID', s, m)
            self.ddid[s:s + m] = self.src.getcol('DATA_DESC_ID', s, m)
            self.ant1[s:s + m] = self.src.getcol('ANTENNA1', s, m)
            self.ant2[s:s + m] = self.src.getcol('ANTENNA2', s, m)
        self.used_ddids = np.unique(self.ddid)

    def ddid_shape(self, ddid):
        return int(self.spw_nchan[self.dd_spw[ddid]]), int(self.pol_ncorr[self.dd_pol[ddid]])

    def shape_groups(self):
        groups = {}
        for d in self.used_ddids:
            groups.setdefault(self.ddid_shape(d), []).append(int(d))
        return groups

    def field_name(self, fid):
        return self.field_names[fid] if 0 <= fid < len(self.field_names) else str(fid)

    def stman(self, col):
        return self.managers[col]['TYPE'] if col in self.managers else '?'


# ----------------------------------------------------------------------------------------
# Layout checks
# ----------------------------------------------------------------------------------------

def dysco_rows_per_block(info):
    """Rows per Dysco block as DyscoStMan will determine it (first change of TIME/FIELD/DDID)."""
    if info.nrow == 0:
        return None
    change = (info.time != info.time[0]) | (info.field != info.field[0]) | \
             (info.ddid != info.ddid[0])
    if not change.any():
        return None
    return int(np.argmax(change))


def baseline_period(info):
    """Length of the repeating ANTENNA1/ANTENNA2 pattern, or None if not strictly periodic."""
    a1, a2 = info.ant1, info.ant2
    if info.nrow < 2:
        return None
    rep = np.nonzero((a1[1:] == a1[0]) & (a2[1:] == a2[0]))[0]
    if len(rep) == 0:
        return None
    period = int(rep[0]) + 1
    nfull = info.nrow // period
    p1, p2 = a1[:period], a2[:period]
    if not (np.array_equal(a1[:nfull * period].reshape(nfull, period), np.broadcast_to(p1, (nfull, period))) and
            np.array_equal(a2[:nfull * period].reshape(nfull, period), np.broadcast_to(p2, (nfull, period)))):
        return None
    rest = info.nrow - nfull * period
    if rest and not (np.array_equal(a1[nfull * period:], p1[:rest]) and
                     np.array_equal(a2[nfull * period:], p2[:rest])):
        return None
    return period


def uvw_derivation(p1, p2):
    """Mimic UvwStMan: walk the baselines of one block in order and derive every antenna's
    UVW from the reference antenna (antenna1 of the first row) or from an antenna derived
    earlier. Returns (ref, {antenna: (row, sign, from_antenna)}, order) or None if a baseline
    cannot be related to the antennas known so far."""
    ref = int(p1[0])
    deriv, order = {ref: None}, [ref]
    for i, (a, b) in enumerate(zip(p1.tolist(), p2.tolist())):
        if a == b or (a in deriv and b in deriv):
            continue
        if a in deriv:
            deriv[b] = (i, 1.0, a)        # U[b] = uvw_i + U[a]
            order.append(b)
        elif b in deriv:
            deriv[a] = (i, -1.0, b)       # U[a] = U[b] - uvw_i
            order.append(a)
        else:
            return None
    return ref, deriv, order


def uvw_reconstruction_error(info, period, deriv, order, max_rows=2_000_000):
    """Max |UVW - (U[a2] - U[a1])| over the MS when UVW is stored per antenna as UvwStMan does."""
    p1, p2 = info.ant1[:period], info.ant2[:period]
    nant = max(int(p1.max()), int(p2.max())) + 1
    auto = (p1 == p2)[None, :, None]
    nblocks = info.nrow // period
    chunk = max(1, max_rows // period)
    worst = 0.0
    for b0 in range(0, nblocks, chunk):
        nb = min(chunk, nblocks - b0)
        uvw = info.src.getcol('UVW', b0 * period, nb * period).reshape(nb, period, 3)
        ant_uvw = np.zeros((nb, nant, 3))
        for ant in order[1:]:
            row, sign, frm = deriv[ant]
            ant_uvw[:, ant] = sign * uvw[:, row] + ant_uvw[:, frm]
        recon = ant_uvw[:, p2] - ant_uvw[:, p1]
        diff = np.where(auto, 0.0, np.abs(recon - uvw))
        worst = max(worst, float(diff.max()))
    return worst


def unequal_corr_flag_fraction(info, rpb):
    """Fraction of sampled (row, channel) cells whose flags differ between correlations."""
    if info.nrow == 0:
        return 0.0
    rpb = min(rpb, info.nrow)
    windows = sample_windows(info, rpb, n_per_group=3, blocks_per_window=1, max_bytes=1e9,
                             bytes_per_row=info.ddid_shape(int(info.ddid[0]))[0] * 4)
    windows = [(w[0], w[1]) for w in windows] + [(max(0, info.nrow - rpb), rpb)]
    unequal = total = 0
    for start, n in windows:
        f = info.src.getcol('FLAG', start, n)
        unequal += int(np.count_nonzero(f.any(axis=2) & ~f.all(axis=2)))
        total += f.shape[0] * f.shape[1]
    return unequal / max(total, 1)


class Plan:
    """Effective settings after checking what the MS layout allows."""


def resolve_dysco_spec(args):
    dist = DISTRIBUTIONS.get(args.distribution.lower())
    if dist is None:
        raise DyscoError(f'Unknown distribution {args.distribution}')
    norm = NORMALIZATIONS.get((args.normalization or 'AF').lower())
    if norm is None:
        raise DyscoError(f'Unknown normalization {args.normalization}')
    return dist, norm


def parse_bitrate(value, name, allow_zero):
    if str(value).lower() == 'auto':
        return 'auto'
    b = int(value)
    if b == 0 and allow_zero:
        return 0
    if b not in DYSCO_BITS:
        raise DyscoError(f'{name} must be one of {DYSCO_BITS}{" or 0" if allow_zero else ""} '
                         f'or "auto" (got {value})')
    return b


def make_plan(info, args):
    p = Plan()
    p.messages = []
    p.distribution, p.normalization = resolve_dysco_spec(args)
    p.databitrate = parse_bitrate(args.databitrate, 'databitrate', True)
    p.weightbitrate = parse_bitrate(args.weightbitrate, 'weightbitrate', True)
    p.disttruncation = float(args.disttruncation)
    p.studenttnu = float(args.studenttnu)

    # Dysco target columns
    req = [c.strip() for c in args.columns.split(',') if c.strip()]
    p.data_cols, p.weight_cols = [], []
    for c in req:
        if c not in info.colnames:
            continue
        if c == 'WEIGHT_SPECTRUM':
            if info.coltype[c] != 'float':
                raise DyscoError('WEIGHT_SPECTRUM must be a float column')
            if p.weightbitrate != 0:
                p.weight_cols.append(c)
        elif info.coltype[c] == 'complex':
            if p.databitrate != 0:
                p.data_cols.append(c)
        else:
            raise DyscoError(f'Column {c} is of type {info.coltype[c]}; Dysco only supports '
                             'complex columns and WEIGHT_SPECTRUM')
    for c in p.data_cols + p.weight_cols:
        if info.stman(c) == 'DyscoStMan':
            warn(f'{c} is already Dysco-compressed; re-compressing adds a second round of '
                 'quantisation noise')

    # Columns that will be on DyscoStMan in the output (targets + untouched Dysco columns)
    p.keep_dysco_cols = [c for c in info.colnames if info.stman(c) == 'DyscoStMan'
                         and c not in p.data_cols + p.weight_cols]
    p.dysco_cols = p.data_cols + p.weight_cols
    if p.keep_dysco_cols:
        warn(f'{", ".join(p.keep_dysco_cols)} already Dysco-compressed and not selected: they are '
             're-encoded with their current settings (adds a little quantisation noise)')

    groups = info.shape_groups()
    p.shape = None
    if p.dysco_cols or p.keep_dysco_cols:
        if len(groups) > 1:
            txt = '; '.join(f'(nchan={k[0]}, ncorr={k[1]}): DDIDs {v}' for k, v in groups.items())
            raise DyscoError('The MS contains data descriptions with different shapes, which '
                             f'cannot share one Dysco column: {txt}. Use --split-by-shape.')
        p.shape = next(iter(groups))
    nchan, ncorr = p.shape if p.shape else (None, None)

    # Dysco block structure
    p.rpb = dysco_rows_per_block(info) if (p.dysco_cols or p.keep_dysco_cols) else None
    p.npad = 0
    if p.dysco_cols or p.keep_dysco_cols:
        if p.rpb is None:
            if p.keep_dysco_cols:
                raise DyscoError('MS consists of one single TIME/FIELD/DDID block; DyscoStMan '
                                 'cannot store this. Decompress it instead.')
            warn('MS consists of one single TIME/FIELD/DDID block; DyscoStMan cannot store '
                 'this, so no Dysco compression is applied')
            p.messages.append('Dysco disabled: single time block')
            p.data_cols, p.weight_cols, p.dysco_cols = [], [], []
        else:
            if p.rpb < 8:
                raise DyscoError(f'Dysco block size would be only {p.rpb} rows because the rows '
                                 'are not grouped per timestep/SPW. Use --reorder.')
            nbl_max = info.nant * (info.nant - 1) // 2
            if p.rpb < 0.25 * nbl_max:
                warn(f'Dysco block size is small ({p.rpb} rows for {info.nant} antennas); '
                     'compression will be less efficient. Consider --reorder.')
            p.npad = (-info.nrow) % p.rpb
            if p.npad:
                if args.no_pad:
                    raise DyscoError(f'nrow={info.nrow} is not a multiple of the Dysco block size '
                                     f'{p.rpb} and --no-pad was given')
                p.messages.append(f'{p.npad} fully flagged padding rows will be appended '
                                  f'(nrow {info.nrow} is not a multiple of the Dysco block size '
                                  f'{p.rpb}); recorded in keyword {PAD_KEYWORD}')
            # AF per-antenna arrays are sized from the first block
            maxant_first = max(info.ant1[:p.rpb].max(), info.ant2[:p.rpb].max())
            maxant = max(info.ant1.max(), info.ant2.max())
            if maxant > maxant_first and p.normalization == 'AF':
                if args.normalization is not None:
                    raise DyscoError(
                        f'AF normalization requested, but the first Dysco block only contains '
                        f'antennas up to index {maxant_first} while the MS has index {maxant}. '
                        'DyscoStMan would corrupt memory. Use --normalization RF (or Row).')
                p.normalization = 'RF'
                p.messages.append(f'normalization AF -> RF: first Dysco block lacks antenna '
                                  f'index {maxant} (AF would overflow)')
            if p.keep_dysco_cols and maxant > maxant_first:
                for c in p.keep_dysco_cols:
                    if info.managers[c]['SPEC'].get('normalization') == 'AF':
                        raise DyscoError(f'{c} is Dysco/AF compressed and not selected; its '
                                         'AF normalization is unsafe for this layout. Add it to '
                                         '--columns or decompress first.')

    # Scalar flags (StokesIStMan)
    p.scalarflags = bool(args.scalarflags) and 'FLAG' in info.colnames
    if p.scalarflags:
        if len(groups) > 1:
            p.scalarflags = False
            p.messages.append('scalarflags disabled: DDIDs with different shapes')
        else:
            fshape = next(iter(groups))
            if fshape[1] != 4:
                p.scalarflags = False
                p.messages.append(f'scalarflags disabled: StokesIStMan needs 4 correlations '
                                  f'(MS has {fshape[1]})')
            else:
                p.flag_shape = fshape
                frac = unequal_corr_flag_fraction(info, p.rpb or 1000)
                if frac > 0 and not args.merge_corr_flags:
                    p.scalarflags = False
                    p.messages.append(
                        f'scalarflags disabled: flags differ between correlations for '
                        f'{100 * frac:.3g}% of the sampled channels (use --merge-corr-flags to '
                        'OR them over correlations and keep scalarflags)')
                elif frac > 0:
                    p.messages.append(f'scalarflags: flags differ between correlations for '
                                      f'{100 * frac:.3g}% of the sampled channels; they will be '
                                      'OR-ed over correlations (--merge-corr-flags)')
    # Antenna compression (AntennaPairStMan)
    period = baseline_period(info)
    p.period = period
    p.antennacompression = bool(args.antennacompression)
    if p.antennacompression and period is None:
        p.antennacompression = False
        p.messages.append('antennacompression disabled: baselines do not follow one strictly '
                          'repeating pattern')
    # UVW compression (UvwStMan)
    p.uvwcompression = bool(args.uvwcompression)
    if p.uvwcompression:
        reason = None
        if period is None:
            reason = 'baselines do not follow one strictly repeating pattern'
        elif (info.nrow + p.npad) % period:
            reason = f'nrow is not a multiple of the baseline pattern ({period} rows)'
        elif p.npad:
            reason = 'padding rows are needed for Dysco'
        elif 'UVW' not in info.colnames:
            reason = 'no UVW column'
        else:
            derivation = uvw_derivation(info.ant1[:period], info.ant2[:period])
            if derivation is None:
                reason = ('baseline order does not allow deriving all antennas from the '
                          'reference antenna')
            else:
                err = uvw_reconstruction_error(info, period, derivation[1], derivation[2])
                if err > args.uvw_tolerance:
                    reason = (f'UVWs are not exact per-antenna differences (max error {err:.3g} m '
                              f'> --uvw-tolerance {args.uvw_tolerance} m), e.g. a DP3 '
                                  'StationAdder superstation (ST001) or weighted UVW averaging')
        if reason:
            p.uvwcompression = False
            p.messages.append(f'uvwcompression disabled: {reason}')
    p.merge_corr_flags = bool(args.merge_corr_flags)
    p.keep_flagged = bool(args.keep_flagged_values)
    return p


# ----------------------------------------------------------------------------------------
# Sampling, noise estimation and automatic bitrates
# ----------------------------------------------------------------------------------------

def sample_windows(info, rpb, n_per_group=3, blocks_per_window=2, max_bytes=2e9,
                   bytes_per_row=1, fields=None):
    """Block-aligned row windows, a few per (FIELD_ID, DATA_DESC_ID) group."""
    nblocks = info.nrow // rpb
    if nblocks < 1:
        return []
    starts = np.arange(nblocks) * rpb
    keys = info.field[starts].astype(np.int64) * 100000 + info.ddid[starts]
    windows = []
    for key in np.unique(keys):
        if fields is not None and key // 100000 not in fields:
            continue
        blocks = np.nonzero(keys == key)[0]
        pick = np.unique(np.linspace(0, len(blocks) - 1, min(n_per_group, len(blocks))).astype(int))
        for i in pick:
            b0 = blocks[i]
            nb = min(blocks_per_window, nblocks - b0)
            windows.append((int(b0 * rpb), int(nb * rpb), int(key // 100000), int(key % 100000)))
    total = sum(w[1] for w in windows) * bytes_per_row
    if total > max_bytes:
        # keep at least one window per group, one block each
        seen, slim = set(), []
        for w in windows:
            k = (w[2], w[3])
            if k not in seen:
                seen.add(k)
                slim.append((w[0], rpb, w[2], w[3]))
        windows = slim
        total = sum(w[1] for w in windows) * bytes_per_row
        if total > max_bytes:
            step = int(math.ceil(total / max_bytes))
            windows = windows[::step]
    return windows


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if x.size < 50:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def estimate_noise(vis, ant1, ant2, tm):
    """Per-real-component thermal noise from channel and time differences (robust MAD)."""
    estimates = []
    if vis.shape[1] > 1:
        dc = vis[:, 1:, :] - vis[:, :-1, :]
        estimates.append(robust_sigma(np.concatenate([dc.real.ravel(), dc.imag.ravel()])) / np.sqrt(2))
    order = np.lexsort((tm, ant2, ant1))
    same = (ant1[order][1:] == ant1[order][:-1]) & (ant2[order][1:] == ant2[order][:-1])
    if same.sum() > 0:
        v = vis[order]
        dt = (v[1:] - v[:-1])[same]
        estimates.append(robust_sigma(np.concatenate([dt.real.ravel(), dt.imag.ravel()])) / np.sqrt(2))
    estimates = [e for e in estimates if np.isfinite(e) and e > 0]
    return min(estimates) if estimates else np.nan


def dysco_error_stats(orig, dec, sigma):
    """Dysco error relative to the thermal noise sigma (per real component).

    Returns (typical, rms, frac_large): 'typical' is a robust (MAD) error estimate that is
    not dominated by a few clipped outliers such as unflagged RFI; 'rms' includes them;
    'frac_large' is the fraction of visibilities with an error above 10 sigma."""
    good = np.isfinite(orig) & np.isfinite(dec)
    if not good.any() or not sigma > 0:
        return np.nan, np.nan, np.nan
    err = dec[good] - orig[good]
    typical = robust_sigma(np.concatenate([err.real, err.imag]))
    rms = np.sqrt(np.mean(np.abs(err) ** 2) / 2)
    frac = float(np.mean(np.abs(err) > 10 * sigma))
    return typical / sigma, rms / sigma, frac


def write_trial_table(path, info, plan, windows, samples, bits_data, bits_weight):
    """Write sample windows into a minimal Dysco table; returns per-window row offsets."""
    nchan, ncorr = plan.shape
    cols = [ct.makescacoldesc('ANTENNA1', 0), ct.makescacoldesc('ANTENNA2', 0),
            ct.makescacoldesc('TIME', 0.0), ct.makescacoldesc('FIELD_ID', 0),
            ct.makescacoldesc('DATA_DESC_ID', 0)]
    dm = {}
    for i, c in enumerate(plan.data_cols + plan.weight_cols):
        vt = 'float' if c == 'WEIGHT_SPECTRUM' else 'complex'
        cols.append(ct.makearrcoldesc(c, 0.0 if vt == 'float' else 0j, shape=[nchan, ncorr],
                                      options=5, valuetype=vt))
        dm[f'*{i + 1}'] = {'TYPE': 'DyscoStMan', 'NAME': f'Trial{c}', 'COLUMNS': [c],
                           'SPEC': dysco_spec(plan, bits_data, bits_weight)}
    t = ct.table(path, ct.maketabdesc(cols), nrow=0, dminfo=dm, ack=False, readonly=False)
    rpb = plan.rpb
    # Block 0 is always the real first block so the antenna count matches the real output.
    segments = [(0, rpb, None)] + [(w[0], w[1], k) for k, w in enumerate(windows)]
    offsets = {}
    row = 0
    blk = 0
    for start, n, k in segments:
        t.addrows(n)
        sl = slice(start, start + n)
        t.putcol('ANTENNA1', info.ant1[sl], row, n)
        t.putcol('ANTENNA2', info.ant2[sl], row, n)
        # Fake TIME so that the Dysco block size is exactly rpb, like in the real output.
        t.putcol('TIME', np.repeat(np.arange(blk, blk + n // rpb, dtype=float), rpb), row, n)
        t.putcol('FIELD_ID', np.zeros(n, np.int32), row, n)
        t.putcol('DATA_DESC_ID', np.zeros(n, np.int32), row, n)
        for c in plan.data_cols + plan.weight_cols:
            arr = samples['first'][c] if k is None else samples[k][c]
            t.putcol(c, arr, row, n)
        if k is not None:
            offsets[k] = row
        row += n
        blk += n // rpb
    t.close()
    return offsets


def dysco_spec(plan, bits_data, bits_weight):
    return {'dataBitCount': int(bits_data or 16), 'weightBitCount': int(bits_weight or 16),
            'distribution': plan.distribution, 'normalization': plan.normalization,
            'distributionTruncation': float(plan.disttruncation),
            'studentTNu': float(plan.studenttnu)}


def prepare_chunk(plan, col, arr, flags):
    """Apply the value conditioning that is also applied when writing the output."""
    if col == 'WEIGHT_SPECTRUM':
        bad = ~np.isfinite(arr) | (arr < 0)
        nbad = int(bad.sum())
        if nbad:
            arr = arr.copy()
            arr[bad] = 0.0
        if flags is not None and not plan.keep_flagged:
            # Zero only channels where all correlations are flagged: Dysco stores the minimum
            # weight over correlations, so zeroing a partly flagged channel would also zero the
            # weight of the unflagged correlations.
            allf = flags.all(axis=2)
            if allf.any():
                arr = arr.copy() if not nbad else arr
                arr[allf] = 0.0
        return arr, nbad
    if flags is not None and not plan.keep_flagged and flags.any():
        arr = arr.copy()
        arr[flags] = np.nan
    return arr, 0


def read_samples(info, plan, windows):
    """Read (conditioned) data/weight samples plus flags for every window."""
    samples = {}
    segs = [('first', 0, plan.rpb)] + [(k, w[0], w[1]) for k, w in enumerate(windows)]
    for key, start, n in segs:
        flags = info.src.getcol('FLAG', start, n) if 'FLAG' in info.colnames else None
        if flags is not None and plan.scalarflags:
            flags = np.broadcast_to(flags.any(axis=2, keepdims=True), flags.shape)
        d = {'FLAG': flags}
        for c in plan.data_cols + plan.weight_cols:
            d[c], _ = prepare_chunk(plan, c, info.src.getcol(c, start, n), flags)
        samples[key] = d
    return samples


def analyse_bitrates(info, plan, args, evaluate_all=False):
    """Empirically determine data/weight bitrates from trial compressions of sample windows.

    Returns (databits, weightbits, report_rows)."""
    if plan.rpb is None or not (plan.data_cols or plan.weight_cols):
        return None, None, []
    nchan, ncorr = plan.shape
    bpr = nchan * ncorr * 8 * max(1, len(plan.data_cols + plan.weight_cols))
    wanted = None
    if args.auto_fields:
        wanted = set()
        for f in args.auto_fields.split(','):
            f = f.strip()
            if f.isdigit():
                wanted.add(int(f))
            elif f in info.field_names:
                wanted.add(info.field_names.index(f))
            elif f:
                raise DyscoError(f'--auto-fields: unknown field {f} (fields: {info.field_names})')
        log('Auto-bitrate: only fields ' + ', '.join(info.field_name(f) for f in sorted(wanted))
            + ' are used')
    windows = sample_windows(info, plan.rpb, n_per_group=args.auto_windows,
                             max_bytes=args.auto_sample_mb * 2**20, bytes_per_row=bpr,
                             fields=wanted)
    if not windows:
        raise DyscoError('No data to sample for the automatic bitrate (check --auto-fields)')
    log(f'Auto-bitrate: {len(windows)} sample windows of ~{int(np.mean([w[1] for w in windows]))} rows '
        f'({fmt_bytes(sum(w[1] for w in windows) * bpr)})')
    samples = read_samples(info, plan, windows)

    # Noise and signal per window, per data column
    stats = []
    for k, w in enumerate(windows):
        sl = slice(w[0], w[0] + w[1])
        for c in plan.data_cols:
            v = samples[k][c]
            sigma = estimate_noise(v, info.ant1[sl], info.ant2[sl], info.time[sl])
            p2 = np.nanmean(np.abs(v) ** 2) if np.isfinite(v).any() else np.nan
            sig = np.sqrt(max(p2 - 2 * sigma ** 2, 0.0)) if np.isfinite(sigma) else np.nan
            stats.append({'win': k, 'col': c, 'field': w[2], 'ddid': w[3], 'sigma': sigma,
                          'snr': sig / (np.sqrt(2) * sigma) if sigma > 0 else np.nan, 'ratio': {}})
        for c in plan.weight_cols:
            wt = samples[k][c]
            wmin = wt.min(axis=2, keepdims=True)
            good = (wt > 0) & np.isfinite(wt)
            polloss = np.sqrt(np.mean(((wt - wmin)[good] / wt[good]) ** 2)) if good.any() else np.nan
            stats.append({'win': k, 'col': c, 'field': w[2], 'ddid': w[3], 'polloss': polloss,
                          'ratio': {}})

    tol = args.noise_tolerance
    max_ratio = math.sqrt((1 + tol) ** 2 - 1)        # eps/sigma allowed for this noise increase
    need_data = plan.databitrate == 'auto' or evaluate_all
    need_weight = plan.weightbitrate == 'auto' or evaluate_all
    tmpdir = tempfile.mkdtemp(prefix='dysco_trial_', dir=args.tmpdir or os.path.dirname(plan.msout))
    chosen_d = chosen_w = None
    try:
        for b in DYSCO_BITS:
            if not evaluate_all and (chosen_d or not need_data or not plan.data_cols) and \
                    (chosen_w or not need_weight or not plan.weight_cols):
                break
            path = os.path.join(tmpdir, f'trial_{b}.tab')
            offsets = write_trial_table(path, info, plan, windows, samples, b, b)
            t = ct.table(path, ack=False)
            for s in stats:
                k, c = s['win'], s['col']
                n = windows[k][1]
                dec = t.getcol(c, offsets[k], n)
                orig = samples[k][c]
                if c == 'WEIGHT_SPECTRUM':
                    ref = np.broadcast_to(orig.min(axis=2, keepdims=True), orig.shape)
                    good = (ref > 0) & np.isfinite(ref)
                    s['ratio'][b] = np.sqrt(np.mean(((dec - ref)[good] / ref[good]) ** 2)) \
                        if good.any() else np.nan
                else:
                    typ, rms, frac = dysco_error_stats(orig, dec, s['sigma'])
                    s['ratio'][b] = typ
                    s.setdefault('ratio_rms', {})[b] = rms
                    s.setdefault('frac_large', {})[b] = frac
            t.close()
            shutil.rmtree(path, ignore_errors=True)
            q = args.auto_group_quantile
            if chosen_d is None and plan.data_cols:
                worst = group_worst(stats, b, data=True, quantile=q)
                if worst <= max_ratio:
                    chosen_d = b
            if chosen_w is None and plan.weight_cols:
                worst = group_worst(stats, b, data=False, quantile=q)
                if worst <= args.weight_tolerance:
                    chosen_w = b
            log(f'  trial {b:2d} bits: eps/sigma = {group_worst(stats, b, True, q):.4f}'
                + (f', weight rel. error = {group_worst(stats, b, False, q):.2e}' if plan.weight_cols else '')
                + (' (worst group)' if q >= 1 else f' ({q:g} quantile over groups)'))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if plan.data_cols and chosen_d is None:
        warn('noise tolerance not reached even at 16 bits; using 16')
        chosen_d = 16
    if plan.weight_cols and chosen_w is None:
        warn('weight tolerance not reached even at 16 bits; using 16')
        chosen_w = 16
    print_analysis(info, plan, stats, chosen_d, chosen_w, max_ratio, tol)
    return chosen_d, chosen_w, stats


def group_worst(stats, b, data=True, quantile=1.0):
    """Max (or the given quantile) over (column, field, ddid) groups of the median ratio over
    the group's windows."""
    groups = {}
    for s in stats:
        if (s['col'] == 'WEIGHT_SPECTRUM') == data or b not in s['ratio']:
            continue
        r = s['ratio'][b]
        if np.isfinite(r):
            groups.setdefault((s['col'], s['field'], s['ddid']), []).append(r)
    if not groups:
        return 0.0
    values = [float(np.median(v)) for v in groups.values()]
    return float(np.quantile(values, quantile, method='higher'))


def print_analysis(info, plan, stats, chosen_d, chosen_w, max_ratio, tol):
    print('\n' + '=' * 96)
    print(f'Dysco bitrate analysis  (distribution={plan.distribution}, '
          f'normalization={plan.normalization}, noise tolerance={100 * tol:.2f}% '
          f'-> eps/sigma <= {max_ratio:.3f})')
    print('=' * 96)
    groups = {}
    for s in stats:
        groups.setdefault((s['col'], s['field'], s['ddid']), []).append(s)
    rows = []
    for (c, f, d), ss in groups.items():
        bits = sorted(set().union(*[s['ratio'].keys() for s in ss]))
        med = {b: np.nanmedian([s['ratio'].get(b, np.nan) for s in ss]) for b in bits}
        if c == 'WEIGHT_SPECTRUM':
            req = next((b for b in bits if med[b] <= plan_weight_tol(plan)), None)
            rows.append((c, f, d, np.nan, np.nanmedian([s['polloss'] for s in ss]), med, req))
        else:
            req = next((b for b in bits if med[b] <= max_ratio), None)
            rows.append((c, f, d, np.nanmedian([s['sigma'] for s in ss]),
                         np.nanmedian([s['snr'] for s in ss]), med, req))
    def need_key(r):
        if all(not np.isfinite(v) for v in r[5].values()):
            return 100          # no usable data (fully flagged): list last
        return -(r[6] or 99)
    rows.sort(key=lambda r: (r[0], need_key(r), r[1], r[2]))
    limit = 40
    hdr_bits = sorted(set().union(*[r[5].keys() for r in rows])) if rows else []
    print(f'{"column":16s} {"field":14s} {"spw":>4s} {"sigma":>10s} {"S/N|polloss":>11s} '
          + ' '.join(f'{b:>7d}b' for b in hdr_bits) + '  need')
    for i, (c, f, d, sigma, snr, med, req) in enumerate(rows):
        if i >= limit:
            print(f'... {len(rows) - limit} more groups not shown')
            break
        spw = info.dd_spw[d] if d < len(info.dd_spw) else d
        print(f'{c:16s} {info.field_name(f)[:14]:14s} {spw:4d} {sigma:10.3e} {snr:11.3g} '
              + ' '.join(f'{med.get(b, np.nan):8.4f}' for b in hdr_bits)
              + '  ' + (str(req) if req else ('n/a' if need_key((c, f, d, sigma, snr, med, req)) == 100
                                              else f'>{max(med)}')))
    print('  (data columns: eps/sigma = typical (robust) Dysco error / thermal noise, per real component;')
    print('   WEIGHT_SPECTRUM: rms relative weight error; polloss = loss from storing min over pols)')
    if plan.data_cols and chosen_d:
        rms = np.nanmax([s.get('ratio_rms', {}).get(chosen_d, np.nan) for s in stats] + [0.0])
        frac = np.nanmax([s.get('frac_large', {}).get(chosen_d, np.nan) for s in stats] + [0.0])
        print(f'At {chosen_d} bits: worst rms eps/sigma incl. outliers = {rms:.4f}, '
              f'max fraction of visibilities with error > 10 sigma = {frac:.2e}')
        if frac > 1e-4:
            print('NOTE: a fraction of the visibilities gets errors > 10 sigma. These are usually '
                  'strong unflagged outliers (RFI) that Dysco clips; flag RFI before compressing. '
                  'The bitrate choice uses the typical (robust) error of the other visibilities.')
    if plan.data_cols:
        print(f'Recommended databitrate  : {chosen_d}')
    if plan.weight_cols:
        print(f'Recommended weightbitrate: {chosen_w}')
    print('=' * 96 + '\n')


def plan_weight_tol(plan):
    return getattr(plan, 'weight_tolerance', 0.005)


# ----------------------------------------------------------------------------------------
# Output construction
# ----------------------------------------------------------------------------------------

def unique_name(name, used):
    base, i = name, 1
    while name in used:
        name = f'{base}_{i}'
        i += 1
    used.add(name)
    return name


def tiled_tile_shape(ncorr, nchan, bytes_per_value, tilesize_kb=1024):
    nrow = max(1, int(tilesize_kb * 1024 / (bytes_per_value * ncorr * nchan)))
    return np.array([ncorr, nchan, nrow], dtype=np.int32)


def new_column_layout(info, plan, mode):
    """Return (coldescs, dminfo) for the columns whose storage manager changes."""
    used = {e['NAME'] for e in info.table.getdminfo().values()}
    descs, dminfo = [], {}

    def add(cols, entry):
        dminfo[f'*{len(dminfo) + 1}'] = dict(entry, COLUMNS=list(cols))

    def fixed_desc(col, shape):
        d = dict(info.coldesc[col])
        d.pop('dataManagerType', None)
        d.pop('dataManagerGroup', None)
        d['option'] = 5
        d['shape'] = np.array(shape, dtype=np.int64)
        d['ndim'] = len(shape)
        d['_c_order'] = True
        return d

    def restore(cols):
        """Move columns back to standard storage managers."""
        ssm_cols = []
        for c in cols:
            if c in ('ANTENNA1', 'ANTENNA2'):
                d = dict(info.coldesc[c])
                d.pop('dataManagerType', None)
                d.pop('dataManagerGroup', None)
                descs.append(ct.makecoldesc(c, d))
                ssm_cols.append(c)
                continue
            shape = list(info.coldesc[c].get('shape', []))
            if not shape:
                shape = list(info.src.getcell(c, 0).shape)
            descs.append(ct.makecoldesc(c, fixed_desc(c, shape)))
            if c == 'UVW':
                tile = np.array([3, 1024], dtype=np.int32)
            else:
                nchan, ncorr = shape
                bpv = BYTES_PER_TYPE.get(info.coltype[c], 8)
                tile = tiled_tile_shape(ncorr, nchan, 8 if c == 'FLAG' else bpv)
                if c == 'FLAG':
                    tile[2] *= 8
            add([c], {'TYPE': 'TiledColumnStMan', 'NAME': unique_name(f'Tiled{c}', used),
                      'SPEC': {'DEFAULTTILESHAPE': tile, 'MAXIMUMCACHESIZE': 0}})
        if ssm_cols:
            add(ssm_cols, {'TYPE': 'StandardStMan', 'NAME': unique_name('SSMAntenna', used),
                           'SPEC': {'BUCKETSIZE': 32768}})

    if mode == 'compress':
        nchan, ncorr = plan.shape if plan.shape else (None, None)
        handled = set()
        for c in plan.dysco_cols + plan.keep_dysco_cols:
            if c in plan.dysco_cols:
                spec = dysco_spec(plan, plan.databitrate, plan.weightbitrate)
            else:
                spec = dict(info.managers[c]['SPEC'])
                if spec.get('distribution') == 'StudentsT':
                    spec['distribution'] = 'StudentT'
            descs.append(ct.makecoldesc(c, fixed_desc(c, [nchan, ncorr])))
            name = {'DATA': 'DyscoData', 'WEIGHT_SPECTRUM': 'DyscoWeightSpectrum'}.get(c, f'Dysco{c}')
            add([c], {'TYPE': 'DyscoStMan', 'NAME': unique_name(name, used), 'SPEC': spec})
            handled.add(c)
        if plan.scalarflags:
            descs.append(ct.makecoldesc('FLAG', fixed_desc('FLAG', list(plan.flag_shape))))
            add(['FLAG'], {'TYPE': 'StokesIStMan', 'NAME': unique_name('StokesIFlag', used), 'SPEC': {}})
            handled.add('FLAG')
        if plan.uvwcompression:
            descs.append(ct.makecoldesc('UVW', fixed_desc('UVW', [3])))
            add(['UVW'], {'TYPE': 'UvwStMan', 'NAME': unique_name('CompressedUvw', used), 'SPEC': {}})
            handled.add('UVW')
        if plan.antennacompression:
            for c in ('ANTENNA1', 'ANTENNA2'):
                d = dict(info.coldesc[c])
                d.pop('dataManagerType', None)
                d.pop('dataManagerGroup', None)
                descs.append(ct.makecoldesc(c, d))
                handled.add(c)
            add(['ANTENNA1', 'ANTENNA2'], {'TYPE': 'AntennaPairStMan',
                                           'NAME': unique_name('AntennaPairStMan', used), 'SPEC': {}})
        # Columns on a special manager that is not requested any more go back to standard ones.
        restore([c for c in info.colnames
                 if info.stman(c) in SPECIAL_STMANS and c not in handled])
    else:
        restore(plan.restore_cols)
    return descs, dminfo


def build_skeleton(info, out_path, descs, dminfo, overwrite):
    if os.path.exists(out_path):
        if not overwrite:
            raise DyscoError(f'{out_path} exists; use --overwrite')
        shutil.rmtree(out_path)
    # Copy without rows: all subtables, keywords, table info and the other columns' managers.
    # Columns on special managers (which cannot be copied empty) are temporarily put on a
    # StandardStMan; they are all replaced below. Hypercube definitions are dropped so the
    # tiled managers start empty.
    copy_dm = {}
    for k, e in info.table.getdminfo().items():
        e = dict(e)
        if e['TYPE'] in SPECIAL_STMANS:
            e = {'TYPE': 'StandardStMan', 'NAME': e['NAME'], 'SPEC': {}, 'COLUMNS': e['COLUMNS']}
        else:
            e['SPEC'] = {a: b for a, b in e['SPEC'].items() if a != 'HYPERCUBES'}
        copy_dm[k] = e
    tc = info.src.copy(out_path, deep=True, valuecopy=True, dminfo=copy_dm, copynorows=True)
    tc.close()
    # copynorows also empties the subtables; copy them again with their rows.
    for name, path in subtable_paths(info.table).items():
        dst = os.path.join(out_path, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        sub = ct.table(path, ack=False)
        sub.copy(dst, deep=True, valuecopy=False).close()   # plain file copy (fast)
        sub.close()
    out = ct.table(out_path, readonly=False, ack=False)
    changed = [ct_name(d) for d in descs]
    if changed:
        out.removecols(changed)
        by_name = {ct_name(d): d for d in descs}
        # Managers that read ANTENNA1/2 when they are created need those columns to exist.
        entries = sorted(dminfo.values(),
                         key=lambda e: 0 if set(e['COLUMNS']) & {'ANTENNA1', 'ANTENNA2'} else 1)
        for entry in entries:
            out.addcols(ct.maketabdesc([by_name[c] for c in entry['COLUMNS']]), entry)
    if PAD_KEYWORD in out.keywordnames():
        out.removekeyword(PAD_KEYWORD)
    return out


def subtable_paths(t):
    """Map subtable keyword -> path for all table-valued keywords of the main table."""
    result = {}
    for kw, val in t.getkeywords().items():
        if isinstance(val, str) and val.startswith('Table: '):
            result[kw] = val[len('Table: '):]
    return result


def ct_name(desc):
    return desc['name']


# ----------------------------------------------------------------------------------------
# Row copying
# ----------------------------------------------------------------------------------------

def column_state(src, col, nrow):
    """'bulk' if cells can be read with getcol, 'empty' if no cell is defined, else 'cells'."""
    if src.isscalarcol(col) or nrow == 0:
        return 'bulk'
    probe = sorted({0, nrow // 2, nrow - 1})
    defined = [src.iscelldefined(col, r) for r in probe]
    if not any(defined):
        return 'empty'
    if all(defined):
        return 'bulk'
    return 'cells'


def copy_cells(src, out, col, start, n, out_start):
    for i in range(n):
        if src.iscelldefined(col, start + i):
            out.putcell(col, out_start + i, src.getcell(col, start + i))


def put_any(src, out, col, start, n, value=None):
    """Copy rows [start, start+n) of col; falls back to per-cell copies for odd columns."""
    try:
        if value is None:
            value = src.getcol(col, start, n)
        out.putcol(col, value, start, n)
    except RuntimeError:
        copy_cells(src, out, col, start, n, start)


class CopyStats:
    def __init__(self):
        self.merged_flags = 0
        self.bad_weights = 0
        self.unequal_flag_chunks = 0


def copy_rows(info, plan, out, write_cols, special_order, chunk_rows, stats):
    src = info.src
    n = info.nrow
    states = {c: column_state(src, c, n) for c in write_cols}
    for c, s in states.items():
        if s == 'empty':
            log(f'  column {c}: no cells defined, left empty')
        elif s == 'cells':
            warn(f'column {c} has partly undefined cells; copied cell by cell (slow)')
    meta = [c for c in META_COLUMNS if c in write_cols]
    late = [c for c in special_order if c in write_cols]
    normal = [c for c in write_cols if c not in meta and c not in late and states[c] != 'empty']
    dysco = set(plan.dysco_cols) if hasattr(plan, 'dysco_cols') else set()
    scalar_flag_out = getattr(plan, 'scalarflags', False)
    t0 = time.time()
    last_log = 0
    for start in range(0, n, chunk_rows):
        m = min(chunk_rows, n - start)
        out.addrows(m)
        # 1. ANTENNA1/2 (AntennaPairStMan needs interleaved writes until its pattern is known)
        if getattr(plan, 'antennacompression', False) and start == 0:
            ncell = min(m, (plan.period or 0) + 1)
            a1 = src.getcol('ANTENNA1', 0, ncell)
            a2 = src.getcol('ANTENNA2', 0, ncell)
            for r in range(ncell):
                out.putcell('ANTENNA1', r, int(a1[r]))
                out.putcell('ANTENNA2', r, int(a2[r]))
            if m > ncell:
                for c in ('ANTENNA1', 'ANTENNA2'):
                    out.putcol(c, src.getcol(c, ncell, m - ncell), ncell, m - ncell)
            meta_rest = [c for c in meta if c not in ('ANTENNA1', 'ANTENNA2')]
        else:
            meta_rest = meta
        for c in meta_rest:
            out.putcol(c, src.getcol(c, start, m), start, m)
        # 2. FLAG (read first, needed for data conditioning)
        flags = None
        if 'FLAG' in info.colnames:
            flags = src.getcol('FLAG', start, m)
            if scalar_flag_out:
                anyf = flags.any(axis=2, keepdims=True)
                if not np.array_equal(np.broadcast_to(anyf, flags.shape), flags):
                    stats.unequal_flag_chunks += 1
                    added = int(np.broadcast_to(anyf, flags.shape).sum() - flags.sum())
                    stats.merged_flags += added
                    flags = np.ascontiguousarray(np.broadcast_to(anyf, flags.shape))
        # 3. ordinary columns
        for c in normal:
            if c == 'FLAG_ROW' and flags is not None and stats.merged_flags:
                fr = src.getcol('FLAG_ROW', start, m) | flags.all(axis=(1, 2))
                out.putcol('FLAG_ROW', fr, start, m)
            elif states[c] == 'cells':
                copy_cells(src, out, c, start, m, start)
            else:
                put_any(src, out, c, start, m)
        # 4. UVW, FLAG and the Dysco columns, in this order
        for c in late:
            if states[c] == 'empty':
                continue
            if c == 'FLAG':
                out.putcol('FLAG', flags, start, m)
            elif c in dysco:
                arr, nbad = prepare_chunk(plan, c, src.getcol(c, start, m), flags)
                stats.bad_weights += nbad
                out.putcol(c, arr, start, m)
            else:
                put_any(src, out, c, start, m)
        done = start + m
        now = time.time()
        if now - last_log > 30 or done == n:
            last_log = now
            rate = done / max(now - t0, 1e-6)
            eta = (n - done) / rate if rate > 0 else 0
            log(f'  {done}/{n} rows ({100.0 * done / n:5.1f}%), {rate:,.0f} rows/s, ETA {eta / 60:.1f} min')


def pad_rows(info, plan, out, write_cols):
    """Append fully flagged copies of the last row so that nrow is a multiple of the block size."""
    npad = plan.npad
    if not npad:
        return
    n = info.nrow
    src = info.src
    out.addrows(npad)
    for c in [c for c in META_COLUMNS if c in write_cols] + \
             [c for c in write_cols if c not in META_COLUMNS]:
        if column_state(src, c, n) == 'empty':
            continue
        if c in ('ANTENNA1', 'ANTENNA2') and plan.antennacompression:
            pat = (info.ant1 if c == 'ANTENNA1' else info.ant2)[:plan.period]
            val = pat[(np.arange(n, n + npad)) % plan.period]
        else:
            try:
                last = src.getcol(c, n - 1, 1)
            except RuntimeError:
                continue
            val = np.repeat(last, npad, axis=0)
            if c == 'FLAG':
                val[...] = True
            elif c == 'FLAG_ROW':
                val[...] = True
            elif c in ('WEIGHT', 'WEIGHT_SPECTRUM'):
                val[...] = 0
            elif c in plan.dysco_cols or c in plan.keep_dysco_cols:
                val[...] = np.nan
        out.putcol(c, val, n, npad)
    out.putkeyword(PAD_KEYWORD, int(npad))


# ----------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------

def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def verify(info, plan, out_path, mode, level, uvw_tol):
    """Compare output to input. Returns (ok, list of messages)."""
    if level == 'none':
        return True, ['verification skipped']
    out = ct.table(out_path, ack=False)
    msgs, ok = [], True
    npad_expected = plan.npad if mode == 'compress' else 0
    if out.nrows() != info.nrow + npad_expected:
        return False, [f'row count {out.nrows()} != expected {info.nrow + npad_expected}']
    if set(out.colnames()) != set(info.colnames):
        return False, [f'column sets differ: {set(out.colnames()) ^ set(info.colnames)}']
    for kw in info.table.keywordnames():
        if kw != PAD_KEYWORD and kw not in out.keywordnames():
            ok = False
            msgs.append(f'keyword {kw} missing')
    out_subs = subtable_paths(out)
    for kw, path in subtable_paths(info.table).items():
        try:
            n_in = ct.table(path, ack=False).nrows()
            n_out = ct.table(out_subs[kw], ack=False).nrows()
        except (KeyError, RuntimeError):
            n_in, n_out = 0, -1
        if n_in != n_out:
            ok = False
            msgs.append(f'subtable {kw} has {n_out} rows instead of {n_in}')
    n = info.nrow
    dysco = set(plan.dysco_cols) if mode == 'compress' else set()
    kept = set(plan.keep_dysco_cols) if mode == 'compress' else set()
    lossy = dysco | kept | ({'UVW'} if mode == 'compress' and plan.uvwcompression else set())
    states = {c: column_state(info.src, c, n) for c in info.colnames}
    # Choose rows to compare
    if level == 'full':
        step = max(1, int(2**28 // max(1, row_bytes(info, info.colnames))))
        windows = [(s, min(step, n - s)) for s in range(0, n, step)]
    else:
        rpb = plan.rpb if mode == 'compress' and plan.rpb else max(1, min(n, 1000))
        ws = sample_windows(info, rpb, n_per_group=2, blocks_per_window=1, max_bytes=5e8,
                            bytes_per_row=row_bytes(info, info.colnames)) if rpb <= n else []
        windows = [(w[0], w[1]) for w in ws[:64]] or [(0, min(n, 1000))]
        windows.append((max(0, n - rpb), min(n, rpb)))
    worst_ratio, worst_rms, worst_frac, worst_w = 0.0, 0.0, 0.0, 0.0
    merged = getattr(plan, 'scalarflags', False) and mode == 'compress'
    for start, m in windows:
        fl_in = info.src.getcol('FLAG', start, m) if 'FLAG' in info.colnames else None
        fl_exp = fl_in
        if merged and fl_in is not None:
            fl_exp = np.broadcast_to(fl_in.any(axis=2, keepdims=True), fl_in.shape)
        for c in info.colnames:
            if states[c] == 'empty' or c in lossy:
                continue
            try:
                a = info.src.getcol(c, start, m)
                b = out.getcol(c, start, m)
            except RuntimeError:
                continue
            if c == 'FLAG':
                a = fl_exp
            elif c == 'FLAG_ROW' and merged:
                a = a | fl_exp.all(axis=(1, 2))
            elif c == 'WEIGHT_SPECTRUM' and mode == 'decompress' and info.stman(c) == 'DyscoStMan':
                pass
            if not arrays_equal(np.asarray(a), np.asarray(b)):
                ok = False
                msgs.append(f'column {c} differs in rows {start}-{start + m}')
        for c in dysco | kept:
            if c in dysco:
                orig, _ = prepare_chunk(plan, c, info.src.getcol(c, start, m), fl_exp)
            else:
                orig = info.src.getcol(c, start, m)
            dec = out.getcol(c, start, m)
            if c == 'WEIGHT_SPECTRUM':
                ref = np.broadcast_to(orig.min(axis=2, keepdims=True), orig.shape)
                good = ref > 0
                if good.any():
                    worst_w = max(worst_w, float(np.sqrt(np.mean(((dec - ref)[good] / ref[good]) ** 2))))
                continue
            if not np.array_equal(np.isfinite(orig), np.isfinite(dec)):
                ok = False
                msgs.append(f'column {c}: finite/NaN pattern differs in rows {start}-{start + m}')
            sl = slice(start, start + m)
            sigma = estimate_noise(orig, info.ant1[sl], info.ant2[sl], info.time[sl])
            typ, rms, frac = dysco_error_stats(orig, dec, sigma)
            if np.isfinite(typ):
                worst_ratio = max(worst_ratio, float(typ))
                worst_rms = max(worst_rms, float(rms))
                worst_frac = max(worst_frac, frac)
    if (dysco | kept) - {'WEIGHT_SPECTRUM'}:
        msgs.append(f'Dysco data: worst sampled typical eps/sigma = {worst_ratio:.4f} '
                    f'(noise increase {100 * (math.sqrt(1 + worst_ratio**2) - 1):.3f}%), '
                    f'rms eps/sigma incl. outliers = {worst_rms:.4f}, '
                    f'max fraction with error > 10 sigma = {worst_frac:.2e}')
        if worst_frac > 1e-4:
            msgs.append('NOTE: some visibilities have errors > 10 sigma; these are usually strong '
                        'unflagged outliers (RFI) that Dysco clips. Flag RFI before compressing.')
    if 'WEIGHT_SPECTRUM' in dysco:
        msgs.append(f'Dysco weights: worst sampled rms relative error = {worst_w:.2e}')
    # Full checks for the lossless metadata managers
    if mode == 'compress' and plan.uvwcompression:
        maxd = 0.0
        for s in range(0, n, 5_000_000):
            m = min(5_000_000, n - s)
            a = info.src.getcol('UVW', s, m)
            b = out.getcol('UVW', s, m)
            auto = info.ant1[s:s + m] == info.ant2[s:s + m]
            a[auto] = 0.0
            maxd = max(maxd, float(np.abs(a - b).max()))
        msgs.append(f'UVW: max |difference| = {maxd:.3e} m')
        if maxd > uvw_tol:
            ok = False
            msgs.append(f'UVW difference exceeds tolerance {uvw_tol} m')
    if mode == 'compress' and plan.antennacompression:
        a1 = np.concatenate([out.getcol('ANTENNA1', s, min(10_000_000, n - s)) for s in range(0, n, 10_000_000)])
        a2 = np.concatenate([out.getcol('ANTENNA2', s, min(10_000_000, n - s)) for s in range(0, n, 10_000_000)])
        if not (np.array_equal(a1, info.ant1) and np.array_equal(a2, info.ant2)):
            ok = False
            msgs.append('ANTENNA1/ANTENNA2 differ')
    if npad_expected:
        f = out.getcol('FLAG_ROW', n, npad_expected) if 'FLAG_ROW' in info.colnames else np.array([True])
        if not f.all():
            ok = False
            msgs.append('padding rows are not flagged')
    out.close()
    return ok, msgs


def row_bytes(info, cols):
    total = 0
    for c in cols:
        d = info.coldesc[c]
        nval = int(np.prod(d['shape'])) if 'shape' in d and len(d.get('shape', [])) else 1
        if not info.table.isscalarcol(c) and 'shape' not in d:
            if c in ('DATA', 'CORRECTED_DATA', 'MODEL_DATA', 'FLAG', 'WEIGHT_SPECTRUM', 'SIGMA_SPECTRUM'):
                nchan, ncorr = next(iter(info.shape_groups()))
                nval = nchan * ncorr
            else:
                nval = 4
        total += nval * BYTES_PER_TYPE.get(info.coltype[c], 8)
    return max(1, int(total))


# ----------------------------------------------------------------------------------------
# Size estimate and replacing the input
# ----------------------------------------------------------------------------------------

def column_disk_bytes(info, col, stman, spec, nrow):
    """Rough on-disk size of a column under the given storage manager."""
    if info.table.isscalarcol(col):
        if stman == 'AntennaPairStMan':
            return 0
        return nrow * BYTES_PER_TYPE.get(info.coltype[col], 8)
    groups = info.shape_groups()
    nchan, ncorr = next(iter(groups)) if groups else (1, 1)
    if col == 'UVW':
        return 0 if stman == 'UvwStMan' else nrow * 24
    nval = nchan * ncorr
    if stman == 'DyscoStMan':
        if col == 'WEIGHT_SPECTRUM':
            return nrow * nchan * spec['weightBitCount'] / 8
        return nrow * nval * 2 * spec['dataBitCount'] / 8
    if stman == 'StokesIStMan':
        return nrow * nchan / 8
    return nrow * nval * BYTES_PER_TYPE.get(info.coltype[col], 8)


def check_disk_space(info, out_path, changes, force):
    """changes: {col: (new_stman, new_spec)}"""
    in_size = dir_size(info.path)
    est = in_size
    for c, (stman, spec) in changes.items():
        old = info.managers.get(c, {})
        est -= column_disk_bytes(info, c, old.get('TYPE', ''), old.get('SPEC', {}), info.nrow_total)
        est += column_disk_bytes(info, c, stman, spec, info.nrow)
    est = max(est, 0) * 1.02
    free = shutil.disk_usage(os.path.dirname(out_path) or '.').free
    log(f'Input size {fmt_bytes(in_size)}, estimated output size {fmt_bytes(est)}, '
        f'free space {fmt_bytes(free)}')
    if est > free and not force:
        raise DyscoError('Not enough free disk space for the output (use --force to try anyway)')
    return in_size, est


def replace_input(msin, msout):
    tmp = msin.rstrip('/') + f'.dysco_ms_orig_{os.getpid()}'
    log(f'Replacing {msin} with {msout}')
    os.rename(msin, tmp)
    try:
        shutil.move(msout, msin)
    except Exception:
        os.rename(tmp, msin)
        raise
    shutil.rmtree(tmp)


# ----------------------------------------------------------------------------------------
# Sub-commands
# ----------------------------------------------------------------------------------------

def default_out(msin, suffix):
    msin = os.path.abspath(msin.rstrip('/'))
    return msin + suffix


def run_compress(args):
    msin = os.path.abspath(args.ms.rstrip('/'))
    probe = MSInfo(msin, read_meta=False)
    groups = None
    if args.split_by_shape:
        probe.read_meta()
        groups = probe.shape_groups()
        if len(groups) == 1:
            groups = None
    probe.table.close()
    if groups:
        if args.replace:
            raise DyscoError('--replace cannot be combined with --split-by-shape')
        base = args.msout or default_out(msin, '.dysco')
        root, ext = os.path.splitext(base)
        log(f'Splitting into {len(groups)} MSs with one data shape each')
        ok_all = True
        for i, (shape, ddids) in enumerate(sorted(groups.items())):
            out = f'{root}.shape{i}_{shape[0]}ch{shape[1]}pol{ext}'
            log(f'--- shape group {i}: nchan={shape[0]} ncorr={shape[1]} DDIDs={ddids} -> {out}')
            ok_all &= compress_one(args, msin, out, ddids)
        return 0 if ok_all else 1
    msout = os.path.abspath((args.msout or default_out(msin, '.dysco')).rstrip('/'))
    return 0 if compress_one(args, msin, msout, None) else 1


def compress_one(args, msin, msout, ddids):
    if msout == msin:
        raise DyscoError('msout must differ from the input (use --replace to replace the input)')
    log(f'Reading metadata of {msin}')
    info = MSInfo(msin, reorder=args.reorder, ddids=ddids)
    log(f'{info.nrow} rows, {len(info.used_ddids)} DDIDs, {len(np.unique(info.field))} fields, '
        f'{info.nant} antennas, shapes {list(info.shape_groups().keys())}')
    plan = make_plan(info, args)
    plan.msout = msout
    plan.weight_tolerance = args.weight_tolerance
    plan.auto_data = plan.databitrate == 'auto'
    plan.auto_weight = plan.weightbitrate == 'auto'
    if plan.auto_data or plan.auto_weight:
        db, wb, _ = analyse_bitrates(info, plan, args)
        if plan.auto_data:
            plan.databitrate = db if db else 10
            if plan.data_cols:
                log(f'AUTO: databitrate = {plan.databitrate} bits (lowest bitrate with <= '
                    f'{100 * args.noise_tolerance:g}% noise increase in every field/SPW)')
        if plan.auto_weight:
            plan.weightbitrate = wb if wb else 12
            if plan.weight_cols:
                log(f'AUTO: weightbitrate = {plan.weightbitrate} bits (lowest bitrate with rms '
                    f'relative weight error <= {args.weight_tolerance:g})')
    print_plan(info, plan)

    descs, dminfo = new_column_layout(info, plan, 'compress')
    changes = {}
    for e in dminfo.values():
        for c in e['COLUMNS']:
            changes[c] = (e['TYPE'], e['SPEC'])
    check_disk_space(info, msout, changes, args.force)
    if args.dry_run:
        log('Dry run: nothing written')
        return True

    special_order = (['UVW'] if 'UVW' in info.colnames else []) + \
                    (['FLAG'] if 'FLAG' in info.colnames else []) + \
                    [c for c in info.colnames if c in plan.dysco_cols or c in plan.keep_dysco_cols]
    chunk_rows = chunk_size(info, plan, args.chunk_mb)
    stats = CopyStats()

    def write(holder):
        out = holder['table'] = build_skeleton(info, msout, descs, dminfo, args.overwrite)
        log(f'Writing {msout} in chunks of {chunk_rows} rows')
        copy_rows(info, plan, out, info.colnames, special_order, chunk_rows, stats)
        pad_rows(info, plan, out, info.colnames)
        out.close()

    write_or_cleanup(write, msout, args.overwrite)
    report_stats(stats)
    return finish(args, info, plan, msin, msout, 'compress')


def chunk_size(info, plan, chunk_mb):
    rb = 0
    groups = info.shape_groups()
    nchan, ncorr = next(iter(groups)) if groups else (1, 1)
    rb = nchan * ncorr * 8
    rows = max(1, int(chunk_mb * 2**20 / rb))
    rpb = getattr(plan, 'rpb', None)
    if rpb:
        rows = max(rpb, rows // rpb * rpb)
    if getattr(plan, 'period', None) and getattr(plan, 'antennacompression', False):
        rows = max(rows, plan.period + 1)
    return rows


def write_or_cleanup(write, msout, overwrite):
    """Run write(); remove the partial output if it fails."""
    if os.path.exists(msout) and not overwrite:
        raise DyscoError(f'{msout} exists; use --overwrite')
    holder = {}
    try:
        write(holder)
    except BaseException:
        log(f'Writing failed; removing partial output {msout}')
        if holder.get('table') is not None:
            try:
                holder['table'].close()
            except Exception:
                pass
        shutil.rmtree(msout, ignore_errors=True)
        raise


def report_stats(stats):
    if stats.merged_flags:
        warn(f'scalarflags: per-correlation flags differed in {stats.unequal_flag_chunks} chunks; '
             f'{stats.merged_flags} visibilities were additionally flagged (OR over correlations)')
    if stats.bad_weights:
        warn(f'{stats.bad_weights} negative or non-finite weights were set to 0')


def finish(args, info, plan, msin, msout, mode):
    log('Verifying output')
    ok, msgs = verify(info, plan, msout, mode, args.verify, args.uvw_tolerance)
    for m in msgs:
        log(f'  {m}')
    in_size = dir_size(info.path)
    out_size = dir_size(msout)
    log(f'Size: input {fmt_bytes(in_size)} -> output {fmt_bytes(out_size)} '
        f'({100.0 * out_size / max(in_size, 1):.1f}%)')
    info.table.close()
    if not ok:
        log('VERIFICATION FAILED; input left untouched')
        return False
    log('Verification passed')
    if mode == 'compress':
        log('Settings used: ' + settings_summary(plan))
        newer = [n for n, on in (('StokesIStMan (scalarflags)', plan.scalarflags),
                                 ('AntennaPairStMan (antennacompression)', plan.antennacompression),
                                 ('UvwStMan (uvwcompression)', plan.uvwcompression)) if on]
        if newer:
            log('NOTE: this MS uses ' + ', '.join(newer) + ', which need casacore >= 3.7 '
                '(DP3/WSClean in the flocs container are fine, but e.g. CASA 6.7 cannot open it). '
                'Use --no-scalarflags (and leave antenna/uvw compression off) for CASA compatibility, '
                'or run "decompress" later.')
    if args.replace:
        if args.verify == 'none':
            warn('--replace requires verification; input not replaced')
        else:
            replace_input(msin, msout)
            log(f'Done: {msin} replaced')
            return True
    log(f'Done: {msout}')
    return True


def settings_summary(plan):
    parts = []
    if plan.data_cols:
        parts.append(f'databitrate={plan.databitrate}'
                     f'{" (auto)" if getattr(plan, "auto_data", False) else ""}')
    else:
        parts.append('data not compressed')
    if plan.weight_cols:
        parts.append(f'weightbitrate={plan.weightbitrate}'
                     f'{" (auto)" if getattr(plan, "auto_weight", False) else ""}')
    else:
        parts.append('weights not compressed')
    if plan.dysco_cols:
        parts.append(f'distribution={plan.distribution}, disttruncation={plan.disttruncation}, '
                     f'normalization={plan.normalization}')
    parts.append(f'scalarflags={plan.scalarflags}, antennacompression={plan.antennacompression}, '
                 f'uvwcompression={plan.uvwcompression}')
    return ', '.join(parts)


def print_plan(info, plan):
    print('-' * 80)
    print('Compression settings')
    if plan.data_cols:
        print(f'  Dysco data columns : {", ".join(plan.data_cols)} ({plan.databitrate} bits'
              f'{", auto" if getattr(plan, "auto_data", False) else ""})')
    if plan.weight_cols:
        print(f'  Dysco weight column: {", ".join(plan.weight_cols)} ({plan.weightbitrate} bits'
              f'{", auto" if getattr(plan, "auto_weight", False) else ""})')
    if not plan.dysco_cols:
        print('  Dysco              : no columns selected')
    else:
        print(f'  distribution={plan.distribution}, disttruncation={plan.disttruncation}, '
              f'normalization={plan.normalization}')
        print(f'  Dysco block size   : {plan.rpb} rows, padding rows: {plan.npad}')
    if plan.keep_dysco_cols:
        print(f'  kept Dysco columns : {", ".join(plan.keep_dysco_cols)} (unchanged settings)')
    print(f'  scalarflags={plan.scalarflags}, antennacompression={plan.antennacompression}, '
          f'uvwcompression={plan.uvwcompression}')
    print(f'  flagged values {"kept" if plan.keep_flagged else "set to NaN (data) / 0 (weights)"}')
    for m in plan.messages:
        print(f'  NOTE: {m}')
    print('-' * 80, flush=True)


def run_decompress(args):
    msin = os.path.abspath(args.ms.rstrip('/'))
    msout = os.path.abspath((args.msout or default_out(msin, '.nodysco')).rstrip('/'))
    if msout == msin:
        raise DyscoError('msout must differ from the input (use --replace to replace the input)')
    log(f'Reading metadata of {msin}')
    info = MSInfo(msin)
    plan = Plan()
    plan.restore_cols = [c for c in info.colnames if info.stman(c) in SPECIAL_STMANS]
    plan.npad = 0
    plan.rpb = None
    plan.dysco_cols, plan.keep_dysco_cols = [], []
    plan.uvwcompression = plan.antennacompression = plan.scalarflags = False
    if not plan.restore_cols and not info.npad_in:
        log('Nothing to decompress: no columns use Dysco/StokesI/Uvw/AntennaPair storage managers')
        return 0
    log(f'Restoring columns {plan.restore_cols} to standard storage managers'
        + (f', removing {info.npad_in} padding rows' if info.npad_in else ''))
    descs, dminfo = new_column_layout(info, plan, 'decompress')
    changes = {c: (e['TYPE'], e['SPEC']) for e in dminfo.values() for c in e['COLUMNS']}
    check_disk_space(info, msout, changes, args.force)
    if args.dry_run:
        log('Dry run: nothing written')
        return 0
    chunk_rows = chunk_size(info, plan, args.chunk_mb)
    stats = CopyStats()

    def write(holder):
        out = holder['table'] = build_skeleton(info, msout, descs, dminfo, args.overwrite)
        log(f'Writing {msout} in chunks of {chunk_rows} rows')
        copy_rows(info, plan, out, info.colnames, [], chunk_rows, stats)
        out.close()

    write_or_cleanup(write, msout, args.overwrite)
    return 0 if finish(args, info, plan, msin, msout, 'decompress') else 1


def run_analyse(args):
    msin = os.path.abspath(args.ms.rstrip('/'))
    log(f'Reading metadata of {msin}')
    info = MSInfo(msin, reorder=args.reorder)
    args.databitrate = 'auto'
    args.weightbitrate = 'auto'
    plan = make_plan(info, args)
    plan.msout = msin
    plan.weight_tolerance = args.weight_tolerance
    if plan.rpb is None:
        log('No Dysco-compressible layout (see notes above)')
        return 1
    print_plan(info, plan)
    analyse_bitrates(info, plan, args, evaluate_all=args.all_bits)
    return 0


def run_info(args):
    msin = os.path.abspath(args.ms.rstrip('/'))
    info = MSInfo(msin, read_meta=args.layout)
    t = info.table
    print(f'MS           : {msin}')
    print(f'Rows         : {info.nrow_total}' + (f' (incl. {info.npad_in} Dysco padding rows)' if info.npad_in else ''))
    print(f'Antennas     : {info.nant}')
    print(f'Fields       : {len(info.field_names)} {info.field_names[:10]}')
    print(f'SPWs         : {len(info.spw_nchan)} (nchan {sorted(set(info.spw_nchan.tolist()))})')
    print(f'Correlations : {sorted(set(info.pol_ncorr.tolist()))}')
    print(f'Size on disk : {fmt_bytes(dir_size(msin))}')
    print('Storage managers:')
    files = {}
    for f in os.listdir(msin):
        if f.startswith('table.f'):
            seq = f[7:].split('_')[0]
            if seq.isdigit():
                files[int(seq)] = files.get(int(seq), 0) + os.path.getsize(os.path.join(msin, f))
    for e in sorted(t.getdminfo().values(), key=lambda e: e['SEQNR']):
        spec = ''
        if e['TYPE'] == 'DyscoStMan':
            s = e['SPEC']
            spec = (f" [data {s.get('dataBitCount')} bit, weight {s.get('weightBitCount')} bit, "
                    f"{s.get('distribution')}, trunc {s.get('distributionTruncation')}, "
                    f"{s.get('normalization')}]")
        print(f'  {e["TYPE"]:18s} {e["NAME"]:22s} {fmt_bytes(files.get(e["SEQNR"], 0)):>10s}  '
              f'{",".join(e["COLUMNS"])}{spec}')
    if args.layout:
        rpb = dysco_rows_per_block(info)
        print(f'Dysco block  : {rpb} rows' + (f', {(-info.nrow) % rpb} padding rows needed' if rpb else ''))
        print(f'Shape groups : {info.shape_groups()}')
        print(f'Baseline pattern period: {baseline_period(info)}')
    t.close()
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common_out(sp):
        sp.add_argument('--msout', help='output MS (default <ms>.dysco or <ms>.nodysco)')
        sp.add_argument('--replace', action='store_true',
                        help='replace the input MS by the output after successful verification')
        sp.add_argument('--overwrite', action='store_true', help='overwrite an existing --msout')
        sp.add_argument('--verify', choices=('sample', 'full', 'none'), default='sample',
                        help='compare output with input on sampled rows (default) or all rows')
        sp.add_argument('--chunk-mb', type=float, default=1024,
                        help='approximate memory per column chunk in MB (default 1024)')
        sp.add_argument('--force', action='store_true', help='ignore the free disk space check')
        sp.add_argument('--dry-run', action='store_true', help='only show what would be done')
        sp.add_argument('--uvw-tolerance', type=float, default=1e-4,
                        help='max allowed UVW difference in m for uvwcompression (default 1e-4)')

    def dysco_opts(sp):
        sp.add_argument('--databitrate', default='10',
                        help=f'bits per visibility float {DYSCO_BITS}, 0 = do not compress data, '
                             'or "auto" (default 10)')
        sp.add_argument('--weightbitrate', default='12',
                        help='bits per WEIGHT_SPECTRUM value, 0 = do not compress weights, or "auto" '
                             '(default 12)')
        sp.add_argument('--distribution', default='TruncatedGaussian',
                        help='Uniform, Gaussian, TruncatedGaussian (default) or StudentT')
        sp.add_argument('--disttruncation', type=float, default=2.5,
                        help='truncation level for TruncatedGaussian (default 2.5)')
        sp.add_argument('--normalization', default=None,
                        help='AF (default), RF or Row. AF falls back to RF if unsafe for the layout')
        sp.add_argument('--studenttnu', type=float, default=0.0, help='nu for the StudentT distribution')
        sp.add_argument('--columns', default=','.join(DEFAULT_COLUMNS),
                        help='comma separated columns to Dysco-compress '
                             f'(default {",".join(DEFAULT_COLUMNS)})')
        sp.add_argument('--scalarflags', action=argparse.BooleanOptionalAction, default=True,
                        help='store one flag per channel for all 4 correlations (default on)')
        sp.add_argument('--antennacompression', action=argparse.BooleanOptionalAction, default=False,
                        help='losslessly compress ANTENNA1/ANTENNA2 (default off)')
        sp.add_argument('--uvwcompression', action=argparse.BooleanOptionalAction, default=False,
                        help='store UVW per antenna (UvwStMan, default off)')
        sp.add_argument('--merge-corr-flags', action='store_true',
                        help='with scalarflags: OR the flags over correlations if they differ '
                             '(otherwise scalarflags is switched off when they differ)')
        sp.add_argument('--keep-flagged-values', action='store_true',
                        help='do not set flagged data to NaN / flagged weights to 0 before '
                             'compressing (DP3 does set them)')
        sp.add_argument('--reorder', action='store_true',
                        help='write rows ordered by TIME, DATA_DESC_ID, ANTENNA1, ANTENNA2')
        sp.add_argument('--no-pad', action='store_true',
                        help='refuse instead of appending flagged padding rows')
        sp.add_argument('--noise-tolerance', type=float, default=0.01,
                        help='auto bitrate: allowed fractional noise increase (default 0.01)')
        sp.add_argument('--weight-tolerance', type=float, default=0.005,
                        help='auto bitrate: allowed rms relative weight error (default 0.005)')
        sp.add_argument('--auto-fields', default='',
                        help='auto bitrate: comma separated field names/ids that determine the '
                             'bitrate (default: all fields, incl. calibrators)')
        sp.add_argument('--auto-group-quantile', type=float, default=1.0,
                        help='auto bitrate: quantile over field/SPW groups that must meet the '
                             'tolerance; 1.0 = every group (default), e.g. 0.95 ignores the worst '
                             '5%% (e.g. RFI-dominated SPWs)')
        sp.add_argument('--auto-windows', type=int, default=3,
                        help='auto bitrate: sample windows per field/SPW (default 3)')
        sp.add_argument('--auto-sample-mb', type=float, default=2000,
                        help='auto bitrate: max sample size in MB (default 2000)')
        sp.add_argument('--tmpdir', help='directory for trial compressions (default: next to output)')
        if not any(a.dest == 'uvw_tolerance' for a in sp._actions):
            sp.add_argument('--uvw-tolerance', type=float, default=1e-4,
                            help='max allowed UVW difference in m for uvwcompression (default 1e-4)')

    c = sub.add_parser('compress', help='Dysco-compress an MS')
    c.add_argument('ms')
    common_out(c)
    dysco_opts(c)
    c.add_argument('--split-by-shape', action='store_true',
                   help='write one output MS per (nchan, ncorr) shape if SPWs differ in shape')

    d = sub.add_parser('decompress', help='move all compressed columns to standard storage managers')
    d.add_argument('ms')
    common_out(d)

    a = sub.add_parser('analyse', help='recommend Dysco bitrates from the noise in the data')
    a.add_argument('ms')
    dysco_opts(a)
    a.add_argument('--all-bits', action='store_true', help='evaluate every bit count')

    i = sub.add_parser('info', help='show MS layout and storage managers')
    i.add_argument('ms')
    i.add_argument('--layout', action='store_true', help='also analyse the row layout (reads metadata)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == 'compress':
            return run_compress(args)
        if args.cmd == 'decompress':
            return run_decompress(args)
        if args.cmd == 'analyse':
            return run_analyse(args)
        return run_info(args)
    except DyscoError as e:
        log(f'ERROR: {e}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
