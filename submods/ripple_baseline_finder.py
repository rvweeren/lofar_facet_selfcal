#!/usr/bin/env python3
"""Find baselines/antennas (and time/frequency ranges) that cause ripples/stripes.

Input: a Measurement Set with a residual column, the MFS residual image and the
MFS (clean) image of the same data. The script

1. characterises the ripple in the residual image (uv band, wavelength, angle),
2. scans the residual visibilities for baselines/antennas with a systematic
   (non-noise) excess, for antennas with a noise excess, for compact holes in the
   flagged uv coverage and for the baselines/times/channels that feed the
   ripple uv band,
3. confirms the candidates by re-imaging the residual data with WSClean without
   deconvolution (candidate baselines/antennas removed, plus control
   baselines/antennas removed, plus a reference), and measures the ripple power
   away from bright emission,
4. writes diagnostic PNGs, a markdown report and results.json.

The input Measurement Set is only read. WSClean works on a working copy of the
MS (written to --work-dir; it needs about the size of the MS). No flags are ever
written to the input MS: the report only suggests actions (TaQL / DP3).

Example:
    python submods/ripple_baseline_finder.py obs.ms \\
        --residual-image run-MFS-residual.fits --image run-MFS-image.fits \\
        --data-column RESIDUAL_DATA --output-dir ripple_report

Documentation (methods, output figures, how to interpret them):
submods/documentation/ripple_baseline_finder.md

Telescope independent (GMRT, MeerKAT, LOFAR, ...): parallel hands (RR/LL or
XX/YY, or Stokes I) are combined, flagged data and NaN are ignored.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label,
    map_coordinates,
)

C_LIGHT = 299792458.0


class FinderError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]
        + ". Reads an MS with residual data plus MFS residual and clean images, "
        "reports bad baselines/antennas and confirms them with WSClean tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("ms", help="Measurement Set with a residual data column")
    p.add_argument("--residual-image", required=True, help="MFS residual FITS image")
    p.add_argument("--image", required=True, help="MFS (clean) image FITS, used to mask bright emission")
    p.add_argument("--data-column", default="RESIDUAL_DATA", help="Residual visibility column")
    p.add_argument("--weight-column", choices=("auto", "WEIGHT_SPECTRUM", "WEIGHT", "none"), default="auto")
    p.add_argument("--output-dir", default="ripple_report", help="Directory for PNGs, report and results.json")
    p.add_argument("--work-dir", help="Directory for the working MS copy and WSClean images (default: <output-dir>/work)")
    p.add_argument("--field-id", type=int, help="FIELD_ID to use (default: the field with most rows)")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")

    g = p.add_argument_group("bright emission mask")
    g.add_argument("--bright-sigma", type=float, default=50.0, help="Bright = smoothed image above this many residual rms")
    g.add_argument("--bright-threshold", type=float, help="Bright threshold in Jy/beam (overrides --bright-sigma)")
    g.add_argument("--bright-margin-arcsec", type=float, default=100.0, help="Margin excluded around bright emission")

    g = p.add_argument_group("ripple band detection (residual image)")
    g.add_argument("--highpass-arcsec", type=float, default=50.0, help="Gaussian high-pass scale")
    g.add_argument("--tile-size", type=int, default=1024, help="FFT tile size in pixels (reduced for small images)")
    g.add_argument("--band-threshold", type=float, default=2.0, help="Whitened tile power above this is ripple band")
    g.add_argument("--band-uv-min", type=float, help="Minimum |uv| of the band (lambda; default 6 tile FFT cells)")
    g.add_argument("--band-uv-max", type=float, help="Maximum |uv| of the band (lambda; default min(uv max of data, Nyquist/4))")
    g.add_argument("--band-min-cells", type=int, default=12, help="Minimum area of a band component in FFT cells")
    g.add_argument("--uv-sign", choices=("wcs", "flip"), default="wcs", help="Map image FFT axes to MS u,v: wcs = from the FITS axes (WSClean convention), flip = opposite u sign")

    g = p.add_argument_group("visibility analysis")
    g.add_argument("--chan-avg", type=int, default=0, help="Channels averaged before analysis (0 = auto, at most ~128 channels)")
    g.add_argument("--uv-cell", type=float, help="uv cell size in lambda for the coherent analysis (default 2.5 / field of view)")
    g.add_argument("--time-blocks", type=int, default=30, help="Time blocks for time/frequency localisation")
    g.add_argument("--freq-blocks", type=int, default=16, help="Frequency blocks for time/frequency localisation")
    g.add_argument("--min-baseline-length", type=float, default=500.0, help="Shortest baseline (m) used for outlier statistics")
    g.add_argument("--z-threshold", type=float, default=6.0, help="Robust z threshold for systematic-excess outliers")
    g.add_argument("--clip-factor", type=float, default=13.0, help="Clip w|r|^2 at this factor times the median")
    g.add_argument("--chunk-elements", type=float, default=1.5e7, help="Visibility elements per read chunk")
    g.add_argument("--max-ripple-samples", type=float, default=2.0e8, help="Maximum samples kept in the ripple band")

    g = p.add_argument_group("WSClean confirmation")
    g.add_argument("--no-wsclean", action="store_true", help="Skip the WSClean confirmation images")
    g.add_argument("--wsclean", default="wsclean", help="WSClean command (may contain a prefix, e.g. 'singularity exec x.sif wsclean')")
    g.add_argument("--wsclean-args", help="Imaging options replacing those taken from the image header HISTORY (without MS, -name, -data-column, -niter)")
    g.add_argument("--wsclean-threads", type=int, default=0, help="Passed to WSClean -j when > 0")
    g.add_argument("--jobs", type=int, default=1, help="Concurrent WSClean runs (each needs its own MS copy)")
    g.add_argument("--max-baseline-candidates", type=int, default=4)
    g.add_argument("--max-antenna-candidates", type=int, default=2)
    g.add_argument("--n-controls", type=int, default=6, help="Control baselines")
    g.add_argument("--n-antenna-controls", type=int, default=3, help="Control antennas (only if antenna candidates exist)")
    g.add_argument("--no-baseline-antennas", action="store_true", help="Do not test the two antennas of a clearly bad baseline as antenna candidates")
    g.add_argument("--extra-baselines", default="", help="Comma list of extra baselines to test, e.g. C05-E02,E03-E04")
    g.add_argument("--extra-antennas", default="", help="Comma list of extra antennas to test")
    g.add_argument("--no-window-test", action="store_true", help="Do not test a time/frequency window variant for the top candidate")
    g.add_argument("--keep-work", action="store_true", help="Keep the working MS copy and all WSClean products")
    g.add_argument("--crop-fraction", type=float, default=0.6, help="Fraction of the image shown in comparison figures")
    return p


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def read_image(path: str):
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header.copy()
        data = np.asarray(hdul[0].data, dtype=np.float32)
    while data.ndim > 2:
        data = data[0]
    return data, header


def robust_rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def history_command(header) -> Optional[str]:
    cards = [str(c) for c in header.get("HISTORY", [])]
    text = "".join(c.ljust(72) for c in cards)
    start = text.find("wsclean ")
    if start < 0:
        return None
    text = text[start:]
    end = text.find("facetselfcal version")
    return (text[:end] if end > 0 else text).strip()


class ImageInfo:
    def __init__(self, header, shape):
        self.ny, self.nx = shape
        self.pix_deg = abs(float(header["CDELT1"]))
        self.pix_rad = math.radians(self.pix_deg)
        self.pix_arcsec = self.pix_deg * 3600.0
        self.su = -1.0 if float(header["CDELT1"]) < 0 else 1.0
        self.sv = 1.0 if float(header["CDELT2"]) > 0 else -1.0
        self.beam_px = float(header.get("BMAJ", 4 * self.pix_deg)) / self.pix_deg
        self.crval = (float(header.get("CRVAL1", np.nan)), float(header.get("CRVAL2", np.nan)))
        self.fov_rad = max(self.nx, self.ny) * self.pix_rad


def make_bright_masks(image, info, rms, args):
    sigma = max(2.0, info.beam_px)
    thr = args.bright_threshold if args.bright_threshold else args.bright_sigma * rms
    smooth = gaussian_filter(np.nan_to_num(image), sigma)
    bright = smooth > thr
    zero = binary_dilation(bright, iterations=3) if bright.any() else bright
    margin_px = args.bright_margin_arcsec / info.pix_arcsec
    exclude = gaussian_filter(bright.astype(np.float32), max(margin_px / 2.0, 1.0)) > 0.02
    return bright, zero, exclude, thr


def tile_size_for(info, requested):
    t = 1
    while t * 2 <= min(requested, min(info.nx, info.ny) // 2):
        t *= 2
    return max(t, 64)


def prepare_residual(resid, rms, zero, info, args):
    hp_px = max(args.highpass_arcsec / info.pix_arcsec, 3.0)
    x = np.where(np.isfinite(resid) & ~zero, resid, 0.0).astype(np.float32)
    x = x - gaussian_filter(x, hp_px)
    x[np.abs(x) > 5 * rms] = 0.0
    x[zero] = 0.0
    return x


def tile_power(x, info, tile, min_valid):
    w = np.hanning(tile)
    win = np.outer(w, w)
    total = np.zeros((tile, tile))
    count = 0
    for y0 in range(0, info.ny - tile + 1, tile // 2):
        for x0 in range(0, info.nx - tile + 1, tile // 2):
            t = x[y0 : y0 + tile, x0 : x0 + tile]
            if (t != 0).mean() < min_valid:
                continue
            total += np.abs(np.fft.fftshift(np.fft.fft2(t * win))) ** 2
            count += 1
    return total, count


def radial_mean(power):
    n = power.shape[0]
    cy = n // 2
    yy, xx = np.indices(power.shape)
    ri = np.hypot(yy - cy, xx - cy).astype(int)
    prof = np.bincount(ri.ravel(), power.ravel()) / np.maximum(np.bincount(ri.ravel()), 1)
    return ri, prof


class RippleBand:
    """Ripple band in the image FFT plane (axes kx, ky in cycles/rad = lambda)."""

    def __init__(self):
        self.P = None
        self.Pw = None
        self.band = None
        self.axis = None
        self.df = None
        self.tile = 0
        self.ntiles = 0
        self.peaks = []
        self.rmin = 0.0
        self.rmax = 0.0
        self.components = 0
        self.extent = (0.0, 0.0)


def detect_band(resid_prepared, info, args, uv_max_data):
    tile = tile_size_for(info, args.tile_size)
    tp, count = None, 0
    for valid in (0.7, 0.5, 0.3):
        tp, count = tile_power(resid_prepared, info, tile, valid)
        if count >= 4:
            break
    if count < 1:
        raise FinderError("No usable image tiles for the ripple analysis (too much bright emission masked?)")
    P = tp / count
    rb = RippleBand()
    rb.tile, rb.ntiles = tile, count
    rb.axis = np.fft.fftshift(np.fft.fftfreq(tile, d=info.pix_rad))
    rb.df = float(rb.axis[1] - rb.axis[0])
    ri, prof = radial_mean(P)
    Pw = gaussian_filter(P, 1.5) / np.maximum(prof[ri], 1e-30)
    kr = np.hypot(*np.meshgrid(rb.axis, rb.axis))
    nyq = 1.0 / (2.0 * info.pix_rad)
    rb.rmin = args.band_uv_min if args.band_uv_min else 6 * rb.df
    rb.rmax = args.band_uv_max if args.band_uv_max else min(uv_max_data, nyq / 4.0)
    cand = (Pw > args.band_threshold) & (kr > rb.rmin) & (kr < rb.rmax)
    lab, nlab = label(cand)
    keep = np.zeros_like(cand)
    excess = []
    for i in range(1, nlab + 1):
        m = lab == i
        if m.sum() >= args.band_min_cells:
            excess.append((float((Pw[m] - 1).sum()), i))
    if excess:
        top = max(e for e, _ in excess)
        for e, i in excess:
            if e >= 0.1 * top:
                keep |= lab == i
    rb.P, rb.Pw, rb.band = P, Pw, keep
    rb.extent = (float(kr[keep].min()), float(kr[keep].max())) if keep.any() else (0.0, 0.0)
    rb.components = len([1 for e, _ in excess if excess and e >= 0.1 * max(x for x, _ in excess)])
    if keep.any():
        idx = np.argsort(np.where(keep, Pw, 0).ravel())[::-1]
        used = []
        for k in idx[:400]:
            y, x = np.unravel_index(k, Pw.shape)
            kx, ky = rb.axis[x], rb.axis[y]
            if kx < 0 or (kx == 0 and ky < 0):
                continue
            if any(math.hypot(kx - a, ky - b) < 4 * rb.df for a, b, _ in used):
                continue
            used.append((kx, ky, float(Pw[y, x])))
            if len(used) >= 6:
                break
        rb.peaks = used
    return rb


def fft_masks(rb, info, shape):
    """Band and control-band masks on the full image FFT grid (fftshifted)."""
    ny, nx = shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=info.pix_rad))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=info.pix_rad))
    KX, KY = np.meshgrid(kx, ky)
    ix = (KX - rb.axis[0]) / rb.df
    iy = (KY - rb.axis[0]) / rb.df
    kr = np.hypot(KX, KY)
    annulus = (kr > rb.rmin) & (kr < rb.rmax)
    inside = map_coordinates(rb.band.astype(np.float32), [iy, ix], order=1, cval=0) > 0.5
    band = gaussian_filter((inside & annulus).astype(np.float32), 2.0)
    ctl = gaussian_filter((~inside & annulus).astype(np.float32), 2.0)
    return band, ctl


def band_filter(image, zero, rms, info, args, masks):
    band, ctl = masks
    x = prepare_residual(image, rms, zero, info, args)
    fx = np.fft.fft2(x)
    b = np.fft.ifft2(fx * np.fft.ifftshift(band)).real
    c = np.fft.ifft2(fx * np.fft.ifftshift(ctl)).real
    return b.astype(np.float32), c.astype(np.float32)


def band_power(band_img, ctl_img, keep):
    return float(np.sqrt(np.mean(band_img[keep] ** 2))), float(np.sqrt(np.mean(ctl_img[keep] ** 2)))


# ----------------------------------------------------------------------------
# Measurement Set scan
# ----------------------------------------------------------------------------
def mjd_seconds_to_str(t: float) -> str:
    return (datetime.datetime(1858, 11, 17) + datetime.timedelta(seconds=float(t))).strftime("%Y-%m-%d %H:%M:%S")


def ms_overview(ms, args):
    from casacore.tables import table

    t = table(ms, ack=False)
    cols = t.colnames()
    if args.data_column not in cols:
        raise FinderError(f"Column {args.data_column} not found in {ms}")
    ant = table(ms + "/ANTENNA", ack=False)
    names = [str(n) for n in ant.getcol("NAME")]
    ant.close()
    field = t.getcol("FIELD_ID")
    if args.field_id is not None:
        fid = args.field_id
    else:
        vals, counts = np.unique(field, return_counts=True)
        fid = int(vals[np.argmax(counts)])
        if len(vals) > 1:
            log(f"Multiple fields present; using FIELD_ID {fid} (use --field-id to choose)")
    sel = t.query(f"FIELD_ID=={fid} AND ANTENNA1!=ANTENNA2")
    if sel.nrows() == 0:
        raise FinderError("No cross-correlation rows for the selected field")
    fld = table(ms + "/FIELD", ack=False)
    phase_dir = np.degrees(fld.getcol("PHASE_DIR")[fid, 0])
    fld.close()
    return t, sel, names, cols, fid, phase_dir


def ddid_info(ms, ddid):
    from casacore.tables import table

    dd = table(ms + "/DATA_DESCRIPTION", ack=False)
    spw, pol = int(dd.getcol("SPECTRAL_WINDOW_ID")[ddid]), int(dd.getcol("POLARIZATION_ID")[ddid])
    dd.close()
    s = table(ms + "/SPECTRAL_WINDOW", ack=False)
    freq = np.asarray(s.getcell("CHAN_FREQ", spw), dtype=np.float64)
    s.close()
    p = table(ms + "/POLARIZATION", ack=False)
    corr = [int(c) for c in p.getcell("CORR_TYPE", pol)]
    p.close()
    sel = [i for i, c in enumerate(corr) if c in (5, 8, 9, 12)]
    if not sel:
        sel = [i for i, c in enumerate(corr) if c == 1]
    if not sel:
        sel = [0]
    return freq, sel


def choose_weight_mode(cols, requested):
    if requested == "none":
        return None
    if requested != "auto":
        if requested not in cols:
            raise FinderError(f"Weight column {requested} not present")
        return requested
    for c in ("WEIGHT_SPECTRUM", "WEIGHT"):
        if c in cols:
            return c
    return None


def read_visibility_chunk(sub, s, n, args, wcol, pol_sel):
    r = np.asarray(sub.getcol(args.data_column, startrow=s, nrow=n))[:, :, pol_sel]
    f = np.asarray(sub.getcol("FLAG", startrow=s, nrow=n))[:, :, pol_sel]
    w = None
    if wcol == "WEIGHT_SPECTRUM":
        try:
            w = np.asarray(sub.getcol("WEIGHT_SPECTRUM", startrow=s, nrow=n))[:, :, pol_sel]
        except RuntimeError:
            w = None
    if w is None and wcol is not None:
        w = np.asarray(sub.getcol("WEIGHT", startrow=s, nrow=n))[:, pol_sel][:, None, :] * np.ones((1, r.shape[1], 1), np.float32)
    if w is None:
        w = np.ones(r.shape, np.float32)
    bad = f | ~np.isfinite(r) | (w <= 0)
    w = np.where(bad, 0.0, w).astype(np.float32)
    r = np.where(bad, 0.0, r)
    wt = w.sum(axis=2)
    ri = (w * r).sum(axis=2) / np.maximum(wt, 1e-30)
    return ri.astype(np.complex64), wt.astype(np.float32)


def average_channels(ri, wt, starts):
    if len(starts) == ri.shape[1]:
        return ri, wt
    wa = np.add.reduceat(wt, starts, axis=1)
    num = np.add.reduceat(wt * ri, starts, axis=1)
    return (num / np.maximum(wa, 1e-30)).astype(np.complex64), wa.astype(np.float32)


class ScanResult:
    pass


def uv_grid_setup(band_rmax, cell):
    nu = int(math.ceil(band_rmax / cell)) + 1
    return nu, 2 * nu


def band_mask_on_uv_grid(rb, info, su, sv, cell, nu, nv):
    u = (np.arange(nu) + 0.5) * cell
    v = (np.arange(nv) - nv // 2 + 0.5) * cell
    U, V = np.meshgrid(u, v)
    ix = (su * U - rb.axis[0]) / rb.df
    iy = (sv * V - rb.axis[0]) / rb.df
    return map_coordinates(rb.band.astype(np.float32), [iy, ix], order=1, cval=0) > 0.5


def scan_ms(ms, args, rb, info, su, sv, cell, hole_cell, chan_avg_arg):
    """One pass over the MS: baseline statistics, uv flag map, ripple-band samples."""
    t, sel, names, cols, fid, phase_dir = ms_overview(ms, args)
    nant = len(names)
    a1_all = sel.getcol("ANTENNA1")
    a2_all = sel.getcol("ANTENNA2")
    time_all = sel.getcol("TIME")
    dd_all = sel.getcol("DATA_DESC_ID")
    pair_key = a1_all.astype(np.int64) * nant + a2_all
    ukeys, pair_inv = np.unique(pair_key, return_inverse=True)
    nbl = len(ukeys)
    pair_map = -np.ones(nant * nant, np.int64)
    pair_map[ukeys] = np.arange(nbl)
    bl_a1, bl_a2 = ukeys // nant, ukeys % nant
    utimes, slot_all = np.unique(time_all, return_inverse=True)
    nslots = len(utimes)
    monotonic = bool(np.all(np.diff(time_all) >= 0))
    if not monotonic:
        log("WARNING: TIME is not monotonic; time-difference noise estimates are disabled")
    NTB = max(2, min(args.time_blocks, nslots))
    wcol = choose_weight_mode(cols, args.weight_column)

    res = ScanResult()
    res.names, res.nant, res.nbl, res.bl_a1, res.bl_a2 = names, nant, nbl, bl_a1, bl_a2
    res.pair_map, res.utimes, res.fid, res.phase_dir = pair_map, utimes, fid, phase_dir
    res.NTB = NTB
    res.slot_to_block = lambda s: np.minimum(np.asarray(s) * NTB // max(nslots, 1), NTB - 1)

    ddids = np.unique(dd_all)
    freq0, _ = ddid_info(ms, int(ddids[0]))
    nch0 = len(freq0)
    avg = chan_avg_arg if chan_avg_arg > 0 else max(1, int(math.ceil(nch0 / 128.0)))
    starts0 = np.arange(0, nch0, avg)
    ncha0 = len(starts0)
    NCB = max(2, min(args.freq_blocks, ncha0))
    cb_starts = np.linspace(0, ncha0, NCB + 1).astype(int)[:-1]
    res.avg, res.NCB, res.cb_starts, res.nch0, res.freq0 = avg, NCB, cb_starts, nch0, freq0
    edges = list(starts0[cb_starts]) + [nch0]
    res.cb_freq = np.array([freq0[edges[j] : edges[j + 1]].mean() for j in range(NCB)])
    res.cb_chan_edges = np.array(edges)

    shape3 = (nbl, NTB, NCB)
    Zs, Zd, Nd = (np.zeros(shape3) for _ in range(3))
    Zsf, Zf, Nf = (np.zeros(shape3) for _ in range(3))
    flag_n = np.zeros(nbl)
    tot_n = np.zeros(nbl)
    len_sum = np.zeros(nbl)
    len_n = np.zeros(nbl)

    nu, nv = uv_grid_setup(rb.rmax, cell)
    ms_band = band_mask_on_uv_grid(rb, info, su, sv, cell, nu, nv)
    nuh = int(math.ceil(rb.rmax / hole_cell)) + 1
    nvh = 2 * nuh
    tot_h = np.zeros(nuh * nvh)
    flg_h = np.zeros(nuh * nvh)
    flg_ant = np.zeros(nant * nuh * nvh)
    samples = {k: [] for k in ("cell", "bl", "slot", "ch", "w", "r")}
    n_samples = 0

    # clip level from a spread-out subsample
    ri_s = []
    sub0 = sel.query(f"DATA_DESC_ID=={int(ddids[0])}")
    freq_d, pol_sel = ddid_info(ms, int(ddids[0]))
    nrow0 = sub0.nrows()
    for off in np.linspace(0, max(nrow0 - 200, 0), 8).astype(int):
        ri, wt = read_visibility_chunk(sub0, int(off), min(200, nrow0), args, wcol, pol_sel)
        z = (wt * np.abs(ri) ** 2)[wt > 0]
        ri_s.append(z)
    zmed = float(np.median(np.concatenate(ri_s))) if ri_s else 1.0
    clip = args.clip_factor * zmed
    res.zmedian, res.clip = zmed, clip
    sub0.close()

    t0 = time.time()
    done_rows = 0
    total_rows = sel.nrows()
    cstarts_cb = cb_starts
    for dd in ddids:
        freq, pol_sel = ddid_info(ms, int(dd))
        if len(freq) != nch0:
            log(f"WARNING: spectral window of DATA_DESC_ID {dd} has a different channel count; skipped")
            continue
        sub = sel.query(f"DATA_DESC_ID=={int(dd)}")
        rows_dd = np.nonzero(dd_all == dd)[0]
        nrow = sub.nrows()
        chunk = max(500, int(args.chunk_elements / (nch0 * 4)))
        starts = np.arange(0, nch0, avg)
        f_avg = np.add.reduceat(freq, starts) / np.diff(np.r_[starts, nch0])
        prev_ri = np.zeros((nbl, ncha0), np.complex64)
        prev_wt = np.zeros((nbl, ncha0), np.float32)
        prev_slot = -10 * np.ones(nbl, np.int64)
        for s in range(0, nrow, chunk):
            n = min(chunk, nrow - s)
            ridx = rows_dd[s : s + n]
            a1, a2 = a1_all[ridx], a2_all[ridx]
            bl = pair_inv[ridx]
            slot = slot_all[ridx]
            uvw = np.asarray(sub.getcol("UVW", startrow=s, nrow=n))
            ri, wt = read_visibility_chunk(sub, s, n, args, wcol, pol_sel)
            ri, wt = average_channels(ri, wt, starts)
            valid = wt > 0
            row_ok = valid.any(axis=1)

            # baseline length and flag statistics (fully flagged integrations excluded)
            np.add.at(len_sum, bl, np.linalg.norm(uvw, axis=1))
            np.add.at(len_n, bl, 1)
            np.add.at(tot_n, bl[row_ok], valid.shape[1])
            np.add.at(flag_n, bl[row_ok], (~valid[row_ok]).sum(axis=1))

            # uv coordinates, folded into u >= 0
            u = uvw[:, 0, None] * f_avg[None, :] / C_LIGHT
            v = uvw[:, 1, None] * f_avg[None, :] / C_LIGHT
            neg = u < 0
            ri_f = np.where(neg, np.conj(ri), ri)
            u = np.where(neg, -u, u)
            v = np.where(neg, -v, v)

            # hole map (coarse), per-antenna flagged counts
            iu = np.floor(u / hole_cell).astype(np.int64)
            iv = np.floor(v / hole_cell).astype(np.int64) + nvh // 2
            inr = row_ok[:, None] & (iu < nuh) & (iv >= 0) & (iv < nvh)
            cid = (iv * nuh + iu)[inr]
            fl = (~valid)[inr]
            tot_h += np.bincount(cid, minlength=nuh * nvh)
            flg_h += np.bincount(cid[fl], minlength=nuh * nvh)
            rr, _ = np.nonzero(inr)
            for aa in (a1, a2):
                flg_ant += np.bincount(aa[rr][fl] * (nuh * nvh) + cid[fl], minlength=nant * nuh * nvh)

            # ripple-band samples (fine grid)
            iu2 = np.floor(u / cell).astype(np.int64)
            iv2 = np.floor(v / cell).astype(np.int64) + nv // 2
            ok2 = valid & (iu2 < nu) & (iv2 >= 0) & (iv2 < nv)
            iu2 = np.where(ok2, iu2, 0)
            iv2 = np.where(ok2, iv2, 0)
            selb = ok2 & ms_band[iv2, iu2]
            rrow, cch = np.nonzero(selb)
            if len(rrow) and n_samples < args.max_ripple_samples:
                samples["cell"].append((iv2 * nu + iu2)[selb].astype(np.int32))
                samples["bl"].append(bl[rrow].astype(np.int32))
                samples["slot"].append(slot[rrow].astype(np.int32))
                samples["ch"].append(cch.astype(np.int16))
                samples["w"].append(wt[selb].astype(np.float32))
                samples["r"].append(ri_f[selb].astype(np.complex64))
                n_samples += len(rrow)

            # time-difference and frequency-difference noise
            bounds = np.flatnonzero(np.diff(slot)) + 1
            for a, b in zip(np.r_[0, bounds], np.r_[bounds, n]):
                sl = int(slot[a])
                ids = bl[a:b]
                tb = int(res.slot_to_block(sl))
                r1, w1 = ri[a:b], wt[a:b]
                if monotonic:
                    hp = prev_slot[ids] == sl - 1
                    if hp.any():
                        i2 = ids[hp]
                        r0, w0, ra, wa = prev_ri[i2], prev_wt[i2], r1[hp], w1[hp]
                        ok = (w0 > 0) & (wa > 0)
                        zs = 0.5 * (w0 * np.abs(r0) ** 2 + wa * np.abs(ra) ** 2)
                        zd = 0.5 * np.minimum(w0, wa) * np.abs(r0 - ra) ** 2
                        zs = np.where(ok, np.minimum(zs, clip), 0)
                        zd = np.where(ok, np.minimum(zd, clip), 0)
                        Zs[i2, tb] += np.add.reduceat(zs, cstarts_cb, axis=1)
                        Zd[i2, tb] += np.add.reduceat(zd, cstarts_cb, axis=1)
                        Nd[i2, tb] += np.add.reduceat(ok.astype(np.float64), cstarts_cb, axis=1)
                    prev_ri[ids], prev_wt[ids], prev_slot[ids] = r1, w1, sl
                okf = (w1[:, 1:] > 0) & (w1[:, :-1] > 0)
                zsf = 0.5 * (w1[:, 1:] * np.abs(r1[:, 1:]) ** 2 + w1[:, :-1] * np.abs(r1[:, :-1]) ** 2)
                zf = 0.5 * np.minimum(w1[:, 1:], w1[:, :-1]) * np.abs(r1[:, 1:] - r1[:, :-1]) ** 2
                pad = np.zeros((len(ids), 1))
                zsf = np.hstack([np.where(okf, np.minimum(zsf, clip), 0), pad])
                zf = np.hstack([np.where(okf, np.minimum(zf, clip), 0), pad])
                nf = np.hstack([okf.astype(np.float64), pad])
                Zsf[ids, tb] += np.add.reduceat(zsf, cstarts_cb, axis=1)
                Zf[ids, tb] += np.add.reduceat(zf, cstarts_cb, axis=1)
                Nf[ids, tb] += np.add.reduceat(nf, cstarts_cb, axis=1)
            done_rows += n
        sub.close()
        log(f"  DATA_DESC_ID {int(dd)} scanned ({done_rows}/{total_rows} rows, {time.time() - t0:.0f}s)")

    res.Zs, res.Zd, res.Nd, res.Zsf, res.Zf, res.Nf = Zs, Zd, Nd, Zsf, Zf, Nf
    res.flag_frac = flag_n / np.maximum(tot_n, 1)
    res.length = len_sum / np.maximum(len_n, 1)
    res.cell, res.nu, res.nv, res.hole_cell, res.nuh, res.nvh = cell, nu, nv, hole_cell, nuh, nvh
    res.tot_h = tot_h.reshape(nvh, nuh)
    res.flg_h = flg_h.reshape(nvh, nuh)
    res.flg_ant = flg_ant.reshape(nant, nvh, nuh)
    res.ms_band = ms_band
    cat = lambda k, dt: np.concatenate(samples[k]) if samples[k] else np.zeros(0, dt)
    res.s_cell, res.s_bl, res.s_slot = cat("cell", np.int32), cat("bl", np.int32), cat("slot", np.int32)
    res.s_ch, res.s_w, res.s_r = cat("ch", np.int16), cat("w", np.float32), cat("r", np.complex64)
    res.n_samples_capped = n_samples >= args.max_ripple_samples
    res.nslots = nslots
    t.close()
    return res


# ----------------------------------------------------------------------------
# Analysis of the scan
# ----------------------------------------------------------------------------
def length_median(x, length, window=31):
    out = np.full(len(x), np.nan)
    ok = np.nonzero(np.isfinite(x))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(length[ok])]
    for i, k in enumerate(order):
        lo, hi = max(0, i - window // 2), min(len(order), i + window // 2 + 1)
        out[k] = np.median(x[order[lo:hi]])
    return out


def pair_name(res, b):
    return f"{res.names[res.bl_a1[b]]}-{res.names[res.bl_a2[b]]}"


def baseline_table(res, args):
    nd = res.Nd.sum((1, 2))
    nf = res.Nf.sum((1, 2))
    valid = (nd > 20) & (nf > 20)
    div = lambda a, b: np.where(b > 0, a / np.maximum(b, 1e-30), np.nan)
    bt = {"valid": valid, "length": res.length, "nd": nd}
    bt["tot"] = np.where(valid, div(res.Zs.sum((1, 2)), nd), np.nan)
    bt["tdn"] = np.where(valid, div(res.Zd.sum((1, 2)), nd), np.nan)
    bt["totf"] = np.where(valid, div(res.Zsf.sum((1, 2)), nf), np.nan)
    bt["fdn"] = np.where(valid, div(res.Zf.sum((1, 2)), nf), np.nan)
    bt["sys_t"] = bt["tot"] - bt["tdn"]
    bt["sys_f"] = bt["totf"] - bt["fdn"]
    longb = valid & (res.length >= args.min_baseline_length)
    bt["long"] = longb
    sys_use = np.where(longb, bt["sys_t"], np.nan)
    bt["ex_t"] = bt["sys_t"] - length_median(sys_use, res.length)
    bt["ex_f"] = bt["sys_f"] - length_median(np.where(longb, bt["sys_f"], np.nan), res.length)
    typical = float(np.nanmedian(bt["tot"][longb])) if longb.any() else 1.0
    ex = bt["ex_t"][longb]
    center = float(np.median(ex)) if len(ex) else 0.0
    scale = max(1.4826 * float(np.median(np.abs(ex - center))) if len(ex) else 0.0, 0.02 * typical)
    bt["zA"] = np.where(longb, (bt["ex_t"] - center) / scale, np.nan)
    bt["typical"], bt["ex_scale"] = typical, scale
    bt["noise_rel"] = bt["tdn"] / length_median(np.where(longb, bt["tdn"], np.nan), res.length)
    bt["noise_rel_f"] = bt["fdn"] / length_median(np.where(longb, bt["fdn"], np.nan), res.length)
    return bt


def antenna_table(res, bt, args):
    rows = []
    for a in range(res.nant):
        m = (res.bl_a1 == a) | (res.bl_a2 == a)
        mv = m & bt["valid"]
        ml = m & bt["long"]
        if mv.sum() == 0:
            rows.append(dict(ant=a, name=res.names[a], n=0))
            continue
        rows.append(
            dict(
                ant=a,
                name=res.names[a],
                n=int(mv.sum()),
                noise_rel=float(np.nanmedian(bt["noise_rel"][mv])),
                noise_rel_f=float(np.nanmedian(bt["noise_rel_f"][mv])),
                frac_out=float(np.mean(bt["zA"][ml] >= 3)) if ml.any() else 0.0,
                mean_ex=float(np.nanmean(bt["ex_t"][ml])) if ml.any() else 0.0,
                flag_frac=float(np.average(res.flag_frac[m])),
            )
        )
    return rows


def ripple_attribution(res):
    out = {"valid": False}
    if len(res.s_w) < 100:
        return out
    uc, inv = np.unique(res.s_cell, return_inverse=True)
    w, r = res.s_w.astype(np.float64), res.s_r
    wr = w * r
    sc = np.bincount(inv, wr.real) + 1j * np.bincount(inv, wr.imag)
    a = (wr * np.conj(sc[inv] - wr)).real
    total_e = float(a.sum())
    nbl = res.nbl
    eb = np.bincount(res.s_bl, a, minlength=nbl)
    wb = np.bincount(res.s_bl, w, minlength=nbl)
    nb = np.bincount(res.s_bl, minlength=nbl)
    inc = float((w**2 * np.abs(r) ** 2).sum())
    out.update(valid=total_e > 0, E=total_e, incoherent=inc, n=len(w), ncells=len(uc), Eb=eb, Wb=wb, nb=nb, a=a, inv=inv, cells=uc)
    if total_e <= 0:
        return out
    out["Eshare"] = eb / total_e
    out["Wshare"] = wb / w.sum()
    out["ratio"] = np.where(out["Wshare"] > 0, out["Eshare"] / np.maximum(out["Wshare"], 1e-30), 0.0)
    # antenna level: coherent power per w^2 for samples involving p versus not
    a1, a2 = res.bl_a1[res.s_bl], res.bl_a2[res.s_bl]
    ncell = len(uc)
    ant_rows = []
    for p in range(res.nant):
        mp = (a1 == p) | (a2 == p)
        share_w = float(w[mp].sum() / w.sum())
        ea = float(a[mp].sum() / (2 * total_e))
        q_in = q_out = np.nan
        if share_w >= 0.01 and share_w <= 0.99:
            for m, key in ((mp, "in"), (~mp, "out")):
                s = np.bincount(inv[m], wr[m].real, minlength=ncell) + 1j * np.bincount(inv[m], wr[m].imag, minlength=ncell)
                wc = np.bincount(inv[m], w[m], minlength=ncell)
                ic = np.bincount(inv[m], (w[m] ** 2) * np.abs(r[m]) ** 2, minlength=ncell)
                g = wc > 0
                val = float((np.abs(s) ** 2 - ic)[g].sum() / (wc[g] ** 2).sum())
                if key == "in":
                    q_in = val
                else:
                    q_out = val
        ant_rows.append(dict(ant=p, name=res.names[p], Wshare=share_w / 2, Eshare=ea, q_in=q_in, q_out=q_out))
    out["ant"] = ant_rows
    return out


def time_freq_marginals(res, att, b):
    m = res.s_bl == b
    tb = res.slot_to_block(res.s_slot[m])
    cb = np.clip(np.searchsorted(res.cb_starts, res.s_ch[m], side="right") - 1, 0, res.NCB - 1)
    a = att["a"][m]
    return np.bincount(tb, a, minlength=res.NTB) / att["E"], np.bincount(cb, a, minlength=res.NCB) / att["E"]


def contiguous_range(marginal, fraction=0.5):
    m = np.asarray(marginal, float)
    if not np.any(m > 0):
        return None
    sel = np.nonzero(m >= fraction * m.max())[0]
    return int(sel.min()), int(sel.max())


def block_to_time(res, blk0, blk1):
    s0 = int(math.ceil(blk0 * res.nslots / res.NTB))
    s1 = int(math.ceil((blk1 + 1) * res.nslots / res.NTB)) - 1
    return float(res.utimes[min(s0, res.nslots - 1)]), float(res.utimes[min(max(s1, s0), res.nslots - 1)])


def block_to_channels(res, c0, c1):
    ch0 = int(res.cb_chan_edges[c0])
    ch1 = min(int(res.cb_chan_edges[c1 + 1]) - 1, res.nch0 - 1)
    return ch0, ch1


def excess_map(res, b, bt):
    with np.errstate(invalid="ignore", divide="ignore"):
        m_all = (res.Zs - res.Zd) / np.where(res.Nd > 5, res.Nd, np.nan)
    L = res.length[b]
    near = np.nonzero((np.abs(res.length - L) < 0.3 * L) & (np.arange(res.nbl) != b) & bt["valid"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(m_all[near], axis=0) if len(near) >= 3 else np.nanmedian(m_all[bt["valid"]], axis=0)
    return m_all[b] - ref, res.Nd[b]


def excess_window(res, b, bt, att=None):
    ex, nd = excess_map(res, b, bt)
    wgt = np.where(np.isfinite(ex), nd, 0)
    pos = np.where(np.isfinite(ex), np.clip(ex, 0, None), 0) * wgt
    out = {"map": ex, "nd": nd, "window": None}
    if pos.sum() <= 0:
        return out
    tm, fm = pos.sum(1), pos.sum(0)
    tr, fr_ = contiguous_range(tm), contiguous_range(fm)
    box = pos[tr[0] : tr[1] + 1, fr_[0] : fr_[1] + 1].sum() / pos.sum()
    area = (tr[1] - tr[0] + 1) * (fr_[1] - fr_[0] + 1) / float(res.NTB * res.NCB)
    t0, t1 = block_to_time(res, *tr)
    ch0, ch1 = block_to_channels(res, *fr_)
    out["window"] = dict(
        blocks_t=tr, blocks_f=fr_, captured=float(box), area=float(area), t0=t0, t1=t1, ch0=ch0, ch1=ch1,
        f0=float(res.freq0[ch0]), f1=float(res.freq0[ch1]), localized=bool(box >= 0.6 and area <= 0.6),
    )
    return out


def ripple_window(res, att, b):
    """Time/frequency box holding most of the ripple-band coherent excess of baseline b."""
    if not att.get("valid"):
        return None
    m = res.s_bl == b
    if m.sum() < 50:
        return None
    tb = res.slot_to_block(res.s_slot[m])
    cb = np.clip(np.searchsorted(res.cb_starts, res.s_ch[m], side="right") - 1, 0, res.NCB - 1)
    grid = np.bincount(tb * res.NCB + cb, att["a"][m], minlength=res.NTB * res.NCB).reshape(res.NTB, res.NCB)
    pos = np.clip(grid, 0, None)
    if pos.sum() <= 0:
        return None
    tr, fr_ = contiguous_range(pos.sum(1), 0.3), contiguous_range(pos.sum(0), 0.3)
    box = pos[tr[0] : tr[1] + 1, fr_[0] : fr_[1] + 1].sum() / pos.sum()
    area = (tr[1] - tr[0] + 1) * (fr_[1] - fr_[0] + 1) / float(res.NTB * res.NCB)
    t0, t1 = block_to_time(res, *tr)
    ch0, ch1 = block_to_channels(res, *fr_)
    return dict(blocks_t=tr, blocks_f=fr_, captured=float(box), area=float(area), t0=t0, t1=t1, ch0=ch0, ch1=ch1,
                f0=float(res.freq0[ch0]), f1=float(res.freq0[ch1]), localized=bool(box >= 0.6 and area <= 0.5))


def find_holes(res, rb, args):
    cnt = res.tot_h
    frac = np.where(cnt > 0, res.flg_h / np.maximum(cnt, 1), 0)
    pos = cnt[cnt > 0]
    if len(pos) == 0:
        return []
    mask = (frac > 0.9) & (cnt >= 0.2 * np.median(pos))
    lab, nlab = label(mask)
    uu = (np.arange(res.nuh) + 0.5) * res.hole_cell
    vv = (np.arange(res.nvh) - res.nvh // 2 + 0.5) * res.hole_cell
    bu = (np.arange(res.nu) + 0.5) * res.cell
    bv = (np.arange(res.nv) - res.nv // 2 + 0.5) * res.cell
    BU, BV = np.meshgrid(bu, bv)
    bcells = np.c_[BU[res.ms_band], BV[res.ms_band]] if res.ms_band.any() else np.zeros((0, 2))
    holes = []
    for i in range(1, nlab + 1):
        m = lab == i
        if m.sum() < 4:
            continue
        iy, ix = np.nonzero(m)
        cu, cv = uu[ix].mean(), vv[iy].mean()
        radius = math.sqrt(m.sum() / math.pi) * res.hole_cell
        flagged = float(res.flg_h[m].sum())
        dist = float(np.min(np.hypot(bcells[:, 0] - cu, bcells[:, 1] - cv))) if len(bcells) else np.nan
        per_ant = res.flg_ant[:, m].sum(axis=1)
        top = np.argsort(per_ant)[::-1][:5]
        holes.append(
            dict(
                u=float(cu), v=float(cv), radius=float(radius), cells=int(m.sum()), flagged=flagged,
                flag_fraction=float(res.flg_h[m].sum() / res.tot_h[m].sum()), dist_to_band=dist,
                adjacent=bool(np.isfinite(dist) and dist < 1.5 * radius + 2 * res.cell),
                antennas=[(res.names[a], float(per_ant[a] / max(per_ant.sum(), 1) * 2)) for a in top],
                mask=m,
            )
        )
    holes.sort(key=lambda h: -h["flagged"])
    return holes[:5]


def select_candidates(res, bt, at, att, args):
    cands, notes = [], []
    zthr = args.z_threshold
    for b in range(res.nbl):
        zA = bt["zA"][b]
        pass_a = bool(np.isfinite(zA) and zA >= zthr and bt["ex_t"][b] >= 0.15 * bt["typical"] and (not att.get("valid") or att["Wshare"][b] >= 0.002))
        ratio = share = 0.0
        pass_c = False
        if att.get("valid"):
            share, ratio = float(att["Eshare"][b]), float(att["ratio"][b])
            pass_c = share >= 0.03 and ratio >= 3.0
        score = (zA / zthr if pass_a else 0.0) + (min(ratio, 30.0) / 3.0 if pass_c else 0.0)
        weak = (zA / zthr if np.isfinite(zA) and zA > 0 else 0.0) + (min(ratio, 30.0) / 3.0 if share > 0.01 else 0.0) * 0.1
        cands.append(dict(b=b, name=pair_name(res, b), zA=float(zA) if np.isfinite(zA) else np.nan, ex=float(bt["ex_t"][b]),
                          Eshare=share, ratio=ratio, pass_a=pass_a, pass_c=pass_c, score=score, weak=weak))
    strong = sorted([c for c in cands if c["score"] > 0], key=lambda c: -c["score"])
    chosen = strong[: args.max_baseline_candidates]
    if len(chosen) < 2:
        extra = sorted([c for c in cands if c["score"] <= 0 and c["weak"] > 0], key=lambda c: -c["weak"])
        for c in extra[: 2 - len(chosen)]:
            c["weak_candidate"] = True
            chosen.append(c)
        if chosen:
            notes.append("Few or no baselines pass the outlier criteria; the weakest-evidence candidates were still tested.")
    names = {c["name"]: c for c in cands}
    for tok in [t for t in args.extra_baselines.split(",") if t]:
        p, q = tok.split("-")
        key = f"{p}-{q}" if f"{p}-{q}" in names else f"{q}-{p}"
        if key in names and names[key] not in chosen:
            names[key]["user"] = True
            chosen.append(names[key])
    ant_c = []
    band_w = {r["name"]: r["Wshare"] for r in att.get("ant", [])}
    for row in at:
        if row.get("n", 0) and row["frac_out"] >= 0.3 and (not att.get("valid") or band_w.get(row["name"], 0) >= 0.01):
            ant_c.append(dict(ant=row["ant"], name=row["name"], frac_out=row["frac_out"], mean_ex=row["mean_ex"]))
    ant_c = sorted(ant_c, key=lambda r: -(r["frac_out"] * max(r["mean_ex"], 0)))[: args.max_antenna_candidates]
    # the antennas of a clearly bad baseline (outlier and ripple-band contribution) are tested as well
    if not args.no_baseline_antennas:
        clear = [c for c in chosen if c["pass_a"] and c["pass_c"]][:2]
        for c in clear:
            for name in c["name"].split("-"):
                ex = next((a for a in ant_c if a["name"] == name), None)
                if ex is not None:
                    ex.setdefault("from_baseline", c["name"])
                    continue
                row = next((r for r in at if r.get("name") == name and r.get("n")), None)
                ant_c.append(dict(ant=res.names.index(name), name=name, frac_out=row["frac_out"] if row else np.nan,
                                  mean_ex=row["mean_ex"] if row else np.nan, from_baseline=c["name"]))
        if clear:
            notes.append("The antennas of the clearly bad baseline(s) " + ", ".join(c["name"] for c in clear) + " were also tested as antennas, to see whether the problem is baseline-specific or antenna-wide.")
    for tok in [t for t in args.extra_antennas.split(",") if t]:
        if tok in res.names and not any(c["name"] == tok for c in ant_c):
            ant_c.append(dict(ant=res.names.index(tok), name=tok, frac_out=np.nan, mean_ex=np.nan, user=True))
    # controls: baselines carrying much weight in the ripple band but not flagged as outliers
    controls = []
    if att.get("valid") and len(chosen):
        clen = np.median([res.length[c["b"]] for c in chosen])
        excl = {c["b"] for c in chosen}
        pool = [c for c in cands if c["b"] not in excl and not (np.isfinite(c["zA"]) and c["zA"] >= zthr / 2)]
        pool.sort(key=lambda c: -att["Wshare"][c["b"]])
        similar = [c for c in pool if 0.5 * clen <= res.length[c["b"]] <= 2.0 * clen and att["Wshare"][c["b"]] > 0]
        rest = [c for c in pool if c not in similar and att["Wshare"][c["b"]] > 0]
        controls = (similar + rest)[: args.n_controls]
    if att.get("valid"):
        for a in ant_c:
            on = (res.bl_a1 == a["ant"]) | (res.bl_a2 == a["ant"])
            a["band_weight"] = float(att["Wshare"][on].sum())
    ant_controls = []
    if ant_c and att.get("valid"):
        cn = {c["name"] for c in ant_c}
        pool = [r for r in att["ant"] if r["name"] not in cn and r["Wshare"] > 0.005]
        frac = {r["name"]: r["frac_out"] for r in at if r.get("n")}
        pool = [r for r in pool if frac.get(r["name"], 0) < 0.1]
        pool.sort(key=lambda r: -r["Wshare"])
        ant_controls = pool[: args.n_antenna_controls]
    return chosen, ant_c, controls, ant_controls, cands, notes


# ----------------------------------------------------------------------------
# WSClean confirmation
# ----------------------------------------------------------------------------
_DROP_ARGS = {
    "-niter": 1, "-nmiter": 1, "-mgain": 1, "-gain": 1, "-threshold": 1, "-auto-threshold": 1,
    "-auto-mask": 1, "-parallel-deconvolution": 1, "-multiscale": 0, "-fits-mask": 1, "-casa-mask": 1,
    "-local-rms": 0, "-local-rms-window": 1, "-local-rms-method": 1, "-save-source-list": 0,
    "-fit-spectral-pol": 1, "-deconvolution-channels": 1, "-no-negative": 0, "-iuwt": 0,
    "-python-deconvolution": 1, "-deconvolution-threads": 1, "-facet-regions": 1, "-apply-facet-solutions": 2,
    "-apply-facet-beam": 0, "-facet-beam-update": 1, "-diagonal-visibilities": 0, "-diagonal-solutions": 0,
    "-data-column": 1, "-name": 1, "-temp-dir": 1, "-j": 1, "-continue": 0, "-predict": 0,
    "-save-first-residual": 0, "-save-weights": 0, "-save-uv": 0, "-update-model-required": 0,
}


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _is_option(tok: str) -> bool:
    return tok.startswith("-") and len(tok) > 1 and not _is_number(tok)


def sanitize_wsclean_command(text: str, ms_basename: str) -> list:
    """Dirty-imaging options from a WSClean command: no deconvolution, no facets/DD."""
    tokens = shlex.split(text)
    if tokens and os.path.basename(tokens[0]) == "wsclean":
        tokens = tokens[1:]
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        nargs = None
        if _is_option(t):
            if t in _DROP_ARGS:
                nargs = _DROP_ARGS[t]
            elif t.startswith("-multiscale"):
                nargs = 1
            elif t.startswith("-facet") or t.startswith("-apply-facet"):
                nargs = 1
        i += 1
        if nargs is not None:
            k = 0
            while k < nargs and i < len(tokens) and not _is_option(tokens[i]):
                i += 1
                k += 1
            continue
        out.append(t)
    while out and (os.path.basename(out[-1].rstrip("/")) == ms_basename or out[-1].rstrip("/").endswith(".ms") or os.path.isdir(out[-1])):
        out.pop()
    return out


def default_wsclean_options(info) -> list:
    return [
        "-size", str(info.nx), str(info.ny), "-scale", f"{info.pix_arcsec:.6g}arcsec", "-weight", "briggs", "0",
        "-pol", "i", "-gridder", "wgridder", "-no-update-model-required",
    ]


def wsclean_base_options(args, hdr_image, hdr_resid, info, ms_name):
    if args.wsclean_args:
        return sanitize_wsclean_command(args.wsclean_args, ms_name), "user (--wsclean-args)"
    for hdr, label_ in ((hdr_image, "image header HISTORY"), (hdr_resid, "residual image header HISTORY")):
        cmd = history_command(hdr)
        if cmd:
            return sanitize_wsclean_command(cmd, ms_name), label_
    return default_wsclean_options(info), "defaults (no WSClean command found in the FITS headers)"


def prepare_work_ms(ms, dst, data_column):
    from casacore.tables import table

    if os.path.exists(dst):
        shutil.rmtree(dst)
    t = table(ms, ack=False)
    t.copy(dst, deep=True)
    t.close()
    d = table(dst, readonly=False, ack=False)
    for col in d.colnames():
        if col == data_column:
            continue
        try:
            if d.getcoldesc(col).get("valueType") == "complex":
                d.removecols(col)
        except RuntimeError:
            pass
    d.flush()
    d.unlock()
    return d


def variant_query(v, names, dt):
    if v["kind"] == "baseline":
        i, j = sorted(names.index(a) for a in v["ants"])
        q = f"ANTENNA1=={i} AND ANTENNA2=={j}"
    elif v["kind"] == "baselineset":
        keys = sorted(1000 * min(names.index(x) for x in pr.split("-")) + max(names.index(x) for x in pr.split("-")) for pr in v["ants"])
        q = "ANTENNA1*1000+ANTENNA2 IN [" + ",".join(str(k) for k in keys) + "]"
    else:
        i = names.index(v["ants"][0])
        q = f"(ANTENNA1=={i} OR ANTENNA2=={i})"
    if v.get("time_range"):
        q += f" AND TIME>={v['time_range'][0] - 0.5 * dt!r} AND TIME<={v['time_range'][1] + 0.5 * dt!r}"
    return q


def run_variants(cfg, variants, k):
    """Run WSClean for each variant on a private working MS copy (worker k)."""
    dst = os.path.join(cfg["work_dir"], f"ms_{k}.ms")
    d = prepare_work_ms(cfg["ms"], dst, cfg["data_column"])
    results = {}
    tmp = os.path.join(cfg["work_dir"], f"tmp_{k}")
    os.makedirs(tmp, exist_ok=True)
    for v in variants:
        t0 = time.time()
        applied = None
        if v["kind"] != "reference":
            q = variant_query(v, cfg["names"], cfg["dt"])
            sel = d.query(q)
            if sel.nrows() == 0:
                sel.close()
                results[v["name"]] = None
                continue
            fl = sel.getcol("FLAG")
            orig = fl.copy()
            if v.get("chan_range"):
                fl[:, v["chan_range"][0] : v["chan_range"][1] + 1, :] = True
            else:
                fl[:] = True
            sel.putcol("FLAG", fl)
            sel.flush()
            sel.close()
            d.flush()
            d.unlock()
            applied = (q, orig)
        prefix = os.path.join(cfg["work_dir"], "img_" + re.sub(r"[^A-Za-z0-9]+", "_", v["name"]))
        cmd = cfg["wsclean"] + ["-niter", "0", "-data-column", cfg["data_column"], "-name", prefix, "-temp-dir", tmp]
        if cfg["threads"] > 0:
            cmd += ["-j", str(cfg["threads"])]
        cmd += cfg["options"] + [dst]
        with open(prefix + ".log", "w") as logf:
            rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
        found = glob.glob(prefix + "-MFS-image.fits") or glob.glob(prefix + "-image.fits")
        for f in glob.glob(prefix + "-*"):
            if f not in found and f.endswith(".fits") and not cfg["keep"]:
                os.remove(f)
        results[v["name"]] = found[0] if (rc == 0 and found) else None
        if applied:
            sel = d.query(applied[0])
            sel.putcol("FLAG", applied[1])
            sel.flush()
            sel.close()
            d.flush()
            d.unlock()
        log(f"  WSClean '{v['name']}' ({'ok' if results[v['name']] else 'FAILED, see ' + prefix + '.log'}, {time.time() - t0:.0f}s)")
    d.close()
    if not cfg["keep"]:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def worker_main(argv) -> int:
    """Entry point of a worker process: run_variants for a JSON job file."""
    job = json.loads(Path(argv[0]).read_text())
    results = run_variants(job["cfg"], job["variants"], job["k"])
    Path(argv[1]).write_text(json.dumps(results))
    return 0


def run_all_variants(cfg, variants, jobs):
    """Run the variants, in fresh worker interpreters when jobs > 1 (no fork: libraries such as MPI/UCX crash after fork)."""
    jobs = max(1, min(jobs, len(variants)))
    parts = [variants[i::jobs] for i in range(jobs)]
    if jobs == 1:
        return run_variants(cfg, parts[0], 0)
    procs = []
    for k in range(jobs):
        jf = os.path.join(cfg["work_dir"], f"job_{k}.json")
        rf = os.path.join(cfg["work_dir"], f"result_{k}.json")
        Path(jf).write_text(json.dumps(dict(cfg=cfg, variants=parts[k], k=k)))
        wlog = open(os.path.join(cfg["work_dir"], f"worker_{k}.log"), "w")
        procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_worker", jf, rf], stdout=wlog, stderr=subprocess.STDOUT), rf, wlog, k))
    out = {}
    for proc, rf, wlog, k in procs:
        rc = proc.wait()
        wlog.close()
        if rc == 0 and os.path.exists(rf):
            out.update(json.loads(Path(rf).read_text()))
        else:
            log(f"WARNING: worker {k} failed (return code {rc}); see {cfg['work_dir']}/worker_{k}.log")
    for k in range(jobs):
        log_text = Path(os.path.join(cfg["work_dir"], f"worker_{k}.log")).read_text()
        if log_text.strip():
            print(log_text, end="", flush=True)
    return out


def build_variants(res, chosen, ant_c, controls, ant_controls, windows, args, att=None, cands=None):
    vs = [dict(name="reference", kind="reference", role="reference", ants=[])]
    for c in chosen:
        vs.append(dict(name=c["name"], kind="baseline", role="candidate", ants=c["name"].split("-")))
    for a in ant_c:
        vs.append(dict(name=f"ant:{a['name']}", kind="antenna", role="candidate", ants=[a["name"]], from_baseline=a.get("from_baseline")))
    for c in controls:
        vs.append(dict(name=c["name"], kind="baseline", role="control", ants=c["name"].split("-")))
    for a in ant_controls:
        vs.append(dict(name=f"ant:{a['name']}", kind="antenna", role="control", ants=[a["name"]]))
    # an antenna that is in most ripple-band samples cannot be judged against ordinary controls: random baseline
    # sets (any baselines except the candidate baselines) that remove the same amount of ripple-band weight
    # show the pure coverage effect
    if att and att.get("valid") and cands:
        excl = {c["b"] for c in chosen}
        for a in ant_c:
            f = a.get("band_weight", 0.0)
            if f < 0.15:
                continue
            pool = [b for b in range(res.nbl) if att["Wb"][b] > 0 and b not in excl]
            for k in range(2):
                rng = np.random.default_rng(1000 * a["ant"] + k)
                sel, cum = [], 0.0
                for b in rng.permutation(pool):
                    sel.append(int(b))
                    cum += float(att["Wshare"][b])
                    if cum >= f:
                        break
                vs.append(dict(name=f"random-set[{a['name']}]#{k + 1}", kind="baselineset", role="control", ants=[pair_name(res, b) for b in sel],
                               matched_for=f"ant:{a['name']}", matched_weight=cum, target_weight=f))
    for c, w in windows:
        vs.append(dict(name=f"{c['name']}@window", kind="baseline", role="window", ants=c["name"].split("-"), time_range=(w["t0"], w["t1"]), chan_range=(w["ch0"], w["ch1"]), parent=c["name"]))
    return vs


def evaluate_variants(images, variants, resid_rms, zero, keep, info, args, masks):
    """Ripple-band rms outside bright emission for every variant image."""
    metrics = {}
    for v in variants:
        path = images.get(v["name"])
        if not path:
            continue
        img, _ = read_image(path)
        b, c = band_filter(img, zero, resid_rms, info, args, masks)
        bp, cp = band_power(b, c, keep)
        metrics[v["name"]] = dict(band=bp, ctl=cp, rms=robust_rms(img[keep]))
    if "reference" in metrics:
        r = metrics["reference"]
        for n, m in metrics.items():
            m["ratio"] = (m["band"] / r["band"]) ** 2
            m["ctl_ratio"] = (m["ctl"] / r["ctl"]) ** 2
    return metrics


def verdicts(variants, metrics):
    ctrl = [metrics[v["name"]]["ratio"] for v in variants if v["role"] == "control" and v["kind"] == "baseline" and v["name"] in metrics]
    ctrl_a = [metrics[v["name"]]["ratio"] for v in variants if v["role"] == "control" and v["kind"] == "antenna" and v["name"] in metrics]
    out = {}
    for v in variants:
        if v["role"] not in ("candidate", "window") or v["name"] not in metrics:
            continue
        r = metrics[v["name"]]["ratio"]
        pool = ctrl_a if (v["kind"] == "antenna" and len(ctrl_a) >= 2) else ctrl
        matched = [metrics[x["name"]]["ratio"] for x in variants if x.get("matched_for") == v["name"] and x["name"] in metrics]
        if matched:
            pool = matched
            med = float(np.median(pool))
            spread, z = np.nan, np.nan
            confirmed = r <= 0.85 and r <= min(pool) - 0.05
        elif len(pool) >= 3:
            med = float(np.median(pool))
            spread = max(1.4826 * float(np.median(np.abs(np.array(pool) - med))), 0.02)
            z = (r - med) / spread
            confirmed = r <= 0.85 and r <= min(pool) - 0.05
        else:
            med, spread, z = np.nan, np.nan, np.nan
            confirmed = r <= 0.7
        if r >= 1.1:
            text = "removal increases ripple power (the data partly cancel the ripple)"
        elif confirmed:
            text = "CONFIRMED: ripple power drops clearly more than for control removals"
        elif matched and r <= 0.85:
            text = f"ripple power drops, but not more than for random removal of the same amount of ripple-band weight (median {med:.2f}): a coverage effect, no evidence for an antenna error"
        elif r <= 0.9:
            text = "partial effect, not clearly above controls"
        else:
            text = "no significant effect"
        out[v["name"]] = dict(ratio=r, control_median=med, control_spread=spread, z=float(z), confirmed=bool(confirmed), text=text, n_controls=len(pool))
    # an antenna tested because of a bad baseline: does removing the whole antenna do more than removing the baseline?
    for v in variants:
        b = v.get("from_baseline")
        if v["name"] not in out or not b or b not in metrics:
            continue
        o, rb_ = out[v["name"]], metrics[b]["ratio"]
        o["baseline"], o["baseline_ratio"] = b, rb_
        o["baseline_specific"] = bool(o["ratio"] >= rb_ - 0.10)
        if o["baseline_specific"]:
            o["conclusion"] = "baseline-specific"
            o["text"] += f"; no stronger than removing only {b} ({rb_:.2f}): the problem is baseline-specific, not antenna-wide"
        elif o["confirmed"]:
            o["conclusion"] = "antenna-wide contribution"
            o["text"] += f"; clearly stronger than removing only {b} ({rb_:.2f}): the antenna contributes beyond this baseline"
        else:
            o["conclusion"] = "inconclusive (coverage effect)"
            o["text"] += f"; stronger than {b} alone ({rb_:.2f}) but not more than removing the same ripple-band weight at random"
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def downsample(a, f):
    if f <= 1:
        return a
    ny, nx = a.shape
    return a[: ny // f * f, : nx // f * f].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def crop_slices(shape, fraction):
    ny, nx = shape
    h, w = int(ny * fraction), int(nx * fraction)
    y0, x0 = (ny - h) // 2, (nx - w) // 2
    return slice(y0, y0 + h), slice(x0, x0 + w)


def show_masked(ax, img, exclude, rms, sl, title, scale=3.0, factor=None):
    a = np.where(np.isfinite(img), img, 0.0)[sl]
    m = exclude[sl].astype(np.float32)
    f = factor or max(1, int(max(a.shape) // 1200))
    a = downsample(a, f)
    m = downsample(m, f) > 0.5
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#d95f02")
    ax.imshow(np.ma.masked_where(m, a), origin="lower", cmap=cmap, vmin=-scale * rms, vmax=scale * rms, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def save_fig(fig, path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def fig_overview(path, resid, rms, exclude, band_img, keep, info, bright_thr):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    full = (slice(None), slice(None))
    show_masked(ax[0], resid, exclude, rms, full, f"Residual image (+/-3 rms, rms = {rms * 1e6:.1f} uJy/beam); orange = bright emission excluded")
    brms = float(np.sqrt(np.mean(band_img[keep] ** 2)))
    show_masked(ax[1], band_img, exclude, brms, full, f"Residual filtered to the ripple uv band (+/-3 x {brms * 1e6:.1f} uJy/beam)")
    save_fig(fig, path)


def fig_spectrum(path, rb, info, su, sv):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    lim = min(rb.axis.max(), 1.4 * rb.rmax) / 1e3
    ext = [rb.axis[0] / 1e3, rb.axis[-1] / 1e3, rb.axis[0] / 1e3, rb.axis[-1] / 1e3]
    lp = np.log10(gaussian_filter(rb.P, 1.0) + 1e-30)
    ax[0].imshow(lp, origin="lower", extent=ext, vmin=np.percentile(lp, 10), vmax=lp.max() - 0.7)
    ax[0].set_title(f"Mean tile power of the residual ({rb.ntiles} tiles of {rb.tile}px), log10")
    ax[1].imshow(rb.Pw, origin="lower", extent=ext, vmin=0.5, vmax=max(3.0, np.percentile(rb.Pw[rb.band], 90) if rb.band.any() else 3.0))
    ax[1].set_title("Power / radial mean (ripple band contour in red)")
    for a in ax:
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel("image FFT kx [klambda]")
        a.set_ylabel("image FFT ky [klambda]")
        if rb.band.any():
            a.contour(rb.axis / 1e3, rb.axis / 1e3, rb.band.astype(float), [0.5], colors="r", linewidths=0.8)
    for kx, ky, p in rb.peaks:
        ax[1].plot([kx / 1e3], [ky / 1e3], "w+", ms=10)
    save_fig(fig, path)


def fig_uv(path, res, rb, holes, su, sv):
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    frac = np.where(res.tot_h > 0, res.flg_h / np.maximum(res.tot_h, 1), np.nan)
    lim = res.nuh * res.hole_cell / 1e3
    ext = [0, lim, -lim, lim]
    im = ax[0].imshow(frac, origin="lower", extent=ext, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax[0], shrink=0.6)
    ax[0].set_title("Flagged fraction in the uv plane (fully flagged integrations excluded)")
    dens = np.log10(np.maximum(res.tot_h - res.flg_h, 0) + 1.0)
    im = ax[1].imshow(dens, origin="lower", extent=ext, vmin=0)
    plt.colorbar(im, ax=ax[1], shrink=0.6)
    ax[1].set_title("log10 unflagged samples per cell")
    bu = (np.arange(res.nu) + 0.5) * res.cell / 1e3
    bv = (np.arange(res.nv) - res.nv // 2 + 0.5) * res.cell / 1e3
    for a in ax:
        a.contour(bu, bv, res.ms_band.astype(float), [0.5], colors="r", linewidths=0.7)
        for k, h in enumerate(holes):
            a.add_patch(plt.Circle((h["u"] / 1e3, h["v"] / 1e3), h["radius"] / 1e3, fill=False, ec="w", ls="--"))
            a.text(h["u"] / 1e3, h["v"] / 1e3, f"H{k + 1}", color="w", ha="center", va="center", fontsize=9)
        a.set_xlim(0, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel("u [klambda]")
        a.set_ylabel("v [klambda]")
    save_fig(fig, path)


def fig_baselines(path, res, bt, at, chosen, args):
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    L = res.length / 1e3
    v = bt["valid"]
    ax[0, 0].scatter(L[v], bt["tot"][v], s=8, label="total w|r|^2")
    ax[0, 0].scatter(L[v], bt["tdn"][v], s=8, label="time-difference noise")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("baseline length [km]")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("Mean weighted residual power per baseline")
    lg = bt["long"]
    ax[0, 1].scatter(L[lg], bt["ex_t"][lg], s=8)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].axhline(args.z_threshold * bt["ex_scale"], color="r", ls="--", lw=0.8)
    for c in chosen:
        b = c["b"]
        if bt["valid"][b]:
            ax[0, 1].annotate(c["name"], (L[b], bt["ex_t"][b]), fontsize=8)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("baseline length [km]")
    ax[0, 1].set_title("Systematic excess (total - time-diff noise) relative to similar-length baselines; red = z threshold")
    rows = [r for r in at if r.get("n")]
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    ax[1, 0].bar(x, [r["noise_rel"] for r in rows], color="C1", label="time-diff noise")
    ax[1, 0].plot(x, [r["noise_rel_f"] for r in rows], "k.", label="freq-diff noise")
    ax[1, 0].axhline(1, color="k", lw=0.5)
    ax[1, 0].axhline(1.25, color="r", ls="--", lw=0.8)
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(names, rotation=90, fontsize=7)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("Antenna noise relative to baselines of similar length (>1.25: weights too high)")
    ax[1, 1].bar(x, [r["frac_out"] for r in rows], color="C2")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(names, rotation=90, fontsize=7)
    ax[1, 1].set_title("Fraction of long baselines with systematic excess z >= 3")
    save_fig(fig, path)


def fig_attribution(path, res, att, chosen):
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    order = np.argsort(-att["Eshare"])[:12]
    x = np.arange(len(order))
    ax[0].bar(x - 0.2, att["Eshare"][order], 0.4, label="share of coherent excess")
    ax[0].bar(x + 0.2, att["Wshare"][order], 0.4, label="share of weight")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([pair_name(res, b) for b in order], rotation=60, ha="right", fontsize=8)
    ax[0].legend()
    ax[0].set_title("Ripple-band uv cells: coherent excess versus weight")
    hours = (res.utimes[np.minimum((np.arange(res.NTB) + 0.5) * res.nslots // res.NTB, res.nslots - 1).astype(int)] - res.utimes[0]) / 3600
    for c in chosen[:4]:
        t_m, f_m = time_freq_marginals(res, att, c["b"])
        ax[1].plot(hours, t_m, label=c["name"])
        ax[2].plot(res.cb_freq / 1e6, f_m, label=c["name"])
    ax[1].set_xlabel("hours since start")
    ax[1].set_title("Share of the ripple-band coherent excess in time")
    ax[2].set_xlabel("frequency [MHz]")
    ax[2].set_title("... and in frequency")
    ax[1].legend(fontsize=8)
    save_fig(fig, path)


def fig_timefreq(path, res, wins, names):
    n = len(wins)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    hours = (res.utimes[-1] - res.utimes[0]) / 3600.0
    for k, (w, name) in enumerate(zip(wins, names)):
        a = ax[k // cols, k % cols]
        ex = w["map"]
        vmax = np.nanpercentile(np.abs(ex), 98) if np.isfinite(ex).any() else 1
        im = a.imshow(ex.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=[0, hours, res.cb_freq[0] / 1e6, res.cb_freq[-1] / 1e6])
        plt.colorbar(im, ax=a)
        a.set_title(f"{name}: systematic excess vs similar baselines", fontsize=9)
        a.set_xlabel("hours since start")
        a.set_ylabel("MHz")
        if w.get("window"):
            wi = w["window"]
            a.add_patch(plt.Rectangle(((wi["t0"] - res.utimes[0]) / 3600, wi["f0"] / 1e6), (wi["t1"] - wi["t0"]) / 3600, (wi["f1"] - wi["f0"]) / 1e6, fill=False, ec="k", lw=1.5))
    for k in range(n, rows * cols):
        ax[k // cols, k % cols].axis("off")
    save_fig(fig, path)


def fig_verification(path, variants, metrics, verdict):
    names = [v["name"] for v in variants if v["name"] in metrics and v["role"] != "reference"]
    roles = {v["name"]: v["role"] for v in variants}
    color = {"candidate": "C3", "control": "C0", "window": "C1"}
    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.45 * len(names) + 3))
    ax[0].barh(y, [metrics[n]["ratio"] for n in names], color=[color[roles[n]] for n in names])
    ax[0].axvline(1.0, color="k", lw=0.8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("ripple-band power relative to the reference (away from bright emission)")
    ax[0].set_title("Red = candidates, orange = time/freq window, blue = controls")
    ax[1].barh(y, [metrics[n]["ctl_ratio"] for n in names], color="0.6")
    ax[1].axvline(1.0, color="k", lw=0.8)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("same for a control uv band at equal radii")
    for i, n in enumerate(names):
        if n in verdict and verdict[n]["confirmed"]:
            ax[0].text(metrics[n]["ratio"] + 0.01, i, "confirmed", va="center", fontsize=8)
    save_fig(fig, path)


def fig_compare(path, ref_img, var_img, name, exclude, zero, rms, info, args, masks, sl):
    keep = ~exclude
    rb_, _ = band_filter(ref_img, zero, rms, info, args, masks)
    vb_, _ = band_filter(var_img, zero, rms, info, args, masks)
    b0 = float(np.sqrt(np.mean(rb_[keep] ** 2)))
    fig, ax = plt.subplots(2, 3, figsize=(21, 14))
    show_masked(ax[0, 0], ref_img, exclude, rms, sl, "reference (all data)")
    show_masked(ax[0, 1], var_img, exclude, rms, sl, f"without {name}")
    show_masked(ax[0, 2], ref_img - var_img, exclude, rms, sl, "difference (reference - without)")
    show_masked(ax[1, 0], rb_, exclude, b0, sl, "ripple-band filtered: reference")
    show_masked(ax[1, 1], vb_, exclude, b0, sl, f"ripple-band filtered: without {name}")
    show_masked(ax[1, 2], rb_ - vb_, exclude, b0, sl, "ripple-band filtered difference")
    fig.suptitle("Orange = bright emission, excluded from all statistics (new fringes appear there because no deconvolution was done)")
    save_fig(fig, path, rect=(0, 0, 1, 0.97))


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def peak_description(rb, su, sv):
    rows = []
    for kx, ky, p in rb.peaks:
        u, v = su * kx, sv * ky
        kr = math.hypot(u, v)
        normal = math.degrees(math.atan2(u, v)) % 180.0
        rows.append(dict(u=u, v=v, uv=kr, wavelength_arcsec=math.degrees(1.0 / kr) * 3600.0, normal_pa=normal, stripe_pa=(normal + 90.0) % 180.0, whitened=p))
    return rows


def taql_suggestions(res, name, kind, window, ms):
    if kind == "antenna":
        i = res.names.index(name.split(":")[-1])
        where = f"ANTENNA1=={i} OR ANTENNA2=={i}"
    else:
        i, j = sorted(res.names.index(a) for a in name.split("@")[0].split("-"))
        where = f"ANTENNA1=={i} AND ANTENNA2=={j}"
    lines = [f'taql "UPDATE {ms} SET FLAG=True WHERE {where}"']
    if window and kind != "antenna":
        lines.append(f'taql "UPDATE {ms} SET FLAG=True WHERE ({where}) AND TIME>={window["t0"] - 1:.1f} AND TIME<={window["t1"] + 1:.1f}"  # time window only (all channels)')
    dp3 = None
    if window and kind != "antenna":
        p, q = name.split("@")[0].split("-")
        dp3 = (
            f"DP3 msin={ms} msout=. steps=[flag] flag.type=preflagger flag.baseline=[[{p},{q}]] "
            f"flag.freqrange=[{window['f0'] / 1e6:.2f}MHz..{window['f1'] / 1e6:.2f}MHz]  "
            f"# restrict also in time with flag.timeofday / a TIME range in TaQL: {mjd_seconds_to_str(window['t0'])} - {mjd_seconds_to_str(window['t1'])} UT"
        )
    return lines, dp3


def write_report(path, ctx):
    a = ctx["args"]
    res, rb, info = ctx["res"], ctx["rb"], ctx["info"]
    L = []
    add = L.append
    verdict = ctx.get("verdict", {})
    confirmed = [n for n, v in verdict.items() if v["confirmed"] and not v.get("baseline_specific")]
    add("# Ripple/stripe source finder report\n")
    add(f"- Measurement Set: `{ctx['ms']}` (read-only), column `{a.data_column}`, field {res.fid}")
    add(f"- Residual image: `{a.residual_image}`; image: `{a.image}`")
    add(f"- Image: {info.nx} x {info.ny} px of {info.pix_arcsec:.3g}\"; residual rms {ctx['rms'] * 1e6:.1f} uJy/beam")
    add(f"- Antennas: {res.nant}; baselines: {res.nbl}; time slots: {res.nslots}; channels: {res.nch0} (averaged by {res.avg}); run time {ctx['elapsed'] / 60:.1f} min\n")
    add("## 1. Summary\n")
    if confirmed:
        for n in confirmed:
            v = verdict[n]
            ref_txt = (f"controls: median {v['control_median']:.2f}, spread {v['control_spread']:.2f}, z = {v['z']:.1f}" if np.isfinite(v["z"])
                       else f"random removal of the same ripple-band weight: median {v['control_median']:.2f}" if np.isfinite(v["control_median"]) else "no controls")
            add(f"- **{n}**: removing it lowers the ripple-band power to {v['ratio']:.2f} of the reference ({ref_txt}).")
        add("\nSee section 8 for suggested actions. Always inspect the comparison figures before flagging.")
    elif verdict:
        add("- **No baseline or antenna was confirmed as the (single) cause of the ripples.** Candidates and controls behave alike or the effect is only partial; see section 6.")
    elif ctx.get("wsclean_skipped"):
        add(f"- WSClean confirmation was not run ({ctx['wsclean_skipped']}); the statistics below are candidates only.")
    else:
        add("- No candidates were tested.")
    for n, v in verdict.items():
        if v.get("baseline"):
            kind_txt = v["conclusion"]
            add(f"- Antenna test for {v['baseline']}: removing the whole antenna {n[4:]} gives {v['ratio']:.2f} (baseline alone {v['baseline_ratio']:.2f}; control antennas: median {v['control_median']:.2f}) -> {kind_txt}.")
    for n in ctx["notes"]:
        add(f"- Note: {n}")
    if ctx["holes"]:
        h = ctx["holes"][0]
        add(f"- Largest flagged hole in the uv plane: centre (u, v) = ({h['u'] / 1e3:.2f}, {h['v'] / 1e3:.2f}) klambda, radius {h['radius'] / 1e3:.2f} klambda, {100 * h['flag_fraction']:.0f}% flagged"
            + (", **adjacent to the ripple band** (missing uv data can create ripples by itself, without any bad baseline)." if h["adjacent"] else "."))
    add("\n## 2. The ripple in the residual image\n")
    add(f"Tile-averaged power spectrum of the high-passed residual ({rb.ntiles} tiles of {rb.tile} px, bright emission masked). Band: whitened power > {a.band_threshold} within {rb.rmin / 1e3:.2f}-{rb.rmax / 1e3:.2f} klambda; detected at |uv| = {rb.extent[0] / 1e3:.2f}-{rb.extent[1] / 1e3:.2f} klambda ({rb.band.sum()} FFT cells, {rb.components} component(s)). uv axes derived from the FITS axes (u = {ctx['su']:+.0f} kx, v = {ctx['sv']:+.0f} ky: {ctx['sign_check']}; the WSClean confirmation tests whether this orientation is right).\n")
    add("| peak | u [klambda] | v [klambda] | wavelength | wave normal PA | stripe PA | whitened power |")
    add("|---|---|---|---|---|---|---|")
    for k, p in enumerate(ctx["peaks"]):
        add(f"| {k + 1} | {p['u'] / 1e3:.2f} | {p['v'] / 1e3:.2f} | {p['wavelength_arcsec']:.0f}\" | {p['normal_pa']:.0f} deg | {p['stripe_pa']:.0f} deg | {p['whitened']:.1f} |")
    add("\nPA is measured from north through east. Figures: `01_residual_overview.png`, `02_ripple_spectrum.png`.\n")
    add("## 3. Flagged holes in the uv plane\n")
    if ctx["holes"]:
        add("| hole | u [klambda] | v [klambda] | radius [klambda] | flagged | distance to band [klambda] | main antennas (share of flagged samples) |")
        add("|---|---|---|---|---|---|---|")
        for k, h in enumerate(ctx["holes"]):
            ants = ", ".join(f"{n} {100 * s:.0f}%" for n, s in h["antennas"])
            add(f"| H{k + 1} | {h['u'] / 1e3:.2f} | {h['v'] / 1e3:.2f} | {h['radius'] / 1e3:.2f} | {100 * h['flag_fraction']:.0f}% | {h['dist_to_band'] / 1e3:.2f} | {ants} |")
    else:
        add("No compact region with > 90% flagged samples was found (figure `03_uv_flags.png`).")
    add("\nA hole made of many baselines is an imaging/flagging problem, not a bad antenna. Check why these data were flagged and re-image with them restored.\n")
    add("## 4. Baseline and antenna statistics (no model needed)\n")
    add(f"Weighted residual power per baseline: total power and the noise from time differences (neighbouring integrations) and frequency differences. The systematic excess = total - time-difference noise, relative to the median of the 31 baselines closest in length; z = robust sigma units (threshold {a.z_threshold}). Baselines shorter than {a.min_baseline_length:.0f} m are not ranked. Figure `04_baseline_stats.png`.\n")
    add("Candidates tested with WSClean come first (z >= threshold and a share of the ripple-band weight); baselines with 0.0% ripple-band weight cannot cause the ripple and are shown only for information.\n")
    add("| baseline | length [m] | total | time-diff noise | freq-diff noise | excess | z | ripple-band weight | ripple-band coherent excess | ratio |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    bt, att = ctx["bt"], ctx["att"]
    order = [c for c in ctx["cands"] if np.isfinite(c["zA"])]
    order = sorted(order, key=lambda c: -c["zA"])
    tested = [c for c in order if c["name"] in ctx["tested_names"]]
    order = tested + [c for c in order if c not in tested][: max(0, 10 - len(tested))]
    for c in order:
        b = c["b"]
        ws = f"{100 * att['Wshare'][b]:.1f}%" if att.get("valid") else "-"
        es = f"{100 * att['Eshare'][b]:.1f}%" if att.get("valid") else "-"
        rt = f"{att['ratio'][b]:.1f}" if att.get("valid") else "-"
        add(f"| {c['name']} | {res.length[b]:.0f} | {bt['tot'][b]:.2f} | {bt['tdn'][b]:.2f} | {bt['fdn'][b]:.2f} | {bt['ex_t'][b]:+.2f} | {c['zA']:.1f} | {ws} | {es} | {rt} |")
    noisy = [r for r in ctx["at"] if r.get("n") and r["noise_rel"] > 1.25]
    add("\nNoisy antennas (white noise higher than the weights say; this does **not** make ripples, reweight instead):")
    add(", ".join(f"{r['name']} ({r['noise_rel']:.2f}x)" for r in sorted(noisy, key=lambda r: -r["noise_rel"])) if noisy else "none (all antenna noise ratios <= 1.25).")
    add("\n## 5. Baselines feeding the ripple band\n")
    if att.get("valid"):
        add(f"{att['n']} samples in {att['ncells']} uv cells of {res.cell:.0f} lambda. Coherent excess = sum over pairs of different samples in a cell of Re(w r1 conj(w r2)); a baseline with a much larger share of coherent excess than of weight is suspicious. Figure `05_ripple_attribution.png`.\n")
        add("| baseline | weight share | coherent excess share | ratio |")
        add("|---|---|---|---|")
        for b in np.argsort(-att["Eshare"])[:10]:
            add(f"| {pair_name(res, b)} | {100 * att['Wshare'][b]:.1f}% | {100 * att['Eshare'][b]:.1f}% | {att['ratio'][b]:.1f} |")
        top_ant = sorted(att["ant"], key=lambda r: -r["Wshare"])[:5]
        add("\nAntennas involved in most ripple-band samples (fraction of the band samples that include the antenna): " + ", ".join(f"{r['name']} {200 * r['Wshare']:.0f}%" for r in top_ant) + ". A large share only reflects the uv coverage, not an error.")
    else:
        add("No coherent excess was found in the ripple-band uv cells.")
    add("\n## 6. Time and frequency ranges\n")
    if ctx["windows_all"]:
        add("Two views per candidate: where the baseline's systematic excess (total - time-difference noise) is largest over the whole data set, and where its contribution to the ripple-band coherent excess sits (blocks >= 50% resp. 30% of the maximum; figures `05_ripple_attribution.png`, `06_timefreq_candidates.png`). A window is called localized if it holds >= 60% of the signal in <= 60% (excess) or 50% (ripple band) of the time-frequency plane.\n")
        add("| baseline | basis | time range (UT) | frequency range [MHz] | captured | box fraction | localized |")
        add("|---|---|---|---|---|---|---|")
        for c, w in ctx["windows_all"]:
            for key, label_ in (("window", "systematic excess"), ("ripple_window", "ripple-band excess")):
                wi = w.get(key)
                if wi is None:
                    add(f"| {c['name']} | {label_} | - | - | - | - | no signal |")
                    continue
                add(f"| {c['name']} | {label_} | {mjd_seconds_to_str(wi['t0'])} - {mjd_seconds_to_str(wi['t1'])} | {wi['f0'] / 1e6:.1f} - {wi['f1'] / 1e6:.1f} | {100 * wi['captured']:.0f}% | {100 * wi['area']:.0f}% | {'yes' if wi['localized'] else 'no'} |")
    else:
        add("No candidates.")
    add("\n## 7. WSClean confirmation (no deconvolution, no facets)\n")
    if ctx["metrics"]:
        add(f"Imaging options from {ctx['options_source']}: `{' '.join(ctx['options'])}`, plus `-niter 0`. Each image removes one baseline/antenna (flagged in a working copy), same imaging otherwise. Metric: rms of the residual filtered to the ripple band, measured **only away from bright emission** (excluded: smoothed image > {ctx['bright_thr'] * 1e3:.2f} mJy/beam plus {a.bright_margin_arcsec:.0f}\" margin = {100 * (1 - ctx['keep_frac']):.1f}% of the image), squared and relative to the reference. A control band at equal |uv| outside the ripple band shows the effect on other scales.\n")
        add("| variant | role | band power vs reference | control band | verdict |")
        add("|---|---|---|---|---|")
        for v in ctx["variants"]:
            m = ctx["metrics"].get(v["name"])
            if m is None:
                add(f"| {v['name']} | {v['role']} | failed | | |")
                continue
            vd = verdict.get(v["name"], {})
            add(f"| {v['name']} | {v['role']} | {m.get('ratio', 1.0):.3f} | {m.get('ctl_ratio', 1.0):.3f} | {vd.get('text', '')} |")
        tied = [(n, v) for n, v in verdict.items() if v.get("baseline")]
        if tied:
            add("\nBaseline versus its antennas (the antennas were tested because the baseline is clearly bad; an antenna removal also removes all of that antenna's other baselines, so it is compared with the control antennas and with the baseline alone):\n")
            add("| baseline alone | antenna removed | band power (baseline) | band power (antenna) | controls (median) | conclusion |")
            add("|---|---|---|---|---|---|")
            for n, v in tied:
                concl = v["conclusion"]
                add(f"| {v['baseline']} | {n[4:]} | {v['baseline_ratio']:.3f} | {v['ratio']:.3f} | {v['control_median']:.3f} | {concl} |")
            add("\nBaseline-specific: removing the whole antenna does not reduce the ripple power more than removing only the baseline (difference < 0.10): flag the baseline, not the antenna. Antenna-wide: the antenna removal is clearly stronger and clearly more than controls. Inconclusive (coverage effect): the antenna is involved in so many ripple-band samples that removing it removes the ripple band data themselves; the control is then a random set of baselines with the same ripple-band weight (`random-set[...]` rows above), and the antenna is only called a culprit if it beats those.")
        add("\nFigures: `07_verification_summary.png` and `08_compare_*.png` (reference, variant, difference, and the ripple-band filtered versions; bright emission masked in orange). Rule: confirmed if the band power drops to <= 0.85 and is at least 0.05 below the lowest control.")
    else:
        add("Not run." + (f" ({ctx['wsclean_skipped']})" if ctx.get("wsclean_skipped") else ""))
    add("\n## 8. Suggested actions (the MS was not modified)\n")
    acts = 0
    for n in sorted({c.split("@")[0] for c in confirmed}):
        kind = "antenna" if n.startswith("ant:") else "baseline"
        win, wname = None, f"{n}@window"
        for c, w in ctx["windows_all"]:
            for key in ("ripple_window", "window"):
                if c["name"] == n and w.get(key) and w[key]["localized"] and win is None:
                    win = w[key]
        lines, dp3 = taql_suggestions(res, n, kind, win, ctx["ms"])
        add(f"- **{n}**" + (" (confirmed)." if n in confirmed else " (only the window variant was confirmed)."))
        add(f"  Flag the whole {kind}:\n  ```\n  {lines[0]}\n  ```")
        if kind == "antenna":
            ac = next((x for x in ctx["ant_c"] if x["name"] == n[4:]), None)
            if ac and ac.get("band_weight", 0) >= 0.15:
                single = [(v["name"], ctx["metrics"][v["name"]]["ratio"]) for v in ctx["variants"]
                          if v["role"] == "control" and v["kind"] == "baseline" and n[4:] in v["ants"] and v["name"] in ctx["metrics"]]
                sing_txt = (" Removing single baselines of this antenna that are not outliers gave band-power ratios of " + ", ".join(f"{b} {r:.2f}" for b, r in single) + ".") if single else ""
                add(f"  **Caution:** {n[4:]} is involved in {100 * ac['band_weight']:.0f}% of the ripple-band samples, so its baselines populate the ripple uv region (flagging it removes {int(((res.bl_a1 == ac['ant']) | (res.bl_a2 == ac['ant'])).sum())} baselines). The ripple may be carried by them because of the uv coverage (see the flagged hole) as well as by an antenna error.{sing_txt} Check this antenna's calibration solutions and the other findings (the bad baseline first) before flagging it.")
        if win is not None:
            wv = verdict.get(wname)
            tail = f" Flagging only this window gave a band-power ratio of {wv['ratio']:.2f} ({wv['text']})." if wv else " This window was not tested with WSClean."
            add(f"  Or flag only the tested time/frequency window ({mjd_seconds_to_str(win['t0'])} - {mjd_seconds_to_str(win['t1'])} UT, {win['f0'] / 1e6:.1f}-{win['f1'] / 1e6:.1f} MHz; the TaQL line below is the time window only, all channels; the DP3 line adds the frequency range, check its syntax for your DP3 version).{tail}")
            for l in lines[1:]:
                add(f"  ```\n  {l}\n  ```")
            if dp3:
                add(f"  ```\n  {dp3}\n  ```")
        acts += 1
    for r in noisy:
        add(f"- {r['name']}: noise {r['noise_rel']:.2f}x higher than its weights imply; reweight (weights x about {1 / r['noise_rel']:.2f}) before imaging.")
        acts += 1
    if ctx["holes"] and ctx["holes"][0]["adjacent"]:
        add("- Investigate the flagged uv hole next to the ripple band and re-image with those data restored (see section 3).")
        acts += 1
    if not acts:
        add("- Nothing to recommend automatically.")
    add("\n## 9. Caveats\n")
    add("- Candidates are tested one at a time without deconvolution; ripples that need several baselines/antennas removed together, or that come from the sky model, deconvolution or DD calibration, may not be confirmed.")
    add("- Time/frequency ranges are where the excess is largest; for the ripple band they are partly set by the uv coverage (which baseline/time/channel falls in a uv cell).")
    add("- The residual image must be phase-centred like the MS and match the data column; the uv orientation follows the FITS axes (section 2). If no candidate or control changes the ripple at all, try `--uv-sign flip`.")
    add("- `results.json` contains the numbers of this report.")
    Path(path).write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def ms_uv_max(ms, field_id):
    from casacore.tables import table

    t = table(ms, ack=False)
    s = t.query(f"FIELD_ID=={field_id} AND ANTENNA1!=ANTENNA2")
    uvw = s.getcol("UVW")
    s.close()
    t.close()
    sp = table(ms + "/SPECTRAL_WINDOW", ack=False)
    fmax = float(np.max([np.max(sp.getcell("CHAN_FREQ", i)) for i in range(sp.nrows())]))
    sp.close()
    return float(np.percentile(np.linalg.norm(uvw[:, :2], axis=1), 99.9) * fmax / C_LIGHT)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items() if k not in ("mask", "map", "a", "inv", "cells")}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size < 5000 else None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        return worker_main(sys.argv[2:])
    parser = build_parser()
    args = parser.parse_args(argv)
    t_start = time.time()
    ms = str(Path(args.ms).expanduser().resolve())
    for pth in (ms, args.residual_image, args.image):
        if not os.path.exists(pth):
            parser.error(f"Not found: {pth}")
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error(f"{out} is not empty; use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir).expanduser().resolve() if args.work_dir else out / "work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        work.relative_to(ms)
        parser.error("--work-dir must be outside the Measurement Set")
    except ValueError:
        pass

    log("Reading images")
    resid, hdr_r = read_image(args.residual_image)
    image, hdr_i = read_image(args.image)
    if resid.shape != image.shape:
        raise FinderError("Residual image and image must have the same shape")
    info = ImageInfo(hdr_r, resid.shape)
    rms = robust_rms(resid)
    bright, zero, exclude, bright_thr = make_bright_masks(image, info, rms, args)
    keep = ~exclude
    log(f"  {info.nx}x{info.ny} px of {info.pix_arcsec:.3g}\", rms {rms * 1e6:.1f} uJy/beam, bright threshold {bright_thr * 1e3:.2f} mJy/beam, excluded {100 * exclude.mean():.1f}% of the image")

    log("Overview of the Measurement Set")
    t, sel, names, cols, fid, phase_dir = ms_overview(ms, args)
    sel.close()
    t.close()
    if np.all(np.isfinite(info.crval)):
        # RA from casacore may be negative (-pi..pi); compare with wrapping
        dra = (info.crval[0] - phase_dir[0] + 180.0) % 360.0 - 180.0
        sep = np.hypot(dra * math.cos(math.radians(phase_dir[1])), info.crval[1] - phase_dir[1]) * 3600
        if sep > 3 * info.pix_arcsec:
            log(f"WARNING: image reference position differs from the MS phase centre by {sep:.1f}\"; the uv analysis assumes they coincide")
    uv_max = ms_uv_max(ms, fid)

    log("Characterising the ripple in the residual image")
    prepared = prepare_residual(resid, rms, zero, info, args)
    rb = detect_band(prepared, info, args, uv_max)
    su, sv = info.su, info.sv
    if args.uv_sign == "flip":
        su = -su
    peaks = peak_description(rb, su, sv)
    fft_m = fft_masks(rb, info, resid.shape) if rb.band.any() else None
    band_img = band_filter(resid, zero, rms, info, args, fft_m)[0] if fft_m else np.zeros_like(resid)
    fig_overview(out / "01_residual_overview.png", resid, rms, exclude, band_img, keep, info, bright_thr)
    fig_spectrum(out / "02_ripple_spectrum.png", rb, info, su, sv)
    if not rb.band.any():
        msg = f"No significant ripple band found (threshold {args.band_threshold}); see 02_ripple_spectrum.png. Try --band-threshold 1.5."
        (out / "ripple_report.md").write_text(f"# Ripple/stripe source finder report\n\n{msg}\n")
        log(msg)
        return 0
    log(f"  ripple band at |uv| = {rb.extent[0] / 1e3:.2f}-{rb.extent[1] / 1e3:.2f} klambda, {len(peaks)} peak(s): " + "; ".join(f"|uv|={p['uv'] / 1e3:.2f}k, {p['wavelength_arcsec']:.0f}\", stripe PA {p['stripe_pa']:.0f}" for p in peaks[:3]))

    cell = args.uv_cell if args.uv_cell else 2.5 / info.fov_rad
    hole_cell = 2.5 * cell
    log(f"Scanning the Measurement Set (uv cell {cell:.0f} lambda)")
    res = scan_ms(ms, args, rb, info, su, sv, cell, hole_cell, args.chan_avg)
    if res.n_samples_capped:
        log("WARNING: ripple-band sample cap reached; attribution is based on the first rows only (raise --max-ripple-samples)")

    log("Analysing")
    bt = baseline_table(res, args)
    at = antenna_table(res, bt, args)
    att = ripple_attribution(res)
    holes = find_holes(res, rb, args)
    chosen, ant_c, controls, ant_controls, cands, notes = select_candidates(res, bt, at, att, args)
    windows_all = []
    for c in chosen:
        w = excess_window(res, c["b"], bt, att)
        w["ripple_window"] = ripple_window(res, att, c["b"])
        windows_all.append((c, w))
    log("  candidate baselines: " + (", ".join(f"{c['name']} (z={c['zA']:.1f})" for c in chosen) or "none"))
    log("  candidate antennas: " + (", ".join(a["name"] for a in ant_c) or "none"))
    log("  control baselines: " + (", ".join(c["name"] for c in controls) or "none"))

    fig_uv(out / "03_uv_flags.png", res, rb, holes, su, sv)
    fig_baselines(out / "04_baseline_stats.png", res, bt, at, chosen, args)
    if att.get("valid") and chosen:
        fig_attribution(out / "05_ripple_attribution.png", res, att, chosen)
    if windows_all:
        fig_timefreq(out / "06_timefreq_candidates.png", res, [w for _, w in windows_all], [c["name"] for c, _ in windows_all])

    ctx = dict(args=args, ms=ms, res=res, rb=rb, info=info, rms=rms, peaks=peaks, holes=holes, bt=bt, at=at, att=att,
               cands=cands, notes=notes, windows_all=windows_all, su=su, sv=sv, sign_check="axes from the FITS header" + (" with u flipped by --uv-sign" if args.uv_sign == "flip" else ""), metrics={}, variants=[], tested_names={c['name'] for c in chosen}, ant_c=ant_c,
               verdict={}, bright_thr=bright_thr, keep_frac=float(keep.mean()), elapsed=0.0)

    wsclean_cmd = shlex.split(args.wsclean)
    skip = None
    if args.no_wsclean:
        skip = "--no-wsclean"
    elif shutil.which(wsclean_cmd[0]) is None:
        skip = f"'{wsclean_cmd[0]}' not found; use --wsclean to give the command"
    elif not (chosen or ant_c):
        skip = "no candidates"
    ctx["wsclean_skipped"] = skip
    if skip:
        log(f"Skipping WSClean confirmation: {skip}")
    else:
        windows = []
        if not args.no_window_test and windows_all:
            c, w = windows_all[0]
            for key in ("ripple_window", "window"):
                if w.get(key) and w[key]["localized"]:
                    windows.append((c, w[key]))
                    break
        variants = build_variants(res, chosen, ant_c, controls, ant_controls, windows, args, att, cands)
        options, source = wsclean_base_options(args, hdr_i, hdr_r, info, os.path.basename(ms.rstrip("/")))
        ctx["options"], ctx["options_source"], ctx["variants"] = options, source, variants
        log(f"Running WSClean for {len(variants)} variants (options from {source})")
        cfg = dict(ms=ms, data_column=args.data_column, work_dir=str(work), names=res.names, wsclean=wsclean_cmd, options=options,
                   threads=args.wsclean_threads, keep=args.keep_work, dt=float(np.median(np.diff(res.utimes))) if res.nslots > 1 else 1.0)
        images = run_all_variants(cfg, variants, args.jobs)
        if images.get("reference"):
            masks = fft_masks(rb, info, resid.shape)
            metrics = evaluate_variants(images, variants, rms, zero, keep, info, args, masks)
            verdict = verdicts(variants, metrics)
            ctx["metrics"], ctx["verdict"] = metrics, verdict
            fig_verification(out / "07_verification_summary.png", variants, metrics, verdict)
            ref_img, _ = read_image(images["reference"])
            sl = crop_slices(resid.shape, args.crop_fraction)
            shown = [v for v in variants if v["role"] in ("candidate", "window")] + [v for v in variants if v["role"] == "control"][:2]
            for v in shown:
                if images.get(v["name"]):
                    img, _ = read_image(images[v["name"]])
                    safe = re.sub(r"[^A-Za-z0-9]+", "_", v["name"])
                    fig_compare(out / f"08_compare_{safe}.png", ref_img, img, v["name"], exclude, zero, rms, info, args, masks, sl)
        else:
            log("WARNING: the reference WSClean image failed; see the logs in the work directory")
            ctx["wsclean_skipped"] = "reference image failed (see logs in the work directory)"

    ctx["elapsed"] = time.time() - t_start
    write_report(out / "ripple_report.md", ctx)
    summary = dict(ms=ms, ripple_peaks=peaks, band=[rb.rmin, rb.rmax], uv_sign=[su, sv], holes=holes, candidates=chosen, antenna_candidates=ant_c,
                   controls=controls, metrics=ctx["metrics"], verdict=ctx["verdict"], noisy_antennas=[r for r in at if r.get("n") and r["noise_rel"] > 1.25],
                   windows=[(c["name"], w["window"], w.get("ripple_window")) for c, w in windows_all])
    (out / "results.json").write_text(json.dumps(to_jsonable(summary), indent=1, default=str))
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True) if not args.work_dir else None
    log(f"Done in {ctx['elapsed'] / 60:.1f} min. Report: {out / 'ripple_report.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FinderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
