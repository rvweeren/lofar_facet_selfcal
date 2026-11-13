"""Noise diode polarisation angle (cross-hand phase) for the MeerKAT calibrator-based calibration (1GC).

Method (katsdpcal Scan.bcross_sol / bcross_to_sky; Plavin, Cotton & Mauch 2020; SARAO knowledge base
"Polarisation calibration: using the noise diode"): the noise diode injects the same signal into the H
and V receptors of an antenna. With the diagonal gains applied (delay, bandpass and phase gains, each
polarisation referenced to the reference antenna; leakage as well), the autocorrelation cross-hand of
antenna a has the phase

    arg XY_aa = delta_a(nu) + theta(nu)

with delta_a the X-Y phase of the diode signal path of antenna a and theta the X-Y phase left by the
referencing, i.e. the cross-hand phase (xfsolve convention X = diag(exp(i theta), 1), theta = X - Y).
The diode-to-sky spline s(nu) of the SDP pipeline (katsdpcal bcross_sky, facetselfcal/data/
meerkat_bcross_sky.json) is defined for the sky H-V phase in the H,V polarisation order of the
archive data: Xf_hv = median_a arg(HV_aa) + s. The feeds are swapped here (X = V, Y = H), so
XY_aa = conj(HV_aa) and

    theta = -Xf_hv = median_a arg(XY_aa) - s(nu).

The diode-on dumps are found in the autocorrelations (the coherence |XY| / sqrt(XX YY) jumps when the
diode is on) or given by the user (--meerkatcal-diode-times). The constant X-Y coupling of the
autocorrelations is removed by subtracting the mean of the diode-off dumps of the same scan.

The diode solve uses separate autocorrelation-only copies of the calibrator scans (<scan>.auto.ms:
feed swap, flagging of the known RFI ranges, band edges and the static RFI mask, frequency averaging
as the calibrator scans), without AOFlagger, which can flag the broadband diode power steps, and
without the residual flagging of the calibrator scan MSs.

Not verified yet against a celestial polarisation angle calibrator on the same observation: the sign
and polarisation order of the katsdpcal spline in this convention follow from the katsdpcal code.
"""
import json
import os
import shutil

import numpy as np
import tables
from casacore.tables import table  # type: ignore

from submods.flagging import meerkat_static_rfimask
from submods.meerkat_cal.prepare import (freq_factor, nchan_divisible, preflag_parset, set_receptor_angle_zero, split_scan,
                                         swap_parset)
from submods.runtime_utils import run, terminal_print

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLINE_FILE = os.path.join(REPO, 'facetselfcal', 'data', 'meerkat_bcross_sky.json')
# fraction of the dumps of an antenna (all calibrator scans) below which its diode-off coherence is taken
OFF_PERCENTILE = 20.
# minimum number of antennas with diode-on dumps for a diode solution
MIN_ANTENNAS = 4


def auto_ms_name(scanms):
    """Name of the autocorrelation copy of a calibrator scan MS (cal/<field>.scanNNN.auto.ms)."""
    return scanms[:-3] + '.auto.ms' if scanms.endswith('.ms') else scanms + '.auto.ms'


def make_autocorr_scan(ms, field_id, scan, outname, band, edgefraction=0.05, maxnchan=4096, rfimask='sdp'):
    """
    Autocorrelation-only copy of one calibrator scan of the archive MS: feed swap, RECEPTOR_ANGLE 0,
    flagging (known RFI ranges, band edges, zeros, and the static RFI mask, here also on the
    autocorrelations) and frequency averaging as the calibrator scans (prepare_calibrator_scan). No
    AOFlagger: the diode is a broadband power step it can flag.

    Parameters
    ----------
    ms : str
        Archive MS (all fields).
    field_id, scan : int
        FIELD_ID and SCAN_NUMBER.
    outname : str
        Output MS.
    band : str
        MeerKAT band.
    edgefraction : float, optional
        Fraction of the band flagged at each edge. Default 0.05.
    maxnchan : int, optional
        Maximum number of channels after averaging (see prepare.freq_factor). Default 4096.
    rfimask : str, optional
        Static RFI mask ('sdp', 'caracal' or 'none'). Default 'sdp'.
    """
    with table(ms, ack=False) as t:
        if t.getdminfo('DATA')['TYPE'] == 'DyscoStMan':
            # Dysco quantises each row relative to its largest values: the weak cross-hands of an
            # autocorrelation are dominated by the quantisation noise of XX and YY (|XY|/XX 0.009 -> 0.23
            # on MeerKAT L-band data), far above the diode coherence step (~0.01-0.1)
            terminal_print(f'WARNING: {ms} DATA is Dysco-compressed: the autocorrelation cross-hands are '
                           'dominated by quantisation noise, noise diode detection and phases are unreliable')
    raw = outname.rstrip('/') + '.raw'
    split_scan(ms, field_id, scan, raw, autocorr_only=True)
    with table(raw + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
    factor = freq_factor(nchan, maxnchan)
    # HI is not flagged (as in the targets, which need a cross-hand phase there)
    pfargs, pfsteps = preflag_parset(raw, band, edgefraction, include_hi=False, rfimask='none')
    swapargs, swapsteps = swap_parset()
    steps = swapsteps + pfsteps
    args = swapargs + pfargs
    maskranges = meerkat_static_rfimask(rfimask, include_hi=False)
    if maskranges:
        args += 'pfa.type=preflagger pfa.freqrange=[' + ','.join(f'{lo}..{hi}MHz' for lo, hi in maskranges) + '] '
        steps.append('pfa')
    if factor > 1:
        args += f'avg.type=averager avg.freqstep={factor} '
        steps.append('avg')
        nread = nchan_divisible(nchan, factor)  # whole channel groups only (top band-edge channels left out)
        if nread != nchan:
            args += f'msin.nchan={nread} '
    if os.path.isdir(outname):
        shutil.rmtree(outname)
    cmd = (f'DP3 msin={raw} msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM msout={outname} '
           'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False ' + args
           + 'steps=[' + ','.join(steps) + ']')
    terminal_print('Autocorrelation scan:', cmd)
    run(cmd, log=True)
    set_receptor_angle_zero(outname)
    shutil.rmtree(raw, ignore_errors=True)


def diode_coherence(ms, maxchan=1024, chunk=4000):
    """
    Per autocorrelation row: time, antenna, the coherence |XY| / sqrt(XX YY) and the power (XX+YY)/2,
    each the median over the unflagged channels (at most maxchan, evenly spaced). The coherence per
    channel does not depend on the X-Y delay (the phase), and the median ignores narrow-band RFI.

    Returns
    -------
    dict
        time, ant (names), coh, power (arrays per autocorrelation row) and interval (dump time, s).
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
    with table(ms + '::ANTENNA', ack=False) as t:
        names = np.array(t.getcol('NAME'))
    inc = max(1, nchan // maxchan)
    with table(ms, ack=False) as t:
        a1, a2 = t.getcol('ANTENNA1'), t.getcol('ANTENNA2')
        times = t.getcol('TIME')
        interval = float(np.median(t.getcol('INTERVAL')))
        rows = np.where(a1 == a2)[0]
        coh = np.full(len(rows), np.nan)
        power = np.full(len(rows), np.nan)
        for i0 in range(0, len(rows), chunk):
            r = rows[i0:i0 + chunk]
            # the autocorrelation rows of an auto MS are contiguous; read the full span and select
            start, stop = int(r[0]), int(r[-1]) + 1
            d = t.getcolslice('DATA', [0, 0], [nchan - 1, 3], [inc, 1], startrow=start, nrow=stop - start)
            f = t.getcolslice('FLAG', [0, 0], [nchan - 1, 3], [inc, 1], startrow=start, nrow=stop - start)
            d, f = d[r - start], f[r - start]
            xx, yy = d[:, :, 0].real, d[:, :, 3].real
            ok = ~np.any(f, axis=2) & (xx > 0) & (yy > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                c = np.where(ok, np.abs(d[:, :, 1]) / np.sqrt(np.where(ok, xx * yy, 1.)), np.nan)
                p = np.where(ok, 0.5 * (xx + yy), np.nan)
            good = ok.any(axis=1)
            coh[i0:i0 + len(r)][good] = np.nanmedian(c[good], axis=1)
            power[i0:i0 + len(r)][good] = np.nanmedian(p[good], axis=1)
    return dict(time=times[rows], ant=names[a1[rows]], coh=coh, power=power, interval=interval)


def _merge_ranges(times, interval):
    """Sorted dump times -> merged [start, end] ranges (dumps closer than 1.5 dump times are joined)."""
    ranges = []
    for t in np.sort(times):
        lo, hi = t - 0.5 * interval, t + 0.5 * interval
        if ranges and lo <= ranges[-1][1] + 0.5 * interval:
            ranges[-1][1] = hi
        else:
            ranges.append([lo, hi])
    return ranges


def detect_diode(mslist, minjump=0.01, nsigma=8., powerfrac=0.5):
    """
    Find the noise diode-on dumps in autocorrelation MSs.

    Per antenna, over the dumps of all MSs: the diode-off coherence is the OFF_PERCENTILE percentile of
    the coherence, its noise 1.4826 x the MAD of the dumps below the median. A dump is diode-on when
    its coherence is more than max(minjump, nsigma x noise) above the off level and, when its MS has
    diode-off dumps of that antenna, its power is at least 1 + powerfrac x (coherence jump) times the
    median diode-off power (the diode adds power: the jump in coherence cannot exceed the relative
    power increase). The diode must be off in most dumps of each antenna (OFF_PERCENTILE).

    Parameters
    ----------
    mslist : list of str
        Autocorrelation MSs (make_autocorr_scan).
    minjump : float, optional
        Minimum coherence jump. Default 0.01.
    nsigma : float, optional
        Detection threshold in units of the noise of the off coherence. Default 8.
    powerfrac : float, optional
        Power test factor, see above. Default 0.5.

    Returns
    -------
    (dict, dict)
        {ms: {antenna: [[start, end], ...]}} (MS TIME, s; only MSs and antennas with diode-on dumps)
        and per-antenna statistics (off level, threshold, number of on dumps, median coherence jump).
    """
    data = {ms: diode_coherence(ms) for ms in mslist}
    ants = sorted(set(np.concatenate([d['ant'] for d in data.values()])))
    ranges, stats = {}, {}
    for ant in ants:
        c = np.concatenate([d['coh'][d['ant'] == ant] for d in data.values()])
        c = c[np.isfinite(c)]
        if len(c) < 3:
            continue
        base = float(np.percentile(c, OFF_PERCENTILE))
        low = c[c <= np.median(c)]
        noise = float(1.4826 * np.median(np.abs(low - np.median(low))))
        thr = base + max(minjump, nsigma * noise)
        non, jumps = 0, []
        for ms, d in data.items():
            sel = (d['ant'] == ant) & np.isfinite(d['coh'])
            on = sel & (d['coh'] > thr)
            off = sel & ~on
            if off.any() and on.any():
                poff = np.median(d['power'][off])
                on &= d['power'] >= poff * (1. + powerfrac * (d['coh'] - base))
            if on.any():
                ranges.setdefault(ms, {})[str(ant)] = _merge_ranges(d['time'][on], d['interval'])
                non += int(on.sum())
                jumps += list(d['coh'][on] - base)
        stats[str(ant)] = dict(off=base, threshold=thr, n_on=non, n_dumps=int(len(c)),
                               jump=float(np.median(jumps)) if jumps else None)
    return ranges, stats


def parse_diode_times(text):
    """
    Parse --meerkatcal-diode-times: comma-separated 'start~end' in UTC (YYYY/MM/DD/HH:MM:SS or ISO) or
    MS TIME (MJD seconds). Returns a list of [start, end] in MS TIME.
    """
    from astropy.time import Time

    def mstime(s):
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            pass
        if s.count('/') == 3:  # CASA YYYY/MM/DD/HH:MM:SS
            date, clock = s.rsplit('/', 1)
            s = date.replace('/', '-') + 'T' + clock
        return float(Time(s, scale='utc').mjd * 86400.)

    out = []
    for item in text.split(','):
        if not item.strip():
            continue
        if '~' not in item:
            raise ValueError(f"--meerkatcal-diode-times: '{item}' is not a 'start~end' range")
        lo, hi = (mstime(x) for x in item.split('~'))
        if hi <= lo:
            raise ValueError(f"--meerkatcal-diode-times: end before start in '{item}'")
        out.append([lo, hi])
    return out


def ranges_from_times(mslist, timeranges):
    """User diode-on time ranges -> {ms: {antenna: ranges}} for every antenna of the MSs they overlap."""
    out = {}
    for ms in mslist:
        with table(ms, ack=False) as t:
            times = t.getcol('TIME')
            interval = float(np.median(t.getcol('INTERVAL')))
        lo, hi = times.min() - 0.5 * interval, times.max() + 0.5 * interval
        sel = [[max(a, lo), min(b, hi)] for a, b in timeranges if b > lo and a < hi]
        if sel:
            with table(ms + '::ANTENNA', ack=False) as t:
                out[ms] = {str(name): sel for name in t.getcol('NAME')}
    return out


def sky_spline_deg(freqs, band):
    """
    The katsdpcal diode-to-sky H-V phase spline (degrees) at freqs (Hz) for a MeerKAT band (UHF, L,
    S0-S4). Outside the spline's frequency range it is extrapolated (as katsdpcal).
    """
    from scipy.interpolate import splev
    key = 'S' if band.startswith('S') else band
    with open(SPLINE_FILE) as f:
        spl = json.load(f)
    if key not in spl:
        raise ValueError(f'No diode-to-sky spline for band {band}')
    knots, coefs, k = np.array(spl[key]['knots'], float), np.array(spl[key]['coefs'], float), int(spl[key]['k'])
    mhz = np.asarray(freqs, float) / 1e6
    if mhz.min() < knots[0] or mhz.max() > knots[-1]:
        terminal_print(f'WARNING: diode-to-sky spline ({key}) extrapolated outside {knots[0]:.0f}-{knots[-1]:.0f} MHz')
    return splev(mhz, (knots, coefs, k))


def _scan_diode(sums):
    """Per-antenna diode cross-hand (on - off, or on without off dumps) and its weight from xfsolve sums."""
    has_off = sums['w_off'] > 0
    with np.errstate(invalid='ignore', divide='ignore'):
        on = sums['d_on'] / sums['w_on']
        off = np.where(has_off, sums['d_off'] / np.where(has_off, sums['w_off'], 1.), 0.)
    d = on - off
    w = np.where(np.isfinite(d) & (np.abs(d) > 0), sums['w_on'], 0.)
    return np.where(w > 0, d, 0.), w


def _robust_phase(u, valid):
    """Median phase over antennas (axis 0) of unit phasors u, around their circular mean."""
    with np.errstate(invalid='ignore'):
        ref = np.angle(np.sum(np.where(valid, u, 0.), axis=0))
        r = np.angle(u * np.exp(-1j * ref)[np.newaxis])
        r = np.where(valid, r, np.nan)
        med = np.nanmedian(r, axis=0)
    return ref + med, r - med[np.newaxis]


def combine_diode(h5list, h5out, band, skyspline=True, min_antennas=3, outlier_deg=30.):
    """
    Combine the per-scan per-antenna diode X-Y phases (xfsolve mode=diode) into one cross-hand phase
    solution theta(nu) = median_a arg D_a(nu) - s(nu) (see the module docstring), with D_a the weighted
    mean over the scans of the diode cross-hand of antenna a. Antennas whose phase deviates from the
    median by more than outlier_deg (median over the channels) are left out and the median is
    recomputed. Channels with fewer than min_antennas antennas are flagged (NaN).

    The output h5parm has the layout of the other xfsolve solutions (phase000, pol XX = theta, YY = 0,
    the same for every antenna), for applycal correction=phase000.

    Parameters
    ----------
    h5list : list of str
        Per-scan xfsolve mode=diode h5parms (same channels), each with <h5parm>.sums.npz.
    h5out : str
        Output h5parm.
    band : str
        MeerKAT band (for the spline).
    skyspline : bool, optional
        Subtract the katsdpcal diode-to-sky spline. Default True.
    min_antennas : int, optional
        Minimum number of antennas per channel. Default 3.
    outlier_deg : float, optional
        Antenna outlier threshold (deg). Default 30.

    Returns
    -------
    dict
        theta (rad, per channel, NaN where flagged), freqs (Hz), the diode phase (rad), the spline
        (deg), number of antennas per channel, antenna spread (deg, 1.4826 MAD per channel), the
        antennas used and the outlier antennas.
    """
    import warnings
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')  # channels without data
    sums = [np.load(h5 + '.sums.npz') for h5 in h5list]
    freqs = sums[0]['freqs']
    names = [str(n) for n in sums[0]['ant_names']]
    total = np.zeros((len(names), len(freqs)), dtype=np.complex128)
    weight = np.zeros((len(names), len(freqs)))
    for s in sums:
        if len(s['freqs']) != len(freqs):
            raise ValueError('combine_diode: the diode solutions have different channels')
        d, w = _scan_diode(s)
        idx = [names.index(str(n)) for n in s['ant_names']]  # same antenna order in practice
        total[idx] += w * d
        weight[idx] += w
    with np.errstate(invalid='ignore', divide='ignore'):
        dio = total / weight
    valid = (weight > 0) & np.isfinite(dio) & (np.abs(dio) > 0)
    u = np.where(valid, dio / np.where(valid, np.abs(dio), 1.), 0.)
    phase, dev = _robust_phase(u, valid)
    with np.errstate(invalid='ignore'):
        antdev = np.degrees(np.nanmedian(np.abs(dev), axis=1))
    outliers = [names[i] for i in np.where(antdev > outlier_deg)[0]]
    if outliers:
        keep = np.array([n not in outliers for n in names])
        valid &= keep[:, np.newaxis]
        phase, dev = _robust_phase(u, valid)
    nant = valid.sum(axis=0)
    with np.errstate(invalid='ignore'):
        spread = np.degrees(1.4826 * np.nanmedian(np.abs(dev), axis=0))
    spline = sky_spline_deg(freqs, band) if skyspline else np.zeros(len(freqs))
    theta = np.angle(np.exp(1j * (phase - np.radians(spline))))
    good = nant >= min_antennas
    theta[~good] = np.nan

    if os.path.exists(h5out):
        os.remove(h5out)
    shutil.copy(h5list[0], h5out)
    with tables.open_file(h5out, 'r+') as H:
        st = H.root.sol000.phase000  # axes time, freq, ant, dir, pol (as written by xfsolve)
        vals = np.zeros(st.val.shape)
        vals[..., 0] = theta[np.newaxis, :, np.newaxis, np.newaxis]
        vals[:, ~good] = np.nan
        wgt = np.ones(st.val.shape)
        wgt[:, ~good] = 0.
        st.val[:] = vals
        st.weight[:] = wgt
    used = [n for i, n in enumerate(names) if valid[i].any()]
    np.savez(h5out + '.sums.npz', mode='diode', freqs=freqs, theta=theta, diode_phase=phase, spline_deg=spline,
             nant=nant, spread_deg=spread)
    terminal_print(f'Diode cross-hand phase from {len(h5list)} scan(s), {len(used)} antennas '
                   f'({len(outliers)} outliers left out: {", ".join(outliers) or "none"}), median antenna '
                   f'spread {np.nanmedian(spread):.1f} deg, sky spline {"applied" if skyspline else "not applied"} '
                   f'-> {h5out}')
    return dict(theta=theta, freqs=freqs, diode_phase=phase, spline_deg=spline, nant=nant, spread_deg=spread,
                antennas=used, outliers=outliers, antenna_deviation_deg=dict(zip(names, antdev.tolist())))


def plot_diode(h5list, result, out, title):
    """Per-antenna diode X-Y phases relative to their median, the median, the spline and theta."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mhz = result['freqs'] / 1e6
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    total, weight = None, None
    for h5 in h5list:
        d, w = _scan_diode(np.load(h5 + '.sums.npz'))
        total = w * d if total is None else total + w * d
        weight = w if weight is None else weight + w
    with np.errstate(invalid='ignore', divide='ignore'):
        dio = total / weight
    for a in range(dio.shape[0]):
        dev = np.degrees(np.angle(dio[a] * np.exp(-1j * result['diode_phase'])))
        ax[0].plot(mhz, dev, lw=0.4, color='0.5')
    ax[0].set_ylim(-60, 60)
    ax[0].set_ylabel('antenna - median (deg)')
    ax[0].set_title(title)
    ax[1].plot(mhz, np.degrees(result['diode_phase']), '.', ms=2, label='diode X-Y phase (median)')
    ax[1].plot(mhz, result['spline_deg'], '-', label='diode-to-sky spline')
    ax[1].set_ylabel('deg')
    ax[1].legend()
    ax[2].plot(mhz, np.degrees(result['theta']), '.', ms=2, color='k')
    ax[2].set_ylabel('cross-hand phase theta (deg)')
    ax[2].set_xlabel('frequency (MHz)')
    plt.tight_layout()
    plt.savefig(out, dpi=80)
    plt.close(fig)
