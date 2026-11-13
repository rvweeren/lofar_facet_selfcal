"""Quality assessment of calibration solutions in h5parms: outlier flagging along frequency or time
(running median, MAD based), bad channels and antennas, and merging of bandpasses. Flagged solutions
are set to NaN with weight 0, so that DP3 applycal flags the data they would be applied to."""
import numpy as np
import pandas as pd
import tables

from submods.runtime_utils import terminal_print


def _axes(soltab):
    return soltab.val.attrs['AXES'].decode().split(',')


def _to_standard(arr, axes, order=('time', 'freq', 'ant', 'dir', 'pol')):
    """Reorder (and add missing length-1) axes to time, freq, ant, dir, pol."""
    for name in order:
        if name not in axes:
            arr = arr[..., np.newaxis]
            axes = axes + [name]
    return np.transpose(arr, [axes.index(name) for name in order])


def _from_standard(arr, axes, order=('time', 'freq', 'ant', 'dir', 'pol')):
    """Inverse of _to_standard."""
    arr = np.transpose(arr, np.argsort([axes.index(name) if name in axes else len(axes) + i
                                        for i, name in enumerate(order)]))
    return arr.reshape(arr.shape[:len(axes)])


def _real_imag(values):
    """Real and imaginary parts with NaN in both where the complex value is NaN (np.nan + 0j has
    imaginary part 0, which a running median or fit would take as data)."""
    bad = np.isnan(values)
    return np.where(bad, np.nan, values.real), np.where(bad, np.nan, values.imag)


def _local_quadratic(values, weights, half):
    """
    Weighted least-squares quadratic fit over +-half samples around every sample, along axis 0.
    Returns the fitted value at each sample, NaN where fewer than 5 samples have weight.
    """
    from scipy.ndimage import correlate1d
    off = np.arange(-half, half + 1, dtype=float)
    yw = weights * np.where(np.isfinite(values), values, 0.)
    S = [correlate1d(weights, off ** p, axis=0, mode='constant') for p in range(5)]
    T = [correlate1d(yw, off ** p, axis=0, mode='constant') for p in range(3)]
    A = np.stack([np.stack([S[p + q] for q in range(3)], -1) for p in range(3)], -2)
    b = np.stack(T, -1)
    npts = correlate1d((weights > 0).astype(float), np.ones(2 * half + 1), axis=0, mode='constant')
    out = np.full(values.shape, np.nan)
    with np.errstate(all='ignore'):
        ok = (npts >= 5) & (np.abs(np.linalg.det(A)) > 1e-12 * np.maximum(S[0], 1.) ** 3)
    if ok.any():
        out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0, 0]
    return out


def running_outliers(values, axis, window, nsigma, niter=2, minscale=0., smooth='median'):
    """
    Outliers of complex or real series along one axis, relative to a running median, or (smooth=
    'quadratic', real series) a robust running quadratic fit.

    The scatter of each series comes from the differences between neighbouring samples, not from the
    residuals from the running median: a running median follows a smooth monotonic series exactly, so
    at high S/N those residuals (and their MAD) are zero and every small wiggle would be an outlier.

    Parameters
    ----------
    values : np.ndarray
        Complex or real values, NaN where already flagged.
    axis : int
        Axis along which the running median is taken.
    window : int
        Running median window (samples).
    nsigma : float
        Threshold in units of the noise of each series (per real component for complex values;
        complex residuals are relative to the running median amplitude).
    niter : int, optional
        Iterations (the running median is recomputed without the flagged samples).
    minscale : float, optional
        Lower limit of the noise (relative for complex values), so that very high S/N solutions are
        not flagged for deviations that do not matter. Default is 0.
    smooth : str, optional
        'median' (default) or 'quadratic'. A running median cuts the peaks and troughs of structure
        narrower than a few windows, which then shows up as outliers at high S/N; the quadratic fit
        over the window follows it. The fit is made robust with bisquare weights (as in LOESS) from
        the residuals relative to the threshold, so outliers do not pull it. The first fit leaves out
        the running median outliers, which include blocks of outliers a quadratic could bend into
        (and peaks of narrow structure, which the fit then puts back). Next to a gap, where the fit has
        samples on one side only, a sample is flagged only if the running median flags it too. niter
        is then the number of reweighting iterations (at least 3).

    Returns
    -------
    np.ndarray of bool
        True for outliers, same shape as values.
    """
    vals = np.moveaxis(values, axis, 0)
    shape = vals.shape
    flat = vals.reshape(shape[0], -1).copy()
    flagged = np.zeros(flat.shape, dtype=bool)
    if shape[0] < 3:
        return np.zeros(values.shape, dtype=bool)
    window = max(3, min(window, shape[0]))
    if smooth == 'quadratic':
        if np.iscomplexobj(flat):
            raise ValueError("running_outliers: smooth='quadratic' needs real values")
        valid = np.isfinite(flat)
        with np.errstate(all='ignore'):
            scale = 1.4826 * np.nanmedian(np.abs(np.diff(flat, axis=0)), axis=0, keepdims=True) / np.sqrt(2)
        thresh = nsigma * np.maximum(np.where(np.isfinite(scale), scale, 0.), minscale)
        median_out = running_outliers(flat, 0, window, nsigma, niter=niter, minscale=minscale)
        weights = (valid & ~median_out).astype(float)
        for _ in range(max(3, niter)):
            fit = _local_quadratic(flat, weights, window // 2)
            resid = np.abs(flat - fit)
            with np.errstate(invalid='ignore'):
                u = np.where(np.isfinite(resid), resid / np.maximum(thresh, 1e-30), np.inf)
            weights = np.where(valid & (u < 1.), (1. - u ** 2) ** 2, 0.)
        from scipy.ndimage import correlate1d
        half = window // 2
        side = np.r_[np.zeros(half + 1), np.ones(half)]  # the half samples after (correlate1d)
        after = correlate1d(valid.astype(float), side, axis=0, mode='constant')
        before = correlate1d(valid.astype(float), side[::-1], axis=0, mode='constant')
        one_sided = (after < 2) | (before < 2)
        with np.errstate(invalid='ignore'):
            flagged = valid & (resid > thresh) & (~one_sided | median_out)
        return np.moveaxis(flagged.reshape(shape), 0, axis)
    for _ in range(niter):
        cur = np.where(flagged, np.nan, flat)
        with np.errstate(all='ignore'):
            if np.iscomplexobj(cur):
                re, im = _real_imag(cur)
                smooth = (pd.DataFrame(re).rolling(window, center=True, min_periods=1).median().to_numpy()
                          + 1j * pd.DataFrame(im).rolling(window, center=True, min_periods=1).median().to_numpy())
                norm = np.maximum(np.abs(smooth), 1e-12)
                resid = np.abs(cur - smooth) / norm
                # |difference| of two complex Gaussian samples: Rayleigh with median 1.1774 * sqrt(2) sigma
                scale = np.nanmedian(np.abs(np.diff(cur, axis=0)) / norm[1:], axis=0, keepdims=True) / (
                    1.1774 * np.sqrt(2))
            else:
                smooth = pd.DataFrame(cur).rolling(window, center=True, min_periods=1).median().to_numpy()
                resid = np.abs(cur - smooth)
                scale = 1.4826 * np.nanmedian(np.abs(np.diff(cur, axis=0)), axis=0, keepdims=True) / np.sqrt(2)
            new = resid > nsigma * np.maximum(scale, minscale)
        if not np.any(new & ~flagged):
            break
        flagged |= new
    out = flagged.reshape(shape)
    return np.moveaxis(out, 0, axis)


def _antenna_flag_fraction(flags, ant_axis=2):
    """
    Fraction of each antenna's solutions that are flagged, counted only over the cells (all axes except
    the antenna axis) that most antennas have valid.

    Cells flagged for (almost) the whole array, such as RFI channels or scan gaps flagged before the QA,
    or channels/time slots flagged for all antennas by the QA itself, say nothing about one antenna.
    Counting them would flag every antenna once more than ant_frac of the solutions are flagged for the
    array (e.g. MeerKAT L-band with ~50% RFI-flagged channels). An antenna that is bad on its own is
    still counted, because the other antennas are valid in those cells.

    Parameters
    ----------
    flags : np.ndarray of bool
        True where flagged.
    ant_axis : int, optional
        Antenna axis. Default is 2 (time, freq, ant, ...).

    Returns
    -------
    np.ndarray
        Flagged fraction per antenna (1 for an antenna without any reference cell).
    """
    ref = np.broadcast_to(np.mean(~flags, axis=ant_axis, keepdims=True) > 0.5, flags.shape)
    other = tuple(i for i in range(flags.ndim) if i != ant_axis)
    nref = np.sum(ref, axis=other)
    nflag = np.sum(flags & ref, axis=other)
    return np.where(nref > 0, nflag / np.maximum(nref, 1), 1.)


def _write_flags(soltab, flags_std, axes, zero_values=None):
    """Set flagged solutions to NaN (or zero_values) and weight 0 in a soltab."""
    val = _to_standard(soltab.val[:], axes)
    wgt = _to_standard(soltab.weight[:], axes).astype(float)
    val[flags_std] = np.nan if zero_values is None else zero_values
    wgt[flags_std] = 0.
    soltab.val[:] = _from_standard(val, axes)
    soltab.weight[:] = _from_standard(wgt, axes)


def qa_gains(h5parm, along='freq', window=15, nsigma=5., chan_frac=0.5, ant_frac=0.5, amp_dex=0.3,
             minscale=3e-3, write=True):
    """
    Flag bad diagonal gain solutions (amplitude000 and/or phase000) in an h5parm.

    Outliers are found per antenna and polarisation along frequency (bandpass) or time (gains),
    relative to a running median of the complex gain. A frequency channel (time slot) is flagged for
    all antennas if more than chan_frac of the antennas and polarisations are flagged there; an
    antenna is flagged completely if more than ant_frac of its solutions are flagged (counted over the
    cells that most antennas have valid, see _antenna_flag_fraction), or if its median amplitude
    deviates more than amp_dex (in log10) from the median over the antennas.

    Parameters
    ----------
    h5parm : str
        h5parm with sol000/amplitude000 and/or sol000/phase000 (axes include ant, and freq or time).
    along : str, optional
        'freq' or 'time'. Default is 'freq'.
    window : int, optional
        Running median window. Default is 15.
    nsigma : float, optional
        Outlier threshold. Default is 5.
    chan_frac, ant_frac : float, optional
        Fractions for flagging whole channels (time slots) and antennas. Default is 0.5.
    amp_dex : float, optional
        Maximum deviation of an antenna's median amplitude from the array median (log10). Default 0.3.
    minscale : float, optional
        Lower limit of the relative noise of the gains (running_outliers); with the default 3e-3 a
        gain is flagged only if it deviates more than 1.5% from the running median. Default is 3e-3.
    write : bool, optional
        Write the flags to the h5parm. With False the h5parm is opened read-only and only the
        statistics are returned (diagnostics, report). Default is True.

    Returns
    -------
    dict
        Flagged fraction before and after, flagged antennas.
    """
    with tables.open_file(h5parm, 'r+' if write else 'r') as H:
        sol = H.root.sol000
        names = [n for n in ('amplitude000', 'phase000') if n in sol._v_children]
        axes = _axes(getattr(sol, names[0]))
        ants = [a.decode() if isinstance(a, bytes) else str(a) for a in getattr(sol, names[0]).ant[:]]
        amp = _to_standard(sol.amplitude000.val[:], axes) if 'amplitude000' in names else None
        phase = _to_standard(sol.phase000.val[:], axes) if 'phase000' in names else None
        weight = _to_standard(getattr(sol, names[0]).weight[:], axes)
        if amp is None:
            gain = np.exp(1j * phase)
        elif phase is None:
            gain = amp.astype(complex)
        else:
            gain = amp * np.exp(1j * phase)
        gain = np.where(np.isfinite(gain) & (weight > 0), gain, np.nan)
        flagged0 = np.isnan(gain)

        axis = 1 if along == 'freq' else 0
        test = gain
        if along == 'freq' and amp is not None:
            # relative to the amplitude spectrum common to the antennas (median per time, direction and
            # polarisation): the running median cannot follow a steep filter roll-off with few channels
            # (historical VLA line data: 7 channels, the last ~30% lower), which flagged the edge
            # channel on most antennas and with it the channel for all
            with np.errstate(all='ignore'):
                shape = np.nanmedian(np.abs(gain), axis=2, keepdims=True)
            test = np.where(shape > 0, gain / shape, np.nan)
        flags = flagged0 | running_outliers(test, axis, window, nsigma, minscale=minscale)
        # channels (time slots) bad for most antennas
        bad_slot = np.mean(flags, axis=(2, 3, 4), keepdims=True) > chan_frac
        flags |= np.broadcast_to(bad_slot, flags.shape)
        # antennas mostly flagged or with deviating amplitudes
        frac_ant = _antenna_flag_fraction(flags)
        with np.errstate(all='ignore'):
            medamp = np.nanmedian(np.where(flags, np.nan, np.abs(gain)), axis=(0, 1, 3, 4))
            dev = np.abs(np.log10(medamp / np.nanmedian(medamp)))
        bad_ant = (frac_ant > ant_frac) | ~np.isfinite(dev) | ((dev > amp_dex) if amp is not None else False)
        flags[:, :, bad_ant] = True
        if write:
            for name in names:
                _write_flags(getattr(sol, name), flags, axes)
    stats = dict(h5parm=h5parm, along=along, flagged_before=float(np.mean(flagged0)),
                 flagged_after=float(np.mean(flags)),
                 flagged_antennas=[a for a, b in zip(ants, bad_ant) if b])
    terminal_print(f"QA {h5parm}: flagged {100 * stats['flagged_before']:.1f}% -> {100 * stats['flagged_after']:.1f}% "
                   f"of the solutions, flagged antennas: {stats['flagged_antennas']}")
    return stats


def qa_real(h5parm, soltab, window=7, nsigma=5., ant_frac=0.5, minscale=0.):
    """
    Flag outliers in time of a real-valued soltab (delay000 or tec000) per antenna.

    Parameters
    ----------
    h5parm : str
        h5parm.
    soltab : str
        Soltab name, e.g. 'delay000'.
    window : int, optional
        Running median window (solution intervals). Default is 7.
    nsigma : float, optional
        Outlier threshold. Default is 5.
    ant_frac : float, optional
        Flag an antenna completely if more than this fraction is flagged (counted over the cells that
        most antennas have valid, see _antenna_flag_fraction). Default is 0.5.
    minscale : float, optional
        Lower limit of the noise, in the units of the soltab (e.g. s for delay000). The noise comes from
        the differences between neighbouring solutions, so for very high S/N solutions, or solution
        intervals that come in groups within a calibrator scan (nearly identical), it is tiny and
        harmless changes between scans become outliers. Default is 0.

    Returns
    -------
    dict
        Flagged fraction and antennas.
    """
    with tables.open_file(h5parm, 'r+') as H:
        st = getattr(H.root.sol000, soltab)
        axes = _axes(st)
        ants = [a.decode() if isinstance(a, bytes) else str(a) for a in st.ant[:]]
        val = _to_standard(st.val[:], axes)
        wgt = _to_standard(st.weight[:], axes)
        val = np.where(np.isfinite(val) & (wgt > 0), val, np.nan)
        flags = np.isnan(val) | running_outliers(val, 0, window, nsigma, minscale=minscale)
        bad_ant = _antenna_flag_fraction(flags) > ant_frac
        flags[:, :, bad_ant] = True
        _write_flags(st, flags, axes)
    stats = dict(h5parm=h5parm, soltab=soltab, flagged=float(np.mean(flags)),
                 flagged_antennas=[a for a, b in zip(ants, bad_ant) if b])
    terminal_print(f"QA {h5parm} {soltab}: flagged {100 * stats['flagged']:.1f}%, antennas {stats['flagged_antennas']}")
    return stats


def freq_ranges(freq, mask):
    """Contiguous runs of True in mask as [low, high] frequency pairs in MHz (channel centres)."""
    ranges, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            ranges.append([float(freq[start]) / 1e6, float(freq[i - 1]) / 1e6])
            start = None
    return ranges


def _running_median(freq, values, use, halfwidth):
    """Median of values[use] within +-halfwidth (Hz) of each channel, NaN where there are none."""
    out = np.full(len(freq), np.nan)
    for i, f in enumerate(freq):
        sel = use & (np.abs(freq - f) <= halfwidth)
        if sel.any():
            out[i] = np.median(values[sel])
    return out


def qa_crosshand(h5parm, constraint=None, min_constraint=0.3, window_mhz=60., nsigma=5., minscale_deg=2.,
                 flip_deg=45., interpolate=True, fill_missing=False):
    """
    QA of a cross-hand phase (xfsolve) solution, which is the same for all antennas and smooth in
    frequency (an offset, the reference antenna's X-Y delay and slow structure).

    1. Solutions the model does not constrain are flagged: constraint is the rms fraction of the model
       polarisation in the cross-hands (|U| / P in the feed frame for linear feeds), where it is small a
       model angle error of a few degrees flips the sign of the model cross-hands relative to the data,
       giving a solution off by 180 deg.
    2. Outliers from a smooth reference are flagged: a global delay and offset plus a running median
       over +-window_mhz / 2 of the residual phase. Both are taken from twice the phase, which a sign
       flip does not change, and the branch is that of most solutions, so flipped solutions deviate by
       180 deg however many neighbours flipped with them (a running median of the phase itself
       follows a block of flips longer than half its window). Flagged are deviations over flip_deg
       or over nsigma times the MAD noise (with a floor of minscale_deg).
    3. With interpolate, flagged solutions between good ones are replaced by a linear interpolation in
       frequency of the good ones (phases unwrapped relative to the delay), instead of flagging the data.
       Flagged solutions outside the outermost good ones stay flagged.

    Parameters
    ----------
    h5parm : str
        xfsolve h5parm (sol000/phase000, theta in pol 0), modified in place.
    constraint : np.ndarray, optional
        Per-frequency constraint fraction (see 1.), None to skip that test.
    min_constraint : float or np.ndarray, optional
        Minimum constraint fraction, or one per frequency. Default 0.3.
    window_mhz : float, optional
        Running median window (MHz). Default 60.
    nsigma : float, optional
        Outlier threshold. Default 5.
    minscale_deg : float, optional
        Lower limit of the noise (deg). Default 2.
    flip_deg : float, optional
        Deviations above this (deg) are always flagged. Default 45.
    interpolate : bool, optional
        Interpolate the flagged solutions between good ones. Default True.
    fill_missing : bool, optional
        With interpolate, also fill the solutions that were missing (NaN, no data) between good ones,
        e.g. the RFI-masked channels of the noise diode solution, which would otherwise flag those
        channels in the targets. Default False.

    Returns
    -------
    dict
        Flagged fraction before and after (and interpolated), and the frequency ranges (MHz) of the
        unconstrained, outlying and interpolated solutions.
    """
    from submods.merge_bandpass import _fit_phase_offset
    with tables.open_file(h5parm, 'r+') as H:
        st = H.root.sol000.phase000
        axes = _axes(st)
        freq = np.array(st.freq[:], dtype=float)
        val = _to_standard(st.val[:], axes)
        wgt = _to_standard(st.weight[:], axes).astype(float)
        theta = val[..., 0, 0, 0]  # (time, freq), the same for all antennas
        valid0 = np.isfinite(theta) & (wgt[..., 0, 0, 0] > 0)
        weak = np.zeros(len(freq), bool) if constraint is None else ~(np.asarray(constraint) >= np.asarray(min_constraint))
        outlier = np.zeros_like(valid0)
        filled = np.zeros_like(valid0)
        for t in range(theta.shape[0]):
            good = valid0[t] & ~weak
            if good.sum() < 3:
                continue
            f0 = float(np.mean(freq[good]))
            offset2, delay2 = _fit_phase_offset(freq, 2. * theta[t], good, True, f0)
            ref = 0.5 * offset2 + np.pi * delay2 * (freq - f0)
            if np.median(np.abs(np.angle(np.exp(1j * (theta[t, good] - ref[good]))))) > 0.5 * np.pi:
                ref += np.pi  # the branch of most solutions
            resid = np.angle(np.exp(1j * (theta[t] - ref)))
            out = np.zeros(len(freq), bool)
            for _ in range(3):
                smooth = 0.5 * _running_median(freq, np.angle(np.exp(2j * resid)), good & ~out, 0.5e6 * window_mhz)
                dev = np.degrees(np.abs(np.angle(np.exp(1j * (resid - smooth)))))
                keep = good & ~out & np.isfinite(dev)
                scale = max(1.4826 * np.median(dev[keep]) if keep.any() else 0., minscale_deg)
                new = good & ~(dev <= min(nsigma * scale, flip_deg))
                if np.array_equal(new, out):
                    break
                out = new
            outlier[t] = out
            keep = good & ~out
            bad = (valid0[t] | fill_missing) & ~keep
            if interpolate and keep.sum() >= 2:
                inside = bad & (freq > freq[keep].min()) & (freq < freq[keep].max())
                phase = np.unwrap(resid[keep])
                theta[t, inside] = ref[inside] + np.interp(freq[inside], freq[keep], phase)
                filled[t] = inside
        flags = (valid0 & (weak[np.newaxis] | outlier)) & ~filled
        val[..., 0] = np.where(filled[..., None, None], theta[..., None, None], val[..., 0])
        val[flags] = np.nan
        wgt[flags] = 0.
        wgt[filled] = 1.
        st.val[:] = _from_standard(val, axes)
        st.weight[:] = _from_standard(wgt, axes)
    usable = (valid0 & ~flags) | filled  # filled includes the missing solutions with fill_missing
    stats = dict(h5parm=h5parm, along='freq', flagged_before=float(1. - np.mean(valid0)),
                 flagged_after=float(1. - np.mean(usable)), interpolated=float(np.mean(filled)),
                 flagged_antennas=[], min_constraint=np.asarray(min_constraint, dtype=float).tolist(),
                 unconstrained_mhz=freq_ranges(freq, (valid0 & weak[np.newaxis]).any(axis=0)),
                 outlier_mhz=freq_ranges(freq, outlier.any(axis=0)),
                 interpolated_mhz=freq_ranges(freq, filled.any(axis=0)),
                 flagged_mhz=freq_ranges(freq, flags.any(axis=0)))
    terminal_print(f"QA {h5parm}: flagged {100 * stats['flagged_before']:.1f}% -> {100 * stats['flagged_after']:.1f}% "
                   f"of the solutions, interpolated {100 * stats['interpolated']:.1f}% (unconstrained by the model: "
                   f"{int((valid0 & weak[np.newaxis]).sum())}, outliers: {int(outlier.sum())})")
    return stats


def _scan_factors(gains, minchan=10):
    """
    Per input, antenna, direction and polarisation, the complex factor between a bandpass and the
    average of all inputs over the channels where all inputs have solutions (median amplitude ratio,
    mean phase difference), normalised to an average of 1. Inputs with fewer than minchan such
    channels get a factor 1.

    Parameters
    ----------
    gains : np.ndarray
        Complex gains (input, time, freq, ant, dir, pol), NaN where flagged.
    minchan : int, optional
        Minimum number of channels with solutions of all inputs. Default 10.

    Returns
    -------
    np.ndarray
        Factors (input, 1, 1, ant, dir, pol).
    """
    import warnings
    common = np.all(np.isfinite(gains), axis=0)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ratio = np.where(common, gains / np.nanmean(gains, axis=0), np.nan)
        amp = np.nanmedian(np.abs(ratio), axis=2, keepdims=True)
        phase = np.angle(np.nansum(ratio / np.abs(ratio), axis=2, keepdims=True))
    factor = amp * np.exp(1j * phase)
    enough = np.sum(common, axis=1, keepdims=True) >= minchan
    factor = np.where(enough & np.isfinite(factor), factor, 1.)
    return factor / np.mean(factor, axis=0)


def merge_bandpasses(h5list, h5out, normalise=True):
    """
    Combine bandpass h5parms (one time slot each, same frequencies and antennas) into one with a
    weighted complex average per channel, antenna and polarisation. Flagged (NaN) solutions are
    ignored; where all inputs are flagged the output is flagged.

    The inputs differ by a broadband factor per antenna and polarisation (calibrator flux models, gain
    changes between scans; J1939-6342 and J0408-6545 by ~6% in amplitude at UHF). Without normalisation
    the average jumps to the level of the remaining inputs in channels where some are flagged (RFI in
    one scan only).

    Parameters
    ----------
    h5list : list of str
        Input h5parms (amplitude000 and phase000), already referenced to the same antenna.
    h5out : str
        Output h5parm, a copy of the first input with the averaged solutions.
    normalise : bool, optional
        Divide each input by its factor to the average (see _scan_factors) before averaging, so that the
        level of the average does not depend on which inputs are flagged. Default True. Use False for
        leakage solutions (off-diagonal terms are not multiplicative).

    Returns
    -------
    np.ndarray
        Per input, the median amplitude ratio to the average (a flux scale consistency check).
    """
    import shutil
    gains, weights = [], []
    for h5 in h5list:
        with tables.open_file(h5) as H:
            axes = _axes(H.root.sol000.amplitude000)
            amp = _to_standard(H.root.sol000.amplitude000.val[:], axes)
            phase = _to_standard(H.root.sol000.phase000.val[:], axes)
            w = _to_standard(H.root.sol000.amplitude000.weight[:], axes)
        g = amp * np.exp(1j * phase)
        ok = np.isfinite(g) & (w > 0)
        gains.append(np.nanmean(np.where(ok, g, np.nan), axis=0, keepdims=True))
        weights.append(np.sum(ok, axis=0, keepdims=True).astype(float))
    gains = np.array(gains)
    weights = np.array(weights)
    factors = _scan_factors(gains) if normalise and len(gains) > 1 else np.ones(1)
    with np.errstate(all='ignore'):
        normalised = gains / factors
        mean = np.nansum(np.where(np.isfinite(normalised), normalised, 0) * weights, axis=0) / np.sum(
            np.where(np.isfinite(normalised), weights, 0), axis=0)
        ratios = np.array([np.nanmedian(np.abs(g) / np.abs(mean)) for g in gains])
    shutil.copy(h5list[0], h5out)
    with tables.open_file(h5out, 'r+') as H:
        for name, values in (('amplitude000', np.abs(mean)), ('phase000', np.angle(mean))):
            st = getattr(H.root.sol000, name)
            axes = _axes(st)
            values = np.where(np.isfinite(mean), values, np.nan)
            if st.val.shape[axes.index('time')] != 1:
                raise ValueError('merge_bandpasses expects bandpass h5parms with one time slot')
            st.val[:] = _from_standard(values, axes)
            st.weight[:] = _from_standard(np.isfinite(mean).astype(float), axes)
    terminal_print('Merged bandpasses ' + ', '.join(h5list) + ' -> ' + h5out
                   + ', median amplitude ratios to the average: ' + str(np.round(ratios, 3)))
    return ratios


def _read_fulljones(H):
    """Complex 2x2 solutions (time, freq, ant, dir, 4) and pol labels from amplitude000/phase000."""
    st = H.root.sol000.amplitude000
    axes = _axes(st)
    pols = [p.decode() if isinstance(p, bytes) else str(p) for p in st.pol[:]]
    amp = _to_standard(st.val[:], axes)
    phase = _to_standard(H.root.sol000.phase000.val[:], axes)
    weight = _to_standard(st.weight[:], axes)
    jones = np.where(np.isfinite(amp) & np.isfinite(phase) & (weight > 0), amp * np.exp(1j * phase), np.nan)
    return jones, axes, pols


def _write_fulljones(H, jones, axes):
    """Write complex 2x2 solutions (standard axis order) to amplitude000/phase000, NaN means flagged."""
    ok = np.all(np.isfinite(jones), axis=-1, keepdims=True) & np.ones(jones.shape, dtype=bool)
    for name, values in (('amplitude000', np.abs(jones)), ('phase000', np.angle(jones))):
        st = getattr(H.root.sol000, name)
        st.val[:] = _from_standard(np.where(ok, values, np.nan), axes)
        st.weight[:] = _from_standard(ok.astype(float), axes)


def fix_leakage_gauge(h5parm):
    """
    Remove the unconstrained common term of leakage solved on an unpolarised calibrator.

    For an unpolarised source the leakages d_X (XY element) and d_Y (YX element) of all antennas can
    be changed to d_X + c and d_Y - c* (a common, to first order unitary, transformation) without
    changing the model visibilities. Per solution cell c = -mean(d_X - d_Y*)/2 over the unflagged
    antennas is added, so that the mean d_X - d_Y* (a common feed rotation and ellipticity) is zero.

    Parameters
    ----------
    h5parm : str
        Full-Jones (leakage) h5parm with amplitude000 and phase000 and pols XX, XY, YX, YY.

    Returns
    -------
    float
        The median |c| that was removed.
    """
    with tables.open_file(h5parm, 'r+') as H:
        jones, axes, pols = _read_fulljones(H)
        ixy, iyx = pols.index('XY'), pols.index('YX')
        dx, dy = jones[..., ixy], jones[..., iyx]
        with np.errstate(all='ignore'):
            c = -np.nanmean(dx - np.conj(dy), axis=2, keepdims=True) / 2.
        c = np.where(np.isfinite(c), c, 0.)
        jones[..., ixy] = dx + c
        jones[..., iyx] = dy - np.conj(c)
        _write_fulljones(H, jones, axes)
    cmed = float(np.median(np.abs(c)))
    terminal_print(f'Leakage gauge fix {h5parm}: median |c| = {cmed:.4f}')
    return cmed


def qa_leakage(h5parm, window=15, nsigma=5., dmax=0.3, chan_frac=0.5, ant_frac=0.5, minscale=2e-3,
               smooth='quadratic'):
    """
    Flag bad leakage solutions: outliers of d_X and d_Y along frequency relative to a robust running
    quadratic fit (or running median), |d| above dmax, channels bad for most antennas and antennas
    that are mostly flagged. A flagged cell is flagged in all four Jones elements, so that applycal
    flags the data.

    Parameters
    ----------
    h5parm : str
        Full-Jones (leakage) h5parm with pols XX, XY, YX, YY.
    window : int, optional
        Window (channels) of the running fit. Default is 15.
    nsigma : float, optional
        Outlier threshold in units of the channel-to-channel noise of d. Default is 5.
    dmax : float, optional
        Maximum |d|. Default is 0.3.
    minscale : float, optional
        Lower limit of that noise: d is known to ~0.1% per channel on J1939-6342, so without it the
        smooth structure of d across the band is flagged. Default is 2e-3 (deviations above 1%).
    chan_frac, ant_frac : float, optional
        Fractions for flagging whole channels and antennas (the antenna fraction is counted over the
        cells that most antennas have valid, see _antenna_flag_fraction). Default is 0.5.
    smooth : str, optional
        Reference of the outlier test, see running_outliers. Default 'quadratic': some antennas have
        a leakage ripple of several percent over ~20 MHz, whose peaks a running median cuts off.

    Returns
    -------
    dict
        Flagged fractions, flagged antennas, median |d| per antenna.
    """
    with tables.open_file(h5parm, 'r+') as H:
        jones, axes, pols = _read_fulljones(H)
        ants = [a.decode() if isinstance(a, bytes) else str(a) for a in H.root.sol000.amplitude000.ant[:]]
        d = jones[..., [pols.index('XY'), pols.index('YX')]]
        flagged0 = np.any(np.isnan(jones), axis=-1)
        # running_outliers normalises complex residuals by |smooth|, so use real and imaginary parts
        re, im = _real_imag(d)
        out = (running_outliers(re, 1, window, nsigma, minscale=minscale, smooth=smooth)
               | running_outliers(im, 1, window, nsigma, minscale=minscale, smooth=smooth))
        with np.errstate(invalid='ignore'):
            flags = flagged0 | np.any(out | (np.abs(d) > dmax), axis=-1)
        bad_chan = np.mean(flags, axis=(2, 3), keepdims=True) > chan_frac
        flags |= np.broadcast_to(bad_chan, flags.shape)
        bad_ant = _antenna_flag_fraction(flags) > ant_frac
        flags[:, :, bad_ant] = True
        jones[flags] = np.nan
        _write_fulljones(H, jones, axes)
        with np.errstate(all='ignore'):
            dmed = np.nanmedian(np.abs(np.where(flags[..., np.newaxis], np.nan, d)), axis=(0, 1, 3, 4))
    stats = dict(h5parm=h5parm, flagged_before=float(np.mean(flagged0)), flagged_after=float(np.mean(flags)),
                 flagged_antennas=[a for a, b in zip(ants, bad_ant) if b],
                 median_abs_d={a: (None if not np.isfinite(v) else float(v)) for a, v in zip(ants, dmed)})
    terminal_print(f"QA leakage {h5parm}: flagged {100 * stats['flagged_before']:.1f}% -> "
                   f"{100 * stats['flagged_after']:.1f}%, antennas {stats['flagged_antennas']}, "
                   f"median |d| {np.nanmedian(dmed):.4f}")
    return stats
