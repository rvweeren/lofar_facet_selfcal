"""Data-driven diagnostics of solution intervals for a finished (or running) facetselfcal run.

Diagnostic only: it reads the h5parms of the run (read-only) and suggests, per solve, per antenna and
per cycle, the solution interval that minimises the total solution error. It changes nothing in the
run.

Method (per solve = one h5parm, per antenna, per quantity: phase and/or log amplitude)
---------------------------------------------------------------------------------
- Read the solutions on a subset of channels (every ``chan_step``-th channel). Phases are referenced
  to the circular mean over the antennas per (time, channel, polarisation), which removes the common
  phase that a DDECal solve leaves free.
- Temporal structure function D(k) = robust mean of the squared difference between solutions k
  intervals apart, pooled over channels and polarisations (robust: (median |difference| / 0.6745)^2).
- Fit D(k) = 2 sigma^2 + A * G_beta(k): a white noise floor (sigma = noise per solution) plus the
  structure function of a power-law process (D_true(tau) = A tau^beta, tau in units of the current
  interval) as seen through solutions that are averages over one interval (G_beta, forward-modelled
  boxcar averaging). Because the averaging is forward-modelled, the inferred variability does not
  depend on the current interval (guard against the ratchet toward long intervals).
- Total error for an interval m times the current one:
      E(m) = sigma^2 / m + A m^beta / ((beta + 1)(beta + 2))
  (noise averages down as 1/m; the second term is the mean squared difference between a power-law
  process and its average over the interval). The optimum is
      m_opt = [sigma^2 (beta + 1)(beta + 2) / (beta A)]^(1 / (beta + 1)).
- Flags: m_opt < 1 is extrapolated below the measured lags; m_opt above the largest measured lag is
  not supported by the data (never lengthen past the measured lags); m_opt below the data
  integration time is not possible.
- Noise cross-check for two-polarisation phase solves: the XX-YY phase difference shares the
  ionosphere/troposphere, so its lag-1 scatter is mostly noise (sigma^2 ~ D_xy(1) / 4).
- Station pairs (fast phase solves): the scatter of the phase difference between two antennas after
  removing a running mean over the slow-solve interval, minus the noise of both. Near zero means the
  two antennas see the same fast corruption, so an antenna constraint would cost little.
- Frequency direction (smoothness kernel): the same per antenna along frequency, on all channels of
  a subset of time slots. The solutions are modelled as per-channel white noise plus a power-law
  frequency structure, both smoothed by the DP3 kernel that was used (exp(-9 (dnu/B)^2), truncated at
  +-B/2), and the suggested kernel width minimises the smoothed noise plus the structure smoothed
  away. The polarisation-difference noise from the time direction is used as fixed noise where
  available (converted to per-channel noise with the kernel).
- solution_qa: qa_gains along time in dry-run mode (no flags written) for selected cycles.

Output: misc/solution_settings.json and plots in plots/solution_settings/, read by the report page
solutions.html.
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import tables

from submods.h5_helpers.solution_qa import _axes, _to_standard, qa_gains

SCHEMA_VERSION = 1
JSON_PATH = os.path.join('misc', 'solution_settings.json')
PLOT_DIR = os.path.join('plots', 'solution_settings')
H5_PATTERN = re.compile(r'^(?P<soltype>[a-z]+?)(?P<slot>\d+)_selfcalcycle(?P<cycle>\d{3})_(?P<ms>.+)\.h5$')
BETA_GRID = np.round(np.arange(0.1, 2.0001, 0.02), 3)
FIT_KMAX = 16  # fit the structure function up to this lag (in intervals): the short lags set the optimum
WRAP_SATURATION = 2.5  # rad^2: above this the robust phase scatter is affected by wrapping
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _wrap(x):
    return np.angle(np.exp(1j * x))


def _robust_ms(x, axis=None):
    """Robust mean square of zero-mean differences: (median |x| / 0.6745)^2, NaN-aware."""
    with np.errstate(all='ignore'):
        return (np.nanmedian(np.abs(x), axis=axis) / 0.6745) ** 2


def _boxcar_powerlaw_sf(lags, betas, nw=801):
    """G_beta(k): structure function at lag k of the averages over unit intervals of a process with
    structure function tau^beta. Shape (len(betas), len(lags))."""
    w = np.linspace(-1., 1., nw)
    kern = 1. - np.abs(w)
    lags = np.asarray(lags, dtype=float)
    out = np.empty((len(betas), len(lags)))
    for i, beta in enumerate(betas):
        f0 = _trapz(kern * np.abs(w) ** beta, w)
        out[i] = [_trapz(kern * np.abs(k + w) ** beta, w) - f0 for k in lags]
    return out


def fit_structure_function(lags, d_obs, gtab=None, fsig=10.):
    """
    Fit D(k) = s + A G_beta(k) (s = 2 sigma^2) with relative-error weighting.

    The noise floor s and the variability A are each tested for significance against a fit without
    them (F-like ratio of the cost improvement to the residual variance per point > fsig). A noise
    floor is not measurable when the solutions change by much more than the noise already at lag 1
    (interval longer than optimal); the variability is not measurable when the solutions are
    noise-dominated at all lags.

    Returns
    -------
    dict or None
        sigma2, A, beta, rel_rms, noise_significant, var_significant, sigma2_upper (upper limit on
        the noise when it is not significant); None if there are too few valid lags.
    """
    lags = np.asarray(lags, dtype=float)
    d_obs = np.asarray(d_obs, dtype=float)
    ok = np.isfinite(d_obs) & (d_obs > 0)
    n = int(np.sum(ok))
    if n < 4:
        return None
    if gtab is None:
        gtab = _boxcar_powerlaw_sf(lags, BETA_GRID)
    from scipy.optimize import nnls
    y = d_obs[ok]
    best = best0 = None
    for i, beta in enumerate(BETA_GRID):
        g = gtab[i, ok]
        design = np.vstack([np.ones_like(y), g]).T / y[:, None]
        coef, _ = nnls(design, np.ones_like(y))
        cost = np.sum(((y - coef[0] - coef[1] * g) / y) ** 2)
        if best is None or cost < best[0]:
            best = (cost, coef, beta)
        a0 = max(np.sum(g / y) / np.sum((g / y) ** 2), 0.)  # no noise floor
        cost0 = np.sum(((y - a0 * g) / y) ** 2)
        if best0 is None or cost0 < best0[0]:
            best0 = (cost0, a0, beta)
    cost, (s, amp), beta = best
    s_only = np.sum(1. / y) / np.sum(1. / y ** 2)  # no variability
    cost_s = np.sum(((y - s_only) / y) ** 2)
    resid_var = max(cost / max(n - 3, 1), 1e-12)
    noise_sig = s > 0 and (best0[0] - cost) / resid_var > fsig
    var_sig = amp > 0 and (cost_s - cost) / resid_var > fsig
    out = dict(sigma2=float(s / 2.), A=float(amp), beta=float(beta), rel_rms=float(np.sqrt(cost / n)),
               noise_significant=bool(noise_sig), var_significant=bool(var_sig), sigma2_upper=None)
    if not var_sig:
        out.update(sigma2=float(s_only / 2.), A=0.)
    elif not noise_sig:
        # variability-only model; noise conservatively below ~50% of D at the shortest lag (tested on synthetic data)
        out.update(A=float(best0[1]), beta=float(best0[2]), sigma2=0., sigma2_upper=float(0.25 * y[0]))
    return out


def fit_fixed_noise(lags, d_obs, sigma2, gtab, fsig=10.):
    """Fit D(k) = 2 sigma2 + A G_beta(k) with the noise fixed (independent estimate); A tested for
    significance against A = 0."""
    d_obs = np.asarray(d_obs, dtype=float)
    ok = np.isfinite(d_obs) & (d_obs > 0)
    n = int(np.sum(ok))
    if n < 3:
        return None
    y = d_obs[ok]
    s = 2. * sigma2
    best = None
    for i, beta in enumerate(BETA_GRID):
        g = gtab[i, ok]
        amp = max(np.sum(g * (y - s) / y ** 2) / np.sum(g ** 2 / y ** 2), 0.)
        cost = np.sum(((y - s - amp * g) / y) ** 2)
        if best is None or cost < best[0]:
            best = (cost, amp, beta)
    cost, amp, beta = best
    cost_s = np.sum(((y - s) / y) ** 2)
    resid_var = max(cost / max(n - 2, 1), 1e-12)
    var_sig = amp > 0 and (cost_s - cost) / resid_var > fsig
    return dict(sigma2=float(sigma2), A=float(amp) if var_sig else 0., beta=float(beta),
                rel_rms=float(np.sqrt(cost / n)), noise_significant=True, var_significant=bool(var_sig),
                sigma2_upper=None)


def optimal_factor(sigma2, amp, beta):
    """Interval factor m (relative to the current interval) that minimises E(m), and E(m)."""
    c = (beta + 1.) * (beta + 2.)

    def err(m):
        return sigma2 / m + amp * m ** beta / c

    if amp <= 0:
        return np.inf, err
    if sigma2 <= 0:
        return 0., err
    return (sigma2 * c / (beta * amp)) ** (1. / (beta + 1.)), err


def parse_solve_commands(log_path):
    """DP3 solve settings per h5parm basename from logs/selfcal.log."""
    settings = {}
    if not os.path.isfile(log_path):
        return settings
    with open(log_path, errors='replace') as f:
        for line in f:
            if 'ddecal.h5parm=' not in line:
                continue
            tokens = dict(re.findall(r'(ddecal\.[A-Za-z0-9_]+)=(\S+)', line))
            h5 = os.path.basename(tokens.get('ddecal.h5parm', ''))
            if not h5:
                continue
            entry = {'mode': tokens.get('ddecal.mode')}
            for key, name, conv in (('ddecal.solint', 'solint_slots', int), ('ddecal.nchan', 'nchan', int),
                                    ('ddecal.smoothnessconstraint', 'smoothness_hz', float)):
                try:
                    entry[name] = conv(tokens[key]) if key in tokens else None
                except ValueError:
                    entry[name] = None
            try:
                entry['smoothnessreffrequency_hz'] = float(tokens.get('ddecal.smoothnessreffrequency', 0.) or 0.)
            except ValueError:
                entry['smoothnessreffrequency_hz'] = None
            ac = tokens.get('ddecal.antennaconstraint')
            entry['antennaconstraint'] = ac
            settings[h5] = entry  # last command wins (restarts)
    return settings


def integration_time(run_root):
    log = os.path.join(run_root, 'logs', 'selfcal.log')
    if os.path.isfile(log):
        with open(log, errors='replace') as f:
            for line in f:
                m = re.search(r'Integration time \[s\]:\s*([0-9.]+)', line)
                if m:
                    return float(m.group(1))
    return None


def _read_solve(h5, nchan_target):
    """Solutions (time, freq, ant, pol) on a channel subset, NaN where flagged, plus metadata."""
    out = {}
    with tables.open_file(h5, 'r') as H:
        sol = H.root.sol000
        names = [n for n in ('phase000', 'amplitude000') if n in sol._v_children]
        if not names:
            return None
        ref = getattr(sol, names[0])
        axes = _axes(ref)
        nf = ref.val.shape[axes.index('freq')] if 'freq' in axes else 1
        step = max(1, nf // nchan_target)
        ants = [a.decode() if isinstance(a, bytes) else str(a) for a in ref.ant[:]]
        antpos = {}
        if 'antenna' in sol._v_children:
            for row in sol.antenna[:]:
                name = row['name'].decode() if isinstance(row['name'], bytes) else str(row['name'])
                antpos[name] = np.asarray(row['position'], dtype=float)
        times = np.asarray(ref.time[:], dtype=float)
        freqs = np.asarray(ref.freq[::step], dtype=float) if 'freq' in axes else np.zeros(1)
        for name in names:
            st = getattr(sol, name)
            sel = [slice(None)] * len(axes)
            if 'freq' in axes:
                sel[axes.index('freq')] = slice(None, None, step)
            val = _to_standard(np.asarray(st.val[tuple(sel)], dtype=float), list(axes))[:, :, :, 0, :]
            wgt = _to_standard(np.asarray(st.weight[tuple(sel)], dtype=float), list(axes))[:, :, :, 0, :]
            val[~np.isfinite(val) | (wgt <= 0)] = np.nan
            out[name] = val
    pos = np.array([antpos.get(a, np.full(3, np.nan)) for a in ants])
    return dict(values=out, times=times, freqs=freqs, ants=ants, pos=pos, chan_step=step, nchan_total=nf)


def _lags(nt, solint_s):
    kmax = int(min(nt // 3, max(10, np.ceil(7200. / max(solint_s, 1.)))))
    if kmax < 4:
        return np.array([], dtype=int)
    return np.unique(np.round(np.logspace(0, np.log10(kmax), 24)).astype(int))


def _structure_functions(x, lags, phase):
    """D(k) per antenna for x (time, freq, ant, pol)."""
    na = x.shape[2]
    d_out = np.full((len(lags), na), np.nan)
    xt = np.moveaxis(x, 2, -1)  # (t, f, p, a)
    for i, k in enumerate(lags):
        d = xt[k:] - xt[:-k]
        if phase:
            d = _wrap(d)
        d_out[i] = _robust_ms(d.reshape(-1, na), axis=0)
    return d_out


def _analyse_quantity(x, lags, solint_s, tint, phase, gtab, noise_indep=None, indep_label='XX-YY',
                      noise_bound=None):
    """Per-antenna fits and optimal factors for one quantity of one solve.

    noise_indep: independent per-antenna noise per solution from the difference of the two
    polarisations (XX-YY phase, XX/YY amplitude ratio), in which a change common to both cancels.
    When available, the fit uses it as a fixed noise floor. If the solutions themselves look much
    noisier than that, they change faster than the interval: real variability that looks like noise
    (it decorrelates between neighbouring intervals), which a structure function alone cannot tell
    apart from noise."""
    na = x.shape[2]
    d_obs = _structure_functions(x, lags, phase)
    m_min = (tint / solint_s) if tint else None
    nfit = max(int(np.sum(lags <= FIT_KMAX)), min(6, len(lags)))
    per_ant = []
    for a in range(na):
        fit_free = fit_structure_function(lags[:nfit], d_obs[:nfit, a], gtab[:, :nfit])
        fit = fit_free
        noise_source = 'fit'
        hidden_fast = False
        capped_by_freq = False
        indep_a = None
        if noise_indep is not None and np.isfinite(noise_indep[a]) and noise_indep[a] > 0:
            indep_a = float(noise_indep[a])
            if noise_bound is not None and np.isfinite(noise_bound[a]) and 0 < noise_bound[a] < indep_a:
                indep_a = float(noise_bound[a])
                capped_by_freq = True
        if fit_free is not None and indep_a is not None:
            fixed = fit_fixed_noise(lags[:nfit], d_obs[:nfit, a], indep_a ** 2, gtab[:, :nfit])
            if fixed is not None:
                hidden_fast = bool(fit_free['noise_significant'] and fit_free['sigma2'] > 4. * indep_a ** 2)
                fit = fixed
                noise_source = indep_label
        entry = dict(d_obs=[None if not np.isfinite(v) else float(v) for v in d_obs[:, a]],
                     noise_rms_freq_bound=None if noise_bound is None or not np.isfinite(noise_bound[a])
                     else float(noise_bound[a]),
                     noise_rms_fit=None if fit_free is None or not fit_free['noise_significant']
                     else float(np.sqrt(fit_free['sigma2'])))
        if fit is None:
            entry.update(valid=False)
            per_ant.append(entry)
            continue
        limit = None
        if hidden_fast:
            # the excess white part is real variability faster than the interval: the optimum is
            # shorter than the used interval (how much shorter cannot be measured from these solutions)
            m_opt, err = optimal_factor(fit['sigma2'], max(fit['A'], 1e-30), fit['beta'])
            m_opt = min(m_opt, 1.)
            limit = 'upper'
        elif not fit['var_significant']:
            m_opt, err = optimal_factor(fit['sigma2'], 0., fit['beta'])
            limit = 'lower'
        elif not fit['noise_significant']:
            if noise_bound is not None and np.isfinite(noise_bound[a]) and noise_bound[a] ** 2 < fit['sigma2_upper']:
                fit = dict(fit, sigma2_upper=float(noise_bound[a] ** 2))
                noise_source = 'frequency bound'
            m_opt, _ = optimal_factor(fit['sigma2_upper'], fit['A'], fit['beta'])
            _, err = optimal_factor(0., fit['A'], fit['beta'])
            limit = 'upper'
        else:
            m_opt, err = optimal_factor(fit['sigma2'], fit['A'], fit['beta'])
        flags = []
        if phase and np.nanmax(d_obs[:, a]) > WRAP_SATURATION:
            flags.append('phase scatter near wrapping')
        if noise_source == 'frequency bound':
            flags.append('noise upper limit from the frequency direction (channels have independent noise)')
        elif noise_source != 'fit':
            flags.append(f'noise from {noise_source}: changes common to both polarisations cancel; any real XX/YY '
                         'difference adds to it, so it is an upper limit on the noise')
        if capped_by_freq:
            flags.append('polarisation-difference noise exceeds the frequency-direction bound: it contains real '
                         'XX/YY changes; the bound is used instead')
        if hidden_fast:
            flags.append('changes faster than the interval: solutions look noisier than the noise '
                         '(real variability, not noise); optimum shorter than the used interval')
        if fit['var_significant'] and (fit['beta'] <= BETA_GRID[0] + 1e-9 or fit['beta'] >= BETA_GRID[-1] - 1e-9):
            flags.append('power-law index at the fit bound (fit unreliable)')
        if limit == 'lower':
            flags.append('no variability measured: optimum longer than the largest measured lag')
        elif limit == 'upper' and not hidden_fast:
            flags.append('noise floor not measurable: optimum is an upper limit')
        if limit is None and m_opt > lags[-1]:
            flags.append('optimum beyond the measured lags')
        if limit != 'lower' and np.isfinite(m_opt) and m_opt < 1:
            flags.append('optimum below the current interval (extrapolated)')
        if m_min is not None and np.isfinite(m_opt) and m_opt < m_min:
            flags.append('optimum below the data integration time')
        m_lo = m_min if m_min is not None else 1e-3
        m_eff = float(np.clip(m_opt, m_lo, lags[-1])) if np.isfinite(m_opt) else float(lags[-1])
        entry.update(valid=True, sigma2=fit['sigma2'], A=fit['A'], beta=fit['beta'], rel_rms=fit['rel_rms'],
                     noise_significant=fit['noise_significant'], var_significant=fit['var_significant'],
                     sigma2_upper=fit['sigma2_upper'], limit=limit, noise_source=noise_source,
                     hidden_fast=hidden_fast,
                     noise_rms=float(np.sqrt(fit['sigma2'])) if fit['noise_significant'] or limit == 'lower' else None,
                     noise_rms_upper=None if fit['sigma2_upper'] is None else float(np.sqrt(fit['sigma2_upper'])),
                     var_rms_now=float(np.sqrt(fit['A'] / ((fit['beta'] + 1) * (fit['beta'] + 2)))),
                     m_opt=float(m_opt) if np.isfinite(m_opt) else None,
                     m_opt_bounded=m_eff,
                     m_next=float(np.clip(m_eff, 0.5, 2.)),
                     err_rms_now=float(np.sqrt(err(1.))), err_rms_opt=float(np.sqrt(err(m_eff))),
                     flags=flags)
        per_ant.append(entry)
    return per_ant


def _percentiles(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None, None
    return tuple(float(p) for p in np.percentile(v, [16, 50, 84]))


def _xxyy_noise(x, phase=True):
    """Per-antenna noise per polarisation from the lag-1 scatter of the XX-YY difference (phase, or
    log amplitude = log of the XX/YY ratio). A change common to both polarisations cancels."""
    if x.shape[3] != 2:
        return None
    dd = x[..., 0] - x[..., 1]  # (t, f, a)
    d1 = dd[1:] - dd[:-1]
    if phase:
        d1 = _wrap(d1)
    return np.sqrt(_robust_ms(d1.reshape(-1, d1.shape[-1]), axis=0) / 4.)


def _pair_excess(phase, noise_rms, window):
    """Excess scatter of antenna-pair phase differences after removing a running mean over window."""
    from scipy.ndimage import uniform_filter1d
    na = phase.shape[2]
    out = np.full((na, na), np.nan)
    for i in range(na - 1):
        d = _wrap(phase[:, :, i:i + 1, :] - phase[:, :, i + 1:, :])  # (t, f, nj, p)
        z = np.exp(1j * d)
        valid = np.isfinite(z)
        z0 = np.where(valid, z, 0.)
        n = uniform_filter1d(valid.astype(float), window, axis=0, mode='nearest')
        with np.errstate(all='ignore'):
            zm = (uniform_filter1d(z0.real, window, axis=0, mode='nearest')
                  + 1j * uniform_filter1d(z0.imag, window, axis=0, mode='nearest')) / n
            r = np.angle(z * np.conj(zm))
        v = _robust_ms(np.moveaxis(r, 2, -1).reshape(-1, na - i - 1), axis=0)
        noise = (noise_rms[i] ** 2 + noise_rms[i + 1:] ** 2) * (1. - 1. / window)
        out[i, i + 1:] = v - noise
        out[i + 1:, i] = out[i, i + 1:]
    return out


def analyse_solve(job):
    """Analyse one h5parm. job: dict(h5, run_root, cycle, slot, soltype, settings, tint, nchan, do_qa,
    slow_window_s)."""
    h5 = job['h5']
    data = _read_solve(h5, job['nchan'])
    if data is None:
        return None
    times = data['times']
    solint_s = float(np.median(np.diff(times))) if len(times) > 1 else None
    rec = dict(cycle=job['cycle'], slot=job['slot'], soltype=job['soltype'],
               h5=os.path.relpath(h5, job['run_root']), ntimes=int(len(times)),
               solint_s=solint_s, nchan_total=int(data['nchan_total']), chan_step=int(data['chan_step']),
               freq_range_mhz=[float(data['freqs'].min() / 1e6), float(data['freqs'].max() / 1e6)],
               settings=job['settings'], quantities={}, spatial=None, qa=None, ants=data['ants'])
    pos = data['pos']
    centre = np.nanmedian(pos, axis=0)
    dist = np.linalg.norm(pos - centre, axis=1)
    rec['ant_dist_m'] = [None if not np.isfinite(d) else float(d) for d in dist]
    if solint_s is None:
        return rec
    lags = _lags(len(times), solint_s)
    rec['lags'] = [int(k) for k in lags]
    if len(lags) == 0:
        rec['note'] = 'too few solution intervals for a structure function'
        return rec
    gtab = _boxcar_powerlaw_sf(lags, BETA_GRID)
    rec['fit_kmax'] = int(lags[max(int(np.sum(lags <= FIT_KMAX)), min(6, len(lags))) - 1])
    vals = data['values']
    try:
        fprep = prepare_frequency(h5, job['settings'])
        fbound = frequency_noise_bound(fprep)
    except Exception as exc:  # noqa: BLE001 - report, do not fail the diagnostic
        fprep, fbound = dict(error=str(exc)), {}
    phase = None
    if 'phase000' in vals:
        phase = vals['phase000']
        z = np.where(np.isfinite(phase), np.exp(1j * phase), 0.)
        ref = np.angle(np.sum(z, axis=2, keepdims=True))
        phase = _wrap(phase - ref)
        xxyy = _xxyy_noise(phase, phase=True)
        per_ant = _analyse_quantity(phase, lags, solint_s, job['tint'], True, gtab,
                                    noise_indep=xxyy, indep_label='XX-YY phase',
                                    noise_bound=fbound.get('phase'))
        if xxyy is not None:
            for a, e in enumerate(per_ant):
                e['noise_rms_xxyy'] = None if not np.isfinite(xxyy[a]) else float(xxyy[a])
        rec['quantities']['phase'] = _summarise(per_ant, 'rad', solint_s)
    if 'amplitude000' in vals:
        with np.errstate(all='ignore'):
            lamp = np.log(vals['amplitude000'])
        lamp[~np.isfinite(lamp)] = np.nan
        xxyy_amp = _xxyy_noise(lamp, phase=False)
        per_ant = _analyse_quantity(lamp, lags, solint_s, job['tint'], False, gtab,
                                    noise_indep=xxyy_amp,
                                    indep_label='XX/YY amplitude ratio', noise_bound=fbound.get('amplitude'))
        if xxyy_amp is not None:
            for a, e in enumerate(per_ant):
                e['noise_rms_xxyy'] = None if not np.isfinite(xxyy_amp[a]) else float(xxyy_amp[a])
        rec['quantities']['amplitude'] = _summarise(per_ant, 'fractional (log amplitude)', solint_s)
    # station pairs for fast phase solves
    if phase is not None and solint_s <= 120. and job.get('do_spatial'):
        # unmeasurable noise (upper limit) counts as 0: conservative, the full pair scatter is excess
        noise = np.array([(e.get('noise_rms') or 0.) if e.get('valid') else np.nan
                          for e in rec['quantities']['phase']['per_antenna']])
        window = max(3, int(round(job['slow_window_s'] / solint_s)))
        excess = _pair_excess(phase, np.nan_to_num(noise, nan=0.), window)
        iu = np.triu_indices(len(data['ants']), 1)
        bl = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)
        rec['spatial'] = dict(window_s=float(window * solint_s),
                              baseline_m=[float(b) for b in bl],
                              excess_rms_signed=[float(np.sign(v) * np.sqrt(abs(v))) if np.isfinite(v) else None
                                                 for v in excess[iu]],
                              pairs=[[int(i), int(j)] for i, j in zip(*iu)])
        exv = excess[iu]
        bins = []
        for lo, hi in ((0., 100.), (100., 300.), (300., 1000.), (1000., 3000.), (3000., 1e9)):
            sel = (bl >= lo) & (bl < hi) & np.isfinite(exv)
            if np.any(sel):
                med = float(np.median(exv[sel]))
                bins.append(dict(min_m=lo, max_m=None if hi >= 1e9 else hi, n=int(np.sum(sel)),
                                 median_excess_rms=float(np.sign(med) * np.sqrt(abs(med)))))
        rec['spatial']['bins'] = bins
    # frequency direction (smoothness kernel), with the polarisation-difference noise per solution from
    # the time direction where available
    noise_indep = {}
    for q, summ in rec['quantities'].items():
        # conservative: the lower of the polarisation-difference noise and the noise the time-direction
        # fit measured (both are upper limits on the true noise)
        vals = []
        for e in summ['per_antenna']:
            if not (e.get('valid') and e.get('noise_source', 'fit') not in ('fit', 'frequency bound')):
                vals.append(None)
                continue
            cands = [v for v in (e.get('noise_rms'), e.get('noise_rms_fit'), e.get('noise_rms_freq_bound'))
                     if v is not None and v > 0]
            vals.append(min(cands) if cands else None)
        if any(v is not None for v in vals):
            noise_indep[q] = np.array([np.nan if v is None else v for v in vals], dtype=float)
    try:
        rec['frequency'] = analyse_frequency(fprep, noise_indep)
        if isinstance(rec['frequency'], dict) and fbound:
            rec['frequency']['noise_sol_bound'] = {q: [None if not np.isfinite(v) else float(v) for v in b]
                                                  for q, b in fbound.items()}
    except Exception as exc:  # noqa: BLE001 - report, do not fail the diagnostic
        rec['frequency'] = dict(error=str(exc))
    if job.get('do_qa'):
        try:
            rec['qa'] = qa_gains(h5, along='time', window=9, write=False)
            rec['qa']['h5parm'] = os.path.relpath(h5, job['run_root'])
            rec['qa']['settings'] = 'qa_gains along time, window 9, nsigma 5, dry run (no flags written)'
        except Exception as exc:  # noqa: BLE001 - report, do not fail the diagnostic
            rec['qa'] = dict(error=str(exc))
    return rec


def _summarise(per_ant, units, solint_s):
    valid = [e for e in per_ant if e.get('valid')]
    summary = dict(units=units, per_antenna=per_ant, n_valid=len(valid))
    for key in ('noise_rms', 'noise_rms_upper', 'noise_rms_xxyy', 'beta', 'var_rms_now', 'm_opt_bounded',
                'm_next', 'err_rms_now', 'err_rms_opt'):
        summary[key] = _percentiles([e.get(key) for e in valid])
    summary['n_upper_limit'] = sum(e.get('limit') == 'upper' for e in valid)
    summary['n_hidden_fast'] = sum(bool(e.get('hidden_fast')) for e in valid)
    summary['noise_rms_fit'] = _percentiles([e.get('noise_rms_fit') for e in valid])
    summary['n_lower_limit'] = sum(e.get('limit') == 'lower' for e in valid)
    m = summary['m_opt_bounded'][1]
    summary['suggested_solint_s'] = None if m is None else float(m * solint_s)
    summary['next_solint_s'] = None if summary['m_next'][1] is None else float(summary['m_next'][1] * solint_s)
    counts = {}
    for e in valid:
        for flag in e['flags']:
            counts[flag] = counts.get(flag, 0) + 1
    summary['flag_counts'] = counts
    return summary


# ---------------------------------------------------------------- frequency direction

FREQ_NTIME = 48  # time slots (evenly spaced) read at full frequency resolution


def dp3_kernel_taps(kernel_ch):
    """
    Normalised weights of the DP3 smoothness kernel (ddecal/constraints/KernelSmoother.h, Gaussian
    kernel with truncation) on a channel grid: exp(-9 (d/B)^2), d within B/2 plus one channel (DP3 adds
    one channel on each side and only cuts at |d| > B). kernel_ch = B in channels; <= 0 means no
    smoothing (a single tap).
    """
    if kernel_ch is None or kernel_ch <= 0:
        return np.array([1.])
    half = int(np.floor(kernel_ch / 2.)) + 1
    u = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-9. * (u / kernel_ch) ** 2)
    k[np.abs(u) > kernel_ch] = 0.
    return k / k.sum()


def _kernel_autocorr(k):
    """a(m) = sum_u k_u k_(u+m) for m = -(n-1) .. n-1, with the offsets m."""
    a = np.correlate(k, k, mode='full')
    m = np.arange(-(len(k) - 1), len(k))
    return a, m


def kernel_noise_sf(k, lags):
    """Structure function of kernel-smoothed white noise of unit variance per channel."""
    a, m = _kernel_autocorr(k)
    c0 = a[m == 0][0]
    out = []
    for lag in lags:
        sel = m == lag
        out.append(2. * (c0 - (a[sel][0] if np.any(sel) else 0.)))
    return np.asarray(out)


def kernel_powerlaw_sf(k, lags, betas):
    """Structure function of a kernel-smoothed process with structure function |d|^beta (d in
    channels): G(D) = F(D) - F(0), F(t) = sum_m a(m) |t + m|^beta. Shape (len(betas), len(lags))."""
    a, m = _kernel_autocorr(k)
    lags = np.asarray(lags, dtype=float)
    out = np.empty((len(betas), len(lags)))
    for i, beta in enumerate(betas):
        f0 = np.sum(a * np.abs(m) ** beta)
        out[i] = [np.sum(a * np.abs(lag + m) ** beta) - f0 for lag in lags]
    return out


def kernel_error_terms(k, beta):
    """For kernel weights k: noise factor sum k^2 (variance of smoothed unit white noise) and the
    variability factor V = sum_u k_u |u|^beta - 1/2 sum_m a(m)|m|^beta, the mean squared difference
    between a process with structure function |d|^beta and its kernel-smoothed version."""
    u = np.arange(len(k)) - (len(k) - 1) // 2
    a, m = _kernel_autocorr(k)
    v = float(np.sum(k * np.abs(u) ** beta) - 0.5 * np.sum(a * np.abs(m) ** beta))
    return float(np.sum(k ** 2)), max(v, 0.)


def optimal_kernel(sigma2_ch, amp, beta, nchan_band, kernel_grid=None):
    """Kernel width (channels; 0 = no smoothing) minimising sigma2_ch * sum k^2 + A V(k), and the
    error function E(width)."""
    if kernel_grid is None:
        kernel_grid = np.r_[0., np.logspace(np.log10(2.), np.log10(max(nchan_band, 3)), 70)]
    cache = {}

    def err(width):
        if width not in cache:
            s2, v = kernel_error_terms(dp3_kernel_taps(width), beta)
            cache[width] = sigma2_ch * s2 + amp * v
        return cache[width]

    vals = [err(w) for w in kernel_grid]
    best = int(np.argmin(vals))
    return float(kernel_grid[best]), err, best, len(kernel_grid)


RESMOOTH_FACTORS = (1.5, 2., 3., 4., 6., 8., 12., 16.)


def _gauss_taps(sigma_ch):
    half = max(1, int(np.ceil(3. * sigma_ch)))
    u = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (u / sigma_ch) ** 2)
    return g / g.sum()


def _resmooth_terms(k_used, kernel_ch, factor):
    """For widening the used kernel by an extra Gaussian (kernel used followed by the extra smoothing
    ~ a kernel factor times wider): sigma of the extra Gaussian (channels), the noise variance factor of
    the composite kernel (sum k_W^2) and of the part an extra smoothing removes from smoothed white noise
    (sum (k_used - k_W)^2)."""
    sig_b = kernel_ch / (3. * np.sqrt(2.)) if kernel_ch else 0.
    sig_w = factor * (sig_b if sig_b else 1.)
    sig_x = np.sqrt(max(sig_w ** 2 - sig_b ** 2, 1e-6))
    g = _gauss_taps(sig_x)
    k_w = np.convolve(k_used, g)
    pad = (len(k_w) - len(k_used)) // 2
    k_u = np.pad(k_used, (pad, len(k_w) - len(k_used) - pad))
    return sig_x, float(np.sum(k_w ** 2)), float(np.sum((k_u - k_w) ** 2)), sig_w * 3. * np.sqrt(2.)


def _resmooth_removed(x, sig_x, phase):
    """Robust mean square per antenna of what an extra Gaussian smoothing (sigma sig_x channels) along
    frequency removes from x (time, freq, ant, pol); NaN-aware (normalised convolution), samples next to
    gaps (less than half of the kernel weight available) are left out."""
    from scipy.ndimage import gaussian_filter1d
    na = x.shape[2]
    valid = np.isfinite(x)
    n = gaussian_filter1d(valid.astype(float), sig_x, axis=1, mode='constant', truncate=3.)
    with np.errstate(all='ignore'):
        if phase:
            z0 = np.where(valid, np.exp(1j * x), 0.)
            sm = (gaussian_filter1d(z0.real, sig_x, axis=1, mode='constant', truncate=3.)
                  + 1j * gaussian_filter1d(z0.imag, sig_x, axis=1, mode='constant', truncate=3.)) / n
            r = np.angle(np.exp(1j * x) * np.conj(sm))
        else:
            sm = gaussian_filter1d(np.where(valid, x, 0.), sig_x, axis=1, mode='constant', truncate=3.) / n
            r = x - sm
    r[~valid | (n < 0.5)] = np.nan
    return _robust_ms(np.moveaxis(r, 2, -1).reshape(-1, na), axis=0)


def _freq_lags(nf):
    kmax = max(4, nf // 3)
    return np.unique(np.round(np.logspace(0, np.log10(kmax), 28)).astype(int))


def _read_solve_freq(h5, ntime_target):
    """Solutions (time, freq, ant, pol) at full frequency resolution on a subset of time slots."""
    out = {}
    with tables.open_file(h5, 'r') as H:
        sol = H.root.sol000
        names = [n for n in ('phase000', 'amplitude000') if n in sol._v_children]
        if not names:
            return None
        ref = getattr(sol, names[0])
        axes = _axes(ref)
        if 'freq' not in axes:
            return None
        nt = ref.val.shape[axes.index('time')] if 'time' in axes else 1
        step = max(1, nt // ntime_target)
        freqs = np.asarray(ref.freq[:], dtype=float)
        for name in names:
            st = getattr(sol, name)
            sel = [slice(None)] * len(axes)
            if 'time' in axes:
                sel[axes.index('time')] = slice(None, None, step)
            val = _to_standard(np.asarray(st.val[tuple(sel)], dtype=float), list(axes))[:, :, :, 0, :]
            wgt = _to_standard(np.asarray(st.weight[tuple(sel)], dtype=float), list(axes))[:, :, :, 0, :]
            val[~np.isfinite(val) | (wgt <= 0)] = np.nan
            out[name] = val
    return dict(values=out, freqs=freqs, time_step=step)


def _freq_structure_functions(x, lags, phase):
    """D(lag in channels) per antenna for x (time, freq, ant, pol): differences along frequency."""
    na = x.shape[2]
    out = np.full((len(lags), na), np.nan)
    xt = np.moveaxis(x, 2, -1)  # (t, f, p, a)
    for i, lag in enumerate(lags):
        d = xt[:, lag:] - xt[:, :-lag]
        if phase:
            d = _wrap(d)
        out[i] = _robust_ms(d.reshape(-1, na), axis=0)
    return out


def _fit_freq(lags, d_obs, nshape, gtab, sigma2_fixed=None, fsig=10.):
    """Fit D = sigma2_ch N(lag) + A G_beta(lag) (N, G: kernel-smoothed noise and power law), with
    relative-error weights and significance tests as in the time direction. sigma2_fixed: independent
    per-channel noise (fixed)."""
    from scipy.optimize import nnls
    ok = np.isfinite(d_obs) & (d_obs > 0)
    n = int(np.sum(ok))
    if n < 4:
        return None
    y = d_obs[ok]
    nn = nshape[ok]
    best = best0 = None
    for i, beta in enumerate(BETA_GRID):
        g = gtab[i, ok]
        if sigma2_fixed is None:
            design = np.vstack([nn, g]).T / y[:, None]
            coef, _ = nnls(design, np.ones_like(y))
            s2, amp = coef
        else:
            s2 = sigma2_fixed
            amp = max(np.sum(g * (y - s2 * nn) / y ** 2) / np.sum(g ** 2 / y ** 2), 0.)
        cost = np.sum(((y - s2 * nn - amp * g) / y) ** 2)
        if best is None or cost < best[0]:
            best = (cost, s2, amp, beta)
        a0 = max(np.sum(g / y) / np.sum((g / y) ** 2), 0.)
        cost0 = np.sum(((y - a0 * g) / y) ** 2)
        if best0 is None or cost0 < best0[0]:
            best0 = (cost0, a0, beta)
    cost, s2, amp, beta = best
    if sigma2_fixed is None:
        s_only = np.sum(nn / y) / np.sum((nn / y) ** 2)
    else:
        s_only = sigma2_fixed
    cost_s = np.sum(((y - s_only * nn) / y) ** 2)
    resid_var = max(cost / max(n - 3, 1), 1e-12)
    var_sig = amp > 0 and (cost_s - cost) / resid_var > fsig
    noise_sig = sigma2_fixed is not None or (s2 > 0 and (best0[0] - cost) / resid_var > fsig)
    out = dict(sigma2_ch=float(s2), A=float(amp), beta=float(beta), rel_rms=float(np.sqrt(cost / n)),
               noise_significant=bool(noise_sig), var_significant=bool(var_sig), sigma2_ch_upper=None)
    if not var_sig:
        out.update(sigma2_ch=float(s_only), A=0.)
    elif not noise_sig:
        out.update(A=float(best0[1]), beta=float(best0[2]), sigma2_ch=0.,
                   sigma2_ch_upper=float(0.25 * y[0] / max(nn[0], 1e-12)))
    return out


def prepare_frequency(h5, settings, ntime=FREQ_NTIME):
    """Read the solutions at full frequency resolution (time subset) and compute the frequency
    structure functions per quantity and antenna, plus the kernel model terms."""
    data = _read_solve_freq(h5, ntime)
    if data is None:
        return None
    freqs = data['freqs']
    nf = len(freqs)
    if nf < 8:
        return dict(note='too few channels for a frequency structure function')
    df = float(np.median(np.abs(np.diff(freqs))))
    settings = settings or {}
    smooth_hz = settings.get('smoothness_hz') or 0.
    kernel_ch = smooth_hz / df if smooth_hz else 0.
    k_used = dp3_kernel_taps(kernel_ch)
    lags = _freq_lags(nf)
    prep = dict(df=df, nf=nf, smooth_hz=float(smooth_hz), kernel_ch=kernel_ch, k_used=k_used, lags=lags,
                nshape=kernel_noise_sf(k_used, lags), gtab=kernel_powerlaw_sf(k_used, lags, BETA_GRID),
                s2_used=float(np.sum(k_used ** 2)), band_ch=(freqs.max() - freqs.min()) / df + 1,
                time_step=int(data['time_step']), settings=settings, x={}, d_obs={})
    for q, name in (('phase', 'phase000'), ('amplitude', 'amplitude000')):
        if name not in data['values']:
            continue
        x = data['values'][name]
        if q == 'phase':
            z = np.where(np.isfinite(x), np.exp(1j * x), 0.)
            x = _wrap(x - np.angle(np.sum(z, axis=2, keepdims=True)))
        else:
            with np.errstate(all='ignore'):
                x = np.log(x)
            x[~np.isfinite(x)] = np.nan
        prep['x'][q] = x
        prep['d_obs'][q] = _freq_structure_functions(x, lags, q == 'phase')
    return prep


def frequency_noise_bound(prep):
    """
    Upper limit on the noise per (smoothed) solution per antenna from the frequency direction: noise in
    different channels is independent, and anything else adds to the structure function, so
    D_obs(lag) >= sigma_ch^2 N(lag) at every lag (N: smoothed-noise shape). This bounds the noise of
    every solve, including scalar ones, independent of the time direction.
    """
    if not prep or 'x' not in prep:
        return {}
    nshape = prep['nshape']
    use = nshape > 1e-3 * np.max(nshape)
    out = {}
    for q, d_obs in prep['d_obs'].items():
        with np.errstate(all='ignore'):
            ratio = np.where(use[:, None], d_obs / nshape[:, None], np.nan)
            bound_ch2 = np.nanmin(ratio, axis=0)
        out[q] = np.sqrt(bound_ch2 * prep['s2_used'])
    return out


def analyse_frequency(prep, noise_sol_indep=None):
    """
    Frequency-direction analysis of one solve: per antenna and quantity, the frequency structure
    function of the solutions, fitted with per-channel white noise plus a power law, both smoothed
    by the DP3 kernel that was used; and the kernel width that minimises noise + structure smoothed
    away, limited by the model-free re-smoothing check for wider kernels.

    noise_sol_indep : dict quantity -> per-antenna noise per (smoothed) solution (independent
        estimate, an upper limit), converted to per-channel noise with the kernel.
    """
    if prep is None or 'x' not in prep:
        return prep
    df, kernel_ch, k_used, lags = prep['df'], prep['kernel_ch'], prep['k_used'], prep['lags']
    nshape, gtab, s2_used, band_ch = prep['nshape'], prep['gtab'], prep['s2_used'], prep['band_ch']
    res = dict(kernel_hz_used=prep['smooth_hz'], chan_width_hz=df, nchan=int(prep['nf']),
               time_step=prep['time_step'], lags_ch=[int(v) for v in lags], quantities={},
               kernel_varies_with_frequency=bool(prep['settings'].get('smoothnessreffrequency_hz')),
               noise_factor_used=s2_used)
    for q, x in prep['x'].items():
        d_obs = prep['d_obs'][q]
        # model-free check for widening: what an extra smoothing removes from the solutions
        resm = []
        for factor in RESMOOTH_FACTORS:
            sig_x, s2_w, s2_diff, width_ch = _resmooth_terms(k_used, kernel_ch, factor)
            if width_ch > band_ch:
                break
            resm.append((width_ch, s2_w, s2_diff, _resmooth_removed(x, sig_x, q == 'phase')))
        indep = (noise_sol_indep or {}).get(q)
        per_ant = []
        for a in range(x.shape[2]):
            fixed = None
            if indep is not None and indep[a] is not None and np.isfinite(indep[a]) and indep[a] > 0:
                fixed = float(indep[a] ** 2 / s2_used)  # per-channel noise from noise per solution
            fit = _fit_freq(lags, d_obs[:, a], nshape, gtab, sigma2_fixed=fixed)
            entry = dict(d_obs=[None if not np.isfinite(v) else float(v) for v in d_obs[:, a]])
            if fit is None:
                entry.update(valid=False)
                per_ant.append(entry)
                continue
            limit = None
            s2c = fit['sigma2_ch']
            if not fit['var_significant']:
                limit = 'lower'
            elif not fit['noise_significant']:
                s2c = fit['sigma2_ch_upper']
                limit = 'upper'
            width, err, ibest, ngrid = optimal_kernel(s2c, fit['A'], fit['beta'], band_ch)
            # re-smoothing check (model-free): widen only as far as the real structure an extra smoothing
            # removes stays below the noise reduction it gives (noise: measured or fixed, else 0)
            # conservative noise for the check: the lower of the fixed (polarisation-difference, an upper
            # limit) and the fitted noise; 0 if neither is measurable
            # (white noise is the beta -> 0 limit of a power law, so the fit alone often cannot separate
            # them; the independent noise is needed)
            if fixed is not None:
                s2_check = fixed
            elif fit['noise_significant']:
                s2_check = fit['sigma2_ch']
            else:
                s2_check = 0.
            check_max = kernel_ch if kernel_ch else 0.
            check_rows = []
            all_ok = True
            for width_ch, s2_w, s2_diff, removed in resm:
                signal_removed = max(removed[a] - s2_check * s2_diff, 0.) if np.isfinite(removed[a]) else np.nan
                noise_gain = s2_check * max(s2_used - s2_w, 0.)
                ok_w = np.isfinite(signal_removed) and signal_removed <= noise_gain
                check_rows.append(dict(kernel_hz=float(width_ch * df),
                                       signal_removed_rms=None if not np.isfinite(signal_removed)
                                       else float(np.sqrt(signal_removed)),
                                       noise_reduction_rms=float(np.sqrt(noise_gain)), ok=bool(ok_w)))
                if not ok_w:
                    all_ok = False
                    break
                check_max = width_ch
            capped = False
            if width > max(kernel_ch, 1.) and width > check_max:
                width = max(check_max, kernel_ch)
                capped = True
            err_measured = None
            if width > max(kernel_ch, 1.):
                for row, (width_ch, s2_w, s2_diff, removed) in zip(check_rows, resm):
                    if abs(width_ch - width) < 1e-6 and row['signal_removed_rms'] is not None:
                        err_measured = float(np.sqrt(s2c * s2_w + fit['A'] * kernel_error_terms(k_used, fit['beta'])[1]
                                                     + row['signal_removed_rms'] ** 2))
            flags = []
            if capped and all_ok:
                flags.append('re-smoothing check passes up to the widest factor tested (x16): suggestion limited to it')
            elif capped:
                flags.append('re-smoothing check: a wider kernel would remove more real structure than noise '
                             '(suggestion capped)')
            if fixed is not None:
                flags.append('noise from the polarisation difference / time-direction fit (lower of the two), '
                             'converted to per-channel noise with the kernel; an upper limit on the noise')
            if limit == 'lower':
                flags.append('no frequency structure measured: widest kernel (lower limit)')
            elif limit == 'upper':
                flags.append('noise not measurable: suggested kernel is an upper limit')
            if fit['var_significant'] and (fit['beta'] <= BETA_GRID[0] + 1e-9 or fit['beta'] >= BETA_GRID[-1] - 1e-9):
                flags.append('power-law index at the fit bound (fit unreliable)')
            if limit is None and ibest == ngrid - 1:
                flags.append('optimum at the widest kernel tested (band width)')
            if kernel_ch and width and width < kernel_ch:
                flags.append('optimum narrower than the used kernel (extrapolated below the smoothed scales)')
            entry.update(valid=True, sigma2_ch=fit['sigma2_ch'], A=fit['A'], beta=fit['beta'], rel_rms=fit['rel_rms'],
                         limit=limit, noise_source='polarisation difference' if fixed is not None else 'fit',
                         noise_ch_rms=float(np.sqrt(fit['sigma2_ch'])) if fit['noise_significant'] or limit == 'lower'
                         else None,
                         noise_ch_rms_upper=None if fit['sigma2_ch_upper'] is None
                         else float(np.sqrt(fit['sigma2_ch_upper'])),
                         noise_sol_rms=float(np.sqrt(fit['sigma2_ch'] * s2_used)),
                         var_rms_now=float(np.sqrt(fit['A'] * kernel_error_terms(k_used, fit['beta'])[1])),
                         kernel_opt_hz=float(width * df),
                         kernel_check_max_hz=float(check_max * df),
                         resmooth_check=check_rows,
                         err_rms_now=float(np.sqrt(s2c * s2_used
                                                   + fit['A'] * kernel_error_terms(k_used, fit['beta'])[1])),
                         err_rms_opt=err_measured if err_measured is not None else float(np.sqrt(err(width))),
                         flags=flags)
            per_ant.append(entry)
        valid = [e for e in per_ant if e.get('valid')]
        summ = dict(per_antenna=per_ant, n_valid=len(valid),
                    n_upper_limit=sum(e.get('limit') == 'upper' for e in valid),
                    n_lower_limit=sum(e.get('limit') == 'lower' for e in valid))
        for key in ('noise_ch_rms', 'noise_ch_rms_upper', 'noise_sol_rms', 'beta', 'var_rms_now', 'kernel_opt_hz',
                    'kernel_check_max_hz', 'err_rms_now', 'err_rms_opt'):
            summ[key] = _percentiles([e.get(key) for e in valid])
        counts = {}
        for e in valid:
            for flag in e['flags']:
                counts[flag] = counts.get(flag, 0) + 1
        summ['flag_counts'] = counts
        res['quantities'][q] = summ
    return res


# ---------------------------------------------------------------- plots

def _plot_solve(rec, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    quantities = [q for q in ('phase', 'amplitude') if q in rec['quantities']]
    if not quantities or not rec.get('lags'):
        return None
    lags_s = np.asarray(rec['lags'], dtype=float) * rec['solint_s']
    fig, axes = plt.subplots(len(quantities), 2, figsize=(11, 3.8 * len(quantities)), squeeze=False)
    gfine_k = np.logspace(0, np.log10(max(rec.get('fit_kmax', rec['lags'][-1]), 2)), 60)
    for row, q in enumerate(quantities):
        summ = rec['quantities'][q]
        ax = axes[row, 0]
        for e in summ['per_antenna']:
            d = np.array([np.nan if v is None else v for v in e['d_obs']], dtype=float)
            ax.plot(lags_s, np.sqrt(d), color='0.6', lw=0.6, alpha=0.6)
        valid = [e for e in summ['per_antenna'] if e.get('valid')]
        if valid:
            med = np.nanmedian([[np.nan if v is None else v for v in e['d_obs']] for e in valid], axis=0)
            ax.plot(lags_s, np.sqrt(med), 'k-', lw=2, label='median over antennas')
            s2 = np.median([e['sigma2'] for e in valid])
            amp = np.median([e['A'] for e in valid])
            beta = np.median([e['beta'] for e in valid])
            g = _boxcar_powerlaw_sf(gfine_k, [beta])[0]
            ax.plot(gfine_k * rec['solint_s'], np.sqrt(2 * s2 + amp * g), 'C1--', lw=1.5,
                    label=f'fit (median, fitted lags): noise {np.sqrt(s2):.3g}, beta {beta:.2f}')
            if s2 > 0:
                ax.axhline(np.sqrt(2 * s2), color='C0', ls=':', lw=1.2, label='noise floor (median)')
            allv = np.array([[np.nan if v is None else v for v in e['d_obs']] for e in summ['per_antenna']],
                            dtype=float)
            finite = allv[np.isfinite(allv) & (allv > 0)]
            if finite.size:
                lo = np.sqrt(finite.min())
                if s2 > 0:
                    lo = min(lo, np.sqrt(2 * s2))
                ax.set_ylim(lo * 0.7, np.sqrt(finite.max()) * 1.4)
        ax.axvline(rec['solint_s'], color='C2', lw=1.2, label=f"used interval {rec['solint_s']:.0f} s")
        if summ.get('suggested_solint_s'):
            ax.axvline(summ['suggested_solint_s'], color='C3', lw=1.2,
                       label=f"suggested {summ['suggested_solint_s']:.0f} s (median)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('lag [s]')
        ax.set_ylabel(f"rms difference [{'rad' if q == 'phase' else 'ln amp'}]")
        ax.set_title(f"{q}: structure function per antenna")
        ax.legend(fontsize=7, loc='lower right')
        ax2 = axes[row, 1]
        dist = np.array([np.nan if d is None else d for d in rec['ant_dist_m']], dtype=float)
        mo = np.array([e.get('m_opt_bounded', np.nan) if e.get('valid') else np.nan for e in summ['per_antenna']])
        ax2.scatter(dist / 1e3, mo * rec['solint_s'], s=14, c='C3')
        ax2.axhline(rec['solint_s'], color='C2', lw=1.2, label='used interval')
        ax2.set_xscale('symlog', linthresh=0.1)
        ax2.set_yscale('log')
        ax2.set_xlabel('distance from array centre [km]')
        ax2.set_ylabel('suggested interval [s]')
        ax2.set_title(f'{q}: suggested interval per antenna')
        ax2.legend(fontsize=7)
    fig.suptitle(f"cycle {rec['cycle']}, {rec['soltype']} (slot {rec['slot']})", fontsize=11)
    fig.tight_layout()
    name = f"sf_{rec['soltype']}{rec['slot']}_cycle{rec['cycle']:03d}.png"
    path = os.path.join(run_root, PLOT_DIR, name)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return os.path.relpath(path, run_root)


def _plot_spatial(rec, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sp = rec.get('spatial')
    if not sp:
        return None
    bl = np.asarray(sp['baseline_m']) / 1e3
    ex = np.array([np.nan if v is None else v for v in sp['excess_rms_signed']])
    summ = rec['quantities']['phase']
    noise, label = summ['noise_rms'][1], 'median solution noise'
    if noise is None:
        noise, label = summ['noise_rms_upper'][1], 'upper limit on the median solution noise'
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(bl, ex, s=4, alpha=0.5, label='antenna pairs')
    ax.axhline(0, color='k', lw=0.8)
    if noise:
        ax.axhspan(-noise, noise, color='C0', alpha=0.12, label=f'+/- {label} ({noise:.3g} rad)')
    ax.set_xscale('log')
    ax.set_xlabel('baseline length [km]')
    ax.set_ylabel('excess pair phase scatter [rad] (signed)')
    ax.set_title(f"cycle {rec['cycle']}, {rec['soltype']}: fast phase differences between antennas "
                 f"(running mean {sp['window_s']:.0f} s removed)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    name = f"pairs_{rec['soltype']}{rec['slot']}_cycle{rec['cycle']:03d}.png"
    path = os.path.join(run_root, PLOT_DIR, name)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return os.path.relpath(path, run_root)


def _plot_history(solves, slot, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    recs = sorted([r for r in solves if r['slot'] == slot and r.get('quantities')], key=lambda r: r['cycle'])
    quantities = [q for q in ('phase', 'amplitude') if any(q in r['quantities'] for r in recs)]
    if not recs or not quantities:
        return None
    fig, axes = plt.subplots(len(quantities), 3, figsize=(13, 3.4 * len(quantities)), squeeze=False)
    for row, q in enumerate(quantities):
        rr = [r for r in recs if q in r['quantities']]
        cyc = np.array([r['cycle'] for r in rr])
        used = np.array([r['solint_s'] for r in rr])
        sug = np.array([[np.nan if v is None else v * r['solint_s'] for v in r['quantities'][q]['m_opt_bounded']]
                        for r in rr], dtype=float)
        ax = axes[row, 0]
        ax.step(cyc, used, where='mid', color='C2', label='used')
        ax.plot(cyc, sug[:, 1], 'o-', color='C3', ms=3, label='suggested (median)')
        ax.fill_between(cyc, sug[:, 0], sug[:, 2], color='C3', alpha=0.2, label='16-84% over antennas')
        ax.set_yscale('log')
        ax.set_xlabel('cycle')
        ax.set_ylabel('interval [s]')
        ax.set_title(f'{q}: interval')
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        nu = np.array([np.nan if r['quantities'][q]['noise_rms_upper'][1] is None
                       else r['quantities'][q]['noise_rms_upper'][1] for r in rr])
        if np.any(np.isfinite(nu)):
            ax.plot(cyc, nu, 'v--', ms=3, color='C0', alpha=0.6, label='noise upper limit (where not measurable)')
        for key, label, col in (('noise_rms', 'noise per solution', 'C0'),
                                ('var_rms_now', 'variability within the used interval', 'C1'),
                                ('err_rms_now', 'total error, used interval', 'k'),
                                ('err_rms_opt', 'total error, suggested interval', 'C3')):
            v = np.array([np.nan if r['quantities'][q][key][1] is None else r['quantities'][q][key][1] for r in rr])
            ax.plot(cyc, v, 'o-', ms=3, color=col, label=label)
        ax.set_yscale('log')
        ax.set_xlabel('cycle')
        ax.set_ylabel('rad' if q == 'phase' else 'ln amp')
        ax.set_title(f'{q}: error budget (median over antennas)')
        ax.legend(fontsize=7)
        ax = axes[row, 2]
        b = np.array([[np.nan if v is None else v for v in r['quantities'][q]['beta']] for r in rr], dtype=float)
        ax.plot(cyc, b[:, 1], 'o-', ms=3)
        ax.fill_between(cyc, b[:, 0], b[:, 2], alpha=0.2)
        ax.axhline(5. / 3., color='0.5', ls=':', label='Kolmogorov 5/3')
        ax.set_xlabel('cycle')
        ax.set_ylabel('beta')
        ax.set_title(f'{q}: power-law index')
        ax.legend(fontsize=7)
    soltypes = sorted({r['soltype'] for r in recs})
    fig.suptitle(f"slot {slot} ({', '.join(soltypes)}) across cycles", fontsize=11)
    fig.tight_layout()
    path = os.path.join(run_root, PLOT_DIR, f'history_slot{slot}.png')
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return os.path.relpath(path, run_root)


def _plot_solve_freq(rec, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fr = rec.get('frequency') or {}
    quantities = [q for q in ('phase', 'amplitude') if q in (fr.get('quantities') or {})]
    if not quantities:
        return None
    df = fr['chan_width_hz']
    lags = np.asarray(fr['lags_ch'], dtype=float)
    k_used = dp3_kernel_taps(fr['kernel_hz_used'] / df if fr['kernel_hz_used'] else 0.)
    fine = np.unique(np.round(np.logspace(0, np.log10(lags[-1]), 80)).astype(int))
    fig, axes = plt.subplots(len(quantities), 2, figsize=(11, 3.8 * len(quantities)), squeeze=False)
    for row, q in enumerate(quantities):
        summ = fr['quantities'][q]
        ax = axes[row, 0]
        allv = np.array([[np.nan if v is None else v for v in e['d_obs']] for e in summ['per_antenna']], dtype=float)
        for v in allv:
            ax.plot(lags * df / 1e6, np.sqrt(v), color='0.6', lw=0.6, alpha=0.6)
        valid = [e for e in summ['per_antenna'] if e.get('valid')]
        if valid:
            vv = np.array([[np.nan if v is None else v for v in e['d_obs']] for e in valid], dtype=float)
            ax.plot(lags * df / 1e6, np.sqrt(np.nanmedian(vv, axis=0)), 'k-', lw=2, label='median over antennas')
            s2 = np.median([e['sigma2_ch'] for e in valid])
            amp = np.median([e['A'] for e in valid])
            beta = np.median([e['beta'] for e in valid])
            model = s2 * kernel_noise_sf(k_used, fine) + amp * kernel_powerlaw_sf(k_used, fine, [beta])[0]
            ax.plot(fine * df / 1e6, np.sqrt(model), 'C1--', lw=1.5,
                    label=f'fit (median): noise/channel {np.sqrt(s2):.3g}, beta {beta:.2f}')
            if s2 > 0:
                ax.plot(fine * df / 1e6, np.sqrt(s2 * kernel_noise_sf(k_used, fine)), 'C0:', lw=1.2,
                        label='smoothed noise (median)')
        if fr['kernel_hz_used']:
            ax.axvline(fr['kernel_hz_used'] / 1e6, color='C2', lw=1.2,
                       label=f"used kernel {fr['kernel_hz_used'] / 1e6:g} MHz")
        sug = summ['kernel_opt_hz'][1]
        if sug:
            ax.axvline(sug / 1e6, color='C3', lw=1.2, label=f'suggested {sug / 1e6:.3g} MHz (median)')
        finite = allv[np.isfinite(allv) & (allv > 0)]
        if finite.size:
            ax.set_ylim(np.sqrt(finite.min()) * 0.7, np.sqrt(finite.max()) * 1.4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('frequency separation [MHz]')
        ax.set_ylabel(f"rms difference [{'rad' if q == 'phase' else 'ln amp'}]")
        ax.set_title(f'{q}: structure function along frequency per antenna')
        ax.legend(fontsize=7, loc='lower right')
        ax2 = axes[row, 1]
        dist = np.array([np.nan if d is None else d for d in rec['ant_dist_m']], dtype=float)
        ko = np.array([e.get('kernel_opt_hz', np.nan) if e.get('valid') else np.nan for e in summ['per_antenna']])
        ax2.scatter(dist / 1e3, np.where(ko > 0, ko, df) / 1e6, s=14, c='C3')
        if fr['kernel_hz_used']:
            ax2.axhline(fr['kernel_hz_used'] / 1e6, color='C2', lw=1.2, label='used kernel')
        ax2.set_xscale('symlog', linthresh=0.1)
        ax2.set_yscale('log')
        ax2.set_xlabel('distance from array centre [km]')
        ax2.set_ylabel('suggested kernel [MHz] (lowest = no smoothing)')
        ax2.set_title(f'{q}: suggested smoothness kernel per antenna')
        ax2.legend(fontsize=7)
    fig.suptitle(f"cycle {rec['cycle']}, {rec['soltype']} (slot {rec['slot']}): frequency direction", fontsize=11)
    fig.tight_layout()
    path = os.path.join(run_root, PLOT_DIR, f"freq_{rec['soltype']}{rec['slot']}_cycle{rec['cycle']:03d}.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return os.path.relpath(path, run_root)


def _plot_history_freq(solves, slot, run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    recs = sorted([r for r in solves if r['slot'] == slot and (r.get('frequency') or {}).get('quantities')],
                  key=lambda r: r['cycle'])
    quantities = [q for q in ('phase', 'amplitude') if any(q in r['frequency']['quantities'] for r in recs)]
    if not recs or not quantities:
        return None
    fig, axes = plt.subplots(len(quantities), 3, figsize=(13, 3.4 * len(quantities)), squeeze=False)
    for row, q in enumerate(quantities):
        rr = [r for r in recs if q in r['frequency']['quantities']]
        cyc = np.array([r['cycle'] for r in rr])
        used = np.array([r['frequency']['kernel_hz_used'] / 1e6 for r in rr])
        sug = np.array([[np.nan if v is None else v / 1e6 for v in r['frequency']['quantities'][q]['kernel_opt_hz']]
                        for r in rr], dtype=float)
        ax = axes[row, 0]
        ax.step(cyc, used, where='mid', color='C2', label='used')
        ax.plot(cyc, sug[:, 1], 'o-', color='C3', ms=3, label='suggested (median)')
        ax.fill_between(cyc, sug[:, 0], sug[:, 2], color='C3', alpha=0.2, label='16-84% over antennas')
        ax.set_yscale('symlog', linthresh=1.)
        ax.set_xlabel('cycle')
        ax.set_ylabel('kernel [MHz] (0 = no smoothing)')
        ax.set_title(f'{q}: smoothness kernel')
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        for key, label, col in (('noise_sol_rms', 'noise per solution (smoothed)', 'C0'),
                                ('var_rms_now', 'structure smoothed away (used kernel)', 'C1'),
                                ('err_rms_now', 'total error, used kernel', 'k'),
                                ('err_rms_opt', 'total error, suggested kernel', 'C3')):
            v = np.array([np.nan if r['frequency']['quantities'][q][key][1] is None
                          else r['frequency']['quantities'][q][key][1] for r in rr])
            ax.plot(cyc, v, 'o-', ms=3, color=col, label=label)
        ax.set_yscale('log')
        ax.set_xlabel('cycle')
        ax.set_ylabel('rad' if q == 'phase' else 'ln amp')
        ax.set_title(f'{q}: error budget along frequency')
        ax.legend(fontsize=7)
        ax = axes[row, 2]
        b = np.array([[np.nan if v is None else v for v in r['frequency']['quantities'][q]['beta']] for r in rr],
                     dtype=float)
        ax.plot(cyc, b[:, 1], 'o-', ms=3)
        ax.fill_between(cyc, b[:, 0], b[:, 2], alpha=0.2)
        ax.axhline(2., color='0.5', ls=':', label='2 (linear phase slope, e.g. delay)')
        ax.set_xlabel('cycle')
        ax.set_ylabel('beta (frequency)')
        ax.set_title(f'{q}: power-law index along frequency')
        ax.legend(fontsize=7)
    soltypes = sorted({r['soltype'] for r in recs})
    fig.suptitle(f"slot {slot} ({', '.join(soltypes)}) across cycles: frequency direction", fontsize=11)
    fig.tight_layout()
    path = os.path.join(run_root, PLOT_DIR, f'history_freq_slot{slot}.png')
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return os.path.relpath(path, run_root)


# ---------------------------------------------------------------- driver

def _sanitize(obj):
    """Replace NaN/inf by None (strict JSON)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


INFO_FLAG_PREFIXES = ('noise from ', 'noise upper limit from the frequency direction')


def _cycle_ranges(cycles):
    cycles = sorted(set(cycles))
    parts, start, prev = [], None, None
    for c in cycles:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            parts.append(f'{start}' if start == prev else f'{start}-{prev}')
            start = prev = c
    if start is not None:
        parts.append(f'{start}' if start == prev else f'{start}-{prev}')
    return ', '.join(parts)


def collect_warnings(solves):
    """Warnings for the report: per solve and quantity, every flag (except informational ones) that
    applies to at least half of the antennas, grouped over the cycles in which it occurs, plus
    solution_qa results that flag everything."""
    groups = {}
    order = []
    for rec in sorted(solves, key=lambda r: (r['slot'], r['cycle'])):
        items = [(q, summ) for q, summ in rec['quantities'].items()]
        freq = rec.get('frequency') or {}
        items += [(q + ' (frequency)', summ) for q, summ in (freq.get('quantities') or {}).items()]
        for q, summ in items:
            n = summ['n_valid']
            for flag, count in summ['flag_counts'].items():
                if flag.startswith(INFO_FLAG_PREFIXES) or not n or count / n < 0.5:
                    continue
                key = (rec['soltype'], rec['slot'], q, flag)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((rec['cycle'], count, n))
    warnings = []
    for key in order:
        soltype, slot, q, flag = key
        items = groups[key]
        counts = [c for _, c, _ in items]
        nants = items[0][2]
        crange = f'{min(counts)}' if min(counts) == max(counts) else f'{min(counts)}-{max(counts)}'
        label = 'cycle' if len(items) == 1 else 'cycles'
        warnings.append(f"{label} {_cycle_ranges([c for c, _, _ in items])} {soltype}{slot} {q}: {flag} "
                        f"({crange} of {nants} antennas)")
    for rec in solves:
        qa_res = rec.get('qa') or {}
        if isinstance(qa_res.get('flagged_after'), float) and qa_res['flagged_after'] >= 0.99:
            warnings.append(f"cycle {rec['cycle']} {rec['soltype']}: solution_qa (dry run) would flag all solutions "
                            f"({100 * qa_res.get('flagged_before', 0):.0f}% were flagged before)")
    return warnings


def analyse_run(run_root='.', nchan=64, nproc=8, qa='last', cycles=None, slow_window_s=600.):
    """Analyse all per-solve h5parms of a run and write misc/solution_settings.json plus plots."""
    run_root = os.path.abspath(run_root)
    os.makedirs(os.path.join(run_root, PLOT_DIR), exist_ok=True)
    os.makedirs(os.path.join(run_root, 'misc'), exist_ok=True)
    settings = parse_solve_commands(os.path.join(run_root, 'logs', 'selfcal.log'))
    tint = integration_time(run_root)
    files = []
    for h5 in sorted(glob.glob(os.path.join(run_root, 'h5_solutions', '*.h5'))):
        m = H5_PATTERN.match(os.path.basename(h5))
        if not m or m.group('soltype') == 'merged':
            continue
        cyc = int(m.group('cycle'))
        if cycles is not None and cyc not in cycles:
            continue
        files.append((h5, m.group('soltype'), int(m.group('slot')), cyc))
    # per solve type (soltype + slot): QA on the last cycle ('last') or all; station pairs on the first
    # and last cycle
    first_last = {}
    for _, st, slot, cyc in files:
        lo, hi = first_last.get((st, slot), (cyc, cyc))
        first_last[(st, slot)] = (min(lo, cyc), max(hi, cyc))
    jobs = []
    for h5, st, slot, cyc in files:
        lo, hi = first_last[(st, slot)]
        do_qa = qa == 'all' or (qa == 'last' and cyc == hi)
        jobs.append(dict(h5=h5, run_root=run_root, cycle=cyc, slot=slot, soltype=st,
                         settings=settings.get(os.path.basename(h5)), tint=tint, nchan=nchan,
                         do_qa=do_qa, do_spatial=cyc in (lo, hi), slow_window_s=slow_window_s))
    if nproc > 1:
        with Pool(nproc) as pool:
            solves = [r for r in pool.map(analyse_solve, jobs) if r is not None]
    else:
        solves = [r for r in map(analyse_solve, jobs) if r is not None]
    for rec in solves:
        rec['tint'] = tint
        rec['plot'] = _plot_solve(rec, run_root)
        rec['spatial_plot'] = _plot_spatial(rec, run_root)
        rec['freq_plot'] = _plot_solve_freq(rec, run_root)
    history = {}
    history_freq = {}
    for slot in sorted({r['slot'] for r in solves}):
        history[str(slot)] = _plot_history(solves, slot, run_root)
        history_freq[str(slot)] = _plot_history_freq(solves, slot, run_root)
    warnings = collect_warnings(solves)
    out = dict(schema_version=SCHEMA_VERSION, generated=datetime.now().isoformat(timespec='seconds'),
               tool='submods/h5_helpers/solution_settings.py', diagnostic_only=True,
               integration_time_s=tint, nchan_used=nchan, slow_window_s=slow_window_s,
               solves=sorted(solves, key=lambda r: (r['cycle'], r['slot'])),
               history_plots=history, history_freq_plots=history_freq, warnings=warnings)
    path = os.path.join(run_root, JSON_PATH)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_sanitize(out), f, indent=1, allow_nan=False)
    os.replace(tmp, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Data-driven solution-interval diagnostics of a facetselfcal run '
                                                 '(read-only; writes misc/solution_settings.json and plots).')
    parser.add_argument('run_directory', nargs='?', default='.')
    parser.add_argument('--nchan', type=int, default=64, help='number of channels to use (subset)')
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--qa', choices=('last', 'all', 'none'), default='last',
                        help='cycles for which solution_qa runs (dry run)')
    parser.add_argument('--cycles', type=int, nargs='+', default=None, help='only these cycles (default: all)')
    parser.add_argument('--slow-window', type=float, default=600.,
                        help='running-mean window [s] for the station-pair analysis (slow-solve interval)')
    args = parser.parse_args(argv)
    path = analyse_run(args.run_directory, nchan=args.nchan, nproc=args.nproc, qa=args.qa,
                       slow_window_s=args.slow_window, cycles=args.cycles)
    print('[facetselfcal] solution settings diagnostics written:', path)


if __name__ == '__main__':
    main()
