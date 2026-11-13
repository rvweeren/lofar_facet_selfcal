r"""
Leakage (D-terms) and calibrator polarisation solved together for circular feeds, the counterpart of
CASA polcal(poltype='Df+QU') and AIPS PCAL, for --vlacal.

On gain-calibrated data of alt-az dishes, to first order in the leakage,

    RL_ij = I_ij [ p e^{-i(chi_i + chi_j)} + d_Ri + conj(d_Lj) ]
    LR_ij = I_ij [ conj(p) e^{+i(chi_i + chi_j)} + d_Li + conj(d_Rj) ]

with chi the parallactic angle (parang.py convention: the sky-frame cross-hands are rotated by
e^{-i(chi_i+chi_j)}), I_ij the Stokes I visibility (taken from the data, (RR + LL)/2, so no model is
needed and gain amplitude and phase errors common to R and L cancel), p = (Q + iU)/I the fractional
polarisation of the calibrator (assumed to follow its Stokes I structure) and d_R, d_L the leakages in the
Jones matrix J = [[1, d_R], [d_L, 1]] (DP3 fulljones, pols XX, XY, YX, YY = RR, RL, LR, LL). The source
term rotates with the parallactic angle and the leakage does not, so both are solvable with enough
parallactic angle coverage; for an unpolarised calibrator p = 0 is fixed. The unknown phase of the
reference antenna between R and L (the R-L phase, solved later with xfsolve) multiplies the cross-hands
and is absorbed in p and d; the leakage solved here is the one to apply before the cross-hand phase.

The system is linear in (d_R, d_L, p) and their conjugates, solved as a real weighted least-squares
problem. Its common term (d_R + c, d_L - conj(c) for all antennas) is not constrained and is fixed as in
fix_leakage_gauge (mean d_R - conj(d_L) = 0).
"""
import os

import numpy as np
import scipy.sparse as sp
from casacore.measures import measures
import casacore.quanta as qa
from casacore.tables import table  # type: ignore
from losoto.h5parm import h5parm

from submods.runtime_utils import _track_resource_phase, terminal_print


def parallactic_angles(ms, times, step=30.):
    """
    Parallactic angle (rad) per time and antenna of the field of a single-field MS (apparent frame, as
    parang.py; 0 for equatorial mounts), computed every step seconds and interpolated.
    """
    with table(ms + '::ANTENNA', ack=False) as t:
        positions = t.getcol('POSITION')
        mounts = [m.upper() for m in t.getcol('MOUNT')]
    with table(ms + '::FIELD', ack=False) as t:
        ra, dec = t.getcol('PHASE_DIR')[0, 0]
    utimes = np.unique(times)
    samples = np.unique(np.concatenate([np.arange(utimes[0], utimes[-1], step), [utimes[-1]]]))
    dm = measures()
    field = dm.direction('J2000', qa.quantity(ra, 'rad'), qa.quantity(dec, 'rad'))
    zenith = dm.direction('AZEL', qa.quantity(0., 'deg'), qa.quantity(90., 'deg'))
    frames = [dm.position('ITRF', *(qa.quantity(p, 'm') for p in pos)) for pos in positions]
    chi = np.zeros((len(samples), len(positions)))
    for it, time in enumerate(samples):
        dm.do_frame(dm.epoch('UTC', qa.quantity(time, 's')))
        for ia, pos in enumerate(frames):
            if mounts[ia] == 'EQUATORIAL':
                continue
            dm.do_frame(pos)
            chi[it, ia] = dm.posangle(dm.measure(field, 'APP'), zenith).get_value('rad')
    chi = np.unwrap(chi, axis=0)
    out = np.empty((len(utimes), len(positions)))
    for ia in range(len(positions)):
        out[:, ia] = np.interp(utimes, samples, chi[:, ia])
    index = np.searchsorted(utimes, times)
    return out[index]  # per row, antenna


def read_calibrator(ms, column, fieldindex):
    """Rows of a calibrator MS: data, weights, flags, antennas, times, parallactic angles."""
    with table(ms, ack=False) as t:
        data = t.getcol(column)
        flag = t.getcol('FLAG')
        weight = t.getcol('WEIGHT_SPECTRUM') if 'WEIGHT_SPECTRUM' in t.colnames() else \
            np.repeat(t.getcol('WEIGHT')[:, None, :], data.shape[1], axis=1)
        a1, a2 = t.getcol('ANTENNA1'), t.getcol('ANTENNA2')
        times = t.getcol('TIME')
    keep = a1 != a2
    chi = parallactic_angles(ms, times[keep])
    rows = np.nonzero(keep)[0]
    return dict(data=data[keep], flag=flag[keep], weight=weight[keep], a1=a1[keep], a2=a2[keep],
                time=times[keep], chi1=chi[np.arange(len(rows)), a1[keep]], chi2=chi[np.arange(len(rows)), a2[keep]],
                field=np.full(len(rows), fieldindex))


def solve_dqu(samples, nant, fitpol, nsigma=5., niter=3):
    """
    Solve the leakage of all antennas and the fractional polarisation of the calibrators that have one
    (fitpol) from the cross-hand samples of one channel block.

    Parameters
    ----------
    samples : dict
        Concatenated samples (RL, LR, I, weight, a1, a2, psi = chi_i + chi_j, field).
    nant : int
        Number of antennas.
    fitpol : list of bool
        Per field whether its polarisation is solved (else fixed at 0).
    nsigma : float, optional
        Clip samples with a normalised residual above nsigma. Default 5.
    niter : int, optional
        Solve and clip iterations. Default 3.

    Returns
    -------
    dict
        dR, dL (complex per antenna, NaN where not constrained), p (per field, NaN if not solved), rms of
        the normalised residuals, number of samples used, number of unconstrained directions beyond the gauge.
    """
    polcol = {}
    for f, fit in enumerate(fitpol):
        if fit:
            polcol[f] = 4 * nant + 2 * len(polcol)
    nunk = 4 * nant + 2 * len(polcol)
    n = len(samples['RL'])
    eq = np.arange(n)
    # complex equations: RL samples are 0..n-1, LR samples n..2n-1
    rows, cols, vals = [], [], []
    I, psi, a1, a2, fld = samples['I'], samples['psi'], samples['a1'], samples['a2'], samples['field']
    terms = [(eq, I, 4 * a1, False), (eq, I, 4 * a2 + 2, True),          # RL: d_Ri, conj(d_Lj)
             (eq + n, I, 4 * a1 + 2, False), (eq + n, I, 4 * a2, True)]  # LR: d_Li, conj(d_Rj)
    hasp = np.array([f in polcol for f in fld])
    if hasp.any():
        pc = np.array([polcol.get(f, 0) for f in fld])
        terms += [(eq[hasp], (I * np.exp(-1j * psi))[hasp], pc[hasp], False),
                  (eq[hasp] + n, (I * np.exp(1j * psi))[hasp], pc[hasp], True)]
    for e, coef, col, conj in terms:
        # coef * z (or coef * conj(z) with sign -1) for the complex unknown at real columns col, col + 1:
        # Re(y) = Re(coef) Re(z) - sign Im(coef) Im(z), Im(y) = Im(coef) Re(z) + sign Re(coef) Im(z)
        sign = -1. if conj else 1.
        rows.append(np.concatenate([2 * e, 2 * e, 2 * e + 1, 2 * e + 1]))
        cols.append(np.concatenate([col, col + 1, col, col + 1]))
        vals.append(np.concatenate([coef.real, -sign * coef.imag, coef.imag, sign * coef.real]))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * n, nunk))
    y = np.concatenate([samples['RL'], samples['LR']])
    yreal = np.empty(4 * n)
    yreal[0::2], yreal[1::2] = y.real, y.imag
    w = np.repeat(np.concatenate([samples['weight'], samples['weight']]), 2)
    good = np.ones(4 * n, dtype=bool)
    for it in range(niter):
        W = sp.diags(w * good)
        N = (A.T @ W @ A).toarray()
        b = A.T @ (w * good * yreal)
        constrained = np.diag(N) > 0
        sol = np.full(nunk, np.nan)
        Nc = N[np.ix_(constrained, constrained)]
        sol[constrained] = np.linalg.lstsq(Nc, b[constrained], rcond=1e-12)[0]
        resid = (yreal - A @ np.nan_to_num(sol)) * np.sqrt(w)
        sigma = 1.4826 * np.median(np.abs(resid[good]))
        newgood = np.abs(resid) < nsigma * sigma
        # a complex sample is clipped as a whole (real and imaginary row)
        newgood = np.repeat(newgood.reshape(-1, 2).all(axis=1), 2)
        if it == niter - 1 or np.array_equal(newgood, good):
            break
        good = newgood
    eig = np.linalg.eigvalsh(Nc)
    degenerate = int(np.sum(eig < 1e-9 * eig.max())) - 2  # the gauge has two real directions
    # formal errors (pseudo-inverse: the gauge directions are left out), scaled by the residual rms
    err = np.full(nunk, np.nan)
    err[constrained] = np.sqrt(np.abs(np.diag(np.linalg.pinv(Nc, rcond=1e-9)))) * sigma
    derr = np.hypot(err[:4 * nant:2], err[1:4 * nant:2])
    perr = np.full(len(fitpol), np.nan)
    for f, col in polcol.items():
        perr[f] = np.hypot(err[col], err[col + 1])
    d = sol[:4 * nant:2] + 1j * sol[1:4 * nant:2]
    dR, dL = d[0::2].copy(), d[1::2].copy()
    p = np.full(len(fitpol), np.nan, dtype=complex)
    for f, col in polcol.items():
        p[f] = sol[col] + 1j * sol[col + 1]
    # gauge: mean d_R - conj(d_L) = 0 over the antennas with a solution (fix_leakage_gauge)
    ok = np.isfinite(dR) & np.isfinite(dL)
    c = -np.mean(dR[ok] - np.conj(dL[ok])) / 2.
    dR[ok] += c
    dL[ok] -= np.conj(c)
    return dict(dR=dR, dL=dL, p=p, rms=float(sigma), nsamples=int(good[0::2].sum() // 2),
                clipped=float(1. - good.mean()), degenerate=max(degenerate, 0), gauge_c=float(abs(c)),
                d_error=float(np.nanmedian(derr)), p_error=perr)


def write_leakage_h5(h5out, template_ms, freqs, time, dR, dL, direction):
    """
    Leakage h5parm in the layout of a DP3 DDECal leakage solve (amplitude000, phase000; axes time, freq,
    ant, dir, pol XX, XY, YX, YY; flagged (NaN, weight 0) where there is no solution).

    Parameters
    ----------
    dR, dL : np.ndarray
        Complex leakages (freq, ant).
    """
    with table(template_ms + '::ANTENNA', ack=False) as t:
        names = list(t.getcol('NAME'))
        positions = t.getcol('POSITION')
    nf, na = dR.shape
    jones = np.zeros((1, nf, na, 1, 4), dtype=complex)
    jones[..., 0] = 1.
    jones[..., 3] = 1.
    jones[0, :, :, 0, 1] = dR
    jones[0, :, :, 0, 2] = dL
    ok = np.all(np.isfinite(jones), axis=-1, keepdims=True) & np.ones(jones.shape, dtype=bool)
    if os.path.exists(h5out):
        os.remove(h5out)
    H = h5parm(h5out, readonly=False)
    ss = H.makeSolset('sol000')
    ss.obj.antenna.append(list(zip(names, positions)))
    ss.obj.source.append([('pointing', list(direction))])
    axes = ['time', 'freq', 'ant', 'dir', 'pol']
    axesvals = [np.array([time]), np.asarray(freqs, dtype=float), names, ['pointing'], ['XX', 'XY', 'YX', 'YY']]
    ss.makeSoltab('amplitude', 'amplitude000', axesNames=axes, axesVals=axesvals,
                  vals=np.where(ok, np.abs(jones), np.nan), weights=ok.astype(float))
    ss.makeSoltab('phase', 'phase000', axesNames=axes, axesVals=axesvals,
                  vals=np.where(ok, np.angle(jones), np.nan), weights=ok.astype(float))
    H.close()


@_track_resource_phase("solve")
def solve_leakage(mslist, fieldnames, fitpol, column, h5out, nchan=1, nsigma=5.):
    """
    Leakage (and calibrator polarisation) from the gain-calibrated scans of one or more calibrators in
    one spectral window, written to a leakage h5parm.

    Parameters
    ----------
    mslist : list of str
        Calibrator scan MSs (same spectral window and antennas), each one field.
    fieldnames : list of str
        Field name of each MS.
    fitpol : dict
        Per field name whether its polarisation is solved (False: unpolarised calibrator).
    column : str
        Gain-calibrated data column (RR, RL, LR, LL).
    h5out : str
        Output h5parm.
    nchan : int, optional
        Channels per solution (0: all channels). Default 1.
    nsigma : float, optional
        Clipping threshold of the solve. Default 5.

    Returns
    -------
    dict
        Per channel block: frequency, fractional polarisation and angle per field (the angle includes the
        unknown R-L phase), median |d|, residual rms, samples, clipped fraction and degeneracies.
    """
    fields = sorted(set(fieldnames), key=fieldnames.index)
    parts = [read_calibrator(ms, column, fields.index(name)) for ms, name in zip(mslist, fieldnames)]
    with table(mslist[0] + '::ANTENNA', ack=False) as t:
        nant = t.nrows()
    with table(mslist[0] + '::SPECTRAL_WINDOW', ack=False) as t:
        chanfreq = t.getcell('CHAN_FREQ', 0)
    with table(mslist[0] + '::FIELD', ack=False) as t:
        direction = t.getcol('PHASE_DIR')[0, 0]
    nch = len(chanfreq)
    step = nch if nchan == 0 else max(1, nchan)
    blocks = [np.arange(s, min(s + step, nch)) for s in range(0, nch, step)]
    data = np.concatenate([p['data'] for p in parts])
    flag = np.concatenate([p['flag'] for p in parts])
    weight = np.concatenate([p['weight'] for p in parts])
    a1 = np.concatenate([p['a1'] for p in parts])
    a2 = np.concatenate([p['a2'] for p in parts])
    psi = np.concatenate([p['chi1'] + p['chi2'] for p in parts])
    fld = np.concatenate([p['field'] for p in parts])
    time = float(np.mean(np.concatenate([p['time'] for p in parts])))
    dR = np.full((len(blocks), nant), np.nan, dtype=complex)
    dL = np.full((len(blocks), nant), np.nan, dtype=complex)
    summary = []
    for ib, chans in enumerate(blocks):
        d = data[:, chans]
        ok = ~np.any(flag[:, chans], axis=-1)
        I = 0.5 * (d[..., 0] + d[..., 3])
        ok &= np.abs(I) > 0
        # weight of the cross-hand samples (1 / sigma^2 of the visibility)
        w = 0.5 * (weight[:, chans, 1] + weight[:, chans, 2])
        rowi, chi = np.nonzero(ok)
        samples = dict(RL=d[rowi, chi, 1], LR=d[rowi, chi, 2], I=I[rowi, chi], weight=w[rowi, chi],
                       a1=a1[rowi], a2=a2[rowi], psi=psi[rowi], field=fld[rowi])
        if len(rowi) < 4 * nant:
            terminal_print(f'WARNING: leakage block {ib}: only {len(rowi)} samples, no solution')
            continue
        res = solve_dqu(samples, nant, [fitpol[f] for f in fields], nsigma=nsigma)
        dR[ib], dL[ib] = res['dR'], res['dL']
        pol = {f: (dict(frac=float(abs(res['p'][i])), angle_deg=float(np.degrees(np.angle(res['p'][i])) / 2.))
                   if fitpol[f] else None) for i, f in enumerate(fields)}
        absd = np.abs(np.concatenate([res['dR'], res['dL']]))
        summary.append(dict(freq=float(np.mean(chanfreq[chans])), polarisation=pol,
                            median_abs_d=float(np.nanmedian(absd)), max_abs_d=float(np.nanmax(absd)),
                            rms=res['rms'], nsamples=res['nsamples'], clipped=res['clipped'],
                            degenerate=res['degenerate'], gauge_c=res['gauge_c']))
        if res['degenerate'] > 0:
            terminal_print(f'WARNING: leakage block {ib}: {res["degenerate"]} unconstrained directions beyond the gauge '
                           '(too little parallactic angle coverage to separate the leakage and the calibrator '
                           'polarisation?)')
    freqs = [float(np.mean(chanfreq[c])) for c in blocks]
    write_leakage_h5(h5out, mslist[0], freqs, time, dR, dL, direction)
    for s in summary:
        pols = ', '.join(f'{f}: p={100 * v["frac"]:.2f}% angle {v["angle_deg"]:.1f} deg' for f, v in
                         s['polarisation'].items() if v is not None)
        terminal_print(f'Leakage {h5out} {s["freq"] / 1e6:.1f} MHz: median |d| {s["median_abs_d"]:.4f}, max '
                       f'{s["max_abs_d"]:.4f}, {s["nsamples"]} samples, {100 * s["clipped"]:.1f}% clipped'
                       + (f'; {pols}' if pols else ''))
    return dict(blocks=summary, fields=fields, fitpol={f: bool(fitpol[f]) for f in fields}, h5=h5out)
