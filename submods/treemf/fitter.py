"""
Coherent fit of the TEC + offset + delay phase model to a station's coherent sum.

Model per channel (block) n, with x = nu / nu0:
    phase(x) = a / x**p + b + c * x,
    p = 1:  a = -K_TEC * TEC / nu0,   c = 2 pi delay nu0
which is DP3's tec+phase+delay convention (phase = -K_TEC TEC / nu + b + 2 pi delay nu).
With p = 2 the first term is a rotation measure on phase-difference (RR conj(LL)) data:
    a = 2 c^2 RM / nu0^2, i.e. phase = 2 RM lambda^2 (+ b),
the RR - LL phase of a differential Faraday rotation (scalarphasediff convention).

Ported from the DP3 delay-fitter investigation (harness.FftFitter, tecfit.vm_refine).
"""
import numpy as np

K_TEC = 8.44797245e9
C_LIGHT = 299792458.0


def kernel_scale(kind, nu0):
    """(power p, scale) of the first model term: a = scale * q for q in TECU or rad/m^2."""
    if kind == "rm":
        return 2, 2 * C_LIGHT ** 2 / nu0 ** 2
    return 1, -K_TEC / nu0


def wrap(p):
    """Wrap phases to [-pi, pi)."""
    return np.mod(p + np.pi, 2 * np.pi) - np.pi


def model(x, a, b, c, p=1):
    return a / x ** p + b + c * x


def to_physical(params, nu0):
    """(a, b, c) rows -> TEC [TECU], offset [rad], delay [s]."""
    params = np.atleast_2d(params)
    return (params[:, 0] * nu0 / -K_TEC, params[:, 1],
            params[:, 2] / (2 * np.pi * nu0))


def from_physical(tec, offset, delay, nu0):
    """TEC [TECU], offset [rad], delay [s] -> (a, b, c) rows."""
    return np.stack([-K_TEC * np.asarray(tec) / nu0, np.asarray(offset),
                     2 * np.pi * np.asarray(delay) * nu0], axis=-1)


def gains(params, x, p=1):
    """Unit-modulus gains (nrow, nchan) of parameter rows (a, b, c)."""
    params = np.atleast_2d(params)
    return np.exp(1j * (params[:, 0:1] / x ** p + params[:, 1:2] + params[:, 2:3] * x))


def vm_refine(x, y, w, a, b, c, iters=50, pen=None, free=(True, True, True), p=1):
    """
    Gauss-Newton on sum w (1 - cos(y - model)) in its sin/cos (IRLS) form.

    Parameters
    ----------
    x, y, w : arrays
        x = nu / nu0, the phase arg(z) and the weight |z| per channel (block).
    a, b, c : float
        Start values.
    iters : int
        Maximum number of iterations.
    pen : tuple or None
        (mean, precision) over (a, b, c): adds sum prec (p - mean)^2 / 2.
    free : tuple of bool
        Which of (a, b, c) are refined; the others stay at their start values.
    p : int
        Power of the first term (a / x**p).

    Returns
    -------
    numpy.ndarray
        The refined (a, b, c).
    """
    par = np.array([a, b, c], float)
    fr = np.asarray(free, bool)
    J = np.vstack([1.0 / x ** p, np.ones_like(x), x]).T[:, fr]
    pm, pp = (np.zeros(3), np.zeros(3)) if pen is None else pen
    pm, pp = pm[fr], pp[fr]
    for _ in range(iters):
        r = y - model(x, *par, p=p)
        g = J.T @ (w * np.sin(r)) - pp * (par[fr] - pm)
        H = (J * (w * np.maximum(np.cos(r), 0.05))[:, None]).T @ J + np.diag(pp)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:  # fewer channels with data than free parameters
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        par[fr] += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return par


class FftFitter:
    """
    Global coherent search: maximise |sum_n z_n exp(-i (a / x_n**p + c x_n))| over a TEC (or RM)
    grid and the delay bins within the delay bound, then refine the best candidates. Without the
    phase offset in the model the objective is Re(sum_n z_n exp(-i (a / x_n**p + c x_n))).
    The grid of the first parameter is `tecs` (TECU, or rad/m^2 for kind 'rm').
    """

    def __init__(self, nu, tec_range=4.0, tec_step=0.005, nfft=2048, topk=3, delay_max=20e-9,
                 fit_tec=True, fit_offset=True, fit_delay=True, kind="tec"):
        """
        Parameters
        ----------
        nu : numpy.ndarray
            Channel (block) frequencies in Hz, increasing, on a regular grid with gaps allowed.
        tec_range : float
            The TEC grid covers +- tec_range TECU (a hard bound on TEC relative to the reference).
        tec_step : float
            TEC grid step (TECU).
        nfft : int
            FFT length that defines the delay bins (at least twice the channel span).
        topk : int
            Number of TEC grid rows refined.
        delay_max : float
            Hard bound on |delay| (s); only the allowed delay bins are computed.
        fit_tec, fit_offset, fit_delay : bool
            Which parameters the model has; the others are fixed at 0.
        kind : str
            'tec' (first term -K_TEC TEC / nu) or 'rm' (2 RM lambda^2 on phase-difference data);
            for 'rm', tec_range and tec_step are the RM bound and grid step (rad/m^2), and a
            tec_step of None chooses the step from the band (see below).
        """
        self.kind = kind
        self.p, self.scale = kernel_scale(kind, nu[0])
        self.free = (fit_tec, fit_offset, fit_delay)
        self.fit_offset = fit_offset
        if not fit_tec:
            tec_range = 0.0
        if not fit_delay:
            delay_max = 0.0
        self.topk = topk
        self.nu, self.x = nu, nu / nu[0]
        self.dnu = np.median(np.diff(nu))
        self.chan = np.rint((nu - nu[0]) / self.dnu).astype(int)
        self.nfft = max(nfft, 2 * (self.chan[-1] + 1))
        step = 0.25
        # phase change across the band of a unit step in the first parameter
        span = abs(self.scale) * (1 - self.x[-1] ** -self.p)
        if tec_step is None:
            # with the offset free it absorbs the mid-band error of a cell, so a cell may move the
            # phase by at most 0.25 rad across the band; without it, anywhere in the band
            tec_step = step / span if fit_offset else step / abs(self.scale)
        if not fit_offset:
            # without a free offset a grid cell must not move the phase by more than ~0.25 rad
            # anywhere in the band, or the cells next to the optimum score below a 2 pi alias
            # (with the offset free, it absorbs most of the cell error)
            self.nfft = max(self.nfft, int(2 ** np.ceil(np.log2(2 * np.pi * nu[-1] / (step * self.dnu)))))
            tec_step = min(tec_step, step / abs(self.scale))
        self.tecs = np.arange(-tec_range, tec_range + tec_step / 2, tec_step)
        self.a = self.scale * self.tecs
        self.E = np.exp(-1j * self.a[:, None] / self.x[None, :] ** self.p)
        fr = np.fft.fftfreq(self.nfft, d=self.dnu)
        k = np.flatnonzero(np.abs(fr) <= delay_max)
        if len(k) == 0:
            k = np.array([0])
        # the allowed FFT bins as a matrix product: the same values as the FFT, far fewer cells
        self.Wk = np.exp(-2j * np.pi * np.outer(self.chan, k) / self.nfft)
        self.taus = fr[k]
        self.cks = 2 * np.pi * self.taus * nu[0]  # c of each delay bin
        # hard bounds on (a, c): the refinement may not leave them
        self.a_bound = abs(self.scale) * tec_range * (1 + 1e-9)
        self.c_bound = 2 * np.pi * nu[0] * max(delay_max, np.abs(self.taus).max()) * (1 + 1e-9)
        self.centre = None

    def set_centre(self, centre):
        """
        Per-antenna centres (s) of the delay window, or None (0): the delay bound of antenna a then
        applies to |delay - centre[a]|. The fits demodulate z by the centre delay, so a window
        costs the same wherever it is.
        """
        self.centre = None if centre is None or not np.any(centre) else np.asarray(centre, float)

    def centre_of(self, a):
        """Delay-window centre (s) of antenna a."""
        return 0.0 if self.centre is None or a is None else float(self.centre[a])

    def shift(self, a):
        """The window centre of antenna a in the units of c."""
        return 2 * np.pi * self.nu[0] * self.centre_of(a)

    def coherence(self, z):
        """F[i, k] = sum_n z_n exp(-i a_i / x_n**p) exp(-2 pi i k n / nfft)."""
        return (self.E * z[None, :]) @ self.Wk

    def synthesis(self, W):
        """Adjoint of coherence: h_n = sum_ik W_ik e^{i a_i / x_n**p} e^{2 pi i k n / nfft}."""
        G = W @ np.conj(self.Wk).T
        return np.sum(np.conj(self.E) * G, axis=0)

    def fit(self, z, a=None, prior=None):
        """
        Best (a, b, c) for a coherent sum z; None without data.

        With a prior (prior.Prior) the objective is sum |z| cos(arg z - model) + sigma^2 log p;
        `a` is the antenna index the prior and the delay-window centre are looked up for.
        """
        if not np.any(np.abs(z) > 0):
            return None
        cs = self.shift(a)
        if cs:
            z = z * np.exp(-1j * cs * self.x)  # fit in the frame of the window centre
        F = self.coherence(z)
        # sum_n z conj(model) = e^{-i c_k} F_ik; its modulus when the offset is free
        S = np.abs(F) if self.fit_offset else np.real(F * np.exp(-1j * self.cks)[None, :])
        pen, boff = None, None
        if prior is not None:
            off_prec = prior.sig2 / prior.sig_off ** 2 if (
                self.fit_offset and np.isfinite(prior.sig_off) and a != prior.ref) else 0.0
            if off_prec > 0:
                # zero-mean Gaussian prior on the offset b: maximise |F| cos(arg F - c - b)
                # - prec b^2 / 2 over b, linearised: b = phi |F| / (|F| + prec)
                phi = wrap(np.angle(F) - self.cks[None, :])
                boff = phi * np.abs(F) / (np.abs(F) + off_prec)
                S = np.abs(F) * np.cos(phi - boff) - 0.5 * off_prec * boff ** 2
            S = S + prior.sig2 * prior.lp(a, self.tecs, self.taus, self.centre_of(a))
            pen = prior.penalty(a)
            pen[0][2] -= cs
        kbest = np.argmax(S, axis=1)
        sbest = S[np.arange(len(self.a)), kbest]
        y, w = np.angle(z), np.abs(z)
        best, bc = None, np.inf
        for i in np.argsort(-sbest)[:self.topk]:
            c = self.cks[kbest[i]]
            if boff is not None:
                b = boff[i, kbest[i]]
            else:
                b = np.angle(F[i, kbest[i]]) - c if self.fit_offset else 0.0
            p = vm_refine(self.x, y, w, self.a[i], b, c, pen=pen, free=self.free, p=self.p)
            if abs(p[0]) > self.a_bound or abs(p[2]) > self.c_bound:
                p = np.array([self.a[i], b, c])  # refined out of the bounds: keep the grid point
            cost = -np.sum(w * np.cos(y - model(self.x, *p, p=self.p)))
            if pen is not None:
                cost += 0.5 * np.sum(pen[1] * (p - pen[0]) ** 2)
            if cost < bc:
                best, bc = p, cost
        if best is not None:
            best[2] += cs
        return best

    def gains(self, params):
        """Unit-modulus gains (nrow, nblk) of parameter rows (a, b, c)."""
        return gains(params, self.x, self.p)

    def physical(self, params):
        """(a, b, c) rows -> first parameter (TECU or rad/m^2), offset [rad], delay [s]."""
        params = np.atleast_2d(params)
        return (params[:, 0] / self.scale, params[:, 1],
                params[:, 2] / (2 * np.pi * self.nu[0]))


def meanfield_update(z, fitter, sig2, T, a=None, prior=None):
    """
    Posterior-mean gain of one station given its coherent sum z (mean-field start-up).

    The conditional posterior over the (TEC, delay) cells is P ~ I0(kappa) p(cell),
    kappa = |F| / (sigma^2 T), F the coherence map of z and p the prior; given the cell
    the offset is von Mises. So
        h(n) = sum_cells P A(kappa) e^{i(arg F - c)} e^{i(a / x_n**p + c x_n)},  A = I1 / I0.
    Without the offset in the model, P ~ exp(Re(e^{-i c} F) / (sigma^2 T)) p(cell) and
        h(n) = sum_cells P e^{i(a / x_n**p + c x_n)}.
    """
    from scipy.special import i0e, i1e
    cs = fitter.shift(a)
    if cs:
        z = z * np.exp(-1j * cs * fitter.x)
    F = fitter.coherence(z)
    if fitter.fit_offset:
        kap = np.abs(F) / (sig2 * T)
        L = np.log(i0e(kap)) + kap
    else:
        L = np.real(F * np.exp(-1j * fitter.cks)[None, :]) / (sig2 * T)
    if prior is not None:
        L = L + prior.lp(a, fitter.tecs, fitter.taus, fitter.centre_of(a))
    P = np.exp(L - L.max())
    P /= P.sum()
    if fitter.fit_offset:
        W = P * (i1e(kap) / i0e(kap)) * np.exp(1j * np.angle(F))
    else:
        W = P * np.exp(1j * fitter.cks)[None, :]
    h = fitter.synthesis(W)
    return h * np.exp(1j * cs * fitter.x) if cs else h
