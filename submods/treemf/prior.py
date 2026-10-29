"""
Per-station priors on TEC and delay for the treeMF solver.

TEC of station a relative to the reference station r is Gaussian:
  geometric   mean 0, width sigma(d_ar) = clip(s100 (d_ar / 100 km)^(5/6), smin, smax), with d_ar
              the pierce-point separation in a thin layer (the Kolmogorov structure function of the
              ionosphere); with cal_sep (km, calibrator-transferred phases) the width saturates at
              sqrt(2) sigma(cal_sep)
  propagated  mean the previous interval's solution, a fixed width
Delay: a hard bound (the fitter's delay grid) and optionally a Gaussian (mean 0, or the previous
interval's delay when propagating).
For the 'rm' models on phase-difference data the TEC prior is a prior on the rotation measure
(rad/m^2, the same geometric form, or a mean from RMextract), and the offset can have a zero-mean
Gaussian prior.

The coherent objective of a station (partners fixed) is sum_ch |z| cos(arg z - model) / sigma^2,
so the fits maximise |F| + sigma^2 log p, and the mean-field posterior is I0(|F| / (sigma^2 T)) p.
"""
import numpy as np

from submods.treemf.fitter import K_TEC

R_E = 6371e3


def source_itrf(pdir, t, pos0):
    """Unit vector towards the source (ra, dec in rad) in ITRF at time t (MJD seconds)."""
    from casacore.measures import measures
    dm = measures()
    dm.do_frame(dm.epoch("UTC", f"{t}s"))
    dm.do_frame(dm.position("ITRF", *[f"{v}m" for v in pos0]))
    d = dm.measure(dm.direction("J2000", f"{pdir[0]}rad", f"{pdir[1]}rad"), "ITRF")
    lon, lat = d["m0"]["value"], d["m1"]["value"]
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def pierce(pos, s, h):
    """Pierce points (n, 3) of the sightlines s from pos through a sphere of radius R_E + h."""
    b = pos @ s
    r2 = np.sum(pos ** 2, axis=1)
    L = -b + np.sqrt(b ** 2 - (r2 - (R_E + h) ** 2))
    return pos + L[:, None] * s


def tangent_xy(pp, i0):
    """East/north offsets (km) of pierce points from pierce point i0."""
    up = pp[i0] / np.linalg.norm(pp[i0])
    east = np.cross([0, 0, 1.0], up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    d = pp - pp[i0]
    return np.stack([d @ east, d @ north], 1) / 1e3


def find_anchor(names, pos):
    """The array centre station: ST001 if present, else a CS002 field, else the station
    closest to the median position."""
    if "ST001" in names:
        return names.index("ST001")
    for i, n in enumerate(names):
        if n.startswith("CS002"):
            return i
    return int(np.argmin(np.linalg.norm(pos - np.median(pos, axis=0), axis=1)))


def geometry(names, pos, pdir, t, height=300e3):
    """
    Pierce-point geometry for the priors.

    Returns
    -------
    dict
        x: east/north pierce-point offsets (km) from the anchor's, at time t;
        dground: ground distance (km) from the anchor; anchor: its index.
    """
    i0 = find_anchor(names, pos)
    s = source_itrf(pdir, t, pos[i0])
    x = tangent_xy(pierce(pos, s, height), i0)
    return dict(x=x, dground=np.linalg.norm(pos - pos[i0], axis=1) / 1e3, anchor=i0)


class Prior:
    """
    Per-interval prior, relative to the reference antenna `ref`. sig2 is the noise variance per
    real/imaginary component (sqrt(w)-scaled units), so that sigma^2 log p is in the units of the
    coherent objective.
    """

    def __init__(self, nant, ref, sig2, nu0, tec_mean=None, tec_sigma=None, delay_mean=None,
                 delay_sigma=None, scale=None, offset_sigma=None):
        """
        Parameters
        ----------
        nant : int
            Number of antennas.
        ref : int
            Reference antenna (no prior on it).
        sig2 : float
            Noise variance per real/imaginary component.
        nu0 : float
            Reference frequency of the fitter (Hz).
        tec_mean, tec_sigma : arrays or None
            Per-antenna TEC prior mean and width (TECU, relative to ref); inf = no TEC prior.
        delay_mean, delay_sigma : arrays or None
            Per-antenna delay prior mean and width (s); inf = no Gaussian delay prior.
        scale : float or None
            The fitter's scale of the first parameter (a = scale * q); default -K_TEC / nu0 (TEC).
        offset_sigma : float or None
            Width (rad) of a zero-mean Gaussian prior on the offset; None = no offset prior.
        """
        self.sig2, self.nu0 = sig2, nu0
        self.scale = -K_TEC / nu0 if scale is None else scale
        self.sig_off = np.inf if offset_sigma is None else float(offset_sigma)
        self.mu = np.zeros(nant) if tec_mean is None else np.asarray(tec_mean, float).copy()
        self.sig = np.full(nant, np.inf) if tec_sigma is None else np.asarray(tec_sigma, float).copy()
        self.mu_tau = np.zeros(nant) if delay_mean is None else np.asarray(delay_mean, float).copy()
        self.sig_tau = (np.full(nant, np.inf) if delay_sigma is None
                        else np.broadcast_to(np.asarray(delay_sigma, float), (nant,)).copy())
        self.ref = ref
        self.sig[ref] = np.inf
        self.sig_tau[ref] = np.inf
        self._cache = {}

    def lp(self, a, tecs, taus, shift=0.0):
        """log prior on the (TEC, delay) grid of a fitter for antenna a, its delays taus + shift
        (s; the window centre); cached per grid."""
        key = (a, id(tecs), id(taus), shift)
        if key not in self._cache:
            lt = np.zeros(len(tecs))
            if a is not None and np.isfinite(self.sig[a]):
                lt = -0.5 * ((tecs - self.mu[a]) / self.sig[a]) ** 2
            ld = np.zeros(len(taus))
            if a is not None and np.isfinite(self.sig_tau[a]):
                ld = -0.5 * ((taus + shift - self.mu_tau[a]) / self.sig_tau[a]) ** 2
            self._cache[key] = lt[:, None] + ld[None, :]
        return self._cache[key]

    def penalty(self, a):
        """For vm_refine: prior mean and precision of (a, b, c), the precision times sig2."""
        mean, prec = np.zeros(3), np.zeros(3)
        if a is not None and np.isfinite(self.sig[a]):
            mean[0] = self.scale * self.mu[a]
            prec[0] = self.sig2 / (self.scale * self.sig[a]) ** 2
        if a is not None and a != self.ref and np.isfinite(self.sig_off):
            prec[1] = self.sig2 / self.sig_off ** 2
        if a is not None and np.isfinite(self.sig_tau[a]):
            mean[2] = 2 * np.pi * self.nu0 * self.mu_tau[a]
            prec[2] = self.sig2 / (2 * np.pi * self.nu0 * self.sig_tau[a]) ** 2
        return mean, prec


def geometric_tec_sigma(geo, ref, s100, smin=0.02, smax=None, cal_sep=None):
    """Per-antenna TEC prior width (TECU) relative to ref from the pierce-point separation."""
    d = np.linalg.norm(geo["x"] - geo["x"][ref], axis=1)
    if cal_sep is not None:
        d, s100 = np.minimum(d, cal_sep), np.sqrt(2) * s100
    return np.clip(s100 * (d / 100.0) ** (5 / 6), smin, np.inf if smax is None else smax)
