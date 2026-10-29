"""
Delay-window centres for the treeMF solver from the data, and a check of the delay bound.

The delay of a station (mostly its clock, after calibrator transfer) changes slowly, while the
TEC changes from interval to interval. So the delay can be estimated from many intervals together,
even when one interval alone cannot set it (treemf.delaycentre=track):
1. Per interval and station, the map log I0(|F(TEC, tau)| / sigma^2) over the TEC bound and a wide
   delay range (delaytrack_search), from the station's baselines to the reference only (unit gain
   there, so no solve is needed), the offset marginalised; F the coherence map of the station's
   sum over its reference baselines. The map is smoothed along TEC by a Gaussian of
   delaytrack_tecwander (in linear space): the interval's TEC may wander by that much around the
   block's TEC.
2. The maps summed over the intervals of a block (delaytrack_block minutes), then the TEC prior
   added once and the TEC marginalised: S_b(tau) = log sum_TEC exp(sum_i map_i) p(TEC); minus its
   median over tau. The prior enters once per block: marginalising the TEC with the prior in every
   interval and summing those profiles counts the prior once per interval, which at low S/N pulls
   the TEC towards the prior mean and moves the delay along the TEC-delay degeneracy (about
   65 ns per TECU at 144 MHz): tens of ns of bias in the simulation. Without the wander (a fixed
   TEC per block) the real ionosphere, whose TEC changes within a block, smears the block sum and
   the prior pulls again (errors up to 28 ns on ILTJ165558). Plan, "Data-driven delay windows".
3. A Viterbi track through the blocks: the path that maximises the summed block profiles and
   moves by at most delaytrack_maxstep per block; interpolated linearly to the intervals.
4. A permutation test: the same with each interval's map shifted circularly along tau by a random
   delay before the block sums, which keeps the noise of the maps but breaks their alignment over
   time. A station's track is used only if its score beats all delaytrack_nperm shuffled scores (a
   false-alarm probability below 1 / (nperm + 1)); otherwise its centre stays 0.

The bound check (treemf.delayadapt): after a solve, the solutions of a station that pile up at
one edge of its delay window within a block mean the window is in the wrong place there; the
window of that station and block is then moved by half the bound towards the pile-up (half, so
that a delay inside the window but near its edge ends up inside, not beyond the other edge) and
the intervals of the block are solved again.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import i0e, logsumexp


def interval_maps(prob, ref_nodes, wide, sig2, prior, nant, wander=0.0):
    """
    Delay-TEC likelihood maps per antenna for one interval.

    Parameters
    ----------
    prob : dict
        From solver.build_nodes.
    ref_nodes : list
        The reference antenna's nodes (unit gain).
    wide : fitter.FftFitter
        Fitter with the TEC grid and the wide delay range of the track.
    sig2 : float
        Noise variance per real/imaginary component.
    prior : prior.Prior or None
        TEC prior (no delay prior).
    nant : int
        Number of antennas.
    wander : float
        Width (TECU) of the Gaussian the maps are smoothed with along TEC (0: none).

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        (nant, ntec, ntau) maps log I0(|F| / sigma^2), summed over an antenna's nodes and smoothed,
        NaN for antennas without baselines to the reference; (nant, ntec) log prior on the TEC
        grid.
    """
    nodes = prob["nodes"]
    maps = np.full((nant, len(wide.tecs), len(wide.taus)), np.nan, np.float32)
    lp = np.zeros((nant, len(wide.tecs)), np.float32)
    if not len(ref_nodes):
        return maps, lp
    refflat = np.concatenate([nodes[k]["members"] for k in ref_nodes])
    ref_set = set(ref_nodes)
    for k, nd in enumerate(nodes):
        if k in ref_set or nd["sens"] <= 0:
            continue
        sel = np.isin(nd["pidx"], refflat)
        if not np.any(sel):
            continue
        z = np.sum(nd["Y"][sel], axis=0)
        if not np.any(np.abs(z) > 0):
            continue
        kap = np.abs(wide.coherence(z)) / sig2
        a = nd["ant"]
        maps[a] = np.nan_to_num(maps[a]) + np.log(i0e(kap)) + kap
        if prior is not None:
            lp[a] = prior.lp(a, wide.tecs, wide.taus)[:, 0]
    if wander > 0:
        cells = wander / abs(wide.tecs[1] - wide.tecs[0])
        for a in range(nant):
            if np.all(np.isfinite(maps[a])):
                m = maps[a].max()
                sm = gaussian_filter1d(np.exp(maps[a].astype(float) - m), cells, axis=0, mode="constant")
                maps[a] = np.log(np.maximum(sm, 1e-300)) + m
    return maps, lp


class BlockStack:
    """
    The block sums of the interval maps per antenna, and the same with each interval's map shifted
    circularly along tau by a random delay (for the permutation test).
    """

    def __init__(self, nblock, nant, ntec, ntau, nperm=20, seed=0):
        self.shape = (nperm + 1, nblock, ntec, ntau)
        self.nperm, self.nant, self.ntau = nperm, nant, ntau
        self.sums = {}  # antenna -> (nperm + 1, nblock, ntec, ntau); index 0 unshifted
        self.lp = np.zeros((nblock, nant, ntec))
        self.count = np.zeros((nblock, nant), int)
        self.rng = np.random.default_rng(seed)

    def add(self, b, maps, lp):
        """Add one interval's maps (nant, ntec, ntau) and log priors (nant, ntec) to block b."""
        for a in range(self.nant):
            if not np.all(np.isfinite(maps[a])):
                continue
            if a not in self.sums:
                self.sums[a] = np.zeros(self.shape, np.float32)
            acc = self.sums[a]
            acc[0, b] += maps[a]
            for p, k in enumerate(self.rng.integers(0, self.ntau, self.nperm)):
                acc[p + 1, b] += np.roll(maps[a], k, axis=1)
            self.lp[b, a] += lp[a]
            self.count[b, a] += 1

    def track(self, taus, maxstep):
        """
        Delay track per antenna (Viterbi through the block profiles, the TEC prior once per block)
        and the permutation test.

        Parameters
        ----------
        taus : numpy.ndarray
            The delays (s) of the map bins, on a regular grid.
        maxstep : float
            Largest change of the track between blocks (s).

        Returns
        -------
        numpy.ndarray, dict
            Track per block and antenna (nblock, nant; s, 0 where not used), and per antenna:
            used (bool), score, null (the largest shuffled score), nint (intervals with data).
        """
        nblock = self.shape[1]
        step = max(1, int(np.ceil(maxstep / abs(taus[1] - taus[0]))))
        cen = np.zeros((nblock, self.nant))
        info = dict(used=np.zeros(self.nant, bool), score=np.zeros(self.nant),
                    null=np.zeros(self.nant), nint=self.count.sum(axis=0))
        for a, acc in self.sums.items():
            cnt = np.maximum(self.count[:, a], 1)
            lp_once = (self.lp[:, a] / cnt[:, None])[:, :, None]  # the prior once per block
            scores = []
            for p in range(self.nperm + 1):
                Sb = logsumexp(acc[p] + lp_once, axis=1)
                Sb[self.count[:, a] == 0] = 0.0
                path, score = viterbi(Sb - np.median(Sb, axis=1, keepdims=True), step)
                if p == 0:
                    best = path
                scores.append(score)
            info["score"][a], info["null"][a] = scores[0], max(scores[1:]) if self.nperm else 0.0
            if scores[0] > info["null"][a]:
                info["used"][a] = True
                cen[:, a] = taus[best]
        return cen, info


def viterbi(S, maxstep):
    """
    Path through the rows of S (nblock, ntau) that maximises the sum of S along it, moving by at
    most maxstep bins between rows: (bin per row, the summed score).
    """
    nb, n = S.shape
    acc = S[0].copy()
    back = np.zeros((nb, n), int)
    idx = np.arange(n)
    for b in range(1, nb):
        best = np.full(n, -np.inf)
        arg = idx.copy()
        for o in range(-maxstep, maxstep + 1):
            src = idx - o
            ok = (src >= 0) & (src < n)
            v = np.full(n, -np.inf)
            v[ok] = acc[src[ok]]
            better = v > best
            best[better] = v[better]
            arg[better] = src[better]
        acc = best + S[b]
        back[b] = arg
    path = np.zeros(nb, int)
    path[-1] = int(np.argmax(acc))
    for b in range(nb - 1, 0, -1):
        path[b - 1] = back[b, path[b]]
    return path, float(acc.max())


def to_intervals(block_track, block_of, nint):
    """Block track (nblock, nant) linearly interpolated to the intervals (nint, nant), with the
    block centres at the mean interval index of each block."""
    nblock = block_track.shape[0]
    mid = np.array([np.mean(np.flatnonzero(block_of == b)) if np.any(block_of == b) else np.nan
                    for b in range(nblock)])
    good = np.isfinite(mid)
    i = np.arange(nint)
    return np.stack([np.interp(i, mid[good], block_track[good, a])
                     for a in range(block_track.shape[1])], axis=1)


def check_bound(delay, weight, centre, bound, block_of, frac=0.1, tol=1e-9):
    """
    The at-bound check: per block and antenna, the fraction of solutions within tol of each edge of
    the window centre +- bound.

    Parameters
    ----------
    delay, weight, centre : numpy.ndarray
        Per interval and antenna (nint, nant): the solved delay (s) and weight, and the window
        centre used, all relative to the interval's reference.
    bound : float
        The delay bound (s).
    block_of : numpy.ndarray
        Block index of each interval.
    frac : float
        Pile-up threshold: at least this fraction at one edge (and at least 3x what a uniform spread
        gives, and 3x the other edge).

    Returns
    -------
    numpy.ndarray, list
        Shift of the window per interval and antenna (s; +-bound / 2 where a pile-up was found,
        else 0), and (block, antenna, side, fraction) of each pile-up.
    """
    dev = delay - centre
    shift = np.zeros_like(delay)
    found = []
    expect = tol / (2 * bound)
    for b in np.unique(block_of):
        rows = block_of == b
        for a in range(delay.shape[1]):
            ok = weight[rows, a] > 0
            if ok.sum() < 5:
                continue
            d = dev[rows, a][ok]
            hi, lo = np.mean(d > bound - tol), np.mean(d < -bound + tol)
            f, other, side = (hi, lo, 1) if hi >= lo else (lo, hi, -1)
            if f >= max(frac, 3 * expect) and f > 3 * other:
                shift[rows, a] = side * bound / 2
                found.append((int(b), a, side, float(f)))
    return shift, found
