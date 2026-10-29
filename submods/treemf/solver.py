"""
treeMF solver for one solution interval: a mean-field ("soft tree") start-up followed by coherent
per-node fits with sequential (Gauss-Seidel) sweeps.

The unknowns are nodes. A node is a set of (time slot, antenna) pairs that share one
(TEC, offset, delay): one per antenna and interval (solint), one per antenna and group of
`factor` sub-solutions (solutions_per_direction with antenna_averaging_factors), and antennas
in an antennaconstraint group share their nodes. A node's coherent sum over its slots and
baselines is
    z(ch) = sum_(slot, antenna in node) sum_partners Y(ch) h_partner(slot, ch),
    Y = d conj(m) (sqrt(w)-scaled data and model),
so the model is fitted to all of the node's data at once.

Ported from the DP3 delay-fitter investigation (harness.tree_meanfield and friends).
"""
import numpy as np

from submods.treemf.fitter import meanfield_update, wrap


def build_nodes(slots, nant, solint, nsub, factors, group_of, blocks):
    """
    Nodes and their terms for one interval.

    Parameters
    ----------
    slots : list
        One entry per time slot of the interval (None for a missing slot), each a dict with
        a1, a2 (antenna indices), d, m (sqrt(w)-scaled data and model, (nbl, nchan)).
    nant : int
        Number of antennas.
    solint : int
        Time slots per interval.
    nsub : int
        Sub-solutions per interval (solutions_per_direction).
    factors : numpy.ndarray
        Per-antenna averaging factor in sub-solutions.
    group_of : numpy.ndarray
        Per-antenna group id (antennaconstraint); antennas with the same id share nodes.
    blocks : numpy.ndarray
        Start channel of each channel block.

    Returns
    -------
    dict
        nodes: list of dicts (ant: antenna for the prior, members: flat (slot, antenna) indices,
        pidx: flat partner indices, Y, M2: per-row terms summed over channel blocks, sens,
        nblk: number of channel blocks with data);
        node_of: (solint, nant) node index of each (slot, antenna); nflat = solint * nant.
    """
    keys, node_of = {}, np.zeros((solint, nant), int)
    for a in range(nant):
        for t in range(solint):
            key = (group_of[a], (t * nsub // solint) // factors[a])
            if key not in keys:
                keys[key] = len(keys)
            node_of[t, a] = keys[key]
    nn = len(keys)
    ant_of = np.zeros(nn, int)
    for a in range(nant - 1, -1, -1):  # the first antenna of a group carries the prior
        ant_of[node_of[:, a]] = a
    parts = [[] for _ in range(nn)]
    for t, sl in enumerate(slots):
        if sl is None:
            continue
        a1, a2, d, m = sl["a1"], sl["a2"], sl["d"], sl["m"]
        y = np.add.reduceat(d * np.conj(m), blocks, axis=1)
        m2 = np.add.reduceat(np.abs(m) ** 2, blocks, axis=1)
        n1, n2 = node_of[t, a1], node_of[t, a2]
        keep = n1 != n2  # baselines within a node carry no information on its phase
        for side, (own, oth, yy) in enumerate(((a1, a2, y), (a2, a1, np.conj(y)))):
            for k in np.unique(node_of[t, own[keep]]):
                sel = keep & (node_of[t, own] == k)
                parts[k].append((t * nant + oth[sel], yy[sel], m2[sel]))
    nodes = []
    nblk = len(blocks)
    for k in range(nn):
        if parts[k]:
            pidx = np.concatenate([p[0] for p in parts[k]])
            Y = np.concatenate([p[1] for p in parts[k]])
            M2 = np.concatenate([p[2] for p in parts[k]])
        else:
            pidx, Y, M2 = np.zeros(0, int), np.zeros((0, nblk), complex), np.zeros((0, nblk))
        members = np.flatnonzero(node_of.ravel() == k)
        nodes.append(dict(ant=ant_of[k], members=members, pidx=pidx, Y=Y, sens=float(M2.sum()),
                          nblk=int(np.sum(M2.sum(axis=0) > 0))))
    return dict(nodes=nodes, node_of=node_of, nflat=solint * nant)


def phasediff_amplitude(slots):
    """
    Model-free amplitude of phase-difference data y = RR conj(LL) (d = sqrt(w) y): the mean
    coherence |<y_n conj(y_n+1)>| of adjacent channels per baseline, which is alpha^2 for
    y = alpha e^{i phi} + noise when phi changes little between adjacent channels (a phase
    gradient common to the channels of a baseline drops out of the modulus). The modulus of a
    noise-only sum, sqrt(pi)/2 times its rms, is subtracted.
    """
    num = den = 0.0
    for sl in slots:
        if sl is None:
            continue
        d, sw = sl["d"], np.sqrt(sl["w"])
        if d.shape[1] < 2:
            continue
        prod = d[:, :-1] * np.conj(d[:, 1:])
        num += np.sum(np.abs(np.sum(prod, axis=1)))
        num -= 0.5 * np.sqrt(np.pi) * np.sum(np.sqrt(np.sum(np.abs(prod) ** 2, axis=1)))
        den += np.sum(sw[:, :-1] * sw[:, 1:])
    return np.sqrt(num / den) if den > 0 and num > 0 else 1e-3


def noise_var(slots):
    """sigma^2 per real/imag component from the data alone: |d|^2 = |m|^2 + 2 sigma^2 on
    average for unit gains."""
    pd = pm = n = 0.0
    for sl in slots:
        if sl is None:
            continue
        ok = sl["w"] > 0
        pd += np.sum(np.abs(sl["d"][ok]) ** 2)
        pm += np.sum(np.abs(sl["m"][ok]) ** 2)
        n += ok.sum()
    if n == 0:
        return 1.0
    pd, pm = pd / n, pm / n
    return max((pd - pm) / 2, 1e-3 * pm)


def _z(node, hflat):
    return np.sum(node["Y"] * hflat[node["pidx"]], axis=0)


def treemf(prob, ref_nodes, fitter, sig2, prior=None, nsweep=30, T0=3.0, maxsweeps=30, tol=1e-4):
    """
    Mean-field start-up, then hard coherent fits and Gauss-Seidel sweeps over the nodes.

    Parameters
    ----------
    prob : dict
        From build_nodes.
    ref_nodes : list
        Nodes fixed at unit gain (the reference antenna's).
    fitter : fitter.FftFitter
        Coherent fitter on the channel blocks.
    sig2 : float
        Noise variance per real/imaginary component.
    prior : prior.Prior or None
        Per-antenna prior.
    nsweep : int
        Number of mean-field sweeps; the temperature falls from T0 to 1 over the first 2/3.
    T0 : float
        Start temperature.
    maxsweeps, tol : int, float
        Hard sweeps until the mean phase change is below tol.

    Returns
    -------
    numpy.ndarray, int
        Parameters (a, b, c) per node relative to the reference, and the number of sweeps.
    """
    nodes, nflat = prob["nodes"], prob["nflat"]
    nn, x = len(nodes), fitter.x
    sens = np.array([nd["sens"] for nd in nodes])
    ref_set = set(ref_nodes)
    order = [k for k in np.argsort(-sens) if k not in ref_set and sens[k] > 0]
    nhot = max(1, (2 * nsweep) // 3)
    h = np.zeros((nn, len(x)), complex)
    hflat = np.zeros((nflat, len(x)), complex)
    for k in ref_nodes:
        h[k] = 1.0
        hflat[nodes[k]["members"]] = 1.0
    for it in range(nsweep):
        T = T0 * (1.0 / T0) ** min(1.0, it / max(nhot - 1, 1))
        for k in order:
            z = _z(nodes[k], hflat)
            if not np.any(np.abs(z) > 0):
                continue
            h[k] = meanfield_update(z, fitter, sig2, T, nodes[k]["ant"], prior)
            hflat[nodes[k]["members"]] = h[k]
    params = np.zeros((nn, 3))
    for k in order:
        f = fitter.fit(_z(nodes[k], hflat), a=nodes[k]["ant"], prior=prior)
        if f is not None and np.all(np.isfinite(f)):
            params[k] = f
    return converge(prob, params, ref_set, fitter, prior, maxsweeps, tol)


def converge(prob, params, ref_set, fitter, prior=None, maxsweeps=30, tol=1e-4):
    """Gauss-Seidel sweeps of hard coherent fits (step 1) until the phases stop changing."""
    nodes, x = prob["nodes"], fitter.x
    params = params.copy()
    for k in ref_set:
        params[k] = 0.0
    g = fitter.gains(params)
    hflat = np.zeros((prob["nflat"], len(x)), complex)
    for k, nd in enumerate(nodes):
        hflat[nd["members"]] = g[k]
    order = [k for k in np.argsort([-nd["sens"] for nd in nodes]) if k not in ref_set]
    g_old = g.copy()
    for it in range(maxsweeps):
        for k in order:
            f = fitter.fit(_z(nodes[k], hflat), a=nodes[k]["ant"], prior=prior)
            if f is None or not np.all(np.isfinite(f)):
                continue
            dlt = f - params[k]
            dlt[1] = wrap(dlt[1])
            params[k] = params[k] + dlt
            g[k] = fitter.gains(params[k])[0]
            hflat[nodes[k]["members"]] = g[k]
        change = np.mean(np.abs(wrap(np.angle(g) - np.angle(g_old))))
        g_old = g.copy()
        if change < tol:
            break
    params[:, 1] = wrap(params[:, 1])
    return params, it + 1


def node_snr(prob, params, fitter, ref_nodes=()):
    """
    Detection significance of each node's fitted model: |sum z e^{-i model}| over its noise-only
    rms per real/imaginary component, sqrt(sum |Y|^2 / 2) over the terms used. That rms is exact
    when noise dominates (an overestimate otherwise) and needs no noise model: for unit-modulus
    phase differences the visibility weights do not describe the noise.
    Only the baselines to the reference nodes (fixed unit gains) are used when a node has them:
    with all partners at their fitted gains, nodes fitted jointly to noise reinforce each other
    and the sum looks significant.
    """
    nodes = prob["nodes"]
    g = fitter.gains(params)
    hflat = np.zeros((prob["nflat"], len(fitter.x)), complex)
    for k, nd in enumerate(nodes):
        hflat[nd["members"]] = g[k]
    refflat = np.concatenate([nodes[k]["members"] for k in ref_nodes]) if len(ref_nodes) else \
        np.zeros(0, int)
    snr = np.zeros(len(nodes))
    for k, nd in enumerate(nodes):
        if nd["sens"] <= 0:
            continue
        sel = np.isin(nd["pidx"], refflat)
        if not np.any(sel):
            sel = np.ones(len(nd["pidx"]), bool)
        Y = nd["Y"][sel]
        noise = np.sqrt(np.sum(np.abs(Y) ** 2) / 2)
        if noise > 0:
            z = np.sum(Y * hflat[nd["pidx"][sel]], axis=0)
            snr[k] = np.abs(np.sum(z * np.conj(g[k]))) / noise
    return snr
