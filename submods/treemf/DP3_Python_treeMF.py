#!/usr/bin/env python3
"""
DP3_Python_treeMF: a Python stand-in for DP3 DDECal in mode tec+phase+delay, solved with treeMF
(a mean-field start-up plus coherent per-station fits, with per-station priors on TEC and delay).
It takes DP3's command line and writes an h5parm in DP3's tec+phase+delay layout, so that
facetselfcal can run it in place of DP3 for one solve:

  DP3_Python_treeMF.py [parset] msin=x.ms msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM \\
      steps=[ddecal] ddecal.type=ddecal ddecal.mode=tec+phase+delay ddecal.modeldatacolumns=[MODEL_DATA] \\
      ddecal.solint=4 ddecal.solutions_per_direction=[4] ddecal.nchan=1 ddecal.uvlambdamin=40000 \\
      ddecal.antenna_averaging_factors=[[ST001]:4,[RS*]:2] ddecal.h5parm=out.h5 \\
      ddecal.treemf.delaymax=20 ddecal.treemf.tecprior=geometric numthreads=32

Supported DP3 keys: msin (one MS), msin.datacolumn, msin.weightcolumn, msin.startchan, msin.nchan,
msin.starttimeslot, msin.ntimes, numthreads, and for the ddecal step: mode (tec+phase+delay),
modeldatacolumns (one column), h5parm, solint, solutions_per_direction, nchan,
antenna_averaging_factors, antennaconstraint, uvlambdamin, uvlambdamax, uvmmin, uvmmax.
Other DP3 solver keys (maxiter, tolerance, stepsize, solveralgorithm, propagatesolutions, ...) are
accepted and ignored.

treeMF keys (<step>.treemf.*):
  model                 the phase model: tec+phase+delay (default), tec+phase, tec+delay, tec,
                        delay+phase or delay; the parameters not in the model are fixed at 0 (and
                        written as 0, so the h5parm always has tec000, phase000 and delay000)
  delaymax              hard bound on |delay| relative to the reference (ns, default 20)
  delaysigma            Gaussian delay prior width (ns, default none)
  tecmax                hard bound on |TEC| relative to the reference: the TEC search range
                        (TECU, default 4)
  tecprior              'geometric' (per-station width from the pierce-point distance) or 'none'
  tecsig100             geometric prior width at 100 km (TECU, default 0.25)
  tecsigmin             minimum prior width (TECU, default 0.02)
  priorcal              calibrator-transferred phases: the width saturates beyond this pierce-point
                        separation (km, default none)
  height                ionospheric layer height (km, default 300)
  propagate             use each interval's solution as the prior mean of the next (default False)
  propagate_tecsigma    TEC prior width around the propagated solution (TECU, default 0.1)
  propagate_delaysigma  delay prior width around the propagated solution (ns, default 5)
  nchunk                with propagate: contiguous chunks solved in parallel (default numthreads);
                        the first interval of each chunk is solved cold
  tiecore               solve the core stations (CS*) as one station (default True): they share the
                        clock and see the same ionosphere, and at long-baseline uv cuts each of
                        them alone sees only a few partners; core stations already in an
                        antennaconstraint group are left there
  ref                   reference antenna: 'core' (ST001 or the most sensitive core station, which
                        is the tied core with tiecore; without core stations the most sensitive
                        within 100 km of the array centre) or 'sens' (the most sensitive)
  nsweep, T0, tecstep   mean-field sweeps (30), start temperature (3), TEC grid step (0.005 TECU)
  delaycentre           centre of each station's delay window (the bound delaymax applies around it):
                        'zero' (default), 'track' (from the data: a delay track per station over
                        the whole MS, see below) or an h5parm with delay000 (e.g. a previous solve,
                        or a simulation truth; nearest solution time, relative to the reference)
  delaytrack_search     track: half-width of the delay search (ns, default 200)
  delaytrack_tecstep    track: TEC grid step of the delay-TEC maps (TECU, default 0.02)
  delaytrack_tecwander  track: how far an interval's TEC may wander around its block's (TECU, the
                        width of the smoothing of the maps along TEC; default 0.1)
  delaytrack_block      track and delayadapt: block length (minutes, default 30)
  delaytrack_maxstep    track: largest change of the track between blocks (ns, default 15)
  delaytrack_nperm      track: shuffles of the permutation test (default 20; a station's track is
                        used only if it beats them all, else its centre stays 0)
  delayadapt            rounds of the bound check (default 0): where a station's solutions pile up at
                        one edge of its delay window within a block (at least delayadapt_frac of
                        them, default 0.1), the window there is moved by delaymax / 2 towards the
                        pile-up (never back to a centre it had) and the intervals of that block are
                        solved again
                        The delay centres and the track are written to <h5parm>.delaycentre.npz.

Differential Faraday rotation (model rm or rm+phase; facetselfcal soltype treemfphasediff, in
place of scalarphasediff): the data are the phase difference RR conj(LL) of the data column
(circular, or linear converted), in which the source structure, TEC and delay cancel, and the
model per station is 2 RM lambda^2 (rm) plus a constant (rm+phase). ddecal.mode may then be
phaseonly (the scalarphasediff command); the model column is not read. The h5parm has the
scalarphasediff layout (phase000, time,freq,ant,dir,pol; XX = 2 RM lambda^2 + offset, YY = 0);
RM and offset per solution are also written to <h5parm>.rm.npz. Keys:
  rmmax                 hard bound on |RM| relative to the reference: the RM search range
                        (rad/m^2, default 5)
  rmstep                RM grid step (rad/m^2, default: from the band, 0.25 rad across it)
  rmprior               'none' (default), 'geometric' (width rmsig100 (d / 100 km)^(5/6), at least
                        rmsigmin) or 'rmextract' (mean from RMextract, width rmsigma; only for data
                        without the RMextract correction applied)
  rmsig100, rmsigmin    geometric RM prior width at 100 km and its minimum (rad/m^2, 0.1, 0.005)
  rmsigma               width around the RMextract mean (rad/m^2, default 0.1)
  rmextract_sign        sign applied to the RMextract differential RM (default 1: DP3's rotation
                        convention, in which LINC applies RMextract as rotationmeasure000; not
                        verified on data)
  rmextract_ionexpath, rmextract_server, rmextract_prefix   passed to RMextract getRM (default
                        ./IONEXdata/, http://chapman.upc.es and uqrg: UPC's rapid maps; the CODE
                        ftp server, RMextract's default, was not reachable from the cluster)
  offsetsigma           rm+phase: zero-mean Gaussian prior width on the offset (rad, default none)
  pdiffamp              'unit' (default: unit-modulus phasors, as the DATA_CIRCULAR_PHASEDIFF
                        column) or 'product' (RR conj(LL) with its amplitude as weight)
  pdiffsource           'rrll' (default: RR conj(LL) from the 4 correlations of the data column)
                        or 'column' (the data column is already facetselfcal's
                        DATA_CIRCULAR_PHASEDIFF; unit amplitude only)
  propagate_rmsigma     RM prior width around the propagated solution (rad/m^2, default 0.05)
  minsnr                detection threshold: a station solution whose fitted model is not
                        significant (coherent signal-to-noise below minsnr) is set to RM 0, offset 0
                        (no correction) and the others are re-solved with it fixed. 'auto' (default):
                        a 1% false-alarm probability over the independent RM cells of the search
                        range; 0 disables it
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from submods.treemf import parset as ps  # noqa: E402
from submods.treemf.delaytrack import BlockStack, check_bound, interval_maps, to_intervals  # noqa: E402
from submods.treemf.fitter import C_LIGHT, FftFitter  # noqa: E402
from submods.treemf.h5out import solution_times, write_h5parm, write_h5parm_phasediff  # noqa: E402
from submods.treemf.msio import MsInfo, read_slots  # noqa: E402
from submods.treemf.prior import Prior, geometric_tec_sigma, geometry  # noqa: E402
from submods.treemf.solver import (build_nodes, converge, node_snr, noise_var,  # noqa: E402
                                   phasediff_amplitude, treemf)

IGNORED = {"type", "solveralgorithm", "maxiter", "tolerance", "stepsize", "propagatesolutions",
           "propagateconvergedonly", "max_tec_delay_wraps", "datause", "detectstalling",
           "approximatetec", "maxapproxiter", "approxtolerance", "flagunconverged",
           "flagdivergedonly", "phasereference", "smoothnessconstraint",
           "smoothnessreffrequency", "smoothnessrefdistance", "smoothnessspectralexponent",
           "antenna_smoothness_factors", "minvisratio", "usebeammodel", "keepmodel"}
MODELS = {"tec+phase+delay": (True, True, True), "tec+phase": (True, True, False),
          "tec+delay": (True, False, True), "tec": (True, False, False),
          "delay+phase": (False, True, True), "delay": (False, False, True),
          "rm": (True, False, False), "rm+phase": (True, True, False)}
TREEMF_DEFAULTS = dict(model="tec+phase+delay", delaymax=20.0, delaysigma=None, tecmax=4.0, tecprior="geometric",
                       tecsig100=0.25, tecsigmin=0.02, priorcal=None, height=300.0,
                       propagate=False, propagate_tecsigma=0.1, propagate_delaysigma=5.0,
                       nchunk=None, tiecore=True, ref="core", nsweep=30, T0=3.0, tecstep=0.005,
                       geometry_time="interval", rmmax=5.0, rmstep=None, rmprior="none",
                       rmsig100=0.1, rmsigmin=0.005, rmsigma=0.1, rmextract_sign=1.0,
                       rmextract_ionexpath="./IONEXdata/", rmextract_server="http://chapman.upc.es",
                       rmextract_prefix="uqrg", offsetsigma=None, pdiffamp="unit", pdiffsource="rrll",
                       propagate_rmsigma=0.05, minsnr="auto", delaycentre="zero",
                       delaytrack_search=200.0, delaytrack_tecstep=0.02,
                       delaytrack_tecwander=0.1, delaytrack_block=30.0,
                       delaytrack_maxstep=15.0,
                       delaytrack_nperm=20, delayadapt=0, delayadapt_frac=0.1)

# shared with the worker processes (set before forking)
G = {}


def as_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "t")


def configure(kv):
    """Check the arguments and collect the configuration."""
    pre = ps.solve_step_prefix(kv)
    msin = ps.parse_list(kv.get("msin", kv.get("msin.name", "")))
    if len(msin) != 1:
        raise ValueError(f"exactly one input MS is supported, got msin={msin}")
    mode = kv.get(pre + "mode", "").lower()
    rm_model = kv.get(pre + "treemf.model", "").startswith("rm")
    if mode not in ("tec+phase+delay", "treemf") and not (
            rm_model and mode in ("phaseonly", "scalarphase", "treemfphasediff")):
        raise ValueError(f"{pre}mode={mode}: only tec+phase+delay is supported (or phaseonly with "
                         f"treemf.model rm or rm+phase)")
    cols = ps.parse_list(kv.get(pre + "modeldatacolumns", "[]"))
    if len(cols) != 1:
        raise ValueError(f"exactly one model data column is supported, got {cols}")
    for k in (pre + "sourcedb", pre + "directions", pre + "idg.regions", pre + "reusemodel"):
        if kv.get(k, "").strip("[] "):
            raise ValueError(f"{k} is not supported (use modeldatacolumns)")
    if pre + "h5parm" not in kv:
        raise ValueError(f"{pre}h5parm is required")
    spd = [int(v) for v in ps.parse_list(kv.get(pre + "solutions_per_direction", "[1]"))] or [1]
    if len(spd) != 1:
        raise ValueError("solutions_per_direction must have one entry (one direction)")
    c = dict(ms=msin[0], pre=pre, h5parm=kv[pre + "h5parm"], modelcolumn=cols[0],
             datacolumn=kv.get("msin.datacolumn", "DATA"),
             weightcolumn=kv.get("msin.weightcolumn", "WEIGHT_SPECTRUM"),
             solint=int(kv.get(pre + "solint", 1)), nsub=spd[0],
             nchan=int(kv.get(pre + "nchan", 1)),
             uvlmin=float(kv.get(pre + "uvlambdamin", 0.0)),
             uvlmax=float(kv.get(pre + "uvlambdamax", np.inf)),
             uvmmin=float(kv.get(pre + "uvmmin", 0.0)), uvmmax=float(kv.get(pre + "uvmmax", np.inf)),
             aaf=kv.get(pre + "antenna_averaging_factors", ""),
             antcon=kv.get(pre + "antennaconstraint", ""),
             nproc=int(kv.get("numthreads", os.cpu_count() or 1)),
             startchan=int(kv.get("msin.startchan", 0)), msnchan=int(kv.get("msin.nchan", 0)),
             starttimeslot=int(kv.get("msin.starttimeslot", 0)), ntimes=int(kv.get("msin.ntimes", 0)))
    tm = dict(TREEMF_DEFAULTS)
    for k, v in kv.items():
        if k.startswith(pre + "treemf."):
            name = k[len(pre) + 7:]
            if name not in tm:
                raise ValueError(f"unknown treeMF key {k}")
            tm[name] = v
    for k in ("delaymax", "tecmax", "tecsig100", "tecsigmin", "height", "propagate_tecsigma",
              "propagate_delaysigma", "T0", "tecstep", "rmmax", "rmsig100", "rmsigmin", "rmsigma",
              "rmextract_sign", "propagate_rmsigma", "delaytrack_search", "delaytrack_tecstep",
              "delaytrack_tecwander",
              "delaytrack_block",
              "delaytrack_maxstep", "delayadapt_frac"):
        tm[k] = float(tm[k])
    tm["delaytrack_nperm"] = int(tm["delaytrack_nperm"])
    tm["delayadapt"] = int(tm["delayadapt"])
    for k in ("delaysigma", "priorcal", "rmstep", "offsetsigma"):
        tm[k] = None if tm[k] in (None, "", "None", "none") else float(tm[k])
    tm["nsweep"] = int(tm["nsweep"])
    tm["propagate"] = as_bool(tm["propagate"])
    tm["tiecore"] = as_bool(tm["tiecore"])
    tm["nchunk"] = c["nproc"] if tm["nchunk"] in (None, "", "None") else int(tm["nchunk"])
    if tm["tecprior"] not in ("geometric", "none"):
        raise ValueError(f"treemf.tecprior must be 'geometric' or 'none', got {tm['tecprior']}")
    if tm["model"] not in MODELS:
        raise ValueError(f"treemf.model must be one of {', '.join(MODELS)}, got {tm['model']}")
    if tm["ref"] not in ("core", "sens"):
        raise ValueError(f"treemf.ref must be 'core' or 'sens', got {tm['ref']}")
    if tm["rmprior"] not in ("none", "geometric", "rmextract"):
        raise ValueError(f"treemf.rmprior must be 'none', 'geometric' or 'rmextract', got {tm['rmprior']}")
    if tm["pdiffamp"] not in ("unit", "product"):
        raise ValueError(f"treemf.pdiffamp must be 'unit' or 'product', got {tm['pdiffamp']}")
    if tm["pdiffsource"] not in ("rrll", "column"):
        raise ValueError(f"treemf.pdiffsource must be 'rrll' or 'column', got {tm['pdiffsource']}")
    for k in ("rmextract_server", "rmextract_prefix"):
        tm[k] = None if tm[k] in (None, "", "None", "none") else tm[k]
    tm["kind"] = "rm" if tm["model"].startswith("rm") else "tec"
    if tm["delaycentre"] != "zero" or tm["delayadapt"] > 0:
        if tm["kind"] == "rm" or not MODELS[tm["model"]][2]:
            raise ValueError("treemf.delaycentre and treemf.delayadapt need a model with a delay")
        if tm["delaycentre"] not in ("zero", "track") and not os.path.exists(tm["delaycentre"]):
            raise ValueError(f"treemf.delaycentre must be 'zero', 'track' or an existing h5parm, "
                             f"got {tm['delaycentre']}")
    tm["minsnr"] = "auto" if str(tm["minsnr"]).lower() == "auto" else float(tm["minsnr"])
    c["treemf"] = tm
    other = []
    for k in kv:
        if k.startswith(pre):
            name = k[len(pre):]
            if name.split(".")[0] in IGNORED or name.startswith("initialsolutions"):
                other.append(k)
            elif not (name.startswith("treemf.") or name in (
                    "mode", "modeldatacolumns", "h5parm", "solint", "solutions_per_direction",
                    "nchan", "antenna_averaging_factors", "antennaconstraint", "uvlambdamin",
                    "uvlambdamax", "uvmmin", "uvmmax")):
                print(f"WARNING: unsupported key {k}={kv[k]} is ignored")
        elif not (k in ("msin", "msin.name", "msin.datacolumn", "msin.weightcolumn", "msin.startchan",
                        "msin.nchan", "msin.starttimeslot", "msin.ntimes", "numthreads", "steps")
                  or k.startswith("msout")):
            print(f"WARNING: unsupported key {k}={kv[k]} is ignored")
    if other:
        print("Ignored DP3 solver keys: " + " ".join(f"{k}={kv[k]}" for k in other))
    if kv.get(pre + "smoothnessconstraint", "0").strip() not in ("0", "0.0", ""):
        print("WARNING: smoothnessconstraint is ignored (the phase model is smooth by construction)")
    return c


def setup(c):
    """Metadata, channel blocks, node layout and the fitter."""
    info = MsInfo(c["ms"], c["startchan"], c["msnchan"], c["starttimeslot"], c["ntimes"])
    nant = len(info.names)
    solint = info.nslot if c["solint"] <= 0 else c["solint"]
    nsub = c["nsub"]
    if solint % nsub:
        raise ValueError(f"solutions_per_direction {nsub} must divide solint {solint}")
    nch = len(info.nu)
    # as DP3: nchan 0 (or more than the channels) is one block, else floor(nch / nchan) blocks
    # with the leftover channels spread over them
    nblk = 1 if c["nchan"] <= 0 or c["nchan"] > nch else nch // c["nchan"]
    blocks = np.array([b * nch // nblk for b in range(nblk)])
    nu_blk = np.array([info.nu[b * nch // nblk:(b + 1) * nch // nblk].mean() for b in range(nblk)])
    if nblk < 3:
        raise ValueError(f"at least 3 channel blocks are needed for the fit, got {nblk}")
    factors = np.array(ps.value_per_station(c["aaf"], info.names, 1, int)) if c["aaf"] else \
        np.ones(nant, int)
    if np.any(factors > nsub) or np.any(factors < 1):
        raise ValueError(f"antenna averaging factors must be in 1..{nsub} (solutions_per_direction)")
    group_of = np.arange(nant)
    if c["antcon"].strip("[] "):
        for gi, grp in enumerate(ps.antenna_groups(c["antcon"])):
            members = set()
            for p in grp:
                members |= ps._match(p, info.names)
            idx = [i for i, n in enumerate(info.names) if n in members]
            if idx:
                if len(set(factors[idx])) > 1:
                    raise ValueError(f"antennaconstraint group {grp} has mixed averaging factors")
                group_of[idx] = nant + gi
    tm = c["treemf"]
    core = [i for i in info.used if info.names[i].startswith("CS") and group_of[i] == i]
    if tm["tiecore"] and len(core) > 1:
        if len(set(factors[core])) > 1:
            print(f"tiecore: the core stations have mixed averaging factors, using {factors[core].min()}")
            factors[core] = factors[core].min()
        group_of[core] = 2 * nant  # beyond the antennaconstraint group ids
        print(f"tiecore: {len(core)} core stations are solved as one station")
    fit_tec, fit_offset, fit_delay = MODELS[tm["model"]]
    if tm["kind"] == "rm":
        fitter = FftFitter(nu_blk, tec_range=tm["rmmax"], tec_step=tm["rmstep"], delay_max=0.0,
                           fit_tec=fit_tec, fit_offset=fit_offset, fit_delay=False, kind="rm")
    else:
        fitter = FftFitter(nu_blk, tec_range=tm["tecmax"], tec_step=tm["tecstep"],
                           delay_max=tm["delaymax"] * 1e-9, fit_tec=fit_tec, fit_offset=fit_offset,
                           fit_delay=fit_delay)
    minsnr = None
    if tm["kind"] == "rm":
        # independent RM cells in the search range: its width over the RM resolution
        # 2 sqrt(3) / (lambda^2 span); the threshold for a 1% false alarm over them (the modulus
        # |F| / sigma of a noise-only cell is Rayleigh: P(> t) = exp(-t^2 / 2))
        lam2 = (C_LIGHT / nu_blk) ** 2
        ncell = max(1.0, 2 * tm["rmmax"] * (lam2.max() - lam2.min()) / (2 * np.sqrt(3)))
        minsnr = float(np.sqrt(2 * np.log(ncell / 0.01)))
    nint = int(np.ceil(info.nslot / solint))
    wide = None
    if tm["delaycentre"] == "track":
        wide = FftFitter(nu_blk, tec_range=tm["tecmax"], tec_step=tm["delaytrack_tecstep"],
                         delay_max=tm["delaytrack_search"] * 1e-9, fit_tec=fit_tec,
                         fit_offset=fit_offset, fit_delay=fit_delay)
    # time blocks of the delay track and the bound check
    block_of = np.floor(np.arange(nint) * solint * info.dt / (60 * tm["delaytrack_block"])).astype(int)
    return dict(info=info, nant=nant, solint=solint, nsub=nsub, blocks=blocks, nu_blk=nu_blk,
                factors=factors, group_of=group_of, fitter=fitter, wide=wide, block_of=block_of,
                nint=nint, minsnr=minsnr)


def prepare(i):
    """
    Data, nodes, reference and TEC (or RM) prior of interval i, shared by the solve and the delay
    track. ref is -1 when no station has data.
    """
    c, S = G["c"], G["S"]
    info, nant, solint, nsub, tm = S["info"], S["nant"], S["solint"], S["nsub"], c["treemf"]
    s0 = i * solint
    rm = tm["kind"] == "rm"
    slots = read_slots(info, s0, min(s0 + solint, info.nslot), c["datacolumn"], c["modelcolumn"],
                       c["weightcolumn"], c["uvlmin"], c["uvlmax"], c["uvmmin"], c["uvmmax"],
                       pdiff=tm["pdiffamp"] if rm else None, pdiff_source=tm["pdiffsource"])
    slots += [None] * (solint - len(slots))
    alpha = None
    if rm:
        # model-free: the signal amplitude of RR conj(LL) from the data, as the model amplitude
        alpha = phasediff_amplitude(slots)
        for sl in slots:
            if sl is not None:
                sl["m"] = sl["m"] * alpha
    prob = build_nodes(slots, nant, solint, nsub, S["factors"], S["group_of"], S["blocks"])
    nodes = prob["nodes"]
    # a node with data in fewer channel blocks than free parameters is not solved (weight 0)
    nfree = sum(S["fitter"].free)
    sparse = [k for k, nd in enumerate(nodes) if 0 < nd["nblk"] < nfree]
    for k in sparse:
        nodes[k]["sens"] = 0.0
    sens_ant = np.zeros(nant)
    for nd in nodes:
        sens_ant[nd["ant"]] += nd["sens"]
    P = dict(slots=slots, prob=prob, sparse=sparse, alpha=alpha, ref=-1)
    if not np.any(sens_ant > 0):
        return P
    tmid = info.t0 + (s0 + (solint - 1) / 2) * info.dt if tm["geometry_time"] == "interval" else info.t0
    geo = geometry(info.names, info.pos, info.pdir, tmid, tm["height"] * 1e3)
    # a node's sensitivity is counted on its first antenna, so the tied core counts as one station
    core = np.array([n == "ST001" or n.startswith("CS") for n in info.names]) & (sens_ant > 0)
    near = (geo["dground"] < 100.0) & (sens_ant > 0)
    if tm["ref"] == "core" and np.any(core | near):
        cand = np.flatnonzero(core if np.any(core) else near)
        ref = int(cand[np.argmax(sens_ant[cand])])
    else:
        ref = int(np.argmax(sens_ant))
    tec_mu, tec_sig = np.zeros(nant), np.full(nant, np.inf)
    if rm:  # the first parameter is the RM (rad/m^2)
        if tm["rmprior"] == "geometric":
            tec_sig = geometric_tec_sigma(geo, ref, tm["rmsig100"], tm["rmsigmin"])
        elif tm["rmprior"] == "rmextract" and G.get("rmx") is not None:
            rmx = rmextract_at(G["rmx"], tmid)
            ok = np.isfinite(rmx) & np.isfinite(rmx[ref])
            tec_mu[ok] = tm["rmextract_sign"] * (rmx[ok] - rmx[ref])
            tec_sig[ok] = tm["rmsigma"]
    elif tm["tecprior"] == "geometric":
        tec_sig = geometric_tec_sigma(geo, ref, tm["tecsig100"], tm["tecsigmin"], None, tm["priorcal"])
    P.update(ref=ref, sig2=noise_var(slots), tec_mu=tec_mu, tec_sig=tec_sig)
    return P


def solve_interval(i, prev):
    """
    Solve interval i. prev: the previous interval's (tec, delay, valid) per antenna, relative to
    its reference, for propagation (or None).

    Returns
    -------
    dict
        tec, phase, delay, weight: (nsub, nant); ref; the delay-window centre used (nant, relative
        to ref); the last sub-solution as `last` for the next interval; diagnostics.
    """
    c, S = G["c"], G["S"]
    nant, solint, nsub, tm = S["nant"], S["solint"], S["nsub"], c["treemf"]
    t0 = time.time()
    rm = tm["kind"] == "rm"
    P = prepare(i)
    prob, alpha = P["prob"], P["alpha"]
    nodes = prob["nodes"]
    out = dict(tec=np.zeros((nsub, nant)), phase=np.zeros((nsub, nant)),
               delay=np.zeros((nsub, nant)), weight=np.zeros((nsub, nant)), ref=-1, last=None,
               nodes=len(nodes), sweeps=0, sparse=len(P["sparse"]), centre=np.zeros(nant))
    if P["ref"] < 0:
        return out
    ref, sig2, tec_mu, tec_sig = P["ref"], P["sig2"], P["tec_mu"], P["tec_sig"]
    nu0 = S["nu_blk"][0]
    fitter = S["fitter"]
    # the delay windows (and the mean of a Gaussian delay prior) around the centres
    cen = None if G.get("centre") is None else G["centre"][i] - G["centre"][i][ref]
    fitter.set_centre(cen)
    dl_mu = np.zeros(nant) if cen is None else cen.copy()
    dl_sig = np.full(nant, np.inf if tm["delaysigma"] is None or rm else tm["delaysigma"] * 1e-9)
    if prev is not None and prev[2][ref]:
        ptec, pdl, pok = prev
        ok = pok.copy()
        tec_mu[ok] = ptec[ok] - ptec[ref]
        tec_sig[ok] = tm["propagate_rmsigma"] if rm else tm["propagate_tecsigma"]
        if not rm:
            dl_mu[ok] = pdl[ok] - pdl[ref]
            dl_sig[ok] = tm["propagate_delaysigma"] * 1e-9
        tec_mu[ref], dl_mu[ref] = 0.0, 0.0
    prior = Prior(nant, ref, sig2, nu0, tec_mu, tec_sig, dl_mu, dl_sig, scale=fitter.scale,
                  offset_sigma=tm["offsetsigma"] if rm else None)
    ref_nodes = sorted(set(prob["node_of"][:, ref]))
    params, nsw = treemf(prob, ref_nodes, fitter, sig2, prior, tm["nsweep"], tm["T0"])
    ndet = 0
    if rm and tm["minsnr"] != 0:
        thr = S["minsnr"] if tm["minsnr"] == "auto" else tm["minsnr"]
        snr = node_snr(prob, params, fitter, ref_nodes)
        weak = {k for k, nd in enumerate(nodes) if nd["sens"] > 0 and k not in ref_nodes and snr[k] < thr}
        if weak:
            # not detected: no correction, and the others re-solved with these fixed at zero
            params, _ = converge(prob, params, set(ref_nodes) | weak, fitter, prior)
        ndet = len(weak)
    tec, off, dl = fitter.physical(params)
    for s in range(nsub):
        k = prob["node_of"][s * solint // nsub]
        out["tec"][s], out["phase"][s], out["delay"][s] = tec[k], off[k], dl[k]
        out["weight"][s] = [1.0 if nodes[kk]["sens"] > 0 or kk in ref_nodes else 0.0 for kk in k]
    out.update(ref=ref, sweeps=nsw, last=(out["tec"][-1].copy(), out["delay"][-1].copy(),
                                          out["weight"][-1] > 0))
    if cen is not None:
        out["centre"] = cen
    out["seconds"] = time.time() - t0
    out["alpha"] = alpha
    out["undetected"] = ndet
    return out


def profile_interval(i):
    """Delay-TEC maps and TEC log priors of interval i for the delay track, and its reference."""
    S = G["S"]
    P = prepare(i)
    if P["ref"] < 0:
        return i, -1, None, None
    ref, wide = P["ref"], S["wide"]
    prior = Prior(S["nant"], ref, P["sig2"], S["nu_blk"][0], P["tec_mu"], P["tec_sig"],
                  scale=wide.scale)
    ref_nodes = sorted(set(P["prob"]["node_of"][:, ref]))
    return (i, ref) + interval_maps(P["prob"], ref_nodes, wide, P["sig2"], prior, S["nant"],
                                    G["c"]["treemf"]["delaytrack_tecwander"])


def load_rmextract(c, info):
    """
    RM per antenna from RMextract (getRM) over the MS time range, for the 'rmextract' RM prior:
    (times, rm (ntime, nant)), or None when RMextract is not available or fails.
    """
    tm = c["treemf"]
    try:
        from RMextract.getRM import getRM
    except ImportError as e:
        print(f"WARNING: rmprior=rmextract but RMextract cannot be imported ({e}); no RM prior mean")
        return None
    kw = dict(ionexPath=tm["rmextract_ionexpath"], timestep=300.0)
    if tm["rmextract_server"]:
        kw["server"] = tm["rmextract_server"]
    if tm["rmextract_prefix"]:
        kw["prefix"] = tm["rmextract_prefix"]
    try:
        res = getRM(MS=c["ms"], **kw)
    except Exception as e:  # network or IONEX problems: run without the prior mean
        print(f"WARNING: RMextract getRM failed ({e}); no RM prior mean")
        return None
    times = np.asarray(res["times"], float)
    rm = np.full((len(times), len(info.names)), np.nan)
    for a, n in enumerate(info.names):
        if n in res["RM"]:
            rm[:, a] = np.asarray(res["RM"][n]).ravel()[:len(times)]
    print(f"RMextract: {len(times)} times, RM for {int(np.sum(np.isfinite(rm[0])))} antennas")
    return times, rm


def rmextract_at(rmx, t):
    """RMextract RM per antenna interpolated to time t."""
    times, rm = rmx
    return np.array([np.interp(t, times, rm[:, a]) if np.all(np.isfinite(rm[:, a])) else np.nan
                     for a in range(rm.shape[1])])


def solve_chunk(intervals):
    """Solve a list of consecutive intervals; with propagation each starts from the previous."""
    res, prev = [], None
    for i in intervals:
        r = solve_interval(i, prev if G["c"]["treemf"]["propagate"] else None)
        res.append((i, r))
        prev = r["last"] if r["ref"] >= 0 else None
    return res


def run_intervals(intervals, t_start, label="intervals"):
    """Solve the given intervals in parallel (in chunks with propagation): {interval: result}."""
    c, tm = G["c"], G["c"]["treemf"]
    n = len(intervals)
    if tm["propagate"]:
        nchunk = max(1, min(tm["nchunk"], n))
        edges = np.linspace(0, n, nchunk + 1).astype(int)
        jobs = [list(intervals[edges[j]:edges[j + 1]]) for j in range(nchunk) if edges[j + 1] > edges[j]]
    else:
        jobs = [[i] for i in intervals]
    results = {}
    nproc = max(1, min(c["nproc"], len(jobs)))
    if nproc == 1:
        for j in jobs:
            results.update(dict(solve_chunk(j)))
    else:
        with mp.get_context("fork").Pool(nproc) as pool:
            for res in pool.imap_unordered(solve_chunk, jobs):
                results.update(dict(res))
                done = len(results)
                if done % max(1, n // 10) == 0 or done == n:
                    print(f"  {done}/{n} {label} done ({time.time() - t_start:.0f} s)", flush=True)
    return results


def delay_track(t_start):
    """
    Delay-window centres from the data (delaycentre=track): per interval and antenna (s), the
    block track and the per-antenna statistics.
    """
    c, S = G["c"], G["S"]
    tm, nint, nant, names = c["treemf"], S["nint"], S["nant"], S["info"].names
    wide, block_of = S["wide"], S["block_of"]
    stack = BlockStack(int(block_of.max()) + 1, nant, len(wide.tecs), len(wide.taus),
                       tm["delaytrack_nperm"])
    nproc = max(1, min(c["nproc"], nint))
    pool = mp.get_context("fork").Pool(nproc) if nproc > 1 else None
    res = pool.imap_unordered(profile_interval, range(nint), chunksize=2) if pool else \
        map(profile_interval, range(nint))
    # maps relative to different references cannot be stacked: keep the first reference's
    ref0, nskip = None, 0
    for i, ref, maps, lp in res:
        if ref < 0:
            continue
        ref0 = ref if ref0 is None else ref0
        if ref != ref0:
            nskip += 1
            continue
        stack.add(block_of[i], maps, lp)
    if pool:
        pool.close()
        pool.join()
    if ref0 is None:
        return np.zeros((nint, nant)), None, None
    blk, inf = stack.track(wide.taus, tm["delaytrack_maxstep"] * 1e-9)
    centre = to_intervals(blk, block_of, nint)
    print(f"delay track ({time.time() - t_start:.0f} s): reference {names[ref0]}"
          f"{f' ({nskip} intervals with another reference left out)' if nskip else ''}, search "
          f"+-{tm['delaytrack_search']:g} ns, TEC wander {tm['delaytrack_tecwander']:g} TECU, "
          f"{blk.shape[0]} blocks of {tm['delaytrack_block']:g} min, "
          f"max step {tm['delaytrack_maxstep']:g} ns; score vs largest of {tm['delaytrack_nperm']} "
          f"shuffled; track per block (ns)")
    for a in range(nant):
        if inf["nint"][a] == 0:
            continue
        trk = " ".join(f"{1e9 * v:6.1f}" for v in blk[:, a]) if inf["used"][a] else "not used (centre 0)"
        print(f"  {names[a]:10} {inf['nint'][a]:5d} int  score {inf['score'][a]:9.1f} / "
              f"{inf['null'][a]:9.1f}  {trk}")
    return centre, blk, inf


def centres_from_h5(fn):
    """Delay-window centres (nint, nant; s) from the delay000 of an h5parm, nearest solution time
    per interval (relative to the reference later, per interval)."""
    import h5py
    S = G["S"]
    info, nint, solint = S["info"], S["nint"], S["solint"]
    with h5py.File(fn, "r") as f:
        ds = f["sol000/delay000/val"]
        axes = ds.attrs["AXES"].decode().split(",")
        v = np.moveaxis(ds[()], [axes.index("time"), axes.index("ant")], [0, 1])
        v = v.reshape(v.shape[0], v.shape[1], -1)[..., 0]
        hn = [a.decode() for a in f["sol000/delay000/ant"][:]]
        ht = f["sol000/delay000/time"][:]
    tmid = info.t0 + (np.arange(nint) * solint + (solint - 1) / 2) * info.dt
    idx = np.abs(tmid[:, None] - ht[None, :]).argmin(1)
    centre = np.zeros((nint, S["nant"]))
    for j, n in enumerate(hn):
        if n in info.names:
            centre[:, info.names.index(n)] = v[idx, j]
    print(f"delay-window centres from {fn} ({len(hn)} antennas, {len(ht)} solution times)")
    return centre


def per_interval(results, key, nint, nant):
    """Per interval and antenna: the last sub-solution's value (key delay/weight) or the centre."""
    out = np.zeros((nint, nant))
    for i, r in results.items():
        out[i] = r[key][-1] if r[key].ndim == 2 else r[key]
    return out


def main(argv=None):
    t_start = time.time()
    kv = ps.read_args(sys.argv[1:] if argv is None else argv)
    print("DP3_Python_treeMF\nParameters:\n" + "\n".join(f"  {k}={v}" for k, v in sorted(kv.items())))
    c = configure(kv)
    S = setup(c)
    info = S["info"]
    G.update(c=c, S=S)
    tm = c["treemf"]
    if tm["kind"] == "rm" and tm["rmprior"] == "rmextract":
        G["rmx"] = load_rmextract(c, info)
    print(f"MS {c['ms']}: {len(info.names)} antennas ({len(info.used)} used), {len(info.nu)} channels "
          f"{info.nu[0] / 1e6:.3f}-{info.nu[-1] / 1e6:.3f} MHz in {len(S['blocks'])} blocks, "
          f"{info.nslot} time slots of {info.dt:.3f} s")
    print(f"solint {S['solint']} slots, {S['nsub']} sub-solutions, {S['nint']} intervals; "
          f"averaging factors (factor: antennas) "
          f"{ {int(k): int(n) for k, n in zip(*np.unique(S['factors'], return_counts=True))} }; "
          + (f"model {tm['model']} on RR conj(LL) ({tm['pdiffamp']} amplitude), RM bound {tm['rmmax']} "
             f"rad/m^2, RM grid step {S['fitter'].tecs[1] - S['fitter'].tecs[0]:.4g} rad/m^2, RM prior "
             f"{tm['rmprior']}, offset prior {tm['offsetsigma']}, "
             if tm["kind"] == "rm" else
             f"model {tm['model']}, delay bound {tm['delaymax']} ns, TEC bound {tm['tecmax']} TECU, TEC prior {tm['tecprior']}"
             f"{' (s100 %.3g TECU)' % tm['tecsig100'] if tm['tecprior'] == 'geometric' else ''}, ")
          + f"propagate {tm['propagate']}")
    nint, nsub, nant = S["nint"], S["nsub"], S["nant"]
    trk_blk, trk_info = None, None
    if tm["kind"] != "rm" and tm["delaycentre"] == "track":
        G["centre"], trk_blk, trk_info = delay_track(t_start)
    elif tm["kind"] != "rm" and tm["delaycentre"] != "zero":
        G["centre"] = centres_from_h5(tm["delaycentre"])
    results = run_intervals(list(range(nint)), t_start)
    piles, visited = [], {}
    if tm["kind"] != "rm":
        bound = tm["delaymax"] * 1e-9
        for rnd in range(tm["delayadapt"] + 1):
            shift, found = check_bound(per_interval(results, "delay", nint, nant),
                                       per_interval(results, "weight", nint, nant),
                                       per_interval(results, "centre", nint, nant), bound,
                                       S["block_of"], tm["delayadapt_frac"])
            if found:
                print(f"delay bound check{'' if rnd == tm['delayadapt'] else f' (round {rnd + 1})'}: "
                      "pile-ups at the window edge (station block side fraction): "
                      + ", ".join(f"{info.names[a]} {b} {'+' if sd > 0 else '-'} {f:.0%}"
                                  for b, a, sd, f in found))
            if not found or rnd == tm["delayadapt"]:
                break
            base = np.zeros((nint, nant)) if G.get("centre") is None else G["centre"]
            # never back to a centre a station had in that block (no oscillation)
            for b, a, sd, f in found:
                rows = S["block_of"] == b
                key = (b, a)
                visited.setdefault(key, {round(1e10 * base[rows, a].mean())})
                new = round(1e10 * (base[rows, a] + shift[rows, a]).mean())
                if new in visited[key]:
                    shift[rows, a] = 0.0
                visited[key].add(new)
            if not np.any(shift):
                break
            piles.append(found)
            G["centre"] = base + shift
            redo = list(range(nint)) if tm["propagate"] else \
                [int(i) for i in np.flatnonzero(np.any(shift != 0, axis=1))]
            print(f"  moving those windows by {tm['delaymax'] / 2:g} ns and solving {len(redo)} intervals again")
            results.update(run_intervals(redo, t_start, f"intervals (round {rnd + 1})"))
    tec = np.zeros((nint * nsub, nant))
    phase, delay, weight = np.zeros_like(tec), np.zeros_like(tec), np.zeros_like(tec)
    for i, r in results.items():
        sl = slice(i * nsub, (i + 1) * nsub)
        tec[sl], phase[sl], delay[sl], weight[sl] = r["tec"], r["phase"], r["delay"], r["weight"]
    times = solution_times(info.t0, info.dt, info.nslot, info.dt * S["solint"] / nsub)
    n = len(times)
    u = info.used
    refs = [info.names[r["ref"]] for r in results.values() if r["ref"] >= 0]
    print(f"reference antenna(s): {sorted(set(refs))}; median sweeps "
          f"{np.median([r['sweeps'] for r in results.values()]):.0f}; median time per interval "
          f"{np.median([r.get('seconds', 0) for r in results.values()]):.1f} s; "
          f"{sum(r['sparse'] for r in results.values())} station solutions not solved (data in "
          f"fewer channel blocks than free parameters)"
          + (f"; {sum(r.get('undetected', 0) for r in results.values())} station solutions not "
             f"detected (coherent S/N below {S['minsnr'] if tm['minsnr'] == 'auto' else tm['minsnr']:.2f}; "
             f"set to no correction)" if tm["kind"] == "rm" else ""))
    parset_text = "".join(f"{k}={v}\n" for k, v in sorted(kv.items()))
    if tm["kind"] == "rm":
        # RR - LL phase per channel: 2 RM lambda^2 + offset
        rmv, off = tec[:n][:, u], phase[:n][:, u]
        ph = 2 * C_LIGHT ** 2 * rmv[:, None, :] / info.nu[None, :, None] ** 2 + off[:, None, :]
        wt = np.repeat(weight[:n][:, None, u], len(info.nu), axis=1)
        write_h5parm_phasediff(c["h5parm"], info.names, info.pos, u, [f"[{c['modelcolumn']}]"],
                               [info.pdir], times, info.nu, ph[..., None], wt[..., None], parset_text)
        np.savez(c["h5parm"] + ".rm.npz", times=times, ant=np.array([info.names[a] for a in u]),
                 rm=rmv, offset=off, weight=weight[:n][:, u],
                 alpha=np.array([r.get("alpha") or np.nan for _, r in sorted(results.items())]))
        dutch = np.array([info.names[a][:2] in ("CS", "RS", "ST") for a in u])
        ok = weight[:n][:, u] > 0
        for lab, sel in (("Dutch", dutch), ("international", ~dutch)):
            v = np.abs(rmv[:, sel][ok[:, sel]])
            if v.size:
                print(f"|RM| {lab}: median {np.median(v):.3g}, 90th percentile {np.percentile(v, 90):.3g} rad/m^2")
    else:
        write_h5parm(c["h5parm"], info.names, info.pos, u, [f"[{c['modelcolumn']}]"], [info.pdir], times,
                     float(info.nu.mean()), tec[:n, u, None], phase[:n, u, None], delay[:n, u, None],
                     weight[:n, u, None], parset_text)
    if tm["kind"] != "rm" and (tm["delaycentre"] != "zero" or tm["delayadapt"] > 0):
        np.savez(c["h5parm"] + ".delaycentre.npz", ant=np.array(info.names),
                 centre=per_interval(results, "centre", nint, nant),
                 refs=np.array([results[i]["ref"] for i in range(nint)]), block_of=S["block_of"],
                 track=np.zeros(0) if trk_blk is None else trk_blk,
                 track_used=np.zeros(0) if trk_info is None else trk_info["used"],
                 track_score=np.zeros(0) if trk_info is None else trk_info["score"],
                 track_null=np.zeros(0) if trk_info is None else trk_info["null"],
                 piles=np.array([(r, b, a, sd, f) for r, fl in enumerate(piles) for b, a, sd, f in fl]))
    print(f"Wrote {c['h5parm']}: {n} solution times x {len(u)} antennas. "
          f"Total time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
