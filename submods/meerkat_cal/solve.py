"""Solves for the MeerKAT calibrator-based calibration: delays with treeMF, DDECal solves without output data,
referencing to the fixed reference antenna, applying h5parms, and combining h5parms."""
import os
import shutil

import numpy as np
import tables
from casacore.tables import makecoldesc, table  # type: ignore
from losoto.h5parm import h5parm

from submods.applycal import applycal
from submods.h5_helpers.h5_info import findrefant_core
from submods.h5_helpers.modify_delay import fix_delayreference
from submods.h5_helpers.modify_phase import fix_phasereference
from submods.h5_helpers.modify_tec import fix_tecreference
from submods.meerkat_cal.prepare import nchan_divisible
from submods.runtime_utils import _track_resource_phase, allocated_cpu_count, run, terminal_print

TREEMF_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'treemf',
                             'DP3_Python_treeMF.py')


def nthreads():
    """Number of threads for DP3 and treeMF."""
    return min(allocated_cpu_count(), 48)


# DDECal solves with many channel blocks scale badly beyond ~16 threads: on a 96-core node a per-channel
# diagonal solve of a 10 min MeerKAT scan spent most of its time in system calls with 48 threads
DDECAL_THREADS = 16


@_track_resource_phase("solve")
def solve_delay(ms, h5out, incol, modelcol='MODEL_DATA', solint=4, model='delay+phase',
                delaymax=20., tecmax=1., nblocks=256):
    """
    Solve for the delay per antenna with treeMF on (XX+YY)/2 (DP3 ddecal.mode=tec+phase+delay with the
    treeMF model 'delay+phase': the TEC is fixed at zero).

    The phase is part of the model because the data have an arbitrary phase per antenna when the
    delay is solved. With the model 'delay' that phase phi has to come from 2 pi nu tau alone, which
    has several near-equivalent solutions about 1/nu (~1.2 ns at UHF) apart for |phi| near 180 deg;
    treeMF then picks different ones in different solution intervals, and the linear interpolation
    in time between them winds the phase across the interval (seen on the phase calibrator).

    Parameters
    ----------
    ms : str
        Measurement Set.
    h5out : str
        Output h5parm (DP3 tec+phase+delay layout: tec000, phase000, delay000).
    incol : str
        Data column.
    modelcol : str, optional
        Model column. Default is MODEL_DATA.
    solint : int, optional
        Solution interval in time slots. Default is 4.
    model : str, optional
        treeMF phase model, e.g. 'delay', 'delay+phase' or 'tec+phase+delay'. Default is 'delay+phase'.
    delaymax : float, optional
        Bound on |delay| (ns) relative to the reference. Default is 20.
    tecmax : float, optional
        Bound on |TEC| (TECU) relative to the reference. Default is 1.
    nblocks : int, optional
        Approximate number of channel blocks treeMF fits over (at least 3). Default is 256.
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = max(1, len(t.getcol('CHAN_FREQ')[0]) // nblocks)
    cmd = ('python ' + TREEMF_SCRIPT + f' msin={ms} msin.datacolumn={incol} msin.weightcolumn=WEIGHT_SPECTRUM '
           f'numthreads={nthreads()} steps=[ddecal] ddecal.type=ddecal ddecal.mode=tec+phase+delay '
           f'ddecal.modeldatacolumns=[{modelcol}] ddecal.solint={solint} ddecal.nchan={nchan} ddecal.h5parm={h5out} '
           f'ddecal.treemf.model={model} ddecal.treemf.delaymax={delaymax} ddecal.treemf.tecmax={tecmax}')
    terminal_print('treeMF delay solve:', cmd)
    run(cmd, log=True)
    check_solutions(h5out)


def average_for_solve(ms, incol, modelcol, out, timestep=0, freqstep=1, nchan=0):
    """
    Average a data and a model column (DP3 averager, same flags and weights for both) into a new MS
    with DATA and MODEL_DATA, for a solve with one solution over the averaged range.

    DDECal goes over every visibility in every iteration, so a per-channel solve over a whole scan
    (2048 channels, ~75 dumps, ~1900 baselines, ~100 iterations) takes hours, and so does a per-dump
    solve over the whole band, which DDECal cannot spread over threads (one channel block); CASA
    pre-averages in the same way. The data must be coherent over the averaged range (delays and
    time-variable phases applied), and averaging data and model the same way keeps the smearing of
    the field sources identical in both.

    Parameters
    ----------
    ms : str
        Measurement Set.
    incol, modelcol : str
        Data and model columns.
    out : str
        Output MS (overwritten).
    timestep : int, optional
        Time slots to average, 0 for the whole MS. Default is 0.
    freqstep : int, optional
        Channels to average. Default is 1.
    nchan : int, optional
        Number of channels to read (DP3 msin.nchan, a multiple of freqstep), 0 for all. Default is 0.
    """
    if timestep == 0:
        with table(ms, ack=False) as t:
            timestep = len(np.unique(t.getcol('TIME')))
    for col, msout in ((incol, out), (modelcol, out + '.model')):
        if os.path.isdir(msout):
            shutil.rmtree(msout)
        cmd = (f'DP3 numthreads={nthreads()} msin={ms} msin.datacolumn={col} msin.weightcolumn=WEIGHT_SPECTRUM '
               f'msout={msout} msout.scalarflags=False steps=[avg] avg.type=averager avg.timestep={timestep} avg.freqstep={freqstep}'
               + (f' msin.nchan={nchan}' if nchan else ''))
        terminal_print('Average for the solve:', cmd)
        run(cmd, log=True)
    with table(out + '.model', ack=False) as tm, table(out, ack=False, readonly=False) as t:
        t.addcols(makecoldesc('MODEL_DATA', t.getcoldesc('DATA')))
        t.putcol('MODEL_DATA', tm.getcol('DATA'))
    shutil.rmtree(out + '.model')


@_track_resource_phase("solve")
def solve_ddecal(ms, h5out, mode, incol, solint=0, nchan=1, modelcol='MODEL_DATA', smoothness=None,
                 uvlambdamin=None, maxiter=150, timeavg=False, maxchan=None):
    """
    DDECal solve against a model column, without writing data.

    Parameters
    ----------
    ms : str
        Measurement Set.
    h5out : str
        Output h5parm.
    mode : str
        DDECal mode (facetselfcal soltype names), e.g. 'complexgain', 'phaseonly', 'amplitudeonly',
        'scalaramplitude', 'leakage'.
    incol : str
        Data column.
    solint : int, optional
        Solution interval in time slots, 0 for one solution. Default is 0.
    nchan : int, optional
        Channels per solution. Default is 1.
    modelcol : str, optional
        Model column. Default is MODEL_DATA.
    smoothness : float or None, optional
        Smoothness constraint kernel (MHz). Default None.
    uvlambdamin : float or None, optional
        Minimum uv distance (lambda). Default None.
    maxiter : int, optional
        Maximum number of iterations. Default is 150.
    timeavg : bool, optional
        For solint=0: solve on data and model averaged over the whole MS in time (average_for_solve).
        Only for data that are coherent in time. Default False.
    maxchan : int or None, optional
        For nchan=0: solve on data and model averaged in frequency by the smallest factor that gives
        at most maxchan channels (average_for_solve). Only for data that are coherent over the
        averaged channels. Default None (no averaging).
    """
    timestep = 0 if (timeavg and solint == 0) else 1
    freqstep, nchan_read = 1, 0
    if maxchan and nchan == 0:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            nchan_ms = len(t.getcol('CHAN_FREQ')[0])
        freqstep = -(-nchan_ms // maxchan)
        nchan_read = nchan_divisible(nchan_ms, freqstep)
        nchan_read = 0 if nchan_read == nchan_ms else nchan_read
    avgms = None
    if timestep != 1 or freqstep > 1:
        avgms = ms.rstrip('/') + '.solveavg'
        average_for_solve(ms, incol, modelcol, avgms, timestep=timestep, freqstep=freqstep, nchan=nchan_read)
        ms, incol, modelcol = avgms, 'DATA', 'MODEL_DATA'
    cmd = (f'DP3 numthreads={min(nthreads(), DDECAL_THREADS)} msin={ms} msin.datacolumn={incol} '
           f'msin.weightcolumn=WEIGHT_SPECTRUM msout= steps=[ddecal] ddecal.type=ddecal ddecal.mode={mode} ddecal.modeldatacolumns=[{modelcol}] '
           f'ddecal.solint={solint} ddecal.nchan={nchan} ddecal.h5parm={h5out} ddecal.maxiter={maxiter} '
           'ddecal.tolerance=1e-5 ddecal.propagatesolutions=True ddecal.flagunconverged=False ')
    if smoothness is not None and smoothness > 0:
        cmd += f'ddecal.smoothnessconstraint={smoothness * 1e6} '
    if uvlambdamin is not None:
        cmd += f'ddecal.uvlambdamin={uvlambdamin} '
    terminal_print('DDECal solve:', cmd)
    run(cmd, log=True)
    if avgms is not None:
        shutil.rmtree(avgms)
    check_solutions(h5out)


def check_solutions(h5parm, maxnan=0.9):
    """
    Stop if almost all solutions of a solve are NaN (a diverged solve that propagated NaN to every later
    solution interval, or data that are all flagged): applying them would flag all the data.

    Parameters
    ----------
    h5parm : str
        h5parm written by the solve.
    maxnan : float, optional
        Largest allowed NaN fraction over all soltabs. Default is 0.9.
    """
    with tables.open_file(h5parm) as H:
        frac = [float(np.mean(~np.isfinite(st.val[:]))) for st in H.root.sol000._v_children.values()
                if hasattr(st, 'val')]
    if frac and min(frac) > maxnan:
        raise RuntimeError(f'{h5parm}: {100 * min(frac):.1f}% of the solutions are NaN, the solve failed or '
                           'the data are (nearly) all flagged; stopping before these solutions flag the data')


def reference(h5in, telescope='MeerKAT'):
    """
    Reference the phases, delays and TEC of an h5parm to the reference antenna of the run
    (findrefant_core returns the antenna fixed by setup_refant / set_refant).

    Parameters
    ----------
    h5in : str
        h5parm.
    telescope : str, optional
        Telescope name for findrefant_core. Default is 'MeerKAT'.

    Returns
    -------
    str
        The reference antenna.
    """
    refant = findrefant_core(h5in, telescope=telescope)
    with tables.open_file(h5in) as H:
        soltabs = list(H.root.sol000._v_children.keys())
    if 'phase000' in soltabs:
        fix_phasereference(h5in, refant)
    if 'delay000' in soltabs:
        fix_delayreference(h5in, refant)
    if 'tec000' in soltabs:
        fix_tecreference(h5in, refant)
    unwrap_phase(h5in)
    return refant


def unwrap_phase(h5in, soltab='phase000'):
    """
    Remove the 2 pi jumps between neighbouring solutions of a phase soltab, for the linear
    interpolation in time and frequency of DP3 applycal. Only whole turns are added, so the applied
    gain at each solution is unchanged; what changes is the interpolation in between: DP3 interpolates
    the stored phase values (not the complex gains) without taking the shorter way round, so a jump
    of e.g. +179 -> -179 deg, or one of the +-360 deg jumps that subtracting the reference antenna's
    phase leaves, winds the applied phase through a full turn between two solutions. CASA applycal
    takes the shorter way between the two solutions it interpolates in time
    (RIorAPArray::trackPhase) and unwraps the phase over the channels before it interpolates in
    frequency (CTPatchedInterp).

    Per antenna, polarisation and direction:
    1. each channel is unwrapped along time, which puts every pair of neighbouring time slots on the
       shorter way, as CASA does; this is local, a noisy or outlying solution only affects the
       interpolation next to it;
    2. each channel is then shifted as a whole (all time slots, so step 1 is kept) by the number of
       turns of the median over time of its difference from the previous channel with a solution.
       With one time slot this is the unwrap over the channels of CASA; with more, the median keeps
       a single outlier from moving a channel.

    Parameters
    ----------
    h5in : str
        h5parm, modified in place.
    soltab : str, optional
        Soltab name. Default is 'phase000'.

    Returns
    -------
    int
        Number of solutions that still differ by more than pi from the previous channel (noise).
    """
    import warnings
    with tables.open_file(h5in, 'r+') as H:
        if soltab not in H.root.sol000._v_children:
            return 0
        st = H.root.sol000._v_children[soltab]
        axes = st.val.attrs['AXES'].decode().split(',')
        vals = st.val[:]
        # time and freq first (length 1 when absent), then the rest
        for name in ('time', 'freq'):
            if name not in axes:
                vals = vals[np.newaxis]
                axes = [name] + axes
        order = [axes.index('time'), axes.index('freq')]
        order += [i for i in range(len(axes)) if i not in order]
        vals = np.transpose(vals, order).copy()
        shape = vals.shape
        vals = vals.reshape(shape[0], shape[1], -1)  # time, freq, rest
        # 1. along time, per channel (over the finite slots)
        flat = vals.reshape(vals.shape[0], -1)  # a view
        finite = np.isfinite(flat)
        full = finite.all(axis=0)
        flat[:, full] = np.unwrap(flat[:, full], axis=0)
        for j in np.nonzero(~full & (finite.sum(axis=0) > 1))[0]:
            flat[finite[:, j], j] = np.unwrap(flat[finite[:, j], j])
        # 2. whole channels, following the previous channel with a solution
        twopi = 2 * np.pi
        last = vals[:, 0].copy()
        for c in range(1, vals.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN: flagged antennas
                turns = np.round(np.nanmedian(vals[:, c] - last, axis=0) / twopi)
            vals[:, c] -= twopi * np.nan_to_num(turns)[None, :]
            last = np.where(np.isfinite(vals[:, c]), vals[:, c], last)
        with np.errstate(invalid='ignore'):
            jumps = int(np.sum(np.abs(np.diff(vals, axis=1)) > np.pi))
        vals = np.transpose(vals.reshape(shape), np.argsort(order))
        st.val[:] = vals.reshape(st.val.shape)
    if jumps:
        terminal_print(f'{h5in}: {jumps} {soltab} solutions differ by more than 180 deg from the previous '
                       'channel (noise)')
    return jumps


def apply(ms, h5list, incol, outcol, invert=True, msout='.', interpolation='linear', updateweights=False):
    """
    Apply (or with invert=False corrupt with) a list of h5parms, see submods.applycal.applycal.

    Solutions that are NaN (flagged by the QA) flag the data they are applied to. With
    interpolation='linear' DP3 also flags the time slots next to a NaN solution, so solutions solved
    on the MS itself (same time grid) are applied with 'nearest': otherwise every in-place apply grows
    the flags around each flagged time slot. 'nearest' also holds in frequency, so solutions in
    channel blocks that need the linear frequency interpolation keep 'linear'.

    Parameters
    ----------
    ms : str
        Measurement Set.
    h5list : list of str
        h5parms, applied in this order.
    incol, outcol : str
        Input and output data columns.
    invert : bool, optional
        True to correct, False to corrupt. Default is True.
    msout : str, optional
        Output MS ('.' to write outcol in ms). Default is '.'.
    interpolation : str, optional
        DP3 applycal interpolation, in time and frequency, as CASA applycal; the phases
        of the h5parms are unwrapped for it (unwrap_phase). Default is 'linear'.
    updateweights : bool, optional
        Scale WEIGHT_SPECTRUM with the applied amplitudes (|g_i|^2 |g_j|^2, as CASA applycal calwt=True):
        the archive weights follow the raw gains. Only for a new msout: in place, WEIGHT_SPECTRUM is
        shared by all columns and repeated applies from the raw data would scale it again. Default False.
    """
    if updateweights and msout == '.':
        raise ValueError('apply: updateweights needs a new output MS (msout), WEIGHT_SPECTRUM is shared')
    # Dysco for the data, the weights uncompressed (see prepare.DYSCO_ARGS)
    applycal(ms, h5list, msincol=incol, msoutcol=outcol, msout=msout, dysco=True, dysco_weightbitrate=0,
             invert=invert, missingantennabehavior='flag', metadata_compression=False,
             interpolation=interpolation, updateweights=updateweights)


def concat_h5_time(h5list, h5out):
    """
    Concatenate h5parms with the same soltabs, frequencies and antennas along time (e.g. the
    solutions of the separate scans of a calibrator), so that they can be interpolated in time.

    Parameters
    ----------
    h5list : list of str
        Input h5parms, in any order (they are sorted by time).
    h5out : str
        Output h5parm.
    """
    starts = []
    for h5 in h5list:
        with tables.open_file(h5) as H:
            st = H.root.sol000._v_children[list(H.root.sol000._v_children.keys())[0]]
            starts.append(np.min(st.time[:]) if 'time' in st._v_children else 0.)
    h5list = [h for _, h in sorted(zip(starts, h5list))]
    if os.path.exists(h5out):
        os.remove(h5out)
    Hin = [h5parm(h5) for h5 in h5list]
    Hout = h5parm(h5out, readonly=False)
    ss_out = Hout.makeSolset('sol000')
    ss0 = Hin[0].getSolset('sol000')
    # as tuples, so that the fields are cast to the dtypes of the new tables (DP3 writes shorter names)
    ss_out.obj.antenna.append([tuple(row) for row in ss0.obj.antenna[:]])
    ss_out.obj.source.append([tuple(row) for row in ss0.obj.source[:]])
    for name in ss0.getSoltabNames():
        sts = [H.getSolset('sol000').getSoltab(name) for H in Hin]
        axes = sts[0].getAxesNames()
        taxis = axes.index('time')
        vals = np.concatenate([st.getValues(retAxesVals=False) for st in sts], axis=taxis)
        weights = np.concatenate([st.getValues(retAxesVals=False, weight=True) for st in sts], axis=taxis)
        axesvals = [np.concatenate([st.getAxisValues('time') for st in sts]) if a == 'time'
                    else sts[0].getAxisValues(a) for a in axes]
        ss_out.makeSoltab(sts[0].getType(), name, axesNames=axes, axesVals=axesvals, vals=vals, weights=weights)
    for H in Hin:
        H.close()
    Hout.close()
    # the scans are interpolated in time when applied to the fields in between
    unwrap_phase(h5out)
    terminal_print(f'Concatenated {len(h5list)} h5parms in time -> {h5out}')


def fold_transfer_terms(bandpass_h5, delay_h5list, phase_h5list, h5out):
    """
    Write the bandpass that is transferred to the other fields: the merged bandpass times the terms
    that were taken out of the bandpass calibrator data before the bandpass solve and are stable in
    time, which the other fields need as well:

    - the delay per antenna (median over the times and bandpass calibrator scans), as a phase slope;
    - the X-Y phase offset per antenna of the per-dump phaseonly solutions (vector mean over time,
      circular mean over the scans), half added to X and half subtracted from Y. Their common phase,
      like the phase000 of the delay solve, drifts with time and is solved on the phase calibrator.

    Without them the X-Y offsets (up to 180 deg) make a Stokes I image of the phase calibrator
    decorrelate, and so does the treeMF delay solve on (XX+YY)/2. The reference antenna's own X-Y phase
    is in neither, that is the cross-hand phase (xfsolve).

    Parameters
    ----------
    bandpass_h5 : str
        Merged bandpass (amplitude000, phase000; pols XX, YY).
    delay_h5list : list of str
        Per-scan delay h5parms (delay000) of the last bandpass cycle.
    phase_h5list : list of str
        Per-scan per-dump phaseonly h5parms (phase000) of the last bandpass cycle.
    h5out : str
        Output h5parm, a copy of bandpass_h5 with the terms folded into phase000.

    Returns
    -------
    dict
        Per antenna the delay (ns) and the X-Y phase offset (deg) that were folded in.
    """
    from submods.h5_helpers.solution_qa import _axes, _from_standard, _to_standard

    def names(values):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]

    with tables.open_file(bandpass_h5) as H:
        ants = names(H.root.sol000.phase000.ant[:])
    delays, xy = [], []
    for h5 in delay_h5list:
        with tables.open_file(h5) as H:
            st = H.root.sol000.delay000
            with np.errstate(all='ignore'):
                values = np.nanmedian(_to_standard(st.val[:], _axes(st)), axis=(0, 1, 3, 4))
            per_ant = dict(zip(names(st.ant[:]), values))
        delays.append([per_ant.get(a, np.nan) for a in ants])
    for h5 in phase_h5list:
        with tables.open_file(h5) as H:
            st = H.root.sol000.phase000
            pols = names(st.pol[:])
            phase = _to_standard(st.val[:], _axes(st))
            diff = phase[..., pols.index('XX')] - phase[..., pols.index('YY')]
            with np.errstate(all='ignore'):
                values = np.nanmean(np.exp(1j * diff), axis=(0, 1, 3))
            per_ant = dict(zip(names(st.ant[:]), values))
        xy.append([per_ant.get(a, np.nan) for a in ants])
    with np.errstate(all='ignore'):
        delay = np.nanmedian(np.array(delays), axis=0)
        offset = np.angle(np.nanmean(np.array(xy), axis=0))
    delay = np.where(np.isfinite(delay), delay, 0.)
    offset = np.where(np.isfinite(offset), offset, 0.)

    shutil.copy(bandpass_h5, h5out)
    with tables.open_file(h5out, 'r+') as H:
        st = H.root.sol000.phase000
        axes = _axes(st)
        pols = names(st.pol[:])
        phase = _to_standard(st.val[:], axes)
        # DP3 applies delay000 as the phase 2 pi nu tau
        phase = phase + 2 * np.pi * st.freq[:][None, :, None, None, None] * delay[None, None, :, None, None]
        phase[..., pols.index('XX')] += offset[None, None, :, None] / 2
        phase[..., pols.index('YY')] -= offset[None, None, :, None] / 2
        st.val[:] = _from_standard(np.angle(np.exp(1j * phase)), axes)
    unwrap_phase(h5out)  # for the linear interpolation in frequency
    terminal_print(f'Transferred bandpass {h5out}: folded in the delays (median |delay| '
                   f'{np.median(np.abs(delay)) * 1e9:.2f} ns) and X-Y phase offsets (median |offset| '
                   f'{np.degrees(np.median(np.abs(offset))):.0f} deg) of the bandpass calibrator')
    return dict(delay_ns={a: float(d * 1e9) for a, d in zip(ants, delay)},
                xy_phase_deg={a: float(np.degrees(o)) for a, o in zip(ants, offset)})


def normalise_amplitudes(h5in):
    """
    Divide the amplitudes in an h5parm by their median over all times, frequencies and antennas, so
    that the absolute flux scale is kept from the bandpass calibrator.

    Parameters
    ----------
    h5in : str
        h5parm with amplitude000.

    Returns
    -------
    float
        The normalisation factor.
    """
    with tables.open_file(h5in, 'r+') as H:
        st = H.root.sol000.amplitude000
        amp = st.val[:]
        ok = np.isfinite(amp) & (st.weight[:] > 0)
        norm = float(np.median(amp[ok]))
        st.val[:] = amp / norm
    terminal_print(f'Normalised the amplitudes in {h5in} by {norm:.4f}')
    return norm


@_track_resource_phase("solve")
def solve_xf(ms, h5out, incol, modelcol='MODEL_FEED', mode='model', nchan=1):
    """
    Cross-hand phase (one solution for the MS) with the xfsolve.py DP3 step.

    Parameters
    ----------
    ms : str
        Measurement Set (calibrated data in incol, feed-frame model in modelcol for mode=model).
    h5out : str
        Output h5parm (phase000, pol XX holds theta).
    incol : str
        Data column.
    modelcol : str, optional
        Model column. Default MODEL_FEED.
    mode : str, optional
        'model' or 'nullv'. Default 'model'.
    nchan : int, optional
        Channels per solution. Default 1.
    """
    cmd = (f'DP3 numthreads={nthreads()} msin={ms} msin.datacolumn={incol} msin.weightcolumn=WEIGHT_SPECTRUM '
           'msout= steps=[xf] xf.type=PythonDPPP xf.python.module=xfsolve xf.python.class=XfSolve '
           f'xf.h5parm={h5out} xf.mode={mode} xf.modelcolumn={modelcol} xf.solint=0 xf.nchan={nchan}')
    terminal_print('xfsolve:', cmd)
    run(cmd, log=True)


def solve_xf_diode(ms, h5out, h5list, diodefile, nchan=1):
    """
    Per-antenna noise diode X-Y phases of an autocorrelation MS (xfsolve.py mode=diode), with the
    calibration h5list applied on the fly (DP3 applycal, linear interpolation, in this order).

    Parameters
    ----------
    ms : str
        Autocorrelation MS (feeds swapped), see diode.make_autocorr_scan.
    h5out : str
        Output h5parm (per-antenna phases) and <h5out>.sums.npz.
    h5list : list of str
        h5parms applied before the solve (time-variable gains, bandpass, leakage).
    diodefile : str
        JSON with the diode-on time ranges per antenna.
    nchan : int, optional
        Channels per solution. Default 1.
    """
    from submods.applycal import applycal_dp3_steps
    acargs, nac = applycal_dp3_steps(ms, h5list, missingantennabehavior='flag', interpolation='linear')
    steps = ['ac%d' % i for i in range(nac)] + ['xf']
    # DP3 updates the input MS when msout is empty: with the applycal steps that would write the
    # calibrated data (and the flags of NaN solutions) into DATA, so the output goes to a scratch MS
    scratch = ms.rstrip('/') + '.xfdiode_tmp'
    cmd = (f'DP3 numthreads={nthreads()} msin={ms} msin.datacolumn=DATA msout={scratch} msout.overwrite=True msout.scalarflags=False '
           'msout.uvwcompression=False msout.antennacompression=False ' + acargs +
           'xf.type=PythonDPPP xf.python.module=xfsolve xf.python.class=XfSolve '
           f'xf.h5parm={h5out} xf.mode=diode xf.diodefile={diodefile} xf.solint=0 xf.nchan={nchan} '
           'steps=[' + ','.join(steps) + ']')
    terminal_print('xfsolve (diode):', cmd)
    try:
        run(cmd, log=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def combine_xf(h5list, h5out):
    """
    Combine cross-hand phase solutions of separate runs (scans) into one solution, exactly as if they
    had been solved together, from the sums xfsolve saves next to its h5parms.

    Parameters
    ----------
    h5list : list of str
        xfsolve h5parms (same frequencies), each with <h5parm>.sums.npz.
    h5out : str
        Output h5parm (a copy of the first input with the combined solution).

    Returns
    -------
    np.ndarray
        theta (radians) per frequency, NaN where there is no solution.
    """
    sums = [np.load(h5 + '.sums.npz') for h5 in h5list]
    total = np.sum([s['sum'][0] for s in sums], axis=0)
    weight = np.sum([s['weight'][0] for s in sums], axis=0)
    theta = np.angle(total) * (0.5 if str(sums[0]['mode']) == 'nullv' else 1.)
    valid = (weight > 0) & (np.abs(total) > 0)
    theta[~valid] = np.nan
    shutil.copy(h5list[0], h5out)
    with tables.open_file(h5out, 'r+') as H:
        st = H.root.sol000.phase000  # axes time, freq, ant, dir, pol (as written by xfsolve)
        vals = np.zeros(st.val.shape)
        vals[..., 0] = theta[np.newaxis, :, np.newaxis, np.newaxis]
        vals[:, ~valid] = np.nan
        wgt = np.ones(st.val.shape)
        wgt[:, ~valid] = 0.
        st.val[:] = vals
        st.weight[:] = wgt
    extra = {}
    if all('upow' in s.files for s in sums):  # model power (xfsolve mode=model), for the QA
        extra = {key: np.sum([s[key][0] for s in sums], axis=0)[np.newaxis] for key in ('upow', 'ppow')}
    np.savez(h5out + '.sums.npz', sum=total[np.newaxis], weight=weight[np.newaxis], mode=sums[0]['mode'],
             freqs=sums[0]['freqs'], **extra)
    unwrap_phase(h5out)  # for the linear interpolation in frequency (between the chunks of nchan > 1)
    terminal_print(f'Combined {len(h5list)} cross-hand phase solutions -> {h5out}')
    return theta
