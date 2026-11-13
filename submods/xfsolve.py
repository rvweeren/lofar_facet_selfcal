"""
This code is based on MockPyStep.py from https://dp3.readthedocs.io/en/latest/steps/PythonStep.html
Cross-hand phase (Xf) solver, the DP3 counterpart of CASA polcal(poltype='Xf') and of the QuartiCal
crosshand_phase and crosshand_phase_null_v gain types.

To use this script in DPPP/DP3, you can save this file in the same folder as from where you run your DPPP command.
Or you save this python file somewhere you like and run:
export PYTHONPATH=/somewhere/you/like:$PYTHONPATH

Example (solve on gain-calibrated, parallactic angle corrected circular data, then apply):
DP3 msin=my.ms msout= steps=[ac,pa,xf] ac.type=applycal ac.parmdb=gains.h5 ac.correction=phase000 \
    pa.type=PythonDPPP pa.python.module=parang pa.python.class=ParAng \
    xf.type=PythonDPPP xf.python.module=xfsolve xf.python.class=XfSolve xf.h5parm=xf.h5
DP3 msin=my.ms msout=. msout.datacolumn=CORRECTED_DATA steps=[ac] ac.type=applycal ac.parmdb=xf.h5 \
    ac.correction=phase000

"""
from subprocess import check_output
import json
import re
import numpy as np
import os
import shutil
import sys
from casacore.tables import table
from losoto.h5parm import h5parm

#hacky way to figure out the DPPP/DP3 version (important to run this script properly)
DP3name = shutil.which('DP3')
if not DP3name:
    DP3name = shutil.which('DPPP')
try:
    rgx = r'[0-9]+(\.[0-9]+)+'
    grep_version_string = str(check_output(DP3name+' --version', shell=True), 'utf-8')
    DP3_VERSION = float(re.search(rgx, grep_version_string).group()[0:3])
except AttributeError:
    print('WARNING: grep of DP3 version failed.')
    DP3_VERSION=0.0 # set to default

if DP3_VERSION > 5.3:
    from dp3 import Fields

try:
    from dppp import DPStep as Step
except:
    if DP3_VERSION >= 6:
        from dp3.pydp3 import Step
    else:
        from dp3 import Step


def get_parset_value(parset, key, default, kind='string'):
    """Read a parset key, trying the old (getString/getInt) and new (get_string/get_int) DP3 bindings."""
    names = {'string': ('getString', 'get_string'), 'int': ('getInt', 'get_int')}[kind]
    for name in names:
        try:
            return getattr(parset, name)(key)
        except AttributeError:
            continue  # DP3 Python bindings have been renamed.
        except RuntimeError:
            return default  # key not present
    return default


class XfSolve(Step):
    r"""
    Solve for the cross-hand phase theta(nu), one value for the whole array per solution cell:

        X = diag(exp(i theta), 1),   V_obs = X V_model X^H

    so XY (RL) is multiplied by exp(i theta), YX (LR) by exp(-i theta) and the parallel hands are unchanged.
    For circular feeds this is the polarisation angle calibration (position angle = theta/2), for linear
    feeds it is the XY-phase. The least-squares solution is closed form:

      mode=model  theta = arg sum w (d_12 conj(m_12) + conj(d_21) m_21)    (needs a polarised model)
      mode=nullv  theta = 0.5 arg sum w d_XY conj(d_YX)                    (linear feeds only, assumes
                                                                            Stokes V = 0, theta mod 180 deg)
      mode=diode  per antenna a: D_a = <c_a>_on - <c_a>_off,  c_a = (d_XY + conj(d_YX)) / 2 of the
                  autocorrelation of antenna a, averaged over the dumps with the noise diode on and off
                  (see below; the cross-correlations are not used)

    The solutions are written to an h5parm (solset sol000, soltab phase000, pol [XX,YY] or [RR,LL] with
    values [theta, 0] for every antenna), to be applied with DP3 applycal correction=phase000.
    The data passes through this step unchanged. An existing h5parm with the same name is overwritten.
    The accumulated sums are also saved (<h5parm>.sums.npz), so that the solutions of separate runs (for
    example the scans of a calibrator) can be combined exactly: theta = arg of the summed sums.

    mode=diode (linear feeds, MeerKAT): the noise diode injects the same signal into both receptors of an
    antenna, so with the diagonal gains applied (referenced per polarisation to one antenna), the
    autocorrelation cross-hand of antenna a has the phase delta_a + theta, with delta_a the X-Y phase
    of the diode signal path of that antenna and theta the X-Y phase of the reference antenna (the
    cross-hand phase). The mean over the diode-off dumps of the same data (scan) is subtracted, which
    removes the constant X-Y coupling (leakage of the system noise, crosstalk) of the autocorrelations;
    an antenna without diode-off dumps uses the diode-on mean alone. The h5parm holds the per-antenna
    phase arg D_a (not a cross-hand phase solution); the combination over antennas and scans and the
    diode-to-sky correction (theta = median_a arg D_a - spline) are done outside DP3
    (submods/meerkat_cal/diode.py), from the sums: per antenna and solution channel the complex sums
    and weights (unit weight per unflagged visibility) of c_a over the diode-on and diode-off dumps,
    and of (XX+YY)/2 (the power) for diagnostics.

    Parset keys (prefix pystep.):
      h5parm       output h5parm (required)
      mode         model (default), nullv or diode
      modelcolumn  MS column with the model visibilities, default MODEL_DATA (mode=model only).
                   It is read from the MS directly, so no averaging or other step that changes the time
                   or channel axes may come before this step (msin.startchan/nchan are fine).
      solint       solution interval in time slots, default 0 (one solution for the whole observation)
      nchan        channels per solution, default 1
      diodefile    JSON file with the diode-on time ranges (mode=diode, required):
                   {antenna name: [[t_start, t_end], ...]} in MS TIME (s); antennas not in the file
                   have no diode-on data
      offguard     dumps within this many seconds of a diode-on range are not used as diode-off dumps
                   (they can be partly on), default the dump time, which excludes the neighbouring
                   dumps (mode=diode)

    Notes:
    - Solve after the gains (and the leakage, as CASA recommends), leakage biases the cross-hands.
    - Circular feeds: X and the parallactic rotation P commute, derotate the data (parang.py mode=correct)
      or corrupt the model (parang.py mode=corrupt).
    - Linear feeds: X acts in the feed frame and does not commute with P. Do not derotate the data, put the
      model in the feed frame instead with parang.py mode=corrupt. mode=nullv works on non-derotated data.
    - Stokes U in the feed frame must be non-zero, the model needs an accurate polarisation angle.
    """

    def __init__(self, parset, prefix):
        """
        Set up the step (constructor). Read the parset here.

        Args:
          parset: Parameter set for the entire pipeline
          prefix: Prefix for this step."
        """

        super().__init__()

        self.h5parm = get_parset_value(parset, prefix + "h5parm", "")
        if not self.h5parm:
            sys.exit("XfSolve: h5parm is required")
        self.mode = get_parset_value(parset, prefix + "mode", "model").lower()
        if self.mode not in ('model', 'nullv', 'diode'):
            sys.exit("XfSolve: mode should be model, nullv or diode, not " + self.mode)
        self.modelcolumn = get_parset_value(parset, prefix + "modelcolumn", "MODEL_DATA")
        self.solint = get_parset_value(parset, prefix + "solint", 0, kind='int')
        self.nchan_sol = max(1, get_parset_value(parset, prefix + "nchan", 1, kind='int'))
        self.diodefile = get_parset_value(parset, prefix + "diodefile", "")
        self.offguard = float(get_parset_value(parset, prefix + "offguard", "-1"))
        if self.mode == 'diode':
            if not self.diodefile:
                sys.exit("XfSolve: mode=diode needs diodefile")
            if self.solint > 0:
                sys.exit("XfSolve: mode=diode solves one solution per run (solint=0)")
            with open(self.diodefile) as f:
                self.diode_ranges = {name: np.array(r, dtype=float).reshape(-1, 2) for name, r in json.load(f).items()}

        self.n_slot = 0  # time slots processed
        self.fetch_uvw = True

    def get_required_fields(self):
        if DP3_VERSION>5.3:
            return (Fields.DATA | Fields.FLAGS | Fields.WEIGHTS | Fields.UVW)
        else:
            pass

    def get_provided_fields(self):
        # the data pass through unchanged: providing nothing keeps DP3 from writing the input column back
        # (msout= updates the input MS; with DATA provided, a Dysco column was rewritten and requantised)
        if DP3_VERSION>5.3:
            return Fields()
        else:
            pass

    def update_info(self, dpinfo):
        """
        Process metadata. This will be called before any call to process.
        Sets up the solution grid and, for mode=model, the access to the model column.

        Args:
          dpinfo: DPInfo object with all metadata, see docs in pydp3.cc
        """

        super().update_info(dpinfo)

        ms = dpinfo.ms_name
        if dpinfo.n_correlations != 4:
            sys.exit("XfSolve: needs 4 correlations, got " + str(dpinfo.n_correlations))
        with table(ms + '::POLARIZATION', ack=False, readonly=True) as t:
            corr_type = t.getcol('CORR_TYPE')[0]
        if np.all(np.isin(corr_type, [5, 6, 7, 8])):
            self.feedtype = 'circular'
        elif np.all(np.isin(corr_type, [9, 10, 11, 12])):
            self.feedtype = 'linear'
        else:
            sys.exit("XfSolve: unsupported correlation types " + str(corr_type))
        if self.mode == 'nullv' and self.feedtype == 'circular':
            sys.exit("XfSolve: mode=nullv needs linear feeds, for circular feeds Stokes V is not in the cross-hands")

        self.ant_names = list(dpinfo.antenna_names)
        self.ant_positions = np.array(dpinfo.antenna_positions)
        self.phase_center = list(dpinfo.phase_center)
        self.ant1 = np.array(dpinfo.first_antenna_indices)
        self.ant2 = np.array(dpinfo.second_antenna_indices)
        self.crosscorr = self.ant1 != self.ant2
        self.autocorr = self.ant1 == self.ant2
        # the baselines this mode uses: the autocorrelations for the diode, else the cross-correlations
        self.used = self.autocorr if self.mode == 'diode' else self.crosscorr
        self.freqs = np.array(dpinfo.channel_frequencies)
        self.start_channel = dpinfo.start_channel
        nchan = len(self.freqs)
        self.fblock = np.arange(nchan) // self.nchan_sol
        self.n_fblock = self.fblock[-1] + 1
        self.n_tblock = 1 if self.solint <= 0 else int(np.ceil(dpinfo.n_times / self.solint))
        if self.mode == 'diode':
            self.setup_diode(dpinfo, ms)

        self.sum = np.zeros((self.n_tblock, self.n_fblock), dtype=np.complex128)
        self.weight = np.zeros((self.n_tblock, self.n_fblock))
        # mode=model: weighted power of the model cross-hands and of the model polarised intensity,
        # their ratio (|U| / P in the feed frame for linear feeds) tells how well the model fixes theta
        self.upow = np.zeros((self.n_tblock, self.n_fblock))
        self.ppow = np.zeros((self.n_tblock, self.n_fblock))
        self.time_sum = np.zeros(self.n_tblock)
        self.time_n = np.zeros(self.n_tblock)

        if self.mode == 'model':
            self.mstable = table(ms, ack=False, readonly=True)
            if self.modelcolumn not in self.mstable.colnames():
                sys.exit("XfSolve: model column " + self.modelcolumn + " not in " + ms)
            times = self.mstable.getcol('TIME')
            if np.any(np.diff(times) < 0):
                sys.exit("XfSolve: the MS is not sorted in time")
            self.ms_times, self.ms_time_start, self.ms_time_nrow = np.unique(times, return_index=True,
                                                                             return_counts=True)
            # MS antenna index -> DP3 antenna index (by name), and DP3 baseline -> buffer index
            with table(ms + '::ANTENNA', ack=False, readonly=True) as t:
                ms_ant_names = list(t.getcol('NAME'))
            self.ms_to_dp3_ant = np.array([self.ant_names.index(n) if n in self.ant_names else -1
                                           for n in ms_ant_names])
            self.n_ant = len(self.ant_names)
            self.bl_index = np.full(self.n_ant * self.n_ant, -1)
            self.bl_index[self.ant1 * self.n_ant + self.ant2] = np.arange(len(self.ant1))

    def setup_diode(self, dpinfo, ms):
        """mode=diode: per-antenna accumulators and the diode-on time ranges per DP3 antenna index."""
        if self.feedtype != 'linear':
            sys.exit("XfSolve: mode=diode needs linear feeds")
        if not np.any(self.autocorr):
            sys.exit("XfSolve: mode=diode needs autocorrelations, " + ms + " has none")
        n_ant = len(self.ant_names)
        shape = (n_ant, self.n_fblock)
        self.d_on = np.zeros(shape, dtype=np.complex128)
        self.d_off = np.zeros(shape, dtype=np.complex128)
        self.w_on = np.zeros(shape)
        self.w_off = np.zeros(shape)
        self.p_on = np.zeros(shape)
        self.p_off = np.zeros(shape)
        self.ndump_on = np.zeros(n_ant, dtype=int)
        self.ndump_off = np.zeros(n_ant, dtype=int)
        self.fblock_start = np.arange(0, len(self.freqs), self.nchan_sol)
        interval = getattr(dpinfo, 'time_interval', None)
        if not interval:
            with table(ms, ack=False, readonly=True) as t:
                interval = float(np.median(t.getcol('INTERVAL', 0, 1000)))
        self.guard = self.offguard if self.offguard >= 0 else float(interval)
        self.on_ranges = [self.diode_ranges.get(name, np.zeros((0, 2))) for name in self.ant_names]
        unknown = sorted(set(self.diode_ranges) - set(self.ant_names))
        if unknown:
            print('XfSolve: antennas in the diodefile but not in the data: ' + ', '.join(unknown))

    def diode_state(self, time):
        """Per antenna: 1 diode on, 0 diode off, -1 not used (next to a diode-on range)."""
        state = np.zeros(len(self.ant_names), dtype=int)
        for a, r in enumerate(self.on_ranges):
            if len(r) == 0:
                continue
            if np.any((r[:, 0] <= time) & (time <= r[:, 1])):
                state[a] = 1
            elif np.any((r[:, 0] - self.guard <= time) & (time <= r[:, 1] + self.guard)):
                state[a] = -1
        return state

    def accumulate_diode(self, time, data, flags):
        """mode=diode: add the autocorrelation cross-hands of one time slot to the on or off sums."""
        d = data[self.autocorr]
        f = flags[self.autocorr]
        ant = self.ant1[self.autocorr]
        c = 0.5 * (d[:, :, 1] + np.conj(d[:, :, 2]))
        p = 0.5 * (d[:, :, 0].real + d[:, :, 3].real)
        w = ~np.any(f, axis=2) & np.isfinite(c) & np.isfinite(p)
        c = np.where(w, c, 0.)
        p = np.where(w, p, 0.)
        w = w.astype(float)
        cs = np.add.reduceat(c, self.fblock_start, axis=1)
        ps = np.add.reduceat(p, self.fblock_start, axis=1)
        ws = np.add.reduceat(w, self.fblock_start, axis=1)
        state = self.diode_state(time)
        for i, a in enumerate(ant):
            if not ws[i].any() or state[a] < 0:
                continue
            if state[a] == 1:
                self.d_on[a] += cs[i]
                self.p_on[a] += ps[i]
                self.w_on[a] += ws[i]
                self.ndump_on[a] += 1
            else:
                self.d_off[a] += cs[i]
                self.p_off[a] += ps[i]
                self.w_off[a] += ws[i]
                self.ndump_off[a] += 1

    def finish_diode(self):
        """mode=diode: write the per-antenna diode X-Y phases (h5parm) and the sums."""
        with np.errstate(invalid='ignore', divide='ignore'):
            on = self.d_on / self.w_on
            off = self.d_off / self.w_off
            dio = np.where(self.w_off > 0, on - off, on)
        valid = (self.w_on > 0) & np.isfinite(dio) & (np.abs(dio) > 0)
        phase = np.where(valid, np.angle(dio), np.nan)  # (ant, freq)
        times = self.time_sum / np.maximum(self.time_n, 1)
        freqs = np.array([np.mean(self.freqs[self.fblock == fb]) for fb in range(self.n_fblock)])
        n_ant = len(self.ant_names)
        vals = np.zeros((1, self.n_fblock, n_ant, 1, 2))
        vals[0, :, :, 0, 0] = phase.T
        vals[0, :, :, 0, 1] = np.where(valid.T, 0., np.nan)
        weights = np.repeat(valid.T[np.newaxis, :, :, np.newaxis, np.newaxis], 2, axis=4).astype(float)
        if os.path.exists(self.h5parm):
            os.remove(self.h5parm)
        H = h5parm(self.h5parm, readonly=False)
        solset = H.makeSolset('sol000')
        solset.obj.antenna.append([(name, pos) for name, pos in zip(self.ant_names, self.ant_positions)])
        solset.obj.source.append([('pointing', self.phase_center)])
        solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                          axesVals=[times[:1], freqs, self.ant_names, ['pointing'], ['XX', 'YY']], vals=vals,
                          weights=weights)
        H.close()
        np.savez(self.h5parm + '.sums.npz', mode=self.mode, freqs=freqs, ant_names=np.array(self.ant_names),
                 time=times[:1], d_on=self.d_on, w_on=self.w_on, d_off=self.d_off, w_off=self.w_off,
                 p_on=self.p_on, p_off=self.p_off, ndump_on=self.ndump_on, ndump_off=self.ndump_off)
        nant = int(np.sum(self.ndump_on > 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            coh = np.nanmedian(np.abs(dio) / (self.p_off / self.w_off), axis=1)
        print('XfSolve: diode X-Y phases of ' + str(nant) + ' of ' + str(n_ant) + ' antennas (' +
              str(int(np.sum((self.ndump_on > 0) & (self.ndump_off == 0)))) + ' without diode-off dumps), '
              'median diode coherence {:.4f}, written to {}'.format(
                  float(np.nanmedian(coh[self.ndump_on > 0])) if nant else np.nan, self.h5parm), flush=True)

    def show(self):
        """Print a summary of the step and its settings"""

        print("\nXfSolve")
        print("  mode:          " + self.mode)
        print("  feed type:     " + self.feedtype)
        if self.mode == 'model':
            print("  modelcolumn:   " + self.modelcolumn)
        if self.mode == 'diode':
            print("  diodefile:     " + self.diodefile)
            print("  offguard:      " + str(self.guard) + " s")
        print("  solint:        " + str(self.solint) + " (time slots, 0 is the whole observation)")
        print("  nchan:         " + str(self.nchan_sol))
        print("  h5parm:        " + self.h5parm)

    def read_model(self, time, shape):
        """Model visibilities for this time slot, in DP3 baseline order (NaN for baselines not in the MS)."""
        i = np.searchsorted(self.ms_times, time)
        for j in (i - 1, i):
            if 0 <= j < len(self.ms_times) and abs(self.ms_times[j] - time) < 1e-3:
                break
        else:
            sys.exit("XfSolve: time " + str(time) + " not found in the MS, steps before XfSolve may not change the time axis")
        start, nrow = int(self.ms_time_start[j]), int(self.ms_time_nrow[j])
        a1 = self.ms_to_dp3_ant[self.mstable.getcol('ANTENNA1', startrow=start, nrow=nrow)]
        a2 = self.ms_to_dp3_ant[self.mstable.getcol('ANTENNA2', startrow=start, nrow=nrow)]
        rows = self.mstable.getcol(self.modelcolumn, startrow=start, nrow=nrow)
        rows = rows[:, self.start_channel:self.start_channel + shape[1], :]
        if rows.shape[1] != shape[1]:
            sys.exit("XfSolve: " + self.modelcolumn + " has fewer channels than the data, steps before XfSolve may not change the channels")
        keep = (a1 >= 0) & (a2 >= 0)
        index = self.bl_index[a1[keep] * self.n_ant + a2[keep]]
        model = np.full(shape, np.nan, dtype=np.complex64)
        model[index[index >= 0]] = rows[keep][index >= 0]
        return model

    def process(self, dpbuffer):
        """
        Process one time slot of data. This function MUST call process_next_step.

        Args:
          dpbuffer: DPBuffer object which can contain data, flags and weights
                    for one time slot.
        """

        data = np.array(dpbuffer.get_data(), copy=False)
        flags = np.array(dpbuffer.get_flags(), copy=False)
        weights = np.array(dpbuffer.get_weights(), copy=False)

        tb = 0 if self.solint <= 0 else min(self.n_slot // self.solint, self.n_tblock - 1)
        self.n_slot += 1
        if np.all(flags[self.used]):
            # nothing to add, e.g. a time slot DP3 inserted in a gap between scans (not in the MS)
            if DP3_VERSION>5.3:
                next_step = self.get_next_step()
                if next_step is not None:
                    next_step.process(dpbuffer)
            else:
                self.process_next_step(dpbuffer)
            return
        self.time_sum[tb] += dpbuffer.get_time()
        self.time_n[tb] += 1

        if self.mode == 'diode':
            self.accumulate_diode(dpbuffer.get_time(), data, flags)
            if DP3_VERSION>5.3:
                next_step = self.get_next_step()
                if next_step is not None:
                    next_step.process(dpbuffer)
            else:
                self.process_next_step(dpbuffer)
            return

        # the cross-hands (XY,YX or RL,LR) of the cross-correlations
        d12, d21 = data[self.crosscorr, :, 1], data[self.crosscorr, :, 2]
        w12 = np.where(flags[self.crosscorr, :, 1], 0., weights[self.crosscorr, :, 1])
        w21 = np.where(flags[self.crosscorr, :, 2], 0., weights[self.crosscorr, :, 2])
        if self.mode == 'model':
            model = self.read_model(dpbuffer.get_time(), data.shape)
            m12, m21 = model[self.crosscorr, :, 1], model[self.crosscorr, :, 2]
            w12 = np.where(np.isfinite(d12) & np.isfinite(m12), w12, 0.)
            w21 = np.where(np.isfinite(d21) & np.isfinite(m21), w21, 0.)
            term = (w12 * np.nan_to_num(d12 * np.conj(m12)) +
                    w21 * np.nan_to_num(np.conj(d21) * m21))
            wsum = w12 + w21
            upow = w12 * np.abs(np.nan_to_num(m12)) ** 2 + w21 * np.abs(np.nan_to_num(m21)) ** 2
            if self.feedtype == 'linear':
                # XY, YX = U +- iV and XX - YY = 2Q in the feed frame
                qpow = 0.5 * wsum * np.abs(np.nan_to_num(model[self.crosscorr, :, 0] - model[self.crosscorr, :, 3])) ** 2
                ppow = upow + qpow
            else:
                ppow = upow  # RL, LR = Q +- iU carry all of the linear polarisation
        else:
            w = np.where(np.isfinite(d12) & np.isfinite(d21), np.minimum(w12, w21), 0.)
            term = w * np.nan_to_num(d12 * np.conj(d21))
            wsum = w
        for fb in range(self.n_fblock):
            sel = self.fblock == fb
            self.sum[tb, fb] += np.sum(term[:, sel])
            self.weight[tb, fb] += np.sum(wsum[:, sel])
            if self.mode == 'model':
                self.upow[tb, fb] += np.sum(upow[:, sel])
                self.ppow[tb, fb] += np.sum(ppow[:, sel])

        # Send processed data to the next step
        if DP3_VERSION>5.3:
            next_step = self.get_next_step()
            if next_step is not None:
                next_step.process(dpbuffer)
        else:
            self.process_next_step(dpbuffer)

    def finish(self):
        """
        Compute the cross-hand phases and write them to the h5parm.
        """

        if self.mode == 'diode':
            self.finish_diode()
            return
        if self.mode == 'model':
            self.mstable.close()
        valid = (self.weight > 0) & (np.abs(self.sum) > 0)
        theta = np.angle(self.sum)
        if self.mode == 'nullv':
            theta *= 0.5
        theta[~valid] = np.nan

        times = self.time_sum / np.maximum(self.time_n, 1)
        freqs = np.array([np.mean(self.freqs[self.fblock == fb]) for fb in range(self.n_fblock)])
        n_ant = len(self.ant_names)
        vals = np.zeros((self.n_tblock, self.n_fblock, n_ant, 1, 2))
        vals[..., 0] = theta[:, :, None, None]
        vals[~valid] = np.nan
        weights = np.repeat(valid[:, :, None, None, None], n_ant, axis=2).repeat(2, axis=4).astype(float)
        pols = ['XX', 'YY'] if self.feedtype == 'linear' else ['RR', 'LL']

        if os.path.exists(self.h5parm):
            os.remove(self.h5parm)
        H = h5parm(self.h5parm, readonly=False)
        solset = H.makeSolset('sol000')
        solset.obj.antenna.append([(name, pos) for name, pos in zip(self.ant_names, self.ant_positions)])
        solset.obj.source.append([('pointing', self.phase_center)])
        solset.makeSoltab('phase', 'phase000', axesNames=['time', 'freq', 'ant', 'dir', 'pol'],
                          axesVals=[times, freqs, self.ant_names, ['pointing'], pols], vals=vals, weights=weights)
        H.close()
        # the accumulated sums, so that solutions of separate runs (e.g. scans) can be combined exactly
        np.savez(self.h5parm + '.sums.npz', sum=self.sum, weight=self.weight, upow=self.upow, ppow=self.ppow,
                 mode=self.mode, freqs=freqs)

        print('XfSolve: cross-hand phase (' + self.mode + ', ' + self.feedtype + ' feeds) solved for ' +
              str(np.sum(valid)) + ' of ' + str(valid.size) + ' solution cells, range ' +
              '{:.2f} to {:.2f} deg, written to {}'.format(np.nanmin(np.degrees(theta)) if np.any(valid) else np.nan,
                                                          np.nanmax(np.degrees(theta)) if np.any(valid) else np.nan,
                                                          self.h5parm), flush=True)
