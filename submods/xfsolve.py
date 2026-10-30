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

    The solutions are written to an h5parm (solset sol000, soltab phase000, pol [XX,YY] or [RR,LL] with
    values [theta, 0] for every antenna), to be applied with DP3 applycal correction=phase000.
    The data passes through this step unchanged. An existing h5parm with the same name is overwritten.

    Parset keys (prefix pystep.):
      h5parm       output h5parm (required)
      mode         model (default) or nullv
      modelcolumn  MS column with the model visibilities, default MODEL_DATA (mode=model only).
                   It is read from the MS directly, so no averaging or other step that changes the time
                   or channel axes may come before this step (msin.startchan/nchan are fine).
      solint       solution interval in time slots, default 0 (one solution for the whole observation)
      nchan        channels per solution, default 1

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
        if self.mode not in ('model', 'nullv'):
            sys.exit("XfSolve: mode should be model or nullv, not " + self.mode)
        self.modelcolumn = get_parset_value(parset, prefix + "modelcolumn", "MODEL_DATA")
        self.solint = get_parset_value(parset, prefix + "solint", 0, kind='int')
        self.nchan_sol = max(1, get_parset_value(parset, prefix + "nchan", 1, kind='int'))

        self.n_slot = 0  # time slots processed
        self.fetch_uvw = True

    def get_required_fields(self):
        if DP3_VERSION>5.3:
            return (Fields.DATA | Fields.FLAGS | Fields.WEIGHTS | Fields.UVW)
        else:
            pass

    def get_provided_fields(self):
        if DP3_VERSION>5.3:
            return (Fields.DATA | Fields.FLAGS | Fields.WEIGHTS | Fields.UVW)
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
        self.freqs = np.array(dpinfo.channel_frequencies)
        self.start_channel = dpinfo.start_channel
        nchan = len(self.freqs)
        self.fblock = np.arange(nchan) // self.nchan_sol
        self.n_fblock = self.fblock[-1] + 1
        self.n_tblock = 1 if self.solint <= 0 else int(np.ceil(dpinfo.n_times / self.solint))

        self.sum = np.zeros((self.n_tblock, self.n_fblock), dtype=np.complex128)
        self.weight = np.zeros((self.n_tblock, self.n_fblock))
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

    def show(self):
        """Print a summary of the step and its settings"""

        print("\nXfSolve")
        print("  mode:          " + self.mode)
        print("  feed type:     " + self.feedtype)
        if self.mode == 'model':
            print("  modelcolumn:   " + self.modelcolumn)
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
        self.time_sum[tb] += dpbuffer.get_time()
        self.time_n[tb] += 1
        self.n_slot += 1

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
        else:
            w = np.where(np.isfinite(d12) & np.isfinite(d21), np.minimum(w12, w21), 0.)
            term = w * np.nan_to_num(d12 * np.conj(d21))
            wsum = w
        for fb in range(self.n_fblock):
            sel = self.fblock == fb
            self.sum[tb, fb] += np.sum(term[:, sel])
            self.weight[tb, fb] += np.sum(wsum[:, sel])

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

        print('XfSolve: cross-hand phase (' + self.mode + ', ' + self.feedtype + ' feeds) solved for ' +
              str(np.sum(valid)) + ' of ' + str(valid.size) + ' solution cells, range ' +
              '{:.2f} to {:.2f} deg, written to {}'.format(np.nanmin(np.degrees(theta)) if np.any(valid) else np.nan,
                                                          np.nanmax(np.degrees(theta)) if np.any(valid) else np.nan,
                                                          self.h5parm), flush=True)
