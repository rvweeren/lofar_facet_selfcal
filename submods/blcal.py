"""
Baseline-based calibration (closure-error correction) as a DP3 Python step, the counterpart of AIPS
BLCAL and CASA blcal.

Antenna-based calibration cannot remove errors that belong to a baseline rather than to an antenna
(closure errors), for example from bandpass mismatches between antennas in old single-channel VLA
continuum data, or from second-order polarisation leakage in the parallel hands. When such errors are
stable in time, they can be measured against a good sky model as one complex factor per baseline:

    B_pq(baseline) = sum w d conj(m) / sum w |m|^2       (d = calibrated data, m = model)

for each parallel hand p = q (XX and YY, or RR and LL), per solution interval and channel block, and
then divided out of the data. The cross hands (XY, YX / RL, LR) are divided by sqrt(B_pp B_qq), so the
polarisation products are not distorted. Factors that are not significant (|B - 1| below minsnr times
their formal error) or that deviate more than maxdev from 1 (more likely bad data or a model error than
a closure error) are set to 1, i.e. not applied.

h5parm and DP3 applycal only handle antenna-based Jones matrices, so the factors are written to their own
HDF5 file (see write_solutions) and applied with this same step in mode=apply.

To use this script in DP3, put it in the folder where you run DP3, or add its folder to PYTHONPATH.

Example (solve on the calibrated data against MODEL_DATA, then apply to DATA and CORRECTED_DATA):
DP3 msin=my.ms msin.datacolumn=CORRECTED_DATA msin.extradatacolumns=[MODEL_DATA] msout= steps=[bl] \
    bl.type=PythonDPPP bl.python.module=blcal bl.python.class=BlCal bl.mode=solve bl.solutions=bl.h5
DP3 msin=my.ms msin.datacolumn=DATA msout=. msout.datacolumn=DATA steps=[bl] \
    bl.type=PythonDPPP bl.python.module=blcal bl.python.class=BlCal bl.mode=apply bl.solutions=bl.h5

Parset keys (prefix bl.):
  mode         solve or apply (required)
  solutions    HDF5 file with the factors (written in mode=solve, read in mode=apply; required)
  modelcolumn  name of the extra data buffer with the model, default MODEL_DATA (mode=solve; it must
               also be given in msin.extradatacolumns)
  solint       time slots per solution, default 0 (one solution for the whole observation)
  nchan        channels per solution, default 0 (one solution for all channels)
  minsnr       factors with |B - 1| < minsnr * sigma are set to 1, default 3
  maxdev       factors with |B - 1| > maxdev are set to 1, default 0.1
The data weights are not changed in mode=apply (|B|^2 differs from 1 by about a percent at most).
"""
import os
import sys

import numpy as np
import tables

try:
    from dp3 import Fields
    from dp3.pydp3 import Step
except ImportError:  # outside DP3 (reading and plotting solutions) the step class is not needed
    Fields = None
    Step = object


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


def parallel_hand_indices(n_correlations):
    """Indices of the parallel hands (XX, YY or RR, LL) for 4, 2 or 1 correlations."""
    return {4: [0, 3], 2: [0, 1], 1: [0]}[n_correlations]


def write_solutions(filename, gain, raw, sigma, weight, valid, antenna_names, antenna_positions, ant1, ant2,
                    chan_freq, chan_block, time_start, time_end, attrs):
    """
    Write baseline-based solutions to HDF5.

    Arrays (n_time blocks, n_baselines, n_freq blocks, n_pol parallel hands):
      gain    applied complex factor (1 where not valid)
      raw     solved complex factor before the safeguards
      sigma   formal error of the factor
      weight  sum of w |model|^2
      valid   True where the factor is applied
    Axes: ant1, ant2 (indices into antenna_names), time_start/time_end per time block (MS TIME, s),
    chan_freq and chan_block (solution block of every channel).
    """
    if os.path.exists(filename):
        os.remove(filename)
    with tables.open_file(filename, mode='w', title='facetselfcal baseline-based calibration') as h:
        for name, value in [('gain', gain), ('raw', raw), ('sigma', sigma), ('weight', weight),
                            ('valid', valid), ('ant1', ant1), ('ant2', ant2),
                            ('antenna_positions', antenna_positions), ('chan_freq', chan_freq),
                            ('chan_block', chan_block), ('time_start', time_start), ('time_end', time_end)]:
            h.create_array('/', name, np.asarray(value))
        h.create_array('/', 'antenna_names', np.array([str(n) for n in antenna_names], dtype='S'))
        for key, value in attrs.items():
            h.root._v_attrs[key] = value


def read_solutions(filename):
    """Read a file written by write_solutions into a dict (antenna_names decoded to str)."""
    with tables.open_file(filename) as h:
        sol = {node.name: node.read() for node in h.list_nodes('/')}
        attrs = {key: h.root._v_attrs[key] for key in h.root._v_attrs._f_list()}
    sol['antenna_names'] = [n.decode() if isinstance(n, bytes) else str(n) for n in sol['antenna_names']]
    sol['attrs'] = attrs
    return sol


class BlCal(Step):
    """Solve (mode=solve) or apply (mode=apply) baseline-based complex factors, see the module docstring."""

    def __init__(self, parset, prefix):
        super().__init__()
        self.mode = get_parset_value(parset, prefix + "mode", "").lower()
        if self.mode not in ('solve', 'apply'):
            sys.exit("BlCal: mode should be solve or apply, not '" + self.mode + "'")
        self.solutions = get_parset_value(parset, prefix + "solutions", "")
        if not self.solutions:
            sys.exit("BlCal: solutions (HDF5 file) is required")
        self.modelcolumn = get_parset_value(parset, prefix + "modelcolumn", "MODEL_DATA")
        self.solint = get_parset_value(parset, prefix + "solint", 0, kind='int')
        self.nchan_sol = get_parset_value(parset, prefix + "nchan", 0, kind='int')
        self.minsnr = float(get_parset_value(parset, prefix + "minsnr", "3"))
        self.maxdev = float(get_parset_value(parset, prefix + "maxdev", "0.1"))
        self.n_slot = 0

    def get_required_fields(self):
        if self.mode == 'solve':
            return Fields.DATA | Fields.FLAGS | Fields.WEIGHTS
        return Fields.DATA

    def get_provided_fields(self):
        # solve: the data pass through unchanged, providing nothing keeps DP3 from writing them back
        return Fields() if self.mode == 'solve' else Fields.DATA

    def update_info(self, dpinfo):
        super().update_info(dpinfo)
        self.ncorr = dpinfo.n_correlations
        self.pp = parallel_hand_indices(self.ncorr)
        self.ant_names = list(dpinfo.antenna_names)
        self.ant1 = np.array(dpinfo.first_antenna_indices)
        self.ant2 = np.array(dpinfo.second_antenna_indices)
        self.freqs = np.array(dpinfo.channel_frequencies)
        nchan = len(self.freqs)
        if self.mode == 'solve':
            self.ant_positions = np.array(dpinfo.antenna_positions)
            per_block = nchan if self.nchan_sol <= 0 else self.nchan_sol
            self.chan_block = np.arange(nchan) // per_block
            self.n_fblock = self.chan_block[-1] + 1
            self.block_start = np.arange(0, nchan, per_block)
            self.n_tblock = 1 if self.solint <= 0 else int(np.ceil(dpinfo.n_times / self.solint))
            shape = (self.n_tblock, len(self.ant1), self.n_fblock, len(self.pp))
            self.num = np.zeros(shape, dtype=np.complex128)
            self.den = np.zeros(shape)
            self.var = np.zeros(shape)
            self.t_start = np.full(self.n_tblock, np.inf)
            self.t_end = np.full(self.n_tblock, -np.inf)
        else:
            self.setup_apply()

    def setup_apply(self):
        """Map the solutions onto the baselines and channels of this run."""
        sol = read_solutions(self.solutions)
        gain = sol['gain']  # (time, baseline, freq block, pol)
        if gain.shape[3] != len(self.pp):
            sys.exit("BlCal: the solutions have %d parallel hands, the data %d" % (gain.shape[3], len(self.pp)))
        name_to_index = {n: i for i, n in enumerate(sol['antenna_names'])}
        key = {(sol['antenna_names'][a], sol['antenna_names'][b]): k
               for k, (a, b) in enumerate(zip(sol['ant1'], sol['ant2']))}
        self.bl_map = np.full(len(self.ant1), -1)
        n_missing = 0
        for k, (a, b) in enumerate(zip(self.ant1, self.ant2)):
            na, nb = self.ant_names[a], self.ant_names[b]
            if (na, nb) in key:
                self.bl_map[k] = key[(na, nb)]
            elif a != b and na in name_to_index and nb in name_to_index:
                n_missing += 1
        if n_missing:
            print('BlCal: %d baselines have no solution, they are not corrected' % n_missing)
        # channel -> solution block, via the nearest solved channel frequency
        idx = np.abs(sol['chan_freq'][None, :] - self.freqs[:, None]).argmin(axis=1)
        self.apply_chan_block = sol['chan_block'][idx]
        self.time_start = sol['time_start']
        # factor 1 for baselines without a solution
        ones = np.ones((gain.shape[0], 1, gain.shape[2], gain.shape[3]), dtype=gain.dtype)
        self.gain = np.concatenate([gain, ones], axis=1)  # index -1 -> ones
        print('BlCal: applying %s (%d of %d factors not 1)' % (self.solutions, int(np.sum(sol['valid'])),
                                                               sol['valid'].size))

    def process(self, dpbuffer):
        if self.mode == 'solve':
            self.accumulate(dpbuffer)
        else:
            self.apply(dpbuffer)
        self.n_slot += 1
        self.get_next_step().process(dpbuffer)

    def accumulate(self, dpbuffer):
        tb = 0 if self.solint <= 0 else min(self.n_slot // self.solint, self.n_tblock - 1)
        time = dpbuffer.get_time()
        self.t_start[tb] = min(self.t_start[tb], time)
        self.t_end[tb] = max(self.t_end[tb], time)
        data = np.array(dpbuffer.get_data(), copy=False)[:, :, self.pp]
        model = np.array(dpbuffer.get_data(self.modelcolumn), copy=False)[:, :, self.pp]
        flags = np.array(dpbuffer.get_flags(), copy=False)[:, :, self.pp]
        weights = np.array(dpbuffer.get_weights(), copy=False)[:, :, self.pp]
        ok = ~flags & (weights > 0) & np.isfinite(data) & np.isfinite(model)
        ok &= (self.ant1 != self.ant2)[:, None, None]
        w = np.where(ok, weights, 0.)
        d = np.where(ok, data, 0.)
        m = np.where(ok, model, 0.)
        r = d - m
        add = lambda x: np.add.reduceat(x, self.block_start, axis=1)  # sum over the channels of a block
        self.num[tb] += add(w * d * np.conj(m))
        self.den[tb] += add(w * np.abs(m) ** 2)
        self.var[tb] += add(w ** 2 * np.abs(m) ** 2 * np.abs(r) ** 2)

    def apply(self, dpbuffer):
        time = dpbuffer.get_time()
        tb = int(np.clip(np.searchsorted(self.time_start, time + 1e-3, side='right') - 1, 0, len(self.time_start) - 1))
        g = self.gain[tb][self.bl_map][:, self.apply_chan_block, :]  # (baseline, channel, parallel hand)
        data = np.array(dpbuffer.get_data(), copy=False)
        if self.ncorr == 4:
            cross = np.sqrt(g[:, :, 0] * g[:, :, 1])
            data[:, :, 0] /= g[:, :, 0]
            data[:, :, 1] /= cross
            data[:, :, 2] /= np.conj(cross)
            data[:, :, 3] /= g[:, :, 1]
        else:
            data[:, :, :] /= g

    def finish(self):
        if self.mode == 'solve':
            self.solve_and_write()
        self.get_next_step().finish()

    def solve_and_write(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            raw = self.num / self.den
            sigma = np.sqrt(self.var) / self.den
        dev = np.abs(raw - 1.)
        solved = (self.den > 0) & np.isfinite(raw) & np.isfinite(sigma)
        significant = solved & (dev >= self.minsnr * sigma)
        valid = significant & (dev <= self.maxdev)
        gain = np.where(valid, raw, 1. + 0j)
        write_solutions(self.solutions, gain, np.where(solved, raw, np.nan), np.where(solved, sigma, np.nan),
                        self.den, valid, self.ant_names, self.ant_positions, self.ant1, self.ant2, self.freqs,
                        self.chan_block, self.t_start, self.t_end,
                        {'solint': self.solint, 'nchan': self.nchan_sol, 'minsnr': self.minsnr,
                         'maxdev': self.maxdev})
        n = int(solved.sum())
        print('BlCal: %d factors solved, %d applied, %d below %.1f sigma, %d above the maximum deviation %.3f'
              % (n, int(valid.sum()), int((solved & ~significant).sum()), self.minsnr,
                 int((significant & (dev > self.maxdev)).sum()), self.maxdev))
        if valid.any():
            print('BlCal: median |B-1| of the applied factors: amplitude %.3f%%, phase %.3f deg'
                  % (100 * np.median(np.abs(np.abs(gain[valid]) - 1)), np.degrees(np.median(np.abs(np.angle(gain[valid]))))))
        print('BlCal: solutions written to ' + self.solutions)


def plot_solutions(filename, outfile):
    """
    Plot the baseline-based factors (amplitude deviation in % and phase in deg) against baseline length,
    per parallel hand; factors that are not applied are shown in grey.

    Parameters
    ----------
    filename : str
        HDF5 solutions file written by BlCal (mode=solve).
    outfile : str
        Output image (png).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sol = read_solutions(filename)
    pos = sol['antenna_positions']
    length = np.linalg.norm(pos[sol['ant1']] - pos[sol['ant2']], axis=1) / 1e3  # km
    raw, valid = sol['raw'], sol['valid']
    npol = raw.shape[3]
    labels = ['XX/RR', 'YY/LL'] if npol == 2 else ['parallel hand %d' % p for p in range(npol)]
    fig, ax = plt.subplots(2, npol, figsize=(6 * npol, 8), squeeze=False)
    for p in range(npol):
        r = raw[..., p]
        v = valid[..., p]
        L = np.broadcast_to(length[None, :, None], r.shape)
        for row, value, unit in [(0, 100 * (np.abs(r) - 1), 'amplitude - 1 [%]'), (1, np.degrees(np.angle(r)), 'phase [deg]')]:
            a = ax[row, p]
            a.scatter(L[~v], value[~v], s=4, c='0.7', label='not applied')
            a.scatter(L[v], value[v], s=6, c='C0', label='applied')
            a.axhline(0, c='k', lw=0.5)
            a.set_xlabel('baseline length [km]')
            a.set_ylabel(unit)
            a.set_title(labels[p])
            if row == 0 and p == 0:
                a.legend(loc='upper right')
    fig.suptitle(os.path.basename(filename))
    fig.tight_layout()
    fig.savefig(outfile, dpi=80)
    plt.close(fig)
