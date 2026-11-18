"""
Closure chi^2 of a sky model against the data, as a DP3 Python step.

Antenna-based gains cancel in closure phases (triangles of baselines) and in closure amplitudes
(quadrangles), so a closure chi^2 measures how well the sky model fits the data independently of any
gain solutions. The usual self-calibration chi^2, sum w |d - g_p conj(g_q) m|^2, can also go down when
the gains absorb errors of the model (unmodelled emission, deconvolution artefacts); the closure chi^2
only goes down when the model itself gets better. That makes it a convergence measure for
self-calibration that the gains cannot fool. Errors that do not factor into antenna gains (baseline-based
errors, direction-dependent effects that differ between antennas) also show up in it.

Closure phases (per parallel hand, XX and YY or RR and LL separately):
    B = V_ra V_ab conj(V_rb) for the independent set of triangles through a reference antenna r,
    averaged (bispectrum averaging, which keeps the gain invariance) over solint time slots and nchan
    channels, for the data and for the model. Residual: arg(B_data conj(B_model)), variance from the
    weights and the model amplitudes, var = |B_m|^-2 sum_k |B_m,k|^2 (1/(2 w |m|^2) summed over the three
    baselines of sample k), for weights w = 1/E|noise|^2 of the complex visibility.
Closure amplitudes (per parallel hand):
    C = |V_ra||V_sb| / (|V_rb||V_sa|) for quadrangles through two reference antennas r and s,
    log C_data - log C_model per sample (only samples with model S/N > minsnr on all four baselines),
    inverse-variance averaged over the same blocks.

chi^2 / N is about 1 for a perfect model and weights w = 1/E|noise|^2 (high S/N); it scales with the
weight normalisation, so compare values from the same data. The references are the antennas with the
most unflagged baselines in the first time slot with data.

The step passes the data through unchanged and writes nothing to the MS. The model comes from named
data buffers, summed when there are several (one model column per direction in a direction-dependent
solve). On its own the step needs msin.extradatacolumns=[MODEL_DATA]. Behind a DDECal step it can reuse
the model DDECal read, without extra I/O: DDECal reads its modeldatacolumns itself (they are not in the
buffer in front of it) and passes them on only with ddecal.keepmodel=true, named <ddecal step>.<column>.

Direction-dependent solves: gains that differ between directions do not cancel in the closures, only the
direction-independent part does. Against the summed model the closure chi^2 then also contains the
differences between the direction-dependent gains; it remains a relative measure between cycles of the
same run, not a pure measure of the model.

Examples:
DP3 msin=my.ms msin.extradatacolumns=[MODEL_DATA] msout= steps=[closurechi2] \
    closurechi2.type=PythonDPPP closurechi2.python.module=closurechi2 closurechi2.python.class=ClosureChi2 closurechi2.output=closure.json
DP3 msin=my.ms msout=. steps=[ddecal,closurechi2] ddecal.modeldatacolumns=[MODEL_DATA] ddecal.keepmodel=true ... \
    closurechi2.type=PythonDPPP closurechi2.python.module=closurechi2 closurechi2.python.class=ClosureChi2 \
    closurechi2.modelcolumns=[ddecal.MODEL_DATA] closurechi2.output=closure.json

Parset keys (prefix <step name>., e.g. closurechi2.):
  modelcolumns names of the model data buffers, summed, default [MODEL_DATA]
  output       JSON file for the results, default closurechi2.json
  solint       time slots per averaging block, default 1
  nchan        channels per averaging block, default 1
  minsnr       closure amplitudes: minimum model S/N (|m| sqrt(w)) on each of the four baselines, default 3
  every        use every k-th time slot only (a cheaper estimate), default 1
  label        free text stored in the JSON file (e.g. the self-calibration cycle), default empty
"""
import json
import sys
import time as _time

import numpy as np

try:
    from dp3 import Fields
    from dp3.pydp3 import Step
except ImportError:  # outside DP3 (reading results) the step class is not needed
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


def read_results(filename):
    """Read a JSON file written by ClosureChi2."""
    with open(filename) as f:
        return json.load(f)


class ClosureChi2(Step):
    """Closure-phase and closure-amplitude chi^2 of the model against the data, see the module docstring."""

    def __init__(self, parset, prefix):
        super().__init__()
        names = get_parset_value(parset, prefix + "modelcolumns", "[MODEL_DATA]")
        self.modelcolumns = [n.strip().strip("'\"") for n in names.strip().strip('[]').split(',') if n.strip()]
        if not self.modelcolumns:
            sys.exit("ClosureChi2: modelcolumns is empty")
        self.output = get_parset_value(parset, prefix + "output", "closurechi2.json")
        self.solint = max(1, get_parset_value(parset, prefix + "solint", 1, kind='int'))
        self.nchan_avg = max(1, get_parset_value(parset, prefix + "nchan", 1, kind='int'))
        self.minsnr = float(get_parset_value(parset, prefix + "minsnr", "3"))
        self.label = get_parset_value(parset, prefix + "label", "")
        self.every = max(1, get_parset_value(parset, prefix + "every", 1, kind='int'))
        self.n_slot = 0
        self.ready = False  # triangles and quadrangles are set up at the first time slot with data
        self.error = None  # set when the step switched itself off after an error
        self.t_process = 0.0

    def get_required_fields(self):
        return Fields.DATA | Fields.FLAGS | Fields.WEIGHTS

    def get_provided_fields(self):
        # the data pass through unchanged: providing nothing keeps DP3 from writing anything back
        return Fields()

    def update_info(self, dpinfo):
        super().update_info(dpinfo)
        self.ms_name = dpinfo.ms_name
        self.pp = parallel_hand_indices(dpinfo.n_correlations)
        self.ant_names = list(dpinfo.antenna_names)
        self.n_ant = len(self.ant_names)
        self.ant1 = np.array(dpinfo.first_antenna_indices)
        self.ant2 = np.array(dpinfo.second_antenna_indices)
        self.freqs = np.array(dpinfo.channel_frequencies)
        nchan = len(self.freqs)
        self.chan_start = np.arange(0, nchan, self.nchan_avg)  # reduceat boundaries
        self.n_fblock = len(self.chan_start)
        # baseline lookup in both orientations: index and whether the stored visibility must be conjugated
        self.bl_index = np.full((self.n_ant, self.n_ant), -1)
        self.bl_conj = np.zeros((self.n_ant, self.n_ant), dtype=bool)
        cross = self.ant1 != self.ant2
        for k in np.where(cross)[0]:
            a, b = self.ant1[k], self.ant2[k]
            self.bl_index[a, b] = k
            self.bl_index[b, a] = k
            self.bl_conj[b, a] = True

    def setup_sets(self, flags):
        """Reference antennas from the first slot with data, then the independent triangles and quadrangles."""
        good = ~np.all(flags[:, :, self.pp], axis=(1, 2))
        count = np.zeros(self.n_ant)
        np.add.at(count, self.ant1[good & (self.ant1 != self.ant2)], 1)
        np.add.at(count, self.ant2[good & (self.ant1 != self.ant2)], 1)
        order = np.argsort(-count)
        r, s = order[0], order[1]
        self.refants = [self.ant_names[r], self.ant_names[s]]
        others = [a for a in range(self.n_ant) if a != r]
        # closure phases: triangles (r, a, b), a < b, a, b != r -> (N-1)(N-2)/2
        tri = [(r, a, b) for i, a in enumerate(others) for b in others[i + 1:]]
        tri = [t for t in tri if self.bl_index[t[0], t[1]] >= 0 and self.bl_index[t[1], t[2]] >= 0
               and self.bl_index[t[0], t[2]] >= 0]
        self.tri = np.array(tri, dtype=int).reshape(-1, 3)
        # closure amplitudes: quadrangles (r, s, a, b), C = |V_ra||V_sb| / (|V_rb||V_sa|), a < b, a, b not r, s
        rest = [a for a in range(self.n_ant) if a not in (r, s)]
        quad = [(r, s, a, b) for i, a in enumerate(rest) for b in rest[i + 1:]]
        quad = [q for q in quad if min(self.bl_index[q[0], q[2]], self.bl_index[q[1], q[3]],
                                       self.bl_index[q[0], q[3]], self.bl_index[q[1], q[2]]) >= 0]
        self.quad = np.array(quad, dtype=int).reshape(-1, 4)
        # baseline indices (and conjugation for the phases) of the triangle and quadrangle sides
        r_, a_, b_ = self.tri.T
        self.t_idx = [self.bl_index[r_, a_], self.bl_index[a_, b_], self.bl_index[r_, b_]]
        self.t_conj = [self.bl_conj[r_, a_], self.bl_conj[a_, b_], self.bl_conj[r_, b_]]
        r_, s_, a_, b_ = self.quad.T
        self.q_idx = [self.bl_index[r_, a_], self.bl_index[s_, b_], self.bl_index[r_, b_], self.bl_index[s_, a_]]
        npp = len(self.pp)
        nt, nq = len(self.tri), len(self.quad)
        # per block accumulators (triangle/quadrangle, frequency block, parallel hand)
        self.bd = np.zeros((nt, self.n_fblock, npp), dtype=np.complex128)
        self.bm = np.zeros((nt, self.n_fblock, npp), dtype=np.complex128)
        self.bv = np.zeros((nt, self.n_fblock, npp))
        self.qs = np.zeros((nq, self.n_fblock, npp))
        self.qw = np.zeros((nq, self.n_fblock, npp))
        # totals
        self.cp_chi2 = 0.0
        self.cp_n = 0
        self.ca_chi2 = 0.0
        self.ca_n = 0
        self.cp_chi2_ant = np.zeros(self.n_ant)
        self.cp_n_ant = np.zeros(self.n_ant)
        self.ca_chi2_ant = np.zeros(self.n_ant)
        self.ca_n_ant = np.zeros(self.n_ant)
        self.ready = True

    def oriented(self, arr, k):
        """Side k of the triangles: arr gathered on its baselines, conjugated where the MS stores it reversed."""
        v = arr[self.t_idx[k]]
        c = self.t_conj[k]
        if c.any():
            v[c] = np.conj(v[c])
        return v

    def block_sum(self, x):
        """Sum over the channels of each frequency block (axis 1)."""
        return np.add.reduceat(x, self.chan_start, axis=1)

    def accumulate(self, data, model, flags, weights):
        pp = self.pp
        d = data[:, :, pp]  # copies, complex64
        m = model[:, :, pp]
        w = weights[:, :, pp]
        m2 = m.real ** 2 + m.imag ** 2
        valid = ~flags[:, :, pp] & (w > 0) & (m2 > 0)
        # variance of arg(V) and of log|V|: 1/(2 w |V|^2) for weights w = 1/E|noise|^2 of the complex visibility
        nvar = np.full(d.shape, np.inf, dtype=np.float32)
        nvar[valid] = 0.5 / (w[valid] * m2[valid])
        # --- closure phases (bispectra of data and model, summed per block)
        if len(self.tri):
            rv = nvar[self.t_idx[0]] + nvar[self.t_idx[1]] + nvar[self.t_idx[2]]
            ok = np.isfinite(rv)
            bdat = self.oriented(d, 0) * self.oriented(d, 1) * np.conj(self.oriented(d, 2))
            bmod = self.oriented(m, 0) * self.oriented(m, 1) * np.conj(self.oriented(m, 2))
            bvar = m2[self.t_idx[0]] * m2[self.t_idx[1]] * m2[self.t_idx[2]] * rv
            bdat[~ok] = 0
            bmod[~ok] = 0
            bvar[~ok] = 0
            self.bd += self.block_sum(bdat)
            self.bm += self.block_sum(bmod)
            self.bv += self.block_sum(bvar)
        # --- closure amplitudes (log ratios per sample, inverse-variance summed per block)
        if len(self.quad):
            d2 = d.real ** 2 + d.imag ** 2
            sel = valid & (d2 > 0)
            ldiff = np.zeros(d.shape, dtype=np.float32)  # log|d| - log|m| per baseline
            ldiff[sel] = 0.5 * np.log(d2[sel] / m2[sel])
            snr_ok = sel & (nvar < 0.5 / self.minsnr ** 2)  # |m| sqrt(w) > minsnr
            i0, i1, i2, i3 = self.q_idx
            ok = snr_ok[i0] & snr_ok[i1] & snr_ok[i2] & snr_ok[i3]
            var = nvar[i0] + nvar[i1] + nvar[i2] + nvar[i3]
            iv = np.zeros(var.shape, dtype=np.float32)
            iv[ok] = 1.0 / var[ok]
            lq = ldiff[i0] + ldiff[i1] - ldiff[i2] - ldiff[i3]
            self.qs += self.block_sum(lq * iv)
            self.qw += self.block_sum(iv)

    def close_block(self):
        """Turn the accumulated block into chi^2 contributions and reset the accumulators."""
        if not self.ready:
            return
        # closure phases
        valid = (np.abs(self.bm) > 0) & (self.bv > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.angle(self.bd * np.conj(self.bm))
            var = self.bv / np.abs(self.bm) ** 2
            chi = np.where(valid, res ** 2 / var, 0.0)
        self.cp_chi2 += chi.sum()
        self.cp_n += int(valid.sum())
        per_tri = chi.sum(axis=(1, 2))
        n_tri = valid.sum(axis=(1, 2))
        for col in range(3):
            np.add.at(self.cp_chi2_ant, self.tri[:, col], per_tri)
            np.add.at(self.cp_n_ant, self.tri[:, col], n_tri)
        # closure amplitudes
        valid = self.qw > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = self.qs / self.qw
            chi = np.where(valid, mean ** 2 * self.qw, 0.0)
        self.ca_chi2 += chi.sum()
        self.ca_n += int(valid.sum())
        per_q = chi.sum(axis=(1, 2))
        n_q = valid.sum(axis=(1, 2))
        for col in range(4):
            np.add.at(self.ca_chi2_ant, self.quad[:, col], per_q)
            np.add.at(self.ca_n_ant, self.quad[:, col], n_q)
        for arr in (self.bd, self.bm, self.bv, self.qs, self.qw):
            arr[...] = 0

    def process(self, dpbuffer):
        # a diagnostic: an error here switches the step off, the data always go on to the next step
        if self.error is None:
            t0 = _time.time()
            try:
                flags = np.array(dpbuffer.get_flags(), copy=False)
                if not self.ready and not np.all(flags):
                    self.setup_sets(flags)
                if self.ready and self.n_slot % self.every == 0 and not np.all(flags):
                    data = np.array(dpbuffer.get_data(), copy=False)
                    model = np.array(dpbuffer.get_data(self.modelcolumns[0]), copy=False)
                    for name in self.modelcolumns[1:]:
                        model = model + np.array(dpbuffer.get_data(name), copy=False)
                    weights = np.array(dpbuffer.get_weights(), copy=False)
                    self.accumulate(data, model, flags, weights)
                self.n_slot += 1
                if self.ready and self.n_slot % self.solint == 0:
                    self.close_block()
            except Exception as e:
                self.error = repr(e)
                print('ClosureChi2: switched off after an error: ' + self.error, flush=True)
            self.t_process += _time.time() - t0
        self.get_next_step().process(dpbuffer)

    def finish(self):
        try:
            self.write_results()
        except Exception as e:
            print('ClosureChi2: no results written: ' + repr(e), flush=True)

    def write_results(self):
        if self.error is None and self.ready and self.n_slot % self.solint != 0:
            self.close_block()
        result = {'ms': self.ms_name, 'label': self.label, 'modelcolumns': self.modelcolumns,
                  'solint': self.solint, 'nchan': self.nchan_avg, 'minsnr': self.minsnr, 'every': self.every,
                  'n_slots': self.n_slot, 'time_s': round(self.t_process, 2), 'error': self.error}
        if self.error is not None:
            self.ready = False  # partial sums are not written
        if self.ready:
            def ratio(x, n):
                return float(x / n) if n > 0 else None
            result.update({
                'refants': self.refants, 'n_triangles': int(len(self.tri)), 'n_quadrangles': int(len(self.quad)),
                'closure_phase': {'chi2': float(self.cp_chi2), 'n': self.cp_n,
                                  'chi2_per_n': ratio(self.cp_chi2, self.cp_n)},
                'closure_amplitude': {'chi2': float(self.ca_chi2), 'n': self.ca_n,
                                      'chi2_per_n': ratio(self.ca_chi2, self.ca_n)},
                'per_antenna': {name: {'cp_chi2_per_n': ratio(self.cp_chi2_ant[i], self.cp_n_ant[i]),
                                       'ca_chi2_per_n': ratio(self.ca_chi2_ant[i], self.ca_n_ant[i])}
                                for i, name in enumerate(self.ant_names)}})
        with open(self.output, 'w') as f:
            json.dump(result, f, indent=1)
        if self.ready:
            print('ClosureChi2 %s: closure phase chi2/N = %s (N=%d), closure amplitude chi2/N = %s (N=%d), %.1f s'
                  % (self.ms_name, result['closure_phase']['chi2_per_n'], self.cp_n,
                     result['closure_amplitude']['chi2_per_n'], self.ca_n, self.t_process))
        elif self.error is None:
            print('ClosureChi2 %s: no unflagged data' % self.ms_name)

    def show(self):
        print("\nClosureChi2")
        print("  modelcolumns:  " + ', '.join(self.modelcolumns))
        print("  output:        " + self.output)
        print("  solint:        " + str(self.solint))
        print("  nchan:         " + str(self.nchan_avg))
        print("  minsnr:        " + str(self.minsnr))
        print("  every:         " + str(self.every))
