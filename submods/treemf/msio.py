"""
Measurement Set access for DP3_Python_treeMF, with DP3's DDECal data semantics:
- datause=single: d = (XX + YY) / 2, m likewise, w = w_XX + w_YY; d and m are scaled by sqrt(w)
- a sample is flagged if any correlation is flagged or not finite (data or model);
  autocorrelations are not used
- uvlambdamin/uvlambdamax per channel (strict inequalities), uvmmin/uvmmax per baseline
- missing time slots are treated as fully flagged
Phase-difference mode (pdiff 'unit' or 'product', for the rm models): y = RR conj(LL) per
visibility from the data column (circular, or linear converted as facetselfcal's polconv does),
normalised to unit modulus ('unit', as facetselfcal's DATA_CIRCULAR_PHASEDIFF column) or kept
('product': the amplitude weights the visibility); d = sqrt(w) y, m = sqrt(w) (no model column:
the source structure cancels in RR conj(LL)); w = w_RR + w_LL.
"""
import numpy as np
from casacore.tables import table

C_LIGHT = 299792458.0


class MsInfo:
    """Metadata of the input MS and the time-slot grid."""

    def __init__(self, ms, startchan=0, nchan=0, starttimeslot=0, ntimes=0):
        self.ms = ms
        ant = table(ms + "/ANTENNA", ack=False)
        self.names = list(ant.getcol("NAME"))
        self.pos = ant.getcol("POSITION")
        ant.close()
        spw = table(ms + "/SPECTRAL_WINDOW", ack=False)
        nu = spw.getcol("CHAN_FREQ")[0]
        spw.close()
        nchan = nchan if nchan > 0 else len(nu) - startchan
        self.chan = slice(startchan, startchan + nchan)
        self.nu = nu[self.chan]
        self.pdir = table(ms + "/FIELD", ack=False).getcol("PHASE_DIR")[0, 0]
        self.corr_type = list(table(ms + "/POLARIZATION", ack=False).getcol("CORR_TYPE")[0])
        t = table(ms, ack=False)
        tm = t.getcol("TIME")
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        self.dt = float(t.getcell("INTERVAL", 0))
        self.ncorr = t.getcell("DATA", 0).shape[1]
        self.colnames = t.colnames()
        t.close()
        cross = a1 != a2
        used = np.unique(np.concatenate([a1[cross], a2[cross]]))
        self.used = used  # antennas in the cross-correlations, MS order
        t0 = tm.min()
        slot = np.rint((tm - t0) / self.dt).astype(int)
        nslot = slot.max() + 1
        self.first_slot = starttimeslot
        self.nslot = nslot - starttimeslot if ntimes <= 0 else min(ntimes, nslot - starttimeslot)
        self.t0 = t0 + starttimeslot * self.dt  # centre of the first used slot
        self.slot = slot - starttimeslot
        self.sorted = bool(np.all(np.diff(tm) >= 0))

    def rows(self, s0, s1):
        """Row numbers of time slots [s0, s1) (relative to the first used slot)."""
        return np.flatnonzero((self.slot >= s0) & (self.slot < s1))


def phasediff(data, corr_type, amp="unit", source="rrll"):
    """
    RR conj(LL) per visibility (nrow, nchan) from 4-correlation data, circular (CORR_TYPE 5-8)
    or linear (9-12, converted as submods/polconv.py lin2circ). With source 'column' the data
    column is already the phase difference (facetselfcal's DATA_CIRCULAR_PHASEDIFF, in
    correlation 0), and only its phase is used.
    """
    if source == "column":
        y = data[..., 0]
        a = np.abs(y)
        return np.where(a > 0, y / np.where(a > 0, a, 1.0), 0.0)
    if len(corr_type) != 4:
        raise ValueError(f"phase-difference mode needs 4 correlations, got CORR_TYPE {corr_type}")
    if list(corr_type) == [5, 6, 7, 8]:
        rr, ll = data[..., 0], data[..., 3]
    elif list(corr_type) == [9, 10, 11, 12]:
        rr = 0.5 * (data[..., 0] - 1j * data[..., 1] + 1j * data[..., 2] + data[..., 3])
        ll = 0.5 * (data[..., 0] + 1j * data[..., 1] - 1j * data[..., 2] + data[..., 3])
    else:
        raise ValueError(f"unsupported CORR_TYPE {corr_type}")
    y = rr * np.conj(ll)
    if amp == "unit":
        a = np.abs(y)
        y = np.where(a > 0, y / np.where(a > 0, a, 1.0), 0.0)
    return y


def read_slots(info, s0, s1, datacolumn, modelcolumn, weightcolumn, uvlmin=0.0, uvlmax=np.inf,
               uvmmin=0.0, uvmmax=np.inf, pdiff=None, pdiff_source="rrll"):
    """
    Data of time slots [s0, s1), one dict per slot (None for a missing slot) with
    a1, a2, d, m (sqrt(w)-scaled, (nbl, nchan)) and w. With pdiff ('unit' or 'product') d is the
    phase difference RR conj(LL) and m = 1 (see the module docstring); the model column is not
    read.
    """
    rows = info.rows(s0, s1)
    out = [None] * (s1 - s0)
    if len(rows) == 0:
        return out
    t = table(info.ms, ack=False)
    if info.sorted and rows[-1] - rows[0] + 1 == len(rows):
        sel, r0, nr = t, int(rows[0]), len(rows)
    else:
        sel, r0, nr = t.selectrows(rows), 0, len(rows)

    def col(name):
        return sel.getcol(name, startrow=r0, nrow=nr)

    a1, a2, uvw = col("ANTENNA1"), col("ANTENNA2"), col("UVW")
    ch = info.chan
    data, flag = col(datacolumn)[:, ch], col("FLAG")[:, ch]
    model = None if pdiff else col(modelcolumn)[:, ch]
    if weightcolumn in info.colnames:
        wspec = col(weightcolumn)[:, ch]
    else:  # as DP3: WEIGHT copied to all channels, WEIGHT == 0 set to 1
        wt = col("WEIGHT")
        wt = np.where(wt == 0, 1.0, wt)
        wspec = np.repeat(wt[:, None, :], data.shape[1], axis=1)
    t.close()
    ip = [0, info.ncorr - 1] if info.ncorr > 1 else [0]
    bad = flag.any(axis=2) | ~np.isfinite(data).all(axis=2)
    if pdiff:
        d = phasediff(data, info.corr_type, pdiff, pdiff_source)
        bad |= ~(np.abs(d) > 0)
        m = np.ones(d.shape)
    else:
        bad |= ~np.isfinite(model).all(axis=2)
        d = data[..., ip].mean(axis=2)
        m = model[..., ip].mean(axis=2)
    w = np.where(bad, 0.0, wspec[..., ip].sum(axis=2))
    bl = np.hypot(uvw[:, 0], uvw[:, 1])
    uvl = bl[:, None] * info.nu[None, :] / C_LIGHT
    w[(uvl < uvlmin) | (uvl > uvlmax)] = 0.0
    w[(bl < uvmmin) | (bl > uvmmax)] = 0.0
    w[a1 == a2] = 0.0
    sw = np.sqrt(w)
    ok = w > 0
    d = np.where(ok, d, 0.0) * sw
    m = np.where(ok, m, 0.0) * sw
    slot = info.slot[rows]
    for s in np.unique(slot):
        r = (slot == s) & (a1 != a2)
        out[s - s0] = dict(a1=a1[r], a2=a2[r], d=d[r], m=m[r], w=w[r])
    return out
