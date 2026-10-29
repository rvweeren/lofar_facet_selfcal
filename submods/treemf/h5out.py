"""
Write treeMF solutions in the h5parm layout of DP3 DDECal (mode tec+phase+delay, DP3 6.6):
  sol000 (attr h5parm_version)
    antenna   compound (name S16, position float32[3]): all MS antennas
    source    compound (name S128, dir float64[2]): the direction(s), e.g. [MODEL_DATA]
    tec000, phase000, delay000 (attrs TITLE, h5parm_version)
      ant (used antennas, S<maxlen>), dir (S128), freq (float64[1], mean channel frequency),
      time (float64), val (float64, time,ant,dir,freq; attrs AXES, HISTORY000),
      weight (float32, attr AXES)
For the rm models on phase-difference data (write_h5parm_phasediff) the layout is that of the
scalarphasediff solve in facetselfcal (DP3 phaseonly on DATA_CIRCULAR_PHASEDIFF after
makephasediffh5): one soltab phase000 with axes time,freq,ant,dir,pol, pol [XX, YY], the RR - LL
phase 2 RM lambda^2 + offset per channel in XX and 0 in YY.
"""
import datetime

import h5py
import numpy as np


# DP3 (HighFive) writes fixed-length strings of exactly the text's length, typed null-terminated,
# and DP3 applycal does not find antenna names in h5py's default null-padded strings. They are
# written as raw bytes: a conversion to the null-terminated type would drop the last character.
def _str_type(n):
    t = h5py.h5t.C_S1.copy()
    t.set_size(max(n, 1))
    t.set_strpad(h5py.h5t.STR_NULLTERM)
    return t


def _str_attr(obj, name, s):
    b = s.encode() if isinstance(s, str) else bytes(s)
    t = _str_type(len(b))
    a = h5py.h5a.create(obj.id, name.encode(), t, h5py.h5s.create(h5py.h5s.SCALAR))
    a.write(np.array(b, dtype=f"S{max(len(b), 1)}"), mtype=t)  # raw bytes, no conversion


def _str_dataset(group, name, strings, size):
    arr = np.array([s.encode() if isinstance(s, str) else s for s in strings], dtype=f"S{size}")
    t = _str_type(size)
    ds = h5py.h5d.create(group.id, name.encode(), t, h5py.h5s.create_simple(arr.shape))
    ds.write(h5py.h5s.ALL, h5py.h5s.ALL, arr, mtype=t)  # raw bytes, no conversion


def _compound_dataset(group, name, strings, ssize, values, vtype, vshape):
    """Records (name S<ssize> null-terminated, value <vtype>[vshape]) as DP3's antenna/source."""
    vdt = np.dtype((vtype, vshape))
    ndt = np.dtype([("name", f"S{ssize}"), ("v", vdt)])
    tid = h5py.h5t.create(h5py.h5t.COMPOUND, ndt.itemsize)
    tid.insert(b"name", 0, _str_type(ssize))
    tid.insert(b"position" if name == "antenna" else b"dir", ssize,
               h5py.h5t.array_create(h5py.h5t.py_create(np.dtype(vtype)), vshape))
    arr = np.zeros(len(strings), dtype=ndt)
    arr["name"] = [s.encode() for s in strings]
    arr["v"] = values
    ds = h5py.h5d.create(group.id, name.encode(), tid, h5py.h5s.create_simple(arr.shape))
    ds.write(h5py.h5s.ALL, h5py.h5s.ALL, arr, mtype=tid)


def solution_times(t_first, dt, nslot, step):
    """
    Solution time centres as DP3 writes them: start + (k + 0.5) step, start the beginning of
    the first slot, for every step that starts within the data.
    """
    start = t_first - dt / 2
    n = int(np.ceil(nslot * dt / step - 1e-9))
    return start + (np.arange(n) + 0.5) * step


def write_h5parm(fn, names, positions, used, dir_names, dir_radec, times, freq, tec, phase,
                 delay, weight, parset_text):
    """
    Parameters
    ----------
    fn : str
        Output file (overwritten).
    names, positions : list, numpy.ndarray
        All MS antenna names and ITRF positions (m).
    used : numpy.ndarray
        Indices of the antennas in the soltabs (MS order).
    dir_names, dir_radec : list
        Direction names (e.g. '[MODEL_DATA]') and (ra, dec) in rad.
    times : numpy.ndarray
        Solution times.
    freq : float
        Frequency written on the freq axis (Hz).
    tec, phase, delay, weight : numpy.ndarray
        (ntime, nused, ndir) values: TECU, rad, s, and weights.
    parset_text : str
        Sorted 'key=value' lines for HISTORY000.
    """
    hist = (datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            + ": CREATE by DP3_Python_treeMF\nstep ddecal. in parset: \n" + parset_text)
    ant_names = [names[a] for a in used]
    with h5py.File(fn, "w") as f:
        ss = f.create_group("sol000")
        _str_attr(ss, "h5parm_version", "1.0")
        _compound_dataset(ss, "antenna", names, 16, np.asarray(positions, np.float32), "<f4", (3,))
        _compound_dataset(ss, "source", dir_names, 128, np.asarray(dir_radec, np.float64), "<f8", (2,))
        axes = "time,ant,dir,freq"
        for name, title, val in (("tec000", "tec", tec), ("phase000", "phase", phase),
                                 ("delay000", "delay", delay)):
            g = ss.create_group(name)
            _str_attr(g, "TITLE", title)
            _str_attr(g, "h5parm_version", "1.0")
            _str_dataset(g, "ant", ant_names, max(len(n) for n in ant_names))
            _str_dataset(g, "dir", dir_names, 128)
            g.create_dataset("freq", data=np.array([freq], dtype=np.float64))
            g.create_dataset("time", data=np.asarray(times, dtype=np.float64))
            v = g.create_dataset("val", data=np.asarray(val, dtype=np.float64)[..., None])
            _str_attr(v, "AXES", axes)
            _str_attr(v, "HISTORY000", hist)
            w = g.create_dataset("weight", data=np.asarray(weight, dtype=np.float32)[..., None])
            _str_attr(w, "AXES", axes)


def write_h5parm_phasediff(fn, names, positions, used, dir_names, dir_radec, times, freqs, phase,
                           weight, parset_text):
    """
    Parameters
    ----------
    fn, names, positions, used, dir_names, dir_radec, times, parset_text
        As write_h5parm.
    freqs : numpy.ndarray
        Channel frequencies (Hz) of the freq axis.
    phase, weight : numpy.ndarray
        (ntime, nfreq, nused, ndir) RR - LL phase (rad) and weights; written to pol XX, with
        pol YY 0.
    """
    hist = (datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            + ": CREATE by DP3_Python_treeMF\nstep ddecal. in parset: \n" + parset_text)
    ant_names = [names[a] for a in used]
    val = np.zeros(np.shape(phase) + (2,))
    val[..., 0] = phase
    wgt = np.repeat(np.asarray(weight, np.float32)[..., None], 2, axis=-1)
    with h5py.File(fn, "w") as f:
        ss = f.create_group("sol000")
        _str_attr(ss, "h5parm_version", "1.0")
        _compound_dataset(ss, "antenna", names, 16, np.asarray(positions, np.float32), "<f4", (3,))
        _compound_dataset(ss, "source", dir_names, 128, np.asarray(dir_radec, np.float64), "<f8", (2,))
        axes = "time,freq,ant,dir,pol"
        g = ss.create_group("phase000")
        _str_attr(g, "TITLE", "phase")
        _str_attr(g, "h5parm_version", "1.0")
        _str_dataset(g, "ant", ant_names, max(len(n) for n in ant_names))
        _str_dataset(g, "dir", dir_names, 128)
        g.create_dataset("freq", data=np.asarray(freqs, dtype=np.float64))
        _str_dataset(g, "pol", ["XX", "YY"], 2)
        g.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        v = g.create_dataset("val", data=val)
        _str_attr(v, "AXES", axes)
        _str_attr(v, "HISTORY000", hist)
        w = g.create_dataset("weight", data=wgt)
        _str_attr(w, "AXES", axes)
