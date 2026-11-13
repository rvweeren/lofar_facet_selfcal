"""Plots of the MeerKAT calibrator-based calibration solutions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tables  # noqa: E402

from submods.h5_helpers.solution_qa import _axes, _to_standard  # noqa: E402


def _read(h5, soltab):
    with tables.open_file(h5) as H:
        st = getattr(H.root.sol000, soltab)
        axes = _axes(st)
        val = _to_standard(st.val[:], axes)
        wgt = _to_standard(st.weight[:], axes)
        freq = st.freq[:] if 'freq' in axes else np.array([0.])
        time = st.time[:] if 'time' in axes else np.array([0.])
        pols = [p.decode() for p in st.pol[:]] if 'pol' in axes else ['scalar']
    return np.where(wgt > 0, val, np.nan), freq, time, pols


def plot_spectrum(h5, outpng, title, timeslot=0):
    """
    Amplitude and phase (degrees) versus frequency of all antennas and polarisations, for one time slot
    (bandpass, leakage, cross-hand phase).

    Parameters
    ----------
    h5 : str
        h5parm with amplitude000 and/or phase000.
    outpng : str
        Output png.
    title : str
        Plot title.
    timeslot : int, optional
        Time slot to plot. Default 0.
    """
    with tables.open_file(h5) as H:
        names = [n for n in ('amplitude000', 'phase000') if n in H.root.sol000._v_children]
    fig, axs = plt.subplots(len(names), 1, figsize=(11, 3.5 * len(names)), sharex=True, squeeze=False)
    for ax, name in zip(axs[:, 0], names):
        val, freq, _, pols = _read(h5, name)
        marker = ',' if len(freq) > 200 else '.'
        for p, pol in enumerate(pols):
            for a in range(val.shape[2]):
                y = val[timeslot, :, a, 0, p]
                if name == 'phase000':
                    y = np.degrees(np.angle(np.exp(1j * y)))
                ax.plot(freq / 1e6, y, marker, color='C%d' % p, alpha=0.5, label=pol if a == 0 else None)
        ax.set_ylabel('amplitude' if name == 'amplitude000' else 'phase (deg)')
        ax.legend(loc='upper right', markerscale=10 if marker == ',' else 1)
    axs[-1, 0].set_xlabel('frequency (MHz)')
    axs[0, 0].set_title(title)
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)
    plt.close(fig)


def plot_timeseries(h5, soltab, outpng, title, scale=1., unit=''):
    """
    A solution (frequency-averaged) versus time for all antennas (delays, TEC, gains).

    Parameters
    ----------
    h5 : str
        h5parm.
    soltab : str
        Soltab name.
    outpng : str
        Output png.
    title : str
        Plot title.
    scale : float, optional
        Factor applied to the values (e.g. 1e9 for delays in ns). Default 1.
    unit : str, optional
        Unit label. Default ''.
    """
    val, _, time, pols = _read(h5, soltab)
    if soltab == 'phase000':
        val = np.angle(np.nanmean(np.exp(1j * val), axis=1, keepdims=True))
        scale, unit = 180. / np.pi, 'deg'
    else:
        val = np.nanmean(val, axis=1, keepdims=True)
    t = (time - time[0]) / 3600.
    fig, ax = plt.subplots(figsize=(11, 4))
    for p, pol in enumerate(pols):
        for a in range(val.shape[2]):
            ax.plot(t, val[:, 0, a, 0, p] * scale, '.', ms=2, color='C%d' % p, label=pol if a == 0 else None)
    ax.set_xlabel('time (h)')
    ax.set_ylabel(soltab.replace('000', '') + (f' ({unit})' if unit else ''))
    ax.set_title(title)
    ax.legend(loc='upper right', markerscale=4)
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)
    plt.close(fig)
