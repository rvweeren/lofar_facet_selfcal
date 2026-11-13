#!/usr/bin/env python
"""Per-antenna plot of an H5parm that has one time slot and one frequency channel.

losoto's PLOT operation needs more than one point per antenna, so it cannot plot
such solutions. This plots every soltab as value versus antenna instead.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tables  # noqa: E402


def _read_soltab(soltab):
    """Return values and weights as (ant, pol, dir) arrays plus the pol and dir labels.

    Every other axis (time, freq) is reduced to its first element; this plotter
    is meant for soltabs where those axes have length one.
    """
    axes = soltab.val.attrs["AXES"]
    if isinstance(axes, bytes):
        axes = axes.decode()
    axes = axes.split(",")
    val = np.asarray(soltab.val[:], dtype=float)
    weight = np.asarray(soltab.weight[:], dtype=float)
    val[weight == 0] = np.nan

    index = tuple(slice(None) if ax in ("ant", "pol", "dir") else 0 for ax in axes)
    val = val[index]
    kept = [ax for ax in axes if ax in ("ant", "pol", "dir")]
    for ax in ("pol", "dir"):
        if ax not in kept:
            val = val[..., np.newaxis]
            kept.append(ax)
    val = np.transpose(val, [kept.index(ax) for ax in ("ant", "pol", "dir")])

    def labels(name, n):
        if name in soltab._v_children:
            return [x.decode() if isinstance(x, bytes) else str(x) for x in soltab._f_get_child(name)[:]]
        return [""] * n

    return val, labels("pol", val.shape[1]), labels("dir", val.shape[2])


def plot_single_solution(h5file, outfile, refant=None):
    """Plot each soltab in sol000 of ``h5file`` as solution value versus antenna.

    Parameters
    ----------
    h5file : str
        H5parm with a single time slot and a single frequency channel.
    outfile : str
        Output PNG. With more than one direction, one file per direction is
        written with ``_dir<name>`` inserted before the extension.
    refant : str, optional
        Reference antenna for phase soltabs. Phases are not referenced when it
        is None, not in the H5parm, or flagged.

    Returns
    -------
    list of str
        The PNG files written.
    """
    with tables.open_file(h5file, mode="r") as H:
        soltabs = []
        for name in sorted(H.root.sol000._v_groups.keys()):
            soltab = H.root.sol000._f_get_child(name)
            soltype = soltab._v_attrs["TITLE"] if "TITLE" in soltab._v_attrs else ""
            if isinstance(soltype, bytes):
                soltype = soltype.decode()
            soltype = soltype or name.rstrip("0123456789")
            antennas = [a.decode() if isinstance(a, bytes) else str(a) for a in soltab.ant[:]]
            val, pols, dirs = _read_soltab(soltab)
            soltabs.append((name, soltype, antennas, val, pols, dirs))

    if not soltabs:
        return []

    ndir = max(s[3].shape[2] for s in soltabs)
    stem, ext = os.path.splitext(outfile)
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    written = []
    for idir in range(ndir):
        nant = max(len(s[2]) for s in soltabs)
        fig, axs = plt.subplots(len(soltabs), 1, squeeze=False,
                                figsize=(max(8.0, 0.22 * nant + 2.0), 3.2 * len(soltabs)))
        dirname = ""
        for ax, (name, soltype, antennas, val, pols, dirs) in zip(axs[:, 0], soltabs):
            d = min(idir, val.shape[2] - 1)
            dirname = dirs[d] or dirname
            values = val[:, :, d]
            is_phase = soltype == "phase"
            title = name
            if is_phase:
                if refant in antennas and np.any(np.isfinite(values[antennas.index(refant)])):
                    values = values - values[antennas.index(refant)][np.newaxis, :]
                    title += f" (ref. {refant})"
                values = np.mod(values + np.pi, 2.0 * np.pi) - np.pi
            x = np.arange(len(antennas))
            npol = values.shape[1]
            offsets = (np.arange(npol) - (npol - 1) / 2.0) * min(0.15, 0.6 / max(npol, 1))
            for ipol in range(npol):
                ax.plot(x + offsets[ipol], values[:, ipol], "o", ms=5,
                        label=pols[ipol] if pols[ipol] else None)
            if np.all(np.isnan(values)):
                ax.text(0.5, 0.5, "all solutions flagged", transform=ax.transAxes,
                        ha="center", va="center")
            if is_phase:
                ax.set_ylim(-np.pi, np.pi)
                ax.set_ylabel("phase [rad]")
            elif soltype == "amplitude":
                ax.set_ylabel("amplitude")
            elif soltype == "rotation":
                ax.set_ylabel("rotation [rad]")
            else:
                ax.set_ylabel(soltype)
            ax.set_xticks(x)
            ax.set_xticklabels(antennas, rotation=90, fontsize=7)
            ax.set_xlim(-0.7, len(antennas) - 0.3)
            ax.grid(alpha=0.3)
            ax.set_title(title, fontsize=10)
            if any(pols):
                ax.legend(fontsize=8, loc="best")

        heading = os.path.basename(h5file) + "\n(single time slot, single channel)"
        if ndir > 1 and dirname:
            heading += f"  dir {dirname}"
        fig.suptitle(heading, fontsize=10)
        fig.tight_layout()
        out = f"{stem}_dir{dirname or idir}{ext}" if ndir > 1 else outfile
        fig.savefig(out, dpi=100)
        plt.close(fig)
        written.append(out)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5file", help="H5parm with one time slot and one frequency channel")
    parser.add_argument("--outfile", default="single_solution.png", help="output PNG")
    parser.add_argument("--refant", default=None, help="reference antenna for phases")
    args = parser.parse_args()
    for out in plot_single_solution(args.h5file, args.outfile, refant=args.refant):
        print(out)


if __name__ == "__main__":
    main()
