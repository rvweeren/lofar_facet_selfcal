#!/usr/bin/env python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tables
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from submods.h5_helpers.h5_info import findrefant_core  # noqa: E402


def phaseplot(phase):
    phase = np.mod(phase + np.pi, 2.0 * np.pi) - np.pi
    return phase


def delay2phase(delay, freq):
    phase = 2.0 * np.pi * delay * freq
    return phase


def TEC2phase(TEC, freq):
    phase = -1.0 * (TEC * 8.44797245e9 / freq)
    return phase


def is_multidir(h5: str = None):
    """
    Check if h5 has multiple directions

    Args:
        h5: h5parm

    Returns: boolean

    """

    with tables.open_file(h5) as H:
        for soltab in H.root.sol000._v_groups.keys():
            dirs = H.root.sol000._f_get_child(soltab).dir[:]
            if len(dirs) > 1:
                return True
    return False


def plot_tecdelayphase(
    h5file: str, outfile: str, telescope: str, plotpoints: bool = False
):
    H = tables.open_file(h5file, mode="r")
    soltabs = list(H.root.sol000._v_children.keys())
    print("Found the following soltabs in your H5 file:", soltabs)

    try:
        delay = H.root.sol000.delay000.val[:]
        antennas = H.root.sol000.delay000.ant[:].tolist()
        times = H.root.sol000.delay000.time[:]
        freq = H.root.sol000.delay000.freq[:][0]
        nodelay = False
    except:
        print("Your solutions contain do NOT contain delay values")
        nodelay = True

    try:
        tec = H.root.sol000.tec000.val[:]
        antennas = H.root.sol000.tec000.ant[:].tolist()
        times = H.root.sol000.tec000.time[:]
        freq = H.root.sol000.tec000.freq[:][0]
        notec = False
    except:
        print("Your solutions contain do NOT contain TEC values")
        notec = True

    try:
        phase = H.root.sol000.phase000.val[:]
        antennas = H.root.sol000.phase000.ant[:].tolist()
        times = H.root.sol000.phase000.time[:]
        freq = H.root.sol000.phase000.freq[:][0]
        containsphase = True
        if notec:
            freq = H.root.sol000.phase000.freq[:]
    except:
        print("Your solutions contain do NOT contain phase values")
        containsphase = False
        pass

    H.close()

    print("Plotting at a frequency of:", freq / 1e6, "MHz")

    times = (times - np.min(times)) / 3600.0  # in hrs since obs start

    ysizefig = float(len(antennas))
    antennas = [x.decode("utf-8") for x in antennas]  # convert to proper string list
    refant = antennas.index(findrefant_core(h5file, telescope=telescope))
    print("--- Using reference antenna:", antennas[refant])

    if containsphase:
        if notec:
            freqidx = int(len(freq) / 2)
            refphase = phase[:, freqidx, refant, 0, 0]
        else:
            if nodelay:
                refphase = phase[:, refant, 0, 0] + TEC2phase(
                    tec[:, refant, 0, 0], freq
                )
            else:
                refphase = (
                    phase[:, refant, 0, 0]
                    + TEC2phase(tec[:, refant, 0, 0], freq)
                    + delay2phase(delay[:, refant, 0, 0], freq)
                )
    else:
        if nodelay:
            refphase = TEC2phase(tec[:, refant, 0, 0], freq)
        else:
            refphase = TEC2phase(tec[:, refant, 0, 0], freq) + delay2phase(
                delay[:, refant, 0, 0], freq
            )

    with tables.open_file(h5file) as H:
        dirs = H.root.sol000.tec000.dir[:]
        N_dir = len(dirs)

    outplotname = outfile
    for direction_id, direction in enumerate(dirs):
        if N_dir > 1:
            outplotname = outfile.replace(
                ".png", "_Dir" + str(direction_id).zfill(2) + ".png"
            )
            print("Creating:", outplotname)

        fig, ax = plt.subplots(
            nrows=len(antennas),
            ncols=1,
            figsize=(9, 1.5 * ysizefig),
            squeeze=True,
            sharex="col",
        )
        figcount = 0

        for antenna_id, antenna in enumerate(antennas):
            # print(figcount)
            if containsphase:
                if notec:
                    phasep = phaseplot(
                        phase[:, freqidx, antenna_id, direction_id, 0] - refphase
                    )
                elif nodelay:
                    phasep = phaseplot(
                        phase[:, antenna_id, direction_id, 0]
                        + TEC2phase(tec[:, antenna_id, direction_id, 0], freq)
                        - refphase
                    )
                else:
                    phasep = phaseplot(
                        phase[:, antenna_id, direction_id, 0]
                        + TEC2phase(tec[:, antenna_id, direction_id, 0], freq)
                        + delay2phase(delay[:, antenna_id, direction_id, 0], freq)
                        - refphase
                    )
            else:
                if nodelay:
                    phasep = phaseplot(
                        TEC2phase(tec[:, antenna_id, direction_id, 0], freq) - refphase
                    )
                else:
                    phasep = phaseplot(
                        TEC2phase(tec[:, antenna_id, direction_id, 0], freq)
                        + delay2phase(delay[:, antenna_id, direction_id, 0], freq)
                        - refphase
                    )

            if plotpoints:
                ax[figcount].plot(times, phasep, ".")
            else:
                ax[figcount].plot(times, phasep)

            ax[figcount].set_ylabel("phase [rad]")
            if figcount == len(antennas) - 1:
                ax[figcount].set_xlabel("time [hr]")
            ax[figcount].set_title(antenna, position=(0.5, 0.75))
            ax[figcount].set_ylim(-np.pi, np.pi)
            figcount += 1

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(pad=1.0)
        plt.savefig(outplotname)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot DPPP tecandphase solutions by combining the TEC and phase offset"
    )
    # imaging options
    parser.add_argument(
        "-h5",
        "--H5file",
        help="H5 solution file, must contain tecandphase solutions",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output figure name (png format)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p", "--plotpoints", help="Plot points instead of lines", action="store_true"
    )
    parser.add_argument("--telescope", help="Telescope name", type=str, required=True)

    # parser.add_argument('-f','--freq', help='Frequnecy at which the phase corrections are plotted', type=float, required=True)
    args = vars(parser.parse_args())
    plot_tecdelayphase(
        h5file=args["H5file"],
        outfile=args["outfile"],
        telescope=args["telescope"],
        plotpoints=args["plotpoints"],
    )


if __name__ == "__main__":
    main()
