"""
This script is used to derive an S/N selection score by using an h5parm with scalarphasediff solutions from facetselfcal.
The procedure and rational is described in Section 3.3 of de Jong et al. (2024)
"""

author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"
__all__ = ['GetSolint']

import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circstd, circvar
from glob import glob
import csv
import sys
from argparse import ArgumentParser
from typing import Union
from numpy.random import normal

try:
    from .selfcal_selection import parse_source_from_h5
except ImportError:
    from selfcal_selection import parse_source_from_h5


def make_utf8(inp):
    """
    Convert input to utf8 instead of bytes

    :param inp: string input
    :return: input in utf-8 format
    """

    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def rad_to_degree(inp):
    """
    Check if radians and convert to degree

    :param inp: two coordinates (RA, DEC)
    :return: output in degrees
    """

    try:
        if abs(inp[0]) < np.pi and abs(inp[1]) < np.pi:
            return inp * 360 / 2 / np.pi % 360
        else:
            return inp
    except ValueError: # Sorry for the ugly code..
        if abs(inp[0][0]) < np.pi and abs(inp[0][1]) < np.pi:
            return inp[0] * 360 / 2 / np.pi % 360
        else:
            return inp[0]


class GetSolint:
    def __init__(self, h5: str = None, optimal_score: float = 0.5, ref_solint: float = 10., station: str = None):
        """
        Get a score based on the phase difference between XX and YY. This reflects the noise in the observation.
        From this score we can determine an optimal solution interval, by fitting a wrapped normal distribution.

        See:
        - https://en.wikipedia.org/wiki/Wrapped_normal_distribution
        - https://en.wikipedia.org/wiki/Yamartino_method
        - https://en.wikipedia.org/wiki/Directional_statistics

        :param h5: h5parm
        :param optimal_score: score to fit solution interval
        :param ref_solint: reference solution interval in minutes
        :param station: station name
        """

        self.h5 = h5
        self.optimal_score = optimal_score
        self.ref_solint = ref_solint
        self.cstd = 0
        self.C = None
        self.station = station
        self.limit = np.pi

    def plot_C(self, title: str = None, saveas: str = None, extrapoints: Union[list, tuple] = None):
        """
        Plot circstd score in function of solint for given C
        """

        plt.figure(figsize=(10, 7), dpi=120)
        # normal_sigmas = np.array([n / 1000 for n in range(1, 10000)])/100
        # values = [circstd(normal(0, n, 300)) for n in normal_sigmas]
        # x = (self.C * self.limit ** 2) / (np.array(normal_sigmas) ** 2)
        # plt.plot(x, values, alpha=0.5)

        bestsolint = self.best_solint
        solints = np.array(range(1, int(max(bestsolint * 200, self.ref_solint * 150)))) / 100
        y = [self.theoretical_curve(float(t)) for t in solints if self.theoretical_curve(float(t))]
        mask = [False if v>np.pi else True for v in y]
        plt.plot(np.array(solints)[mask], np.array(y)[mask],
                 color='#2ca02c', linewidth=2, linestyle='--')
        plt.plot([0, 1000], [1.75, 1.75], linestyle='dotted', linewidth=2, color='black')
        plt.scatter([self.ref_solint], [self.cstd], label=f'Solint={int(round(self.ref_solint, 0))}min',
                    s=250, marker='X', edgecolor='darkblue', zorder=5, alpha=0.8, color='black')
        plt.scatter([bestsolint], [self.optimal_score], color='#d62728', label=f'Solint={round(self.best_solint,2)}min',
                    s=250, marker='*', edgecolor='black', zorder=5, alpha=0.9)
        if extrapoints is not None:
            plt.scatter(extrapoints[0], extrapoints[1], color='orange', label='Other Measurements',
                s=80, marker='o', edgecolor='black', zorder=5, alpha=0.7)
        plt.xlim(0, max(bestsolint * 1.5, self.ref_solint * 1.5))
        plt.xlabel("Solint (min)", fontsize=14)
        plt.ylabel(r"$\sigma_{c}$ (rad)", fontsize=16)

        plt.xticks(fontsize=13)  # Setting the font size for x-ticks
        plt.yticks(fontsize=13)  # Setting the font size for y-ticks

        plt.legend(frameon=True, loc='upper right', fontsize=12, fancybox=True, shadow=True)
        if title is not None:
            plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        if saveas is not None:
            plt.savefig(saveas)
        else:
            plt.show()

        return self

    def _get_circvar(self, cst: float = None):
        """
        Get circvar score

        return: circular variance
        """

        if cst >= self.limit ** 2:
            return 999 # replacement for infinity
        else:
            return - np.log(1 - cst / (self.limit ** 2))

    @property
    def _get_C(self):
        """
        Get constant defining the normal circular distribution

        :return: C
        """

        if self.cstd == 0:
            self.get_phasediff_score(station=self.station)
        return self._get_circvar(self.cstd ** 2) * np.sqrt(self.ref_solint)

    def get_phasediff_score(self, station: str = None):
        """
        Calculate score for phasediff

        :return: circular standard deviation score
        """

        H = tables.open_file(self.h5)

        stations = [make_utf8(s) for s in list(H.root.sol000.antenna[:]['name'])]

        if station is None or station == 'international':
            stations_idx = [stations.index(stion) for stion in stations if
                            ('DE' == stion[0:2]) |
                            ('PL' == stion[0:2]) |
                            ('FR' == stion[0:2]) |
                            ('SE' == stion[0:2]) |
                            ('UK' == stion[0:2]) ]
        elif station == 'all':
            stations_idx = [stations.index(stion) for stion in stations]
        else:
            stations_idx = [stations.index(station)]

        axes = str(H.root.sol000.phase000.val.attrs["AXES"]).replace("b'", '').replace("'", '').split(',')
        axes_idx = sorted({ax: axes.index(ax) for ax in axes}.items(), key=lambda x: x[1], reverse=True)

        phase = H.root.sol000.phase000.val[:] * H.root.sol000.phase000.weight[:]
        H.close()

        phasemod = phase % (2 * np.pi)

        for ax in axes_idx:
            if ax[0] == 'pol':  # YX should be zero
                phasemod = phasemod.take(indices=0, axis=ax[1])
            elif ax[0] == 'dir':  # there should just be one direction
                if phasemod.shape[ax[1]] == 1:
                    phasemod = phasemod.take(indices=0, axis=ax[1])
                else:
                    sys.exit('ERROR: This solution file should only contain one direction, but it has ' +
                             str(phasemod.shape[ax[1]]) + ' directions')
            elif ax[0] == 'freq':  # faraday corrected
                if phasemod.shape[ax[1]] == 1:
                    print("WARNING: only 1 frequency --> Skip frequency diff for Faraday correction (score will be less accurate)")
                else:
                    phasemod = np.diff(phasemod, axis=ax[1])
            elif ax[0] == 'ant':  # take only international stations
                phasemod = phasemod.take(indices=stations_idx, axis=ax[1])

        phasemod[phasemod == 0] = np.nan

        self.cstd = circstd(phasemod, nan_policy='omit')
        self.cvar = circvar(phasemod, nan_policy='omit')

        return circstd(phasemod, nan_policy='omit')

    @property
    def best_solint(self):
        """
        Get optimal solution interval from phasediff, given C

        :return: value corresponding with increase solution interval
        """

        if self.cstd == 0:
            self.get_phasediff_score(station=self.station)
        self.C = self._get_C
        optimal_cirvar = self.optimal_score ** 2
        return (self.C / (self._get_circvar(optimal_cirvar)))**2

    def theoretical_curve(self, t):
        """
        Theoretical curve based on circ statistics
        :return: circular std
        """

        if self.C is None:
            self.C = self._get_C
        return self.limit * np.sqrt(1 - np.exp(-(self.C / np.sqrt(t))))


def generate_csv(h5s: list = None, ref_solint: int = 10, optimal_score: float = 1.8, make_plot: bool = True):
    """
    Generate CSV file with phasediff score

    Args:
        h5s: Input h5parms
        ref_solint: Reference solution interval
        optimal_score: Optimal phasediff score
        make_plot: Make phasediff plot
    """

    h5s = h5s
    if len(h5s)==1 and ' ' in h5s[0]:
        h5s = h5s[0].split(" ")
    elif h5s is None:
        h5s = glob("P*_phasediff/phasediff0*.h5")

    with open('phasediff_output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["source", "spd_score", "spd_score_all", "best_solint", 'RA', 'DEC'])
        for h5 in h5s:
            S = GetSolint(h5, optimal_score, ref_solint)
            std_all = S.get_phasediff_score(station='all')
            std_int = S.get_phasediff_score(station='international')
            solint = S.best_solint
            H = tables.open_file(h5)
            dir = rad_to_degree(H.root.sol000.source[:]['dir'])

            writer.writerow([parse_source_from_h5(h5), std_int, std_all, solint, dir[0], dir[1]])
            if make_plot:
                S.plot_C(saveas='phasediff.png')
            H.close()

    # sort output
    df = pd.read_csv('phasediff_output.csv').sort_values(by='spd_score')
    df.to_csv('phasediff_output.csv', index=False)



def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Determine phasediff scores")
    parser.add_argument('--h5', nargs='+', help='selfcal phasediff solutions', default=None)
    parser.add_argument('--make_plot', action='store_true', help='make phasediff plot')
    parser.add_argument('--optimal_score', help='optimal score between 0 and pi', default=1.8, type=float)
    return parser.parse_args()


def main():

    args = parse_args()
    generate_csv(h5s = args.h5, optimal_score = args.optimal_score)


if __name__ == '__main__':
    main()
