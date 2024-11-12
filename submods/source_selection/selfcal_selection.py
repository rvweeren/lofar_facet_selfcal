"""
This script is used to select the best self-calibration cycle.
It will return a few plots and a csv with the statistics for each self-calibration cycle.
This is described in Section 3.3 of de Jong et al. (2024)

You can run this script in the folder with your facetselfcal output on 1 source as
python selfcal_quality.py --fits *.fits --h5 merged_selfcal*.h5

Alternatively (for testing), you can also run the script on multiple sources at the same time with
python selfcal_quality.py --parallel --root_folder_parallel <my_path>
where '<my_path>' is the path directed to where all calibrator source folders are located
"""

__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl), Robert Jan Schlimbach (robert-jan.schlimbach@surf.nl)"

import logging
import functools
import re
import csv
from pathlib import Path
from os import sched_getaffinity, system
import sys

from joblib import Parallel, delayed
import tables
from scipy.stats import circstd, linregress
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from astropy.io import fits
import pandas as pd
from typing import Union

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class SelfcalQuality:
    def __init__(self, h5s: list, fitsim: list, station: str):
        """
        Determine quality of selfcal from facetselfcal.py

        :param fitsim: fits images
        :param h5s: h5parm solutions
        :param folder: path to directory where selfcal ran
        :param station: which stations to consider [dutch, remote, international, debug]
        """

        # merged selfcal h5parms
        self.h5s = h5s
        assert len(self.h5s) != 0, "No h5 files given"

        # selfcal images
        self.fitsfiles = fitsim
        assert len(self.fitsfiles) != 0, "No fits files found"

        # select all sources
        sources = []
        for h5 in self.h5s:
            filename = h5.split('/')[-1]
            sources.append(parse_source_from_h5(filename))
        self.sources = set(sources)
        assert len(self.sources) > 0, "No sources found"

        self.main_source = list(self.sources)[0].split("_")[-1]

        # for phase/amp evolution
        self.station = station

        self.station_codes = (
            ('CS',) if self.station == 'dutch' else
            ('RS',) if self.station == 'remote' else
            ('CS', 'RS') if self.station == 'alldutch' else
            ('IE', 'SE', 'PL', 'UK', 'LV', 'DE')  # i.e.: if self.station == 'international'
        )

    # def image_entropy(self, fitsfile: str = None):
    #     """
    #     Calculate entropy of image
    #
    #     :param fitsfile:
    #
    #     :return: image entropy value
    #     """
    #
    #     with fits.open(fitsfile) as f:
    #         image = f[0].data
    #
    #     while image.ndim > 2:
    #         image = image[0]
    #     image = np.sqrt((image - self.minp) / (self.maxp - self.minp)) * 255
    #     image = image.astype(np.uint8)
    #     val = entropy(image, disk(6)).sum()
    #     print(f"Entropy: {val}")
    #     return val

    def filter_stations(self, station_names):
        """Generate indices of filtered stations"""

        if self.station == 'debug':
            return list(range(len(station_names)))

        output_stations = [
            i for i, station_name in enumerate(station_names)
            if any(station_code in station_name for station_code in self.station_codes)
        ]

        return output_stations

    def print_station_names(self, h5):
        """Print station names"""

        with tables.open_file(h5) as H:
            station_names = H.root.sol000.antenna[:]['name']

        stations = map(make_utf8, station_names)

        stations_used = ', '.join([
            station_name for station_name in stations
            if any(station_code in station_name for station_code in self.station_codes)
        ])
        logger.debug(f'Used the following stations: {stations_used}')
        return self

    def get_solution_scores(self, h5_1: str, h5_2: str = None):
        """
        Get solution scores

        :param h5_1: solution file 1
        :param h5_2: solution file 2

        :return: phase_score --> circular std phase difference score
                 amp_score --> std amp difference score
        """

        def extract_data(tables_path):
            with tables.open_file(tables_path) as f:
                axes = make_utf8(f.root.sol000.phase000.val.attrs['AXES']).split(',')
                pols = f.root.sol000.phase000.pol[:] if 'pol' in axes else ['XX']
                amps = (
                    f.root.sol000.amplitude000.val[:]
                    if 'amplitude000' in list(f.root.sol000._v_groups.keys())
                    else np.ones(f.root.sol000.phase000.val.shape)
                )

                return (
                    [make_utf8(station) for station in f.root.sol000.antenna[:]['name']],
                    make_utf8(f.root.sol000.phase000.val.attrs['AXES']).split(','),
                    pols,
                    f.root.sol000.phase000.val[:],
                    f.root.sol000.phase000.weight[:],
                    amps,
                )

        def filter_params(station_indices, axes, *parameters):
            return tuple(
                np.take(param, station_indices, axes)
                for param in parameters
            )

        def weighted_vals(vals, weights):
            return np.nan_to_num(vals) * weights

        station_names1, axes1, phase_pols1, *params1 = extract_data(h5_1)

        antenna_selection = functools.partial(filter_params, self.filter_stations(station_names1), axes1.index('ant'))
        phase_vals1, phase_weights1, amps1 = antenna_selection(*params1)

        prep_phase_score = weighted_vals(phase_vals1, phase_weights1)
        prep_amp_score = weighted_vals(amps1, phase_weights1)

        if h5_2 is not None:
            _, _, phase_pols2, *params2 = extract_data(h5_2)
            phase_vals2, phase_weights2, amps2 = antenna_selection(*params2)

            min_length = min(len(phase_pols1), len(phase_pols2))
            assert 0 < min_length <= 4

            indices = [0] if min_length == 1 else [0, -1]

            if 'pol' in axes1:
                prep_phase_score, prep_amp_score, phase_vals2, phase_weights2, amps2 = filter_params(
                    indices, axes1.index('pol'), prep_phase_score, prep_amp_score, phase_vals2, phase_weights2, amps2
                )

            # np.seterr(all='raise')
            prep_phase_score = np.subtract(prep_phase_score, weighted_vals(phase_vals2, phase_weights2))
            prep_amp_score = np.divide(
                prep_amp_score,
                weighted_vals(amps2, phase_weights2),
                out=np.zeros_like(prep_amp_score),
                where=phase_weights2 != 0
            )

        phase_score = circstd(prep_phase_score[prep_phase_score != 0], nan_policy='omit')
        amp_score = np.std(prep_amp_score[prep_amp_score != 0])

        return phase_score, amp_score

    def solution_stability(self):
        """
        Get solution stability scores and make figure

        :return:    bestcycle --> best cycle according to solutions
                    accept --> accept this selfcal
        """

        # loop over sources to get scores
        for k, source in enumerate(self.sources):
            logger.debug(source)

            sub_h5s = sorted([h5 for h5 in self.h5s if source in h5])

            phase_scores = []
            amp_scores = []

            for m, sub_h5 in enumerate(sub_h5s):
                number = get_cycle_num(sub_h5)

                phase_score, amp_score = self.get_solution_scores(sub_h5, sub_h5s[m - 1] if number > 0 else None)

                phase_scores.append(phase_score)
                amp_scores.append(amp_score)

            if k == 0:
                total_phase_scores = [phase_scores]
                total_amp_scores = [amp_scores]

            total_phase_scores = np.append(total_phase_scores, [phase_scores], axis=0)
            total_amp_scores = np.append(total_amp_scores, [amp_scores], axis=0)

        # plot
        finalphase, finalamp = (np.mean(score, axis=0) for score in (total_phase_scores, total_amp_scores))

        return finalphase, finalamp

    @staticmethod
    def solution_accept_reject(finalphase, finalamp):

        bestcycle = int(np.array(finalphase).argmin())

        # selection based on slope
        phase_decrease = linreg_slope(finalphase[:bestcycle+1])
        if not all(v == 0 for v in finalamp) and len(finalamp) >= 3:
            # amplitude solves start typically later than phase solves
            start_cycle = 0
            for a in finalamp:
                if a==0:
                    start_cycle += 1
            if bestcycle-start_cycle > 3:
                amp_decrease = linreg_slope([i for i in finalamp if i != 0][start_cycle:bestcycle+1])
            else:
                amp_decrease = 0
        else:
            amp_decrease = 0
        accept = (
                (phase_decrease <= 0
                or amp_decrease <= 0)
                and bestcycle >= 1
                and finalphase[bestcycle] < 1
                and finalphase[0] > finalphase[bestcycle]
        )
        return bestcycle, accept

    @staticmethod
    def image_stability(rmss, minmaxs):
        """
        Determine image stability

        :param minmaxs: absolute values of min/max for each self-cal cycle
        :param rmss: rms (noise) for each self-cal cycle

        :return: bestcycle --> best solution cycle
                 accept    --> accept this selfcal
        """

        # metric scores
        combined_metric = min_max_norm(rmss) * min_max_norm(minmaxs)

        # best cycle
        bestcycle = select_cycle(combined_metric)

        # getting slopes for selection
        rms_slope, minmax_slope = linregress(list(range(len(rmss[:bestcycle+1]))), rmss[:bestcycle+1]).slope, linregress(
            list(range(len(rmss[:bestcycle+1]))),
            np.array(
                minmaxs[:bestcycle+1])).slope

        # acceptance criteria
        accept = ((rms_slope <= 0
                  or minmax_slope <= 0)
                  and rmss[0] > rmss[bestcycle]
                  and minmaxs[0] > minmaxs[bestcycle]
                  and bestcycle >= 1)

        return bestcycle, accept

    def peak_flux_constraint(self):
        """
        Validate if the peak flux is larger than 100 times the local rms
        """
        return get_peakflux(self.fitsfiles[0])/get_rms(self.fitsfiles[0]) > 100


def parse_source_from_h5(h5):
    """
    Parse sensible output names
    """
    h5 = h5.split("/")[-1]
    if 'ILTJ' in h5:
        matches = re.findall(r'ILTJ\d+\..\d+\+\d+.\d+_L\d+', h5)
        if len(matches)==0:
            matches = re.findall(r'ILTJ\d+\..\d+\+\d+.\d+', h5)
            if len(matches)==0:
                print("WARNING: Difficulty with parsing the source name form " + h5)
                output = (re.sub('(\D)\d{3}\_', '', h5).
                          replace("merged_", "").
                          replace('addCS_', '').
                          replace('selfcalcyl', '').
                          replace('selfcalcyle', '').
                          replace('.ms', '').
                          replace('.copy', '').
                          replace('.phaseup', '').
                          replace('.h5', '').
                          replace('.dp3', '').
                          replace('-concat', '').
                          replace('.phasediff','').
                          replace('_uv','').
                          replace('scalarphasediff0_sky',''))
                print('Parsed into ' + h5)
                return output
        output = matches[0]
    elif 'selfcalcyle' in h5:
        matches = re.findall(r'selfcalcyle\d+_(.*?)\.', h5)
        output = matches[0]
    else:
        print("WARNING: Difficulty with parsing the source name form "+h5)
        output = (re.sub('(\D)\d{3}\_', '', h5).
                  replace("merged_", "").
                  replace('addCS_', '').
                  replace('selfcalcyl', '').
                  replace('selfcalcyle', '').
                  replace('.ms', '').
                  replace('.copy', '').
                  replace('.phaseup', '').
                  replace('.h5', '').
                  replace('.dp3', '').
                  replace('-concat', '').
                  replace('.phasediff', '').
                  replace('_uv', '').
                  replace('scalarphasediff0_sky', ''))
        print('Parsed into '+h5)

    return output


def min_max_norm(lst):
    """Normalize list values between 0 and 1"""

    # find the minimum and maximum
    min_value = min(lst)
    max_value = max(lst)

    # normalize
    normalized_floats = [(x - min_value) / (max_value - min_value) for x in lst]

    return np.array(normalized_floats)


def linreg_slope(values=None):
    """
    Fit linear regression and return slope

    :param values: Values

    :return: linear regression slope
    """

    return linregress(list(range(len(values))), values).slope


def get_minmax(inp: Union[str, np.ndarray]):
    """
    Get min/max value

    :param inp: fits file name or numpy array

    :return: minmax --> pixel min/max value
    """
    if isinstance(inp, str):
        with fits.open(inp) as hdul:
            data = hdul[0].data
    else:
        data = inp

    minmax = np.abs(data.min() / data.max())

    logger.debug(f"min/max: {minmax}")
    return minmax

def get_peakflux(inp: Union[str, np.ndarray]):
    """
    Get min/max value

    :param inp: fits file name or numpy array

    :return: minmax --> pixel min/max value
    """
    if isinstance(inp, str):
        with fits.open(inp) as hdul:
            data = hdul[0].data
    else:
        data = inp

    mx = data.max()

    logger.debug(f"Peak flux: {mx}")
    return mx


def select_cycle(cycles=None):
    """
    Select best cycle

    :param cycles: rms or minmax cycles

    :return: best cycle
    """

    b, best_cycle = 0, 0
    for n, c in enumerate(cycles[1:]):
        if c > cycles[n - 1]:
            b += 1
        else:
            b = 0
            best_cycle = n + 1
        if b == 2:
            break
    return best_cycle


def get_rms(inp: Union[str, np.ndarray], maskSup: float = 1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS

    :param inp: fits file name or numpy array
    :param maskSup: mask threshold

    :return: rms --> rms of image
    """

    if isinstance(inp, str):
        with fits.open(inp) as hdul:
            data = hdul[0].data
    else:
        data = inp

    mIn = np.ndarray.flatten(data)
    m = mIn[np.abs(mIn) > maskSup]
    rmsold = np.std(m)
    diff = 1e-1
    cut = 3.
    med = np.median(m)

    for i in range(10):
        ind = np.where(np.abs(m - med) < rmsold * cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms - rmsold) / rmsold) < diff:
            break
        rmsold = rms

    logger.debug(f'rms: {rms}')

    return rms  # jy/beam


def get_cycle_num(fitsfile: str = None) -> int:
    """
    Parse cycle number

    :param fitsfile: fits file name
    """

    cycle_num = int(float(re.findall(r"selfcalcyle(\d+)", fitsfile.split('/')[-1])[0]))
    assert cycle_num >= 0
    return cycle_num


def make_utf8(inp=None):
    """
    Convert input to utf8 instead of bytes

    :param inp: string input
    """

    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def make_figure(vals1=None, vals2=None, label1=None, label2=None, plotname=None, bestcycle=None):
    """
    Make figure (with optionally two axis)

    :param vals1: values 1
    :param vals2: values 2
    :param label1: label corresponding to values 1
    :param label2: label corresponding to values 2
    :param plotname: plot name
    :param bestcycle: plot best cycle
    """

    plt.style.use('ggplot')

    fig, ax1 = plt.subplots()

    color = 'red'
    ax1.set_xlabel('cycle')
    ax1.set_ylabel(label1, color="tab:"+color)
    ax1.plot([i for i in range(len(vals1))], vals1, color='dark'+color, linewidth=2, marker='s',
             markerfacecolor="tab:"+color, markersize=3, alpha=0.7, dash_capstyle='round', dash_joinstyle='round')
    ax1.tick_params(axis='y', labelcolor="tab:"+color)
    ax1.grid(False)
    ax1.plot([bestcycle, bestcycle], [0, max(vals1)], linestyle='--', color='black')
    ax1.set_ylim(0, max(vals1))

    if vals2 is not None:

        ax2 = ax1.twinx()

        color = 'blue'
        ax2.set_ylabel(label2, color="tab:"+color)
        ax2.plot([i for i, v in enumerate(vals2) if v!=0], [v for v in vals2 if v!=0], color='dark'+color, linewidth=2,
                 marker='s', markerfacecolor="tab:"+color, markersize=3, alpha=0.7, dash_capstyle='round',
                 dash_joinstyle='round')
        ax2.tick_params(axis='y', labelcolor="tab:"+color)
        ax2.grid(False)
        ax2.set_ylim(0, max(vals2))

    fig.tight_layout()

    plt.savefig(plotname, dpi=150)


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser(description='Determine selfcal quality')
    parser.add_argument('--fits', nargs='+', help='selfcal fits images')
    parser.add_argument('--h5', nargs='+', help='h5 solutions')
    parser.add_argument('--station', type=str, help='', default='international',
                        choices=['dutch', 'remote', 'alldutch', 'international', 'debug'])
    parser.add_argument('--parallel', action='store_true', help='run parallel over multiple sources (for testing)')
    parser.add_argument('--root_folder_parallel', type=str, help='root folder (if parallel is on), '
                                                                 'which is the path to where all calibrator source folders are located')
    return parser.parse_args()


def main(h5s: list = None, fitsfiles: list = None, station: str = 'international'):
    """
    Main function

    Input:
        - List of h5 files
        - List of fits files
        - Station type

    Returns:
        - Source name
        - Accept source (total)
        - Best cycle (total)
        - Accept according to solutions
        - Best cycle according to solutions
        - Accept according to images
        - Best cycle according to images
        - Best h5parm
    """
    sq = SelfcalQuality(h5s, fitsfiles, station)

    assert len(sq.h5s) > 1 and len(sq.fitsfiles) > 1, "Need more than 1 h5 or fits file"

    finalphase, finalamp = sq.solution_stability()
    bestcycle_solutions, accept_solutions = sq.solution_accept_reject(finalphase, finalamp)

    rmss = [get_rms(fts) * 1000 for fts in sq.fitsfiles]
    minmaxs = [get_minmax(fts) for fts in sq.fitsfiles]

    bestcycle_image, accept_image_stability = sq.image_stability(rmss, minmaxs)
    accept_peak = sq.peak_flux_constraint()

    best_cycle = int(round((bestcycle_solutions + bestcycle_image - 1)//2, 0))
    # final accept
    accept = (accept_image_stability
              and accept_peak
              and accept_solutions
              and (rmss[best_cycle] < rmss[0] or minmaxs[best_cycle] < minmaxs[0])
              )

    logger.info(
        f"{sq.main_source} | "
        f"Best cycle according to images: {bestcycle_image}, accept image: {accept_image_stability}. "
        f"Best cycle according to solutions: {bestcycle_solutions}, accept solution: {accept_solutions}. "
    )

    logger.info(
        f"{sq.main_source} | accept: {accept}, best solutions: {sq.h5s[best_cycle]}"
    )

    fname = f'./selection_output/selfcal_performance_{sq.main_source}.csv'
    system(f'mkdir -p ./selection_output')
    with open(fname, 'w') as textfile:
        # output csv
        csv_writer = csv.writer(textfile)
        csv_writer.writerow(['solutions', 'dirty'] + [str(i) for i in range(len(sq.fitsfiles))])

        # best cycle based on phase solution stability
        csv_writer.writerow(['phase', np.nan] + list(finalphase))
        csv_writer.writerow(['amp', np.nan] + list(finalamp))

        csv_writer.writerow(['min/max'] + minmaxs + [np.nan])
        csv_writer.writerow(['rms'] + rmss + [np.nan])

    make_figure(finalphase, finalamp, 'Phase stability', 'Amplitude stability', f'./selection_output/solution_stability_{sq.main_source}.png', best_cycle)
    make_figure(rmss, minmaxs, 'RMS (mJy/beam)', '$|min/max|$', f'./selection_output/image_stability_{sq.main_source}.png', best_cycle)

    df = pd.read_csv(fname).set_index('solutions').T
    df.to_csv(fname, index=False)

    return sq.main_source, accept, best_cycle, accept_solutions, bestcycle_solutions, accept_image_stability, bestcycle_image, sq.h5s[best_cycle]


def calc_all_scores(sources_root, stations='international'):
    """
    For parallel calculation (for testing purposes only)
    """

    def get_solutions(item):
        item = Path(item)
        if not (item.is_dir() and any(file.suffix == '.h5' for file in item.iterdir())):
            return None
        star_folder, star_name = item, item.name

        try:
            return main(list(map(str, sorted(star_folder.glob('merged*.h5')))),
                        list(map(str, sorted(star_folder.glob('*MFS-*image.fits')))), stations)
        except Exception as e:
            logger.warning(f"skipping {star_folder} due to {e}")
            return star_name, None, None, None, None, None, None

    all_files = [p for p in Path(sources_root).iterdir() if p.is_dir()]

    results = Parallel(n_jobs=len(sched_getaffinity(0)))(delayed(get_solutions)(f) for f in all_files)

    results = filter(None, results)

    fname = f'./selection_output/selfcal_performance.csv'
    system(f'mkdir -p ./selection_output')
    with open(fname, 'w') as textfile:
        csv_writer = csv.writer(textfile)
        csv_writer.writerow(
            ['source', 'accept', 'bestcycle', 'accept_solutions', 'bestcycle_solutions', 'accept_images', 'bestcycle_images', 'best_h5']
        )

        for res in results:
             # output csv
            csv_writer.writerow(res)


if __name__ == '__main__':
    args = parse_args()

    output_folder='./selection_output'
    system(f'mkdir -p {output_folder}')

    if args.parallel:
        print(f"Running parallel in {args.root_folder_parallel}")
        if args.root_folder_parallel is not None:
            calc_all_scores(args.root_folder_parallel)
        else:
            sys.exit("ERROR: if parallel, you need to specify --root_folder_parallel")
    else:
        main(args.h5, args.fits, args.station)
