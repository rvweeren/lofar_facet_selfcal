__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import csv
from argparse import ArgumentParser
import pandas as pd
from scipy.stats import circstd
import tables
import numpy as np
from sys import exit

try:
    from validate_lofar_vlbi_imgs import parse_source_id
except ImportError:
    from .validate_lofar_vlbi_imgs import parse_source_id


def make_utf8(inp):
    """
    Convert input to utf8 instead of bytes

    :param inp: string input
    """

    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def get_phase_score(h5):
    """
    Get phase score, which is based on the circular standard deviation

    Args:
        h5: h5parm solution file

    Returns: circular standard deviation
    """

    with tables.open_file(h5) as H:
        axes = make_utf8(H.root.sol000.phase000.val.attrs["AXES"]).split(',')
        phase = np.take(H.root.sol000.phase000.val[:] * H.root.sol000.phase000.weight[:],
                        [0], axis=axes.index('pol'))
        ref_phase = np.take(phase, [0], axis=axes.index('ant'))
        phase -= ref_phase
        p = np.diff(phase, axis=axes.index('freq'))
        return circstd(p)


def get_amp_score(h5):
    """
    Get amplitude score, which is based on the values that are larger than 2.0 and below 0.5

    Args:
        h5: h5parm solution file

    Returns: % > 2.0 and <0.5, standard deviation
    """

    with tables.open_file(h5) as H:
        axes = make_utf8(H.root.sol000.phase000.val.attrs["AXES"]).split(',')
        amps = np.take(H.root.sol000.amplitude000.val[:] * H.root.sol000.amplitude000.weight[:],
                       [0], axis=axes.index('pol'))
        std = np.std(amps)
        score = 1 - len(amps[(amps > 2.0) | (amps < 0.5)])/amps.size
        return score, std


def get_val_scores(solutions):
    """
    Get calibration solution validation scores

    Args:
        solutions: input h5parms
    """

    # Get validation metrics
    with open('validation_solutions.csv', 'w') as csvfile:
        fieldnames = ['Source_id', 'amp_score', 'amp_std', 'phase_circstd', 'sol_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over FITS images
        for sol in solutions:
            id = parse_source_id(sol)
            if id=='':
                id = sol.replace('.h5','')
            print(id)
            amp_score, amp_std = get_amp_score(sol)
            phase_circstd = get_phase_score(sol)
            sol_score = amp_score * (1-phase_circstd) * (1-amp_std)

            writer.writerow({
                'Source_id': id,
                'amp_score': amp_score,
                'amp_std': amp_std,
                'phase_circstd': phase_circstd,
                'sol_score': sol_score
            })


def sol_quality(csv_table, return_error):
    """
    Use calibration solution quality scores to determine if a direction is stable or not.

    Args:
        csv_table: CSV with image-based scores
        return_error: Return error if more than 1 direction is not accepted

    Returns: filtered DataFrame
    """

    df = pd.read_csv(csv_table)
    df['comment'] = ''
    df['accept_solutions'] = True
    # 'bad' qualifications
    mask = (
            (df['phase_circstd'] > 0.025) |
            (df['sol_score'] < 0.75) |
            ((df['amp_std'] > 0.2) & (df['amp_score']<0.985))
    )
    df.loc[mask, 'accept_solutions'] = False
    df.loc[df.amp_score<0.99, 'comment'] += 'Large amps,'
    df.loc[df.phase_circstd>0.02, 'comment'] += 'Noisy phases,'
    df.loc[df.amp_std>0.15, 'comment'] += 'Noisy amps'

    df.to_csv(csv_table)

    if return_error and np.sum(df['accept_solutions'])<len(df):
        exit(f"ERROR: Following directions have bad solutions: \n"
             f"{'\n'.join(list(df[~df.accept_solutions].Source_id))}")


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Get validation scores for calibration solutions")
    parser.add_argument('h5parms', nargs='+', help='h5parms with amplitude and phase solutions', default=None)
    parser.add_argument('--return_error', action='store_true', help='Return error if more than 1 directions incorrect.')
    return parser.parse_args()


def main():
    args = parse_args()
    get_val_scores(args.h5parms)
    sol_quality("validation_solutions.csv", args.return_error)


if __name__ == '__main__':
    main()