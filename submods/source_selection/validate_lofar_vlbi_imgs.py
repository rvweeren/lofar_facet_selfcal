__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import re
import csv
from argparse import ArgumentParser
import pandas as pd

try:
    from selfcal_selection import get_minmax, get_peakflux, get_rms
except ImportError:
    from .selfcal_selection import get_minmax, get_peakflux, get_rms


def parse_source_id(inp_str: str = None):
    """
    Parse ILTJhhmmss.ss±ddmmss.s source_id string

    Args:
        inp_str: ILTJ source_id

    Returns: parsed output

    """

    try:
        parsed_inp = re.findall(r'ILTJ\d+\..\d+\+\d+.\d+', inp_str)[0]
    except IndexError:
        parsed_inp = ''

    return parsed_inp


def get_val_scores(images, model, model_cache):
    """
    Get validation scores

    Args:
        images: input FITS images
        model: neural network model name
        model_cache: local cache for model
    """

    # Get neural network model
    try:
        try:
            from image_score import get_nn_model, predict_nn
        except ImportError:
            from .image_score import get_nn_model, predict_nn
        nn_model = get_nn_model(model=model, cache=model_cache)
    except Exception:
        nn_model = None
        print("Fail to load neural network model.")

    # Get validation metrics
    with open('validation_images.csv', 'w') as csvfile:
        fieldnames = ['Source_id', 'Peak_flux', 'Dyn_range', 'RMS', "NN_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over FITS images
        for image in images:
            id = parse_source_id(image)
            if id=='':
                id = image.replace('.fits','')
            print(id)
            minmax = get_minmax(image)
            rms = get_rms(image)
            peak = get_peakflux(image)
            if nn_model is not None:
                nn_score = predict_nn(image, nn_model)
            else:
                nn_score = 999.
            print(f"NN score: {nn_score}")
            writer.writerow({
                'Source_id': id,
                'Peak_flux': peak,
                'Dyn_range': minmax,
                'RMS': rms,
                'NN_score': nn_score
            })


def filter_sources(csv_table):
    """
    Filter DataFrame with image-based scores

    Args:
        csv_table: CSV with image-based scores

    Returns: filtered DataFrame
    """

    df = pd.read_csv(csv_table)

    # Filter for weak self-calibration sources
    if df['NN_score'].min() != 999.:
        df_filt = df[~((df['Dyn_range'] > 0.04) | (df['NN_score']>0.5) & (df['Peak_flux']<0.03))]
    else:
        df_filt = df[df['Dyn_range'] < 0.035]

    df_filt.to_csv(csv_table, index=False)


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Get validation scores for images")
    parser.add_argument('images', nargs='+', help='Images', default=None)
    parser.add_argument('--model', help='Model name', default='surf/dino_big_lora_tune_posclsreg_may_O2_aug_099')
    parser.add_argument('--cache', help='Cache folder with model', default='.cache/cortexchange')

    return parser.parse_args()


def main():
    args = parse_args()
    get_val_scores(args.images, model=args.model, model_cache=args.cache)
    filter_sources("validation_images.csv")


if __name__ == '__main__':
    main()