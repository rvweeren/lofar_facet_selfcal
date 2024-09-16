import sys
from glob import glob
from .selfcal_selection import get_minmax, get_rms
from argparse import ArgumentParser

try:
    from cortex.predictors import StopPredictor
except ImportError:
    sys.exit('ERROR: Missing cortex.predictors --> please install pip install "git+https://github.com/jurjen93/lofar_helpers@inference_script#subdirectory=neural_networks"')


def early_stopping(folder='.', ampsolve=True, predictor=None):
    """
    Determine if early stopping is allowed and return a self-calibration quality score.

    Args:
        folder: selfcal folder
        ampsolve: used amplitude solve or not

    Returns:
        self-calibration score

    """

    if predictor is None:
        sys.exit("ERROR: No predictor given.")

    if folder == '.':
        ims = sorted(glob(f"*_0??-MFS-I-image.fits")+glob(f"*_0??-MFS-image.fits"))
    else:
        ims = sorted(glob(f"{folder}*/*_0??-MFS-image.fits")+glob(f"{folder}*/*_0??-MFS-I-image.fits"))

    if len(ims)==0:
        sys.exit("ERROR: No images given?")

    minmaxs = []
    rmss = []
    preds = []
    preds_err = []

    if ampsolve:
        cycle_min = 4
    else:
        cycle_min = 2

    selfcal_score = 1

    for n, im in enumerate(ims):
        print(f"Determine early-stopping for {im}")
        minmax = get_minmax(im)
        rms = get_rms(im)
        pred, pred_err = [float(i) for i in predictor.predict(im)]

        print(minmax, rms, pred, pred_err)

        minmaxs.append(minmax)
        rmss.append(rms)
        preds.append(pred)
        preds_err.append(pred_err)

        if n>0:
            selfcal_score = rms/rmss[0] * minmax/minmaxs[0] * pred/preds[0]

        if n>cycle_min:

            if (pred<0.3
                and rms<rmss[0]
                and minmax<minmaxs[0]
                and pred_err<0.2):
                print('Early-stopping criteria 1: Converged')
                return n, selfcal_score

            elif (pred<0.25
                and minmax<minmaxs[0]
                and pred_err<0.2):
                print('Early-stopping criteria 2: Converged')
                return n, selfcal_score

            elif (pred<0.4
                and min(minmaxs)==minmax
                and pred_err<0.2):
                print('Early-stopping criteria 3: Converged')
                return n, selfcal_score

            elif (pred<0.4
                and n>=5
                and rms<rmss[0]
                and minmax<minmaxs[0]
                and minmax<minmaxs[1]
                and minmax<minmaxs[2]
                and pred_err<0.2):
                print('Early-stopping criteria 4: Converged')
                return n, selfcal_score

            if (pred>0.6
                and n>6):
                print('Early-stopping criteria 5: Diverged')

            elif (pred>0.7
                and rms>rmss[0]
                and n>3):
                print('Early-stopping criteria 6: Diverged')
                return n, selfcal_score

            elif (pred>0.5
                and minmax>minmaxs[0]
                and n>3):
                print('Early-stopping criteria 7: Diverged')
                return n, selfcal_score

    print(f"No early-stopping reached before cycle {n}")

    return n, selfcal_score


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser()
    parser.add_argument('folder', help='Selfcal folder with images', default='.')
    parser.add_argument('--cache', help='Cache folder for model', default='.cache')
    parser.add_argument('--model', help='Neural network model', default='version_7743995_4__model_resnext101_64x4d__lr_0.001__normalize_0__dropout_p_0.25__use_compile_1')
    parser.add_argument('--device', help='CPU or GPU', default='cpu')
    parser.add_argument('--conservative', action='store_true', help='More conservative selection. Recommended when Amplitudes are solved as well.')

    return parser.parse_args()


def main():
    """
    Main function
    """

    args = parse_args()

    predictor = StopPredictor(cache=args.cache,
                              device=args.device,
                              model=args.model,
                              variational_dropout=5)

    best_cycle, selfcal_score = early_stopping(args.folder, args.ampsolve, predictor)

    print(f"Best cycle: {best_cycle}\nSelfcal score: {selfcal_score}")

if __name__ == "__main__":
    main()