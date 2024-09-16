import sys
from glob import glob
from .selfcal_selection import get_minmax, get_rms
from argparse import ArgumentParser

try:
    from cortex.predictors import StopPredictor
except ImportError:
    sys.exit('ERROR: Missing cortex.predictors --> please install pip install "git+https://github.com/jurjen93/lofar_helpers@inference_script#subdirectory=neural_networks"')


def early_stopping(folder='.', predictor=None):
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



    for n, im in enumerate(ims):
        print(f"Determine early-stopping for {im}")
        minmax = get_minmax(im)
        rms = get_rms(im)
        pred, pred_err = [float(i) for i in predictor.predict(im)]

        minmaxs.append(minmax)
        rmss.append(rms)
        preds.append(pred)
        preds_err.append(pred_err)

        if n>0:
            selfcal_score = rms/rmss[0] * minmax/minmaxs[0] * pred/preds[0]
            print(selfcal_score)
        else:
            signal = get_signal(im)

            if signal>7:
                cycle_min = 4
            else:
                cycle_min = 2

            selfcal_score = 1

        print(minmax, rms, pred, pred_err, signal)

        if n>cycle_min:

            if (pred<0.3
                and rms<rmss[0]
                and minmax<minmaxs[0]
                and pred_err<0.2):
                print('Early-stopping criteria 1: Converged')
                return n, selfcal_score, signal, 'converged'

            elif (pred<0.25
                and minmax<minmaxs[0]
                and pred_err<0.2):
                print('Early-stopping criteria 2: Converged')
                return n, selfcal_score, signal, 'converged'

            elif (pred<0.4
                and min(minmaxs)==minmax
                and pred_err<0.2):
                print('Early-stopping criteria 3: Converged')
                return n, selfcal_score, signal, 'converged'

            elif (pred<0.4
                and n>=5
                and rms/rmss[0]<1.05
                and minmax<minmaxs[0]
                and minmax<minmaxs[1]
                and minmax<minmaxs[2]
                and pred_err<0.2):
                print('Early-stopping criteria 4: Converged')
                return n, selfcal_score, signal, 'converged'

            elif (pred<0.5
                and n>=5
                and rms/rmss[0]<1.05
                and minmax<minmaxs[0]
                and minmax<minmaxs[n-1]
                and pred_err<0.1):
                print('Early-stopping criteria 5: Converged')
                return n, selfcal_score, signal, 'converged'

            elif pred<0.15:
                print('Early-stopping criteria 6: Converged')
                return n, selfcal_score, signal, 'converged'

            if (pred>0.75
                and n>7):
                print('Early-stopping criteria 7: Diverged')
                return n, selfcal_score, signal, 'diverged'

            elif (pred>0.75
                and rmss[0]/rms<0.95
                and n>5):
                print('Early-stopping criteria 8: Diverged')
                return n, selfcal_score, signal, 'diverged'

            elif (pred>0.65
                and minmax>minmaxs[0]
                and rmss[0]/rms<0.95
                and n>6):
                print('Early-stopping criteria 9: Diverged')
                return n, selfcal_score, signal, 'diverged'

            elif (pred>0.6
                and rmss[0]/rms<0.7):
                print('Early-stopping criteria 10: Diverged')
                return n, selfcal_score, signal, 'diverged'

            elif rmss[0]/rms<0.5:
                print('Early-stopping criteria 11: Diverged')
                return n, selfcal_score, signal, 'diverged'

            elif minmaxs[0]/minmax<0.5:
                print('Early-stopping criteria 12: Diverged')
                return n, selfcal_score, signal, 'diverged'


    print(f"No early-stopping reached before cycle {n}")
    return n, selfcal_score, signal, 'none'


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

    best_cycle, selfcal_score, _, _ = early_stopping(args.folder, args.ampsolve, predictor)

    print(f"Best cycle: {best_cycle}\nSelfcal score: {selfcal_score}")

if __name__ == "__main__":
    main()
