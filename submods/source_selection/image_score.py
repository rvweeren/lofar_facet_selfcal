"""
This script requires https://github.com/sara-nl/cortExchange to pull neural network models.
"""

author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

from cortexchange.architecture import get_architecture, Architecture
from argparse import ArgumentParser
import os
import warnings

# Suppress all UserWarnings containing 'xFormers'
warnings.filterwarnings("ignore", message="xFormers is disabled*")
warnings.filterwarnings("ignore", message="xFormers is not available*")

def get_nn_model(model: str = 'surf/dino_big_lora_tune_posclsreg_may_O2_aug_099', # old model: surf/dino_big_lora_default_pos_november_09876
                 device: str = 'cpu',
                 cache: str = ".cache/cortexchange",
                 architecture: str = 'surf/TransferLearningV2'): # old architecture: #surf/TransferLearning
    """
    Get Neural Network model for prediction

    Args:
        model: Model name
        device: Device name (CPU or GPU)

    Returns:
        Model
    """

    os.environ['TORCH_HOME'] = cache
    TransferLearning: type(Architecture) = get_architecture(architecture)

    return TransferLearning(device=device, model_name=model)


def predict_nn(image: str = None, model=None):
    """
    Predict image score
    Args:
        image: Fits image

    Returns:
        Prediction score
    """

    if model is None:
        model = get_nn_model()

    torch_tensor=model.prepare_data(image)

    return float(model.predict(torch_tensor)[0])


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Get image score with Neural Network model")
    parser.add_argument('images', nargs='+', help='Images', default=None)
    parser.add_argument('--cache', help='Cache folder with model', default='.cache/cortexchange')
    parser.add_argument('--device', help='CPU or GPU', default='cpu')
    parser.add_argument('--model', help='Model name', default='surf/dino_big_lora_tune_posclsreg_may_O2_aug_099')

    return parser.parse_args()


def main():
    args = parse_args()
    model = get_nn_model(model=args.model, cache=args.cache, device=args.device)
    for im in args.images:
        print(im)
        print(predict_nn(im, model))


if __name__ == '__main__':
    main()
