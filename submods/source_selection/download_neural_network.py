"""
This script requires https://github.com/sara-nl/cortExchange to pull neural network models.
"""

author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

from argparse import ArgumentParser
import os
import warnings

# Suppress all UserWarnings containing 'xFormers'
warnings.filterwarnings("ignore", message="xFormers is disabled*")
warnings.filterwarnings("ignore", message="xFormers is not available*")


def get_nn_model(model: str = 'surf/dinov2_vitb14_lora_O2_aug_0984',
                 device: str = 'cpu',
                 cache: str = ".cache/cortexchange",
                 architecture: str = 'surf/TransferLearningV3'):
    """
    Get Neural Network model for prediction

    Args:
        model: Model name
        device: Device name (CPU or GPU)
        cache: CortExchange model cache
        architecture: Architecture name

    Returns:
        Model
    """

    from cortexchange.architecture import get_architecture, Architecture
    from cortexchange.wdclient import init_downloader

    os.environ['TORCH_HOME'] = os.path.realpath(cache)

    init_downloader(
        url="https://researchdrive.surf.nl/public.php/webdav/",
        login="WsSxVZHPqHlKcvY",
        password="PublicAccess1!",
        cache=os.path.realpath(cache)
    )

    TransferLearning: type(Architecture) = get_architecture(architecture)

    return TransferLearning(device=device, model_name=model)


def main():
    parser = ArgumentParser("Download neural network")
    parser.add_argument('--cache_directory', help='Cache folder with model', default='.cache/cortexchange')
    parser.add_argument('--device', help='CPU or GPU', default='cpu')
    parser.add_argument('--model', help='Model name', default='surf/dino_big_lora_tune_posclsreg_may_O2_aug_099')

    args = parser.parse_args()

    get_nn_model(
        model=args.model,
        cache=args.cache_directory,
        device=args.device
    )


if __name__ == '__main__':
    main()
