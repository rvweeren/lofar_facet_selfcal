"""
Script to download neural network.
"""

from argparse import ArgumentParser

try:
    from .image_score import get_nn_model as download_model
except ImportError:
    from image_score import get_nn_model as download_model

def main():
    parser = ArgumentParser("Download neural network")
    parser.add_argument('--cache_directory', help='Cache folder with model', default='.cache/cortexchange')
    parser.add_argument('--device', help='CPU or GPU', default='cpu')
    parser.add_argument('--model', help='Model name', default='surf/dino_big_lora_tune_posclsreg_may_O2_aug_099')
    args = parser.parse_args()
    download_model(model=args.model, cache=args.cache_directory, device=args.device)

if __name__ == '__main__':
    main()
