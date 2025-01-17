from cortexchange.wdclient import init_downloader
from cortexchange.architecture import get_architecture, Architecture
from argparse import ArgumentParser


def get_nn_model(model: str = 'surf/dinov2_09814', device: str = 'cpu', cache: str = ".cache/cortexchange"):
    """
    Get Neural Network model for prediction

    Args:
        model: Model name
        device: Device name (CPU or GPU)

    Returns:
        Model
    """
    init_downloader(url="https://researchdrive.surfsara.nl/public.php/webdav/",
                    login="WsSxVZHPqHlKcvY",
                    password="PublicAccess1!",
                    cache=cache)
    TransferLearning: type(Architecture) = get_architecture('surf/TransferLearning')
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
    return model.predict(torch_tensor)


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Get image score with Neural Network model")
    parser.add_argument('images', nargs='+', help='Images', default=None)
    parser.add_argument('--cache', help='Cache folder with model', default='.cache/cortexchange')
    parser.add_argument('--device', help='CPU or GPU', default='cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    model = get_nn_model(cache='.cache/cortexchange', device=args.device)
    for im in args.images:
        print(im)
        predict_nn(im, model)


if __name__ == '__main__':
    main()
