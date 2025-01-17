from cortexchange.wdclient import init_downloader
from cortexchange.architecture import get_architecture, Architecture


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
