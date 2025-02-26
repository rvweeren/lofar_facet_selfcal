from astropy.io import fits

def log_fits(fits_file, config):
    """
    For FAIR data storage, we want to store the facetselfcal config file in the output fits file

    Args:
        fits_file: fits image
        config: configuration file
    """

    # Read the content of the config file
    with open(config, "r") as f:
        config_data = f.read()

    with fits.open(fits_file) as f:
        header = f[0].header

    pass


def read_config_fits(fits_file):
    """
    Read configuration file from fits file

    Args:
        fits_file: fits image
    """

    pass
