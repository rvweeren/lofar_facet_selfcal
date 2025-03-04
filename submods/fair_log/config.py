"""
Module for enhancing reproducibility by logging calibration parameters to merged h5parm.
"""

__author__ = "Jurjen de Jong (jurjendejong@strw.leidenuniv.nl)"

import tables
from argparse import ArgumentParser


def add_config_to_h5(h5parm: str = None, config: str = None):
    """
    Adding parameter configurations to h5parm for reproducibility.

    Args:
        h5parm: h5parm solution file
        config: configuration file
    """

    # Read the content of the config file
    if is_valid_filename(config):
        with open(config, "r") as f:
            config_data = f.read()
    else:
        config_data = config

    # Store it as an attribute in the h5parm file
    with tables.open_file(h5parm, mode="a") as h5f:
        h5f.root._v_attrs.config = config_data


def add_version_to_h5(h5parm: str = None, version: str = None):
    """
    Adding parameter configurations to h5parm for reproducibility.

    Args:
        h5parm: h5parm solution file
        version: facet_selfcal version
    """

    # Store it as an attribute in the h5parm file
    with tables.open_file(h5parm, mode="a") as h5f:
        h5f.root._v_attrs.facetselfcal_version = version


def get_config_from_h5(h5parm: str = None, output: str = None, printlog: bool = True):
    """
    Read/write configuration file from h5parm.

    Args:
        h5parm: h5parm solution file
        output: text file name for writing config file to
        printlog: print the configuration file
    """

    # Read config file from h5parm
    with tables.open_file(h5parm, mode="r") as h5f:
        config = getattr(h5f.root._v_attrs, "config", None)
        if config is None:
            return config

        # Write configuration parameters to output txt file
        if output is not None:
            with open(output, "w") as file:
                file.write(config)
        # Read only
        elif printlog:
            print(config)

    return config


def get_facetselfcal_version_from_h5(h5parm: str = None, printversion: bool = True):
    """
    Read facetselfcal version from h5parm.

    Args:
        h5parm: h5parm solution file
        printversion: print version name
    """

    # Read config file from h5parm
    with tables.open_file(h5parm, mode="r") as h5f:
        version = getattr(h5f.root._v_attrs, "facetselfcal_version", None)
        if version is not None and printversion:
            print(f"facetselfcal version: "+version)

    return version


def is_valid_filename(filename):
    try:
        with open(filename, "r") as f:
            pass  # Try opening
        return True
    except OSError:
        return False  # Invalid file name


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """

    parser = ArgumentParser("Log configuration parameters for reproducibility")
    parser.add_argument('h5', help='h5parm file name.')
    parser.add_argument('--write_to', help='Output file name to write configuration file to.')
    return parser.parse_args()


def main():
    args = parse_args()
    get_config_from_h5(args.h5, args.write_to)
    get_facetselfcal_version_from_h5(args.h5)


if __name__ == '__main__':
    main()
