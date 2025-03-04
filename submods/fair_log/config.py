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
    with open(config, "r") as f:
        config_data = f.read()

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


def get_config_from_h5(h5parm: str = None, output: str = None):
    """
    Read/write configuration file from h5parm.

    Args:
        h5parm: h5parm solution file
        output: text file name for writing config file to
    """

    # Read config file from h5parm
    with tables.open_file(h5parm, mode="r") as h5f:
        config = getattr(h5f.root._v_attrs, "config", None)

        # Write configuration parameters to output txt file
        if output is not None:
            with open(output, "w") as file:
                file.write(config)
        # Read only
        else:
            print(config)

    return config


def get_facetselfcal_version_from_h5(h5parm: str = None):
    """
    Read facetselfcal version from h5parm.

    Args:
        h5parm: h5parm solution file
    """

    # Read config file from h5parm
    with tables.open_file(h5parm, mode="r") as h5f:
        version = getattr(h5f.root._v_attrs, "facetselfcal_version", None)
        print(f"facetselfcal version: "+version)

    return version


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
