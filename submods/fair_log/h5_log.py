import tables

def log_config(h5parm, config):
    """
    For FAIR data storage, we want to store the facetselfcal config file in the output h5

    Args:
        h5parm: h5parm solution file
        config: configuration file
    """

    # Read the content of the config file
    with open(config, "r") as f:
        config_data = f.read()

    # Store it as an attribute in the H5parm file
    with tables.open_file(h5parm, mode="a") as h5f:
        h5f.root._v_attrs.config = config_data


def read_config_h5(h5parm):
    """
    Read configuration file from h5parm

    Args:
        h5parm: h5parm solution file
    """
    with tables.open_file(h5parm, mode="r") as h5f:
        print(getattr(h5f.root._v_attrs, "config", None))
