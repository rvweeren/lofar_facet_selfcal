def get_double_slice(values, idx: list = None, axes: list = None):
    """
    Get double slices

    :param values: numpy array
    :param idx: list of indices
    :param axes: list of axes corresponding to indices

    return slice
    """

    ax1, ax2 = axes
    id1, id2 = idx
    l = list([slice(None)] * ax1 + [id1] + [slice(None)] * (len(values.shape) - ax1 - 1))
    l[ax2] = id2
    return tuple(l)

def get_slices(values, id: int = None, axis: int = None):
    """
    Get slice

    :param values: numpy array
    :param id: ID from axis
    :param axis: axis corresponding to given id

    :return: slice
    """
    return tuple([slice(None)] * axis + [id] + [slice(None)] * (len(values.shape) - axis - 1))

def get_double_slice_len(values, idx: list = None, axes: list = None):
    """
    Get double slices

    :param values: numpy array
    :param idx: list of indices
    :param axes: list of axes corresponding to indices

    return slice
    """

    ax1, ax2 = axes
    id1, id2 = idx
    l = list([slice(None)] * ax1 + [id1] + [slice(None)] * (len(values) - ax1 - 1))
    l[ax2] = id2
    return tuple(l)