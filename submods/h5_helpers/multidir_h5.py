import tables
import numpy as np
try:
    from .slicing import get_slices
except:
    from slicing import get_slices


def same_weights_multidir(h5: str = None):
    """
    Enforce same weights for all directions in multidir h5parm

    Args:
        h5: multidir-h5
    """

    with tables.open_file(h5, "r+") as H:
        for soltab in H.root.sol000._v_groups.keys():
            st = H.root.sol000._f_get_child(soltab)
            weights = st.weight[:]
            weights[(weights > 0) & (weights < 1)] = 1 # convert small weights to 1
            axes = st.weight.attrs["AXES"].decode("utf8").split(',')
            if 'dir' in axes:
                H.root.sol000._f_get_child(soltab).weight[:] = np.prod(weights, axis=axes.index("dir"), keepdims=True)
            else:
                print(f"WARNING: no directions in solution table: {h5}/{soltab}")


def is_multidir(h5: str = None):
    """
    Check if h5 has multiple directions

    Args:
        h5: h5parm

    Returns: boolean

    """

    with tables.open_file(h5) as H:
        for soltab in H.root.sol000._v_groups.keys():
            dirs = H.root.sol000._f_get_child(soltab).dir[:]
            if len(dirs) > 1:
                return True
    return False
