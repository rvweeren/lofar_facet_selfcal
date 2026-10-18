import numpy as np
import tables
import sys


def copy_over_source_direction_h5(h5ref, h5):
    """
    Replace source direction in h5 with the one in h5ref
    """
    with tables.open_file(h5ref) as Href:
        refsource = Href.root.sol000.source[:]

    overwrite_table(h5, 'source', refsource)
    return


def overwrite_table(h5, table, new_arr):
    """
    Overwrite h5 source or antenna table

    :param h5: h5parm table
    :param table: table name (antenna or source)
    :param new_arr: new values
    """

    with tables.open_file(h5, 'r+') as T:

        ss = T.root._f_get_child('sol000')
        ss._f_get_child(table)._f_remove()
        if table == 'source':
            values = np.array(new_arr, dtype=[('name', 'S128'), ('dir', '<f4', (2,))])
            title = 'Source names and directions'
        elif table == 'antenna':
            title = 'Antenna names and positions'
            values = np.array(new_arr, dtype=[('name', 'S16'), ('position', '<f4', (3,))])
        else:
            sys.exit('ERROR: table needs to be source or antenna')

        T.create_table(ss, table, values, title=title)

    return


def set_weights_h5_to_one(h5parm):
    """
    Set weights for the solutions that have valid numbers to 1.0
    This is useful for bandpass solutions because the losoto time median preserves the time depedent flagging otherwise

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file whose valid solution weights are updated.
    """
    with tables.open_file(h5parm) as H:
        soltabs = list(H.root.sol000._v_children.keys())
    H = tables.open_file(h5parm, mode='a')
    
    if 'phase000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.phase000.val[:])
         # update weights to 1.0
         H.root.sol000.phase000.weight[goodvals] = 1.0

    if 'amplitude000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.amplitude000.val[:])
         # update weights to 1.0
         H.root.sol000.amplitude000.weight[goodvals] = 1.0

    if 'tec000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.tec000.val[:])
         # update weights to 1.0
         H.root.sol000.tec000.weight[goodvals] = 1.0

    if 'delay000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.delay000.val[:])
         # update weights to 1.0
         H.root.sol000.delay000.weight[goodvals] = 1.0

    if 'rotation000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.rotation000.val[:])
         # update weights to 1.0
         H.root.sol000.rotation000.weight[goodvals] = 1.0

    if 'rotationmeasure000' in soltabs:
         goodvals =  np.isfinite(H.root.sol000.rotationmeasure000.val[:])
         # update weights to 1.0
         H.root.sol000.rotationmeasure000.weight[goodvals] = 1.0

    H.flush()
    H.close()
    return
