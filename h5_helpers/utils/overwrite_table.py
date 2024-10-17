import numpy as np
import tables
import sys

def overwrite_table(h5, table, new_arr):
    """
    Overwrite h5 source or antenna table

    :param h5: h5parm table
    :param table: table name (antenna or source)
    :param new_arr: new values
    """

    T = tables.open_file(h5, 'r+')

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

    T.close()

    return