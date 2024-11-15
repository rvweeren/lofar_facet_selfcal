"""
Split out multi-dir h5
"""

import tables
import numpy as np
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import system as commandline


def split_h5_idx(multi_dir_h5, h5_out_name, idx):
    """
    Split h5 for specific index

    :param multi_dir_h5: h5parm with multiple directions
    :param h5_out_name: h5parm output name
    :param idx: index of direction to split
    """

    shutil.copy(multi_dir_h5, h5_out_name+'.tmp')  # Make a separate copy for each direction

    try:
        # open both input and output h5parms
        with tables.open_file(multi_dir_h5, 'r') as h5_in, tables.open_file(h5_out_name+'.tmp', 'r+') as h5_out:

            # iterate over solution sets (e.g., sol000, sol001, etc.)
            for solset in h5_out.root._v_groups.keys():
                ss = h5_out.root._f_get_child(solset)

                # remove original 'source' table in the output file if it exists
                if 'source' in ss:
                    ss._f_get_child('source')._f_remove()

                # create a new 'source' table with the specified direction index
                values = np.array([(b'Dir00', h5_in.root.sol000.source[:][idx][1])],
                                  dtype=[('name', 'S128'), ('dir', '<f4', (2,))])
                h5_out.create_table(ss, 'source', values, title='Source names and directions')

                # iterate over solution tables (e.g., phase000, amplitude000, etc.)
                for soltab in ss._v_groups.keys():
                    st = ss._f_get_child(soltab)

                    # create new .dir array
                    st.dir._f_remove()
                    h5_out.create_array(st, 'dir', np.array([b'Dir00'], dtype='|S5'))

                    # process the 'val' and 'weight' datasets
                    for axes in ['val', 'weight']:
                        st_axes = st._f_get_child(axes)
                        AXES = st_axes.attrs['AXES']
                        dir_idx = AXES.decode('utf8').split(',').index('dir')

                        # extract values for the specified direction index
                        allvals = h5_in.root._f_get_child(solset)._f_get_child(soltab)._f_get_child(axes)[:]
                        newvals = np.take(allvals, indices=[idx], axis=dir_idx)
                        del allvals  # Free memory

                        # map the data type to the corresponding atom type
                        dtype_map = {
                            'float16': tables.Float16Atom,
                            'float32': tables.Float32Atom,
                            'float64': tables.Float64Atom,
                        }
                        valtype = str(st_axes.dtype)
                        atomtype = dtype_map.get(valtype, tables.Float64Atom)()

                        # remove the old dataset and create a new one with the selected values
                        st_axes._f_remove()
                        h5_out.create_array(st, axes, newvals.astype(valtype), atom=atomtype)
                        st._f_get_child(axes).attrs['AXES'] = AXES

        print(f'Repack {h5_out_name}')
        commandline(f'h5repack {h5_out_name}.tmp {h5_out_name} && rm {h5_out_name}.tmp')

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def process_direction(multi_dir_h5, n, dir_name):
    """Helper function to process each direction split."""
    dir_str = dir_name.decode('utf8')
    print(f"Splitting {dir_str}")
    split_h5_idx(multi_dir_h5, f"{dir_str}.h5", n)
    print(f"Splitted {dir_str}")


def split_multidir(multi_dir_h5):
    """
    Split all directions from the multi directional h5parm file.

    :param multi_dir_h5: h5parm containing multiple directions.
    """

    # open the h5parm and read the directions
    with tables.open_file(multi_dir_h5) as h5:
        dirs = h5.root.sol000.source[:]['name']

    # ProcessPoolExecutor to run the splits concurrently
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_direction, multi_dir_h5, n, dir) for n, dir in enumerate(dirs)]

        # wait for all threads to complete and handle exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred while processing a direction: {exc}")
