import tables
import pickle
import numpy as np


def update_sourcedirname_h5_dde(h5, modeldatacolumns):
    """
    Replace direction names in the H5 file with modeldatacolumns names.

    Parameters:
    h5 (str): Path to the H5 file.
    modeldatacolumns (list): List of direction names to update.
    """
    modeldatacolumns_outname = [f'DIL{str(mm_id).zfill(2)}' for mm_id in range(len(modeldatacolumns))]
    # Create output names with 5 characters each

    with tables.open_file(h5, mode='a') as H:
        # Update source directions
        for direction_id, direction in enumerate(modeldatacolumns):
            H.root.sol000.source[direction_id] = (
                modeldatacolumns[direction_id], H.root.sol000.source[direction_id]['dir'])
            print(H.root.sol000.source[direction_id]['name'], modeldatacolumns[direction_id])

        # Update directions in various sol sets
        for sol_set in ['phase000', 'amplitude000', 'tec000', 'rotation000']:
            try:
                getattr(H.root.sol000, sol_set).dir[:] = modeldatacolumns_outname
                print(f'Updated direction names in {sol_set}:', modeldatacolumns_outname)
            except AttributeError:
                pass

    print('Done updating direction names.')

def update_sourcedir_h5_dde(h5, sourcedirpickle, dir_id_kept=None):
    f = open(sourcedirpickle, 'rb')
    sourcedir = pickle.load(f)
    f.close()

    if dir_id_kept is not None:
        print('Before directions')
        print(sourcedir)
        sourcedir = np.copy(sourcedir[dir_id_kept][:])
        print('After directions')
        print(sourcedir)

    H = tables.open_file(h5, mode='a')
    for direction_id, direction in enumerate(np.copy(H.root.sol000.source[:])):
        H.root.sol000.source[direction_id] = (
            direction['name'], [sourcedir[direction_id, 0], sourcedir[direction_id, 1]])
    H.close()

    return