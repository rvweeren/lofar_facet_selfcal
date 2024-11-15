import tables
import numpy as np

def fix_tecreference(h5parm, refant):
    """ Tec reference values with respect to a reference station
    Args:
      h5parm: h5parm file
      refant: reference antenna
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.tec000.val.attrs['AXES'].decode().split(',')

    tec = H.root.sol000.tec000.val[:]
    refant_idx = np.where(H.root.sol000.tec000.ant[:].astype(str) == refant)  # to deal with byte strings
    print(refant_idx, refant)
    antennaxis = axisn.index('ant')

    print('Referencing tec to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        tecn = tec - tec[refant_idx[0], ...]
    if antennaxis == 1:
        tecn = tec - tec[:, refant_idx[0], ...]
    if antennaxis == 2:
        tecn = tec - tec[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        tecn = tec - tec[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        tecn = tec - tec[:, :, :, :, refant_idx[0], ...]
    tec = np.copy(tecn)

    # fill values back in
    H.root.sol000.tec000.val[:] = np.copy(tec)

    H.flush()
    H.close()
    return