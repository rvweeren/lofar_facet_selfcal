import tables
import numpy as np


def fix_delayreference(h5parm, refant):
    """ Delay reference values with respect to a reference station
    Args:
      h5parm: h5parm file
      refant: reference antenna
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.delay000.val.attrs['AXES'].decode().split(',')

    delay = H.root.sol000.delay000.val[:]
    refant_idx = np.where(H.root.sol000.delay000.ant[:].astype(str) == refant)  # to deal with byte strings
    print(refant_idx, refant)
    antennaxis = axisn.index('ant')

    print('Referencing delay to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        delayn = delay - delay[refant_idx[0], ...]
    if antennaxis == 1:
        delayn = delay - delay[:, refant_idx[0], ...]
    if antennaxis == 2:
        delayn = delay - delay[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        delayn = delay - delay[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        delayn = delay - delay[:, :, :, :, refant_idx[0], ...]
    delay = np.copy(delayn)

    # fill values back in
    H.root.sol000.delay000.val[:] = np.copy(delay)

    H.flush()
    H.close()
    return
