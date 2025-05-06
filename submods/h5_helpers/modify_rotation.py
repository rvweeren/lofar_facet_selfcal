import tables
import numpy as np


def rotationmeasure_to_phase(H5filein, H5fileout, dejump=False):
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5in = tables.open_file(H5filein, mode='r')
    H5out = tables.open_file(H5fileout, mode='a')
    c = 2.99792458e8

    if dejump:
        rotationmeasure = H5in.root.sol000.rotationmeasure001.val[:]
    else:
        rotationmeasure = H5in.root.sol000.rotationmeasure000.val[:]
    phase = H5in.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    freq = H5in.root.sol000.phase000.freq[:]
    wav = c / freq

    print('FR step: Shape rotationmeasure000/1', rotationmeasure.shape)
    print('FR step: Shape phase000', phase.shape)

    for antenna_id, antennatmp in enumerate(H5in.root.sol000.phase000.ant[:]):
        for freq_id, freqtmp in enumerate(freq):
            phase[:, freq_id, antenna_id, 0, 0] = \
                2. * rotationmeasure[antenna_id, :] * (wav[freq_id]) ** 2  # notice the factor of 2 because of RR-LL
    # note slice over time-axis
    phase[
        ..., -1] = 0.0  # assume last axis is pol-axis, set YY to zero because losoto residual computation changes this froom 0.0 (it divides the poll diff over the XX and YY phases and does not put it all in XX)

    H5out.root.sol000.phase000.val[:] = phase
    H5out.flush()
    H5out.close()
    H5in.close()
    return


def fix_weights_rotationh5(h5parm):
    """
     DP3 bug causing weird weight values in rotation000, fix these
     https://github.com/lofar-astron/DP3/issues/327
     """
    with tables.open_file(h5parm, mode='a') as H:
        # Update rotation000 weights
        weights = H.root.sol000.rotation000.weight[:]
        weights[(weights > 0.0) & (weights < 1.0)] = 1.0
        H.root.sol000.rotation000.weight[:] = np.copy(weights)
        H.flush()

    # Function to update weights for phase000 and amplitude000 safely
    def update_weights(h5file, sol_type):
        try:
            with tables.open_file(h5file, mode='a') as H:
                weights = getattr(H.root.sol000, sol_type).weight[:]
                weights[(weights > 0.0) & (weights < 1.0)] = 1.0
                getattr(H.root.sol000, sol_type).weight[:] = np.copy(weights)
        except AttributeError:
            pass

    # Update phase000 weights
    update_weights(h5parm, 'phase000')

    # Update amplitude000 weights
    update_weights(h5parm, 'amplitude000')

def fix_weights_rotationmeasureh5(h5parm):
    """
     DP3 bug causing weird weight values in rotationmeasure000, fix these
     https://github.com/lofar-astron/DP3/issues/327
     """
    with tables.open_file(h5parm, mode='a') as H:
        # Update rotationmeasure000 weights
        weights = H.root.sol000.rotationmeasure000.weight[:]
        weights[(weights > 0.0) & (weights < 1.0)] = 1.0
        H.root.sol000.rotationmeasure000.weight[:] = np.copy(weights)
        H.flush()

    # Function to update weights for phase000 and amplitude000 safely
    def update_weights(h5file, sol_type):
        try:
            with tables.open_file(h5file, mode='a') as H:
                weights = getattr(H.root.sol000, sol_type).weight[:]
                weights[(weights > 0.0) & (weights < 1.0)] = 1.0
                getattr(H.root.sol000, sol_type).weight[:] = np.copy(weights)
        except AttributeError:
            pass

    # Update phase000 weights
    update_weights(h5parm, 'phase000')

    # Update amplitude000 weights
    update_weights(h5parm, 'amplitude000')


def fix_rotationreference(h5parm, refant):
    """ Phase reference rotation values with respect to a reference station
    Args:
      h5parm: h5parm file
      refant: reference antenna
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.rotation000.val.attrs['AXES'].decode().split(',')

    rotation = H.root.sol000.rotation000.val[:]
    refant_idx = np.where(H.root.sol000.rotation000.ant[:].astype(str) == refant)  # to deal with byte strings
    print(refant_idx, refant)
    antennaxis = axisn.index('ant')

    print('Referencing rotation to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        rotationn = rotation - rotation[refant_idx[0], ...]
    if antennaxis == 1:
        rotationn = rotation - rotation[:, refant_idx[0], ...]
    if antennaxis == 2:
        rotationn = rotation - rotation[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        rotationn = rotation - rotation[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        rotationn = rotation - rotation[:, :, :, :, refant_idx[0], ...]
    rotation = np.copy(rotationn)

    # fill values back in
    H.root.sol000.rotation000.val[:] = np.copy(rotation)

    H.flush()
    H.close()
    return

def fix_rotationmeasurereference(h5parm, refant):
    """ Phase reference rotationmeasure values with respect to a reference station
    Args:
      h5parm: h5parm file
      refant: reference antenna
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.rotationmeasure000.val.attrs['AXES'].decode().split(',')

    rotationmeasure = H.root.sol000.rotationmeasure000.val[:]
    refant_idx = np.where(H.root.sol000.rotationmeasure000.ant[:].astype(str) == refant)  # to deal with byte strings
    print(refant_idx, refant)
    antennaxis = axisn.index('ant')

    print('Referencing rotationmeasure to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        rotationmeasuren = rotationmeasure - rotationmeasure[refant_idx[0], ...]
    if antennaxis == 1:
        rotationmeasuren = rotationmeasure - rotationmeasure[:, refant_idx[0], ...]
    if antennaxis == 2:
        rotationmeasuren = rotationmeasure - rotationmeasure[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        rotationmeasuren = rotationmeasure - rotationmeasure[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        rotationmeasuren = rotationmeasure - rotationmeasure[:, :, :, :, refant_idx[0], ...]
    rotationmeasure = np.copy(rotationmeasuren)

    # fill values back in
    H.root.sol000.rotationmeasure000.val[:] = np.copy(rotationmeasure)

    H.flush()
    H.close()
    return
