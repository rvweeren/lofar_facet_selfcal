from losoto import h5parm
import numpy as np
import tables

def remove_nans(parmdb, soltab):
    """
    Remove NaN values in an h5parm solution table by setting default values based on the solution type.

    Args:
      parmdb (str): Path to the h5parm file.
      soltab (str): Solution table name (e.g., amplitude000, phase000, rotation000).
    """
    H5 = h5parm.h5parm(parmdb, readonly=False)
    solset = H5.getSolset('sol000').getSoltab(soltab)
    vals = solset.getValues()[0]
    weights = solset.getValues(weight=True)[0]

    # Identify NaN indices
    idxnan = np.where(~np.isfinite(vals))
    print('Found some NaNs, flagging them...')

    # Set default values based on solution type
    default_values = {
        'phase': 0.0,
        'amplitude': 1.0,
        'rotation': 0.0
    }
    sol_type = solset.getType()
    vals[idxnan] = default_values.get(sol_type, 0.0)
    weights[idxnan] = 0.0

    # Update values and weights in the h5parm file
    solset.setValues(weights, weight=True)
    solset.setValues(vals)
    H5.close()

def removenans_fulljones(parmdb):
    """ Remove nan values in full jones h5parm

    Args:
      parmdb: h5parm file
    """
    H = tables.open_file(parmdb, mode='a')
    amplitude = H.root.sol000.amplitude000.val[:]
    weights = H.root.sol000.amplitude000.weight[:]
    phase = H.root.sol000.phase000.val[:]
    weights_p = H.root.sol000.phase000.weight[:]

    # XX
    amps_xx = amplitude[..., 0]
    weights_xx = weights[..., 0]
    idx = np.where((~np.isfinite(amps_xx)))
    amps_xx[idx] = 1.0
    weights_xx[idx] = 0.0
    phase_xx = phase[..., 0]
    weights_p_xx = weights_p[..., 0]
    phase_xx[idx] = 0.0
    weights_p_xx[idx] = 0.0

    # XY
    amps_xy = amplitude[..., 1]
    weights_xy = weights[..., 1]
    idx = np.where((~np.isfinite(amps_xy)))
    amps_xy[idx] = 0.0
    weights_xy[idx] = 0.0
    phase_xy = phase[..., 1]
    weights_p_xy = weights_p[..., 1]
    phase_xy[idx] = 0.0
    weights_p_xy[idx] = 0.0

    # XY
    amps_yx = amplitude[..., 2]
    weights_yx = weights[..., 2]
    idx = np.where((~np.isfinite(amps_yx)))
    amps_yx[idx] = 0.0
    weights_yx[idx] = 0.0
    phase_yx = phase[..., 2]
    weights_p_yx = weights_p[..., 2]
    phase_yx[idx] = 0.0
    weights_p_yx[idx] = 0.0

    # YY
    amps_yy = amplitude[..., 3]
    weights_yy = weights[..., 3]
    idx = np.where((~np.isfinite(amps_yy)))
    amps_yy[idx] = 1.0
    weights_yy[idx] = 0.0
    phase_yy = phase[..., 3]
    weights_p_yy = weights_p[..., 3]
    phase_yy[idx] = 0.0
    weights_p_yy[idx] = 0.0

    amplitude[..., 0] = amps_xx
    amplitude[..., 1] = amps_xy
    amplitude[..., 2] = amps_yx
    amplitude[..., 3] = amps_yy

    weights[..., 0] = weights_yy
    weights[..., 1] = weights_xy
    weights[..., 2] = weights_yx
    weights[..., 3] = weights_yy
    H.root.sol000.amplitude000.val[:] = amplitude
    H.root.sol000.amplitude000.weight[:] = weights

    phase[..., 0] = phase_xx
    phase[..., 1] = phase_xy
    phase[..., 2] = phase_yx
    phase[..., 3] = phase_yy

    weights_p[..., 0] = weights_p_yy
    weights_p[..., 1] = weights_p_xy
    weights_p[..., 2] = weights_p_yx
    weights_p[..., 3] = weights_p_yy
    H.root.sol000.phase000.val[:] = phase
    H.root.sol000.phase000.weight[:] = weights_p
    H.close()
    return
