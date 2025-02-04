import numpy as np
from losoto import h5parm
import tables


def flaglowamps_fulljones(parmdb, lowampval=0.1, flagging=True, setweightsphases=True):
    """
    flag bad amplitudes in H5 parmdb, those with values < lowampval
    assume pol-axis is present (can handle length 2 (diagonal), or 4 (fulljones))
    """
    H5 = h5parm.h5parm(parmdb, readonly=False)
    amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    amps_xx = amps[..., 0]
    amps_yy = amps[..., -1]  # so this also works for pol axis length 1
    weights_xx = weights[..., 0]
    weights_yy = weights[..., -1]
    idx_xx = np.where(amps_xx < lowampval)
    idx_yy = np.where(amps_yy < lowampval)

    if flagging:  # no flagging
        weights_xx[idx_xx] = 0.0
        weights_yy[idx_yy] = 0.0
        print('Settting some weights to zero in flaglowamps_fulljones')

    amps_xx[idx_xx] = 1.0
    amps_yy[idx_yy] = 1.0

    weights[..., 0] = weights_xx
    weights[..., -1] = weights_yy
    amps[..., 0] = amps_xx
    amps[..., -1] = amps_yy

    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights, weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        phases_xx = phases[..., 0]
        phases_yy = phases[..., -1]
        weights_p_xx = weights_p[..., 0]
        weights_p_yy = weights_p[..., -1]

        if flagging:  # no flagging
            weights_p_xx[idx_xx] = 0.0
            weights_p_yy[idx_yy] = 0.0
            phases_xx[idx_xx] = 0.0
            phases_yy[idx_yy] = 0.0

            weights_p[..., 0] = weights_p_xx
            weights_p[..., -1] = weights_p_yy
            phases[..., 0] = phases_xx
            phases[..., -1] = phases_yy

            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p, weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)
    H5.close()
    return


def flaglowamps(parmdb, lowampval=0.1, flagging=True, setweightsphases=True):
    """
    flag bad amplitudes in H5 parmdb, those with values < lowampval
    """
    H5 = h5parm.h5parm(parmdb, readonly=False)
    amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    idx = np.where(amps < lowampval)
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    if flagging:
        weights[idx] = 0.0
        print('Settting some weights to zero in flaglowamps')
    amps[idx] = 1.0
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights, weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        if flagging:
            weights_p[idx] = 0.0
            phases[idx] = 0.0
            # print(idx)
            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p, weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)

    # H5.getSolset('sol000').getSoltab('phase000').flush()
    # H5.getSolset('sol000').getSoltab('amplitude000').flush()
    H5.close()
    return


def flaghighamps(parmdb, highampval=10., flagging=True, setweightsphases=True):
    """
    flag bad amplitudes in H5 parmdb, those with values > highampval
    """
    H5 = h5parm.h5parm(parmdb, readonly=False)
    amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    idx = np.where(amps > highampval)
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    if flagging:
        weights[idx] = 0.0
        print('Settting some weights to zero in flaghighamps')
    amps[idx] = 1.0
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights, weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        if flagging:
            weights_p[idx] = 0.0
            phases[idx] = 0.0
            # print(idx)
            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p, weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)

    # H5.getSolset('sol000').getSoltab('phase000').flush()
    # H5.getSolset('sol000').getSoltab('amplitude000').flush()
    H5.close()
    return


def flaghighamps_fulljones(parmdb, highampval=10., flagging=True, setweightsphases=True):
    """
    flag bad amplitudes in H5 parmdb, those with values > highampval
    assume pol-axis is present (can handle 2 (diagonal), or 4 (fulljones))
    """
    H5 = h5parm.h5parm(parmdb, readonly=False)
    amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
    weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    amps_xx = amps[..., 0]
    amps_yy = amps[..., -1]  # so this also works for pol axis length 1
    weights_xx = weights[..., 0]
    weights_yy = weights[..., -1]
    idx_xx = np.where(amps_xx > highampval)
    idx_yy = np.where(amps_yy > highampval)

    if flagging:  # no flagging
        weights_xx[idx_xx] = 0.0
        weights_yy[idx_yy] = 0.0
        print('Settting some weights to zero in flaghighamps_fulljones')

    amps_xx[idx_xx] = 1.0
    amps_yy[idx_yy] = 1.0

    weights[..., 0] = weights_xx
    weights[..., -1] = weights_yy
    amps[..., 0] = amps_xx
    amps[..., -1] = amps_yy

    H5.getSolset('sol000').getSoltab('amplitude000').setValues(weights, weight=True)
    H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)

    # also put phases weights and phases to zero
    if setweightsphases:
        phases = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]
        weights_p = H5.getSolset('sol000').getSoltab('phase000').getValues(weight=True)[0]
        phases_xx = phases[..., 0]
        phases_yy = phases[..., -1]
        weights_p_xx = weights_p[..., 0]
        weights_p_yy = weights_p[..., -1]

        if flagging:  # no flagging
            weights_p_xx[idx_xx] = 0.0
            weights_p_yy[idx_yy] = 0.0
            phases_xx[idx_xx] = 0.0
            phases_yy[idx_yy] = 0.0

            weights_p[..., 0] = weights_p_xx
            weights_p[..., -1] = weights_p_yy
            phases[..., 0] = phases_xx
            phases[..., -1] = phases_yy

            H5.getSolset('sol000').getSoltab('phase000').setValues(weights_p, weight=True)
            H5.getSolset('sol000').getSoltab('phase000').setValues(phases)
    H5.close()
    return


def flag_bad_amps(parmdb, setweightsphases=True, flagamp1=True, flagampxyzero=True):
    """
    flag bad amplitudes in H5 parmdb, those with amplitude==1.0
    """
    # check if full jones
    H = tables.open_file(parmdb, mode='a')
    amplitude = H.root.sol000.amplitude000.val[:]
    weights = H.root.sol000.amplitude000.weight[:]
    if amplitude.shape[-1] == 4:
        fulljones = True
    else:
        fulljones = False

    if not fulljones:
        idx = np.where(amplitude <= 0.0)
        amplitude[idx] = 1.0
        if flagamp1:
            idx = np.where(amplitude == 1.0)
            weights[idx] = 0.0

        H.root.sol000.amplitude000.val[:] = amplitude
        H.root.sol000.amplitude000.weight[:] = weights
        # also put phases weights and phases to zero
        if setweightsphases:
            phase = H.root.sol000.phase000.val[:]
            weights_p = H.root.sol000.phase000.weight[:]
            phase[idx] = 0.0
            weights_p[idx] = 0.0
            H.root.sol000.phase000.val[:] = phase
            H.root.sol000.phase000.weight[:] = weights_p
        H.close()

    if fulljones:
        if setweightsphases:
            phase = H.root.sol000.phase000.val[:]
            weights_p = H.root.sol000.phase000.weight[:]

        # XX
        amps_xx = amplitude[..., 0]
        weights_xx = weights[..., 0]
        idx = np.where(amps_xx <= 0.0)
        amps_xx[idx] = 1.0
        if flagamp1:
            idx = np.where(amps_xx == 1.0)
            weights_xx[idx] = 0.0
        if setweightsphases:
            phase_xx = phase[..., 0]
            weights_p_xx = weights_p[..., 0]
            phase_xx[idx] = 0.0
            weights_p_xx[idx] = 0.0

        # XY
        amps_xy = amplitude[..., 1]
        weights_xy = weights[..., 1]
        idx = np.where(amps_xy == 1.0)
        amps_xy[idx] = 0.0
        if flagampxyzero:
            idx = np.where(amps_xy == 0.0)  # we do not want this if we resetsols
            weights_xy[idx] = 0.0
        if setweightsphases:
            phase_xy = phase[..., 1]
            weights_p_xy = weights_p[..., 1]
            phase_xy[idx] = 0.0
            weights_p_xy[idx] = 0.0

        # YX
        amps_yx = amplitude[..., 2]
        weights_yx = weights[..., 2]
        idx = np.where(amps_yx == 1.0)
        amps_yx[idx] = 0.0
        if flagampxyzero:
            idx = np.where(amps_yx == 0.0)
            weights_yx[idx] = 0.0
        if setweightsphases:
            phase_yx = phase[..., 2]
            weights_p_yx = weights_p[..., 2]
            phase_yx[idx] = 0.0
            weights_p_yx[idx] = 0.0

        # YY
        amps_yy = amplitude[..., 3]
        weights_yy = weights[..., 3]
        idx = np.where(amps_yy <= 0.0)
        amps_yy[idx] = 1.0
        if flagamp1:
            idx = np.where(amps_yy == 1.0)
            weights_yy[idx] = 0.0
        if setweightsphases:
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

        if setweightsphases:
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
