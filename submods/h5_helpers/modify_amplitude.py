import numpy as np
import tables
from losoto import h5parm
from .slicing import get_double_slice, get_double_slice_len
import sys


def get_median_amp(h5):
    # assume pol-axis is present (can handle length, 1 (scalar), 2 (diagonal), or 4 (fulljones))
    H = tables.open_file(h5)
    amplitude = H.root.sol000.amplitude000.val[:]
    weights = H.root.sol000.amplitude000.weight[:]
    if amplitude.shape[-1] == 4:
        fulljones = True
    else:
        fulljones = False
    H.close()

    print('Amplitude and Weights shape:', weights.shape, amplitude.shape)
    amps_xx = amplitude[..., 0]
    amps_yy = amplitude[..., -1]  # so this also works for pol axis length 1
    weights_xx = weights[..., 0]
    weights_yy = weights[..., -1]

    idx_xx = np.where(weights_xx != 0.0)
    idx_yy = np.where(weights_yy != 0.0)
    amps_xx_tmp = amps_xx[idx_xx]
    amps_yy_tmp = amps_yy[idx_yy]

    idx_xx = np.where(amps_xx_tmp != 1.0)  # remove 1.0, these can be "resetsols" values
    idx_yy = np.where(amps_yy_tmp != 1.0)  # remove 1.0, these can be "resetsols" values
    if any(map(len, idx_xx)) and any(map(len, idx_yy)):
        medamps = 0.5 * (10 ** (np.nanmedian(np.log10(amps_xx_tmp[idx_xx]))) + 10 ** (
            np.nanmedian(np.log10(amps_yy_tmp[idx_yy]))))
    else:
        medamps = 1.
    print('Median  Stokes I amplitude of ', h5, ':', medamps)

    if fulljones:
        amps_xy = amplitude[..., 1]
        amps_yx = amplitude[..., 2]
        weights_xy = weights[..., 1]
        weights_yx = weights[..., 2]
        idx_xy = np.where(weights_xy != 0.0)
        idx_yx = np.where(weights_yx != 0.0)

        amps_xy_tmp = amps_xy[idx_xy]
        amps_yx_tmp = amps_yx[idx_yx]

        idx_xy = np.where(amps_xy_tmp > 0.0)
        idx_yx = np.where(amps_yx_tmp > 0.0)

        medamps_cross = 0.5 * (10 ** (np.nanmedian(np.log10(amps_xy_tmp[idx_xy]))) + 10 ** (
            np.nanmedian(np.log10(amps_yx_tmp[idx_yx]))))
        print('Median amplitude of XY+YX ', h5, ':', medamps_cross)

    print('Median Stokes I amplitude of ' + h5 + ': ' + str(medamps))
    return medamps


def get_all_antennas_from_h5list(h5list):
    """
    Get all unique antenna names in a list of h5 files.
    """

    # Initialize solution flags
    has_solutions = {
        'amplitude000': True,
        'phase000': True,
        'tec000': True,
        'rotation000': True,
        'rotationmeasure000': True
    }

    antennas = []

    # Check available solutions in the first file
    with tables.open_file(h5list[0], mode='a') as H:
        for sol_type in has_solutions.keys():
            try:
                antennas = getattr(H.root.sol000, sol_type).ant[:]
                break  # Use the first available solution type
            except tables.NoSuchNodeError:
                has_solutions[sol_type] = False  # Mark as unavailable

        print('Reading:', h5list[0], '-- Number of antennas:', len(antennas))

    # If there are additional files, continue gathering antennas
    for h5 in h5list[1:]:
        with tables.open_file(h5, mode='a') as H:
            for sol_type, available in has_solutions.items():
                if available:
                    try:
                        ants = getattr(H.root.sol000, sol_type).ant[:]
                        antennas = np.concatenate((antennas, ants.flatten()), axis=0)
                        break
                    except tables.NoSuchNodeError:
                        continue  # Try the next solution type if missing

            print('Reading:', h5, '-- Number of antennas:', len(ants))

    return np.unique(antennas)


def h5_has_dir(h5):
    """Check if any sol set contains 'dir' in its AXES attributes."""
    with tables.open_file(h5) as H:
        for sol_type in ['phase000', 'amplitude000', 'tec000', 'rotation000', 'rotationmeasure000']:
            try:
                if 'dir' in getattr(H.root.sol000, sol_type).val.attrs['AXES'].decode().split(','):
                    return True
            except AttributeError:
                continue
    return False


def normamplitudes_withmatrix(parmdblist):
    """
    Normalize global amplitues per direction and per antenna
    """
    H = tables.open_file(parmdblist[0])
    directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdb in the list
    H.close()

    allantenna = get_all_antennas_from_h5list(parmdblist)
    matrix_amps, matrix_weights = get_matrix_forampnorm(parmdblist)
    print(matrix_amps.shape)
    # sys.exit()
    # create norm factors
    norm_array = np.zeros((matrix_amps.shape[1], matrix_amps.shape[2]))
    # matrix_weights[h5_id,ant_id,dir_id]
    for ant_id, antenna in enumerate(allantenna):
        for dir_id, direction in enumerate(directions):
            norm_array[ant_id, dir_id] = np.sum(matrix_amps[:, ant_id, dir_id]) / np.sum(
                matrix_weights[:, ant_id, dir_id])

    # write result
    for h5_id, h5 in enumerate(parmdblist):
        H = tables.open_file(h5, mode='a')
        amps = H.root.sol000.amplitude000.val[:]
        weights = H.root.sol000.amplitude000.weight[:]
        ants_inh5 = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
        diraxis = axisn.index('dir')
        antaxis = axisn.index('ant')

        # matrix_weights[h5_id,ant_id,dir_id]
        for antenna in allantenna:
            ant_id = np.where(ants_inh5 == antenna)[0]
            print(h5, antenna)
            if len(ant_id) > 0:
                ant_id = ant_id[0]
                for dir_id, direction in enumerate(directions):
                    print(ant_id, dir_id, norm_array[ant_id, dir_id])
                    slice_obj = get_double_slice(amps, [ant_id, dir_id], [antaxis, diraxis])
                    amps[slice_obj] = 10 ** (np.log10(amps[slice_obj]) - norm_array[ant_id, dir_id])
        H.root.sol000.amplitude000.val[:] = amps
        H.flush()
        H.close()
    return


def normamplitudes(parmdb, norm_per_ms=False, norm_per_ant=False):
    """
    Normalize H5 amplitudes

    :param paramdb: list of h5 files
    :param norm_per_ms : boolean
    :param norm_per_ant : boolean

    return None
    """
    has_dir = h5_has_dir(parmdb[0])
    H = tables.open_file(parmdb[0])
    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
    if has_dir:
        directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdb in the list
    H.close()

    # ---------------------- THIS IS FOR norm_per_ant=False ----------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    if norm_per_ms and not norm_per_ant:
        for parmdbi in parmdb:
            H = tables.open_file(parmdbi, mode='a')
            if not has_dir:
                amps = H.root.sol000.amplitude000.val[:]
                weights = H.root.sol000.amplitude000.weight[:]
                idx = np.where(weights != 0.0)
                amps = np.log10(amps)
                # logger.info(parmdbi + ' Mean amplitudes before normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
                amps = amps - (np.nanmean(amps[idx]))
                # logger.info(parmdbi + ' Mean amplitudes after normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
                amps = 10 ** (amps)
                H.root.sol000.amplitude000.val[:] = amps
                H.flush()
                H.close()
            else:
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                diraxis = axisn.index('dir')
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]
                for dir_id, direction in enumerate(directions):
                    print('Normalizing direction', direction, 'Axis entry number', axisn.index('dir'))
                    slice_obj = [slice(None)] * diraxis + [dir_id] + [slice(None)] * (ampsfull.ndim - diraxis - 1)
                    amps = ampsfull[tuple(slice_obj)]
                    weights = weightsfull[tuple(slice_obj)]

                    idx = np.where(weights != 0.0)
                    amps = np.log10(amps)
                    # logger.info(
                    #     'Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                    #         10 ** (np.nanmean(amps[idx]))))
                    print('Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                        10 ** (np.nanmean(amps[idx]))))
                    amps = amps - (np.nanmean(amps[idx]))
                    # logger.info(
                    #     'Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes after normalization: ' + str(
                    #         10 ** (np.nanmean(amps[idx]))))
                    amps = 10 ** (amps)
                    # put back vales in ampsfull
                    ampsfull[tuple(slice_obj)] = amps

                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
        print('Return with norm_per_ant=False')
        return  # norm_per_ms

    if not has_dir and not norm_per_ant:
        for i, parmdbi in enumerate(parmdb):
            H = tables.open_file(parmdbi, mode='r')
            ampsfull = H.root.sol000.amplitude000.val[:]
            weightsfull = H.root.sol000.amplitude000.weight[:]
            idx = np.where(weightsfull != 0.0)
            # logger.info(parmdbi + '  Normfactor: ' + str(10 ** (np.nanmean(np.log10(ampsfull[idx])))))
            if i == 0:
                amps = np.ndarray.flatten(ampsfull[idx])
            else:
                amps = np.concatenate((amps, np.ndarray.flatten(ampsfull[idx])), axis=0)

            H.close()
        normmin = (np.nanmean(np.log10(amps)))
        # logger.info('Global normfactor: ' + str(10 ** normmin))
        # now write the new H5 files
        for parmdbi in parmdb:
            H = tables.open_file(parmdbi, mode='a')
            ampsfull = H.root.sol000.amplitude000.val[:]
            ampsfull = (np.log10(ampsfull)) - normmin
            ampsfull = 10 ** ampsfull
            H.root.sol000.amplitude000.val[:] = ampsfull
            H.flush()
            H.close()
    elif not norm_per_ant:
        for dir_id, direction in enumerate(directions):
            for i, parmdbi in enumerate(parmdb):
                H = tables.open_file(parmdbi, mode='r')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]
                diraxis = axisn.index('dir')
                H.close()

                slice_obj = [slice(None)] * diraxis + [dir_id] + [slice(None)] * (ampsfull.ndim - diraxis - 1)
                ampsi = ampsfull[tuple(slice_obj)]
                weights = weightsfull[tuple(slice_obj)]

                idx = np.where(weights != 0.0)
                if i == 0:
                    amps = np.ndarray.flatten(ampsi[idx])
                else:
                    amps = np.concatenate((amps, np.ndarray.flatten(ampsi[idx])), axis=0)
            normmin = (np.nanmean(np.log10(amps)))
            # logger.info('Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
            print('Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
            for parmdbi in parmdb:
                H = tables.open_file(parmdbi, mode='a')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                ampsfull = H.root.sol000.amplitude000.val[:]
                diraxis = axisn.index('dir')
                # put back vales in ampsfull
                slices = tuple(slice(None) if i != diraxis else dir_id for i in range(diraxis + 1))
                ampsfull[slices] = np.log10(ampsfull[slices]) - normmin
                ampsfull[slices] = 10 ** ampsfull[slices]
                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
        print('Return with norm_per_ant=False')
        return  # return because we do not need to go to the norm_per_an=True part

    # ---------------------- THIS IS FOR norm_per_ant=True ----------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    allantenna = get_all_antennas_from_h5list(parmdb)

    if norm_per_ms and norm_per_ant:
        for parmdbi in parmdb:
            H = tables.open_file(parmdbi, mode='a')
            ants_inh5 = H.root.sol000.amplitude000.ant[:]
            antaxis = axisn.index('ant')
            if not has_dir:
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]

                for antenna in allantenna:
                    ant_id = np.where(ants_inh5 == antenna)[0]
                    if len(ant_id) > 0:
                        ant_id = ant_id[0]
                        print('Doing:', antenna)
                        print('antenna index: ', )
                        slice_obj = [slice(None)] * antaxis + [ant_id] + [slice(None)] * (ampsfull.ndim - diraxis - 1)
                        amps = ampsfull[tuple(slice_obj)]
                        weights = weightsfull[tuple(slice_obj)]

                        idx = np.where(weights != 0.0)
                        amps = np.log10(amps)
                        # logger.info(str(antenna) + ' ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                        #     10 ** (np.nanmean(amps[idx]))))
                        amps = amps - (np.nanmean(amps[idx]))
                        # logger.info(str(antenna) + ' ' + parmdbi + ' Mean amplitudes after normalization: ' + str(
                        #     10 ** (np.nanmean(amps[idx]))))
                        amps = 10 ** (amps)

                        ampsfull[tuple(slice_obj)] = amps
                    else:
                        print('Skipping this antenna as it is not present in the h5', antenna)

                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
            else:
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                diraxis = axisn.index('dir')
                antaxis = axisn.index('ant')
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]

                for antenna in allantenna:
                    ant_id = np.where(ants_inh5 == antenna)[0]
                    if len(ant_id) > 0:
                        for dir_id, direction in enumerate(directions):
                            print(str(antenna) + ' ' + 'Normalizing direction', direction)

                            slice_obj = get_double_slice(ampsfull, [ant_id, dir_id], [antaxis, diraxis])

                            amps = ampsfull[slice_obj]
                            weights = weightsfull[slice_obj]

                            idx = np.where(weights != 0.0)
                            amps = np.log10(amps)
                            # logger.info(str(antenna) + ' ' + 'Direction:' + str(
                            #     dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                            #     10 ** (np.nanmean(amps[idx]))))
                            print('Direction:' + str(
                                dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                                10 ** (np.nanmean(amps[idx]))))
                            amps = amps - (np.nanmean(amps[idx]))
                            # logger.info(str(antenna) + ' ' + 'Direction:' + str(
                            #     dir_id) + '  ' + parmdbi + ' Mean amplitudes after normalization: ' + str(
                            #     10 ** (np.nanmean(amps[idx]))))
                            amps = 10 ** (amps)
                            # put back vales in ampsfull
                            ampsfull[slice_obj] = amps
                    else:
                        print('Skipping this antenna as it is not present in the h5', antenna)

                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
        print('Return with norm_per_ant=True')
        return  # norm_per_ms

    if not has_dir and norm_per_ant:
        print('Not implemented  "if not has_dir and norm_per_ant" ')
        sys.exit()

    elif norm_per_ant:
        for antenna in allantenna:
            for dir_id, direction in enumerate(directions):
                if 'amps' in locals():
                    del amps
                for i, parmdbi in enumerate(parmdb):
                    H = tables.open_file(parmdbi, mode='r')
                    ants_inh5 = H.root.sol000.amplitude000.ant[:]
                    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                    # ampsfull = H.root.sol000.amplitude000.val[:]
                    # weightsfull = H.root.sol000.amplitude000.weight[:]
                    diraxis = axisn.index('dir')
                    antaxis = axisn.index('ant')

                    ant_id = np.where(ants_inh5 == antenna)[0]
                    if len(ant_id) > 0:
                        slice_obj = get_double_slice_len(axisn, [ant_id, dir_id], [antaxis, diraxis])
                        ampsi = H.root.sol000.amplitude000.val[slice_obj]
                        weights = H.root.sol000.amplitude000.weight[slice_obj]

                        idx = np.where(weights != 0.0)
                        if 'amps' in locals():  # in this case amps was already made
                            amps = np.concatenate((amps, np.ndarray.flatten(ampsi[idx])), axis=0)
                        else:  # create amps
                            amps = np.ndarray.flatten(ampsi[idx])
                    H.close()

                normmin = (np.nanmean(np.log10(amps)))

                # logger.info(str(antenna) + ' Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
                print(str(antenna) + ' Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
                for parmdbi in parmdb:
                    print('Write to:', parmdbi)
                    H = tables.open_file(parmdbi, mode='a')
                    ants_inh5 = H.root.sol000.amplitude000.ant[:]
                    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                    # ampsfull = H.root.sol000.amplitude000.val[:]
                    diraxis = axisn.index('dir')
                    antaxis = axisn.index('ant')

                    # put back vales in ampsfull
                    ant_id = np.where(ants_inh5 == antenna)[0]
                    if len(ant_id) > 0:
                        slice_obj = get_double_slice_len(axisn, [ant_id, dir_id], [antaxis, diraxis])
                        ampsfull = H.root.sol000.amplitude000.val[slice_obj]

                        ampsfull = 10 ** (np.log10(ampsfull) - normmin)
                        # ampsfull = 10**ampsfull
                        H.root.sol000.amplitude000.val[slice_obj] = ampsfull
                    H.flush()
                    H.close()
        # else:
        #    print('Skipping this antenna as it is not present in the h5', antenna)

    print('Return with norm_per_ant=True')
    return


def normamplitudes_old2(parmdb, norm_per_ms=False, norm_per_ant=False):
    has_dir = h5_has_dir(parmdb[0])
    H = tables.open_file(parmdb[0])
    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
    if has_dir:
        directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdb in the list
    H.close()

    if norm_per_ms:
        for parmdbi in parmdb:
            H = tables.open_file(parmdbi, mode='a')
            if not has_dir:
                amps = H.root.sol000.amplitude000.val[:]
                weights = H.root.sol000.amplitude000.weight[:]
                idx = np.where(weights != 0.0)
                amps = np.log10(amps)
                # logger.info(parmdbi + ' Mean amplitudes before normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
                amps = amps - (np.nanmean(amps[idx]))
                # logger.info(parmdbi + ' Mean amplitudes after normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
                amps = 10 ** (amps)
                H.root.sol000.amplitude000.val[:] = amps
                H.flush()
                H.close()
            else:
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                diraxis = axisn.index('dir')
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]
                for dir_id, direction in enumerate(directions):
                    print('Normalizing direction', direction, 'Axis entry number', axisn.index('dir'))
                    if diraxis == 0:
                        amps = ampsfull[dir_id, ...]
                        weights = weightsfull[dir_id, ...]
                    if diraxis == 1:
                        amps = ampsfull[:, dir_id, ...]
                        weights = weightsfull[:, dir_id, ...]
                    if diraxis == 2:
                        amps = ampsfull[:, :, dir_id, ...]
                        weights = weightsfull[:, :, dir_id, ...]
                    if diraxis == 3:
                        amps = ampsfull[:, :, :, dir_id, ...]
                        weights = weightsfull[:, :, :, dir_id, ...]
                    if diraxis == 4:
                        amps = ampsfull[:, :, :, :, dir_id, ...]
                        weights = weightsfull[:, :, :, :, dir_id, ...]

                    # amp[get_double_slice(amp, [antennaid, dirid], [antennaxis, diraxis])

                    idx = np.where(weights != 0.0)
                    amps = np.log10(amps)
                    # logger.info(
                    #     'Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                    #         10 ** (np.nanmean(amps[idx]))))
                    print('Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes before normalization: ' + str(
                        10 ** (np.nanmean(amps[idx]))))
                    amps = amps - (np.nanmean(amps[idx]))
                    # logger.info(
                    #     'Direction:' + str(dir_id) + '  ' + parmdbi + ' Mean amplitudes after normalization: ' + str(
                    #         10 ** (np.nanmean(amps[idx]))))
                    amps = 10 ** (amps)

                    # put back vales in ampsfull
                    if diraxis == 0:
                        ampsfull[dir_id, ...] = amps
                    if diraxis == 1:
                        ampsfull[:, dir_id, ...] = amps
                    if diraxis == 2:
                        ampsfull[:, :, dir_id, ...] = amps
                    if diraxis == 3:
                        ampsfull[:, :, :, dir_id, ...] = amps
                    if diraxis == 4:
                        ampsfull[:, :, :, :, dir_id, ...] = amps
                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
        return  # norm_per_ms

    if not has_dir:
        for i, parmdbi in enumerate(parmdb):
            H = tables.open_file(parmdbi, mode='r')
            ampsfull = H.root.sol000.amplitude000.val[:]
            weightsfull = H.root.sol000.amplitude000.weight[:]
            idx = np.where(weightsfull != 0.0)
            # logger.info(parmdbi + '  Normfactor: ' + str(10 ** (np.nanmean(np.log10(ampsfull[idx])))))
            if i == 0:
                amps = np.ndarray.flatten(ampsfull[idx])
            else:
                amps = np.concatenate((amps, np.ndarray.flatten(ampsfull[idx])), axis=0)

            H.close()
        normmin = (np.nanmean(np.log10(amps)))
        # logger.info('Global normfactor: ' + str(10 ** normmin))
        # now write the new H5 files
        for parmdbi in parmdb:
            H = tables.open_file(parmdbi, mode='a')
            ampsfull = H.root.sol000.amplitude000.val[:]
            ampsfull = (np.log10(ampsfull)) - normmin
            ampsfull = 10 ** ampsfull
            H.root.sol000.amplitude000.val[:] = ampsfull
            H.flush()
            H.close()
    else:
        for dir_id, direction in enumerate(directions):
            for i, parmdbi in enumerate(parmdb):
                H = tables.open_file(parmdbi, mode='r')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                ampsfull = H.root.sol000.amplitude000.val[:]
                weightsfull = H.root.sol000.amplitude000.weight[:]
                diraxis = axisn.index('dir')
                H.close()
                if diraxis == 0:
                    ampsi = ampsfull[dir_id, ...]
                    weights = weightsfull[dir_id, ...]
                if diraxis == 1:
                    ampsi = ampsfull[:, dir_id, ...]
                    weights = weightsfull[:, dir_id, ...]
                if diraxis == 2:
                    ampsi = ampsfull[:, :, dir_id, ...]
                    weights = weightsfull[:, :, dir_id, ...]
                if diraxis == 3:
                    ampsi = ampsfull[:, :, :, dir_id, ...]
                    weights = weightsfull[:, :, :, dir_id, ...]
                if diraxis == 4:
                    ampsi = ampsfull[:, :, :, :, dir_id, ...]
                    weights = weightsfull[:, :, :, :, dir_id, ...]
                idx = np.where(weights != 0.0)
                if i == 0:
                    amps = np.ndarray.flatten(ampsi[idx])
                else:
                    amps = np.concatenate((amps, np.ndarray.flatten(ampsi[idx])), axis=0)
            normmin = (np.nanmean(np.log10(amps)))
            # logger.info('Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
            print('Global normfactor directon:' + str(dir_id) + ' ' + str(10 ** normmin))
            for parmdbi in parmdb:
                H = tables.open_file(parmdbi, mode='a')
                axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
                ampsfull = H.root.sol000.amplitude000.val[:]
                diraxis = axisn.index('dir')
                # put back vales in ampsfull
                if diraxis == 0:
                    ampsfull[dir_id, ...] = (np.log10(ampsfull[dir_id, ...])) - normmin
                    ampsfull[dir_id, ...] = 10 ** ampsfull[dir_id, ...]
                if diraxis == 1:
                    ampsfull[:, dir_id, ...] = (np.log10(ampsfull[:, dir_id, ...])) - normmin
                    ampsfull[:, dir_id, ...] = 10 ** ampsfull[:, dir_id, ...]
                if diraxis == 2:
                    ampsfull[:, :, dir_id, ...] = (np.log10(ampsfull[:, :, dir_id, ...])) - normmin
                    ampsfull[:, :, dir_id, ...] = 10 ** ampsfull[:, :, dir_id, ...]
                if diraxis == 3:
                    ampsfull[:, :, :, dir_id, ...] = (np.log10(ampsfull[:, :, :, dir_id, ...])) - normmin
                    ampsfull[:, :, :, dir_id, ...] = 10 ** ampsfull[:, :, :, dir_id, ...]
                if diraxis == 4:
                    ampsfull[:, :, :, :, dir_id, ...] = (np.log10(ampsfull[:, :, :, :, dir_id, ...])) - normmin
                    ampsfull[:, :, :, :, dir_id, ...] = 10 ** ampsfull[:, :, :, :, dir_id, ...]
                H.root.sol000.amplitude000.val[:] = ampsfull
                H.flush()
                H.close()
    return


def normamplitudes_old(parmdb, norm_per_ms=False):
    """
    normalize amplitude solutions to one
    """

    if norm_per_ms:
        for parmdbi in parmdb:
            H5 = h5parm.h5parm(parmdbi, readonly=False)
            amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
            weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
            idx = np.where(weights != 0.0)

            amps = np.log10(amps)
            # logger.info(parmdbi + ' Mean amplitudes before normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
            amps = amps - (np.nanmean(amps[idx]))
            # logger.info(parmdbi + ' Mean amplitudes after normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
            amps = 10 ** (amps)

            H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)
            H5.close()
        return

    if len(parmdb) == 1:
        H5 = h5parm.h5parm(parmdb[0], readonly=False)
        amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
        weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
        idx = np.where(weights != 0.0)

        amps = np.log10(amps)
        # logger.info('Mean amplitudes before normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
        amps = amps - (np.nanmean(amps[idx]))
        # logger.info('Mean amplitudes after normalization: ' + str(10 ** (np.nanmean(amps[idx]))))
        amps = 10 ** (amps)

        H5.getSolset('sol000').getSoltab('amplitude000').setValues(amps)
        H5.close()

    else:
        # amps = []
        for i, parmdbi in enumerate(parmdb):
            H5 = h5parm.h5parm(parmdbi, readonly=True)
            ampsi = np.copy(H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0])
            weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]
            idx = np.where(weights != 0.0)
            # logger.info(parmdbi + '  Normfactor: ' + str(10 ** (np.nanmean(np.log10(ampsi[idx])))))
            if i == 0:
                amps = np.ndarray.flatten(ampsi[idx])
            else:
                amps = np.concatenate((amps, np.ndarray.flatten(ampsi[idx])), axis=0)

            # print np.shape(amps), parmdbi
            H5.close()
        normmin = (np.nanmean(np.log10(amps)))
        # logger.info('Global normfactor: ' + str(10 ** normmin))
        # now write the new H5 files
        for parmdbi in parmdb:
            H5 = h5parm.h5parm(parmdbi, readonly=False)
            ampsi = np.copy(H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0])
            ampsi = (np.log10(ampsi)) - normmin
            ampsi = 10 ** ampsi
            H5.getSolset('sol000').getSoltab('amplitude000').setValues(ampsi)
            H5.close()
    return


def get_matrix_forampnorm(parmdblist):
    assert type(parmdblist) == list, 'input is not list'  # use only lists as input
    allantenna = get_all_antennas_from_h5list(parmdblist)
    with tables.open_file(parmdblist[0]) as H:
        directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdblist in the list

    N_h5 = len(parmdblist)
    N_ant = len(allantenna)
    N_direction = len(directions)

    matrix_amps = np.zeros((N_h5, N_ant, N_direction))  # sum of log10 of amps
    matrix_weights = np.zeros((N_h5, N_ant, N_direction))  # number if input values
    # matrix_slope = np.zeros((N_h5,N_ant,N_direction)) # of mean slope log10 amps

    for h5_id, h5 in enumerate(parmdblist):
        H = tables.open_file(h5)
        amps = H.root.sol000.amplitude000.val[:]
        weights = H.root.sol000.amplitude000.weight[:]
        ants_inh5 = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
        diraxis = axisn.index('dir')
        antaxis = axisn.index('ant')
        H.close()

        for antenna in allantenna:
            ant_id = np.where(ants_inh5 == antenna)[0]
            print(h5, antenna)
            if len(ant_id) > 0:
                ant_id = ant_id[0]
                for dir_id, direction in enumerate(directions):
                    # print(str(antenna) + ' ' +'Normalizing direction', direction)
                    slice_obj = get_double_slice(amps, [ant_id, dir_id], [antaxis, diraxis])
                    amps_sel = amps[slice_obj]
                    weights_sel = weights[slice_obj]
                    idx = np.where(weights_sel != 0.0)
                    weights_sel[idx] = 1.0

                    matrix_amps[h5_id, ant_id, dir_id] = np.sum(np.log10(amps_sel[idx]))
                    matrix_weights[h5_id, ant_id, dir_id] = np.sum(weights_sel[idx])
                    # print(np.sum(weights_sel[idx]))
                    # matrix_slope[h5_id,ant_id,dir_id] = np.mean(np.diff(np.log10(amps_sel[idx])))
    return matrix_amps, matrix_weights


def normslope_withmatrix(parmdblist):
    """
    Global slope normalization per direction
    input: list of h5 solution files
    """
    assert type(parmdblist) == list, 'input is not list'  # use only lists as input
    H = tables.open_file(parmdblist[0])
    directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdb in the list
    H.close()

    matrix_slope, matrix_weights = get_matrix_forslopenorm(parmdblist)

    # create norm factors
    norm_array = np.zeros((matrix_slope.shape[1]))
    # matrix_weights[h5_id,ant_id,dir_id]
    for dir_id, direction in enumerate(directions):
        norm_array[dir_id] = np.average(matrix_slope[:, dir_id], weights=matrix_weights[:, dir_id])

    # write result
    for h5_id, h5 in enumerate(parmdblist):
        H = tables.open_file(h5, mode='a')
        ampsfull = H.root.sol000.amplitude000.val[:]
        weightsfull = H.root.sol000.amplitude000.weight[:]
        ants_inh5 = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
        diraxis = axisn.index('dir')
        freqaxis = axisn.index('freq')
        frequencies = H.root.sol000.amplitude000.freq[:]

        for dir_id, direction in enumerate(directions):
            print(h5, dir_id, norm_array[dir_id])
            for freq_id, frequency in enumerate(frequencies):
                slice_obj = get_double_slice(ampsfull, [freq_id, dir_id], [freqaxis, diraxis])

                amps = ampsfull[slice_obj]
                ampsfull[slice_obj] = 10 ** (np.log10(amps) - (norm_array[dir_id] * float(freq_id)))

        H.root.sol000.amplitude000.val[:] = ampsfull
        H.flush()
        H.close()
    return


def get_matrix_forslopenorm(parmdblist):
    assert type(parmdblist) == list, 'input is not list'  # use only lists as input

    H = tables.open_file(parmdblist[0])
    directions = H.root.sol000.amplitude000.dir[:]  # should be the same for all parmdblist in the list
    H.close()

    N_h5 = len(parmdblist)
    # N_ant = len(allantenna)
    N_direction = len(directions)

    # matrix_amps = np.zeros((N_h5,N_direction)) # sum of log10 of amps
    matrix_weights = np.zeros((N_h5, N_direction))  # number if input values
    matrix_slope = np.zeros((N_h5, N_direction))  # of mean slope log10 amps

    for h5_id, h5 in enumerate(parmdblist):
        H = tables.open_file(h5)
        ampsfull = H.root.sol000.amplitude000.val[:]
        # weightsfull = H.root.sol000.amplitude000.weight[:]
        ants_inh5 = H.root.sol000.amplitude000.ant[:]
        axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
        diraxis = axisn.index('dir')
        freqaxis = axisn.index('freq')
        H.close()

        for dir_id, direction in enumerate(directions):
            slice_obj = [slice(None)] * diraxis + [dir_id] + [slice(None)] * (ampsfull.ndim - diraxis - 1)
            amps = ampsfull[tuple(slice_obj)]
            # weights = weightsfull[tuple(slice_obj)]

            matrix_slope[h5_id, dir_id] = np.mean(np.diff(np.log10(amps), axis=freqaxis))
            matrix_weights[h5_id, dir_id] = amps.size / N_direction
            print('Direction:', dir_id, 'Slope:', matrix_slope[h5_id, dir_id])
    return matrix_slope, matrix_weights
