"""Modify phase solutions in h5parm files (reference antenna, phase-diff and
CDF handling, copying scalar phases)."""
import os
import shutil
from pathlib import Path

import numpy as np
import tables
from losoto import h5parm  # type: ignore

from ..h5_merger import merge_h5
from ..runtime_utils import get_logger, terminal_print


def makephasediffh5(phaseh5, refant):
    """
    Adjusts the phase solutions in a HDF5 file by referencing them to a specified reference antenna.

    This function opens the given HDF5 file containing phase calibration solutions, computes the phase
    difference with respect to the specified reference antenna, and updates the phase solutions accordingly.
    It is intended for use with scalarphase/phaseonly solutions and does not support tecandphase solutions
    where the frequency axis is missing.

    Parameters
    ----------
    phaseh5 : str
        Path to the HDF5 file containing phase calibration solutions.
    refant : str
        Name of the reference antenna to which phases will be referenced.

    Notes
    -----
    - The function modifies the HDF5 file in place.
    - Only the XX polarization is updated; the YY polarization is set to zero.
    - The function assumes the structure of the HDF5 file matches the expected LOFAR calibration format.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5pol = tables.open_file(phaseh5, mode='a')

    phase_pol = H5pol.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    phase_pol_tmp = np.copy(phase_pol)
    # antenna   = H5pol.root.sol000.phase000.ant[:]
    terminal_print('Shape to make phase diff array', phase_pol.shape)
    terminal_print('Using refant:', refant)
    get_logger().info('Refant for XX/YY or RR/LL phase-referencing' + refant)

    # Reference phases so that we correct the phase difference with respect to a reference station
    refant_idx = np.where(H5pol.root.sol000.phase000.ant[:].astype(str) == refant)
    phase_pol_tmp_ref = phase_pol_tmp - phase_pol_tmp[:, :, refant_idx[0], :, :]

    phase_pol[:, :, :, :, 0] = phase_pol_tmp_ref[:, :, :, :, 0]  # XX
    phase_pol[:, :, :, :, -1] = 0.0 * phase_pol_tmp_ref[:, :, :, :, 0]  # YY

    H5pol.root.sol000.phase000.val[:] = phase_pol
    H5pol.flush()
    H5pol.close()
    return


def phasediffh5_to_linear(phaseh5):
    """
    Convert a phase-difference H5 (from makephasediffh5) to a linear full-Jones H5 for linear data.

    The correction diag(exp(i*phi), 1) on (R, L) becomes, in linear, a rotation by phi/2 times a
    common phase exp(i*phi/2), so it has to be applied as a full-Jones matrix.

    Parameters
    ----------
    phaseh5 : str
        Path to the phase-difference H5 file (circular, pol axis RR, LL).

    Returns
    -------
    str
        Path to the linear full-Jones H5, ``linearfulljones_<name>`` next to the input.
    """
    h5_lin = os.path.join(os.path.dirname(phaseh5), 'linearfulljones_' + os.path.basename(phaseh5))
    Path(h5_lin).unlink(missing_ok=True)
    terminal_print('Converting phase-difference solutions to linear full-Jones:', h5_lin)
    merge_h5(h5_out=h5_lin, h5_tables=phaseh5, circ2lin=True, propagate_weights=True)
    return h5_lin


def makephaseCDFh5(phaseh5, backup=True, testscfactor=1.):
    """
    Convert phase solutions to cumulative frequency-difference values.

    Parameters
    ----------
    phaseh5 : str
        H5 solution file to modify.
    backup : bool, optional
        Create a ``.psbackup`` copy first.
    testscfactor : float, optional
        Scaling factor for cumulative phases.

    Returns
    -------
    None
        The H5 file is modified in place.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    if backup:
        if os.path.isfile(phaseh5 + '.psbackup'):
            Path(phaseh5 + '.psbackup').unlink(missing_ok=True)
        shutil.copy(phaseh5, phaseh5 + '.psbackup')    

    H5 = tables.open_file(phaseh5, mode='a')

    phaseCDF = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    terminal_print('Shape to make phase CDF array', phaseCDF.shape)
    nfreq = len(H5.root.sol000.phase000.freq[:])
    for ff in range(nfreq - 1):
        # reverse order so phase increase towards lower frequnecies
        phaseCDF[:, nfreq - ff - 2, ...] = np.copy(
            phaseCDF[:, nfreq - ff - 2, ...] + (testscfactor * phaseCDF[:, nfreq - ff - 1, ...]))

    terminal_print('Phase CDF array shape:', phaseCDF.shape)
    H5.root.sol000.phase000.val[:] = phaseCDF
    H5.flush()
    H5.close()
    return


def makephaseCDFh5_h5merger(phaseh5, ms, modeldatacolumns, backup=True, testscfactor=1.):
    """
    Merge phase solutions and convert them to cumulative phase values.

    Parameters
    ----------
    phaseh5 : str
        H5 solution file to merge and modify.
    ms : str or list of str
        Measurement Set input for H5 merging.
    modeldatacolumns : list
        Model columns used to determine merge mode.
    backup : bool, optional
        Create a ``.psbackup`` copy first.
    testscfactor : float, optional
        Scaling factor for cumulative phases.

    Returns
    -------
    None
        The merged H5 file is modified in place.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    # if soltypein == 'scalarphase_slope':
    #   single_pol_merge = True
    # if soltypein == 'phaseonly_slope':
    #   single_pol = False
    # if soltypein not in ['scalarphase_slope','phaseonly_slope']:
    #   print('Wrong input in makephaseCDFh5_h5merger')
    #   sys.exit()

    if len(modeldatacolumns) > 1:
        merge_all_in_one = False
    else:
        merge_all_in_one = True

    if backup:
        if os.path.isfile(phaseh5 + '.psbackup'):
            Path(phaseh5 + '.psbackup').unlink(missing_ok=True)
        shutil.copy(phaseh5, phaseh5 + '.psbackup')
    # going to overwrite phaseh5, first move to new file
    if os.path.isfile(phaseh5 + '.in'):
        Path(phaseh5 + '.in').unlink(missing_ok=True)
    Path(phaseh5).rename(f"{phaseh5}.in")
    
    merge_h5(h5_out=phaseh5, h5_tables=phaseh5 + '.in', ms_files=ms, merge_all_in_one=merge_all_in_one,
             propagate_weights=True)
    H5 = tables.open_file(phaseh5, mode='a')

    phaseCDF = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    phaseCDF_tmp = np.copy(phaseCDF)
    terminal_print('Shape to make phase CDF array', phaseCDF.shape)
    nfreq = len(H5.root.sol000.phase000.freq[:])
    for ff in range(nfreq - 1):
        # reverse order so phase increase towards lower frequnecies
        phaseCDF[:, nfreq - ff - 2, ...] = np.copy(
            phaseCDF[:, nfreq - ff - 2, ...] + (testscfactor * phaseCDF[:, nfreq - ff - 1, ...]))

    terminal_print('Phase CDF array shape:', phaseCDF.shape)
    H5.root.sol000.phase000.val[:] = phaseCDF
    H5.flush()
    H5.close()
    return


def copyoverscalarphase(scalarh5, phasexxyyh5):
    """
    Copies scalar phase solutions from a HDF5 file (scalarh5) to the XX and YY polarization axes
    of another HDF5 file (phasexxyyh5).

    This function is intended for use with scalar phase or phase-only solutions, and does not work
    for tecandphase solutions where the frequency axis is missing for phase000.

    Parameters
    ----------
    scalarh5 : str
        Path to the input HDF5 file containing scalar phase solutions.
    phasexxyyh5 : str
        Path to the HDF5 file with XX and YY polarization axes to be updated.

    Notes
    -----
    - The function assumes the input files follow the structure produced by LOFAR calibration software.
    - The phase values from the scalar file are copied to both the XX (pol=0) and YY (pol=-1) axes
    for each antenna and direction.
    - The function modifies the phasexxyyh5 file in place.
    """
    # note for scalarphase/phaseonly solve, does not work for tecandphase as freq axis is missing there for phase000
    H5 = tables.open_file(scalarh5, mode='r')
    H5pol = tables.open_file(phasexxyyh5, mode='a')

    phase = H5.root.sol000.phase000.val[:]  # time, freq, ant, dir
    phase_pol = H5pol.root.sol000.phase000.val[:]  # time, freq, ant, dir, pol
    antenna = H5.root.sol000.phase000.ant[:]
    terminal_print('Shapes for pol copy', phase.shape, phase_pol.shape)

    for ant in range(len(antenna)):
        phase_pol[:, :, ant, :, 0] = phase[:, :, ant, :]  # XX
        phase_pol[:, :, ant, :, -1] = phase[:, :, ant, :]  # YY

    H5pol.root.sol000.phase000.val[:] = phase_pol[:, :, :]
    H5pol.flush()

    H5.close()
    H5pol.close()
    return


def fix_phasereference(h5parm, refant):
    """
    Phase reference values with respect to a reference station

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file containing phase solutions.
    refant : str
        Reference antenna name used to phase-reference the solutions.
    """

    H = tables.open_file(h5parm, mode='a')

    axisn = H.root.sol000.phase000.val.attrs['AXES'].decode().split(',')

    phase = H.root.sol000.phase000.val[:]
    refant_idx = np.where(H.root.sol000.phase000.ant[:].astype(str) == refant)  # to deal with byte strings
    terminal_print('Reference antenna index:', refant_idx, 'Reference antenna name:', refant)
    antennaxis = axisn.index('ant')

    terminal_print('Referencing phase to ', refant, 'Axis entry number', axisn.index('ant'))
    if antennaxis == 0:
        phasen = phase - phase[refant_idx[0], ...]
    if antennaxis == 1:
        phasen = phase - phase[:, refant_idx[0], ...]
    if antennaxis == 2:
        phasen = phase - phase[:, :, refant_idx[0], ...]
    if antennaxis == 3:
        phasen = phase - phase[:, :, :, refant_idx[0], ...]
    if antennaxis == 4:
        phasen = phase - phase[:, :, :, :, refant_idx[0], ...]
    phase = np.copy(phasen)

    # fill values back in
    H.root.sol000.phase000.val[:] = np.copy(phase)

    H.flush()
    H.close()
    return


def reset_phase000(h5parm):
    """
    Reset phase000 solutions to zero

    Parameters
    ----------
    h5parm : str
        Path to the H5Parm file containing the phase solutions.
    """
    # check if phase000 exists
    with tables.open_file(h5parm) as Hcheck:
        soltabs = list(Hcheck.root.sol000._v_children.keys())
    if 'phase000' not in soltabs:
        terminal_print('No phase000 found in ', h5parm)    
        return

    H = tables.open_file(h5parm, mode='r+')
    phase = H.root.sol000.phase000.val[:]
    phase[:] = 0.0
    H.root.sol000.phase000.val[:] = np.copy(phase)
    H.flush()
    H.close()
    return


def change_refant(parmdb, soltab):
    """
    Change the reference antenna, if needed, for phase solutions.

    Parameters
    ----------
    parmdb : str
        Path to the H5Parm / parmdb.
    soltab : str, optional
        Name of the solution table.
    """
    with h5parm.h5parm(parmdb, readonly=False) as H5:
        phases = H5.getSolset('sol000').getSoltab(soltab).getValues()[0]
        weights = H5.getSolset('sol000').getSoltab(soltab).getValues(weight=True)[0]
        axesnames = H5.getSolset('sol000').getSoltab(soltab).getAxesNames()
        terminal_print('axesname', axesnames)
        # print 'SHAPE', np.shape(weights)#, np.size(weights[:,:,0,:,:])

        antennas = list(H5.getSolset('sol000').getSoltab(soltab).getValues()[1]['ant'])
        # print antennas

        if 'pol' in axesnames:
            idx0 = np.where((weights[:, :, 0, :, :] == 0.0))[0]
            idxnan = np.where((~np.isfinite(phases[:, :, 0, :, :])))[0]

            refant = ' '
            tmpvar = float(np.size(weights[:, :, 0, :, :]))
        else:
            idx0 = np.where((weights[:, 0, :, :] == 0.0))[0]
            idxnan = np.where((~np.isfinite(phases[:, 0, :, :])))[0]

            refant = ' '
            tmpvar = float(np.size(weights[:, 0, :, :]))

        if ((float(len(idx0)) / tmpvar) > 0.5) or ((float(len(idxnan)) / tmpvar) > 0.5):
            get_logger().info('Trying to changing reference anntena')

            for antennaid, antenna in enumerate(antennas[1::]):
                terminal_print('Antenna name:', antenna)
                if 'pol' in axesnames:
                    idx0 = np.where((weights[:, :, antennaid + 1, :, :] == 0.0))[0]
                    idxnan = np.where((~np.isfinite(phases[:, :, antennaid + 1, :, :])))[0]
                    tmpvar = float(np.size(weights[:, :, antennaid + 1, :, :]))
                else:
                    idx0 = np.where((weights[:, antennaid + 1, :, :] == 0.0))[0]
                    idxnan = np.where((~np.isfinite(phases[:, antennaid + 1, :, :])))[0]
                    tmpvar = float(np.size(weights[:, antennaid + 1, :, :]))

                terminal_print(
                    'Zero-weight sample indices:',
                    idx0,
                    'Non-finite phase sample indices:',
                    idxnan,
                    'Fraction of zero-weight samples:',
                    float(len(idx0)) / tmpvar,
                )
                if ((float(len(idx0)) / tmpvar) < 0.5) and ((float(len(idxnan)) / tmpvar) < 0.5):
                    get_logger().info('Found new reference anntena,' + str(antenna))
                    refant = antenna
                    break

        if refant != ' ':
            for antennaid, antenna in enumerate(antennas):
                if 'pol' in axesnames:
                    phases[:, :, antennaid, :, :] = phases[:, :, antennaid, :, :] - phases[:, :, antennas.index(refant),
                                                                                    :,
                                                                                    :]
                else:
                    # phases[:,antennaid,:,:] = phases[:,antennaid,:,:] - phases[:,antennas.index(refant),:,:]
                    phases[:, :, antennaid, :] = phases[:, :, antennaid, :] - phases[:, :, antennas.index(refant), :]
            H5.getSolset('sol000').getSoltab(soltab).setValues(phases)
    return
