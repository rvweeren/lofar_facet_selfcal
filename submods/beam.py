"""Beam correction and polarisation-basis conversion (linear <-> circular) of
Measurement Sets with DP3 and losoto."""
import fnmatch
import multiprocessing

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from casacore.tables import table  # type: ignore
from losoto import h5parm  # type: ignore

from submods.h5_helpers.h5_info import check_phaseup, findrefant_core, force_close
from submods.losoto_helpers import create_losoto_beamcorparset, losotolofarbeam
from submods.ms_columns import remove_column_ms
from submods.ms_fixes import fix_uvw
from submods.ms_info import check_phaseup_station
from submods.runtime_utils import get_logger, run, terminal_print


def set_beamcor(ms, beamcor_var, args):
    """
    Determines whether to apply beam correction for a measurement set (MS).

    Parameters
    ----------
    ms : str
        The path to the measurement set (MS) file.
    beamcor_var : str
        A string indicating whether to apply beam correction.
        Possible values are:
        - 'no': Do not apply beam correction.
        - 'yes': Apply beam correction.
        - 'auto': Automatically determine based on observation data.
    args : dict
        Parsed facetselfcal options, passed on to :func:`beam_keywords`.

    Returns
    -------
    bool
        True if beam correction should be applied, False otherwise.

    Notes
    -----
    Behavior:
    - If `beamcor_var` is 'no', beam correction is not applied.
    - If `beamcor_var` is 'yes', beam correction is applied.
    - If `beamcor_var` is 'auto', the function checks:
    - If the telescope is not LOFAR, beam correction is not applied.
    - If the telescope is LOFAR, it calculates the angular separation between
    the phase center and the applied beam direction. If the separation is
    less than 10 arcseconds, beam correction is not applied; otherwise, it is applied.
    - The function uses the `astropy.coordinates.SkyCoord` class to calculate angular separation.
    - Beam keywords are added to the MS if they are missing, using the `beam_keywords` function.
    - Logs information about the decision process and angular separation.
    """

    if beamcor_var == 'no':
        get_logger().info('Run DP3 applybeam: no')
        return False
    if beamcor_var == 'yes':
        get_logger().info('Run DP3 applybeam: yes')
        return True

    t = table(ms + '::OBSERVATION', ack=False)
    if t.getcol('TELESCOPE_NAME')[0] != 'LOFAR':
        t.close()
        get_logger().info('Run DP3 applybeam: no (because we are not using LOFAR observations)')
        return False
    t.close()

    # If we arrive here beamcor_var was set to auto and we are using a LOFAR observation
    # so we assume that beam was taken out in the field center only if user has set 'auto'
    # In case we have old prefactor data and no beam keywords available beam_keywords(ms) will add these
    tmpvar =  beam_keywords(ms, args) # just put output in tmp variable (not used)

    # now check if beam was taken out in the current phase center
    t = table(ms, readonly=True, ack=False)
    beamdir = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_DIR')

    t2 = table(ms + '::FIELD', ack=False)
    phasedir = t2.getcol('PHASE_DIR').squeeze()
    t.close()
    t2.close()

    c1 = SkyCoord(beamdir['m0']['value'] * units.radian, beamdir['m1']['value'] * units.radian, frame='icrs')
    c2 = SkyCoord(phasedir[0] * units.radian, phasedir[1] * units.radian, frame='icrs')
    # Calculate angular separation between phase center and direction in arcseconds
    angsep = c1.separation(c2).to(units.arcsec)

    # angular_separation is recent astropy functionality, do not use, instead use the older SkyCoord.seperation
    # angsep = 3600.*180.*astropy.coordinates.angular_separation(phasedir[0], phasedir[1], beamdir['m0']['value'], beamdir['m1']['value'])/np.pi

    terminal_print('Angular separation between phase center and applied beam direction is', angsep.value, '[arcsec]')
    get_logger().info(
        'Angular separation between phase center and applied beam direction is:' + str(angsep.value) + ' [arcsec]')

    # of less than 10 arcsec than do beam correction
    if angsep.value < 10.0:
        get_logger().info('Run DP3 applybeam: no')
        return False
    else:
        get_logger().info('Run DP3 applybeam: yes')
        return True


def create_beamcortemplate(ms):
    """
    Create a DP3 gain H5 template solution file to be filled with LoSoTo.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    str
        Path to the created H5 template solution file.
    """
    H5name = ms + '_templatejones.h5'

    cmd = "DP3 numthreads=" + str(
        np.min([multiprocessing.cpu_count(), 24])) + " msin=" + ms + " msin.datacolumn=DATA msout=. "
    # cmd += 'msin.modelcolumn=DATA '
    cmd += "steps=[ddecal] ddecal.type=ddecal "
    cmd += "ddecal.maxiter=1 ddecal.nchan=1 "
    cmd += "ddecal.modeldatacolumns='[DATA]' "
    cmd += "ddecal.mode=complexgain ddecal.h5parm=" + H5name + " "
    cmd += "ddecal.solint=10 ddecal.solveralgorithm=directioniterative "
    cmd += "ddecal.datause=dual"  # extra speedup
    # cmd += "ddecal.usedualvisibilities=True" # extra speedup
    terminal_print('Command:', cmd)
    run(cmd)

    return H5name


def fixbeam_ST001(H5name):
    """
    Fix the beam calibration solutions for station ST001 in an H5Parm file.

    This function handles a special case where ST001 station exists in the solution set.
    It replaces ST001's amplitude and phase solutions with values from a reference 
    remote station (RS).

    Parameters
    ----------
    H5name : str
        Path to the H5Parm file containing calibration solutions.

    Returns
    -------
    bool
        True if ST001 was present in the antenna list and corrections were applied,
        False otherwise.

    Notes
    -----
    - Opens the H5Parm file in read-write mode
    - Searches for ST001 in the antenna list of solution set 'sol000'
    - If ST001 exists:
    - Uses the first RS* station as a reference
    - Copies amplitude values from the reference RS station to ST001
    - Sets phase values for ST001 to 0.0
    - Updates both 'amplitude000' and 'phase000' solution tables
    - The H5Parm file is properly closed before returning

    The function assumes the H5Parm structure contains:
    - A solution set named 'sol000'
    - Solution tables 'amplitude000' and 'phase000'
    - At least one RS* station if ST001 is present
    """

    H5 = h5parm.h5parm(H5name, readonly=False)

    ants = H5.getSolset('sol000').getAnt().keys()
    antsrs = fnmatch.filter(ants, 'RS*')
    ST001 = False

    if 'ST001' in ants:
        ST001 = True
        amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()
        ampvals = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
        phasevals = H5.getSolset('sol000').getSoltab('phase000').getValues()[0]

        idx = np.where(amps[1]['ant'] == 'ST001')[0][0]
        idxrs = np.where(amps[1]['ant'] == antsrs[0])[0][0]
        # idx106 = np.where(amps[1]['ant'] == 'RS106HBA')[0][0]
        # idx305 = np.where(amps[1]['ant'] == 'RS305HBA')[0][0]
        # idx508 = np.where(amps[1]['ant'] == 'RS508HBA')[0][0]
        # idx406 = np.where(amps[1]['ant'] == 'RS406HBA')[0][0]
        # idxnonecheck = np.where(amps[1]['ant'] == 'blaHBA')[0][0]

        # ampvals[:,:, idx, 0,:] = 1.0  # set amplitude to 1.
        # phasevals[:,:, idx, 0,:] = 0.0 # set phase to 0.

        ampvals[:, :, idx, 0, :] = ampvals[:, :, idxrs, 0, :]
        phasevals[:, :, idx, 0, :] = 0.0

        H5.getSolset('sol000').getSoltab('amplitude000').setValues(ampvals)
        H5.getSolset('sol000').getSoltab('phase000').setValues(phasevals)

    H5.close()

    return ST001


def beamcor_and_lin2circ(ms, args, msout='.', dysco=True, beam=True, lin2circ=False,
                         circ2lin=False, losotobeamlib='stationresponse', 
                         update_poltable=True, idg=False, metadata_compression=True):
    """
    Correct an MS for the beam in the phase center (array factor only) and convert polarizations.

    Parameters
    ----------
    ms : str
        Input Measurement Set path.
    args : dict
        Parsed facetselfcal options; ``phasediff_only``, ``telescope`` and
        ``skip_solution_plotting`` are used.
    msout : str, optional
        Output Measurement Set path.
    dysco : bool, optional
        Use Dysco compression.
    beam : str, optional
        Beam correction option ('yes', 'no', or 'auto').
    lin2circ : bool, optional
        Convert linear to circular polarizations.
    circ2lin : bool, optional
        Convert circular to linear polarizations.
    losotobeamlib : str, optional
        Beam library to use.
    update_poltable : bool, optional
        Update POLARIZATION table.
    idg : bool, optional
        Image Domain Gridder flag.
    metadata_compression : bool, optional
        Compress MS metadata.
    """

    # check if there are applybeam corrections in the header
    # should be there unless a very old DP3 version has been used
    usedppp = beam_keywords(ms, args)

    losoto = 'losoto'
    # taql = 'taql'
    H5name = create_beamcortemplate(ms)
    phasedup = check_phaseup(H5name)  # in case no beamcor is done we still need this

    if (lin2circ or circ2lin):
        tp = table(ms + '::POLARIZATION', ack=False)
        polinfo = tp.getcol('CORR_TYPE')
        if lin2circ:  # so in this case input must be linear
            if not np.array_equal(np.array([[9, 10, 11, 12]]), polinfo):
                terminal_print('Polarization metadata:', polinfo)
                raise Exception('Input data is not linear, cannot convert to circular')
        if circ2lin:  # so in this case input must be circular
            if not np.array_equal(np.array([[5, 6, 7, 8]]), polinfo):
                terminal_print('Polarization metadata:', polinfo)
                raise Exception('Input data is not circular, cannot convert to linear')
        tp.close()

    if beam:
        tp = table(ms + '::POLARIZATION', ack=False)
        polinfo = tp.getcol('CORR_TYPE')
        tp.close()
        if np.array_equal(np.array([[5, 6, 7, 8]]), polinfo):  # so we have circular data
            raise Exception('Cannot do DP3 beam correction on input data that is circular')

    if lin2circ and circ2lin:
        terminal_print('Wrong input in function, both lin2circ and circ2lin are True')
        raise Exception('Wrong input in function, both lin2circ and circ2lin are True')

    if beam and not args['phasediff_only']:
        losotolofarbeam(H5name, 'phase000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True,
                        beamlib=losotobeamlib)
        losotolofarbeam(H5name, 'amplitude000', ms, useElementResponse=False, useArrayFactor=True, useChanFreq=True,
                        beamlib=losotobeamlib)

        phasedup = fixbeam_ST001(H5name)
        parset = create_losoto_beamcorparset(ms, refant=findrefant_core(H5name, telescope=args['telescope']))
        force_close(H5name)

        # print('Phase up dataset, cannot use DPPP beam, do manual correction')
        if not args["skip_solution_plotting"]:
            cmdlosoto = losoto + ' ' + H5name + ' ' + parset
            terminal_print('LoSoTo command:', cmdlosoto)
            get_logger().info(cmdlosoto)
            run(cmdlosoto)

    if usedppp and not phasedup:
        cmddppp = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=DATA '
        cmddppp += 'msout=' + msout + ' '
        cmddppp += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        
        if not metadata_compression and msout != '.':
            cmddppp += 'msout.uvwcompression=False '
            cmddppp += 'msout.antennacompression=False '
            cmddppp += 'msout.scalarflags=False '
        
        if check_phaseup_station(ms):
            if msout != '.': cmddppp += 'msout.uvwcompression=False ' # only when writing new MS:
        if msout == '.':
            cmddppp += 'msout.datacolumn=CORRECTED_DATA '
        if (lin2circ or circ2lin) and beam:
            cmddppp += 'steps=[beam,pystep] '
            if idg:
                cmddppp += 'beam.type=applybeam beam.updateweights=False '  # weights
            else:
                cmddppp += 'beam.type=applybeam beam.updateweights=True '  # weights
            cmddppp += 'beam.direction=[] '  # correction for the current phase center
            # cmddppp += 'beam.beammode= ' default is full, will undo element as well(!)

            cmddppp += 'pystep.python.module=polconv '
            cmddppp += 'pystep.python.class=PolConv '
            cmddppp += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmddppp += 'pystep.lin2circ=1 '
            if circ2lin:
                cmddppp += 'pystep.circ2lin=1 '

        if beam and not (lin2circ or circ2lin):
            cmddppp += 'steps=[beam] '
            if idg:
                cmddppp += 'beam.type=applybeam beam.updateweights=False '  # weights
            else:
                cmddppp += 'beam.type=applybeam beam.updateweights=True '  # weights
            cmddppp += 'beam.direction=[] '  # correction for the current phase center
            # cmddppp += 'beam.beammode= ' default is full, will undo element as well(!)

        if (lin2circ or circ2lin) and not beam:
            cmddppp += 'steps=[pystep] '
            cmddppp += 'pystep.python.module=polconv '
            cmddppp += 'pystep.python.class=PolConv '
            cmddppp += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmddppp += 'pystep.lin2circ=1 '
            if circ2lin:
                cmddppp += 'pystep.circ2lin=1 '

        if dysco:
            cmddppp += 'msout.storagemanager=dysco '
            cmddppp += 'msout.storagemanager.weightbitrate=16 '

        terminal_print('DP3 applybeam/polconv:', cmddppp)
        run(cmddppp)
        if msout == '.':
            # run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
            run("DP3 msin=" + ms + " msout=. msin.datacolumn=CORRECTED_DATA msout.datacolumn=DATA steps=[]", log=True)
            remove_column_ms(ms, 'CORRECTED_DATA')  # remove the column so it does not get used in the next step
    else:
        cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=DATA '
        cmd += 'msout=' + msout + ' '
        if check_phaseup_station(ms): 
            if msout != '.': cmd += 'msout.uvwcompression=False ' # only when writing new MS
        cmd += 'msin.weightcolumn=WEIGHT_SPECTRUM '
        if msout == '.':
            cmd += 'msout.datacolumn=CORRECTED_DATA '
        
        if not metadata_compression and msout != '.':
            cmd += 'msout.uvwcompression=False '
            cmd += 'msout.antennacompression=False '
            cmd += 'msout.scalarflags=False '

        if (lin2circ or circ2lin) and beam:
            cmd += 'steps=[ac1,ac2,pystep] '
            cmd += 'pystep.python.module=polconv '
            cmd += 'pystep.python.class=PolConv '
            cmd += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmd += 'pystep.lin2circ=1 '
            if circ2lin:
                cmd += 'pystep.circ2lin=1 '

            cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
            cmd += 'ac1.type=applycal ac2.type=applycal '
            cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 '
            if idg:
                cmd += 'ac2.updateweights=False '
            else:
                cmd += 'ac2.updateweights=True '
        if beam and not (lin2circ or circ2lin):
            cmd += 'steps=[ac1,ac2] '
            cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
            cmd += 'ac1.type=applycal ac2.type=applycal '
            cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 '
            if idg:
                cmd += 'ac2.updateweights=False '
            else:
                cmd += 'ac2.updateweights=True '
        if (lin2circ or circ2lin) and not beam:
            cmd += 'steps=[pystep] '
            cmd += 'pystep.python.module=polconv '
            cmd += 'pystep.python.class=PolConv '
            cmd += 'pystep.type=PythonDPPP '
            if lin2circ:
                cmd += 'pystep.lin2circ=1 '
            if circ2lin:
                cmd += 'pystep.circ2lin=1 '

        if dysco:
            cmd += 'msout.storagemanager=dysco '
            cmd += 'msout.storagemanager.weightbitrate=16 '
        terminal_print('DP3 applycal/polconv:', cmd)
        run(cmd, log=True)
        if msout != '.':
            fix_uvw([msout])
        if msout == '.':
            # run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
            cmdcopycolumn = "DP3 msin=" + ms + " msout=. msin.datacolumn=CORRECTED_DATA msout.datacolumn=DATA steps=[]"
            if dysco:
                cmdcopycolumn += ' msout.storagemanager=dysco '
                cmdcopycolumn += ' msout.storagemanager.weightbitrate=16 '
            run(cmdcopycolumn, log=True) # use DP3 and not taql so column beam keyword are copied over
            remove_column_ms(ms, 'CORRECTED_DATA')  # remove the column so it does not get used in the next step
               
    # update ms POLTABLE
    if (lin2circ or circ2lin) and update_poltable:
        tp = table(ms + '::POLARIZATION', readonly=False, ack=True)
        if lin2circ:
            tp.putcol('CORR_TYPE', np.array([[5, 6, 7, 8]], dtype=np.int32))  # FROM LIN-->CIRC
        if circ2lin:
            tp.putcol('CORR_TYPE', np.array([[9, 10, 11, 12]], dtype=np.int32))  # FROM CIRC-->LIN
        tp.close()

    return


def beam_keywords(ms, args, add_beamkeywords=True):
    """
    Check for beam application keywords in a Measurement Set (MS).

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    args : dict
        Parsed facetselfcal options; ``telescope`` is used.
    add_beamkeywords : bool, optional
        Add beam keywords if missing.

    Returns
    -------
    bool
        True if beam keywords are present or were added.
    """

    applybeam_info = False
    with table(ms, readonly=True, ack=False) as t:
        try:
            beammode = t.getcolkeyword('DATA', 'LOFAR_APPLIED_BEAM_MODE')
            applybeam_info = True
            terminal_print(f'DP3 applybeam was used for MS {ms}')
        except:
            applybeam_info = False
            message = (
                f'No applybeam beam keywords were found in MS {ms}. '
                'Possibly an old DP3 version was used in prefactor.'
            )
            terminal_print(message)
            get_logger().warning(message)
            message = (
                f'Adding keywords manually to MS {ms}, '
                'assuming the beam was taken out in the pointing center'
            )
            terminal_print(message)
            get_logger().warning(message)
   
    if not applybeam_info and add_beamkeywords and args['telescope'] == 'LOFAR':
        with table(ms + '::FIELD', readonly=True, ack=False) as t:
            ref_direction = t.getcol('REFERENCE_DIR').squeeze()
            cmddppp = 'DP3 msin=' + ms + ' msout=. steps=[sb] sb.type=setbeam sb.beammode=default '  
            cmddppp += 'sb.direction=['+  str(ref_direction[0]) +',' +  str(ref_direction[1]) + ']'
            terminal_print('DP3 command:', cmddppp)
            run(cmddppp)
            applybeam_info = True

    return applybeam_info


def beamcormodel(ms, dysco=True):
    """
    Create MODEL_DATA_BEAMCOR column for beam-corrupted model data.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    dysco : bool, optional
        Use Dysco compression.
    """
    H5name = ms + '_templatejones.h5'

    cmd = 'DP3 numthreads=' + str(multiprocessing.cpu_count()) + ' msin=' + ms + ' msin.datacolumn=MODEL_DATA msout=. '
    cmd += 'msout.datacolumn=MODEL_DATA_BEAMCOR steps=[ac1,ac2] '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '
    cmd += 'ac1.parmdb=' + H5name + ' ac2.parmdb=' + H5name + ' '
    cmd += 'ac1.type=applycal ac2.type=applycal '
    cmd += 'ac1.correction=phase000 ac2.correction=amplitude000 ac2.updateweights=False '
    cmd += 'ac1.invert=False ac2.invert=False '  # Here we corrupt with the beam !
    terminal_print('DP3 applycal:', cmd)
    run(cmd, log=True)

    return
