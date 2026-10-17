"""Determine and format solution intervals (solint) and channel intervals
(nchan) for calibration, including automatic choices based on the data
and model statistics."""
import re

import astropy.stats
import numpy as np
import scipy.stats
from casacore.tables import table, taql  # type: ignore

from submods.runtime_utils import get_logger, terminal_print
from submods.smearing import get_uvwmax


def tweak_solints(solints, solvalthresh=11, ms_ntimes=None, verbose=False):
    """
    Modify solution intervals so they can be factorized by 2 or 3.

    Parameters
    ----------
    solints : list
        Input solution intervals.
    solvalthresh : float
        Threshold above which factorization is applied.
    ms_ntimes : int, optional
        Total number of time slots in the Measurement Set.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    list
        Modified solution intervals.
    """
    solints_return = []

    # shorten solint to length of MS at most
    if ms_ntimes is not None:
        solints_copy = []
        for solint in solints:
    
            if solint > ms_ntimes:
                solints_copy.append(ms_ntimes)
            else:
                solints_copy.append(solint)
        solints = solints_copy
    
    # range to 15 gives a mximum solint of 24576 below which should be more than enough    
    possible_solints = sorted([2**i + 2**(i-1) for i in range(1,15)] + [2**i for i in range(0,15)])
    if verbose: terminal_print('Possible solints:', possible_solints)
    if ms_ntimes is not None:
        possible_solints = remove_bad_endrounding(possible_solints, ms_ntimes)

    for sol in solints:
        solints_return.append(find_nearest(possible_solints, sol))

    return solints_return


def tweak_solints_single(solint, ms_ntimes, solvalthresh=11):
    """
    def tweak_solints_single(solint, ms_ntimes, solvalthresh=11):
    Adjusts the given solution interval (`solint`) to avoid having a small number
    of leftover time slots near the end of the measurement set (ms).

    Parameters
    ----------
    solint : int
        The initial solution interval to be adjusted.
    ms_ntimes : int
        The total number of time slots in the measurement set.
    solvalthresh : int, optional
        Threshold value for the solution interval. If `solint` is less than this
        threshold, it is returned unchanged. Default is 11.

    Returns
    -------
    int
        A modified solution interval that minimizes leftover time slots while
        being as close as possible to the original `solint`.
    """
    if np.max(solint) < solvalthresh:
        return solint

    possible_solints = np.arange(1, 2 * solint)
    possible_solints = remove_bad_endrounding(possible_solints, ms_ntimes)

    return find_nearest(possible_solints, solint)


def remove_bad_endrounding(solints, ms_ntimes, ignorelessthan=13, fraction_lastslot=0.2):
    """
    Filters a list of solution intervals (solints) to remove those that result in
    significant rounding errors when dividing the total number of timeslots (ms_ntimes)
    by the solution interval. Additionally, excludes solution intervals smaller than
    a specified threshold.

    Parameters
    ----------
    solints : list of int
        A list of possible solution intervals to evaluate.
    ms_ntimes : int
        The total number of timeslots in the measurement set (MS).
    ignorelessthan : int, optional
        The minimum solution interval to consider.
        Solution intervals smaller than this value will be excluded. Defaults to 13.
    fraction_lastslot : float, optional
        The maximum allowed fraction of leftover
        time slots after division. Solution intervals resulting in a leftover
        fraction greater than this value will be excluded. Defaults to 0.2.

    Returns
    -------
    list of int
        A filtered list of solution intervals that meet the criteria.
    """
    solints_out = []
    for solint in solints:
        if (float(ms_ntimes) / float(solint)) - (
                np.floor(float(ms_ntimes) / float(solint))) > fraction_lastslot or solint < ignorelessthan:
            solints_out.append(solint)
    return solints_out


def find_nearest(array, value):
    """
    Find the nearest value in an array to a given target value.

    Parameters
    ----------
    array : array-like
        The input array to search. It will be converted to a NumPy array if not already one.
    value : float or int
        The target value to find the closest match for in the array.

    Returns
    -------
    float or int
        The value from the array that is closest to the target value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def format_solint(solint, ms, return_ntimes=False):
    """
    Format the solution interval for DP3 calls.

    Parameters
    ----------
    solint : int or str
        input solution interval.
    ms : str
        measurement set to extract the integration time from.
    return_ntimes : bool, optional
        Also return the number of unique time slots in the Measurement Set.

    Returns
    -------
    str or tuple
        Processed solution interval; when ``return_ntimes`` is true, also includes
        the number of unique time slots.
    """
    if str(solint).isdigit():
        if return_ntimes:
            with table(ms, readonly=True, ack=False) as t:
                time = np.unique(t.getcol('TIME'))
            return str(solint), len(time)
        else:
            return str(solint)
    else:
        with table(ms, readonly=True, ack=False) as t:
            time = np.unique(t.getcol('TIME'))
            tint = np.abs(time[1] - time[0])
        if 's' in solint:
            solintout = int(np.rint(float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'm' in solint:
            solintout = int(np.rint(60. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if 'h' in solint:
            solintout = int(np.rint(3600. * float(re.findall(r'[+-]?\d+(?:\.\d+)?', solint)[0]) / tint))
        if solintout < 1:
            solintout = 1
        if return_ntimes:
            return str(solintout), len(time)
        else:
            return str(solintout)


def format_nchan(nchan, ms):
    """
    Format the solution interval for DP3 calls.

    Parameters
    ----------
    nchan : int or str
        input solution interval along the frequency axis.
    ms : str
        measurement set to extract the frequnecy resolution from.

    Returns
    -------
    solintout : str
        processed frequency solution interval.
    """
    if str(nchan).isdigit():
        return str(nchan)
    else:
        with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
            chanw = np.median(t.getcol('CHAN_WIDTH'))
        if 'Hz' in nchan:
            nchanout = int(np.rint(float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if 'kHz' in nchan:
            nchanout = int(np.rint(1e3 * float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if 'MHz' in nchan:
            nchanout = int(np.rint(1e6 * float(re.findall(r'[+-]?\d+(?:\.\d+)?', nchan)[0]) / chanw))
        if nchanout < 1:
            nchanout = 1
        return str(nchanout)


def setinitial_solint(mslist, options):
    """
    Expand user input solution parameters to all Measurement Sets.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    options : dict
        Dictionary of self-calibration options.

    Returns
    -------
    tuple
        Expanded lists for soltypes, solints, nchans, smoothness, and constraints.
    """

    nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
        smoothnessspectralexponent_list, smoothnessrefdistance_list, antennaconstraint_list, resetsols_list, \
        resetdir_list, normamps_list, soltypecycles_list, uvmin_list, uvmax_list, uvminim_list, uvmaxim_list, \
        solve_msinnchan_list, solve_msinstartchan_list, antenna_averaging_factors_list, \
        antenna_smoothness_factors_list = \
        ([] for _ in range(20))

    # make here uvminim_list and uvmaxim_list, because the have just the length of mslist
    for ms_id, ms in enumerate(mslist):
        try:
            uvminim = options.uvminim[ms_id]
        except:
            uvminim = options.uvminim  # apparently we just have a float and not a list
        uvminim_list.append(uvminim)
    for ms_id, ms in enumerate(mslist):
        try:
            uvmaxim = options.uvmaxim[ms_id]
        except:
            uvmaxim = options.uvmaxim  # apparently we just have a float and not a list
        uvmaxim_list.append(uvmaxim)

    for soltype_id, soltype in enumerate(options.soltype_list):
        nchan_ms = []  # list with len(mslist)
        solint_ms = []  # list with len(mslist)
        antennaconstraint_list_ms = []  # list with len(mslist)
        resetsols_list_ms = []  # list with len(mslist)
        resetdir_list_ms = []  # list with len(mslist)
        normamps_list_ms = []  # list with len(mslist)
        smoothnessconstraint_list_ms = []  # list with len(mslist)
        smoothnessreffrequency_list_ms = []  # list with len(mslist)
        smoothnessspectralexponent_list_ms = []  # list with len(mslist)
        smoothnessrefdistance_list_ms = []  # list with len(mslist)
        BLsmooth_list_ms = []  # list with len(mslist)
        soltypecycles_list_ms = []  # list with len(mslist)
        uvmin_list_ms = []  # list with len(mslist)
        uvmax_list_ms = []  # list with len(mslist)
        # uvminim_list_ms = []  # list with len(mslist)
        # uvmaxim_list_ms = []  # list with len(mslist)
        solve_msinnchan_list_ms = []  # list with len(mslist)
        solve_msinstartchan_list_ms = []  # list with len(mslist)
        antenna_averaging_factors_list_ms = []  # list with len(mslist)
        antenna_smoothness_factors_list_ms = []  # list with len(mslist)

        for ms in mslist:
            # use try statement in case the user did not provide all the info for certain soltypes

            # solint
            try:
                solint = options.solint_list[soltype_id]
            except:
                solint = 1

            # nchan
            try:
                nchan = options.nchan_list[soltype_id]
            except:
                nchan = 1  # if nothing is set use 1

            # smoothnessconstraint
            try:
                smoothnessconstraint = options.smoothnessconstraint_list[soltype_id]
            except:
                smoothnessconstraint = 0.0

            # BLsmooth
            try:
                BLsmooth = options.BLsmooth_list[soltype_id]
            except:
                BLsmooth = False

            # smoothnessreffrequency
            try:
                smoothnessreffrequency = options.smoothnessreffrequency_list[soltype_id]
            except:
                smoothnessreffrequency = 0.0

            # smoothnessspectralexponent
            try:
                smoothnessspectralexponent = options.smoothnessspectralexponent_list[soltype_id]
            except:
                smoothnessspectralexponent = -1.0

            # smoothnessrefdistance
            try:
                smoothnessrefdistance = options.smoothnessrefdistance_list[soltype_id]
            except:
                smoothnessrefdistance = 0.0

            # antennaconstraint
            try:
                antennaconstraint = options.antennaconstraint_list[soltype_id]
            except:
                antennaconstraint = None

            # resetsols
            try:
                resetsols = options.resetsols_list[soltype_id]
            except:
                resetsols = None

            # resetdir
            try:
                resetdir = options.resetdir_list[soltype_id]
            except:
                resetdir = None

            # normamps
            try:
                normamps = options.normamps_list[soltype_id]
            except:
                normamps = 'normamps'

            # uvmin
            try:
                uvmin = options.uvmin[soltype_id]
            except:
                uvmin = None

            # uvmax
            try:
                uvmax = options.uvmax[soltype_id]
            except:
                uvmax = None
            
            # solve_msinnchan
            try:
                solve_msinnchan = options.solve_msinnchan_list[soltype_id]
            except:
                solve_msinnchan = 'all'
            
            # solve_msinstartchan
            try:
                solve_msinstartchan = options.solve_msinstartchan_list[soltype_id]
            except:
                solve_msinstartchan = 0

            # antenna_averaging_factors
            try:
                antenna_averaging_factors = options.antenna_averaging_factors_list[soltype_id]
            except:
                antenna_averaging_factors = None    
            
            # antenna_smoothness_factors
            try:
                antenna_smoothness_factors = options.antenna_smoothness_factors_list[soltype_id]
            except:
                antenna_smoothness_factors = None

            # uvminim
            # try:
            #  uvminim = options.uvminim[soltype_id]
            # except:
            #  uvminim = 80.

            # uvmaxim
            # try:
            #  uvmaxim = options.uvmaxim[soltype_id]
            # except:
            #  uvmaxim = None

            # soltypecycles
            soltypecycles = options.soltypecycles_list[soltype_id]

            # force nchan 1 for tec(andphase) solve and in case smoothnessconstraint is invoked
            # if soltype == 'tec' or  soltype == 'tecandphase' or smoothnessconstraint > 0.0:
            if soltype == 'tec' or soltype == 'tecandphase' or soltype == 'tec+phase' or soltype == 'tec+delay' or soltype == 'tec+phase+delay' or soltype == 'delay':
                nchan = 1

            nchan_ms.append(nchan)
            solint_ms.append(solint)
            smoothnessconstraint_list_ms.append(smoothnessconstraint)
            BLsmooth_list_ms.append(BLsmooth)
            smoothnessreffrequency_list_ms.append(smoothnessreffrequency)
            smoothnessspectralexponent_list_ms.append(smoothnessspectralexponent)
            smoothnessrefdistance_list_ms.append(smoothnessrefdistance)
            antennaconstraint_list_ms.append(antennaconstraint)
            resetsols_list_ms.append(resetsols)
            resetdir_list_ms.append(resetdir)
            normamps_list_ms.append(normamps)

            soltypecycles_list_ms.append(soltypecycles)
            uvmin_list_ms.append(uvmin)
            uvmax_list_ms.append(uvmax)
            # uvminim_list_ms.append(uvminim)
            # uvmaxim_list_ms.append(uvmaxim)
            solve_msinnchan_list_ms.append(solve_msinnchan)
            solve_msinstartchan_list_ms.append(solve_msinstartchan)
            antenna_averaging_factors_list_ms.append(antenna_averaging_factors)
            antenna_smoothness_factors_list_ms.append(antenna_smoothness_factors)

        nchan_list.append(nchan_ms)  # list of lists
        solint_list.append(solint_ms)  # list of lists
        antennaconstraint_list.append(antennaconstraint_list_ms)  # list of lists
        resetsols_list.append(resetsols_list_ms)  # list of lists
        resetdir_list.append(resetdir_list_ms)  # list of lists
        normamps_list.append(normamps_list_ms)  # list of lists
        BLsmooth_list.append(BLsmooth_list_ms)  # list of lists
        smoothnessconstraint_list.append(smoothnessconstraint_list_ms)  # list of lists
        smoothnessreffrequency_list.append(smoothnessreffrequency_list_ms)  # list of lists
        smoothnessspectralexponent_list.append(smoothnessspectralexponent_list_ms)  # list of lists
        smoothnessrefdistance_list.append(smoothnessrefdistance_list_ms)
        uvmin_list.append(uvmin_list_ms)  # list of lists
        uvmax_list.append(uvmax_list_ms)  # list of lists
        # uvminim_list.append(uvminim_list_ms) # list of lists
        # uvmaxim_list.append(uvmaxim_list_ms)      # list of lists
        solve_msinnchan_list.append(solve_msinnchan_list_ms)  # list of lists
        solve_msinstartchan_list.append(solve_msinstartchan_list_ms)  # list of lists
        antenna_averaging_factors_list.append(antenna_averaging_factors_list_ms)  # list of lists
        antenna_smoothness_factors_list.append(antenna_smoothness_factors_list_ms)  # list of lists

        soltypecycles_list.append(soltypecycles_list_ms)

    terminal_print('soltype:', options.soltype_list, mslist)
    terminal_print('nchan:', nchan_list)
    terminal_print('solint:', solint_list)
    terminal_print('BLsmooth:', BLsmooth_list)
    terminal_print('smoothnessconstraint:', smoothnessconstraint_list)
    terminal_print('smoothnessreffrequency:', smoothnessreffrequency_list)
    terminal_print('smoothnessspectralexponent:', smoothnessspectralexponent_list)
    terminal_print('smoothnessrefdistance:', smoothnessrefdistance_list)
    terminal_print('antennaconstraint:', antennaconstraint_list)
    terminal_print('resetsols:', resetsols_list)
    terminal_print('resetdir:', resetdir_list)
    terminal_print('normamps:', normamps_list)
    terminal_print('soltypecycles:', soltypecycles_list)
    terminal_print('uvmin:', uvmin_list)
    terminal_print('uvmax:', uvmax_list)
    terminal_print('uvminim:', uvminim_list)
    terminal_print('uvmaxim:', uvmaxim_list)
    terminal_print('solve_msinnchan:', solve_msinnchan_list)
    terminal_print('solve_msinstartchan:', solve_msinstartchan_list)
    terminal_print('antenna_averaging_factors:', antenna_averaging_factors_list)
    terminal_print('antenna_smoothness_factors:', antenna_smoothness_factors_list)

    get_logger().info('soltype: ' + str(options.soltype_list) + ' ' + str(mslist))
    get_logger().info('nchan: ' + str(options.nchan_list))
    get_logger().info('solint: ' + str(options.solint_list))
    get_logger().info('BLsmooth: ' + str(options.BLsmooth_list))
    get_logger().info('smoothnessconstraint: ' + str(options.smoothnessconstraint_list))
    get_logger().info('smoothnessreffrequency: ' + str(options.smoothnessreffrequency_list))
    get_logger().info('smoothnessspectralexponent: ' + str(options.smoothnessspectralexponent_list))
    get_logger().info('smoothnessrefdistance: ' + str(options.smoothnessrefdistance_list))
    get_logger().info('antennaconstraint: ' + str(options.antennaconstraint_list))
    get_logger().info('resetsols: ' + str(options.resetsols_list))
    get_logger().info('resetdir: ' + str(options.resetdir_list))
    get_logger().info('normamps: ' + str(options.normamps_list))
    get_logger().info('soltypecycles: ' + str(options.soltypecycles_list))
    get_logger().info('uvmin: ' + str(options.uvmin))
    get_logger().info('uvmax: ' + str(options.uvmax))
    get_logger().info('uvminim: ' + str(options.uvminim))
    get_logger().info('uvmaxim: ' + str(options.uvmaxim))
    get_logger().info('solve_msinnchan: ' + str(options.solve_msinnchan_list))
    get_logger().info('solve_msinstartchan: ' + str(options.solve_msinstartchan_list))
    get_logger().info('antenna_averaging_factors: ' + str(options.antenna_averaging_factors_list))
    get_logger().info('antenna_smoothness_factors: ' + str(options.antenna_smoothness_factors_list))

    return nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, \
           smoothnessreffrequency_list, smoothnessspectralexponent_list, \
           smoothnessrefdistance_list, antennaconstraint_list, resetsols_list, \
           resetdir_list, soltypecycles_list, uvmin_list, uvmax_list, uvminim_list, \
           uvmaxim_list, normamps_list, solve_msinnchan_list, \
           solve_msinstartchan_list, antenna_averaging_factors_list, \
           antenna_smoothness_factors_list


def getms_amp_stats(ms, datacolumn='DATA', uvcutfraction=0.666, robustsigma=True):
    """
    Measure the logarithmic RR/LL amplitude scatter in an MS.

    Parameters
    ----------
    ms : str
        Measurement Set path.
    datacolumn : str, optional
        Visibility column to inspect.
    uvcutfraction : float, optional
        Fraction of the maximum UV distance.
    robustsigma : bool, optional
        Use sigma-clipped statistics.

    Returns
    -------
    float
        Amplitude-noise estimate in log10 units.
    """
    uvdismod = get_uvwmax(ms) * uvcutfraction
    t = taql(
        'SELECT ' + datacolumn + ',UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    flags = t.getcol('FLAG')
    data = t.getcol(datacolumn)
    data = np.ma.masked_array(data, flags)
    t.close()

    amps_rr = np.abs(data[:, :, 0])
    amps_ll = np.abs(data[:, :, 3])

    # remove zeros from LL
    idx = np.where(amps_ll != 0.0)
    amps_ll = amps_ll[idx]
    amps_rr = amps_rr[idx]

    # remove zeros from RR
    idx = np.where(amps_rr != 0.0)
    amps_ll = amps_ll[idx]
    amps_rr = amps_rr[idx]

    amplogratio = np.log10(amps_rr / amps_ll)  # we assume Stokes V = 0, so RR = LL
    if robustsigma:
        logampnoise = astropy.stats.sigma_clipping.sigma_clipped_stats(amplogratio)[2]
    else:
        logampnoise = np.std(amplogratio)
    terminal_print(
        'Measurement Set:',
        ms,
        'Log-amplitude noise:',
        logampnoise,
        'Mean log-amplitude ratio:',
        np.mean(amplogratio),
    )
    return logampnoise


def getms_phase_stats(ms, datacolumn='DATA', uvcutfraction=0.666):
    """
    Measure the circular RR/LL phase scatter in an MS.

    Parameters
    ----------
    ms : str
        Measurement Set path.
    datacolumn : str, optional
        Visibility column to inspect.
    uvcutfraction : float, optional
        Fraction of the maximum UV distance.

    Returns
    -------
    float
        Circular phase-noise estimate in radians.
    """
    uvdismod = get_uvwmax(ms) * uvcutfraction
    t = taql(
        'SELECT ' + datacolumn + ',UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    flags = t.getcol('FLAG')
    data = t.getcol(datacolumn)
    data = np.ma.masked_array(data, flags)
    t.close()

    phase_rr = np.angle(data[:, :, 0])
    phase_ll = np.angle(data[:, :, 3])

    # remove zeros from LL (flagged data)
    idx = np.where(phase_ll != 0.0)
    phase_ll = phase_ll[idx]
    phase_rr = phase_rr[idx]

    # remove zeros from RR (flagged data)
    idx = np.where(phase_rr != 0.0)
    phase_ll = phase_ll[idx]
    phase_rr = phase_rr[idx]

    phasediff = np.mod(phase_rr - phase_ll, 2. * np.pi)
    phasenoise = scipy.stats.circstd(phasediff, nan_policy='omit')

    terminal_print(
        'Measurement Set:',
        ms,
        'Phase noise:',
        phasenoise,
        'Circular mean phase difference:',
        scipy.stats.circmean(phasediff, nan_policy='omit'),
    )
    return phasenoise


def getmsmodelinfo(ms, modelcolumn, fastrms=False, uvcutfraction=0.333):
    """
    Measure summary statistics of a model visibility column.

    Parameters
    ----------
    ms : str
        Measurement Set path.
    modelcolumn : str
        Model visibility column to inspect.
    fastrms : bool, optional
        Use the fast RMS calculation.
    uvcutfraction : float, optional
        Fraction of the maximum UV distance.

    Returns
    -------
    tuple
        Model amplitude and phase summary values.
    """
    t = table(ms + '::SPECTRAL_WINDOW', ack=False)
    chanw = np.median(t.getcol('CHAN_WIDTH'))
    freq = np.median(t.getcol('CHAN_FREQ'))
    nfreq = len(t.getcol('CHAN_FREQ')[0])
    t.close()
    uvdismod = get_uvwmax(ms) * uvcutfraction  # take range [uvcutfraction*uvmax - 1.0uvmax]

    HBA_upfreqsel = 0.75  # select only freqcencies above 75% of the available bandwidth
    freqct = 1000e6
    # the idea is that for HBA if you are far out in the beam the noise gets up much more at the higher freqs and the model flux goes down due to the spectral index. In this way we get more conservative solints

    t = taql(
        'SELECT ' + modelcolumn + ',DATA,UVW,TIME,FLAG FROM ' + ms + ' WHERE SQRT(SUMSQR(UVW[:2])) > ' + str(uvdismod))
    model = np.abs(t.getcol(modelcolumn))
    flags = t.getcol('FLAG')
    data = t.getcol('DATA')
    terminal_print('Compute visibility noise of the dataset with robust sigma clipping', ms)
    get_logger().info('Compute visibility noise of the dataset with robust sigma clipping: ' + ms)
    if fastrms:  # take only every fifth element of the array to speed up the computation
        if freq > freqct:  # HBA
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(
                data[0:data.shape[0]:5, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3],
                mask=flags[0:data.shape[0]:5, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3])[2]  # use XY and YX
        else:
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[0:data.shape[0]:5, :, 1:3],
                                                                     mask=flags[0:data.shape[0]:5, :, 1:3])[
                2]  # use XY and YX
    else:
        if freq > freqct:  # HBA
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(
                data[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3],
                mask=flags[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 1:3])[2]  # use XY and YX
        else:
            noise = astropy.stats.sigma_clipping.sigma_clipped_stats(data[:, :, 1:3],
                                                                     mask=flags[:, :, 1:3])[2]  # use XY and YX

    model = np.ma.masked_array(model, flags)
    if freq > freqct:  # HBA:
        flux = np.ma.mean((model[:, int(np.floor(float(nfreq) * HBA_upfreqsel)):-1, 0] + model[:, int(np.floor(
            float(nfreq) * HBA_upfreqsel)):-1,
                                                                                         3]) * 0.5)  # average XX and YY (ignore XY and YX, they are zero, or nan, in other words this is Stokes I)
    else:
        flux = np.ma.mean(
            (model[:, :, 0] + model[:, :, 3]) * 0.5)  # average XX and YY (ignore XY and YX, they are zero, or nan)
    time = np.unique(t.getcol('TIME'))
    tint = np.abs(time[1] - time[0])
    terminal_print('Integration time visibilities', tint)
    get_logger().info('Integration time visibilities [s]: ' + str(tint))
    t.close()

    del data, flags, model
    terminal_print('Noise visibilities:', noise, 'Jy')
    terminal_print('Flux in model:', flux, 'Jy')
    terminal_print('UV-selection to compute model flux:', str(uvdismod / 1e3), 'km')
    get_logger().info('Noise visibilities: ' + str(noise) + 'Jy')
    get_logger().info('Flux in model: ' + str(flux) + 'Jy')
    get_logger().info('UV-selection to compute model flux: ' + str(uvdismod / 1e3) + 'km')

    return noise, flux, tint, chanw


def return_soltype_index(soltype_list, soltype, occurence=1, onetectypeoccurence=False):
    """
    Returns the index of a specified solution type in a list of solution types.

    Parameters
    ----------
    soltype_list : list of str
        A list of solution types.
    soltype : str
        The solution type to search for.
    occurence : int, optional
        The occurrence of the solution type to find. Defaults to 1.
    onetectypeoccurence : bool, optional
        If True, treats 'tecandphase' and 'tec' as equivalent. Defaults to False.

    Returns
    -------
    int or None
        The index of the specified solution type in the list, or None if not found.
    """
    if onetectypeoccurence:
        if soltype == 'tecandphase' or soltype == 'tec':
            soltype_list = [sol.replace('tecandphase', 'tec') for sol in soltype_list]
            soltype = 'tec'
        if soltype == 'tec+phase':
            soltype_list = [sol.replace('tec+phase', 'tec') for sol in soltype_list]
            soltype = 'tec'    

    sol_index = None
    count = 0
    for sol_id, sol in enumerate(soltype_list):
        if sol == soltype:
            sol_index = sol_id
            count = count + 1
            if occurence == count:
                return sol_index
            else:
                sol_index = None
    return sol_index


def auto_determinesolints(mslist, soltype_list, longbaseline, LBA,
                          innchan_list=None, insolint_list=None,
                          uvdismod=None, modelcolumn='MODEL_DATA', redo=False,
                          inBLsmooth_list=None,
                          insmoothnessconstraint_list=None, insmoothnessreffrequency_list=None,
                          insmoothnessspectralexponent_list=None,
                          insmoothnessrefdistance_list=None,
                          inantennaconstraint_list=None, inresetsols_list=None, inresetdir_list=None,
                          innormamps_list=None,
                          insoltypecycles_list=None, tecfactorsolint=1.0, gainfactorsolint=1.0,
                          gainfactorsmoothness=1.0, phasefactorsolint=1.0, delaycal=False):
    """
    Determine solution time and frequency intervals based on compact flux and noise.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    soltype_list : list of str
        List of solution types.
    longbaseline : bool
        Whether long baselines are present.
    LBA : bool
        Whether data is LOFAR LBA.
    innchan_list : list of int
        Initial channel intervals.
    insolint_list : list of int or str
        Initial solution time intervals.
    uvdismod : float
        UV distance model limit.
    modelcolumn : str
        Model visibility column name.
    redo : bool
        Whether to redo determination.
    inBLsmooth_list : list of bool
        Baseline smoothing flags.
    insmoothnessconstraint_list : list of float
        Smoothness constraints in MHz.
    insmoothnessreffrequency_list : list of float
        Reference frequencies for smoothness.
    insmoothnessspectralexponent_list : list of float
        Spectral exponents for smoothness.
    insmoothnessrefdistance_list : list of float
        Reference distances for smoothness.
    inantennaconstraint_list : list of str
        Antenna constraint lists.
    inresetsols_list : list of str
        Antenna reset lists.
    inresetdir_list : list of str
        Direction reset lists.
    innormamps_list : list of str
        Amplitude normalization options.
    insoltypecycles_list : list of int
        Self-calibration cycles for solution types.
    tecfactorsolint : float
        Scaling factor for TEC solution intervals.
    gainfactorsolint : float
        Scaling factor for gain solution intervals.
    gainfactorsmoothness : float
        Scaling factor for smoothness.
    phasefactorsolint : float
        Scaling factor for phase solution intervals.
    delaycal : bool
        Delay calibration flag.

    Returns
    -------
    tuple
        Updated solint, nchan, smoothness, and constraint lists.
    """
    # 1 find the first tec/tecandphase in the soltype_list
    # set the solints there based on this code
    # update antennaconstraint if needed

    # find the first (scalar)complexgain
    # compute solints

    # based on scalarcomplexgain determine what to do
    # proceed normally, as is now
    # do an 'all' constrained solve
    # no complexgain solves at all, how to do this? update soltype_list here, or set

    # soltype_list == 'tec'

    for ms_id, ms in enumerate(mslist):
        if longbaseline:
            noise, flux, tint, chanw = getmsmodelinfo(ms, modelcolumn, uvcutfraction=0.075)
        else:
            noise, flux, tint, chanw = getmsmodelinfo(ms, modelcolumn)
        for soltype_id, soltype in enumerate(soltype_list):

            ######## TEC or TECANDPHASE ######
            ######## for first occurence of tec(andphase) #######
            if soltype in ['tec', 'tecandphase'] and \
                    ((soltype_id == return_soltype_index(soltype_list, 'tec', occurence=1, onetectypeoccurence=True)) or \
                     (soltype_id == return_soltype_index(soltype_list, 'tecandphase', occurence=1,
                                                         onetectypeoccurence=True))):

                if LBA:
                    if longbaseline:
                        solint_sf = 3.0e-3 * tecfactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 4.0e-2 * tecfactorsolint  # 0.5e-3 # for tecandphase and coreconstraint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.5e-2 * tecfactorsolint  # for tecandphase, no coreconstraint
                    else:  # for -- HBA dutch --
                        solint_sf = 4.0e-2 * tecfactorsolint  # for tecandphase, no coreconstraint

                if soltype == 'tec':
                    solint_sf = solint_sf / np.sqrt(2.)  # tec and coreconstraint

                # trigger antennaconstraint_phase core if solint > tint
                if not longbaseline and (tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3) > tint):
                    terminal_print(
                        'Estimated solution interval (s):',
                        tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3),
                    )
                    solint_sf = solint_sf / 30.
                    terminal_print('Trigger_antennaconstraint core:', soltype, ms)
                    get_logger().info('Trigger_antennaconstraint core: ' + soltype + ' ' + ms)
                    inantennaconstraint_list[soltype_id][ms_id] = 'core'
                    # do another pertubation, a slow solve of the core stations
                    # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
                    if (tint * solint_sf * ((noise / flux) ** 2) * (
                            chanw / 390.625e3) < 720.0):  # less than 12 min now, also doing
                        inantennaconstraint_list[soltype_id + 1][ms_id] = 'remote'  # or copy over input ??
                        insoltypecycles_list[soltype_id + 1][ms_id] = insoltypecycles_list[soltype_id][
                            ms_id]  # do + 1 here??
                        insolint_list[soltype_id + 1][ms_id] = int(
                            np.rint(10. * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)))
                        if insolint_list[soltype_id + 1][ms_id] < 1:
                            insolint_list[soltype_id + 1][ms_id] = 1
                    else:
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None  # or copy over input
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                # round to nearest integer
                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                # frequency scaling is need because if we avearge in freqeuncy the solint should not change for a tec(andphase) solve

                if (longbaseline) and (not LBA) and (soltype == 'tec') \
                        and (soltype_list[1] == 'tecandphase'):
                    if solint < 0.5 and (solint * tint < 16.):  # so less then 16 sec
                        terminal_print('Longbaselines bright source detected: changing from tec to tecandphase solve')
                        insoltypecycles_list[soltype_id][ms_id] = 999
                        insoltypecycles_list[1][ms_id] = 0

                if solint < 1:
                    solint = 1
                if (float(solint) * tint / 3600.) > 0.5:  # so check if larger than 30 min
                    terminal_print('Warning, it seems there is not enough flux density on the longer baselines for solving')
                    get_logger().warning(
                        'Warning, it seems there is not enough flux density on the longer baselines for solving')
                    solint = np.rint(0.5 * 3600. / tint)  # max is 30 min

                terminal_print(
                    'Unrounded solution interval (time slots):',
                    solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3),
                    'Using tec(andphase) solint:',
                    'Solution interval(s):',
                    solint,
                )
                get_logger().info(str(solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) + '-- Using tec(andphase) solint:' + str(solint))
                terminal_print('Using tec(andphase) solint [s]:', float(solint) * tint)
                get_logger().info('Using tec(andphase) solint [s]: ' + str(float(solint) * tint))

                insolint_list[soltype_id][ms_id] = int(solint)
                innchan_list[soltype_id][ms_id] = 1

            ######## SCALARPHASE or PHASEONLY ######
            ######## for first occurence of tec(andphase) #######
            if soltype in ['scalarphase', 'phaseonly'] and \
                    (insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
                    ((soltype_id == return_soltype_index(soltype_list, 'scalarphase', occurence=1,
                                                         onetectypeoccurence=True)) or
                     (soltype_id == return_soltype_index(soltype_list, 'phaseonly', occurence=1,
                                                         onetectypeoccurence=True))):

                if LBA:
                    if longbaseline:
                        solint_sf = 3.0e-3 * phasefactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 4.0e-2 * phasefactorsolint  # 0.5e-3 # for tecandphase and coreconstraint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.5e-2 * phasefactorsolint  # for tecandphase, no coreconstraint
                    else:  # for -- HBA dutch --
                        solint_sf = 4.0e-2 * phasefactorsolint  # for tecandphase, no coreconstraint

                if soltype == 'scalarphase':
                    solint_sf = solint_sf / np.sqrt(2.)  # decrease solint if scalarphase

                # trigger antennaconstraint_phase core if solint > tint
                # needs checking, this might be wrong, this assumes we use [scalarphase/phaseonly,scalarphase/phaseonly, (scalar)complexgain] so 3 steps.....
                if not longbaseline and (tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3) > tint):
                    terminal_print(
                        'Estimated solution interval (s):',
                        tint * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3),
                    )
                    solint_sf = solint_sf / 30.
                    terminal_print('Trigger_antennaconstraint core:', soltype, ms)
                    get_logger().info('Trigger_antennaconstraint core: ' + soltype + ' ' + ms)
                    inantennaconstraint_list[soltype_id][ms_id] = 'core'
                    # do another pertubation, a slow solve of the core stations
                    # if (tint*solint_sf* ((noise/flux)**2) * (chanw/390.625e3) < 360.0): # less than 6 min now, also doing constraint remote
                    if (tint * solint_sf * ((noise / flux) ** 2) * (
                            chanw / 390.625e3) < 720.0):  # less than 12 min now, also doing
                        inantennaconstraint_list[soltype_id + 1][ms_id] = 'remote'  # or copy over input ??
                        insoltypecycles_list[soltype_id + 1][ms_id] = insoltypecycles_list[soltype_id][
                            ms_id]  # do + 1 here??
                        insolint_list[soltype_id + 1][ms_id] = int(
                            np.rint(10. * solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)))
                        if insolint_list[soltype_id + 1][ms_id] < 1:
                            insolint_list[soltype_id + 1][ms_id] = 1
                    else:
                        insoltypecycles_list[soltype_id + 1][ms_id] = 999

                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None  # or copy over input

                # round to nearest integer
                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                # frequency scaling is needed because if we avearge in freqeuncy the solint should not change for a (scalar)phase solve with smoothnessconstraint

                if solint < 1:
                    solint = 1
                if (float(solint) * tint / 3600.) > 0.5:  # so check if larger than 30 min
                    terminal_print('Warning, it seems there is not enough flux density on the longer baselines for solving')
                    get_logger().warning(
                        'Warning, it seems there is not enough flux density on the longer baselines for solving')
                    solint = np.rint(0.5 * 3600. / tint)  # max is 30 min

                terminal_print(
                    'Unrounded solution interval (time slots):',
                    solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3),
                    'Using (scalar)phase solint:',
                    'Solution interval(s):',
                    solint,
                )
                get_logger().info(str(solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) + '-- Using (scalar)phase solint:' + str(solint))
                terminal_print('Using (scalar)phase solint [s]:', float(solint) * tint)
                get_logger().info('Using (scalar)phase solint [s]: ' + str(float(solint) * tint))

                insolint_list[soltype_id][ms_id] = int(solint)
                innchan_list[soltype_id][ms_id] = 1  # because we use smoothnessconstraint

            ######## COMPLEXGAIN or SCALARCOMPLEXGAIN or AMPLITUDEONLY or SCALARAMPLITUDE ######
            # requires smoothnessconstraint
            # for first occurence of (scalar)complexgain
            terminal_print(
                'Smoothness constraints:',
                insmoothnessconstraint_list,
                'Solution type index:',
                soltype_id,
                'Measurement Set index:',
                ms_id,
            )
            if soltype in ['complexgain', 'scalarcomplexgain'] and (
                    insmoothnessconstraint_list[soltype_id][ms_id] > 0.0) and \
                    ((soltype_id == return_soltype_index(soltype_list, 'complexgain', occurence=1)) or
                     (soltype_id == return_soltype_index(soltype_list, 'scalarcomplexgain', occurence=1))):

                if longbaseline:
                    thr_disable_gain = 24.  # 32. #  72.
                else:
                    thr_disable_gain = 64.  # 32. #  72.

                thr_SM15Mhz = 1.5
                thr_gain_trigger_allantenna = 32.  # 16. # 8.

                tgain_max = 4.  # do not allow ap solves that are more than 4 hrs
                tgain_min = 0.3333  # check if less than 20 min, min solint is 20 min

                innchan_list[soltype_id][ms_id] = 1

                if LBA:
                    if longbaseline:
                        solint_sf = 0.4 * gainfactorsolint  # untested
                    else:  # for -- LBA dutch --
                        solint_sf = 10.0 * gainfactorsolint

                else:  # for -- HBA --
                    if longbaseline:
                        solint_sf = 0.8 * gainfactorsolint  #
                    else:  # for -- HBA dutch --
                        solint_sf = 0.8 * gainfactorsolint  #

                solint = np.rint(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3))
                terminal_print(
                    'Unrounded solution interval (time slots):',
                    solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3),
                    'Computes gain solint:',
                    'Solution interval(s):',
                    solint,
                    ' ',
                )
                get_logger().info(
                    str(solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)) + ' Computes gain solint: ' + str(
                        solint))
                terminal_print('Computes gain solint [hr]:', float(solint) * tint / 3600.)
                get_logger().info('Computes gain solint [hr]: ' + str(float(solint) * tint / 3600.))

                # do not allow very short ap solves
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) < tgain_min:  # check if less than tgain_min (20 min)
                    solint = np.rint(tgain_min * 3600. / tint)  # minimum tgain_min is 20 min
                    terminal_print('Setting gain solint to 20 min (the min value allowed):', float(solint) * tint / 3600.)
                    get_logger().info(
                        'Setting gain solint to 20 min (the min value allowed): ' + str(float(solint) * tint / 3600.))

                # do not allow ap solves that are more than tgain_max (4) hrs
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) > tgain_max:  # so check if larger than 4 hrs
                    terminal_print('Warning, it seems there is not enough flux density for gain solving')
                    get_logger().warning('Warning, it seems there is not enough flux density for gain solving')
                    solint = np.rint(tgain_max * 3600. / tint)  # max is tgain_max (4) hrs

                # trigger 15 MHz smoothnessconstraint
                # print('TEST:', ((solint_sf*((noise/flux)**2)*(chanw/390.625e3))*tint/3600.))
                if ((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) < thr_SM15Mhz:  # so check if smaller than 2 hr
                    insmoothnessconstraint_list[soltype_id][ms_id] = 5.0*gainfactorsmoothness
                else:
                    terminal_print('Increasing smoothnessconstraint to 15 MHz')
                    get_logger().info('Increasing smoothnessconstraint to 15 MHz')
                    insmoothnessconstraint_list[soltype_id][ms_id] = 15.0*gainfactorsmoothness

                # trigger nchan=0 solve because not enough S/N
                if not longbaseline and (((solint_sf * ((noise / flux) ** 2) * (
                        chanw / 390.625e3)) * tint / 3600.) > thr_gain_trigger_allantenna):
                    inantennaconstraint_list[soltype_id][ms_id] = 'all'
                    solint = np.rint(
                        2.0 * 3600. / tint)  # 2 hrs nchan=0 solve (do not do bandpass because slope can diverge)
                    innchan_list[soltype_id][
                        ms_id] = 0  # no frequency dependence, smoothnessconstraint will be turned of in runDPPPbase
                    terminal_print('Triggering antennaconstraint all:', soltype, ms)
                    get_logger().info('Triggering antennaconstraint all: ' + soltype + ' ' + ms)
                else:
                    if inantennaconstraint_list[soltype_id][ms_id] != 'alldutch':
                        inantennaconstraint_list[soltype_id][ms_id] = None

                # completely disable slow solve if the solints get too long, target is too faint
                if (((solint_sf * ((noise / flux) ** 2) * (chanw / 390.625e3)) * tint / 3600.) > thr_disable_gain):
                    insoltypecycles_list[soltype_id][ms_id] = 999
                    terminal_print('Disabling solve:', soltype, ms)
                    get_logger().info('Disabling solve: ' + soltype + ' ' + ms)
                else:
                    insoltypecycles_list[soltype_id][
                        ms_id] = 3  # set to user input value? problem because not retained now (for example value can have been set to 999 above)

                insolint_list[soltype_id][ms_id] = int(solint)


    terminal_print('soltype:', soltype_list, mslist)
    terminal_print('nchan:', innchan_list)
    terminal_print('solint:', insolint_list)
    terminal_print('BLsmooth:', inBLsmooth_list)
    terminal_print('smoothnessconstraint:', insmoothnessconstraint_list)
    terminal_print('smoothnessreffrequency:', insmoothnessreffrequency_list)
    terminal_print('smoothnessspectralexponent_list:', insmoothnessspectralexponent_list)
    terminal_print('smoothnessrefdistance_list:', insmoothnessrefdistance_list)
    terminal_print('antennaconstraint:', inantennaconstraint_list)
    terminal_print('resetsols:', inresetsols_list)
    terminal_print('resetdir:', inresetdir_list)
    terminal_print('normamps:', innormamps_list)
    terminal_print('soltypecycles:', insoltypecycles_list)

    get_logger().info('soltype: ' + str(soltype_list) + ' ' + str(mslist))
    get_logger().info('nchan: ' + str(innchan_list))
    get_logger().info('solint: ' + str(insolint_list))
    get_logger().info('BLsmooth: ' + str(inBLsmooth_list))
    get_logger().info('smoothnessconstraint: ' + str(insmoothnessconstraint_list))
    get_logger().info('smoothnessreffrequency: ' + str(insmoothnessreffrequency_list))
    get_logger().info('smoothnessspectralexponent: ' + str(insmoothnessspectralexponent_list))
    get_logger().info('smoothnessrefdistance: ' + str(insmoothnessrefdistance_list))
    get_logger().info('antennaconstraint: ' + str(inantennaconstraint_list))
    get_logger().info('resetsols: ' + str(inresetsols_list))
    get_logger().info('resetdir: ' + str(inresetdir_list))
    get_logger().info('soltypecycles: ' + str(insoltypecycles_list))

    return innchan_list, insolint_list, inBLsmooth_list, insmoothnessconstraint_list, insmoothnessreffrequency_list, insmoothnessspectralexponent_list, insmoothnessrefdistance_list, inantennaconstraint_list, inresetsols_list, inresetdir_list, insoltypecycles_list, innormamps_list


