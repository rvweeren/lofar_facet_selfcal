"""Inspect and smooth h5parm solutions: deviating antennas, FFT delay search,
amplitude noise and losoto smoothing, plot marker size."""
import matplotlib
matplotlib.use('Agg')
import astropy.units as units  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tables  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from casacore.tables import table  # type: ignore # noqa: E402
from losoto import h5parm  # type: ignore # noqa: E402

from ..image_utils import findrms  # noqa: E402
from ..losoto_helpers import create_losoto_mediumsmoothparset  # noqa: E402
from ..runtime_utils import get_logger, run, terminal_print  # noqa: E402
from .h5_info import (amplitude_leakage_paramdb, check_soltabs,  # noqa: E402
                      fulljonesparmdb, ntimesH5, number_freqchan_h5)


def FFTdelayfinder(h5, refant):
    """
    Estimate and plot antenna delays from phase solutions.

    Parameters
    ----------
    h5 : str
        H5 solution file containing ``sol000/phase000``.
    refant : str
        Reference antenna name.

    Returns
    -------
    None
        The delay plot is displayed and no value is returned.
    """
    from scipy.fftpack import fft, fftfreq
    H = tables.open_file(h5)
    upsample_factor = 10

    # reference to refant
    refant_idx = np.where(H.root.sol000.phase000.ant[:] == refant)
    phase = H.root.sol000.phase000.val[:]
    phasen = phase - phase[:, :, refant_idx[0], :]

    phasecomplex = np.exp(phasen * 1j)
    freq = H.root.sol000.phase000.freq[:]
    timeaxis = H.root.sol000.phase000.time[:]
    timeaxis = timeaxis - np.min(timeaxis)

    delayaxis = fftfreq(upsample_factor * freq.size,
                        d=np.abs(freq[1] - freq[0]) / float(upsample_factor))

    for ant_id, ant in enumerate(H.root.sol000.phase000.ant[:]):
        delay = 0.0 * H.root.sol000.phase000.time[:]
        terminal_print('FFT delay finding for:', ant)
        for time_id, time in enumerate(H.root.sol000.phase000.time[:]):
            delay[time_id] = delayaxis[
                np.argmax(np.abs(fft(phasecomplex[time_id, :, ant_id, 0], n=upsample_factor * len(freq))))]
        plt.plot(timeaxis / 3600., delay * 1e9)
    plt.ylim(-2e-6 * 1e9, 2e-6 * 1e9)
    plt.ylabel('Delay [ns]')
    plt.xlabel('Time [hr]')
    # plt.title(ant)
    plt.show()
    H.close()
    return


def compute_markersize(H5file):
    """
    Computes matplotlib markersize for an H5parm.

    Parameters
    ----------
    H5file : str
        path to an H5parm.

    Returns
    -------
    markersize : int
        marker size.
    """
    ntimes = ntimesH5(H5file)
    markersize = 2
    if ntimes < 450:
        markersize = 4
    if ntimes < 100:
        markersize = 10
    if ntimes < 50:
        markersize = 15
    
    if ntimes == 1:
        markersize = 2
        nfreqs = number_freqchan_h5(H5file)
        if nfreqs < 450:
            markersize = 4
        if nfreqs < 100:
            markersize = 10
        if nfreqs < 50:
            markersize = 15
    
    return markersize


def findamplitudenoise(parmdb):
    """
    Find the amplitude noise in a parmdb and return the non-clipped RMS value.

    Parameters
    ----------
    parmdb : str
        Path to the parmdb / H5Parm.

    Returns
    -------
    float
        RMS amplitude noise.
    """
    with h5parm.h5parm(parmdb, readonly=True) as H5:
        amps = H5.getSolset('sol000').getSoltab('amplitude000').getValues()[0]
        weights = H5.getSolset('sol000').getSoltab('amplitude000').getValues(weight=True)[0]

    idx = np.where(weights != 0.0)

    amps = amps[idx]
    amps = amps[np.isfinite(amps)]
    amps = np.log10(np.ndarray.flatten(amps))

    noise = findrms(amps)

    get_logger().info('Noise and clipped noise' + str(parmdb) + ' ' + str(np.std(amps)) + ' ' + str(noise))

    # do not return clipped noise, we are intersted in finding data with high outliers
    return np.std(amps)


def smoothsols(parmdb, ms, longbaseline, includesphase=True, skip_plotting: bool = False):
    """
    Smooth calibration solutions for a baseline class.

    Parameters
    ----------
    parmdb : str
        Parameter database path.
    ms : str
        Measurement Set path.
    longbaseline : bool
        Whether long-baseline smoothing is required.
    includesphase : bool, optional
        Whether phase solutions are present.

    Returns
    -------
    str
        Generated smoothing parset path.
    """
    losoto = 'losoto'

    cmdlosoto = losoto + ' ' + parmdb + ' '
    noise = findamplitudenoise(parmdb)
    smooth = False
    if noise >= 0.1:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '9', longbaseline, includesphase=includesphase, skip_solution_plotting=skip_plotting)
        smooth = True
    if noise < 0.1 and noise >= 0.08:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '7', longbaseline, includesphase=includesphase, skip_solution_plotting=skip_plotting)
        smooth = True
    if noise < 0.08 and noise >= 0.07:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '5', longbaseline, includesphase=includesphase, skip_solution_plotting=skip_plotting)
        smooth = True
    if noise < 0.07 and noise >= 0.04:
        cmdlosoto += create_losoto_mediumsmoothparset(ms, '3', longbaseline, includesphase=includesphase, skip_solution_plotting=skip_plotting)
        smooth = True
    if smooth:
        terminal_print('LoSoTo command:', cmdlosoto)
        get_logger().info(cmdlosoto)
        run(cmdlosoto)
    return


def find_bad_deviating_antennas(h5, ms, threshold=0.075):
    """
    Find antennas with high deviation in amplitude solutions.
    Useful for MeerKAT observations to deal with antennas that have a bad pointing accuracy
    First compute the median amplitude solutions, by taking the median along the antenna axis of the h5
    The find antennas that deviate more than threshold (default 30%) from the median

    Parameters
    ----------
    h5 : str
        Path to the H5 parmdb file.
    ms : str
        Measurement Set used to obtain the field reference direction.
    threshold : float
        Deviation threshold to identify bad antennas.

    Returns
    -------
    list
        List of bad antenna names.
    """
    fulljones = fulljonesparmdb(h5)  # True/False
    amplitudeleakage = amplitude_leakage_paramdb(h5)  # True/False
    hasphase, hasamps, hasrotation, hastec, hasrotationmeasure, hasdelay = check_soltabs(h5)
    if not hasamps:
        terminal_print('No amplitude000 solutions found in', h5)
        raise Exception('No amplitude000 solutions found in ' + h5)
    if fulljones or amplitudeleakage:
        terminal_print('Amplitude solutions are fulljones or have amplitude leakage, cannot determine bad antennas based on amplitude deviations')
        raise Exception('Amplitude solutions are fulljones or have amplitude leakage, cannot determine bad antennas based on amplitude deviations')

    with table(ms + '::FIELD', readonly=True, ack=False) as t:
        ra_ref, dec_ref = t.getcol('REFERENCE_DIR').squeeze()
        center_ms = SkyCoord(ra_ref * units.radian, dec_ref * units.radian, frame='icrs')
        #print(center_ms)
    H = tables.open_file(h5)

    amps = H.root.sol000.amplitude000.val[:]
    weights = H.root.sol000.amplitude000.weight[:]
    antennas = H.root.sol000.amplitude000.ant[:]
    freqs = H.root.sol000.amplitude000.freq[:]
    directions = H.root.sol000.source[:]

    axisn = H.root.sol000.amplitude000.val.attrs['AXES'].decode().split(',')
    ndir = len(H.root.sol000.amplitude000.dir[:])
    H.close()
    badants = []
    # Find the antenna axis
    antennaxis = axisn.index('ant')

    # maximum radius in arcmin for MeerKAT at highest frequency to trust solutions
    max_radius = 2.0 * 57.5 * (1.5e9/np.max(freqs))/2 # radius of 2.0xFWHM_maxfreq arcmin

    # Compute median amplitude solutions along the antenna axis
    median_amps = np.median(amps, axis=antennaxis, keepdims=True)
    terminal_print('Median amplitudes shape:', median_amps.shape)
    terminal_print('Amplitudes shape:', amps.shape)
    # Calculate deviation from median for each antenna
    # the deviation array should have the same shape as amps
    deviation = np.log10(amps) - np.log10(median_amps)   
    terminal_print('Deviation shape:', deviation.shape)
    
    
    if False: # for testing
        # set backend to TkAgg for interactive plotting
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
    
        for antid, ant in enumerate(antennas):
            #plt.imshow(deviation[:,:,antid,1].T , aspect='auto', cmap='viridis', vmin=0.0, vmax=0.02)
            #plt.colorbar(label='Mean log10 deviation from median amplitude')
            #plt.title('Antenna: ' + (ant.decode() if isinstance(ant, bytes) else ant) +' ' + np.median(deviation[:,:,antid,1]).astype(str))
            #plt.ylabel('Frequency channels')
            #plt.xlabel('Time slots')
            #plt.show()
            ww = weights[:,:,antid,1]
            dd = deviation[:,:,antid,1]
            # get values where weights are zero
            idx = np.where(ww == 0.)
            dn = dd.copy()
            dn[idx] = np.nan 
            terminal_print('Antenna: ' + (ant.decode() if isinstance(ant, bytes) else ant) +' ' + str(np.median(deviation[:,:,antid,1])) +' with weights: ' + str(np.nanmedian(dn)))

        matplotlib.use('Agg')  # reset backend to non-interactive
    
    # now create an array with one value, taking the median over time and frequency for each antenna, keep all other axis like polarization and direction in the array
    # --only-- take median along the time,freq axis(!)
    timeaxis = axisn.index('time')
    freqaxis = axisn.index('freq')
    terminal_print('Time axis:', timeaxis, 'Freq axis:', freqaxis)

    # also take the weights into, so do not consider data points with zero weight 
    idx = np.where(weights == 0.)
    # put in NaN values where weights are zero
    deviation[idx] = np.nan

    deviation_per_ant = np.nanmean(deviation, axis=(timeaxis, freqaxis), keepdims=False)
    terminal_print('Mean deviation per antenna shape:', deviation_per_ant.shape)
    terminal_print('Min/max values deviation per antenna', np.nanmin(deviation_per_ant), np.nanmax(deviation_per_ant))
    #print('Deviation per antenna shape:', deviation_per_ant)
    if 'pol' in axisn:
        polaxis = axisn.index('pol')
        # find number of polarizations
        npol = amps.shape[polaxis]
        for pol_id in range(npol):
            for direction_id, direction in enumerate(directions):
                for ant_id, ant in enumerate(antennas):
                    # check if median deviation value is finite and larger than +threshold and smaller than -threshold
                    if np.isfinite(deviation_per_ant[ant_id, direction_id, pol_id]):
                        ra_dir, dec_dir = direction[1]
                        c2 = SkyCoord(ra_dir * units.radian, dec_dir * units.radian, frame='icrs')
                        angsep = center_ms.separation(c2).to(units.arcmin).value
                        # check angular separation to center because solutions far out might be quite bad due to primary beam effects
                        if angsep <= max_radius and (np.abs(deviation_per_ant[ant_id, direction_id, pol_id]) > threshold):
                            terminal_print('Antenna, direction, polarization:', ant.decode() if isinstance(ant, bytes) else ant, direction_id, pol_id, 'Mean deviation:', deviation_per_ant[ant_id, direction_id, pol_id], 'marked as bad antenna')
                            terminal_print('Angular separation solution direction to center (arcmin):', angsep, 'max radius:', max_radius)
                            # convert ant to normal string if it is byte string
                            badants.append(ant.decode() if isinstance(ant, bytes) else ant)
    else: # for h5 without polarization axis
            for direction_id, direction in enumerate(directions):
                for ant_id, ant in enumerate(antennas):
                    ra_dir, dec_dir = direction[1]
                    c2 = SkyCoord(ra_dir * units.radian, dec_dir * units.radian, frame='icrs')
                    angsep = center_ms.separation(c2).to(units.arcmin).value
                    # check angular separation to center because solutions far out might be quite bad due to primary beam effects
                    if angsep <= max_radius and np.isfinite(deviation_per_ant[ant_id, direction_id]):
                        if (np.abs(deviation_per_ant[ant_id, direction_id]) > threshold):
                            terminal_print('Antenna, direction, polarization:', ant.decode() if isinstance(ant, bytes) else ant, direction_id, 'Mean deviation:', deviation_per_ant[ant_id, direction_id], 'marked as bad antenna')
                            terminal_print('Angular separation solution direction to center (arcmin):', angsep, 'max radius:', max_radius)
                            # convert ant to normal string if it is byte string
                            badants.append(ant.decode() if isinstance(ant, bytes) else ant)

    # only keep the unique bad antennas
    badants = list(set(badants)) 
    return badants
