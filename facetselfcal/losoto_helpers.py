import os
import logging
import losoto
import numpy as np

from astropy import units
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord
from astropy.time import Time
from casacore.tables import taql
from losoto import h5parm
from pathlib import Path

from .plotting.plot_tecdelayphase import plot_tecdelayphase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/selfcal.log", delay=True)
formatter = logging.Formatter(
    "%(levelname)s:%(asctime)s ---- %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


def create_losoto_beamcorparset(ms: str, refant: str = "CS003HBA0"):
    """
    Create a LoSoTo parset to fill the beam correction values.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    refant : str
        Reference antenna name.

    Returns
    -------
    str
        Path to the generated LoSoTo parset.
    """
    parset = "losoto_parsets/losotobeam.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = [XX,YY]\n")
        f.write("soltab = [sol000/*]\n\n\n")

        f.write("[plotphase]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/phase000]\n")
        f.write("axesInPlot = [time,freq]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-0.5,0.5]\n")
        f.write("prefix = solution_plots_%s/phases_beam\n" % os.path.basename(ms))
        f.write("refAnt = %s\n\n\n" % refant)

        f.write("[plotamp]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/amplitude000]\n")
        f.write("axesInPlot = [time,freq]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [0.2,1]\n")
        f.write("prefix = solution_plots_%s/amplitudes_beam\n" % os.path.basename(ms))
    return parset


def losotolofarbeam(
    parmdb: str,
    soltabname: str,
    ms: str,
    inverse: bool = False,
    useElementResponse: bool = True,
    useArrayFactor: bool = True,
    useChanFreq: bool = True,
    beamlib: str = "stationresponse",
):
    """
    Do the beam correction via this imported losoto operation

    Parameters
    ----------
    parmdb : str
        path to the h5parm corrections will be stored in.
    soltabname : str
        name of the soltab corrections will be stored in.
    ms : str
        path to the MS (used to determine the stations present).
    inverse : bool
        calculate the inverse beam correction (i.e. undo the beam).
    useElementResponse : bool
        correct for the "element beam" (distance to the tile beam centre).
    useArrayFactor : bool
        correct for the "array factor" (sensitivity loss as function of distance to the pointing centre).
    useChanFreq : bool
        calculate a beam correction for every channel.
    beamlib : str
        beam calculation mode. Can be 'stationresponse' to use the LOFARBeam library (deprecated) or everybeam to use the EveryBeam library.

    Returns
    -------
    int or None
        Returns one when the requested beam operation cannot proceed; otherwise
        updates the solution table and returns ``None``.
    """

    H5 = h5parm.h5parm(parmdb, readonly=False)
    soltab = H5.getSolset("sol000").getSoltab(soltabname)
    times = soltab.getAxisValues("time")
    numants = taql("select gcount(*) as numants from " + ms + "::ANTENNA").getcol(
        "numants"
    )[0]
    H5ants = len(soltab.getAxisValues("ant"))
    if numants != H5ants:
        H5.close()
        raise ValueError(
            "Number of antennas in Measurement Set does not match number of antennas in H5parm."
        )

    if (beamlib.lower() == "stationresponse") or (beamlib.lower() == "lofarbeam"):
        from lofar.stationresponse import stationresponse  # type: ignore

        sr = stationresponse(
            ms, inverse, useElementResponse, useArrayFactor, useChanFreq
        )

        for vals, coord, selection in soltab.getValuesIter(
            returnAxes=["ant", "time", "pol", "freq"], weight=False
        ):
            vals = losoto.lib_operations.reorderAxes(
                vals, soltab.getAxesNames(), ["ant", "time", "freq", "pol"]
            )

            for stationnum in range(numants):
                logger.debug(
                    "stationresponse working on station number %i" % stationnum
                )
                for itime, time in enumerate(times):
                    beam = sr.evaluateStation(time=time, station=stationnum)
                    # Reshape from [nfreq, 2, 2] to [nfreq, 4]
                    beam = beam.reshape(beam.shape[0], 4)

                    if soltab.getAxisLen("pol") == 2:
                        beam = beam[:, [0, 3]]  # get only XX and YY

                    if soltab.getType() == "amplitude":
                        vals[stationnum, itime, :, :] = np.abs(beam)
                    elif soltab.getType() == "phase":
                        vals[stationnum, itime, :, :] = np.angle(beam)
                    else:
                        logger.error(
                            "Beam prediction works only for amplitude/phase solution tables."
                        )
                        return 1

            vals = losoto.lib_operations.reorderAxes(
                vals,
                ["ant", "time", "freq", "pol"],
                [
                    ax
                    for ax in soltab.getAxesNames()
                    if ax in ["ant", "time", "freq", "pol"]
                ],
            )
            soltab.setValues(vals, selection)
    elif beamlib.lower() == "everybeam":
        from tqdm import tqdm
        from joblib import Parallel, delayed, parallel_backend
        import psutil

        freqs = soltab.getAxisValues("freq")

        # Obtain direction to calculate beam for.
        dirs = taql("SELECT REFERENCE_DIR,PHASE_DIR FROM {ms:s}::FIELD".format(ms=ms))
        ra_ref, dec_ref = dirs.getcol("REFERENCE_DIR").squeeze()
        ra, dec = dirs.getcol("PHASE_DIR").squeeze()
        reference_xyz = list(
            zip(*radec_to_xyz(ra_ref * units.rad, dec_ref * units.rad, times))
        )
        phase_xyz = list(zip(*radec_to_xyz(ra * units.rad, dec * units.rad, times)))

        for vals, coord, selection in soltab.getValuesIter(
            returnAxes=["ant", "time", "pol", "freq"], weight=False
        ):
            vals = losoto.lib_operations.reorderAxes(
                vals, soltab.getAxesNames(), ["ant", "time", "freq", "pol"]
            )
            stationloop = tqdm(range(numants))
            stationloop.set_description("Stations processed: ")
            for stationnum in range(numants):
                stationloop.update()
                logger.debug("everybeam working on station number %i" % stationnum)
                # Parallelise over channels to speed things along.
                with parallel_backend(
                    "loky", n_jobs=len(psutil.Process().cpu_affinity())
                ):
                    results = Parallel()(
                        delayed(process_channel_everybeam)(
                            f,
                            stationnum=stationnum,
                            useElementResponse=useElementResponse,
                            useArrayFactor=useArrayFactor,
                            useChanFreq=useChanFreq,
                            ms=ms,
                            freqs=freqs,
                            times=times,
                            ra=ra,
                            dec=dec,
                            ra_ref=ra_ref,
                            dec_ref=dec_ref,
                            reference_xyz=reference_xyz,
                            phase_xyz=phase_xyz,
                        )
                        for f in range(len(freqs))
                    )
                    for freqslot in results:
                        ifreq, beam = freqslot
                        if soltab.getAxisLen("pol") == 2:
                            beam = beam.reshape((beam.shape[0], 4))[
                                :, [0, 3]
                            ]  # get only XX and YY
                        if soltab.getType() == "amplitude":
                            vals[stationnum, :, ifreq, :] = np.abs(beam)
                        elif soltab.getType() == "phase":
                            vals[stationnum, :, ifreq, :] = np.angle(beam)
                        else:
                            logger.error(
                                "Beam prediction works only for amplitude/phase solution tables."
                            )
                            return 1
            vals = losoto.lib_operations.reorderAxes(
                vals,
                ["ant", "time", "freq", "pol"],
                [
                    ax
                    for ax in soltab.getAxesNames()
                    if ax in ["ant", "time", "freq", "pol"]
                ],
            )
            soltab.setValues(vals, selection)

    else:
        H5.close()
        raise ValueError("Unsupported beam library specified")

    H5.close()
    return


def process_channel_everybeam(
    ifreq,
    stationnum,
    useElementResponse,
    useArrayFactor,
    useChanFreq,
    ms,
    freqs,
    times,
    ra,
    dec,
    ra_ref,
    dec_ref,
    reference_xyz,
    phase_xyz,
):
    """
    Calculate EveryBeam response values for one channel and station.

    Parameters
    ----------
    ifreq : int
        Frequency index.
    stationnum : int
        Station index.
    useElementResponse : bool
        Include element response.
    useArrayFactor : bool
        Include array factor.
    useChanFreq : bool
        Use channel frequencies.
    ms : str
        Measurement Set path.
    freqs : ndarray
        Channel frequencies in hertz.
    times : ndarray
        Observation times for the response calculation.
    ra : float
        Right ascension of the requested direction, in radians.
    dec : float
        Declination of the requested direction, in radians.
    ra_ref : float
        Right ascension of the reference direction, in radians.
    dec_ref : float
        Declination of the reference direction, in radians.
    reference_xyz : ndarray
        Reference-direction Cartesian coordinates for each time sample.
    phase_xyz : ndarray
        Phase-center Cartesian coordinates for each time sample.

    Returns
    -------
    tuple
        Channel index and calculated beam response.
    """
    import everybeam  # type: ignore

    if useElementResponse and useArrayFactor:
        # print('Full (element+array_factor) beam correction requested. Using use_differential_beam=False.')
        obs = everybeam.load_telescope(
            ms, use_differential_beam=False, use_channel_frequency=useChanFreq
        )
    elif not useElementResponse and useArrayFactor:
        # print('Array factor beam correction requested. Using use_differential_beam=True.')
        obs = everybeam.load_telescope(
            ms, use_differential_beam=True, use_channel_frequency=useChanFreq
        )
    elif useElementResponse and not useArrayFactor:
        # print('Element beam correction requested.')
        # Not sure how to do this with EveryBeam.
        raise NotImplementedError(
            "Element beam only correction is not implemented in facetselfcal."
        )

    # print(f'Processing channel {ifreq}')
    freq = freqs[ifreq]
    timeslices = np.empty((len(times), 2, 2), dtype=np.complex128)
    for itime, time in enumerate(times):
        # timeloop.update()
        if not useElementResponse and useArrayFactor:
            # Array-factor-only correction.
            beam = obs.array_factor(
                times[itime], stationnum, freq, phase_xyz[itime], reference_xyz[itime]
            )
        else:
            beam = obs.station_response(
                time=time, station_idx=stationnum, freq=freq, ra=ra, dec=dec
            )
        # beam = beam.reshape(4)
        timeslices[itime] = beam
    return ifreq, timeslices


def radec_to_xyz(ra, dec, time):
    """
    Convert ra and dec coordinates to ITRS coordinates for LOFAR observations.

    Parameters
    ----------
    ra : astropy Quantity
        right ascension
    dec : astropy Quantity
        declination
    time : float
        MJD time in seconds

    Returns
    -------
    pointing_xyz : ndarray
        NumPy array containing the X, Y and Z coordinates
    """
    obstime = Time(time / 3600 / 24, scale="utc", format="mjd")
    loc_LOFAR = EarthLocation(
        lon=0.11990128407256424,
        lat=0.9203091252660295,
        height=6364618.852935438 * units.m,
    )

    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_altaz = dir_pointing.transform_to(
        AltAz(obstime=obstime, location=loc_LOFAR)
    )
    dir_pointing_xyz = dir_pointing_altaz.transform_to(ITRS)

    pointing_xyz = np.asarray(
        [dir_pointing_xyz.x, dir_pointing_xyz.y, dir_pointing_xyz.z]
    )
    return pointing_xyz


def tecandphaseplotter(h5, ms, telescope="LOFAR", outplotname="plot.png"):
    """
    Make TEC and phase plots.

    Parameters
    ----------
    h5 : str
        path to the H5parm to plot.
    ms : str
        path to th ecorresponding Measurement Set.
    telescope : str
        telescope name.
    outplotname : str
        name of the output plot.

    Returns
    -------
    None
    """
    if not os.path.isdir(
        "solution_plots_%s" % os.path.basename(ms)
    ):  # needed because if this is the first plot this directory does not yet exist
        os.makedirs("solution_plots_%s" % os.path.basename(ms), exist_ok=True)
    plot_tecdelayphase(
        h5file=h5,
        outfile=f"solution_plots_{os.path.basename(ms)}/{os.path.basename(outplotname)}_nolosoto.png",
        telescope=telescope,
    )
