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
from submods.h5_helpers.modify_amplitude import get_median_amp

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


def create_losoto_delayparset(
    ms: str,
    refant: str = "CS003HBA0",
    outplotname: str = "fastdelay",
    markersize: int = 2,
):
    """
    Create a LoSoTo parset for plotting delay solutions.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    refant : str, optional
        Reference antenna name.
    outplotname : str, optional
        Base name for the output plot.
    markersize : int, optional
        Plot marker size.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_plotfastdelay.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = []\n")
        f.write("Ncpu = 0\n\n\n")
        f.write("[plotdelay]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/delay000]\n")
        f.write("axesInPlot = [time]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-0.2,0.2]\n")
        f.write("figSize=[120,20]\n")
        f.write("markerSize=%s\n" % int(markersize))
        f.write(
            "prefix = solution_plots_%s/%s\n"
            % (os.path.basename(ms), os.path.basename(outplotname))
        )
        f.write("refAnt = %s\n" % refant)

    return parset


def create_losoto_fastphaseparset(
    ms,
    refant="CS003HBA0",
    onechannel=False,
    onepol=False,
    outplotname="fastphase",
    onetime=False,
    markersize=2,
):
    """
    Create a LoSoTo parset for plotting fast phase solutions.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    refant : str, optional
        Reference antenna name.
    onechannel : bool, optional
        Plot one channel along time only.
    onepol : bool, optional
        Omit polarization-specific plots.
    outplotname : str, optional
        Base name for the output plot.
    onetime : bool, optional
        Plot one time along frequency only.
    markersize : int, optional
        Plot marker size.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_plotfastphase.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = [XX,YY]\n")
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")

        f.write("[plotphase]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/phase000]\n")
        if onechannel:
            f.write("axesInPlot = [time]\n")
            if not onepol:
                f.write("axisInCol = pol\n")
        if onetime:
            f.write("axesInPlot = [freq]\n")
            if not onepol:
                f.write("axisInCol = pol\n")
        if onechannel or onetime:
            f.write("markerSize=%s\n" % int(markersize))
        if not onechannel and not onetime:
            f.write("axesInPlot = [time,freq]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-3.14,3.14]\n")
        f.write("figSize=[120,20]\n")
        f.write(
            "prefix = solution_plots_%s/%s\n"
            % (os.path.basename(ms), os.path.basename(outplotname))
        )
        f.write("refAnt = %s\n" % refant)

        if not onepol:
            f.write("[plotphasediff]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = [sol000/phase000]\n")
            if onechannel:
                f.write("axesInPlot = [time]\n")
            if onetime:
                f.write("axesInPlot = [freq]\n")
            if onechannel or onetime:
                f.write("markerSize=%s\n" % int(markersize))
            if not onechannel and not onetime:
                f.write("axesInPlot = [time,freq]\n")
            f.write("axisInTable = ant\n")
            f.write("minmax = [-3.14,3.14]\n")
            f.write("figSize=[120,20]\n")
            f.write(
                "prefix = solution_plots_%s/%spoldiff\n"
                % (os.path.basename(ms), os.path.basename(outplotname))
            )
            f.write("refAnt = %s\n" % refant)
            f.write("axisDiff=pol\n")

    return parset


def create_losoto_FRparset(
    ms: str,
    refant: str = "CS001LBA",
    freqminfitFR: float = 20e6,
    outplotname: str = "FR",
    onlyplotFRfit: bool = False,
    dejump: bool = False,
    skip_solution_plotting: bool = False,
):
    """
    Create a LoSoTo parset to fit Faraday Rotation on the phase difference.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    refant : str
        Reference antenna name.
    freqminfitFR : float, optional
        Minimum frequency for fit in MHz.
    outplotname : str, optional
        Output plot path.
    onlyplotFRfit : bool, optional
        Only plot the fit without applying.
    dejump : bool, optional
        Dejump phase solutions.
    skip_solution_plotting : bool, optional
        Skip plotting of the solutions.

    Returns
    -------
    str
        Path to the generated LoSoTo parset.
    """
    parset = "losoto_parsets/losotoFR.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("[duplicate]\n")
        f.write("operation = DUPLICATE\n")
        f.write("soltab = sol000/phase000\n")
        f.write("soltabOut = phaseOrig000\n\n\n")

        # should not be needed as already done but just in case....
        f.write("[reset]\n")
        f.write("operation = RESET\n")
        f.write("soltab = sol000/phase000\n")
        f.write("pol = YY\n")
        f.write("dataVal = 0.0\n\n\n")

        if not skip_solution_plotting:
            f.write("[plotphase]\n")
            f.write("pol = [XX,YY]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = [sol000/phase000]\n")
            f.write("axesInPlot = [time,freq]\n")
            f.write("axisInTable = ant\n")
            f.write("minmax = [-3.14,3.14]\n")
            f.write(
                "prefix = solution_plots_%s/%s\n"
                % (
                    os.path.basename(ms),
                    os.path.basename(outplotname) + "phases_beforeFR",
                )
            )
            f.write("refAnt = %s\n\n\n" % refant)

        f.write("[faraday]\n")
        f.write("operation = FARADAY\n")
        f.write("soltab = sol000/phase000\n")
        f.write("refAnt = %s\n" % refant)
        f.write("maxResidual = 2\n")
        f.write("freq.minmaxstep = [%s,1e9]\n" % str(freqminfitFR))
        f.write("soltabOut = rotationmeasure000\n\n\n")

        if not skip_solution_plotting:
            f.write("[plotFR]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = sol000/rotationmeasure000\n")
            f.write("axesInPlot = [time]\n")
            f.write("axisInTable = ant\n")
            f.write(
                "prefix = solution_plots_%s/%s\n\n\n"
                % (os.path.basename(ms), os.path.basename(outplotname) + "FR")
            )

        if dejump:
            f.write("[frdejump]\n")
            f.write("operation = FRJUMP\n")
            f.write("soltab = sol000/rotationmeasure000\n")
            f.write("soltabOut = rotationmeasure001\n")
            f.write("clipping = [%s,1e9]\n\n\n" % str(freqminfitFR))

            if not skip_solution_plotting:
                f.write("[plotFR_dejump]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = sol000/rotationmeasure001\n")
                f.write("axesInPlot = [time]\n")
                f.write("axisInTable = ant\n")
                f.write(
                    "prefix = solution_plots_%s/%s\n\n\n"
                    % (
                        os.path.basename(ms),
                        os.path.basename(outplotname) + "FRdejumped",
                    )
                )

        f.write("[residuals]\n")
        f.write("operation = RESIDUALS\n")
        f.write("soltab = sol000/phase000\n")
        if dejump:
            f.write("soltabsToSub = rotationmeasure001\n\n\n")
        else:
            f.write("soltabsToSub = rotationmeasure000\n\n\n")

        if not skip_solution_plotting:
            f.write("[plotRES]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = sol000/phase000\n")
            f.write("axesInPlot = [time,freq]\n")
            f.write("AxisInTable = ant\n")
            f.write("AxisDiff = pol\n")
            f.write("plotFlag = True\n")
            f.write(
                "prefix = solution_plots_%s/%s\n"
                % (
                    os.path.basename(ms),
                    os.path.basename(outplotname) + "residualphases_afterFR",
                )
            )
            f.write("refAnt = %s\n" % refant)
            f.write("minmax = [-3.14,3.14]\n\n\n")

    return parset


def create_losoto_FRparsetplotfit(
    ms: str,
    refant: str = "CS001LBA",
    outplotname: str = "FR",
):
    """
    Create a LoSoTo parset to fit Faraday Rotation on the phase difference.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    refant : str
        Reference antenna name.
    outplotname : str
        Output plot path.

    Returns
    -------
    str
        Path to the generated LoSoTo parset.
    """
    parset = "losoto_parsets/losotoFR_plotresult.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("[plotFRresult]\n")
        f.write("pol = [XX,YY]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/phase000]\n")
        f.write("axesInPlot = [time,freq]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-3.14,3.14]\n")
        f.write(
            "prefix = solution_plots_%s/%s\n"
            % (os.path.basename(ms), os.path.basename(outplotname) + "phases_fitFR")
        )
        f.write("refAnt = %s\n\n\n" % refant)

    return parset


def create_losoto_rotationparset(
    ms: str,
    refant: str = "CS003HBA0",
    onechannel: bool = False,
    outplotname: str = "rotation",
    markersize: int = 2,
):
    """
    Create a LoSoTo parset for plotting rotation solutions.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    refant : str, optional
        Reference antenna name.
    onechannel : bool, optional
        Plot one channel along time only.
    outplotname : str, optional
        Base name for the output plot.
    markersize : int, optional
        Plot marker size.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_plotrotation.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = [XX,YY]\n")
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")
        f.write("[plotrotation]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/rotation000]\n")
        f.write("markerSize=%s\n" % int(markersize))
        if onechannel:
            f.write("axesInPlot = [time]\n")
        else:
            f.write("axesInPlot = [time,freq]\n")
        f.write("axisInTable = ant\n")
        f.write(
            "minmax = [-1.57,1.57]\n"
        )  # rotation needs to be plotted from -pi/2 to pi/2
        f.write("figSize=[120,20]\n")
        f.write(
            "prefix = solution_plots_%s/%s\n"
            % (os.path.basename(ms), os.path.basename(outplotname))
        )
        f.write("refAnt = %s\n" % refant)

    return parset


def create_losoto_tecparset(
    ms: str,
    refant: str = "CS003HBA0",
    outplotname: str = "fasttec",
    markersize: int = 2,
):
    """
    Create a LoSoTo parset for plotting TEC solutions.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    refant : str, optional
        Reference antenna name.
    outplotname : str, optional
        Base name for the output plot.
    markersize : int, optional
        Plot marker size.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_plotfasttec.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = []\n")
        f.write("Ncpu = 0\n\n\n")
        f.write("[plottec]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/tec000]\n")
        f.write("axesInPlot = [time]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-0.2,0.2]\n")
        f.write("figSize=[120,20]\n")
        f.write("markerSize=%s\n" % int(markersize))
        f.write(
            "prefix = solution_plots_%s/%s\n"
            % (os.path.basename(ms), os.path.basename(outplotname))
        )
        f.write("refAnt = %s\n" % refant)

    return parset


def create_losoto_tecandphaseparset(
    ms: str,
    refant: str = "CS003HBA0",
    outplotname: str = "fasttecandphase",
    markersize: int = 2,
):
    """
    Create a LoSoTo parset for plotting TEC and phase solutions.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    refant : str, optional
        Reference antenna name.
    outplotname : str, optional
        Base name for the output plot.
    markersize : int, optional
        Plot marker size.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_plotfasttecandphase.parset"
    Path(parset).unlink(missing_ok=True)
    with open(parset, "w") as f:
        f.write("pol = []\n")
        f.write("Ncpu = 0\n\n\n")
        f.write("[plottecandphase]\n")
        f.write("operation = PLOT\n")
        f.write("soltab = [sol000/phase000]\n")
        f.write("axesInPlot = [time]\n")
        f.write("axisInTable = ant\n")
        f.write("minmax = [-3.14,3.14]\n")
        f.write("soltabToAdd = tec000\n")
        f.write("figSize=[120,20]\n")
        f.write("markerSize=%s\n" % int(markersize))
        f.write("prefix = solution_plots_%s/fasttecandphase\n" % os.path.basename(ms))
        f.write("refAnt = %s\n" % refant)

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


def create_losoto_flag_apgridparset(
    ms: str,
    flagging: bool = True,
    maxrms: float = 7.0,
    maxrmsphase: float = 7.0,
    includesphase: bool = True,
    refant: str = "CS003HBA0",
    onechannel: bool = False,
    medamp: float = 2.5,
    flagphases: bool = True,
    onepol: bool = False,
    outplotname: str = "slowamp",
    fulljones: bool = False,
    onetime: bool = False,
    markersize: int = 2,
    skip_solution_plotting: bool = False,
):
    """
    Create a LoSoTo parset for grid-based solution flagging.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the output context.
    flagging : bool, optional
        Enable flagging operations.
    maxrms : float, optional
        Maximum amplitude RMS.
    maxrmsphase : float, optional
        Maximum phase RMS.
    includesphase : bool, optional
        Whether phase solutions are present.
    refant : str, optional
        Reference antenna name.
    onechannel : bool, optional
        Flag one channel along time.
    medamp : float, optional
        Median amplitude used for plotting limits.
    flagphases : bool, optional
        Enable phase flagging.
    onepol : bool, optional
        Omit polarization-specific plots.
    outplotname : str, optional
        Base name for output plots.
    fulljones : bool, optional
        Use full-Jones solution settings.
    onetime : bool, optional
        Flag one time along frequency.
    markersize : int, optional
        Plot marker size.
    skip_solution_plotting : bool, optional
        Skip plotting of the solutions.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_flag_apgrid.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        # f.write('pol = []\n')
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")

        if not skip_solution_plotting:
            f.write("[plotamp]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = [sol000/amplitude000]\n")
            if onechannel:
                f.write("axesInPlot = [time]\n")
                if not onepol:
                    f.write("axisInCol = pol\n")
            if onetime:
                f.write("axesInPlot = [freq]\n")
                if not onepol:
                    f.write("axisInCol = pol\n")
            if onechannel or onetime:
                f.write("markerSize=%s\n" % int(markersize))
            if not onechannel and not onetime:
                f.write("axesInPlot = [time,freq]\n")
            f.write("axisInTable = ant\n")
            # if longbaseline:
            #  f.write('minmax = [0,2.5]\n')
            # else:
            f.write("minmax = [%s,%s]\n" % (str(medamp / 4.0), str(medamp * 2.5)))
            # f.write('minmax = [0,2.5]\n')
            f.write(
                "prefix = solution_plots_%s/%samp\n\n\n"
                % (os.path.basename(ms), os.path.basename(outplotname))
            )

            if fulljones:
                f.write("[plotampXYYX]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/amplitude000]\n")
                f.write("pol = [XY, YX]\n")
                if onechannel:
                    f.write("axesInPlot = [time]\n")
                if onetime:
                    f.write("axesInPlot = [freq]\n")
                if onechannel or onetime:
                    f.write("markerSize=%s\n" % int(markersize))
                if not onetime and not onechannel:
                    f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [%s,%s]\n" % (str(0.0), str(0.5)))
                f.write(
                    "prefix = solution_plots_%s/%sampXYYX\n\n\n"
                    % (os.path.basename(ms), os.path.basename(outplotname))
                )

            if includesphase:
                f.write("[plotphase]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/phase000]\n")
                if onechannel:
                    f.write("axesInPlot = [time]\n")
                    if not onepol:
                        f.write("axisInCol = pol\n")
                if onetime:
                    f.write("axesInPlot = [freq]\n")
                    if not onepol:
                        f.write("axisInCol = pol\n")
                if onechannel or onetime:
                    f.write("markerSize=%s\n" % int(markersize))
                if not onetime and not onechannel:
                    f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [-3.14,3.14]\n")
                f.write(
                    "prefix = solution_plots_%s/%sphase\n"
                    % (os.path.basename(ms), os.path.basename(outplotname))
                )
                f.write("refAnt = %s\n\n\n" % refant)

                if not onepol and not fulljones:
                    f.write("[plotphasediff]\n")
                    f.write("operation = PLOT\n")
                    f.write("soltab = [sol000/phase000]\n")
                    if onechannel:
                        f.write("axesInPlot = [time]\n")
                    if onetime:
                        f.write("axesInPlot = [freq]\n")
                    if onechannel or onetime:
                        f.write("markerSize=%s\n" % int(markersize))
                    if not onetime and not onechannel:
                        f.write("axesInPlot = [time,freq]\n")
                    f.write("axisInTable = ant\n")
                    f.write("minmax = [-3.14,3.14]\n")
                    f.write("figSize=[120,20]\n")
                    f.write(
                        "prefix = solution_plots_%s/%spoldiff\n"
                        % (os.path.basename(ms), os.path.basename(outplotname))
                    )
                    f.write("refAnt = %s\n" % refant)
                    f.write("axisDiff=pol\n\n\n")

        if flagging:
            f.write("[flagamp]\n")
            f.write("soltab = [sol000/amplitude000]\n")
            f.write("operation = FLAG\n")
            if onechannel:
                f.write("axesToFlag = [time]\n")
            if onetime:
                f.write("axesToFlag = [freq]\n")
            if not onetime and not onechannel:
                f.write("axesToFlag = [time,freq]\n")
            f.write("mode = smooth\n")
            f.write("maxCycles = 3\n")
            f.write("windowNoise = 7\n")
            f.write("maxRms = %s\n" % str(maxrms))
            if onechannel:
                f.write("order  = [5]\n\n\n")
            if onetime:
                f.write("order  = [5]\n\n\n")
            if not onetime and not onechannel:
                f.write("order  = [5,5]\n\n\n")

            if includesphase and flagphases:
                f.write("[flagphase]\n")
                f.write("soltab = [sol000/phase000]\n")
                f.write("operation = FLAG\n")
                if onechannel:
                    f.write("axesToFlag = [time]\n")
                if onetime:
                    f.write("axesToFlag = [freq]\n")
                if not onetime and not onechannel:
                    f.write("axesToFlag = [time,freq]\n")
                f.write("mode = smooth\n")
                f.write("maxCycles = 3\n")
                f.write("windowNoise = 7\n")
                f.write("maxRms = %s\n" % str(maxrmsphase))
                if onechannel:
                    f.write("order  = [5]\n\n\n")
                if onetime:
                    f.write("order  = [5]\n\n\n")
                if not onetime and not onechannel:
                    f.write("order  = [5,5]\n\n\n")

            if not skip_solution_plotting:
                f.write("[plotampafter]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/amplitude000]\n")
                if onechannel:
                    f.write("axesInPlot = [time]\n")
                    if not onepol:
                        f.write("axisInCol = pol\n")
                if onetime:
                    f.write("axesInPlot = [freq]\n")
                    if not onepol:
                        f.write("axisInCol = pol\n")
                if onechannel or onetime:
                    f.write("markerSize=%s\n" % int(markersize))
                if not onetime and not onechannel:
                    f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                # f.write('minmax = [0,2.5]\n')
                f.write("minmax = [%s,%s]\n" % (str(medamp / 4.0), str(medamp * 2.5)))
                f.write(
                    "prefix = solution_plots_%s/%sampfl\n\n\n"
                    % (os.path.basename(ms), os.path.basename(outplotname))
                )

                if includesphase and flagphases:
                    f.write("[plotphase_after]\n")
                    f.write("operation = PLOT\n")
                    f.write("soltab = [sol000/phase000]\n")
                    if onechannel:
                        f.write("axesInPlot = [time]\n")
                        if not onepol:
                            f.write("axisInCol = pol\n")
                    if onetime:
                        f.write("axesInPlot = [freq]\n")
                        if not onepol:
                            f.write("axisInCol = pol\n")
                    if onechannel or onetime:
                        f.write("markerSize=%s\n" % int(markersize))
                    if not onetime and not onechannel:
                        f.write("axesInPlot = [time,freq]\n")
                    f.write("axisInTable = ant\n")
                    f.write("minmax = [-3.14,3.14]\n")
                    f.write(
                        "prefix = solution_plots_%s/%sphasefl\n"
                        % (os.path.basename(ms), os.path.basename(outplotname))
                    )
                    f.write("refAnt = %s\n" % refant)

    return parset


def create_losoto_flag_ap_only(
    ms: str,
    maxrms: float = 7.0,
    maxrmsphase: float = 7.0,
    includesphase: bool = True,
    onechannel: bool = False,
    flagphases: bool = True,
    onetime: bool = False,
):
    """
    Create a LoSoTo parset for amplitude and phase flagging.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the output context.
    maxrms : float, optional
        Maximum amplitude RMS.
    maxrmsphase : float, optional
        Maximum phase RMS.
    includesphase : bool, optional
        Whether phase solutions are present.
    onechannel : bool, optional
        Flag one channel along time.
    flagphases : bool, optional
        Enable phase flagging.
    onetime : bool, optional
        Flag one time along frequency.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_flag_ap_only.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")

        f.write("[flagamp]\n")
        f.write("soltab = [sol000/amplitude000]\n")
        f.write("operation = FLAG\n")
        if onechannel:
            f.write("axesToFlag = [time]\n")
        if onetime:
            f.write("axesToFlag = [freq]\n")
        if not onetime and not onechannel:
            f.write("axesToFlag = [time,freq]\n")
        f.write("mode = smooth\n")
        f.write("maxCycles = 3\n")
        f.write("windowNoise = 7\n")
        f.write("maxRms = %s\n" % str(maxrms))
        if onechannel:
            f.write("order  = [5]\n\n\n")
        if onetime:
            f.write("order  = [5]\n\n\n")
        if not onetime and not onechannel:
            f.write("order  = [5,5]\n\n\n")

        if includesphase and flagphases:
            f.write("[flagphase]\n")
            f.write("soltab = [sol000/phase000]\n")
            f.write("operation = FLAG\n")
            if onechannel:
                f.write("axesToFlag = [time]\n")
            if onetime:
                f.write("axesToFlag = [freq]\n")
            if not onetime and not onechannel:
                f.write("axesToFlag = [time,freq]\n")
            f.write("mode = smooth\n")
            f.write("maxCycles = 3\n")
            f.write("windowNoise = 7\n")
            f.write("maxRms = %s\n" % str(maxrmsphase))
            if onechannel:
                f.write("order  = [5]\n\n\n")
            if onetime:
                f.write("order  = [5]\n\n\n")
            if not onetime and not onechannel:
                f.write("order  = [5,5]\n\n\n")

    return parset


def create_losoto_bandpassparset(
    intype: str,
    ms: str,
    h5: str,
    skip_solution_plotting: bool = False,
):
    """
    Create losoto parset than takes median along the time axis
    Can be used to create a bandpass

    Parameters
    ----------
    intype : str
        set "phase" or "amplitude" or amplitude and phase ("a&p") smoothing, input should be one of these strings
    ms : str
        Measurement Set path used to name solution plots.
    h5 : str
        H5Parm containing the calibration solutions.
    skip_solution_plotting : bool, optional
        Skip plotting of the solutions.

    Returns
    -------
    str
        Path to the generated LoSoTo parset.
    """
    assert intype == "phase" or intype == "amplitude" or intype == "a&p"
    parset = "losoto_parsets/losoto_bandpass.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")

        if intype == "amplitude" or intype == "a&p":
            # compute median amplitude for the h5
            medamp = get_median_amp(h5)

            f.write("[bandpassamp]\n")
            f.write("operation = SMOOTH\n")
            f.write("soltab = [sol000/amplitude000]\n")
            f.write("axesToSmooth = [time] # axes to smooth\n")
            f.write("mode = median\n")
            f.write("replace = False\n")
            f.write("log = True\n")

            if not skip_solution_plotting:
                f.write("\n[plotamp]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/amplitude000]\n")
                f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [0,%s]\n" % str(medamp * 2.5))
                f.write(
                    "prefix = solution_plots_%s/bandpass_amps\n\n\n"
                    % os.path.basename(ms)
                )

        if intype == "phase" or intype == "a&p":
            f.write("[bandpassphase]\n")
            f.write("operation = SMOOTH\n")
            f.write("soltab = [sol000/phase000]\n")
            f.write("axesToSmooth = [time] # axes to smooth\n")
            f.write("mode = median\n")
            f.write("replace = False\n")
            f.write("log = False\n")

            if not skip_solution_plotting:
                f.write("\n[plotphase]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/phase000]\n")
                f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [-3.14,3.14]\n")
                f.write(
                    "prefix = solution_plots_%s/bandpass_phases\n\n\n"
                    % os.path.basename(ms)
                )

    return parset


def create_losoto_mediumsmoothparset(
    ms: str,
    boxsize: int,
    longbaseline: bool,
    includesphase: bool = True,
    refant: str = "CS003HBA0",
    onechannel: bool = False,
    outplotname: str = "runningmedian",
    skip_solution_plotting: bool = False,
):
    """
    Create a LoSoTo parset for medium-scale solution smoothing.

    Parameters
    ----------
    ms : str
        Measurement Set path used in the plot prefix.
    boxsize : int
        Smoothing window size.
    longbaseline : bool
        Whether long-baseline settings are used.
    includesphase : bool, optional
        Whether phase solutions are present.
    refant : str, optional
        Reference antenna name.
    onechannel : bool, optional
        Use one-channel plotting settings.
    outplotname : str, optional
        Base name for the output plot.
    skip_solution_plotting : bool, optional
        Skip plotting of the solutions.

    Returns
    -------
    str
        Generated parset path.
    """
    parset = "losoto_parsets/losoto_mediansmooth.parset"
    Path(parset).unlink(missing_ok=True)

    with open(parset, "w") as f:
        f.write("pol = []\n")
        f.write("soltab = [sol000/*]\n")
        f.write("Ncpu = 0\n\n\n")

        if includesphase:
            f.write("[smoothphase]\n")
            f.write("soltab = [sol000/phase000]\n")
            f.write("operation= SMOOTH\n")
            if onechannel:
                f.write("axesToSmooth = [time]\n")
                f.write("size = [%s]\n" % (boxsize))
            else:
                f.write("axesToSmooth = [freq,time]\n")
                f.write("size = [%s,%s]\n" % (boxsize, boxsize))
            f.write("mode = runningmedian\n\n\n")

        f.write("[smoothamp]\n")
        f.write("soltab = [sol000/amplitude000]\n")
        f.write("operation= SMOOTH\n")
        if onechannel:
            f.write("axesToSmooth = [time]\n")
            f.write("size = [%s]\n" % (boxsize))
        else:
            f.write("axesToSmooth = [freq,time]\n")
            f.write("size = [%s,%s]\n" % (boxsize, boxsize))
        f.write("mode = runningmedian\n\n\n")

        if not skip_solution_plotting:
            f.write("[plotamp_after]\n")
            f.write("operation = PLOT\n")
            f.write("soltab = [sol000/amplitude000]\n")
            if onechannel:
                f.write("axesInPlot = [time]\n")
            else:
                f.write("axesInPlot = [time,freq]\n")
            f.write("axisInTable = ant\n")
            if longbaseline:
                f.write("minmax = [0,2.5]\n")
            else:
                f.write("minmax = [0,2.5]\n")
            f.write("prefix = solution_plots_%s/amps_smoothed\n\n\n" % os.path.basename(ms))

            if includesphase:
                f.write("[plotphase_after]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/phase000]\n")
                if onechannel:
                    f.write("axesInPlot = [time]\n")
                else:
                    f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [-3.14,3.14]\n")
                f.write(
                    "prefix = solution_plots_%s/phases_smoothed\n\n\n"
                    % os.path.basename(ms)
                )
                f.write("refAnt = %s\n" % refant)

                f.write("[plotphase_after1rad]\n")
                f.write("operation = PLOT\n")
                f.write("soltab = [sol000/phase000]\n")
                if onechannel:
                    f.write("axesInPlot = [time]\n")
                else:
                    f.write("axesInPlot = [time,freq]\n")
                f.write("axisInTable = ant\n")
                f.write("minmax = [-1,1]\n")
                f.write(
                    "prefix = solution_plots_%s/phases_smoothed1rad\n"
                    % os.path.basename(ms)
                )
                f.write("refAnt = %s\n" % refant)

    return parset
