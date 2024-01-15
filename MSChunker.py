"""
Portable module to chunk MeasurementSets fractionally by time.

Code adapted from Rapthor: https://git.astron.nl/RD/rapthor
"""
__author__ = "Frits Sweijen"
__license__ = "GPLv3"

import argparse
import logging
import os
import subprocess

import casacore.tables as pt
import numpy as np
from astropy.time import Time

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(name)s ---- %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def normalize_ra(num: float) -> float:
    """
    Normalize RA to be in the range [0, 360).

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The RA in degrees to be normalized.

    Returns
    -------
    res : float
        RA in degrees in the range [0, 360).
    """
    lower = 0.0
    upper = 360.0
    res = num
    if num > upper or num == lower:
        num = lower + abs(num + upper) % (abs(lower) + abs(upper))
    if num < lower or num == upper:
        num = upper - abs(num - lower) % (abs(lower) + abs(upper))
    res = lower if num == upper else num

    return res


def normalize_dec(num: float) -> float:
    """
    Normalize Dec to be in the range [-90, 90].

    Based on https://github.com/phn/angles/blob/master/angles.py

    Parameters
    ----------
    num : float
        The Dec in degrees to be normalized.

    Returns
    -------
    res : float
        Dec in degrees in the range [-90, 90].
    """
    lower = -90.0
    upper = 90.0
    res = num
    total_length = abs(lower) + abs(upper)
    if num < -total_length:
        num += np.ceil(num / (-2 * total_length)) * 2 * total_length
    if num > total_length:
        num -= np.floor(num / (2 * total_length)) * 2 * total_length
    if num > upper:
        num = total_length - num
    if num < lower:
        num = -total_length - num
    res = num

    return res


def concat_time_command(msfiles: list, output_file: str) -> list:
    """
    Construct command to concatenate files in time using TAQL

    Parameters
    ----------
    msfiles : list of str
        List of MS filenames to be concatenated
    output_file : str
        Filename of output concatenated MS

    Returns
    -------
    cmd : list of str
        Command to be executed by subprocess.run()
    """
    cmd = [
        "taql",
        "select",
        "from",
        "[{}]".format(",".join(msfiles)),
        "giving",
        '"{}"'.format(output_file),
        "AS",
        "PLAIN",
    ]
    return cmd


def convert_mjd(mjd_sec: float) -> str:
    """
    Converts MJD to casacore MVTime

    Parameters
    ----------
    mjd_sec : float
        MJD time in seconds

    Returns
    -------
    mvtime : str
        Casacore MVTime string
    """
    t = Time(mjd_sec / 3600 / 24, format="mjd", scale="utc")
    date, hour = t.iso.split(" ")
    year, month, day = date.split("-")
    d = t.datetime
    month = d.ctime().split(" ")[1]

    return "{0}{1}{2}/{3}".format(day, month, year, hour)


class Observation(object):
    """
    The Observation object contains various MS-related parameters

    Parameters
    ----------
    ms_filename : str
        Filename of the MS file
    starttime : float, optional
        The start time of the observation (in MJD seconds). If None, the start time
        is the start of the MS file
    endtime : float, optional
        The end time of the observation (in MJD seconds). If None, the end time
        is the end of the MS file
    startfreq : float, optional
        The start freq of the observation (in Hz). If None, the start freq
        is the start of the MS file
    endfreq : float, optional
        The end freq of the observation (in Hz). If None, the end freq
        is the end of the MS file
    """

    def __init__(
        self,
        ms_filename: str,
        starttime: float = None,
        endtime: float = None,
        startfreq: float = None,
        endfreq: float = None,
    ):
        self.ms_filename = ms_filename
        self.name = os.path.basename(self.ms_filename.rstrip("/"))
        self.log = logging.getLogger("Observation:{}".format(self.name))
        self.log.setLevel(logging.INFO)
        self.starttime = starttime
        self.endtime = endtime
        self.startfreq = startfreq
        self.endfreq = endfreq
        self.parameters = {}
        self.scan_ms()

        # Define the infix for filenames
        if self.startsat_startofms and self.goesto_endofms:
            # Don't include starttime if observation covers full MS
            self.infix = ""
        else:
            # Include starttime to avoid naming conflicts
            self.infix = ".mjd{}".format(int(self.starttime))

    def scan_ms(self):
        """
        Scans input MS and stores info
        """
        # Get time info
        tab = pt.table(self.ms_filename, ack=False)
        if self.starttime is None:
            self.starttime = np.min(tab.getcol("TIME"))
        else:
            valid_times = np.where(tab.getcol("TIME") >= self.starttime)[0]
            if len(valid_times) == 0:
                raise ValueError(
                    "Start time of {0} is greater than the last time in the "
                    "MS".format(self.starttime)
                )
            self.starttime = tab.getcol("TIME")[valid_times[0]]

        # DPPP takes ceil(startTimeParset - startTimeMS), so ensure that our start time is
        # slightly less than the true one (if it's slightly larger, DPPP sets the first
        # time to the next time, skipping the first time slot)
        self.starttime -= 0.1
        if self.starttime > np.min(tab.getcol("TIME")):
            self.startsat_startofms = False
        else:
            self.startsat_startofms = True
        if self.endtime is None:
            self.endtime = np.max(tab.getcol("TIME"))
        else:
            valid_times = np.where(tab.getcol("TIME") <= self.endtime)[0]
            if len(valid_times) == 0:
                raise ValueError(
                    "End time of {0} is less than the first time in the "
                    "MS".format(self.endtime)
                )
            self.endtime = tab.getcol("TIME")[valid_times[-1]]
        if self.endtime < np.max(tab.getcol("TIME")):
            self.goesto_endofms = False
        else:
            self.goesto_endofms = True
        self.timepersample = tab.getcell("EXPOSURE", 0)
        self.numsamples = int(
            np.ceil((self.endtime - self.starttime) / self.timepersample)
        )
        tab.close()

        # Get frequency info
        sw = pt.table(self.ms_filename + "::SPECTRAL_WINDOW", ack=False)
        self.referencefreq = sw.col("REF_FREQUENCY")[0]
        channels = sw.col("CHAN_FREQ")[0]
        if self.startfreq is None:
            self.startfreq = np.min(channels)
        if (self.endfreq is None) or (self.endfreq > channels.max()):
            self.endfreq = np.max(channels)
        self.startchan = int(np.argwhere(channels == self.startfreq))
        self.endchan = int(np.argwhere(channels == self.endfreq))
        if self.endfreq >= channels.max():
            self.endchan += 1
        self.numchannels = sw.col("NUM_CHAN")[0]
        self.channelwidth = sw.col("CHAN_WIDTH")[0][0]
        sw.close()

        # Get pointing info
        obs = pt.table(self.ms_filename + "::FIELD", ack=False)
        self.ra = normalize_ra(np.degrees(float(obs.col("REFERENCE_DIR")[0][0][0])))
        self.dec = normalize_dec(np.degrees(float(obs.col("REFERENCE_DIR")[0][0][1])))
        obs.close()

        # Get station names and diameter
        ant = pt.table(self.ms_filename + "::ANTENNA", ack=False)
        self.stations = ant.col("NAME")[:]
        self.diam = float(ant.col("DISH_DIAMETER")[0])
        if "HBA" in self.stations[0]:
            self.antenna = "HBA"
        elif "LBA" in self.stations[0]:
            self.antenna = "LBA"
        else:
            self.log.warning(
                "Antenna type not recognized (only LBA and HBA data "
                "are supported at this time)"
            )
        ant.close()

        # Find mean elevation and FOV
        el_values = pt.taql(
            "SELECT mscal.azel1()[1] AS el from " + self.ms_filename + " limit ::10000"
        ).getcol("el")
        self.mean_el_rad = np.mean(el_values)


class MSChunker:
    """Handles chunking a MeasurementSet in time."""

    def __init__(self, msin: list, fraction: float = 1.0):
        """Handles chunking a MeasurementSet

        Parameters
        ----------
        msin : list
            List of MS filenames to be chunked.
        fraction : float
            Fraction of time to take.
        """
        self.time_fraction = fraction
        if type(msin) is str:
            mslist = [msin]
        else:
            mslist = msin
        self.full_observations = []
        self.mschunks = {}
        for ms in mslist:
            obs = Observation(ms)
            self.full_observations.append(obs)
            self.mschunks[obs.name] = {"chunks": [], "parsets": []}
        self.log = logging.getLogger("MSChunker")
        self.log.setLevel(logging.INFO)

    def chunk_observations_in_time(self, mintime: float, data_fraction: float = 1.0):
        """
        Break observations into smaller time chunks.

        Chunking is done if the specified data_fraction < 1.

        Parameters
        ----------
        data_fraction : float, optional
            Fraction of data to use during processing
        """
        if data_fraction < 1.0:
            self.log.info("Calculating time chunks")
            self.observations = []
            for obs in self.full_observations:
                tottime = obs.endtime - obs.starttime
                if data_fraction < min(1.0, mintime / tottime):
                    obs.log.warning(
                        "The specified value of data_fraction ({0:0.3f}) results in a "
                        "total time for this observation that is less than the "
                        "minimum timestep. The data fraction will be increased "
                        "to {1:0.3f} to ensure the minimum timestep requirement is "
                        "met.".format(data_fraction, min(1.0, mintime / tottime))
                    )
                nchunks = int(np.ceil(data_fraction / (mintime / tottime)))
                if nchunks == 1:
                    # Center the chunk around the midpoint (which is generally the most
                    # sensitive, near transit)
                    midpoint = obs.starttime + tottime / 2
                    chunktime = min(tottime, max(mintime, data_fraction * tottime))
                    if chunktime < tottime:
                        sub_obs = Observation(
                            obs.ms_filename,
                            starttime=midpoint - chunktime / 2,
                            endtime=midpoint + chunktime / 2,
                        )
                        self.observations.append(sub_obs)
                    else:
                        self.observations.append(obs)
                else:
                    obs.log.info("Splitting MS in {:d} chunks.".format(nchunks))
                    steptime = (
                        mintime * (tottime / mintime - nchunks) / nchunks
                        + mintime
                        + 0.1
                    )
                    starttimes = np.arange(obs.starttime, obs.endtime, steptime)
                    endtimes = np.arange(
                        obs.starttime + mintime, obs.endtime + mintime, steptime
                    )
                    for starttime, endtime in zip(starttimes, endtimes):
                        if endtime > obs.endtime:
                            starttime = obs.endtime - mintime
                            endtime = obs.endtime
                        sub_obs = Observation(
                            obs.ms_filename, starttime=starttime, endtime=endtime
                        )
                        if sub_obs.name.endswith(".ms"):
                            sub_obs.name = sub_obs.name.replace(
                                ".ms", "_mjd{:f}.ms".format(sub_obs.starttime)
                            )
                        if sub_obs.name.endswith(".MS"):
                            sub_obs.name = sub_obs.name.replace(
                                ".MS", "_mjd{:f}.MS".format(sub_obs.starttime)
                            )
                        else:
                            sub_obs.name = sub_obs.name + "_mjd{:f}".format(
                                sub_obs.starttime
                            )
                        self.observations.append(sub_obs)
                        self.mschunks[obs.name]["chunks"].append(sub_obs)
        else:
            self.observations = self.full_observations[:]

    def chunk_observations_in_freq(self, nchan: int = 0):
        """
        Break observations into smaller frequency chunks.

        Chunking is done if the specified number of channels > 0.

        Parameters
        ----------
        nchan : int, optional
            Fraction of data to use during processing
        """
        if nchan > 0:
            self.log.info("Calculating frequency chunks")
            self.observations = []
            for obs in self.full_observations:
                if nchan > obs.numchannels:
                    obs.log.warning(
                        "The specified number of channels exceeds the number of channels in the MeasurementSet."
                    )
                nchunks = int(obs.numchannels / nchan) if nchan > 0 else 1
                obs.log.info("Splitting MS in {:d} chunks.".format(nchunks))
                if nchunks == 1:
                    self.observations.append(obs)
                else:
                    startfreqs = np.arange(
                        obs.startfreq, obs.endfreq, obs.channelwidth * nchan
                    )
                    endfreqs = np.arange(
                        obs.startfreq + obs.channelwidth * nchan,
                        obs.endfreq + obs.channelwidth * nchan,
                        obs.channelwidth * nchan,
                    )
                    for startfreq, endfreq in zip(startfreqs, endfreqs):
                        sub_obs = Observation(
                            obs.ms_filename, startfreq=startfreq, endfreq=endfreq
                        )
                        if sub_obs.name.endswith(".ms"):
                            sub_obs.name = sub_obs.name.replace(
                                ".ms",
                                "_{:f}MHz.ms".format(
                                    (sub_obs.startfreq + sub_obs.endfreq) / 2e6
                                ),
                            )
                        elif sub_obs.name.endswith(".MS"):
                            sub_obs.name = sub_obs.name.replace(
                                ".MS",
                                "_{:f}MHz.MS".format(
                                    (sub_obs.startfreq + sub_obs.endfreq) / 2e6
                                ),
                            )
                        else:
                            sub_obs.name = sub_obs.name + "_{:f}MHz".format(
                                (sub_obs.startfreq + sub_obs.endfreq) / 2e6
                            )
                        self.observations.append(sub_obs)
                        self.mschunks[obs.name]["chunks"].append(sub_obs)
        else:
            self.observations = self.full_observations[:]

    def make_parsets_time(self):
        """
        Generate parsets to create the time chunks.
        """
        PARSET = """numthreads=12

msin = {name}
msin.starttime = {stime}
msin.endtime = {etime}

msout = {name_out}
msout.storagemanager = dysco

steps=[]
"""
        self.log.info("Writing chunk parsets")
        for fobs in self.full_observations:
            for i, obs in enumerate(self.mschunks[fobs.name]["chunks"]):
                pname = "split_" + fobs.name + "_chunk{:02d}.parset".format(i)
                with open(pname, "w") as f:
                    parset = PARSET.format(
                        name=fobs.name,
                        stime=convert_mjd(obs.starttime),
                        etime=convert_mjd(obs.endtime),
                        name_out=obs.name,
                    )
                    # self.mschunks[obs.name]['parsets'][obs.name + "_{:f}".format(obs.starttime)] = pname
                    self.mschunks[fobs.name]["parsets"].append(pname)
                    f.write(parset)

    def make_parsets_freq(self):
        """
        Generate parsets to create the time chunks.
        """
        PARSET = """numthreads=12

msin = {name}
msin.startchan = {sfreq}
msin.nchan = {nchan}

msout = {name_out}
msout.storagemanager = dysco

steps=[]
"""
        self.log.info("Writing chunk parsets")
        for fobs in self.full_observations:
            for i, obs in enumerate(self.mschunks[fobs.name]["chunks"]):
                pname = "split_" + fobs.name + "_chunk{:02d}.parset".format(i)
                with open(pname, "w") as f:
                    parset = PARSET.format(
                        name=fobs.name,
                        sfreq=obs.startchan,
                        nchan=(obs.endchan - obs.startchan),
                        name_out=obs.name,
                    )
                    # self.mschunks[obs.name]['parsets'][obs.name + "_{:f}".format(obs.starttime)] = pname
                    self.mschunks[fobs.name]["parsets"].append(pname)
                    f.write(parset)

    def run_parsets(self):
        """
        Run all the parsets, generating chunks.
        """
        self.log.info("Writing parsets")
        for mschunk in self.mschunks.keys():
            print(mschunk)
            for parset in self.mschunks[mschunk]["parsets"]:
                self.log.info("Running {:s}".format(parset))
                subprocess.run(["DP3", parset])

    def concat_chunks(self) -> list:
        """
        Concatenate all generated chunks.

        Returns
        -------
        msout : list
            Name of the concatenated MeasurementSet as <input name>.<int(data_fraction*100)>pc.ms'
        """
        self.log.info("Concatenating chunks")
        msouts = []
        for mschunk in self.mschunks.keys():
            chunknames = [obs.name for obs in self.mschunks[mschunk]["chunks"]]
            msout = mschunk + ".{:d}pc.ms".format(int(self.time_fraction * 100))
            cmd = concat_time_command(chunknames, msout)
            subprocess.run(cmd)
            self.log.info("Concatenated MS written to {:s}".format(msout))
            msouts.append(msout)
        return msouts


def chunk_and_concat(
    mslist, mode: str, fraction: float = 1.0, mintime: int = 1, nchan: int = 0
) -> list:
    """
    Main entry point. Splits a MeasurementSet in time chunks and concatenate all generated chunks.
    Parameters
    ----------
    mslist : list
        List of MS filenames to be chunked.
    fraction : float
        Fraction of time to take.
    mintime : int
        Minimum time span per chunk in seconds.
    mode : str
        Mode of chunking: time or frequency.

    Returns
    -------
    msout : list
        List of MS chunks.
    """
    if mode == "time":
        data = MSChunker(mslist, fraction)
        data.chunk_observations_in_time(
            data_fraction=data.time_fraction, mintime=mintime
        )
        data.make_parsets_time()
        epochs = []
        for epoch in data.mschunks:
            epochs.append([obs.name for obs in data.mschunks[epoch]["chunks"]])
        data.run_parsets()
        data.concat_chunks()
        return epochs
    elif mode == "frequency":
        data = MSChunker(mslist)
        data.chunk_observations_in_freq(nchan)
        data.make_parsets_freq()
        epochs = []
        for epoch in data.mschunks:
            epochs.append([obs.name for obs in data.mschunks[epoch]["chunks"]])
        data.run_parsets()
        return epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk a MeasurementSet in time or frequency."
    )
    parser.add_argument("ms", nargs="+", help="Input MeasurementSet(s).")
    parser.add_argument(
        "--timefraction",
        type=float,
        default=1.0,
        help="Fraction of data to split off. Default: 1.0",
    )
    parser.add_argument(
        "--mintime",
        required=False,
        type=int,
        default=-1,
        help="Minimum time in seconds. Default: -1 (all)",
    )
    parser.add_argument(
        "--chan_per_chunk",
        required=False,
        type=int,
        default=0,
        help="Number of channels per output frequency chunk. Default: 0 (all)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["time", "frequency"],
        help="Chunk in time or frequency.",
    )

    options = parser.parse_args()
    if (options.timefraction < 1) and (options.chan_per_chunk > 0):
        print("Splitting time and frequency simultaneously is not supported.")
    elif options.timefraction < 1:
        chunk_and_concat(
            options.ms,
            fraction=options.timefraction,
            mintime=options.mintime,
            mode=options.mode,
        )
    elif options.chan_per_chunk > 0:
        chunk_and_concat(options.ms, mode=options.mode, nchan=options.chan_per_chunk)
