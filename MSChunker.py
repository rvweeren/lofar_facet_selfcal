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

from astropy.time import Time

import casacore.tables as pt
import numpy as np

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(name)s ---- %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def normalize_ra(num):
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


def normalize_dec(num):
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
        num += ceil(num / (-2 * total_length)) * 2 * total_length
    if num > total_length:
        num -= floor(num / (2 * total_length)) * 2 * total_length
    if num > upper:
        num = total_length - num
    if num < lower:
        num = -total_length - num
    res = num

    return res


def concat_time_command(msfiles, output_file):
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


def convert_mjd(mjd_sec):
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
    """

    def __init__(self, ms_filename, starttime=None, endtime=None):
        self.ms_filename = ms_filename
        self.name = os.path.basename(self.ms_filename.rstrip("/"))
        self.log = logging.getLogger("Observation:{}".format(self.name))
        self.log.setLevel(logging.INFO)
        self.starttime = starttime
        self.endtime = endtime
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
        self.startfreq = np.min(sw.col("CHAN_FREQ")[0])
        self.endfreq = np.max(sw.col("CHAN_FREQ")[0])
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

    def __init__(self, msin, fraction):
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
            mslist = [mslist]
        else:
            mslist = msin
        self.full_observations = []
        for ms in mslist:
            self.full_observations.append(Observation(ms))
        self.chunks = {}
        self.log = logging.getLogger("MSChunker")
        self.log.setLevel(logging.INFO)

    def chunk_observations(self, mintime, data_fraction=1.0):
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
                        "slow-gain timestep. The data fraction will be increased "
                        "to {1:0.3f} to ensure the slow-gain timestep requirement is "
                        "met.".format(data_fraction, min(1.0, mintime / tottime))
                    )
                nchunks = int(np.ceil(data_fraction / (mintime / tottime)))
                if nchunks == 1:
                    # Center the chunk around the midpoint (which is generally the most
                    # sensitive, near transit)
                    midpoint = obs.starttime + tottime / 2
                    chunktime = min(tottime, max(mintime, data_fraction * tottime))
                    if chunktime < tottime:
                        self.observations.append(
                            Observation(
                                obs.ms_filename,
                                starttime=midpoint - chunktime / 2,
                                endtime=midpoint + chunktime / 2,
                            )
                        )
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
                        self.observations.append(
                            Observation(
                                obs.ms_filename, starttime=starttime, endtime=endtime
                            )
                        )
        else:
            self.observations = self.full_observations[:]

    def make_parsets(self):
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
        for i, obs in enumerate(self.observations):
            pname = "split_" + obs.name + "_chunk{:02d}.parset".format(i)
            with open(pname, "w") as f:
                parset = PARSET.format(
                    name=obs.name,
                    stime=convert_mjd(obs.starttime),
                    etime=convert_mjd(obs.endtime),
                    name_out=obs.name + "_{:f}".format(obs.starttime),
                )
                self.chunks[obs.name + "_{:f}".format(obs.starttime)] = pname
                f.write(parset)

    def run_parsets(self):
        """
        Run all the parsets, generating chunks.
        """
        for chunk, parset in self.chunks.items():
            self.log.info("Running {:s}".format(parset))
            subprocess.run(["DP3", parset])

    def concat_chunks(self):
        """
        Concatenate all generated chunks.

        Returns
        -------
        msout : str
            Name of the concatenated MeasurementSet as <input name>.<int(data_fraction*100)>pc.ms'
        """
        self.log.info("Concatenating chunks")
        msout = self.observations[0].name + ".{:d}pc.ms".format(
            int(self.time_fraction * 100)
        )
        cmd = concat_time_command(list(self.chunks.keys()), msout)
        subprocess.run(cmd)
        return msout


def chunk_and_concat(mslist, fraction, mintime):
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

    Returns
    -------
    msout : str
        Name of the concatenated MeasurementSet as <input name>.<int(data_fraction*100)>pc.ms'
    """
    data = MSChunker(mslist, fraction)
    data.chunk_observations(data_fraction=data.time_fraction, mintime=mintime)
    data.make_parsets()
    data.run_parsets()
    msout = data.concat_chunks()
    return msout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a MeasurementSet in time.")
    parser.add_argument("ms", nargs="+", help="Input MeasurementSet(s).")
    parser.add_argument(
        "--timefraction",
        type=float,
        default=0.2,
        help="Fraction of data to split off. Default: 0.2",
    )
    parser.add_argument(
        "--mintime", required=True, type=int, help="Minimum time in seconds."
    )

    options = parser.parse_args()
    chunk_and_concat(options.ms, options.timefraction, options.mintime)
