"""Functions that flag Measurement Set data (DP3 preflagger, taql, aoflagger, CASA)
and report flagging statistics."""
import os
import shutil
from pathlib import Path

import numpy as np
from casacore.tables import makecoldesc, table, taql  # type: ignore
from losoto import h5parm  # type: ignore

from submods.ms_fixes import fix_uvw
from submods.runtime_utils import get_logger, run, terminal_print

# facetselfcal package directory (holds flagging_strategies/), same as datapath in main.py
datapath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal')


def aoflagger_column(mslist, aoflagger_strategy=None, column='CORRECTED_DATA'):
    """
    Runs AOFlagger on a list of Measurement Sets (MS) using DP3, with optional custom flagging strategy and data column.
    This function checks if a custom AOFlagger strategy file is provided and exists. If the strategy file requests
    flagging of circular polarisation data (RR, RL, LR, LL) and the MS contains such data, it temporarily replaces
    these polarisations with linear equivalents (XX, XY, YX, YY) in a copy of the strategy file, as AOFlagger cannot
    flag circular polarisations in DP3. The function then runs DP3 with AOFlagger on each MS in the list.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Set paths to process.
    aoflagger_strategy : str, optional
        Path or name of the AOFlagger strategy file to use. If not provided,
        the default strategy is used. If a name is given and not a file path, it is assumed to be in the
        'flagging_strategies' directory under 'datapath'.
    column : str, optional
        Name of the data column in the MS to flag. Defaults to 'CORRECTED_DATA'.

    Warnings
    --------
    - If the strategy file requests RR/RL/LR/LL flagging and the MS contains circular polarisation data,
    a warning is printed and the strategy file is temporarily modified to use XX/XY/YX/YY instead.

    Notes
    -----
    Side Effects:
    - May create and remove temporary strategy files in the current working directory.
    - Executes system commands for file manipulation and running DP3.
    Prints:
    - Warnings about unsupported polarisation flagging.
    - The command being executed for AOFlagger.
    """

    if aoflagger_strategy is not None and not os.path.isfile(aoflagger_strategy):
        aoflagger_strategy = f'{datapath}/flagging_strategies/' + aoflagger_strategy

    # check if MS has circular polarisation data and RR/RL/LR/LL flagging is requested
    if aoflagger_strategy is not None:
        with table(mslist[0] + '::POLARIZATION', ack=False) as t:
            corr_type = t.getcol('CORR_TYPE')
            with open(aoflagger_strategy) as myfile:
                if ("LL" in myfile.read() or "RR" in myfile.read() or \
                    "LR" in myfile.read() or "RL" in myfile.read()) and np.array_equal(np.array([[5, 6, 7, 8]]), corr_type):
                    terminal_print("\033[33m" + "WARNING: AOFlagger cannot flag RR/RL/LR/LL data in DP3" + "\033[0m")
                    terminal_print("\033[33m" + "WARNING: Will temporarily replace RR/RL/LR/LL with XX/XY/YX/YY" + "\033[0m")
                    if os.path.isfile('tmp.' + os.path.basename(aoflagger_strategy)):
                        Path('tmp.' + os.path.basename(aoflagger_strategy)).unlink(missing_ok=True)
                    shutil.copy(aoflagger_strategy, 'tmp.' + os.path.basename(aoflagger_strategy))
                    # using linux sed command to replace the strings in the file

                    target_file = Path(f"tmp.{Path(aoflagger_strategy).name}")
                    if target_file.is_file():
                        text = target_file.read_text()
                        # Chain the replacements exactly like the sed commands did
                        text = (text.replace("RR", "XX")
                                    .replace("RL", "XY")
                                    .replace("LR", "YX")
                                    .replace("LL", "YY"))
                        target_file.write_text(text)
                    aoflagger_strategy = 'tmp.' + os.path.basename(aoflagger_strategy)
        
    # run DP3 with AOFlagger
    for ms in mslist:
        cmd = 'DP3 msin=' + ms + ' msin.datacolumn=' + column + ' msout=. '
        cmd += 'ao.type=aoflag '
        cmd += 'ao.keepstatistics=False '
        cmd += 'ao.memoryperc=50 '
        cmd += 'ao.overlapperc=10 '
        if aoflagger_strategy is not None:
            cmd += 'ao.strategy=' +  aoflagger_strategy + ' '         
        cmd += 'steps=[ao] '
        terminal_print('Running AOFlagger on ' + ms + ' with strategy: ' + aoflagger_strategy)
        terminal_print('Command:', cmd)
        run(cmd, log=True)
        # remove temporary strategy file if it was created
    if aoflagger_strategy is not None and os.path.isfile('tmp.' + os.path.basename(aoflagger_strategy)):
        Path('tmp.' + os.path.basename(aoflagger_strategy)).unlink(missing_ok=True)


def flag_shadowed_antenna(mslist): 
    """
    Flag shadowed antennas in a CASA Measurement Set using casapy flagdata.
    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets to process.
        This function constructs and runs a command to execute the casapy flagdata task via a helper script.
    """
    # for standalone running
    datapathc = os.path.dirname(os.path.abspath(__file__))
    submodpathc = '/'.join(datapathc.split('/')[0:-1])+'/submods'
    
    # check if mslist is of type list, if not convert it to a list
    if not isinstance(mslist, list):
        mslist = [mslist]

    for ms in mslist:
        cmdcasa = f'python {submodpathc}/casa_flagshadowed.py '
        cmdcasa += '--ms=' + ms
        run(cmdcasa)


def flag_antenna_taql(ms, antennaname):
    """
    Flags all data in a Measurement Set (MS) corresponding to a specific antenna, identified by its name, using TaQL (Table Query Language).

    Parameters
    ----------
    ms : str
        The path to the Measurement Set (MS) to be modified. This should include the full directory name of the MS.
    antennaname : str or list
        The name of the antenna(s) to flag. This name should match exactly with the entry in the ANTENNA table of the MS.

    Returns
    -------
    None

    Notes
    -----
    Functionality:
    - Constructs a TaQL query to update the `FLAG` column in the MS's `MAIN` table.
    - Flags all rows where `ANTENNA1` or `ANTENNA2` corresponds to the antenna with the specified name.
    - Executes the constructed TaQL query using the `run` function (assumes `run` is defined elsewhere in your codebase to execute shell commands).
    - This function modifies the MS in-place; ensure you have a backup if needed.
    - The `run` function must be defined and capable of executing the constructed TaQL command in the appropriate environment.
    - Requires that the specified `antennaname` exists in the MS's ANTENNA table; otherwise, no rows will be flagged.

    Examples
    --------
    Suppose you have a Measurement Set `observation.ms` and want to flag an antenna named `DE601HBA`:

    ```python
    from facetselfcal import *
    flag_antenna_taql("observation.ms", "DE601HBA")
    ```

    This will flag all rows in `observation.ms` where either `ANTENNA1` or `ANTENNA2` corresponds to the antenna named `DE601`.

    """
    # check is input is string or list, if string convert to list
    if isinstance(antennaname, str):
        antennaname = [antennaname]
    for ant in antennaname:
        cmd = "taql 'UPDATE " + ms + ' '
        cmd += "SET FLAG=true WHERE ANTENNA1 IN (SELECT ROWID() FROM " + ms
        cmd += "::ANTENNA WHERE NAME=\"" + ant + "\") OR ANTENNA2 IN "
        cmd += "(SELECT ROWID() FROM " + ms + "::ANTENNA WHERE NAME=\"" + ant + "\")' "
        terminal_print('Command:', cmd)
    run(cmd, taql=True)
    return


def return_antennas_highflaggingpercentage(ms, percentage=85):
    """
    Identifies antennas with a high percentage of flagged data in a Measurement Set (MS).

    This function queries the provided Measurement Set (MS) to find antennas where the
    percentage of flagged data exceeds the specified threshold. It uses the TaQL (Table Query
    Language) to perform the query and returns a list of antenna names that meet the criteria.

    Parameters
    ----------
    ms : str
        The path to the Measurement Set (MS) to be analyzed.
    percentage : float, optional
        The flagging percentage threshold. Antennas with a
        flagging percentage greater than this value will be returned. Default is 0.85
        (85%).

    Returns
    -------
    list
        A list of antenna names (str) that have a flagging percentage above the specified
        threshold.

    Examples
    --------
    >>> flagged_antennas = return_antennas_highflaggingpercentage("path/to/ms", 0.9)
    Finding stations with a flagging percentage above 90.0 ....
    Found: ['ANT1', 'ANT2']

    """
    terminal_print('Finding stations with a flagging percentage above ' + str(percentage) + ' ....')
    t = taql(""" SELECT antname, gsum(numflagged) AS numflagged, gsum(numvis) AS numvis,
       gsum(numflagged)/gsum(numvis) as percflagged FROM [[ SELECT mscal.ant1name() AS antname, ntrue(FLAG) AS numflagged, count(FLAG) AS numvis FROM """ + ms + """ ],[ SELECT mscal.ant2name() AS antname, ntrue(FLAG) AS numflagged, count(FLAG) AS numvis FROM """ + ms + """ ] ] GROUP BY antname HAVING percflagged > """ + str(
        percentage/100.) + """ """)

    flaggedants = [row["antname"] for row in t]
    terminal_print('Found:', flaggedants)
    t.close()
    return flaggedants


def clip_DATA(mslist, clipvalue):
    """
    Flags values in DATA above a certain threshold and sets corresponding WEIGHT_SPECTRUM values to 0.0.

    Parameters
    ----------
    mslist : list
        a list of Measurement Sets to iterate over and fix outlier values of.
    clipvalue : float
        value for DATA above which WEIGHT_SPECTRUM will be set to 0.

    Returns
    -------
    None
    """

    # assert that clipvalue is a float and is larger than 0
    if clipvalue <= 0.0:
        raise ValueError('clipvalue must be larger than 0.0')

    if isinstance(mslist, str):
        mslist = [mslist]

    for ms in mslist:
        terminal_print('Clipping DATA manually', ms, clipvalue)
        cmdtaql = f'taql "UPDATE {ms} SET WEIGHT_SPECTRUM[abs(DATA) > {clipvalue}] = 0.0"'
        terminal_print('Running command:', cmdtaql)
        run(cmdtaql, log=True, taql=True)
    return


def create_backup_flag_col(ms, flagcolname='FLAG_BACKUP'):
    """
    Creates a backup of the FLAG column.

    Parameters
    ----------
    ms : str
        path to the Measurement Set.
    flagcolname : str
        name of the output column.

    Returns
    -------
    None
    """
    cname = 'FLAG'
    flags = []
    with table(ms, readonly=False, ack=True) as t:
        if flagcolname not in t.colnames():
            flags = t.getcol('FLAG')
            terminal_print('Adding flagging column', flagcolname, 'to', ms)
            desc = t.getcoldesc(cname)
            newdesc = makecoldesc(flagcolname, desc)
            newdmi = t.getdminfo(cname)
            newdmi['NAME'] = flagcolname
            t.addcols(newdesc, newdmi)
            t.putcol(flagcolname, flags)
    del flags
    return


def uvmaxflag(msin, uvmax):
    """
    Flags visibilities in a Measurement Set (MS) with UV distances greater than a specified maximum.

    Parameters
    ----------
    msin : str
        Path to the input Measurement Set.
    uvmax : float
        Maximum allowed UV distance (in wavelengths). Visibilities with UV distances greater than this value will be flagged.

    Returns
    -------
    None

    Notes
    -----
    Side Effects:
    Executes a DP3 command to flag data in the input MS based on the specified UV distance threshold.
    """
    cmd = 'DP3 msin=' + msin + ' msout=. steps=[f] f.type=uvwflag f.uvlambdamax=' + str(uvmax)
    terminal_print('Command:', cmd)
    run(cmd)
    return


def runaoflagger(mslist, strategy=None):
    """
    Run aoglagger on a Measurement Set.

    Parameters
    ----------
    mslist : list
        list of Measurement Sets to iterate over.
    strategy : str or None, optional
        AOFlagger strategy name or path; use the default strategy when None.

    Returns
    -------
    None
    """
    for ms in mslist:
        if strategy is not None:
            if os.path.isfile(strategy): # try full location first
                cmd = 'aoflagger -strategy ' + strategy + ' ' + ms
            else: # try strategy in flagging_strategies
                cmd = 'aoflagger -strategy ' + f'{datapath}/flagging_strategies/' + strategy + ' ' + ms
        else:
            cmd = 'aoflagger ' + ms
        terminal_print('Command:', cmd)
        run(cmd)
    return


def flagms_startend(ms, tecsolsfile, tecsolint):
    """
    Parameters
    ----------
    ms : str
        Path to the Measurement Set to trim.
    tecsolsfile : str
        H5Parm file containing TEC solutions used to find valid start and end times.
    tecsolint : int or str
        TEC solution interval used to convert solution slots to Measurement Set rows.
    """

    taql = 'taql'

    msout = ms + '.cut'

    H5 = h5parm.h5parm(tecsolsfile)
    tec = H5.getSolset('sol000').getSoltab('tec000').getValues()
    tecvals = tec[0]

    axis_names = H5.getSolset('sol000').getSoltab('tec000').getAxesNames()
    time_ind = axis_names.index('time')
    ant_ind = axis_names.index('ant')

    # ['time', 'ant', 'dir', 'freq']
    reftec = tecvals[:, 0, 0, 0]

    # print np.shape( tecvals[:,:,0,0]), np.shape( reftec[:,None]),
    tecvals = tecvals[:, :, 0, 0] - reftec[:, None]  # reference to zero

    times = tec[1]['time']

    # print tecvals[:,0]

    goodtimesvec = []

    for timeid, time in enumerate(times):
        tecvals[timeid, :]

        # print timeid, np.count_nonzero( tecvals[timeid,:])
        goodtimesvec.append(np.count_nonzero(tecvals[timeid, :]))

    goodstartid = np.argmax(np.array(goodtimesvec) > 0)
    goodendid = len(goodtimesvec) - np.argmax(np.array(goodtimesvec[::-1]) > 0)

    terminal_print('First good solutionslot,', goodstartid, ' out of', len(goodtimesvec))
    terminal_print('Last good solutionslot,', goodendid, ' out of', len(goodtimesvec))
    H5.close()

    if (goodstartid != 0) or (goodendid != len(goodtimesvec)):  # only do if needed to save some time
        cmd = taql + " ' select from " + ms + " where TIME in (select distinct TIME from " + ms
        cmd += " offset " + str(goodstartid * int(tecsolint))
        cmd += " limit " + str((goodendid - goodstartid) * int(tecsolint)) + ") giving "
        cmd += msout + " as plain'"

        terminal_print('Command:', cmd)
        run(cmd, taql=True)
        fix_uvw([msout])
        shutil.rmtree(ms, ignore_errors=True)
        time.sleep(2)  # give some time to remove the MS
        Path(msout).rename(ms)
    return


def flag_autocorr(mslist):
    """
    Flag autocorrelations in MS.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    """
    for ms in mslist:
       cmd = 'DP3 msin=' + ms + ' msout=. steps=[pr] '
       cmd += 'pr.type=preflagger pr.corrtype=auto'
       get_logger().info(cmd)
       run(cmd)
    return


def flag_antenna_timerange_ms(ms, timerange, antenna=''):
    """
    Flag time range of of a specific antennas in an MS with DP3 preflagger
    Parameters
    ----------
    ms : str
        Measurement Set to flag
    timerange : str
        Relative timerange to flag in format 'starttime..endtime'
    antenna : str
        Antenna name to flag. Default is an empty string, which means all antennas.
    Returns
    -------
    None
        ------
    Notes
    -----

    Examples
    --------
    flag_antenna_timerange_ms('mydata.ms', '01:23:45..01:45:00', 'CS001HBA0')
    See https://dp3.readthedocs.io/en/latest/steps/PreFlagger.html
    Ranges of times (using .. or +-) since the start of the observation. A time can be given like 1:30:0 or 20s.

    """
   
    cmd = 'DP3 msin=' + ms + ' msout=. steps=[pr] '
    cmd += 'pr.type=preflagger pr.reltime="[' + timerange + ']" ' 
    if antenna != '':
        cmd += 'pr.baseline=' + antenna + '"&&*"'
    terminal_print('Command:', cmd)
    run(cmd)
    return


def flag_uGMRT_badfreqs(mslist, args):
    """
    Flag known bad frequency ranges for uGMRT bands.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    args : dict
        Parsed facetselfcal options; ``args['telescope']`` is used.
    """
    
    if args['telescope'] != 'GMRT':
        return

    for ms in mslist:
        t = table(ms + '::SPECTRAL_WINDOW', ack=False)
        freq = np.median(t.getcol('CHAN_FREQ')[0]) / 1e6  # in MHz
        t.close()
        if freq > 100 and freq < 250:  # band2
            freqranges = '[99MHz..128MHz,167MHz..188MHz,243MHz..301MHz]'
        elif freq >= 250 and freq < 500:  # band3
            #freqranges = '[360MHz..380MHz,390MHz..391MHz]'
            freqranges = '[347MHz..349.7MHz,355MHz..380MHz,390MHz..391MHz]'
        elif freq >= 500 and freq < 800:  # band4    
            #freqranges = '[580MHz..620MHz,730MHz..750MHz]'
            freqranges = None
        elif freq >= 800 and freq < 1450:  # band5
            freqranges = None
            #freqranges = '[950MHz..1050MHz,1200MHz..1280MHz]'
        else:
            freqranges = None
        if freqranges is not None:
            cmd = 'DP3 msin=' + ms + ' msout=. steps=[pr] '
            cmd += 'pr.type=preflagger pr.freqrange=' + freqranges
            terminal_print('Flagging uGMRT known bad frequency ranges: ', freqranges)
            terminal_print('Command:', cmd)
            run(cmd)
    return


def compute_flagging_statistics(mslist, chunk_size=10000):
    """Log flagged-sample percentages and fully flagged antennas per MS.

    FLAG data is read in bounded row chunks. FLAG_ROW counts every sample in
    that row as flagged.
    """
    if isinstance(mslist, (str, os.PathLike)):
        mslist = [mslist]
    chunk_size = max(1, int(chunk_size))

    for ms in mslist:
        try:
            with table(ms, readonly=True, ack=False) as ms_table:
                columns = set(ms_table.colnames())
                required_columns = {"FLAG", "ANTENNA1", "ANTENNA2"}
                missing_columns = sorted(required_columns - columns)
                if missing_columns:
                    raise KeyError(
                        "missing required columns: {}".format(", ".join(missing_columns))
                    )

                total_rows = ms_table.nrows()
                if total_rows == 0:
                    get_logger().info(
                        "No visibility rows in MS %s; flagging statistics unavailable.",
                        ms,
                    )
                    continue

                with table(
                    ms + "::ANTENNA", readonly=True, ack=False
                ) as antenna_table:
                    antenna_names = [
                        name.decode("utf-8", errors="replace").strip()
                        if isinstance(name, bytes)
                        else str(name).strip()
                        for name in antenna_table.getcol("NAME")
                    ]

                has_flag_row = "FLAG_ROW" in columns
                total_sample_count = 0
                flagged_sample_count = 0
                antenna_sample_counts = np.zeros(len(antenna_names), dtype=np.int64)
                antenna_flagged_counts = np.zeros(len(antenna_names), dtype=np.int64)

                for start_row in range(0, total_rows, chunk_size):
                    rows_in_chunk = min(chunk_size, total_rows - start_row)
                    flags = np.asarray(
                        ms_table.getcol(
                            "FLAG", startrow=start_row, nrow=rows_in_chunk
                        ),
                        dtype=bool,
                    )
                    if flags.ndim < 2 or flags.shape[0] != rows_in_chunk:
                        raise ValueError(
                            "FLAG column returned an unexpected shape {}".format(
                                flags.shape
                            )
                        )

                    samples_per_row = int(np.prod(flags.shape[1:]))
                    if samples_per_row == 0:
                        continue
                    flagged_per_row = np.count_nonzero(
                        flags, axis=tuple(range(1, flags.ndim))
                    ).astype(np.int64, copy=False)

                    if has_flag_row:
                        flag_rows = np.asarray(
                            ms_table.getcol(
                                "FLAG_ROW", startrow=start_row, nrow=rows_in_chunk
                            ),
                            dtype=bool,
                        ).reshape(-1)
                        if flag_rows.size != rows_in_chunk:
                            raise ValueError("FLAG_ROW column returned an unexpected shape")
                        flagged_per_row[flag_rows] = samples_per_row

                    total_sample_count += int(flags.size)
                    flagged_sample_count += int(flagged_per_row.sum())

                    for antenna_column in ("ANTENNA1", "ANTENNA2"):
                        antenna_ids = np.asarray(
                            ms_table.getcol(
                                antenna_column,
                                startrow=start_row,
                                nrow=rows_in_chunk,
                            ),
                            dtype=np.int64,
                        ).reshape(-1)
                        if antenna_ids.size != rows_in_chunk:
                            raise ValueError(
                                "{} column returned an unexpected shape".format(
                                    antenna_column
                                )
                            )
                        valid_ids = (antenna_ids >= 0) & (
                            antenna_ids < len(antenna_names)
                        )
                        np.add.at(
                            antenna_sample_counts,
                            antenna_ids[valid_ids],
                            samples_per_row,
                        )
                        np.add.at(
                            antenna_flagged_counts,
                            antenna_ids[valid_ids],
                            flagged_per_row[valid_ids],
                        )

                if total_sample_count == 0:
                    get_logger().info(
                        "No FLAG samples in MS %s; flagging statistics unavailable.",
                        ms,
                    )
                    continue

                flagged_percentage = 100.0 * flagged_sample_count / total_sample_count
                fully_flagged_antennas = [
                    antenna_names[index]
                    for index in range(len(antenna_names))
                    if antenna_sample_counts[index] > 0
                    and antenna_flagged_counts[index] == antenna_sample_counts[index]
                ]
                get_logger().info(
                    "Flagging statistics for MS %s: %.2f%% flagged (%d/%d samples).",
                    ms,
                    flagged_percentage,
                    flagged_sample_count,
                    total_sample_count,
                )
                get_logger().info(
                    "Fully flagged antennas for MS %s: %s",
                    ms,
                    ", ".join(fully_flagged_antennas) or "None",
                )
        except Exception as exc:
            get_logger().warning(
                "Could not compute flagging statistics for MS %s: %s", ms, exc
            )
