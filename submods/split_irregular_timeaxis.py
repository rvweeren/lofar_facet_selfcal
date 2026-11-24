#!/usr/bin/env python

# code adapted from the orginal version by Erik Osinga
import casacore.tables as pt
import numpy as np
import os
import shutil
from astropy.time import Time
import sys
import argparse
import ast

# metadata columns of an added row that are taken from an existing row at the same time stamp
_SAME_TIME_COLUMNS = ['TIME_CENTROID', 'FIELD_ID', 'SCAN_NUMBER', 'STATE_ID', 'OBSERVATION_ID',
                      'ARRAY_ID', 'PROCESSOR_ID', 'INTERVAL', 'EXPOSURE']


def time_baseline_grid(times, ant1, ant2):
    """
    Describe the time-baseline grid of the rows of a Measurement Set.

    Parameters
    ----------
    times, ant1, ant2 : numpy.ndarray
        TIME, ANTENNA1 and ANTENNA2 columns.

    Returns
    -------
    dict
        ``itime``/``ibl``: time and baseline index of each row (baselines in ANTENNA1, ANTENNA2
        order); ``ntime``/``nbl``: number of unique time stamps and baselines; ``nduplicate``:
        number of rows that repeat a (time, baseline) pair; ``missing``: (time index, baseline
        index) arrays of the absent pairs, ordered by time and then baseline; ``first_time`` and
        ``first_bl``: row number of the first row of each time stamp and of each baseline.
    """
    utime, first_time, itime = np.unique(times, return_index=True, return_inverse=True)
    nant = int(max(np.max(ant1), np.max(ant2))) + 1
    ubl, first_bl, ibl = np.unique(ant1.astype(np.int64) * nant + ant2, return_index=True,
                                   return_inverse=True)
    itime, ibl = itime.ravel(), ibl.ravel()
    ntime, nbl = len(utime), len(ubl)
    present = np.zeros((ntime, nbl), dtype=bool)
    present[itime, ibl] = True
    nduplicate = len(times) - int(np.count_nonzero(present))
    return {'utime': utime, 'itime': itime, 'ibl': ibl, 'ntime': ntime, 'nbl': nbl,
            'first_time': first_time, 'first_bl': first_bl,
            'nduplicate': nduplicate, 'missing': np.nonzero(~present)}


def _put_by_rowblock(t, colname, value, rows_per_block=None):
    """Fill an array column with a constant value, in row blocks to limit memory."""
    shape = t.getcell(colname, 0).shape if t.nrows() > 0 else ()
    if rows_per_block is None:
        rows_per_block = max(1, int(50e6 // max(1, int(np.prod(shape)))))
    dtype = t.getcell(colname, 0).dtype
    for start in range(0, t.nrows(), rows_per_block):
        nrow = min(rows_per_block, t.nrows() - start)
        t.putcol(colname, np.full((nrow,) + shape, value, dtype=dtype), startrow=start, nrow=nrow)


def _uvw_sign(t, rows, tolerance=1e-2):
    """
    Find the sign that converts mscal.UVWJ2000() to the UVW convention of the MS.

    Returns +1 or -1, or None if neither sign reproduces the stored UVW of the given rows.
    """
    tsel = t.selectrows(rows)
    stored = tsel.getcol('UVW')
    computed = pt.taql('calc mscal.UVWJ2000() from $tsel')
    tsel.close()
    computed = np.asarray(computed).reshape(stored.shape)
    scale = np.median(np.linalg.norm(stored, axis=1))
    if scale == 0.0:
        return None
    for sign in (1., -1.):
        if np.median(np.linalg.norm(stored - sign * computed, axis=1)) / scale < tolerance:
            return sign
    return None


def regularise_ms(ms_path, outname, dryrun=False):
    """
    Write a regular copy of a Measurement Set: the same baselines at every time stamp.

    DP3 needs every time stamp to have the same baselines. Rows of baselines that are absent at
    a time stamp (for example antennas that were offline) are added as fully flagged rows with
    zero data and weight. Their metadata (FIELD_ID, SCAN_NUMBER, INTERVAL, ...) are taken from
    an existing row at the same time stamp and their UVW coordinates are computed for that time.
    The output is a standalone MS sorted on TIME, ANTENNA1, ANTENNA2; the input MS is not changed.
    (casacore's msregularize is not used: its output references the input MS, and its added rows
    copy UVW, scan and weights from a row at a different time.)

    Parameters
    ----------
    ms_path : str
        Input Measurement Set (one spectral window).
    outname : str
        Output Measurement Set. An existing one is overwritten.
    dryrun : bool, optional
        If True, do not write anything and return ``outname``, which must exist (restarted run).

    Returns
    -------
    tuple
        ``(outname, nadded)``, with ``nadded`` the number of added rows (None for a dry run).
    """
    if dryrun:
        if not os.path.isdir(outname):
            raise FileNotFoundError(outname + ' does not exist; it is created by a run with --start=0')
        return outname, None
    if os.path.abspath(outname) == os.path.abspath(ms_path):
        raise ValueError('The regularised MS cannot overwrite its input: ' + ms_path)
    addname = outname + '_add'
    for name in (outname, addname):
        if os.path.isdir(name):
            shutil.rmtree(name)

    t = pt.table(ms_path, ack=False)
    times = t.getcol('TIME')
    if len(np.unique(t.getcol('DATA_DESC_ID'))) != 1:
        t.close()
        raise ValueError(ms_path + ': regularisation needs an MS with one spectral window (DATA_DESC_ID)')
    grid = time_baseline_grid(times, t.getcol('ANTENNA1'), t.getcol('ANTENNA2'))
    if grid['nduplicate'] > 0:
        t.close()
        raise ValueError(ms_path + ': %d rows repeat a time-baseline pair, cannot regularise'
                         % grid['nduplicate'])
    mtime, mbl = grid['missing']
    nadded = len(mtime)
    nbl = grid['nbl']

    tables = [t]
    if nadded > 0:
        # an existing row of the same baseline gives antennas, feeds and cell shapes
        tsel = t.selectrows(grid['first_bl'][mbl])
        tsel.copy(addname, deep=True)
        tsel.close()
        tadd = pt.table(addname, readonly=False, ack=False)
        same_time = grid['first_time'][mtime]
        tadd.putcol('TIME', grid['utime'][mtime])
        for col in _SAME_TIME_COLUMNS:
            if col in tadd.colnames():
                tadd.putcol(col, t.getcol(col)[same_time])
        tadd.putcol('FLAG_ROW', np.ones(nadded, dtype=bool))
        _put_by_rowblock(tadd, 'FLAG', True)
        for col in tadd.colnames():
            if col in ('FLAG', 'FLAG_ROW', 'FLAG_CATEGORY', 'UVW', 'TIME') + tuple(_SAME_TIME_COLUMNS):
                continue
            if not tadd.iscelldefined(col, 0) or tadd.isscalarcol(col):
                continue
            dtype = tadd.coldatatype(col)
            if dtype in ('complex', 'dcomplex') or col in ('WEIGHT', 'WEIGHT_SPECTRUM'):
                _put_by_rowblock(tadd, col, 0)
        tables.append(tadd)
        print(f'{ms_path}: adding {nadded} flagged rows for baselines absent at some time stamps')

    tcomb = pt.table(tables, ack=False)
    tsort = tcomb.sort('TIME,ANTENNA1,ANTENNA2')
    tsort.copy(outname, deep=True)
    tsort.close()
    tcomb.close()
    for tab in tables:
        tab.close()
    if os.path.isdir(addname):
        shutil.rmtree(addname)

    if nadded > 0:
        # rows of the full grid are sorted on time and baseline, so the added rows are known
        added_rows = mtime.astype(np.int64) * nbl + mbl
        tout = pt.table(outname, readonly=False, ack=False)
        is_added = np.zeros(tout.nrows(), dtype=bool)
        is_added[added_rows] = True
        existing = np.nonzero(~is_added)[0]
        sample = existing[np.linspace(0, len(existing) - 1, min(len(existing), 2000)).astype(np.int64)]
        sign = _uvw_sign(tout, sample)
        if sign is not None:
            tsel = tout.selectrows(added_rows)
            uvw = np.asarray(pt.taql('calc mscal.UVWJ2000() from $tsel')).reshape(nadded, 3)
            tsel.putcol('UVW', sign * uvw)
            tsel.close()
        else:
            # interpolate in time from the existing rows of each baseline
            print('UVW of the added rows interpolated in time (mscal.UVWJ2000 does not match the stored UVW)')
            uvw = tout.getcol('UVW').reshape(grid['ntime'], nbl, 3)
            present = ~is_added.reshape(grid['ntime'], nbl)
            for ibl in np.unique(mbl):
                ok = present[:, ibl]
                for k in range(3):
                    uvw[~ok, ibl, k] = np.interp(grid['utime'][~ok], grid['utime'][ok], uvw[ok, ibl, k])
            tout.putcol('UVW', uvw.reshape(-1, 3))
        tout.close()

    with pt.table(outname, ack=False) as tout:
        check = time_baseline_grid(tout.getcol('TIME'), tout.getcol('ANTENNA1'), tout.getcol('ANTENNA2'))
    if check['nduplicate'] > 0 or len(check['missing'][0]) > 0:
        raise RuntimeError(outname + ' is not regular after regularisation')
    return outname, nadded


def mjd_to_mvtime(mjd_time):
    '''
    Function to convert MJD to MVTime string format (e.g., 19Feb2010/14:01:23)
    Modified Julian Date
    '''
    # Convert MJD time to datetime
    dt = Time(mjd_time / 3600. / 24., format='mjd').to_datetime()

    # Convert datetime to the required MVTime format (e.g., 19Feb2010/14:01:23)
    mvtime_str = dt.strftime("%d%b%Y/%H:%M:%S.%f")[:-3]  # Keep milliseconds with 3 digits
    return mvtime_str


def split_ms(ms_path, overwrite=False, prefix='', dysco=True, return_mslist=False, dryrun=False, metadata_compression=False):
    # Open the Measurement Set
    t = pt.table(ms_path, ack=False)

    # Extract the TIME column
    time = t.getcol('TIME')

    # Find the unique time steps and calculate the time differences
    unique_time = np.unique(time)
    # last one can be different, because its the last one ;) 
    time_diff = np.diff(unique_time)[:-1]

    median_timestep = np.median(time_diff)
    print(f"Median timestep in ms is {median_timestep} seconds")

    # Set a threshold for irregular time gaps (20% difference to median time okay??)
    diff_times_medsub = np.abs(time_diff - median_timestep)  # from facetselfcal
    # Find the indices where time steps are irregular
    breakpoints, = np.where(diff_times_medsub > 0.2 * median_timestep)  # 20% tolerance

    # Define the output folder
    output_folder = 'split_measurements'
    os.makedirs(output_folder, exist_ok=True)

    # Add start and end times for the chunks (including the start and end times of the full MS)
    start_times = np.insert(unique_time[breakpoints + 1], 0, unique_time[0])  # Add the start of the MS
    end_times = np.append(unique_time[breakpoints], unique_time[-1])  # Add the end of the MS
    ## Now we are skipping steps where we would just split off an MS
    ## that only has a single timestep. Which is good. 
    ## but the numbering will be a bit weird, reflecting how many timesteps are not regular

    nsplits = len(start_times) - np.sum(start_times == end_times)

    print(f"Will split the MS into {nsplits} parts")
    successful_files = []

    with open(f'{output_folder}/all_mses.txt', 'w') as ftxt:
        # Loop over breakpoints and create DP3 config files for each chunk
        for i, (start_time_mjd, end_time_mjd) in enumerate(zip(start_times, end_times)):

            timediff = end_time_mjd-start_time_mjd

            if timediff <=0 or timediff < 2*median_timestep:
                # Skip invalid ranges where start_time is not less than end_time (or off by floating errors)
                # or if we'd write an MS with 1 timestep only.
                print(f"Skipping part {i} because invalid range: start_time={start_time_mjd}, end_time={end_time_mjd}, {timediff=} seconds")
                continue 

            # Convert the MJD times to MVTime string format
            start_time = mjd_to_mvtime(start_time_mjd)
            end_time = mjd_to_mvtime(end_time_mjd)
            if prefix == '':
                chunk_name = f"{prefix}chunk_{str(i + 1).zfill(3)}.ms"
            else:
                chunk_name = f"{prefix}_chunk_{str(i + 1).zfill(3)}.ms"  # add underscore to prevent cluttering
            print(f"Splitting between {start_time} and {end_time}")

            # Write DP3 config file for each chunk
            config_file = f"{output_folder}/split_chunk_{i + 1}.parset"
            with open(config_file, 'w') as f:
                f.write(f"msin={ms_path}\n")
                f.write(f"msin.starttime={start_time}\n")
                f.write(f"msin.endtime={end_time}\n")
                if dysco:
                    f.write("msout.storagemanager=dysco\n")
                f.write(f"msout={output_folder}/{chunk_name}\n")
                f.write("steps=[]\n")
                if not metadata_compression:
                    f.write("msout.uvwcompression=False\n")
                    f.write("msout.antennacompression=False\n")
                    f.write("msout.scalarflags=False\n")

            # Run DP3 for each chunk
            if overwrite and os.path.isdir(output_folder + '/' + chunk_name) and not dryrun:
                os.system('rm -rf ' + output_folder + '/' + chunk_name)
            if not dryrun:
                result = os.system(f"DP3 {config_file}")
            else:
                result = 0  # set for the next step
            if result != 0:
                print(f"DP3 failed for {chunk_name}. Exiting.")
                sys.exit(1)
            else:
                successful_files.append(chunk_name)

        # write ms files with a space between them
        for ms_file in successful_files:
            ftxt.write(f"{ms_file} ")

    print(f"Measurement Set has been split into {i + 1} time chunks of which {len(successful_files)} are good splits.")
    print(f"Output in {output_folder}")
    t.close()
    if return_mslist:
        return [output_folder + '/' + ms_file for ms_file in successful_files]
    return


def main():
    parser = argparse.ArgumentParser(
        description='Split MS with an irregular time-axis (for example from the MeerKAT CARACal pipeline)')
    parser.add_argument('--ms', help='Measurement Set', type=str, required=True)
    parser.add_argument('--overwrite', help='Overwrite existing Measurement Sets', action='store_true')
    parser.add_argument('--nodysco', help='Turn off Dysco compression', action='store_false')
    parser.add_argument('--prefix', help='Extra string to attach to the output names of the splited MS', type=str,
                        default='')
    parser.add_argument('--metadatacompression',
                        help='Use MS metadata compression (flags, uvw coordinates, and antenna table). The default is True. Will be turned of always for non-LOFAR data.',
                        type=ast.literal_eval,
                        default=False)
    
    args = parser.parse_args()

    # Make sure MS is regularised (same baselines at every time stamp).
    ms_path = args.ms.rstrip('/')
    with pt.table(ms_path, ack=False) as t:
        grid = time_baseline_grid(t.getcol('TIME'), t.getcol('ANTENNA1'), t.getcol('ANTENNA2'))
    if len(grid['missing'][0]) > 0:
        ms_path, _ = regularise_ms(ms_path, os.path.basename(ms_path) + '.regularised')
    # Do the splitting
    split_ms(ms_path, overwrite=args.overwrite, prefix=args.prefix, dysco=args.nodysco, metadata_compression=args.metadatacompression)


if __name__ == "__main__":
    main()
