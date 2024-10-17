#!/usr/bin/env python

# code adapted from the orginal version by Erik Osinga
import casacore.tables as pt
import numpy as np
import os
from astropy.time import Time
from casacore.tables import table, msregularize
import sys
import argparse

def regularize_ms(ms_path, overwrite=False, dryrun=False):
    regularised = False
    if ".reg" in ms_path:
        regularised=True

    if not regularised and not dryrun:
        if os.path.isdir(ms_path.replace(".ms",".reg.ms")) and overwrite:
            os.system('rm -rf ' + ms_path.replace(".ms",".reg.ms"))
        try:
            # Apply the msregularize function to make the MS regular
            msregularize(ms_path, ms_path.replace(".ms",".reg.ms"))
            print(f"Measurement Set {ms_path} has been regularized.")
            ms_path = ms_path.replace(".ms",".reg.ms")
        except:
           pass 
    return ms_path

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

def split_ms(ms_path, overwrite=False, prefix='', dysco=True, return_mslist=False, dryrun=False):
    # Open the Measurement Set
    t = pt.table(ms_path)

    # Extract the TIME column
    time = t.getcol('TIME')

    # Find the unique time steps and calculate the time differences
    unique_time = np.unique(time)
    # last one can be different, because its the last one ;) 
    time_diff = np.diff(unique_time)[:-1]

    median_timestep = np.median(time_diff)
    print(f"Median timestep in ms is {median_timestep} seconds")

    # Set a threshold for irregular time gaps (20% difference to median time okay??)
    diff_times_medsub = np.abs(time_diff - median_timestep) # from facetselfcal
    # Find the indices where time steps are irregular
    breakpoints, = np.where(diff_times_medsub > 0.2*median_timestep) # 20% tolerance


    # Define the output folder
    output_folder = 'split_measurements'
    os.makedirs(output_folder, exist_ok=True)

    # Add start and end times for the chunks (including the start and end times of the full MS)
    start_times = np.insert(unique_time[breakpoints + 1], 0, unique_time[0])  # Add the start of the MS
    end_times = np.append(unique_time[breakpoints], unique_time[-1])  # Add the end of the MS
    ## Now we are skipping steps where we would just split off an MS
    ## that only has a single timestep. Which is good. 
    ## but the numbering will be a bit weird, reflecting how many timesteps are not regular

    nsplits = len(start_times)-np.sum(start_times == end_times)

    print(f"Will split the MS into {nsplits} parts")
    successful_files = []

    with open(f'{output_folder}/all_mses.txt', 'w') as ftxt:
        # Loop over breakpoints and create DP3 config files for each chunk
        for i, (start_time_mjd, end_time_mjd) in enumerate(zip(start_times, end_times)):
            if start_time_mjd >= end_time_mjd:
                print(f"Skipping invalid range: start_time={start_time_mjd}, end_time={end_time_mjd}")
                continue  # Skip invalid ranges where start_time is not less than end_time

            # Convert the MJD times to MVTime string format
            start_time = mjd_to_mvtime(start_time_mjd)
            end_time = mjd_to_mvtime(end_time_mjd)
            if prefix == '':
               chunk_name = f"{prefix}chunk_{str(i+1).zfill(3)}.ms"
            else:
               chunk_name = f"{prefix}_chunk_{str(i+1).zfill(3)}.ms" # add underscore to prevent cluttering
            print(f"Splitting between {start_time} and {end_time}")

            # Write DP3 config file for each chunk
            config_file = f"{output_folder}/split_chunk_{i+1}.parset"
            with open(config_file, 'w') as f:
                f.write(f"msin={ms_path}\n")
                f.write(f"msin.starttime={start_time}\n")
                f.write(f"msin.endtime={end_time}\n")
                if dysco:
                    f.write(f"msout.storagemanager=dysco\n")
                f.write(f"msout={output_folder}/{chunk_name}\n")
                f.write("steps=[]\n")
        
            # Run DP3 for each chunk
            if overwrite and os.path.isdir(output_folder + '/' + chunk_name) and not dryrun:
                os.system('rm -rf ' + output_folder + '/' +  chunk_name)    
            if not dryrun:
                result = os.system(f"DP3 {config_file}")
            else:
                result = 0 # set for the next step    
            if result != 0:
                print(f"DP3 failed for {chunk_name}. Exiting.")
                sys.exit(1)
            else:
                successful_files.append(chunk_name)

        # write ms files with a space between them
        for ms_file in successful_files:
            ftxt.write(f"{ms_file} ")

    print(f"Measurement Set has been split into {i+1} time chunks.")
    print(f"Output in {output_folder}")
    t.close()
    if return_mslist:
        return  [output_folder + '/' + ms_file for ms_file in successful_files]
    return

def main():
   parser = argparse.ArgumentParser(description='Split MS with an irregular time-axis (for example from the MeerKAT CARACal pipeline)')
   parser.add_argument('--ms', help='Measurement Set', type=str, required=True)
   parser.add_argument('--overwrite', help='Overwrite existing Measurement Sets', action='store_true')
   parser.add_argument('--nodysco', help='Turn off Dysco compression', action='store_false')
   parser.add_argument('--prefix', help='Extra string to attach to the output names of the splited MS', type=str, default='')
   args = parser.parse_args()  
   
   # Make sure MS is regularised. 
   ms_path = regularize_ms(args.ms, overwrite=args.overwrite)
   # Do the splitting
   split_ms(ms_path, overwrite=args.overwrite, prefix=args.prefix, dysco=args.nodysco)


if __name__ == "__main__":
   main()
