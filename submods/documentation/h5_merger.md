## Merge solutions with ```h5_merger```

-------------------------------

With ```h5_merger``` it is possible to merge H5parm solution files (typically with extension .h5).
Within these solution sets there has to be a source, antenna, and solXXX tables. 
They should also contain 'amplitude000' and/or 'phase000' solution tables.
This script has also other functionalities, such as time and frequency averaging or linear to circular conversion and vice versa. 
These can be independently used from the merging functionality. You can find all the options below.

### Requirements

```h5_mergers.py``` uses the following libraries:
* ```losoto```
* ```casacore.tables```
* ```numpy```
* ```scipy```
* ```tables```

-------------------------------

# Examples

Examples below demonstrate how to use ```h5_merger``` for specific cases:

#### 1) Merge series of h5 files:

```h5_merger -in *.h5 -out out.h5```\

#### 2) Merge series of h5 files with frequency and time axis from a measurement set:

```h5_merger -in *.h5 -out out.h5 -ms <YOUR_MS>```\
The input MS can also be multiple measurement sets (where the script will generate one time and freq axis stacking all MS).

#### 3) Merge series of h5 files with specific frequency and time axis from one h5:

```h5_merger -in *.h5 -out out.h5 --h5_time_freq <H5>```\
This will take the time and freq axis from H5 to interpolate to.
It is also possible to give a boolean to ```--h5_time_freq``` which means that all input h5 files will be used to generate the time and freq axis from (this is the default if ```-ms``` is not used).

#### 4) Merge all h5 files in one direction:

```h5_merger -in *.h5 -out out.h5 --merge_all_in_one```\
The script merges by default the same directions with each other, but if the user wishes to merge all h5 files in one direction, it is possible to add the ```--merge_all_in_one``` option.

#### 5) Merge all h5 files and do not convert tec:

```h5_merger -in *.h5 -out out.h5 --keep_tec```\
The script converts by default TEC input to phases (see Equation 1 in Sweijen et al +22), but the user can also turn this conversion off.

#### 6) Convert circular to linear polarization and vice versa:

```h5_merger -in *.h5 -out out.h5 --circ2lin```\
or\
```h5_merger -in *.h5 -out out.h5 --lin2circ```\
```--circ2lin``` and ```-lin2circ``` can be added to any type of merge, as this conversion will be done at the end of the algorithm.

#### 7) Merge h5 solutions with different freq and time axis:

```h5_merger -in *.h5 -out out.h5 --merge_diff_freq```\
This is for example useful if you wish to merge solution files from different frequency bands together into 1 big solution file.

#### 8) Add Core Stations back to output:

```h5_merger -in *.h5 -out out.h5 -ms <YOUR_MS> --add_cs```\
This functionality will replace the super station (ST001) in the h5 file with the core stations from a given MS file.

#### 9) Return only stations from specific MS file:

```h5_merger -in *.h5 -out out.h5 -ms <YOUR_MS> --add_ms_stations```\
This functionality will return in the output H5 file only the stations that are present in the MS file. 
For international stations default values will be returned (amplitudes=1 for diagonals, amplitudes=0 for off-diagonals, phases=0).

Combinations of all the above are possible to merge more exotic cases, such as for example a merge like this:\
```h5_merger -in *.h5 -out test.h5 -ms <YOUR_MS> --h5_time_freq=true --add_ms_stations --merge_diff_freq --merge_all_in_one --keep_tec --circ2lin```

-------------------------------

#### Contact
Let me know if you are using this script or other scripts and have any issues or suggestions for improvements.
Email: jurjendejong(AT)strw.leidenuniv.nl
