# LOFAR, MeerKAT, and ASKAP facet self-calibration
General direction-independent and direction-dependent self-calibration: 
 - refinement self-calibration for individual 'extracted' datasets 
 - full field of view self-calibration and extraction of regions of interest

If you use facetselfcal or extraction for scientific work, please cite van Weeren et al. (2021, A&A, 651, 115) paper: \
https://ui.adsabs.harvard.edu/abs/2021A%26A...651A.115V/abstract 

Requirements:
- Container with all standard LOFAR software can be downloaded here: https://tikk3r.github.io/flocs/ 

Installation:
\
`git clone https://github.com/rvweeren/lofar_facet_selfcal.git`
\
`pip install git+https://github.com/rvweeren/lofar_facet_selfcal.git`
\
\
With pip install, you install ```facetselfcal```, ```h5_merger```, ```ds9facetgenerator```, ```sub_sources_outside_region```
as command line functionalities. Note that facetselfcal is also included in the Singularity container and can be called directly using ```facetselfcal```. However, by cloning the master version from GitHub, you are guaranteed to have the latest version. If you do not care about having the latest version you only need the download Singularity container to start using facetselfcal and the installation step can be skipped.
\
\
Some basic help is given by:\
`python /<path>/lofar_facet_selfcal/facetselfcal.py -h`, or `facetselfcal -h` if you want to use the pre-installed version in the Singularity container.


# LOFAR
Usage examples:
- HBA Dutch baselines for extracted LoTSS data from the ddf-pipeline:\
`python /<path>/lofar_facet_selfcal/facetselfcal.py -b yourDS9extractbox.reg --auto -i yourimagename yourextracted.ms`

- Standard auto settings:\
`python /<path>/lofar_facet_selfcal/facetselfcal.py --imsize=1600 --auto -i yourimagename yourextracted.ms` 

- With a config file (see an example in data/example_config.txt)):\
`python /<path>/lofar_facet_selfcal/facetselfcal.py --config=yourconfig.txt yourextracted.ms`

HBA international baselines
- delaycalibrator
- target source

LBA Dutch baselines
 - widefield
 - decameter band

# Extraction
Information on how to use the [extract option](https://github.com/rvweeren/lofar_facet_selfcal/wiki/FACETSELFCAL-OVERVIEW).

# MeerKAT
- Support for UHF, L-band, and S-band
- direction-independent and direction-dependent self-calibration
Specific MeerKAT information can be found [here](https://github.com/rvweeren/lofar_facet_selfcal/wiki/FACETSELFCAL-OVERVIEW#meerkat-data-reduction).

# uGMRT
- Some notes on [uGMRT processing](https://github.com/rvweeren/lofar_facet_selfcal/wiki/uGMRT-processing)
