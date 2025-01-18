# LOFAR and MeerKAT facet self-calibration
General direction-independent and direction-dependent self-calibration: 
 - refinement self-calibration for individual 'extracted' datasets 
 - full field of view self-calibration and extraction of regions of interest

If you use facetselfcal or extraction for scientific work, please cite van Weeren et al. (2021, A&A, 651, 115) paper: \
https://ui.adsabs.harvard.edu/abs/2021A%26A...651A.115V/abstract 

Requirements:
- Container with all standard LOFAR software: https://tikk3r.github.io/flocs/ 

Installation:
\
`git clone https://github.com/rvweeren/lofar_facet_selfcal.git`
\
`pip install git+https://github.com/rvweeren/lofar_facet_selfcal.git`
\
\
(with pip install, you install ```facetselfcal```, ```h5_merger```, ```ds9facetgenerator```, ```sub_sources_outside_region```
as command line functionalities)
\
\
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
 - < 30 MHz
 - < 30 MHz

MeerKAT
- UHF, L-band, and S-band
- direction-independent and direction-dependent self-calibration
