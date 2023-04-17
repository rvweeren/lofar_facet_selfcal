# lofar_facet_selfcal
Selfcalibration for individual LOFAR sources and facets. If you use facetselfcal or extraction for scientific work, please cite the van Weeren et al. (2021, A&A, 651, 115) paper. 

Requirements:
h5_merger.py from https://github.com/jurjen93/lofar_helpers

Usage:

HBA Dutch baselines (e.g., LoTSS)

facetselfcal.py -b yourDS9extractbox.reg --auto -i yourimagename yourextracted.ms

or 

facetselfcal.py --imsize=1600 --auto -i yourimagename yourextracted.ms
 

HBA international baselines

- delaycalibrator
- target source

LBA Dutch baselines
 - < 30 MHz
 - < 30 MHz

