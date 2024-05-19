# lofar_facet_selfcal
Selfcalibration for individual LOFAR sources and facets. If you use facetselfcal or extraction for scientific work, please cite the van Weeren et al. (2021, A&A, 651, 115) paper. 

Requirements:
- https://github.com/jurjen93/lofar_helpers 
- Container with all standard LOFAR software: https://tikk3r.github.io/flocs/ 

Installation:
create a directory and\
`git clone https://github.com/jurjen93/lofar_helpers.git`\
`git clone https://github.com/rvweeren/lofar_facet_selfcal.git`\
\
\
Usage:\
\
HBA Dutch baselines for extracted LoTSS data from the ddf-pipeline:

`python /<path>/lofar_facet_selfcal/facetselfcal.py --helperscriptspath="/<path>/lofar_facet_selfcal/" --helperscriptspathh5merge="/<path>/lofar_helpers/" -b yourDS9extractbox.reg --auto -i yourimagename yourextracted.ms`

or 

`python /<path>/lofar_facet_selfcal/facetselfcal.py --helperscriptspath="/<path>/lofar_facet_selfcal/" --helperscriptspathh5merge="/<path>/lofar_helpers/" --imsize=1600 --auto -i yourimagename yourextracted.ms`
 

HBA international baselines

- delaycalibrator
- target source

LBA Dutch baselines
 - < 30 MHz
 - < 30 MHz

