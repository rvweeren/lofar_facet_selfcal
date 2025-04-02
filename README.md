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


# MeerKAT
- Support for UHF, L-band, and S-band
- direction-independent and direction-dependent self-calibration

**Direction independent selfcalibration examples** 

L-band example from SDP-pipeline output:
`python /<path>/lofar_facet_selfcal/facetselfcal.py -i imageDI --forwidefield --noarchive --fitspectralpol=9 --solint-list="['1min']" --soltype-list="['scalarphase']" --soltypecycles-list=[0] --smoothnessconstraint-list=[100.] --imsize=12000 --channelsout=12 --niter=45000 --stop=3 --multiscale --useaoflagger --aoflagger-strategy=default_StokesQUV.lua --multiscale-start=0 --parallelgridding=2 --msinnchan=3600 --msinstartchan=60 --avgfreqstep=2 MyTarget.ms` 

L-band example from SDP-pipeline output with factor 2 frequency averaging:
`python /<path>/lofar_facet_selfcal/facetselfcal.py -i imageDI --forwidefield --noarchive --fitspectralpol=9 --solint-list="['1min']" --soltype-list="['scalarphase']" --soltypecycles-list=[0] --smoothnessconstraint-list=[100.] --imsize=12000 --channelsout=12 --niter=45000 --stop=3 --multiscale --useaoflagger --aoflagger-strategy=default_StokesQUV.lua --multiscale-start=0 --parallelgridding=2 --msinnchan=1800 --msinstartchan=30 MyTarget.ms` 

UHF-band example from SDP-pipeline output:
`python /<path>/lofar_facet_selfcal/facetselfcal.py -i imageDI --forwidefield --noarchive --fitspectralpol=9 --solint-list="['32sec']" --soltype-list="['scalarphase']" --soltypecycles-list=[0] --smoothnessconstraint-list=[50.] --imsize=12000 --channelsout=12 --niter=45000 --stop=3 --multiscale --useaoflagger --aoflagger-strategy=default_StokesQUV.lua --multiscale-start=0 --parallelgridding=2 --msinnchan=3420 --msinstartchan=220 --avgfreqstep=2 MyTarget.ms` 

UHF-band example from SDP-pipeline output with factor 2 frequency averaging:
`python /<path>/lofar_facet_selfcal/facetselfcal.py -i imageDI --forwidefield --noarchive --fitspectralpol=9 --solint-list="['32sec']" --soltype-list="['scalarphase']" --soltypecycles-list=[0] --smoothnessconstraint-list=[50.] --imsize=12000 --channelsout=12 --niter=45000 --stop=3 --multiscale --useaoflagger --aoflagger-strategy=default_StokesQUV.lua --multiscale-start=0 --parallelgridding=2 --msinnchan=1710 --msinstartchan=110 MyTarget.ms` 
