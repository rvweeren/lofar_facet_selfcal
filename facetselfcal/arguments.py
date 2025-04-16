import ast
import argparse
import numpy as np
import re


def option_parser():
    parser = argparse.ArgumentParser(description='Self-Calibrate a facet from a LOFAR observation')

    imagingparser = parser.add_argument_group("-------------------------Imaging Settings-------------------------")
    # Imaging settings
    imagingparser.add_argument('--imager',
                               help="Imager to use WSClean or DDFACET. The default is WSCLEAN.",
                               default='WSCLEAN',
                               type=str)
    imagingparser.add_argument('-i', '--imagename',
                               help='Prefix name for image. This is by default "image".',
                               default='image',
                               type=str)
    imagingparser.add_argument('--imsize',
                               help='Image size, required if boxfile is not used. The default is None.',
                               type=int)
    imagingparser.add_argument('-n', '--niter',
                               help='Number of iterations. This is computed automatically if None.',
                               default=None,
                               type=int)
    imagingparser.add_argument('--maskthreshold',
                               help="Mask noise thresholds used from image 1 to N made by MakeMask.py/breizorro. This is by default [5.0,4.5,4.5,4.5,4.0].",
                               default=[5.0, 4.5, 4.5, 4.5, 4.0],
                               type=arg_as_list)
    imagingparser.add_argument('--localrmswindow',
                               help="local-rms-window parameter for automasking in WSClean (in units of psfs), default=0 (0 means it is not used; suggested value 50)",
                               default=0,
                               type=int)
    imagingparser.add_argument('--removenegativefrommodel',
                               help="Remove negative clean components in model predict. This is by default turned off at selfcalcycle 2. See also option autoupdate-removenegativefrommodel.",
                               type=ast.literal_eval,
                               default=True)
    imagingparser.add_argument('--autoupdate-removenegativefrommodel',
                               help="Turn off removing negative clean components at selfcalcycle 2 (for high dynamic range imaging it is better to keep all clean components). The default is True.",
                               type=ast.literal_eval,
                               default=True)
    imagingparser.add_argument('--fitsmask',
                               help='Fits mask for deconvolution (needs to match image size). The fitsmask stays fixed during selfcal. If this is not provided automasking is used in combination with MakeMask.py/breizorro. If set to "nofitsmask" then only WSCLean auto-masking is used.',
                               type=str)
    imagingparser.add_argument('--fitsmask-start',
                               help='Fits mask for deconvolution for image000 (needs to match image size). For subsequent selfcal cycles automasking is used with MakeMask.py/breizorro.',
                               type=str)
    imagingparser.add_argument('--DS9cleanmaskregionfile',
                               help='A DS9 region file (with WCS coordinates) that will be added to the clean mask used in combination with breizorro.',
                               type=str)
    imagingparser.add_argument('--robust',
                               help='Briggs robust parameter for imagaging. The default is -0.5. Also allowed are the strings uniform or naturual which will override Briggs weighting.',
                               default=-0.5,
                               type=str_or_float)
    imagingparser.add_argument('--multiscale-start',
                               help='Start multiscale deconvolution at this selfcal cycle. This is by default 1.',
                               default=1,
                               type=int)

    imagingparser.add_argument('--uvminim',
                               help='Inner uv-cut for imaging in lambda. The default is 80.',
                               default=80.,
                               type=floatlist_or_float)
    imagingparser.add_argument('--uvmaxim',
                               help='Outer uv-cut for imaging in lambda. The default is None',
                               default=None,
                               type=floatlist_or_float)
    imagingparser.add_argument('--pixelscale', '--pixelsize',
                               help='Pixels size in arcsec. Typically, 3.0 for LBA and 1.5 for HBA for the Dutch stations (these are also the default values).',
                               type=float)
    imagingparser.add_argument('--channelsout',
                               help='Number of channels out during imaging (see WSClean documentation). This is by default 6.',
                               default=6,
                               type=int)
    imagingparser.add_argument('--mgain',
                               help='Deconvolution --mgain setting for WSCLean, see WSClean documentation. The default value is 0.75',
                               default=0.75,
                               type=float)
    imagingparser.add_argument('--nmiter',
                               help='Deconvolution --nmiter setting for WSCLean, see WSClean documentation. The default value is 12',
                               default=12,
                               type=int)    
    imagingparser.add_argument('--multiscale',
                               help='Use multiscale deconvolution (see WSClean documentation).',
                               action='store_true')
    imagingparser.add_argument('--multiscalescalebias',
                               help='Multiscalescale bias scale parameter for WSClean (see WSClean documentation). This is by default 0.75.',
                               default=0.75,
                               type=float)
    imagingparser.add_argument('--multiscalemaxscales',
                               help='Multiscalescale max scale parameter for WSClean (see WSClean documentation). Default 0 (means set automatically).',
                               default=0,
                               type=int)

    imagingparser.add_argument('--paralleldeconvolution',
                               help="Parallel-deconvolution size for WSCLean (see WSClean documentation). The default is 0 which means the parallel deconvolution value is determined automatically in facetselfcal. For large images, values around 1000-2000 usually work well. For any value below zero the option is turned off.",
                               default=0,
                               type=int)
    imagingparser.add_argument('--parallelgridding',
                               help="Parallel-gridding for WSClean (see WSClean documentation). The default is 0 which means it is set automatically",
                               default=0,
                               type=int)
    imagingparser.add_argument('--deconvolutionchannels',
                               help="Deconvolution channels value for WSClean (see WSClean documentation). This is by default 0 (means deconvolution-channels equals channels-out).",
                               default=0,
                               type=int)
    imagingparser.add_argument('--idg',
                               help="Use the Image Domain gridder (see WSClean documentation).",
                               action='store_true')
    imagingparser.add_argument('--fitspectralpol',
                               help="Use fit-spectral-pol in WSClean (see WSClean documentation) with this order. The default is 3. fit-spectral-pol can be disabled by setting it to a value less than 1",
                               default=3,
                               type=int)
    imagingparser.add_argument('--ddpsfgrid',
                               help="Value for option -dd-psf-grid with WSClean (integer, by default this value is None and the option is not used",
                               type=int, default=None)
    imagingparser.add_argument("--gapchanneldivision",
                               help='Use the -gap-channel-division option in wsclean imaging and predicts (default is not to use it)',
                               action='store_true')
    imagingparser.add_argument('--taperinnertukey',
                               help="Value for taper-inner-tukey in WSClean (see WSClean documentation), useful to supress negative bowls when using --uvminim. Typically values between 1.5 and 4.0 give good results. The default is None.",
                               default=None,
                               type=float)
    imagingparser.add_argument('--makeimage-ILTlowres-HBA',
                               help='Make 1.2 arcsec tapered image as quality check of ILT 1 arcsec imaging.',
                               action='store_true')
    imagingparser.add_argument('--makeimage-fullpol',
                               help='Make Stokes IQUV version for quality checking.',
                               action='store_true')
    imagingparser.add_argument('--groupms-h5facetspeedup',
                               help='Speed up DDE imaging with h5s',
                               action='store_true')
    imagingparser.add_argument('--DP3-BDA-imaging',
                               help='Speed up DDE facet-imaging by compressing MS with time-BDA.',
                               action='store_true')
    imagingparser.add_argument("--update-channelsout",
                               help='Change --channelsout automatically if there is high peak flux.',
                               action='store_true')
    imagingparser.add_argument("--update-fitspectralpol",
                               help='Change --fitspectralpol automatically if there is high peak flux.',
                               action='store_true')

    calibrationparser = parser.add_argument_group(
        "-------------------------Calibration Settings-------------------------")
    # Calibration options
    calibrationparser.add_argument('--avgfreqstep',
                                   help="Extra DP3 frequency averaging to speed up a solve. This is done before any other correction and could be useful for long baseline infield calibrators. Allowed are integer values or for example '195.3125kHz'; options for units: 'Hz', 'kHz', or 'MHz'. The default is None.",
                                   type=str_or_int,
                                   default=None)
    calibrationparser.add_argument('--avgtimestep',
                                   help="Extra DP3 time averaging to speed up a solve. This is done before any other correction and could be useful for long baseline infield calibrators. Allowed are integer values or for example '16.1s'; options for units: 's' or 'sec'. The default is None.",
                                   type=str_or_int,
                                   default=None)
    calibrationparser.add_argument('--msinnchan',
                                   help="Before averaging, only take this number of input channels. The default is None.",
                                   type=int,
                                   default=None)
    calibrationparser.add_argument('--msinstartchan',
                                   help="Before averaging, start channel for --msinnchan. The default is 0. ",
                                   type=int,
                                   default=0)
    calibrationparser.add_argument('--msinntimes',
                                   help="DP3 msin.ntimes setting. This is mainly used for testing purposes. The default is None.",
                                   type=int,
                                   default=None)
    calibrationparser.add_argument('--autofrequencyaverage-calspeedup',
                                   help="Update April 24: Avoid usage because of corrupt vs correct. Try extra averaging during some selfcalcycles to speed up calibration.",
                                   action='store_true')
    calibrationparser.add_argument('--autofrequencyaverage',
                                   help='Try frequency averaging if it does not result in bandwidth smearing',
                                   action='store_true')

    calibrationparser.add_argument('--phaseupstations',
                                   help="Phase up to a superstation. Possible input: 'core' or 'superterp'. The default is None.",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--phaseshiftbox',
                                   help="DS9 region file to shift the phasecenter to. This is by default None.",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--weightspectrum-clipvalue',
                                   help="Extra option to clip WEIGHT_SPECTRUM values above the provided number. Use with care and test first manually to see what is a fitting value. The default is None.",
                                   type=float,
                                   default=None)
    calibrationparser.add_argument('-u', '--uvmin',
                                   help="Inner uv-cut for calibration in lambda. The default is 80 for LBA and 350 for HBA.",
                                   type=floatlist_or_float,
                                   default=None)
    calibrationparser.add_argument('--uvmax',
                                   help="Outer uv-cut for calibration in lambda. The default is None",
                                   type=floatlist_or_float,
                                   default=None)
    calibrationparser.add_argument('--uvminscalarphasediff',
                                   help='Inner uv-cut for scalarphasediff calibration in lambda. The default is equal to input for --uvmin.',
                                   type=float,
                                   default=None)
    calibrationparser.add_argument("--update-uvmin",
                                   help='Update uvmin automatically for the Dutch array.',
                                   action='store_true')
    calibrationparser.add_argument("--update-multiscale",
                                   help='Switch to multiscale automatically if large islands of emission are present.',
                                   action='store_true')
    calibrationparser.add_argument("--soltype-list",
                                   type=arg_as_list,
                                   default=['tecandphase', 'tecandphase', 'scalarcomplexgain'],
                                   help="List with solution types. Possible input: 'complexgain', 'scalarcomplexgain', 'scalaramplitude', 'amplitudeonly', 'phaseonly', 'fulljones', 'rotation', 'rotation+diagonal', 'rotation+diagonalphase','rotation+diagonalamplitude',                   'rotation+scalar','rotation+scalaramplitude','rotation+scalarphase', 'tec', 'tecandphase', 'scalarphase', 'scalarphasediff', 'scalarphasediffFR', 'phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin', 'scalarphase_phmin', 'scalarphase_slope', 'phaseonly_slope'. The default is [tecandphase,tecandphase,scalarcomplexgain].")
    calibrationparser.add_argument("--solint-list",
                                   type=check_strlist_or_intlist,
                                   default=[1, 1, 120],
                                   help="Solution interval corresponding to solution types (in same order as soltype-list input). The default is [1,1,120].")
    calibrationparser.add_argument("--nchan-list",
                                   type=arg_as_list,
                                   default=[1, 1, 1],
                                   help="Number of channels corresponding to solution types (in same order as soltype-list input). The default is [1,1,1].")
    calibrationparser.add_argument("--smoothnessconstraint-list",
                                   type=arg_as_list,
                                   default=[0., 0., 5.],
                                   help="List with frequency smoothness values in MHz (in same order as soltype-list input). The default is [0.,0.,5.].")
    calibrationparser.add_argument("--smoothnessreffrequency-list",
                                   type=arg_as_list,
                                   default=[0., 0., 0.],
                                   help="List with optional reference frequencies in MHz for the smoothness constraint (in same order as soltype-list input). When unequal to 0, the size of the smoothing kernel will vary over frequency by a factor of smoothnessreffrequency*(frequency^smoothnessspectralexponent). The default is [0.,0.,0.].")
    calibrationparser.add_argument("--smoothnessspectralexponent-list",
                                   type=arg_as_list,
                                   default=[-1., -1., -1.],
                                   help="If smoothnessreffrequency is not equal to zero then this parameter determines the frequency scaling law. It is typically useful to take -2 for scalarphasediff, otherwise -1 (1/nu). The default is [-1.,-1.,-1.].")
    calibrationparser.add_argument("--smoothnessrefdistance-list",
                                   type=arg_as_list,
                                   default=[0., 0., 0.],
                                   help="If smoothnessrefdistance is not equal to zero then this parameter determines the freqeuency smoothness reference distance in units of km, with the smoothness scaling with distance. See DP3 documentation. The default is [0.,0.,0.].")
    calibrationparser.add_argument("--antennaconstraint-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="List with constraints on the antennas used (in same order as soltype-list input). Possible input: 'superterp', 'coreandfirstremotes', 'core', 'remote', 'distantremote', 'all', 'international', 'alldutch', 'core-remote', 'coreandallbutmostdistantremotes, alldutchandclosegerman', 'alldutchbutnoST001'. The default is [None,None,None].")
    calibrationparser.add_argument("--resetsols-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="Values of these stations will be rest to 0.0 (phases), or 1.0 (amplitudes), default None, possible settings are the same as for antennaconstraint-list (alldutch, core, etc)). The default is [None,None,None].")
    calibrationparser.add_argument("--resetdir-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="Values of these directions will be rest to 0.0 (phases), or 1.0 (amplitudes) for DDE solves. The default is [None,None,None]. It requires --facetdirections being set a user defined direction list so the directions are known. An example would be '[None,[1,4],None]', meaning that directions 1 and 4 are being reset, counting starts at zero in the second solve in the pertubation list.")
    calibrationparser.add_argument("--soltypecycles-list",
                                   type=arg_as_list,
                                   default=[0, 999, 3],
                                   help="Selfcalcycle where step from soltype-list starts. The default is [0,999,3].")

    calibrationparser.add_argument("--BLsmooth-list",
                                   type=arg_as_list,
                                   default=[False, False, False],
                                   help="Employ BLsmooth, this is a list of length soltype-list. For example --BLsmooth-list='[True, False,True]'. Default is all False.")
    calibrationparser.add_argument('--dejumpFR',
                                   help='Dejump Faraday solutions when using scalarphasediffFR.',
                                   action='store_true')
    calibrationparser.add_argument('--usemodeldataforsolints',
                                   help='Determine solints from MODEL_DATA.',
                                   action='store_true')
    calibrationparser.add_argument("--preapplyH5-list",
                                   type=arg_as_list,
                                   default=[None],
                                   help="Update April 2024: Avoid usage because of corrupt vs correct. List of H5 files to preapply (one for each MS). The default is [None].")
    calibrationparser.add_argument("--preapplybandpassH5-list",
                                   type=arg_as_list,
                                   default=[None],
                                   help="List of possible h5parm files to preapply. For each MS, the closest h5parm in time in the list will be the one that preapplied. Times do not have to overlap between the h5parm and the MS. WEIGHT_SPECTRUM will be updated based on the amplitude values (if present). It is assumed (and not checked) that there is 'perfect' frequency ovelap and the solutions are constant along the time-axis of the h5parm. Note that DATA will be overwritten via a correct step, and these h5parms are not merged in the merged output solutions files. A list of length 1 with a glob-like string containing * or ? is also allowed, e.g. ['mybandpass*.h5']")
    calibrationparser.add_argument('--normamps',
                                   help='Normalize global amplitudes to 1.0. The default is True (False if fulljones is used). Note that if set to False --normamps-list is ignored.',
                                   type=ast.literal_eval,
                                   default=True)
    calibrationparser.add_argument('--normampsskymodel',
                                   help='Normalize global amplitudes to 1.0 when solving against an external skymodel. The default is False (turned off if fulljones is used). Note that this parameter is False (the default) --normamps-list is ignored for the solve against the skymodel.',
                                   type=ast.literal_eval,
                                   default=False)
    calibrationparser.add_argument('--normamps-per-ms',
                                   help='Normalize amplitudes to 1.0 for each MS separately, by default this is not done',
                                   action='store_true')
    calibrationparser.add_argument("--normamps-list",
                                   type=arg_as_list,
                                   default=['normamps', 'normamps', 'normamps'],
                                   help="List with amplitude normalization options. Possible input: 'normamps', 'normslope', 'normamps_per_ant, 'normslope+normamps', 'normslope+normamps_per_ant', or None. The default is [normamps,normamps,normamps,etc]. Only has an effect if the corresponding soltype outputs and amplitude000 table (and is not fulljones).")

    # Expert settings
    calibrationparser.add_argument('--tecfactorsolint',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--gainfactorsolint',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--phasefactorsolint',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--compute-phasediffstat',
                                   help='Get phasediff statistics for long-baseline calibrator dataset (see de Jong et al. 2024)',
                                   action='store_true')
    calibrationparser.add_argument('--early-stopping',
                                   help='Automatic decision to perform early stopping during self-calibration based on image quality and solution stability. Currently only optimized for LOFAR HBA data for VLBI imaging.',
                                   action='store_true')
    calibrationparser.add_argument('--nn-model-cache',
                                   help='Cache storage for Neural Network model for early stopping. If not given, it will download the model. This needs to be given in combination with --early-stopping.',
                                   default='.cache/cortexchange')
    calibrationparser.add_argument('--QualityBasedWeights',
                                   help='Experts only.',
                                   action='store_true')
    calibrationparser.add_argument('--QualityBasedWeights-start',
                                   help='Experts only.',
                                   type=int,
                                   default=5)
    calibrationparser.add_argument('--QualityBasedWeights-dtime',
                                   help='QualityBasedWeights timestep in units of minutes (default 5)',
                                   type=float,
                                   default=5.0)
    calibrationparser.add_argument('--QualityBasedWeights-dfreq',
                                   help='QualityBasedWeights frequency in units of MHz (default 5)',
                                   type=float,
                                   default=5.0)
    calibrationparser.add_argument('--ncpu-max-DP3solve',
                                   help='Maximum number of threads for DP3 solves, default=64 (too high value can result in BLAS errors)',
                                   type=int,
                                   default=64)
    calibrationparser.add_argument('--DDE',
                                   help='Experts only.',
                                   action='store_true')
    calibrationparser.add_argument('--Nfacets',
                                   help='Number of directions to solve into when --DDE is used. Directions are found automatically. Only used if --facetdirections is not set. Keep to default (=0) if you want to use --targetFlux instead',
                                   type=int,
                                   default=0)
    calibrationparser.add_argument('--targetFlux',
                                   help='targetFlux in Jy for groupalgorithm to create facet directions when --DDE is set (default = 2.0). Directions are found automatically. Only used if --facetdirections is not set. Ignored when --NFacets is set to > 0',
                                   type=float,
                                   default=2.0)
    calibrationparser.add_argument('--facetdirections',
                                   help='Experts only. ASCII csv file containing facet directions. File needs two columns with decimal degree RA and Dec. Default is None.',
                                   type=str,
                                   default=None)
    calibrationparser.add_argument('--DDE-predict',
                                   help='Type of DDE predict to use. Options: DP3 or WSCLEAN, default=WSCLEAN (note: option WSCLEAN will use a lot of disk space as there is one MODEL column per direction written to the MS)',
                                   type=str,
                                   default='WSCLEAN')
    calibrationparser.add_argument('--disable-primary-beam',
                                   help='For WSCLEAN imaging and predicts disable the primary beam corrections (so run with "apparent" images only)',
                                   action='store_true')

    blsmoothparser = parser.add_argument_group("-------------------------BLSmooth Settings-------------------------")
    # BLsmooth settings
    blsmoothparser.add_argument("--iontimefactor",
                                help='BLsmooth ionfactor. The default is 0.01. Larger is more smoothing (see BLsmooth documentation).',
                                type=float,
                                default=0.01)
    blsmoothparser.add_argument("--ionfreqfactor",
                                help='BLsmooth tecfactor. The default is 1.0. Larger is more smoothing (see BLsmooth documentation).',
                                type=float,
                                default=1.0)
    blsmoothparser.add_argument("--blscalefactor",
                                help='BLsmooth blscalefactor. The default is 1.0 (see BLsmooth documentation).',
                                type=float,
                                default=1.0)
    blsmoothparser.add_argument('--blsmooth_chunking_size',
                                type=int,
                                help='Chunking size for blsmooth. Larger values are slower but save on memory, lower values are faster. The default is 8.',
                                default=8)

    flaggingparser = parser.add_argument_group("-------------------------Flagging Settings-------------------------")
    # Flagging options
    flaggingparser.add_argument('--doflagging',
                                help='Flag on complexgain solutions via rms outlier detection (True/False, default=True). The default is True (will be set to False if --forwidefield is set).',
                                type=ast.literal_eval,
                                default=True)
    flaggingparser.add_argument('--clipsolutions',
                                help='Flag amplitude solutions above --clipsolhigh and below  --clipsollow (will be set to False if --forwidefield is set).',
                                action='store_true')
    flaggingparser.add_argument('--clipsolhigh',
                                help='Flag amplitude solutions above this value, only done if --clipsolutions is set.',
                                default=1.5,
                                type=float)
    flaggingparser.add_argument('--clipsollow',
                                help='Flag amplitude solutions below this value, only done if --clipsolutions is set.',
                                default=0.667,
                                type=float)
    flaggingparser.add_argument('--restoreflags',
                                help='Restore flagging column after each selfcal cycle, only relevant if --doflagging=True.',
                                action='store_true')
    flaggingparser.add_argument('--remove-flagged-from-startend',
                                help='Remove flagged time slots at the start and end of an observations. Do not use if you want to combine DD solutions later for widefield imaging.',
                                action='store_true')
    flaggingparser.add_argument('--flagslowamprms',
                                help='RMS outlier value to flag on slow amplitudes. The default is 7.0.',
                                default=7.0,
                                type=float)
    flaggingparser.add_argument('--flagslowphaserms',
                                help='RMS outlier value to flag on slow phases. The default 7.0.',
                                default=7.0,
                                type=float)
    flaggingparser.add_argument('--doflagslowphases',
                                help='If solution flagging is done also flag outliers phases in the slow phase solutions. The default is True.',
                                type=ast.literal_eval,
                                default=True)
    flaggingparser.add_argument('--useaoflagger',
                                help='Run AOflagger on input data.',
                                action='store_true')
    flaggingparser.add_argument('--aoflagger-strategy',
                                help='Use this strategy for AOflagger (options are: "default_StokesV.lua", "LBAdefaultwideband.lua", "default_StokesQUV.lua")',
                                default=None,
                                type=str)
    flaggingparser.add_argument('--useaoflaggerbeforeavg',
                                help='Flag with AOflagger before (True) or after averaging (False). The default is True.',
                                type=ast.literal_eval,
                                default=True)
    flaggingparser.add_argument('--flagtimesmeared',
                                help='Flag data that is severely time smeared. Warning: expert only',
                                action='store_true')
    flaggingparser.add_argument('--removeinternational',
                                help='Remove the international stations if present',
                                action='store_true')
    flaggingparser.add_argument('--removemostlyflaggedstations',
                                help='Remove the staions that have a flaging percentage above 85 percent',
                                action='store_true')

    startmodelparser = parser.add_argument_group(
        "-------------------------Starting model Settings-------------------------")
    # Startmodel
    startmodelparser.add_argument('--skymodel',
                                  help='Skymodel for first selfcalcycle. The default is None.',
                                  type=arg_as_str_or_list)
    startmodelparser.add_argument('--skymodelsource',
                                  help='Source name in skymodel. The default is None (means the skymodel only contains one source/patch).',
                                  type=str)
    startmodelparser.add_argument('--skymodelpointsource',
                                  help='If set, start from a point source in the phase center with the flux density given by this parameter. The default is None (means do not use this option).',
                                  type=arg_as_float_or_list,
                                  default=None)
    startmodelparser.add_argument('--wscleanskymodel',
                                  help='WSclean basename for model images (for a WSClean predict). The default is None.',
                                  type=arg_as_str_or_list,
                                  default=None)
    startmodelparser.add_argument('--fix-model-frequencies',
                                  help='Force predict and imaging wsclean commands to divide on freqencies set by wsclean skymodel',
                                  action='store_true')
    startmodelparser.add_argument('--predictskywithbeam',
                                  help='Predict the skymodel with the beam array factor.',
                                  action='store_true')
    startmodelparser.add_argument('--startfromtgss',
                                  help='Start from TGSS skymodel for positions (boxfile required).',
                                  action='store_true')
    startmodelparser.add_argument('--startfromvlass',
                                  help='Start from VLASS skymodel for ILT phase-up core data.',
                                  action='store_true')
    startmodelparser.add_argument('--startfromgsm',
                                  help='Start from LINC GSM skymodel.',
                                  action='store_true')
    startmodelparser.add_argument('--tgssfitsimage',
                                  help='Start TGSS fits image for model (if not provided use SkyView). The default is None.',
                                  type=str)
    startmodelparser.add_argument('--startfromimage',
                                  help='Start FITS image for model. The default is None.',
                                  action='store_true')


    # General options
    parser.add_argument('-b', '--boxfile',
                        help='DS9 box file. You need to provide a boxfile to use --startfromtgss. The default is None.',
                        type=str)
    parser.add_argument('--beamcor',
                        help='Correct the visibilities for beam in the phase center, options: yes, no, auto (default is auto, auto means beam is taken out in the curent phase center, tolerance for that is 10 arcsec)',
                        type=str,
                        default='auto')
    parser.add_argument('--losotobeamcor-beamlib',
                        help="Beam library to use when not using DP3 for the beam correction. Possible input: 'stationresponse', 'lofarbeam' (identical and deprecated). The default is 'stationresponse'.",
                        type=str,
                        default='stationresponse')
    parser.add_argument('--docircular',
                        help='Convert linear to circular correlations.',
                        action='store_true')
    parser.add_argument('--dolinear',
                        help='Convert circular to linear correlations.',
                        action='store_true')
    parser.add_argument('--forwidefield',
                        help='Keep solutions such that they can be used for widefield imaging/screens.',
                        action='store_true')
    parser.add_argument('--remove-outside-center',
                        help='Subtract sources that are outside the central parts of the FoV, square box is used in the phase center with sizes of 3.0, 2.0, 1.5 degr for MeerKAT UHF, L, and S-band, repspectively. In case you want something else set --remove-outside-center-box. In case of a --DDE solve the solution closest to the box center is applied.',
                        action='store_true')
    parser.add_argument('--remove-outside-center-box',
                        help='User defined box DS9 region file to subtract sources that are outside this part of the image, see also --remove-outside-center. If "keepall" is set then no subtract is done and everything is kept, this is mainly useful if you are already working on box-extracted data. If number is given a boxsize of this size (degr) will be used in the phase center. In case of a --DDE solve the solution closest to the box center is applied (unless "keepall" is set).',
                        type=str_or_float,
                        default=None)
    parser.add_argument('--single-dual-speedup',
                        help='Speed up calibration and imaging if possible using datause=single/dual in DP3 and -scalar/diagonal-visibilities in WSClean. Requires a recent (mid July 2024) DP3 and WSClean versions. Default is True. Set to --single-dual-speedup=False to disable to speed-up',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument('--dysco',
                        help='Use Dysco data compression. The default is True.',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument('--modelstoragemanager',
                        help='String input option for the compression of MODEL_DATA. The default is stokes_i. This will be turned off (set to None) automatically if the solve types do not allow for stokes_i compression. Set to None if you want to turn off MODEL_DATA compression entirely.',
                        type=str,
                        default='stokes_i')
    parser.add_argument('--resetweights',
                        help='If you want to ignore weight_spectrum_solve.',
                        action='store_true')
    parser.add_argument('--start',
                        help='Start selfcal cycle at this iteration number. The default is 0.',
                        default=0,
                        type=int)
    parser.add_argument('--stop',
                        help='Stop selfcal cycle at this iteration number. The default is 10.',
                        default=10,
                        type=int)
    parser.add_argument('--stopafterskysolve',
                        help='Stop calibration after solving against external skymodel.',
                        action='store_true')
    parser.add_argument('--stopafterpreapply',
                        help='Stop after preapply of solutions',
                        action='store_true')
    parser.add_argument('--bandpassMeerKAT',
                        help='Stop calibration after solving against an external skymodel and compute bandpass on merged solution file. Requires an external skymodel to be provided.',
                        action='store_true')
    parser.add_argument('--noarchive',
                        help='Do not archive the data.',
                        action='store_true')
    parser.add_argument('--skipbackup',
                        help='Leave the original MS intact and work always work on a DP3 copied dataset.',
                        action='store_true')
    parser.add_argument('--keepmodelcolumns',
                        help='Leave the MODEL_DATA-type columns in the MS. By default these are removed to save disk space.',
                        action='store_true')
    parser.add_argument('--phasediff_only',
                        help='For finding only the phase difference, we want to stop after calibrating and before imaging',
                        action='store_true')
    parser.add_argument('--helperscriptspath',
                        help='Path to file location pulled from https://github.com/rvweeren/lofar_facet_selfcal.',
                        default='/net/rijn/data2/rvweeren/LoTSS_ClusterCAL/',
                        type=str)
    parser.add_argument('--helperscriptspathh5merge',
                        help='Path to file location pulled from https://github.com/jurjen93/lofar_helpers.',
                        default=None,
                        type=str)
    parser.add_argument('--configpath',
                        help='Path to user config file which will overwrite command line arguments',
                        default='facetselfcal_config.txt',
                        type=str)
    parser.add_argument('--auto',
                        help='Trigger fully automated processing (HBA only for now).',
                        action='store_true')
    parser.add_argument('--delaycal',
                        help='Trigger settings suitable for ILT delay calibration, HBA-ILT only - still under construction.',
                        action='store_true')
    parser.add_argument('--targetcalILT',
                        help="Type of automated target calibration for HBA international baseline data when --auto is used. Options are: 'tec', 'tecandphase', 'scalarphase'. The default is 'scalarphase'.",
                        default='scalarphase',
                        type=str)
    parser.add_argument('--stack',
                        help='Stacking of visibility data for multiple sources to increase S/N. Solve on stacked MSs that have the same time axis and imaging all MSs with the same phase center together - still under construction.',
                        action='store_true')
    parser.add_argument('--testing',
                        help='Skip MS validation for code testing.',
                        action='store_true')
    parser.add_argument('ms', nargs='+', help='msfile(s)')

    return parser.parse_args()


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def arg_as_str_or_list(s):
    if "[" not in s and "]" not in s:
        return str(s)

    v = ast.literal_eval(s)
    if isinstance(v, list):
        return v

    raise argparse.ArgumentTypeError(f'Argument "{s}" is not a string or list')


def arg_as_float_or_list(s):
    try:
        return float(s)
    except ValueError:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        raise argparse.ArgumentTypeError(f'Argument "{s}" is not a float or list')


def str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if isinstance(arg, str):
        return arg
    raise argparse.ArgumentTypeError("Input must be an int or string")


def str_or_float(arg):
    try:
        return float(arg)  # try convert to float
    except ValueError:
        pass
    if isinstance(arg, str):
        return arg
    raise argparse.ArgumentTypeError("Input must be an float or string")


def floatlist_or_float(argin):
    if argin is None:
        return argin
    try:
        return float(argin)  # try convert to float
    except ValueError:
        pass

    arg = ast.literal_eval(argin)

    if type(arg) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (argin))

    # check for float list
    if all([(isinstance(item, float) or isinstance(item, int)) for item in arg]):
        return arg
    else:
        raise argparse.ArgumentTypeError("This needs to be a float or list of floats")


def check_strlist_or_intlist(argin):
    """ Check if the argument is a list of integers or a list of strings with correct formatting.

    Args:
        argin (str): input string to check.
    Returns:
        arg (list): properly formatted list extracted from the output.
    """

    # check if input is a list and make proper list format
    arg = ast.literal_eval(argin)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (argin))

    # check for integer list
    if all([isinstance(item, int) for item in arg]):
        if np.min(arg) < 1:
            raise argparse.ArgumentTypeError("solint_list cannot contain values smaller than 1")
        else:
            return arg
    # so not an integer list, so now check for string list
    if all([isinstance(item, str) for item in arg]):
        # check if string contains numbers
        for item2 in arg:
            # print(item2)
            if not any([ch.isdigit() for ch in item2]):
                raise argparse.ArgumentTypeError("solint_list needs to contain some number characters, not only units")
            # check in the number in there is smaller than 1
            # print(re.findall(r'[+-]?\d+(?:\.\d+)?',item2)[0])
            if float(re.findall(r'[+-]?\d+(?:\.\d+)?', item2)[0]) <= 0.0:
                raise argparse.ArgumentTypeError("numbers in solint_list cannot be smaller than zero")
            # check if string contains proper time formatting
            if ('hr' in item2) or ('min' in item2) or ('sec' in item2) or ('h' in item2) or ('m' in item2) or (
                    's' in item2) or ('hour' in item2) or ('minute' in item2) or ('second' in item2):
                pass
            else:
                raise argparse.ArgumentTypeError(
                    "solint_list needs to have proper time formatting (h(r), m(in), s(ec))")
        return arg
    else:
        raise argparse.ArgumentTypeError(
            "solint_list must be a list of positive integers or a list of properly formatted strings")
