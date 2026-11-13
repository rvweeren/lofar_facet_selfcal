"""Command-line argument definitions and value conversion helpers."""

import ast
import argparse
import numpy as np
import re


def option_parser():
    """
    Create the command-line parser and parse the supplied arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
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
                               help='Image size, required if boxfile is not used. The default is None. If a text file is provided it should contain a single integer which is the image size to use.',
                               type=str_or_int)
    imagingparser.add_argument('-n', '--niter',
                               help='Number of iterations. This is computed automatically if None.',
                               default=None,
                               type=int)
    imagingparser.add_argument('--maskthreshold',
                               help="Mask noise thresholds used from image 1 to N made by MakeMask.py/breizorro. This is by default [5.0,4.5,4.5,4.5,4.0].",
                               default=[5.0, 4.5, 4.5, 4.5, 4.0],
                               type=arg_as_list)
    imagingparser.add_argument('--automask-threshold',
                               help="WSClean automask thresholds for each selfcal cycle. This is by default [2.5, 2.5, 2.5, 2.5, 2.5, ...]. If maskthreshold is lower than this value for a selfcal cycle, the maskthreshold value is used instead. A value of <0.5 turns off automasking.",
                               default=[2.5, 2.5, 2.5, 2.5, 2.5],
                               type=arg_as_list)
    add_bool_arg(imagingparser, 'mask-extended', help='Add an extra masking step for very extended emission. For LOFAR-NL and MeerKAT only. Image size needs to be at least 1600x1600 pixels', default=None)                           
    imagingparser.add_argument('--mask-extended-start',
                               help='Start --mask-extended emission at this selfcal cycle. Default 2.',
                               default=2,
                               type=int)
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
    imagingparser.add_argument('--DS9cleanmaskregionfile-exclude',
                               help='A DS9 region file (with WCS coordinates) that will exclude this region end up in the clean mask (i.e., to ensure that artefacts are not cleaned). Only works if the cleaning is employing a fitsmask, by default this is the case as we use breizorro to make masks automatically.',
                               type=str)                           
    imagingparser.add_argument('--robust',
                               help='Briggs robust parameter for imaging. The default is -0.5. Also allowed are the strings uniform or natural which will override Briggs weighting.',
                               default=None,
                               type=str_or_float)
    imagingparser.add_argument('--multiscale-start',
                               help='Start multiscale deconvolution at this selfcal cycle. This is by default 1.',
                               default=1,
                               type=int)
    imagingparser.add_argument('--uvminim',
                               help='Inner uv-cut for imaging in lambda. The default is 80 for LOFAR and 10 for all other telescopes.',
                               default=None,
                               type=floatlist_or_float)
    imagingparser.add_argument('--uvmaxim',
                               help='Outer uv-cut for imaging in lambda. The default is None',
                               default=None,
                               type=floatlist_or_float)
    imagingparser.add_argument('--pixelscale', '--pixelsize',
                               help='Pixel size in arcsec. Typically, 3.0 for LBA and 1.5 for HBA for the Dutch stations \
                               (these are also the default values). For LOFAR ILT the defaults are 0.04 and 0.08 for \
                               HBA and LBA, respectively. For MeerKAT the defaults are 1.8, 1.0, and 0.5 for UHF, L, \
                               and S-band, respectively. For ASKAP the defaults are 1.5 and 1.0 for the lower and upper \
                               parts of the band, respectively. For the GMRT the defaults are 3, 1.25, 0.75, 0.35, for \
                               band2, 3, 4, and 5, respectively.',
                               type=float)
    imagingparser.add_argument('--channelsout',
                               help='Number of channels out during imaging (see WSClean documentation). \
                               This is by default 6 for LOFAR and 12 for MeerKAT UHF/L-band, and \
                               8 for MeerKAT S-band.',
                               default='auto',
                               type=str_or_int)
    imagingparser.add_argument('--mgain',
                               help='Deconvolution --mgain setting for WSCLean, see WSClean documentation. The default value is 0.75',
                               default=0.75,
                               type=float)
    imagingparser.add_argument('--nmiter',
                               help='Deconvolution --nmiter setting for WSCLean, see WSClean documentation. A list gives one value per self-calibration cycle, the last value is used for the later cycles (e.g. [5,5,12] limits the first two cycles to 5 major iterations). The default value is 12',
                               default=12,
                               type=int_or_intlist)    
    add_bool_arg(imagingparser, 'multiscale', help='Use multiscale deconvolution (see WSClean documentation; default=None).', default=None)                           
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
                               default='auto',
                               type=str_or_int)
    imagingparser.add_argument('--auto-ddpsfgrid',
                               help="Choose the WSClean -dd-psf-grid size automatically after each imaging cycle (off by default; --ddpsfgrid set by hand wins). A source at a distance dtheta from the centre of its PSF cell sees the uv coordinates shifted by w*dtheta (non-coplanar baselines), so its PSF differs by about 2 pi w_rms dtheta beam within the main lobe. The allowed error near the brightest source is (5/DR)^(1/3), because PSF errors are corrected iteratively (CLEAN major cycles). The grid size is the smallest odd number (at most 7) of cells of that size over the image; the parallel deconvolution sub-images are set to at most imsize/N, because WSClean only uses direction-dependent PSFs with parallel deconvolution.",
                               default=False,
                               action=argparse.BooleanOptionalAction)
    imagingparser.add_argument('--ddpsfgrid',
                               help="Value for option -dd-psf-grid with WSClean (integer, by default this value is None and the option is not used",
                               type=int, default=None)
    imagingparser.add_argument("--gapchanneldivision",
                               help='Use the -gap-channel-division option in wsclean imaging and predicts (default is not to use it)',
                               action='store_true')
    imagingparser.add_argument('--taperinnertukey',
                               help="Value for taper-inner-tukey in WSClean (see WSClean documentation), useful to suppress negative bowls when using --uvminim. Typically values between 1.5 and 4.0 give good results. The default is None.",
                               default=None,
                               type=float)
    imagingparser.add_argument('--tapertukey',
                               help="Width in lambda of an outer Tukey taper of the imaging weights (WSClean -taper-tukey). The weights roll off smoothly over this range up to --uvmaxim, which must also be set. Use it to avoid the sharp edge of the uv coverage, where the few outermost visibilities get a large weight with Briggs weighting and the clean model of a bright compact source is least accurate. For example --uvmaxim=45000 --tapertukey=15000 for MeerKAT L-band. The default is None.",
                               default=None,
                               type=float)
    imagingparser.add_argument('--tapergaussian',
                               help="FWHM in arcsec of a Gaussian taper of the imaging weights (WSClean -taper-gaussian). This lowers the weight of the long baselines and gives a lower resolution image with more sensitivity to extended emission. Increase --pixelscale to match the larger restoring beam (about 4 pixels per beam). Only used with WSClean. The default is None.",
                               default=None,
                               type=float)
    imagingparser.add_argument('--auto-tapertukey',
                               help="Switch on an outer Tukey taper of the imaging weights automatically (see --tapertukey). After each imaging cycle the dynamic range of the MFS image (peak over clipped rms) is checked. Once it is at least 50000 and the Briggs robust value is 0 or lower (or uniform), the following images use --uvmaxim set to the longest uv distance in the data and --tapertukey set to a third of that, and --niter is doubled. Once switched on, the taper stays on. It is not used if --uvmaxim or --tapertukey is set by hand, with --stack, or with DDFacet. The default is off.",
                               default=False,
                               action=argparse.BooleanOptionalAction)
    imagingparser.add_argument('--makeimage-ILTlowres-HBA',
                               help='Make 1.2 arcsec tapered image as quality check of ILT 1 arcsec imaging.',
                               action='store_true')
    imagingparser.add_argument('--makeimage-fullpol',
                               help='Make Stokes IQUV version for quality checking. Only done for the first (000) and last selfcal cycle (stop-1).',
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
    imagingparser.add_argument("--wgridderaccuracy",
                               help='Accuracy for wgridder in WSClean (see WSClean documentation). The default is 0.0001 which is a good value for high resolution imaging.',
                               type=float,
                               default=None)
    imagingparser.add_argument('--ncpu-max-WSClean',
                               help='Maximum number of threads for WSClean, default=None. None means WSClean can use all available threads, i.e., all CPU cores.',
                               type=int,
                               default=None)
    imagingparser.add_argument('--enlarge-safe-FoV-diameter',
                               help='Expert option to enlarge the safe FoV diameter for -apply-facet-beam/-apply-primary-beam. Default is 1.0 (no enlargement).',
                               type=float,
                               default=1.0)

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
                                   help="DP3 msin.ntimes setting. The default is None.",
                                   type=int,
                                   default=None)
    calibrationparser.add_argument('--msinstarttimeslot',
                                   help="DP3 msin.starttimeslot setting. The default is None.",
                                   type=int,
                                   default=None)                                   
    add_bool_arg(calibrationparser, 'autofrequencyaverage', help='Try frequency averaging if it does not result in bandwidth smearing', default=None)

    calibrationparser.add_argument('--phaseupstations',
                                   help="Phase up to a superstation. Possible input: 'core' or 'superterp'. The default is None.",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--refant',
                                   help="Reference antenna for all solves (phase referencing, resetting solutions and plots), for example 'ea10'. By default one reference antenna is chosen at the start of the run, the least flagged of the preferred (core) antennas present in all MSs (ST001 if --phaseupstations is set), and used for the whole run. A fixed reference antenna keeps the cross-hand (R-L, X-Y) phase consistent between solves, which matters for polarisation calibration. For MSs from different telescopes give one antenna per telescope, comma-separated (for example 'm000,ea10'); by default one is chosen per telescope. The default is None.",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--blcal-cycles',
                                   help="Selfcal cycles after whose (antenna-based) calibration a baseline-based calibration is done, for example [6]. Closure errors that antenna gains cannot absorb (for example from bandpass mismatches in old single-channel VLA continuum data) are solved as one complex factor per baseline and parallel hand (XX/YY or RR/LL; the cross hands get the geometric mean) against the model of that cycle, and divided into DATA and CORRECTED_DATA of the working MSs, so all later cycles use the corrected data. Only use it with a good model, it can absorb model errors. Not for --DDE, --idg or --stack. The solutions are in blcal/, the plots in solution_plots_<ms>/. The default is None (no baseline-based calibration).",
                                   default=None,
                                   type=arg_as_list)
    calibrationparser.add_argument('--blcal-telescopes',
                                   help="Only do the baseline-based calibration (--blcal-cycles) for the MSs of these telescopes (TELESCOPE_NAME), for example ['VLA']. The default is None (all MSs).",
                                   default=None,
                                   type=arg_as_list)
    calibrationparser.add_argument('--blcal-solint',
                                   help="Solution interval of the baseline-based calibration, as a number of time slots or a time like '30min' or '2h'. The default is None (one solution for the whole observation, like AIPS BLCAL).",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--blcal-nchan',
                                   help="Channels per baseline-based solution. The default is 0 (one solution for all channels).",
                                   default=0,
                                   type=int)
    calibrationparser.add_argument('--blcal-minsnr',
                                   help="Baseline-based factors B with |B-1| below this many times their formal error are not applied (set to 1). The default is 3.0.",
                                   default=3.0,
                                   type=float)
    calibrationparser.add_argument('--blcal-maxdev',
                                   help="Baseline-based factors with |B-1| larger than this are not applied (likely bad data or a model error rather than a closure error). The default is 0.1.",
                                   default=0.1,
                                   type=float)
    calibrationparser.add_argument('--phaseshiftbox',
                                   help="DS9 region file to shift the phasecenter to. This is by default None.",
                                   default=None,
                                   type=str)
    calibrationparser.add_argument('--weightspectrum-clipvalue',
                                   help="Extra option to clip WEIGHT_SPECTRUM values above the provided number. Use with care and test first manually to see what is a fitting value. The default is None.",
                                   type=float,
                                   default=None)
    calibrationparser.add_argument('--data-clipvalue',
                                   help="Extra option to flag DATA values above the provided number (this is done by setting WEIGHT_SPECTRUM to 0). Use with care and test first manually to see what is a fitting value. The default is None.",
                                   type=float,
                                   default=None)                                   
    calibrationparser.add_argument('-u', '--uvmin',
                                   help="Inner uv-cut for calibration in lambda. The default is 80 for LBA, 350 for HBA, and 250 for MWA.",
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
    add_bool_arg(calibrationparser, 'update-uvmin', help='Update uvmin automatically when extended emission is found (Dutch LOFAR array and MWA).', default=None)
    add_bool_arg(calibrationparser, 'update-multiscale', help='Switch to multiscale automatically if large islands of emission are present.', default=None)
    calibrationparser.add_argument("--soltype-list",
                                   type=arg_as_list,
                                   default=['tecandphase', 'tecandphase', 'scalarcomplexgain'],
                                   help="List with solution types. Possible input: 'complexgain', 'scalarcomplexgain', 'scalaramplitude', 'amplitudeonly', 'phaseonly', 'fulljones', 'rotation', 'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude', 'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase', 'faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase' , 'tec', 'tecandphase', 'scalarphase', 'scalarphasediff', 'scalarphasediffFR', 'phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin', 'scalarphase_phmin', 'scalarphase_slope', 'phaseonly_slope', 'leakage', 'leakageamplitude', 'tec+phase', 'tec+delay', 'tec+phase+delay', 'treemf' (tec+phase+delay solved with the Python treeMF solver instead of DP3, see the --treemf-* options), 'treemfphasediff' (in place of scalarphasediff: a per-station rotation measure fitted to RR-LL with the Python treeMF solver, see the --treemfphasediff-* options). Note: treemf and treemfphasediff are experimental, meant for expert testing, and will likely be replaced in a future version. The default is [tecandphase,tecandphase,scalarcomplexgain].")
    calibrationparser.add_argument("--solint-list",
                                   type=check_strlist_or_intlist,
                                   default=[1, 1, 120],
                                   help="Solution interval corresponding to solution types (in same order as soltype-list input). The default is [1,1,120].")
    calibrationparser.add_argument("--nchan-list",
                                   type=arg_as_list,
                                   default=[1, 1, 1],
                                   help="Number of channels corresponding to solution types (in same order as soltype-list input). The default is [1,1,1].")
    calibrationparser.add_argument("--solve-msinnchan-list",
                                   type=arg_as_list,
                                   default=['all', 'all', 'all'],
                                   help="Number of channels selected for the solve via msin.nchan. By default this is ['all','all','all'] which means all channels are used in the solve. If this is not 'all' the solve step will always run with ddecal.nchan=0, which means there will be no frequency dependence. Solution will be applied to the full channel range of the MS. Example is [50,'all',150]")
    calibrationparser.add_argument("--solve-msinstartchan-list",
                                   type=arg_as_list,
                                   default=[0, 0, 0],
                                   help="Start channel selected for the solve via msin.startchan. Only relevant is --solve-msinnchan-list is set. By default --solve-msinstartchan-list is [0,0,0] which means msin.startchan=0 in the solve.")    
    calibrationparser.add_argument("--smoothnessconstraint-list",
                                   type=arg_as_list,
                                   default=[0., 0., 5.],
                                   help="List with frequency smoothness values in MHz (in same order as soltype-list input). Must be 0 for the tec/delay-type and faradayrotation soltypes, whose fitted frequency model is already smooth. The default is [0.,0.,5.].")
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
                                   help="If smoothnessrefdistance is not equal to zero then this parameter determines the frequency smoothness reference distance in units of km, with the smoothness scaling with distance. See DP3 documentation. The default is a list of zeros")
    
    calibrationparser.add_argument("--antenna-averaging-factors-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="Averaging factors per antenna to allow antenna-dependent solution intervals. One list entry per soltype. A list entry can contain multiple antenna groups. Example is ['core:6,remote:3', None,'closeinternational:10']. The default is [None,None,None]. Groups are superterp, corebutsuperterp, firstremotes,midremotes,distantremotes,closeinternational,distantinternational, superstation")   
    calibrationparser.add_argument("--antenna-smoothness-factors-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="A list of extra smoothing factors per antenna group in the same format as antenna-averaging-factors-list but these value can be float, e.g. ['core:1,remote:0.5', None,'closeinternational:0.1']. Values need to be less or equal than 1 and cause less smoothing. The default is [None,None,None]. Groups are superterp, corebutsuperterp, firstremotes,midremotes,distantremotes,closeinternational,distantinternational, superstation") 
    calibrationparser.add_argument("--antennaconstraint-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="List with constraints on the antennas used (in same order as soltype-list input). Possible input: 'superterp', 'coreandfirstremotes', 'core', 'remote', 'distantremote', 'all', 'international', 'alldutch', 'core-remote', 'coreandallbutmostdistantremotes, alldutchandclosegerman', 'alldutchbutnoST001'. The default is [None,None,None].")
    calibrationparser.add_argument("--resetsols-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="Values of these stations will be reset to 0.0 (phases), or 1.0 (amplitudes), default None, possible settings are the same as for antennaconstraint-list (alldutch, core, etc.). The default is [None,None,None].")
    calibrationparser.add_argument("--resetdir-list",
                                   type=arg_as_list,
                                   default=[None, None, None],
                                   help="Values of these directions will be reset to 0.0 (phases), or 1.0 (amplitudes) for DDE solves. The default is [None,None,None]. It requires --facetdirections to be set to a user-defined direction list so the directions are known. An example would be '[None,[1,4],None]', meaning that directions 1 and 4 are being reset; counting starts at zero in the second solve in the perturbation list.")
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
                                   help="List of possible h5parm files to preapply. For each MS, the closest h5parm in time in the list will be the one that is preapplied. Times do not have to overlap between the h5parm and the MS. It is assumed (and not checked) that there is 'perfect' frequency overlap and the solutions are constant along the time axis of the h5parm. Note that DATA will be overwritten via a correct step, and these h5parms are not merged in the merged output solution files. A list of length 1 with a glob-like string containing * or ? is also allowed, e.g. ['mybandpass*.h5']")
    calibrationparser.add_argument('--preapplybandpassH5-updateweights',
                                   help='This determines whether WEIGHT_SPECTRUM will be updated based on the amplitude values when --preapplybandpassH5-list is set. By default this is True.',type=ast.literal_eval,
                                   default=True)
    calibrationparser.add_argument('--normamps',
                                   help='Normalize global amplitudes to 1.0. The default is True (False if fulljones/leakage/leakageamplitude is used). Note that if set to False --normamps-list is ignored.',
                                   type=ast.literal_eval,
                                   default=True)
    calibrationparser.add_argument('--normampsskymodel',
                                   help='Normalize global amplitudes to 1.0 when solving against an external skymodel. The default is False (turned off if fulljones/leakage/leakageamplitude is used). Note that this parameter is False (the default) --normamps-list is ignored for the solve against the skymodel.',
                                   type=ast.literal_eval,
                                   default=False)
    calibrationparser.add_argument('--normamps-per-ms',
                                   help='Normalize amplitudes to 1.0 for each MS separately, by default this is not done',
                                   action='store_true')
    calibrationparser.add_argument("--normamps-list",
                                   type=arg_as_list,
                                   default=['normamps', 'normamps', 'normamps'],
                                   help="List with amplitude normalization options. Possible input: 'normamps', 'normslope', 'normamps_per_ant, 'normslope+normamps', 'normslope+normamps_per_ant', or None. The default is [normamps,normamps,normamps,etc]. Only has an effect if the corresponding soltype outputs and amplitude000 table (and is not fulljones/leakage/leakageamplitude).")
    calibrationparser.add_argument('--startfrominitialsolutions',
                                   help='Use the solutions from the previous selfcalcycle in DP3 as initial solutions for the solve (if possible). This could speed up convergence in the solve.',
                                   action='store_true') 

    # Expert settings
    calibrationparser.add_argument('--tecfactorsolint',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--gainfactorsolint',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--gainfactorsmoothness',
                                   help='Experts only.',
                                   type=float,
                                   default=1.0)
    calibrationparser.add_argument('--ampresetvalfactor',
                                   help='Experts only.',
                                   type=float,
                                   default=10.)  
    calibrationparser.add_argument('--flag-ampresetvalfactor',
                                   help='Flag solutions outside this range. Experts only.',
                                   action='store_true')
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
    calibrationparser.add_argument('--stop-noise-increase-factor',
                                   help='Stop self-calibration when the noise of the image of a selfcal cycle is more than this factor above the lowest noise of the earlier cycles. Such an increase means the calibration diverged. Selfcal then stops before the solve (and any flagging) of that cycle, so the run can be restarted with --start from the best cycle. Set to 0 to switch the check off. The default is 5.0.',
                                   type=float,
                                   default=5.0)
    calibrationparser.add_argument('--nn-model-cache',
                                   help='Cache storage for Neural Network model for early stopping. This only works if --early-stopping.',
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
                                   help='Experts only. ASCII csv file containing facet directions. File needs at least two columns with decimal degree RA and Dec. Default is None.',
                                   type=arg_as_str_or_list,
                                   default=None)
    calibrationparser.add_argument('--DDE-predict',
                                   help='Type of DDE predict to use. Options: DP3 or WSCLEAN, default=WSCLEAN (note: option WSCLEAN will use a lot of disk space as there is one MODEL column per direction written to the MS)',
                                   type=str,
                                   default='WSCLEAN')
    calibrationparser.add_argument('--disable-primary-beam',
                                   help='For WSCLEAN imaging and predicts disable the primary beam corrections (so run with "apparent" images only)',
                                   action='store_true')
    calibrationparser.add_argument('--reduce-h5size',
                                   help='Save h5 file at the highest time and frequency grid based on the solints, instead of the time and frequency resolution of the MS (the default). This option is ignored for tec/tecandphase solves. Enabling this option can potentially reduce the accuracy of the solutions applied. For example if solint 3min and 4min are combined, the 4min solutions cannot be sampled perfectly onto a 3min solution grid.',
                                   action='store_true')
    calibrationparser.add_argument('--treemf-model',
                                   help="For soltype treemf (experimental, for expert testing; likely to be replaced in a future version): the phase model fitted per station. Parameters not in the model are fixed at zero. The default is 'tec+phase+delay'.",
                                   type=str,
                                   choices=['tec+phase+delay', 'tec+phase', 'tec+delay', 'tec', 'delay+phase', 'delay'],
                                   default='tec+phase+delay')
    calibrationparser.add_argument('--treemf-delaymax',
                                   help='For soltype treemf: bound on |delay| relative to the reference station in ns. The default is 20.',
                                   type=float,
                                   default=20.0)
    calibrationparser.add_argument('--treemf-delaysigma',
                                   help='For soltype treemf: width of an optional Gaussian delay prior in ns (in addition to --treemf-delaymax). The default is None (no Gaussian prior).',
                                   type=float,
                                   default=None)
    calibrationparser.add_argument('--treemf-tecmax',
                                   help='For soltype treemf: bound on |TEC| relative to the reference station in TECU (the TEC search range). The default is 4.',
                                   type=float,
                                   default=4.0)
    calibrationparser.add_argument('--treemf-tecprior',
                                   help="For soltype treemf: TEC prior. 'geometric' gives each station a zero-mean prior whose width grows with its ionospheric pierce-point distance from the reference station, 'none' uses only the --treemf-tecmax bound. The default is 'geometric'.",
                                   type=str,
                                   choices=['geometric', 'none'],
                                   default='geometric')
    calibrationparser.add_argument('--treemf-tecsig100',
                                   help='For soltype treemf with --treemf-tecprior geometric: prior width in TECU at 100 km pierce-point separation; it scales with separation^(5/6). The default is 0.25.',
                                   type=float,
                                   default=0.25)
    add_bool_arg(calibrationparser, 'treemf-propagate',
                 help='For soltype treemf: use each solution interval as the prior mean for the next one (with widths --treemf-propagate-tecsigma and --treemf-propagate-delaysigma). Without it every interval is solved independently.',
                 default=False)
    calibrationparser.add_argument('--treemf-propagate-tecsigma',
                                   help='For soltype treemf with --treemf-propagate: TEC prior width around the previous solution in TECU. The default is 0.1.',
                                   type=float,
                                   default=0.1)
    calibrationparser.add_argument('--treemf-propagate-delaysigma',
                                   help='For soltype treemf with --treemf-propagate: delay prior width around the previous solution in ns. The default is 5.',
                                   type=float,
                                   default=5.0)
    add_bool_arg(calibrationparser, 'treemf-tiecore',
                 help='For soltype treemf: solve all core stations (CS*) as one station; they share the clock and see the same ionosphere. The tied core is then the reference station. Core stations in an --antennaconstraint group stay in that group.',
                 default=True)
    calibrationparser.add_argument('--treemf-delaycentre',
                                   help="For soltype treemf: centre of each station's delay window (--treemf-delaymax applies around it). 'zero' centres the windows on zero delay; 'track' estimates a slowly varying delay per station from the data first (the delay likelihoods of all solution intervals stacked in --treemf-delaytrack-block minute blocks, with a test per station against shuffled data; stations that fail it keep zero), so that a narrow window can follow large and drifting residual clock delays; or an h5parm with a delay000 soltab (e.g. an earlier treemf solve) whose delays are used as centres. The default is 'zero'.",
                                   type=str,
                                   default='zero')
    calibrationparser.add_argument('--treemf-delaytrack-search',
                                   help='For soltype treemf with --treemf-delaycentre track: half-width in ns of the delay range searched for the track. The default is 200.',
                                   type=float,
                                   default=200.0)
    calibrationparser.add_argument('--treemf-delaytrack-tecwander',
                                   help='For soltype treemf with --treemf-delaycentre track: how far (TECU) the TEC of a solution interval may wander around that of its block when the delay likelihoods are stacked. The default is 0.1.',
                                   type=float,
                                   default=0.1)
    calibrationparser.add_argument('--treemf-delaytrack-block',
                                   help='For soltype treemf with --treemf-delaycentre track or --treemf-delayadapt: block length in minutes over which the delay is taken as constant. The default is 30.',
                                   type=float,
                                   default=30.0)
    calibrationparser.add_argument('--treemf-delaytrack-maxstep',
                                   help='For soltype treemf with --treemf-delaycentre track: largest change of the delay track between consecutive blocks in ns. The default is 15.',
                                   type=float,
                                   default=15.0)
    calibrationparser.add_argument('--treemf-delayadapt',
                                   help="For soltype treemf: number of rounds of the delay-bound check. Where a station's solutions pile up at one edge of its delay window within a block, the window there is moved by half of --treemf-delaymax towards the pile-up and that block is solved again. The default is 0 (only report pile-ups in the log).",
                                   type=int,
                                   default=0)
    calibrationparser.add_argument('--treemfphasediff-model',
                                   help="For soltype treemfphasediff (used in place of scalarphasediff; experimental, for expert testing; likely to be replaced in a future version): the model fitted per station to the RR-LL phase difference. 'rm' is a differential Faraday rotation, 2 RM lambda^2; 'rm+phase' adds a constant R-L phase offset (a residual frame rotation, e.g. without the full beam applied). The default is 'rm+phase'.",
                                   type=str,
                                   choices=['rm', 'rm+phase'],
                                   default='rm+phase')
    calibrationparser.add_argument('--treemfphasediff-rmmax',
                                   help='For soltype treemfphasediff: bound on |RM| relative to the reference station in rad/m^2 (the RM search range). The default is 5.',
                                   type=float,
                                   default=5.0)
    calibrationparser.add_argument('--treemfphasediff-rmprior',
                                   help="For soltype treemfphasediff: RM prior. 'none' uses only --treemfphasediff-rmmax; 'geometric' a zero-mean prior whose width grows with the pierce-point distance from the reference station (--treemfphasediff-rmsig100); 'rmextract' a prior mean from RMextract (GPS TEC maps and the geomagnetic field) with width --treemfphasediff-rmsigma, only for data that do not already have the RMextract Faraday-rotation correction applied (needs RMextract and IONEX downloads). The default is 'none'.",
                                   type=str,
                                   choices=['none', 'geometric', 'rmextract'],
                                   default='none')
    calibrationparser.add_argument('--treemfphasediff-rmsig100',
                                   help='For soltype treemfphasediff with --treemfphasediff-rmprior geometric: prior width in rad/m^2 at 100 km pierce-point separation; it scales with separation^(5/6). The default is 0.1.',
                                   type=float,
                                   default=0.1)
    calibrationparser.add_argument('--treemfphasediff-rmsigma',
                                   help='For soltype treemfphasediff with --treemfphasediff-rmprior rmextract: prior width around the RMextract value in rad/m^2. The default is 0.1.',
                                   type=float,
                                   default=0.1)
    calibrationparser.add_argument('--treemfphasediff-offsetsigma',
                                   help='For soltype treemfphasediff with model rm+phase: width in rad of a zero-mean Gaussian prior on the R-L phase offset. The default is None (no prior).',
                                   type=float,
                                   default=None)
    calibrationparser.add_argument('--treemfphasediff-amp',
                                   help="For soltype treemfphasediff: 'unit' uses the phase of RR conj(LL) only (as scalarphasediff does), 'product' keeps its amplitude as a weight. The default is 'unit'.",
                                   type=str,
                                   choices=['unit', 'product'],
                                   default='unit')
    add_bool_arg(calibrationparser, 'treemfphasediff-propagate',
                 help='For soltype treemfphasediff: use each solution interval as the prior mean for the next one (with width --treemfphasediff-propagate-rmsigma).',
                 default=False)
    calibrationparser.add_argument('--treemfphasediff-propagate-rmsigma',
                                   help='For soltype treemfphasediff with --treemfphasediff-propagate: RM prior width around the previous solution in rad/m^2. The default is 0.05.',
                                   type=float,
                                   default=0.05)
    add_bool_arg(calibrationparser, 'treemfphasediff-tiecore',
                 help='For soltype treemfphasediff: solve all core stations (CS*) as one station (they see the same ionosphere). Core stations in an --antennaconstraint group stay in that group.',
                 default=True)
    
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
                                help='Remove flagged time slots at the start and end of an observation. Do not use if you want to combine DD solutions later for widefield imaging.',
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
    flaggingparser.add_argument('--flagsolutionsbeforeresetsolsall',
                                help='In case all solutions are reset via --resetsols-list carry out rms outlier solution flagging \
                                      before resetting the solutions. This can be useful for DI solves to find bad data but without \
                                      ever applying the solutions. In this way, the visibility data corresponding to the bad \
                                      solutions will be flagged, but the solution themselves will be reset afterwards.',
                                action='store_true')                                
    flaggingparser.add_argument('--flag-antenna-list',
                                help='List of antenna names to flag',
                                type=ast.literal_eval)


    # AOflagger settings
    add_bool_arg(flaggingparser, 'aoflagger', help='Run AOflagger on input data.', default=None, second_name='useaoflagger')
    add_bool_arg(flaggingparser, 'aoflaggerbeforeavg',
                                help='Flag with AOflagger before averaging. The default is True.',
                                default=True)
    add_bool_arg(flaggingparser, 'aoflagger-correcteddata', default=None, second_name='useaoflagger-correcteddata',
                                help='Run AOflagger on the CORRECTED_DATA column after calibration.')
    add_bool_arg(flaggingparser, 'aoflagger-residualdata', default=None, second_name='useaoflagger-residualdata',
                                help='Run AOflagger on the RESIDUAL_DATA column.')
    add_bool_arg(flaggingparser, 'aoflagger-afterbandpassapply', default=None, second_name='useaoflagger-afterbandpassapply',
                                help='Run AOflagger on DATA column after preapply bandpass solution.')
    flaggingparser.add_argument('--aoflagger-strategy', 
                                help='Use this strategy for AOflagger on DATA, default=None (means flag without a custom strategy). For custom-made strategies see: https://github.com/rvweeren/lofar_facet_selfcal/tree/main/facetselfcal/flagging_strategies',
                                default=None, type=str)
    flaggingparser.add_argument('--aoflagger-strategy-correcteddata',
                                help='Use this strategy for AOflagger on CORRECTED_DATA, default=None (means flag without a custom strategy). For custom-made strategies see: https://github.com/rvweeren/lofar_facet_selfcal/tree/main/facetselfcal/flagging_strategies',type=str)
    flaggingparser.add_argument("--aoflagger-correcteddata-selfcalcycle-list",
                                type=arg_as_list, default=[1], help="Select list of  selfcalcycle where --aoflagger-correcteddata is used. Only values larger than 1 are allowed in the list. The default is [1].")
    flaggingparser.add_argument('--aoflagger-strategy-residualdata',
                                help='Use this strategy for AOflagger on RESIDUAL_DATA, default=None (means flag without a custom strategy). For custom-made strategies see: https://github.com/rvweeren/lofar_facet_selfcal/tree/main/facetselfcal/flagging_strategies',type=str)
    flaggingparser.add_argument("--aoflagger-residualdata-selfcalcycle-list",
                                type=arg_as_list, default=[2], help="Select list of  selfcalcycle where --aoflagger-residualdata is used. Only values larger than 1 are allowed in the list. The default is [2].")
    flaggingparser.add_argument('--aoflagger-strategy-afterbandpassapply',
                                help='Use this strategy for AOflagger after applying the bandpass solutions, default=None (means flag without a custom strategy). For custom-made strategies see: https://github.com/rvweeren/lofar_facet_selfcal/tree/main/facetselfcal/flagging_strategies',type=str)
    add_bool_arg(flaggingparser, 'auto-flag-antennas', default=None,
                                help='Automatically flag antennas that show deviating amplitude solutions (for MeerKAT only; default=None).') 
    add_bool_arg(flaggingparser, 'flagtimesmeared', default=False,
                                help='Flag data that is severely time smeared (default=False). Warning: expert only.')
    add_bool_arg(flaggingparser, 'removeinternational', default=False,
                                help='Remove the international stations if present (default=False).')
    add_bool_arg(flaggingparser, 'removemostlyflaggedstations', default=None,
                                   help='Remove the stations that have a flagging percentage above --removemostlyflaggedstations-percentage (default=None).')
    flaggingparser.add_argument('--removemostlyflaggedstations-percentage',
                                help='If --removemostlyflaggedstations is set this determines the percentage above which stations are removed, default=85', type=float, default=85)

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
    startmodelparser.add_argument('--skymodelsetjy',
                                  help='3C48, 3C147, 3C138, 3C286 skymodels following the Perley-Butler 2017 flux density scale, based on the VLA clean-component model images that CASA setjy uses. By default the model is predicted with DP3 (see --setjy-method). For 3C286 and 3C138 polarization information is also included: the polarization angle from Perley et al. 2026 (ApJS 283, 82), the polarization fraction of 3C286 from Hugo & Perley 2024 and of 3C138 from Perley et al. 2026. Skymodel is automatically selected based on the pointing center of the MS',
                                  action='store_true')
    startmodelparser.add_argument('--setjy-method',
                                  help='How --skymodelsetjy computes the model: "dp3" predicts the skymodels shipped in facetselfcal/data/setjy with DP3 (no CASA needed), "casa" runs CASA setjy. The default is "dp3".',
                                  type=str,
                                  choices=['dp3', 'casa'],
                                  default='dp3')
    startmodelparser.add_argument('--keepusingstartingskymodel',
                                  help='Keep using the initial skymodel for each selfcalcycle. Can be useful for bandpass calibration where some extra flagging rounds on CORRECTED_DATA are done. Relevant for --wscleanskymodel, --skymodelsetjy, --skymodel, --skymodelpointsource, --startfromtgss, --startfromgsm, --startfromimage, --startfromvlass', action='store_true')
    
    startmodelparser.add_argument('--fix-model-frequencies',
                                  help='Force predict and imaging WSClean commands to divide on frequencies set by the WSClean skymodel',
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
                        help='Correct the visibilities for the beam at the phase center; options: yes, no, or auto. This is a LOFAR-specific option. (Default is auto; auto means the LOFAR beam is taken out at the current phase center, with a tolerance of 10 arcsec.)',
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
                        help='Subtract sources that are outside the central parts of the FoV. A square box is used at the phase center with sizes of 3.0, 2.0, and 1.5 deg for MeerKAT UHF, L, and S-band, respectively. To use a different size, set --remove-outside-center-box. In case of a --DDE solve, the solution closest to the box center is applied.',
                        action='store_true')
    parser.add_argument('--remove-outside-center-box', '--remove-outside-center-region',
                        help='User defined DS9 region file to subtract sources that are outside \
                        this part of the image, see also --remove-outside-center. If "keepall" is \
                        set then no subtract is done and everything is kept, this is mainly useful \
                        if you are already working on extracted data. If number is given a \
                        boxsize of this size (degr) will be used in the phase center. The option \
                        "auto" will determine the boxsize automatically based on the artifacts \
                        detected in last selfcal image the image. In case of a --DDE solve the \
                        solution closest to the center is applied (unless of a --DDE solve the \
                        solution closest to the center is applied (unless "keepall" is set). Box, \
                        circle, and ellipse formats are supported.',
                        type=str_or_float,
                        default=None)
    parser.add_argument('--remove-outside-center-auto-minboxsize',
                        help='Required minimum box size (degrees) when automatically determining the \
                        box size for --remove-outside-center-box="auto". Default is the FWHM of the \
                        primary beam for MeerKAT',
                        type=float,
                        default=None)
    parser.add_argument('--remove-outside-center-avgtimestep',
                        help='Timestep for averaging the MS after an extraction step (default=1, which means no time averaging).',
                        type=int,
                        default=1)                        
    parser.add_argument('--remove-outside-center-avgfreqstep',
                        help='Frequency step for averaging the MS after an extraction step (default=1, which means no frequency averaging).',
                        type=int,
                        default=1)
    parser.add_argument('--remove-outside-center-noddcor',
                        help='Do not applycal DD solutions from closest direction to the box center (if applicable).',
                        action='store_true')                      
    parser.add_argument('--single-dual-speedup',
                        help='Speed up calibration and imaging if possible using datause=single/dual in DP3 and -scalar/diagonal-visibilities in WSClean. Requires a recent (mid July 2024) DP3 and WSClean versions. Default is True. Set to --single-dual-speedup=False to disable to speed-up',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument('--split-fieldname',
                        help='The FIELD name to split on. By default, the data are not split, but if you have multiple fields in your MS and want to split on those, you can set this to the name of the FIELD column entry that you want to split on. This is useful, for example, for MeerKAT L-band calibrator observations where there are often multiple fields (e.g. target, primary calibrator, and phase calibrator).',
                        type=str,
                        default=None)
    parser.add_argument('--dysco',
                        help='Use Dysco data compression. The default is True.',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument('--metadata-compression',
                        help='Use MS metadata compression (flags, uvw coordinates, and antenna table). The default is True. Will be turned of always for non-LOFAR data.',
                        type=ast.literal_eval,
                        default=True)
    parser.add_argument('--modelstoragemanager',
                        help='String input option for the compression of MODEL_DATA. The default is "auto". Other options are "sisco", "sisco_stokes_i", "sisco_diagonal" or "stokes_i"; option provided might be turned off automatically if the solve types do not allow for this type of compression. Set to "None" if you want to turn off MODEL_DATA compression entirely.',
                        type=str,
                        default='auto')
    parser.add_argument('--resetweights',
                        help='If you want to ignore weight_spectrum_solve.',
                        action='store_true')
    parser.add_argument('--start',
                        help='Start selfcal cycle at this iteration number. The default is 0.',
                        default=0,
                        type=int)
    parser.add_argument('--stop',
                        help='Stop selfcal cycle at this iteration number. If the value is not set, it defaults to 10. The exception is for MeerKAT data where it defaults to 2, or 6 in case of a --DDE solve.',
                        default=None,
                        type=int)
    parser.add_argument('--stopafterskysolve',
                        help='Stop calibration after solving against external skymodel.',
                        action='store_true')
    parser.add_argument('--stopafterpreapply',
                        help='Stop after preapply of solutions',
                        action='store_true')
    parser.add_argument('--bandpass',
                        help='Compute bandpass for MeerKAT or uGMRT primary calibrator observation. Stop calibration after solving against an external skymodel and then compute bandpass on merged solution file. Requires an external skymodel to be provided, or automatic selection via --skymodelsetjy (see the help for this option). For MeerKAT if no skymodel is provided or skymodelsetjy was not invoked, it will automatically select the ones for J0408-6545 and J1939-6342 provided by facetselfcal (both for UHF and L-band).',
                        action='store_true')
    parser.add_argument('--bandpass-stop',
                        help='Selfcalcycle after which to stop if --bandpass is used (default 0)', default=0, type=int)

    meerkatcalparser = parser.add_argument_group(
        "-------------------------MeerKAT calibrator-based calibration (1GC, --meerkatcal) Settings-------------------------")
    meerkatcalparser.add_argument('--meerkatcal',
                                  help='Calibrate MeerKAT data from the archive MS (all fields) without CASA: feed swap, flagging, delay (treeMF), bandpass, leakage and polarisation angle (cross-hand phase) calibration, transfer to the target(s), parallactic angle and (optionally) ionospheric Faraday rotation correction. Writes calibrated target MSs (sky frame) for a facetselfcal run. Give one archive MS with all fields (the field roles can be overridden with --meerkatcal-bpcal, --meerkatcal-phasecal, --meerkatcal-polcal and --meerkatcal-targets), or already split MSs: the calibrator MSs with --meerkatcal-bpcal-ms, --meerkatcal-phasecal-ms and --meerkatcal-polcal-ms and the target MSs as input.',
                                  action='store_true')
    meerkatcalparser.add_argument('--meerkatcal-level', default='auto', type=str,
                                  choices=['auto', 'standard', 'leakage', 'full'],
                                  help="Calibration level: 'standard' (delay, bandpass, gains), 'leakage' (also leakage from the unpolarised bandpass calibrator), 'full' (also the polarisation angle) or 'auto' (the highest level the observed calibrators allow). The default is 'auto'.")
    meerkatcalparser.add_argument('--meerkatcal-angle', default='auto', type=str,
                                  choices=['auto', 'celestial', 'diode', 'nullv'],
                                  help="Source of the polarisation angle (cross-hand phase) calibration for level 'full': 'celestial' (3C286 or 3C138), 'diode' (noise diode autocorrelations), 'nullv' (Stokes V = 0 on a polarised field), or 'auto' (celestial if a polarisation calibrator was observed, else diode if noise diode data is found). The default is 'auto'.")
    meerkatcalparser.add_argument('--meerkatcal-bpcal', default=None, type=arg_as_list,
                                  help='Bandpass/flux calibrator field name(s), for example "[\'J1939-6342\']". By default from the scan intents and calibrator positions.')
    meerkatcalparser.add_argument('--meerkatcal-phasecal', default=None, type=arg_as_list,
                                  help='Phase (gain) calibrator field name(s). By default from the scan intents.')
    meerkatcalparser.add_argument('--meerkatcal-polcal', default=None, type=arg_as_list,
                                  help='Polarisation calibrator field name(s) (3C286 or 3C138). By default from the scan intents and calibrator positions.')
    meerkatcalparser.add_argument('--meerkatcal-targets', default=None, type=arg_as_list,
                                  help='Target field name(s). By default all fields with a TARGET scan intent.')
    meerkatcalparser.add_argument('--meerkatcal-bpcal-ms', default=None, type=arg_as_list,
                                  help='For already split input instead of the archive MS: the MS(s) with the bandpass/flux calibrator (J1939-6342 or J0408-6545), for example "[\'J0408-6545.ms\']". Every field in these MSs is used as a bandpass calibrator. The target MSs are given as input, the other calibrators with --meerkatcal-phasecal-ms and --meerkatcal-polcal-ms. A field may be spread over several MSs (for example one MS per scan) and an MS may contain several fields of the same role. All MSs must be raw data (feeds not swapped) of one observation, with the same channels and antennas. Required for split input; the field name options (--meerkatcal-bpcal etc.) cannot be used with it.')
    meerkatcalparser.add_argument('--meerkatcal-phasecal-ms', default=None, type=arg_as_list,
                                  help='For already split input: the MS(s) with the phase (gain) calibrator(s), see --meerkatcal-bpcal-ms.')
    meerkatcalparser.add_argument('--meerkatcal-polcal-ms', default=None, type=arg_as_list,
                                  help='For already split input: the MS(s) with the polarisation calibrator (3C286 or 3C138, matched by position), see --meerkatcal-bpcal-ms.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-ionorm', default=False,
                 help='Correct ionospheric Faraday rotation (polarisation angle calibrator and targets) with spinifex, or with --meerkatcal-rmfile (giving that file switches the correction on). Off by default: in tests on MeerKAT data the time variation of the spinifex (IONEX map) RM over a night was not supported by the data and made the 3C286 polarisation angles worse. The default is False.')
    meerkatcalparser.add_argument('--meerkatcal-rmfile', default=None, type=str,
                                  help='Text file with the ionospheric rotation measure to use instead of spinifex (for example from ALBUS): columns MJD seconds, RM (rad/m^2), optionally a third column with the field name.')
    meerkatcalparser.add_argument('--meerkatcal-timeres', default=8., type=float,
                                  help='Time resolution (s) of the calibrated target MSs: the target scans are averaged in time by the integer factor closest to this resolution divided by the dump time (at least 1, so 8 s dumps are kept and 2 s dumps are averaged by 4). 0 keeps the native resolution. The default is 8.')
    meerkatcalparser.add_argument('--meerkatcal-nchan', default=2048, type=int,
                                  help='Maximum number of channels of the calibrated target MSs: the target scans are averaged in frequency by the smallest integer factor that gives at most this many channels (e.g. 32k and 4k to 2k). 0 keeps the native resolution. The default is 2048.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-target-exactweights', default=False,
                 help='Keep the weights of the calibrated target MSs exact (uncompressed, separate per correlation). By default the target data and weights are Dysco compressed as facetselfcal writes its MSs, which stores one weight for all correlations and makes the MSs about half the size. The calibrator MSs always keep exact weights (the polarisation solves need them). The default is False.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-target-concat', default=True,
                 help='Concatenate the calibrated scans of each target in time into one MS, targets/<target>.cal.ms (the per-scan MSs are removed). The MS has time gaps where other fields were observed; facetselfcal splits it at those gaps itself (see --timesplitbefore). With --no-meerkatcal-target-concat one MS per target scan is written instead, targets/<target>.scanNNN.cal.ms. A restart of a finished run in the same directory concatenates the per-scan MSs it wrote. The default is True.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-weightgain', default=False,
                 help='WARNING: normally not to be used. Divide out a time-variable gain that is the same for all antennas, channels and polarisations, measured per dump from the archive weights (WEIGHT_SPECTRUM, which follows the raw system gain): G(t) = sqrt(w_ref / w(t)), applied to the data (divided by G) and the weights (multiplied by G^2) of all calibrator and target scans before any calibration, with w(t) the median weight of a dump after dividing out the channel shape of the weights and w_ref a reference level for the whole observation (so the flux scale still comes from the bandpass calibrator). Only for data with an array-wide gain problem, i.e. target amplitudes that jump or ramp by the same factor on all baselines within and between scans, which the per-scan calibrator solutions cannot follow. In normal data the weights also follow the system temperature (elevation, a bright calibrator in the beam), so the correction then adds gain errors of a few percent, different for calibrators and targets. It must be given from the start of a run (the prepare stage). The default is False.')
    meerkatcalparser.add_argument('--meerkatcal-calnchan', default=4096, type=int,
                                  help='Maximum number of channels of the calibrator scans: after flagging at the native resolution, the scans are averaged in frequency by the smallest integer factor that gives at most this many channels (e.g. 32k to 4k or 2k). 0 keeps the native resolution. The default is 4096.')
    meerkatcalparser.add_argument('--meerkatcal-rfimask', default='sdp', type=str, choices=['sdp', 'caracal', 'none'],
                                  help="Static RFI mask flagged before AOFlagger on baselines shorter than 1 km (not on autocorrelations): 'sdp' (the MeerKAT SDP pipeline static mask: GSM 935-960 MHz, GNSS 1146-1301 MHz, 1519-1608 MHz), 'caracal' (the more aggressive caracal L-band mask, about 40%% of L-band on those baselines, adding the aviation lines at 1030-1141 MHz, 1462-1492 MHz and Iridium) or 'none'. The band edges and the band-specific ranges are flagged on all baselines regardless. The default is 'sdp'.")
    meerkatcalparser.add_argument('--meerkatcal-flagcycles', default=2, type=int,
                                  help='Number of solve and flag cycles on the calibrators (the bandpass calibrator delays are solved in the first cycle only). The default is 2.')
    meerkatcalparser.add_argument('--meerkatcal-edgefraction', default=0.05, type=float,
                                  help='Fraction of the band flagged at each edge. The default is 0.05.')
    meerkatcalparser.add_argument('--meerkatcal-leakage-bandwidth', default=0., type=float,
                                  help='Bandwidth (MHz) per leakage (D) solution, rounded to whole channels. 0 gives one solution per channel. The default is 0.')
    meerkatcalparser.add_argument('--meerkatcal-xf-bandwidth', default=10., type=float,
                                  help="Bandwidth (MHz) per cross-hand phase (Xf) solution, rounded to whole channels. The solutions are interpolated linearly in frequency when applied. 0 gives one solution per channel. The default is 10.")
    meerkatcalparser.add_argument('--meerkatcal-xf-minfrac', default=None, type=float,
                                  help='Minimum rms fraction of the model polarisation of the polarisation calibrator in the feed-frame cross-hands (|U|/P, all scans together) for a cross-hand phase solution. Below it the model angle is so close to a zero of the feed-frame Stokes U that a model angle or ionospheric RM error flips the solution by 180 degrees (90 degrees in the polarisation angle), however bright the calibrator, so those solutions are flagged (see --meerkatcal-xf-interpolate). By default it depends on frequency: sin(2 delta) with delta = 1 deg + 0.5 rad/m^2 lambda^2 (a model error plus an ionospheric RM error), e.g. 0.25 at 650 MHz, 0.14 at 925 MHz, 0.08 at 1.4 GHz, 0.05 at 2.5 GHz. A value gives a fixed fraction at all frequencies. The default is None (frequency dependent).')
    meerkatcalparser.add_argument('--meerkatcal-diode-times', default=None, type=str,
                                  help="Noise diode-on time ranges for the diode polarisation angle calibration (--meerkatcal-angle diode), instead of finding them in the autocorrelations of the calibrator scans: comma-separated 'start~end' in UTC (YYYY/MM/DD/HH:MM:SS, as CASA, or ISO YYYY-MM-DDTHH:MM:SS) or in MS TIME (MJD seconds), for all antennas. The time ranges can be read from the archive metadata with katdal (sensor <antenna>_dig_<band>_band_noise_diode). By default the diode-on dumps are found automatically.")
    meerkatcalparser.add_argument('--meerkatcal-diode-minjump', default=0.01, type=float,
                                  help='Minimum increase of the autocorrelation coherence |XY|/sqrt(XX YY) (median over the channels) of a dump above the diode-off level of its antenna for the automatic noise diode detection. The default is 0.01.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-diode-skyspline', default=True,
                 help='Correct the noise diode cross-hand phase to the sky with the per-band diode-to-sky spline of the MeerKAT SDP calibration pipeline (katsdpcal bcross_sky). Without it the cross-hand phase is that of the diode signal path. The default is True.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-xf-interpolate', default=True,
                 help='Replace flagged cross-hand phase solutions (not constrained by the model, or outliers from the smooth frequency dependence) by a linear interpolation in frequency of the good solutions on both sides, instead of flagging the data there. The default is True.')
    add_bool_arg(meerkatcalparser, 'meerkatcal-phasecal-selfcal', default=True,
                 help='Build the phase calibrator model with a short facetselfcal DI run. If False a point source at the phase centre is used. The default is True.')
    meerkatcalparser.add_argument('--meerkatcal-phasecal-nchan', default=1024, type=int,
                                  help='Frequency resolution of the data imaged by the facetselfcal run that builds the phase calibrator model (--meerkatcal-phasecal-selfcal), as a number of channels over the full nominal band (UHF 544 MHz, L 856 MHz, S 875 MHz): the scans are averaged in frequency to about that channel width, e.g. 4x for 4k data and 2x for the 2k archive product with 1024, also when the archive trimmed the band edges (a Stokes I model; a few times faster imaging than at 4k channels, with small bandwidth smearing). The gain solves use the scans at the calibrator resolution (--meerkatcal-calnchan). 0 keeps that resolution. The default is 1024.')
    meerkatcalparser.add_argument('--meerkatcal-stopafter', default=None, type=str,
                                  choices=['discover', 'prepare', 'bandpass', 'leakage', 'phasecal', 'polangle', 'polcheck', 'targets'],
                                  help="Stop after this stage: 'discover', 'prepare', 'bandpass', 'leakage', 'phasecal', 'polangle', 'polcheck' or 'targets'. Finished stages are skipped when the run is restarted in the same directory.")

    vlacalparser = parser.add_argument_group(
        "-------------------------Historical VLA calibrator-based calibration (1GC, --vlacal) Settings-------------------------")
    vlacalparser.add_argument('--vlacal',
                              help='Calibrate historical (pre-2010, VA antennas) VLA data without CASA, from the MS that CASA importvla makes of the archive files (all fields and spectral windows): quack, flagging, VLA gain curve, flux scale and bandpass (Perley-Butler 2017), phase calibrator gains, transfer to the target(s) and parallactic angle correction; optionally (--vlacal-level) leakage with the calibrator polarisation (D+QU) and the polarisation angle (R-L phase) on 3C286/3C138. Writes one calibrated MS (sky frame, no Dysco compression) per target and spectral window for a facetselfcal run. L, C and X band.',
                              action='store_true')
    vlacalparser.add_argument('--vlacal-level', default='none', type=str, choices=['none', 'auto', 'leakage', 'full'],
                              help="Calibration level: 'none' (gains and flux scale only, no polarisation calibration), 'leakage' (also the leakage), 'full' (also the polarisation angle) or 'auto' (the highest level the observed calibrators allow). The default is 'none': polarisation calibration only when asked for, even if suitable calibrators were observed.")
    vlacalparser.add_argument('--vlacal-band', default=None, type=str, choices=['L', 'C', 'X'],
                              help='Band to calibrate. Needed for band-switching observations (spectral windows in more than one band): run each band in its own directory. By default the only band in the MS.')
    vlacalparser.add_argument('--vlacal-fluxcal', default=None, type=arg_as_list,
                              help='Flux density (and bandpass) calibrator field name(s), for example "[\'0137+331\']". By default the fields at the position of 3C48, 3C138, 3C147 or 3C286.')
    vlacalparser.add_argument('--vlacal-phasecal', default=None, type=arg_as_list,
                              help='Phase (gain) calibrator field name(s). By default all other fields with a VLA calibrator code (FIELD CODE not blank). Each target gets the phase calibrator nearest on the sky.')
    vlacalparser.add_argument('--vlacal-polcal', default=None, type=arg_as_list,
                              help='Polarisation angle calibrator field name(s) (3C286, or 3C138 at L-band). By default from the calibrator positions.')
    vlacalparser.add_argument('--vlacal-leakcal', default=None, type=arg_as_list,
                              help='Leakage calibrator field name(s). By default the unpolarised calibrators (3C84, OQ208, 3C147 at L-band) and the phase calibrators observed over at least --vlacal-min-parang of parallactic angle. The polarisation of a field is solved with the leakage unless it is one of the unpolarised calibrators.')
    vlacalparser.add_argument('--vlacal-targets', default=None, type=arg_as_list,
                              help='Target field name(s). By default all fields with a blank VLA calibrator code.')
    vlacalparser.add_argument('--vlacal-quack', default=10., type=float,
                              help='Seconds flagged at the start of every scan. The default is 10.')
    add_bool_arg(vlacalparser, 'vlacal-gaincurve', default=True,
                 help='Correct the elevation dependence of the antenna gains with the VLA gain curves (the CASA ones). The default is True.')
    add_bool_arg(vlacalparser, 'vlacal-flag-mixed-baselines', default=True,
                 help='In EVLA-transition data (VA and EA antennas) flag the VLA-EVLA baselines, which have closure errors in continuum mode. The default is True.')
    vlacalparser.add_argument('--vlacal-min-parang', default=60., type=float,
                              help='Minimum parallactic angle range (deg) of a polarised (or unknown) leakage calibrator, to separate its polarisation from the leakage. The default is 60.')
    vlacalparser.add_argument('--vlacal-leakage-nchan', default=1, type=int,
                              help='Channels per leakage solution (0: one per spectral window). The default is 1.')
    vlacalparser.add_argument('--vlacal-flagcycles', default=2, type=int,
                              help='Number of solve and flag cycles on the calibrators. The default is 2.')
    vlacalparser.add_argument('--vlacal-timeres', default=0., type=float,
                              help='Time resolution (s) of the calibrated target MSs (integer averaging factor closest to this resolution divided by the integration time). 0 keeps the native resolution. The default is 0.')
    vlacalparser.add_argument('--vlacal-target-clip', default=6., type=float,
                              help='Outlier threshold (robust standard deviations) for flagging the target scans on Stokes V and the cross-hands. The default is 6.')
    add_bool_arg(vlacalparser, 'vlacal-phasecal-selfcal', default=True,
                 help='Build the phase calibrator model with a short facetselfcal DI run. If False a point source at the phase centre is used. The default is True.')
    vlacalparser.add_argument('--vlacal-stopafter', default=None, type=str,
                              help="Stop after this stage: 'discover', 'prepare', 'fluxcal', 'phasecal', 'leakage', 'polangle', 'polcheck' or 'targets'. Finished stages are skipped when the run is restarted in the same directory.")
    add_bool_arg(parser, 'timesplitbefore', default=None,
                 help='Split the MS in time when the time axis is not contiguous. If this option is not set, facetselfcal automatically splits the MS only when the time gaps are large. This is mainly useful for MeerKAT, uGMRT, and (E)VLA observations.')
    parser.add_argument('--noarchive',
                        help='Do not archive the data.',
                        action='store_true')
    parser.add_argument('--skipbackup',
                        help='Leave the original MS intact and always work on a copy made by DP3.',
                        action='store_true')
    parser.add_argument('--skip-solution-plotting', action='store_true',
                 help='Skip creating solution plots of the solutions with losoto')
    parser.add_argument('--skip-Ateam-plotting', action='store_true', 
                    help='Skip creating A-team distance plots')
    parser.add_argument('--skip-time-coverage-plotting', action='store_true',
                    help='Skip creating time coverage plots')
    parser.add_argument('--skip-UV-coverage-plotting',
                        dest='skip_uv_coverage_plotting',
                        help='Skip creating UV-coverage plots for input measurement sets.',
                        action='store_true')
    parser.add_argument('--keepmodelcolumns',
                        help='Leave the MODEL_DATA-type columns in the MS. By default these are removed to save disk space.',
                        action='store_true')
    parser.add_argument('--createresidualdatacolumn',
                        help='Create a RESIDUAL_DATA column in the MS at the last selfcal cycle.',
                        action='store_true')
    parser.add_argument('--compute-weightspectrum',
                        help='Set the WEIGHT_SPECTRUM column in the MS based on the measured variances in RESIDUAL_DATA. Mainly useful for uGMRT observations. This is done at the end of each selfcal cycle. Note: it is important that the visibility units are in Jy for this to work properly.',
                        action='store_true')                        
    parser.add_argument('--phasediff_only',
                        help='For finding only the phase difference, we want to stop after calibrating and before imaging',
                        action='store_true')
    parser.add_argument('--configpath',
                        help='Path to user config file which will overwrite command line arguments',
                        default='facetselfcal_config.txt',
                        type=str)
    parser.add_argument('--auto',
                        help='Trigger fully automated processing for HBA extracted datasets, HBA long baseline facet-calibrators, MeerKAT L-band DI/DD, and MeerKAT UHF-band DI/DD. For DD --auto-directions functionality is used.',
                        action='store_true')
    parser.add_argument('--auto-directions',
                        help='Automatically determine directions for DD solves (under development).',
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


def add_bool_arg(parser, name, help, default=None, second_name=None):
    """
    Add mutually exclusive positive and negative boolean options to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the options are added.
    name : str
        Primary option name without leading dashes.
    help : str
        Help text for the positive option.
    default : bool, optional
        Default value for the parsed option.
    second_name : str, optional
        Alias without leading dashes.
    """
    group = parser.add_mutually_exclusive_group(required=False)
    if second_name is not None:
        group.add_argument('--' + name, '--' + second_name, dest=name.replace('-', '_'), action='store_true', help=help)
    else:
        group.add_argument('--' + name, dest=name.replace('-', '_'), action='store_true', help=help)
    group.add_argument('--no-' + name, dest=name.replace('-', '_'), action='store_false', help='Disable/set to False ' + name)
    parser.set_defaults(**{name.replace('-', '_'):default})

def arg_as_list(s):
    """
    Parse a string containing a Python list and reject other values.

    Parameters
    ----------
    s : str
        String representation of a list.

    Returns
    -------
    list
        Parsed list.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a list.
    """
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def arg_as_str_or_list(s):
    """
    Parse a string as either a plain string or a list.

    Parameters
    ----------
    s : str
        String value or string representation of a list.

    Returns
    -------
    str or list
        Parsed string or list.

    Raises
    ------
    argparse.ArgumentTypeError
        If a bracketed value is not a list.
    """
    if "[" not in s and "]" not in s:
        return str(s)

    v = ast.literal_eval(s)
    if isinstance(v, list):
        return v

    raise argparse.ArgumentTypeError(f'Argument "{s}" is not a string or list')


def arg_as_float_or_list(s):
    """
    Parse a string as either a floating-point value or a list.

    Parameters
    ----------
    s : str
        String representation of a float or list.

    Returns
    -------
    float or list
        Parsed float or list.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is neither a float nor a list.
    """
    try:
        return float(s)
    except ValueError:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
        raise argparse.ArgumentTypeError(f'Argument "{s}" is not a float or list')


def str_or_int(arg):
    """
    Parse an argument as an integer, falling back to a string.

    Parameters
    ----------
    arg : str
        Value to parse.

    Returns
    -------
    int or str
        Parsed integer or original string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is neither an integer nor a string.
    """
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if isinstance(arg, str):
        return arg
    raise argparse.ArgumentTypeError("Input must be an int or string")


def str_or_float(arg):
    """
    Parse an argument as a float, falling back to a string.

    Parameters
    ----------
    arg : str
        Value to parse.

    Returns
    -------
    float or str
        Parsed float or original string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is neither a float nor a string.
    """
    try:
        return float(arg)  # try convert to float
    except ValueError:
        pass
    if isinstance(arg, str):
        return arg
    raise argparse.ArgumentTypeError("Input must be a float or string")


def int_or_intlist(argin):
    """
    Parse an argument as an integer or a list of integers.

    Parameters
    ----------
    argin : str or int
        String representation of an integer or a list of integers.

    Returns
    -------
    int or list of int
        Parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not an integer or a list of integers.
    """
    if isinstance(argin, int):
        return argin
    try:
        arg = ast.literal_eval(str(argin))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Argument \"%s\" is not an integer or a list of integers" % (argin))
    if isinstance(arg, int) and not isinstance(arg, bool):
        return arg
    if isinstance(arg, list) and len(arg) > 0 and all(isinstance(item, int) and not isinstance(item, bool) for item in arg):
        return arg
    raise argparse.ArgumentTypeError("Argument \"%s\" is not an integer or a list of integers" % (argin))


def floatlist_or_float(argin):
    """
    Parse an argument as a float or a list of numeric values.

    Parameters
    ----------
    argin : str or None
        String representation of a float or numeric list.

    Returns
    -------
    float, list, or None
        Parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a float or numeric list.
    """
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
    """
    Check a list of positive integers or formatted time strings.

    Parameters
    ----------
    argin : str
        String representation of the list to check.

    Returns
    -------
    list
        Parsed list of positive integers or formatted time strings.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a valid list.
    """

    # check if input is a list and make proper list format
    arg = ast.literal_eval(argin)
    if type(arg) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (argin))
    if not arg:
        raise argparse.ArgumentTypeError("solint_list cannot be empty")

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
