#!/usr/bin/env python
# rotationmeasure updates
# HTML/plots:add UV-coverage plots with shadeMS
# python3 -m venv $PWD; bin/pip install git+https://github.com/rvweeren/shadeMS/@symlog-uv-coverage
# use phase slope fitting for bandpass step?
# https://ui.adsabs.harvard.edu/abs/2022ApJ...932..110K/abstract
# python /net/rijn/data2/rvweeren/software/lofar_facet_selfcal/submods/MSChunker.py --timefraction=0.15 --mintime=1200 --mode=time L765157.ms.copy
# run with less disk-space usage, remove all but merged h5
# continue splitting functions in facetselfcal in separate modules
# time, timefreq, freq med/avg steps (via losoto)
# BDA step DP3
# Stacking check that freq and time axes are identical
# scalarphasediff solve WEIGHT_SPECTRUM_PM should not be dysco compressed! Or not update weights there...
# BLsmooth cannot smooth more than bandwidth and time smearing allows, not checked now
# bug related to sources-pb.txt in facet imaging being empty if no -apply-beam is used
# fix RR-LL referencing for flaged solutions, check for possible superterp reference station
# phase detrending.
# BLsmooth constant smooth for gain solves
# use scalarphasediff sols stats for solints? test amplitude stats as well
# uvmin, uvmax, per ms per soltype

# antidx = 0
# taql("select ANTENNA1,ANTENNA2,gntrue(FLAG)/(gntrue(FLAG)+gnfalse(FLAG)) as NFLAG from L656064_129_164MHz_uv_pre-cal.concat.ms WHERE (ANTENNA1=={:d} OR ANTENNA2=={:d}) AND ANTENNA1!=ANTENNA2".format(antidx, antidx)).getcol('NFLAG')
#read_MeerKAT_wscleanmodel_5spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J1939-6342_UHF.txt','J1939-6342_UHF.skymodel')
#read_MeerKAT_wscleanmodel_5spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J1939-6342_L.txt','J1939-6342_L.skymodel')
#read_MeerKAT_wscleanmodel_4spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J0408-6545_UHF.txt','J0408-6545_UHF.skymodel')
#read_MeerKAT_wscleanmodel_3spix('/net/lofar4/data2/rvweeren/MeerKAT/calibrated/J0408-6545_L.txt','J0408-6545_L.skymodel')


# example:
# python facetselfal.py -b box_18.reg --forwidefield --avgfreqstep=2 --avgtimestep=2 --smoothnessconstraint-list="[0.0,0.0,5.0]" --antennaconstraint-list="['core']" --solint-list=[1,20,120] --soltypecycles-list="[0,1,3]" --soltype-list="['tecandphase','tecandphase','scalarcomplexgain']" test.ms

# Standard library imports
import ast
import configparser
import copy
import glob
import logging
import multiprocessing
import os
import os.path
import re
import shutil
import subprocess
import sys
import tempfile
from functools import wraps
from pathlib import Path

# avoid numexpr warning when cpu_count() exceeds its default NUMEXPR_MAX_THREADS (64)
os.environ.setdefault('NUMEXPR_MAX_THREADS', str(multiprocessing.cpu_count()))

import math
import time

# Third party imports
import losoto  # type: ignore
import losoto.lib_operations  # type: ignore
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyregion
import tables
from astropy.io import fits
from astropy.table import Table
from casacore.tables import table  # type: ignore

# Required for running with relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from arguments import option_parser

from submods.fair_log.config import add_config_to_h5, add_version_to_h5
from submods.h5_helpers.flagging import (flag_bad_amps, flaghighamps,
                                         flaghighamps_fulljones, flaglowamps,
                                         flaglowamps_fulljones)
from submods.h5_helpers.h5_info import (amplitude_leakage_paramdb,
                                        findrefant_core, force_close,
                                        fulljonesparmdb, has0coordinates,
                                        is_scalar_array_for_wsclean,
                                        ntimesH5, number_freqchan_h5)
from submods.h5_helpers.modify_amplitude import (get_median_amp,
                                                 normamplitudes,
                                                 normamplitudes_withmatrix,
                                                 normslope_withmatrix)
from submods.h5_helpers.modify_delay import fix_delayreference
from submods.h5_helpers.modify_phase import (fix_phasereference,
                                             makephaseCDFh5_h5merger,
                                             makephasediffh5)
from submods.h5_helpers.modify_rotation import (fix_rotationmeasurereference,
                                                fix_rotationreference,
                                                fix_weights_rotationh5,
                                                fix_weights_rotationmeasureh5,
                                                rotationmeasure_to_phase)
from submods.h5_helpers.modify_tec import fix_tecreference
from submods.h5_helpers.multidir_h5 import is_multidir, same_weights_multidir
from submods.h5_helpers.multidir_merge import (
    copy_over_solutions_from_skipped_directions, merge_splitted_h5_ordered)
from submods.h5_helpers.nan_values import (remove_nans,
                                           removenans_amplitude_leakage,
                                           removenans_fulljones)
from submods.h5_helpers.overwrite_table import (copy_over_source_direction_h5,
                                                set_weights_h5_to_one)
from submods.h5_helpers.reset_h5parm_structure import fix_h5
from submods.h5_helpers.reset_solutions import (resetsolsfordir,
                                                resetsolsforstations)
from submods.h5_helpers.split_h5 import split_multidir
from submods.h5_helpers.update_sources import (update_sourcedir_h5_dde,
                                               update_sourcedirname_h5_dde)
from submods.h5_merger import merge_h5
from submods.source_selection.early_stopping import early_stopping
from submods.source_selection.phasediff_output import GetSolint
from submods.source_selection.phasediff_output import \
    generate_csv as generate_phasediff_csv
from submods.source_selection.selfcal_selection import get_images_solutions
from submods.runtime_utils import (_log_machine_info, _run_predict_command,
                                   _track_resource_phase,
                                   _with_run_directory_lock,
                                   get_resource_monitor, print_title, run,
                                   set_logger, set_resource_monitor,
                                   terminal_print)
from submods.ms_fixes import (fix_antenna_info_gmrt, fix_bad_weightspectrum,
                              fix_equidistant_times, fix_GMRT_weights,
                              fix_time_axis_gmrt, fix_uvw,
                              fix_variable_reference_pointing,
                              fix_zero_weight_spectrum, get_telescope_from_ms,
                              getVarianceRRLL, remove_flagged_data_startend,
                              remove_syspower, set_metadata_compression,
                              updateCrosshand)
from submods.fits_utils import flatten
from submods.calibrator_models import setjy_casa
from submods.data_import import download_MWA_beam_model
from submods.vla_helpers import get_vla_configuration, get_vla_maxconfiguration
# Not used in this file, but kept importable from facetselfcal.main for
# standalone use, e.g. "from facetselfcal.main import split_ms_spws". These are
# functions that used to be defined in this file, so that "from
# facetselfcal.main import *" as used in the wiki
# (https://github.com/rvweeren/lofar_facet_selfcal/wiki) keeps working.
from submods.calibrator_models import (  # noqa: F401
    calculate_evpa_3C286, calculate_pfrac_3C286,
    read_MeerKAT_wscleanmodel_3spix, read_MeerKAT_wscleanmodel_4spix,
    read_MeerKAT_wscleanmodel_5spix, set_polarised_model_3C286)
from submods.data_import import convert_lta_to_uvfits, gmrt_uvfits2ms  # noqa: F401
from submods.vla_helpers import (  # noqa: F401
    collect_all_frequencies, get_EVLA_IF_pair, split_ms_spws)
from submods.flagging import (  # noqa: F401
    flag_antenna_timerange_ms, flag_shadowed_antenna, flagms_startend,
    runaoflagger, uvmaxflag)
from submods.h5_helpers.flagging import (  # noqa: F401
    flag_h5_phasediff)
from submods.h5_helpers.modify_amplitude import (  # noqa: F401
    copyovergain, reset_gains_noncore)
from submods.h5_helpers.modify_phase import (  # noqa: F401
    change_refant, copyoverscalarphase, makephaseCDFh5, reset_phase000)
from submods.image_utils import (  # noqa: F401
    max_area_of_island, rename_models)
from submods.h5_helpers.general_utils import make_utf8  # noqa: F401
from submods.runtime_utils import (  # noqa: F401
    dp3_command_phases, dp3_workflow_step_kinds)
from submods.split_irregular_timeaxis import regularize_ms, split_ms  # noqa: F401
from utils.parsers import parse_history  # noqa: F401
from submods.antenna_constraints import MeerKAT_antconstraint  # noqa: F401
from submods.h5_helpers.h5_info import check_phaseup, check_soltabs  # noqa: F401
from submods.input_checks import (  # noqa: F401
    is_ms_regularized, is_stokesdiagonal_modeltype_allowed,
    is_stokesi_modeltype_allowed, removenonms, test_antenna_averaging_factors)
from submods.skymodels import grab_coord_MS  # noqa: F401
from submods.antenna_constraints import max_in_str  # noqa: F401
from submods.runtime_utils import which  # noqa: F401
from submods.solints import remove_bad_endrounding  # noqa: F401
from submods.imaging_params import (  # noqa: F401
    declination_sensivity_factor, get_fractional_bandwidth, getdeclinationms,
    is_binary, round_up_to_even, write_RMsynthesis_weights)
from submods.facet_directions import (  # noqa: F401
    MeerKAT_autodetect_highDR, create_homogenized_facetdirections,
    determine_compactsource_flux, split_facetdirections)
from submods.image_utils import (  # noqa: F401
    get_image_dynamicrange, getlargestislandsize, makemask_extended)
from submods.losoto_helpers import (  # noqa: F401
    create_losoto_beamcorparset, losotolofarbeam)
from submods.beam import (  # noqa: F401
    beam_keywords, beamcormodel, create_beamcortemplate, fixbeam_ST001)
from submods.antenna_constraints import (  # noqa: F401
    filter_baseline_str_removestations)
from submods.flagging import (  # noqa: F401
    return_antennas_highflaggingpercentage)
from submods.ms_columns import _get_stack_groups, mscolexist  # noqa: F401
from submods.skymodels import getregioncenter  # noqa: F401
from submods.ms_info import (  # noqa: F401
    get_chan_freqs, getAntennas, number_of_unique_obsids)
from submods.ms_operations import (  # noqa: F401
    add_dummyms, bda_mslist, concat_ms_from_same_obs, create_pointing_list,
    findfreqavg, getobsmslist, removestartendms, split_columns, timebase)
from submods.h5_helpers.solution_tools import (  # noqa: F401
    FFTdelayfinder, findamplitudenoise, smoothsols)
from submods.applycal import (  # noqa: F401
    closest_arrayvals, find_closest_ddsol, find_closest_H5time_toms,
    h5flags2ms)
from submods.ms_columns import (  # noqa: F401
    normalize_data_bymodel, stackMS_taql)
from submods.ms_fixes import (  # noqa: F401
    check_equidistant_times, fix_twopol_ms, is_two_pol_ms,
    parse_bad_freq_ranges, remove_antennas)
from submods.plotting import (  # noqa: F401
    plot_ms_time_coverage, plotimage_aplpy, selfcal_animatedgif)
from submods.skymodels import (  # noqa: F401
    getGSM, makeBBSmodelforFITS, makeBBSmodelforTGSS, makeBBSmodelforVLASS,
    mslist_return_stack)
from submods.smearing import (  # noqa: F401
    get_resolution, get_time_preavg_factor_LTAdata, get_uvwmax, isfloat,
    smearing_bandwidth, smearing_time, smearing_time_ms,
    smearing_time_ms_imsize)
from submods.solints import (  # noqa: F401
    find_nearest, getms_amp_stats, getms_phase_stats, getmsmodelinfo,
    return_soltype_index)
from submods.plotting import (check_large_timegaps_ms,
                              create_Ateam_seperation_plots, plot_UV_coverage,
                              plotimage, plotimage_astropy)
from submods.image_utils import (checkforzerocleancomponents,
                                 clean_up_images, findrms, fix_fpb_images,
                                 frequencies_from_models, get_image_size,
                                 getimsize, groupskymodel, gzip_model_images,
                                 mask_region, mask_region_inv,
                                 modify_freqs_from_ms, removenegativefrommodel,
                                 removeneNaNfrommodel)
from submods.ms_columns import (addcol, create_phase_column,
                                create_phase_slope, create_phasediff_column,
                                create_weight_spectrum,
                                create_weight_spectrum_modelratio,
                                remove_column_ms, stackwrapper)
from submods.beam import beamcor_and_lin2circ, set_beamcor
from submods.model_columns import (create_empty_fitsimage,
                                   create_MODEL_DATA_PDIFF, predictsky,
                                   remove_model_columns,
                                   set_DDE_predict_skymodel_solve,
                                   set_modelstoragemanager,
                                   updatemodelcols_includedir)
from submods.input_checks import (check_antenna_factors, check_applyfacetbeam,
                                  check_equidistant_freqs,
                                  check_for_highmem_longsolint,
                                  check_if_ms_exists, check_pointing_centers,
                                  check_valid_ms, inputchecker)
from submods.imaging_params import (autodetect_highDR, calculate_restoringbeam,
                                    makemaskthresholdlist, multiscale_trigger,
                                    niter_from_imsize, set_channelsout,
                                    set_fitsmask_restart, set_fitspectralpol,
                                    update_channelsout, update_fitsmask,
                                    update_fitspectralpol, update_uvmin)
from submods.facet_directions import (add_bright_source_to_catalog,
                                      add_peak_total_flux_to_catalog,
                                      add_source_to_catalog,
                                      auto_determine_extractregion,
                                      calibration_error_map,
                                      create_calibration_error_catalog,
                                      create_facet_directions,
                                      filter_catalog_on_flux,
                                      get_number_of_sources_in_catalog,
                                      getregionsize, merge_catalogs,
                                      parse_facetdirections,
                                      update_calibration_error_catalog,
                                      write_compactsource_flux,
                                      write_ds9_regions)
from submods.ms_info import (check_phaseup_station, checklongbaseline,
                             get_frequencies_from_ms)
from submods.ms_operations import (_write_stack_imaging_group_manifest,
                                   applycal_restart_di, archive, average,
                                   cleanup, concat_ms_wsclean_facetimaging,
                                   create_splitted_ms, get_startchan_nchan,
                                   makemslist, nested_mslistforimaging,
                                   phaseup, split_multidir_ms)
from submods.antenna_constraints import antennaconstraintstr
from submods.h5_helpers.solution_tools import (compute_markersize,
                                                find_bad_deviating_antennas)
from submods.applycal import (applycal, build_applycal_dde_cmd,
                              corrupt_modelcolumns, create_mergeparmdbname,
                              preapply, preapply_bandpass, time_match_mstoH5)
from submods.skymodels import (set_MeerKAT_bandpass_skymodel,
                               set_skymodels_external_surveys)
from submods.solints import (auto_determinesolints, format_nchan,
                             format_solint, setinitial_solint, tweak_solints,
                             tweak_solints_single)
from submods.smearing import (bandwidthsmearing,
                              compute_distance_to_pointingcenter,
                              report_bandwidth_smearing)
from submods.primary_beam import (MeerKAT_pbcor, VLA_pbcor, uGMRT_pbcor,
                                  write_primarybeam_info)
from submods.flagging import (aoflagger_column, clip_DATA,
                              compute_flagging_statistics,
                              create_backup_flag_col, flag_antenna_taql,
                              flag_autocorr, flag_smeared_data,
                              flag_uGMRT_badfreqs)

# Modules
from submods.losoto_helpers import create_losoto_delayparset, create_losoto_fastphaseparset, create_losoto_FRparset, create_losoto_FRparsetplotfit, create_losoto_tecandphaseparset, create_losoto_rotationparset, tecandphaseplotter, create_losoto_tecparset, create_losoto_flag_apgridparset, create_losoto_bandpassparset, create_losoto_mediumsmoothparset, create_losoto_flag_ap_only

# Set logger
os.makedirs('logs', exist_ok=True) 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('logs/selfcal.log', delay=True)
formatter = logging.Formatter('%(levelname)s:%(asctime)s ---- %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)
set_logger(logger)  # run() in submods.runtime_utils logs via this logger

_REPORT_RUN_INITIALIZED = False

_RUN_MODE_MARKER = "facetselfcal_mode.txt"
_RUN_MODES = frozenset(("bandpass", "DDE", "standard"))


def _run_mode_from_args(args):
    dde = args.get("DDE")
    bandpass = args.get("bandpass")
    if not isinstance(dde, bool) or not isinstance(bandpass, bool):
        raise ValueError("The DDE and bandpass mode flags must be booleans.")
    if dde and bandpass:
        raise ValueError("--DDE and --bandpass cannot be used together.")
    if bandpass:
        return "bandpass"
    if dde:
        return "DDE"
    return "standard"


def _run_mode_from_config(config_path):
    mode_flags = {}
    with Path(config_path).open(encoding="utf-8") as config_file:
        for line in config_file:
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or key not in ("DDE", "bandpass"):
                continue
            try:
                mode_flags[key] = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    "Could not parse the {} flag in {}.".format(key, config_path)
                ) from exc

    if "DDE" not in mode_flags:
        raise ValueError(
            "{} does not contain a DDE mode flag.".format(config_path)
        )
    mode_flags.setdefault("bandpass", False)
    return _run_mode_from_args(mode_flags)


def _validate_run_mode(recorded_mode, requested_mode, source):
    if recorded_mode not in _RUN_MODES:
        raise ValueError(
            "{} contains unrecognized run mode {!r}; expected one of {}.".format(
                source, recorded_mode, ", ".join(sorted(_RUN_MODES))
            )
        )
    if recorded_mode != requested_mode:
        raise ValueError(
            "This directory is already associated with the {!r} mode ({}) and "
            "cannot be used for a {!r} run. Use a new, clean run directory to "
            "change modes.".format(recorded_mode, source, requested_mode)
        )


def _ensure_run_mode_marker(run_directory, requested_mode):
    if requested_mode not in _RUN_MODES:
        raise ValueError("Unrecognized requested run mode {!r}.".format(requested_mode))

    run_directory = Path(run_directory)
    misc_directory = run_directory / "misc"
    misc_directory.mkdir(parents=True, exist_ok=True)
    marker_path = misc_directory / _RUN_MODE_MARKER

    if marker_path.exists():
        recorded_mode = marker_path.read_text(encoding="utf-8").strip()
        _validate_run_mode(recorded_mode, requested_mode, marker_path)
        return

    legacy_config = run_directory / "full_config.txt"
    if legacy_config.is_file():
        recorded_mode = _run_mode_from_config(legacy_config)
        _validate_run_mode(recorded_mode, requested_mode, legacy_config)

    try:
        with marker_path.open("x", encoding="utf-8") as marker_file:
            marker_file.write(requested_mode + "\n")
            marker_file.flush()
            os.fsync(marker_file.fileno())
    except FileExistsError:
        recorded_mode = marker_path.read_text(encoding="utf-8").strip()
        _validate_run_mode(recorded_mode, requested_mode, marker_path)


try:
    from .monitor import ResourceMonitor
except ImportError:
    try:
        from monitor import ResourceMonitor
    except ImportError:
        from facetselfcal.monitor import ResourceMonitor


def _configure_selfcal_log(start):
    """Configure the self-calibration log for the requested run segment.

    Parameters
    ----------
    start : int
        First self-calibration cycle. Zero archives any existing log and
        starts a new file; a positive value appends to the existing log.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the existing log cannot be archived or the new log cannot be opened.
    """
    global file_handler

    log_path = Path('logs/selfcal.log')
    logger.removeHandler(file_handler)
    file_handler.close()

    if start == 0 and log_path.exists():
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        archive_path = log_path.with_name(f'selfcal_{timestamp}.log')
        archive_index = 1
        while archive_path.exists():
            archive_path = log_path.with_name(f'selfcal_{timestamp}_{archive_index}.log')
            archive_index += 1
        log_path.rename(archive_path)

    mode = 'w' if start == 0 else 'a'
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _prepare_html_overview(start):
    """Remove a previous HTML report before starting a new run.

    Parameters
    ----------
    start : int
        First self-calibration cycle. Existing report output is removed only
        when this value is zero.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the existing report output cannot be removed.
    """
    if start != 0:
        return

    report_directory = Path('html_overview')
    if report_directory.is_symlink() or report_directory.is_file():
        report_directory.unlink()
    elif report_directory.is_dir():
        shutil.rmtree(report_directory)


matplotlib.use('Agg')

# For NFS mounted disks
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def parse_input_args(namespace):
    """
    Serialize an argument namespace as ``name=value`` strings.

    Parameters
    ----------
    namespace : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list of str
        Serialized namespace entries.
    """
    return [f"{k}={repr(v)}" for k, v in vars(namespace).items()]

def insert_history_ms(ms_path, parameters=[], message='parameters', app='facetselfcal', appver='1.0.0', origin='facetselfcal (https://github.com/rvweeren/lofar_facet_selfcal)'):
    """
    Insert history in MeasurementSet

    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set history table.
    parameters : list of str, optional
        Parameter entries to record in the history table.
    message : str, optional
        History message to record.
    app : str, optional
        Name of the application that produced the history entry.
    appver : str, optional
        Application version to record.
    origin : str, optional
        Software origin or URL to record.

    Returns
    -------
    None
        The history entry is appended to the Measurement Set.
    """
    history_table_path = ms_path.rstrip('/') + '/HISTORY'
    with table(history_table_path, readonly=False, ack=False) as t:

        nrows = len(t)
        t.addrows(1)

        now = time.time()
        columns = t.colnames()

        if 'TIME' in columns:
            t.putcell('TIME', nrows, now)
        if 'OBSERVATION_ID' in columns:
            t.putcell('OBSERVATION_ID', nrows, 0)
        if 'MESSAGE' in columns:
            t.putcell('MESSAGE', nrows, message)
        if 'PRIORITY' in columns:
            t.putcell('PRIORITY', nrows, 'NORMAL')
        if 'ORIGIN' in columns:
            t.putcell('ORIGIN', nrows, origin+" "+appver)
        if 'APPLICATION' in columns:
            t.putcell('APPLICATION', nrows, app)
        if 'CLI_COMMAND' in columns:
            t.putcell('CLI_COMMAND', nrows, [])
        if 'APP_PARAMS' in columns:
            t.putcell('APP_PARAMS', nrows, parameters)


def write_processing_history(cmd, version, imagebasename):
    """
    Updates the FITS headers of images matching the given basename with processing history.

    This function writes the command used for processing and the facetselfcal version into the
    primary header of each FITS file whose name matches the provided image basename pattern.
    It also adds citation information as comments in the header.

    Parameters
    ----------
    cmd : str
        The command used for processing, typically the command-line invocation.
    version : str
        The version string of facetselfcal.
    imagebasename : str
        The base name pattern to match FITS image files.

    Notes
    -----
    - Only the portion of the command after 'facetselfcal.py' is recorded, if present.
    - FITS files are identified using glob with the pattern '{imagebasename}*image*.fits'.
    - The function modifies FITS files in place.
    """
    # strip everyting before the facetselcal.py string in cmd
    cmd = cmd.strip() 
    cmd = 'facetselfcal.py' + cmd.split('facetselfcal.py')[1] if 'facetselfcal.py' in cmd else cmd
    imagelist = glob.glob( imagebasename + '*image*.fits')
    for image in imagelist:
        terminal_print('Updating FITS header:', image)
        logger.info('Updating FITS header facetselfcal command: ' + image)
        with fits.open(image, mode='update') as hdul:
            # Add the command used for processing to the primary header
            hdul[0].header['HISTORY'] = "facetselfcal version: " + version
            hdul[0].header['HISTORY'] = cmd
            hdul[0].header['COMMENT'] = "========================================================================"
            hdul[0].header['COMMENT'] = "       If you use facetselfcal for scientific work, please cite:        "
            hdul[0].header['COMMENT'] = "                 van Weeren et al. (2021, A&A, 651, 115)                "
            hdul[0].header['COMMENT'] = "========================================================================"


    # Still need to update args['wscleanskymodel'] to include prefix


def _remove_previous_cycle_products(start_cycle, stop_cycle, imagename,
                                    preserved_inputs=(), remove_best_outputs=False):
    """Remove stale generated products from cycles a run will replace."""
    if start_cycle < 0 or stop_cycle is None or stop_cycle <= start_cycle:
        return []

    image_prefix = Path(imagename).name
    image_prefixes = tuple(
        dict.fromkeys((image_prefix, "box_" + image_prefix, "best_" + image_prefix))
    )
    protected_paths = set()
    protected_prefixes = set()

    def remember_input(value):
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                remember_input(item)
            return
        if not isinstance(value, (str, os.PathLike)):
            return
        input_path = Path(value).expanduser()
        try:
            resolved = os.path.normcase(str(input_path.resolve()))
        except (OSError, RuntimeError):
            return
        protected_paths.add(resolved)
        protected_prefixes.add(resolved.rstrip(os.sep))

    for preserved_input in preserved_inputs:
        remember_input(preserved_input)

    def cycle_for_product(path, product_kind):
        if product_kind in ("h5", "solution_plot"):
            match = re.search(r"selfcalcycle(\d+)", path.name, re.IGNORECASE)
            if match:
                return int(match.group(1))
            if product_kind == "solution_plot" and start_cycle == 0:
                return 0
            if (
                product_kind == "h5"
                and remove_best_outputs
                and re.match(r"best_.*solutions\.h5$", path.name, re.IGNORECASE)
            ):
                return start_cycle
            return None
        if product_kind in ("directions", "facet_regions"):
            match = re.fullmatch(r"directions_(\d+)\.(?:txt|reg)", path.name, re.IGNORECASE)
            return int(match.group(1)) if match else None
        for prefix in image_prefixes:
            if not prefix:
                continue
            match = re.match(
                r"^{}(\d+)(?=[-_.]|$)".format(re.escape(prefix)),
                path.name,
                re.IGNORECASE,
            )
            if match:
                return int(match.group(1))
        return None

    product_directories = [
        (Path("h5_solutions"), "h5"),
        (Path("fits_images"), "image"),
        (Path("clean_masks"), "image"),
        (Path("plots"), "image"),
        (Path("errormaps_dd"), "image"),
        (Path("directions"), "directions"),
        (Path("facet_regions"), "facet_regions"),
    ]
    product_directories.extend(
        (path, "solution_plot")
        for path in Path(".").glob("solution_plots_*")
        if path.is_dir()
    )

    removed_products = []
    for directory, product_kind in product_directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if product_kind == "image" and directory == Path("plots") and (
                path.name.lower().startswith("ateam_")
                or path.name.lower().endswith(".time_coverage.png")
            ):
                continue
            cycle = cycle_for_product(path, product_kind)
            if cycle is None or cycle < start_cycle:
                continue
            try:
                resolved = os.path.normcase(str(path.resolve()))
            except (OSError, RuntimeError):
                continue
            if resolved in protected_paths or any(
                resolved.startswith(prefix + os.sep) for prefix in protected_prefixes
            ):
                continue
            path.unlink(missing_ok=True)
            removed_products.append(path)

    return removed_products


# temporary function to check rounding issues with antenna averaging factors, remove later


def logbasicinfo(args, fitsmask, mslist, version, inputsysargs):
    """
    Prints basic information to the screen.

    Parameters
    ----------
    args : iterable
        list of input arguments.
    fitsmask : str
        name of the user-provided FITS mask.
    mslist : list
        list of input measurement sets.
    version : str
        facetselfcal version string.
    inputsysargs : list of str
        Original command-line arguments passed to the program.
    """
    logger.info(' '.join(map(str, inputsysargs)))

    logger.info('Imsize:                    ' + str(args['imsize']))
    logger.info('Pixelscale [arcsec]:       ' + str(args['pixelscale']))
    logger.info('Niter:                     ' + str(args['niter']))
    logger.info('Uvmin:                     ' + str(args['uvmin']))
    logger.info('Multiscale:                ' + str(args['multiscale']))
    logger.info('Beam correction:           ' + str(args['beamcor']))
    logger.info('IDG:                       ' + str(args['idg']))
    logger.info('Widefield:                 ' + str(args['forwidefield']))
    logger.info('Flagslowamprms:            ' + str(args['flagslowamprms']))
    logger.info('flagslowphaserms:          ' + str(args['flagslowphaserms']))
    logger.info('Do linear:                 ' + str(args['dolinear']))
    logger.info('Do circular:               ' + str(args['docircular']))
    if args['boxfile'] is not None:
        logger.info('Boxfile:                   ' + args['boxfile'])
    logger.info('Mslist:                    ' + ' '.join(map(str, mslist)))
    logger.info('User specified clean mask: ' + str(fitsmask))
    logger.info('Threshold for MakeMask:    ' + str(args['maskthreshold']))
    logger.info('Briggs robust:             ' + str(args['robust']))
        
    for ms in mslist:
        logger.info(' === ' + ms + ' ===')
        logger.info('Telescope:                 ' + args['telescope'])
        if get_telescope_from_ms(ms) == 'EVLA' or get_telescope_from_ms(ms) == 'VLA':
            logger.info('VLA configuration:         ' + get_vla_configuration(ms))
        with table(ms, readonly=True, ack=False) as t:            
            time = np.unique(t.getcol('TIME'))
            logger.info('Integration time [s]:      {:.2f}'.format(np.abs(time[1] - time[0])))
            logger.info('Observation duration [hr]: {:.2f}'.format((np.max(time)-np.min(time))/3600.))
        with table(ms + '::SPECTRAL_WINDOW', readonly=True, ack=False) as t:
            chanw = np.median(t.getcol('CHAN_WIDTH'))
            freqs = t.getcol('CHAN_FREQ')[0]
            nfreq = len(t.getcol('CHAN_FREQ')[0])
            logger.info('Number of channels:        {:d}'.format(nfreq))
            logger.info('Bandwidth [MHz]:           {:.2f}'.format((np.max(freqs)-np.min(freqs))/1e6))
            logger.info('Channel width [kHz]:       {:.2f}'.format(chanw/1e3))
            logger.info('Start frequnecy [MHz]:     {:.2f}'.format(np.min(freqs)/1e6))      
            logger.info('End frequency [MHz]:       {:.2f}'.format(np.max(freqs)/1e6))
        logger.info('================')
    return


def write_facet_directions(catalogfile, freq, facetdirections = 'directions.txt', ds9_region='directions.reg', telescope = 'LOFAR'):
    """
    Writes facet directions to a text file and generates a DS9 region file.
    This function processes a FITS catalog file to extract source information
    (RA and DEC) and writes it to a specified text file in a format suitable
    for self-calibration. It also generates a DS9 region file for visualization.

    Parameters
    ----------
    catalogfile : str
        Path to the FITS catalog file containing source information.
    freq : float
        Frequency in Hz.
    facetdirections : str, optional
        Name of the output text file containing facet directions.
        Default is 'directions.txt'.
    ds9_region : str, optional
        Name of the output DS9 region file. Default is 'directions.reg'.
    telescope : str, optional
        Name of the telescope (e.g., 'LOFAR', 'MeerKAT'). Default is 'LOFAR'.

    Notes
    -----
    Outputs:
    - A text file (`facetdirections`) containing RA, DEC, self-calibration
    cycle, solution intervals, smoothness values, inclusion flags, and
    direction labels for each source.
    - A DS9 region file (`ds9_region`) for visualizing the source positions.
    - The function uses hardcoded values for self-calibration cycles, solution
    intervals, smoothness values, and inclusion flags.
    - The first `N_bright` sources are treated with different solution intervals
    and smoothness values compared to the rest.
    - Beyond `N_normal` sources, a different set of inclusion flags is applied.
    - The function assumes the FITS catalog contains columns 'RA' and 'DEC'.

    Examples
    --------
    write_facet_directions('source_catalog.fits', freq, 'output_directions.txt', 'output_regions.reg')

    """
    hdu_list = fits.open(catalogfile)
    catalog = Table(hdu_list[1].data)
    #print(catalog.columns)
    #print(catalog['Peak_flux'])
    hdu_list.close()
    selfcalcycle = 0 # hard coded at zero is ok
    
    if telescope == 'LOFAR':
        # HBA Dutch settings
        solints_vbright = ["'16sec'", "'5min'","'192sec'","'60min'"]
        solints_bright = ["'32sec'", "'10min'","'192sec'","'60min'"]
        solints = ["'32sec'", "'20min'","'192sec'","'60min'"]
        smoothing_vbright = [10, 2, 35, 10]
        smoothing_bright = [25, 5, 35, 15]
        smoothing = [50, 15, 35, 50]
        #N_bright = 4 # first N_bright get less smoothness and solint
        #N_normal = 45 # beyond this we have pertubative directions
        #N_amps   = 35 # directions for scalarcomplexgain solve
        
        inclusion_flags = [True,True,True,True]
        inclusion_flags_no_amps = [True,False,True,False]
        inclusion_flags_pert = [False,False,True,False]
    elif telescope == 'MeerKAT':
        solints_vbright = ["'32sec'", "'5min'"]
        solints_bright = ["'32sec'", "'10min'"]
        solints = ["'64sec'", "'30min'"]
        smoothing_vbright = [50, 5]
        smoothing_bright = [75 ,20]
        if freq < 1.0e9: # UHF band
            smoothing = [100, 100]
            smoothing_faint = [200,  200]
            solints_faint = ["'128sec'", "'30min'"]
        elif freq<= 1.7e9: # L band
            smoothing = [200, 100]
            smoothing_faint = [300, 200]   
            solints_faint = ["'192sec'", "'30min'"]
        else: # S band
            smoothing = [300, 100]
            smoothing_faint = [400, 200]
            solints_faint = ["'256sec'", "'30min'"]
        
        inclusion_flags = [True,True]
    else:
        raise ValueError('Telescope not supported', telescope)   

    write_ds9_regions(catalog['RA'], catalog['DEC'], filename=ds9_region) 
  
    with open(facetdirections, "w") as f:
        f.write('#RA DEC start solints smoothness soltypelist_includedir\n')
        for source_counter,source in enumerate(catalog):
            if telescope == 'LOFAR':
                # use AFLUX to determine very bright sources
                if source['AFLUX'] > 5.0*((freq/144e6)**(-0.7)): # very bright source
                    smoothnessvals = smoothing_vbright
                    solintvals    = solints_vbright
                    inclusion_flagsvals = inclusion_flags 
                elif source['AFLUX'] > 0.6*((freq/144e6)**(-0.7)): # bright source
                    smoothnessvals = smoothing_bright  
                    solintvals    = solints_bright
                    inclusion_flagsvals = inclusion_flags
                elif source['AFLUX'] > 0.25*((freq/144e6)**(-0.7)): # normal source
                    smoothnessvals = smoothing  
                    solintvals    = solints
                    inclusion_flagsvals = inclusion_flags
                elif source['AFLUX'] > 0.12*((freq/144e6)**(-0.7)): # somewhat faint source (do not solve for scalarcomplex gains, but do solve for phases)
                    smoothnessvals = smoothing  
                    solintvals    = solints
                    inclusion_flagsvals = inclusion_flags_no_amps
                else: # faint source
                    smoothnessvals = smoothing # does not matter because we only do pertubative phases, but keep it the same as normal sources for consistency  
                    solintvals    = solints # does not matter because we only do pertubative phases, but keep it the same as normal sources for consistency
                    inclusion_flagsvals = inclusion_flags_pert
                #if source_counter < N_bright: 
                #    smoothnessvals = smoothing_bright
                #    solintvals    = solints_bright
                #else:
                #    smoothnessvals = smoothing
                #    solintvals = solints
                #if source_counter < N_amps:
                #    inclusion_flagsvals = inclusion_flags # include all solve types
                #elif source_counter >= N_amps and source_counter < N_normal: 
                #    inclusion_flagsvals = inclusion_flags_no_amps # skip amp solving, but include first phases
                #else:
                #    inclusion_flagsvals = inclusion_flags_pert # pertubative phases only
            elif telescope == 'MeerKAT':
                # use AFLUX to determine very bright sources
                if source['AFLUX'] > 0.25*((freq/1.28e9)**(-0.7)): # very bright source
                    smoothnessvals = smoothing_vbright
                    solintvals    = solints_vbright
                elif source['AFLUX'] > 0.05*((freq/1.28e9)**(-0.7)): # bright source
                    smoothnessvals = smoothing_bright  
                    solintvals    = solints_bright
                elif source['AFLUX'] > 0.03*((freq/1.28e9)**(-0.7)): # normal source
                    smoothnessvals = smoothing  
                    solintvals    = solints
                else: # faint source
                    smoothnessvals = smoothing_faint  
                    solintvals    = solints_faint
                inclusion_flagsvals = inclusion_flags # include all solve types         

            line = f"{source['RA']} {source['DEC']} {selfcalcycle} " \
                   f"[{','.join(solintvals)}] " \
                   f"[{','.join(str(s) for s in smoothnessvals)}] " \
                   f"[{','.join(str(i) for i in inclusion_flagsvals)}] " \
                   f"# direction Dir{source_counter:02d}\n"
            f.write(line)


def auto_direction(selfcalcycle=0, freq=150e6, pixelscale=None, imsize=None, telescope=None, imagename=None, idg=None, channelsout=None):
    """
    Automatically determines and processes calibration directions for self-calibration cycles.

    Parameters
    ----------
    selfcalcycle : int, optional
        The self-calibration cycle number (default is 0). Determines the parameters for
        artifact catalog creation and filtering.
    freq : float, optional
        The frequency in Hz (default is 150e6 Hz). Used for solution setting purposes.
    pixelscale : float, optional
        The pixel scale in arcsec (default is None). If provided, it overrides the global `args` dictionary for standalone usage. Default is None.
    imsize : int, optional
        The image size in pixels (default is None). If provided, it overrides the global `args` dictionary for standalone usage. Default is None.
    telescope : str, optional
        The name of the telescope (not used in current implementation). Default is None.
    imagename : str, optional
        The base name of the image file (without cycle number and suffix). If provided,
        it overrides the global `args` dictionary for standalone usage. Default is None.
    idg : bool, optional
        Indicates whether the image is an IDG image (default is None). If provided,
        it overrides the global `args` dictionary for standalone usage.
    channelsout : int, optional
        The number of output channels (default is None). If provided, it overrides
        the global `args` dictionary for standalone usage. Default is None.

    Returns
    -------
    str
        The filename of the facet directions text file generated for the given self-calibration cycle.
        Description:
        This function performs the following steps:
        1. Sets parameters (`keep_N_brightest`, `distance`, `N_dir_max`) based on the self-calibration cycle.
        2. Constructs filenames for input and output files, including error maps, catalogs, and plots.
        3. Generates an error map from the input image.
        4. Creates an artifact sources catalog from the error map.
        5. Updates the artifact catalog by filtering sources based on brightness, distance, and merging with
        the previous catalog.
        6. Writes facet direction files for self-calibration and visualization.
        7. Plots the error map with overlaid artifact directions.

    Notes
    -----
    - The function uses global `args` to retrieve input parameters such as `imagename`, `idg`, and `channelsout`.
    - The error map and artifact catalog are processed using external helper functions:
    `calibration_error_map`, `create_calibration_error_catalog`, `update_calibration_error_catalog`,
    `write_facet_directions`, and `plotimage_astropy`.
    - The plot uses the noise level from the first error map (`imagename000-errormap.fits`) for consistent scaling.
    Dependencies:
    - Requires the `astropy.io.fits` module for handling FITS files.
    - Assumes the presence of helper functions for error map generation, catalog creation, and plotting.
    """
    global args # this is specifically needed here
    # otherwise during compilation it sees the args = {'imagename': imagename} removing the connection to the global args

    # set input parameters for standalone usage
    if imagename is not None:
        args = {'imagename': imagename}
    if idg is not None:
        args |= {'idg': idg}
    if channelsout is not None:
        args |= {'channelsout': channelsout}
    if pixelscale is not None:
        args |= {'pixelscale': pixelscale}    
    if imsize is not None:
        args |= {'imsize': imsize}

    imaged_area = ((args['pixelscale']/3600.)*float(args['imsize']))**2 # in sqaure degrees

    if telescope == 'LOFAR':
        thresh_pix = 7.5
        thresh_isl = 7.5
        min_peakflux = 0.02
        if selfcalcycle == 0:
            keep_N_brightest = 15 
            distance = 20
            N_dir_max = np.max([3, int(np.round((15.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 1:
            keep_N_brightest = 20 
            distance = 20
            N_dir_max = np.max([3, int(np.round((25.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 2:
            keep_N_brightest = 30
            distance = 15
            N_dir_max = np.max([3, int(np.round((35.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 3:
            keep_N_brightest = 40   
            distance = 15
            N_dir_max = np.max([3, int(np.round((45.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 4:
            keep_N_brightest = 40   
            distance = 15
            N_dir_max = np.max([3, int(np.round((45.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 5:
            keep_N_brightest = 50   
            distance = 10
            N_dir_max = np.max([3, int(np.round((60.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 6:
            keep_N_brightest = 60   
            distance = 10
            N_dir_max = np.max([3, int(np.round((70.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 7:
            keep_N_brightest = 70   
            distance = 10
            N_dir_max = np.max([3, int(np.round((80.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 8:
            keep_N_brightest = 80   
            distance = 10
            N_dir_max = np.max([3, int(np.round((90.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle == 9:
            keep_N_brightest = 90   
            distance = 10
            N_dir_max = np.max([3, int(np.round((100.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
        if selfcalcycle >= 10:
            keep_N_brightest = 100   
            distance = 10
            N_dir_max = np.max([3, int(np.round((100.*imaged_area/44.) + 0.5))]) # set N_dir_max proportional to imaged area
    elif telescope == 'MeerKAT':
        
        thresh_pix = 4.5 # MeerKAT smaller artifacts, so use lower thresholds
        thresh_isl = 4.5
        if freq > 1.7e9: # S band
            freq_scaling = 2.
            min_peakflux = 0.002
        elif freq > 1.0e9: # L band
            freq_scaling = 1.5
            if selfcalcycle <=1:
                min_peakflux = 0.02
            elif selfcalcycle == 2:
                min_peakflux = 0.015 # lower min_peakflux for higher cycles because we can go deeper
            else:
                min_peakflux = 0.01 # lower min_peakflux for higher cycles because we can go deeper    
        else: # UHF band
            freq_scaling = 1.0   
            min_peakflux = 0.02 
        if selfcalcycle == 0:
            keep_N_brightest = 5
            distance = (20./freq_scaling) # this assumes a reasonable field of view
            N_dir_max = 3
        if selfcalcycle == 1:
            thresh_pix = 4. # lower thresholds for higher cycles
            thresh_isl = 4.  
            keep_N_brightest = 10 
            distance = (20./freq_scaling)
            N_dir_max = 6
        if selfcalcycle >= 2:
            thresh_pix = 3.5 # lower thresholds for higher cycles
            thresh_isl = 3.5
        if selfcalcycle == 2:
            keep_N_brightest = 15
            distance = (15./freq_scaling)
            N_dir_max = 10
        if selfcalcycle >= 3:
            keep_N_brightest = 20   
            distance = (10./freq_scaling)
            N_dir_max = 15
    else:
        raise Exception('Telescope not supported for auto_directions:', telescope)    
    
    # set image name input to compute error map
    if args['idg']:
        fitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + '-MFS-image.fits'
    else:
        fitsimage = args['imagename'] + str(selfcalcycle).zfill(3) +  '-MFS-image.fits'    
    if args['channelsout'] == 1:
        fitsimage = fitsimage.replace('-MFS', '').replace('-I', '')    
   
    # for standalone running
    # create errormaps_dd directory if it does not exist
    if not os.path.isdir('errormaps_dd'):
        os.mkdir('errormaps_dd')
    # create plots directory if it does not exist
    if not os.path.isdir('plots'):
        os.mkdir('plots')   
    # create directions directory if it does not exist
    if not os.path.isdir('directions'):
        os.mkdir('directions')
    # create facet_regions directory if it does not exist
    if not os.path.isdir('facet_regions'):
        os.mkdir('facet_regions')     

    # set input/output names
    outputerrormap1 =  'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap1.fits'
    outputerrormap2 =  'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap2.fits'
    outputerrormap3 =  'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap3.fits'
    outputcatalog1 =   'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap1.srl.fits'
    outputcatalog2 =   'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap2.srl.fits'
    outputcatalog3 =   'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap3.srl.fits'
    outputcatalog  =   'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap.merged.srl.fits'
    outputcatalog_filtered =  'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap.srl.filtered.fits'
    facetdirections = 'directions/directions_' + str(selfcalcycle).zfill(3) + '.txt'
    directions_reg =  'facet_regions/directions_' + str(selfcalcycle).zfill(3) + '.reg'
    plots_dir = 'plots'
    outplotname1 =  os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap1.png')
    outplotname2 =  os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap2.png')
    outplotname3 =  os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-errormap3.png')
    outputfluxcatalog = 'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + '-compactsource-flux.fits'
    
    with fits.open(fitsimage) as hdul:
        pixsize = 3600. * (hdul[0].header['CDELT2']) # in arcsec
        match_radius = pixsize*31.*3./60.

    if selfcalcycle > 0:
        previous_catalog =  'errormaps_dd/' + os.path.basename(args['imagename']) + str(selfcalcycle-1).zfill(3) + '-errormap.srl.filtered.fits'
        if not os.path.isfile(previous_catalog) or not os.path.isfile('errormaps_dd/' + os.path.basename(args['imagename']) + str(0).zfill(3) + '-errormap1.fits'):
            terminal_print('One of both of these files are missing:',previous_catalog, 'errormaps_dd/' + os.path.basename(args['imagename']) + str(0).zfill(3) + '-errormap1.fits')    
            raise Exception('Missing files')
    else:
        previous_catalog = None  
    
    # create compact source catalog from which we can get peak fluxes
    write_compactsource_flux(fitsimage, outputfluxcatalog)
    # add AFLUX and TFLUX columns to error catalog
    add_peak_total_flux_to_catalog(outputfluxcatalog, outputfluxcatalog, match_radius=-1) # negative match radius means just copy over Peak_flux and Total_flux without matching

    # make the error map
    terminal_print('Making artifact maps from:', fitsimage)

    # compute the calibration error map, run 1 with small kernel to pick up small scale errors
    calibration_error_map(fitsimage,  outputerrormap1, kernelsize=31, rebin=31)
    # compute the calibration error map, run 2 with medium kernel to pick up medium scale errors
    calibration_error_map(fitsimage,  outputerrormap2, kernelsize=31*2, rebin=31*2)
    # compute the calibration error map, run 3 with large kernel to pick up large scale errors
    calibration_error_map(fitsimage,  outputerrormap3, kernelsize=31*4, rebin=31*4)

    # create the artifact sources catalog
    empty_catalog1 = create_calibration_error_catalog(outputerrormap1, outputcatalog1, thresh_pix=thresh_pix, thresh_isl=thresh_isl)
    empty_catalog2 = create_calibration_error_catalog(outputerrormap2, outputcatalog2, thresh_pix=max([thresh_pix-0.5,3.5]), thresh_isl=max([thresh_pix-0.5,3.5]))
    empty_catalog3 = create_calibration_error_catalog(outputerrormap3, outputcatalog3, thresh_pix=max([thresh_pix-0.5,3.5]), thresh_isl=max([thresh_pix-0.5,3.5]))

    # create list of catalogs to merge
    catalog_list = []
    if not empty_catalog1:
        add_peak_total_flux_to_catalog(outputcatalog1, outputfluxcatalog, match_radius=match_radius) # add peak fluxes to catalog for filtering later on
        catalog_list.append(outputcatalog1)
    if not empty_catalog2:
        add_peak_total_flux_to_catalog(outputcatalog2, outputfluxcatalog, match_radius=match_radius) # add peak fluxes to catalog for filtering later on
        catalog_list.append(outputcatalog2)
    if not empty_catalog3:
        add_peak_total_flux_to_catalog(outputcatalog3, outputfluxcatalog, match_radius=match_radius) # add peak fluxes to catalog for filtering later on
        catalog_list.append(outputcatalog3)
    if len(catalog_list) > 1:
        terminal_print('Merging artifact source catalogs')
        merge_catalogs(catalog_list, outputcatalog)
        # remove duplicates and sources that are close to each other
        update_calibration_error_catalog(outputcatalog, outputcatalog, distance=distance, keep_N_brightest=keep_N_brightest, N_dir_max=N_dir_max)
    if len(catalog_list) == 1:
        terminal_print('Only one artifact source catalog found, using this one directly')
        shutil.copy(catalog_list[0], outputcatalog)

    if len(catalog_list) == 0:
        terminal_print('No artifact sources found in any catalog')
        empty_catalog = True
    else:
        empty_catalog = False

    # check if just one/two sources were found in catalog in selfcalcycle 0
    if selfcalcycle == 0 and not empty_catalog:
        if get_number_of_sources_in_catalog(outputcatalog) == 1:
            # we will add two extra sources to the catalog, at the location of a bright source in the image
            terminal_print('\033[93m' + 'Only one artifact source found, adding two extra sources at the location of a bright source in the outputfluxcatalog' + '\033[0m')
            add_source_to_catalog(outputcatalog, outputfluxcatalog, distance=distance)
            add_source_to_catalog(outputcatalog, outputfluxcatalog, distance=distance)
        elif get_number_of_sources_in_catalog(outputcatalog) == 2:   
            # we will add an third extra source to the catalog, at the location of a bright source in the image
            terminal_print('\033[93m' + 'Only two artifact sources found, adding an extra source at the location of a bright source in the outputfluxcatalog' + '\033[0m')
            add_source_to_catalog(outputcatalog, outputfluxcatalog, distance=distance)
        # in principle we can repeat adding sources, for now stick to one/two extra sources         
   
    #if False: # set to True for testing so see if add_source_to_catalog works 
    #    add_source_to_catalog(outputcatalog, outputfluxcatalog, distance=distance)  

    # add very bright sources to catalog to ensure they are included
    if selfcalcycle > 0 and empty_catalog: # do nothing since no no artifact sources are found (maybe the image is close to perfect)
       terminal_print('No artifact sources found, it seems the image quality is very good') # do nothing because outputcatalog does not exist, so we cannot add bright sources to it, but this is also not needed since the image quality is already very good
    else:
       add_bright_source_to_catalog(outputcatalog, outputfluxcatalog, freq)

    # update the artifact catalog (keep only N-brightest, remove closely seperated sources, merge with previous catalog)
    if not empty_catalog: 
        update_calibration_error_catalog(outputcatalog,outputcatalog_filtered, distance=distance, keep_N_brightest=keep_N_brightest, previous_catalog=previous_catalog, N_dir_max=N_dir_max)
    elif previous_catalog is not None: # this can happend when the calibration is already very good and no new artifact sources are found
        terminal_print('No new sources found, copying previous catalog to current cycle')
        if os.path.isfile(outputcatalog_filtered): # delete existing filtered catalog if it exists
            Path(outputcatalog_filtered).unlink(missing_ok=True)
        shutil.copy(previous_catalog, outputcatalog_filtered)
    else:
        # print in green text
        terminal_print('\033[92m' + 'No artefact sources found on first image, cannot continue, there is no need for DDE calibration' + '\033[0m')
        sys.exit(0)

    # find peak fluxes from compact source catalog and add to filtered catalog (put in AFLUX column)
    # add_peak_total_flux_to_catalog(outputcatalog_filtered, outputfluxcatalog, match_radius=match_radius)

    # filter out sources too low peak flux
    filter_catalog_on_flux(outputcatalog_filtered, freq, telescope, min_peakflux=min_peakflux)

    # write facet direction file (for facetselfcal), also output directions.reg for visualization
    write_facet_directions(outputcatalog_filtered, freq, facetdirections=facetdirections, 
                           ds9_region=directions_reg, telescope=telescope)

    # plot, # hardcode minmax to always usage image000 so it is easier to compare
    with fits.open('errormaps_dd/' + os.path.basename(args['imagename']) + str(0).zfill(3) + '-errormap1.fits') as hdulist:
        imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
        plotminmax = [-2.*imagenoise, 35.*imagenoise]
    plotimage_astropy(outputerrormap1, outplotname1, mask=None, regionfile=directions_reg, regioncolor='red', minmax=plotminmax, regionalpha=1.0)

    # plot, # hardcode minmax to always usage image000 so it is easier to compare
    with fits.open('errormaps_dd/' + os.path.basename(args['imagename']) + str(0).zfill(3) + '-errormap2.fits') as hdulist:
        imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
        plotminmax = [-2.*imagenoise, 35.*imagenoise]
    plotimage_astropy(outputerrormap2, outplotname2, mask=None, regionfile=directions_reg, regioncolor='red', minmax=plotminmax, regionalpha=1.0)
        # plot, # hardcode minmax to always usage image000 so it is easier to compare
    with fits.open('errormaps_dd/' + os.path.basename(args['imagename']) + str(0).zfill(3) + '-errormap3.fits') as hdulist:
        imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
        plotminmax = [-2.*imagenoise, 35.*imagenoise]
    plotimage_astropy(outputerrormap3, outplotname3, mask=None, regionfile=directions_reg, regioncolor='red', minmax=plotminmax, regionalpha=1.0)
    return facetdirections


def tmpmakeantresidual(mslist, selfcalcycle, multiscale, fitsmask_list, restoringbeam, \
                       automaskthreshold_selfcalcycle, wsclean_h5list, facetregionfile):
    """
    Create an antenna-residual image for a self-calibration cycle.

    Parameters
    ----------
    mslist : list of str
        Measurement Sets to image.
    selfcalcycle : int
        Self-calibration cycle number.
    multiscale : bool
        Whether to use multiscale deconvolution.
    fitsmask_list : list
        FITS masks used for imaging.
    restoringbeam : str or None
        Restoring-beam setting passed to the imager.
    automaskthreshold_selfcalcycle : list
        Per-cycle automask thresholds.
    wsclean_h5list : list
        H5 solutions used for prediction.
    facetregionfile : str
        Facet-region file for DDE imaging.

    Returns
    -------
    None
        Residual images are written to disk.
    """
    #msname = '44_101_23sep2023_b4_gwb.ms.hypergiant.copy.subtracted.avg'
    #cmdwsclean = 'wsclean -no-update-model-required -minuv-l 10.0 -size 5736 5736 -reorder -weight briggs 0.0 -parallel-reordering 4 -mgain 0.75 -data-column RESIDUAL_DATA  -join-channels -channels-out 8 -parallel-gridding 6 -fit-spectral-pol 5 -pol i -gridder wgridder -wgridder-accuracy 0.0001 -no-min-grid-resolution -facet-regions facet_regions/facets.reg -apply-facet-solutions merged_selfcalcycle008_44_101_23sep2023_b4_gwb.ms.hypergiant.copy.subtracted.avg.h5 amplitude000,phase000 -diagonal-visibilities -name imageDD_009 -scale 0.75arcsec -nmiter 1 -niter 1'

    # full residual
    #os.system(cmdwsclean + ' -name imageDD_009_residualfull ' + msname)
    makeimage(mslist, args['imagename'] + str(selfcalcycle).zfill(3) + '_residual_full',
                      args['pixelscale'], args['imsize'],
                      args['channelsout'], args['niter'], args['robust'],
                      multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[0],
                      uvminim=args['uvminim'], predict=not args['stopafterskysolve'],
                      fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                      restoringbeam=restoringbeam, automask=automaskthreshold_selfcalcycle[selfcalcycle],
                      removenegativecc=args['removenegativefrommodel'],
                      paralleldeconvolution=args['paralleldeconvolution'],
                      parallelgridding=args['parallelgridding'],
                      h5list=wsclean_h5list, forced_imcol='RESIDUAL_DATA',
                      facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                      stack=args['stack'],
                      disable_primarybeam_image=args['disable_primary_beam'],
                      disable_primarybeam_predict=args['disable_primary_beam'],
                      fulljones_h5_facetbeam=not args['single_dual_speedup'])
 
    # collect all antenna names in all MS, append to antenna_names
    antenna_names = []
    for ms in mslist:
        ant_table = table(ms+'::ANTENNA', ack=False)
        antenna_names.extend(ant_table.getcol('NAME'))
        ant_table.close()
    
    terminal_print('Antenna names:', antenna_names)
    for ant in antenna_names:
        for ms in mslist:
            if os.path.isdir(ms + '_' + ant):
                shutil.rmtree(ms + '_' + ant, ignore_errors=True)
            shutil.copytree(ms, ms + '_' + ant)    
            # flag atenna
            flag_antenna_taql(ms + '_'  + ant, [ant])
        
        
        #os.system(cmdwsclean + ' -name imageDD_009_residual_' + ant + ' ' + msname + '_' + ant)
        makeimage(mslist, args['imagename'] + str(selfcalcycle).zfill(3) + '_residual_' + ant,
                      args['pixelscale'], args['imsize'],
                      args['channelsout'], args['niter'], args['robust'],
                      multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[0],
                      uvminim=args['uvminim'], predict=not args['stopafterskysolve'],
                      fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                      restoringbeam=restoringbeam, automask=automaskthreshold_selfcalcycle[selfcalcycle],
                      removenegativecc=args['removenegativefrommodel'],
                      paralleldeconvolution=args['paralleldeconvolution'],
                      parallelgridding=args['parallelgridding'],
                      h5list=wsclean_h5list, forced_imcol='RESIDUAL_DATA',
                      facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                      stack=args['stack'],
                      disable_primarybeam_image=args['disable_primary_beam'],
                      disable_primarybeam_predict=args['disable_primary_beam'],
                      fulljones_h5_facetbeam=not args['single_dual_speedup'])
        for ms in mslist:
            shutil.rmtree(ms + '_' + ant, ignore_errors=True)
        # remove psf, residual images, model images, dirty, and beam images
        stackstr = ''
        clean_up_images(args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '_residual_' + ant, model=True)


        #run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,0]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        #run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,3]=(0.5+0i)'")  # because I = RR+LL/2 (this is tricky because we work with phase diff)
        #run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,1]=(0+0i)'")
        #run("taql" + " 'update " + ms + " set MODEL_DATA_PDIFF[,2]=(0+0i)'")
        # code above is not sisco proof, as it cannot write to one index at a time so do it in one go:


def check_frequency_averaging_steps(mslist, options):
    """Plan and validate frequency averaging before any DP3 average step."""
    global args

    preview_options = copy.deepcopy(options)
    backup_will_run = not options["skipbackup"]
    backup_will_create_copy = backup_will_run and options["start"] == 0
    setup_mslist = mslist
    if options["start"] > 0 and backup_will_run:
        setup_mslist = [os.path.basename(ms) + ".copy" for ms in mslist]

    backup_nchan = options["msinnchan"] if backup_will_create_copy else None
    backup_startchan = options["msinstartchan"] if backup_will_create_copy else 0
    spectral_window_metadata = None
    if backup_nchan is not None:
        with table(mslist[0] + "::SPECTRAL_WINDOW", readonly=True, ack=False) as spw_table:
            preview_freqs = np.asarray(spw_table.getcol("CHAN_FREQ")[0], dtype=float)
            preview_widths = np.asarray(spw_table.getcol("CHAN_WIDTH")[0], dtype=float)
        if (
            backup_nchan > 0
            and backup_startchan >= 0
            and backup_startchan + backup_nchan <= len(preview_freqs)
        ):
            spectral_window_metadata = (
                preview_freqs[backup_startchan:backup_startchan + backup_nchan],
                preview_widths[backup_startchan:backup_startchan + backup_nchan],
            )

    if backup_will_run:
        preview_options["msinntimes"] = None
        preview_options["msinstarttimeslot"] = None
        preview_options["msinnchan"] = None
        preview_options["msinstartchan"] = 0
        preview_options["removeinternational"] = False
        preview_options["removemostlyflaggedstations"] = False
        preview_options["phaseshiftbox"] = None
        preview_options["flag_antenna_list"] = None
        if options["aoflagger"] and options["aoflaggerbeforeavg"]:
            preview_options["aoflagger"] = False

    original_args = args
    args = preview_options
    try:
        setup_result = basicsetup(
            setup_mslist,
            spectral_window_metadata=spectral_window_metadata,
            create_directories=False,
        )
        resolved_options = copy.deepcopy(args)
    finally:
        args = original_args
    lba = setup_result[1]

    remove_step = options["remove_outside_center_avgfreqstep"]
    check_remove_step = options["remove_outside_center"]
    if check_remove_step and (
        isinstance(remove_step, bool)
        or not isinstance(remove_step, int)
        or remove_step < 1
    ):
        raise ValueError(
            "--remove-outside-center-avgfreqstep must be a positive integer"
        )

    requested_step = options["avgfreqstep"]
    if isinstance(requested_step, bool):
        raise ValueError("--avgfreqstep must be a positive integer or frequency resolution")
    if isinstance(requested_step, int) and requested_step < 0:
        raise ValueError("--avgfreqstep cannot be negative")

    planned_steps = []
    for ms in setup_mslist:
        with table(ms, readonly=True, ack=False) as ms_table:
            active_ddids = np.unique(ms_table.getcol("DATA_DESC_ID"))
        with table(ms + "::DATA_DESCRIPTION", readonly=True, ack=False) as dd_table:
            spw_ids_by_ddid = dd_table.getcol("SPECTRAL_WINDOW_ID")
        active_spw_ids = sorted({
            int(spw_ids_by_ddid[int(ddid)]) for ddid in active_ddids
        })
        with table(ms + "::SPECTRAL_WINDOW", readonly=True, ack=False) as spw_table:
            channel_counts = np.asarray(spw_table.getcol("NUM_CHAN")).reshape(-1)
            channel_freqs = spw_table.getcol("CHAN_FREQ")
            channel_widths = spw_table.getcol("CHAN_WIDTH")

        if not active_spw_ids:
            raise ValueError("Cannot determine active spectral windows in {!r}".format(ms))

        def rows(values):
            values = np.asarray(values)
            if values.dtype == object:
                return [np.asarray(row, dtype=float).reshape(-1) for row in values]
            if values.ndim == 1:
                return [np.asarray(values, dtype=float).reshape(-1)]
            return [np.asarray(row, dtype=float).reshape(-1) for row in values]

        freqs_by_spw = rows(channel_freqs)
        widths_by_spw = rows(channel_widths)
        backup_freqs = {}
        backup_widths = {}
        if backup_will_create_copy and backup_nchan is not None:
            for spw_id in active_spw_ids:
                available = int(channel_counts[spw_id])
                if (
                    backup_nchan <= 0
                    or backup_startchan < 0
                    or backup_startchan + backup_nchan > available
                ):
                    raise ValueError(
                        "--msinnchan={} with --msinstartchan={} exceeds the {} "
                        "available channels in SPW {} of {!r}".format(
                            backup_nchan, backup_startchan, available, spw_id, ms
                        )
                    )
                backup_freqs[spw_id] = freqs_by_spw[spw_id][
                    backup_startchan:backup_startchan + backup_nchan
                ]
                backup_widths[spw_id] = widths_by_spw[spw_id][
                    backup_startchan:backup_startchan + backup_nchan
                ]

        requested_nchan = resolved_options["msinnchan"]
        start_channel = resolved_options["msinstartchan"]
        if (
            backup_will_run
            and not backup_will_create_copy
            and options["msinnchan"] is not None
        ):
            # The preview clears input selection, but the requested selection
            # still applies when validating the channels used by averaging.
            requested_nchan = options["msinnchan"]
            start_channel = options["msinstartchan"]
        selected_metadata = {}

        for spw_id in active_spw_ids:
            available = int(channel_counts[spw_id])
            if spw_id >= len(freqs_by_spw) or spw_id >= len(widths_by_spw):
                raise ValueError("Missing channel metadata for SPW {} in {!r}".format(spw_id, ms))
            frequencies = freqs_by_spw[spw_id][:available]
            widths = widths_by_spw[spw_id][:available]
            if len(frequencies) != available or len(widths) != available:
                raise ValueError("Incomplete channel metadata for SPW {} in {!r}".format(spw_id, ms))

            base_frequencies = backup_freqs.get(spw_id, frequencies)
            base_widths = backup_widths.get(spw_id, widths)
            if requested_nchan is None:
                selected_count = len(base_frequencies)
                selected_frequencies = base_frequencies
                selected_widths = base_widths
            else:
                if (
                    requested_nchan <= 0
                    or start_channel < 0
                    or start_channel + requested_nchan > len(base_frequencies)
                ):
                    raise ValueError(
                        "--msinnchan={} with --msinstartchan={} exceeds the {} "
                        "available channels in SPW {} of {!r}".format(
                            requested_nchan, start_channel, len(base_frequencies), spw_id, ms
                        )
                    )
                selected_count = requested_nchan
                selected_frequencies = base_frequencies[
                    start_channel:start_channel + requested_nchan
                ]
                selected_widths = base_widths[
                    start_channel:start_channel + requested_nchan
                ]

            selected_metadata[spw_id] = (
                len(base_frequencies),
                selected_count,
                base_frequencies,
                base_widths,
                selected_frequencies,
                selected_widths,
            )

        if (
            requested_step is None
            and resolved_options["autofrequencyaverage"]
            and not lba
        ):
            first_spw = active_spw_ids[0]
            metadata = selected_metadata[first_spw]
            input_count = requested_nchan or metadata[0]
            smear = bandwidthsmearing(
                float(np.median(metadata[3])),
                float(np.min(metadata[2])),
                float(resolved_options["imsize"]),
                verbose=False,
            )
            step = 0
            for factor in range(2, 21):
                if smear < (1.0 / factor) and input_count % factor == 0:
                    step = factor
        elif requested_step is not None:
            step = requested_step
        else:
            step = 1 if resolved_options["aoflagger"] else 0

        planned_steps.append(step)

        for spw_id in active_spw_ids:
            _, input_count, _, _, _, selected_widths = selected_metadata[spw_id]
            if isinstance(step, str) and not step.strip().isdigit():
                match = re.fullmatch(
                    r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(Hz|kHz|MHz)\s*",
                    step,
                )
                if match is None:
                    raise ValueError(
                        "Cannot validate frequency resolution {!r} from --avgfreqstep".format(step)
                    )
                resolution = float(match.group(1)) * {
                    "Hz": 1.0,
                    "kHz": 1e3,
                    "MHz": 1e6,
                }[match.group(2)]
                channel_width = abs(float(selected_widths[0]))
                if channel_width == 0:
                    raise ValueError("Zero channel width in SPW {} of {!r}".format(spw_id, ms))
                factor = max(1, int(resolution / channel_width + 0.5))
            else:
                factor = int(step)
                if factor < 0:
                    raise ValueError("--avgfreqstep cannot be negative")
                factor = max(1, factor)

            factor = min(factor, input_count)
            if input_count % factor != 0:
                raise ValueError(
                    "Frequency averaging for {!r} would leave a non-integer "
                    "channel count: SPW {} has {} selected channels and "
                    "--avgfreqstep={!r} resolves to a factor of {}".format(
                        ms, spw_id, input_count, step, factor
                    )
                )

            output_count = input_count // factor
            if check_remove_step:
                extraction_factor = min(remove_step, output_count)
                if output_count % extraction_factor != 0:
                    raise ValueError(
                        "--remove-outside-center-avgfreqstep={} cannot evenly "
                        "average the {} channels left in SPW {} of {!r} after "
                        "--avgfreqstep={!r} (factor {})".format(
                            remove_step, output_count, spw_id, ms, step, factor
                        )
                    )

    return planned_steps


def create_residual_data_column(mslist, imagebasename, pixsize, imsize,
                       channelsout, single_dual_speedup=True,
                       outcol='RESIDUAL_DATA', dysco=True,
                       idg=False, h5list=[], facetregionfile=None,
                       disable_primary_beam=False, parallelgridding=1):
    """
    Create residual-data products for the supplied Measurement Sets.

    Parameters
    ----------
    mslist : list
        Measurement Set paths.
    imagebasename : str
        Base name for generated images.
    pixsize : float
        Image pixel scale.
    imsize : int
        Image size in pixels.
    channelsout : int
        Number of output channels.
    single_dual_speedup : bool, optional
        Use the single/dual-polarization prediction speedup.
    outcol : str, optional
        Name of the output residual-data column.
    dysco : bool, optional
        Store a newly created output column using Dysco compression.
    idg : bool, optional
        Use the Image Domain Gridder for prediction.
    h5list : list of str, optional
        H5Parm solution paths; a non-empty list enables direction-dependent prediction.
    facetregionfile : str or None, optional
        Facet region file used for direction-dependent prediction.
    disable_primary_beam : bool, optional
        Disable primary-beam correction during prediction.
    ddcor : bool, optional
        Apply direction-dependent correction to extracted data.
    modelstoragemanager : str or None, optional
        WSClean model storage manager used for prediction.
    parallelgridding : int, optional
        Number of parallel gridding processes.
    metadata_compression : bool, optional
        Compress Measurement Set metadata when creating derived data.

    Returns
    -------
    None
        Results are written to disk.
    """
    # get imageheader to check frequency
    stepsize = 100000
    if len(h5list) != 0:
        datacolumn = 'DATA'  # for DDE
    else:
        datacolumn = 'CORRECTED_DATA'
 
    # predict the model non-DDE case
    if len(h5list) == 0:
        makeimage(mslist, imagebasename, pixsize, imsize,
                  channelsout, onlypredict=True,
                  idg=idg, disable_primarybeam_predict=disable_primary_beam,
                  fulljones_h5_facetbeam=not single_dual_speedup, 
                  parallelgridding=parallelgridding)
    else:  # so this we are in DDE mode as h5list is not empty
        makeimage(mslist, imagebasename, pixsize, imsize,
                  channelsout, onlypredict=True, squarebox='keepall',
                  idg=idg, h5list=h5list, facetregionfile=facetregionfile,
                  disable_primarybeam_predict=disable_primary_beam,
                  fulljones_h5_facetbeam=not single_dual_speedup, 
                  parallelgridding=parallelgridding)
     
    # write new data column with residuals
    for ms in mslist:
        # check if outcol exists, if not create it with DP3
        with table(ms) as t:
            colnames = t.colnames()
        if outcol not in colnames:
            cmd = 'DP3 msin=' + ms + ' msout=. steps=[] msout.datacolumn=' + outcol + ' '
            cmd += 'msin.datacolumn=' + datacolumn + ' '
            if dysco:
                cmd += 'msout.storagemanager=dysco '
                cmd += 'msout.storagemanager.weightbitrate=16 '
            terminal_print('Command:', cmd)
            run(cmd)
        t = table(ms, readonly=False)
        if t.nrows() < stepsize: stepsize = t.nrows()
        for row in range(0, t.nrows(), stepsize):
            terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
            data = t.getcol(datacolumn, startrow=row, nrow=stepsize, rowincr=1)
            model = t.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
            t.putcol(outcol, data - model, startrow=row, nrow=stepsize, rowincr=1)
        t.close()
    return


# losotolofarbeam('P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive_templatejones.h5', 'amplitude000', 'P214+55_PSZ2G098.44+56.59.dysco.sub.shift.avg.weights.ms.archive', inverse=False, useElementResponse=False, useArrayFactor=True, useChanFreq=True)


# flagms_startend('P215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archive','phaseonlyP215+50_PSZ2G089.52+62.34.dysco.sub.shift.avg.weights.ms.archivesolsgrid_9.h5', 2)


# removestartendms('P219+50_PSZ2G084.10+58.72.dysco.sub.shift.avg.weights.ms.archive',endtime='16-Apr-2015/02:14:47.0')
# removestartendms('P223+50_PSZ2G084.10+58.72.dysco.sub.shift.avg.weights.ms.archive',starttime='24-Feb-2015/22:16:00.0')
# removestartendms('P223+52_PSZ2G088.98+55.07.dysco.sub.shift.avg.weights.ms.archive',starttime='19-Feb-2015/22:40:00.0')
# removestartendms('P223+55_PSZ2G096.14+56.24.dysco.sub.shift.avg.weights.ms.archive',starttime='31-Mar-2015/20:11:00.0')
# removestartendms('P227+53_PSZ2G088.98+55.07.dysco.sub.shift.avg.weights.ms.archive',starttime='19-Feb-2015/22:40:00.0')


def prepare_DDE(imagebasename, selfcalcycle, mslist,
                DDE_predict='DP3', restart=False, disable_IDG_DDE_predict=True,
                telescope='LOFAR', skyview=None, wscleanskymodel=None,
                skymodel=None):
    """
    Prepare direction-dependent calibration and imaging inputs.

    Parameters
    ----------
    imagebasename : str
        Base image name.
    selfcalcycle : int
        Current self-calibration cycle.
    mslist : list
        Measurement Set paths.
    DDE_predict : str, optional
        Prediction method used for direction-dependent calibration.
    restart : bool, optional
        Reuse products from the preceding cycle when preparing the solve.
    disable_IDG_DDE_predict : bool, optional
        Disable IDG for direction-dependent prediction.
    telescope : str, optional
        Telescope used to select beam and imaging behavior.
    skyview : str or None, optional
        External sky-view FITS image used to initialize the facets.
    wscleanskymodel : str or None, optional
        WSClean model-image basename used as the initial model.
    skymodel : str or None, optional
        Sky-model file used as the initial model.

    Returns
    -------
    tuple
        Five values: model-data column names, DDE sky model, solution intervals,
        smoothness constraints, and per-direction solve-selection flags.
    """
    if telescope == 'LOFAR' and not disable_IDG_DDE_predict:
        idg = True  # predict WSCLEAN with beam using IDG (wsclean facet mode with h5 is not efficient here)
    else:
        idg = False

   
    solints, smoothness, soltypelist_includedir = create_facet_directions(imagebasename, selfcalcycle, args=args,
                                                              targetFlux=args['targetFlux'], ms=mslist[0], imsize=args['imsize'],
                                                              pixelscale=args['pixelscale'], numClusters=args['Nfacets'],
                                                              facetdirections=args['facetdirections'], restart=restart)
    
    # testing
    #if solints is not None and isinstance(args['facetdirections'], str):
    #    soltypelist_includedir = np.swapaxes(np.array([soltypelist_includedir] * len(mslist)), 1, 0).T.tolist()
    #print(soltypelist_includedir.tolist())
    #print(type(soltypelist_includedir))
    #print((np.array(soltypelist_includedir)).shape)
    #sys.exit()
    
    # --- start CREATE fits_images/facets.fits -----
    # remove previous fits_images/facets.fits if needed
    if os.path.isfile('fits_images/facets.fits'):
        Path('fits_images/facets.fits').unlink(missing_ok=True)
    if skyview == None:
        if not restart and wscleanskymodel is None and skymodel is None:
            shutil.copy(imagebasename + str(selfcalcycle).zfill(3) + '-MFS-image.fits', 'fits_images/facets.fits')
        if not restart and wscleanskymodel is not None:
            create_empty_fitsimage(mslist[0], int(args['imsize']), float(args['pixelscale']), 'fits_images/facets.fits')
        if not restart and skymodel is not None:
            create_empty_fitsimage(mslist[0], int(args['imsize']), float(args['pixelscale']), 'fits_images/facets.fits')
    else:
        shutil.copy(skyview, 'fits_images/facets.fits')

    if restart:  # in that case we also have a previous image avaialble
        shutil.copy(imagebasename + str(selfcalcycle - 1).zfill(3) + '-MFS-image.fits', 'fits_images/facets.fits')

    # --- end CREATE fits_images/facets.fits -----

    # FILL in fits_images/facets.fits with values, every facets get a constant value, for lsmtool
    hdu = fits.open('fits_images/facets.fits')
    hduflat = flatten(hdu)
    region = pyregion.open('facet_regions/facets.reg')
    dirofinput = os.path.dirname('facet_regions/facets.reg')
    if dirofinput == '':
        dirofinput = '.'
    
    for facet_id, facet in enumerate(region):
        region[facet_id:facet_id + 1].write(dirofinput + '/facet' + str(facet_id) + '.reg')  # split facet from region file
        r = pyregion.open(dirofinput + '/facet' + str(facet_id) + '.reg')
        terminal_print('Filling fits_images/facets.fits with:', dirofinput + '/facet' + str(facet_id) + '.reg')
        manualmask = r.get_mask(hdu=hduflat)
        if len(hdu[0].data.shape) == 4:
            hdu[0].data[0][0][np.where(manualmask == True)] = facet_id
        else:
            hdu[0].data[np.where(manualmask == True)] = facet_id
    hdu.writeto('fits_images/facets.fits', overwrite=True)

    if restart:
        # restart with DDE_predict=DP3 because then only the variable modeldatacolumns is made
        # So the wsclean predict step is skipped in makeimage but variable modeldatacolumns is created
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facet_regions/facets.reg',
                                     DDE_predict='DP3',
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        # selfcalcycle-1 because makeimage has not yet produced an image at this point
        if args['fitspectralpol'] > 0 and DDE_predict == 'DP3':
            dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle - 1).zfill(3) + '-sources.txt', 'fits_images/facets.fits')
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off
    elif skyview is not None:
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facet_regions/facets.reg',
                                     DDE_predict=DDE_predict,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        if args['fitspectralpol'] > 0:
            dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle).zfill(3) + '-sources.txt', 'fits_images/facets.fits')  # imagebasename
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off

    elif wscleanskymodel is not None:  # DDE wscleanskymodel solve at the start

        nonpblist = glob.glob(wscleanskymodel + '-????-model.fits')
        pblist = glob.glob(wscleanskymodel + '-????-model-pb.fits')

        if len(pblist) > 0:
            channelsout_forpredict = len(pblist)
        else:
            channelsout_forpredict = len(nonpblist)

        modeldatacolumns = makeimage(mslist, wscleanskymodel,
                                     args['pixelscale'], args['imsize'], channelsout_forpredict, predict=True,
                                     onlypredict=True, facetregionfile='facet_regions/facets.reg',
                                     DDE_predict='WSCLEAN', idg=idg,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        # assume there is no model for DDE wscleanskymodel solve at the start
        # since we are making image000 afterwards anyway setting a dummy now is ok
        dde_skymodel = 'dummy.skymodel'
        terminal_print('Model data columns:', modeldatacolumns)
    else:
        modeldatacolumns = makeimage(mslist, imagebasename + str(selfcalcycle).zfill(3),
                                     args['pixelscale'], args['imsize'], args['channelsout'], predict=True,
                                     onlypredict=True, facetregionfile='facet_regions/facets.reg',
                                     DDE_predict=DDE_predict, idg=idg,
                                     disable_primarybeam_image=args['disable_primary_beam'],
                                     disable_primarybeam_predict=args['disable_primary_beam'],
                                     fulljones_h5_facetbeam=not args['single_dual_speedup'], parallelgridding=args['parallelgridding'], selfcalcycle=selfcalcycle)
        if args['fitspectralpol'] > 0 and DDE_predict == 'DP3':
            dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle).zfill(3) + '-sources.txt', 'fits_images/facets.fits')
        else:
            dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off
    # check if -pb version of source list exists
    # needed because image000 does not have a pb version as no facet imaging is used, however, if IDG is used it does exist and hence the following does also handfle that
    if telescope == 'LOFAR' and wscleanskymodel is None:  # not for MeerKAT because WSClean still has a bug if no primary beam is used, for now assume we do not use a primary beam for MeerKAT
        if os.path.isfile('fits_images/' + imagebasename + str(selfcalcycle).zfill(3) + '-sources-pb.txt'):
            if args['fitspectralpol'] > 0:
                dde_skymodel = groupskymodel(imagebasename + str(selfcalcycle).zfill(3) + '-sources-pb.txt',
                                             'fits_images/facets.fits')
            else:
                dde_skymodel = 'dummy.skymodel'  # no model exists if spectralpol is turned off

    # to create the correct dimensions in case args['facetdirections'] is a string and not a list (in that case the same solints, smoothness and soltypelist_includedir is used for all directions)
    if isinstance(args['facetdirections'], str):
        if solints is not None:
            solints = np.swapaxes(np.array([solints] * len(mslist)), 1, 0).T.tolist()
        if smoothness is not None:
            smoothness = np.swapaxes(np.array([smoothness] * len(mslist)), 1, 0).T.tolist()
        if soltypelist_includedir is not None:
            soltypelist_includedir = np.swapaxes(np.array([soltypelist_includedir] * len(mslist)), 1, 0).T.tolist()
    if isinstance(args['facetdirections'], list):
        if solints is not None:
            solints = solints.tolist()
        if smoothness is not None:
            smoothness = smoothness.tolist()
        if soltypelist_includedir is not None:
            soltypelist_includedir = soltypelist_includedir.tolist()

    return modeldatacolumns, dde_skymodel, solints, smoothness, soltypelist_includedir


# this version corrupts the MODEL_DATA column
def calibrateandapplycal(mslist, selfcalcycle, solint_list, nchan_list,
                         soltypecycles_list, smoothnessconstraint_list,
                         smoothnessreffrequency_list, smoothnessspectralexponent_list,
                         smoothnessrefdistance_list,
                         antennaconstraint_list, resetsols_list, resetdir_list, normamps_list,
                         BLsmooth_list,
                         solve_msinnchan_list, solve_msinstartchan_list,
                         antenna_averaging_factors_list, antenna_smoothness_factors_list,
                         soltypelist_includedir,
                         normamps=False, normamps_per_ms=False, skymodel=None,
                         predictskywithbeam=False,
                         longbaseline=False,
                         skymodelsource=None,
                         skymodelpointsource=None, wscleanskymodel=None, skymodelsetjy=False,
                         mslist_beforephaseup=None,
                         modeldatacolumns=[], dde_skymodel=None,
                         DDE_predict='WSCLEAN', telescope='LOFAR',
                         mslist_beforeremoveinternational=None):
    """
    Solve and apply calibration solutions to Measurement Sets.

    Parameters
    ----------
    mslist : list
        Measurement Set paths.
    selfcalcycle : int
        Current self-calibration cycle.
    solint_list : list
        Solution intervals.
    nchan_list : list
        Solution channel counts.
    soltypecycles_list : list
        Cycles at which each configured solution type is solved.
    smoothnessconstraint_list : list
        Spatial smoothness constraints for each solution type and Measurement Set.
    smoothnessreffrequency_list : list
        Reference frequencies for the smoothness constraints.
    smoothnessspectralexponent_list : list
        Spectral exponents used to scale smoothness constraints.
    smoothnessrefdistance_list : list
        Reference distances used by the smoothness constraints.
    antennaconstraint_list : list
        Antenna-selection constraints for each solve.
    resetsols_list : list
        Solution types to reset before solving.
    resetdir_list : list
        Direction selections to reset before solving.
    normamps_list : list
        Per-solve amplitude-normalization settings.
    BLsmooth_list : list
        Per-solve baseline-smoothing settings.
    solve_msinnchan_list : list
        Channel counts used for each solve.
    solve_msinstartchan_list : list
        Starting channel indices used for each solve.
    antenna_averaging_factors_list : list
        Per-solve antenna averaging factors.
    antenna_smoothness_factors_list : list
        Per-solve antenna smoothness factors.
    soltypelist_includedir : list
        Per-direction selection of solution types.
    normamps : bool, optional
        Normalize amplitudes when solving against the supplied sky model.
    normamps_per_ms : bool, optional
        Normalize amplitudes independently for each Measurement Set.
    skymodel : str, list of str, or None, optional
        Sky-model file or files used for prediction and calibration.
    predictskywithbeam : bool, optional
        Include the beam model when predicting the sky model.
    longbaseline : bool, optional
        Whether the data include long baselines.
    skymodelsource : str, list of str, or None, optional
        Source selection passed to sky-model prediction.
    skymodelpointsource : float or None, optional
        Flux density of a point-source model, when requested.
    wscleanskymodel : str, list of str, or None, optional
        WSClean model-image basename or basenames used for prediction.
    skymodelsetjy : bool, optional
        Use CASA setjy to create the initial model.
    mslist_beforephaseup : list of str or None, optional
        Original Measurement Set paths before phase-up processing.
    modeldatacolumns : list of str, optional
        Direction-dependent model-data column names.
    dde_skymodel : str or None, optional
        Grouped sky model used for direction-dependent prediction.
    DDE_predict : str, optional
        Prediction backend for direction-dependent calibration.
    telescope : str, optional
        Telescope name used to configure calibration behavior.
    mslist_beforeremoveinternational : list of str or None, optional
        Measurement Set paths before international stations were removed.

    Returns
    -------
    list of str
        H5Parm paths required for WSClean prediction in later processing, or an
        empty list when no such solutions are needed.
    """
    ## --- start STACK code ---
    if args['stack']:
        # create MODEL_DATA because in case it does not exist (needed in case user gives external model(s))
        # only for first (=0) selfcalcycle cycle and if user provides a model
        if ((skymodel is not None) or (skymodelpointsource is not None)
            or (wscleanskymodel is not None)) and selfcalcycle == 0:
            for ms_id, ms in enumerate(mslist):  # do the predicts (only used for stacking)
                terminal_print('Doing sky predict for stacking...')
                if skymodel is not None and type(skymodel) is str:
                    predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam,
                               sources=skymodelsource, modelstoragemanager=args['modelstoragemanager'])
                if skymodel is not None and type(skymodel) is list:
                    predictsky(ms, skymodel[ms_id], modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam,
                               sources=skymodelsource, modelstoragemanager=args['modelstoragemanager'])
                if wscleanskymodel is not None and type(wscleanskymodel) is str and not args['phasediff_only']:
                    makeimage([ms], wscleanskymodel, 1., 1.,
                              len(glob.glob(wscleanskymodel + '-????-model.fits')),
                              0, 0.0, onlypredict=True, idg=False,
                              fulljones_h5_facetbeam=not args['single_dual_speedup'])
                if wscleanskymodel is not None and type(wscleanskymodel) is list and not args['phasediff_only']:
                    makeimage([ms], wscleanskymodel[ms_id], 1., 1.,
                              len(glob.glob(wscleanskymodel[ms_id] + '-????-model.fits')),
                              0, 0.0, onlypredict=True, idg=False,
                              fulljones_h5_facetbeam=not args['single_dual_speedup'])

                nchan = len(get_frequencies_from_ms(ms))
                if skymodelpointsource is not None and type(skymodelpointsource) is float:
                    # create MODEL_DATA (no dysco!)
                    if args['modelstoragemanager'] is None:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
                        run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                    else:
                        if args['modelstoragemanager'] == 'sisco_stokes_i':
                            # create temporary column with the phase diff to avoid DP3 error std exception detected: Diagonal storage modes cannot store data for which the 2nd and 3rd correlation are non-zero
                            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]', log=True)
                            run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                            run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i steps=[]', log=True)
                            # remove temporary column
                            remove_column_ms(ms, 'MODEL_DATA_TMP')
                        elif args['modelstoragemanager'] == 'sisco_diagonal':
                            # create temporary column with the phase diff to avoid DP3 error std exception detected: Diagonal storage modes cannot store data for which the 2nd and 3rd correlation are non-zero
                            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]', log=True)
                            run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                            run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal steps=[]', log=True)
                            # remove temporary column
                            remove_column_ms(ms, 'MODEL_DATA_TMP')
                        else:
                            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + args['modelstoragemanager'] + ' steps=[]', log=True)
                            run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                        
                    # do the predict with taql
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource) + "+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource) + "+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'", log=True)
                    # code above is not sisco proof, as it cannot write to one index at a time so do it in one go:
                    #run("taql 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource) + "+0i),MODEL_DATA[,1]=(0+0i),MODEL_DATA[,2]=(0+0i),MODEL_DATA[,3]=(" + str(skymodelpointsource) + "+0i)'")


                if skymodelpointsource is not None and type(skymodelpointsource) is list:
                    # create MODEL_DATA (no dysco!)
                    if args['modelstoragemanager'] is None:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
                        run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource[ms_id]) + "+0i,0+0i,0+0i," + str(skymodelpointsource[ms_id]) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                    elif args['modelstoragemanager'] == 'sisco_stokes_i':
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]', log=True)
                        run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource[ms_id]) + "+0i,0+0i,0+0i," + str(skymodelpointsource[ms_id]) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                        run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i steps=[]', log=True)
                        # remove temporary column
                        remove_column_ms(ms, 'MODEL_DATA_TMP')    
                    elif args['modelstoragemanager'] == 'sisco_diagonal':
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]', log=True)
                        run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource[ms_id]) + "+0i,0+0i,0+0i," + str(skymodelpointsource[ms_id]) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                        run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal steps=[]', log=True)
                        # remove temporary column
                        remove_column_ms(ms, 'MODEL_DATA_TMP')
                    else:
                        run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + args['modelstoragemanager'] + ' steps=[]', log=True)
                        run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource[ms_id]) + "+0i,0+0i,0+0i," + str(skymodelpointsource[ms_id]) + "+0i], [" + str(nchan) + ",4])'", log=True, taql=True)
                    # do the predict with taql
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource[ms_id]) + "+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource[ms_id]) + "+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'", log=True)
                    #run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'", log=True)
                    # code above is not sisco proof, as it cannot write to one index at a time so do it in one go:
                    # nchan = len(get_frequencies_from_ms(ms))
                    

        # do the stack and normalization
        # mslist_stacked = stacked MS (one per common time axis / timestack); mss_timestacks = list of MSs that were used to create each stack
        mslist_stacked, mss_timestacks = stackwrapper(mslist, msout_prefix='stack', column_to_normalise='DATA')
        mslist_orig = mslist[:]  # so we can use it for the applycal
        mslist = mslist_stacked

        # set MODEL_DATA to point source in stacks
        for ms in mslist:
            t = table(ms, ack=False)
            if 'MODEL_DATA' not in t.colnames():
                t.close()
                run(f'DP3 msin={ms} msout=. msout.datacolumn=MODEL_DATA steps=[]', log=True)
            else:
                t.close()
            terminal_print(f'Predict point source for  {ms}')
            # do the predict with taql
            #run(f"taql 'update {ms} set MODEL_DATA[,0]=(1.0+0i)'", log=True)
            #run(f"taql 'update {ms} set MODEL_DATA[,3]=(1.0+0i)'", log=True)
            #run(f"taql 'update {ms} set MODEL_DATA[,1]=(0+0i)'", log=True)
            #run(f"taql 'update {ms} set MODEL_DATA[,2]=(0+0i)'", log=True)
            # code above is not sisco proof, as it cannot write to one index at a time so do it in one go:
            nchan = len(get_frequencies_from_ms(ms))
            run(f"taql 'update {ms} set MODEL_DATA=array([1.0+0i,0+0i,0+0i,1.0+0i], [{nchan},4])'", log=True, taql=True)


        # set all these to None to avoid skymodel predicts in runDPPPbase()
        skymodelpointsource = None
        wscleanskymodel = None
        skymodel = None
        skymodelsetjy = False
    ## --- end STACK code ---

    if len(modeldatacolumns) > 1:
        merge_all_in_one = False
    else:
        merge_all_in_one = True

    soltypecycles_list_array = np.array(soltypecycles_list)  # needed to slice (slicing does not work in nested l
    pertubation = []  # len(mslist)
    for _ in mslist:
        pertubation.append(False)

    parmdbmergelist = [[] for _ in
                       range(len(mslist))]  # [[],[],[],[]] nested list length mslist used for Jurjen's h5_merge
    # LOOP OVER THE ENTIRE SOLTYPE LIST (so includes pertubations via a pre-applycal)
    for soltypenumber, soltype in enumerate(args['soltype_list']):
        # SOLVE LOOP OVER MS
        parmdbmslist = []
        for msnumber, ms in enumerate(mslist):
            # check we are above far enough in the selfcal to solve for the extra pertubation
            if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]:
                terminal_print('selfcalcycle, soltypenumber', selfcalcycle, soltypenumber)
                if (soltypenumber < len(args['soltype_list']) - 1):

                    terminal_print(
                        'Self-calibration cycle:',
                        selfcalcycle,
                        'Next solution-type cycle:',
                        soltypecycles_list[soltypenumber + 1][msnumber],
                    )
                    terminal_print('_______________________')
                    terminal_print(
                        'Remaining solution-type cycles:',
                        soltypecycles_list_array,
                        'Current solution-type index:',
                        soltypenumber,
                        'Number of solution-type cycles:',
                        len(soltypecycles_list_array),
                    )
                    terminal_print('Array soltypecycles_list ahead',
                          soltypecycles_list_array[soltypenumber + 1:len(soltypecycles_list_array[:, 0]), msnumber])
                    # if (selfcalcycle >= soltypecycles_list[soltypenumber+1][msnumber]): # this looks one soltype ahead...hmmm, not good
                    if selfcalcycle >= np.min(
                            soltypecycles_list_array[soltypenumber + 1:len(soltypecycles_list_array[:, 0]),
                            msnumber]):  # this looks all soltype ahead
                        pertubation[msnumber] = True
                    else:
                        pertubation[msnumber] = False
                else:
                    pertubation[msnumber] = False

                if ((skymodel is not None) or (skymodelpointsource is not None) or (
                        wscleanskymodel is not None) or (args['skymodelsetjy'])):
                    parmdb = 'h5_solutions/' + soltype + str(soltypenumber) + '_skyselfcalcycle' + str(selfcalcycle).zfill(
                        3) + '_' + os.path.basename(ms) + '.h5'
                else:
                    parmdb = 'h5_solutions/' + soltype + str(soltypenumber) + '_selfcalcycle' + str(selfcalcycle).zfill(
                        3) + '_' + os.path.basename(ms) + '.h5'

                # set create_modeldata to False it was already prediceted before
                create_modeldata = True
                if (soltypenumber >= 1) or (selfcalcycle > 0 and not args['keepusingstartingskymodel']): 
                    create_modeldata = False

                runDPPPbase(ms, solint_list[soltypenumber][msnumber], nchan_list[soltypenumber][msnumber], parmdb,
                            soltype,
                            uvmin=args['uvmin'],
                            SMconstraint=smoothnessconstraint_list[soltypenumber][msnumber],
                            SMconstraintreffreq=smoothnessreffrequency_list[soltypenumber][msnumber],
                            SMconstraintspectralexponent=smoothnessspectralexponent_list[soltypenumber][msnumber],
                            SMconstraintrefdistance=smoothnessrefdistance_list[soltypenumber][msnumber],
                            antennaconstraint=antennaconstraint_list[soltypenumber][msnumber],
                            resetsols=resetsols_list[soltypenumber][msnumber],
                            resetdir=resetdir_list[soltypenumber][msnumber],
                            restoreflags=args['restoreflags'], flagging=args['doflagging'], skymodel=skymodel,
                            flagslowphases=args['doflagslowphases'], flagslowamprms=args['flagslowamprms'],
                            flagslowphaserms=args['flagslowphaserms'],
                            predictskywithbeam=predictskywithbeam, BLsmooth=BLsmooth_list[soltypenumber][msnumber],
                            skymodelsource=skymodelsource,
                            skymodelpointsource=skymodelpointsource, wscleanskymodel=wscleanskymodel,
                            iontimefactor=args['iontimefactor'], ionfreqfactor=args['ionfreqfactor'], blscalefactor=args['blscalefactor'], 
                            dejumpFR=args['dejumpFR'], uvminscalarphasediff=args['uvminscalarphasediff'],
                            create_modeldata=create_modeldata,
                            selfcalcycle=selfcalcycle, dysco=args['dysco'], blsmooth_chunking_size=args['blsmooth_chunking_size'],
                            soltypenumber=soltypenumber, ampresetvalfactor=args['ampresetvalfactor'],
                            flag_ampresetvalfactor=args['flag_ampresetvalfactor'],
                            clipsolutions=args['clipsolutions'], clipsolhigh=args['clipsolhigh'],
                            clipsollow=args['clipsollow'], uvmax=args['uvmax'], modeldatacolumns=modeldatacolumns,
                            preapplyH5_dde=parmdbmergelist[msnumber], dde_skymodel=dde_skymodel,
                            DDE_predict=DDE_predict, ncpu_max=args['ncpu_max_DP3solve'], soltype_list=args['soltype_list'],
                            DP3_dual_single=args['single_dual_speedup'], soltypelist_includedir=soltypelist_includedir[soltypenumber][msnumber],
                            normamps=normamps, modelstoragemanager=args['modelstoragemanager'], skymodelsetjy=skymodelsetjy,
                            solve_msinnchan=solve_msinnchan_list[soltypenumber][msnumber],
                            solve_msinstartchan=solve_msinstartchan_list[soltypenumber][msnumber],
                            antenna_averaging_factors=antenna_averaging_factors_list[soltypenumber][msnumber],
                            antenna_smoothness_factors=antenna_smoothness_factors_list[soltypenumber][msnumber], 
                            auto_flag_antennas=args['auto_flag_antennas'])
                parmdbmslist.append(parmdb)
                parmdbmergelist[msnumber].append(parmdb)  # for h5_merge

        # NORMALIZE amplitudes
        if normamps and (soltype in ['complexgain', 'scalarcomplexgain', 'rotation+diagonal',
                                     'rotation+diagonalamplitude', 'rotation+scalar',
                                     'rotation+scalaramplitude', 'amplitudeonly', 'scalaramplitude',
                                     'faradayrotation+diagonal', 'faradayrotation+diagonalamplitude',
                                     'faradayrotation+scalar', 'faradayrotation+scalaramplitude']) and len(
            parmdbmslist) > 0:
            terminal_print('Doing global amplitude-type normalization')

            # [soltypenumber][msnumber] msnumber=0 is ok because we  do all ms at once
            if normamps_list[soltypenumber][0] == 'normamps':  #
                terminal_print('Performing global amplitude normalization')
                normamplitudes(parmdbmslist,
                               norm_per_ms=normamps_per_ms)  # list of h5 for different ms, all same soltype

            if normamps_list[soltypenumber][0] == 'normamps_per_ant':
                terminal_print('Performing global amplitude normalization per antenna')
                normamplitudes_withmatrix(parmdbmslist)

            if normamps_list[soltypenumber][0] == 'normslope':
                terminal_print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)

            if normamps_list[soltypenumber][0] == 'normslope+normamps':
                terminal_print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)  # first do the slope
                normamplitudes(parmdbmslist, norm_per_ms=normamps_per_ms)

            if normamps_list[soltypenumber][0] == 'normslope+normamps_per_ant':
                terminal_print('Performing global slope normalization')
                normslope_withmatrix(parmdbmslist)  # first do the slope
                normamplitudes_withmatrix(parmdbmslist)

        if args['phasediff_only']:
            continue

        # APPLYCAL or PRE-APPLYCAL or CORRUPT
        count = 0
        for msnumber, ms in enumerate(mslist):
            if selfcalcycle >= soltypecycles_list[soltypenumber][msnumber]:  # Check cycle condition
                terminal_print(
                    'Pertubation[msnumber]:',
                    pertubation[msnumber],
                    'Parmdbmslist[count]:',
                    parmdbmslist[count],
                    'Measurement Set index:',
                    msnumber,
                    'Perturbation count:',
                    count,
                )

                if pertubation[msnumber]:  # Indicates another solve follows after this
                    if soltypenumber == 0:
                        if len(modeldatacolumns) > 1:
                            if DDE_predict == 'WSCLEAN':
                                corrupt_modelcolumns(ms, parmdbmslist[count], modeldatacolumns, modelstoragemanager=args['modelstoragemanager'])  # For DDE
                        else:
                            corrupt_modelcolumns(ms, parmdbmslist[count], ['MODEL_DATA'], modelstoragemanager=args['modelstoragemanager'])  # Saves disk space
                    else:
                        if len(modeldatacolumns) > 1:
                            if DDE_predict == 'WSCLEAN':
                                corrupt_modelcolumns(ms, parmdbmslist[count], modeldatacolumns, modelstoragemanager=args['modelstoragemanager'])  # For DDE
                        else:
                            corrupt_modelcolumns(ms, parmdbmslist[count], ['MODEL_DATA'], modelstoragemanager=args['modelstoragemanager'])  # Saves disk space
                else:  # This is the last solve; no other perturbation
                    # Reverse solution tables ([::-1]) to switch from corrupt to correct; essential as fulljones and diagonal solutions do not commute
                    applycal(ms, parmdbmergelist[msnumber][::-1], msincol='DATA', msoutcol='CORRECTED_DATA',
                             dysco=args['dysco'], modeldatacolumns=modeldatacolumns)  # Saves disk space

                count += 1  # Extra counter because parmdbmslist can be shorter than mslist as soltypecycles_list goes per ms

    if args['phasediff_only']:
        return []

    wsclean_h5list = []

    for msnumber, ms in enumerate(mslist):
        if ((skymodel is not None) or (skymodelpointsource is not None) or (
                wscleanskymodel is not None) or (args['skymodelsetjy'])):
            parmdbmergename = 'h5_solutions/merged_skyselfcalcycle' + str(selfcalcycle).zfill(3) + '_' + os.path.basename(
                ms) + '.h5'
            parmdbmergename_pc = 'h5_solutions/merged_skyselfcalcycle' + str(selfcalcycle).zfill(
                3) + '_linearfulljones_' + os.path.basename(ms) + '.h5'
        else:
            parmdbmergename = 'h5_solutions/merged_selfcalcycle' + str(selfcalcycle).zfill(3) + '_' + os.path.basename(ms) + '.h5'
            parmdbmergename_pc = 'h5_solutions/merged_selfcalcycle' + str(selfcalcycle).zfill(
                3) + '_linearfulljones_' + os.path.basename(ms) + '.h5'
        if os.path.isfile(parmdbmergename):
            Path(parmdbmergename).unlink(missing_ok=True)
        if os.path.isfile(parmdbmergename_pc):
            Path(parmdbmergename_pc).unlink(missing_ok=True)
        wsclean_h5list.append(parmdbmergename)

        # add extra from preapplyH5_list
        if args['preapplyH5_list'][0] is not None:
            preapplyh5parm = time_match_mstoH5(args['preapplyH5_list'], ms)
            # replace the source direction coordinates so that the merge goes correctly
            copy_over_source_direction_h5(parmdbmergelist[msnumber][0], preapplyh5parm)
            parmdbmergelist[msnumber].append(preapplyh5parm)

        if is_scalar_array_for_wsclean(parmdbmergelist[msnumber]):
            single_pol_merge = True
        else:
            single_pol_merge = False

        # reset h5parm structure
        if not merge_all_in_one:  # only for a DDE solve
            parmdbmergelist[msnumber] = fix_h5(parmdbmergelist[msnumber])

        terminal_print(
            'Merged ParmDB:',
            parmdbmergename,
            'Parmdbmergelist[msnumber]:',
            parmdbmergelist[msnumber],
            'Measurement Set:',
            ms,
        )
        if args['reduce_h5size'] and ('tec' not in args['soltype_list']) \
            and ('tecandphase' not in args['soltype_list']) \
            and ('tec+phase' not in args['soltype_list']) \
            and ('tec+delay' not in args['soltype_list']) \
            and ('delay' not in args['soltype_list']) \
            and ('tec+phase+delay' not in args['soltype_list']):
            merge_h5(h5_out=parmdbmergename, h5_tables=parmdbmergelist[msnumber][::-1],
                     merge_all_in_one=merge_all_in_one,
                     propagate_weights=True, single_pol=single_pol_merge)
        else:
            merge_h5(h5_out=parmdbmergename, h5_tables=parmdbmergelist[msnumber][::-1], ms_files=ms,
                     convert_tec=True, merge_all_in_one=merge_all_in_one,
                     propagate_weights=True, single_pol=single_pol_merge)
        # add CS stations back for superstation
        if mslist_beforephaseup is not None:
            terminal_print('mslist_beforephaseup: ' + mslist_beforephaseup[msnumber])
            if is_scalar_array_for_wsclean([parmdbmergename]):
                single_pol_merge = True
            else:
                single_pol_merge = False
            merge_h5(h5_out=parmdbmergename.replace("selfcalcycle",
                                                    "addCS_selfcalcycle"), h5_tables=parmdbmergename,
                     ms_files=mslist_beforephaseup[msnumber], convert_tec=True,
                     merge_all_in_one=merge_all_in_one, single_pol=single_pol_merge,
                     propagate_weights=True, add_cs=True)

        # make LINEAR solutions from CIRCULAR (never do a single_pol merge here!)
        if ('scalarphasediff' in args['soltype_list']) or ('scalarphasediffFR' in args['soltype_list']) or args['docircular']:
            merge_h5(h5_out=parmdbmergename_pc, h5_tables=parmdbmergename, circ2lin=True,
                     propagate_weights=True)
            # add CS stations back for superstation
            if mslist_beforephaseup is not None:
                merge_h5(h5_out=parmdbmergename_pc.replace("selfcalcycle",
                                                           "addCS_selfcalcycle"),
                         h5_tables=parmdbmergename_pc,
                         ms_files=mslist_beforephaseup[msnumber], convert_tec=True,
                         merge_all_in_one=merge_all_in_one,
                         propagate_weights=True, add_cs=True)

        if False:
            # testing only to check if merged H5 file is correct and makes a good image
            applycal(ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])

        # plot merged solution file
        # print('single_pol_merge',single_pol_merge)
        losotoparset = create_losoto_flag_apgridparset(ms, flagging=False,
                                                       medamp=get_median_amp(parmdbmergename),
                                                       outplotname=
                                                       parmdbmergename.split('_' + os.path.basename(ms) + '.h5')[0],
                                                       refant=findrefant_core(parmdbmergename, telescope=telescope),
                                                       fulljones=fulljonesparmdb(parmdbmergename),
                                                       onepol=is_scalar_array_for_wsclean([parmdbmergename]), 
                                                       skip_solution_plotting=args["skip_solution_plotting"])
        run('losoto ' + parmdbmergename + ' ' + losotoparset)
        force_close(parmdbmergename)

        ## --- start STACK code ---
        if args['stack']:
            # for each stacked MS (one per common time axis), apply the solutions to each direction MS
            for orig_ms in mss_timestacks[msnumber]:
                applycal(orig_ms, parmdbmergename, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=args['dysco'])
        ## --- end STACK code ---

    if args['QualityBasedWeights'] and selfcalcycle >= args['QualityBasedWeights_start']:
        for ms in mslist:
            run('python3 NeReVar.py --filename=' + ms + \
                ' --dt=' + str(args['QualityBasedWeights_dtime']) + ' --dnu=' + str(args['QualityBasedWeights_dfreq']) + \
                ' --DiagDir=solution_plots_' + os.path.basename(ms) + '/NeReVar/ --basename=_selfcalcycle' + str(selfcalcycle).zfill(
                3) + ' --modelcol=MODEL_DATA')


    if len(modeldatacolumns) > 0:
        np.save('misc/wsclean_h5list' + str(selfcalcycle).zfill(3) + '.npy', wsclean_h5list)
        return wsclean_h5list
    else:
        return []


@_track_resource_phase("solve")
def runDPPPbase(ms, solint, nchan, parmdb, soltype, uvmin=1.,
                SMconstraint=0.0, SMconstraintreffreq=0.0,
                SMconstraintspectralexponent=-1.0, SMconstraintrefdistance=0.0, antennaconstraint=None,
                resetsols=None, resetdir=None,
                resetdir_list=[None], restoreflags=False,
                maxiter=100, tolerance=1e-4, flagging=False, skymodel=None, flagslowphases=True,
                flagslowamprms=7.0, flagslowphaserms=7.0, incol='DATA',
                predictskywithbeam=False, BLsmooth=False, skymodelsource=None,
                skymodelpointsource=None, wscleanskymodel=None, iontimefactor=0.01, ionfreqfactor=1.0,
                blscalefactor=1.0, dejumpFR=False, uvminscalarphasediff=0, selfcalcycle=0, dysco=True,
                blsmooth_chunking_size=8, soltypenumber=0, create_modeldata=True,
                clipsolutions=False, clipsolhigh=1.5, clipsollow=0.667,
                ampresetvalfactor=10., flag_ampresetvalfactor=False, uvmax=None,
                modeldatacolumns=[], solveralgorithm='directioniterative', solveralgorithm_dde='directioniterative',
                preapplyH5_dde=[],
                dde_skymodel=None, DDE_predict='WSCLEAN', beamproximitylimit=240.,
                ncpu_max=24, bdaaverager=False, DP3_dual_single=True, soltype_list=None, soltypelist_includedir=None,
                normamps=True, modelstoragemanager=None, pixelscale=None, imsize=None, skymodelsetjy=False,
                solve_msinnchan='all', solve_msinstartchan=0,
                antenna_averaging_factors=None, antenna_smoothness_factors=None, auto_flag_antennas=False, max_tec_delay_wraps=15):
    """
    Run the base DP3 calibration and solution-processing workflow.

    Parameters
    ----------
    ms : str
        Measurement Set path.
    solint : str or float
        Solution interval.
    nchan : int
        Number of solution channels.
    parmdb : str
        Parameter database path.
    soltype : str
        Calibration solution type.
    uvmin : float, optional
        Minimum baseline length used by the solve.
    SMconstraint : float, optional
        Spatial smoothness constraint strength.
    SMconstraintreffreq : float, optional
        Reference frequency for the smoothness constraint.
    SMconstraintspectralexponent : float, optional
        Spectral exponent for scaling the smoothness constraint.
    SMconstraintrefdistance : float, optional
        Reference distance for the smoothness constraint.
    antennaconstraint : str or None, optional
        Antenna subset or constraint passed to the solver.
    resetsols : str or None, optional
        Solution types to reset before solving.
    resetdir : str or None, optional
        Direction to reset before solving.
    resetdir_list : list, optional
        Per-direction reset selections.
    restoreflags : bool, optional
        Restore the saved flag column before solving.
    maxiter : int, optional
        Maximum solver iterations.
    tolerance : float, optional
        Solver convergence tolerance.
    flagging : bool, optional
        Run the configured solution-flagging steps.
    skymodel : str or None, optional
        Sky-model file used for prediction.
    flagslowphases : bool, optional
        Flag low signal-to-noise phase solutions.
    flagslowamprms : float, optional
        RMS threshold for flagging low-amplitude solutions.
    flagslowphaserms : float, optional
        RMS threshold for flagging low-phase solutions.
    incol : str, optional
        Input visibility column used by DP3.
    predictskywithbeam : bool, optional
        Include the beam model during sky prediction.
    BLsmooth : bool, optional
        Smooth visibilities across baselines before solving.
    skymodelsource : str or None, optional
        Source selection passed to sky-model prediction.
    skymodelpointsource : float or None, optional
        Flux density of a point-source model.
    wscleanskymodel : str or None, optional
        WSClean model-image basename used for prediction.
    iontimefactor : float, optional
        Time-smoothing factor for ionospheric solutions.
    ionfreqfactor : float, optional
        Frequency-smoothing factor for ionospheric solutions.
    blscalefactor : float, optional
        Baseline-smoothing scale factor.
    dejumpFR : bool, optional
        Remove phase jumps when fitting Faraday rotation.
    uvminscalarphasediff : float, optional
        Minimum baseline length for scalar phase-difference solving.
    selfcalcycle : int, optional
        Current self-calibration cycle number.
    dysco : bool, optional
        Use Dysco storage for generated Measurement Set columns.
    blsmooth_chunking_size : int, optional
        Number of baselines processed per smoothing chunk.
    soltypenumber : int, optional
        Index of the solution type being processed.
    create_modeldata : bool, optional
        Create model-data columns before solving.
    clipsolutions : bool, optional
        Clip solution values to the configured limits.
    clipsolhigh : float, optional
        Upper solution clipping threshold.
    clipsollow : float, optional
        Lower solution clipping threshold.
    ampresetvalfactor : float, optional
        Amplitude factor used when resetting solutions.
    flag_ampresetvalfactor : bool, optional
        Flag amplitudes outside the reset-factor limits.
    uvmax : float or None, optional
        Maximum baseline length used by the solve.
    modeldatacolumns : list of str, optional
        Model-data columns to use for direction-dependent solving.
    solveralgorithm : str, optional
        Solver algorithm for direction-independent calibration.
    solveralgorithm_dde : str, optional
        Solver algorithm for direction-dependent calibration.
    preapplyH5_dde : list of str, optional
        H5Parm solutions to apply before the direction-dependent solve.
    dde_skymodel : str or None, optional
        Grouped sky model for direction-dependent prediction.
    DDE_predict : str, optional
        Prediction backend for direction-dependent calibration.
    beamproximitylimit : float, optional
        Beam-proximity limit passed to DP3 prediction.
    ncpu_max : int, optional
        Maximum number of CPU threads for DP3.
    bdaaverager : bool, optional
        Use baseline-dependent averaging.
    DP3_dual_single : bool, optional
        Use the DP3 dual/single-polarization solve path.
    soltype_list : list of str or None, optional
        Configured calibration solution types.
    soltypelist_includedir : list or None, optional
        Per-direction solution-type selection flags.
    normamps : bool, optional
        Normalize solution amplitudes.
    modelstoragemanager : str or None, optional
        Storage manager for model-data columns.
    pixelscale : float or None, optional
        Image pixel size in arcseconds for model prediction.
    imsize : int or None, optional
        Image size in pixels for model prediction.
    skymodelsetjy : bool, optional
        Use CASA setjy when creating the model data.
    solve_msinnchan : int or str, optional
        Number of input channels to include in the solve.
    solve_msinstartchan : int, optional
        First input channel included in the solve.
    antenna_averaging_factors : list or None, optional
        Per-antenna averaging factors.
    antenna_smoothness_factors : list or None, optional
        Per-antenna smoothness factors.
    auto_flag_antennas : bool, optional
        Automatically flag antennas with poor solutions.
    max_tec_delay_wraps : int, optional
        Maximum number of TEC/delay phase wraps accepted by the solver.

    Returns
    -------
    None
        Calibration solutions and associated products are written to disk.
    """
    soltypein = soltype  # save the input soltype is as soltype could be modified (for example by scalarphasediff)

    modeldata = 'MODEL_DATA'  # the default, update if needed for scalarphasediff and phmin solves
    if BLsmooth:
        # Open the measurement set and check column names
        with table(ms, ack=False) as t:
            colnames = t.colnames()

        # If 'SMOOTHED_DATA' column doesn't exist, run the BLsmooth command
        if 'SMOOTHED_DATA' not in colnames:
            blsmooth_command = (
                f"python {submodpath}/BLsmooth.py -n 8 -c {blsmooth_chunking_size} -i {incol} "
                f"-o SMOOTHED_DATA -f {iontimefactor} -s {blscalefactor} "
                f"-u {ionfreqfactor} {ms}"
            )
            terminal_print()
            run(blsmooth_command)

        incol = 'SMOOTHED_DATA'

    if skymodelsetjy and create_modeldata and len(modeldatacolumns) == 0:
        setjy_casa(ms)
    
     
    if skymodel is not None and create_modeldata and len(modeldatacolumns) == 0:
        predictsky(ms, skymodel, modeldata='MODEL_DATA', predictskywithbeam=predictskywithbeam, sources=skymodelsource, modelstoragemanager=modelstoragemanager)

    # if wscleanskymodel is not None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR' and create_modeldata:
    if wscleanskymodel is not None and create_modeldata and len(modeldatacolumns) == 0:
        makeimage([ms], wscleanskymodel, 1., 1., len(glob.glob(wscleanskymodel + '-????-model.fits')),
                  0, 0.0, onlypredict=True, idg=False)

    # if skymodelpointsource is not None and soltypein != 'scalarphasediff' and soltypein != 'scalarphasediffFR' and create_modeldata:
    if skymodelpointsource is not None and create_modeldata and len(modeldatacolumns) == 0:
        nchan_taql = len(get_frequencies_from_ms(ms))
        # create MODEL_DATA (no dysco!)
        if modelstoragemanager is None:
            run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA steps=[]')
            run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan_taql) + ",4])'", log=True, taql=True) 
        else:
            if modelstoragemanager == 'sisco_stokes_i':
                run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]')
                run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan_taql) + ",4])'", log=True, taql=True)
                run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=stokes_i steps=[]')
                # remove temporary column
                remove_column_ms(ms, 'MODEL_DATA_TMP')
            elif modelstoragemanager == 'sisco_diagonal':
                run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA_TMP steps=[]')
                run("taql 'update " + ms + " set MODEL_DATA_TMP=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan_taql) + ",4])'", log=True, taql=True)
                run('DP3 msin=' + ms + ' msout=. msin.datacolumn=MODEL_DATA_TMP msout.datacolumn=MODEL_DATA msout.storagemanager=sisco msout.storagemanager.sisco_mode=diagonal steps=[]')
                # remove temporary column
                remove_column_ms(ms, 'MODEL_DATA_TMP')
            else:
                run('DP3 msin=' + ms + ' msout=. msout.datacolumn=MODEL_DATA msout.storagemanager=' + modelstoragemanager + ' steps=[]')
                run("taql 'update " + ms + " set MODEL_DATA=array([" + str(skymodelpointsource) + "+0i,0+0i,0+0i," + str(skymodelpointsource) + "+0i], [" + str(nchan_taql) + ",4])'", log=True, taql=True) 

        # do the predict with taql
        #run("taql" + " 'update " + ms + " set MODEL_DATA[,0]=(" + str(skymodelpointsource) + "+0i)'")
        #run("taql" + " 'update " + ms + " set MODEL_DATA[,3]=(" + str(skymodelpointsource) + "+0i)'")
        #run("taql" + " 'update " + ms + " set MODEL_DATA[,1]=(0+0i)'")
        #run("taql" + " 'update " + ms + " set MODEL_DATA[,2]=(0+0i)'")
        # code above is not sisco proof, as it cannot write to one index at a time so do it in one go:
               

    if soltype == 'scalarphasediff' or soltype == 'scalarphasediffFR':
        # PM means point source model adjusted weights
        create_weight_spectrum(ms, 'WEIGHT_SPECTRUM_PM', updateweights_from_thiscolumn='MODEL_DATA',
                               updateweights=False)  # always do to re-initialize WEIGHT_SPECTRUM_PM (because stack.MS is re-created each selfcalcycle, also MODEL_DATA changes
        # for now use updateweights=False, seems to give better results for scalarphasediff type solves
        # updateweights means WEIGHT_SPECTRUM_PM is updated based on MODEL_DATA**2, however so far this does not seem to give better results
        # update July 2024, tested on Perseus LBA with updateweights True and False, again difference is small

        # check if colnames are there each time because of stack.MS
        t = table(ms, ack=False)
        colnames = t.colnames()
        t.close()  # needs a close here because below were are writing columns potentially
        if 'DATA_CIRCULAR_PHASEDIFF' not in colnames:
            create_phasediff_column(ms, incol=incol, dysco=dysco)
        if 'MODEL_DATA_PDIFF' not in colnames:
            create_MODEL_DATA_PDIFF(ms, modelstoragemanager)  # make a point source
        soltype = 'phaseonly'  # do this type of solve, maybe scalarphase is fine? 'scalarphase' #
        incol = 'DATA_CIRCULAR_PHASEDIFF'
        modeldata = 'MODEL_DATA_PDIFF'

    if soltype in ['phaseonly_phmin', 'rotation_phmin', 'tec_phmin', 'tecandphase_phmin', 'scalarphase_phmin']:
        create_phase_column(ms, incol=incol, outcol='DATA_PHASEONLY', dysco=dysco)
        create_phase_column(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASEONLY', dysco=False)
        soltype = soltype.split('_phmin')[0]
        incol = 'DATA_PHASEONLY'
        modeldata = 'MODEL_DATA_PHASEONLY'

    if soltype in ['phaseonly_slope', 'scalarphase_slope']:
        create_phase_slope(ms, incol=incol, outcol='DATA_PHASE_SLOPE', ampnorm=True, dysco=dysco)
        create_phase_slope(ms, incol='MODEL_DATA', outcol='MODEL_DATA_PHASE_SLOPE', ampnorm=True, dysco=dysco)
        soltype = soltype.split('_slope')[0]
        incol = 'DATA_PHASE_SLOPE'
        modeldata = 'MODEL_DATA_PHASE_SLOPE'
        # udpate weights according to weights * (MODEL_DATA/MODEL_DATA_PHASE_SLOPE)**2
        create_weight_spectrum_modelratio(ms, 'WEIGHT_SPECTRUM_PM',
                                          updateweights=True, originalmodel='MODEL_DATA',
                                          newmodel='MODEL_DATA_PHASE_SLOPE', backup=True)

    if soltype in ['phaseonly', 'complexgain', 'fulljones', 'rotation+diagonal', 'amplitudeonly', \
                   'rotation+diagonalamplitude', 'rotation+diagonalphase', \
                   'faradayrotation+diagonal', 'faradayrotation+diagonalphase', \
                   'faradayrotation+diagonalamplitude', 'leakage', 'leakageamplitude']:  # for 1D plotting
        onepol = False
    if soltype in ['scalarphase', 'tecandphase', 'tec', 'scalaramplitude',
                   'scalarcomplexgain', 'rotation', 'rotation+scalar',
                   'rotation+scalarphase', 'rotation+scalaramplitude',
                   'faradayrotation', 'faradayrotation+scalar',
                   'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase',
                   'tec+phase', 'tec+delay', 'tec+phase+delay', 'delay']:
        onepol = True

    if restoreflags:
        cmdtaql = "'update " + ms + " set FLAG=FLAG_BACKUP'"
        terminal_print("Restore flagging column: " + "taql " + cmdtaql)
        run("taql " + cmdtaql, taql=True)

    t = table(ms + '::SPECTRAL_WINDOW', ack=False)
    freq = np.median(t.getcol('CHAN_FREQ')[0])
    t.close()

    t = table(ms + '::ANTENNA', ack=False)
    antennasms = t.getcol('NAME')
    t.close()

    t = table(ms, readonly=True, ack=False)
    ms_ntimes = len(np.unique(t.getcol('TIME')))
    t.close()

    if args['telescope'] == 'LOFAR':
        if freq > 100e6:
            HBAorLBA = 'HBA'
        else:
            HBAorLBA = 'LBA'
        terminal_print('This is', HBAorLBA, 'data')
    else:
        HBAorLBA = 'other'
    terminal_print('Measurement set', os.path.basename(ms.rstrip('/')), 'contains antennas:', antennasms)

    # determine if phases needs to be included, important if slowgains do not contain phase solutions
    includesphase = True
    if soltype == 'scalaramplitude' or soltype == 'amplitudeonly' or soltype == 'leakageamplitude' \
            or soltype == 'rotation+diagonalphase' or soltype == 'rotation+scalarphase' \
            or soltype == 'faradayrotation+diagonalphase' or soltype == 'faradayrotation+scalarphase':
        includesphase = False

    # figure out which weight_spectrum column to use
    if soltypein == 'scalarphasediff' or soltypein == 'scalarphasediffFR' or \
            soltypein == 'phaseonly_slope' or soltypein == 'scalarphase_slope':
        weight_spectrum = 'WEIGHT_SPECTRUM_PM'
    else:
        # check for WEIGHT_SPECTRUM_SOLVE from DR2 products
        t = table(ms, ack=False)
        if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
            weight_spectrum = 'WEIGHT_SPECTRUM_SOLVE'
        else:
            weight_spectrum = 'WEIGHT_SPECTRUM'
        t.close()

        # check for previous old parmdb and remove them
    if os.path.isfile(parmdb):
        terminal_print('H5 file exists  ', parmdb)
        Path(parmdb).unlink(missing_ok=True)

    cmd = 'DP3 numthreads=' + str(np.min([multiprocessing.cpu_count(), ncpu_max])) + \
          ' msin=' + ms + ' msout=. '
    
    if solve_msinnchan != 'all':
        SMconstraint = 0.0  # set SMconstraint to 0 so that DP3 does not solve for frequency dependence
        nchan = 0  # set nchan to 0 so that DP3 does not solve for frequency dependence

    if soltype == 'rotation+diagonal':
        cmd += 'ddecal.rotationdiagonalmode=diagonal '
    if soltype == 'rotation+diagonalamplitude':
        cmd += 'ddecal.rotationdiagonalmode=diagonalamplitude '
    if soltype == 'rotation+diagonalphase':
        cmd += 'ddecal.rotationdiagonalmode=diagonalphase '
    if soltype == 'rotation+scalar':  # =scalarcomplexgain
        cmd += 'ddecal.rotationdiagonalmode=scalar '
    if soltype == 'rotation+scalaramplitude':
        cmd += 'ddecal.rotationdiagonalmode=scalaramplitude '
    if soltype == 'rotation+scalarphase':
        cmd += 'ddecal.rotationdiagonalmode=scalarphase '

    if soltype == 'faradayrotation+diagonal':
        cmd += 'ddecal.faradaydiagonalmode=diagonal '
    if soltype == 'faradayrotation+diagonalamplitude':
        cmd += 'ddecal.faradaydiagonalmode=diagonalamplitude '
    if soltype == 'faradayrotation+diagonalphase':
        cmd += 'ddecal.faradaydiagonalmode=diagonalphase '
    if soltype == 'faradayrotation+scalar':
        cmd += 'ddecal.faradaydiagonalmode=scalar '
    if soltype == 'faradayrotation+scalaramplitude':
        cmd += 'ddecal.faradaydiagonalmode=scalaramplitude '
    if soltype == 'faradayrotation+scalarphase':
        cmd += 'ddecal.faradaydiagonalmode=scalarphase '


    # deal with the special cases because DP3 only knows soltype rotation+diagonal (it uses rotationdiagonalmode)
    if soltype in ['rotation+diagonalamplitude', 'rotation+diagonalphase', \
                   'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase']:
        cmd += 'ddecal.mode=rotation+diagonal '
        # cmd += 'ddecal.rotationreference=True '
    elif soltype in ['faradayrotation+diagonalamplitude', 'faradayrotation+diagonalphase', \
                   'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
        cmd += 'ddecal.mode=faradayrotation '  # 
    else:
        cmd += 'ddecal.mode=' + soltype + ' '

    # note there are two things to check before we can use ddecal.datause=single/dual
    # 1 current solve needs to be of the correct type
    # 2 previous solves should not violate the assumptions of the current single/dual solve
    if soltype_list is not None:
        terminal_print(f"soltype_list slice: {soltype_list[0:soltypenumber]}")
        terminal_print('Solution type:', soltype, 'Solution type list:', soltype_list)
        if DP3_dual_single:
            if soltype in ['complexgain', 'amplitudeonly', 'phaseonly'] \
                    and 'fulljones' not in soltype_list[0:soltypenumber] \
                    and 'leakage' not in soltype_list[0:soltypenumber] \
                    and 'leakageamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonal' not in soltype_list[0:soltypenumber]:
                cmd += 'ddecal.datause=dual '
            if soltype in ['scalarcomplexgain', 'scalaramplitude', \
                           'tec', 'tecandphase', 'scalarphase','tec+phase', \
                           'tec+delay', 'tec+phase+delay', 'delay'] \
                    and 'fulljones' not in soltype_list[0:soltypenumber] \
                    and 'leakage' not in soltype_list[0:soltypenumber] \
                    and 'leakageamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'rotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'rotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonalamplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalar' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalaramplitude' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+scalarphase' not in soltype_list[0:soltypenumber] \
                    and 'faradayrotation+diagonal' not in soltype_list[0:soltypenumber] \
                    and 'complexgain' not in soltype_list[0:soltypenumber] \
                    and 'amplitudeonly' not in soltype_list[0:soltypenumber] \
                    and 'phaseonly' not in soltype_list[0:soltypenumber]: \
                    cmd += 'ddecal.datause=single '

    cmd += 'msin.weightcolumn=' + weight_spectrum + ' '
    if bdaaverager and pixelscale is not None and imsize is not None:
        cmd += 'steps=[bda,ddecal] ddecal.type=ddecal bda.type=bdaaverager '
    else:
        cmd += 'steps=[ddecal] ddecal.type=ddecal '
    if dysco:
        cmd += 'msout.storagemanager=dysco '
        cmd += 'msout.storagemanager.weightbitrate=16 '

    modeldatacolumns_solve = []  # empty, will be filled below if applicable
    dir_id_kept = []  # empty, will be filled below if applicable
    if len(modeldatacolumns) > 0:
        if DDE_predict == 'DP3' and soltypelist_includedir is not None:
            terminal_print('DDE_predict with soltypelist_includedir is not supported')
            raise Exception('DDE_predict with soltypelist_includedir is not supported')

        if soltypelist_includedir is not None:
            modeldatacolumns_solve, sourcedir_removed, dir_id_kept = updatemodelcols_includedir(modeldatacolumns,
                                                                                                soltypelist_includedir,
                                                                                                ms, modelstoragemanager=modelstoragemanager)

        if DDE_predict == 'DP3':
            cmd += 'ddecal.sourcedb=' + dde_skymodel + ' '
            if args['telescope'] == 'LOFAR':  # predict with array factor for LOFAR data
                cmd += 'ddecal.usebeammodel=True '
                cmd += 'ddecal.usechannelfreq=True ddecal.beammode=array_factor '
                cmd += 'ddecal.beamproximitylimit=' + str(beamproximitylimit) + ' '
        else:
            if len(modeldatacolumns_solve) > 0:  # >0 (and not > 1 because we can have 1 direction left)
                cmd += "ddecal.modeldatacolumns='[" + ','.join(map(str, modeldatacolumns_solve)) + "]' "
            else:
                cmd += "ddecal.modeldatacolumns='[" + ','.join(map(str, modeldatacolumns)) + "]' "
        if len(modeldatacolumns) > 1:  # so we are doing a dde solve
            cmd += 'ddecal.solveralgorithm=' + solveralgorithm_dde + ' '
        else:  # in case the list still has length 1
            cmd += 'ddecal.solveralgorithm=' + solveralgorithm + ' '
    else:
        cmd += "ddecal.modeldatacolumns='[" + modeldata + "]' "
        cmd += 'ddecal.solveralgorithm=' + solveralgorithm + ' '

    cmd += 'ddecal.maxiter=' + str(int(maxiter)) + ' ddecal.propagatesolutions=True '
    # important to avoid propagating very bad solutions which then screw all subsequent solutions
    cmd += 'ddecal.propagateconvergedonly=True '
    # Do list comprehension if solint is a list
    if antenna_averaging_factors is not None:
        max_aaf = max(int(x.split(':')[1]) for x in antenna_averaging_factors.split(','))
    if type(solint) != list:
        solint = [solint] # convert to list for easier handling below
    if type(solint) == list and len(solint) > 1:
        if len(dir_id_kept) > 0:
            terminal_print('Solution interval(s):', solint)
            terminal_print('Retained direction indices:', dir_id_kept)
            solint = [solint[i] for i in dir_id_kept]  # overwrite solint, selecting on the directions kept
        solints = [int(format_solint(x, ms)) for x in solint]
        solints = tweak_solints(solints, ms_ntimes=ms_ntimes)
        lcm = math.lcm(*solints)
        if antenna_averaging_factors is not None: # also means max_aaf is defined
            if max_aaf > max([int(lcm / i) for i in solints]):     
                upscale_factor = int(np.ceil(max_aaf/max([int(lcm / i) for i in solints])))
                if lcm * upscale_factor < 4096: # avoid too large solints
                    lcm = lcm * upscale_factor
                    terminal_print('Updated divisors and lcm:', [int(lcm / i) for i in solints], lcm)
                else:     
                    terminal_print('max_aaf', max_aaf, 'is too large for the divisors (=solutions_per_direction)')
                    terminal_print('divisors', [int(lcm / i) for i in solints], 'lcm', lcm)
                    terminal_print('This is not allowed by DP3')
                    sys.exit(1)
        divisors = [int(lcm / i) for i in solints]
        cmd += 'ddecal.solint=' + str(lcm) + ' '
        cmd += 'ddecal.solutions_per_direction=' + "'" + str(divisors).replace(' ', '') + "' "
    else: # in this case we just have a single solint value
        solints =format_solint(solint[0], ms)  # create the integer str
        solints =[int(tweak_solints_single(int(solints), ms_ntimes))]
        lcm = math.lcm(*solints)
        if antenna_averaging_factors is not None: # also means max_aaf is defined
            terminal_print('max_aaf', max_aaf, 'udating lcm')
            lcm = lcm * max_aaf  # increase the lcm by the maximum antenna averaging factor
        divisors = [int(lcm / i) for i in solints] # divisors
        cmd += 'ddecal.solint=' + str(lcm) + ' '
        cmd += 'ddecal.solutions_per_direction=' + "'" + str(divisors).replace(' ', '') + "' "
    cmd += 'ddecal.nchan=' + format_nchan(nchan, ms) + ' '
    cmd += 'ddecal.h5parm=' + parmdb + ' '


    # format antenna_averaging_factors
    if antenna_averaging_factors is not None:
        antenna_averaging_factors_splitstr = antenna_averaging_factors.split(',')
        groupstr_all = []
        antenna_averaging_factors_new = [] 
        for antgroup in antenna_averaging_factors_splitstr:
            groupstr = antennaconstraintstr(antgroup.split(':')[0], antennasms, HBAorLBA, telescope=args['telescope'], useforresetsols=True)
            groupstr_all = groupstr_all + groupstr
            antenna_averaging_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + antgroup.split(':')[1])
                
        if len(groupstr_all) != len(set(groupstr_all)):
            terminal_print('There are duplicate antennas in antenna_averaging_factors, please check your input')
            raise Exception('There are duplicate antennas in antenna_averaging_factors, please check your input')
        groupstr_complement = list(set(groupstr_all) ^ set (antennasms))  # get the complement of the antenna group
        if len(groupstr_complement) > 0: 
            antenna_averaging_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:1')  # add the complement antennas with average factor 1
            
        cmd += 'ddecal.antenna_averaging_factors=' + '[' + ','.join(antenna_averaging_factors_new) + '] '

    if bdaaverager  and pixelscale is not None and imsize is not None:
        cmd += 'bda.frequencybase= ' + 'bda.minchannels=' + format_nchan(nchan, ms) + ' '
        cmd += 'bda.timebase= ' + 'bda.maxinterval=' + int(lcm / np.max(divisors)) + ' '
        

    # preapply H5 from previous pertubation for DDE solves with DP3
    if (len(modeldatacolumns) > 1) and (len(preapplyH5_dde) > 0):
        if DDE_predict == 'DP3':
            cmd += build_applycal_dde_cmd(preapplyH5_dde) + ' '
        else:
            cmd += 'msin.datacolumn=DATA '  # to prevent solving out of CORRECTED_PREAPPLY$N
    else:
        cmd += 'msin.datacolumn=' + incol + ' '

    # SET UVMIN
    if soltypein == 'scalarphasediff' or soltypein == 'scalarphasediffFR':
        if uvminscalarphasediff is not None:
            cmd += 'ddecal.uvlambdamin=' + str(uvminscalarphasediff) + ' '
        else:
            if uvmin != 0:
                cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '
    else:
        if uvmin != 0:
            cmd += 'ddecal.uvlambdamin=' + str(uvmin) + ' '
        if uvmax is not None:  # no need to see uvlambdamax for scalarphasediff solves since there we always solve against a point source
            cmd += 'ddecal.uvlambdamax=' + str(uvmax) + ' '

    if antennaconstraint is not None:
        cmd += 'ddecal.antennaconstraint=' + antennaconstraintstr(antennaconstraint, antennasms, HBAorLBA,
                                                                  telescope=args['telescope']) + ' '

    # format antenna_smoothness_factors
    if antenna_smoothness_factors is not None:
        antenna_smoothness_factors_splitstr = antenna_smoothness_factors.split(',')
        groupstr_all = []
        smoothness_factors = []
        for antgroup in antenna_smoothness_factors_splitstr:
            smoothness_factors.append(antgroup.split(':')[1])
        terminal_print('smoothness_factors', smoothness_factors)
        smoothness_factors_float = list(map(float, smoothness_factors))
        antenna_smoothness_factors_new = [] 
        for antgroup in antenna_smoothness_factors_splitstr:

            groupstr = antennaconstraintstr(antgroup.split(':')[0], antennasms, HBAorLBA, telescope=args['telescope'], useforresetsols=True)
            groupstr_all = groupstr_all + groupstr
            if np.max(smoothness_factors_float) <= 1.0:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + antgroup.split(':')[1])
            elif (np.max(smoothness_factors_float) > 1.0) and args["DDE"]:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + \
                                                      str(float(antgroup.split(':')[1])/np.max(smoothness_factors_float)))
            elif (np.max(smoothness_factors_float) > 1.0) and (not args["DDE"]):
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr)) + ']:' + antgroup.split(':')[1])
                  
        if len(groupstr_all) != len(set(groupstr_all)):
            terminal_print('There are duplicate antennas in antenna_smoothness_factors, please check your input')
            raise Exception('There are duplicate antennas in antenna_smoothness_factors, please check your input')
        groupstr_complement = list(set(groupstr_all) ^ set (antennasms))  # get the complement of the antenna group
        if len(groupstr_complement) > 0: 
            if (np.max(smoothness_factors_float) <= 1.0):
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:1.0')  # add the complement antennas with factor 1.0
            elif (np.max(smoothness_factors_float) > 1.0) and args["DDE"]:
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:'
                                                      + str(1.0/np.max(smoothness_factors_float)))  # add the complement antennas with factor 1.0/maximum smooth
            elif (np.max(smoothness_factors_float) > 1.0) and (not args["DDE"]):
                antenna_smoothness_factors_new.append('[' + ','.join(map(str, groupstr_complement)) + ']:'
                                                      + str(np.max(smoothness_factors_float)))  # add the complement antennas with factor maximum smooth
            
        if args["DDE"] and (np.max(smoothness_factors_float) > 1.0): # handle the case where the smoothness factors are larger than 1.0
            if type(SMconstraint) == list:
                SMconstraint = [sf*np.max(smoothness_factors_float) for sf in SMconstraint]  # increase SMconstraint with the maximum smoothness factor
            else:
                SMconstraint = np.max(smoothness_factors_float) * SMconstraint  # increase SM

        cmd += 'ddecal.antenna_smoothness_factors=' + '[' + ','.join(antenna_smoothness_factors_new) + '] '


    if np.max(SMconstraint) > 0.0 and nchan != 0:
        if type(SMconstraint) == list:
            if len(dir_id_kept) > 0:
                terminal_print('Smoothness constraints:', SMconstraint)
                terminal_print('Retained direction indices:', dir_id_kept)
                SMconstraint = [SMconstraint[i] for i in dir_id_kept]  # overwrite SMconstraint, selecting on the directions kept
            max_smconstraint = float(np.max(SMconstraint))
            smoothness_dd_factors = [float(ddsf) / max_smconstraint for ddsf in SMconstraint]  
            cmd += 'ddecal.smoothness_dd_factors=' + "'" + str(smoothness_dd_factors).replace(' ', '') + "' "
        cmd += 'ddecal.smoothnessconstraint=' + str(np.max(SMconstraint) * 1e6) + ' '
        cmd += 'ddecal.smoothnessreffrequency=' + str(SMconstraintreffreq * 1e6) + ' '
        cmd += 'ddecal.smoothnessspectralexponent=' + str(SMconstraintspectralexponent) + ' '
        cmd += 'ddecal.smoothnessrefdistance=' + str(SMconstraintrefdistance * 1e3) + ' '  # input units in km

    if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'tec+phase', \
                   'tec+delay', 'tec+phase+delay', 'delay', 'rotation+scalarphase', \
                   'rotation+diagonalphase', 'faradayrotation+scalarphase', \
                   'faradayrotation+diagonalphase', 'faradayrotation']:
        cmd += 'ddecal.tolerance=' + str(tolerance) + ' '
        if soltype in ['tecandphase', 'tec']:
            cmd += 'ddecal.approximatetec=True '
            cmd += 'ddecal.stepsize=0.2 '
            cmd += 'ddecal.maxapproxiter=45 '
            cmd += 'ddecal.approxtolerance=6e-3 '
        if soltype in ['tec+delay', 'tec+phase+delay']: 
            cmd += 'ddecal.max_tec_delay_wraps=' + str(max_tec_delay_wraps) + ' '
    if soltype in ['complexgain', 'scalarcomplexgain', 'scalaramplitude', 'amplitudeonly', \
                   'rotation+diagonal', 'fulljones', 'rotation+scalar', \
                   'rotation+diagonalamplitude', 'rotation+scalaramplitude', \
                   'faradayrotation+diagonal', 'faradayrotation+scalar', \
                   'faradayrotation+diagonalamplitude', 'faradayrotation+scalaramplitude', \
                   'leakage', 'leakageamplitude']:
        cmd += 'ddecal.tolerance=' + str(tolerance) + ' '  # for now the same as phase soltypes
    # cmd += 'ddecal.detectstalling=False '


    # DETERMINE IF WE CAN USE SOLUTIONS FROM PREVIOUS SELFCAL CYCLE
    if args['startfrominitialsolutions']:
        terminal_print('Checking if a solution file from a previous selfcalcycle is present...')
        # update_sourcedir_h5_dde() will cause issues? 
        initialsolutions_exist = False
        initialsolutions_directions_equal = False
        current_ndir = len(modeldatacolumns)
        if (len(modeldatacolumns_solve) > 0) and (len(modeldatacolumns) != len(modeldatacolumns_solve)):
            # means that we require parmdb + .backup 
            # number directions being solved for is different as soltypelist_includedir functionality is used    
            current_ndir = len(modeldatacolumns_solve)
            parmdbtmp = parmdb + '.backup'
        else:
            parmdbtmp = parmdb
            
        if selfcalcycle == 0:  # do deal with the "skyselfcalcycle if it exists"
            previous_parmdb = parmdbtmp.replace('selfcalcycle'+str(0).zfill(3),'skyselfcalcycle'+str(0).zfill(3))
            if os.path.isfile(previous_parmdb): initialsolutions_exist = True
                    
        else:
            previous_parmdb = parmdbtmp.replace('selfcalcycle'+str(selfcalcycle).zfill(3), 'selfcalcycle'+str(selfcalcycle-1).zfill(3))
            if os.path.isfile(previous_parmdb): initialsolutions_exist = True
                
        if initialsolutions_exist: # we autatically guarantee that the soltype is the same because we check for the parmdb name which contains the soltype in there 
        # check n_dir equals current number of dirctions to solve
            terminal_print('Found initial solution file:',previous_parmdb)
            with tables.open_file(previous_parmdb) as Hprev:
                for soltab in Hprev.root.sol000._v_groups.keys():
                    previous_ndir = len(Hprev.root.sol000._f_get_child(soltab).dir[:])
                terminal_print('Number of directions in ' + previous_parmdb + ':', 'Previous ndir:', previous_ndir)     
            if previous_ndir <= 1 and current_ndir <=1: #this is a DI solve
                initialsolutions_directions_equal = True
            else:
               if len(previous_ndir) == len(current_ndir): 
                   initialsolutions_directions_equal = True # this is a DD solve
                   # number of directions did not change from previous selfcalcycle     

        if initialsolutions_exist and initialsolutions_directions_equal: 
            # if we get here it means we can can use ddecal.initialsolutions
            cmd += 'ddecal.initialsolutions.h5parm=' + previous_parmdb + ' '
            if soltypein == 'fulljones' or soltypein == 'leakage' or soltypein == 'leakageamplitude': 
                # just for extra safety 
                # cmd += 'ddecal.initialsolutions.gaintype=fulljones ' 
                # for all soletypes DP3 should be able to figure it out in principle by itself 
                if soltypein == 'leakageamplitude':
                    cmd += 'initialsolutions.soltab=[amplitude000] '
                else:    
                    cmd += 'initialsolutions.soltab=[amplitude000,phase000] '
            # set ddecal.initialsolutions.soltab
            else:
                with tables.open_file(previous_parmdb) as Hprev:
                   soltabs = list(Hprev.root.sol000._v_groups.keys())
                   if 'error000' in soltabs: soltabs.remove('error000')
                   
                cmd += "ddecal.initialsolutions.soltab='[" + ','.join(map(str, soltabs)) + "]' "
    #print('^^^^^^^^^^^')
    #print('^^^^^^^^^^^')
    #print(soltypein)
    #if soltypein == 'tec+phase+delay':
    #    cmd += "ddecal.initialsolutions.h5parm=h5_solutions/scalarphase1_skyselfcalcycle000_J174713+653236_DI.concat.ms.avg.phaseup.h5 ddecal.initialsolutions.soltab='phase000' "


    ms_tmp = None # set to None so that we can check if it is created later
    if solve_msinnchan != 'all':
        # if we are solving for aa select channel range, we need to create a temporary MS with the average function
        
        # create a list of the same columns as in the original MS 
        columns_to_create = []
        if incol != 'DATA':
            columns_to_create.append(incol)
        if len(modeldatacolumns) == 0:
            columns_to_create.append(modeldata)    
        else:
            if len(modeldatacolumns_solve) > 0:
                for mdc in modeldatacolumns_solve:
                    columns_to_create.append(mdc)
            else:
                for mdc in modeldatacolumns:
                    columns_to_create.append(mdc)
        columns_to_create.append('DATA') # to create DATA because XY and YX were set to zero above
        
        ms_tmp = create_splitted_ms(ms, columns_to_create, solve_msinnchan=solve_msinnchan, 
                                    solve_msinstartchan=solve_msinstartchan, dysco=dysco, 
                                    modelstoragemanager=modelstoragemanager, incol=incol, 
                                    metadata_compression=args['metadata_compression'])

        # now replace the MS in the cmd command string with the new temporary MS
        cmd = cmd.replace('msin=' + ms, 'msin=' + ms_tmp)
        
    
    # RUN THE SOLVE WITH DP3
    terminal_print('DP3 solve:', cmd)
    logger.info('DP3 solve: ' + cmd)
    run(cmd)

    if ms_tmp is not None:
        # remove the temporary MS
        terminal_print('Removing temporary MS:', ms_tmp)
        shutil.rmtree(ms_tmp, ignore_errors=True)

    if selfcalcycle == 0 and (soltypein == "scalarphasediffFR" or soltypein == "scalarphasediff") and not args['phasediff_only']:
        shutil.copy(parmdb, parmdb + '.scbackup')

    if (len(modeldatacolumns_solve) > 0) and (len(modeldatacolumns) != len(modeldatacolumns_solve)):
        # fix coordinates otherwise h5merge will merge all directions into one when add_directions is done (as all coordinates are the same up to this point)
        update_sourcedir_h5_dde(parmdb, 'facet_regions/facetdirections.p', dir_id_kept=dir_id_kept)

        # we need to add back the extra direction into the h5 file
        outparmdb = 'h5_solutions/' + 'adddirback' + os.path.basename(parmdb)
        if os.path.isfile(outparmdb):
            Path(outparmdb).unlink(missing_ok=True)  # remove the file if it exists
        merge_h5(h5_out=outparmdb, h5_tables=parmdb, add_directions=sourcedir_removed.tolist(), propagate_weights=False, convert_tec=False)

        # now we split them all into separate h5 per direction so we can reorder and fill them
        terminal_print('Splitting directions into separate h5')
        split_multidir(outparmdb)

        # fill the added emtpy directions with the closest ones that were solved for
        terminal_print('Copy over solutions from skipped directions')
        copy_over_solutions_from_skipped_directions(modeldatacolumns, dir_id_kept)

        # create backup of parmdb and remove orginal and cleanup
        shutil.copy(parmdb, parmdb + '.backup')
        Path(parmdb).unlink(missing_ok=True)
        Path(outparmdb).unlink(missing_ok=True)

        # merge h5 files in order of the directions in facet_regions/facetdirections.p and recreate parmdb
        # clean up previously splitted directions inside this function
        terminal_print('Merge h5 files in correct order and recreate parmdb')
        merge_splitted_h5_ordered(modeldatacolumns, parmdb, clean_up=True)

        # fix direction names
        update_sourcedirname_h5_dde(parmdb, modeldatacolumns)
        # sys.exit()

    if len(modeldatacolumns) > 1:  # and DDE_predict == 'WSCLEAN':
        update_sourcedir_h5_dde(parmdb, 'facet_regions/facetdirections.p')

    if has0coordinates(parmdb):
        logger.warning('Direction coordinates are zero in: ' + parmdb)

    # Rotation checking
    if soltype in ['rotation', 'rotation+diagonal', 'rotation+diagonalphase', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude', 'rotation+scalarphase']:
        remove_nans(parmdb, 'rotation000')
        fix_weights_rotationh5(parmdb)
        refant = findrefant_core(parmdb, telescope=args['telescope'])
        force_close(parmdb)
        fix_rotationreference(parmdb, refant)

    # Faraday rotation checking
    if soltype in ['faradayrotation', 'faradayrotation+diagonal', 'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude',
                   'faradayrotation+scalar', 'faradayrotation+scalaramplitude', 'faradayrotation+scalarphase']:
        remove_nans(parmdb, 'rotationmeasure000')
        fix_weights_rotationmeasureh5(parmdb)
        refant = findrefant_core(parmdb, telescope=args['telescope'])
        force_close(parmdb)
        fix_rotationmeasurereference(parmdb, refant)

    # tec checking
    if soltype in ['tec', 'tecandphase', 'tec+phase', 'tec+delay', 'tec+phase+delay']:
        remove_nans(parmdb, 'tec000')
        refant = findrefant_core(parmdb, telescope=args['telescope'])
        fix_tecreference(parmdb, refant)
        force_close(parmdb)

    # delay checking
    if soltype in ['tec+delay', 'tec+phase+delay', 'delay']:
        remove_nans(parmdb, 'delay000')
        refant = findrefant_core(parmdb, telescope=args['telescope'])
        fix_delayreference(parmdb, refant)
        force_close(parmdb)    

    # phase checking
    if soltype in ['rotation+diagonal', 'rotation+diagonalphase',
                   'rotation+scalar', 'rotation+scalarphase',
                   'scalarcomplexgain', 'complexgain', 'scalarphase',
                   'phaseonly', 'faradayrotation+diagonal', 'faradayrotation+scalar', 'faradayrotation+diagonalphase', 'faradayrotation+scalarphase']:
        remove_nans(parmdb, 'phase000')
        refant = findrefant_core(parmdb, telescope=args['telescope'])
        fix_phasereference(parmdb, refant)
        force_close(parmdb)

    if int(maxiter) == 1:  # this is a template solve only
        terminal_print('Template solve, not going to make plots or do solution flagging')
        return

    outplotname = parmdb.split('_' + os.path.basename(ms) + '.h5')[0]

    if incol == 'DATA_CIRCULAR_PHASEDIFF':
        terminal_print('Manually updating H5 to get the phase difference correct')
        refant = findrefant_core(parmdb, telescope=args['telescope'])  # phase matrix plot
        force_close(parmdb)
        makephasediffh5(parmdb, refant)
    if incol == 'DATA_CIRCULAR_PHASEDIFF' and soltypein == 'scalarphasediffFR':
        terminal_print('Fiting for Faraday Rotation with losoto on the phase differences')
        # work with copies H5 because losoto changes the format splitting off the length 1 direction axis creating issues
        # with H5merge (also add additional solution talbes which we do not want)
        shutil.copy(parmdb, 'FRcopy' + parmdb)
        losoto_parsetFR = create_losoto_FRparset(ms, refant=findrefant_core(parmdb, telescope=args['telescope']), outplotname=outplotname,dejump=dejumpFR,skip_solution_plotting=args["skip_solution_plotting"])
        run('losoto ' + 'FRcopy' + parmdb + ' ' + losoto_parsetFR)
        rotationmeasure_to_phase('FRcopy' + parmdb, parmdb, dejump=dejumpFR)
        if not args["skip_solution_plotting"]:
            run('losoto ' + parmdb + ' ' + create_losoto_FRparsetplotfit(ms, refant=findrefant_core(parmdb, telescope=args['telescope']),
                                                                     outplotname=outplotname))
        force_close(parmdb)

    if incol == 'DATA_PHASE_SLOPE':
        terminal_print('Manually updating H5 to get the cumulative phase')
        # makephaseCDFh5(parmdb)
        makephaseCDFh5_h5merger(parmdb, ms, modeldatacolumns)

    if resetsols is not None and resetsols != 'all':
        # if resetsols == 'all' we postpone the reset to after the  autoauto_flag_antennas step further below
        if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'fulljones', \
                       'complexgain', 'scalarcomplexgain', 'rotation+diagonal', \
                       'rotation+diagonalamplitude', 'rotation+diagonalphase', \
                       'rotation+scalar', 'rotation+scalarphase', 'rotation+scalaramplitude', \
                       'faradayrotation', 'faradayrotation+diagonal', \
                       'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', \
                       'faradayrotation+scalar', 'faradayrotation+scalaramplitude', \
                       'faradayrotation+scalarphase', 'leakage', 'tec+phase', 'tec+delay', \
                       'tec+phase+delay', 'delay']:
            refant = findrefant_core(parmdb, telescope=args['telescope'])
            force_close(parmdb)
        else:
            refant = None
        resetsolsforstations(parmdb, antennaconstraintstr(resetsols, antennasms, HBAorLBA, useforresetsols=True,
                                                          telescope=args['telescope']), refant=refant, telescope=args['telescope'])

    if resetdir is not None:
        if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'fulljones', \
                       'complexgain', 'scalarcomplexgain', 'rotation+diagonal', \
                       'rotation+diagonalamplitude', 'rotation+diagonalphase', \
                       'rotation+scalar', 'rotation+scalarphase', 'rotation+scalaramplitude', \
                       'faradayrotation', 'faradayrotation+diagonal', \
                       'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude',\
                       'faradayrotation+scalar', 'faradayrotation+scalaramplitude', \
                       'faradayrotation+scalarphase', 'leakage', 'tec+phase', 'tec+delay', 'tec+phase+delay', 'delay']:
            refant = findrefant_core(parmdb, telescope=args['telescope'])
            force_close(parmdb)
        else:
            refant = None

        resetsolsfordir(parmdb, resetdir, refant=refant, telescope=args['telescope'])

    if number_freqchan_h5(parmdb) > 1:
        onechannel = False
    else:
        onechannel = True

    # Check for bad values (amplitudes/fulljones)
    if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude', \
                   'fulljones', 'rotation+diagonal', 'rotation+diagonalamplitude', \
                   'rotation+scalar', 'rotation+scalaramplitude', 'faradayrotation+diagonal', \
                   'faradayrotation+scalaramplitude', 'faradayrotation+diagonalamplitude', \
                   'faradayrotation+scalar', 'leakage', 'leakageamplitude']:
        if resetdir is not None or resetsols is not None:
            flag_bad_amps(parmdb, setphases=includesphase, flagamp1=False,
                          flagampxyzero=False)  # otherwise it flags the solutions which where reset
        else:
            # if leakage or leakageamplitude we set flagamp1=False
            flag_bad_amps(parmdb, setphases=includesphase, flagamp1=(not (soltype == 'leakage' or soltype == 'leakageamplitude')))
        if soltype == 'fulljones' or soltype =='leakage': 
            removenans_fulljones(parmdb)
        elif soltype == 'leakageamplitude':
            removenans_amplitude_leakage(parmdb)
        else:
            remove_nans(parmdb, 'amplitude000')
        medamp = get_median_amp(parmdb)

        if soltype != 'amplitudeonly' and soltype != 'scalaramplitude' and soltype != 'leakageamplitude' \
                and soltype != 'rotation+diagonalamplitude' \
                and soltype != 'rotation+scalaramplitude' \
                and soltype != 'faradayrotation+scalaramplitude' \
                and soltype != 'faradayrotation+diagonalamplitude' :
            remove_nans(parmdb, 'phase000')

        if soltype == 'fulljones' or soltype == 'leakage' or soltype == 'leakageamplitude':
            # if normamps: #and all(rsl is None for rsl in resetsols_list) and all(rdl is None for rdl in resetdir_list):
            # otherwise you get too much setting to 1 due to large amp deviations, in particular fullones on raw data which has very high correlator amps (with different ILT vals), also resets in that case cause issues (resets are ok if the amplitudes are close to 1). Hence using the normamps test seems the most logical choice
            flaglowamps_fulljones(parmdb, lowampval=medamp / ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                                  setphases=includesphase)
            flaghighamps_fulljones(parmdb, highampval=medamp * ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                                   setphases=includesphase)
        else:
            # if normamps: #and all(rsl is None for rsl in resetsols_list) and all(rdl is None for rdl in resetdir_list):
            # otherwise you get too much setting to 1 due to large amp deviations, in particular fullones on raw data which has very high correlator amps (with different ILT vals), also resets in that case cause issues (resets are ok if the amplitudes are close to 1).  Hence using the normamps test seems the most logical choice
            flaglowamps(parmdb, lowampval=medamp / ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor), setphases=includesphase)
            flaghighamps(parmdb, highampval=medamp * ampresetvalfactor, flagging=(flagging or flag_ampresetvalfactor),
                         setphases=includesphase)

        if (soltype == 'fulljones' or soltype == 'leakage' or soltype == 'leakageamplitude') and clipsolutions:
            terminal_print('Fulljones/leakage/leakageamplitude and solution clipping not supported')
            raise Exception('Fulljones/leakage/leakageamplitude and clipsolutions not implemtened')
        if clipsolutions:
            flaglowamps(parmdb, lowampval=clipsollow, flagging=True, setphases=True)
            flaghighamps(parmdb, highampval=clipsolhigh, flagging=True, setphases=True)

    # find bad antennas based on solution stats
    if auto_flag_antennas:
        if args['telescope'] == 'MeerKAT': # only MeerKAT for now
            if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude']:
                flag_ant_list = find_bad_deviating_antennas(parmdb, ms)
                for ant in flag_ant_list:
                    logger.info('Auto-flagging bad MeerKAT antenna based on solution stats: ' + str(ant) + ' in ' + ms)
                    # print in orange color to make it more visible in the logs
                    terminal_print('\033[33mAuto-flagging bad MeerKAT antennas based on solution stats: ' + str(ant) + ' in ' + ms + '\033[0m')
                    # use taql function to flag antennas
                    flag_antenna_taql(ms, ant)
            # take take care of outliers in the solutions here as well
            # in particular for data near zero  declination which might be affected by geostationary RFI which causes high amplitude outliers in the solutions
            if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude']:
                medamp = get_median_amp(parmdb)
                flaglowamps(parmdb, lowampval=medamp / 1.5, flagging=False, setphases=True)
                flaghighamps(parmdb, highampval=medamp * 1.5, flagging=False, setphases=True)        

    if resetsols == 'all':
        if soltype in ['phaseonly', 'scalarphase', 'tecandphase', 'tec', 'rotation', 'fulljones', \
                       'leakage', 'leakageamplitude', 'complexgain', 'scalarcomplexgain', \
                       'rotation+diagonal', 'rotation+diagonalamplitude', \
                       'rotation+diagonalphase', 'rotation+scalar', 'rotation+scalarphase',\
                       'rotation+scalaramplitude', 'faradayrotation', 'faradayrotation+diagonal', \
                       'faradayrotation+diagonalphase', 'faradayrotation+diagonalamplitude', \
                       'faradayrotation+scalar', 'faradayrotation+scalaramplitude', \
                       'faradayrotation+scalarphase', 'tec+phase', 'tec+delay', 'tec+phase+delay', 'delay']:
            refant = findrefant_core(parmdb, telescope=args['telescope'])
            force_close(parmdb)
        else:
            refant = None
       
        # if requested do flagging of solutions here before resetting all solutions (because then no bad solution will be found anymore)
        # in this way the flags will make it to the merged solution file so that bad visibilities will be flagged for DI solves
        # only do this for DI solve types
        if not args['DDE'] and flagging and not onechannel and ntimesH5(parmdb) > 1 and \
            soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', \
            'scalaramplitude','rotation+diagonal', 'rotation+diagonalamplitude', \
            'rotation+scalar', 'rotation+scalaramplitude', 'faradayrotation+diagonal', \
            'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', \
            'faradayrotation+scalaramplitude'] and args['flagsolutionsbeforeresetsolsall']:
        
            losotoparset = create_losoto_flag_ap_only(ms, maxrms=flagslowamprms,
                                                      maxrmsphase=flagslowphaserms,
                                                      includesphase=includesphase, onechannel=onechannel,
                                                      flagphases=flagslowphases, onetime=ntimesH5(parmdb)==1)
  
            shutil.copy(parmdb, parmdb + '.allresetsolbackup_gridflagging')
            terminal_print('losoto ' + parmdb + ' ' + losotoparset)
            run('losoto ' + parmdb + ' ' + losotoparset, log=True)

        # make a backup of the parmdb before resetting all solutions
        shutil.copy(parmdb, parmdb + '.allresetsolbackup')
        resetsolsforstations(parmdb, antennaconstraintstr(resetsols, antennasms, HBAorLBA, useforresetsols=True,
                                                          telescope=args['telescope']), refant=refant, telescope=args['telescope'])

    # ---------------------------------
    # ---------------------------------
    # makes plots and do LOSOTO flagging
    if soltype in ['rotation', 'rotation+diagonal', 'rotation+diagonalamplitude',
                   'rotation+scalar', 'rotation+scalaramplitude',
                   'rotation+scalarphase', 'rotation+diagonalphase'] and not args["skip_solution_plotting"]:
            losotoparset_rotation = create_losoto_rotationparset(ms, onechannel=onechannel, outplotname=outplotname + 'ROT',
                                                                 refant=findrefant_core(parmdb, telescope=args['telescope']))  # phase matrix plot
            force_close(parmdb)
            cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_rotation
            if onechannel and (ntimesH5(parmdb) == 1): 
                terminal_print('Skipping losoto rotation plot because only one time and one frequency channel')
            else:
                terminal_print('LoSoTo command:', cmdlosoto)
                logger.info(cmdlosoto)
                run(cmdlosoto)

    if soltype in ['rotation+scalarphase', 'rotation+diagonalphase', 'faradayrotation+scalarphase','faradayrotation+diagonalphase'] and not args["skip_solution_plotting"]:
            losotoparset_phase = create_losoto_fastphaseparset(ms, onechannel=onechannel, onepol=onepol,
                                                               outplotname=outplotname,
                                                               refant=findrefant_core(parmdb, telescope=args['telescope']), onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb))  # phase matrix plot
            cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_phase
            force_close(parmdb)

            # cannot plot if only one time AND one frequency channel
            if onechannel and (ntimesH5(parmdb) == 1): 
                terminal_print('Skipping losoto phase plot for rotation+scalarphase or rotation+diagonalphase because only one time and one frequency channel')
            else:
                terminal_print('LoSoTo command:', cmdlosoto)
                logger.info(cmdlosoto)
                run(cmdlosoto)

    if soltype in ['phaseonly', 'scalarphase'] and not args['phasediff_only'] and not args["skip_solution_plotting"]:
            losotoparset_phase = create_losoto_fastphaseparset(ms, onechannel=onechannel, onepol=onepol,
                                                               outplotname=outplotname,
                                                               refant=findrefant_core(parmdb, telescope=args['telescope']), onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb))  # phase matrix plot
            cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_phase
            force_close(parmdb)
            if onechannel and (ntimesH5(parmdb) == 1): 
                terminal_print('Skipping losoto phase plot for phaseonly or scalarphase because only one time and one frequency channel')
            else:
                terminal_print('LoSoTo command:', cmdlosoto)
                logger.info(cmdlosoto)
                run(cmdlosoto)

    if soltype in ['tecandphase', 'tec', 'tec+phase', 'tec+delay', 'tec+phase+delay', 'delay'] and not args['skip_solution_plotting']:
        tecandphaseplotter(parmdb, ms, telescope=args['telescope'],
                           outplotname=outplotname)  # use own plotter because losoto cannot add tec and phase

    if soltype in ['tec'] and not args["skip_solution_plotting"]:
        losotoparset_tec = create_losoto_tecparset(ms, outplotname=outplotname,
                                                   refant=findrefant_core(parmdb, telescope=args['telescope']),
                                                   markersize=compute_markersize(parmdb))
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset_tec
        terminal_print('LoSoTo command:', cmdlosoto)
        logger.info(cmdlosoto)
        run(cmdlosoto)
        force_close(parmdb)

    if soltype in ['scalarcomplexgain', 'complexgain', 'amplitudeonly', 'scalaramplitude', \
                   'fulljones', 'rotation+diagonal', 'rotation+diagonalamplitude', \
                   'rotation+scalar', 'rotation+scalaramplitude', 'faradayrotation+diagonal', \
                   'faradayrotation+diagonalamplitude', 'faradayrotation+scalar', \
                   'faradayrotation+scalaramplitude', 'leakage', 'leakageamplitude']:
        terminal_print('Do flagging?:', flagging)
        if flagging and not onechannel and ntimesH5(parmdb) > 1 :
            if soltype == 'fulljones' or soltype == 'leakage' or soltype == 'leakageamplitude':
                terminal_print('Fulljones/leakage/leakageamplitude and flagging not implemtened')
                raise Exception('Fulljones/leakage/leakageamplitude and flagging not implemtened')
            else:
                losotoparset = create_losoto_flag_apgridparset(ms, flagging=True, maxrms=flagslowamprms,
                                                               maxrmsphase=flagslowphaserms,
                                                               includesphase=includesphase, onechannel=onechannel,
                                                               medamp=medamp, flagphases=flagslowphases, onepol=onepol,
                                                               outplotname=outplotname, refant=findrefant_core(parmdb, telescope=args['telescope']), onetime=ntimesH5(parmdb)==1, markersize=compute_markersize(parmdb), skip_solution_plotting=args["skip_solution_plotting"])
                force_close(parmdb)
        else:
            losotoparset = create_losoto_flag_apgridparset(ms, flagging=False, includesphase=includesphase,
                                                           onechannel=onechannel, medamp=medamp, onepol=onepol,
                                                           outplotname=outplotname,
                                                           refant=findrefant_core(parmdb, telescope=args['telescope']),
                                                           fulljones=(fulljonesparmdb(parmdb) or amplitude_leakage_paramdb(parmdb)),onetime=ntimesH5(parmdb)==1,markersize=compute_markersize(parmdb), skip_solution_plotting=args["skip_solution_plotting"])
            force_close(parmdb)

        # MAKE losoto command
        if flagging:
            shutil.copy(parmdb, parmdb + '.backup')
        cmdlosoto = 'losoto ' + parmdb + ' ' + losotoparset
        if onechannel and (ntimesH5(parmdb) == 1): 
            terminal_print('Skipping losoto amplitude plot because only one time and one frequency channel')
        else:
            terminal_print('LoSoTo command:', cmdlosoto)
            logger.info(cmdlosoto)
            run(cmdlosoto)
    
    if is_multidir(parmdb):
        same_weights_multidir(parmdb)
    
    if len(tables.file._open_files.filenames) >= 1:  # for debugging
        terminal_print('End runDPPPbase, some HDF5 files are not closed:', tables.file._open_files.filenames)
        force_close(parmdb)

    # remove any add_dir*.h5 files that might have been created by h5_merger.py
    for h5add in glob.glob("add_dir_*.h5") + glob.glob("add_dir.h5"):
        Path(h5add).unlink(missing_ok=True)
    return


def remove_outside_box(mslist, imagebasename, pixsize, imsize,
                       channelsout, single_dual_speedup=True,
                       outcol='SUBTRACTED_DATA', dysco=True, userbox=None,
                       idg=False, h5list=[], facetregionfile=None,
                       disable_primary_beam=False, ddcor=True, parallelgridding=1,
                       metadata_compression=True, avgfreqstep=1, avgtimestep=1):
    """
    Removes emission outside a specified box region from measurement sets (MS) by predicting and subtracting the model
    within the defined region. Optionally applies direction-dependent calibration corrections and manages output columns.
    Parameters
    ----------
    mslist : list of str
        List of measurement set file paths to process.
    imagebasename : str
        Basename for the image FITS file (expects '-MFS-image.fits' suffix).
    pixsize : float
        Pixel size for imaging (arcseconds or degrees, depending on context).
    imsize : int
        Image size (number of pixels per side).
    channelsout : int
        Number of output channels for imaging.
    single_dual_speedup : bool, optional
        If True, enables speedup for single/dual polarization (default: True).
    outcol : str, optional
        Name of the output data column to store subtracted/extracted data (default: 'SUBTRACTED_DATA').
    dysco : bool, optional
        If True, uses DYSCO storage manager for output (default: True).
    userbox : float, str, or None, optional
        User-specified box size in degrees, a region file, 'keepall', or None to use default (default: None).
    idg : bool, optional
        If True, uses IDG for imaging (default: False).
    h5list : list of str, optional
        List of H5 calibration tables for DDE calibration (default: []).
    facetregionfile : str or None, optional
        Path to facet region file for DDE calibration (default: None).
    disable_primary_beam : bool, optional
        If True, disables primary beam correction during prediction (default: False).
    ddcor : bool, optional
        If True, applies direction-dependent corrections after subtraction (default: True).
    parallelgridding : int, optional
        Number of parallel gridding threads (default: 1).
    metadata_compression : bool, optional
        If True, enables metadata compression for output (default: True).
    avgfreqstep : int, optional
        Step size for frequency averaging (default: 1).
    avgtimestep : int, optional
        Step size for time averaging (default: 1).
    Returns
    -------
    None
    Notes
    -----
    - If `userbox` is a float, it is interpreted as the box size in degrees.
    - If `userbox` is a region file or 'keepall', it controls the region to keep or disables subtraction, respectively.
    - The function creates a region file for the box if needed, predicts the model, subtracts it from the data, and averages the result.
    - If DDE calibration tables are provided (`h5list`), direction-dependent corrections are applied.
    - Temporary columns and files may be created and removed to manage disk space.
    """
    # get imageheader to check frequency
    hdul = fits.open(imagebasename + '-MFS-image.fits')
    header = hdul[0].header

    if len(h5list) != 0:
        datacolumn = 'DATA'  # for DDE
    else:
        datacolumn = 'CORRECTED_DATA'

    if (header['CRVAL3'] < 500e6):  # means we have LOFAR?, just a random value here
        boxsize = 1.5  # degr
    if (header['CRVAL3'] > 500e6) and (header['CRVAL3'] < 1.0e9):  # UHF-band
        boxsize = 3.0  # degr
    if (header['CRVAL3'] >= 1.0e9) and (header['CRVAL3'] < 1.7e9):  # L-band
        boxsize = 2.0  # degr
    if (header['CRVAL3'] >= 1.7e9) and (header['CRVAL3'] < 4.0e9):  # S-band
        boxsize = 1.5  # degr
        
    try:
        boxsize = float(userbox) # userbox is a number
        # overwrite boxsize value here with user specified number
        userbox = 'templatebox.reg' # set this as we are going to use the template region file                 
    except ValueError:
        if userbox is not None and userbox != 'keepall' and userbox != 'auto':
            # so we have a user provided regionfile
            if not os.path.isfile(userbox):
                raise Exception('User provided region file ' + userbox + ' does not exist')
            boxsize = getregionsize(userbox)  # get the boxsize from the user provided region file    
        
    if userbox == 'auto':
        if args['remove_outside_center_auto_minboxsize'] is not None:
            min_extract_size = args['remove_outside_center_auto_minboxsize']
        elif args['telescope'] == 'MeerKAT':
            min_extract_size = (57.5/60)*(1.5e9/header['CRVAL3'])  # MeerKAT FWHM primary beam in degr
        elif args['telescope'] == 'GMRT':
            min_extract_size = (32.25/60)*(1.08e9/header['CRVAL3'])  # GMRT FWHM primary beam in degr
        elif args['telescope'] == 'ASKAP':
            min_extract_size = 1.83*(850e6/header['CRVAL3'])  # ASKAP FWHM primary beam in degr
        elif args['telescope'] == 'EVLA' or args['telescope'] == 'VLA':
            min_extract_size = (30.0/60)*(1.4e9/header['CRVAL3'])  # EVLA/VLA FWHM primary beam in degr
        else:
            min_extract_size = 1.0  # degr     
        boxsize = auto_determine_extractregion(imagebasename + '-MFS-image.fits', min_extract_size=min_extract_size)
        userbox = 'templatebox.reg' # set this as we are going to use the template region file
    # create square box file templatebox.reg

    region_string = """
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1
    fk5
    box(137.2868914,-9.6412498,7200.000",7200.000",0)
    """
    r = pyregion.parse(region_string)
    r.write('templatebox.reg')
    r = pyregion.open('templatebox.reg')
    r[0].coord_list[0] = header['CRVAL1']  # units degr
    r[0].coord_list[1] = header['CRVAL2']  # units degr
    r[0].coord_list[2] = boxsize  # units degr
    r[0].coord_list[3] = boxsize  # units degr
    r.write('templatebox.reg')
    hdul.close()
    # predict the model non-DDE case
    if len(h5list) == 0:
        if userbox is None:
            makeimage(mslist, imagebasename, pixsize, imsize,
                      channelsout, onlypredict=True, squarebox='templatebox.reg',
                      idg=idg, disable_primarybeam_predict=disable_primary_beam,
                      fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
            phaseshiftbox = 'templatebox.reg'
        else:
            if userbox != 'keepall':
                makeimage(mslist, imagebasename, pixsize, imsize,
                          channelsout, onlypredict=True, squarebox=userbox,
                          idg=idg, disable_primarybeam_predict=disable_primary_beam,
                          fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
                phaseshiftbox = userbox
            else:
                phaseshiftbox = None  # so option keepall was set by the user
    else:  # so this we are in DDE mode as h5list is not empty
        if userbox is None:
            makeimage(mslist, imagebasename, pixsize, imsize,
                      channelsout, onlypredict=True, squarebox='templatebox.reg',
                      idg=idg, h5list=h5list, facetregionfile=facetregionfile,
                      disable_primarybeam_predict=disable_primary_beam,
                      fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
            phaseshiftbox = 'templatebox.reg'
        else:
            if userbox != 'keepall':
                makeimage(mslist, imagebasename, pixsize, imsize,
                          channelsout, onlypredict=True, squarebox=userbox,
                          idg=idg, h5list=h5list, facetregionfile=facetregionfile,
                          disable_primarybeam_predict=disable_primary_beam,
                          fulljones_h5_facetbeam=not single_dual_speedup, parallelgridding=parallelgridding)
                phaseshiftbox = userbox
            else:
                phaseshiftbox = None  # so option keepall was set by the user
    # write new data column (if keepall was not set) where rest of the field outside the box is removed
    if phaseshiftbox is not None:
        stepsize = 100000
        
        for ms in mslist:
            # check if outcol exists, if not create it with DP3
            t = table(ms)
            colnames = t.colnames()
            t.close()
            if outcol not in colnames:
                cmd = 'DP3 msin=' + ms + ' msout=. steps=[] msout.datacolumn=' + outcol + ' '
                cmd += 'msin.datacolumn=' + datacolumn + ' '
                if dysco:
                    cmd += 'msout.storagemanager=dysco '
                    cmd += 'msout.storagemanager.weightbitrate=16 '
                terminal_print('Command:', cmd)
                run(cmd)
            t = table(ms, readonly=False)
            if t.nrows() < stepsize: stepsize = t.nrows()
            for row in range(0, t.nrows(), stepsize):
                terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                data = t.getcol(datacolumn, startrow=row, nrow=stepsize, rowincr=1)
                model = t.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
                t.putcol(outcol, data - model, startrow=row, nrow=stepsize, rowincr=1)
            t.close()
        # do not average here in case we are applying the DD solutions later, so if len(h5list) != 0 and ddcor and userbox != 'keepall' (because otherwise we apply the solution on averaged data later which is less ideal)
        average(mslist, args=args, freqstep=([1] * len(mslist) if len(h5list) != 0 and ddcor and userbox != 'keepall' else [avgfreqstep] * len(mslist)), timestep=(1 if len(h5list) != 0 and ddcor and userbox != 'keepall' else avgtimestep), \
                phaseshiftbox=phaseshiftbox, dysco=dysco, make_extract=True,
                dataincolumn=outcol, metadata_compression=metadata_compression)
        remove_column_ms(mslist, outcol) # remove SUBTRACTED_DATA to free up space
    else:  # so have have "keepall", no subtract, just a copy
        average(mslist, args=args, freqstep=[avgfreqstep] * len(mslist), timestep=avgtimestep,
                phaseshiftbox=phaseshiftbox, dysco=dysco, make_extract=True,
                dataincolumn=datacolumn, metadata_compression=metadata_compression)
        # applycal of closest direction (in multidir h5)
        
    if len(h5list) != 0 and not ddcor and userbox != 'keepall':
    # in this case we do not apply the solutions, however, the flagging information in the merged solution files need to be applied to the extracted MS to make sure that bad data is flagged
    # we apply the clostest direction in the multidir h5 into a temporary column and then remove the column afterwards so that the flags are applied but the original data is kept
    # the temporary column is called DATA_TMP and is removed at the end of this step to free up space
        for ms_id, ms in enumerate(mslist):
            applycal(os.path.basename(ms) + '.extracted', h5list[ms_id], find_closestdir=True, \
                     dysco=dysco, metadata_compression=metadata_compression, msoutcol='DATA_TMP')
            # remove DATA_TMP column to free up space
            remove_column_ms([os.path.basename(ms) + '.extracted'], 'DATA_TMP')

    
    # applycal of closest direction (in multidir h5)
    if len(h5list) != 0 and ddcor and userbox != 'keepall':
        for ms_id, ms in enumerate(mslist):
            ms = os.path.basename(ms)
            if os.path.isdir(ms + '.extracted_ddcor'):
                shutil.rmtree(ms + '.extracted_ddcor', ignore_errors=True)
                time.sleep(2)  # wait for the directory to be removed
            applycal(ms + '.extracted',h5list[ms_id], find_closestdir=True, 
                     msout=ms + '.extracted_ddcor', dysco=dysco, metadata_compression=metadata_compression)
            with table(ms + '.extracted') as t:
                if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():  # check if WEIGHT_SPECTRUM_SOLVE is present otherwise this is not needed
                    terminal_print('Going to copy over WEIGHT_SPECTRUM_SOLVE')
                    # Make a WEIGHT_SPECTRUM from WEIGHT_SPECTRUM_SOLVE
                    with table(ms + '.extracted_ddcor', readonly=False) as t2:
                        terminal_print('Adding WEIGHT_SPECTRUM_SOLVE')
                        addcol(t2, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_SOLVE')
                        imweights = t.getcol('WEIGHT_SPECTRUM_SOLVE')
                        t2.putcol('WEIGHT_SPECTRUM_SOLVE', imweights)
            # remove uncorrected file to save disk space
            shutil.rmtree(ms + '.extracted', ignore_errors=True)
            if avgfreqstep > 1 or avgtimestep > 1:
                average([ms + '.extracted_ddcor'], args=args, freqstep=[avgfreqstep], timestep=avgtimestep, \
                         dysco=dysco, metadata_compression=metadata_compression)
                shutil.rmtree(ms + '.extracted_ddcor', ignore_errors=True)
                os.rename(ms + '.extracted_ddcor.avg', ms + '.extracted_ddcor')            
     
    # print the imsize for the user
    if userbox is not None and userbox != 'keepall':
        terminal_print('Box size used for remove-outside-center: {} degrees'.format(boxsize))
        # round up the the nearest even integer
        imsize_to_use = int(boxsize * 3600.0 / pixsize)
        if imsize_to_use % 2 != 0:
            imsize_to_use += 1
        if imsize_to_use > imsize:
            # print in orange because this is a warning
            terminal_print('\033[33mWarning: the box size used for remove-outside-center is larger than the original image size, using original image size instead\033[0m')
            imsize_to_use = imsize
        terminal_print('Imsize to use after this extract step: {}'.format(imsize_to_use))
    elif userbox == 'keepall':
        terminal_print('No box used for remove-outside-center, entire field kept as per user request')
        terminal_print('Imsize to use after this extract step: {}'.format(imsize))
    else: # --remove-outside-center-box was not set
        terminal_print('Box size used for remove-outside-center: {} degrees'.format(boxsize))
        # round up the the nearest even integer
        imsize_to_use = int(boxsize * 3600.0 / pixsize)
        if imsize_to_use % 2 != 0:
            imsize_to_use += 1
        if imsize_to_use > imsize:
            # print in orange because this is a warning
            terminal_print('\033[33mWarning: the box size used for remove-outside-center is larger than the original image size, using original image size instead\033[0m')
            imsize_to_use = imsize    
        terminal_print('Imsize to use after this extract step: {}'.format(imsize_to_use))   
    
    # write imsize_to_use to a file so that it can be used in the next steps of the pipeline
    with open('misc/imsize_after_extract.txt', 'w') as f:
        if userbox == 'keepall':
            f.write(str(imsize))
        else:
            f.write(str(imsize_to_use))

    output_suffix = (
        '.extracted_ddcor'
        if len(h5list) != 0 and ddcor and userbox != 'keepall'
        else '.extracted'
    )
    with open('misc/extracted_ms.txt', 'w') as f:
        for ms in mslist:
            f.write(os.path.basename(ms) + output_suffix + '\n')
  
    # remove templatebox.reg if it exists to clean things up
    if os.path.exists('templatebox.reg'):
        os.remove('templatebox.reg')

    # gzip model images back
    gzip_model_images(imagebasename)

    return


@_track_resource_phase("imaging")
def makeimage(mslist, imageout, pixsize, imsize, channelsout, niter=100000, robust=-0.5,
              uvtaper=None, multiscale=False, predict=True, onlypredict=False, fitsmask=None,
              idg=False, uvminim=80, fitspectralpol=3,
              restoringbeam=15, automask=2.5,
              removenegativecc=True, usewgridder=True, paralleldeconvolution=0,
              parallelgridding=1, forced_imcol=None,
              fullpol=False, selfcalcycle=None,
              uvmaxim=None, h5list=[], facetregionfile=None, squarebox=None,
              DDE_predict='WSCLEAN', DDEimaging=False,
              nosmallinversion=False,
              stack=False, disable_primarybeam_predict=False, disable_primarybeam_image=False,
              facet_beam_update_time=120,
              singlefacetpredictspeedup=True, forceimagingwithfacets=True,
              fulljones_h5_facetbeam=False, sharedfacetreads=False):
    """
    Image Measurement Sets using WSClean or DDFacet.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets to image.
    imageout : str
        Output image basename.
    pixsize : float
        Pixel size in arcseconds.
    imsize : int or list of int
        Image dimensions in pixels.
    channelsout : int
        Number of output channels.
    niter : int
        Number of clean iterations.
    robust : float
        Briggs robust weighting parameter.
    uvtaper : float
        Outer uv taper in arcseconds.
    multiscale : bool
        Use multiscale deconvolution.
    predict : bool
        Run predict on model.
    onlypredict : bool
        Only predict without imaging.
    fitsmask : str
        Path to FITS mask.
    idg : bool
        Use Image Domain Gridder.
    uvminim : float
        Minimum uv cut in lambda.
    fitspectralpol : int
        Fit spectral polynomial order.
    restoringbeam : str
        Restoring beam specification.
    automask : float
        Automask threshold.
    removenegativecc : bool
        Remove negative clean components.
    usewgridder : bool
        Use W-gridder.
    paralleldeconvolution : int
        Parallel deconvolution sub-image size.
    parallelgridding : int
        Parallel gridding setting.
    forced_imcol : str
        Force imaging of this data column.
    fullpol : bool
        Image all 4 polarizations.
    selfcalcycle : int
        Current self-calibration cycle number.
    uvmaxim : float
        Maximum uv cut in lambda.
    h5list : list of str
        List of H5Parm solution files.
    facetregionfile : str
        DS9 region file defining facets.
    squarebox : bool
        Use square bounding box.
    DDE_predict : str
        Predict method ('DP3' or 'WSCLEAN').
    DDEimaging : bool
        Perform DDE facet imaging.
    nosmallinversion : bool
        Disable small inversion in WSClean.
    stack : bool
        Image stacked MS.
    disable_primarybeam_predict : bool
        Disable primary beam in predict.
    disable_primarybeam_image : bool
        Disable primary beam in imaging.
    facet_beam_update_time : float
        Beam update interval for facet beam.
    singlefacetpredictspeedup : bool
        Speed up single facet predict.
    forceimagingwithfacets : bool, optional
        Force imaging with facetregionfile even if len(h5list) == 0.
    fulljones_h5_facetbeam : bool, optional
        Apply FullJones solutions in facet beam.
    sharedfacetreads : bool, optional
        Share facet visibility reads.

    Returns
    -------
    list of str or None
        Model-data column names created for prediction, or ``None`` when no
        model columns are produced.
    """
    
    if args['telescope'] in ['MeerKAT', 'GMRT']:
        # static primary beam, this should speed up the imaging
        facet_beam_update_time = 12*3600 # set an 12 hr

    # since the addition of the -model-storage-manager the option -model-column can be used without hitting a DP3 error 
    #Related to this was that WSClean up to now uses the same storage manager (read: file) for multiple model columns. While this is 'legal', it turned out that Dp3's DDECal with those model columns as input didn't work with this (due to complicated multi-threading issues).
    if '-model-storage-manager' in subprocess.check_output(['wsclean'], text=True):
        predict_inmodelcol = True
    else:
        predict_inmodelcol = False

    if '-shared-facet-reads' in subprocess.check_output(['wsclean'], text=True) and \
       '-shared-facet-writes' in subprocess.check_output(['wsclean'], text=True):
        sharedfacetreads = True
    sharedfacetreads = False # for now force it to False
    # it seems to slow the imaging down, instead of speed it up/bugs are present in older wsclean versions
    
    if args['modelstoragemanager'] == 'stokes_i':
        modelstoragemanagerwsclean = 'stokes-i' # because WSclean uses a different name than DP3
    elif args['modelstoragemanager'] == 'sisco_stokes_i':
        modelstoragemanagerwsclean = 'sisco-stokes-i' # because WSclean uses a different name than DP3
    elif args['modelstoragemanager'] == 'sisco_diagonal':
        modelstoragemanagerwsclean = 'sisco-diagonal' # because WSclean uses a different name than DP3
    else:
        modelstoragemanagerwsclean =  args['modelstoragemanager']

    fitspectrallogpol = False  # for testing Perseus
    msliststring = ' '.join(map(str, mslist))
    if idg:
        parallelgridding = 1
    
    # update the fitsmask if user provided DS9cleanmaskregionfile-exclude
    if fitsmask is not None and args['DS9cleanmaskregionfile_exclude'] is not None:
        # make a new fitsmask with the excluded regions masked out
        # input is overwritten
        terminal_print('Updating fitsmask', fitsmask, 'with excluded regions from', args['DS9cleanmaskregionfile_exclude'])
        logging.info('Updating fitsmask {} with excluded regions from {}'.format(fitsmask, args['DS9cleanmaskregionfile_exclude']))
        mask_region(fitsmask, args['DS9cleanmaskregionfile_exclude'], fitsmask)


    #if telescope != 'LOFAR' and not onlypredict and facetregionfile is not None:
    #    nosmallinversion = True
    if args['telescope'] != 'LOFAR':
        nosmallinversion = True  

    #  --- DI predict only without facets ---
    # for example to subtract region of sky for the --remove-outside-center option
    if onlypredict and facetregionfile is None:
        if predict:
            if squarebox is not None:
                for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
                    terminal_print(
                        'Model image:',
                        model,
                        'Output box mask:',
                        'fits_images/box_' + os.path.basename(model),
                    )
                    mask_region(model, squarebox, 'fits_images/box_' + os.path.basename(model))

            cmd = 'wsclean -predict '
            if args['ncpu_max_WSClean'] is not None:
                cmd += '-j ' + str(args['ncpu_max_WSClean']) + ' '
            if channelsout > 1:

                if args['fix_model_frequencies']: # Implementing model manipulation
                    freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                    mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs, args) # Adjust models to fit incoming ms
                    #Overwrite relevant args if needed 
                    if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                        args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                        imageout = args['wscleanskymodel']
                    channelsout = len(mod_freqs)
                    cmd += '-channel-division-frequencies ' + mod_freq_string + ' '
                
                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if not disable_primarybeam_predict:
                    if args['telescope'] == 'LOFAR':
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(args['wgridderaccuracy']) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '
            
            if args['modelstoragemanager'] is not None:
                cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

            if squarebox is None:
                cmd += '-name ' + imageout + ' ' + msliststring
            else:
                cmd += '-name ' + 'fits_images/box_' + os.path.basename(imageout) + ' ' + msliststring
            terminal_print('PREDICT STEP: ', cmd)
            logger.info('PREDICT STEP: ' + cmd)
            _run_predict_command(cmd)

            # remove box_ model files to save space
            if squarebox is not None:
                for model in sorted(glob.glob('fits_images/box_' + os.path.basename(imageout) + '-????-*model*.fits')):
                    Path(model).unlink(missing_ok=True)
        return
    #  --- end predict only ---

    #  --- DDE-NO-FACET-LOOP CORRUPT-predict, so everything ends up in MODEL_DATA and is corrupted
    # for example for --remove-outside-center option with --DDE
    if onlypredict and facetregionfile is not None and predict and squarebox is not None:
        # do the masking
        for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
            if squarebox != 'keepall':
                terminal_print(
                    'Square mask region:',
                    squarebox,
                    'Model image:',
                    model,
                    'Output box mask:',
                    'fits_images/box_' + os.path.basename(model),
                )
                mask_region(model, squarebox, 'fits_images/box_' + os.path.basename(model))

            # predict with wsclean
        cmd = 'wsclean -predict '
        if args['ncpu_max_WSClean'] is not None:
            cmd += '-j ' + str(args['ncpu_max_WSClean']) + ' '
        if channelsout > 1:
            cmd += '-channels-out ' + str(channelsout) + ' '
            if args['gapchanneldivision']:
                cmd += '-gap-channel-division '
        if idg:
            cmd += '-gridder idg -idg-mode cpu '
            if not disable_primarybeam_predict:
                if args['telescope'] == 'LOFAR':
                    cmd += '-grid-with-beam -use-differential-lofar-beam '
                    cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
            # cmd += '-pol iquv '
            cmd += '-pol i '
            cmd += '-padding 1.8 '
        else:
            if usewgridder:
                cmd += '-gridder wgridder '
                cmd += '-wgridder-accuracy ' + str(args['wgridderaccuracy']) + ' '
                if nosmallinversion:
                    cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
            if parallelgridding > 1:
                cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

        cmd += '-facet-regions ' + facetregionfile + ' '
        cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list)) + ' amplitude000,phase000 '
        if sharedfacetreads: cmd += '-shared-facet-reads -shared-facet-writes '

        if not fulljones_h5_facetbeam:
            if not is_scalar_array_for_wsclean(h5list):
                cmd += '-diagonal-visibilities '  # different XX and YY solutions
            else:
                cmd += '-scalar-visibilities '  # scalar solutions

        if args['telescope'] in ['LOFAR', 'MeerKAT', 'GMRT', 'VLA', 'EVLA', 'MWA']:
            if not disable_primarybeam_predict:
                cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                if args['telescope'] == 'LOFAR': cmd += '-use-differential-lofar-beam '

        if args['modelstoragemanager'] is not None:
            cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

        if squarebox != 'keepall':
            cmd += '-name fits_images/box_' + os.path.basename(imageout) + ' ' + msliststring
        else:
            cmd += '-name ' + imageout + ' ' + msliststring    
        
        if DDE_predict == 'WSCLEAN':
            terminal_print('DDE PREDICT STEP: ', cmd)
            logger.info('DDE PREDICT STEP: ' + cmd)
            _run_predict_command(cmd)
        # remove box_ model files to save space
        if squarebox != 'keepall':
            for model in sorted(glob.glob('fits_images/box_' + os.path.basename(imageout) + '-????-*model*.fits')):
                Path(model).unlink(missing_ok=True)
        return
        #  --- NO-FACET-LOOP end DDE CORRUPT-predict only ---

    #  --- DDE-FACET-LOOP predict, so all facets are predicted  ---
    #  --- each facet gets its own model column ---
    if onlypredict and facetregionfile is not None and predict:
        # step 1 open facetregionfile
        modeldatacolumns_list = []
        r = pyregion.open(facetregionfile)
        dirofinput = os.path.dirname(facetregionfile)
        if dirofinput == '':
            dirofinput = '.'
        for facet_id, facet in enumerate(r):
            r[facet_id:facet_id + 1].write(dirofinput + '/facet' + str(facet_id) + '.reg')  # split facet from region file

            # step 2 mask outside of region file
            if not singlefacetpredictspeedup:  # not needed, because WSClean will do the facet cutting
                for model in sorted(glob.glob(imageout + '-????-*model*.fits')):
                    modelout = 'fits_images/facet_' + os.path.basename(model)
                    if DDE_predict == 'WSCLEAN':
                        terminal_print('Model image:', model, 'Output model:', modelout)
                        mask_region_inv(model, dirofinput + '/facet' + str(facet_id) + '.reg', modelout)

            # step 3 predict with wsclean
            cmd = 'wsclean -predict '
            if args['ncpu_max_WSClean'] is not None:
                cmd += '-j ' + str(args['ncpu_max_WSClean']) + ' '
            if predict_inmodelcol:  # directly predict the right column name
                cmd += '-model-column MODEL_DATA_DD' + str(facet_id) + ' '
            if args['modelstoragemanager'] is not None:
                cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '
            # if not usewgridder and not idg:
            #  cmd += '-padding 1.8 '
            if channelsout > 1:
                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if not disable_primarybeam_predict:
                    if args['telescope'] == 'LOFAR':
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(args['wgridderaccuracy']) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

            # NEW CODE FOR SPEEDUP
            if singlefacetpredictspeedup:
                cmd += '-facet-regions ' + dirofinput + '/facet' + str(facet_id) + '.reg' + ' '
                if args['telescope'] in ['LOFAR', 'MeerKAT', 'GMRT', 'VLA', 'EVLA', 'MWA']:
                    if not disable_primarybeam_predict:
                        # check if -model-fpb.fits is there for image000 (in case image000 was made without facets)
                        if selfcalcycle == 0:
                            fix_fpb_images(imageout)    
                        cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                        if args['telescope'] == 'LOFAR': cmd += '-use-differential-lofar-beam '
                        if not fulljones_h5_facetbeam:
                            # cmd += '-diagonal-visibilities ' # different XX and YY solutions
                            cmd += '-scalar-visibilities '  # scalar solutions

                cmd += '-name ' + imageout + ' ' + msliststring
            else:
                cmd += '-name facet_' + imageout + ' ' + msliststring

            if DDE_predict == 'WSCLEAN':
                terminal_print('DDE PREDICT STEP: ', cmd)
                logger.info('DDE PREDICT STEP: ' + cmd)
                _run_predict_command(cmd)

            # step 4 copy over to MODEL_DATA_DDX
            for ms in mslist:
                cmddppp = 'DP3 msin=' + ms + ' msin.datacolumn=MODEL_DATA msout=. steps=[] '
                cmddppp += 'msout.datacolumn=MODEL_DATA_DD' + str(facet_id)
                if DDE_predict == 'WSCLEAN' and not predict_inmodelcol:
                    run(cmddppp)
            modeldatacolumns_list.append('MODEL_DATA_DD' + str(facet_id))

        return modeldatacolumns_list
    #  --- end DDE predict only ---

    # ----- MAIN IMAGING PART ------
    for file_path in Path('.').glob(imageout + '-*.fits'):
        file_path.unlink(missing_ok=True)
    imcol = 'CORRECTED_DATA'
    # check for all ms in mslist
    # situation can be that only some ms have CORRECTED_DATA based on what happens in the beamcor
    # so some ms can have a beam correction and others not
    # for example because of different directions where the beam was applied
    for ms in mslist:
        t = table(ms, readonly=True, ack=False)
        colnames = t.colnames()
        if 'CORRECTED_DATA' not in colnames:  # for first imaging run
            imcol = 'DATA'
        t.close()
    
    # if set explicitly by user override
    if forced_imcol is not None:
        imcol = forced_imcol

    baselineav = str(1.5e3 * 60000. * 2. * np.pi * 1.5 / (24. * 60. * 60 * float(imsize)))

    if args['imager'] == 'WSCLEAN':
        cmd = 'wsclean '
        if args['ncpu_max_WSClean'] is not None:
            cmd += '-j ' + str(args['ncpu_max_WSClean']) + ' '
        cmd += '-no-update-model-required '
        cmd += '-minuv-l ' + str(uvminim) + ' '
        if uvmaxim is not None:
            cmd += '-maxuv-l ' + str(uvmaxim) + ' '
        cmd += '-size ' + str(int(imsize)) + ' ' + str(int(imsize)) + ' -reorder '
        if type(robust) is not str:
            cmd += '-weight briggs ' + str(robust)
        else:
            cmd += '-weight ' + str(robust)
        # cmd += ' -clean-border 1 ' # not needed anymore for WSCleand
        cmd += ' -parallel-reordering 4 '
        # -weighting-rank-filter 3 -fit-beam
        cmd += '-mgain ' + str(args['mgain']) + ' -data-column ' + imcol + ' '
        # if not usewgridder and not idg:
        #  cmd += '-padding 1.4 '
        if channelsout > 1:
            if args['fix_model_frequencies']: # Implementing model manipulation
                freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs, args) # Adjust models to fit incoming ms
                #Overwrite relevant args
                if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                    args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                    imageout = args['wscleanskymodel']
                channelsout = len(mod_freqs)
                cmd += '-channel-division-frequencies ' + mod_freq_string + ' '
            cmd += ' -join-channels -channels-out ' + str(channelsout) + ' '
            if args['gapchanneldivision']:
                cmd += '-gap-channel-division '
        if paralleldeconvolution > 0:
            cmd += '-parallel-deconvolution ' + str(paralleldeconvolution) + ' '
        if parallelgridding > 1:
            cmd += '-parallel-gridding ' + str(parallelgridding) + ' '
        if args['deconvolutionchannels'] > 0 and channelsout > 1:
            cmd += '-deconvolution-channels ' + str(args['deconvolutionchannels']) + ' '
        if automask > 0.5:
            cmd += '-auto-mask ' + str(automask) + ' -auto-threshold 0.5 '  # to avoid automask 0
        if args['localrmswindow'] > 0:
            cmd += '-local-rms-window ' + str(args['localrmswindow']) + ' '

        if args['ddpsfgrid'] is not None:
            cmd += '-dd-psf-grid ' + str(args['ddpsfgrid']) + ' ' + str(args['ddpsfgrid']) + ' '

        if multiscale:
            # cmd += '-multiscale '+' -multiscale-scales 0,4,8,16,32,64 -multiscale-scale-bias 0.6 '
            # cmd += '-multiscale '+' -multiscale-scales 0,6,12,16,24,32,42,64,72,128,180,256,380,512,650 '
            cmd += '-multiscale '
            cmd += '-multiscale-scale-bias ' + str(args['multiscalescalebias']) + ' '
            if args['multiscalemaxscales'] == 0:
                cmd += '-multiscale-max-scales ' + str(int(np.rint(np.log2(float(imsize)) - 3))) + ' '
            else:  # use value set by user
                cmd += '-multiscale-max-scales ' + str(int(args['multiscalemaxscales'])) + ' '
        if fitsmask is not None and fitsmask != 'nofitsmask':
            if os.path.isfile(fitsmask):
                shape = get_image_size(fitsmask)
                if (shape[0] == int(imsize)) and (shape[1] == int(imsize)): # to allow for a restart with different imsize
                    cmd += '-fits-mask ' + fitsmask + ' '
            else:
                terminal_print('fitsmask: ', fitsmask, 'does not exist')
                raise Exception('fitsmask does not exist')
        if uvtaper is not None:
            cmd += '-taper-gaussian ' + uvtaper + ' '
        if args['taperinnertukey'] is not None:
            cmd += '-taper-inner-tukey ' + str(args['taperinnertukey']) + ' '

        if (fitspectralpol > 0) and not (fullpol):
            cmd += '-save-source-list '

        if (fitspectralpol > 0) and (channelsout > 1) and not (fullpol):
            if fitspectrallogpol:
                cmd += '-fit-spectral-log-pol ' + str(fitspectralpol) + ' '
            else:
                cmd += '-fit-spectral-pol ' + str(fitspectralpol) + ' '
        if idg:
            cmd += '-gridder idg -idg-mode cpu '
            if not disable_primarybeam_image:
                if args['telescope'] == 'LOFAR':
                    cmd += '-grid-with-beam -use-differential-lofar-beam '
                    cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
            # cmd += '-pol iquv -link-polarizations i '
            cmd += '-pol i '
            cmd += '-padding 1.4 '
        else:
            if fullpol:
                cmd += '-pol iquv -join-polarizations '
            else:
                cmd += '-pol i '
            if len(h5list) == 0 and not args['groupms_h5facetspeedup']:  # only use baseline-averaging if there are no h5 facet-solutions
                # if not forceimagingwithfacets:
                if facetregionfile is None: # do not allow baseline-averaging with facets
                    # facet imaging works with  baseline-averaging but is very slow/inefficient there is no speedup, rather an order of magnitude slow down
                    cmd += '-baseline-averaging ' + baselineav + ' '
            if usewgridder:
                cmd += '-gridder wgridder '
                cmd += '-wgridder-accuracy ' + str(args['wgridderaccuracy']) + ' '
                if nosmallinversion:
                    cmd += '-no-min-grid-resolution '  # '-no-small-inversion '

        if len(h5list) > 0:
            cmd += '-facet-regions ' + facetregionfile + ' '
            if sharedfacetreads: cmd += '-shared-facet-reads -shared-facet-writes '
            if args['groupms_h5facetspeedup'] and len(mslist) > 1:
                mslist_concat, h5list_concat = concat_ms_wsclean_facetimaging(mslist, h5list=h5list, concatms=selfcalcycle is not None and ((selfcalcycle+1 in args['aoflagger_correcteddata_selfcalcycle_list']) or (selfcalcycle+1 in args['aoflagger_residualdata_selfcalcycle_list'])))
                cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list_concat)) + ' '
                cmd += ' amplitude000,phase000 '
            else:
                cmd += '-apply-facet-solutions ' + ','.join(map(str, h5list)) + ' amplitude000,phase000 '

            if not fulljones_h5_facetbeam and not fullpol:  # full-pol needs all correlations
                if not is_scalar_array_for_wsclean(h5list):
                    cmd += '-diagonal-visibilities '  # different XX and YY solutions
                else:
                    cmd += '-scalar-visibilities '  # scalar solutions

            if args['telescope'] in ['LOFAR', 'MeerKAT', 'GMRT', 'VLA', 'EVLA', 'MWA']:
                if not disable_primarybeam_image:
                    cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                    if args['telescope'] == 'LOFAR': cmd += '-use-differential-lofar-beam '
        elif forceimagingwithfacets and facetregionfile is not None:  # so h5list is zero, but we still want facet imaging
            if args['groupms_h5facetspeedup'] and len(mslist) > 1:
                mslist_concat, h5list_concat_tmp = concat_ms_wsclean_facetimaging(mslist, concatms=selfcalcycle is not None and ((selfcalcycle+1 in args['aoflagger_correcteddata_selfcalcycle_list']) or (selfcalcycle+1 in args['aoflagger_residualdata_selfcalcycle_list'])))
            cmd += '-facet-regions ' + facetregionfile + ' '
            if sharedfacetreads: cmd += '-shared-facet-reads -shared-facet-writes '
            if args['telescope'] in ['LOFAR', 'MeerKAT', 'GMRT', 'VLA', 'EVLA', 'MWA'] and not disable_primarybeam_image:
                cmd += '-apply-facet-beam -facet-beam-update ' + str(facet_beam_update_time) + ' '
                if args['telescope'] == 'LOFAR': cmd += '-use-differential-lofar-beam '
                if not fulljones_h5_facetbeam and not fullpol:  # full-pol needs all correlations
                    # cmd += '-diagonal-visibilities ' # different XX and YY solutions
                    cmd += '-scalar-visibilities '  # scalar solutions
        else:
            if args['telescope'] == 'LOFAR' and not check_phaseup_station(mslist[0]) and not idg:
                if not disable_primarybeam_image:
                    cmd += '-apply-primary-beam -use-differential-lofar-beam '
                    cmd += '-facet-beam-update ' + str(facet_beam_update_time) + ' '
            if args['telescope'] in ['MeerKAT', 'GMRT', 'VLA', 'EVLA', 'MWA'] and not idg and not disable_primarybeam_image:
                cmd += '-apply-primary-beam '


        cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec '
        if args['groupms_h5facetspeedup'] and len(mslist) > 1 and facetregionfile is not None:
            msliststring_concat = ' '.join(map(str, mslist_concat))
            terminal_print('WSCLEAN IMAGING: ', cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)
            logger.info('WSCLEAN IMAGING: ' + cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)
            run(cmd + ' -nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring_concat)
        else:
            terminal_print('WSCLEAN IMAGING: ', cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)
            logger.info('WSCLEAN IMAGING: ' + cmd + '-nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)
            #if imageout != 'imageDD_auto_003':
            run(cmd + ' -nmiter ' + str(args['nmiter']) + ' -niter ' + str(niter) + ' ' + msliststring)

        clean_up_images(imageout)

        # do manual  pbcor for MeerKAT/GMRT images if -applybeam or -apply-facet-beam was not used
        if args['telescope'] in ['MeerKAT', 'GMRT', 'VLA', 'EVLA']:
            if '-apply-facet-beam' not in cmd and '-apply-primary-beam' not in cmd:
                terminal_print('Doing manual primary beam correction for MeerKAT/GMRT image')
                if os.path.isfile(imageout + '-MFS-image-pb.fits'):
                    outfile = (imageout + '-MFS-image-pb.fits').replace('-MFS-image-pb.fits', '-MFS-image-manualpb.fits')
                    if args['telescope'] == 'MeerKAT':
                        MeerKAT_pbcor(imageout + '-MFS-image-pb.fits', outfile, ms=mslist[0])
                    elif args['telescope'] == 'GMRT':
                        uGMRT_pbcor(imageout + '-MFS-image-pb.fits', outfile, ms=mslist[0])
                    elif args['telescope'] in ['VLA', 'EVLA']:
                        VLA_pbcor(imageout + '-MFS-image-pb.fits', outfile, ms=mslist[0])    
                else: #so this must be a run without facets
                    if fullpol:
                        outfile = (imageout + '-MFS-I-image.fits').replace('-MFS-I-image.fits', '-MFS-I-image-manualpb.fits')
                        if args['telescope'] == 'MeerKAT':
                            MeerKAT_pbcor(imageout + '-MFS-I-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] == 'GMRT':
                            uGMRT_pbcor(imageout + '-MFS-I-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] in ['VLA', 'EVLA']:
                            VLA_pbcor(imageout + '-MFS-I-image.fits', outfile, ms=mslist[0])
                        outfile = (imageout + '-MFS-Q-image.fits').replace('-MFS-Q-image.fits', '-MFS-Q-image-manualpb.fits')
                        if args['telescope'] == 'MeerKAT':
                            MeerKAT_pbcor(imageout + '-MFS-Q-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] == 'GMRT':
                            uGMRT_pbcor(imageout + '-MFS-Q-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] in ['VLA', 'EVLA']:
                            VLA_pbcor(imageout + '-MFS-Q-image.fits', outfile, ms=mslist[0])    
                        outfile = (imageout + '-MFS-U-image.fits').replace('-MFS-U-image.fits', '-MFS-U-image-manualpb.fits')
                        if args['telescope'] == 'MeerKAT':
                            MeerKAT_pbcor(imageout + '-MFS-U-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] == 'GMRT':
                            uGMRT_pbcor(imageout + '-MFS-U-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] in ['VLA', 'EVLA']:
                            VLA_pbcor(imageout + '-MFS-U-image.fits', outfile, ms=mslist[0])        
                        outfile = (imageout + '-MFS-V-image.fits').replace('-MFS-V-image.fits', '-MFS-V-image-manualpb.fits')
                        if args['telescope'] == 'MeerKAT':
                            MeerKAT_pbcor(imageout + '-MFS-V-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] == 'GMRT':
                            uGMRT_pbcor(imageout + '-MFS-V-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] in ['VLA', 'EVLA']:
                            VLA_pbcor(imageout + '-MFS-V-image.fits', outfile, ms=mslist[0])  

                    else:
                        outfile = (imageout + '-MFS-image.fits').replace('-MFS-image.fits', '-MFS-image-manualpb.fits')
                        if args['telescope'] == 'MeerKAT':
                            MeerKAT_pbcor(imageout + '-MFS-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] == 'GMRT':
                            uGMRT_pbcor(imageout + '-MFS-image.fits', outfile, ms=mslist[0])
                        elif args['telescope'] in ['VLA', 'EVLA']:
                            VLA_pbcor(imageout + '-MFS-image.fits', outfile, ms=mslist[0])    

        # write info about how the primary beam correction was done to the FITS header and processing history
        write_processing_history(' '.join(map(str, sys.argv)), facetselfcal_version, imageout)
        write_primarybeam_info(cmd, imageout, telescope=args['telescope'])
        
        # REMOVE nagetive model components, these are artifacts (only for Stokes I)
        if removenegativecc:
            if idg:
                removenegativefrommodel(sorted(glob.glob(imageout + '-????-*model*.fits')))  # only Stokes I
            else:
                removenegativefrommodel(sorted(glob.glob(imageout + '-????-*model*.fits')))

        # Remove NaNs from array (can happen if certain channels from channels-out are flagged, or from -apply-facet-beam far into the sidelobes)
        if idg:
            removeneNaNfrommodel(glob.glob(imageout + '-????-*model*.fits'))  # only Stokes I
        else:
            removeneNaNfrommodel(glob.glob(imageout + '-????-*model*.fits'))

        # Check is anything was cleaned. If not, stop the selfcal to avoid obscure errors later
        if channelsout > 1:
            model_allzero = checkforzerocleancomponents(glob.glob(imageout + '-????-*model*.fits'))  # only Stokes I
        else:
            model_allzero = checkforzerocleancomponents(glob.glob(imageout + '-*model*.fits'))
        if model_allzero:
            logger.error("All channel maps models were zero: Stopping the selfcal")
            terminal_print("All channel maps models were zero: Stopping the selfcal")
            sys.exit(1)

        if predict and len(h5list) == 0 and not DDEimaging:
            # We check for DDEimaging to avoid a predict for image000 in a --DDE run
            # because at that moment there is no h5list yet, avoiding an unnecessary DI-type predict
            cmd = 'wsclean -predict '  # -size '
            if args['ncpu_max_WSClean'] is not None:
                cmd += '-j ' + str(args['ncpu_max_WSClean']) + ' '
            if channelsout > 1:
                if args['fix_model_frequencies']: # Implementing model manipulation
                    freq_string, freqs = frequencies_from_models(args['wscleanskymodel']) # Get frequency info from models
                    mod_freq_string, mod_freqs = modify_freqs_from_ms(mslist, freqs, args) # Adjust models to fit incoming ms
                    #Overwrite relevant args
                    if len(glob.glob("tmp_" + args['wscleanskymodel'] + "*")) > 0:
                        args['wscleanskymodel'] = "tmp_" + args['wscleanskymodel']
                        imageout = args['wscleanskymodel']
                    channelsout = len(mod_freqs)
                    cmd += '-channel-division-frequencies ' + mod_freq_string + ' '

                cmd += '-channels-out ' + str(channelsout) + ' '
                if args['gapchanneldivision']:
                    cmd += '-gap-channel-division '
            if idg:
                cmd += '-gridder idg -idg-mode cpu '
                if args['telescope'] == 'LOFAR':
                    if not disable_primarybeam_predict:
                        cmd += '-grid-with-beam -use-differential-lofar-beam '
                        cmd += '-beam-aterm-update ' + str(facet_beam_update_time) + ' '
                # cmd += '-pol iquv '
                cmd += '-pol i '
                cmd += '-padding 1.8 '
            else:
                if usewgridder:
                    cmd += '-gridder wgridder '
                    cmd += '-wgridder-accuracy ' + str(args['wgridderaccuracy']) + ' '
                    if nosmallinversion:
                        cmd += '-no-min-grid-resolution '  # '-no-small-inversion '
                if parallelgridding > 1:
                    cmd += '-parallel-gridding ' + str(parallelgridding) + ' '

                # Needs multi-col predict
                # if h5 is not None:
                #     cmd += '-facet-regions ' + facetregionfile + ' '
                #     cmd += '-apply-facet-solutions ' + h5 + ' amplitude000,phase000 '
                #     if args['telescope'] == 'LOFAR':
                #         cmd += '-apply-facet-beam -facet-beam-update 600 -use-differential-lofar-beam '
                #         cmd += '-diagonal-solutions '
                if args['modelstoragemanager'] is not None:
                    cmd += '-model-storage-manager ' + modelstoragemanagerwsclean + ' '

                cmd += '-name ' + imageout + ' -scale ' + str(pixsize) + 'arcsec ' + msliststring
                terminal_print('PREDICT STEP: ', cmd)
                logger.info('PREDICT STEP: ' + cmd)
                _run_predict_command(cmd)

        if args['imager'] == 'DDFACET':
            makemslist(mslist)
            # restoringbeam = '15'
            cmd = 'DDF.py --Data-MS=mslist.txt --Deconv-PeakFactor=0.001 --Data-ColName=' + imcol + ' ' + \
                  '--Parallel-NCPU=32 --Output-Mode=Clean --Deconv-CycleFactor=0 ' + \
                  '--Deconv-MaxMinorIter=' + str(niter) + ' --Deconv-MaxMajorIter=5 ' + \
                  '--Deconv-Mode=SSD --Weight-Robust=' + str(robust) + ' --Image-NPix=' + str(int(imsize)) + ' ' + \
                  '--CF-wmax=50000 --CF-Nw=100 --Beam-Model=None --Beam-LOFARBeamMode=A --Beam-NBand=1 ' + \
                  '--Output-Also=onNeds --Image-Cell=' + str(pixsize) + ' --Facets-NFacets=1 --Freq-NDegridBand=1 ' + \
                  '--Deconv-RMSFactor=3.0 --Deconv-FluxThreshold=0.0 --Data-Sort=1 --Cache-Dir=. --Freq-NBand=2 ' + \
                  '--GAClean-MinSizeInit=10 --Facets-DiamMax=1.5 --Facets-DiamMin=0.1 ' + \
                  '--Cache-Dirty=auto --Weight-ColName=WEIGHT_SPECTRUM --Output-Name=' + imageout + ' ' + \
                  '--Comp-BDAMode=1 --DDESolutions-DDModeGrid=AP --DDESolutions-DDModeDeGrid=AP --Cache-Reset=1 ' + \
                  '--RIME-ForwardMode=BDA-degrid --Predict-ColName=MODEL_DATA --Selection-UVRange=[0.1,2000.] ' + \
                  '--Output-RestoringBeam=' + str(restoringbeam) + ' --Mask-SigTh=5.0 '
            if fitsmask is not None and fitsmask != 'nofitsmask':
                cmd += '--Mask-External=' + fitsmask + ' --Mask-Auto=0 '
            else:
                cmd += '--Mask-Auto=1 '

            terminal_print('Command:', cmd)
            run(cmd)


# print getdeclinationms('1E216.dysco.sub.shift.avg.weights.set0.ms')


# to remove H5/h5 and other files out of a wildcard selection if needed


def basicsetup(mslist, spectral_window_metadata=None, create_directories=True):
    """
    Perform basic setup and metadata checks for input MSs.

    Parameters
    ----------
    mslist : list
        Measurement Set paths.

    Returns
    -------
    None
        Setup state is recorded in the workflow context.
    """
    
    if create_directories:
        os.makedirs('losoto_parsets', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('facet_regions', exist_ok=True)
        os.makedirs('directions', exist_ok=True)
        os.makedirs('h5_solutions', exist_ok=True)
        os.makedirs('fits_images', exist_ok=True)
        os.makedirs('clean_masks', exist_ok=True)
        os.makedirs('errormaps_dd', exist_ok=True)
        os.makedirs('misc', exist_ok=True)

    if args['compute_weightspectrum']:
        args['createresidualdatacolumn'] = True  # force creation of residual data column if weight spectrum is computed

    longbaseline = checklongbaseline(mslist[0])
    if args['removeinternational']:
        terminal_print('Forcing longbaseline to False as --removeinternational has been specified')
        longbaseline = False
        # Determine HBA or LBA
    if spectral_window_metadata is None:
        t = table(mslist[0] + '::SPECTRAL_WINDOW', ack=False)
        freqs = t.getcol('CHAN_FREQ')[0]
        chan_width = np.median(t.getcol('CHAN_WIDTH')[0])
        t.close()
    else:
        freqs, chan_widths = spectral_window_metadata
        chan_width = np.median(chan_widths)
    freq = np.median(freqs)

    if args['multiscale'] is not None:
        # do not allow update_multiscale to be set automatically further below if multiscale is specified
        args['update_multiscale'] = False
   
    if args['telescope'] == 'LOFAR':
        if freq < 100e6:
            LBA = True
            HBAorLBA = 'HBA'
        else:
            LBA = False
            HBAorLBA = 'LBA'
    else:
        HBAorLBA = 'other'
        LBA = False
        HBA = False

    if args['DDE']: args['forwidefield'] = True  # force widefield imaging if DDE

    if args['telescope'] != 'LOFAR':
        args['noarchive'] = True  # force noarchive for these telescopes

    # set some default values if not provided
    if args['uvmin'] is None:
        if longbaseline:
            args['uvmin'] = 20000.
        else:
            if LBA:
                if freq >= 40e6:
                    args['uvmin'] = 80.
                if freq < 40e6:
                    args['uvmin'] = 60.
            elif args['telescope'] == 'EVLA' or args['telescope'] == 'VLA':
                args['uvmin'] = 1. # assume we can use all baselines
            else:
                args['uvmin'] = 350.

    if type(args['uvminim']) is not list:
        if args['uvminim'] is None:
            if args['telescope'] == 'LOFAR':
                if args['DDE']:
                    args['uvminim'] = 10.  # for DDE we want to go to smaller scales
                else:
                    args['uvminim'] = 80.  # the default since a long time so keep it for now, but could be reduced for HBA in the future 
            else:
                args['uvminim'] = 10.  # MeerKAT/uGMRT/EVLA/ASKAP for example

    if args['pixelscale'] is None and args['telescope'] == 'LOFAR':
        if LBA:
            if longbaseline:
                args['pixelscale'] = 0.08
            else:
                args['pixelscale'] = np.rint(3.0 * 54e6 / freq)
        else:
            if longbaseline:
                args['pixelscale'] = 0.04
            else:
                args['pixelscale'] = 1.5
    elif args['pixelscale'] is None and args['telescope'] == 'MeerKAT':
        if freq < 1e9:  # UHF-band
            args['pixelscale'] = 1.8
        elif freq < 2e9:  # L-band
            args['pixelscale'] = 1.
        elif freq < 4e9:  # S-band
            args['pixelscale'] = 0.5
    elif args['pixelscale'] is None and args['telescope'] == 'ASKAP':
        if freq < 1e9:  # UHF-band
            args['pixelscale'] = 2.0
        elif freq < 1.4e9:  # L-band-low
            args['pixelscale'] = 1.5
        elif freq < 2.0e9:  # L-band-high
            args['pixelscale'] = 1.0
    elif args['pixelscale'] is None and args['telescope'] == 'GMRT':
        if freq < 250e6:  # band2
            args['pixelscale'] = 3.0
        elif freq >= 250e6 and freq < 500e6:  # band3
            args['pixelscale'] = 1.25
        elif freq >= 500e6 and freq < 1e9:  # band4
            args['pixelscale']  = 0.75
        elif freq >= 1e9:  # band5
            args['pixelscale'] = 0.35
    elif args['pixelscale'] is None and (args['telescope'] == 'EVLA' or args['telescope'] == 'VLA'):
        # get the configuration
        if get_vla_maxconfiguration(mslist) == 'A':
            res_factor = 1.0
        if get_vla_maxconfiguration(mslist) == 'B':
            res_factor = 3.0
        if get_vla_maxconfiguration(mslist) == 'C':
            res_factor = 9.0
        if get_vla_maxconfiguration(mslist) == 'D':
            res_factor = 27.0
        if freq > 50e6 and freq < 100e6:  # 4-band, 74 MHz centre
            args['pixelscale'] = 6.0*res_factor
        elif freq > 200e6 and freq < 500e6:  # P-band, 350 MHz centre
            args['pixelscale'] = 1.3*res_factor
        elif freq > 1e9 and freq < 2e9:  # L-band, 1.5 GHz centre
            args['pixelscale'] = 0.3*res_factor
        elif freq >= 2e9 and freq < 4e9:  # S-band, 3.0 GHz centre
            args['pixelscale'] = 0.15*res_factor
        elif freq >= 4e9 and freq < 8e9:  # C-band, 6.0 GHz centre
            args['pixelscale'] = 0.075*res_factor
        elif freq >= 8e9 and freq < 12e9:  # X-band, 10 GHz centre
            args['pixelscale'] = 0.045*res_factor
        elif freq >= 12e9 and freq < 18e9:  # Ku-band, 15 GHz centre
            args['pixelscale'] = 0.03*res_factor
        elif freq >= 18e9 and freq < 26e9:  # K-band, 22 GHz centre
            args['pixelscale'] = 0.020*res_factor
        elif freq >= 26e9 and freq < 40e9:  # Ka-band, 33 GHz centre
            args['pixelscale'] = 0.014*res_factor
        elif freq >= 40e9 and freq < 50e9:  # Q-band, 45 GHz centre
            args['pixelscale'] = 0.01*res_factor

    elif args['pixelscale'] is None:
        terminal_print('pixelscale not set and cannot be determined for telescope', args['telescope'])
        raise Exception('pixelscale not set and cannot be determined for telescope')    

    if args['robust'] is None:
        if args['telescope'] == 'LOFAR':
            args['robust'] = -0.5
        elif args['telescope'] == 'MeerKAT':
            if freq > 1.7e9:  # S-band
                args['robust'] = 0.0
            else:
                args['robust'] = -0.5
        elif args['telescope'] == 'ASKAP':
            args['robust'] = -0.5
        elif args['telescope'] == 'GMRT':
            args['robust'] = -0.0
        elif args['telescope'] == 'EVLA' or args['telescope'] == 'VLA':    
            args['robust'] = 0.0
        elif args['telescope'] == 'MWA':
            args['robust'] = -0.5 
        else:
            args['robust'] = 0.0       

    # if imsize is a filename (so a string) then read the imsize for there
    if isinstance(args['imsize'], str):
        with open(args['imsize'], 'r') as f:
            args['imsize'] = int(f.read().strip())

    if (args['delaycal'] or args['auto']) and longbaseline and not LBA:
        if args['imsize'] is None:
            args['imsize'] = 2048

    if args['boxfile'] is not None:
        if args['DDE']:
            args['imsize'] = getimsize(args['boxfile'], args['pixelscale'], increasefactor=1.025)
        else:
            args['imsize'] = getimsize(args['boxfile'], args['pixelscale'])
    
    if args['auto'] and args['telescope'] == 'MeerKAT' and not args['DDE']:
        if args['imsize'] is None:
            if args['bandpass']: 
                args['imsize'] = 1024 # default for MeerKAT when doing bandpass
            else:
                args['imsize'] = 12000 # default for MeerKAT in DI auto mode 

    if args['paralleldeconvolution'] == 0: # means determine automatically
        if args['imsize'] > 1600 and args['telescope'] == 'MeerKAT':
            args['paralleldeconvolution'] = 1200
        elif args['imsize'] > 1600 and args['telescope'] == 'GMRT':
            args['paralleldeconvolution'] = 1200
        elif args['imsize'] > 1600: 
            args['paralleldeconvolution'] =  np.min([1800, int(args['imsize'] / 2)])

    if args['niter'] is None:
        args['niter'] = niter_from_imsize(args['imsize'], args['paralleldeconvolution'])

    if args['auto'] and args['telescope'] == 'LOFAR' and not longbaseline:
        if args['update_uvmin'] is None: # so not set by user, so we can set it to True in auto
            args['update_uvmin'] = True
        args['usemodeldataforsolints'] = True
        args['forwidefield'] = True
        if args['autofrequencyaverage'] is None: # so not set by user, so we can set it to True in auto
            args['autofrequencyaverage'] = True
        if args['start'] == 0 and args['stop'] is None:
            args['stop'] = 10
        if LBA:
            args['BLsmooth_list'] = [True] * len(args['soltype_list'])
        else:
            if args['update_multiscale'] is None: # so not set by user, so we can set it to True in auto
                args['update_multiscale'] = True  # HBA only
        if args['DDE']:
            args['usemodeldataforsolints'] = False
            if args['facetdirections'] is None: # allow auto mode with user-provided facet directions
                args['auto_directions'] = True  
            if args['mask_extended'] is None: # so not set by user, so we can set it to True in auto
                args['mask_extended'] = True 
            if args['update_multiscale'] is None: # so not set by user, so we can set it to True in auto
                args['update_multiscale'] = False  # always use multiscale for DDE if not set by user
            if args['multiscale'] is None: # so not set by user, so we can set it to True in auto
                args['multiscale'] = True
            args['multiscale_start'] = 0
            args['soltypecycles_list'] = [0, 2, 4, 999]
            args['soltype_list'] = ['scalarphase', 'scalarcomplexgain', 'scalarphase', 'scalarcomplexgain'] 
            args['solint_list']  = ["'32sec'","'20min'","'192sec'","'60min'"]
            args['smoothnessconstraint_list'] = [25,15,25,15]
         
            args['antenna_averaging_factors_list'] = ['superterp:8,corebutsuperterp:4,closeremote:2,mediumremote:1,distantremote:1',None,None,None]
            args['antenna_smoothness_factors_list'] = ['core:1,remote:0.5',None,None,None]
            args['antennaconstraint_list'] = [None,None,'core',None]
            args['forwidefield'] = True

    # MeerkAT auto settings
    if args['auto'] and args['telescope'] == 'MeerKAT':
        if isinstance(args['channelsout'], str) and args['channelsout'] == 'auto':
            args['channelsout'] = set_channelsout(mslist, args)
        if isinstance(args['fitspectralpol'], str) and args['fitspectralpol'] == 'auto':
            args['fitspectralpol'] = set_fitspectralpol(args['channelsout'])
        
        args['update_uvmin'] = False
        args['usemodeldataforsolints'] = False
        args['forwidefield'] = True
        if args['auto_flag_antennas'] is None: # so not set by user, so we can set it to True in auto
            args['auto_flag_antennas'] = True # will be invoked for soltype scalarcomplexgain, complexgain, scalaramplitude, amplitudeonly
        if not args['bandpass']: 
            if args['autofrequencyaverage'] is None: # so not set by user, so we can set it to True in auto
                args['autofrequencyaverage'] = True
        if not args['bandpass']: 
            if args['multiscale'] is None: # so not set by user, so we can set it to True in auto
                args['multiscale'] = True
        args['multiscale_start'] = 0
        args['noarchive'] = True
        if args['uvminim'] is None: # so not set by user, so we can set it to 10 in auto
            args['uvminim'] = 10.
        if not args['bandpass']: 
            if args['mask_extended'] is None: # so not set by user, so we can set it to True in auto
                args['mask_extended'] = True
        
        # set the start and end frequencies for the self-calibration based on the central frequency of the observation, so we can determine the channels to use for self-calibration if not set by user
        # channels outside these ranges will be removed from the MS
        if not args['DDE']:
            if (freq >= 1.0e9) and (freq < 1.6e9):  # L-band
                startfreq = 906e6
                endfreq = 1655e6
            elif freq < 1.0e9:  # UHF-band 
                startfreq = 595e6
                endfreq = 1049e6
            # for S0 to S4 bands
            elif freq < 2.3e9:  # S0 
                startfreq = 1.90e9 # might need some tuning
                endfreq   = 2.50e9 # might need some tuning
            elif freq < 2.52e9:  # S1
                startfreq = 2.046e9 # tuning done
                endfreq   = 2.773e9 # tuning done
            elif freq < 2.73e9:  # S2
                startfreq = 2.25e9 # might need some tuning
                endfreq   = 2.95e9 # might need some tuning
            elif freq < 2.95e9:  # S3
                startfreq = 2.45e9 # might need some tuning
                endfreq   = 3.15e9 # might need some tuning
            else:  # S4
                startfreq = 2.65e9 # might need some tuning
                endfreq   = 3.35e9 # might need some tuning
            args['msinstartchan'], args['msinnchan'] = get_startchan_nchan(freqs, startfreq, endfreq)
        
        
        if not args['DDE'] and not args['bandpass']:
            args['flag_ampresetvalfactor'] = True       
            if args['aoflagger'] is None: # so no set by user, so we can set it to True in auto
                args['aoflagger'] = True
            if args['aoflagger_strategy'] is None: # so not set by user, so we can set it to default strategy in auto
                args['aoflagger_strategy'] = 'defaultMeerKAT_StokesQUV.lua' #'default_StokesQUV.lua'
            if args['preapplybandpassH5_list'][0] is None:
                args['soltype_list'] = ['scalarphase', 'scalarcomplexgain', 'scalarcomplexgain', 'scalarcomplexgain']
            else: 
                args['soltype_list'] = ['phaseonly', 'complexgain', 'scalarcomplexgain', 'complexgain'] # becuause XX and YY may not be perfectly calibrated
            args['soltypecycles_list'] = [0, 999, 999, 1] # will be updated automatically in case highDR is detected
            args['solint_list'] = ['1min', '10min', '2min', '30min']
            args['resetsols_list'] = [None, None, None, 'all'] 
            
            # last (fourth) entry just of soltype_list is for auto_flag_antennas 
            # all solutions will be reset, so nothing happens apart from possible flagging of bad antennas
            # if highDR is detected then args['soltypecycles_list'] = [0, 2, 3, 999]

            if args['start'] == 0 and args['stop'] is None:
                if args['preapplybandpassH5_list'][0] is None: # SDP-fully-calibrated data
                    args['stop'] = 3 # this means we can do restarting runs as well
                else:   
                    args['stop'] = 4 # one extra cycle because we are using bandpass solutions without full SDP-calibrated products
            
            if freq < 1.0e9:  # UHF-band
                args['smoothnessconstraint_list'] = [50., 5.0, 50., 50.]
            if (freq >= 1.0e9) and (freq < 1.7e9):  # L-band
                args['smoothnessconstraint_list'] = [100., 5.0, 75., 75.] 
            if (freq >= 1.7e9) and (freq < 4.0e9):  # S-band
                args['smoothnessconstraint_list'] = [150., 7.5, 100., 100.] 
            
 
            # tmp
            #args['flagsolutionsbeforeresetsolsall'] = True
            #args['nchan_list'] = [1, 1, 1, 4]
            #args['solint_list'] = ['1min', '10min', '2min', '4min']
            #args['smoothnessconstraint_list'] = [100., 5.0, 75., 0.]
            #args['clipsolutions'] = True
            #args['clipsolhigh'] = 1.2
            #args['clipsollow'] = 1/1.2
            #args['flagslowamprms'] = 10.0
            #args['doflagging'] = True
            #args['forwidefield'] = False
            #args['doflagslowphases'] = False
            
        else: # so this is a DDE run  
            #if args['aoflagger_residualdata'] is None:
            #    args['aoflagger_residualdata'] = True
            #    args['aoflagger_strategy_residualdata'] = 'default_StokesI.lua'
            #    args['aoflagger_residualdata_selfcalcycle_list'] = [2]

            args['soltype_list'] = ['scalarphase','scalarcomplexgain']
            if args['start'] == 0 and args['stop'] is None:
                args['stop'] = 5 # this means we can do restarting runs as well
            args['soltypecycles_list'] = [0,0]
            args['solint_list'] = ['1min','30min']
            args['nchan_list'] = [1,1]
            args['smoothnessconstraint_list'] = [100.,100.]
            if args['facetdirections'] is None: # allow auto mode with user-provided facet directions
                args['auto_directions'] = True
            args['normamps_list'] = [None,'normamps']
        if args['bandpass']:
            args['keepusingstartingskymodel'] = True
            args['solint_list'] = ['8sec','2min','2min']
            args['soltype_list'] = ['phaseonly','complexgain']
            args['smoothnessconstraint_list'] = [50.,0.0]
            args['soltypecycles_list'] = [0,0]
            args['imsize'] = 1024
            args['niter'] = 10000
            if args['aoflagger'] is None: # so no set by user, so we can set it to True in auto
                args['aoflagger'] = True
            if args['aoflagger_strategy'] is None: # so not set by user, so we can set it to default strategy in auto
                args['aoflagger_strategy'] = 'defaultMeerKAT_StokesQUV.lua'
            args['nchan_list'] = [1,1]

    if args['auto'] and longbaseline and not args['delaycal']:
        args['update_uvmin'] = False
        args['usemodeldataforsolints'] = True
        args['forwidefield'] = True
        if args['autofrequencyaverage'] is None: # so not set by user, so we can set it to True in auto
            args['autofrequencyaverage'] = True
        if args['update_multiscale'] is None: # so not set by user, so we can set it to True in auto
            args['update_multiscale'] = True
        if args['start'] == 0 and args['stop'] is None:
            args['stop'] = 10
        args['soltypecycles_list'] = [0, 3]
        args['soltype_list'] = [args['targetcalILT'], 'scalarcomplexgain']
        if args['targetcalILT'] == 'tec' or args['targetcalILT'] == 'tecandphase':
            args['smoothnessconstraint_list'] = [0.0, 5.0]
        else:
            args['smoothnessconstraint_list'] = [10.0, 5.0]
            args['smoothnessreffrequency_list'] = [120.0, 0.0]
            args['smoothnessspectralexponent_list'] = [-1.0, -1.0]
            args['smoothnessrefdistance_list'] = [0.0, 0.0]
        args['uvmin'] = 20000

        if LBA:
            args['BLsmooth_list'] = [True] * len(args['soltype_list'])

    if args['delaycal'] and LBA:
        terminal_print('Option automated delaycal can only be used for HBA')
        raise Exception('Option automated delaycal can only be used for HBA')
    if args['delaycal'] and not longbaseline:
        terminal_print('Option automated delaycal can only be used for longbaseline data')
        raise Exception('Option automated delaycal can only be used for longbaseline data')

    if args['delaycal'] and longbaseline and not LBA:
        args['update_uvmin'] = False
        # args['usemodeldataforsolints'] = True # NEEDS SPECIAL SETTINGS to be implemented
        args['solint_list'] = ['5min','32sec','1hr']
        args['forwidefield'] = True
        if args['autofrequencyaverage'] is None: # so not set by user, so we can set it to True here
            args['autofrequencyaverage'] = True
        args['update_multiscale'] = True

        args['soltypecycles_list'] = [0, 0, 3]
        args['soltype_list'] = ['scalarphasediff', 'scalarphase', 'scalarcomplexgain']
        args['smoothnessconstraint_list'] = [8.0, 2.0, 15.0]
        args['smoothnessreffrequency_list'] = [120., 144., 0.0]
        args['smoothnessspectralexponent_list'] = [-2.0, -1.0, -1.0]
        args['smoothnessrefdistance_list'] = [0.0, 0.0, 0.0]
        args['antennaconstraint_list'] = ['alldutch', None, None]
        args['nchan_list'] = [1, 1, 1]
        args['uvmin'] = 40000
        if args['start'] == 0 and args['stop'] is None:
            args['stop'] = 8
        args['maskthreshold'] = [5]
        args['docircular'] = True

    # reset tecandphase -> tec for LBA
    if LBA and args['usemodeldataforsolints']:
        args['soltype_list'][1] = 'tec'
        # args['soltype_list'][0] = 'tec'

        if freq < 30e6:
            # args['soltype_list'] = ['tecandphase','tec']    # no scalarcomplexgain in the list, do not use "tec" that gives rings around sources for some reason
            args['soltype_list'] = ['tecandphase', 'tecandphase']  # no scalarcomplexgain in the list

    if args['forwidefield']:
        args['doflagging'] = False
        args['clipsolutions'] = False

    # set stop in case it was not set above
    if args['start'] == 0 and args['stop'] is None:
        args['stop'] = 10
 
    args['imagename'] = 'fits_images/' + os.path.basename(args['imagename']) + '_'
    if args['fitsmask'] is not None:
        fitsmask = args['fitsmask']
    else:
        if args['fitsmask_start'] is not None and args['start'] == 0:
            fitsmask = args['fitsmask_start']
        else:    
            fitsmask = None

    if args['boxfile'] is not None:
        outtarname = (args['boxfile'].split('/')[-1]).split('.reg')[0] + '.tar.gz'
    else:
        outtarname = 'calibrateddata' + '.tar.gz'

    maskthreshold_selfcalcycle = makemaskthresholdlist(args['maskthreshold'], args['stop'] + 1) # add extra in case of an extract step 
    automaskthreshold_selfcalcycle = makemaskthresholdlist(args['automask_threshold'], args['stop'] + 1) # add one extra in case of an extract step


    # ensure automaskthreshold_selfcalcycle is always smaller than maskthreshold_selfcalcycle
    for selfcalcycle_id, mval in enumerate(maskthreshold_selfcalcycle):
        if automaskthreshold_selfcalcycle[selfcalcycle_id] > mval:
            automaskthreshold_selfcalcycle[selfcalcycle_id] = maskthreshold_selfcalcycle[selfcalcycle_id]

    # idgin = args['idg'] # store here as we update args['idg'] at some point to made image000 for selfcalcycle 0 in when --DDE is enabled

    if isinstance(args['channelsout'], str) and args['channelsout'] == 'auto':
        args['channelsout'] = set_channelsout(mslist, args)

    if type(args['fitspectralpol']) is str:
        if args['fitspectralpol'] == 'auto':
            args['fitspectralpol'] = set_fitspectralpol(args['channelsout'])
        else:
            raise Exception("fitspectralpol needs to be an integer or 'auto'")

    if args['parallelgridding'] == 0: # means the user asks to set it out automatically
        if args['imsize'] > 20000:
            args['parallelgridding'] = 1
        if args['imsize'] < 18000:
            args['parallelgridding'] = 2
        if args['imsize'] < 12500:
            args['parallelgridding'] = 4
        if args['imsize'] < 6000:
            args['parallelgridding'] = 6

    if args['wgridderaccuracy'] is None:
        args['wgridderaccuracy'] = 1e-4

    # set flagging paramters explicitly here to False if not set by user (to avoid they are left at None which can cause issues in some functions if not checked properly)
    if args['aoflagger'] is None:
        args['aoflagger'] = False
    if args['aoflagger_correcteddata'] is None:
        args['aoflagger_correcteddata'] = False
    if args['aoflagger_residualdata'] is None:
        args['aoflagger_residualdata'] = False
    if args['aoflagger_afterbandpassapply'] is None:
        args['aoflagger_afterbandpassapply'] = False
    if args['auto_flag_antennas'] is None:
        args['auto_flag_antennas'] = False
    if args['removemostlyflaggedstations'] is None:
        args['removemostlyflaggedstations'] = False
    if args['flagtimesmeared'] is None:
        args['flagtimesmeared'] = False
    if args['removeinternational'] is None:
        args['removeinternational'] = False    

    # set boolean/None type parameters explicitly to False if they are None to avoid issues in some functions if not checked properly
    if args['autofrequencyaverage'] is None:
        args['autofrequencyaverage'] = False
    if args['mask_extended'] is None:
        args['mask_extended'] = False
    if args['multiscale'] is None:
        args['multiscale'] = False   
    if args['update_uvmin'] is None:
        args['update_uvmin'] = False
    if args['update_multiscale'] is None:
        args['update_multiscale'] = False

    return longbaseline, LBA, HBAorLBA, freq, fitsmask, \
        maskthreshold_selfcalcycle, automaskthreshold_selfcalcycle, outtarname


def compute_phasediffstat(mslist, args, nchan='1953.125kHz', solint='10min'):
    """
    Compute phasediff statistics using circular standard deviations on the solutions of a scalarphasediff solve for
    international stations.

    The procedure and rational is described in Section 3.3 of de Jong et al. (2024)


    ISSUE?: Seems that the phasediff scores are different compared to what is generated by VLBI pipeline?

    Parameters
    ----------
    mslist : list of str
        Measurement Sets on which to calculate phase-difference statistics.
    args : dict
        Parsed facetselfcal options controlling conversion and calibration.
    nchan : int or str, optional
        Channel count or channel-width setting passed to the calibration solve.
    solint : int or str, optional
        Solution interval used for the phase-difference solve.
    """

    mslist_input = mslist[:]  # make a copy

    # Verify if we are in circular pol basis and do beam correction
    for ms in mslist:
        beamcor_and_lin2circ(ms, args, dysco=args['dysco'],
                             beam=set_beamcor(ms, args['beamcor'], args),
                             lin2circ=True,
                             losotobeamlib=args['losotobeamcor_beamlib'],
                             metadata_compression=args['metadata_compression'])

    # Phaseup if needed
    if args['phaseupstations']:
        mslist = phaseup(mslist, datacolumn='DATA', superstation=args['phaseupstations'],
                         dysco=args['dysco'], metadata_compression=args['metadata_compression'])

    # Solve and get best solution interval
    for ms_id, ms in enumerate(mslist):
        scorelist = []
        parmdb = 'h5_solutions/scalarphasediffstat' + '_' + os.path.basename(ms) + '.h5'
        runDPPPbase(ms, str(solint) + 'min', nchan, parmdb=parmdb, soltype='scalarphasediff', uvminscalarphasediff=0.0,
                    dysco=args['dysco'], modelstoragemanager=args['modelstoragemanager'], uvmin=20000)

        # Reference solution interval
        ref_solint = solint

        terminal_print('Image scores:', scorelist)

        # Set optimal std score
        optimal_score = 1.75

        if type(ref_solint) == str:
            if 'min' in ref_solint:
                ref_solint = float(re.findall(r'-?\d+', ref_solint)[0])
            elif ref_solint[-1] == 's' or ref_solint[-1] == 'sec':
                ref_solint = float(re.findall(r'-?\d+', ref_solint)[0]) // 60
            else:
                sys.exit("ERROR: ref_solint needs to be a float with solution interval in minutes "
                         "or string ending on min (minutes) or s/sec (seconds)")

        S = GetSolint(parmdb, optimal_score=optimal_score, ref_solint=ref_solint)

        # Write to STAT SCORE to original MS DATA-col header mslist_input
        with table(mslist_input[ms_id], readonly=False) as t:
            t.putcolkeyword('DATA', 'SCALARPHASEDIFF_STAT', S.cstd)

        if args['phasediff_only']:
            generate_phasediff_csv(glob.glob("h5_solutions/scalarphasediffstat*.h5"))
        
    return


def make_fullpol_image(mslistim, cycle, stackstr, fitsmask, automask, multiscale, h5list, facetregionfile):
    """
    Make a Stokes IQUV image for quality checking (--makeimage-fullpol).

    Parameters
    ----------
    mslistim : list of str
        Measurement Sets to image.
    cycle : int
        Selfcal cycle number used in the output image name.
    stackstr : str
        Suffix for stacked imaging (empty string if not stacking).
    fitsmask : str or None
        Clean mask.
    automask : float
        WSClean auto-mask threshold.
    multiscale : bool
        Use multiscale cleaning.
    h5list : list of str
        h5parm solutions to apply during facet imaging.
    facetregionfile : str or None
        Facet region file for DDE imaging.
    """
    makeimage(mslistim, args['imagename'] + 'fullpol' + str(cycle).zfill(3) + stackstr,
              args['pixelscale'], args['imsize'],
              args['channelsout'], args['niter'], args['robust'],
              multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask,
              uvminim=args['uvminim'], uvmaxim=args['uvmaxim'], fitspectralpol=0,
              automask=automask, removenegativecc=False, predict=False,
              paralleldeconvolution=args['paralleldeconvolution'],
              parallelgridding=args['parallelgridding'],
              fullpol=True,
              h5list=h5list, facetregionfile=facetregionfile,
              stack=args['stack'],
              disable_primarybeam_image=args['disable_primary_beam'],
              disable_primarybeam_predict=args['disable_primary_beam'],
              fulljones_h5_facetbeam=not args['single_dual_speedup'])


###############################
############## MAIN ###########
###############################

def main():
    """
    Run the facet self-calibration command-line workflow.

    Returns
    -------
    None
        Processing follows the command-line arguments.

        An offline HTML report is generated shortly after startup and refreshed
        during and after processing.
    """

    global _REPORT_RUN_INITIALIZED

    options = option_parser()

    # Temporary warning
    #if options.helperscriptspathh5merge is not None or options.helperscriptspath is not None:
    #    print("WARNING: --helperscriptspath and --helperscriptspathh5merge are not being used anymore")

    # If a config file exists, then read the information. Priotise specified config over default.
    if os.path.isfile(options.configpath):
        config = options.configpath
        terminal_print("A config file (%s) exists, using it. This contains:" % config)
        parser = configparser.ConfigParser()
        # Preserve upper case in options
        parser.optionxform = str
        with open(config) as f:
            parser.read_string("[DEFAULT]\n" + f.read())
        for k, v in parser["DEFAULT"].items():
            terminal_print(f"{k} = {v}")
            if k not in vars(options).keys():
                raise KeyError(
                    "Encountered invalid option {:s} in config file {:s}.".format(
                        k, os.path.abspath(config)
                    )
                )
            if os.path.exists(os.path.expanduser(v)):
                setattr(options, k, v)
            else:
                setattr(options, k, ast.literal_eval(v))

    global args
    args = vars(options)
    try:
        requested_run_mode = _run_mode_from_args(args)
        _ensure_run_mode_marker(Path.cwd(), requested_run_mode)
    except ValueError as exc:
        terminal_print("Cannot start facetselfcal: {}".format(exc))
        raise SystemExit(2) from None

    _configure_selfcal_log(args['start'])
    _log_machine_info()
    _prepare_html_overview(args['start'])

    try:
        resource_monitor = ResourceMonitor(interval=15.0, start_cycle=args['start'])
        resource_monitor.start()
        set_resource_monitor(resource_monitor)
    except Exception as exc:
        logger.warning("Could not start resource monitor: %s", exc)
        set_resource_monitor(None)
    
    with open("full_config.txt", "w") as file:
        for key, value in args.items():
            file.write(f"{key} = {value}\n")
    _REPORT_RUN_INITIALIZED = True

    if args['stack']:
        args['dysco'] = False  # no dysco compression allowed as multiple various steps violate the assumptions that need to be valid for proper dysco compression
        args['metadata_compression'] = False  # no metadata compression allowed
        args['noarchive'] = True
    if args['skymodelsetjy']:
        args['dysco'] = False  # no dysco compression allowed as CASA does not work with dysco compression
        args['modelstoragemanager'] = None  # no model compression allowed as CASA does not work with dysco compression
    args |= {'telescope': get_telescope_from_ms(sorted(args['ms']))} # get telescope from MS and add to args
    
    global submodpath, datapath, facetselfcal_version
    datapath = os.path.dirname(os.path.abspath(__file__))
    submodpath = '/'.join(datapath.split('/')[0:-1])+'/submods'
    shutil.copy(submodpath + '/polconv.py', '.')

    facetselfcal_version = '20.3.0'
    print_title(facetselfcal_version)
    _write_html_overview(
        Path.cwd(), status="running", announce_browser=True
    )

    # copy h5s locally
    for h5parm_id, h5parmdb in enumerate(args['preapplyH5_list']):
        if h5parmdb is not None:
            shutil.copy(h5parmdb, '.')
            args['preapplyH5_list'][h5parm_id] = h5parmdb.split('/')[-1]  # update input list to local location

    # deal with glob-like string input for preapplybandpassH5_list
    if  len(args['preapplybandpassH5_list']) == 1 and \
        args['preapplybandpassH5_list'][0] is not None and \
        ('*' in args['preapplybandpassH5_list'][0] or '?' in args['preapplybandpassH5_list'][0]): 
             args['preapplybandpassH5_list'] = glob.glob(args['preapplybandpassH5_list'][0])
             assert len(args['preapplybandpassH5_list']) >= 1 # assert that something is found
             terminal_print('Found these bandpass solutions', args['preapplybandpassH5_list'])

    # reorder lists based on sorted(args['ms'])
    if type(args['skymodel']) is list:
        args['skymodel'] = [x for _, x in sorted(zip(args['ms'], args['skymodel']))]
    if type(args['wscleanskymodel']) is list:
        args['wscleanskymodel'] = [x for _, x in sorted(zip(args['ms'], args['wscleanskymodel']))]
    if type(args['skymodelpointsource']) is list:
        args['skymodelpointsource'] = [x for _, x in sorted(zip(args['ms'], args['skymodelpointsource']))]
    mslist = sorted(args['ms'])

    # check if all MS exist
    check_if_ms_exists(mslist)

    # turn of metadata_compression for non-LOFAR data if needed
    set_metadata_compression(mslist, args)

    # remove trailing slashes
    mslist_tmp = []
    for ms in mslist:
        mslist_tmp.append(ms.rstrip('/'))
    mslist = mslist_tmp[:]  # .copy()

    # remove ms which are too short (to catch Elais-N1 case of 600s of data)
    if not args['testing']:
        check_valid_ms(mslist)
        # check if ms channels are equidistant in freuqency (to confirm DP3 concat was used properly)
        check_equidistant_freqs(mslist)
        # do some input checking
        inputchecker(args, mslist)

    # Gunzip wscleanskymodels if needed
    if args['wscleanskymodel'] is not None and args['start'] == 0:
        for gz in glob.glob(args['wscleanskymodel'] + '-????-model.fits.gz') + glob.glob(args['wscleanskymodel'] + '-????-model-pb.fits.gz') + glob.glob(args['wscleanskymodel'] + '-????-model-fpb.fits.gz'):
            subprocess.run(['gunzip', '-f', gz], check=True)

    # SPLIT MS IF REQUESTED
    if args['split_fieldname'] is not None: 
        # if args['start'] > 0, we are in a restarting run, so we do not want to split the MS again
        # is args['start'] == 0, we are in the first run, so we do want to split the MS so dryrun needs to be False
        mslist = split_multidir_ms(mslist, field_names=[args['split_fieldname']], dryrun=args['start'] != 0)

    # TEST ONLY REMOVE
    if False:
        #modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, candidate_soltypelist_includedir = (
        #        prepare_DDE(args['imagename'], 0, ['P192+27_Coma.dysco.sub.shift.avg.weights.ms.archive0.copy','P192+27_Coma.dysco.sub.shift.avg.weights.ms.archive1.copy','P195+27_Coma.dysco.sub.shift.avg.weights.ms.archive0.copy'],
        #                    DDE_predict=args['DDE_predict'], telescope='LOFAR'))
        #sys.exit()

        modeldatacolumnsin = ['MODEL_DATA_DD' + str(i) for i in range(23)]
        soltypenumber = 0
        dirs, solints, smoothness, soltypelist_includedir = parse_facetdirections(args['facetdirections'], 0, args=args)
        soltypelist_includedir = np.swapaxes(np.array([soltypelist_includedir] * len(mslist)), 1, 0).T.tolist()      
        msnumber=0
        soltypenumber=1
        modeldatacolumns, sourcedir_removed, id_kept = updatemodelcols_includedir(modeldatacolumnsin,
                                                                                  soltypelist_includedir[soltypenumber][msnumber], mslist[0],
                                                                                  dryrun=True)
        # print(len(sourcedir_removed))
        # for ddir in sourcedir_removed:
        sourcedir_removed = sourcedir_removed.tolist()
        terminal_print('First removed source direction (RA/Dec):', sourcedir_removed[0])
        terminal_print('Model data columns:', modeldatacolumns)
        copy_over_solutions_from_skipped_directions(modeldatacolumnsin, id_kept)
        merge_splitted_h5_ordered(modeldatacolumnsin, 'test.h5', clean_up=False)
        sys.exit()


    # fix bad GMRT weights for LR and LR (because they are set to zero if created by fixuGMRT_revised.py)
    # zero weights result in fully flagged data when doing time/freq averaging in DP3
    if args['start'] == 0:
        fix_GMRT_weights(mslist, args)
        if not args['skip_uv_coverage_plotting']:
            plot_UV_coverage(mslist)
        # fix MS that have a variable reference pointing (like the ELVA)
        fix_variable_reference_pointing(mslist)


    # fix irregular time axes or avoid bloated MS, do this first before any other step
    # in case of multiple scans on the calibrator this avoids DP3 adding of lot flagged data in between
    if args['bandpass']:
        # args['skipbackup'] will set based on whether all MS were splitted
        # in case all MS were splitted a backup is not needed anymore
        args['flag_ampresetvalfactor'] = True  # set to True to flag bad bandpass solutions which values are too high or low
        if args['skymodel'] is None and args['skymodelpointsource'] is None \
                and args['wscleanskymodel'] is None and not args['skymodelsetjy']:
            args['skymodel'] = set_MeerKAT_bandpass_skymodel(mslist[0]) # try to set skymodel automatically    
        mslist, args['skipbackup'] = fix_equidistant_times(mslist, args['start'] != 0, args, 
                                                           dysco=args['dysco'], metadata_compression=args['metadata_compression'])
        
        # some old MeerKAT data has zero WEIGHT_SPECTRUM values which need to be fixed first
        # only do it for the banpass calibrator for now as the issue has only been found for calibrator data so far
        fix_zero_weight_spectrum(mslist)
 
    if args['timesplitbefore'] is None: # in this case we let facetselfcal decide by itself whether to split or not
        os.makedirs('plots', exist_ok=True)
        args['timesplitbefore'] = check_large_timegaps_ms(
            mslist[0],
            skip_time_coverage_plotting=args['skip_time_coverage_plotting'])
        for ms in mslist[1:]:
            check_large_timegaps_ms(
                ms,
                skip_time_coverage_plotting=args['skip_time_coverage_plotting'])

    if args['timesplitbefore']:
        mslist, args['skipbackup'] = fix_equidistant_times(mslist, args['start'] != 0, args, 
                                                           dysco=args['dysco'], metadata_compression=args['metadata_compression'])

    # cut ms if there are flagged times at the start or end of the ms
    if args['remove_flagged_from_startend']:
        mslist = sorted(remove_flagged_data_startend(mslist))

    if args['auto'] and args['telescope'] == 'MeerKAT' and not args['DDE']:
        if args['removemostlyflaggedstations'] is None:  # not set by user, so we can decide based on auto settings
            args['removemostlyflaggedstations'] = True  # for MeerKAT auto remove mostly flagged stations

    avgfreqstep = check_frequency_averaging_steps(mslist, args)

    if not args['skipbackup']:  # work on copy of input data as a backup
        terminal_print('Creating a copy of the data and work on that....')
        mslist = average(mslist, args=args, freqstep=[1] * len(mslist), timestep=1, start=args['start'], makecopy=True,
                         dysco=args['dysco'], aoflagger=(args['aoflagger'] and args['aoflaggerbeforeavg']),
                         aoflagger_strategy=args['aoflagger_strategy'], metadata_compression=args['metadata_compression'],
                         msinnchan=args['msinnchan'], msinstartchan=args['msinstartchan'], msinntimes=args['msinntimes'], 
                         msinstarttimeslot=args['msinstarttimeslot'],
                         removeinternational=args['removeinternational'], phaseshiftbox=args['phaseshiftbox'],
                         removemostlyflaggedstations=args['removemostlyflaggedstations'], flag_antennas=args['flag_antenna_list'])
        args['msinntimes'] = None # since we use it above set msinntimes to to None now
        args['msinstarttimeslot'] = None  # since we use it above set msinstarttimeslot to None now
        args['msinnchan'] = None  # since we use it above set msinnchan to None now
        args['msinstartchan'] = 0  # since we use it above set msinstartchan to 0 now
        args['removeinternational'] = False  # since we use it above set removeinternational to False now
        args['removemostlyflaggedstations'] = False  # since we use it above set removemostlyflaggedstations to False now
        args['phaseshiftbox'] = None  # since we use it above set phaseshiftbox to None now
        args['flag_antenna_list'] = None  # since we use it above set flag_antenna_list to None now
        if args['aoflagger'] and args['aoflaggerbeforeavg']:
            args['aoflagger'] = False # turn off now because we used it above    

    # remove the SYSPOWER column from a measurement set (relevant for (E)VLA data)
    # keeping this column will slow DP3 down significantly
    remove_syspower(mslist)

    # flag known bad frequencies for uGMRT data
    if args['start'] == 0: flag_uGMRT_badfreqs(mslist, args)

    # take out bad WEIGHT_SPECTRUM values if weightspectrum_clipvalue is set
    if args['weightspectrum_clipvalue'] is not None and args['start'] == 0:
        fix_bad_weightspectrum(mslist, clipvalue=args['weightspectrum_clipvalue'])
    if args['data_clipvalue'] is not None and args['start'] == 0:
        clip_DATA(mslist, clipvalue=args['data_clipvalue'])

    # create Ateam plots
    if not args['phasediff_only'] and not args['skip_Ateam_plotting']:
        create_Ateam_seperation_plots(mslist, start=args['start'])

    # flag shadowed antennas for MeerKAT, VLA, EVLA, ATCA, GMRT, WSRT, ASKAP
    if args['start'] == 0:
        for ms in mslist:
            if get_telescope_from_ms(ms) in ['MeerKAT', 'VLA', 'EVLA', 'ATCA', 'GMRT', 'WSRT', 'ASKAP']:
                run('python ' + submodpath + '/shadow_flag.py --ms=' + ms)

    # fix UVW coordinates (for time averaging with MeerKAT data)
    if args['start'] == 0: fix_uvw(mslist)

    # fix antenna info for GMRT data (nothing happens for other telescopes)
    if args['start'] == 0: fix_antenna_info_gmrt(mslist, args)

    # fix irregular time axes if needed (do this after flagging)
    # in case --bandpass was set this step should not do anything because
    # the MS were already splitted before in case of gaps (so they will be regular now)
    # the [0] at the end is to avoid the extra output returned by fix_equidistant_times
    mslist = fix_equidistant_times(mslist, args['start'] != 0, args, \
                                   dysco=args['dysco'], metadata_compression=args['metadata_compression'])[0]

    # for the GMRT fix possible bad UVW coordinates from DP3 inserting flagged timeslots at gaps, do this before any DP3 time averaging
    # these bad UVW coodinates cause issues when time averaging the data in DP3, so we need to fix them first (becasue that can combine flagged and unflagged visibilities and then the UVW coordinates are wrong for the unflagged averaged visibilities)
    if args['start'] == 0:
        for ms in mslist:
            if get_telescope_from_ms(ms) == 'GMRT':
                run('python ' + submodpath + '/fix_uvw_with_offset.py ' + ms + ' --flagged-only')

    # fix TIME axis for GMRT data (nothing happens for other telescopes)
    if args['start'] == 0: fix_time_axis_gmrt(mslist, args)

    # reset weights if requested
    if args['resetweights']:
        for ms in mslist:
            cmd = "'update " + ms + " set WEIGHT_SPECTRUM=WEIGHT_SPECTRUM_SOLVE'"
            run("taql " + cmd, taql=True)

    # SETUP VARIOUS PARAMETERS
    longbaseline, LBA, HBAorLBA, freq, fitsmask, maskthreshold_selfcalcycle, \
        automaskthreshold_selfcalcycle, outtarname = basicsetup(mslist)

    removed_products = _remove_previous_cycle_products(
        args['start'],
        args['stop'],
        args['imagename'],
        preserved_inputs=(
            fitsmask,
            args.get('fitsmask'),
            args.get('fitsmask_start'),
            args.get('facetdirections'),
            args.get('preapplyH5_list'),
            args.get('preapplybandpassH5_list'),
            args.get('skymodel'),
            args.get('skymodelpointsource'),
            args.get('wscleanskymodel'),
            args.get('boxfile'),
            args.get('DS9cleanmaskregionfile'),
            args.get('DS9cleanmaskregionfile_exclude'),
        ),
        remove_best_outputs=args['early_stopping'],
    )
    if removed_products:
        terminal_print(
            'Removed', len(removed_products), 'stale products from cycles',
            args['start'], 'and later'
        )
        logger.info(
            'Removed %d stale generated products from cycles %d and later',
            len(removed_products), args['start']
        )

    # SET MODEL STORAGE MANAGER
    args['modelstoragemanager'] = set_modelstoragemanager(args['telescope'], args)

    # avgfreqstep was planned and validated before the first average() call.

    # COMPUTE PHASE-DIFF statistic
    if args['compute_phasediffstat']:
        if longbaseline and len(args['soltype_list'])<2: # of more than one soltype list, phasediff stat is generated later on in the script
            compute_phasediffstat(mslist, args)
            if args['phasediff_only']:
                return
        else:
            logger.info("--compute-phasediffstat requested but no long-baselines in dataset.")

    # set once here, preserve original mslist in case --removeinternational was set
    if args['removeinternational']:
        # used for h5_merge add_ms_stations option
        mslist_beforeremoveinternational = mslist[:]  # copy by slicing otherwise list refers to original
    else:
        mslist_beforeremoveinternational = None

    if not check_pointing_centers(mslist, args): args['phaseshiftbox'] = 'align' # set phaseshiftbox

    # AVERAGE if requested/possible
    mslist = average(mslist, args=args, freqstep=avgfreqstep, timestep=args['avgtimestep'],
                     start=args['start'], msinnchan=args['msinnchan'], msinstartchan=args['msinstartchan'],
                     phaseshiftbox=args['phaseshiftbox'], msinntimes=args['msinntimes'], msinstarttimeslot=args['msinstarttimeslot'],
                     dysco=args['dysco'], removeinternational=args['removeinternational'],
                     removemostlyflaggedstations=args['removemostlyflaggedstations'], aoflagger=args['aoflagger'], aoflagger_strategy=args['aoflagger_strategy'], aoflaggerbeforeavg=args['aoflaggerbeforeavg'],
                     metadata_compression=args['metadata_compression'], flag_antennas=args['flag_antenna_list'])

    for ms in mslist:
        compute_distance_to_pointingcenter(ms, HBAorLBA=HBAorLBA, warn=longbaseline, returnval=False)

    # this needs to be done after averaging because of time averaging changing the intervals
    if args['facetdirections'] is not None:
        # Check facet directions for high memory/long solution intervals
        facetdirections_list = args['facetdirections'] if isinstance(args['facetdirections'], list) else [args['facetdirections']]
        for facetdirections in facetdirections_list:
            check_for_highmem_longsolint(mslist, facetdirections, args=args)

    # extra flagging if requested
    #if args['start'] == 0 and args['aoflagger'] and not args['aoflaggerbeforeavg']:
    #    runaoflagger(mslist, strategy=args['aoflagger_strategy'])

    # compute bandwidth smearing
    report_bandwidth_smearing(mslist, args['imsize'])
 
    # backup flagging column for option --restoreflags if needed
    if args['restoreflags']:
        for ms in mslist:
            create_backup_flag_col(ms)

    # LOG INPUT SETTINGS
    if args['stack']:
        _write_stack_imaging_group_manifest(mslist)
    logbasicinfo(args, fitsmask, mslist, facetselfcal_version, sys.argv)
    _write_html_overview(Path.cwd(), status="running", quiet=True)

    # Make starting skymodel from TGSS or VLASS survey if requested

    args, tgssfitsfile = set_skymodels_external_surveys(args, mslist)

    nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
        smoothnessspectralexponent_list, smoothnessrefdistance_list, \
        antennaconstraint_list, resetsols_list, resetdir_list, soltypecycles_list, \
        uvmin_list, uvmax_list, uvminim_list, uvmaxim_list, normamps_list, solve_msinnchan_list, \
        solve_msinstartchan_list, antenna_averaging_factors_list, antenna_smoothness_factors_list = \
        setinitial_solint(mslist, options)

    check_antenna_factors(antenna_averaging_factors_list, antenna_smoothness_factors_list, \
                          mslist, args['facetdirections'], args)

    # Get restoring beam for DDFACET in case it is needed
    restoringbeam = calculate_restoringbeam(mslist, LBA)

    # set once here, will be updated in the loop below if phaseup is requested
    if args['phaseupstations'] is not None:
        # used for h5_merge add_CS option
        mslist_beforephaseup = mslist[:]  # note copy by slicing otherwise list refers to original
    else:
        mslist_beforephaseup = None

    # extra smearing flagging
    if args['flagtimesmeared'] and args['start'] == 0:
        for ms in mslist:
            flag_smeared_data(ms)
    
    # flag autocorrelations for MeerKAT (these are not flagged by the SDP pipeline)
    if args['telescope'] == 'MeerKAT' and args['start'] == 0:
        flag_autocorr(mslist)
  
    wsclean_h5list = []
    facetregionfile = None
    soltypelist_includedir = (np.array([[None] * len(args['soltype_list'])] * len(mslist))).T.tolist() # so that this variable is at least defined for DI solves or in case is is not in the facetdirections input
    modeldatacolumns = []
    if args['stack']:
        fitsmask_list = [None] * len(mslist)
    else:
        fitsmask_list = [
            fitsmask]  # *len(mslist) last part not needed because of the enumerate(nested_mslistforimaging(mslist, stack=args['stack']))

    if args['groupms_h5facetspeedup'] and args['start'] == 0 and len(mslist) > 1:
        concat_ms_wsclean_facetimaging(mslist)
    

    # create ./facet_regions/facets.reg so we have it avaialble for image000
    # so that we can use WSClean facet mode, but without having h5 DDE solutions
    if args['facetdirections'] is not None and args['start'] == 0:
        facetdirections_list = args['facetdirections'] if isinstance(args['facetdirections'], list) else [args['facetdirections']]
        create_facet_directions(None, 0, args=args, ms=mslist[0], imsize=args['imsize'],
                                pixelscale=args['pixelscale'], facetdirections=facetdirections_list[0])
        facetregionfile = 'facet_regions/facets.reg'  # so when making image000 we can use it without having h5 DDE solutions

    if args['start'] > 0 and  args['stop'] == args['start'] and args['remove_outside_center']:
        terminal_print('Only doing an imaging and extract step')
        remove_outside_center_only = True
    else:
        remove_outside_center_only = False

    if args['start'] > 0 and  args['stop'] == args['start'] and args['createresidualdatacolumn']:
        terminal_print('Only creating RESIDUAL_DATA')
        createresidualdatacolumn_only = True
    else:
        createresidualdatacolumn_only = False

    # check if we can use -apply-facet-beam or disable_primary_beam needs to be set
    check_applyfacetbeam(mslist, args['imsize'], args['pixelscale'], args['telescope'], args, \
                         enlarge_safe_FoV_diameter=args['enlarge_safe_FoV_diameter'])

    # download MWS primary beam model if needed
    if args['telescope'] == 'MWA' and not args['disable_primary_beam']:
        download_MWA_beam_model()

    # compute flagging statistics for the MS list and log the results to selfcal.log
    # reports the total percentage of data flagged in the MS and list the antennas that are fully flagged
    if args['start'] == 0:
        compute_flagging_statistics(mslist)

    # Insert MS history from facetselfcal
    for ms in mslist:
        insert_history_ms(ms, parse_input_args(options), appver=facetselfcal_version)

    # ----- START SELFCAL LOOP -----
    for i in range(args['start'], 999):  # large number, will break when i == args['stop']-1
        logger.info('Starting self-calibration cycle %d' % i)
        if get_resource_monitor() is not None:
            get_resource_monitor().set_cycle(i)
        # UPDATE REMOVENEGATIVEFROMMODEL SETTING, 
        # for high dynamic range it is better to keep negative clean components (based on a very clear 3C84 test case)
        if args['autoupdate_removenegativefrommodel'] and i > 1 and not args['DDE']:
            args['removenegativefrommodel'] = False
        if args['autoupdate_removenegativefrommodel'] and args[
            'DDE']:  # never remove negative clean components for a DDE solve
            args['removenegativefrommodel'] = False
        if args['autoupdate_removenegativefrommodel'] and args['telescope'] == 'MeerKAT':
            # never remove negative clean components for MeerKAT as the image quality is already very good
            args['removenegativefrommodel'] = False

        # AUTOMATICALLY PICKUP PREVIOUS MASK (in case of a restart) and trigger multiscale
        if (i > 0) and (args['fitsmask'] is None):
            fitsmask, fitsmask_list = set_fitsmask_restart(i, mslist, args)
            # update to multiscale cleaning if large island is present
            args['multiscale'] = multiscale_trigger(fitsmask, args)
            # update uvmin if allowed/requested
            update_uvmin(fitsmask, longbaseline, LBA, args)
            # update channelsout
            args['channelsout'] = update_channelsout(i - 1, mslist, args)
            # update fitspectralpol
            args['fitspectralpol'] = update_fitspectralpol(args)

        # BEAM CORRECTION AND/OR CONVERT TO CIRCULAR/LINEAR CORRELATIONS
        for ms in mslist:
            if ((args['docircular'] or args['dolinear']) or (set_beamcor(ms, args['beamcor'], args))) and (i == 0):
                beamcor_and_lin2circ(ms, args, dysco=args['dysco'],
                                     beam=set_beamcor(ms, args['beamcor'], args),
                                     lin2circ=args['docircular'],
                                     circ2lin=args['dolinear'],
                                     losotobeamlib=args['losotobeamcor_beamlib'], idg=args['idg'],
                                     metadata_compression=args['metadata_compression'])

        # PHASE-UP if requested
        if args['phaseupstations'] is not None:
            if (i == 0) or (i == args['start']):
                mslist = phaseup(mslist, datacolumn='DATA', superstation=args['phaseupstations'],
                                 start=i, dysco=args['dysco'], metadata_compression=args['metadata_compression'])

        # PRE-APPLY BANDPASS-TYPE SOLUTIONS
        if (args['preapplybandpassH5_list'][0]) is not None and i == 0:
            preapply_bandpass(args['preapplybandpassH5_list'], mslist, dysco=args['dysco'], 
                              updateweights=args['preapplybandpassH5_updateweights'])
            # write keyword to MS to indicate bandpass has been applied
            for ms in mslist:
                with table(ms, readonly=False) as t:
                    t.putcolkeyword('DATA', 'BANDPASS_APPLIED', True)
            # read out the keyword FAKE_RLLR for the DATA column if this keyword exists
            for ms in mslist:
                fake_rllr = False # initialize variable
                with table(ms, readonly=True, ack=False) as t:
                    if 'FAKE_RLLR' in t.colkeywordnames('DATA'): # inserted by fixuGMRT_revised.py for uGMRT data
                        fake_rllr = t.getcolkeyword('DATA', 'FAKE_RLLR')
                        terminal_print('This is a MS were fake cross-hand correlations were added to the DATA column')
                        terminal_print('Because we apply a bandpass solution we need to regenerate the fake cross-hand correlations in the DATA column because otherwise the noise is not flat')
                        logger.info('Regenerating fake cross-hand correlations in the DATA column for MS %s' % ms)
                if fake_rllr:
                    variance = getVarianceRRLL(ms)  # get variance of RR and LL correlations
                    updateCrosshand(ms, variance)  # update the cross-hand correlations in the DATA column based on the RR and LL variance         
            # importantly, run AOFlagger after applying the updateCrosshand update because otherwise the bandpass structure can be imposed on the noise if fake noise was injected in the cross-hand correlations. This intrduces major unwanted flagging
            if args['aoflagger_afterbandpassapply']:
                aoflagger_column(mslist, aoflagger_strategy=args['aoflagger_strategy_afterbandpassapply'], column='DATA')

        # PRE-APPLY SOLUTIONS (from a nearby direction for example)
        if (args['preapplyH5_list'][0]) is not None and i == 0:
            preapply(args['preapplyH5_list'], mslist, dysco=args['dysco'])

        if args['stopafterpreapply']:
            terminal_print('Stopping as requested via --stopafterpreapply')
            return

        # REMOVE EXISTING MODEL COLUMNS IN CASE OF a RESTART:
        # this can be important in case the model column storage manager has changed
        if args['start'] != 0: remove_model_columns(mslist)
        
        # CALIBRATE AGAINST THE INITAL SKYMODEL (selfcalcycle 0) IF REQUESTED
        if (args['skymodel'] is not None or args['skymodelpointsource'] is not None
            or args['wscleanskymodel'] is not None or args['skymodelsetjy']) and (i == 0):
            # Function that
            # add patches for DDE predict
            # also do prepare_DDE
            if args['DDE']:
                modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, candidate_soltypelist_includedir = (
                    prepare_DDE(args['skymodel'], i, mslist,
                                DDE_predict='DP3', restart=False, skyview=tgssfitsfile,
                                wscleanskymodel=args['wscleanskymodel'], skymodel=args['skymodel']))

                if candidate_solints is not None:
                    solint_list = candidate_solints
                if candidate_smoothness is not None:
                    smoothnessconstraint_list = candidate_smoothness
                if candidate_soltypelist_includedir is not None:
                    soltypelist_includedir = candidate_soltypelist_includedir
            
            else:
                dde_skymodel = None
            wsclean_h5list = calibrateandapplycal(mslist, i, solint_list, nchan_list, 
                                                  soltypecycles_list, smoothnessconstraint_list,
                                                  smoothnessreffrequency_list,
                                                  smoothnessspectralexponent_list, smoothnessrefdistance_list,
                                                  antennaconstraint_list, resetsols_list, resetdir_list,
                                                  normamps_list, BLsmooth_list, solve_msinnchan_list, solve_msinstartchan_list,
                                                  antenna_averaging_factors_list, antenna_smoothness_factors_list,
                                                  soltypelist_includedir,
                                                  normamps=args['normampsskymodel'],
                                                  skymodel=args['skymodel'],
                                                  predictskywithbeam=args['predictskywithbeam'],
                                                  longbaseline=longbaseline,
                                                  skymodelsource=args['skymodelsource'],
                                                  skymodelpointsource=args['skymodelpointsource'],
                                                  wscleanskymodel=args['wscleanskymodel'], skymodelsetjy=args['skymodelsetjy'],
                                                  mslist_beforephaseup=mslist_beforephaseup,
                                                  telescope=args['telescope'],
                                                  modeldatacolumns=modeldatacolumns, dde_skymodel=dde_skymodel,
                                                  DDE_predict=set_DDE_predict_skymodel_solve(args['wscleanskymodel']),
                                                  mslist_beforeremoveinternational=mslist_beforeremoveinternational)

        # Generate phasediff stat CSV here if more than 2
        if args['compute_phasediffstat'] and len(args['soltype_list'])>=2:
            if args['solint_list'][0]=='10min':
                generate_phasediff_csv(glob.glob("h5_solutions/scalarphasediff*.h5"))
            else:
                terminal_print("WARNING: Cannot generate phasediff CSV because solution interval for scalarphasediff is not 10min")
        if args['phasediff_only']:
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return

        # SET MULTISCALE
        if args['multiscale'] and i >= args['multiscale_start']:
            multiscale = True
        else:
            multiscale = False

        # RESTART FOR A DDE RUN, set modeldatacolumns and dde_skymodel
        if args['DDE'] and args['start'] != 0 and i == args['start']:
            if args['auto_directions']: args['facetdirections'] = 'directions/directions_' + str(i-1).zfill(3) + '.txt'
           
            modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, candidate_soltypelist_includedir = (
                prepare_DDE(args['imagename'], i, mslist,
                            DDE_predict=args['DDE_predict'], restart=True, telescope=args['telescope']))
            wsclean_h5list = list(np.load('misc/wsclean_h5list' + str(i-1).zfill(3) + '.npy'))
            # re-create facet_regions/facets.reg here
            # in a restart the number of directions from the previous facet_regions/facets.reg might not match that in the h5
            create_facet_directions(args['imagename'], i, args=args, ms=mslist[0], imsize=args['imsize'],
                            pixelscale= args['pixelscale'], via_h5=True, h5=wsclean_h5list[0])

        # RESTART FOR A DI RUN, set CORRECTED_DATA from previous selfcal cycle
        if not args['DDE'] and  args['start'] != 0 and i == args['start']: 
            applycal_restart_di(mslist, i, args)

        #  --- START IMAGING PART ---
        for msim_id, mslistim in enumerate(nested_mslistforimaging(mslist, stack=args['stack'])):
            if args['stack']:
                stackstr = '_stack' + str(msim_id).zfill(2)
            else:
                stackstr = ''  # empty string
            if len(modeldatacolumns) > 1:
                facetregionfile = 'facet_regions/facets.reg'
            # else:
            # if args['DDE'] and i == 0: # we are making image000 without having DDE solutions yet
            # if telescope == 'LOFAR' and not args['disable_IDG_DDE_predict']: # so image000 has model-pb
            #    args['idg'] = True
            makeimage(mslistim, args['imagename'] + str(i).zfill(3) + stackstr,
                      args['pixelscale'], args['imsize'],
                      args['channelsout'], args['niter'], args['robust'],
                      multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[msim_id],
                      uvminim=args['uvminim'], predict=not args['stopafterskysolve'],
                      fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                      restoringbeam=restoringbeam, automask=automaskthreshold_selfcalcycle[i],
                      removenegativecc=args['removenegativefrommodel'],
                      paralleldeconvolution=args['paralleldeconvolution'],
                      parallelgridding=args['parallelgridding'],
                      h5list=wsclean_h5list, selfcalcycle=i,
                      facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                      stack=args['stack'],
                      disable_primarybeam_image=args['disable_primary_beam'],
                      disable_primarybeam_predict=args['disable_primary_beam'],
                      fulljones_h5_facetbeam=not args['single_dual_speedup'])
            
            if remove_outside_center_only or createresidualdatacolumn_only:

                if createresidualdatacolumn_only:
                    create_residual_data_column(mslist, args['imagename'] + str(i).zfill(3), 
                                           args['pixelscale'], args['imsize'], 
                                           args['channelsout'], 
                                           single_dual_speedup=args['single_dual_speedup'],
                                           dysco=args['dysco'], idg=args['idg'],
                                           h5list=wsclean_h5list, facetregionfile=facetregionfile,
                                           disable_primary_beam=args['disable_primary_beam'], 
                                           parallelgridding=args['parallelgridding'])
                    if args['compute_weightspectrum']:
                        for ms in mslist:
                            cmdw = ['python', f'{submodpath}/uGMRTSetWeights.py', '-f', '12', ms]
                            subprocess.run(cmdw, check=True) 

                if remove_outside_center_only:
                    remove_outside_box(mslist, args['imagename'] + str(i).zfill(3), args['pixelscale'],
                                       args['imsize'], args['channelsout'], single_dual_speedup=args['single_dual_speedup'],
                                       dysco=args['dysco'], userbox=args['remove_outside_center_box'], idg=args['idg'],
                                       h5list=wsclean_h5list, facetregionfile=facetregionfile,
                                       disable_primary_beam=args['disable_primary_beam'], 
                                       parallelgridding=args['parallelgridding'],
                                       metadata_compression=args['metadata_compression'], 
                                       avgtimestep=args['remove_outside_center_avgtimestep'],
                                       avgfreqstep=args['remove_outside_center_avgfreqstep'],
                                       ddcor=not args['remove_outside_center_noddcor'])
                  
                if not args['keepmodelcolumns']: remove_model_columns(mslist)
                return
            
            # args['idg'] = idgin # set back
            if args['makeimage_ILTlowres_HBA']:
                if args['phaseupstations'] is None:
                    briggslowres = -1.5
                else:
                    briggslowres = -0.5
                makeimage(mslistim, args['imagename'] + '1.2arcsectaper' + str(i).zfill(3) + stackstr,
                          args['pixelscale'], args['imsize'],
                          args['channelsout'], args['niter'], briggslowres, uvtaper='1.2arcsec',
                          multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask_list[msim_id],
                          uvminim=args['uvminim'], uvmaxim=args['uvmaxim'], fitspectralpol=args['fitspectralpol'],
                          automask=automaskthreshold_selfcalcycle[i], removenegativecc=False,
                          predict=False,
                          paralleldeconvolution=args['paralleldeconvolution'],
                          parallelgridding=args['parallelgridding'],
                          h5list=wsclean_h5list, stack=args['stack'],
                          disable_primarybeam_image=args['disable_primary_beam'],
                          disable_primarybeam_predict=args['disable_primary_beam'],
                          fulljones_h5_facetbeam=not args['single_dual_speedup'])
            # only make full-pol images for the first and last selfcal cycle to save compute time
            if args['makeimage_fullpol'] and (i == 0 or i == args['stop'] - 1):
                make_fullpol_image(mslistim, i, stackstr, fitsmask_list[msim_id],
                                   automaskthreshold_selfcalcycle[i], multiscale, wsclean_h5list, facetregionfile)

            # PLOT IMAGE
            plotimage(i, args, stackstr, mask=fitsmask_list[msim_id], regionfile='facet_regions/facets.reg' if (args['DDE'] and facetregionfile is not None) else None)
        #  --- END IMAGING PART ---

        # RUN AOFLAGGER ON THE RESIDUAL DATA COLUMN IF REQUESTED
        if i in args['aoflagger_residualdata_selfcalcycle_list'] and args['aoflagger_residualdata']:
            create_residual_data_column(mslist, args['imagename'] + str(i).zfill(3), 
                                        args['pixelscale'], args['imsize'], 
                                        args['channelsout'], 
                                        single_dual_speedup=args['single_dual_speedup'],
                                        dysco=args['dysco'], idg=args['idg'],
                                        h5list=wsclean_h5list, facetregionfile=facetregionfile,
                                        disable_primary_beam=args['disable_primary_beam'], 
                                        parallelgridding=args['parallelgridding'])
            
            aoflagger_column(mslist, aoflagger_strategy=args['aoflagger_strategy_residualdata'], column='RESIDUAL_DATA')

        modeldatacolumns = []
        if args['DDE']:
            if args['auto_directions']: args['facetdirections'] = auto_direction(i, freq=freq, telescope=args['telescope'])
            modeldatacolumns, dde_skymodel, candidate_solints, candidate_smoothness, \
            candidate_soltypelist_includedir = prepare_DDE(args['imagename'], i, mslist, \
            DDE_predict=args['DDE_predict'], telescope=args['telescope'])

            if candidate_solints is not None:
                solint_list = candidate_solints
            if candidate_smoothness is not None:
                smoothnessconstraint_list = candidate_smoothness
            if candidate_soltypelist_includedir is not None:
                soltypelist_includedir = candidate_soltypelist_includedir

        else:
            dde_skymodel = None
        # compress model images with gzip to save space
        gzip_model_images(args['imagename'] + str(i).zfill(3))

        if args['stopafterskysolve']:
            terminal_print('Stopping as requested via --stopafterskysolve')
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return
        
        # COMPUTE BANDPASS IF REQUESTED
        if args['bandpass'] and args['bandpass_stop'] == 0: 
            terminal_print('Stopping as requested via --bandpass and compute bandpass')
            for parmdb_id, parmdb in enumerate(create_mergeparmdbname(mslist, i, skymodelsolve=True)):
                run('losoto ' + parmdb + ' ' + create_losoto_bandpassparset('a&p', mslist[parmdb_id], parmdb, skip_solution_plotting=args["skip_solution_plotting"]))
                set_weights_h5_to_one(parmdb)
                if os.path.isfile(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_')):
                    # remove existing bandpass file
                    Path(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_')).unlink(missing_ok=True)
                Path(parmdb).rename(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_'))    
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return

        # RUN AOFLAGGER ON THE CORRECTED DATA COLUMN IF REQUESTED
        if i in args['aoflagger_correcteddata_selfcalcycle_list'] and args['aoflagger_correcteddata']:
            aoflagger_column(mslist, aoflagger_strategy=args['aoflagger_strategy_correcteddata'], column='CORRECTED_DATA')

        # CHECK FOR HIGH DYNAMIC RANGE DATA AND ADJUST SETTINGS (DI-SOLVES ONLY)
        soltypecycles_list, solint_list, smoothnessconstraint_list, automaskthreshold_selfcalcycle, maskthreshold_selfcalcycle = autodetect_highDR(i, mslist, args['telescope'], soltypecycles_list, solint_list, smoothnessconstraint_list, args)

        # REDETERMINE SOLINTS IF REQUESTED
        if (i >= 0) and (args['usemodeldataforsolints']):
            terminal_print('Recomputing solints .... ')
            nchan_list, solint_list, BLsmooth_list, smoothnessconstraint_list, smoothnessreffrequency_list, \
                smoothnessspectralexponent_list, smoothnessrefdistance_list, \
                antennaconstraint_list, resetsols_list, resetdir_list, \
                soltypecycles_list, normamps_list = \
                auto_determinesolints(mslist, args['soltype_list'],
                                      longbaseline, LBA,
                                      innchan_list=nchan_list, insolint_list=solint_list,
                                      insmoothnessconstraint_list=smoothnessconstraint_list,
                                      insmoothnessreffrequency_list=smoothnessreffrequency_list,
                                      insmoothnessspectralexponent_list=smoothnessspectralexponent_list,
                                      insmoothnessrefdistance_list=smoothnessrefdistance_list,
                                      inantennaconstraint_list=antennaconstraint_list,
                                      inresetsols_list=resetsols_list,
                                      inresetdir_list=resetdir_list,
                                      innormamps_list=normamps_list,
                                      inBLsmooth_list=BLsmooth_list,
                                      insoltypecycles_list=soltypecycles_list, redo=True,
                                      tecfactorsolint=args['tecfactorsolint'],
                                      gainfactorsolint=args['gainfactorsolint'],
                                      gainfactorsmoothness=args['gainfactorsmoothness'],
                                      phasefactorsolint=args['phasefactorsolint'],
                                      delaycal=args['delaycal'])

        # CALIBRATE AND APPLYCAL
        wsclean_h5list = calibrateandapplycal(mslist, i, solint_list, nchan_list, 
                                              soltypecycles_list,
                                              smoothnessconstraint_list, smoothnessreffrequency_list,
                                              smoothnessspectralexponent_list, smoothnessrefdistance_list,
                                              antennaconstraint_list, resetsols_list, resetdir_list,
                                              normamps_list, BLsmooth_list,
                                              solve_msinnchan_list, solve_msinstartchan_list,
                                              antenna_averaging_factors_list, antenna_smoothness_factors_list,
                                              soltypelist_includedir,
                                              normamps=args['normampsskymodel'] if args['keepusingstartingskymodel'] else args['normamps'],
                                              skymodel=args['skymodel'] if args['keepusingstartingskymodel'] else None,
                                              skymodelpointsource=args['skymodelpointsource'] if args['keepusingstartingskymodel'] else None,
                                              wscleanskymodel=args['wscleanskymodel'] if args['keepusingstartingskymodel'] else None, 
                                              skymodelsetjy=args['skymodelsetjy'] if args['keepusingstartingskymodel'] else False,
                                              longbaseline=longbaseline,
                                              predictskywithbeam=args['predictskywithbeam'], skymodelsource=args['skymodelsource'],                                          
                                              mslist_beforephaseup=mslist_beforephaseup, telescope=args['telescope'],
                                              modeldatacolumns=modeldatacolumns, dde_skymodel=dde_skymodel,
                                              DDE_predict=args['DDE_predict'],
                                              mslist_beforeremoveinternational=mslist_beforeremoveinternational)


        if args['bandpass'] and i >=args['bandpass_stop']: 
            terminal_print('Stopping as requested via --bandpass and compute bandpass')
            for parmdb_id, parmdb in enumerate(create_mergeparmdbname(mslist, i, skymodelsolve=True)):
                run('losoto ' + parmdb + ' ' + create_losoto_bandpassparset('a&p', mslist[parmdb_id], parmdb, skip_solution_plotting=args["skip_solution_plotting"]))
                set_weights_h5_to_one(parmdb)
                if os.path.isfile(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_')):
                    # remove existing bandpass file
                    Path(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_')).unlink(missing_ok=True)
                Path(parmdb).rename(parmdb.replace('h5_solutions/merged_', 'h5_solutions/bandpass_'))
            if not args['keepmodelcolumns']: remove_model_columns(mslist)
            return

        # UPDATE UVMIN IF ALLOWED/REQUESTED
        update_uvmin(fitsmask, longbaseline, LBA, args)

        # UPDATE FITSMASK IF ALLOWED/REQUESTED
        fitsmask, fitsmask_list, imagename = update_fitsmask(fitsmask, maskthreshold_selfcalcycle, i, args, mslist, args['telescope'], longbaseline)

        # UPDATE TO MULTISCALE CLEANING IF LARGE ISLAND IS PRESENT
        args['multiscale'] = multiscale_trigger(fitsmask, args)

        # UPDATE CHANNELSOUT
        args['channelsout'] = update_channelsout(i, mslist, args)

        # UPDATE FITSPECTRALPOL
        args['fitspectralpol'] = update_fitspectralpol(args)

        # Get additional diagnostics and/or early-stopping --> in particular useful for calibrator selection and automation
        stop_for_early_stopping = False
        if args['early_stopping'] and len(mslist)>1:
            logger.info("WARNING: --early-stopping not yet developed for multiple input MeasurementSets.\nSkipping early-stopping evaluation.")

        elif args['early_stopping']:
            images, mergedh5 = get_images_solutions('fits_images', 'h5_solutions')
            stop_for_early_stopping = early_stopping(station='international' if longbaseline else 'alldutch',
                                                     cycle=i,
                                                     start_cycle=args['start'],
                                                     end_cycle=args['stop'],
                                                     nn_model_cache=args['nn_model_cache'],
                                                     skip_neural_network=args['nn_model_cache'] is None,
                                                     images=images,
                                                     mergedh5=mergedh5)

        _write_html_overview(Path.cwd(), status="running", cycle=i)
        if stop_for_early_stopping:
            break

        # STOP IF REQUESTED
        if i == args['stop']-1:
            terminal_print('Reached requested stop selfcal cycle')
            break    

    # Early stopping means the final cycle had no full-pol image made, image the data
    # with the final solutions applied instead (early stopping is single-MS only, so no stacking)
    if args['makeimage_fullpol'] and stop_for_early_stopping and i not in (0, args['stop'] - 1):
        make_fullpol_image(mslistim, i + 1, '', fitsmask,
                           automaskthreshold_selfcalcycle[i], multiscale, wsclean_h5list, facetregionfile)

    # Collect h5 files
    h5s = []
    if args['early_stopping']:
        h5s += glob.glob('h5_solutions/best_*solutions.h5')
    else:
        h5s += [f for ms in mslist for f in glob.glob('h5_solutions/merged_*selfcalcycle???_' + os.path.basename(ms) + '.h5')]
    # Write config file to merged h5parms
    for h5 in h5s:
        # Write all input parameters and facetselfcal version to h5parm
        add_config_to_h5(h5, 'full_config.txt')
        add_version_to_h5(h5, facetselfcal_version)

    # remove sources outside central region after selfcal (to prepare for DDE solves)
    # or create RESIDUAL_DATA column if requested
    if args['remove_outside_center'] or args['createresidualdatacolumn']:
        # make image after calibration so the calibration and images match
        # normally we would finish with calibration and not have the subsequent image, make this i+1 image here
        makeimage(mslistim, args['imagename'] + str(i + 1).zfill(3),
                  args['pixelscale'], args['imsize'],
                  args['channelsout'], args['niter'], args['robust'],
                  multiscale=multiscale, idg=args['idg'], fitsmask=fitsmask,
                  uvminim=args['uvminim'], predict=False,
                  fitspectralpol=args['fitspectralpol'], uvmaxim=args['uvmaxim'],
                  restoringbeam=restoringbeam, automask=automaskthreshold_selfcalcycle[i],
                  removenegativecc=args['removenegativefrommodel'],
                  paralleldeconvolution=args['paralleldeconvolution'],
                  parallelgridding=args['parallelgridding'],
                  h5list=wsclean_h5list, selfcalcycle=i + 1,
                  facetregionfile=facetregionfile, DDEimaging=args['DDE'],
                  disable_primarybeam_image=args['disable_primary_beam'],
                  disable_primarybeam_predict=args['disable_primary_beam'],
                  fulljones_h5_facetbeam=not args['single_dual_speedup'])

        # create RESIDUAL_DATA column if requested
        if args['createresidualdatacolumn']:
            create_residual_data_column(mslist, args['imagename'] + str(i + 1).zfill(3), 
                                        args['pixelscale'], args['imsize'], 
                                        args['channelsout'], 
                                        single_dual_speedup=args['single_dual_speedup'],
                                        dysco=args['dysco'], idg=args['idg'],
                                        h5list=wsclean_h5list, facetregionfile=facetregionfile,
                                        disable_primary_beam=args['disable_primary_beam'], 
                                        parallelgridding=args['parallelgridding'])
            if args['compute_weightspectrum']:
                for ms in mslist:
                    cmdw = ['python', f'{submodpath}/uGMRTSetWeights.py', '-f', '12', ms]
                    subprocess.run(cmdw, check=True) 
                    
        if args['remove_outside_center']:
            remove_outside_box(mslist, args['imagename'] + str(i + 1).zfill(3), 
                               args['pixelscale'],
                               args['imsize'], args['channelsout'], 
                               single_dual_speedup=args['single_dual_speedup'],
                               dysco=args['dysco'],
                               userbox=args['remove_outside_center_box'], idg=args['idg'],
                               h5list=wsclean_h5list, facetregionfile=facetregionfile,
                               disable_primary_beam=args['disable_primary_beam'], 
                               parallelgridding=args['parallelgridding'], 
                               metadata_compression=args['metadata_compression'],
                               avgtimestep=args['remove_outside_center_avgtimestep'],
                               avgfreqstep=args['remove_outside_center_avgfreqstep'],
                               ddcor=not args['remove_outside_center_noddcor'])
                   

        # compress model images with gzip to save space
        gzip_model_images(args['imagename'] + str(i + 1).zfill(3))

    # REMOVE MODEL_DATA type columns after selfcal
    if not args['keepmodelcolumns']: remove_model_columns(mslist)

    # CLEAN UP SOME FILES TO AVOID CLUTTER
    Path('polconv.py').unlink(missing_ok=True)

    # ARCHIVE DATA AFTER SELFCAL if requested
    if not longbaseline and not args['noarchive']:
        if not LBA:
            if args['DDE']:
                mergedh5_i = glob.glob('h5_solutions/merged_selfcalcycle' + str(i).zfill(3) + '*.h5')
                archive(mslist, outtarname, args['boxfile'], fitsmask, imagename, args, dysco=args['dysco'],
                        mergedh5_i=mergedh5_i, facetregionfile=facetregionfile, metadata_compression=args['metadata_compression'])
            else:
                archive(mslist, outtarname, args['boxfile'], fitsmask, imagename, args, dysco=args['dysco'], metadata_compression=args['metadata_compression'])
            cleanup(mslist)


def _write_html_overview(
    run_directory,
    status,
    error=None,
    cycle=None,
    announce_browser=False,
    quiet=False,
):
    """Generate or refresh the offline HTML overview without aborting a run.

    Parameters
    ----------
    run_directory : str or pathlib.Path
        Directory containing the facetselfcal run products.
    status : str
        Current run status to display in the report.
    error : str or None, optional
        Error summary to include for a failed run.
    cycle : int or None, optional
        Completed cycle that triggered this refresh, if any.
    announce_browser : bool, optional
        Highlight the absolute overview path and suggest opening it in a browser.
    quiet : bool, optional
        Suppress the success message while still updating the report.

    Returns
    -------
    None
    """
    try:
        if __package__:
            from .report import generate_html_overview
        else:
            from facetselfcal.report import generate_html_overview

        report_index = generate_html_overview(
            run_directory, status=status, error=error
        )
        if announce_browser:
            terminal_print(
                "\033[1;36mHTML overview is ready. You can open this page in a "
                "browser: {}\033[0m".format(Path(report_index).resolve())
            )
        elif quiet:
            return
        elif cycle is None:
            terminal_print("Offline HTML overview written to", report_index)
        else:
            terminal_print(
                f"Offline HTML overview updated after self-cal cycle {cycle} at",
                report_index,
            )
    except Exception:
        logger.exception("Could not generate the offline HTML overview")
        terminal_print(
            "Could not generate the offline HTML overview; see logs/selfcal.log"
        )


def _with_html_report(run_function):
    """Wrap a workflow function with final offline-report generation.

    Parameters
    ----------
    run_function : callable
        Zero-argument workflow function whose result and exceptions are kept.

    Returns
    -------
    callable
        Wrapper that generates a final report after an initialized run.
    """
    @wraps(run_function)
    def wrapped():
        """Run the workflow and record its final status in the report.

        Returns
        -------
        object
            Value returned by ``run_function``.

        Raises
        ------
        KeyboardInterrupt
            Re-raised after recording an interrupted run.
        SystemExit
            Re-raised after recording the stop or failure status.
        Exception
            Re-raised after recording the workflow failure.
        """
        global _REPORT_RUN_INITIALIZED

        _REPORT_RUN_INITIALIZED = False
        run_directory = Path.cwd()
        status = "unknown"
        error = None

        try:
            result = run_function()
        except KeyboardInterrupt:
            status = "interrupted"
            error = "Processing was interrupted by the user."
            raise
        except SystemExit as exc:
            status = "stopped" if exc.code in (None, 0) else "failed"
            if status == "failed":
                error = f"SystemExit: {exc}"
            raise
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            status = "completed"
            return result
        finally:
            resource_monitor = get_resource_monitor()
            if resource_monitor is not None:
                try:
                    resource_monitor.stop()
                except Exception:
                    logger.exception("Could not stop resource monitoring cleanly")
            if _REPORT_RUN_INITIALIZED:
                _write_html_overview(run_directory, status=status, error=error)

    return wrapped


main = _with_run_directory_lock(_with_html_report(main))


if __name__ == "__main__":
    main()
