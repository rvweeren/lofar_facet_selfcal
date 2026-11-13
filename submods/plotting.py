"""Plotting: self-calibration images, uv-coverage, MS time coverage and
time-gap checks, A-team separation plots."""
import json
import os
import pickle
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from casacore.tables import table  # type: ignore # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from submods.fits_utils import flatten  # noqa: E402
from submods.image_utils import findrms, get_image_size  # noqa: E402
from submods.runtime_utils import allocated_cpu_count, get_logger, run, terminal_print  # noqa: E402

# submods directory, same value as submodpath in main.py
submodpath = os.path.dirname(os.path.abspath(__file__))

# Image-centre plots at full resolution: "<plot>-zoom.png", ZOOM_PLOT_SIZE pixels wide
ZOOM_PLOT_SUFFIX = '-zoom.png'
ZOOM_PLOT_SIZE = 800
# Cutouts around the DDE directions per cycle, with a JSON manifest (read by the HTML overview)
DIRECTION_CUTOUT_DIR = os.path.join('plots', 'direction_cutouts')
DIRECTION_CUTOUT_BEAMS = 40.0
FACETDIRECTIONS_PICKLE = os.path.join('facet_regions', 'facetdirections.p')


def _get_ms_time_coverage(ms):
    """
    Return time slots, full-baseline slots, and the normal time step.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.

    Returns
    -------
    tuple
        Unique time slots, boolean mask of full-baseline slots, and normal step.
    """
    with table(ms, readonly=True, ack=False) as ms_table:
        if ms_table.nrows() == 0:
            return np.array([]), np.array([], dtype=bool), None
        times = np.asarray(ms_table.getcol('TIME'), dtype=float)

    unique_times, row_counts = np.unique(times, return_counts=True)
    if unique_times.size < 2:
        return unique_times, np.ones(unique_times.size, dtype=bool), None

    # A regular baseline grid has the same number of rows at every time.
    # The mode is robust to a small number of incomplete edge slots.
    normal_row_count = np.bincount(row_counts).argmax()
    full_baseline = row_counts == normal_row_count
    time_steps = np.diff(unique_times)
    positive_steps = time_steps[time_steps > 0]
    normal_step = float(np.median(positive_steps)) if positive_steps.size else None
    return unique_times, full_baseline, normal_step


def _find_ms_time_gaps(ms, timegap_threshold=1800, relative_threshold=0.5,
                       ignore_gap=60):
    """
    Find significant gaps longer than ``ignore_gap`` seconds in an MS.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    timegap_threshold : float, optional
        Threshold in seconds for significant gaps.
    relative_threshold : float, optional
        Relative threshold against neighboring observations.
    ignore_gap : float, optional
        Ignore gaps shorter than this value in seconds.

    Returns
    -------
    tuple
        Unique times, full-baseline mask, and list of significant gaps.
    """
    times, full_baseline, normal_step = _get_ms_time_coverage(ms)
    if normal_step is None or times.size < 2:
        return times, full_baseline, []

    gaps = []
    short_gap_duration = 0.0
    have_plotted_gap = False
    for index, time_delta in enumerate(np.diff(times)):
        if not (full_baseline[index] and full_baseline[index + 1]):
            short_gap_duration = 0.0
            have_plotted_gap = False
            continue
        missing_duration = float(time_delta - normal_step)
        if missing_duration < 0:
            continue
        if missing_duration < ignore_gap:
            if have_plotted_gap:
                short_gap_duration += missing_duration
            continue

        missing_duration += short_gap_duration
        short_gap_duration = 0.0

        left = index
        while (left > 0 and full_baseline[left - 1] and
               times[left] - times[left - 1] <= 1.5 * normal_step):
            left -= 1
        right = index + 1
        while (right < full_baseline.size - 1 and full_baseline[right + 1] and
               times[right + 1] - times[right] <= 1.5 * normal_step):
            right += 1
        left_duration = times[index] - times[left] + normal_step
        right_duration = times[right] - times[index + 1] + normal_step
        target_duration = left_duration + right_duration
        relative_limit = relative_threshold * target_duration
        exceeds_relative_limit = (target_duration > 0 and
                      missing_duration >= relative_limit)

        if missing_duration >= timegap_threshold or exceeds_relative_limit:
            gaps.append((times[index], times[index + 1], missing_duration))
        have_plotted_gap = True

    return times, full_baseline, gaps


def check_large_timegaps_ms(ms, timegap_threshold=1200, relative_threshold=0.5,
                            ignore_gap=60, skip_time_coverage_plotting=False):
    """
    Return whether an MS contains a significant all-baseline time gap.

    The normal cadence is estimated from the median difference between unique
    TIME values. A gap is the excess over that cadence. It is significant when
    it exceeds ``timegap_threshold`` seconds or ``relative_threshold`` times
    the combined neighboring continuous observations.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    timegap_threshold : float, optional
        Threshold in seconds for significant gaps.
    relative_threshold : float, optional
        Relative threshold against neighboring observations.
    ignore_gap : float, optional
        Ignore gaps shorter than this value in seconds.
    skip_time_coverage_plotting : bool, optional
        Do not generate a time-coverage plot.

    Returns
    -------
    bool
        True if significant gaps exist, False otherwise.
    """
    _, _, gaps = _find_ms_time_gaps(
        ms, timegap_threshold, relative_threshold, ignore_gap)
    ms_basename = os.path.basename(ms.rstrip(os.sep))
    plot_path = os.path.join('plots', f'{ms_basename}.time_coverage.png')
    if not skip_time_coverage_plotting:
        plot_ms_time_coverage(ms, plot_path, timegap_threshold, relative_threshold,
                              ignore_gap)
    return bool(gaps)


def plot_ms_time_coverage(ms, output_path=None, timegap_threshold=1200,
                          relative_threshold=0.5, ignore_gap=60):
    """
    Plot MS time coverage and annotate significant gaps.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    output_path : str, optional
        Output path for the plot.
    timegap_threshold : float, optional
        Threshold in seconds for significant gaps.
    relative_threshold : float, optional
        Relative threshold against neighboring observations.
    ignore_gap : float, optional
        Ignore gaps shorter than this value in seconds.

    Returns
    -------
    str or matplotlib.figure.Figure
        The output path when saved to disk, otherwise the Matplotlib figure.
    """
    times, full_baseline, gaps = _find_ms_time_gaps(
        ms, timegap_threshold, relative_threshold, ignore_gap)
    if times.size == 0:
        raise ValueError(f'Measurement Set has no TIME rows: {ms}')

    figure, axis = plt.subplots(figsize=(14, 4.5))
    time_hours = (times - times[0]) / 3600.0
    positive_steps = np.diff(times)
    positive_steps = positive_steps[positive_steps > 0]
    normal_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
    significant_gap_keys = {(gap_start, gap_end) for gap_start, gap_end, _ in gaps}
    shorter_gaps = []
    for index, time_delta in enumerate(np.diff(times)):
        if (full_baseline[index] and full_baseline[index + 1] and
                time_delta > 1.5 * normal_step):
            missing_duration = time_delta - normal_step
            if missing_duration < ignore_gap:
                continue
            gap_key = (times[index], times[index + 1])
            if gap_key not in significant_gap_keys:
                shorter_gaps.append((times[index], times[index + 1],
                                     missing_duration))
    axis.plot(time_hours, full_baseline.astype(int), drawstyle='steps-mid',
              color='tab:blue', linewidth=1.5)

    # Label each continuous coverage segment so the relative threshold can be
    # compared directly with the duration of the adjacent observations.
    segment_start = 0
    for index, time_delta in enumerate(np.diff(times)):
        if (not full_baseline[index] or not full_baseline[index + 1] or
                time_delta > 1.5 * normal_step):
            segment_end = index
            segment_duration = times[segment_end] - times[segment_start] + normal_step
            segment_middle = (times[segment_start] + times[segment_end]) / 2
            axis.text((segment_middle - times[0]) / 3600.0, 1.04,
                      f'{segment_duration / 60:.1f} min',
                      ha='center', va='bottom', color='black', fontsize=7,
                      rotation=90,
                      bbox=dict(facecolor='white', alpha=0.75,
                                edgecolor='none', pad=1))
            segment_start = index + 1

    segment_end = len(times) - 1
    segment_duration = times[segment_end] - times[segment_start] + normal_step
    segment_middle = (times[segment_start] + times[segment_end]) / 2
    axis.text((segment_middle - times[0]) / 3600.0, 1.04,
              f'{segment_duration / 60:.1f} min',
              ha='center', va='bottom', color='black', fontsize=7,
              rotation=90,
              bbox=dict(facecolor='white', alpha=0.75,
                        edgecolor='none', pad=1))

    for gap_start, gap_end, missing_duration in shorter_gaps:
        start_hours = (gap_start - times[0]) / 3600.0
        end_hours = (gap_end - times[0]) / 3600.0
        axis.axvspan(start_hours, end_hours, color='orange', alpha=0.25)
        axis.annotate(f'{missing_duration / 60:.1f} min',
                      xy=((start_hours + end_hours) / 2, 0.5),
                      ha='center', va='center', color='darkorange',
                      rotation=90, fontsize=7,
                      bbox=dict(facecolor='white', alpha=0.75,
                                edgecolor='none', pad=1))

    for gap_start, gap_end, missing_duration in gaps:
        start_hours = (gap_start - times[0]) / 3600.0
        end_hours = (gap_end - times[0]) / 3600.0
        axis.axvspan(start_hours, end_hours, color='tab:red', alpha=0.2)
        axis.annotate(f'{missing_duration / 60:.1f} min',
                      xy=((start_hours + end_hours) / 2, 0.5),
                      ha='center', va='center', color='tab:red',
                      rotation=90, fontsize=7,
                      bbox=dict(facecolor='white', alpha=0.75,
                                edgecolor='none', pad=1))

    axis.set_xlabel('Time since start (hours)')
    axis.set_ylabel('Complete baseline slot')
    axis.set_yticks([0, 1])
    axis.set_yticklabels(['incomplete', 'complete'])
    axis.set_ylim(-0.05, 1.2)
    axis.set_title(f'MS time coverage: {os.path.basename(ms.rstrip(os.sep))}')
    axis.legend(handles=[
        Patch(facecolor='tab:red', alpha=0.2,
              label='No visibility data (significant gap)'),
        Patch(facecolor='orange', alpha=0.25,
              label='No visibility data (shorter gap)')
    ], loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8,
        frameon=True)
    figure.tight_layout(rect=(0, 0, 0.8, 1))

    if output_path is not None:
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
    return figure


def plot_UV_coverage(mslist):
    """Create a shadeMS UV-coverage plot for each input Measurement Set."""
    try:
        os.makedirs('plots', exist_ok=True)
    except OSError as exc:
        get_logger().error('Could not create plots directory for UV-coverage plots: %s', exc)
        return

    for ms in mslist:
        ms_path = os.fspath(ms)
        ms_basename = os.path.basename(os.path.normpath(ms_path))
        ms_extension = os.path.splitext(ms_basename)[1]
        plot_path = os.path.join('plots', f'UVcoverage_{ms_basename}.png')
        command = [
            'shadems',
            '--xaxis', 'u',
            '--yaxis', 'v',
            '--cmap', 'kbc',
            '-j', str(max(1, allocated_cpu_count() // 2)),
            '--dir', 'plots',
            '--png', f'UVcoverage_{{ms}}{ms_extension}.png',
            '--fontsize', '20',
            ms_path,
        ]
        get_logger().info('Generating UV-coverage plot for %s', ms_path)
        try:
            result = subprocess.run(
                command,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            get_logger().error(
                'Cannot generate UV-coverage plots because shadems is unavailable: %s',
                exc,
            )
            return
        except OSError as exc:
            get_logger().error('Could not run shadems for %s: %s', ms_path, exc)
            continue

        if result.returncode != 0:
            error_detail = (result.stderr or '').strip()
            get_logger().error(
                'shadems failed for %s with exit status %d%s',
                ms_path,
                result.returncode,
                ': ' + error_detail if error_detail else '',
            )
            continue
        if not os.path.isfile(plot_path):
            get_logger().error(
                'shadems completed for %s but did not create the expected plot %s',
                ms_path,
                plot_path,
            )
            continue
        get_logger().info('Saved UV-coverage plot for %s to %s', ms_path, plot_path)


def selfcal_animatedgif(fitsstr, outname):
    """
    Generates an animated GIF from a FITS file using the DS9 visualization tool.

    This function constructs a command to run DS9 with specific parameters to
    create an animated GIF from the provided FITS file. The GIF is saved to the
    specified output file.

    Parameters
    ----------
    fitsstr : str
        The path to the input FITS file.
    outname : str
        The name of the output GIF file.

    Returns
    -------
    None
    """
    limit_min = -250e-6
    limit_max = 2.5e-2
    cmd = 'ds9 ' + fitsstr + ' '
    cmd += '-single -view basic -frame first -geometry 800x800 -zoom to fit -sqrt -scale limits '
    cmd += str(limit_max) + ' ' + str(limit_max) + ' '
    cmd += '-cmap ch05m151008 -colorbar lock yes -frame lock wcs '
    cmd += '-lock scalelimits yes -movie frame gif 10 '
    cmd += outname + ' -quit'
    run(cmd)
    return


def plotimage_astropy(fitsimagename, outplotname, mask=None, regionfile=None, \
                      cmap='bone', regioncolor='yellow', minmax=None, regionalpha=0.6):
    """
    Plot a FITS image with Astropy-aware coordinates and regions.

    Parameters
    ----------
    fitsimagename : str
        FITS image path.
    outplotname : str
        Output plot path.
    mask : str, optional
        FITS mask path.
    regionfile : str, optional
        Region file to overlay.
    cmap : str, optional
        Matplotlib colormap.
    regioncolor : str, optional
        Region overlay color.
    minmax : tuple, optional
        Display minimum and maximum.
    regionalpha : float, optional
        Region overlay transparency.

    Returns
    -------
    None
        The plot is written to disk.
    """

    # image noise info
    hdulist = fits.open(fitsimagename)
    imagenoiseinfo = findrms(np.ndarray.flatten(hdulist[0].data))
    
    try:
        get_logger().info(fitsimagename + ' Max image: ' + str(np.max(np.ndarray.flatten(hdulist[0].data))))
        get_logger().info(fitsimagename + ' Min image: ' + str(np.min(np.ndarray.flatten(hdulist[0].data))))
        get_logger().info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
    except Exception:
        pass # so we can also use the function without a logger open

    hdulist = flatten(fits.open(fitsimagename))

    data = fits.getdata(fitsimagename)
    head = fits.getheader(fitsimagename)
    f = plt.figure()
    ax = f.add_subplot(111, projection=WCS(hdulist.header) ) #, slices=('x', 'y', 0, 0))
    if minmax is None:
        img = ax.imshow(data[0, 0, :, :], cmap=cmap, vmax=16 * imagenoiseinfo, vmin=-6 * imagenoiseinfo)
        ax.set_title(fitsimagename + ' (noise = {:.3g} mJy/beam)'.format(imagenoiseinfo * 1e3),fontsize=6)
    else:
        img = ax.imshow(data[0, 0, :, :], cmap=cmap, vmax=minmax[1], vmin=minmax[0])
        #ax.set_title(fitsimagename, fontsize=6)
        ax.set_title(fitsimagename + ' (noise = {:.3g} mJy/beam)'.format(imagenoiseinfo * 1e3),fontsize=6)
    
    ax.grid(True)
    ax.coords[0].set_axislabel_position('b') # for some reason this needs to be hardcoded, otherwise RA gets on top axis
    ax.coords[0].set_ticks_position('bt')
    ax.coords[0].set_ticklabel_position('b') # for some reason this needs to be hardcoded, otherwise RA gets on top axis
    
    ax.set_xlabel('Right Ascension (J2000)')
    ax.set_ylabel('Declination (J2000)')
    try:
        from astropy.visualization.wcsaxes import add_beam, add_scalebar
        add_beam(ax, header=hdulist.header,  frame=True) 
    except Exception as e:
        terminal_print(f"Cannot plot beam on image, failed with error: {e}. Skipping.")

    cbar = plt.colorbar(img)
    cbar.set_label('Flux (mJy beam$^{-1}$)')
 
    try: 
        if regionfile is not None:
            import regions  # type: ignore

            # if regionfile is a string
            if isinstance(regionfile, str):
                ds9regions = regions.Regions.read(regionfile, format='ds9')
                for ds9region in ds9regions:
                    reg = ds9region.to_pixel(WCS(hdulist.header))
                    reg.plot(ax=ax, color=regioncolor, alpha=regionalpha)
            if isinstance(regionfile, list):
                for rf in regionfile:
                    ds9regions = regions.Regions.read(rf, format='ds9')
                    for ds9region in ds9regions:
                        reg = ds9region.to_pixel(WCS(hdulist.header))
                        reg.plot(ax=ax, color=regioncolor, alpha=regionalpha)        
    except Exception as e:
        terminal_print(f"Cannot overplot facets, failed with error: {e}. Skipping.")
        
    if mask is not None:
        maskdata = fits.getdata(mask)[0, 0, :, :]
        ax.contour(maskdata, colors='red', levels=[0.1 * imagenoiseinfo], filled=False, alpha=0.6, linewidths=1)

    if os.path.isfile(outplotname + '.png'):
        Path(outplotname + '.png').unlink(missing_ok=True)  # remove existing file to avoid permission issues
    plt.savefig(outplotname, dpi=450, format='png')
    plt.close()
    return


def plotimage_plain(fitsimagename, outplotname, minmax, cmap='bone', max_pixels=2500):
    """
    Write only the image pixels of a FITS image to a PNG file.

    The PNG has no axes, grid, title, colorbar, beam, regions or mask, and
    one PNG pixel corresponds to a fixed block of image pixels. Plots of the
    same field made with the same ``minmax`` therefore line up exactly, which
    the HTML report uses to blink between self-calibration cycles.

    Parameters
    ----------
    fitsimagename : str
        FITS image path.
    outplotname : str
        Output PNG path.
    minmax : sequence of float
        Display minimum and maximum.
    cmap : str, optional
        Matplotlib colormap.
    max_pixels : int, optional
        Larger images are block-averaged so neither PNG axis exceeds this size.

    Returns
    -------
    None
        The plot is written to disk.
    """
    data = np.asarray(fits.getdata(fitsimagename), dtype=np.float32)
    while data.ndim > 2:
        data = data[0]

    factor = int(np.ceil(max(data.shape) / float(max_pixels)))
    if factor > 1:
        ny = (data.shape[0] // factor) * factor
        nx = (data.shape[1] // factor) * factor
        blocks = data[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN blocks stay NaN
            data = np.nanmean(blocks, axis=(1, 3))

    if os.path.isfile(outplotname):
        Path(outplotname).unlink(missing_ok=True)  # remove existing file to avoid permission issues
    plt.imsave(outplotname, data, cmap=cmap, vmin=minmax[0], vmax=minmax[1],
               origin='lower', format='png')


def plotimage(selfcalcycle, args, stackstr='', mask=None, regionfile=None):
    """
    Plot an image using Astropy, falling back to APLpy if necessary.

    The image and reference noise image are selected from the configured
    imager and self-calibration cycle. The resulting PNG is written to the
    ``plots`` directory.

    Parameters
    ----------
    selfcalcycle : int
        Self-calibration cycle number used to select the
        image.
    args : dict
        Parsed facetselfcal options; ``imager``, ``idg``, ``imagename`` and
        ``channelsout`` select the image to plot.
    stackstr : str, optional
        Filename suffix for stacked images.
    mask : str or None, optional
        FITS clean-mask image to overlay as red
        contours. It is only drawn when it has the size of the image:
        WSClean imaging skips a mask with another size (e.g. after a
        restart with a different --imsize), so no mask was used then.
    regionfile : str or None, optional
        DS9 facet-region file to overlay
        for DDE mode.
    """
    plotpngimage, plotfitsimage, plotfitsimage000 = _plotimage_names(selfcalcycle, args, stackstr)

    if mask == 'nofitsmask' or (mask is not None and not os.path.isfile(mask)):
        mask = None
    if mask is not None and get_image_size(mask) != get_image_size(plotfitsimage):
        terminal_print('Not drawing mask ' + mask + ' in the plot: its size differs from ' + plotfitsimage
                       + ', so it was not used for this image')
        mask = None

    # find noise of image000, so we can directly compare images between selfcal cycles by having a hardcoded range
    # (a run restarted with --start>0 in a directory without the cycle-0 image uses the noise of this image)
    if not os.path.isfile(plotfitsimage000):
        plotfitsimage000 = plotfitsimage
    hdulist = fits.open(plotfitsimage000)
    imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
    plotminmax = [-3.*imagenoise, 10.*imagenoise]
    hdulist.close()

    try:
        plotimage_astropy(plotfitsimage, plotpngimage, mask, regionfile=regionfile, minmax=plotminmax)
    except Exception as e:
        terminal_print(f"Astropy plotting failed with error: {e}. Switching to aplpy.")
        plotimage_aplpy(plotfitsimage, plotpngimage, mask, plotfitsimage)

    # Same image and display range without overlays, for blinking between cycles
    plainpngimage = os.path.splitext(plotpngimage)[0] + '-plain.png'
    try:
        plotimage_plain(plotfitsimage, plainpngimage, plotminmax)
    except Exception as e:
        terminal_print(f"Plain image plot failed with error: {e}. Skipping.")

    # Image centre at full resolution, for blinking between cycles in the HTML overview
    zoompngimage = os.path.splitext(plotpngimage)[0] + ZOOM_PLOT_SUFFIX
    try:
        plotimage_zoom(plotfitsimage, zoompngimage, plotminmax)
    except Exception as e:
        terminal_print(f"Zoomed image plot failed with error: {e}. Skipping.")

    # Cutouts around the DDE directions for all cycles imaged so far
    if args.get('DDE') and os.path.isfile(FACETDIRECTIONS_PICKLE):
        try:
            plotdirectioncutouts(selfcalcycle, args, plotminmax, stackstr=stackstr)
        except Exception as e:
            terminal_print(f"Direction cutout plots failed with error: {e}. Skipping.")

    # earlier cycles imaged by an older facetselfcal version (before a restart) lack the plain/zoom plots
    backfill_plot_variants(selfcalcycle, args, stackstr=stackstr)


def backfill_plot_variants(upto_cycle, args, stackstr=''):
    """
    Make the missing plain and zoom PNGs of earlier selfcal cycles.

    Cycles imaged by an older facetselfcal version (for example before a restart with --start)
    have only the main PNG. The HTML overview needs the "-plain.png" and "-zoom.png" variants to
    blink and compare cycles. They are made here from the FITS images, if those still exist, with
    the same display range as plotimage.

    Parameters
    ----------
    upto_cycle : int
        Cycles 0 .. upto_cycle - 1 are checked.
    args : dict
        Parsed facetselfcal options (see plotimage).
    stackstr : str, optional
        Filename suffix for stacked images.
    """
    plotminmax = None
    for cycle in range(upto_cycle):
        plotpngimage, plotfitsimage, plotfitsimage000 = _plotimage_names(cycle, args, stackstr)
        if plotpngimage is None:
            return
        variants = [(os.path.splitext(plotpngimage)[0] + '-plain.png', plotimage_plain),
                    (os.path.splitext(plotpngimage)[0] + ZOOM_PLOT_SUFFIX, plotimage_zoom)]
        missing = [(png, func) for png, func in variants if not os.path.isfile(png)]
        if not missing:
            continue
        # the channel count can differ between cycles: try the image name with and without MFS
        if not os.path.isfile(plotfitsimage):
            alternative = plotfitsimage.replace('-MFS-image', '-image') if '-MFS-' in plotfitsimage \
                else plotfitsimage.replace('-image.fits', '-MFS-image.fits')
            if not os.path.isfile(alternative):
                continue
            plotfitsimage = alternative
        if plotminmax is None:
            name000 = plotfitsimage000 if os.path.isfile(plotfitsimage000) else plotfitsimage
            with fits.open(name000) as hdulist:
                imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
            plotminmax = [-3. * imagenoise, 10. * imagenoise]
        for png, func in missing:
            try:
                func(plotfitsimage, png, plotminmax)
                terminal_print('Made missing plot ' + png + ' for selfcal cycle ' + str(cycle))
            except Exception as e:
                terminal_print(f"Plot {png} failed with error: {e}. Skipping.")


def _plotimage_names(selfcalcycle, args, stackstr=''):
    """
    Return the PNG name, FITS image and cycle-0 FITS image used by ``plotimage``.

    Parameters
    ----------
    selfcalcycle : int
        Self-calibration cycle number.
    args : dict
        Parsed facetselfcal options; ``imager``, ``idg``, ``imagename`` and
        ``channelsout`` select the image.
    stackstr : str, optional
        Filename suffix for stacked images.

    Returns
    -------
    tuple of str
        ``(plotpngimage, plotfitsimage, plotfitsimage000)``.
    """
    plots_dir = 'plots'
    plotpngimage = plotfitsimage = plotfitsimage000 = None
    if args['imager'] == 'WSCLEAN':
        if args['idg']:
            plotpngimage = os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + stackstr + '.png')
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '-MFS-image.fits'
        else:
            plotpngimage = os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle).zfill(3) + stackstr + '.png')
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '-MFS-image.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '-MFS-image.fits'
        if args['imager'] == 'DDFACET':
            plotpngimage = os.path.join(plots_dir, os.path.basename(args['imagename']) + str(selfcalcycle) + '.png')
            plotfitsimage = args['imagename'] + str(selfcalcycle).zfill(3) + stackstr + '.app.restored.fits'
            plotfitsimage000 = args['imagename'] + str(0).zfill(3) + stackstr + '.app.restored.fits'

        if args['channelsout'] == 1:
            plotpngimage = plotpngimage.replace('-MFS', '').replace('-I', '')
            plotfitsimage = plotfitsimage.replace('-MFS', '').replace('-I', '')    
            plotfitsimage000 = plotfitsimage000.replace('-MFS', '').replace('-I', '')
    return plotpngimage, plotfitsimage, plotfitsimage000


def _image_plane(hdulist):
    """Return the first 2D image plane of a FITS HDU list (memory-mapped when possible)."""
    data = hdulist[0].data
    while data.ndim > 2:
        data = data[0]
    return data


def _cycle_image_noise(selfcalcycle, args, stackstr, noise_history):
    """
    Return the noise of the selfcal image of one cycle, using and filling a cache.

    Parameters
    ----------
    selfcalcycle : int
        Self-calibration cycle number.
    args : dict
        Parsed facetselfcal options (see ``_plotimage_names``).
    stackstr : str
        Filename suffix for stacked images.
    noise_history : dict
        Cache ``{(stackstr, cycle): rms}``, updated in place.

    Returns
    -------
    float or None
        The image noise, or None if the image does not exist or has no usable pixels.
    """
    key = (stackstr, selfcalcycle)
    if key in noise_history:
        return noise_history[key]
    fitsimage = _plotimage_names(selfcalcycle, args, stackstr)[1]
    if fitsimage is None or not os.path.isfile(fitsimage):
        return None
    with fits.open(fitsimage, memmap=True) as hdulist:
        data = np.asarray(_image_plane(hdulist), dtype=np.float32).ravel()
    data = data[np.isfinite(data)]
    rms = float(findrms(data)) if data.size > 0 else None
    if rms is not None and not (np.isfinite(rms) and rms > 0):
        rms = None
    noise_history[key] = rms
    return rms


def check_noise_divergence(selfcalcycle, args, stackstr, noise_history, factor):
    """
    Compare the image noise of a selfcal cycle with the lowest noise of the earlier cycles.

    A large increase means that the calibration diverged. The caller then stops
    selfcal before the solve and flagging of this cycle.

    Parameters
    ----------
    selfcalcycle : int
        Self-calibration cycle whose image was just made.
    args : dict
        Parsed facetselfcal options (see ``_plotimage_names``).
    stackstr : str
        Filename suffix for stacked images.
    noise_history : dict
        Cache ``{(stackstr, cycle): rms}``, updated in place. Earlier cycles
        missing from it (for example after a restart with ``--start``) are read
        from their FITS images if these still exist.
    factor : float or None
        Maximum allowed ratio of the noise to the lowest earlier noise.
        None or a value <= 0 switches the check off.

    Returns
    -------
    tuple
        ``(diverged, ratio, best_cycle, history)``: whether the ratio exceeds
        ``factor``, the ratio (None if it could not be computed), the earlier
        cycle with the lowest noise, and ``{cycle: rms}`` for this stack.
    """
    history = {}
    if factor is None or factor <= 0 or selfcalcycle == 0:
        return False, None, None, history
    for cycle in range(selfcalcycle + 1):
        try:
            rms = _cycle_image_noise(cycle, args, stackstr, noise_history)
        except Exception as e:  # an unreadable image should never stop selfcal
            get_logger().warning('Noise check: could not read the image of cycle %d: %s' % (cycle, e))
            rms = None
        if rms is not None:
            history[cycle] = rms
    earlier = {cycle: rms for cycle, rms in history.items() if cycle < selfcalcycle}
    if selfcalcycle not in history or not earlier:
        return False, None, None, history
    best_cycle = min(earlier, key=earlier.get)
    ratio = history[selfcalcycle] / earlier[best_cycle]
    return ratio > factor, ratio, best_cycle, history


def plotimage_zoom(fitsimagename, outplotname, minmax, cmap='bone', size=ZOOM_PLOT_SIZE):
    """
    Write the central part of a FITS image at full resolution to a PNG file.

    One PNG pixel is one image pixel, so the HTML overview can show the image
    centre at 1:1 and blink between self-calibration cycles. Images with
    fewer than 1.5 times ``size`` pixels along both axes are skipped, because
    the plain PNG of those already shows enough detail.

    Parameters
    ----------
    fitsimagename : str
        FITS image path.
    outplotname : str
        Output PNG path.
    minmax : sequence of float
        Display minimum and maximum.
    cmap : str, optional
        Matplotlib colormap.
    size : int, optional
        Width and height of the central part in image pixels.

    Returns
    -------
    bool
        True if the PNG was written, False if the image is too small.
    """
    with fits.open(fitsimagename, memmap=True) as hdulist:
        plane = _image_plane(hdulist)
        ny, nx = plane.shape
        if max(nx, ny) < 1.5 * size:
            return False
        x0 = max(0, (nx - size) // 2)
        y0 = max(0, (ny - size) // 2)
        data = np.asarray(plane[y0:y0 + size, x0:x0 + size], dtype=np.float32)

    if os.path.isfile(outplotname):
        Path(outplotname).unlink(missing_ok=True)  # remove existing file to avoid permission issues
    plt.imsave(outplotname, data, cmap=cmap, vmin=minmax[0], vmax=minmax[1],
               origin='lower', format='png')
    return True


def _cutout_box(header, ra, dec, size):
    """Return the pixel box (x0, y0) of a ``size`` x ``size`` cutout centred on (ra, dec) in degrees."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(header).celestial
    x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(round(x)) - size // 2, int(round(y)) - size // 2


def _read_cutout(plane, x0, y0, size):
    """Return a ``size`` x ``size`` cutout, NaN-padded where it extends beyond the image."""
    ny, nx = plane.shape
    cutout = np.full((size, size), np.nan, dtype=np.float32)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + size, nx), min(y0 + size, ny)
    if xa >= xb or ya >= yb:
        return None
    cutout[ya - y0:yb - y0, xa - x0:xb - x0] = plane[ya:yb, xa:xb]
    return cutout


def plot_direction_cutouts(images, directions, outdir, minmax, manifest_name='cutouts.json',
                           frame_prefix='', cmap='bone', beam_factor=DIRECTION_CUTOUT_BEAMS,
                           min_pixels=64, max_pixels=400):
    """
    Write PNG cutouts around directions for a series of self-calibration images.

    One cutout is made per direction and image, all with the same linear
    display range ``minmax`` (the range of the full image plots), so cutouts
    of different directions and cycles can be compared and animated in the
    HTML overview. A direction keeps the cutout size it got when it was first
    plotted. The sizes, display range and frames are recorded in a JSON
    manifest in ``outdir``. Existing cutouts are only remade when their FITS
    image is newer or the display range changed.

    Parameters
    ----------
    images : dict
        Self-calibration cycle (int) -> FITS image path.
    directions : sequence of (float, float)
        Direction RA and Dec in degrees.
    outdir : str
        Output directory for the PNGs and the manifest.
    minmax : sequence of float
        Display minimum and maximum (Jy/beam).
    manifest_name : str, optional
        Manifest filename in ``outdir``.
    frame_prefix : str, optional
        Prefix of the PNG filenames.
    cmap : str, optional
        Matplotlib colormap.
    beam_factor : float, optional
        Cutout width in units of the restoring-beam major axis.
    min_pixels, max_pixels : int, optional
        Limits on the cutout width in image pixels.

    Returns
    -------
    dict
        The manifest.
    """
    os.makedirs(outdir, exist_ok=True)
    manifest_path = os.path.join(outdir, manifest_name)
    vmin, vmax = float(minmax[0]), float(minmax[1])
    manifest = {'directions': []}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            pass
    if (manifest.get('stretch') != 'linear'
            or not np.allclose([manifest.get('vmin', np.nan), manifest.get('vmax', np.nan)],
                               [vmin, vmax], rtol=1e-6, atol=0.0)):
        # cutouts made with another display range: remake them all
        manifest = {'directions': [
            {key: entry[key] for key in ('index', 'ra', 'dec', 'size', 'pixel_arcsec') if key in entry}
            for entry in manifest.get('directions', [])
        ]}
    manifest.update({'stretch': 'linear', 'vmin': vmin, 'vmax': vmax})
    entries = manifest.setdefault('directions', [])

    def find_entry(ra, dec):
        for entry in entries:
            dra = (entry['ra'] - ra + 180.0) % 360.0 - 180.0
            if np.hypot(dra * np.cos(np.radians(dec)), entry['dec'] - dec) < 1.0 / 3600.0:
                return entry
        return None

    cycles = sorted(images)
    if not cycles:
        return manifest
    latest = cycles[-1]
    for ra, dec in directions:
        ra, dec = float(ra) % 360.0, float(dec)
        entry = find_entry(ra, dec)
        if entry is None:
            entry = {'index': len(entries), 'ra': ra, 'dec': dec, 'frames': {}}
            entries.append(entry)
        frames = entry.setdefault('frames', {})
        # The latest cycle sets the cutout size of a new direction
        for cycle in [latest] + [c for c in cycles if c != latest]:
            fitsname = images[cycle]
            key = str(cycle).zfill(3)
            pngname = '{}dir{:02d}_{}.png'.format(frame_prefix, entry['index'], key)
            pngpath = os.path.join(outdir, pngname)
            if (frames.get(key) == pngname and os.path.isfile(pngpath)
                    and os.path.getmtime(pngpath) >= os.path.getmtime(fitsname)):
                continue
            with fits.open(fitsname, memmap=True) as hdulist:
                header = hdulist[0].header
                if 'size' not in entry:
                    pixel = abs(float(header['CDELT2']))
                    bmaj = float(header.get('BMAJ', 0.0)) or 10.0 * pixel
                    size = int(np.clip(beam_factor * bmaj / pixel, min_pixels, max_pixels))
                    entry['size'] = size + size % 2
                    entry['pixel_arcsec'] = pixel * 3600.0
                box = _cutout_box(header, ra, dec, entry['size'])
                cutout = None if box is None else _read_cutout(_image_plane(hdulist), box[0], box[1], entry['size'])
            if cutout is None or not np.any(np.isfinite(cutout)):
                continue
            if os.path.isfile(pngpath):
                Path(pngpath).unlink(missing_ok=True)
            plt.imsave(pngpath, cutout, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', format='png')
            frames[key] = pngname

    tmp_path = manifest_path + '.tmp'
    with open(tmp_path, 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    os.replace(tmp_path, manifest_path)
    return manifest


def plotdirectioncutouts(selfcalcycle, args, minmax, stackstr=''):
    """
    Write cutouts around the current DDE directions for cycles 0 to ``selfcalcycle``.

    The directions are read from ``facet_regions/facetdirections.p``. Earlier
    cycles are included so a direction added in a later cycle also shows how
    it looked before it was calibrated. The PNGs and the manifest go to
    ``plots/direction_cutouts`` (see ``plot_direction_cutouts``).

    Parameters
    ----------
    selfcalcycle : int
        Current self-calibration cycle.
    args : dict
        Parsed facetselfcal options.
    minmax : sequence of float
        Display minimum and maximum (Jy/beam), the range of the image plots.
    stackstr : str, optional
        Filename suffix for stacked images.
    """
    with open(FACETDIRECTIONS_PICKLE, 'rb') as handle:
        positions = np.degrees(np.asarray(pickle.load(handle), dtype=float))
    images = {}
    for cycle in range(selfcalcycle + 1):
        fitsname = _plotimage_names(cycle, args, stackstr)[1]
        if fitsname and os.path.isfile(fitsname):
            images[cycle] = fitsname
    manifest_name = 'cutouts{}.json'.format(stackstr)
    if selfcalcycle == 0:  # a new run: start from an empty manifest
        Path(os.path.join(DIRECTION_CUTOUT_DIR, manifest_name)).unlink(missing_ok=True)
    plot_direction_cutouts(images, positions.reshape(-1, 2), DIRECTION_CUTOUT_DIR, minmax,
                           manifest_name=manifest_name,
                           frame_prefix='{}_'.format(stackstr.strip('_')) if stackstr else '')


def plotimage_aplpy(fitsimagename, outplotname, mask=None, rmsnoiseimage=None):
    """
    Plot a FITS image with APLpy and save it as a PNG.

    The image RMS is estimated from ``rmsnoiseimage`` when supplied, or from
    the plotted image otherwise. The output filename is formed by appending
    ``.png`` to ``outplotname``.

    Parameters
    ----------
    fitsimagename : str
        Path to the FITS image to plot.
    outplotname : str
        Output filename without the ``.png`` suffix.
    mask : str or None, optional
        FITS clean-mask image to overlay as red
        contours.
    rmsnoiseimage : str or None, optional
        FITS image from which to
        estimate the plotting noise range.
    """
    import aplpy

    # image noise for plotting
    if rmsnoiseimage is None:
        hdulist = fits.open(fitsimagename)
    else:
        hdulist = fits.open(rmsnoiseimage)
    imagenoise = findrms(np.ndarray.flatten(hdulist[0].data))
    hdulist.close()

    # image noise info
    hdulist = fits.open(fitsimagename)
    imagenoiseinfo = findrms(np.ndarray.flatten(hdulist[0].data))
    get_logger().info(fitsimagename + ' Max image: ' + str(np.max(np.ndarray.flatten(hdulist[0].data))))
    get_logger().info(fitsimagename + ' Min image: ' + str(np.min(np.ndarray.flatten(hdulist[0].data))))
    hdulist.close()

    f = aplpy.FITSFigure(fitsimagename, slices=[0, 0])
    f.show_colorscale(vmax=16 * imagenoise, vmin=-6 * imagenoise, cmap='bone')
    f.set_title(fitsimagename + ' (noise = {:.3g} mJy/beam)'.format(imagenoiseinfo * 1e3))
    try:  # to work around an aplpy error
        f.add_beam()
        f.beam.set_frame(True)
        f.beam.set_color('white')
        f.beam.set_edgecolor('black')
        f.beam.set_linewidth(1.)
    except:
        pass

    f.add_grid()
    f.grid.set_color('white')
    f.grid.set_alpha(0.5)
    f.grid.set_linewidth(0.2)
    f.add_colorbar()
    f.colorbar.set_axis_label_text('Flux (mJy beam$^{-1}$)')
    if mask is not None:
        try:
            f.show_contour(mask, colors='red', levels=[0.1 * imagenoise], filled=False, smooth=1, alpha=0.6,
                           linewidths=1)
        except:
            pass
    Path(outplotname + '.png').unlink(missing_ok=True)
    f.save(outplotname, dpi=120, format='png')
    get_logger().info(fitsimagename + ' RMS noise: ' + str(imagenoiseinfo))
    return


def create_Ateam_seperation_plots(mslist, start=0):
    """
    Create A-team and Sun, Moon, Jupiter separation plots.

    Parameters
    ----------
    mslist : list of str
        List of Measurement Sets.
    start : int, optional
        Starting self-calibration cycle.
    """
    if start != 0:
        return
    os.makedirs('plots', exist_ok=True)
    for ms in mslist:
        outputname = 'plots/Ateam_' + os.path.basename(ms) + '.png'
        try:
            run(f'python {submodpath}/check_Ateam_separation_mod.py --outputimage={outputname} {ms}')
        except Exception:
            terminal_print(f"check_Ateam_separation_mod.py does not exist")
    return
