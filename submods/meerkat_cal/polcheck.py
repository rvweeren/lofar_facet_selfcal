"""Check of the polarisation calibration on the polarisation calibrator(s): the calibrated scans (all
corrections as for the targets, in the sky frame) are imaged in Stokes I, Q, U and V in frequency
chunks with WSClean, the integrated I, Q, U and V of the calibrator are measured in an aperture shaped
like the restoring beam (centred on the Stokes I peak, see _beam_aperture), the polarisation fraction,
EVPA and V/I are compared with the 3C286/3C138 model (which is for the whole source), and the RM is
fitted.

The cross-hand phase was solved on the same data, so the EVPA agreement is not an independent test of
the absolute angle. It does test the frequency structure, the ionospheric RM over the scans (the
solution is one per observation), the leakage (V/I and the polarisation fraction) and the parallactic
angle correction."""
import glob
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from casacore.tables import table  # noqa: E402

from submods.applycal import applycal_dp3_steps  # noqa: E402
from submods.calibrator_models import POLARISED_CALIBRATOR_MODELS  # noqa: E402
from submods.meerkat_cal.prepare import DYSCO_ARGS, nchan_divisible  # noqa: E402
from submods.ms_fixes import fix_uvw  # noqa: E402
from submods.runtime_utils import run, terminal_print  # noqa: E402

C = 299792458.
# warn above these values
EVPA_TOLERANCE = {'UHF': 5., 'L': 3.}  # degrees, median |measured - model|; S-band uses the L value
VFRAC_TOLERANCE = 0.01
PFRAC_TOLERANCE = 0.2  # relative
# Flux aperture: the restoring-beam ellipse with semi-axes APERTURE_FWHM times the beam FWHM (it holds
# 1 - 2**(-4 APERTURE_FWHM**2) = 99.8% of the flux of a point source), each axis widened by SOURCE_EXTENT
# arcsec for the arcsecond-scale structure of the calibrators, which the VLA resolves in its extended
# configurations. The integrated values are the aperture sum divided by the beam area in pixels.
APERTURE_FWHM = 1.5
SOURCE_EXTENT = 3.  # arcsec


def calibrated_scan(ms, h5chain, out, freqstep=4, nthreads=24, dysco=True):
    """
    Write a calibrated copy (sky frame) of a polarisation calibrator scan.

    Parameters
    ----------
    ms : str
        Prepared calibrator scan (feeds swapped, DATA column).
    h5chain : list of str
        h5parms to apply, outermost Jones term first (time-variable gains ... cross-hand phase, then the
        ionospheric RM).
    out : str
        Output MS.
    freqstep : int, optional
        Frequency averaging. Default 4.
    nthreads : int, optional
        DP3 threads. Default 24.
    dysco : bool, optional
        Dysco compression of the output data (DYSCO_ARGS, exact weights). Default True.
    """
    if os.path.isdir(out):
        shutil.rmtree(out)
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        nchan = len(t.getcol('CHAN_FREQ')[0])
    nread = nchan_divisible(nchan, freqstep)
    acargs, nac = applycal_dp3_steps(ms, h5chain, missingantennabehavior='flag', interpolation='linear',
                                     updateweights=True)  # the archive weights follow the raw gains
    steps = ['ac%d' % i for i in range(nac)] + ['pa', 'avg']
    cmd = (f'DP3 numthreads={nthreads} msin={ms} msin.datacolumn=DATA msin.weightcolumn=WEIGHT_SPECTRUM msout={out} '
           + (f'msin.nchan={nread} ' if nread != nchan else '')
           + 'msout.uvwcompression=False msout.antennacompression=False msout.scalarflags=False '  # fix_uvw rewrites UVW
           + (DYSCO_ARGS if dysco else '') + acargs + 'pa.type=PythonDPPP pa.python.module=parang pa.python.class=ParAng pa.mode=correct '
           f'pa.receptorangle=0 avg.type=averager avg.freqstep={freqstep} steps=[' + ','.join(steps) + ']')
    terminal_print('Calibrated polarisation calibrator scan:', cmd)
    run(cmd, log=True)
    fix_uvw([out])  # time slots inserted by DP3


def image_iquv(mslist, prefix, nchunks=8, pixelscale=1.8, imsize=1024, nthreads=24):
    """
    WSClean Stokes I, Q, U, V images in frequency chunks (joined polarisation cleaning).

    Parameters
    ----------
    mslist : list of str
        Calibrated MSs (sky frame, DATA column).
    prefix : str
        WSClean -name.
    nchunks : int, optional
        Number of frequency chunks. Default 8.
    pixelscale : float, optional
        Pixel size (arcsec). Default 1.8.
    imsize : int, optional
        Image size (pixels). Default 1024.
    nthreads : int, optional
        Threads. Default 24.
    """
    cmd = (f'wsclean -name {prefix} -pol IQUV -join-polarizations -channels-out {nchunks} -size {imsize} {imsize} '
           f'-scale {pixelscale}arcsec -niter 10000 -mgain 0.8 -auto-threshold 3 -auto-mask 6 -weight briggs 0 '
           f'-minuv-l 100 -no-update-model-required -data-column DATA -j {nthreads} ' + ' '.join(mslist))
    terminal_print('Polarisation calibrator images:', cmd)
    run(cmd, log=True)


def _beam_aperture(header, shape, pix, extent=SOURCE_EXTENT):
    """
    Elliptical aperture shaped like the restoring beam (see APERTURE_FWHM) around a pixel.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Image header with the restoring beam (BMAJ, BMIN, BPA in degrees) and CDELT1/2.
    shape : tuple of int
        Image shape (ny, nx).
    pix : tuple of int
        Aperture centre (y, x).
    extent : float, optional
        Widening of each axis (arcsec). Default SOURCE_EXTENT.

    Returns
    -------
    (numpy.ndarray of bool, float)
        The aperture and the beam area in pixels (aperture sum / beam area = flux density).
    """
    bmaj, bmin = header['BMAJ'] * 3600., header['BMIN'] * 3600.
    bpa = np.radians(header.get('BPA', 0.))
    dx, dy = header['CDELT1'] * 3600., header['CDELT2'] * 3600.
    yy, xx = np.indices(shape)
    east, north = (xx - pix[1]) * dx, (yy - pix[0]) * dy  # arcsec (CDELT1 < 0: RA increases to the left)
    along = east * np.sin(bpa) + north * np.cos(bpa)  # major axis, position angle east of north
    across = east * np.cos(bpa) - north * np.sin(bpa)
    inside = ((along / (APERTURE_FWHM * bmaj + extent)) ** 2
              + (across / (APERTURE_FWHM * bmin + extent)) ** 2) <= 1.
    return inside, np.pi * bmaj * bmin / (4. * np.log(2.)) / abs(dx * dy)


def _image_flux(fitsfile, pix=None, box=20):
    """
    Integrated flux density of the calibrator in a beam-shaped aperture (_beam_aperture) and its noise.

    Parameters
    ----------
    fitsfile : str
        Restored WSClean image.
    pix : tuple of int or None, optional
        Aperture centre (y, x). Default None: the Stokes I peak within box pixels of the image centre.
    box : int, optional
        Half size of the peak search box (pixels). Default 20.

    Returns
    -------
    (float, float, tuple or None, float, float)
        Flux density (Jy); its noise (Jy: the image rms, from the top and bottom eighth of the image, times
        the square root of the number of beams in the aperture); the aperture centre; the frequency and
        bandwidth (Hz). NaN and None for a chunk without unflagged data.
    """
    with fits.open(fitsfile) as h:
        data = np.squeeze(h[0].data)
        header = h[0].header
    freq, bandwidth = header.get('CRVAL3'), header.get('CDELT3')
    ny, nx = data.shape
    if pix is None:
        cy, cx = ny // 2, nx // 2
        sub = data[cy - box:cy + box, cx - box:cx + box]
        if not np.isfinite(sub).any():  # a chunk without unflagged data (WSClean writes NaN)
            return np.nan, np.nan, None, freq, bandwidth
        iy, ix = np.unravel_index(np.nanargmax(sub), sub.shape)
        pix = (cy - box + iy, cx - box + ix)
    if not header.get('BMAJ') or not header.get('BMIN'):  # no restoring beam (no data in the chunk)
        return np.nan, np.nan, None, freq, bandwidth
    inside, beam = _beam_aperture(header, data.shape, pix)
    edge = np.concatenate([data[:ny // 8].ravel(), data[-ny // 8:].ravel()])
    rms = 1.4826 * np.nanmedian(np.abs(edge - np.nanmedian(edge)))
    flux = np.nansum(data[inside]) / beam
    return float(flux), float(rms * np.sqrt(inside.sum() / beam)), pix, freq, bandwidth


def model_polarisation(source, freq, bandwidth, nsample=32, rm=None):
    """
    Model polarisation fraction and EVPA averaged (as Q + iU, Stokes I weighting ignored) over a chunk.

    Parameters
    ----------
    rm : array_like or None, optional
        Ionospheric RM (rad/m^2) at the times of the observation: the model is rotated by -RM lambda^2
        (averaged over these times), i.e. it gets the ionospheric Faraday rotation correction that is
        applied to the data. Default None (intrinsic model).

    Returns
    -------
    (float, float)
        Polarisation fraction and EVPA (degrees).
    """
    nu = (freq + (np.arange(nsample) + 0.5 - nsample / 2) * bandwidth / nsample) / 1e9
    calc_pfrac, calc_evpa = POLARISED_CALIBRATOR_MODELS[source]
    p = calc_pfrac(nu) * np.exp(2j * calc_evpa(nu))
    if rm is not None:
        lam2 = (C / (nu * 1e9)) ** 2
        p = p * np.mean(np.exp(-2j * np.outer(np.atleast_1d(rm), lam2)), axis=0)
    pm = np.mean(p)
    return float(np.abs(pm)), float(np.degrees(0.5 * np.angle(pm)))


def measure(prefix, source, nchunks, rm=None):
    """
    Measure the integrated I, Q, U, V of the calibrator per chunk (_image_flux, the same aperture in all
    four Stokes images) and compare with the model.

    Parameters
    ----------
    rm : dict or None, optional
        Ionospheric RM (the json of make_rm_h5: source, times, rm) that was corrected in the data. If
        given, each chunk also gets rm_correction, the EVPA rotation of that correction (-RM lambda^2,
        averaged over the times and the chunk), evpa_uncorr, the measured EVPA with the correction
        undone, and dev_evpa_uncorr, its difference from the model; comparing dev_evpa_uncorr with
        dev_evpa shows whether the correction brings the data closer to the model.

    Returns
    -------
    dict
        Per chunk lists (freq_mhz, I and I_rms: flux density and its noise in Jy, pfrac, pfrac_model, evpa, evpa_model, dev_evpa, vfrac, ...), the
        fitted RM and intrinsic angle, summary numbers and image file names.
    """
    rows = []
    for c in range(nchunks):
        tag = f'{prefix}-{c:04d}'
        if not os.path.isfile(f'{tag}-I-image.fits'):
            continue
        i, irms, pix, freq, bw = _image_flux(f'{tag}-I-image.fits')
        if pix is None:
            continue
        q, qrms = _image_flux(f'{tag}-Q-image.fits', pix)[:2]
        u, urms = _image_flux(f'{tag}-U-image.fits', pix)[:2]
        v, vrms = _image_flux(f'{tag}-V-image.fits', pix)[:2]
        if not np.isfinite(i) or i <= 5 * irms:
            continue
        sigma = 0.5 * (qrms + urms)
        ppol = np.sqrt(max(q * q + u * u - sigma * sigma, 0.))  # Ricean bias correction
        pm, evpam = model_polarisation(source, freq, bw)
        evpa = float(np.degrees(0.5 * np.arctan2(u, q)))
        rows.append(dict(freq_mhz=freq / 1e6, I=i, I_rms=irms, Q=q, U=u, V=v, pfrac=ppol / i,
                         pfrac_err=sigma / i, pfrac_model=pm, evpa=evpa,
                         evpa_err=float(np.degrees(0.5 * sigma / max(ppol, 1e-12))), evpa_model=evpam,
                         dev_evpa=float((evpa - evpam + 90.) % 180. - 90.), vfrac=v / i))
        if rm is not None:
            evpac = model_polarisation(source, freq, bw, rm=rm['rm'])[1]
            correction = (evpac - evpam + 90.) % 180. - 90.
            rows[-1].update(rm_correction=correction, evpa_uncorr=evpa - correction,
                            dev_evpa_uncorr=float((evpa - correction - evpam + 90.) % 180. - 90.))
    result = dict(source=source, chunks=rows, images=sorted(glob.glob(prefix + '-[0-9][0-9][0-9][0-9]-[IP]*-image.fits')))
    if rm is not None:
        result.update(rm_ion=float(np.mean(rm['rm'])), rm_ion_source=rm.get('source', ''))
    if len(rows) >= 2:
        lam2 = np.array([(C / (r['freq_mhz'] * 1e6)) ** 2 for r in rows])
        evpa = np.radians([r['evpa'] for r in rows])
        # unwrap the EVPA (period 180 deg) along lambda^2 before the fit
        evpa = 0.5 * np.unwrap(2 * evpa)
        w = 1. / np.maximum(np.radians([r['evpa_err'] for r in rows]), np.radians(0.1)) ** 2
        A = np.vstack([lam2, np.ones_like(lam2)]).T
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], evpa * np.sqrt(w), rcond=None)
        dev = np.radians([r['dev_evpa'] for r in rows])
        cdev, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], dev * np.sqrt(w), rcond=None)
        result.update(rm=float(coef[0]), evpa0_deg=float(np.degrees(coef[1])), residual_rm=float(cdev[0]))
    if rows:
        if rm is not None:
            result.update(median_abs_dev_evpa_uncorr=float(np.median([abs(r['dev_evpa_uncorr']) for r in rows])))
        result.update(median_abs_dev_evpa=float(np.median([abs(r['dev_evpa']) for r in rows])),
                      median_abs_vfrac=float(np.median([abs(r['vfrac']) for r in rows])),
                      median_rel_dev_pfrac=float(np.median([abs(r['pfrac'] / r['pfrac_model'] - 1.)
                                                            for r in rows if r['pfrac_model'] > 0])))
    return result


def verdict(result, band):
    """Warnings for a polcheck result (empty list: pass)."""
    warnings = []
    if not result['chunks']:
        return ['no chunk with a significant Stokes I detection']
    tol = EVPA_TOLERANCE.get(band, EVPA_TOLERANCE['L'])
    if result['median_abs_dev_evpa'] > tol:
        warnings.append(f"EVPA differs from the model by {result['median_abs_dev_evpa']:.1f} deg (median, tolerance {tol})")
    if result['median_abs_vfrac'] > VFRAC_TOLERANCE:
        warnings.append(f"|V/I| = {100 * result['median_abs_vfrac']:.2f}% (median, tolerance {100 * VFRAC_TOLERANCE:.0f}%)")
    if result['median_rel_dev_pfrac'] > PFRAC_TOLERANCE:
        warnings.append(f"polarisation fraction differs from the model by {100 * result['median_rel_dev_pfrac']:.0f}% "
                        f"(median, tolerance {100 * PFRAC_TOLERANCE:.0f}%)")
    return warnings


def plot(result, outpng):
    """Polarisation fraction, EVPA and V/I versus frequency, measured and model."""
    rows = result['chunks']
    if not rows:
        return
    f = np.array([r['freq_mhz'] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].errorbar(f, [100 * r['pfrac'] for r in rows], [100 * r['pfrac_err'] for r in rows], fmt='o', label='measured')
    axs[0].plot(f, [100 * r['pfrac_model'] for r in rows], 's-', mfc='none', label='model')
    axs[0].set_ylabel('polarisation fraction (%)')
    axs[0].legend()
    # within 90 deg of the model (the EVPA is defined modulo 180 deg)
    corrected = 'measured, corrected' if 'rm_ion' in result else 'measured'
    axs[1].errorbar(f, [r['evpa_model'] + r['dev_evpa'] for r in rows], [r['evpa_err'] for r in rows], fmt='o',
                    label=corrected)
    if 'rm_ion' in result:
        axs[1].errorbar(f, [r['evpa_model'] + r['dev_evpa_uncorr'] for r in rows], [r['evpa_err'] for r in rows],
                        fmt='o', mfc='none', label=f"measured, uncorrected (without the {result['rm_ion_source']} "
                        f"correction, mean RM {result['rm_ion']:.2f})")
    axs[1].plot(f, [r['evpa_model'] for r in rows], 's-', mfc='none', label='model')
    axs[1].set_ylabel('EVPA (deg)')
    axs[1].legend()
    axs[2].plot(f, [100 * r['vfrac'] for r in rows], 'o')
    axs[2].axhline(0, color='k', lw=0.5)
    axs[2].set_ylabel('V/I (%)')
    axs[2].set_xlabel('frequency (MHz)')
    axs[0].set_title(f"{result['source']}: median |dEVPA| {result.get('median_abs_dev_evpa', np.nan):.1f} deg"
                     + (f", RM {result['rm']:.2f} rad/m$^2$" if 'rm' in result else ''))
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)
    plt.close(fig)


def thumbnails(prefix, nchunks, outdir):
    """PNG thumbnails of Stokes I and P = |Q + iU| per chunk (centre 256 pixels)."""
    os.makedirs(outdir, exist_ok=True)
    out = []
    for c in range(nchunks):
        tag = f'{prefix}-{c:04d}'
        if not os.path.isfile(f'{tag}-I-image.fits'):
            continue
        imgs = {s: np.squeeze(fits.getdata(f'{tag}-{s}-image.fits')) for s in 'IQU'}
        ny, nx = imgs['I'].shape
        sl = (slice(ny // 2 - 128, ny // 2 + 128), slice(nx // 2 - 128, nx // 2 + 128))
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax, (title, img) in zip(axs, (('I', imgs['I']), ('P', np.hypot(imgs['Q'], imgs['U'])))):
            ax.imshow(img[sl], origin='lower', cmap='magma')
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        png = os.path.join(outdir, os.path.basename(tag) + '.png')
        fig.savefig(png, dpi=80)
        plt.close(fig)
        out.append(png)
    return out
