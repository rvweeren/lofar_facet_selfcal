#!/usr/bin/env python3
"""
Convert the VLA calibrator model images used by CASA setjy into DP3 skymodels.

Reads the clean-component model images <source>_<band>.im from casa-data
(nrao/VLA/CalModels) and the Perley-Butler 2017 coefficient table
(nrao/VLA/standards/PerleyButler2017Coeffs), and writes

  facetselfcal/data/setjy/<source>_<band>.skymodel    (DP3 skymodel format)
  facetselfcal/data/setjy/PerleyButler2017Coeffs.json

Every image pixel that is kept becomes a POINT component. The component fluxes are
fractions of the total, multiplied by the Perley-Butler 2017 spectrum
log10(S/Jy) = a0 + a1*log10(nu/GHz) + a2*log10(nu/GHz)**2 + ..., which DP3 represents
exactly with ReferenceFrequency=1e9 and LogarithmicSI=true. This is what CASA setjy
does with a model image and a flux density standard (scalebychan=True): the image
structure is kept and its total flux is scaled to the standard in every channel.
The skymodels use the coefficients of the most recent epoch in the table; for the
time-variable sources (3C48, 3C138, 3C147) setjy_dp3 in calibrator_models.py switches
to the epoch nearest to the observation, using the JSON file.

This only needs to be run when NRAO updates the models; the output is part of the
repository. Run it in a container with python-casacore and the casadata package, e.g.
  singularity exec flocs.sif python submods/make_setjy_skymodels.py
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import re

import astropy.units as units
import numpy as np
from astropy.coordinates import Angle
from casacore.images import image  # type: ignore
from casacore.tables import table  # type: ignore

SOURCES = ['3C48', '3C138', '3C147', '3C286']
BANDS = ['L', 'S', 'C', 'X', 'U', 'K', 'A', 'Q']
HEADER = ('Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, '
          'MajorAxis, MinorAxis, Orientation\n')


def default_casadata_path():
    """Return the nrao/VLA directory of the casadata package, or None if it is not installed."""
    spec = importlib.util.find_spec('casadata')
    if spec is None or spec.origin is None:
        return None
    return os.path.join(os.path.dirname(spec.origin), '__data__', 'nrao', 'VLA')


def read_pb2017_coeffs(vladir):
    """
    Read the Perley-Butler 2017 coefficients of the setjy calibrators.

    Parameters
    ----------
    vladir : str
        Path to the nrao/VLA directory of casa-data.

    Returns
    -------
    dict
        Epochs (decimal years), coefficients per source and epoch, and the table keywords.
    """
    with table(os.path.join(vladir, 'standards', 'PerleyButler2017Coeffs'), ack=False) as t:
        keywords = {k: str(v) for k, v in t.getkeywords().items()}
        epochs = [round(float(e), 4) for e in t.getcol('Epoch')]
        coeffs = {}
        for source in SOURCES:
            c = t.getcol(source + '_coeffs')
            # stored as float32, round to the published precision
            coeffs[source] = [[round(float(v), 6) for v in row] for row in c]
    return {'table': 'nrao/VLA/standards/PerleyButler2017Coeffs', 'keywords': keywords,
            'epochs': epochs, 'coeffs': coeffs}


def image_components(imagename, fluxfraction=0.999, maxcomponents=1000):
    """
    Select the brightest pixels of a model image and return their positions and flux fractions.

    Parameters
    ----------
    imagename : str
        CASA model image (Jy/pixel, Stokes I, one channel).
    fluxfraction : float, optional
        Keep the smallest set of pixels that holds this fraction of the summed absolute flux.
    maxcomponents : int, optional
        Maximum number of components.

    Returns
    -------
    ra, dec : np.ndarray
        Component positions in radians.
    fraction : np.ndarray
        Component flux divided by the total flux of the kept components (sums to 1).
    stats : dict
        Number of pixels and fraction of absolute and net flux kept.
    """
    im = image(imagename)
    data = im.getdata()[0, 0]  # axes: spectral, stokes, y (dec), x (ra)
    flat = data.ravel().astype(np.float64)
    order = np.argsort(np.abs(flat))[::-1]
    cumulative = np.cumsum(np.abs(flat[order])) / np.sum(np.abs(flat))
    ncomp = min(int(np.searchsorted(cumulative, fluxfraction)) + 1, maxcomponents)
    keep = order[:ncomp]
    iy, ix = np.unravel_index(keep, data.shape)
    ra = np.empty(ncomp)
    dec = np.empty(ncomp)
    for i in range(ncomp):
        # toworld takes and returns values in numpy axis order: freq, stokes, dec, ra
        world = im.toworld([0, 0, int(iy[i]), int(ix[i])])
        dec[i], ra[i] = world[2], world[3]
    stats = {'npix': flat.size, 'ncomp': ncomp, 'absflux_kept': float(cumulative[ncomp - 1]),
             'netflux_kept': float(flat[keep].sum() / flat.sum())}
    return ra, dec, flat[keep] / flat[keep].sum(), stats


def write_skymodel(outfile, source, band, ra, dec, fraction, coeffs, epoch, stats, casadata_version):
    """Write the components as a DP3 skymodel with the Perley-Butler 2017 spectrum."""
    spectralindex = '[' + ', '.join(f'{c:.6g}' for c in coeffs[1:]) + ']'
    with open(outfile, 'w') as f:
        f.write(HEADER)
        f.write(f'# {source} {band}-band model for setjy_dp3 (facetselfcal), made with make_setjy_skymodels.py\n')
        f.write(f'# Structure: casa-data nrao/VLA/CalModels/{source}_{band}.im (casadata {casadata_version}), '
                f'{stats["ncomp"]} of {stats["npix"]} pixels, holding {100 * stats["absflux_kept"]:.3f}% of the '
                f'absolute flux; component fluxes rescaled to the total\n')
        f.write(f'# Spectrum: Perley & Butler 2017 (ApJS 230, 7), coefficients of epoch {epoch}: '
                f'log10(S/Jy) = sum_i a_i log10(nu/GHz)^i, a = {list(coeffs)}\n')
        for i in range(len(ra)):
            rastr = Angle(ra[i] * units.rad).wrap_at(360 * units.deg).to_string(unit=units.hourangle, sep=':', precision=6, pad=True)
            decstr = Angle(dec[i] * units.rad).to_string(unit=units.deg, sep='.', precision=5, pad=True, alwayssign=True)
            flux = fraction[i] * 10**coeffs[0]
            f.write(f'{source}_{band}_{i:04d}, POINT, {rastr}, {decstr}, {flux:.8g}, {spectralindex}, true, 1000000000.0,,,\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--casadata', default=default_casadata_path(),
                        help='Path to the nrao/VLA directory of casa-data. Default: from the casadata Python package.')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         'facetselfcal', 'data', 'setjy'),
                        help='Output directory. Default: facetselfcal/data/setjy in this repository.')
    parser.add_argument('--fluxfraction', type=float, default=0.999,
                        help='Fraction of the absolute image flux the kept components must hold. Default: 0.999.')
    parser.add_argument('--maxcomponents', type=int, default=1000,
                        help='Maximum number of components per model. Default: 1000.')
    args = parser.parse_args()
    if args.casadata is None or not os.path.isdir(args.casadata):
        raise SystemExit('casa-data nrao/VLA directory not found, give it with --casadata')
    try:
        casadata_version = importlib.metadata.version('casadata')
    except importlib.metadata.PackageNotFoundError:
        casadata_version = 'unknown'

    os.makedirs(args.outdir, exist_ok=True)
    pb2017 = read_pb2017_coeffs(args.casadata)
    pb2017['casadata_version'] = casadata_version
    # one list of numbers per line, so that each epoch's coefficients are on a single line
    text = re.sub(r'\[\s+([-0-9.,\s]+?)\s+\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']',
                  json.dumps(pb2017, indent=1))
    with open(os.path.join(args.outdir, 'PerleyButler2017Coeffs.json'), 'w') as f:
        f.write(text + '\n')

    epoch = pb2017['epochs'][-1]
    for source in SOURCES:
        coeffs = pb2017['coeffs'][source][-1]
        while len(coeffs) > 1 and coeffs[-1] == 0.0:  # drop the zero padding of the table
            coeffs = coeffs[:-1]
        for band in BANDS:
            ra, dec, fraction, stats = image_components(os.path.join(args.casadata, 'CalModels', f'{source}_{band}.im'),
                                                        args.fluxfraction, args.maxcomponents)
            outfile = os.path.join(args.outdir, f'{source}_{band}.skymodel')
            write_skymodel(outfile, source, band, ra, dec, fraction, coeffs, epoch, stats, casadata_version)
            print(f'{outfile}: {stats["ncomp"]} components, {100 * stats["absflux_kept"]:.3f}% of |flux|, '
                  f'{100 * stats["netflux_kept"]:.3f}% of net flux')


if __name__ == '__main__':
    main()
