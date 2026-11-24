"""Solve weights from the imaging weights (--solve-imaging-weights).

WSClean -store-imaging-weights writes IMAGING_WEIGHT_SPECTRUM: the factor that the weighting
(Briggs/uniform, uv range, tapers) applies on top of the noise weights, 1 for natural weighting.
The solve weights are the noise weights times that factor, so that the self-calibration fits the
data with the same relative weights as the image.
"""
import numpy as np
from casacore.tables import table, makecoldesc, maketabdesc

IMAGING_SOLVE_WEIGHT_COLUMN = 'WEIGHT_SPECTRUM_IMAGING'
SIGNATURE_KEYWORD = 'FACETSELFCAL_IMAGING_WEIGHTS'


def noise_weight_column(colnames):
    """The noise-weight column: WEIGHT_SPECTRUM_SOLVE for LoTSS-DR2 products, else WEIGHT_SPECTRUM."""
    return 'WEIGHT_SPECTRUM_SOLVE' if 'WEIGHT_SPECTRUM_SOLVE' in colnames else 'WEIGHT_SPECTRUM'


def imaging_weights_signature(ms, colname=IMAGING_SOLVE_WEIGHT_COLUMN):
    """Weighting the solve-weight column of ``ms`` was made with, or None if there is no such column."""
    with table(ms, ack=False) as t:
        if colname not in t.colnames():
            return None
        try:
            return t.getcolkeyword(colname, SIGNATURE_KEYWORD)
        except RuntimeError:
            return None


def _add_weight_column(t, colname):
    """Add ``colname`` shaped like WEIGHT_SPECTRUM, in its own storage manager."""
    desc = t.getcoldesc('WEIGHT_SPECTRUM')
    desc['name'] = colname
    dminfo = t.getdminfo('WEIGHT_SPECTRUM')
    used = [dm['NAME'] for dm in t.getdminfo().values()]
    name = next('SolveImagingWeights%d' % i for i in range(1000) if 'SolveImagingWeights%d' % i not in used)
    if 'dysco' in dminfo['TYPE'].lower():  # Dysco needs the data column, store the weights uncompressed
        dminfo = {'TYPE': 'StandardStMan', 'SPEC': {}}
    dminfo['NAME'] = name
    for key in ('dataManagerType', 'dataManagerGroup'):
        desc.pop(key, None)
    t.addcols(maketabdesc(makecoldesc(colname, desc)), dminfo)


def _parallel_hands(ncorr):
    """Indices of the parallel-hand correlations (XX/YY or RR/LL) for 1, 2 or 4 correlations."""
    return {1: [0], 2: [0, 1], 4: [0, 3]}[ncorr]


def write_imaging_solve_weights(ms, signature, colname=IMAGING_SOLVE_WEIGHT_COLUMN, chunksize=100000):
    """
    Write the solve weights = noise weights x imaging-weight factor and drop IMAGING_WEIGHT_SPECTRUM.

    The factor is normalized to a mean of 1 over the unflagged parallel-hand visibilities, so that
    the solve weights keep the scale of the noise weights. WSClean stores a factor only for the
    correlations of the Stokes I image (zero for the cross-hands); the cross-hands get the mean
    factor of the parallel hands, so that full-Jones, rotation and leakage solves keep them.

    Parameters
    ----------
    ms : str
        Measurement Set with IMAGING_WEIGHT_SPECTRUM written by WSClean.
    signature : str
        Description of the weighting, stored as a column keyword to tell when to recompute.
    colname : str, optional
        Output column.
    chunksize : int, optional
        Rows per read/write.

    Returns
    -------
    float
        Fraction of the unflagged parallel-hand visibilities with a non-zero imaging weight.
    """
    with table(ms, readonly=False, ack=False) as t:
        colnames = t.colnames()
        if 'IMAGING_WEIGHT_SPECTRUM' not in colnames:
            raise RuntimeError('IMAGING_WEIGHT_SPECTRUM missing in ' + ms + ' (WSClean -store-imaging-weights)')
        base = noise_weight_column(colnames)
        if colname not in colnames:
            _add_weight_column(t, colname)
        nrows = t.nrows()

        # mean factor over the unflagged parallel hands (the normalization)
        total, count, nonzero = 0.0, 0, 0
        par = None
        for r0 in range(0, nrows, chunksize):
            nr = min(chunksize, nrows - r0)
            factor = t.getcol('IMAGING_WEIGHT_SPECTRUM', r0, nr)
            if par is None:
                par = _parallel_hands(factor.shape[-1])
            good = (t.getcol(base, r0, nr) > 0) & ~t.getcol('FLAG', r0, nr)
            fpar, gpar = factor[..., par], good[..., par]
            total += float(fpar[gpar].sum(dtype=np.float64))
            count += int(gpar.sum())
            nonzero += int((fpar[gpar] > 0).sum())
        if count == 0 or total <= 0:
            raise RuntimeError('No unflagged data with a non-zero imaging weight in ' + ms)
        norm = count / total

        for r0 in range(0, nrows, chunksize):
            nr = min(chunksize, nrows - r0)
            factor = t.getcol('IMAGING_WEIGHT_SPECTRUM', r0, nr) * norm
            if factor.shape[-1] == 4:
                f0, f3 = factor[..., 0], factor[..., 3]
                fcross = np.where((f0 > 0) & (f3 > 0), 0.5 * (f0 + f3), np.maximum(f0, f3))
                factor[..., 1] = fcross
                factor[..., 2] = fcross
            t.putcol(colname, (t.getcol(base, r0, nr) * factor).astype(np.float32), r0, nr)
        t.putcolkeyword(colname, SIGNATURE_KEYWORD, signature)
        t.removecols('IMAGING_WEIGHT_SPECTRUM')
    return nonzero / count
