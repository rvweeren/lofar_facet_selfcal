"""Merge and complete per-direction h5parm files for direction-dependent calibration."""
import os
import pickle
from pathlib import Path

import astropy.units as units
import numpy as np
import tables
from astropy.coordinates import SkyCoord

from ..h5_merger import merge_h5
from ..runtime_utils import terminal_print


def merge_splitted_h5_ordered(modeldatacolumnsin, parmdb_out, clean_up=False):
    """
    Merge direction H5 files in the order of their sky directions.

    Parameters
    ----------
    modeldatacolumnsin : list
        Direction model columns used to determine
        the input H5 file sequence.
    parmdb_out : str
        Output merged H5 file path.
    clean_up : bool, optional
        Remove the input direction H5 files after
        merging.

    Returns
    -------
    None
        The merged H5 file is written to ``parmdb_out``.
    """
    h5list_sols = []
    for colid, coln in enumerate(modeldatacolumnsin):
        h5list_sols.append('Dir' + str(colid).zfill(2) + '.h5')
    terminal_print('These are the h5 that need merging:', h5list_sols)
    if os.path.isfile(parmdb_out):
        Path(parmdb_out).unlink(missing_ok=True)

    f = open('facet_regions/facetdirections.p', 'rb')
    sourcedir = pickle.load(f)  # units are radian
    f.close()
    parmdb_merge_list = []

    for direction in sourcedir:
        terminal_print('Direction coordinates (RA/Dec, rad):', direction)
        c1 = SkyCoord(direction[0] * units.radian, direction[1] * units.radian, frame='icrs')
        distance = 1e9
        # print(c1)
        for hsol in h5list_sols:
            H5 = tables.open_file(hsol, mode='r')
            dd = H5.root.sol000.source[:][0]
            H5.close()
            ra, dec = dd[1]
            c2 = SkyCoord(ra * units.radian, dec * units.radian, frame='icrs')
            angsep = c1.separation(c2).to(units.degree)
            # print(c2)

            # print(hsol, angsep.value, '[degree]')
            if angsep.value < distance:
                distance = angsep.value
                matchging_h5 = hsol
        parmdb_merge_list.append(matchging_h5)
        terminal_print('separation direction entry and h5 entry is:', distance, matchging_h5)
        assert abs(distance) < 0.1/3600  # there should always be a close to perfect match (0.1arcsec)

    merge_h5(h5_out=parmdb_out, h5_tables=parmdb_merge_list, propagate_weights=True, convert_tec=False)

    if clean_up:
        for h5 in h5list_sols:
            Path(h5).unlink(missing_ok=True)
    return


def copy_over_solutions_from_skipped_directions(modeldatacolumnsin, id_kept):
    """
    Copy over solutions from skipped directions.

    Parameters
    ----------
    modeldatacolumnsin : list of str
        All model data column names.
    id_kept : list of int
        Indices of the model data columns kept in the solve.
    """
    h5list_sols = []
    h5list_empty = []
    for colid, coln in enumerate(modeldatacolumnsin):
        if colid >= len(id_kept):
            h5list_empty.append('Dir' + str(colid).zfill(2) + '.h5')
        else:
            h5list_sols.append('Dir' + str(colid).zfill(2) + '.h5')
    terminal_print('These h5 have solutions:', h5list_sols)
    terminal_print('These h5 are empty:', h5list_empty)

    # fill the empty directions (those that were removed and not solve) with the closest valid solutions
    for h5 in h5list_empty:
        hempty = tables.open_file(h5, mode='a')
        direction = hempty.root.sol000.source[:][0]
        ra, dec = direction[1]
        c1 = SkyCoord(ra * units.radian, dec * units.radian, frame='icrs')
        # print(c1)

        distance = 1e9
        for h5sol in h5list_sols:
            hsol = tables.open_file(h5sol, mode='r')
            directionsol = hsol.root.sol000.source[:][0]
            rasol, decsol = directionsol[1]
            c2 = SkyCoord(rasol * units.radian, decsol * units.radian, frame='icrs')
            angsep = c1.separation(c2).to(units.degree)
            # print(h5, h5sol, angsep.value, '[degree]')
            if angsep.value < distance:
                distance = angsep.value
                matchging_h5 = h5sol
            hsol.close()
        terminal_print(
            h5 + ' needs solutions copied from ' + matchging_h5,
            'Angular separation (deg):',
            distance,
        )

        # copy over the values
        with tables.open_file(matchging_h5, mode='r') as hmatch:

            for sol_type in ['phase000', 'amplitude000', 'tec000', 'rotation000', 'rotationmeasure000', 'delay000']:
                try:
                    getattr(hempty.root.sol000, sol_type).val[:] = np.copy(getattr(hmatch.root.sol000, sol_type).val[:])
                    terminal_print(f'Copied over {sol_type}')
                except AttributeError:
                    pass

        hempty.flush()
        hempty.close()
    return
