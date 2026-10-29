"""Create and modify Measurement Set columns (weight spectra, phase and
phase-difference columns, model-normalised data) and stack MSs."""
import logging
import os
import shutil
import time

import numpy as np
from casacore.tables import makecoldesc, table, taql  # type: ignore

from submods.runtime_utils import run, terminal_print


def mscolexist(ms, colname):
    """
    Check if a column exists in a measurement set.

    This function verifies whether a specified column name exists in a given
    measurement set directory. It returns `True` if the column exists and `False`
    otherwise.

    Parameters
    ----------
    ms : str
        The path to the measurement set directory.
    colname : str
        The name of the column to check for existence.

    Returns
    -------
    bool
        `True` if the column exists in the measurement set, `False` otherwise.

    Raises
    ------
    None
        This function does not raise any exceptions, but it assumes that the
        `os` module and `table` class from `casacore.tables` are properly imported.
    """
    if os.path.isdir(ms):
        with table(ms, readonly=True, ack=False) as t:
            colnames = t.colnames()
            if colname in colnames:  # check if the column is in the list
                exist = True
            else:
                exist = False
    else:
        exist = False  # ms does not exist
    return exist


def create_phase_slope(inmslist, incol='DATA', outcol='DATA_PHASE_SLOPE',
                       ampnorm=False, dysco=False, testscfactor=1., crosshandtozero=True):
    """
    Creates a new column to solve for a phase slope from.

    Parameters
    ----------
    inmslist : list
        list of input measurement sets.
    incol : str
        name of the input column to copy (meta)data from.
    outcol : str
        name of the output column that will be created.
    ampnorm : bool
        If True, only takes phases from the input visibilities and sets their amplitude to 1.
    dysco : bool
        dysco compress the output column.
    testscfactor : float, optional
        Scale factor applied to adjacent-channel phase differences.
    crosshandtozero : bool, optional
        Set cross-hand correlations to zero in the output column.

    Returns
    -------
    None
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                terminal_print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)
            data = t.getcol(incol)
            dataslope = np.copy(data)
            for ff in range(data.shape[1] - 1):
                if ampnorm:
                    dataslope[:, ff, 0] = np.copy(
                        np.exp(1j * testscfactor * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                    dataslope[:, ff, 3] = np.copy(
                        np.exp(1j * testscfactor * (np.angle(data[:, ff, 3]) - np.angle(data[:, ff + 1, 3]))))
                    if crosshandtozero:
                        dataslope[:, ff, 1] = 0. * np.exp(1j * 0)
                        dataslope[:, ff, 2] = 0. * np.exp(1j * 0)
                else:
                    dataslope[:, ff, 0] = np.copy(np.abs(data[:, ff, 0]) * np.exp(
                        1j * testscfactor * (np.angle(data[:, ff, 0]) - np.angle(data[:, ff + 1, 0]))))
                    dataslope[:, ff, 3] = np.copy(np.abs(data[:, ff, 3]) * np.exp(
                        1j * testscfactor * (np.angle(data[:, ff, 3]) - np.angle(data[:, ff + 1, 3]))))
                    if crosshandtozero:
                        dataslope[:, ff, 1] = 0. * np.exp(1j * 0)
                        dataslope[:, ff, 2] = 0. * np.exp(1j * 0)

            # last freq set to second to last freq because difference reduces length of freq axis with one
            dataslope[:, -1, :] = np.copy(dataslope[:, -2, :])
            t.putcol(outcol, dataslope)

        del data, dataslope
    return


def stackwrapper(inmslist: list, msout_prefix: str = 'stack', column_to_normalise: str = 'DATA') -> None:
    """Normalize and stack measurement sets by their common time axes.

    Parameters
    ----------
    inmslist : list
        Input Measurement Set paths.
    msout_prefix : str, optional
        Prefix for generated stacked Measurement Sets.
    column_to_normalise : str, optional
        Visibility column used to normalize the input data.

    Returns
    -------
    tuple of list
        Stacked Measurement Set paths and their corresponding input groups.

    Raises
    ------
    TypeError
        If ``inmslist`` is not a list.
    """
    if type(inmslist) is not list:
        raise TypeError('Incorrect input type for inmslist')
    terminal_print('Adding weight spectrum to stack')
    create_weight_spectrum(inmslist, 'WEIGHT_SPECTRUM_PM', updateweights=True,
                           updateweights_from_thiscolumn='MODEL_DATA')
    terminal_print('Attempting to normalise data to point source')
    normalize_data_bymodel(inmslist, outcol='DATA_NORM', incol=column_to_normalise,
                           modelcol='MODEL_DATA')
    terminal_print('Stacking datasets')

    start = time.time()
    msout_stacks, mss_timestacks = stackMS_taql(inmslist, outputms_prefix=msout_prefix, incol='DATA_NORM', outcol='DATA', weightref='WEIGHT_SPECTRUM_PM')
    now = time.time()
    terminal_print(f'Stacking took {now - start} seconds')

    assert len(msout_stacks) == len(mss_timestacks)

    return msout_stacks, mss_timestacks


def create_weight_spectrum_modelratio(inmslist, outweightcol, updateweights=False,
                                      originalmodel='MODEL_DATA', newmodel='MODEL_DATA_PHASE_SLOPE', backup=True):
    """
    Create a weight column and optionally update it from model ratios.

    Parameters
    ----------
    inmslist : str or list of str
        Measurement Set path or paths.
    outweightcol : str
        Output weight-column name.
    updateweights : bool, optional
        Scale weights using the two model
        columns when available.
    originalmodel : str, optional
        Original model-column name.
    newmodel : str, optional
        Replacement model-column name.
    backup : bool, optional
        Preserve the source weights in a backup column.

    Returns
    -------
    None
        Measurement Sets are modified in place.
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    stepsize = 1000000
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            weightref = 'WEIGHT_SPECTRUM'
            if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
                weightref = 'WEIGHT_SPECTRUM_SOLVE'  # for LoTSS-DR2 datasets
            if backup and ('WEIGHT_SPECTRUM_BACKUP' not in t.colnames()):
                #desc = t.getcoldesc(weightref)
                #desc['name'] = 'WEIGHT_SPECTRUM_BACKUP'
                #t.addcols(desc)
                addcol(t, weightref, 'WEIGHT_SPECTRUM_BACKUP', write_outcol=True) 
            if outweightcol not in t.colnames():
                terminal_print('Adding', outweightcol, 'to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = outweightcol
                #t.addcols(desc)
                addcol(t, weightref, outweightcol)
            # os.system('DP3 msin={ms} msin.datacolumn={weightref} msout=. msout.datacolumn={outweightcol} steps=[]')
            if t.nrows() < stepsize: stepsize = t.nrows()  # if less than stepsize rows, do not loop
            for row in range(0, t.nrows(), stepsize):
                terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                weight = t.getcol(weightref, startrow=row, nrow=stepsize, rowincr=1).astype(np.float64)
                if updateweights and originalmodel in t.colnames() and newmodel in t.colnames():
                    model_orig = t.getcol(originalmodel, startrow=row, nrow=stepsize, rowincr=1).astype(np.complex256)
                    model_new = t.getcol(newmodel, startrow=row, nrow=stepsize, rowincr=1).astype(np.complex256)

                    model_orig[:, :, 1] = model_orig[:, :, 0]  # make everything XX/RR
                    model_orig[:, :, 2] = model_orig[:, :, 0]  # make everything XX/RR
                    model_orig[:, :, 3] = model_orig[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 1] = model_new[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 2] = model_new[:, :, 0]  # make everything XX/RR
                    model_new[:, :, 3] = model_new[:, :, 0]  # make everything XX/RR
                else:
                    model_orig = 1.
                    model_new = 1.
                terminal_print('Mean weights input', np.nanmean(weight))
                terminal_print('Mean weights change factor', np.nanmean((np.abs(model_orig)) ** 2))
                t.putcol(outweightcol, (weight * (np.abs(model_orig / model_new)) ** 2).astype(np.float64), startrow=row,
                         nrow=stepsize, rowincr=1)
                # print(weight.shape, model_orig.shape)
        terminal_print()
        del weight, model_orig, model_new


def addcol(t, incol, outcol, write_outcol=False, dysco=False):
    """
    Adds a new column to a table by copying the description and data from an existing column.
    If the output column already exists, it will not be added again.
    Optionally, the data from the input column can be copied to the output column.

    Parameters
    ----------
    t : table
        the table to which the column will be added.
    incol : str
        name of the input column to copy (meta)data from.
    outcol : str
        name of the output column that will be created.
    write_outcol : bool
        If True, copy the data from the input column to the output column.
    dysco : bool
        If True, use Dysco compression for the new column.

    Returns
    -------
    None
    """
    
    if outcol not in t.colnames():
        logging.info('Adding column: ' + outcol)
        coldmi = t.getdminfo(incol)
        if dysco:
            coldmi['NAME'] = 'Dysco' + outcol
        else:
            coldmi['NAME'] = outcol
        try:
            t.addcols(makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
        except:
            coldmi['TYPE'] = "StandardStMan"  # DyscoStMan"
            t.addcols(makecoldesc(outcol, t.getcoldesc(incol)), coldmi)
    if (outcol != incol) and write_outcol:
        # copy over the columns
        taql("UPDATE $t SET " + outcol + "=" + incol)


def create_weight_spectrum(inmslist, outweightcol, updateweights=False, 
                           updateweights_from_thiscolumn='MODEL_DATA', backup=True):
    """
    Creates or updates a weight spectrum column in one or more Measurement Sets (MS).
    This function adds a new weight spectrum column (`outweightcol`) to the specified MS files,
    optionally backing up the existing weight spectrum column and updating weights based on a model data column.

    Parameters
    ----------
    inmslist : str or list of str
        Path(s) to the input Measurement Set(s). Can be a single string or a list of strings.
    outweightcol : str
        Name of the output weight spectrum column to be created or updated.
    updateweights : bool, optional
        If True, update the weights using the specified model data column. Default is False.
    updateweights_from_thiscolumn : str, optional
        Name of the column from which to update weights (typically 'MODEL_DATA'). Default is 'MODEL_DATA'.
    backup : bool, optional
        If True, create a backup of the original weight spectrum column as 'WEIGHT_SPECTRUM_BACKUP'. Default is True.

    Notes
    -----
    - For LoTSS-DR2 datasets, uses 'WEIGHT_SPECTRUM_SOLVE' if available.
    - The function processes the MS in steps to handle large datasets efficiently.
    - The weights are updated by multiplying with the squared absolute value of the model data if `updateweights` is True.
    - Requires the `table` and `addcol` utilities, and NumPy for array operations.
    """

    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    stepsize = 1000000
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            weightref = 'WEIGHT_SPECTRUM'
            if 'WEIGHT_SPECTRUM_SOLVE' in t.colnames():
                weightref = 'WEIGHT_SPECTRUM_SOLVE'  # for LoTSS-DR2 datasets
            if backup and ('WEIGHT_SPECTRUM_BACKUP' not in t.colnames()):
                terminal_print('Adding WEIGHT_SPECTRUM_BACKUP to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = 'WEIGHT_SPECTRUM_BACKUP'
                #t.addcols(desc)
                addcol(t, weightref, 'WEIGHT_SPECTRUM_BACKUP', write_outcol=True)

            if outweightcol not in t.colnames():
                terminal_print('Adding', outweightcol, 'to', ms, 'based on', weightref)
                #desc = t.getcoldesc(weightref)
                #desc['name'] = outweightcol
                #t.addcols(desc)
                addcol(t, weightref, outweightcol)

            # os.system('DP3 msin={ms} msin.datacolumn={weightref} msout=. msout.datacolumn={outweightcol} steps=[]')
            if t.nrows() < stepsize: stepsize = t.nrows() # if less than stepsize rows, do not loop
            for row in range(0, t.nrows(), stepsize):
                terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                weight = t.getcol(weightref, startrow=row, nrow=stepsize, rowincr=1).astype(np.float64)
                if updateweights and updateweights_from_thiscolumn in t.colnames():
                    model = t.getcol(updateweights_from_thiscolumn, startrow=row, nrow=stepsize, rowincr=1).astype(
                        np.complex256)
                    model[:, :, 1] = model[:, :, 0]  # make everything XX/RR
                    model[:, :, 2] = model[:, :, 0]  # make everything XX/RR
                    model[:, :, 3] = model[:, :, 0]  # make everything XX/RR
                else:
                    model = 1.
                terminal_print('Mean weights input', np.nanmean(weight))
                terminal_print('Mean weights change factor', np.nanmean((np.abs(model)) ** 2))
                t.putcol(outweightcol, (weight * (np.abs(model)) ** 2).astype(np.float64), startrow=row, nrow=stepsize,
                         rowincr=1)
                # print(weight.shape, model.shape)
        terminal_print()
        del weight, model


def normalize_data_bymodel(inmslist, outcol='DATA_NORM', incol='DATA', modelcol='MODEL_DATA', stepsize=1000000):
    """
    Normalize visibility data by model data.

    Parameters
    ----------
    inmslist : str or list of str
        List of input Measurement Set(s).
    outcol : str, optional
        Name of the output column for normalized data. Default is 'DATA_NORM'.
    incol : str, optional
        Name of the input column containing original data. Default is 'DATA'.
    modelcol : str, optional
        Name of the model column. Default is 'MODEL_DATA'.
    stepsize : int, optional
        Step size for processing rows. Default is 1000000.

    Returns
    -------
    None
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                terminal_print('Adding', outcol, 'to', ms, 'based on', incol)
                # store uncompressed in its own StandardStMan; without an explicit
                # data manager casacore can bind the new array column to a
                # scalar-only manager (e.g. AntennaPairStMan in DP3-written MSs)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = {'TYPE': 'StandardStMan', 'NAME': outcol, 'SPEC': {}}
                t.addcols(newdesc, newdmi)
            for row in range(0, t.nrows(), stepsize):
                data = t.getcol(incol, startrow=row, nrow=stepsize, rowincr=1)
                if modelcol in t.colnames():
                    model = t.getcol(modelcol, startrow=row, nrow=stepsize, rowincr=1)
                    terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                    # print(np.max(abs(model)))
                    # print(np.min(abs(model)))
                    np.divide(data, model, out=data, where=np.abs(model) > 0)
                    t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)
                else:
                    t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)


def _get_stack_groups(inmslist, outputms_prefix='stack', verbose=False):
    """Return ordered stacked-MS names and their input MS groups."""
    starttimelist = []
    mss_timestacks = []
    for ms in inmslist:
        with table(ms, ack=False) as t:
            starttime = t.TIME[0]
        try:
            group = starttimelist.index(starttime)
            if verbose:
                terminal_print('group', group)
                terminal_print(f'append {ms} to {starttime}: {mss_timestacks[group]}')
            mss_timestacks[group].append(ms)
        except ValueError:
            starttimelist.append(starttime)
            mss_timestacks.append([ms])
        if verbose:
            terminal_print(f'new list {ms} to {starttime}')

    if verbose:
        terminal_print('Unique Measurement Set start times:', starttimelist)
        terminal_print(f'Found {len(starttimelist)} groups of MSs with same time axis.')
        terminal_print(f'Groups: {mss_timestacks}.')

    return [
        (f'{outputms_prefix}_t{group_id:02d}.MS', ms_group)
        for group_id, ms_group in enumerate(mss_timestacks)
    ]


def stackMS_taql(inmslist: list, outputms_prefix: str = 'stack', incol: str = 'DATA_NORM', outcol: str = 'DATA',
                 weightref: str = 'WEIGHT_SPECTRUM_PM', outcol_weight: str = 'WEIGHT_SPECTRUM'):
    """
    Stack a list of MSes - per group with same time axis, one stacked MS is created.

    Parameters
    ----------
    inmslist : list
        List of input Measurement Sets to stack.
    outputms_prefix : str
        Name of the output MS.
    incol : str
        Column to stack from the individual MSes.
    outcol : str
        Name of the stacked data column in the output MS.
    weightref : str
        Name of the weight column to stack from the individual files.
    outcol_weight : str
        Name of the stacked weight column in the output MS.
    Returns
    -------
    msout_stacked: list of stacked MS names
        mss_timestacks: list of input MS grouped in timestacks
    """

    stack_groups = _get_stack_groups(inmslist, outputms_prefix, verbose=True)
    msout_stacked = [stacked_ms for stacked_ms, _ in stack_groups]
    mss_timestacks = [ms_group for _, ms_group in stack_groups]
    for outputms, inmslist_timestack in stack_groups:
        terminal_print(f'Using input column {incol}')
        terminal_print(f'Writing to {outputms}')
        if not isinstance(inmslist_timestack, list):
            if os.path.isdir(outputms):  # delete MS if it exists
                shutil.rmtree(outputms, ignore_errors=True)
            shutil.copytree(inmslist_timestack, outputms)
            terminal_print("WARNING: Stacking was performed on only one MS, so not really a meaningful stack")
            return True
        if os.path.isdir(outputms):  # delete MS if it exists
            shutil.rmtree(outputms, ignore_errors=True)
        shutil.copytree(inmslist_timestack[0], outputms)

        TAQLSTR = f'UPDATE {outputms} SET DATA = ('
        sum_clause = ' + '.join(
            [f'ms{idx:02d}.DATA_NORM * ms{idx:02d}.WEIGHT_SPECTRUM_PM' for idx in range(1, len(inmslist_timestack) + 1)])
        sum_weight_clause = ' + '.join([f'ms{idx:02d}.WEIGHT_SPECTRUM_PM' for idx in range(1, len(inmslist_timestack) + 1)])
        from_clause = ', '.join([f'{ms} AS ms{idx:02d}' for idx, ms in enumerate(inmslist_timestack, start=1)])

        taql_query = f'{TAQLSTR} {sum_clause}) / ({sum_weight_clause}) FROM {from_clause}'

        terminal_print('Stacking DATA')
        terminal_print('TAQL query:', taql_query)
        taql(taql_query)

        terminal_print('Stacking WEIGHT_SPECTRUM')
        terminal_print(f'UPDATE {outputms} SET {outcol_weight} = ({sum_weight_clause}) FROM {from_clause}')
        taql(f'UPDATE {outputms} SET {outcol_weight} = ({sum_weight_clause}) FROM {from_clause}')

    return msout_stacked, mss_timestacks


def create_phasediff_column(inmslist, incol='DATA', outcol='DATA_CIRCULAR_PHASEDIFF', dysco=True, stepsize=1000000):
    """
    Creates a new column for the phase difference solve.

    Parameters
    ----------
    inmslist : list
        list of input Measurement Sets.
    incol : str
        name of the input column to copy (meta)data from.
    outcol : str
        name of the output column that will be created.
    dysco : bool
        dysco compress the output column.
    stepsize : int
        step size for row looping in casacore tables getcol/putcol
    """

    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms + '::POLARIZATION', ack=False) as tp:
            linear = np.array_equal(tp.getcol('CORR_TYPE'), [[9, 10, 11, 12]])
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                terminal_print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)

            for row in range(0, t.nrows(), stepsize):
                terminal_print("Doing {} out of {}, (step: {})".format(row, t.nrows(), stepsize))
                data = t.getcol(incol, startrow=row, nrow=stepsize, rowincr=1)
                if linear:  # form RR and LL from XX, XY, YX, YY (as polconv.py lin2circ)
                    rr = data[:, :, 0] - 1j * data[:, :, 1] + 1j * data[:, :, 2] + data[:, :, 3]
                    ll = data[:, :, 0] + 1j * data[:, :, 1] - 1j * data[:, :, 2] + data[:, :, 3]
                    phasediff = np.angle(rr) - np.angle(ll)
                    del rr, ll
                else:
                    phasediff = np.copy(np.angle(data[:, :, 0]) - np.angle(data[:, :, 3]))  # RR - LL
                data[:, :, 0] = 0.5 * np.exp(
                    1j * phasediff)  # because I = RR+LL/2 (this is tricky because we work with phase diff)
                data[:, :, 3] = data[:, :, 0]
                t.putcol(outcol, data, startrow=row, nrow=stepsize, rowincr=1)
                del data
                del phasediff

    if False:
        # data = t.getcol(incol)
        # t.putcol(outcol, data)
        # t.close()

        time.sleep(2)
        cmd = "taql 'update " + ms + " set "
        cmd += outcol + "[,0]=0.5*EXP(1.0i*(PHASE(" + incol + "[,0])-PHASE(" + incol + "[,3])))'"
        # cmd += outcol + "[,3]=" + outcol + "[,0],"
        # cmd += outcol + "[,1]=0+0i,"
        # cmd += outcol + "[,2]=0+0i'"
        terminal_print('Command:', cmd)
        run(cmd, taql=True)
        cmd = "taql 'update " + ms + " set "
        # cmd += outcol + "[,0]=0.5*EXP(1.0i*(PHASE(" + incol + "[,0])-PHASE(" + incol + "[,3]))),"
        cmd += outcol + "[,3]=" + outcol + "[,0]'"
        # cmd += outcol + "[,1]=0+0i,"
        # cmd += outcol + "[,2]=0+0i'"
        terminal_print('Command:', cmd)
        run(cmd, taql=True)
    return


def create_phase_column(inmslist, incol='DATA', outcol='DATA_PHASEONLY', dysco=True):
    """
    Creates a new column containging visibilities with their original phase, but unity amplitude.

    Parameters
    ----------
    inmslist : list
        list of input Measurement Sets.
    incol : str
        name of the input column to copy (meta)data from.
    outcol : str
        name of the output column that will be created.
    dysco : bool
        dysco compress the output column.
    """
    if not isinstance(inmslist, list):
        inmslist = [inmslist]
    for ms in inmslist:
        with table(ms, readonly=False, ack=True) as t:
            if outcol not in t.colnames():
                terminal_print('Adding', outcol, 'to', ms)
                desc = t.getcoldesc(incol)
                newdesc = makecoldesc(outcol, desc)
                newdmi = t.getdminfo(incol)
                if dysco:
                    newdmi['NAME'] = 'Dysco' + outcol
                else:
                    newdmi['NAME'] = outcol
                t.addcols(newdesc, newdmi)
            data = t.getcol(incol)
            data[:, :, 0] = np.copy(np.exp(1j * np.angle(data[:, :, 0])))  # because I = xx+yy/2
            data[:, :, 3] = np.copy(np.exp(1j * np.angle(data[:, :, 3])))  # because I = xx+yy/2
            t.putcol(outcol, data)
        del data
    return


def remove_column_ms(mslist, colname):
    """
    Remove a column from a Measurement Set or a list of Measurement Sets.

    Parameters
    ----------
    mslist : str/list
        Input Measurement Set(s) as a string or a list of strings.
    colname : str
        Column name to be removed.
    """
    mslist = [mslist] if isinstance(mslist, str) else mslist

    # assert that colname is a string
    assert isinstance(colname, str), "colname must be a string" 

    for ms in mslist:
        with table(ms, readonly=False, ack=False) as ts:
            ts.removecols(colname)
