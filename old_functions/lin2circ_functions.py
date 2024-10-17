def circular(ms, linear=False, dysco=True):
    """
    convert to circular correlations
    """
    taql = 'taql'
    scriptn = f'python lin2circ.py'
    if linear:
        cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --lincol=CORRECTED_DATA --back'
    else:
        cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --outcol=CORRECTED_DATA'
    if not dysco:
        cmdlin2circ += ' --nodysco'
    print(cmdlin2circ)
    run(cmdlin2circ)
    run(taql + " 'update " + ms + " set DATA=CORRECTED_DATA'")
    return


def preapplydelay(H5filelist, mslist, applydelaytype, dysco=True):
    ''' Pre-apply a given list of h5parms to a measurement set, specifically intended for post-delay calibration sources.

    Args:
        H5filelist (list): list of h5parms to apply.
        mslist (list): list of measurement set to apply corrections to.
        applydelaytype (str): 'linear' or 'circular' to indicate the polarisation type of the solutions.
        dysco (bool): dysco compress the circular data column or not.
    Returns:
        None
    '''
    for ms in mslist:
        parmdb = time_match_mstoH5(H5filelist, ms)
        # from LINEAR to CIRCULAR
        if applydelaytype == 'circular':
            scriptn = f'python lin2circ.py'
            cmdlin2circ = scriptn + ' -i ' + ms + ' --column=DATA --outcol=DATA_CIRC'
            if not dysco:
                cmdlin2circ += ' --nodysco'
            run(cmdlin2circ)
            # APPLY solutions
            applycal(ms, parmdb, msincol='DATA_CIRC', msoutcol='CORRECTED_DATA', dysco=dysco)
        else:
            applycal(ms, parmdb, msincol='DATA', msoutcol='CORRECTED_DATA', dysco=dysco)
        # from CIRCULAR to LINEAR
        if applydelaytype == 'circular':
            cmdlin2circ = scriptn + ' -i ' + ms + ' --column=CORRECTED_DATA --lincol=DATA --back'
            if not dysco:
                cmdlin2circ += ' --nodysco'
            run(cmdlin2circ)
        else:
            run("taql 'update " + ms + " set DATA=CORRECTED_DATA'")
    return