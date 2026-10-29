"""
facetselfcal hooks for the treemf and treemfphasediff soltypes: turn the DP3 solve command built
by runDPPPbase (tec+phase+delay, or the scalarphasediff phaseonly solve) into a DP3_Python_treeMF
command with the same arguments plus the treeMF settings.
"""
import os

TREEMF_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DP3_Python_treeMF.py")


def treemf_command(cmd, args):
    """
    Parameters
    ----------
    cmd : str
        The DP3 solve command (starting with 'DP3 ', with ddecal.mode=tec+phase+delay).
    args : dict
        facetselfcal options (treemf_*).

    Returns
    -------
    str
        The same command run by DP3_Python_treeMF, with the ddecal.treemf.* keys appended.
    """
    stripped = cmd.lstrip()
    if not stripped.startswith("DP3 "):
        raise ValueError("expected a DP3 command, got: " + cmd[:80])
    new = "python " + TREEMF_SCRIPT + " " + stripped[4:].rstrip() + " "
    new += "ddecal.treemf.model=" + str(args["treemf_model"]) + " "
    new += "ddecal.treemf.delaymax=" + str(args["treemf_delaymax"]) + " "
    new += "ddecal.treemf.tecmax=" + str(args["treemf_tecmax"]) + " "
    new += "ddecal.treemf.tecprior=" + str(args["treemf_tecprior"]) + " "
    new += "ddecal.treemf.tecsig100=" + str(args["treemf_tecsig100"]) + " "
    if args.get("treemf_delaysigma") is not None:
        new += "ddecal.treemf.delaysigma=" + str(args["treemf_delaysigma"]) + " "
    new += "ddecal.treemf.propagate=" + str(bool(args["treemf_propagate"])) + " "
    new += "ddecal.treemf.propagate_tecsigma=" + str(args["treemf_propagate_tecsigma"]) + " "
    new += "ddecal.treemf.propagate_delaysigma=" + str(args["treemf_propagate_delaysigma"]) + " "
    new += "ddecal.treemf.tiecore=" + str(bool(args["treemf_tiecore"])) + " "
    if args.get("treemf_delaycentre", "zero") != "zero":
        new += "ddecal.treemf.delaycentre=" + str(args["treemf_delaycentre"]) + " "
        new += "ddecal.treemf.delaytrack_search=" + str(args["treemf_delaytrack_search"]) + " "
        new += "ddecal.treemf.delaytrack_maxstep=" + str(args["treemf_delaytrack_maxstep"]) + " "
        new += "ddecal.treemf.delaytrack_tecwander=" + str(args["treemf_delaytrack_tecwander"]) + " "
    if args.get("treemf_delaycentre", "zero") != "zero" or args.get("treemf_delayadapt", 0) > 0:
        new += "ddecal.treemf.delaytrack_block=" + str(args["treemf_delaytrack_block"]) + " "
    if args.get("treemf_delayadapt", 0) > 0:
        new += "ddecal.treemf.delayadapt=" + str(args["treemf_delayadapt"]) + " "
    return new


def treemfphasediff_command(cmd, args, source_column=None):
    """
    facetselfcal hook for the treemfphasediff soltype: turn the DP3 scalarphasediff solve command
    (mode phaseonly on DATA_CIRCULAR_PHASEDIFF) into a DP3_Python_treeMF command that fits a
    rotation measure (rm or rm+phase) to RR conj(LL).

    Parameters
    ----------
    cmd : str
        The DP3 solve command built by runDPPPbase for the phase-difference solve.
    args : dict
        facetselfcal options (treemfphasediff_*).
    source_column : str or None
        The (circular) data column the phase-difference column was made from; RR conj(LL) is
        then formed from it. None: read the phase-difference column itself (unit amplitude only).

    Returns
    -------
    str
        The command run by DP3_Python_treeMF, with the ddecal.treemf.* keys appended.
    """
    stripped = cmd.lstrip()
    if not stripped.startswith("DP3 "):
        raise ValueError("expected a DP3 command, got: " + cmd[:80])
    body = stripped[4:].rstrip()
    amp = str(args["treemfphasediff_amp"])
    if source_column is not None:
        body = body.replace("msin.datacolumn=DATA_CIRCULAR_PHASEDIFF",
                            "msin.datacolumn=" + source_column)
        source = "rrll"
    else:
        source, amp = "column", "unit"
    new = "python " + TREEMF_SCRIPT + " " + body + " "
    new += "ddecal.treemf.model=" + str(args["treemfphasediff_model"]) + " "
    new += "ddecal.treemf.rmmax=" + str(args["treemfphasediff_rmmax"]) + " "
    new += "ddecal.treemf.rmprior=" + str(args["treemfphasediff_rmprior"]) + " "
    new += "ddecal.treemf.rmsig100=" + str(args["treemfphasediff_rmsig100"]) + " "
    new += "ddecal.treemf.rmsigma=" + str(args["treemfphasediff_rmsigma"]) + " "
    if args.get("treemfphasediff_offsetsigma") is not None:
        new += "ddecal.treemf.offsetsigma=" + str(args["treemfphasediff_offsetsigma"]) + " "
    new += "ddecal.treemf.pdiffamp=" + amp + " "
    new += "ddecal.treemf.pdiffsource=" + source + " "
    new += "ddecal.treemf.propagate=" + str(bool(args["treemfphasediff_propagate"])) + " "
    new += "ddecal.treemf.propagate_rmsigma=" + str(args["treemfphasediff_propagate_rmsigma"]) + " "
    new += "ddecal.treemf.tiecore=" + str(bool(args["treemfphasediff_tiecore"])) + " "
    return new
