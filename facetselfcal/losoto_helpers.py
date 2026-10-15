import os
from .plotting.plot_tecdelayphase import plot_tecdelayphase


def tecandphaseplotter(h5, ms, telescope="LOFAR", outplotname="plot.png"):
    """
    Make TEC and phase plots.

    Parameters
    ----------
    h5 : str
        path to the H5parm to plot.
    ms : str
        path to th ecorresponding Measurement Set.
    telescope : str
        telescope name.
    outplotname : str
        name of the output plot.

    Returns
    -------
    None
    """
    if not os.path.isdir(
        "solution_plots_%s" % os.path.basename(ms)
    ):  # needed because if this is the first plot this directory does not yet exist
        os.makedirs("solution_plots_%s" % os.path.basename(ms), exist_ok=True)
    plot_tecdelayphase(
        h5file=h5,
        outfile=f"solution_plots_{os.path.basename(ms)}/{os.path.basename(outplotname)}_nolosoto.png",
        telescope=telescope,
    )
