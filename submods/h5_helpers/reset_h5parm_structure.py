import os
from ..h5_merger import merge_h5


def fix_h5(h5_list: list = None, overwrite: bool = False):
    """
    Reset h5parms to h5_merger structure. Necessary when merging h5parms with different file structures.
    """

    reset_h5s = []
    for h5file in h5_list:
        out_h5parm = h5file.replace('.h5', '.reset.h5')
        if os.path.isfile(out_h5parm) and overwrite:
            os.system('rm -f ' + out_h5parm)
        # copy to get a clean h5 with standard dimensions
        merge_h5(h5_out=out_h5parm, h5_tables=h5file, convert_tec=False)

        # overwrite original
        if overwrite:
            os.system('cp -f ' + out_h5parm + ' ' + h5file)
            os.system('rm -f ' + out_h5parm)
        reset_h5s.append(out_h5parm)

    return reset_h5s
