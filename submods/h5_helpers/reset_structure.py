import os
from ..h5_merger import merge_h5

def fix_h5(h5_list):
    """
    Fix for h5_merger that cannot handle multi-dir merges where both h5 with and without pol-axis are included
    """
    for h5file in h5_list:
        outparmdb = h5file.replace('.h5', '.tmp.h5')
        if os.path.isfile(outparmdb):
            os.system('rm -f ' + outparmdb)
        # copy to get a clean h5 with standard dimensions
        merge_h5(h5_out=outparmdb, h5_tables=h5file, propagate_flags=True)

        # overwrite original
        os.system('cp -f ' + outparmdb + ' ' + h5file)
        os.system('rm -f ' + outparmdb)
    return