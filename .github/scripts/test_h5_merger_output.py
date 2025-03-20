import tables
import numpy as np

with tables.open_file("merged.h5") as H:
    assert H.root.sol000.phase000.val[0,0,0,0,0].round(2) == 1.22

with tables.open_file("fulljonesmerged.h5") as H:
    assert np.all(H.root.sol000.phase000.val[0,0,0,0].round(3) == np.array([ 0.138, -2.417, -0.199,  0.039]))
    assert np.all(H.root.sol000.phase000.val[0,0,0,0].round(3) == np.array([0.987, 0.105, 0.17 , 0.94 ]))
