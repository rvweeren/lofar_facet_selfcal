import tables
import numpy as np

with tables.open_file("merged.h5") as H:
    t = H.root.sol000.phase000.val[0,0,0,0,0].round(2)
    print(t)
    assert t == 1.22

with tables.open_file("fulljonesmerged.h5") as H:
    t = H.root.sol000.phase000.val[0,0,0,0].round(2)
    print(t)
    assert np.all(t == np.array([ 0.14, -2.42, -0.2 ,  0.04]))
    t = H.root.sol000.phase000.val[0,0,0,0].round(2)
    print(t)
    assert np.all(t == np.array([0.99, 0.11, 0.17, 0.94]))
