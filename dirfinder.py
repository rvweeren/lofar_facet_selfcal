import numpy as np
from astropy.io import ascii
import lsmtool
from sklearn.cluster import k_means
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import pickle
import glob
import argparse
import ast


def sep(x, y, a, b): return np.sqrt((x-a)**2+(y-b)**2)


def match_cluster(clust, source):
    posits = source
    retlist = []
    idxlist = []
    for i in range(len(clust)):
        seplist = []
        for j in range(len(posits)):
            seplist.append(
                sep(clust[i][0], clust[i][1], posits[j][0], posits[j][1]))
        halt = True
        while halt:
            best_fit = np.argmin(seplist)
            if best_fit in idxlist:
                seplist[best_fit] = 1e10
            else:
                halt = False
        retlist.append(posits[np.argmin(seplist)])
        idxlist.append(best_fit)
    return np.asarray(retlist)


def cluster_example(posits):
    PatchPositions = np.zeros_like(posits)
    PatchPositions[:, 0] = (posits[:, 0]*u.deg).to(u.rad).value
    PatchPositions[:, 1] = (posits[:, 1]*u.deg).to(u.rad).value
    if os.path.isfile("_facet_dirs.p"):
        # Remove file
        os.remove("_facet_dirs.p")
    with open("_facet_dirs.p", "wb") as f:
        pickle.dump(PatchPositions, f)

    cmd = f'python ds9facetgenerator.py --ms {glob.glob("*avg")[0]} --h5 _facet_dirs.p --DS9regionout _outfile.reg --imsize 8192 --pixelscale 8.0 --plottesselation'
    print(cmd)


def main(args):
    imagename, fluxcuts = args['image'], args['fluxcuts']
    fluxcuts = ast.literal_eval(fluxcuts)
    if '.txt' not in imagename:
        # Run pybdsf
        import bdsf
        bdsf.process_image(imagename, rms_box=[160, 40], rms_map=True)
        bdsf.write_catalog(format='bbs', catalog_type='gaul',
                           bbs_patches='single', outfile='_tmp_catalog.txt')
        imagename = '_tmp_catalog.txt'
    L = lsmtool.load(imagename)
    sumI = np.sum(L.getColValues('I', aggregate='sum'))

    L.group('tessellate', targetFlux=fluxcuts[0])
    posits1 = np.array([[x[0].value, x[1].value]
                       for x in L.getPatchPositions().values()])
    L.group('tessellate', targetFlux=fluxcuts[1])
    posits2 = np.array([[x[0].value, x[1].value]
                       for x in L.getPatchPositions().values()])
    L.group('tessellate', targetFlux=fluxcuts[2])
    posits3 = np.array([[x[0].value, x[1].value]
                       for x in L.getPatchPositions().values()])

    print(
        f"Total flux: {sumI}; Amount of facets: {len(posits1)}, {len(posits2)}, {len(posits3)}")
    pos_2 = match_cluster(posits2, posits1)
    pos_3 = match_cluster(posits3, pos_2)

    timestringarray = []
    for x in posits1:
        timestring = "["
        if x in pos_3:
            timestring += "'16sec','1min,"
        else:
            timestring += "None,None,"
        if x in pos_2:
            timestring += "'2min',"
        else:
            timestring += "None,"
        timestring += "'4min',"
        if x in pos_3:
            timestring += "'60min']"
        else:
            timestring += "None]"
        timestringarray.append(timestring)

    with open(args['outname'], 'w') as f:
        f.write("RA DEC solints\n")
        for i, pos in enumerate(posits1):
            crd = SkyCoord(*pos, unit=('deg', 'deg'))
            radra = str(crd.ra.to('deg').value)
            raddec = str(crd.dec.to('deg').value)
            writestring = f'{radra} {raddec} {timestringarray[i]}\n'
            f.write(writestring)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str,
                        required=True, help='Image to use for clustering')
    parser.add_argument('--fluxcuts', '-f', type=str,
                        default="[40,100,240]", help='Flux cuts to use for clustering')
    parser.add_argument('--outname', '-o', type=str,
                        default='directions.txt', help='Output name for directions file')
    res = parser.parse_args()
    res = vars(res)
    main(res)
