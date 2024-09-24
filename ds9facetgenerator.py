#!/usr/bin/python3
from scipy.spatial import Voronoi #, voronoi_plot_2d
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import argparse
import sys
import casacore.tables as pt

from shapely.geometry import Polygon
from shapely.geometry import Point
import shapely.geometry
import shapely.ops
import tables
import pickle

def read_dir_fromh5(h5):
    """
    Read in the direction info from a H5 file
    Parameters
    ----------
    h5 : str
        h5 filename
    
        Delta in degrees for sky grid
    
    Returns 
    ----------    
    sourcedir: numpy array
    contains directions (ra, dec in units of radians)    
    """
    
    # try if this is a pickle file
    try:
      f = open(h5, 'rb')
      sourcedir = pickle.load(f)
      f.close()
      print(sourcedir)
      return sourcedir
    except:
      pass

    H5 = tables.open_file(h5, mode='r') 
    sourcedir = H5.root.sol000.source[:]['dir'] 
    if len(sourcedir) < 2:
        print('Error: H5 seems to contain only one direction')
        sys.exit(1)
    H5.close()
    return sourcedir

def makeWCS(centreX, centreY, refRA, refDec, crdelt=None):
    """
    Makes simple WCS object.
    Parameters
    ----------
    centreX : int
        Centre x pixel
    centreY : int
        Centre y pixel
    refRA : float
        Reference RA in degrees
    refDec : float
        Reference Dec in degrees
    crdelt: float, optional
        Delta in degrees for sky grid
    Returns
    -------
    w : astropy.wcs.WCS object
        A simple TAN-projection WCS object for specified reference position
    """

    w = WCS(naxis=2)
    w.wcs.crpix = [centreX, centreY]
    if crdelt is None:
        crdelt = 0.066667  # 4 arcmin
    w.wcs.cdelt = np.array([-crdelt, crdelt])
    w.wcs.crval = [refRA, refDec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.set_pv([(2, 1, 45.0)])
    return w



def tessellate(x_pix, y_pix, w, dist_pix, bbox, plot_tesselation=True):
    """
    Returns Voronoi tessellation vertices
    Parameters
    ----------
    x_pix : array
        Array of x pixel values for tessellation centers
    y_pix : array
        Array of y pixel values for tessellation centers
    w : WCS object
        WCS for transformation from pix to world coordinates
    dist_pix : float
        Distance in pixels from center to outer boundary of facets
    plot_tesselation : bool
        Plot tesselation

    Returns
    -------
    verts : list
        List of facet vertices in (RA, Dec)
    """

    # Get x, y coords for directions in pixels. We use the input calibration sky
    # model for this, as the patch positions written to the h5parm file by DPPP may
    # be different
    xy = []
    for RAvert, Decvert in zip(x_pix, y_pix):
        xy.append((RAvert, Decvert))

    # Generate array of outer points used to constrain the facets
    nouter = 64
    means = np.ones((nouter, 2)) * np.array(xy).mean(axis=0)
    offsets = []
    angles = [np.pi/(nouter/2.0)*i for i in range(0, nouter)]
    for ang in angles:
        offsets.append([np.cos(ang), np.sin(ang)])
    scale_offsets = dist_pix * np.array(offsets)
    outer_box = means + scale_offsets

    # Tessellate and clip
    points_all = np.vstack([xy, outer_box])
    vor = Voronoi(points_all)

    #if plot_tesselation:
        #fig = voronoi_plot_2d(vor)
        #plt.show()

    lines = [
        shapely.geometry.LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line
    ]
    polygons = [poly for poly in shapely.ops.polygonize(lines)]

    clipped_polygons = []
    for polygon in polygons:
        # facet_poly = Polygon(facet)
        clipped_polygons.append(polygon_intersect(bbox, polygon))


    if plot_tesselation:
        import matplotlib.pyplot as plt
        [plt.plot(*poly.exterior.xy) for poly in clipped_polygons]
        plt.xlabel('Right Ascension [pixels]')
        plt.ylabel('Declination [pixels]')
        plt.axis('square')
        plt.tight_layout()
        plt.show()

    verts = []
    for poly in clipped_polygons:
        verts_xy = poly.exterior.xy
        verts_deg = []
        for x, y in zip(verts_xy[0], verts_xy[1]):
            # x_y = np.array([[y, x, 0.0, 0.0]])
            ra_deg, dec_deg = w.wcs_pix2world(x, y, 1)
            verts_deg.append((ra_deg, dec_deg))
        verts.append(verts_deg)

    # Reorder to match the initial ordering
    ind = []
    for poly in polygons:
        for j, (xs, ys) in enumerate(zip(x_pix, y_pix)):
            if poly.contains(shapely.geometry.Point(xs, ys)):
                ind.append(j)
                break
    verts = [verts[i] for i in ind]
    # return verts
    return [Polygon(vert) for vert in verts]

def generate_centroids(xmin, ymin, xmax, ymax, npoints_x, npoints_y, distort_x=0.0, distort_y=0.0):
    """
    Generate centroids for the Voronoi tessellation. These points are essentially
    generated from a distorted regular grid.

    Parameters
    ----------
    xmin : float
        Min-x pixel index, typically 0
    ymin : float
        Min-y pixel index, typically 0
    xmax : float
        Max-x pixel index, typically image width
    ymax : float
        Max-y pixel index, typically image height
    npoints_x : int
        Number of points to generate in width direction
    npoints_y : int
        Number of points to generate in height direction
    distort_x : float, optional
        "Cell width" fraction by which to distort the x points, by default 0.0
    distort_y : float, optional
        "Cell height" fraction by which to distory the y points, by default 0.0

    Returns
    -------
    X,Y : np.1darray
        Flattened arrays with X,Y coordinates
    """

    x_int = np.linspace(xmin, xmax, npoints_x)
    y_int = np.linspace(ymin, ymax, npoints_y)

    np.random.seed(0)

    # Strip the points on the boundary
    x = x_int[1:-1]
    y = y_int[1:-1]
    X, Y = np.meshgrid(x, y)

    xtol = np.diff(x)[0]
    dX = np.random.uniform(low=-distort_x*xtol, high=distort_x*xtol, size=X.shape)
    X = X + dX

    ytol = np.diff(y)[0]
    dY = np.random.uniform(low=-distort_x*ytol, high=distort_y*ytol, size=Y.shape)
    Y = Y + dY
    return X.flatten(), Y.flatten()


def reorder_facets(facets, ra, dec):
    print('\n---Reorder Polygons to match order in the H5 solution table---')
    facets_out = []
    for direction_id, _ in enumerate((ra)):
       # find closest facet
       distances = []
       for _, facet in enumerate(facets):          
          distances.append(facet.distance(Point(ra[direction_id],dec[direction_id])))
       mindist = np.argmin(distances)
       facets_out.append(facets[mindist])

    return facets_out

def polygon_intersect(poly1, poly2):
    """
    Returns the intersection of polygon2 with polygon1
    """
    clip = poly1.intersection(poly2)
    return clip

def write_ds9(fname, polygons):
    """
    Write ds9 regions file, given a list of polygons
    """

    # Write header
    header = ['# Region file format: DS9 version 4.1', 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1', "fk5", "\n"]
    with open(fname, "w") as f:
        f.writelines('\n'.join(header))
        polygon_strings = []
        for polygon in polygons:
            poly_string="polygon("
            xv,yv = polygon.exterior.xy
            for (x, y) in zip(xv[:-1], yv[:-1]):
                poly_string = f'{poly_string}{x:.5f},{y:.5f},'
            # Strip trailing comma
            poly_string = poly_string[:-1] + ")"
            polygon_strings.append(poly_string)
        f.write("\n".join(polygon_strings))


def main(args):
    
    # get phase centre from the ms in units of degrees
    t = pt.table(args.ms + '::FIELD', ack=False)
    phasedir = t.getcol('PHASE_DIR').squeeze()
    cphasedir = SkyCoord(ra=phasedir[0]*u.rad, dec=phasedir[1]*u.rad) # astropy coordinate
    phaseCentreRa =  cphasedir.ra.degree
    phaseCentreDec = cphasedir.dec.degree

    # Pixel "resolution" (in degrees!)
    dl_dm = args.pixelscale/60.0/60.0 # in units of degree 
    
    # Image size (in pixels)
    xmin = 0
    xmax = args.imsize
    ymin = 0
    ymax =  args.imsize
    centreX = (xmax - xmin) // 2 + 1
    centreY = (ymax - ymin) // 2 + 1

    # To cut the Voronoi tesselation on the bounding box, we need
    # a "circumscribing circle"
    dist_pix = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)

    # Tesselation input, points below will define the
    # Voronoi centroids. Note that the outer points
    # are stripped. So the number of interior points
    # effectively is (npoints_x - 2) * (npoints_y - 2)
    # npoints_x = 10
    # npoints_y = npoints_x

    # Distortion fraction, double-sided. i.e. if distort = 0.5,
    # the maximum displacement of an interior point is 1 "cell size"
    # distort_x = 0.35
    # distort_y = 0.35

    # load in the directions from the H5
    sourcedir = read_dir_fromh5(args.h5)

    # make ra and dec arrays and coordinates c
    ralist = sourcedir[:,0]
    declist = sourcedir[:,1]
    #print(ralist*u.rad)
    #print(declist*u.rad)

    c = SkyCoord(ra=ralist*u.rad, dec=declist*u.rad)
    #print(c.ra.degree)
    #print(c.dec.degree)

    # Make World Coord Stystem transform object
    w = makeWCS(centreX, centreY, phaseCentreRa, phaseCentreDec, dl_dm)
    
    # convert fromo ra,dec to x,y pixel
    x, y = w.wcs_world2pix(c.ra.degree, c.dec.degree, 1)
    
    if (np.max(x) >= xmax-1.) or (np.min(x) <= xmin) or (np.max(y) >=ymax-1.) or (np.min(y) <= ymin):
        print('You are feeding in a direction which sits outside the image region covered by --imsize')
        print('\n',x,'\n',y)
        sys.exit()
        
    
    # Generate coordinates
    #x, y = generate_centroids(xmin, ymin, xmax, ymax, npoints_x, npoints_y, distort_x, distort_y)
    

    bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    facets = tessellate(x, y, w, dist_pix, bbox, plot_tesselation=args.plottesselation)
    
    facets_out = reorder_facets(facets, c.ra.degree, c.dec.degree)

    write_ds9(args.DS9regionout, facets_out)
    #write_ds9(args.DS9regionout, facets)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Make DS9 Voroni region tesselation region file for WSClean')
   parser.add_argument('--ms', help='Measurement Set', type=str, required=True)
   parser.add_argument('--h5', help='Multi-dir solution file with directions', type=str, required=True)
   parser.add_argument('--DS9regionout', help='Output DS9 region file name (default=facets.reg)', type=str, default='facets.reg')
   parser.add_argument('--imsize', help='image size, required if boxfile is not used', type=int, default=8192)
   parser.add_argument('--pixelscale', help='pixels size in arcsec, default=1.5', type=float, default=1.5)
   parser.add_argument('--plottesselation', help='Plot tesselation', action='store_true')
   args = parser.parse_args()  
   main(args)
    
