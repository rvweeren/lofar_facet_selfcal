#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search VLASS for a given RA and Dec
Downloads a cutout of the best quicklook subtile from CADC
(the NRAO VLASS quicklook image cache is offline)
Original code by Anna Ho
Edited by Roland Timmerman
"""


#Imports
import sys
import subprocess
import argparse
import numpy as np
import requests
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astroquery.cadc import Cadc

#Settings to potentially tweak
crop = True
crop_scale = 256  # pixels; quicklook images have 1 arcsec pixels


def sign(x):
    """
    Returns the mathematical sign of a number
    Input: x (int/float)
    Returns: +1 or -1
    """
    
    if x<0:
        return -1
    else:
        return 1


def subtile_center(observation_id):
    """
    Centre of a subtile from its CADC observationID
    Input: observation_id (str), e.g. 'VLASS3.2.T25t16.J210759+603000'
    Returns: SkyCoord object
    """
    pos_raw = observation_id.split(".")[3]
    sep = "-" if "-" in pos_raw else "+"
    ra_raw, dec_raw = pos_raw.split(sep)
    rah = "00" if ra_raw[1:3] == "24" else ra_raw[1:3]
    return SkyCoord("{}h{}m{}s".format(rah, ra_raw[3:5], ra_raw[5:]),
                    "{}{}d{}m{}s".format(sep, dec_raw[:2], dec_raw[2:4], dec_raw[4:]), frame='icrs')


def search_subtiles(c, cadc):
    """
    Search CADC for the VLASS quicklook subtiles containing the input coordinate
    Input: c (SkyCoord object), cadc (Cadc object)
    Returns: the query result row of the subtile to use: from the latest epoch, the one
    whose centre is closest to the coordinate
    """
    results = cadc.query_region(c, radius=1 * u.arcsec, collection="VLASS")
    results = results[np.array([str(p).endswith(".quicklook") for p in results["productID"]], dtype=bool)]
    if len(results) == 0:
        raise IndexError("Zero VLASS quicklook subtiles available for the given coordinate")
    #observationID e.g. 'VLASS3.2.T25t16.J210759+603000' -> epoch (3, 2)
    epoch = [(int(str(o).split(".")[0][5:]), int(str(o).split(".")[1])) for o in results["observationID"]]
    latest = np.array([e == max(epoch) for e in epoch])
    results = results[latest]
    dist = [c.separation(subtile_center(str(o))).deg for o in results["observationID"]]
    return results[[int(np.argmin(dist))]]


def get_cutout(imname, c, crop_scale):
    #Define output name
    output_fits = "vlass_poststamp.fits"
    
    #Get header info
    hdu_list = fits.open(imname)
    header = hdu_list[0].header
    data = hdu_list[0].data[0,0,:,:]
    
    #Obtain header and drop useless axes
    wcs = WCS(header)
    wcs = wcs.dropaxis(2).dropaxis(2)
    
    pixel_coords = skycoord_to_pixel(SkyCoord(c.ra.deg, c.dec.deg, unit='deg', frame='icrs'), wcs)
    
    if pixel_coords[0] < 0  or pixel_coords[1] < 0 or pixel_coords[0] > data.shape[0]  or pixel_coords[1] > data.shape[1]:
        subprocess.call("rm -f {}".format(imname), shell=True)
        raise Exception("Requested coordinate not within the downloaded VLASS cutout")
        
    #Produce a cutout
    cutout = Cutout2D(data, c, (crop_scale, crop_scale), wcs=wcs)
        
    #Update the HDU
    hdu_list[0].data = cutout.data
    new_header = cutout.wcs.to_header()
    hdu_list[0].header.update(new_header)
    hdu_list[0].header.set('NAXIS', 4)
    hdu_list[0].header.insert('NAXIS2', ('NAXIS3', 1), after=True)
    hdu_list[0].header.insert('NAXIS3', ('NAXIS4', 1), after=True)
    hdu_list[0].header.remove('WCSAXES', ignore_missing=True)
    hdu_list[0].header.remove('MJDREF', ignore_missing=True)
    hdu_list[0].header.remove('MJD-OBS', ignore_missing=True)
    
    #Write the new fits
    hdu_list.writeto(output_fits, overwrite=True)
    
    #Cleanup
    subprocess.call("rm -f {}".format(imname), shell=True)

    return output_fits


def search_vlass(c, crop=False, crop_scale=256):
    """ 
    Searches the VLASS catalog for a source

    Parameters
    ----------
    c: coordinates as SkyCoord object
    crop: crop the image to crop_scale x crop_scale pixels around c
    crop_scale: cutout size in pixels; CADC is asked for a cutout of this radius (arcsec),
                so that the crop fits in it
    """
    cadc = Cadc()
    subtile = search_subtiles(c, cadc)
    print("Using VLASS subtile {}".format(subtile["observationID"][0]))

    url = cadc.get_image_list(subtile, c, crop_scale * u.arcsec)[0]
    imname = "{}.I.iter1.image.pbcor.tt0.subim.fits".format(subtile["observationID"][0])
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(imname, "wb") as f:
        f.write(response.content)
    if crop:    
        out = get_cutout(imname, c, crop_scale=crop_scale)
        return out
    else:
        return imname


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=\
        '''
        Searches VLASS for a source.
        User needs to supply RA (in decimal degrees), Dec (in decimal degrees)
        
        Usage: vlass_search.py <RA [deg]> <Dec [deg]> or vlass_search.py <MS>
        ''', formatter_class=argparse.RawTextHelpFormatter)
    
    #Test coords: 232.517541667 60.4054444
    try:
        ra = 15*(float(sys.argv[1]) + float(sys.argv[2])/60 + float(sys.argv[3])/3600)
        dec = float(sys.argv[4]) + sign(float(sys.argv[4]))*float(sys.argv[5])/60 + sign(float(sys.argv[4]))*float(sys.argv[6])/3600
    except IndexError:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
    except ValueError:
        pi = 3.14159265358979
        import casacore.tables as pt
        table = pt.table(sys.argv[1]+"::FIELD")
        direction = table.getcol('PHASE_DIR').squeeze()
        ra=(direction[0]%(2*pi))/pi*180
        dec=direction[1]/pi*180
        table.close()
    except Exception:
        raise TypeError("Incorrect inputs given. Usage: vlass_search.py <RA [deg]> <Dec [deg]> or vlass_search.py <MS>")
        
    c = SkyCoord(ra, dec, unit='deg')

    search_vlass(c, crop=crop, crop_scale=crop_scale)
