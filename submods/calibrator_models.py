"""Calibrator source models: setjy (CASA, or DP3 with the shipped Perley-Butler 2017
models) with the polarised 3C286 model, and reading MeerKAT WSClean calibrator models."""
import json
import os
from pathlib import Path

import astropy.units as units
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.time import Time
from casacore.tables import table, taql  # type: ignore

from submods.model_columns import predictsky
from submods.runtime_utils import run, terminal_print


# J2000 positions (degrees) of the calibrators that setjy_casa and setjy_dp3 can model
SETJY_CALIBRATORS = {'3C147': (85.650575, 49.852009),
                     '3C286': (202.784534, 30.509155),
                     '3C138': (80.291192, 16.639459),
                     '3C48': (24.422081, 33.159759)}

# upper frequency edge (Hz) of the VLA bands that have a model image; below 1.7 GHz
# (including UHF and lower frequencies) the L-band model is used
SETJY_BANDS = [('L', 1.7e9), ('S', 4.0e9), ('C', 8.0e9), ('X', 12.0e9),
               ('U', 18.0e9), ('K', 26.5e9), ('A', 40.0e9), ('Q', 50.0e9)]


def find_setjy_calibrator(ms):
    """
    Find which setjy calibrator is observed in a Measurement Set and the band of its model.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set. It must contain a single field (fieldid=0).

    Returns
    -------
    source : str
        '3C147', '3C286', '3C138' or '3C48'.
    band : str
        VLA band letter of the model (L, S, C, X, U, K, A or Q).

    Raises
    ------
    Exception
        If the field is not one of the calibrators or the frequency is above 50 GHz.
    """
    with table(ms + '::SPECTRAL_WINDOW', ack=False) as t:
        midfreq = np.mean(t.getcol('CHAN_FREQ')[0])
    band = next((name for name, upper in SETJY_BANDS if midfreq < upper), None)
    if band is None:
        terminal_print('No setjy model available above 50 GHz')
        raise Exception('No setjy model available above 50 GHz')

    with table(ms + '::FIELD', ack=False) as t:
        adir = t.getcol('DELAY_DIR')[0][0][:]
    cdatta = SkyCoord(adir[0]*units.radian, adir[1]*units.radian, frame='icrs')
    for source, (ra, dec) in SETJY_CALIBRATORS.items():
        if cdatta.separation(SkyCoord(ra*units.deg, dec*units.deg, frame='icrs')) < 0.05*units.deg:
            return source, band
    terminal_print('No calibrator found in MS that matches the coordinates of 3C147, 3C138, 3C286, or 3C48: cannot use setjy')
    raise Exception('No calibrator found in MS that matches the coordinates of 3C147, 3C138, 3C286, or 3C48: cannot use setjy')


def setjy_casa(ms):
    """
    Selects and applies the appropriate CASA setjy model image for a known calibrator present in the Measurement Set (MS).
    This function determines which standard calibrator (3C147, 3C286, 3C138, or 3C48) is present in the MS by comparing the field coordinates.
    It then selects the correct model image based on the observing frequency band and runs the CASA setjy procedure
    using a helper script. If 3C286 is detected, it also sets the polarised model for this calibrator.
    Parameters
    ----------
    ms : str
        Path to the Measurement Set (MS) directory.
    Raises
    ------
    Exception
    If no known calibrator is found in the MS field coordinates.
    Notes
    -----
    - Assumes the MS contains only a single source (fieldid=0).
    - Requires access to model images named according to the calibrator and frequency band.
    - Relies on external scripts and functions: `casa_setjy.py` and `set_polarised_model_3C286`.
    - setjy_dp3 does the same without CASA.
    """
 
    datapathc = os.path.dirname(os.path.abspath(__file__))
    submodpathc = '/'.join(datapathc.split('/')[0:-1])+'/submods'

    # find which calibrators is present in the MS (this cannot be a mulitsource MS), so fieldid is 0
    source, band = find_setjy_calibrator(ms)
    modelimage = source + '_' + band + '.im'
    terminal_print('Using model image for CASA setjy: ' + modelimage)
    cmdsetjy = f'python {submodpathc}/casa_setjy.py '
    cmdsetjy += '--ms=' + ms + ' --fieldid=0 --modelimage=' + modelimage + ' '
    terminal_print('CASA setjy command:', cmdsetjy)
    run(cmdsetjy)
    if source == '3C286':
        terminal_print('Setting polarised model for 3C286')
        set_polarised_model_3C286(ms, chunksize=1000)


def setjy_dp3(ms, modelstoragemanager=None):
    """
    Predict the Perley-Butler 2017 model of a VLA flux density calibrator into MODEL_DATA with DP3.

    This is the CASA-free counterpart of setjy_casa. The calibrator (3C147, 3C286, 3C138 or 3C48)
    and band are found as in setjy_casa. The model is a DP3 skymodel shipped in
    facetselfcal/data/setjy, made by make_setjy_skymodels.py from the CASA model images
    (nrao/VLA/CalModels). Its components carry the Perley-Butler 2017 spectrum, so the model is
    scaled per channel like CASA setjy with scalebychan=True. For the time-variable calibrators
    (3C48, 3C138, 3C147) the coefficients of the epoch nearest to the observation are used, as
    CASA does by default. If 3C286 is detected, the polarised model is set as well.

    Parameters
    ----------
    ms : str
        Path to the Measurement Set. It must contain a single field (fieldid=0).
    modelstoragemanager : str or None, optional
        Storage manager for MODEL_DATA, passed to predictsky.
    """
    source, band = find_setjy_calibrator(ms)
    setjydir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'facetselfcal', 'data', 'setjy')
    skymodel = os.path.join(setjydir, source + '_' + band + '.skymodel')

    with table(ms, ack=False) as t:
        obsyear = Time(np.mean(t.getcol('TIME')) / 86400., format='mjd').decimalyear
    epoch, coeffs = pb2017_coefficients(source, obsyear, setjydir)
    terminal_print(f'setjy_dp3: {source}, {band}-band model, Perley-Butler 2017 coefficients of epoch {epoch} '
                   f'(observation {obsyear:.2f}): {coeffs}')

    _, coeffs_skymodel = pb2017_coefficients(source, np.inf, setjydir)  # the shipped skymodels use the latest epoch
    if coeffs != coeffs_skymodel:
        skymodel_epoch = os.path.basename(skymodel).replace('.skymodel', f'_PB2017_{epoch}.skymodel')
        rescale_setjy_skymodel(skymodel, skymodel_epoch, coeffs_skymodel[0], coeffs)
        skymodel = skymodel_epoch

    terminal_print('Using skymodel for DP3 setjy: ' + skymodel)
    predictsky(ms, skymodel, modeldata='MODEL_DATA', modelstoragemanager=modelstoragemanager)
    if source == '3C286':
        terminal_print('Setting polarised model for 3C286')
        set_polarised_model_3C286(ms, chunksize=1000)


def pb2017_coefficients(source, year, setjydir):
    """
    Perley-Butler 2017 coefficients of a calibrator for the epoch nearest to a date.

    Parameters
    ----------
    source : str
        Calibrator name, e.g. '3C147'.
    year : float
        Date as a decimal year (np.inf gives the latest epoch).
    setjydir : str
        Directory with PerleyButler2017Coeffs.json.

    Returns
    -------
    epoch : float
        The selected epoch (decimal year).
    coeffs : list of float
        Coefficients a_i of log10(S/Jy) = sum_i a_i log10(nu/GHz)^i, without trailing zeros.
    """
    with open(os.path.join(setjydir, 'PerleyButler2017Coeffs.json')) as f:
        pb2017 = json.load(f)
    epochs = np.array(pb2017['epochs'])
    index = len(epochs) - 1 if np.isinf(year) else int(np.argmin(np.abs(epochs - year)))
    coeffs = list(pb2017['coeffs'][source][index])
    while len(coeffs) > 1 and coeffs[-1] == 0.0:  # the table pads with zeros
        coeffs.pop()
    return float(epochs[index]), coeffs


def rescale_setjy_skymodel(skymodel, outfile, a0_skymodel, coeffs):
    """
    Write a copy of a setjy skymodel with a different Perley-Butler 2017 spectrum.

    Parameters
    ----------
    skymodel : str
        Skymodel made by make_setjy_skymodels.py (ReferenceFrequency 1 GHz, LogarithmicSI true).
    outfile : str
        Output skymodel.
    a0_skymodel : float
        Zeroth coefficient used in the input skymodel.
    coeffs : list of float
        New coefficients a_0, a_1, ...
    """
    scale = 10**(coeffs[0] - a0_skymodel)
    spectralindex = '[' + ', '.join(f'{c:.6g}' for c in coeffs[1:]) + ']'
    with open(skymodel) as fin, open(outfile, 'w') as fout:
        for line in fin:
            if line.startswith('Format') or line.startswith('#'):
                fout.write(line)
                continue
            # Name, POINT, Ra, Dec, I, [SpectralIndex], LogarithmicSI, ReferenceFrequency,,,
            head, rest = line.split('[', 1)
            fields = head.rstrip(', ').split(', ')
            fields[4] = f'{float(fields[4]) * scale:.8g}'
            fout.write(', '.join(fields) + ', ' + spectralindex + rest.split(']', 1)[1])


def set_polarised_model_3C286(ms, chunksize=1000):
    """
    Calculates and inserts a polarised model for the 3C286 calibrator source into a Measurement Set (MS).
    This function performs the following steps:
    1. Obtains channel frequency data in GHz from the MS.
    2. Identifies the field ID corresponding to the 3C286 source (J1331+3030).
    3. Calculates the polarisation fraction and electric vector position angle (EVPA) for each frequency channel.
    4. Computes the Stokes IQUV values based on the initial Stokes I image, assuming Stokes V = 0.
    5. Converts the Stokes IQUV values to the XX, XY, YX, YY correlation model.
    6. Updates the MODEL_DATA column in the MS with the computed polarised model, processing the data in chunks.
    The polarisation model is computed as:
    Q = I * pfrac * cos(2 * EVPA)
    U = I * pfrac * sin(2 * EVPA)
    XY = U + iV (with V assumed to be 0)
    YX = U - iV (with V assumed to be 0)

    Parameters
    ----------
    ms : str
        Path to the Measurement Set.
    chunksize : int, optional
        Number of rows to process per chunk. Default is 1000.

    Returns
    -------
    None

    Notes
    -----
    - Assumes the presence of helper functions `calculate_evpa_3C286` and `calculate_pfrac_3C286`.
    - Requires the `tqdm`, `numpy`, and `casacore.tables` libraries.
    - Only updates the model for the 3C286 source (J1331+3030).
    """
    #obtain channel frequency data in GHz
    from tqdm import tqdm
    terminal_print('obtaining channel frequencies...')
    with table(ms + '::SPECTRAL_WINDOW', ack=False, readonly=True) as t:
        freqs = t.getcol('CHAN_FREQ') / 1e9     #freqs in GHz
        freqs = freqs.flatten()                 #freqs in 1D array
    
    # get polarization information to determine if linear or circular polarisation is used
    with table(ms + '::POLARIZATION', ack=False, readonly=True) as t:
        corr_type = t.getcol('CORR_TYPE') 

    #obtain field_id corresponding to J1331+3030 (3C286)
    with table(ms+'::FIELD', ack=False, readonly=True) as t:
        names = np.array(t.getcol('NAME'))
        try:
            source_id = np.array(t.getcol('SOURCE_ID'))[names == 'J1331+3030'][0]
        except:
            source_id = np.array(t.getcol('SOURCE_ID'))[names == '3C286'][0]
        terminal_print('source_id = ', source_id)

    #calculate polarisation characteristics
    terminal_print('calculating EVPA and polarisation fraction...')
    evpa  = calculate_evpa_3C286(freqs)
    pfrac = calculate_pfrac_3C286(freqs)

    #adjusting the existing model and input it back into the table
    with table(ms, ack=False, readonly=False) as t:
        with taql("select * from {} where FIELD_ID=={}".format(ms, source_id)) as tt:
            nrow = tt.nrows()   #the number of rows in the selected table

            #the number of chunks ( nrow//chunksize (+ 1 if chunksize not a
            #perfect divisor of nrow) )
            nchunk = nrow // chunksize + int(nrow % chunksize > 0)

            for ci in tqdm(range(nchunk), desc='looping over chunks'):
                cl = ci * chunksize     #the starting row of the chunk

                #for the last chunk the amount of rows is not necessarily
                #the chunksize. This corrects that
                crow = min(nrow - cl, chunksize)   #the length of the chunk

                #obtain the stokes I model (crow, nchan, ncor)
                model = tt.getcol('MODEL_DATA', startrow=cl, nrow=crow)

                #need to copy otherwise adjusted value will end up in I
                I = np.copy(model[:,:,0])

                #calculate the polarised model
                if np.array_equal(np.array([[9, 10, 11, 12]]), corr_type): # linear polarisation
                    model[:,:,0] = I + I*pfrac*np.cos(2*evpa) # XX
                    model[:,:,1] = I*pfrac*np.sin(2*evpa) # XY
                    model[:,:,2] = I*pfrac*np.sin(2*evpa) # YX
                    model[:,:,3] = I - I*pfrac*np.cos(2*evpa) # YY
                elif np.array_equal(np.array([[5, 6, 7, 8]]), corr_type): # circular polarisation
                    Q = I*pfrac*np.cos(2*evpa)
                    U = I*pfrac*np.sin(2*evpa)
                    model[:,:,0] = I # RR
                    model[:,:,1] =  (Q + (complex(0,1)*U)) # RL
                    model[:,:,2] =  (Q - (complex(0,1)*U)) # LR
                    model[:,:,3] = I # LL
                else:
                    raise ValueError("Unknown correlation type in POLARIZATION table. Expected linear or circular polarisation.")

                #inserting the polarised model
                tt.putcol('MODEL_DATA', model, startrow=cl, nrow=crow)
                
    terminal_print('successfully computed and inserted model')


def calculate_pfrac_3C286(nu):
    """
    Calculate the model polarization fraction (P) for 3C286 as a function of frequency.
    This function computes the polarization fraction of the radio source 3C286 based on a model
    with coefficients adopted from Hugo (2024), as implemented by Maarten Elion. The calculation
    is performed differently for frequencies above and below 1.1 GHz.
    Parameters
    ----------
    nu : np.ndarray
        Array of frequencies in GHz.
    Returns
    -------
    pfrac : np.ndarray
        Array of polarization fractions corresponding to the input frequencies.
    Notes
    -----
    - The model uses different coefficients for frequency ranges above and below 1.1 GHz.
    - The calculation is based on the wavelength corresponding to each frequency.
    """
    c = 2.99792e8   #speed of light [m/s]
    
    C0_high    = 0.080
    C2_high    = -0.053
    Clog2_high = -0.015

    C0_low     = 0.029
    C2_low     = -0.172
    Clog2_low  = -0.067

    #the wavelength in m
    wavelength = c / (nu*1e9) 

    #seperate the different frequency ranges
    mask_high = nu >= 1.1
    mask_low  = nu < 1.1

    #initialize solution array
    pfrac = np.zeros(nu.shape)

    #calculate polarization fraction higher range
    pfrac[mask_high] = C0_high + C2_high*wavelength[mask_high]**2 + \
                        Clog2_high*np.log10(wavelength[mask_high]**2)

    #calculate polarization fraction higher range
    pfrac[mask_low] = C0_low + C2_low*wavelength[mask_low]**2 + \
                        Clog2_low*np.log10(wavelength[mask_low]**2)

    return pfrac


def calculate_evpa_3C286(nu):
    """
    Calculate the model Electric Vector Polarization Angle (EVPA) for 3C286 in radians.
    This function computes the EVPA based on the observing frequency using a piecewise model
    with coefficients adopted from Hugo (2024), as implemented by Maarten Elion.
    Parameters
    ----------
    nu : np.ndarray
        Array of frequencies in GHz.
    Returns
    -------
    EVPA : np.ndarray
        Array of model EVPAs in radians, corresponding to the input frequencies.
    Notes
    -----
    - For frequencies >= 1.7 GHz, a quadratic model in wavelength squared is used.
    - For frequencies < 1.7 GHz, a more complex model involving log10 of frequency is used.
    - The output EVPA is in radians.
    """
    c = 2.99792e8   #speed of light [m/s]
    
    C0_high   = 32.64
    C2_high   = -85.37

    C0_low    = 29.53
    C2_low_nu = 4005.88
    C2_low    = -39.38

    #the wavelength in m
    wavelength = c / (nu*1e9) 

    #seperate the different frequency ranges
    mask_high = nu >= 1.7
    mask_low  = nu < 1.7

    #initialize solution array
    EVPA = np.zeros(nu.shape)
    
    #calculate the EVPA higher range
    EVPA[mask_high] = C0_high + C2_high*(wavelength[mask_high])**2

    #calculate the EVPA lower range
    EVPA[mask_low] = C0_low + wavelength[mask_low]**2 * (
                        C2_low_nu*(np.log10(nu[mask_low]))**3 + C2_low)

    #convert to radians
    EVPA *= np.pi/180

    return EVPA


def read_MeerKAT_wscleanmodel_5spix(filename, outfile):
    """
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 5-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb

    Parameters
    ----------
    filename : str
        input filename
    outfile : str
        ouptput filename
    """
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = data['col11'].astype(str)
    data['col11'] = np.char.add(data['col11'], ',')

    #add , after third spectral index element    
    data['col12'] = data['col12'].astype(str)
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add , after third spectral index element    
    data['col13'] = np.char.add(data['col13'], ',')
    
    #add , after LogarithmicSI element    
    data['col14'] = np.char.add(data['col14'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col15'] = data['col15'].astype(str)
    data['col15'] = np.char.add(data['col15'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # add formatting line

    # 1. Define the exact text header to insert (no shell quotes needed)
    header = "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation\n"
    out_path = Path(outfile)

    # 2. Safely read the existing file content, prepend the header, and save it back
    if out_path.is_file():
        original_content = out_path.read_text()
        out_path.write_text(header + original_content)

    return


def read_MeerKAT_wscleanmodel_4spix(filename, outfile):
    """
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 4-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb

    Parameters
    ----------
    filename : str
        input filename
    outfile : str
        ouptput filename
    """
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = data['col11'].astype(str)
    data['col11'] = np.char.add(data['col11'], ',')
    
    #add , after third spectral index element    
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add , after LogarithmicSI element    
    data['col13'] = np.char.add(data['col13'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col14'] = data['col14'].astype(str)
    data['col14'] = np.char.add(data['col14'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # 1. Define the exact text header to insert (no shell quotes needed)
    header = "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation\n"
    out_path = Path(outfile)

    # 2. Safely read the existing file content, prepend the header, and save it back
    if out_path.is_file():
        original_content = out_path.read_text()
        out_path.write_text(header + original_content)

    return


def read_MeerKAT_wscleanmodel_3spix(filename, outfile):
    """
    Read in skymodel from https://github.com/ska-sa/katsdpcal/tree/master/katsdpcal/conf/sky_models
    This model is for 3-spectral index terms
    These are used by the SDP pipeline
    (code can only handle the wsclean format models provided there)
    The function reformats the file so it can be used in DP3 and/or makesourcedb

    Parameters
    ----------
    filename : str
        input filename
    outfile : str
        ouptput filename
    """
    assert filename !=  outfile # prevent overwriting the input
    data = ascii.read(filename, delimiter=' ')

    # remove these as they are part of the skymodel format
    data.remove_column('col5')
    data.remove_column('col6')
    data.remove_column('col7')
    
    # make all sources type POINT
    data['col2'][:] =  'POINT,'
    
    # replace : with . for the declination
    data['col4'] = np.char.replace(data['col4'], ":", ".")
    
    # add , after the StokesI flux (first convert to str format)
    data['col8'] = data['col8'].astype(str)
    data['col8'] = np.char.add(data['col8'], ',')
    
    #add , after first spectral index element
    data['col9'] = np.char.add(data['col9'], ',')    

    #add , after second spectral index element    
    data['col10'] = data['col10'].astype(str)
    data['col10'] = np.char.add(data['col10'], ',')
    
    #add , after third spectral index element    
    data['col11'] = np.char.add(data['col11'], ',')


    #add , after LogarithmicSI element    
    data['col12'] = np.char.add(data['col12'], ',')
    
    #add ,,, after ReferenceFrequency element (extra comma because there are no Gaussian components)    
    data['col13'] = data['col13'].astype(str)
    data['col13'] = np.char.add(data['col13'], ',,,')
    
    ascii.write(data, outfile, overwrite=True, format='fast_no_header')
    
    # add formatting line


    header = "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency, MajorAxis, MinorAxis, Orientation\n"
    out_path = Path(outfile)

    if out_path.is_file():
        original_content = out_path.read_text()
        out_path.write_text(header + original_content)

    return
