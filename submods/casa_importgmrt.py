#!/usr/bin/env python

import numpy as np
import os

if not os.path.isdir('casadata/data'):
    os.system('mkdir -p casadata/data')

from casaconfig import config
config.datapath=['casadata/data']
config.measurespath='casadata/data'
import casatasks.importgmrt as importgmrt
import argparse


def run_importgmrt(uvfits, msout, flagfile=''):
    """import GMRT uvfits file into CASA Measurement Set format
    Parameters
    ----------
    uvfits : str
        Path to the input uvfits file
    msout : str
        Path to the output Measurement Set
    flagfile : str, optional
        Path to the flag file, by default ''"""
    print('Importing GMRT uvfits file into CASA MS format...')
    importgmrt(fitsfile=uvfits, flagfile=flagfile, vis=msout)

    return


def argparser():
    parser = argparse.ArgumentParser(description='Run CASA importgmrt task')
    parser.add_argument('--uvfits', help='Input UVFITS file', type=str, required=True)
    parser.add_argument('--msout', help='Measurement Set output', type=str, required=True)
    parser.add_argument('--flagfile', help='Flag file', type=str, default='')
    return parser.parse_args()

def main():
    args = argparser()
    print('Running casapy importgmrt...')
    run_importgmrt(args.uvfits, args.msout, args.flagfile)
    

if __name__ == "__main__":
    main()
