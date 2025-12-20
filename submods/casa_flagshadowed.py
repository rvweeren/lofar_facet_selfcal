#!/usr/bin/env python

import numpy as np
import os

if not os.path.isdir('casadata/data'):
    os.system('mkdir -p casadata/data')

from casaconfig import config
config.datapath=['casadata/data']
config.measurespath='casadata/data'
import casatasks.flagdata as flagdata
import casatasks.flagmanager as flagmanager
import argparse

def run_flagshadowed(ms):
    """flag shadowed data in a measurement set using CASA flagdata task"""
   
    print('Flagging shadowed antennas in ' + ms)
    flagdata(vis=ms, mode='shadow', flagbackup=False)
    
    return


def argparser():
    parser = argparse.ArgumentParser(description='Run CASA flagdata to flag shadowed antennas in a Measurement Set.')
    parser.add_argument('--ms', help='Measurement Set', type=str, required=True)
    return parser.parse_args()

def main():
    args = argparser()
    print('Running casapy flagdata to flag shadowed antennas...')
    run_flagshadowed(args.ms)
    
if __name__ == "__main__":
    main()
