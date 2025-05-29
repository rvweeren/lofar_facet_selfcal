#!/usr/bin/env python

import numpy as np
import os

if not os.path.isdir('casadata/data'):
    os.system('mkdir -p casadata/data')

from casaconfig import config
config.datapath=['casadata/data']
config.measurespath='casadata/data'
import casatasks.setjy as setjy
import casatasks.flagdata as flagdata
import casatasks.flagmanager as flagmanager
import argparse


def run_setjy(ms, fieldid, modelimage):
    """unflags all data and puts in stokes I model"""
    #unflaggin all data in order for setjy to compute model
    #for all data points
    flagdata(vis=ms, mode='unflag', flagbackup=True)
    print('unflagged all data')

    #set model I data from image
    setjy(vis=ms, field=fieldid, standard='Perley-Butler 2017',
          model=modelimage, scalebychan=True, usescratch=True)
    
    print('set the stokes I image model')

    #restore the flags to their original state
    myflaglist = flagmanager(ms)
    flag_keys  = np.array(list(myflaglist.keys()), dtype=str)

    #filter out the key corresponding to the MS path
    #and convert to integer array
    mask = flag_keys != 'MS'
    keys = np.array(flag_keys[mask], dtype=int)

    #get the key corresponding to the desired flag version
    #this is the key corresponding the the latest flag version
    #or in other words the highest key
    key = np.max(keys)

    #restore the flags
    flagmanager(vis=ms, mode='restore',
                versionname=myflaglist[key]['name'])
    print('restored flags to original state')
    return


def argparser():
    parser = argparse.ArgumentParser(description='Run CASA setjy')
    parser.add_argument('--ms', help='Measurement Set', type=str, required=True)
    parser.add_argument('--fieldid', help='Field ID name or number. Default=J1331+3030', type=str, default='J1331+3030')
    parser.add_argument('--modelimage', help='CASA model image, for example 3C286_L.im. Default is an empty string.', type=str, default='')
    return parser.parse_args()

def main():
    args = argparser()
    print('Running casapy setjy...')
    run_setjy(args.ms, args.fieldid, args.modelimage)
    

if __name__ == "__main__":
    main()
