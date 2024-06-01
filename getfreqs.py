#!/usr/bin/env python

import casacore.tables as pt
import numpy as np
import argparse
from astropy.io import ascii
# import sys
import os.path

def uniqueobsid(mslist):
   tmp = [] 
   for ms in mslist:
      tmp.append(os.path.basename(ms).split('_')[0])
   return list(np.unique(tmp))

parser = argparse.ArgumentParser(description='Create "obsid"_freqs.npy')
parser.add_argument('-m','--mslist', help='DR2 mslist file, default=big-mslist.txt, Note this should not be big-mslist.txt', default='mslist.txt', type=str)
args = vars(parser.parse_args())

msfiles   = ascii.read(args['mslist'],data_start=0)
msfiles   = list(msfiles[:][msfiles.colnames[0]]) # convert to normal list of strings
#if len(msfiles) != 6:
#    print('Hmmm, expecting you have 6 of them, but there are', len(msfiles))
#    print('You will need to edit this script in order to proceed')
#    sys.exit()

print(msfiles)
#print(uniqueobsid(msfiles))

for obsid in uniqueobsid(msfiles):
   filenamefreqs = obsid + 'freqs.npy'

   # update mslist to only this obsid
   mslistobsid = [i for i in msfiles if i.startswith(obsid+'_')]
   print(mslistobsid)
   freqs=[]
   for ms in mslistobsid:
      t = pt.table(ms+'/SPECTRAL_WINDOW', readonly=True, ack=False)
      # Check freqs due to https://github.com/lofar-astron/DP3/issues/217
      freqest1 = np.mean(t.getcol('CHAN_FREQ')[0])
      freqest2 = t[0]['REF_FREQUENCY']
      if abs(freqest1-freqest2) > 0.1E6:
         freqs.append(freqest1)
         print('Freq [Hz]',freqest1) 
      else:
         freqs.append(freqest2)
         print('Freq [Hz]',freqest2) 
      t.close()   
   np.save(filenamefreqs, freqs)
