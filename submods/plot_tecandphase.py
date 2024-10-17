#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import os, sys
import numpy as np
import argparse
import tables



def phaseplot(phase):
    phase = np.mod(phase+np.pi, 2.*np.pi) - np.pi
    return phase

def TEC2phase(TEC,freq):
    phase = -1.*(TEC*8.44797245e9/freq)
    return phase

parser = argparse.ArgumentParser(description='Plot DPPP tecandphase solutions by combining the TEC and phase offset')
#imaging options
parser.add_argument('-h5','--H5file', help='H5 solution file, must contain tecandphase solutions', type=str, required=True)
parser.add_argument('-o','--outfile', help='Output figure name (png format)', type=str, required=True)
parser.add_argument('-p','--plotpoints', help='Plot points instead of lines', action='store_true')

#parser.add_argument('-f','--freq', help='Frequnecy at which the phase corrections are plotted', type=float, required=True)
args = vars(parser.parse_args())

H=tables.open_file(args['H5file'], mode='r')


try:
   tec = H.root.sol000.tec000.val[:]
   antennas = H.root.sol000.tec000.ant[:].tolist()
   times= H.root.sol000.tec000.time[:]   
   freq = H.root.sol000.tec000.freq[:][0]
   notec = False
except:    
   print('Your solutions contain do NOT contain TEC values')
   #sys.exit()
   notec = True
try:
   phase = H.root.sol000.phase000.val[:]
   antennas = H.root.sol000.phase000.ant[:].tolist()
   times= H.root.sol000.phase000.time[:]
   freq    = H.root.sol000.phase000.freq[:][0]
   containsphase = True
   if notec:
     freq = H.root.sol000.phase000.freq[:]    
except:    
   print('Your solutions contain do NOT contain phase values')
   containsphase = False
   pass
   

print('Plotting at a frequency of:', freq/1e6, 'MHz')

times = (times-np.min(times))/3600. # in hrs since obs start

#print(tec.shape, phase.shape)
ysizefig = float(len(antennas))
refant = 0
antennas=[x.decode('utf-8') for x in antennas] # convert to proper string list

if 'ST001' in antennas:
    refant = antennas.index('ST001')
    print('ST001 detected, switching to refant:', refant)


fig, ax = plt.subplots(nrows=len(antennas), ncols=1,  figsize=(9, 1.5*ysizefig),  squeeze=True, sharex='col')
figcount = 0

if containsphase:
  if notec: 
    freqidx =   int(len(freq)/2)
    refphase = phase[:,freqidx,refant,0,0] 
  else:
    refphase = phase[:,refant,0,0] + TEC2phase(tec[:,refant,0,0], freq)
else:
 refphase = TEC2phase(tec[:,refant,0,0], freq)   

#print('Here')
for antenna_id, antenna in enumerate(antennas):
    #if antenna_id != refant:

    print(figcount)
    if containsphase:
        
      if notec: 
        phasep = phaseplot(phase[:,freqidx,antenna_id,0,0] - refphase)  
      else:
        phasep = phaseplot(phase[:,antenna_id,0,0] + TEC2phase(tec[:,antenna_id,0,0], freq) - refphase)
    else:
     phasep = phaseplot(TEC2phase(tec[:,antenna_id,0,0], freq) - refphase)
     
    if args['plotpoints']:
      ax[figcount].plot(times, phasep, '.')
      #ax[figcount].plot(times, phasep) 
    else:
      ax[figcount].plot(times, phasep)   
    
    ax[figcount].set_ylabel('phase [rad]')
    if figcount == len(antennas)-1:
      ax[figcount].set_xlabel('time [hr]')
    #print(type(antenna))  
    ax[figcount].set_title(antenna, position=(0.5, 0.75))
    ax[figcount].set_ylim(-np.pi,np.pi)
    figcount += 1

#plt.subplots_adjust(wspace=0, hspace=0)  
plt.tight_layout(pad=1.0)    
plt.savefig(str(args['outfile']))    
