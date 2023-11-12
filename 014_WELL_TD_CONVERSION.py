import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from wavelet import ricker
from scipy.signal import convolve
from wiggles import  wiggles
from scipy.signal import hilbert

well = np.loadtxt('WELL.LAS',skiprows=86)
depth = well[:,0]
RHOB = well[:,1]
DT = well[:,12]

well = pd.DataFrame(np.vstack((depth,RHOB,DT)).T, columns=['DEPTH', 'RHOB', 'DT'])
well[well==-999.2500]=np.nan
well = well.dropna(axis=0,how='any')
well['VP'] = 1000000/well.DT
well['AI'] = well.VP*well.RHOB

AI = well.AI.values
R=[]
for i in range(len(AI)-1):
    R.append((AI[i+1]-AI[i])/(AI[i+1]+AI[i]))
R.append(0)

well['R'] = R
depthW = well.DEPTH.values
chk = np.loadtxt('CHK.txt', skiprows=1)
timeC = chk[:,1] *2
depthC = chk[:,0]

f = interp1d(depthC,timeC,kind='linear', fill_value='exrapolate')
timeW = f(depthW)
well['TWT'] = timeW
f2 = interp1d(well.TWT.values,well.R.values,kind='linear', fill_value='exrapolate')
timedt = np.arange(846,1561,2)
Rdt = f2(timedt)
t1,w = ricker(30, 250, 2)
SS = convolve(Rdt,w, mode="same")
###PHASE ROTATION
w = -90  #degree of rotation
w= w*np.pi / 180  # radian
SS = hilbert(SS)
SS = np.cos(w) * SS.real - np.sin(w) * SS.imag

###########
SS = np.vstack((SS,SS,SS,SS,SS)).T
nsample = SS.shape[0]
well=well[['RHOB', 'VP', 'AI', 'R' ,'TWT']]
data = well.values
mneumonics = [ 'RHOB', 'VP', 'AI', 'R' ,'SYNTHETICS']
rows,cols = 1,5
fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,10), sharey=True)
for i in range(cols):
    if i < cols-1:
        ax[i].plot(data[:,i], data[:,4], linewidth = '0.5', color='b')
        ax[i].set_ylim(max(data[:,4]), min(data[:,4]))
        ax[i].set_title('%s' % mneumonics[i])
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    elif i==cols-1:
        # ax[i].plot(SS, timedt, linewidth='0.5', color='b')  #one trace
        # ax[i].imshow(SS, aspect='auto', extent=[0, 1, max(data[:,4]), min(data[:,4])], cmap='seismic')  #image color density
        ###attemp for wiggle plot
        dt = 2
        x, y1, y2 = wiggles(SS, dt)
        x = x+timedt[0]
        ax[i].plot(y1, x, color='black', linewidth=0.5)
        ax[i].fill_betweenx(x, y1, y2, where=(y1 >= y2), color='k', linewidth=0)
        ax[i].set_ylim((dt * nsample)+timedt[0], timedt[0])
        ax[i].set_title('%s' % mneumonics[i])
plt.show()
