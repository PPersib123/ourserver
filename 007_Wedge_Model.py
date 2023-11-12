import numpy as np
import matplotlib.pyplot as plt
from wavelet import ricker
from scipy.signal import convolve
dt = 4
ntrace = 51
ns = 200
t1,w = ricker(15, 100, dt)

# plt.plot(t1,w)
# plt.show()




traces = []
thickness=[]
for i in range(ntrace):
    R = np.zeros(ns)
    R[50] = 0.8
    R[51+i] = -0.7
    thickness.append((51+i)-50)
    Tr = convolve(R,w, mode="same")
    traces.append(Tr)
traces = np.asarray(traces).T

np.save('wedgemodel',traces)

A = traces[50,:]
x = np.arange(0,ntrace,1)
plt.figure(1)
plt.imshow(traces,cmap='seismic',interpolation='bilinear', aspect='auto', extent=[0,50,ns*dt,0])
plt.xlabel('CMP NO')
plt.ylabel('TWT [ms]')
plt.title('Wedge Model')
plt.xlim(0,50)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(x,A)
plt.xlabel('CMP NO')
plt.ylabel('Amplitude')
plt.xlim(0,50)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.subplot(3,1,2)
plt.plot(x,np.array(thickness)*dt)
plt.xlabel('CMP NO')
plt.ylabel('Thickness in TWT[ms]')
plt.xlim(0,50)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


plt.subplot(3,1,3)
plt.plot(np.array(thickness)*dt,A)
plt.xlabel('Thickness in TWT[ms]')
plt.ylabel('Amplitude')
plt.xlim(0,200)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

detuning = np.vstack((np.array(thickness)*dt,A+0.22)).T

np.save('detuning', detuning)
plt.show()