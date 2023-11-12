import numpy as np
import h5py

traces = np.load('volve.npy')
t = 200
m = 200
n = 200

f=h5py.File('volve.vol','w')
data=np.reshape(traces,(t,m,n))
dset=f.create_dataset('seis', data=data, maxshape=(t,m,n))  #TIME-INLINE-XLINE
dset.resize((t,m,n))


