import numpy as np
import matplotlib.pyplot as plt
from wiggles import  wiggles

traces = np.load('seismic2d.npy')
traces = traces[200:250, 100:150]
nsample = traces.shape[0]
ntraces = traces.shape[1]
dt=4

x,y1,y2 = wiggles(traces,dt)
fig, ax = plt.subplots()
ax.plot(y1, x, color='black', linewidth=0.5)
ax.fill_betweenx(x, y1, y2, where=(y1 >= y2), color='k', linewidth=0)
plt.xlim(min(y1), max(y1))
plt.ylim(dt * nsample, 0)
plt.show()