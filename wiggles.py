from interpintercept import interpolated_intercept
import numpy as np
def wiggles(traces,dt):
    sc=np.std(traces - traces.mean(axis=0))*3
    nsample = traces.shape[0]
    ntraces = traces.shape[1]
    x1 = np.linspace(1, nsample, nsample)  # time axis
    y2 = 0 * x1  # time axis
    zertt = []
    for m in range(ntraces):
        y1 = traces[:, m:(m + 1)]
        y1 = np.array(y1).flatten()
        x, y = interpolated_intercept(x1, y1, y2)
        x = np.array(x).flatten()
        zertt.append(x)
    x1 = np.linspace(1, nsample, nsample)  # time axis
    x1[0]=np.nan
    x = []
    y1 = []
    y2 = []
    for n in range(ntraces):
        traces1 = traces[:, n:(n + 1)].flatten()  ############### trace nth
        traces1 = np.hstack((traces1, 0 * zertt[n]))  ############### trace nth
        time = np.hstack((x1, zertt[n]))  ############### trace nth
        timetrace = np.vstack((time, traces1)).T
        timetrace = timetrace[timetrace[:, 0].argsort()]  # sort based on col 1
        x.extend(timetrace[:, 0])
        y1.extend(n * sc + timetrace[:, 1])
        y2.extend(n * sc + 0*timetrace[:, 1])
    x=dt*np.array(x)
    y1=np.array(y1)
    y2=np.array(y2)
    return x,y1,y2
