import numpy as np
def ricker(f, tn, dt):  #f is frequency in hertz, tn is no of samples, dt is sampling rate...all unit in msec
    t = np.arange(-np.floor(tn / 2), np.floor(tn / 2) + dt, dt) / 1000
    w = (1 - 2 * np.pi ** 2 * f ** 2 * t ** 2) * np.exp(-np.pi ** 2 * f ** 2 * t ** 2)
    return t, w

def ormsby(f1, f2, f3, f4, tn, dt):
    t = np.arange(-np.floor(tn / 2), np.floor(tn / 2) + dt, dt) / 1000
    a4 = (np.pi * f4) ** 2 / (np.pi * f4 - np.pi * f3) * np.sinc(f4 * t)**2
    a3 = (np.pi * f3) ** 2 / (np.pi * f4 - np.pi * f3) * np.sinc(f3 * t)**2
    a2 = (np.pi * f2) ** 2 / (np.pi * f2 - np.pi * f1) * np.sinc(f2 * t)**2
    a1 = (np.pi * f1) ** 2 / (np.pi * f2 - np.pi * f1) * np.sinc(f1 * t)**2
    w = (a4 - a3) - (a2 - a1)
    w = w/np.max(w)
    return t, w