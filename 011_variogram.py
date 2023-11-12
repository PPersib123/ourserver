import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('dataporosity.txt',skiprows=1)

X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

dL = 1200
maxLag = 6000

a = 2400
C0 = 0.00085
nu = 0
typev = 'spherical'
# typev = 'gaussian'
# typev = 'exponential'


t = np.int(dL/2)
L = np.arange(0, maxLag+dL, dL)

V = np.zeros((len(L),))
for j in np.arange(0, len(L), 1):
    S = 0
    N = 0
    for i in np.arange(0, len(X), 1):
        X1 = X[0:len(X)-i]
        X2 = X[i:len(X)]
        Y1 = Y[0:len(X) - i]
        Y2 = Y[i:len(X)]
        D = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)
        n = np.where((D > L[j]-t) & (D <= L[j]+t))[0]
        S = S + np.sum((Z[n] - Z[n+i])**2)
        N = N + len(n)
    if N > 0:
        V[j] = S/(2*N)

plt.plot(L, V, '.')
plt.show()


Lv = np.arange(0, np.max(L)+1, 1)
if typev == 'spherical':
    M = np.zeros((len(Lv),))
    n = np.max(np.where(Lv <= a)[0])
    M[0:n + 1] = C0 * (3 / 2 * Lv[0:n + 1] / a - 1 / 2 * (Lv[0:n + 1] / a) ** 3)
    M[n + 1:None] = C0
    M = M + nu
    M[0] = 0
    plt.plot(L, V, '.')
    plt.plot(Lv, M)
    plt.legend(('Estimated Variogram', 'Spherical Variogram Model'))
elif typev == 'gaussian':
    M = C0*(1 - np.exp(-3*Lv/a))
    M = M + nu
    M[0] = 0
    plt.plot(L, V, '.')
    plt.plot(Lv, M)
    plt.legend(('Estimated Variogram', 'Gaussian Variogram Model'))
elif typev == 'exponential':
    M = C0*(1 - np.exp(-3*(Lv/a)**2))
    M = M + nu
    M[0] = 0
    plt.plot(L, V, '.')
    plt.plot(Lv, M)
    plt.legend(('Estimated Variogram', 'Exponential Variogram Model'))

data = [typev, C0, a, nu]

with open('param.txt', 'w') as f:
    for item in data:
        f.write("%s\n" % item)


plt.xlim(np.min(Lv), np.max(Lv)*1.05)
plt.ylim(0, np.max(V)*1.1)
plt.xlabel('Lag (m)')
plt.ylabel('Variogram')
plt.show()
