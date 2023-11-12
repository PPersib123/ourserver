import numpy as np
import matplotlib.pyplot as plt
from varmod import varmod

nx = 300
ny = 300

data = np.loadtxt('dataporosity.txt',skiprows=1)
X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

datav = []
with open('param.txt', 'r') as filehandle:
    for line in filehandle:
        param = line[:-1]
        datav.append(param)

type = datav[0]
C0 = np.asarray(datav[1]).astype(float)
a = np.asarray(datav[2]).astype(float)
nu = np.asarray(datav[3]).astype(float)
n = len(Z)

x = np.linspace(np.min(X), max(X), nx)
y = np.linspace(np.min(Y), max(Y), ny)

u = np.zeros((ny, nx))

x1, x2 = np.meshgrid(X, X)
y1, y2 = np.meshgrid(Y, Y)
Lu = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
Cu = C0 - varmod(Lu, a, C0, nu, type)

Xv = np.matmul(np.asarray([X]).T, np.ones((1, nx)))
Yv = np.matmul(np.asarray([Y]).T, np.ones((1, nx)))
xj = np.matmul(np.ones((n, 1)), np.asarray([x]))
mi = np.mean(Z)
for i in np.arange(0, ny, 1):
    yj = np.ones((n, nx))*y[i]
    L = np.sqrt((xj - Xv) ** 2 + (yj - Yv) ** 2)
    Cr = C0 - varmod(L, a, C0, nu, type)
    w = np.matmul(np.linalg.inv(Cu), Cr)
    w0 = mi*(1-np.sum(w, axis=0))
    u[i, :] = w0 + np.matmul(Z, w)

np.savetxt('kriging_result.txt', u)
plt.figure(1)
plt.imshow(u, aspect='auto', interpolation='bilinear', extent=[np.min(x), np.max(x), np.min(y), np.max(y)], origin='lower')
plt.title('Simple Kriging')
plt.show()
