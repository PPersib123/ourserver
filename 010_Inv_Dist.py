import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

data = np.loadtxt('dataporosity.txt',skiprows=1)

nx = 300
ny = 300
p = 2

X = data[:, 0]
Y = data[:, 1]
Z = data[:, 2]

K = len(Z)

x = np.linspace(np.min(X), max(X), nx)
y = np.linspace(np.min(Y), max(Y), ny)

[x,y] = np.meshgrid(x,y)
x = x.flatten()
y = y.flatten()
datae = np.vstack((x,y)).T

tree = neighbors.KDTree(data[:,0:2], leaf_size=X.shape[0] + 1)
dist, ind = tree.query(datae, k=K)

Z0 = Z[ind]
d = 1 / dist ** p
u = np.sum(d * Z0, axis=1) / np.sum(d, axis=1)

u = u.reshape(ny,nx)

plt.figure(1)
plt.imshow(u, aspect='auto', interpolation='bilinear', extent=[np.min(x), np.max(x), np.min(y), np.max(y)], origin='lower')
plt.title('Inverse Distance')
plt.show()
