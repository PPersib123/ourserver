import numpy as np
X = np.genfromtxt('matrix1.txt')
Y = np.genfromtxt('matrix2.txt')
Z = np.genfromtxt('matrix3.txt')
# X
# 1   2
# 6   2
# Y
# 4   5
# 2   6

# matrix addition
result = np.mat(X) + np.mat(Y)


# matrix substruction
result = np.mat(X) - np.mat(Y)

# matrix multiplication
result = np.mat(X) * np.mat(Y)


# matrix multiplication elemenwise
result = np.multiply(np.mat(X),np.mat(Y))

# matrix inversion (Gauss) only for non singular matrix
# Z
# 6   1
# 2   3

from numpy.linalg import  inv, svd
result = inv(np.mat(Z))

print(result)
U,s,V=svd(Z,full_matrices=True)
# print(X)
# print(10*'=')
# print(np.transpose(U))
# print(s)
so=[]
for i in range(2):
    if s[i] != 0:
        so.append(1/s[i])
    else:
        so.append(0)

so=np.array(so)

# result = np.mat(V) * np.mat(so)

z=np.zeros((2,2))
z[0][0]=so[0]

result = (np.mat(V) * np.mat(z))*np.mat(np.transpose(U))


# Marquardt-Lavenberg
I=np.eye(2)
# print(I)

J=np.mat(np.transpose(Z))*np.mat(Z)+0.001*I

result = inv(np.mat(J))*np.mat(np.transpose(Z))
