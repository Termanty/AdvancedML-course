#########################################
#
# AML Exercise 3.3
# Tero Mantyla, 014039702
#
# python 3

import urllib
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from matplotlib import pyplot as plt
import math

url = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_3_3.csv'
# data = np.loadtxt(urllib.request.urlopen(url), delimiter=',')
data = np.loadtxt('exercise_3_3.csv', delimiter=',')
X = data
n = len(X) # number of data points

# Whitening the data
# X = whiten(X)

# Ploting the data
# plt.figure(1)
# plt.plot(X[:,0], X[:,1], 'o')

# Ploting k-means
centroids,_ = kmeans(X,2)
idx,_ = vq(X, centroids)
plt.figure(2)
plt.title('pure k-means')
plt.axis('equal')
plt.plot(X[idx==0,0], X[idx==0,1], 'or')
plt.plot(X[idx==1,0], X[idx==1,1], 'ob')

# pairwise distance matrix
# faster way would be to do this
# from scipy.spatial import distance
# D_alt = distance.squareform(distance.pdist(X))
i = np.floor_divide(range(n**2),n)
j = np.remainder(range(n**2),n)
dis = np.sqrt(np.sum((X[i]-X[j])**2, axis=1)).reshape(n,n)


## (a) Implementing two types of adjancency matricies W, W1
W = np.zeros_like(dis, dtype='int')
W[dis < 0.5] = 1

A = 9
W1 = np.zeros_like(dis, dtype='int')
indeces = np.argpartition(dis, A, axis=1)[:,:A]
i = np.ravel(indeces)
j = np.floor_divide(range(len(i)), A)
W1[i,j] = 1
W1[j,i] = 1


## (b) eigenvalues and -vectors for Laplacian L = D-W
D = np.diag(sum(W))
D1 = np. diag(sum(W1))
L = D - W
L1 = D1 - W1

Eig_val, Eig_vec = np.linalg.eig(L)
Eig_val1, Eig_vec1 = np.linalg.eig(L1)

def sortEigen( evals, evecs ):
    tempvec = [(evals[i],i) for i in range(len(evals))]
    tempvec.sort()
    return [tempvec[i][1] for i in range(len(evecs))]

dsorder = sortEigen(Eig_val, Eig_vec)
dsorder1 = sortEigen(Eig_val1, Eig_vec1)

plt.figure(3)
style = [['r','b'],['--r','--b'],['-.r','-.b'],[':r',':b']]
plt.subplot(121)
plt.title('Eigenvector, e<0.5')
for i in range(4):
    plt.plot(range(120), Eig_vec[dsorder[i]], style[i][0])
plt.subplot(122)
plt.title('Eigenvector, A=8')
for i in range(4):
    plt.plot(range(120), Eig_vec1[dsorder1[i]], style[i][1])


## (c) Representing the data in 
M = 2
Y = Eig_vec[:,:M]
Y1 = Eig_vec1[:,:M]

plt.figure(4)
plt.subplot(121)
plt.title('Y, e<0.5')
plt.plot(Y[:,0], Y[:,1], 'or')
plt.subplot(122)
plt.title('Y, A=8')
plt.plot(Y1[:,0], Y1[:,1], 'ob')


## (d) Clustering the data with k-means using the new representation Y
centroids,_ = kmeans(Y,2)
idx,_ = vq(Y, centroids)
plt.figure(6)
plt.subplot(121)
plt.title('Spectral with e<0.5') 
plt.axis('equal')
plt.plot(X[idx==0,0], X[idx==0,1], 'or')
plt.plot(X[idx==1,0], X[idx==1,1], 'ob')

centroids,_ = kmeans(Y1,2)
idx,_ = vq(Y1, centroids)
plt.subplot(122)
plt.title('Spectral with A=8') 
plt.axis('equal')
plt.plot(X[idx==0,0], X[idx==0,1], 'or')
plt.plot(X[idx==1,0], X[idx==1,1], 'ob')




plt.show()
