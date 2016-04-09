#########################################
#
# Exercise 3.3
# Tero Mantyla, 014039702
#
# Use python 3

import urllib
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from matplotlib import pyplot as plt
import math

url = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_3_3.csv'
data = np.loadtxt(urllib.request.urlopen(url), delimiter=',')
X = data


# Ploting the data and k-means
plt.figure(1)
plt.plot(data[:,0], data[:,1], 'o')
centroids,_ = kmeans(X,2)
idx,_ = vq(X, centroids)
plt.figure(2)
plt.plot(data[idx==0,0], data[idx==0,1], 'or')
plt.plot(data[idx==1,0], data[idx==1,1], 'ob')


# pairwise distance matrix
# faster way would be to do this
# from scipy.spatial import distance
# D_alt = distance.squareform(distance.pdist(X))
i = np.floor_divide(range(len(X)**2),len(X))
j = np.remainder(range(len(X)**2),len(X))
vec = X[i]-X[j]
D = np.sqrt(np.sum(vec**2, axis=1)).reshape(len(X),len(X))

# (a) Implementing two types of adjancency matricies W, W2


# (b) eigenvalues and -vectors for Laplacian L = D-W


# (c) Representing the data in 


# (d) Clustering the data with k-means using the new representation Y


# Comparing clustering solutions



# plotting kmeans
plt.show()
