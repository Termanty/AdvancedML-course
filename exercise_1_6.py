#################################################
# Exercise 1.6
#
# Tero Mantyla, 014093702

import math
import numpy as np
from matplotlib import pyplot as plt
import urllib

url = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_1_data.csv'
response = urllib.urlopen(url)
X = np.loadtxt(response, delimiter=',').T

sigma = np.cov(X)
v, w = np.linalg.eig(sigma)

W = w[:,0:2]
newX = W.T.dot(X)

plt.figure(1)
plt.subplot(211)
plt.plot(newX[0,0:200], newX[1,0:200], 'o', markersize=7, color='blue', alpha=0.5)
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.title('Data with two most significant component')

plt.subplot(212)
errors = []
plt.title('Reconstruction Error')
for i in range(2,6):
    W_reduced = w[:,0:i]   
    X_reconst = np.dot(X.T, np.dot(W_reduced, W_reduced.T))
    rec_err = np.sum((X.T - X_reconst)**2)
    errors.append(rec_err)

plt.plot(range(2,6), errors, 'o', markersize=7, linestyle='-', color='red')
plt.xlim([2,5])
plt.ylim([0,40])

plt.show()

