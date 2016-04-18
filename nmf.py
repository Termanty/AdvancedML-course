###############################################
#
# AML Exercise 4.3
# Tero Mäntylä
# 014093702
#
# python 3

import numpy as np
import scipy
import urllib
from matplotlib import pyplot as plt


# url = 'https://cs.helsinki.fi/u/aklami/teaching/AML/exercise_4.csv'
# data = np.loadtxt(urllib.request.urlopen(url), delimiter=',')
data = np.loadtxt('exercise_4.csv', delimiter=',')
X = data[:3500]
X_val = data[3500:]
K = 64

def getCost(O, R):
    n = len(O)
    cost = 0
    diff = O - R
    for i in range(n):
        cost += np.dot(diff[i], diff[i]) / n
    return cost


## NMF
W = np.random.rand(3500, K)
H = np.random.rand(K, 256) / 10
iterations = 50
costs = []
for i in range(iterations):
    H_new = H * np.dot(W.T, X) / np.dot(np.dot(W.T, W), H)
    W_new = W * np.dot(X, H_new.T) / np.dot(W, np.dot(H_new, H_new.T))
    costs.append(getCost(X, np.dot(W_new, H)))
    H = H_new
    W = W_new

# validation
costs_validation = []
XH = np.dot(X_val, H.T)
HH = np.dot(H, H.T)
W_val = np.random.rand(500, K)
for i in range(iterations):
    W_new = W_val * XH / np.dot(W_val, HH)
    costs_validation.append(getCost(X_val, np.dot(W_new, H)))
    W_val = W_new


## PCA
X_mean = np.mean(X, axis=0, keepdims=True)
X0 = X - X_mean 
e, EV = np.linalg.eig(np.dot(X0.T, X0))

def sortEigen( evals, evecs ):
    tempvec = [(evals[i],i) for i in range(len(evals))]
    tempvec.sort()
    return [tempvec[i][1] for i in range(len(evecs))]

dsorder = sortEigen( e, EV )
EH = EV[:,dsorder[:K]]
Z_proj = np.dot(X0, EH)
X_reconst = np.dot(Z_proj, EH.T) + X_mean 
cost_PCA_training = getCost(X, X_reconst)

X_val_mean = np.mean(X_val, axis=0, keepdims=True)
X0_val = X_val - X_val_mean
Z_proj_val = np.dot(X0_val, EH)
X_reconst_val = np.dot(Z_proj_val, EH.T) + X_val_mean
cost_PCA_val = getCost(X_val, X_reconst_val)


## Printing costs
print('NMF training error:   ' + str(costs[-1]))
print('NMF validation error: ' + str(costs_validation[-1]))
print('PCA training error:   ' + str(cost_PCA_training))
print('PCA validation error: ' + str(cost_PCA_val))


## Plotting 
plt.figure(1)
plt.plot(range(len(costs)), costs)
plt.plot(range(len(costs_validation)), costs_validation, 'r')

plt.figure(2)
for i in range(8):
    for j in range(8):
        plt.subplot(8,8, i*8+j)
        plt.imshow(H[i*8+j].reshape(16,16), cmap='Greys_r')
        plt.axis('off')

plt.figure(5)
for i in range(8):
    for j in range(8):
        plt.subplot(8,8, i*8+j)
        plt.imshow(EH[:,i*8+j].reshape(16,16), cmap='Greys_r')
        plt.axis('off')


#plt.figure(3)
#plt.imshow(data[0].reshape(16,16), cmap='Greys_r')
#plt.figure(4)
#plt.imshow(np.dot(W[0], H).reshape(16,16), cmap='Greys_r')


plt.show()
