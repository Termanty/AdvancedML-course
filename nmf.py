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


url = 'https://cs.helsinki.fi/u/aklami/teaching/AML/exercise_4.csv'
# data = np.loadtxt(urllib.request.urlopen(url), delimiter=',')
data = np.loadtxt('exercise_4.csv', delimiter=',')

X = data[:3500]
X_test = data[3501:]

K = 64
W = np.random.rand(3500, K)
H = np.random.rand(K, 256) / 10

costs = []
for i in range(2000):
    H_new = H * np.dot(W.T, X) / np.dot(np.dot(W.T, W), H)
    W_new = W * np.dot(X, H_new.T) / np.dot(W, np.dot(H_new, H_new.T))
    diff = X - np.dot(W, H)
    cost = 0
    for i in range(3500):
        cost += np.dot(diff[i], diff[i])
    costs.append(cost)
    print(cost)
    H = H_new
    W = W_new



plt.figure(1)
plt.plot(range(len(costs)), costs)


plt.figure(2)
for i in range(8):
    for j in range(8):
        plt.subplot(8,8, i*8+j)
        plt.imshow(H[i*8+j].reshape(16,16), cmap='Greys_r')
        plt.axis('off')

plt.figure(3)
plt.imshow(data[0].reshape(16,16), cmap='Greys_r')
plt.figure(4)
plt.imshow(np.dot(W[0], H).reshape(16,16), cmap='Greys_r')


plt.show()
