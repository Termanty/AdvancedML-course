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
import mnist_load_show as mnist

X, y = mnist.read_mnist_training_data(2000)

print(X.shape)

e, EV = np.linalg.eig(np.dot(X.T, X))

def sortEigen( evals, evecs ):
    tempvec = [(evals[i],i) for i in range(len(evals))]
    tempvec.sort()
    return [tempvec[i][1] for i in range(len(evecs))]

dsorder = sortEigen(e, EV)
EV2 = EV[:,dsorder[:2]]
print(np.std(EV2, axis=0))
print(EV2.shape)
