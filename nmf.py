###############################################
#
# AML Exercise 4.3
# Tero Mäntylä
# 014093702
#
# python 3

import numpy as np
import urllib
from matplotlib import pyplot as plt


url = 'https://cs.helsinki.fi/u/aklami/teaching/AML/exercise_4.csv'
data = np.loadtxt(urllib.request.urlopen(url), delimiter=',')
# data = np.loadtxt('exercise_4.csv', delimiter=',')

print(data.shape)


