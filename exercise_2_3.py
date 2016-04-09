#########################################33
# Exercise 2.2
# Tero Mantyla, 014039702

from __future__ import division
import urllib
import numpy as np
from matplotlib import pyplot as plt
import math

url = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_2_3.csv'
data = np.loadtxt(urllib.urlopen(url), delimiter=',')

X = data[:,0:-1] 
y = data[:,-1]


def cost_f(theta, X, y):
    n = len(y)
    p = np.dot(X, theta.T)
    return 1/n * np.dot((p-y).T, p-y)

def grad_f(theta, X, y):
    n = len(y)
    return 2/n * np.dot((np.dot(X, theta.T) - y), X)

def grad_desc(X, y, alpha, stop=0.01):
    theta = np.array([-0.3, 1.5])
    thetas = []
    costs = []
    while True:
        cost = cost_f(theta, X, y)
        costs.append(cost)
        thetas.append(theta)
        grad = grad_f(theta, X, y)
        theta = theta - alpha*grad
        if np.dot(grad.T, grad) < stop or len(costs) > 1000:
            break
    return [theta, costs, thetas]

def grad_desc_newton(X, y, stop=0.01):
    theta = np.array([-0.3, 1.5])
    H_inv = np.linalg.inv(np.dot(X.T, X))
    thetas = []
    costs = []
    while True:
        cost = cost_f(theta, X, y)
        costs.append(cost)
        thetas.append(theta)
        grad = grad_f(theta, X, y)
        theta = theta - np.dot(H_inv, grad)
        if np.dot(grad.T, grad) < stop or len(costs) > 1000:
            break
    return [theta, costs, thetas]

# Plotting help
def create_contour():
    I = np.arange(-3.0, 3.0, 0.05)
    J = np.arange(-3.0, 3.0, 0.05)
    Z = np.zeros((len(I),len(J)))
    for i in range(len(I)):
        for j in range(len(J)):
            theta = np.array([I[i], J[j]])
            Z[i,j] = cost_f(theta, X, y)
    levels=np.arange(0.1,2.2,0.3)
    I1, J1 = np.meshgrid(I, J)
    plt.contour(I1, J1, Z.T, levels)


#  1.  ###################################################
theta, costs, thetas = grad_desc(X, y, 0.1)

plt.figure(1)
create_contour()
plt.plot([i[0] for i in thetas], [i[1] for i in thetas], 'ro')
print("\nGradient descent with fixed step size: Good value is 0.1")
print("Algorithm used %d iterations" %len(costs))

#  2.  ###################################################
theta, costs, thetas = grad_desc_newton(X, y)
print("\n\nNewton's method: %d iteration" % len(costs))
print('I actually have no idea how and why this works')

plt.figure(2)
create_contour()
plt.plot([i[0] for i in thetas], [i[1] for i in thetas], 'ro')


#  3.  ###################################################
# running out of time to implement this.


plt.show()
