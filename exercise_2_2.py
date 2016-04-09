#########################################33
# Exercise 2.2
# Tero Mantyla, 014039702



from __future__ import division
import urllib
import numpy as np
from matplotlib import pyplot as plt

url_train = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_2_2_train.csv'
url_test = 'https://www.cs.helsinki.fi/u/aklami/teaching/AML/exercise_2_2_test.csv'
data_train = np.loadtxt(urllib.urlopen(url_train), delimiter=',')
data_test = np.loadtxt(urllib.urlopen(url_test), delimiter=',')

X = data_train[:,0:-1] 
y = data_train[:,-1]
X_test = data_test[:,0:-1] 
y_test = data_test[:,-1]

X_20 = X[0:20,:]
y_20 = y[0:20]
X_100 = X[0:100,:]
y_100 = y[0:100]
X_180 = X[0:180,:]
y_180 = y[0:180]


def cost_f(theta, X, y, c=0):
    n = len(y)
    p = np.dot(X, theta.T)
    return 1/n * (np.dot((p-y).T, p-y) + c*np.dot(theta.T, theta))

def grad_f(theta, X, y):
    n = len(y)
    return 2/n * np.dot((np.dot(X, theta.T) - y), X)

def grad_desc(X, y, alpha, c=0, stop=0.001):
    n = len(y)
    theta = np.zeros_like(X[0,:])
    thetas = []
    costs = []
    while True:
        cost = cost_f(theta, X, y, c)
        costs.append(cost)
        thetas.append(theta)
        grad = grad_f(theta, X, y)
        theta = (1-alpha*c/n)*theta - alpha*grad
        if np.dot(grad.T, grad) < stop or len(costs) > 10000:
            break
    return [theta, costs, thetas]

def grad_desc_early(X, y, X_va, y_va, alpha, stop=0.001):
    theta = np.zeros_like(X[0,:])
    thetas = [theta]
    costs = [cost_f(theta, X_va, y_va)]     
    while True:
        grad = grad_f(theta, X, y)
        theta = theta - alpha * grad 
        thetas.append(theta)
        cost = cost_f(theta, X_va, y_va)     
        costs.append(cost)
        if (costs[-2] - costs[-1] < stop):
            break 
    return [theta, costs, thetas]

def plotter(name, costs, thetas, sub):
    plt.subplot(sub)
    plt.title(name)
    plt.plot(range(len(costs)), costs, '-b', label='Train error')
    plt.plot(range(len(costs)), [cost_f(t, X_test, y_test) for t in thetas], '-r', label='Test error') 
    plt.legend(loc='upper right')
    


# (a) #############################################################
theta_ERM_20, costs_20, thetas_20 = grad_desc(X_20, y_20, 0.5)
theta_ERM_100, costs_100, thetas_100 = grad_desc(X_100, y_100, 0.5)
theta_ERM_180, costs_180, thetas_180 = grad_desc(X_180, y_180, 0.5)

print("\n-- ERM --")
print("split\t\t# iter\ttrain err\test err")
print("20/180\t\t%d\t%.4f\t\t%.4f" % (len(costs_20), costs_20[-1], cost_f(theta_ERM_20, X_test, y_test)))
print("100/100\t\t%d\t%.4f\t\t%.4f" % (len(costs_100), costs_100[-1], cost_f(theta_ERM_100, X_test, y_test)))
print("180/20\t\t%d\t%.4f\t\t%.4f" % (len(costs_180), costs_180[-1], cost_f(theta_ERM_180, X_test, y_test)))

plt.figure(1)
plotter('ERM 20/180', costs_20, thetas_20, 221)
plotter('ERM 100/100', costs_100, thetas_100, 222)
plotter('ERM 180/20', costs_180, thetas_180, 223)


# (b) #############################################################
X_20_val = X[20:,:]
y_20_val = y[20:]
X_100_val = X[100:,:]
y_100_val = y[100:]
X_180_val = X[180:,:]
y_180_val = y[180:]

theta_early_20, costs_20, thetas_20 = grad_desc_early(X_20, y_20, X_20_val, y_20_val, 0.5)
theta_early_100, costs_100, thetas_100 = grad_desc_early(X_100, y_100, X_100_val, y_100_val, 0.5)
theta_early_180, costs_180, thetas_180 = grad_desc_early(X_180, y_180, X_180_val, y_180_val, 0.5)

print("\n-- early --")
print("split\t\t# iter\ttrain err\test err")
print("20/180\t\t%d\t%.4f\t\t%.4f" % (len(costs_20), costs_20[-1], cost_f(theta_early_20, X_test, y_test)))
print("100/100\t\t%d\t%.4f\t\t%.4f" % (len(costs_100), costs_100[-1], cost_f(theta_early_100, X_test, y_test)))
print("180/20\t\t%d\t%.4f\t\t%.4f" % (len(costs_180), costs_180[-1], cost_f(theta_early_180, X_test, y_test)))

plt.figure(2)
plotter('early 20/180', costs_20, thetas_20, 221)
plotter('early 100/100', costs_100, thetas_100, 222)
plotter('early 180/20', costs_180, thetas_180, 223)


# (c) ############################################################
regularize_coefficients = [0.001, 0.003, 0.01, 0.03, 0.1]
rc_thetas_20 = []
rc_thetas_100 = []
rc_thetas_180 = []
for rc in regularize_coefficients: 
    theta_20, costs_20, thetas_20 = grad_desc(X_20, y_20, 0.5, rc)
    rc_thetas_20.append(theta_20) 
    theta_100, costs_100, thetas_100 = grad_desc(X_100, y_100, 0.5, rc)
    rc_thetas_100.append(theta_100) 
    theta_180, costs_180, thetas_180 = grad_desc(X_180, y_180, 0.5, rc)
    rc_thetas_180.append(theta_180) 

rc_costs_20 = []
rc_costs_100 = []
rc_costs_180 = []
for i in range(len(regularize_coefficients)): 
    rc_costs_20.append(cost_f(rc_thetas_20[i], X_test, y_test))
    rc_costs_100.append(cost_f(rc_thetas_100[i], X_test, y_test))
    rc_costs_180.append(cost_f(rc_thetas_180[i], X_test, y_test))

print("\n-- regularization --")
print("split\t\t0.001\t0.003\t0.01\t0.03\t0.1")
print("20/180\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (rc_costs_20[0],rc_costs_20[1],rc_costs_20[2],rc_costs_20[3],rc_costs_20[4]))
print("100/100\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (rc_costs_100[0],rc_costs_100[1],rc_costs_100[2],rc_costs_100[3],rc_costs_100[4]))
print("180/20\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (rc_costs_180[0],rc_costs_180[1],rc_costs_180[2],rc_costs_180[3],rc_costs_180[4]))

print('\nBest split to use is 180 for training and rest for testing.')
print('This way we get lowest error in test data set - best regularization')

plt.subplot(224)
plt.title("lambdas")
plt.plot(regularize_coefficients, rc_costs_20) 
plt.plot(regularize_coefficients, rc_costs_100) 
plt.plot(regularize_coefficients, rc_costs_180) 
plt.xscale('log')
plt.show()

