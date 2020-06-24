import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os
#from computeCost import normalize,computeCost2,GradientDecent2 #, normalize
import ComputeCost2
from mpl_toolkits.mplot3d import Axes3D



# tests = np.genfromtxt('../ex2data2.txt',delimiter=',')
# print(tests)

# Theta = np.zeros(3)
# tests,admitted = np.hsplit(tests,[2])
# test1, test2 = np.hsplit(tests,[1])
# size = admitted.size

# print(test1)
# print(test2)
# print(admitted)

# plt.scatter(test1,test2,c=admitted)
# #plt.plot(exam1,exam2)
# plt.show()
data = np.loadtxt(os.path.join('..', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

ComputeCost2.plotData(X,y)

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'], loc='upper right')

X = ComputeCost2.mapFeature(X[:,0],X[:,1])




# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = ComputeCost2.costFunctionReg(initial_theta, X, y, lambda_)

#print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('HERE WE ARE')
print(list(map('{:.4f}%'.format,cost)))

print('Expected cost (approx)       : 0.693\n')

print('Gradient at initial theta (zeros) - first five values only:')
print(list(map('{:.4f}%'.format,grad[:5])))
#print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))

print('Expected gradients (approx) - first five values only:')
print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = ComputeCost2.costFunctionReg(test_theta, X, y, 10)

print('------------------------------\n')
print('Cost at test theta    : {:.2f}'.format(cost))
print('Expected cost (approx): 3.16\n')

print('Gradient at test theta - first five values only:')
print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
print('Expected gradients (approx) - first five values only:')
print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

