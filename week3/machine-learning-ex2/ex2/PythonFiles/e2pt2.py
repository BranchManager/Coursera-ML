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

