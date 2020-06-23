import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
#from computeCost import normalize,computeCost2,GradientDecent2 #, normalize
import ComputeCost
from mpl_toolkits.mplot3d import Axes3D
#############################################################################################################################
# def sigmoid(z):
#     return 1.0/(1 +  np.e**(-z))

# def costFunction(theta,X,y):
#     m = len(y) 
#     J =(np.sum(-y*np.log(sigmoid(np.dot(X,theta)))-
#        (1-y)*(np.log(1-sigmoid(np.dot(X,theta)))))/m)
#     grad = (np.sum((sigmoid(np.dot(X,theta))-y)[:,None]*X,axis=0)/m)
#     print("gitEXAMS")
#     print(X)
#     print("BEFORE    !")
#     print((sigmoid(np.dot(X,theta))-y)[:,None])
#     print("AFTER      !")
#     print((sigmoid(np.dot(X,theta))-y)[:,None]*X)

#     #exit()
#     return (J, grad)


# data = pd.read_csv('../ex2data1.txt',names=['x1','x2','y'])
# X = np.asarray(data[["x1","x2"]])
# y = np.asarray(data["y"])
# print(X)
# print("This is Y")
# print(y)

# X = np.hstack((np.ones_like(y)[:,None],X))
# initial_theta = np.zeros(3)
# #cost, grad = costFunction(initial_theta, X, y)

# print(X)
# print(initial_theta)
# print(y)


# cost, grad = costFunction(initial_theta, X, y)

# print('Cost at initial theta (zeros): \n', cost)
# print('Gradient at initial theta (zeros): \n',grad)
# exit()
################################################################################################################################3
data = np.genfromtxt('../ex2data1.txt',delimiter=',')
print(data)

Theta = np.zeros(3)
exams,admitted = np.hsplit(data,[2])
exam1, exam2 = np.hsplit(exams,[1])
size = admitted.size

ones = np.ones((len(admitted),1))
print(ones)

exams = np.concatenate((ones,exams),axis=1)

print(exams)
print(admitted)
print(Theta)
#exit()
plt.scatter(exam1,exam2,c=admitted)
#plt.plot(exam1,exam2)


plt.show()

#print(exams.T)
z=np.dot(exams,Theta.T)
print(ComputeCost.hThetaof(z))


print(ComputeCost.JofTheta(Theta,exams,admitted.flatten()))

#print(ComputeCost.JofTheta(admitted,size,z))
#print(ComputeCost.GradDescent(Theta,exams,admitted))

print(Theta)
a = admitted.flatten()
print(a)
print(np.transpose(a))
print("NUMBER  !!!")
#exit()
#temp = opt.fmin_tnc(func=ComputeCost.JofTheta,x0=Theta,fprime=ComputeCost.GradDescent,args=(exams,admitted))
temp = opt.minimize(ComputeCost.JofTheta,Theta,(exams,admitted.flatten()),jac=True,method='TNC',options={'maxiter':400})
print(temp.x)
optTheta=temp.x

#ComputeCost.plotDecisionBoundary(optTheta,exams,admitted)

plot_x = [np.min(exams[:,1]-2), np.max(exams[:,2]+2)]
plot_y = -1/optTheta[2]*(optTheta[0] 
          + np.dot(optTheta[1],plot_x))  
mask = admitted.flatten() == 1
adm = plt.scatter(exams[mask][:,1], exams[mask][:,2])
not_adm = plt.scatter(exams[~mask][:,1], exams[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

z=np.dot([1,45,85],optTheta)
print(ComputeCost.hThetaof(z))