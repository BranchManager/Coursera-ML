import numpy as np
import math
def computeCost(X,y,theta):
    m = len(y)
    a = 1/(2*m)

    sum = 0

    print('hello')
    print(len(y))
    for i in range(len(y)):
         func = (theta[0][0] + theta[1][0]*X[i][0])-y[i][0]
         #print(func)
         func2 = math.pow(func,2)
         sum = sum+func2
    return a*sum
     #   sum = sum + theta
def GradientDescent(X,y,theta,alpha,I):
     #print(np.dot(X,theta))
     # m = len(y)
     # ones=np.ones((m,1))
     # xx = np.hstack((ones,X))
     # print(xx)
     # print(np.dot(xx,theta))
     m = len(y)
     ones = np.ones((m,1))
     Xx = np.hstack((ones, X))

     for i in range(I):
          #print("i is now "+str(i)+)
          print(computeCost(X,y,theta))
          sumu = 0
          sumu2 = 0
          for l in range(m):
               func = (theta[0][0] + theta[1][0]*X[l][0])-y[l][0]
               #sumu = func + sumu 
               func2 = func*X[0][0]
               func3 = func*X[l][0]

               sumu = sumu + func2
               sumu2 = sumu2+func3
          
          theta[0][0] = theta[0][0] - (alpha/m) *sumu
          theta[1][0] = theta[1][0] - (alpha/m) *sumu2
          #derv=theta-(alpha/m)*func2
          #theta = theta - derv
          print(theta)
          print(i)
          print('cheese')
          #print(Xx)
          #print(theta)
          # temp = np.dot(Xx, theta) - y
          # temp = np.dot(Xx.T, temp)
          # theta = theta - (alpha/m) * temp

          
          print(theta)
     return theta

def normalize(X,mean):
     # x1 = []
     # print(X)
     # newx = X.flatten()
     # std =np.std(newx)
     # print("STD Dev and Mean")
     # print(str(std)+"   "+str(mean))
     # #print(newx)
     # #exit()
     # print("normalize")
     # for i in range(len(newx)):
     #      print(newx[i])
     #      new_norm = (newx[i]-mean)/std
     #      x1.append(new_norm)
     #      print(new_norm)
     # return x1
     newx=X.flatten()
     std = np.std(newx)

     X_norm = (newx-mean)/std
     print(X_norm)
     return X_norm
     #exit()
     

def computeCost2(X,y,theta):
     print("Compute Cost")
     tempTheta = theta.flatten()
     print(X)
     print(y)
     print(tempTheta)
    # exit()

     pred = np.dot(tempTheta,X.T)-y
     print(pred)
     f = np.sum(np.power(pred,2))/(2*len(y))

     print(f)
     return f

def GradientDecent2(X,y,theta,alpha,iter):
     m = len(y)
     J_his=[]
     for i in range(iter):
          temp = np.dot(X, theta) - y
          temp = np.dot(X.T, temp)
          theta = theta - (alpha/m) * temp
          J_his.append(computeCost2(X,y,theta))
     return theta, J_his

      


