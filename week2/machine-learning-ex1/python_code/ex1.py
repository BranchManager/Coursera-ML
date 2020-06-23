import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost, GradientDescent

eye=np.identity(5)
#identity matrix asf floats
print(eye)

eye = np.eye(5,dtype=int)
print(eye) #identity as integers
initTheta = np.zeros((2,1))

#the below will read in on delimiter in this case a comma
data = np.genfromtxt('ex1data1.txt',delimiter=',')
print(data)

#the below function splits the data array into 2 arrays, along 1 axis
xs,ys=np.split(data,2,axis=1)
print(xs)
print(ys)

#exit()
plt.plot(xs,ys,'x')
plt.ylabel('Profit')
plt.xlabel('Population')
#plt.plot()
#plt.show()

costcomputed=computeCost(xs,ys,initTheta)
print(costcomputed)
print(initTheta)

newTheta = GradientDescent(xs,ys,initTheta,0.01,2000)

plt.scatter(xs,ys,marker='x')
yfit = [newTheta[0][0]+newTheta[1][0] * i[0] for i in xs ]
plt.plot(xs,yfit)
#plt.show()
print('hello')

predict1 = np.dot([1,3.5],newTheta)
predict2 = np.dot([1,7],newTheta)

plt.plot([3.5],predict1,'or')
plt.plot([7],predict2,'or')
plt.show()
print(predict1)
print(predict2)


