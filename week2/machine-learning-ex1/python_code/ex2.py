import numpy as np
import matplotlib.pyplot as plt
from computeCost import normalize,computeCost2,GradientDecent2 #, normalize
from mpl_toolkits.mplot3d import Axes3D

initTheta = np.zeros((3,1))

print(initTheta)


#the below will read in on delimiter in this case a comma
data = np.genfromtxt('ex1data2.txt',delimiter=',')
print(data)

#the below function splits the data array into 2 arrays, along 1 axis
xs,ys=np.hsplit(data,[2])
print(xs)
print(ys)
print(xs[:,1])
print(len(xs))
print(len(ys))

fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xs[:,1],xs[:,0],ys,c=ys)
plt.show()

#exit()
means = np.mean(xs,axis=0)
print(means)



bdrm_mean = means[1]
area_mean = means[0]
print(area_mean)

g = (xs-np.mean(xs))/np.std(xs)
print(g)
#exit()

# areas,bedrooms = np.hsplit(xs,2)
# print("This is xs")
# print(xs)
# Xs = (xs-np.mean(xs))/np.std(xs)
# print(Xs)

# norm_areas = normalize(areas,area_mean)
# norm_bedrooms = normalize(bedrooms,bdrm_mean)
# print(" The following are normalized areas and bedrooms /n")
# print(norm_areas)
# print(norm_bedrooms)



# a=np.asarray(norm_areas)
# b = np.asarray(norm_bedrooms)
# print("normalized values \n")
# ar = a.reshape((len(a),1))
# br = b.reshape((len(a),1))
# print(ar)
# print(br)
# #exit()
# normarr = np.concatenate((ar,br),axis=1)
# print(normarr)

ones = np.ones((len(xs),1))
print(ones)

NewXs = np.concatenate((ones,g),axis=1)
#print(normarr)
print(g)
print(NewXs)
print(initTheta)
#exit()

costs=[]
cost = computeCost2(NewXs,ys,initTheta)
costs.append(cost)

J_his=[]

newTheta,J_his=GradientDecent2(NewXs,ys,initTheta,0.01,400)

print("J history")
print(J_his)
print(newTheta)

plt.plot(J_his)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.show()


x_s = normalize(np.array([1650,3]), np.mean(xs))
x_s = np.append(np.ones(1),x_s)
x_sample = np.asarray(x_s)
print(x_sample)

price_prediction = np.dot(newTheta.T,x_sample)
print(price_prediction)
exit()


print(norm_areas)
print(len(norm_areas))