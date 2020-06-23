import numpy as np
import matplotlib.pyplot as plt

def hThetaof(z):
    return 1/(1+np.exp(-z))

def JofTheta(Theta,exams,admit):
    
    #exit()

#def JofTheta(admit,size,z):
    z=np.dot(exams,Theta.T)
    #print(admit.T)
   # exit()
    size = admit.size
    #print(size)
    #exit()
    cost = (1/size)*(np.sum(np.multiply(-admit,np.log(hThetaof(z)))-(np.multiply((1-admit ),np.log(1-hThetaof(z))))))
    print(Theta)
    print(exams)
    print(admit)
    print(z)
    print(cost)
    #exit()
#def GradDescent(Theta,exams,admit):
    z=np.dot(exams,Theta.T)
    size = admit.size
    print(z)
    print("EXAMS")
    print(exams)
    print(admit)
    print(hThetaof(z))
    HTofX = (hThetaof(z)-admit)
    ar = np.asarray(HTofX)
    arT=np.transpose(np.matrix(ar))
    print(arT)
    print(arT.T)
    print()
    print(arT.T*exams)
    #exit()
    gradient = (1/size)*(arT.T*exams)
    print("THis is THE Gradient")
    print(gradient)
    #exit()
    return cost, gradient

##############################################################################################
def mapFeatureVector(X1,X2):
    """
    Feature mapping function to polynomial features. Maps the two features
    X1,X2 to quadratic features used in the regularization exercise. X1, X2
    must be the same size.returns new feature array with interactions and quadratic terms
    """
    
    degree = 6
    output_feature_vec = np.ones(len(X1))[:,None]

    for i in range(1,7):
        for j in range(i+1):
            new_feature = np.array(X1**(i-j)*X2**j)[:,None]
            output_feature_vec = np.hstack((output_feature_vec,new_feature))
   
    return output_feature_vec

def plotData(X, y):
    pos = X[np.where(y==1)]
    neg = X[np.where(y==0)]
    fig, ax = plt.subplots()
    ax.plot(pos[:,0],pos[:,1],"k+",neg[:,0],neg[:,1],"yo")
    return (fig, ax)


def plotDecisionBoundary(theta,X,y):
    """X is asssumed to be either:
        1) Mx3 matrix where the first column is all ones for the intercept
        2) MxN with N>3, where the first column is all ones
    """
    fig, ax = plotData(X[:,1:],y)
    """
    if len(X[0]<=3):
        # Choose two endpoints and plot the line between them
        plot_x = np.array([min(X[:,1])-2,max(X[:,2])+2])
        ax.plot(plot_x,plot_y)
        ax.legend(['Admitted','Fail','Pass'])
        ax.set_xbound(30,100)
        ax.set_ybound(30,100)
    else:
    """

    # Create grid space
    u = np.linspace(-1,1.5,50)
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    
    # Evaluate z = theta*x over values in the gridspace
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = np.dot(mapFeatureVector(np.array([u[i]]),
		      np.array([v[j]])),theta)
    
    # Plot contour
    ax.contour(u,v,z,levels=[0])

    return (fig,ax)











