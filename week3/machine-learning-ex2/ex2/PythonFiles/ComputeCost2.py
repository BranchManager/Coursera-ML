import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
        
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = 1/(1+np.exp(-z))

    # =============================================================
    return g
    #return 1/(1+np.exp(-z))

#def JofTheta(Theta,exams,admit):
    
    #exit()

#def JofTheta(admit,size,z):
 #   z=np.dot(exams,Theta.T)
    #print(admit.T)
   # exit()
  #  size = admit.size
    #print(size)
    #exit()
   # cost = (1/size)*(np.sum(np.multiply(-admit,np.log(hThetaof(z)))-(np.multiply((1-admit ),np.log(1-hThetaof(z))))))
    #print(Theta)

def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset. 
    
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """
    # Create New Figure
    fig = plt.figure()

    # ====================== YOUR CODE HERE ======================
    print(X)

    X1, X2 = np.hsplit(X,[1])
    plt.scatter(X1,X2,c=y)
    plt.show()
    # ============================================================


def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total 
        number of polynomial features. 
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    
    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================
    z=np.dot(X,theta.T)
    #cost = (1/size)*(np.sum(np.multiply(-admit,np.log(hThetaof(z)))-(np.multiply((1-admit ),np.log(1-hThetaof(z))))))
    #A = np.multiply(-y,np.log(sigmoid(z)))
    #E = np.multiply((1-y),np.log(1-sigmoid))
    #exit()

    B = (np.sum(np.multiply(-y,np.log(sigmoid(z)))-np.multiply((1-y),np.log(1-sigmoid(z)))))
    J = (1/m)*B+np.multiply((lambda_/(2*m)),(np.square(theta)))


    grad = ((1/m)*(np.sum(sigmoid(z)-y)))+((lambda_/m)*theta)

    
    
    
    # =============================================================
    return J, grad

def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.
    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.
    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.
    degree: int, optional
        The polynomial degree.
    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)