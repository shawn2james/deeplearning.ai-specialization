import numpy as np
import matplotlib.pyplot as plt
from util_funcs import *

class LogisticRegressor():
    """
    Creates a Logistic Regression model with the specified parameters

    Parameters:
          X --- Feature Matrix of the training set with training examples as columns and features as rows
          y --- A vector containing corresponding outputs for the training set stacked in rows
      alpha --- learning rate of the model to apply in gradient descent
    n_iters --- No. of iterations to apply gradient descent algorithm for

    Available methods:
    
    """
    def __init__(self, X, y, alpha, n_iters, print_cost=True, plot_cost_funcs = False):
        self.X=  X
        self.y = y
        self.m = X.shape[1]
        self.alpha = alpha
        self.n_iters = n_iters
        # Weights
        self.W = np.zeros(shape=(X.shape[0], 1))*0.01
        # Bias
        self.b = 0
        self.print_cost = print_cost
        self.plot_cost_funcs = plot_cost_funcs

    self.learn()
    
    def linear_activation_forward():
        """
        Calculates the Sigmoid Function with respect to the given X
        """
        Z = np.dot(self.W.T, self.X)+ self.b
        A = sigmoid(Z)

        return (Z, A)

    def linear_backward(A, y):
        dZ = np.sum(A-y)
        db = (1/self.m)dZ
        dW = (1/self.m)*np.dot(self.X, dZ)
        return db, dW

    def cost_function(A, y):
        J = np.sum(np.dot(y, np.log(A))+np.dot((1-y), np.log(1-A))
        return -J/self.m
        
    def learn():
        cost_funcs = []
        for i in self.n_iters:
            Z, A = self.linear_activation_forward()
            cost = self.cost_function(A, y)
            cost_funcs.append(cost)
            db, dW = self.linear_backward(A, y)
            self.W = self.W - alpha*dW
            self.b = self.b - alpha*self.db

            if self.plot_cost_funcs and n_iters%100==0:
                plt.plot(cost_funcs, range(1, n_iters))
                plt.show()    
