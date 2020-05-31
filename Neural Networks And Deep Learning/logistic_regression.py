import numpy as np
import matplotlib.pyplot as plt
from util_funcs import *
from sklearn.metrics import accuracy_score

class LogisticRegressor():
    """
    Creates a Logistic Regression model with the specified parameters

    Parameters:
              X --- Feature Matrix of the training set with training examples as columns and features as rows
              y --- A vector containing corresponding outputs for the training set stacked in rows
          alpha --- learning rate of the model to apply in gradient descent
        n_iters --- No. of iterations to apply gradient descent algorithm for
     print_cost --- Boolean that determines whether to print the cost function on every 100 iterations

    Available methods:
    
    """
    def __init__(self, X, y, alpha, n_iters, print_cost=True, plot_cost_funcs = False):
        self.X=  X
        self.y = y
        self.m = X.shape[1]
        self.alpha = alpha
        self.n_iters = n_iters
        # Weights
        self.W = np.zeros(shape=(X.shape[0], 1))
        # Bias
        self.b = 0
        self.print_cost = print_cost
        self.plot_cost_funcs = plot_cost_funcs
        
        self.learn()
    
    def linear_activation_forward(self):
        """
        Calculates the Sigmoid Function with respect to the given X
        """
        Z = np.dot(self.W.T, self.X)+ self.b
        A = sigmoid(Z)

        return (Z, A)

    def linear_backward(self, A, y):
        """
        Executes backpropagation and returns the derivatives of weights and bias
        which is used for gradient descent update
        """
        dZ = A.T-y
        db = (1/self.m)*np.sum(dZ)
        dW = (1/self.m)*np.dot(self.X, dZ)
        return db, dW

    def cost_function(self, A, y):
        """
        calculates the cost function of the activations 
        """
        J = np.sum(y*np.log(A))+(1-y)*np.log(1-A)
        return -J[0]/self.m
        
    def learn(self):
        """
        Learning the data and applying forward and backward propagations
        for the specified no. of iterations
        """
        for i in range(self.n_iters):
            Z, A = self.linear_activation_forward()
            cost = self.cost_function(A, self.y)
            db, dW = self.linear_backward(A, self.y)
            self.W = self.W - self.alpha*dW
            self.b = self.b - self.alpha*db

            if i%100==0 and self.print_cost:
                    print("Cost Function at iteration", i, "=", cost[0])
            
    def predict_proba(self, X_test):
        """
        Returns the probability that each training example belongs to class 1  
        """
        y_test_pred = []
        for i in X_test.T:
            test_pred = sigmoid(np.dot(self.W.T, i)+self.b)
            y_test_pred.append(test_pred[0])
    
        return np.array(y_test_pred)
    
    def predict(self, X_test):
        """
        Returns the prediction of the model on the given test data
        """
        y_test_pred = self.predict_proba(X_test)
        return y_test_pred > 0.5
    
    def accuracy(self, y_true, y_pred):
        """
        returns the accuracy in percentage using sklearn.metrics.accuracy_score()
        """
        return str(accuracy_score(y_true, y_pred)*100)+"%"
