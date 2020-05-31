import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
    return np.maximum(X, 0)

def test():
    print("lskdjflskjf")

