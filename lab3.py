import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

l = 100

iris = ds.load_iris()
X = iris.data[:l]
Y = iris.target[:l] * 2 - 1

X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

#print(X)
#print(Y)

w = np.zeros(5);

eps = 0.005
n = 0.01
prev_w = w

dw = eps
while(dw >= eps):
    prev_w = w.copy()
    for j in range(5):
        sum = 0
        for i in range(l):
            sum += X[i,j] * Y[i] * sigmoid(-Y[i]*w.dot(X[i]))
        w[j] = w[j] + n * (1 / l) * sum
        
    dw = math.sqrt(((prev_w - w)**2).sum())
    
print(X.dot(w))
print(np.array([1, 5.7, 2.8, 4.1, 1.3]).dot(w))

