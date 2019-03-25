import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

l = 100

iris = ds.load_iris()
X = iris.data[:l,1:3]
Y = iris.target[:l] * 2 - 1

X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.45, random_state = 1)

#print(X)
#print(Y)

w = np.zeros(3);

eps = 0.005
n = 0.01
prev_w = w

dw = eps
while(dw >= eps):
    prev_w = w.copy()
    for j in range(3):
        sum = 0
        for i in range(X_train.shape[0]):
            sum += X_train[i,j] * Y_train[i] * sigmoid(-Y_train[i]*w.dot(X_train[i]))
        w[j] = w[j] + n * (1 / l) * sum
        
    dw = math.sqrt(((prev_w - w)**2).sum())
    

Y_predict = X.dot(w)

colors_red = X[Y_predict > 0]
colors_blue = X[Y_predict <= 0]

x2 = (-w[0] - w[1] * X) / w[2]

plt.scatter(colors_red[:,1], colors_red[:,2], color="red")
plt.scatter(colors_blue[:,1], colors_blue[:,2], color="blue")
plt.plot(X, x2, color="green")
plt.show()
