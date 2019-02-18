import numpy as np
import matplotlib.pyplot as plt

#X = np.array([1, 2, 3, 4, 5])
#Y = np.array([0.1, 0.3, 0.2, 0.4, 0.35])
X = []
Y = []

k = 0
for line in open('data_apple.csv'):
    x, y = line.split(',')
    X.append(k)
    Y.append(float(y))
    k += 1

X = np.array(X)
Y = np.array(Y)

X_mean = X.mean()
X2_mean = (X * X).mean()
Y_mean = Y.mean()
XY_mean = (X * Y).mean()

denominator = X2_mean - X_mean**2
a = (XY_mean - X_mean * Y_mean) / denominator
b = (Y_mean * X2_mean - X_mean * XY_mean) / denominator

Y_predict = a*X+b

plt.scatter(X, Y, color="red")
plt.plot(X, Y_predict, color="blue")
plt.show()