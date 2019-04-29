import pandas as pd
import math
import numpy as np

def Sigmoid(x):
  return 1 / (1 + math.exp(-x))

df = pd.read_csv("classif_test_data.txt", header=None)
data = df.values

X = data[:,:4]
Y = data[:,4:] * 2 - 1

l = Y.shape[0]

X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

w = np.zeros(5)

n = 0.1
for k in range(200):
  sum = 0
  for j in range(5):
    for i in range(l):
      sum += Y[i] * X[i, j] * Sigmoid(-Y[i] * w.dot(X[i]))
    
    w[j] = w[j] + n * (1.0 / l) * sum

y_predict = X.dot(w)
y_predict_b = []
for x in y_predict:
  y_predict_b.append(1 if x > 0 else -1)

print(*y_predict_b)

sum = 0
for i in range(l):
  sum += 1 if Y[i] == y_predict_b[i] else 0

A = sum / l
print(A)