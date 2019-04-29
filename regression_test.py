import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

x = []
y = []

boston = ds.load_boston()
x = boston.data
y = boston.target
#
#for line in open('winequalityN.csv'):
#    l = line.split(',')
#    l = np.char.replace(l, "\n", "")
#    l[l == ""] = 0
#    
#    if(l[0] == 'white'):
#        l[0] = 0
#    else:
#        l[0] = 1
#    
#    x.append(l[:12].astype(float))
#    y.append(l[12].astype(float))
#
x = np.array(x)
y = np.array(y)


w = np.linalg.solve(x.T.dot(x), x.T.dot(y))

y_predict = x.dot(w)

R_2 = 1 - (((y - y_predict)**2).sum()) / (((y - y_predict.mean())**2).sum())

print(R_2)

plt.scatter(y, y_predict)
plt.plot([y.min(), y.max()], [y.min(), y.max()], c='r')
plt.show()

plt.hist(x)