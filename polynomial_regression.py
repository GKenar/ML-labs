import numpy as np
import matplotlib.pyplot as plt

def Func(x, B, k):
    result = 0
    for i in range(k):
        result += (x**i) * B[i]
    return result

k = 4 #Степень многочлена

#rnd generator
np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

x = np.array(x)
y = np.array(y)


mem = {}
A = []
for i in range(k):
    A.append([])
    for j in range(k):
        if not mem.get(i + j):
            mem[i + j] = (x**(i + j)).mean()
        A[i].append(mem[i + j])
        
A = np.array(A)

C = []
for i in range(k):
    C.append((x**i*y).mean())

C = np.array(C)

B = np.linalg.solve(A, C)

x = np.sort(x)
y_predict = Func(x, B, k)

plt.scatter(x, y, color='m')
plt.plot(x, y_predict, color="red")
plt.show()