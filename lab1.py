import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.optimize as so

#1
A = np.array([[1, 2, 3], [-2, 3, 0], [5, 1, 4]])
B = np.array([[1, 2, 3], [2, 4, 6], [3, 7, 2]])
u = np.array([-4, 1, 1])
v = np.array([3, 2, 10])

#2
C = np.random.random((100, 100))
w = np.random.random((100, 1))

#3
print(A + B) # A + B
print(np.dot(A, B)) # AB
print(A.dot(A).dot(A)) # A^3
print(A.dot(B).dot(A)) # ABA
print(np.dot(v.T, A.T).dot(u + 2*v)) # v^(T)*A^(T)*(u+2v)
print(u.dot(v)) # u*v
print(C.dot(w)) # Cw
print(w.T.dot(C)) # w^(T)*C

#4
t = np.array(range(0, 20))
t = np.tile(t, (20, 1))
result = t * t.T
print(result)


#5
sio.savemat('matrixes', {"A": A, "B": B, "u": u, "v": v})
checkMatA = sio.loadmat('matrixes')["A"]
print(checkMatA)

#6
sumResult = 0
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        t = A[i, j]
        if t > 0:
            sumResult += t
            
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        t = B[i, j]
        if t > 0:
            sumResult += t

print(sumResult)

#7
oneRow = []
for row in A:
    for el in row:
        oneRow.append(el)
        
oneRowStep2 = []
for i in range(1, len(oneRow), 2):
    oneRowStep2.append(oneRow[i])
    
print(oneRow)
print(oneRowStep2)

#8
invA = np.linalg.inv(A)
#invB = np.linalg.inv(B)
#invC = np.linalg.inv(C)
pinvA = np.linalg.pinv(A)
pinvB = np.linalg.pinv(B)
pinvC = np.linalg.pinv(C)

print(A.dot(invA), '\n')
print(B.dot(pinvB), '\n') #?
print(C.dot(pinvC), '\n')

#9
A = np.array([[32, 7, -6], [-5, -20, 3], [0, 1, -3]])
b = np.array([12, 3, 7])
x = np.linalg.solve(A, b)
print("Корни: ", x)
print("Проверка: ", A.dot(x))

#10
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Собственные векторы (столбец - вектор):\n", eigenvectors, '\n')
print("Собственные значения:\n", eigenvalues, '\n')

#11
def f1(x):
    return 5 * (x - 2)**4 - 1 / (x**2 + 8)
def f2(x):
    return 4 * (x[0] - 3*x[1])**2 + 7 * x[0] ** 4

min = so.minimize(f1, 0.0, method='BFGS')
print('min f1(x) =', min.x)
print('Проверка: f1(x) =', f1(min.x))

min = so.minimize(f2, [5.0, 2.0], method='BFGS')
print('min f2(x) =', min.x)
print('Проверка: f2(x) =', f2(min.x))

#12
def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

x = np.linspace(-5, 5, 50)
g1y = g1(x)
g2y = g2(x)

plt.plot(x, g1y)
plt.plot(x, g2y)

plt.xlabel('x')
plt.ylabel('g1(x), g2(x)')
plt.legend(['g1', 'g2'])
plt.title('Plot')

plt.show()

#13
def g1(x):
    return x**5 - 2*x**4 + 3*x - 7
def g2(x):
    return x**5 + 2*x**4 - 3*x - 7

plt.subplot(1, 3, 1)
plt.plot(x, g1y)
plt.xlabel('x')
plt.ylabel('g1(x)')
plt.legend(['g1'])
plt.title('Plot #1')

plt.subplot(1, 3, 3)
plt.plot(x, g2y)
plt.xlabel('x')
plt.ylabel('g2(x)')
plt.legend(['g2'])
plt.title('Plot #2')

plt.show()

#14
g1x0 = so.brentq(g1, -5, 5)
g2x0 = so.brentq(g2, -5, 5)

print(g1x0)
print(g2x0)



