import numpy as np
import pandas as pd
import math

def normalize(Data, investigated_x):
    _max = max(np.max(Data), np.max(investigated_x))
    _min = min(np.min(Data), np.min(investigated_x))
    
    Data = (Data - _min) / (_max - _min)
    investigated_x = (investigated_x - _min) / (_max - _min)

def range(current_el, investigated_x):
    return [current_el[13], dist(current_el[:13], investigated_x)]

def dist(x, y):
    return math.sqrt(((x - y)**2).sum())

data = pd.read_csv("wine.csv", header=None)

X = np.array(data.values[:,:13])
Y = np.array([data.values[:,13]])

#Сколько брать для сравнения:
k = math.floor(math.sqrt(X.shape[0]))
#исследуемая точка:
undef_x = np.array([12.7, 3.87, 2.4, 23, 101, 2.83, 2.55, .43, 1.95, 2.57, 1.19, 3.13, 463]) 
#ожидаемый класс:
class_expect = 2

#нормализация:
normalize(X, undef_x) 

#добавляем данные о классах:
X = np.concatenate((X, Y.T), axis = 1)

#находим расстояние до каждой точки и сортируем
ranges = [range(t, undef_x) for t in X]
ranges = np.array(sorted(ranges, key=lambda x: x[1]))

#выбираем первые k классов и считаем вероятность
first_k_elements = ranges[:k,:]
probability = first_k_elements[first_k_elements[:,0] == class_expect].shape[0] / first_k_elements.shape[0]

#Выводим ближайшие точки
print(first_k_elements)

print("Вероятность принадлежности к классу:")
print(probability)

