#! python3
#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np

series = [5, 10, 12, 13, 12, 10, 10.5]

def simple(series):
    return series[-1]

def average(series, n=None):
    if n is None:
        return average(series, len(series))
    return float(sum(series[-n:]))/n

def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

weights = [0.1, 0.2, 0.3, 0.4]

plt.title('Result')
plt.grid()
plt.plot(series, 'o-')
plt.plot([len(series)-1, len(series)], [series[-1], simple(series)],'o--c', label="Simple")
plt.plot([len(series)-1, len(series)], [series[-1], average(series)], 'g--o', label="Simple Average")
plt.plot([len(series)-1, len(series)], [series[-1], average(series, 5)], 'r--o', label="Moving Average(n=5)")
plt.plot([len(series)-1, len(series)], [series[-1], average(series, 3)], 'm--o', label="Moving Average(n=3)")
plt.plot([len(series)-1, len(series)], [series[-1], weighted_average(series, weights)], 'b--o', label="Weighted Moving Average")
plt.legend()
plt.show()
