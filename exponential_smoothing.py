#! python3
#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import math
from time import *
from importandexportData import importandexportData

#series for single and double exponential smoothing
#series = [5, 10, 12, 13, 12, 10, 10.5]

#series for triple exponential smoothing
series = [131, 122, 130, 132, 141, 149, 154, 148, 138, 140, 132, 130, 118,
110, 121, 125, 128, 136, 142, 139, 128, 132, 128, 127, 122, 114, 122, 119,
134, 136, 141, 137, 123, 125, 122, 121, 118, 115, 118, 120, 127, 130, 141,
132, 121, 125, 119, 127, 118, 110, 118, 122, 129, 133, 147, 134, 124, 129,
123, 128, 119, 109, 118, 122, 132, 135, 145, 139, 132, 131, 127, 133]

# simple exponential smoothing, neglect sum of weighted
def simple_exponential_smoothing(alpha, num):
    result = []
    sum = 0
    for n in range(1, num+2):
        result.append(math.pow(alpha,n))
        sum+=math.pow(alpha,n)
    print(alpha,sum)
    return result

def rough_exponential_smoothing(alpha, series):
    result = [15]
    for n in range(0, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n])
    return result

def single_exponential_smoothing(alpha, num):
    result = [alpha]
    for n in range(1, num):
        result.append(alpha*math.pow((1-alpha), n))
    return result

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    level, trend = series[0], series[1] - series[0] #intial value for level zero and trend zero
    for n in range(0, len(series)+2):
        if n >= len(series): # we are forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

# for triple exponential smoothing
# Average method to calculate 
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    for j in range(n_seasons):    # compute season averages
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    for i in range(slen):         # compute initial values
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i-1]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            last_trend, trend = trend, beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-last_smooth-last_trend) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

def optimize_param(series, slen, sse_range):
    precision = 0.001             # train precision
    l = 0.0                       # alpha
    b = 0.0                       # beta
    s = 0.0                       # gamma
    SSE = 1e10                    # temporary big value
    result = []                   # forecast result
    begin_time = time()           # record time
    for alpha in np.arange(0+precision,1-precision,precision):
        for beta in np.arange(0+precision,1-precision,precision):
            for gamma in np.arange(0+precision,1-precision,precision):
                temp = 0
                result = triple_exponential_smoothing(series, slen, alpha, beta, gamma, 0)
                for i in range(len(series)-sse_range*slen,len(series)):
                    temp += math.pow((result[i] - series[i]), 2)
                if temp < SSE:
                    SSE = temp
                    l = alpha
                    b = beta
                    s = gamma
            print("Current alpha:", alpha, "beta:", beta, "gamma:", gamma, "param:", l,b,s, "SSE=", SSE, "temp:", temp)
    end_time = time()
    print("Final param is:",l,b,s, "train time is:", end_time-begin_time)
    return {l,b,s}

# simple_exponential_smoothing with different alpha
# plt.title('Simple Exponential Smoothing')
# plt.plot(simple_exponential_smoothing(0.8, 8), 'b--o', label="Alpha = 0.8")
# plt.plot(simple_exponential_smoothing(0.6, 8), 'm--o', label="Alpha = 0.6")
# plt.plot(simple_exponential_smoothing(0.5, 8), 'c--o', label="Alpha = 0.5")
# plt.plot(simple_exponential_smoothing(0.4, 8), 'g--o', label="Alpha = 0.4")
# plt.plot(simple_exponential_smoothing(0.2, 8), 'r--o', label="Alpha = 0.2")

# single_exponential_smoothing with different alpha
# plt.title('Single Exponential Smoothing')
# plt.plot(single_exponential_smoothing(0.8, 8), 'b--o', label="Alpha = 0.8")
# plt.plot(single_exponential_smoothing(0.6, 8), 'm--o', label="Alpha = 0.6")
# plt.plot(single_exponential_smoothing(0.4, 8), 'g--o', label="Alpha = 0.4")
# plt.plot(single_exponential_smoothing(0.2, 8), 'r--o', label="Alpha = 0.2")

# double_exponential_smoothing with different alpha
# plt.title('Double Exponential Smoothing')
# plt.plot(series, 'o-', linewidth=2, label="Original series")
# plt.plot(double_exponential_smoothing(series, 0.8, 0.8), 'b--o', linewidth=1, label="Alpha = 0.8, Beta = 0.8")
# plt.plot(double_exponential_smoothing(series, 0.6, 0.6), 'm--o', linewidth=1, label="Alpha = 0.6, Beta = 0.6")
# plt.plot(double_exponential_smoothing(series, 0.4, 0.4), 'g--o', linewidth=1, label="Alpha = 0.4, Beta = 0.4")
# plt.plot(double_exponential_smoothing(series, 0.2, 0.2), 'r--o', linewidth=1, label="Alpha = 0.2, Beta = 0.2")

#triple_exponential_smoothing
plt.title('Triple Exponential Smoothing')
plt.plot(triple_exponential_smoothing(series, 12, 0.041, 0.821, 0.021, 20), 'y.--', linewidth=1, label="Forcarst series, MSE=6.54")
plt.plot(series, 'c.-', linewidth=1, label="Original series")

#National Import and Export Data
# plt.title('National Import and Export Monthly Data')
# plt.plot(importandexportData, 'c.-', linewidth=1, label="Original series")
# plt.plot(triple_exponential_smoothing(importandexportData, 12, 0.071, 0.065, 0.001, 12), 'y.--', linewidth=1, label="Forcarst series, MSE=0.82")

plt.grid()
plt.legend()
plt.show()

#try to optimize parameters, it takes long time
#optimize_param(series,12,2)
#optimize_param(importandexportData,12,1)
