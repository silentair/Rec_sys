﻿import numpy as np

''' normalization '''
def Normalize(vector,param1,param2,method='min-max'):
    if method == 'min-max':
        maxVal = param1
        minVal = param2
        normalized_data = (vector - minVal) / (maxVal - minVal)
    elif method == 'z-score':
        mean = param1
        var = param2
        normalized_data = (vector - mean) / var
    else:
        print('please check your choose the correct normalization method')
        raise ValueError
    return normalized_data

def Denormalize(vector,param1,param2,method='min-max'):
    if method == 'min-max':
        maxVal = param1
        minVal = param2
        denormalized_data = vector * (maxVal - minVal) + minVal
    elif method == 'z-score':
        mean = param1
        var = param2
        denormalized_data = vector * var + mean
    else:
        print('please check your choose the correct normalization method')
        raise ValueError
    return denormalized_data

''' similarity '''
def Euclidean_sim(vector_a,vector_b):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return 1 / (1 + np.linalg.norm(vector_a-vector_b))

def Pearson_sim(vector_a,vector_b):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    #var_a = np.std(tf_a)
    #var_b = np.std(tf_b)
    #cov_ab = np.cov(a,b,ddof=0)

    return 0.5 * np.corrcoef(vector_a,vector_b)[0][1] + 0.5

def Cosin_sim(vector_a,vector_b):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return np.sum(vector_a * vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))

''' evaluation '''
def MAE(pred,real):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return np.sum(np.abs(pred-real)) / len(pred)

def RMAE(pred,real):
    if len(pred) != len(real):
        print('inputs must have same size!')
        raise ValueError
    return np.sqrt(np.sum((pred-real)**2) / len(pred))