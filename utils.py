import numpy as np

def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    res = numerator / denominator
    return res

def sigmoid(x):
    numerator = 1
    denominator = 1 + np.exp(-x) 
    res = numerator / denominator
    return res