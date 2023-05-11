import numpy as np

def tan_h():
    pass

def softmax(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    res = numerator / denominator
    return res