import torch

def softmax(x):
    numerator = torch.exp(x)
    denominator = torch.sum(numerator)
    res = numerator / denominator
    return res