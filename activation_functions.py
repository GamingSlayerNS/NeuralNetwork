import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    # return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    return np.tanh(x)


def tanhPrime(x):
    return 1-np.tanh(x)**2


def relu(x):
    return np.maximum(0, x)


def reluPrime(x):
    return np.where(x > 0, 1, 0)
    # return (x > 0).astype(x.dtype)


def leakyRelu(x, alpha=0.1):
    return max(x, alpha * x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))