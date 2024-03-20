import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    # return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    return np.tanh(x)


def tanhPrime(x):
    return 1-np.tanh(x)**2


def relu(x):
    return max(0, x)


def leakyRelu(x, alpha=0.1):
    return max(x, alpha * x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))