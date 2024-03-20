import math
import numpy as np


def sigmoid(self, x):
    return 1 / (1 + math.exp(-x))


def tanh(self, x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def relu(self, x):
    return max(0, x)


def leakyRelu(self, x, alpha=0.1):
    return max(x, alpha * x)


def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x))