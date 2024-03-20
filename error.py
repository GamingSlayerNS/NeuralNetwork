import numpy as np


def mse(targetY, predictedY):
    return np.mean(np.power(targetY - predictedY, 2))


def derivativeMSE(targetY, predictedY):
    return (2 * (predictedY - targetY)) / targetY.size
