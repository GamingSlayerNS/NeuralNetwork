# This is an implementation of a simple Neural Network.

# Press Shift+F10 to execute it in PyCharm.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import numpy as np
import pandas

from network import NeuralNetwork
from activation_functions import relu, reluPrime, sigmoid, sigmoidPrime, tanh, tanhPrime
from activation_layer import ActivationLayer
from hidden_layer import HiddenLayer
from error import mse, derivativeMSE

from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# metadata
print(iris.metadata)

# variable information
print(iris.variables)
# print(iris.data)


# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    # neuralNetwork = OldNeuralNetwork(numInNodes=2, numHiddenNodes=4, numOutNodes=2, learningRate=0.6, activationFunction=1)
    neuralNetwork = NeuralNetwork()

    # Cleanup Iris data

    # InputData
    trainingDataX = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    trainingDataY = np.array([[[0]], [[1]], [[1]], [[0]]])
    # trainingDataX = X[y != 2].copy()
    # trainingDataY = y[y != 2].copy()
    # print(trainingDataX)

    # Create NeuralNetwork
    neuralNetwork.add(HiddenLayer(2, 4))
    neuralNetwork.add(ActivationLayer(relu, reluPrime))
    neuralNetwork.add(HiddenLayer(4, 2))
    neuralNetwork.add(ActivationLayer(relu, reluPrime))
    neuralNetwork.add(HiddenLayer(2, 1))
    neuralNetwork.add(ActivationLayer(sigmoid, sigmoidPrime))

    # Train Model
    neuralNetwork.useErrorFunction(mse, derivativeMSE)
    neuralNetwork.train(trainingDataX, trainingDataY, epochs=1000, learningRate=0.1)

    # Test Model
    outputLayer = neuralNetwork.classify(trainingDataX)

    # Display Output
    print("")
    print(outputLayer)
