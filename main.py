# This is an implementation of a simple Neural Network.

# Press Shift+F10 to execute it in PyCharm.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import numpy as np
import pandas

from network import NeuralNetwork
from activation_functions import sigmoid
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
#print(iris.variables)
#print(iris.data)


# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    # neuralNetwork = OldNeuralNetwork(numInNodes=2, numHiddenNodes=4, numOutNodes=2, learningRate=0.6, activationFunction=1)
    neuralNetwork = NeuralNetwork()

    # InputData
    trainingDataX = np.array()
    trainingDataY = np.array()

    # Create NeuralNetwork
    neuralNetwork.add(HiddenLayer())

    # Train Model
    neuralNetwork.useActivationFunction(mse, derivativeMSE())
    neuralNetwork.train(trainingDataX, trainingDataY, epochs=500, learningRate=0.1)

    # Test Model
    outputLayer = neuralNetwork.classify()

    # Display Output
    print(outputLayer)
