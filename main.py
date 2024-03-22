# This is an implementation of a simple Neural Network. Naxel Santiago & Danny Bao
# Press Shift+F10 to execute it in PyCharm.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import numpy as np
import pandas
from sklearn.datasets import load_iris

from network import Network
from activation_functions import relu, reluPrime, sigmoid, sigmoidPrime, tanh, tanhPrime
from activation_layer import ActivationLayer
from hidden_layer import HiddenLayer
from error import mse, derivativeMSE

from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = load_iris()


class NeuralNetwork:
    def __init__(self, numInNodes, numHiddenNodes1, numHiddenNodes2, numOutNodes, activationFunction, learningRate, momentum, epochs):
        self.network = Network()
        self.numInNodes = numInNodes
        self.numHiddenNodes1 = numHiddenNodes1
        self.numHiddenNodes2 = numHiddenNodes2
        self.numOutNodes = numOutNodes
        self.activationFunction = activationFunction
        self.learningRate = learningRate
        self.momentum = momentum
        self.epochs = epochs

    def cleanData(self, xFeatures, yClass):
        # Cleanup Iris data

        # OldInputData
        # trainingDataX = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        # trainingDataY = np.array([[[0]], [[1]], [[1]], [[0]]])
        print(xFeatures.reshape(xFeatures.shape[0], 1, xFeatures.shape[1]))
        print(yClass.reshape(yClass.shape[0], 1, 1))
        reshapedX = xFeatures.reshape(xFeatures.shape[0], 1, xFeatures.shape[1])
        reshapedY = yClass.reshape(yClass.shape[0], 1, 1)
        return reshapedX, reshapedY

    def initiateNetwork(self):
        # Create NeuralNetwork
        self.network.add(HiddenLayer(self.numInNodes, self.numHiddenNodes1))
        self.network.add(ActivationLayer(sigmoid, sigmoidPrime))
        self.network.add(HiddenLayer(self.numHiddenNodes1, self.numHiddenNodes2))
        self.network.add(ActivationLayer(sigmoid, sigmoidPrime))
        self.network.add(HiddenLayer(self.numHiddenNodes2, self.numOutNodes))
        self.network.add(ActivationLayer(sigmoid, sigmoidPrime))

    def trainModel(self, trainingDataX, trainingDataY):
        # Train Model
        self.network.useErrorFunction(mse, derivativeMSE)
        self.network.train(trainingDataX, trainingDataY, epochs=self.epochs, learningRate=self.learningRate,
                           momentum=self.momentum)

    def testModel(self, trainingDataX):
        # Test Model
        self.outputLayer = self.network.classify(trainingDataX)

    def renderNeuralNetwork(self):
        # Display Output
        print("")
        print(self.outputLayer)


# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    neuralNetwork = NeuralNetwork(numInNodes=4, numHiddenNodes1=8, numHiddenNodes2=4, numOutNodes=1,
                                  activationFunction=1, learningRate=0.1, momentum=0, epochs=100)

    # Reduce data to two classes
    yClass = iris.target[iris.target != 2]
    xFeatures = iris.data[iris.target != 2]
    trainingDataX, trainingDataY = neuralNetwork.cleanData(xFeatures, yClass)

    # metadata
    # print(iris.metadata)

    # variable information
    # print(iris.variables)

    neuralNetwork.initiateNetwork()
    neuralNetwork.trainModel(trainingDataX, trainingDataY)
    neuralNetwork.testModel(trainingDataX)
    neuralNetwork.renderNeuralNetwork()

