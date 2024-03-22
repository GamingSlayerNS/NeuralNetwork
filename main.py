# This is an implementation of a simple Neural Network. Naxel Santiago & Danny Bao
# Press Shift+F10 to execute it in PyCharm.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import sys

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

    def preprocessData(self, xFeatures, yClass):
        # Example Input Data:
        # reshapedX = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        # reshapedY = np.array([[[0]], [[1]], [[1]], [[0]]])

        # Cleanup Iris data
        print("Dataset Features: ")
        print(xFeatures)
        print("\nDataset Classes: ")
        print(yClass)
        reshapedX = xFeatures.reshape(xFeatures.shape[0], 1, xFeatures.shape[1])
        reshapedY = yClass.reshape(yClass.shape[0], 1, 1)

        # Shuffle Data
        print("\nShuffling Data...")
        indices = np.arange(reshapedX.shape[0])
        np.random.shuffle(indices)
        shuffledX = reshapedX[indices]
        shuffledY = reshapedY[indices]

        # Split Data as 80% for training and 20% for testing
        print("\nSplitting Data 80%/20%...")
        split_size = int(shuffledX.shape[0] * 0.8)
        trainX = shuffledX[:split_size]
        trainY = shuffledY[:split_size]
        testX = shuffledX[split_size:]
        testY = shuffledY[split_size:]

        return trainX, trainY, testX, testY

    def initiateNetwork(self):
        activationFunctionsList = {
            'sigmoid': (sigmoid, sigmoidPrime),
            'tanh': (tanh, tanhPrime),
            'relu': (relu, reluPrime)
        }

        # Create NeuralNetwork
        print("\nGenerating NeuralNetwork...")
        if self.activationFunction not in activationFunctionsList:
            print("Error: Invalid activation function.")
            sys.exit(1)
        else:
            activation, activationPrime = activationFunctionsList[self.activationFunction]
            self.network.add(HiddenLayer(self.numInNodes, self.numHiddenNodes1, "1st Hidden"))
            self.network.add(ActivationLayer(activation, activationPrime))
            self.network.add(HiddenLayer(self.numHiddenNodes1, self.numHiddenNodes2, "2nd Hidden"))
            self.network.add(ActivationLayer(activation, activationPrime))
            self.network.add(HiddenLayer(self.numHiddenNodes2, self.numOutNodes, "Output"))
            self.network.add(ActivationLayer(activation, activationPrime))

    def trainModel(self, trainingDataX, trainingDataY):
        # Train Model
        self.network.useErrorFunction(mse, derivativeMSE)
        self.network.train(trainingDataX, trainingDataY, epochs=self.epochs, learningRate=self.learningRate,
                           momentum=self.momentum)

    def testModel(self, trainingDataX):
        # Test Model
        self.outputLayer = self.network.classify(trainingDataX)

    def renderOutput(self, actualY, type):
        # Display Output
        np.set_printoptions(precision=6, suppress=True)
        output = np.array([item.ravel() for item in self.outputLayer]).T
        actual = actualY.reshape(actualY.shape[0], -1).T
        print("\n", type, "Output Predicted: ")
        print(output)
        print("\n", type, "Actual: ")
        print(actual)

        # Calculate Accuracy
        print("\n", type, "Model Accuracy: ")
        predictedLabels = (output > 0.5).astype(int)
        accuracy = (predictedLabels == actual).sum() / actual.size
        print(f"Accuracy = {accuracy * 100:.2f}%")


# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    neuralNetwork = NeuralNetwork(numInNodes=4, numHiddenNodes1=8, numHiddenNodes2=4, numOutNodes=1,
                                  activationFunction='sigmoid', learningRate=0.1, momentum=0.75, epochs=100)

    # Reduce data to two classes, 100 iris flowers total
    xFeatures = iris.data[iris.target != 2]
    yClass = iris.target[iris.target != 2]
    trainingDataX, trainingDataY, testDataX, testDataY = neuralNetwork.preprocessData(xFeatures, yClass)

    neuralNetwork.initiateNetwork()
    neuralNetwork.trainModel(trainingDataX, trainingDataY)
    neuralNetwork.testModel(trainingDataX)
    neuralNetwork.renderOutput(trainingDataY, type='TrainingData')
    neuralNetwork.testModel(testDataX)
    neuralNetwork.renderOutput(testDataY, type='TestingData')

