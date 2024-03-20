# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
from scipy.stats import beta
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


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
    def __init__(self, numInNodes, numHiddenNodes, numOutNodes, learningRate, activationFunction):
        self.numInNodes = numInNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutNodes = numOutNodes
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.weightMatrices()

    def weightMatrices(self):
        rad = 1 / np.sqrt(self.numInNodes)
        X = truncated_normal(mean=0, sd=1, low=rad, upp=rad)
        print(self.numHiddenNodes)
        self.weightsInHidden = X.rvs(self.numHiddenNodes, self.numInNodes)
        rad = 1 / np.sqrt(self.numHiddenNodes)
        X = truncated_normal(mean=0, sd=1, low=rad, upp=rad)
        self.weightsHiddenOut = beta.rvs(self.numOutNodes, self.numHiddenNodes)

    def train(self, inputVector, targetVector):
        pass

    def run(self, inputVector):
        inputVector = np.array(inputVector, ndmin=2).T
        inputHidden = activation_function(self.weightsInHidden @ inputVector)
        outputVector = activation_function(self.weightsHiddenOut @ inputHidden)
        return outputVector

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

# Press the green button in the gutter to run the Neural Network.
if __name__ == '__main__':
    neuralNetwork = NeuralNetwork(numInNodes=2, numHiddenNodes=4, numOutNodes=2, learningRate=0.6, activationFunction=1)
    #neuralNetwork.run([3, 4])


