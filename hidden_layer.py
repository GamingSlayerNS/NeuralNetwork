import numpy as np


class HiddenLayer:
    def __init__(self, inputSize, outputSize):
        self.input = None
        self.output = None
        self.weights = np.random.rand(inputSize, outputSize) - 0.5
        self.bias = np.random.rand(1, outputSize) - 0.5

    def forwardPropagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningRate):
        inputError = np.dot(outputError, self.weights.T)
        weightError = np.dot(self.input.T, outputError)
        # Update Weights
        # print("Updating Weights...")
        self.weights -= learningRate * weightError
        self.bias -= learningRate * outputError
        return inputError
