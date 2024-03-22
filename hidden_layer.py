import numpy as np


class HiddenLayer:
    def __init__(self, inputSize, outputSize, name):
        # Relu uses He Weight Activation
        self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2 / inputSize)
        self.bias = np.zeros((1, outputSize))
        self.deltaWeight = 0
        self.deltaWeightPrevIteration = 0
        # Sigmoid (Line 10) and Tanh (Line 11) use Xavier Weight Activation
        # self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(1 / (inputSize + outputSize))
        # self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(1 / inputSize)
        # self.bias = np.random.rand(1, outputSize) - 0.5
        print(name, " Layer (", inputSize, "x", outputSize, "): ")
        print("Weights: ", self.weights)
        print("Bias: ", self.bias)

    def forwardPropagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backwardPropagation(self, outputError, learningRate, momentum):
        inputError = np.dot(outputError, self.weights.T)
        weightError = np.dot(self.input.T, outputError)
        # Update Weights
        # print("Updating Weights...")
        self.deltaWeight = learningRate * weightError
        self.weights -= self.deltaWeight + (momentum * self.deltaWeightPrevIteration)
        self.deltaWeightPrevIteration = self.deltaWeight
        self.bias -= learningRate * outputError
        return inputError
