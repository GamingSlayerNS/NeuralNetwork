import numpy as np


class Layer:
    def __init__(self, inputSize, outputSize):
        self.input = None
        self.output = None
        self.weights = np.random.rand(inputSize)

    def forwardPropagation(self, input):
        print("Initiating BackwardPropagation...")

    def backwardPropagation(self, outputError, learningRate):
        print("Initiating ForwardPropagation...")
