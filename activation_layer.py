class ActivationLayer:
    def __init__(self, activation, activationPrime):
        self.activation = activation
        self.activationPrime = activationPrime

    def forwardPropagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backwardPropagation(self, outputError, learningRate, momentum):
        return self.activationPrime(self.input) * outputError
