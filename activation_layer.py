class ActivationLayer:
    def __init__(self, activation, activationPrime):
        self.input = None
        self.output = None
        self.activation = activation
        self.activationPrime = activationPrime

    def forwardPropagation(self, input):
        print("Initiating ActivationFunction...")
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backwardPropagation(self, outputError):
        print("Initiating BackwardPropagation...")
        return self.activationPrime(self.input) * outputError
