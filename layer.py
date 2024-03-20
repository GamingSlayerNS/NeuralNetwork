class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forwardPropagation(self, input):
        print("Initiating ForwardPropagation...")

    def backwardPropagation(self, outputError, learningRate):
        print("Initiating BackwardPropagation...")
