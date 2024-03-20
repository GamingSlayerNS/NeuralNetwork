class OldNeuralNetwork:
    def __init__(self, numInNodes, numHiddenNodes, numOutNodes, learningRate, activationFunction):
        self.numInNodes = numInNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutNodes = numOutNodes
        self.learningRate = learningRate
        self.activationFunction = activationFunction
        self.weightMatrices()


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.weight = None
        self.deltaWeight = None

    def add(self, layer):
        self.layers.append(layer)

    def useActivationFunction(self, weight, deltaWeight):
        self.weight = weight
        self.deltaWeight = deltaWeight

    def train(self, xTrain, yTrain, epochs, learningRate):
        nodes = len(xTrain)

        # Initiate Train
        print("Start Training...")
        for i in range(epochs):
            errorForward = 0
            for j in range(nodes):
                # Initiate ForwardPropagation
                output = xTrain[j]
                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                # Initiate BackwardPropagation
                errorBackward = self.layers

    def classify(self, inputData):
        nodes = len(inputData)
        final = []

        for i in range(nodes):
            # Initiate ForwardPropagation
            output = inputData[i]
            for layer in self.layers:
                output = layer.forwardPropagation(output)
            final.append(output)
        return final
