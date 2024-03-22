class Network:
    def __init__(self):
        self.layers = []
        self.errorFunction = None
        self.deltaErrorFunction = None

    def add(self, layer):
        self.layers.append(layer)

    def useErrorFunction(self, errorFunction, deltaErrorFunction):
        self.errorFunction = errorFunction
        self.deltaErrorFunction = deltaErrorFunction

    def train(self, trainingDataX, trainingDataY, epochs, learningRate, momentum):
        nodes = len(trainingDataX)

        # Initiate Train
        print("Start Training...")
        for i in range(epochs):
            errorForward = 0
            # Initiate ForwardPropagation
            # print("Forwardpass...")
            for j in range(nodes):
                output = trainingDataX[j]
                for layer in self.layers:
                    output = layer.forwardPropagation(output)

                # Calculate Error
                errorForward += self.errorFunction(trainingDataY[j], output)

                # Initiate BackwardPropagation
                # print("Backwardpass...")
                errorBackward = self.deltaErrorFunction(trainingDataY[j], output)
                for layer in reversed(self.layers):
                    errorBackward = layer.backwardPropagation(errorBackward, learningRate, momentum)

            errorForward /= nodes
            print('epoch: %d/%d   Training Loss: %f' % (i+1, epochs, errorForward))

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
