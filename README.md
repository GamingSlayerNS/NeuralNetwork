# Simple Neural Network Implementation

This project demonstrates a simple neural network implementation using Python. The network is tested on a modified Iris dataset, reduced to a binary classification task.

## Prerequisites

Ensure you have Python installed on your system. The project requires the following Python libraries:

- numpy
- pandas
- sklearn

You can install these packages using pip:

## How to Run

1. Clone the repository or download the source code.
2. Navigate to the directory containing `main.py`.
3. Run the script using Python:

## Configuration

The neural network is configurable through the `NeuralNetwork` class instantiation in `main.py`. You can set the following parameters:

- `numInNodes`: Number of input nodes.
- `numHiddenNodes1`: Number of nodes in the first hidden layer.
- `numHiddenNodes2`: Number of nodes in the second hidden layer.
- `numOutNodes`: Number of output nodes.
- `activationFunction`: Activation function to use ('sigmoid', 'tanh', 'relu').
- `learningRate`: Learning rate for the network.
- `momentum`: Momentum for the network.
- `epochs`: Number of epochs to train the network.

Example of instantiation:

```python
neuralNetwork = NeuralNetwork(numInNodes=4, numHiddenNodes1=8, numHiddenNodes2=4, numOutNodes=1,
                              activationFunction='sigmoid', learningRate=0.1, momentum=0.75, epochs=100)

## Authors

Naxel Santiago
Danny Bao