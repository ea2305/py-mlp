import numpy as np
from random import random

# Simple MLP Implementation
class Mlp:
    def __init__(self, numInputs = 3, numHidden = [3, 5], numOutputs = 2):
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs
        self.weights = []
        self.activations = []
        self.derivatives = []

        layers = [self.numInputs] + self.numHidden + [self.numOutputs]

        # weights configuration
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(weight)

            derivative = np.zeros((layers[i], layers[i + 1]))
            self.derivatives.append(derivative)

        for i in range(len(layers)):
            activation = np.zeros(layers[i])
            self.activations.append(activation)      

    def fwdPropagation(self, inputs):
        activations = inputs
        self.activations[0] = inputs

        for (i, weight) in enumerate(self.weights):
            # inputs
            netInputs = np.dot(activations, weight)
            # activations
            activations = self.activationFunction(netInputs)
            self.activations[i + 1] = activations

        return activations
    
    def backpropagation(self, error, debug = False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]

            delta = error * self.sigmoid_derivative(activations)
            delta_reshape = delta.reshape(delta.shape[0], -1).T

            current_activation = self.activations[i]
            current_activation_reshape = current_activation.reshape(current_activation.shape[0], -1)
            self.derivatives[i] = np.dot(current_activation_reshape, delta_reshape)
            error = np.dot(delta, self.weights[i].T)

            if debug:
                print("derivatives for W{}: {}".format(i, self.derivatives[i]))
        
        return error

    def gradientDescent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivative = self.derivatives[i]
            weights += derivative * learning_rate

    def training(self, inputs, targets, epochs, learningRate, debug = False):
        for i in range(epochs):
            sumError = 0
            for input, target in zip(inputs, targets):
                output = self.fwdPropagation(input)
                error = target - output
                self.backpropagation(error)
                self.gradientDescent(learningRate)

                # report error
                sumError += self._mse(target, output)
            if debug:
                print("Error: {} at epoch {}".format(sumError / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output)**2)

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def activationFunction(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])
    # create mlp
    mlp = Mlp(2, [5], 1)

    # train
    mlp.training(inputs, targets, 100, 0.1)

    #dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.fwdPropagation(input)
    print("")
    print("Network x:{} + y:{} = {}".format(input[0], input[1], output[0]))