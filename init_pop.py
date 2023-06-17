import numpy as np


class Model:
    def __init__(self, num_of_layers, num_of_neurons, af, init_weights=True):
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons
        self.weights = []
        self.biases = []
        self.fitness = 0
        self.elite = False
        if af == "sigmoid":
            self.activation_function = self.sigmoid
        elif af == "relu":
            self.activation_function = self.relu
        elif af == "leaky_relu":
            self.activation_function = self.leaky_relu
        elif af == "sign":
            self.activation_function = self.sign
        else:
            self.activation_function = self.leaky_relu
        if init_weights:
            self.init_weights()

    def init_weights(self):
        for i in range(self.num_of_layers - 1):
            self.weights.append(np.random.uniform(-1, 1, size=(self.num_of_neurons[i], self.num_of_neurons[i + 1])))
            self.biases.append(np.random.uniform(-1, 1, size=(self.num_of_neurons[i + 1])))

    def sigmoid(self, x):
        # the function returns the sigmoid of the input
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def sign(self, x):
        if x > 0:
            return 1
        else:
            return -1


def init_pop(pop_size, num_of_layers, num_of_neurons, activation_function):
    population = []
    for i in range(pop_size):
        population.append(Model(num_of_layers, num_of_neurons, activation_function))
    return population
