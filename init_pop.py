import numpy as np


class model:
    def __init__(self, num_of_layers, num_of_neurons):
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons
        self.weights = []
        self.biases = []
        self.fitness = 0
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_of_layers - 1):
            self.weights.append(np.random.randn(self.num_of_neurons[i], self.num_of_neurons[i + 1]))
            self.biases.append(np.random.randn(self.num_of_neurons[i + 1])) # todo: maybe change shape


def init_pop(pop_size, num_of_layers, num_of_neurons):
    pop = []
    for i in range(pop_size):
        pop.append(model(num_of_layers, num_of_neurons))
    return pop
