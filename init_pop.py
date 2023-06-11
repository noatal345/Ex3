import numpy as np


class model:
    def __init__(self, num_of_layers, num_of_neurons, init_weights=True):
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons
        self.weights = []
        self.biases = []
        self.fitness = 0
        self.elite = False
        if init_weights:
            self.init_weights()

    def init_weights(self):
        for i in range(self.num_of_layers - 1):
            self.weights.append(np.random.uniform(-1, 1, size=(self.num_of_neurons[i], self.num_of_neurons[i + 1])))
            self.biases.append(np.random.uniform(-1, 1, size=(self.num_of_neurons[i + 1])))


def init_pop(pop_size, num_of_layers, num_of_neurons):
    pop = []
    for i in range(pop_size):
        pop.append(model(num_of_layers, num_of_neurons))
    return pop
