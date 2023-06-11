import numpy as np


class model:
    def __init__(self, num_of_layers, num_of_neurons):
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons
        self.weights = np.array([])
        self.biases = np.array([])
        self.fitness = 0
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_of_layers - 1):
            new_weights = np.random.uniform(-1, 1, (self.num_of_neurons[i], self.num_of_neurons[i + 1]))
            new_biases = np.random.uniform(-1, 1, (1, self.num_of_neurons[i + 1]))
            self.weights = np.concatenate((self.weights, new_weights.flatten()))
            self.biases = np.concatenate((self.biases, new_biases.flatten()))


def init_pop(pop_size, num_of_layers, num_of_neurons):
    pop = []
    for i in range(pop_size):
        pop.append(model(num_of_layers, num_of_neurons))
    return pop
