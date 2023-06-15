from conf import *
from init_pop import Model as model_class
import numpy as np


def sigmoid(x):
    # the function returns the sigmoid of the input
    return 1 / (1 + np.exp(-x))


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def forward(sample, model):
    # This function is in charge of the forward propagation of the sample through the model
    # returns the prediction of the model 0 or 1
    x = sample.copy()
    for i in range(model.num_of_layers - 1):
        x = np.dot(x, model.weights[i]) + model.biases[i]
        x = leaky_relu(x)
    # return a binary output
    if x > 0.5:
        x = 1
    else:
        x = 0
    return x


def fitness(dataset, model):
    if model.elite:
        return model.fitness
    sum_correct = 0
    for i in range(len(dataset)):
        pred = forward(dataset[i][0], model)
        if pred == dataset[i][1]:
            sum_correct += 1
    model.fitness = sum_correct / len(dataset)
    return model.fitness


def calc_probabilities(population):
    sum_fitness = sum([model.fitness for model in population])
    probabilities = [model.fitness / sum_fitness for model in population]
    return probabilities


def crossover(parent1, parent2):
    child = model_class(parent1.num_of_layers, parent1.num_of_neurons, init_weights=False)
    crossover_layer = np.random.randint(0, parent1.num_of_layers - 1)
    crossover_layer_neuron = np.random.randint(0, parent1.num_of_neurons[crossover_layer + 1])
    for i in range(crossover_layer):
        child.weights.append(parent1.weights[i].copy())
        child.biases.append(parent1.biases[i].copy())
    for i in range(crossover_layer, parent1.num_of_layers - 1):
        child.weights.append(parent2.weights[i].copy())
        child.biases.append(parent2.biases[i].copy())
    for i in range(crossover_layer_neuron):
        child.weights[crossover_layer].T[i] = parent1.weights[crossover_layer].T[i].copy()
    for i in range(crossover_layer_neuron, parent1.num_of_neurons[crossover_layer + 1]):
        child.weights[crossover_layer].T[i] = parent2.weights[crossover_layer].T[i].copy()
    return child


# todo: maybe change the way the mutation is done (so that it will be more efficient)
def mutate(model):
    for i in range(model.num_of_layers - 1):
        for j in range(model.num_of_neurons[i + 1]):
            if np.random.rand() < MUTATION_RATE:
                model.weights[i].T[j] += np.random.uniform(-1, 1) * MUTATION_FACTOR
    return


def generate_next_generation(population):
    probabilities = calc_probabilities(population)
    # get the indexes of the elite models
    elite_indexes = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:ELITE_SIZE]
    new_population = []
    # print("Elite indexes: " + str(elite_indexes))
    for i in range(POPULATION_SIZE):
        if i in elite_indexes:
            population[i].elite = True
            new_population.append(population[i])
            continue
        parent1 = np.random.choice(population, p=probabilities)
        parent2 = np.random.choice(population, p=probabilities)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    return new_population