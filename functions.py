from main import forward
from conf import *
from init_pop import model as model_class
import numpy as np


def fittness(dataset, model):
    sum_correct = 0
    for i in range(len(dataset)):
        pred = forward(dataset[i][0], model)
        if pred == dataset[i][1]:
            sum_correct += 1
    model.fitness = sum_correct / len(dataset)


def calc_probabilities(population):
    sum_fitness = sum([model.fitness for model in population])
    probabilities = [model.fitness / sum_fitness for model in population]
    return probabilities


def crossover(parent1, parent2):
    child = model_class(parent1.num_of_layers, parent1.num_of_neurons, init_weights=False)
    crossover_point = np.random.randint(0, parent1.num_of_layers - 1)
    child.weights = parent1.weights[:crossover_point] + parent2.weights[crossover_point:]
    child.biases = parent1.biases[:crossover_point] + parent2.biases[crossover_point:]
    return child


# todo: maybe change the way the mutation is done (so that it will be more efficient)
def mutate(model):
    for i in range(model.num_of_layers - 1):
        for j in range(model.num_of_neurons[i + 1]):
            if np.random.rand() < MUTATION_RATE:
                model.weights[i].T[j] += np.random.uniform(-1, 1, size=(model.num_of_neurons[i + 1])) * MUTATION_FACTOR
    return


def generate_next_generation(population):
    probabilities = calc_probabilities(population)
    # get the indexes of the elite models
    elite_indexes = np.argsort(probabilities)[-ELITE_SIZE:]
    new_population = []
    for i in range(POPULATION_SIZE):
        if i in elite_indexes:
            new_population.append(population[i])
            continue
        parent1 = np.random.choice(population, p=probabilities)
        parent2 = np.random.choice(population, p=probabilities)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    return new_population
