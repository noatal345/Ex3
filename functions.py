from init_pop import Model as model_class
import numpy as np
import random
from readfiles import *
import init_pop
from init_pop import Model
from tqdm import tqdm


# This function is in charge of the forward propagation of the sample through the model
# returns the prediction of the model 0 or 1
def forward(sample, model):
    x = sample.copy()
    for i in range(model.num_of_layers - 1):
        x = np.dot(x, model.weights[i]) + model.biases[i]
        x = model.activation_function(x)
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
    # normalize the probabilities

    return probabilities


def crossover(parent1, parent2):
    child = model_class(parent1.num_of_layers, parent1.num_of_neurons, parent1.activation_function, init_weights=False)
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
def mutate(model, mutation_rate, mutation_factor):
    for i in range(model.num_of_layers - 1):
        for j in range(model.num_of_neurons[i + 1]):
            if np.random.rand() < mutation_rate:
                model.weights[i].T[j] += np.random.uniform(-1, 1) * mutation_factor
    return


def generate_next_generation(population, elite_size, population_size, mutation_rate, mutation_factor):
    probabilities = calc_probabilities(population)
    # get the indexes of the elite models
    elite_indexes = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:elite_size]
    new_population = []
    for i in range(population_size):
        if i in elite_indexes:
            population[i].elite = True
            new_population.append(population[i])
            continue
        parent1 = np.random.choice(population, p=probabilities)
        parent2 = np.random.choice(population, p=probabilities)
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate, mutation_factor)
        new_population.append(child)
    return new_population


def save_results(model, file_name):
    # use numpy.savetxt to save the weights and biases to a txt file
    file = open(file_name, "w")
    file.write(str(model.num_of_layers) + "\n")
    file.write(" ".join(map(str, model.num_of_neurons)) + "\n")
    for i in range(model.num_of_layers - 1):
        np.savetxt(file, model.weights[i], fmt="%f")
    file.write("*\n")
    for i in range(model.num_of_layers - 1):
        # write the biases to the file without the brackets
        file.write(" ".join(map(str, model.biases[i])) + "\n")
    file.close()


def split_samples(samples, train_ratio=0.8):
    # split the samples into train and test samples
    random.shuffle(samples)
    train_samples = samples[:int(len(samples) * train_ratio)]
    test_samples = samples[int(len(samples) * train_ratio):]
    return train_samples, test_samples


def test(population, test_samples, population_size):
    # test all the models in the population and return the best model
    fitness_lst = [fitness(test_samples, population[i]) for i in range(population_size)]
    best_model = population[fitness_lst.index(max(fitness_lst))]
    print("Test Best Fitness(accuracy): " + str(max(fitness_lst)))
    print(fitness_lst)
    return best_model


def train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate, mutation_factor):
    # The algorithm will run for 100 generations and return the best model
    # for i in range(num_of_generations+1):
    for i in tqdm(range(num_of_generations+1)):
        # calculate the fitness of each individual in the population
        fitness_lst = [fitness(train_samples, population[i]) for i in range(population_size)]
        if i % 20 == 0:
            print("Generation: " + str(i) + " Best Fitness(accuracy): " + str(max(fitness_lst)) + " Average Fitness: "
                  + str(sum(fitness_lst) / len(fitness_lst)))
        # generate the next generation
        population = generate_next_generation(population, elite_size, population_size, mutation_rate, mutation_factor)
    return population


# This genetic algorithm train all the models and then return the model with the best accuracy on the test samples
def genetic_algorithm(population, train_samples, test_samples, num_of_generations, population_size, elite_size,
                      mutation_rate, mutation_factor):
    population = train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate,
                       mutation_factor)
    # test all the models in the population and save the best model
    best_model = test(population, test_samples, population_size)
    return best_model


# This genetic algorithm train all the models and then test only the best model on the test samples
def genetic_algorithm2(population, train_samples, test_samples, num_of_generations, population_size,
                       elite_size, mutation_rate, mutation_factor):
    population = train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate,
                       mutation_factor)
    # calculate the fitness of each individual in the population
    fitness_lst = [fitness(train_samples, population[i]) for i in range(population_size)]
    # get the best model from the population
    best_model = population[fitness_lst.index(max(fitness_lst))]
    # test the best model
    fitness_lst = [fitness(test_samples, best_model)]
    print("\nTest Best Fitness(accuracy): " + str(round(max(fitness_lst), 4)))
    # print(fitness_lst)
    return best_model, str(max(fitness_lst))


def start(population, nn_train_samples, nn_test_samples, num_of_generations, population_size, elite_size,
          mutation_rate, mutation_factor):
    # Get the best model out of 10 iterations of the genetic algorithm
    best_model_lst = []
    i = 1
    while i < 11:
        # print iteration number in format string
        print("Iteration number ", i)
        # apply the genetic algorithm on the population of models to choose the best one.
        model, accuracy = genetic_algorithm2(population, nn_train_samples, nn_test_samples, num_of_generations,
                                             population_size, elite_size, mutation_rate, mutation_factor)
        best_model_lst.append((model, accuracy))
        i += 1

    best_model = max(best_model_lst, key=lambda x: x[1])[0]
    print("All models accuracy list:", [m[1] for m in best_model_lst])
    print("The best model accuracy is:", max([m[1] for m in best_model_lst]))
    return best_model


def load_model(file_name):
    # load the weights and biases from a txt file
    file = open(file_name, "r")
    num_of_layers = int(file.readline())
    num_of_neurons = list(map(int, file.readline().split()))
    # create a new model with the parameters define in the wnet txt file.
    model = Model(num_of_layers, num_of_neurons, False)
    # get the weights and biases from the wnet txt file.
    for i in range(model.num_of_layers - 1):
        # reset weights
        weights = []
        for j in range(model.num_of_neurons[i]):
            # make thw weights np.array
            weights.append(np.array(list(map(float, file.readline().split()))))
        model.weights.append(np.array(weights))
    file.readline()
    for i in range(model.num_of_layers - 1):
        model.biases.append(np.array(list(map(float, file.readline().split()))))
    file.close()
    return model
