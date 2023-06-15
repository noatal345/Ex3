from readfiles import *
import init_pop
from functions import *
import random


def save_results(model, file_name="wnet.txt"):
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


def test(population, test_samples):
    # test all the models in the population and return the best model
    fitness_lst = [fitness(test_samples, population[i]) for i in range(POPULATION_SIZE)]
    best_model = population[fitness_lst.index(max(fitness_lst))]
    print("***\n Test Best Fitness(accuracy): " + str(max(fitness_lst)) + "\n***")
    print(fitness_lst)
    return best_model


def train(population, train_samples):
    # The algorithm will run for 100 generations and return the best model
    for i in range(151):
        # calculate the fitness of each individual in the population
        fitness_lst = [fitness(train_samples, population[i]) for i in range(POPULATION_SIZE)]
        # generate the next generation
        population = generate_next_generation(population)
        # print the best fitness evrey 10 generations
        if i % 10 == 0:
            # print the fitness_lst
            print(fitness_lst)
            print("Generation: " + str(i) + " Best Fitness(accuracy): " + str(max(fitness_lst)))
            print("Generation: " + str(i) + " Avg Fitness(accuracy): " + str(sum(fitness_lst)/len(fitness_lst)))
    return population


def genetic_algorithm(population, train_samples, test_samples):
    population = train(population, train_samples)
    # test all the models in the population and save the best model
    best_model = test(population, test_samples)
    return best_model

def genetic_algorithm2(population, train_samples, test_samples):
    population = train(population, train_samples)
    # calculate the fitness of each individual in the population
    fitness_lst = [fitness(train_samples, population[i]) for i in range(POPULATION_SIZE)]
    # get the best model from the population
    best_model = population[fitness_lst.index(max(fitness_lst))]
    # test the best model
    fitness_lst = [fitness(test_samples, best_model)]
    print("***\n Test Best Fitness(accuracy): " + str(max(fitness_lst)) + "\n***")
    print(fitness_lst)
    return best_model


name_of_file = "nn0.txt"
# TODO This program should receive one or 2 files ?
# This is the main program
# read the files and save the samples and classifications into lists of tuples
nn0_list_of_samples = read_files(name_of_file, True)

# splite the samples into train and test samples
nn0_train_samples, nn0_test_samples = split_samples(nn0_list_of_samples, 0.8)

# Initialize the population
population = init_pop.init_pop(POPULATION_SIZE, NUM_OF_LAYERS, [16, 8, 4, 1])

# apply the genetic algorithm on the population of models to choose the best one.
model = genetic_algorithm2(population, nn0_train_samples, nn0_test_samples)

save_results(model, "wnet.txt")

