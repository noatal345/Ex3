from functions import *
from conf1 import *
# TODO This program should receive 1 file and split in or 2 files train and test ?


# read the file and save the samples and classifications to a lists of tuples
nn_list_of_samples = read_files(name_of_file, True)

# splite the samples into train and test samples
nn_train_samples, nn_test_samples = split_samples(nn_list_of_samples, train_ratio)

# Initialize the population
population = init_pop.init_pop(population_size, num_of_layers, neurons_lst, leaky_relu)

# start the genetic algorithm
model = start(population, nn_train_samples, nn_test_samples, num_of_generations, population_size, elite_size,
          mutation_rate, mutation_factor)
# save the results to a file
save_results(model, "wnet1.txt")

