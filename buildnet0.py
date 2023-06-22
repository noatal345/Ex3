from functions import *
from conf0 import *
from init_pop import *
# TODO This program should receive 1 file and split in or 2 files train and test ?


# read the file and save the samples and classifications to a lists of tuples
nn_list_of_samples = read_files(name_of_file, True)

# splite the samples into train and test samples
nn_train_samples, nn_test_samples = split_samples(nn_list_of_samples, train_ratio)

# Initialize the population
population = init_pop(population_size, num_of_layers, neurons_lst, "leaky_relu", True)

# start the genetic algorithm
model, train_fitness_lst_lst, train_avg_fitness_lst_lst, test_fitness_lst, test_plt_lst = start(population,
                                                                                                nn_train_samples,
                                                                                                nn_test_samples,
                                                                                                num_of_generations,
                                                                                                population_size,
                                                                                                elite_size,
                                                                                                mutation_rate,
                                                                                                mutation_factor)
# save the results to a file
save_results(model, "wnet0.txt")


# save all 3 lists to a file
file = open("plots00.txt", "w")
file.write("train_fitness_lst_lst:\n")
for i in range(len(train_fitness_lst_lst)):
    file.write(str(train_fitness_lst_lst[i]) + "\n")
file.write("train_avg_fitness_lst_lst:\n")
for i in range(len(train_avg_fitness_lst_lst)):
    file.write(str(train_avg_fitness_lst_lst[i]) + "\n")
file.write("test_fitness_lst:\n")
file.write(str(test_fitness_lst) + "\n")
file.write("test_plt_lst:\n")
for i in range(len(test_plt_lst)):
    file.write(str(test_plt_lst[i]) + "\n")
file.close()
