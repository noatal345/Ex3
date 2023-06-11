from readfiles import *
import init_pop
from functions import *
from conf import *

# This is the main program
# read the files and save the samples and classifications into lists of tuples
nn0_list_of_samples = read_files("nn0.txt")
nn1_list_of_samples = read_files("nn1.txt")
# Initialize the population
population = init_pop.init_pop(POPULATION_SIZE, NUM_OF_LAYERS, [16, 8, 4, 1])

# for each generation:
for i in range(100):
    # calculate the fitness of each individual in the population
    fitness_lst = [fitness(nn0_list_of_samples, population[i]) for i in range(POPULATION_SIZE)]
    # print the fitness_lst
    print(fitness_lst)
    # generate the next generation
    population = generate_next_generation(population)
    # print the best fitness in the generation
    print("Generation: " + str(i) + " Best Fitness: " + str(max(fitness_lst)))
