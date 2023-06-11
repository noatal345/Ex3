from readfiles import *
import init_pop
from conf import *



# This is the main program
# read the files and save the samples and classifications into lists of tuples
nn0_list_of_samples = read_files("nn0.txt")
nn1_list_of_samples = read_files("nn1.txt")
# Initialize the population
population = init_pop.init_pop(POPULATION_SIZE, NUM_OF_LAYERS, [16, 8, 4, 1])


