from readfiles import *
import init_pop
from functions import *


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


# This is the main program
# read the files and save the samples and classifications into lists of tuples
nn0_list_of_samples = read_files("nn0.txt", True)
nn1_list_of_samples = read_files("nn1.txt", True)


# Initialize the population
population = init_pop.init_pop(POPULATION_SIZE, NUM_OF_LAYERS, [16, 8, 4, 1])

# The algorithm will run for 100 generations and return the best model
# TODO delete this
sample = nn0_list_of_samples[0][0]
model = population[7]
save_results(model, "wnet.txt")