from readfiles import *
import init_pop
from conf import *


def sigmoid(x):
    # the function returns the sigmoid of the input
    return 1 / (1 + np.exp(-x))


def forward(sample, model):
    # This function is in charge of the forward propagation of the sample through the model
    # returns the prediction of the model 0 or 1
    x = sample.copy()
    for i in range(model.num_of_layers - 1):
        x = np.dot(x, model.weights[i]) + model.biases[i]
        x = sigmoid(x)
    # return a binary output
    if x > 0.5:
        x = 1
    else:
        x = 0
    return x


# This is the main program
# read the files and save the samples and classifications into lists of tuples
nn0_list_of_samples = read_files("nn0.txt")
nn1_list_of_samples = read_files("nn1.txt")
# Initialize the population
population = init_pop.init_pop(POPULATION_SIZE, NUM_OF_LAYERS, [16, 8, 4, 1])


