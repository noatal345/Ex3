from readfiles import *
from init_pop import Model
from functions import *


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


def main():
    # read the wnet txt file to create a model with the parameters define in the wnet txt file.
    model = load_model("wnet.txt")

    # create an empty result file
    file = open("result.txt", "w")
    unclassified_samples = read_files("unclassified.txt", False)

    for i in range(len(unclassified_samples)):
        # for every sample in the unclassified file
        prediction = forward(unclassified_samples[i], model)
        # write the sample without the brackets and the prediction in the same line
        file.write("".join(map(str, unclassified_samples[i])) + " " + str(prediction) + "\n")


if __name__ == "__runnet__":
    main()
