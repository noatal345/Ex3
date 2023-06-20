from functions import *


# read the wnet1.txt file and create a NN model with the parameters defined in the file.
model = load_model("wnet0.txt")

# create an empty result file to save the classification of the unclassified samples
file = open("result0.txt", "w")

# save the unclassified samples in a list of tuples
unclassified_samples = read_files("testnet0.txt", False)

# predict the classification of the unclassified samples
for i in range(len(unclassified_samples)):
    # for every sample in the unclassified file
    prediction = forward(unclassified_samples[i], model)
    # write the sample without the brackets and the prediction in the same line
    file.write("".join(map(str, unclassified_samples[i])) + " " + str(prediction) + "\n")
