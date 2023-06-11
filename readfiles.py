import numpy as np


def read_files(name_of_file):
    # This program is in charge of reading the files and saving the sample and classification into numpy arrays
    # open nn0.txt file
    file = open(name_of_file, "r")
    # read the file line by line the first argument of each line is the sample and the second is the classification
    # the sample is converted to numpy array of 16 integers and the classification is converted to integer
    # the sample and classification are saved into a tuple and then saved into a list of tuples
    list_of_samples = []
    for line in file:
        sample, classification = line.split()
        # convert each char-digit of the sample to int and store in the numpy array
        sample = np.array([int(digit) for digit in sample])
        classification = int(classification)
        list_of_samples.append((sample, classification))
    return list_of_samples

