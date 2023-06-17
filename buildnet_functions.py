# from readfiles import *
# import init_pop
# from functions import *
# import random
#
#
# def save_results(model, file_name):
#     # use numpy.savetxt to save the weights and biases to a txt file
#     file = open(file_name, "w")
#     file.write(str(model.num_of_layers) + "\n")
#     file.write(" ".join(map(str, model.num_of_neurons)) + "\n")
#     for i in range(model.num_of_layers - 1):
#         np.savetxt(file, model.weights[i], fmt="%f")
#     file.write("*\n")
#     for i in range(model.num_of_layers - 1):
#         # write the biases to the file without the brackets
#         file.write(" ".join(map(str, model.biases[i])) + "\n")
#     file.close()
#
#
# def split_samples(samples, train_ratio=0.8):
#     # split the samples into train and test samples
#     random.shuffle(samples)
#     train_samples = samples[:int(len(samples) * train_ratio)]
#     test_samples = samples[int(len(samples) * train_ratio):]
#     return train_samples, test_samples
#
#
# def test(population, test_samples, population_size):
#     # test all the models in the population and return the best model
#     fitness_lst = [fitness(test_samples, population[i]) for i in range(population_size)]
#     best_model = population[fitness_lst.index(max(fitness_lst))]
#     print("***\n Test Best Fitness(accuracy): " + str(max(fitness_lst)) + "\n***")
#     print(fitness_lst)
#     return best_model
#
#
# def train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate, mutation_factor):
#     # The algorithm will run for 100 generations and return the best model
#     for i in range(num_of_generations):
#         # calculate the fitness of each individual in the population
#         # fitness_lst = [fitness(train_samples, population[i]) for i in range(population_size)]
#         # generate the next generation
#         population = generate_next_generation(population, elite_size, population_size, mutation_rate, mutation_factor)
#         # print the best fitness evrey 10 generations
#         # if i % 10 == 0:
#         #     print(fitness_lst)
#         #     print("Generation: " + str(i) + " Best Fitness(accuracy): " + str(max(fitness_lst)))
#         #     print("Generation: " + str(i) + " Avg Fitness(accuracy): " + str(sum(fitness_lst)/len(fitness_lst)))
#     return population
#
#
# # This genetic algorithm train all the models and then return the model with the best accuracy on the test samples
# def genetic_algorithm(population, train_samples, test_samples, num_of_generations, population_size, elite_size,
#                       mutation_rate, mutation_factor):
#     population = train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate,
#                        mutation_factor)
#     # test all the models in the population and save the best model
#     best_model = test(population, test_samples, population_size)
#     return best_model
#
#
# # This genetic algorithm train all the models and then test only the best model on the test samples
# def genetic_algorithm2(population, train_samples, test_samples, num_of_generations, population_size,
#                        elite_size, mutation_rate, mutation_factor):
#     population = train(population, train_samples, num_of_generations, population_size, elite_size, mutation_rate,
#                        mutation_factor)
#     # calculate the fitness of each individual in the population
#     fitness_lst = [fitness(train_samples, population[i]) for i in range(population_size)]
#     # get the best model from the population
#     best_model = population[fitness_lst.index(max(fitness_lst))]
#     # test the best model
#     fitness_lst = [fitness(test_samples, best_model)]
#     print("***\n Test Best Fitness(accuracy): " + str(max(fitness_lst)) + "\n***")
#     # print(fitness_lst)
#     return best_model, str(max(fitness_lst))
#
#
# def start(population, nn_train_samples, nn_test_samples, num_of_generations, population_size, elite_size,
#           mutation_rate, mutation_factor):
#     # Get the best model out of 10 iterations of the genetic algorithm
#     best_model_lst = []
#     i = 0
#     while i < 10:
#         # print iteration number in format string
#         print("iteration number:", i)
#         # apply the genetic algorithm on the population of models to choose the best one.
#         model, accuracy = genetic_algorithm2(population, nn_train_samples, nn_test_samples, num_of_generations,
#                                              population_size, elite_size, mutation_rate, mutation_factor)
#         best_model_lst.append((model, accuracy))
#         i += 1
#
#     best_model = max(best_model_lst, key=lambda x: x[1])[0]
#     print("best models accuracy list:", [m[1] for m in best_model_lst])
#     print("The best model accuracy is:", max([m[1] for m in best_model_lst]))
#     return best_model
