import matplotlib.pyplot as plt

# read all 3lists from the plots.txt file
file = open("plots00.txt", "r")
train_fitness_lst_lst = []
train_avg_fitness_lst_lst = []
test_fitness_lst = []
# skip the "train_fitness_lst_lst:" line
file.readline()
for i in range(3):
    lst = eval(file.readline())
    lst = [x*100 for x in lst]
    train_fitness_lst_lst.append(lst)
# skip the "train_avg_fitness_lst_lst:" line
file.readline()
for i in range(3):
    train_avg_fitness_lst_lst.append(eval(file.readline()))
# skip the "test_fitness_lst:" line
file.readline()
test_fitness_lst = eval(file.readline())
test_fitness_lst = [lst[0]*100 for lst in test_fitness_lst]
# skip the "test_plt_lst:" line
file.readline()
test_plt_lst = eval(file.readline())
for i in range(len(test_plt_lst)):
    test_plt_lst[i] = [x*100 for x in test_plt_lst[i]]
# test_plt1 = eval(file.readline())
# test_plt2 = eval(file.readline())
# test_plt3 = eval(file.readline())
# print(test_plt1)
# print(test_plt2)
# print(test_plt3)

# test_plt1 = [x*100 for x in test_plt1]
# test_plt2 = [x*100 for x in test_plt2]
# test_plt3 = [x*100 for x in test_plt3]
# file.close()
# print(test_plt1)
# print(test_plt2)
# print(test_plt3)
# test_plt_lst = []
# test_plt_lst.append(test_plt1)
# test_plt_lst.append(test_plt2)
# test_plt_lst.append(test_plt3)


def plot_train():
    # plot the results of train_fitness_lst_lst
    for i in range(len(train_fitness_lst_lst)):
        plt.plot(train_fitness_lst_lst[i])
    # define the y axis between 0 and 100 with percentage sign %
    plt.yticks(range(0, 101, 20), ['{}%'.format(x) for x in range(0, 101, 20)])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Fitness of train samples nn1')
    plt.show()


def plot_avg():
    # plot the results of train_avg_fitness_lst_lst
    for i in range(len(train_avg_fitness_lst_lst)):
        plt.plot(train_avg_fitness_lst_lst[i])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Average fitness of train samples nn1')
    plt.show()


def plot_test():
    print(test_fitness_lst)
    # plot the results of test_fitness_lst as dots
    plt.plot(test_fitness_lst, 'ro')
    plt.ylabel('Fitness')
    # define x axis values to be the number of test samples
    plt.xticks(range(len(test_fitness_lst)))
    # define y axis between 0 and 100 with percentage sign %
    plt.yticks(range(95, 101), ['{}%'.format(x) for x in range(95, 101)])
    plt.xlabel('Iteration')
    plt.title('Fitness of test samples nn1')
    plt.show()


# def plot_tests():
#     # plot the results of test_plt_lst
#     plt.plot(test_plt_lst)
#     plt.ylabel('Fitness')
#     plt.xlabel('Generation')
#     plt.title('test every 50 generations nn1')
#     plt.show()


def plot_train_and_test():
    # color the train results in green
    plt.plot(train_fitness_lst_lst[0], 'g')
    plt.plot(train_fitness_lst_lst[1], 'g')
    plt.plot(train_fitness_lst_lst[2], 'g')

    # plot the test results on the same plot as the train results with the correct x-axis values
    for i, test_lst in enumerate(test_plt_lst):
        x_values = [x * 50 for x in range(len(test_lst))]
        plt.plot(x_values, test_lst, 'r')

    # add a legend to the plot Train in green and Test in blue
    plt.legend(['Train 1', 'Train 2', 'Train 3', 'Test 1', 'Test 2', 'Test 3'], loc='best', bbox_to_anchor=(1, 0.5))
    # define the y-axis between 0 and 100 with percentage sign %
    plt.yticks(range(50, 101, 20), ['{}%'.format(x) for x in range(50, 101, 20)])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Train and Test Fitness nn0')
    plt.show()


# plot_train()
# plot_avg()
# plot_test()
# plot_tests()
plot_train_and_test()
