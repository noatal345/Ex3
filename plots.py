import matplotlib.pyplot as plt

# read all 3lists from the plots.txt file
file = open("plots.txt", "r")
train_fitness_lst_lst = []
train_avg_fitness_lst_lst = []
test_fitness_lst = []
# skip the "train_fitness_lst_lst:" line
file.readline()
for i in range(10):
    train_fitness_lst_lst.append(eval(file.readline()))
# skip the "train_avg_fitness_lst_lst:" line
file.readline()
for i in range(10):
    train_avg_fitness_lst_lst.append(eval(file.readline()))
# skip the "test_fitness_lst:" line
file.readline()
test_fitness_lst = eval(file.readline())
file.close()


def plot_train():
    # plot the results of train_fitness_lst_lst
    for i in range(len(train_fitness_lst_lst)):
        plt.plot(train_fitness_lst_lst[i])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Fitness of train samples')
    plt.show()


def plot_avg():
    # plot the results of train_avg_fitness_lst_lst
    for i in range(len(train_avg_fitness_lst_lst)):
        plt.plot(train_avg_fitness_lst_lst[i])
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.title('Average fitness of train samples')
    plt.show()


def plot_test():
    print(test_fitness_lst)
    # plot the results of test_fitness_lst
    plt.plot(test_fitness_lst)
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')
    plt.title('Fitness of test samples')
    plt.show()


plot_train()
plot_avg()
plot_test()
