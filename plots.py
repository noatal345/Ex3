import matplotlib.pyplot as plt

# read all 3lists from the plots.txt file
file = open("plots.txt", "r")
train_fitness_lst_lst = []
train_avg_fitness_lst_lst = []
test_fitness_lst = []
for i in range(2):
    train_fitness_lst_lst.append(eval(file.readline()))
for i in range(2):
    train_avg_fitness_lst_lst.append(eval(file.readline()))
test_fitness_lst = eval(file.readline())
file.close()


def plot_train():
    # plot the results of train_fitness_lst_lst
    for i in range(len(train_fitness_lst_lst)):
        plt.plot(train_fitness_lst_lst[i])
    plt.ylabel('fitness')
    plt.xlabel('generation')
    plt.title('fitness of train samples')
    plt.show()


def plot_avg():
    # plot the results of train_avg_fitness_lst_lst
    for i in range(len(train_avg_fitness_lst_lst)):
        plt.plot(train_avg_fitness_lst_lst[i])
    plt.ylabel('fitness')
    plt.xlabel('generation')
    plt.title('average fitness of train samples')
    plt.show()


def plot_test():
    # plot the results of test_fitness_lst
    plt.plot(test_fitness_lst)
    plt.ylabel('fitness')
    plt.xlabel('iteration')
    plt.title('fitness of test samples')
    plt.show()


plot_train()
plot_avg()
plot_test()
