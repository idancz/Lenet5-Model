import matplotlib.pyplot as plt
from train_and_eval import *

global train_accuracy_dict
global test_accuracy_dict


# ploting all accuracies
def accuracy_plot(model_name):
    accuracy_plot_fig = plt.figure()
    plt.plot(train_accuracy_dict[model_name], label='Train')
    plt.plot(test_accuracy_dict[model_name], label='Test')
    plt.title(model_name + " Accuracy Graph")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy [%]")
    plt.grid(True)
    plt.legend(frameon=False)
    accuracy_plot_fig.set_size_inches((12, 8))
    plt.show()
    print("\n")


# generating summary table
def summary_table():
    print("################ - FashionMNIST LeNet5 Summary Table - ################")
    print("_______________________________________________________________________")
    print("Model\t\t\tBest Train Accuracy[%]\t  Best Test Accuracy[%]")
    print("_______________________________________________________________________")
    for k in test_accuracy_dict:
        l = len(k)
        print(f'{k}{" " * (20 - l)}\t{max(train_accuracy_dict[k]):.4f}\t\t\t  {max(test_accuracy_dict[k]):.4f}')
    print("_______________________________________________________________________")
    print("#######################################################################")
