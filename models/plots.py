import matplotlib
# matplotlib.use('Agg')
import toml
from matplotlib import pyplot as plt

import experiments

config = toml.load('config.toml')
experiments.utilities.fix_seeds_unique()

def plot_model_loss(history_dict, name):

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "ro", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("LOSS - " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig( config['path']['plots_prefix'] + 'loss/' + name + '.png')
    plt.clf()
    # plt.show()


def plot_model_accuracy(history_dict, name):
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, acc, "ro", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("ACCURACY - " + name)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig( config['path']['plots_prefix'] + 'accuracy/' + name + '.png')
    plt.clf()
    # plt.show()