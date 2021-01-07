import numpy as np
from random import random
import data_handler as dh
from NeuralNetwork import NeuralNetwork
from NeuralNetwork import Sigmoid
import matplotlib.pyplot as plt


def plot_history(history, c, label):
    plt.ylabel(label)
    plt.xlabel('Iteration')
    plt.plot(history, c=c, label=label)
    plt.legend()
    plt.show()


def main():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    sizes, x, y = dh.get_data('train.txt')
    mu, sigma = dh.get_normalization(x)
    x = dh.apply_normalization(x, mu, sigma)

    nn = NeuralNetwork(sizes)

    alpha = 0.5
    costs = nn.train(x, y, alpha, max_epochs=300, eps=1e-7)

    y_pred, mse = nn.test(x, y)
    print("MSE:", mse)
    plot_history(costs, 'orange', 'Mean Square Error')

    np.save('weights.npy', nn.weights, allow_pickle=True)
    np.save('sizes.npy', nn.sizes, allow_pickle=True)
    np.save('normalization.npy', (mu, sigma), allow_pickle=True)
    print("Model parameters saved.")

if __name__ == '__main__':
    main()
