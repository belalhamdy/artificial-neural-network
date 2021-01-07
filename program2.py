import numpy as np
from NeuralNetwork import NeuralNetwork
import data_handler as dh

def main():
    best_weights = np.load('weights.npy', allow_pickle=True)

    sizes = np.load('sizes.npy', allow_pickle=True)
    mu, sigma = np.load('normalization.npy', allow_pickle=True)

    sizes, x, y = dh.get_data('train.txt')
    x = dh.apply_normalization(x, mu, sigma)

    activation = (lambda x: x, lambda y: 1)
    network = NeuralNetwork(sizes, weights=best_weights, activation=activation)

    y_pred, mse = network.test(x, y)

    print(np.array(y_pred))
    print(mse)


if __name__ == '__main__':
    main()
