import numpy as np
from NeuralNetwork import NeuralNetwork
import data_handler as dh


def main():
    best_weights = np.load('weights.npy', allow_pickle=True)

    sizes = np.load('sizes.npy', allow_pickle=True)
    mu, sigma = np.load('normalization.npy', allow_pickle=True)

    sizes, x, y, range_y = dh.get_data('train.txt', return_range=True)
    x = dh.apply_normalization(x, mu, sigma)

    network = NeuralNetwork(sizes, weights=best_weights)

    y_pred, mse = network.test(x, y)
    y_pred *= range_y
    print("Prediction for the given data: ")
    for y_val in y_pred:
        print(y_val)

    print("Mean Square Error:", mse)


if __name__ == '__main__':
    main()
