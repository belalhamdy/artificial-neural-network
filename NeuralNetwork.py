import numpy as np
from random import random

Sigmoid = (lambda z: 1 / (1 + np.exp(-z)), lambda y: y * (1 - y))


class NeuralNetwork:
    @staticmethod
    def __calc_rand(sz):
        org = (-1 if random() <= 0.5 else 1) * 1 / sz
        return org + random() - 0.5

    def __init__(self, sizes, weights=None, activation=Sigmoid):
        self.__f, self.__diff = activation
        self.sizes = sizes
        self.o = len(self.sizes) - 1
        self.a = [np.zeros(v) for v in sizes]
        if weights is None:
            self.weights = []
            for i in range(len(sizes) - 1):
                self.weights.append(np.array(
                    [
                        [NeuralNetwork.__calc_rand(sizes[i+1]) for w in range(sizes[i] + 1)]
                        for q in range(sizes[i + 1])
                    ]
                ))
        else:
            self.weights = weights


    def __net(self, cur, h):
        cur = np.concatenate(([1], cur))
        return np.dot(self.weights[h], cur)

    def mean_square_error_example(self, y):
        return np.sum((self.a[self.o] - y) ** 2) / (2 * len(y))

    def forward_propagation_example(self, x):
        self.a[0] = cur = x
        for i, w in enumerate(self.weights):
            cur = np.concatenate(([1], cur))
            net = np.dot(w, cur)
            cur = self.a[i + 1] = self.__f(net)

        return cur

    def backward_propagation_example(self, y, alpha):
        deriv = []
        delta = (self.a[self.o] - y) * self.__diff(self.a[self.o])
        deriv.append(delta)
        for i in range(self.o, 1, -1):
            part = np.dot(self.weights[i - 1].T, delta)
            part = part[1:]
            delta = part * self.__diff(self.a[i - 1])
            deriv.append(delta)

        deriv.reverse()
        deltas = np.array(deriv)

        for h in range(self.o):
            for j in range(len(self.weights[h])):
                self.weights[h][j] -= alpha * deltas[h][j] * np.concatenate(([1], self.a[h]))

    def train_example(self, x, y, alpha):
        self.forward_propagation_example(x)
        self.backward_propagation_example(y, alpha)

    def test(self, x, y):
        mse = 0
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.forward_propagation_example(x[i]))
            mse += self.mean_square_error_example(y[i])

        mse /= len(x)
        return y_pred, mse

    def train(self, x, y, alpha, max_epochs=500, eps=1e-7):
        mse = []
        last_cost = float('inf')
        for _ in range(max_epochs):
            cur_mse = 0
            for i in range(len(x)):
                self.train_example(x[i], y[i], alpha)
                cur_mse += self.mean_square_error_example(y[i])
            cur_mse /= len(x)
            mse.append(cur_mse)
            if abs(cur_mse - last_cost) < eps:
                break
            last_cost = cur_mse

        return mse
