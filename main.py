import numpy as np
from random import random


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = []
        self.o = len(self.sizes) - 1
        self.a = [np.zeros(v) for v in sizes]
        for i in range(len(sizes) - 1):
            self.weights.append(np.array([[random() for w in range(sizes[i] + 1)] for q in range(sizes[i + 1])]))

    def net(self, cur, h):
        cur = np.concatenate(([1], cur))
        return np.dot(self.weights[h], cur)

    def f(self, cur):
        return 1 / (1 + np.exp(-cur))

    def diff_f(self, y):
        return y * (1 - y)

    def forward_propagation(self, input):
        self.a[0] = cur = input
        for i, w in enumerate(self.weights):
            cur = np.concatenate(([1], cur))
            net = np.dot(w, cur)
            cur = self.a[i + 1] = self.f(net)

    def backward_propagation(self, y, alpha):
        deriv = []
        delta = (self.a[self.o] - y) * self.diff_f(self.a[self.o])
        deriv.append(delta)
        for i in range(self.o, 1, -1):
            part = np.dot(self.weights[i - 1].T, delta)
            part = part[1:]
            delta = part * self.diff_f(self.a[i - 1])
            deriv.append(delta)

        deriv.reverse()
        deltas = np.array(deriv)

        for h in range(self.o):
            for j in range(len(self.weights[h])):
                self.weights[h][j] -= alpha * deltas[h][j] * np.concatenate(([1], self.a[h]))

    def process(self, input, y):
        self.forward_propagation(input)
        self.backward_propagation(y, 0.3)


def main():
    sizes = [2, 2, 1]
    data = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = [1, 0, 0, 1]

    crash = Network(sizes)

    crash.weights = [np.array([[0.3, -0.9, 1], [-1.2, 1, 1]]), np.array([[0, 1, 0.8]])]

    for i in [1]:
        crash.process(np.array(data[i]), np.array(y[i]))

    print(crash.weights)


if __name__ == '__main__':
    main()
