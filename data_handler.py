import pandas as pd
import numpy as np


def get_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return mu, sigma


def apply_normalization(x, mu, sigma):
    return np.divide(np.subtract(x, mu), sigma)


def get_data(filename):
    f = open(filename, mode='r')
    sizes = [int(s) for s in f.readline().split(' ')]
    n = int(f.readline())
    f.close()

    df = pd.read_csv(filename, header=None, skiprows=2, delim_whitespace=True,)
    x = df.iloc[:, : sizes[0]].values
    y = df.iloc[:, -sizes[-1]:].values
    assert(len(x) == len(y) == n)
    # y = (y-np.mean(y,axis=0))/np.std(y,axis=0)
    return sizes, x, y
