import pandas as pd


def get_data(filename):
    f = open(filename, mode='r')
    sizes = [int(s) for s in f.readline().split(' ')]
    n = int(f.readline())
    f.close()

    df = pd.read_csv(filename, header=None, skiprows=2, delim_whitespace=True,)
    X = df.iloc[:, : sizes[0]].values
    y = df.iloc[:, sizes[0]:].values
    return sizes, n, X, y
