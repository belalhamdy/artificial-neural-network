import numpy as np
import program1 as bp


def main():
    best_weights = np.load('weights.npy', allow_pickle=True)
    sizes = np.load('sizes.npy', allow_pickle=True)
    network = bp.Network(sizes, best_weights)



if __name__ == '__main__':
    main()
