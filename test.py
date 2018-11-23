from neuralnetwork import NeuralNetwork
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def generate_test_dataset():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    print(X, y)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    # generate_test_dataset()
    network = NeuralNetwork(2, 3, 2)
    print(network.feed_forward(np.array([1, 2])))
