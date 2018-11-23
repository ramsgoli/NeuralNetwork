from neuralnetwork import NeuralNetwork
#from masternetwork import Network
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
    X = np.array([[1], [2]])
    y = np.array([[0.99], [0.01]])

    network.train(X, y)
