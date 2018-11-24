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
    X_train = np.array([[5, 2], [5, 3], [0, 33], [1, 35]])
    y_train = np.array([[0.99, 0.01], [0.98, 0.02], [0.01, 0.99], [0.01, 0.98]])

    network.SGD(X_train, y_train, 5, debug=True)

    network.fit(np.array([[0, 35], [4, 1]]))
