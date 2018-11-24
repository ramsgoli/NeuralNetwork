from neuralnetwork import NeuralNetwork, shuffle
import mnist_loader
#from masternetwork import Network
import numpy as np
import matplotlib.pyplot as plt

def generate_test_dataset():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    print(X, y)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    # generate_test_dataset()
    X_train, y_train, X_test, y_test = mnist_loader.load_data_wrapper()
    print(X_train.shape, X_test.shape)

    network = NeuralNetwork(784, 30, 10)

    network.SGD(X_train, y_train, 10, 30, 3.0, X_test=X_test, y_test=y_test)

    # network.fit(np.array([[0, 35], [4, 1]]))
