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
    train, validation, test = mnist_loader.load_data_wrapper()
    X_train, y_train = train[0], train[1]
    X_validate, y_validate = validation[0], validation[1]
    X_test, y_test = test[0], test[1]

    network = NeuralNetwork(784, 30, 10)
    network.SGD(X_train, y_train, 10, 30, 0.5, l2=0.5, X_test=X_test, y_test=y_test)

    # plot accuracies on test data
    plt.plot(network.test_accuracies)
    plt.show()

