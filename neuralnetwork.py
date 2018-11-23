import numpy as np

class NeuralNetwork:
    def __init__(self, i_size, h_size, o_size):
        self.i_size = i_size  # input layer size
        self.o_size = o_size  # output layer size
        self.h_size = h_size  # hidden layer size
        self.b1 = np.random.randn(h_size)
        self.b2 = np.random.randn(o_size)
        self.w1 = np.random.randn(i_size, h_size)
        self.w2 = np.random.randn(h_size, o_size)

    def train(self, X, y):
        self.X = X
        self.y = y

    def total_error(self, output):
        return np.square(self.training - output).sum()

    def feed_forward(self, X):
        """
        Returns the activations of a single training example
        """
        z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = self.a1.dot(self.w2) + self.b2
        self.a2 = sigmoid(z2)
        return self.a2

    def back_propagate(self, z1, a1, z2, output, y):
        delta_2 = (output - y) * sigmoid_prime(z2)
        delta_1 = self.w2.T.dot(nabla_2) * sigmoid_prime(z1)
        partial_b2 = delta_2
        partial_w2 = np.dot(delta_2, a1.T)

    def fit(self, X):
        return feed_forward(X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
