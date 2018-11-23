import numpy as np

class NeuralNetwork:
    def __init__(self, i_size, h_size, o_size):
        self.i_size = i_size  # input layer size
        self.o_size = o_size  # output layer size
        self.h_size = h_size  # hidden layer size
        self.b1 = np.random.randn(h_size, 1)
        self.b2 = np.random.randn(o_size, 1)
        self.w1 = np.random.randn(h_size, i_size)
        self.w2 = np.random.randn(o_size, h_size)

    def train(self, X, y, epochs=30):
        self.X = X
        self.y = y
        errors = []

        for i in range(epochs):
            # feed forward
            errors.append(self.total_error(X, y))
            z1, a1, z2, a2 = self.feed_forward(X)
            grad_weights, grad_biases = self.back_propagate(z1, a1, z2, a2, X, y)

            # update weights and biases
            self.w1 -= grad_weights[0]
            self.w2 -= grad_weights[1]

            self.b1 -= grad_biases[0]
            self.b2 -= grad_biases[1]

        print(errors)

    def total_error(self, X, y):
        output = self.feed_forward(X)[3]
        return np.square(y - output).sum()

    def feed_forward(self, X):
        """
        Returns the activations of a single training example
        """
        z1 = np.dot(self.w1, X) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = sigmoid(z2)
        return z1, a1, z2, a2

    def back_propagate(self, z1, a1, z2, a2, inputs, y):
        delta_2 = (a2 - y) * sigmoid_prime(z2)
        delta_1 = np.dot(self.w2.T, delta_2) * sigmoid_prime(z1)
        partial_b2 = delta_2
        partial_b1 = delta_1
        partial_w2 = np.dot(delta_2, a1.T)
        partial_w1 = np.dot(delta_1, inputs.T)

        partials_weights = [partial_w1, partial_w2]
        partials_biases = [partial_b1, partial_b2]

        return partials_weights, partials_biases

    def fit(self, X):
        return feed_forward(X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))
