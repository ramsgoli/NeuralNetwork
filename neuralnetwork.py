import numpy as np

class NeuralNetwork:
    def __init__(self, i_size, h_size, o_size):
        self.i_size = i_size  # input layer size
        self.o_size = o_size  # output layer size
        self.h_size = h_size  # hidden layer size
        self.b1 = np.random.randn(1, h_size)
        self.b2 = np.random.randn(1, o_size)
        self.w1 = np.random.randn(i_size, h_size)
        self.w2 = np.random.randn(h_size, o_size)

    def SGD(self, X, y, batch_size, epochs=30, eta=3.0):
        pass

    def train(self, X, y, epochs=30, eta=3.0):
        X, y = shuffle(X, y)
        errors = []

        for i in range(epochs):
            # feed forward
            errors.append(self.total_error(X, y))

            for X_train, y_train in zip(X, y):
                X_train, y_train = X_train.reshape(1, len(X_train)), y_train.reshape(1, len(y_train))
                z1, a1, z2, a2 = self.feed_forward(X_train)
                grad_weights, grad_biases = self.back_propagate(z1, a1, z2, a2, X_train, y_train)

                # update weights and biases
                self.w1 -= eta * grad_weights[0]
                self.w2 -= eta * grad_weights[1]

                self.b1 -= eta * grad_biases[0]
                self.b2 -= eta * grad_biases[1]

        #print(errors)

    def total_error(self, X, y):
        output = self.feed_forward(X)[3]
        print(output, y)
        return np.square(y - output).sum() / (2 * len(X))

    def feed_forward(self, X):
        """
        Returns the activations of a single training example
        """
        z1 = np.dot(X, self.w1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid(z2)
        return z1, a1, z2, a2

    def back_propagate(self, z1, a1, z2, a2, inputs, y):
        delta_2 = (a2 - y) * sigmoid_prime(z2)
        delta_1 = np.dot(delta_2, self.w2.T) * sigmoid_prime(z1)
        partial_b2 = delta_2
        partial_b1 = delta_1
        partial_w2 = np.dot(a1.T, delta_2)
        partial_w1 = np.dot(inputs.T, delta_1)

        partials_weights = [partial_w1, partial_w2]
        partials_biases = [partial_b1, partial_b2]

        return partials_weights, partials_biases

    def fit(self, X):
        return feed_forward(X)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def shuffle(X, y):
    dummy = np.concatenate((X, y), axis=1)
    np.random.shuffle(dummy)
    return dummy[:, :X.shape[1]], dummy[:, X.shape[1]:]
