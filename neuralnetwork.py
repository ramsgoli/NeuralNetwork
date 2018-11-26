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
        self.test_accuracies = []


    def SGD(self, X, y,
            batch_size=10,
            epochs=30,
            eta=3.0,
            l2=0,
            X_test=None,
            y_test=None,
            debug=True):

        n = len(X)

        for x in range(epochs):
            if debug:
                print("Training Epoch #{}".format(x))

            X_train, y_train = shuffle(X, y)

            for k in range(0, n, batch_size):
                batch = [X_train[k:k+batch_size], y_train[k:k+batch_size]]
                self.update_mini_batch(batch, eta, l2, n)

            if X_test is not None and y_test is not None:
                num_correct = self.evaluate(X_test, y_test)
                total = X_test.shape[0]

                print("Epoch {}: correct predictions: {}/{}".format(x, num_correct, total))
                self.test_accuracies.append(num_correct/total)

        print("Done training")


    def update_mini_batch(self, batch, eta, l2, n):
        nabla_d_1 = np.zeros(self.w1.shape)
        nabla_d_2 = np.zeros(self.w2.shape)
        nabla_b_1 = np.zeros(self.b1.shape)
        nabla_b_2 = np.zeros(self.b2.shape)

        len_batch = len(batch[0])

        for X_train, y_train in zip(batch[0], batch[1]):
            X_train, y_train = X_train.reshape(1, len(X_train)), y_train.reshape(1, len(y_train))
            z1, a1, z2, a2 = self.feed_forward(X_train)
            grad_weights, grad_biases = self.back_propagate(z1, a1, z2, a2, X_train, y_train)

            nabla_d_1 += grad_weights[0]
            nabla_d_2 += grad_weights[1]
            nabla_b_1 += grad_biases[0]
            nabla_b_2 += grad_biases[1]

        # update weights and biases with regularization
        self.w1 = (1 - eta*l2/n) * self.w1 - (eta/len_batch) * nabla_d_1
        self.w2 = (1 - eta*l2/n) * self.w2 - (eta/len_batch) * nabla_d_2

        self.b1 += (eta/len_batch) * nabla_b_1
        self.b2 += (eta/len_batch) * nabla_b_2


    def feed_forward(self, X):
        """
        Returns the activations of a single training example
        """
        z1 = np.dot(X, self.w1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        # a2 = sigmoid(z2)
        a2 = softmax(z2)
        return z1, a1, z2, a2


    def back_propagate(self, z1, a1, z2, a2, inputs, y):
        delta_2 = a2 - y  # using log-likelihood/cross-entropy
        delta_1 = np.dot(delta_2, self.w2.T) * sigmoid_prime(z1)

        partial_b2 = delta_2
        partial_b1 = delta_1
        partial_w2 = np.dot(a1.T, delta_2)
        partial_w1 = np.dot(inputs.T, delta_1)

        partials_weights = [partial_w1, partial_w2]
        partials_biases = [partial_b1, partial_b2]

        return partials_weights, partials_biases


    def cross_entropy_cost(self, X, y):
        a = self.fit(X)
        cost = -np.sum(y * np.log(a))

        return cost/X.shape(0)


    def fit(self, X):
        z1, a1, z2, a2 = self.feed_forward(X)
        return a2


    def evaluate(self, X_test, y_test):
        test_results = [(np.argmax(self.fit(X)), y) for X, y in zip(X_test, y_test)]
        return sum(int(x==y) for x, y in test_results)


def softmax(z2):
    return np.exp(z2) / np.sum(np.exp(z2), axis=1).reshape(len(z2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def shuffle(X, y):
    dummy = np.concatenate((X, y), axis=1)
    np.random.shuffle(dummy)
    return dummy[:, :X.shape[1]], dummy[:, X.shape[1]:]
