import numpy as np
from sklearn.utils import shuffle


def activation(z):
    return 1 if z >= 0 else 0


def shuffle(train_set, train_set_labels):
    tmp_set1, tmp_set2 = shuffle(train_set, train_set_labels, random_state=0)
    return tmp_set1, tmp_set2


class Perceptron:
    def __init__(self, speciality_value, lr=0.01, epochs=15):
        self.W = None
        self.B = None

        self.speciality_value = speciality_value
        self.lr = lr
        self.epochs = epochs

    def mini_batch(self, train_set, train_set_labels, batch_size=32):
        train_set, train_set_labels = shuffle(train_set, train_set_labels)
        train_set_labels = np.where(train_set_labels == self.speciality_value, 1, 0)

        number_of_batches = len(train_set[0]) // batch_size
        shuffle()

        self.W = np.random.rand(len(train_set[0]) + 1)
        self.B = np.random.rand(1)
        self.W[0] = self.B

        for i in range(0, self.epochs):
            for j in range(0, number_of_batches):
                batch = train_set[j * batch_size:(j + 1) * batch_size]
                batch_labels = train_set_labels[j * batch_size:(j + 1) * batch_size]

                weights_delta = np.zeros(len(train_set[0]) + 1)
                bias_delta = 0
                weights_delta[0] = bias_delta

                for x, t in zip(batch, batch_labels):
                    x = np.insert(x, 0, 1)
                    z = np.dot(x, self.W) + self.B
                    y = activation(z)
                    weights_delta += (t - y) * x * self.lr
                    bias_delta += (t - y) * self.lr

                self.W += weights_delta
                self.B += bias_delta

            print(f"[{self.speciality_value}] Epoch: {i + 1} Accuracy: {self.accuracy(train_set, train_set_labels, True)}")

    def accuracy(self, test_set, test_set_labels, running=False):
        if not running:
            test_set_labels = np.where(test_set_labels == self.speciality_value, 1, 0)
        correct = 0
        for x, t in zip(test_set, test_set_labels):
            x = np.insert(x, 0, 1)
            z = np.dot(x, self.W) + self.B
            y = activation(z)
            if y == t:
                correct += 1
        return correct / len(test_set_labels)
