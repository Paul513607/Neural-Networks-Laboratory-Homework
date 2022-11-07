import numpy as np
from sklearn.utils import shuffle

def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)


def shuffle_sets(train_set, train_set_labels):
    tmp_set1, tmp_set2 = shuffle(train_set, train_set_labels, random_state=0)
    return [tmp_set1, tmp_set2]


def sigmoid_activation(z: np.ndarray):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_activation_derivative(z: np.ndarray):
    return sigmoid_activation(z) * (1 - sigmoid_activation(z))


def cross_entropy(y: np.ndarray, y_hat: np.ndarray):
    return -np.sum(y * np.log(y_hat))


def cross_entropy_derivative(y: np.ndarray, y_hat: np.ndarray):
    return y - y_hat


def softmax_activation(z: np.ndarray):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_activation_derivative(z: np.ndarray):
    return softmax_activation(z) * (1 - softmax_activation(z))


def convert_label_to_array(label, size):
    arr = np.zeros(size)
    arr[label] = 1
    return arr
