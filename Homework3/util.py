import random

import numpy as np
from scipy.ndimage.interpolation import shift
import cv2


def accuracy(y, t):
    return np.sum(y == t) / len(y)


def shuffle_sets(train_set, train_set_labels):
    temp = list(zip(train_set, train_set_labels))
    random.shuffle(temp)
    train_set, train_set_labels = zip(*temp)
    return np.array(train_set), np.array(train_set_labels)


def sigmoid_activation(z: np.ndarray):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_activation_derivative(z: np.ndarray):
    return sigmoid_activation(z) * (1 - sigmoid_activation(z))


def cross_entropy(y: np.ndarray, t: np.ndarray):
    return -np.sum(y * np.log(t))


def cross_entropy_derivative(y: np.ndarray, t: np.ndarray):
    return y - t


def softmax_activation(z: np.ndarray):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_activation_derivative(z: np.ndarray):
    return softmax_activation(z) * (1 - softmax_activation(z))


def convert_label_to_array(label, size):
    arr = np.zeros(size)
    arr[label] = 1
    return arr


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, order=0)
    return shifted_image.reshape(784)


def rotate_image(image, angle):
    image = image.reshape((28, 28))
    rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((14, 14), angle, 1), (28, 28))
    return rotated_image.reshape(784)
