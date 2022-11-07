from typing import Callable

import numpy as np
import util

from Layer import Layer


class Network:
    layer_sizes: [int]
    learning_rate: float
    epochs: int
    layers: [Layer]

    def __init__(self, layer_sizes, learning_rate, epochs):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        for idx, input_size in enumerate(layer_sizes[:-1]):
            # the current layers nr of outputs(nodes) is the next layers number of inputs
            self.layers += [Layer(input_size, layer_sizes[idx + 1])]

    def make_copy(self):
        copy = Network(self.layer_sizes, self.learning_rate, self.epochs)
        for idx, layer in enumerate(self.layers):
            layer_in_copy = copy.layers[idx]
            layer_in_copy.weights = np.copy(layer.weights)
            layer_in_copy.biases = np.copy(layer.biases)
        return copy

    def feed_forward(self, training_set):
        training_input = training_set
        for layer in self.layers[:-1]:
            output = layer.eval(training_input)
            layer.activation_for_layer(output, util.sigmoid_activation)
            training_input = layer.activation
        output_layer = self.layers[-1]
        output = output_layer.eval(training_input)
        output_layer.activation_for_layer(output, util.sigmoid_activation)
        return output_layer.activation

    def backpropagation(self, train_set, train_labels, lin_factor, friction_param, dataset_size):
        # Compute the error for the final layer
        output_layer_activation = self.feed_forward(train_set)
        t = np.array([util.convert_label_to_array(label, self.layers[-1].neurons_count) for label in train_labels])
        output_layer_err = util.cross_entropy_derivative(output_layer_activation, t)
        self.layers[-1].error = output_layer_err

        # Compute the error for the hidden layers
        for index, layer in reversed(list(enumerate(self.layers[:-1]))):
            next_layer = self.layers[index + 1]
            layer.error = np.multiply(util.sigmoid_activation_derivative(layer.activation),
                                      np.dot(next_layer.error, next_layer.weights.T))

            next_layer.cost_gradient_w = np.dot(layer.activation.T, next_layer.error)
            next_layer.cost_gradient_b = next_layer.error

            # Update layer friction
            if next_layer.friction is None:
                next_layer.friction = np.zeros(next_layer.cost_gradient_w.shape)
            next_layer.friction = friction_param * next_layer.friction - (self.learning_rate / len(train_set)) * next_layer.cost_gradient_w

            # Update the weights and biases for next layer
            next_layer.weights = (1 - self.learning_rate * lin_factor / dataset_size) * \
                                 next_layer.weights + next_layer.friction
            next_layer.biases += np.sum((-self.learning_rate) * next_layer.cost_gradient_b, axis=0)

        # Compute the error for the first layer
        first_layer = self.layers[0]
        first_layer.cost_gradient_w = np.dot(train_set.T, first_layer.error)
        first_layer.cost_gradient_b = first_layer.error

        # Update first layer friction
        if first_layer.friction is None:
            first_layer.friction = np.zeros(first_layer.cost_gradient_w.shape)
        first_layer.friction = friction_param * first_layer.friction - \
                               (self.learning_rate / len(train_set)) * first_layer.cost_gradient_w

        # Update the weights and biases for first layer
        first_layer.weights = (1 - self.learning_rate * lin_factor / dataset_size) * \
                                first_layer.weights + first_layer.friction
        first_layer.biases += np.sum((-self.learning_rate) * first_layer.cost_gradient_b, axis=0)

    def train(self, training_set, training_labels, batch_size, validation_set, validation_labels):
        epoch_map = {}
        for epoch in range(self.epochs):
            for batch in range(0, len(training_set), batch_size):
                self.backpropagation(training_set[batch:batch + batch_size], training_labels[batch:batch + batch_size],
                                     5, 0.9, len(training_set[0]))
            print(f"Epoch {epoch + 1} completed")
            copy = self.make_copy()
            epoch_map[epoch] = (copy, self.validate(validation_set, validation_labels))
        return epoch_map

    def validate(self, validation_set, validation_labels):
        correct = 0
        for idx, image in enumerate(validation_set):
            prediction = self.feed_forward(image)
            if np.argmax(prediction) == validation_labels[idx]:
                correct += 1
        return correct, len(validation_set) - correct, correct / len(validation_set)

    def test(self, test_set, test_labels):
        correct = 0
        for idx, image in enumerate(test_set):
            prediction = self.feed_forward(image)
            if np.argmax(prediction) == test_labels[idx]:
                correct += 1
        return correct / len(test_set)
