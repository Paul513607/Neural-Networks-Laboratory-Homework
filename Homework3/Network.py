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
            layer.compute_activ(output, util.sigmoid_activation)
            training_input = layer.activation
        output_layer = self.layers[-1]
        output = output_layer.eval(training_input)
        output_layer.compute_activ(output, util.softmax_activation)
        return output_layer.activation

    def backpropagation(self, train_set, train_labels, lin_factor, beta, dataset_size):
        output_layer = self.layers[-1]
        hidden_layer = self.layers[-2]

        # Compute the error for the output layer
        output_layer_activation = self.feed_forward(train_set)
        t = np.array([util.convert_label_to_array(label, self.layers[-1].neurons_count) for label in train_labels])
        output_layer_err = util.cross_entropy_derivative(output_layer_activation, t)
        output_layer.error = output_layer_err

        # Compute the gradients for the output layer
        output_layer.gradient_w = np.dot(hidden_layer.activation.T, output_layer.error)
        output_layer.gradient_b = output_layer.error

        output_layer.friction = beta * output_layer.friction \
                                - (self.learning_rate / len(train_set)) * output_layer.gradient_w

        # Update the weights and biases for output layer
        output_layer.weights = (1 - self.learning_rate * lin_factor / dataset_size) * output_layer.weights \
                               + output_layer.friction
        output_layer.biases += np.sum((-self.learning_rate) * output_layer.gradient_b, axis=0)

        # Compute the error for the hidden layer
        hidden_layer.error = np.multiply(util.sigmoid_activation_derivative(hidden_layer.activation),
                                         np.dot(output_layer.error, output_layer.weights.T))

        # Compute the gradients for the hidden layer
        hidden_layer.gradient_w = np.dot(train_set.T, hidden_layer.error)
        hidden_layer.gradient_b = hidden_layer.error

        hidden_layer.friction = beta * hidden_layer.friction \
                                - (self.learning_rate / len(train_set)) * hidden_layer.gradient_w

        # Update the weights and biases for hidden layer
        hidden_layer.weights = (1 - self.learning_rate * lin_factor / dataset_size) * \
                                hidden_layer.weights + hidden_layer.friction
        hidden_layer.biases += np.sum((-self.learning_rate) * hidden_layer.gradient_b, axis=0)

    def train(self, training_set, training_labels, batch_size, validation_set, validation_labels):
        training_set, training_labels = util.shuffle_sets(training_set, training_labels)
        epoch_map = {}
        for epoch in range(self.epochs):
            for batch in range(0, len(training_set), batch_size):
                self.backpropagation(training_set[batch:batch + batch_size], training_labels[batch:batch + batch_size],
                                     5, 0.9, len(training_set))
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
