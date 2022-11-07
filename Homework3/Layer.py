import math
from typing import Callable

import numpy as np

class Layer:
    inputs_count: int
    neurons_count: int
    weights: np.ndarray
    biases: np.ndarray
    activation: np.ndarray
    error: np.ndarray
    friction: np.ndarray
    cost_gradient_w: np.ndarray
    cost_gradient_b: np.ndarray

    def __init__(self, inputs_count: int, neurons_count: int):
        self.error = None
        self.cost_gradient_w = None
        self.cost_gradient_b = None
        self.friction = None
        self.activation = np.zeros((1, 1))
        self.inputs_count = inputs_count
        self.neurons_count = neurons_count
        self.weights = np.random.normal(0, 1 / math.sqrt(inputs_count), size=(self.inputs_count, self.neurons_count))
        self.biases = np.random.rand(1, self.neurons_count)

    def eval(self, input: np.ndarray):
        z = (input @ self.weights) + self.biases
        return z

    def activation_for_layer(self, output: np.ndarray, activation_function: Callable[[np.ndarray], np.ndarray]):
        activation = []
        for row in output:
            activation += [activation_function(row)]
        self.activation = np.row_stack(activation)
