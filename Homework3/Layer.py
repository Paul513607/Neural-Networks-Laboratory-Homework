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
    gradient_w: np.ndarray
    gradient_b: np.ndarray

    def __init__(self, inputs_count: int, neurons_count: int):
        self.error = None
        self.gradient_w = None
        self.gradient_b = None
        self.friction = np.zeros((inputs_count, neurons_count))
        self.activation = np.zeros((1, 1))
        self.inputs_count = inputs_count
        self.neurons_count = neurons_count
        self.weights = np.random.normal(0, 1 / math.sqrt(inputs_count), size=(self.inputs_count, self.neurons_count))
        self.biases = np.random.rand(1, self.neurons_count)

    def eval(self, input_data: np.ndarray):
        z = (input_data @ self.weights) + self.biases
        return z

    def compute_activ(self, output: np.ndarray, activation_function: Callable[[np.ndarray], np.ndarray]):
        activation = []
        for row in output:
            activation += [activation_function(row)]
        self.activation = np.row_stack(activation)
        
    def get_activation(self, input_data, activation_function: Callable[[np.ndarray], np.ndarray]):
        self.activation = activation_function(self.eval(input_data))
        return self.activation
