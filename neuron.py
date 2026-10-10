import numpy as np
import activations

class Neuron:
    def __init__(self, n_inputs, activation_func = activations.step_function):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)
        self.activation = activation_func

    def forward(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)