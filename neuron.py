import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand(1)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias