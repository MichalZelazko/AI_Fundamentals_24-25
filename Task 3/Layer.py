import numpy as np

class Layer:
    def __init__(self, num_of_inputs, num_of_neurons, activation="sigmoid"):
        self.weights = np.random.randn(num_of_neurons, num_of_inputs) * np.sqrt(2.0 / (num_of_inputs + num_of_neurons))
        self.biases = np.random.randn(num_of_neurons, 1)
        self.activation = activation
        self.outputs = None
        self.not_activated_outputs = None
        self.inputs = None
        self.deltas = None

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-np.clip(s, -500, 500)))

    def activate(self, s):
        if self.activation == 'sigmoid':
            return self.sigmoid(s)
        elif self.activation == 'heaviside':
            return np.where(s >= 0, 1, 0)
        elif self.activation == 'relu':
            return np.maximum(0, s)
        elif self.activation == 'leaky_relu':
            return np.where(s > 0, s, 0.01 * s)
        elif self.activation == 'tanh':
            return np.tanh(s)
        elif self.activation == 'sin':
            return np.sin(s)
        elif self.activation == 'sign':
            return np.sign(s)

    def derivative(self, s):
        if self.activation == 'sigmoid':
            return self.sigmoid(s) * (1 - self.sigmoid(s))
        elif self.activation == 'heaviside':
            return 1
        elif self.activation == 'relu':
            return np.where(s > 0, 1, 0)
        elif self.activation == 'leaky_relu':
            return np.where(s > 0, 1, 0.01)
        elif self.activation == 'tanh':
            return 1 - np.tanh(s) ** 2
        elif self.activation == 'sin':
            return np.cos(s)
        elif self.activation == 'sign':
            return 1

    def forward(self, x):
        self.inputs = x
        z = np.dot(self.weights, x) + self.biases
        self.not_activated_outputs = z
        self.outputs = self.activate(z)
        return self.outputs

    def compute_deltas(self, delta_next_layer, weights_next_layer):
        z = np.dot(self.weights, self.inputs) + self.biases
        self.deltas = np.dot(weights_next_layer.T, delta_next_layer) * self.derivative(self.outputs)
        return self.deltas