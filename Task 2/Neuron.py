import numpy as np

class Neuron:
    def __init__(self, input_size, activation='heaviside', learning_rate=0.1):
        self.weights = np.random.randn(input_size)  # Initialize random weights
        self.bias = np.random.randn()  # Initialize random bias
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.activation, self.activation_derivative = self.get_activation(activation)

    # Activation Functions and their Derivatives
    def heaviside(s):
        return np.where(s >= 0, 1, 0)

    def sigmoid(s, beta=1):
        return 1 / (1 + np.exp(-beta * s))

    def sigmoid_derivative(s, beta=1):
        sig = sigmoid(s, beta)
        return beta * sig * (1 - sig)

    def relu(s):
        return np.maximum(0, s)

    def relu_derivative(s):
        return np.where(s > 0, 1, 0)

    def get_activation(self, activation):
        if activation == 'heaviside':
            return self.heaviside, lambda s: 1  # Derivative is 1 for Heaviside
        elif activation == 'sigmoid':
            return self.sigmoid, self.sigmoid_derivative
        elif activation == 'relu':
            return self.relu, self.relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def predict(self, x):
        # Calculate the weighted sum (s = w^T * x + b)
        s = np.dot(self.weights, x) + self.bias
        return self.activation(s)  # Pass 's' to the activation function

    def train(self, x, d):
        # Calculate the weighted sum (s = w^T * x + b)
        s = np.dot(self.weights, x) + self.bias
        y = self.activation(s)  # Activation function applied to 's'
        error = d - y
        self.weights += self.learning_rate * error * self.activation_derivative(s) * x
        self.bias += self.learning_rate * error * self.activation_derivative(s)