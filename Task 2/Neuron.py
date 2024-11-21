import numpy as np

class Neuron:
    def __init__(self, input_size, activation='heaviside', learning_rate_min=0.001, learning_rate_max=0.1):
        self.weights = np.random.randn(input_size + 1)  # +1 for the bias
        self.activation = activation
        self.learning_rate_min = learning_rate_min
        self.learning_rate_max = learning_rate_max

    # Activation functions
    def heaviside(self, s):
        return np.where(s >= 0, 1, 0)

    def sigmoid(self, s, beta=1):
        return 1 / (1 + np.exp(-beta * s))

    def sigmoid_derivative(self, s, beta=1):
        return beta * self.sigmoid(s, beta) * (1 - self.sigmoid(s, beta))

    def tanh(self, s):
        return np.tanh(s)

    def tanh_derivative(self, s):
        return (1/np.cosh(s)) ** 2

    def sin(self, s):
        return np.sin(s)

    def sin_derivative(self, s):
        return np.cos(s)

    def sign(self, s):
        return np.sign(s)

    def relu(self, s):
        return np.maximum(0, s)

    def relu_derivative(self, s):
        return np.where(s > 0, 1, 0)

    def leaky_relu(self, s, alpha=0.01):
        return np.where(s > 0, s, alpha * s)

    def leaky_relu_derivative(self, s, alpha=0.01):
        return np.where(s > 0, 1, alpha)

    def get_learning_rate(self, epoch, total_epochs):
        return self.learning_rate_min + (self.learning_rate_max - self.learning_rate_min) * \
               (1 + np.cos((epoch / total_epochs) * np.pi))

    def activate(self, s):
        if self.activation == 'heaviside':
            return self.heaviside(s)
        elif self.activation == 'sigmoid':
            return self.sigmoid(s)
        elif self.activation == 'sin':
            return self.sin(s)
        elif self.activation == 'tanh':
            return self.tanh(s)
        elif self.activation == 'sign':
            return self.sign(s)
        elif self.activation == 'relu':
            return self.relu(s)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu(s)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def derivative(self, s):
        if self.activation == 'heaviside':
            return 1
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative(s)
        elif self.activation == 'sin':
            return self.sin_derivative(s)
        elif self.activation == 'tanh':
            return self.tanh_derivative(s)
        elif self.activation == 'sign':
            return 1
        elif self.activation == 'relu':
            return self.relu_derivative(s)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu_derivative(s)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def forward(self, x):
        x_with_bias = np.append(x, 1)
        s = np.dot(self.weights, x_with_bias)
        return self.activate(s)


    def train(self, X, y, epochs=100):        
        for epoch in range(epochs):
            current_lr = self.get_learning_rate(epoch, epochs)
            for x_i, d in zip(X, y):
                x_i_with_bias = np.append(x_i, 1)
                s = np.dot(self.weights, x_i_with_bias)
                y_pred = self.activate(s)
                derivative = self.derivative(s)
                error = d - y_pred
                self.weights += current_lr * error * derivative * x_i_with_bias