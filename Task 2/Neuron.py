# import numpy as np


# class Neuron:
#     def __init__(self, n_inputs, activation_function='heaviside', learning_rate=0.01):
#         # Initialize weights randomly
#         self.weights = np.random.randn(n_inputs)
#         self.bias = np.random.randn()
#         self.activation_function = activation_function
#         self.learning_rate = learning_rate

#     def activate(self, s):
#         """Apply the chosen activation function."""
#         if self.activation_function == 'heaviside':
#             return np.heaviside(s, 1)
#         elif self.activation_function == 'sigmoid':
#             return 1 / (1 + np.exp(-s))
#         elif self.activation_function == 'sin':
#             return np.sin(s)
#         elif self.activation_function == 'tanh':
#             return np.tanh(s)
#         elif self.activation_function == 'sign':
#             return np.sign(s)
#         elif self.activation_function == 'relu':
#             return np.maximum(0, s)
#         elif self.activation_function == 'leaky_relu':
#             return np.where(s > 0, s, 0.01 * s)
#         else:
#             raise ValueError(f"Unknown activation function: {self.activation_function}")

#     def derivative(self, s):
#         """Compute the derivative of the activation function."""
#         if self.activation_function == 'heaviside':
#             return 1
#         elif self.activation_function == 'sigmoid':
#             sigmoid_value = self.activate(s)
#             return sigmoid_value * (1 - sigmoid_value)
#         elif self.activation_function == 'sin':
#             return np.cos(s)
#         elif self.activation_function == 'tanh':
#             return 1 - np.tanh(s) ** 2
#         elif self.activation_function == 'sign':
#             return 0  # Derivative of sign is technically undefined, treated as 0
#         elif self.activation_function == 'relu':
#             return np.where(s > 0, 1, 0)
#         elif self.activation_function == 'leaky_relu':
#             return np.where(s > 0, 1, 0.01)
#         else:
#             raise ValueError(f"Unknown activation function: {self.activation_function}")

#     def forward(self, x):
#         """Forward pass through the neuron."""
#         s = np.dot(self.weights, x) + self.bias
#         return self.activate(s)

#     def train(self, x, d):
#         """Train the neuron using one sample."""
#         s = np.dot(self.weights, x) + self.bias
#         y = self.activate(s)
#         error = d - y

#         # Update weights and bias using the weight update formula
#         self.weights += self.learning_rate * error * self.derivative(s) * x
#         self.bias += self.learning_rate * error * self.derivative(s)

#         return y  # Return the prediction after the update

# import numpy as np
# import matplotlib.pyplot as plt

# class Neuron:
#     def __init__(self, learning_rate=0.01, activation='sigmoid', beta=1.0):
#         self.weights = None
#         self.bias = None
#         self.learning_rate = learning_rate
#         self.activation = activation
#         self.beta = beta
        
#     def activation_function(self, s):
#         if self.activation == 'heaviside':
#             return np.where(s >= 0, 1, 0)
#         elif self.activation == 'sigmoid':
#             return 1 / (1 + np.exp(-self.beta * s))
#         elif self.activation == 'sin':
#             return np.sin(s)
#         elif self.activation == 'tanh':
#             return np.tanh(s)
#         elif self.activation == 'sign':
#             return np.sign(s)
#         elif self.activation == 'relu':
#             return np.maximum(0, s)
#         elif self.activation == 'leaky_relu':
#             return np.where(s > 0, s, 0.01 * s)
    
#     def activation_derivative(self, s):
#         if self.activation == 'heaviside':
#             return 1  # Assumed derivative as per requirements
#         elif self.activation == 'sigmoid':
#             fx = self.activation_function(s)
#             return self.beta * fx * (1 - fx)
#         elif self.activation == 'sin':
#             return np.cos(s)
#         elif self.activation == 'tanh':
#             return 1 - np.tanh(s) ** 2
#         elif self.activation == 'sign':
#             return 0
#         elif self.activation == 'relu':
#             return np.where(s > 0, 1, 0)
#         elif self.activation == 'leaky_relu':
#             return np.where(s > 0, 1, 0.01)
#         else:
#             raise ValueError("Derivative not implemented for this activation function")
    
#     def fit(self, X, y, epochs=100, variable_learning_rate=False):
#         if self.weights is None:
#             self.weights = np.random.randn(X.shape[1])
#             self.bias = np.random.randn()
        
#         self.history = []
#         for epoch in range(epochs):
#             if variable_learning_rate:
#                 curr_lr = self._variable_learning_rate(epoch, epochs)
#             else:
#                 curr_lr = self.learning_rate
                
#             for xi, yi in zip(X, y):
#                 # Forward pass
#                 s = np.dot(self.weights, xi) + self.bias
#                 y_pred = self.activation_function(s)
                
#                 # Backward pass
#                 error = yi - y_pred
#                 derivative = self.activation_derivative(s)
                
#                 # Update weights and bias
#                 self.weights += curr_lr * error * derivative * xi
#                 self.bias += curr_lr * error * derivative
            
#             # Calculate accuracy for this epoch
#             accuracy = np.mean(self.predict(X) == y)
#             self.history.append(accuracy)
    
#     def predict(self, X):
#         s = np.dot(X, self.weights) + self.bias
#         return self.activation_function(s)
    
#     def _variable_learning_rate(self, n, n_max):
#         eta_min = 0.01
#         eta_max = 0.1
#         return eta_min + (eta_max - eta_min) * (1 + np.cos(n/n_max * np.pi))
    
#     def plot_decision_boundary(self, X, y):
#         fig, ax = plt.subplots(figsize=(8, 8))
        
#         # Create a mesh grid
#         x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#         y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                             np.linspace(y_min, y_max, 100))
        
#         # Get predictions for all points in the mesh
#         Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
        
#         # Plot decision boundary
#         ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
        
#         # Plot training points
#         ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
#         ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)
        
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#         ax.legend()
        
#         return fig

import numpy as np

class Neuron:
    def __init__(self, input_size, activation='heaviside', learning_rate=0.01):
        self.weights = np.random.randn(input_size + 1)  # +1 for the bias
        self.activation = activation
        self.learning_rate = learning_rate

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
        return 1 - np.tanh(s) ** 2

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

    # Forward pass
    def forward(self, x):
        x_with_bias = np.append(x, 1)  # Add bias term
        s = np.dot(self.weights, x_with_bias)

        if self.activation == 'heaviside':
            return self.heaviside(s)
        elif self.activation == 'sigmoid':
            return self.sigmoid(s)
        elif self.activation == 'tanh':
            return self.tanh(s)
        elif self.activation == 'sin':
            return self.sin(s)
        elif self.activation == 'sign':
            return self.sign(s)
        elif self.activation == 'relu':
            return self.relu(s)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu(s)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    # Training function
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for x_i, d in zip(X, y):
                x_i_with_bias = np.append(x_i, 1)  # Add bias term
                s = np.dot(self.weights, x_i_with_bias)

                # Predicted output and derivative
                if self.activation == 'heaviside':
                    y_pred = self.heaviside(s)
                    derivative = 1  # Heaviside derivative is 1 for simplicity
                elif self.activation == 'sigmoid':
                    y_pred = self.sigmoid(s)
                    derivative = self.sigmoid_derivative(s)
                elif self.activation == 'tanh':
                    y_pred = self.tanh(s)
                    derivative = self.tanh_derivative(s)
                elif self.activation == 'sin':
                    y_pred = self.sin(s)
                    derivative = self.sin_derivative(s)
                elif self.activation == 'sign':
                    y_pred = self.sign(s)
                    derivative = 0  # Derivative of sign is 0
                elif self.activation == 'relu':
                    y_pred = self.relu(s)
                    derivative = self.relu_derivative(s)
                elif self.activation == 'leaky_relu':
                    y_pred = self.leaky_relu(s)
                    derivative = self.leaky_relu_derivative(s)
                else:
                    raise ValueError(f"Unsupported activation function: {self.activation}")

                # Error
                error = d - y_pred
                
                # Weight update
                self.weights += self.learning_rate * error * derivative * x_i_with_bias

