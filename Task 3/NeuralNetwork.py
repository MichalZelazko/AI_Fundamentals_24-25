from Layer import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid", learning_rate=0.1):
        self.layers = []
        self.learning_rate = learning_rate
        # for i in range(1, len(layer_sizes)):
        #     self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activation))
        # Use different activations for hidden and output layers
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i], activation))
        
        # Output layer always uses sigmoid for binary classification
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], "sigmoid"))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, y_true):
        output_layer = self.layers[-1]
        output_error = output_layer.outputs - y_true
        output_layer.deltas = output_error * output_layer.derivative(output_layer.not_activated_outputs)

        for i in reversed(range(len(self.layers) - 1)):
            next_layer = self.layers[i + 1]
            self.layers[i].compute_deltas(next_layer.deltas, next_layer.weights)

        for layer in self.layers:
            layer.weights -= self.learning_rate * np.dot(layer.deltas, layer.inputs.T)
            layer.biases -= self.learning_rate * np.sum(layer.deltas, axis=1, keepdims=True)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                x_i = x_i.reshape(-1, 1)
                y_i = y_i.reshape(-1, 1)
                self.forward(x_i)
                self.backpropagate(y_i)

    def predict(self, X):
        probabilities = []
        for x_i in X:
            x_i = x_i.reshape(-1, 1)
            output = self.forward(x_i)
            probabilities.append(output.flatten())
        predictions = np.argmax(probabilities, axis=1)
        return np.array(predictions)