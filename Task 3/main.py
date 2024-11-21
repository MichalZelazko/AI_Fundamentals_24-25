import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from GaussianClass import GaussianClass
from NeuralNetwork import NeuralNetwork

def main():
    st.title("Neural Network")

    # Sidebar configuration for selecting parameters
    st.sidebar.header("Settings")
    
    num_of_modes_class_0 = st.sidebar.number_input("Number of Modes for Class 0", min_value=1, max_value=10, value=2)
    num_of_modes_class_1 = st.sidebar.number_input("Number of Modes for Class 1", min_value=1, max_value=10, value=2)
    samples_per_mode = st.sidebar.number_input("Number of Samples per Mode", min_value=10, max_value=1000, value=100)
    mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
    cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))
    activation_function = st.sidebar.selectbox(
        "Activation Function", 
        ["relu", "leaky_relu", "sigmoid", "sin", "tanh", "sign", "heaviside"]
    )
    
    # Slider for configuring the neural network structure
    num_of_hidden_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=3, value=2)
    neurons_per_hidden_layer = st.sidebar.slider("Neurons per Hidden Layer", min_value=2, max_value=10, value=5)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f")
    epochs = st.sidebar.slider("Number of Epochs", 100, 1000, 300, step=100)
    
    # Generate data and train neural network when button is clicked
    if st.sidebar.button("Generate Data and Train Neural Network"):
        class_0 = GaussianClass(class_label=0, n_modes=num_of_modes_class_0, n_samples=samples_per_mode, mean_range=mean, cov_range=cov)
        class_1 = GaussianClass(class_label=1, n_modes=num_of_modes_class_1, n_samples=samples_per_mode, mean_range=mean, cov_range=cov)
        # Generate data for Class 0 and Class 1
        class_0_data = class_0.generate_data()
        class_1_data = class_1.generate_data()
        
        # Prepare labels for both classes
        class_0_labels = np.zeros((class_0_data.shape[0], 2))
        class_0_labels[:, 0] = 1  # Class 0 is [1, 0]
        
        class_1_labels = np.zeros((class_1_data.shape[0], 2))
        class_1_labels[:, 1] = 1  # Class 1 is [0, 1]
        
        # Combine data and labels
        data = np.vstack([class_0_data, class_1_data])
        labels = np.vstack([class_0_labels, class_1_labels])

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        
        # Set up the neural network
        input_size = 2
        output_size = 2
        layer_sizes = [input_size] + [neurons_per_hidden_layer] * num_of_hidden_layers + [output_size]
        neural_net = NeuralNetwork(layer_sizes, learning_rate=learning_rate, activation=activation_function)
        neural_net.train(data, labels, epochs=epochs)
        
        # Visualization using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the generated data
        ax.scatter(class_0_data[:, 0], class_0_data[:, 1], color='red', label='Class 0', alpha=0.6)
        ax.scatter(class_1_data[:, 0], class_1_data[:, 1], color='blue', label='Class 1', alpha=0.6)
        
        # Generate a meshgrid for the decision boundary
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict class probabilities for each point in the grid
        predictions = neural_net.predict(grid)
        # zz = np.argmax(predictions, axis=1).reshape(xx.shape)
        zz = predictions.reshape(xx.shape)
        
        
        # Plot the decision boundary
        ax.contourf(xx, yy, zz, levels=[-0.01, 0.5, 1.01], colors=['#FF0000', '#0000FF'], alpha=0.3)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title("Neural Network Decision Boundary")
        ax.legend()
        ax.grid(True)
        
        # Display the plot in Streamlit
        st.pyplot(fig)

if __name__ == '__main__':
    main()