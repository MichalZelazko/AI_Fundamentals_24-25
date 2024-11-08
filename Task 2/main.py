import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from GaussianClass import GaussianClass
from Neuron import Neuron

def main():
    # Sidebar inputs
    st.sidebar.write("Parameters")
    n_modes = st.sidebar.number_input("Number of Modes:", min_value=1, max_value=3, value=1)
    n_samples = st.sidebar.number_input("Number of Samples:", min_value=10, max_value=500, value=100)
    mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
    cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))

    # Activation function selection
    activation_func = st.sidebar.selectbox("Activation Function:", 
                                         ['heaviside', 'sigmoid', 'sin', 'tanh', 'sign', 'relu', 'leaky_relu'])

    # Learning rate inputs
    st.sidebar.write("Learning Rate Parameters")
    learning_rate_min = st.sidebar.slider("Minimum Learning Rate", 0.001, 0.1, 0.001, step=0.001, format="%.3f")
    learning_rate_max = st.sidebar.slider("Maximum Learning Rate", 0.01, 0.5, 0.1, step=0.01, format="%.2f")
    epochs = st.sidebar.slider("Number of Epochs", 100, 1000, 500, step=100)

    # Initialize session state for data
    if 'class_0_data' not in st.session_state:
        st.session_state.class_0_data = None
        st.session_state.class_1_data = None

    # Create instances for Class 0 and Class 1
    class_0 = GaussianClass(class_label=0, n_modes=n_modes, n_samples=n_samples, 
                           mean_range=mean, cov_range=cov)
    class_1 = GaussianClass(class_label=1, n_modes=n_modes, n_samples=n_samples, 
                           mean_range=mean, cov_range=cov)

    # Button to generate data
    if st.sidebar.button("Generate Data"):
        st.session_state.class_0_data = class_0.generate_data()
        st.session_state.class_1_data = class_1.generate_data()

    # If data has been generated, display the plot
    if st.session_state.class_0_data is not None and st.session_state.class_1_data is not None:
        # Create figure for data and decision boundary
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot Class 0 and Class 1 data
        ax.scatter(st.session_state.class_0_data[:, 0], st.session_state.class_0_data[:, 1], 
                  color='red', label='Class 0', alpha=0.6)
        ax.scatter(st.session_state.class_1_data[:, 0], st.session_state.class_1_data[:, 1], 
                  color='blue', label='Class 1', alpha=0.6)

        # Prepare training data
        X = np.vstack([st.session_state.class_0_data, st.session_state.class_1_data])
        y = np.hstack([np.zeros(st.session_state.class_0_data.shape[0]), 
                      np.ones(st.session_state.class_1_data.shape[0])])

        # Train the neuron
        neuron = Neuron(input_size=2, activation=activation_func, 
                       learning_rate_min=learning_rate_min, 
                       learning_rate_max=learning_rate_max)
        neuron.train(X, y, epochs=epochs)
        
        # Plot decision boundary
        xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([neuron.forward(x) for x in grid])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=[-0.01, 0.5, 1.01], 
                    colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title("Data and Decision Boundary")
        ax.legend()
        ax.grid(True)
        
        # Display the figure
        st.pyplot(fig)
    else:
        st.write("Click the 'Generate Data' button to start!")

if __name__ == "__main__":
    main()