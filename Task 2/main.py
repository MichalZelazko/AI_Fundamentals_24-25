import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from GaussianClass import GaussianClass
from Neuron import Neuron

st.sidebar.write("Parameters")
n_modes = st.sidebar.number_input("Number of Modes:", min_value=1, max_value=10, value=2)
n_samples = st.sidebar.number_input("Number of Samples:", min_value=10, max_value=500, value=100)

mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))

# Learning parameters
activation_function = st.sidebar.selectbox("Activation Function", ['heaviside', 'sigmoid'])
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)

# Initialize session state for data if not already present
if 'class_0_data' not in st.session_state:
    st.session_state.class_0_data = None
    st.session_state.class_1_data = None

# Create instances for Class 0 and Class 1
class_0 = GaussianClass(class_label=0, n_modes=n_modes, n_samples=n_samples, mean_range=mean, cov_range=cov)
class_1 = GaussianClass(class_label=1, n_modes=n_modes, n_samples=n_samples, mean_range=mean, cov_range=cov)

# Button to generate data
if st.sidebar.button("Generate Data"):
    # Generate data for both classes and store them in session state
    st.session_state.class_0_data = class_0.generate_data()
    st.session_state.class_1_data = class_1.generate_data()

# If data has been generated, display the plot and train the neuron
if st.session_state.class_0_data is not None and st.session_state.class_1_data is not None:
    data_0 = st.session_state.class_0_data
    data_1 = st.session_state.class_1_data
    
    # Combine data and labels for training
    X = np.vstack((data_0, data_1))
    y = np.hstack((np.zeros(data_0.shape[0]), np.ones(data_1.shape[0])))  # 0 for class 0, 1 for class 1

    # Initialize and train neuron
    neuron = Neuron(input_size=2, activation=activation_function, learning_rate=learning_rate)
    
    # Training loop (single epoch for simplicity)
    for x_sample, y_sample in zip(X, y):
        neuron.train(x_sample, y_sample)
    
    # Plot the data and decision boundary
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot Class 0 data (in red)
    ax.scatter(data_0[:, 0], data_0[:, 1], color='red', label='Class 0', alpha=0.6)
    
    # Plot Class 1 data (in blue)
    ax.scatter(data_1[:, 0], data_1[:, 1], color='blue', label='Class 1', alpha=0.6)

    # Create a mesh to plot the decision boundary
    xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict class for each point in the grid
    Z = np.array([neuron.predict(point) for point in grid])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title("Generated Data and Decision Boundary")
    ax.legend()

    # Display the plot
    st.pyplot(fig)
else:
    st.write("Click the button to generate data!")