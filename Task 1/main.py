import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import GaussianClass

# Sidebar inputs
st.sidebar.write("Parameters")
n_modes = st.sidebar.number_input("Number of Modes:", min_value=1, max_value=10, value=2)
n_samples = st.sidebar.number_input("Number of Samples:", min_value=10, max_value=500, value=100)

mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))

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

# If data has been generated, display the plot
if st.session_state.class_0_data is not None and st.session_state.class_1_data is not None:
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot Class 0 data (in red)
    ax.scatter(st.session_state.class_0_data[:, 0], st.session_state.class_0_data[:, 1], color='red', label='Class 0', alpha=0.6)

    # Plot Class 1 data (in blue)
    ax.scatter(st.session_state.class_1_data[:, 0], st.session_state.class_1_data[:, 1], color='blue', label='Class 1', alpha=0.6)

    # Set plot limits based on mean and covariance range
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    ax.set_title("Generated Data from Two Classes")
    ax.legend()

    # Display the plot
    st.pyplot(fig)
else:
    st.write("Click the button to generate data!")