# from matplotlib.colors import ListedColormap
# import numpy as np
# import streamlit as st
# import matplotlib.pyplot as plt
# from GaussianClass import GaussianClass
# from Neuron import Neuron

# st.sidebar.write("Parameters")
# n_modes = st.sidebar.number_input("Number of Modes:", min_value=1, max_value=10, value=2)
# n_samples = st.sidebar.number_input("Number of Samples:", min_value=10, max_value=500, value=100)
# mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
# cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))

# # Activation function selection
# activation_function = st.sidebar.selectbox("Activation Function", ('heaviside', 'sigmoid', 'sin', 'tanh', 'sign', 'relu', 'leaky_relu'))
# learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.01, 0.001, "%4f")

# # Initialize session state for data if not already present
# if 'class_0_data' not in st.session_state:
#     st.session_state.class_0_data = None
#     st.session_state.class_1_data = None

# # Create instances for Class 0 and Class 1
# class_0 = GaussianClass(class_label=0, n_modes=n_modes, n_samples=n_samples, mean_range=mean, cov_range=cov)
# class_1 = GaussianClass(class_label=1, n_modes=n_modes, n_samples=n_samples, mean_range=mean, cov_range=cov)

# # Generate data button
# if st.sidebar.button("Generate Data"):
#     st.session_state.class_0_data = class_0.generate_data()
#     st.session_state.class_1_data = class_1.generate_data()

# # If data is generated, show plot and decision boundary
# if st.session_state.class_0_data is not None and st.session_state.class_1_data is not None:
#     # Combine class data
#     X = np.vstack((st.session_state.class_0_data, st.session_state.class_1_data))
#     y = np.hstack((np.zeros(len(st.session_state.class_0_data)), np.ones(len(st.session_state.class_1_data))))

#     # Initialize the neuron with the selected activation function
#     neuron = Neuron(n_inputs=2, activation_function=activation_function, learning_rate=learning_rate)

#     # Train the neuron on the dataset
#     for i in range(len(X)):
#         neuron.train(X[i], y[i])

#     # Plotting the decision boundary
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Create a grid of points to evaluate the model
#     x_min, x_max = -6, 6
#     y_min, y_max = -6, 6
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # Classify each point in the grid using the trained neuron
#     Z = np.array([neuron.forward(point) for point in grid_points])
#     Z = Z.reshape(xx.shape)

#     # Define custom colormap for decision boundary
#     cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    
#     # Plot the decision boundary as a contour plot
#     ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

#     # Plot the actual data points
#     ax.scatter(st.session_state.class_0_data[:, 0], st.session_state.class_0_data[:, 1], color='red', label='Class 0', alpha=0.8)
#     ax.scatter(st.session_state.class_1_data[:, 0], st.session_state.class_1_data[:, 1], color='blue', label='Class 1', alpha=0.8)

#     ax.set_title(f"Decision Boundary using {activation_function} Activation")
#     ax.legend()

#     # Show the plot
#     st.pyplot(fig)
# else:
#     st.write("Click the button to generate data!")


import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from GaussianClass import GaussianClass
from Neuron import Neuron

# Update the Streamlit interface
def main():
    # Sidebar inputs
    st.sidebar.write("Parameters")
    n_modes = st.sidebar.number_input("Number of Modes:", min_value=1, max_value=10, value=2)
    n_samples = st.sidebar.number_input("Number of Samples:", min_value=10, max_value=500, value=100)
    mean = st.sidebar.slider("Mean Range", -5.0, 5.0, (-5.0, 5.0))
    cov = st.sidebar.slider("Covariance Range", 0.0, 1.0, (0.05, 0.25))

    # Activation function selection
    activation_func = st.sidebar.selectbox("Activation Function:", ['heaviside', 'sigmoid', 'sin', 'tanh', 'sign', 'relu', 'leaky_relu'])

    # Learning rate input
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

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
        
        # Prepare training data
        X = np.vstack([st.session_state.class_0_data, st.session_state.class_1_data])
        y = np.hstack([np.zeros(st.session_state.class_0_data.shape[0]), np.ones(st.session_state.class_1_data.shape[0])])

        # Train the neuron
        neuron = Neuron(input_size=2, activation=activation_func, learning_rate=learning_rate)
        neuron.train(X, y, epochs=100)
        
        # Plot decision boundary
        xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-6, 6, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([neuron.forward(x) for x in grid])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, levels=[-0.01, 0.5, 1.01], colors=['#FFAAAA', '#AAAAFF'], alpha=0.3)
        st.pyplot(fig)
    else:
        st.write("Click the button to generate data!")


if __name__ == "__main__":
    main()