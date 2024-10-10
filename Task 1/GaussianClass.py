import numpy as np

class GaussianClass:
    def __init__(self, class_label, n_modes, n_samples, mean_range, cov_range):
        self.class_label = class_label
        self.n_modes = n_modes
        self.n_samples = n_samples
        self.mean_range = mean_range
        self.cov_range = cov_range

    def generate_data(self):
        data = []
        for _ in range(self.n_modes):
            mean = np.random.uniform(self.mean_range[0], self.mean_range[1], 2)
            cov = np.random.uniform(self.cov_range[0], self.cov_range[1]) * np.eye(2)
            samples = np.random.multivariate_normal(mean, cov, self.n_samples)
            data.append(samples)
        
        return np.vstack(data)