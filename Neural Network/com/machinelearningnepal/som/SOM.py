import numpy as np

class SOM:

    def __init__(self, net_x_dim, net_y_dim, num_features):
        self.network_dimensions = np.array([net_x_dim, net_y_dim])
        self.init_radius = min(self.network_dimensions[0], self.network_dimensions[1])
        self.num_features = num_features
        self.initialize()

    def initialize(self):
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.num_features))

    def train(self, data, num_epochs=100, init_learning_rate=0.01, resetWeights=False):
        if resetWeights:
            self.initialize()
        num_rows = data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = num_epochs / np.log(self.init_radius)

        for i in range(1, num_epochs + 1):
            radius = self.decay_radius(i)
            learning_rate = self.decay_learning_rate(init_learning_rate, i, num_epochs)
            vis_interval = int(num_epochs/10)
            np.random.shuffle(indices)
            for record in indices:
                row_t = data[record, :]
                bmu, bmu_idx = self.find_bmu(row_t)
                for x in range(self.network_dimensions[0]):
                    for y in range(self.network_dimensions[1]):
                        weight = self.net[x, y, :].reshape(1, self.num_features)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                        if w_dist <= radius ** 2:
                            influence = SOM.calculate_influence(w_dist, radius)
                            new_w = weight + (learning_rate * influence * (row_t - weight))
                            self.net[x, y, :] = new_w.reshape(1, self.num_features)


    def calculate_influence(distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def find_bmu(self, row_t):
        bmu_idx = np.array([0, 0])
        min_dist = np.iinfo(np.int).max
        for x in range(self.network_dimensions[0]):
            for y in range(self.network_dimensions[1]):
                weight_k = self.net[x, y, :].reshape(1, self.num_features)
                sq_dist = np.sum((weight_k - row_t) ** 2)
                if sq_dist < min_dist:
                    min_dist = sq_dist
                    bmu_idx = np.array([x, y])
        bmu = self.net[bmu_idx[0], bmu_idx[1], :].reshape(1, self.num_features)
        return bmu, bmu_idx

    def predict(self, data):
        bmu, bmu_idx = self.find_bmu(data)
        return bmu, bmu_idx

    def decay_radius(self, iteration):
        return self.init_radius * np.exp(-iteration / self.time_constant)

    def decay_learning_rate(self, initial_learning_rate, iteration, num_iterations):
        return initial_learning_rate * np.exp(-iteration / num_iterations)

