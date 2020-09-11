import numpy as np

class SOM:

    def __init__(self, net_x_dim, net_y_dim, num_features):
        self.net_dim = np.array([net_x_dim, net_y_dim])
        self.init_radius = min(self.net_dim[0], self.net_dim[1])
        self.n_feat = num_features
        self.initialize()

    def initialize(self):
        self.net = np.random.random\
            ([self.net_dim[0],self.net_dim[1], self.n_feat])

    def train(self, data, n_epochs=100, learn_rate=0.01, rWeight=False):
        if rWeight:
            self.initialize()
        num_rows = data.shape[0]
        indices = np.arange(num_rows)
        self.time_constant = n_epochs / np.log(self.init_radius)

        for i in range(1, n_epochs + 1):
            radius = self.decay_radius(i)
            learning_rate = self.decay_learning_rate(learn_rate, i, n_epochs)
            vis_interval = int(n_epochs/10)
            np.random.shuffle(indices)
            for record in indices:
                row_t = data[record, :]
                bmu, bmu_idx = self.find_bmu(row_t)
                for x in range(self.net_dim[0]):
                    for y in range(self.net_dim[1]):
                        weight = self.net[x, y, :].reshape(1, self.n_feat)
                        w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                        if w_dist <= radius ** 2:
                            influence = self.calc_inf(w_dist, radius)
                            new_w = weight + (learning_rate * influence * (row_t - weight))
                            self.net[x, y, :] = new_w.reshape(1, self.n_feat)


    def calc_inf(self, distance, radius):
        return np.exp(-distance / (2 * (radius ** 2)))

    def find_bmu(self, row_t):
        bmu_idx = np.array([0, 0])
        min_dist = np.iinfo(np.int).max
        for x in range(self.net_dim[0]):
            for y in range(self.net_dim[1]):
                weight_k = self.net[x, y, :].reshape(1, self.n_feat)
                sq_dist = np.sum((weight_k - row_t) ** 2)
                if sq_dist < min_dist:
                    min_dist = sq_dist
                    bmu_idx = np.array([x, y])
        bmu = self.net[bmu_idx[0], bmu_idx[1], :].reshape(1, self.n_feat)
        return bmu, bmu_idx

    def predict(self, data):
        bmu, bmu_idx = self.find_bmu(data)
        return bmu, bmu_idx

    def decay_radius(self, iteration):
        return self.init_radius * np.exp(-iteration / self.time_constant)

    def decay_learning_rate(self,learn_rate, iteration, num_iterations):
        return learn_rate * np.exp(-iteration / num_iterations)

