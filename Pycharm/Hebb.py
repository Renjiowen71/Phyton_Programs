import numpy as np


class Hebb(object):

    def __init__(self, no_of_inputs):
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        y = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if y > 0:
            return 1
        else:
            return -1

    def train(self, training_inputs, labels):
        for inputs, label in zip(training_inputs, labels):
            self.weights[1:] +=  label * inputs
            self.weights[0] +=  label