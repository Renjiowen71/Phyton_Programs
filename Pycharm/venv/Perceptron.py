import numpy as np
class Perceptron(object):

    def __init__(self, no_of_inputs, threshold, learning_rate):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        theta = 0.5
        y = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if y > theta:
            return 1
        elif y > -theta:
            return 0
        else:
            return -1

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction!=label:
                    self.weights[1:] += self.learning_rate * label * inputs
                    self.weights[0] += self.learning_rate * label