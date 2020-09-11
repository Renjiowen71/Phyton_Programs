from random import seed
from random import random
from math import exp

class BackPropagation(object):

    def __init__(self, n_inputs, n_hidden, n_outputs, threshold, learning_rate):
        self.network = self.initialize_network(n_inputs, n_hidden, n_outputs)
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs

    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights':[(random()-0.5) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights':[(random()-0.5) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 2.0 / (1.0 + exp(-activation))-1

    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def transfer_derivative(self, output):
        return (1.0+output) * (1.0 - output)/2

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += self.learning_rate * neuron['delta']

    def train_network(self, train):
        for epoch in range(self.threshold):
            sum_error = 0
            i=0
            for row in train:
                outputs = self.forward_propagate( row)
                expected = row[-1]
                self.backward_propagate_error(expected)
                self.update_weights(row)


    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs


    def predicts(self, row):
        outputs = self.predict(row)
        for i in range(0,self.n_outputs):
            outputs[i] = round(outputs[i])
        return outputs