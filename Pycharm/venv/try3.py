from random import seed
from random import random
from math import exp
import numpy as np

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[(random()-0.5) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[(random()-0.5) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return 2.0 / (1.0 + exp(-activation))-1

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return (output +1.0)* (1.0 - output)/2

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row[:-1])
			expected = row[-1]
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f' % (epoch, l_rate))


def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs

net = initialize_network(15,11,3)
for layer in net:
    print(layer)
train = []
train.append(np.array([-1,1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1,[-1,-1,-1]]))
train.append(np.array([1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,[-1,-1,1]]))
train.append(np.array([-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,[-1,1,-1]]))
train.append(np.array([1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,[-1,1,1]]))
train.append(np.array([1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1,[1,-1,-1]]))
train.append(np.array([1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,[1,-1,1]]))
train.append(np.array([-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,[1,1,-1]]))
train.append(np.array([1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1,1,[1,1,1]]))

train_network(net, train,1,1000,0.05)

print(predict(net,[-1,1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1]))
print(predict(net,[1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1]))
print(predict(net,[-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,1,]))
print(predict(net,[1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1]))
print(predict(net,[1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,1,1]))
print(predict(net,[1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1]))
print(predict(net,[-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,1,1,1]))
print(predict(net,[1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1,1]))

