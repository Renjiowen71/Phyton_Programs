import numpy as np
from Perceptron import Perceptron

train = []
train.append(np.array([1, 1]))
train.append(np.array([1, -1]))
train.append(np.array([-1, 1]))
train.append(np.array([-1, -1]))

labels = np.array([-1, 1, -1, -1])

perceptron = Perceptron(2,10,1)
perceptron.train(train, labels)

inputs = np.array([1, 1])
print(perceptron.predict(inputs))

#=> 1

inputs = np.array([1, -1])
print(perceptron.predict(inputs))
#=> 0
inputs = np.array([-1, 1])
print(perceptron.predict(inputs))
inputs = np.array([-1, -1])
print(perceptron.predict(inputs))