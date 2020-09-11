from random import seed
from random import random
from math import exp
import numpy as np

weights3 = np.zeros(3)
weights3 = [(random()-0.5) for i in range(3)]
weights2 = np.zeros(3)
weights2 = [(random()-0.5) for i in range(3)]
weights1 = np.zeros(3)
weights1 = [(random()-0.5) for i in range(3)]
print(weights1)
print(weights2)

train = []
train.append(np.array([1, 1,-1]))
train.append(np.array([1, -1,1]))
train.append(np.array([-1, 1,1]))
train.append(np.array([-1, -1,-1]))

lrate = 0.25
epoch = 10000

for i in range(epoch):
    for row in train:
        zin1 = np.dot(weights1[1:],row[:-1])+weights1[0]