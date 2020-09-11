import numpy as np
from ClassBackPropagation import BackPropagation

bp1 = BackPropagation(2,2,1,1000,0.05)
bp2 = BackPropagation(2,2,1,10000,0.25)
bp3 = BackPropagation(2,2,1,25000,0.5)
bp2.network = bp1.network
bp3.network = bp1.network

train = []
train.append(np.array([1, 1,[1]]))
train.append(np.array([1, -1,[-1]]))
train.append(np.array([-1, 1,[-1]]))
train.append(np.array([-1, -1,[1]]))

print('Condition 1')
bp1.train_network(train)
inputs = np.array([1, 1])
print(bp1.predict(inputs))
inputs = np.array([-1, 1])
print(bp1.predict(inputs))
inputs = np.array([1, -1])
print(bp1.predict(inputs))
inputs = np.array([-1, -1])
print(bp1.predict(inputs))

print('Condition 2')
bp2.train_network(train)
inputs = np.array([1, 1])
print(bp2.predict(inputs))
inputs = np.array([-1, 1])
print(bp2.predict(inputs))
inputs = np.array([1, -1])
print(bp2.predict(inputs))
inputs = np.array([-1, -1])
print(bp2.predict(inputs))

print('Condition 3')
bp3.train_network(train)
inputs = np.array([1, 1])
print(bp3.predict(inputs))
inputs = np.array([-1, 1])
print(bp3.predict(inputs))
inputs = np.array([1, -1])
print(bp3.predict(inputs))
inputs = np.array([-1, -1])
print(bp3.predict(inputs))