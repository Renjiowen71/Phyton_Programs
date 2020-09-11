import numpy as np
from Hebb import Hebb

train = []
train.append(np.array([1, 1]))
train.append(np.array([1, -1]))
train.append(np.array([-1, 1]))
train.append(np.array([-1, -1]))

labels = np.array([1, 1, 1, -1])

hebb= Hebb(2)
hebb.train(train, labels)

inputs = np.array([1, 1])
print(hebb.predict(inputs))

inputs = np.array([1, -1])
print(hebb.predict(inputs))
