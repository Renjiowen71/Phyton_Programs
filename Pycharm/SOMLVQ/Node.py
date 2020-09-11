from random import *


class Node:

    def __init__(self, FV_size=10, PV_size=10, Y=0, X=0):
        self.FV_size = FV_size
        self.PV_size = PV_size
        self.FV = [0.0] * FV_size  # Feature Vector
        self.PV = [0.0] * PV_size  # Prediction Vector
        self.X = X  # X location
        self.Y = Y  # Y location

        for i in range(FV_size):
            self.FV[i] = random()  # randrange(1,3,1) # Assign a random number from 0 to 1

        for i in range(PV_size):
            self.PV[i] = random()  # randrange(1,3,1) # Assign a random number from 0 to 1
