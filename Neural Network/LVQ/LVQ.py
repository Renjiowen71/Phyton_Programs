import numpy as np
from numpy import zeros
class LVQ:
    def __init__(self, vectorLength, trainingPatterns, decayRate, minAlpha, patternArray,
                 target):
        self.mNumberOfClusters = self.cluster(target)
        self.mVectorLength = vectorLength
        self.mTrainingPatterns = trainingPatterns
        self.mDecayRate = decayRate
        self.mMinimumAlpha = minAlpha
        self.alpha = 0.6
        self.d = []  # Network nodes. The "clusters"
        self.w = []  # Weight matrix.
        self.mPatterns = patternArray
        self.mTarget = target
        return

    def initialize_arrays(self):
        self.d = [0.0] * self.mNumberOfClusters
        for i in range(self.mNumberOfClusters):
            self.w.append([0.0] * self.mVectorLength)
            for j in range(self.mVectorLength):
                self.w[i][j] = self.mPatterns[i][j]

        return

    def compute_input(self, Array):
        self.d = [0.0] * self.mNumberOfClusters

        for i in range(self.mNumberOfClusters):
            for j in range(self.mVectorLength):
                self.d[i] += (self.w[i][j] - Array[j])**2

        return

    def compute_input_2D(self, array, vector_number):
        self.d = [0.0] * self.mNumberOfClusters

        for i in range(self.mNumberOfClusters):
            for j in range(self.mVectorLength):
                self.d[i] += (self.w[i][j] - array[vector_number][j])**2

        return

    def get_cluster(self, array_att):
        # Compute input for all nodes.
        self.compute_input(array_att)

        return self.get_minimum(self.d)

    def get_clusters(self, inputPatterns, array1_att):
        a = zeros((1, inputPatterns))
        for i in range (inputPatterns):
            inputpattern = array1_att[i,:]
            self.compute_input(inputpattern)
            a[0][i] = self.get_minimum(self.d)
        return a


    def get_minimum(self, nodeArray):
        minimum = 0;
        done = False

        while not done:
            found = False
            for i in range(self.mNumberOfClusters):
                if i != minimum:
                    if nodeArray[i] < nodeArray[minimum]:
                        minimum = i
                        found = True

            if found == False:
                done = True

        return minimum

    def update_weights(self, vector_number, dmin):
        for i in range(self.mVectorLength):
            if dmin == self.mTarget[0][vector_number]:
                self.w[dmin][i] += (self.alpha * (self.mPatterns[vector_number][i] - self.w[dmin][i]))
            else:
                self.w[dmin][i] -= (self.alpha * (self.mPatterns[vector_number][i] - self.w[dmin][i]))

        return

    def cluster(self,target):
        clus = 0;
        for i in range (target.size):
            if clus<target[0][i]:
                clus = target[0][i]
        return clus+1

    def training(self):
        while self.alpha > self.mMinimumAlpha:
            for i in range(self.mTrainingPatterns):
                self.compute_input_2D(self.mPatterns, i)
                dmin = self.get_minimum(self.d)
                self.update_weights(i, dmin)
            self.alpha = self.mDecayRate * self.alpha

        return

