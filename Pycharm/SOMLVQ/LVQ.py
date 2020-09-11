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

    def compute_input(self, vectorArray):
        self.d = [0.0] * self.mNumberOfClusters

        for i in range(self.mNumberOfClusters):
            for j in range(self.mVectorLength):
                self.d[i] += (self.w[i][j] - vectorArray[j])**2

        return

    def compute_input_2D(self, vectorArray, vectorNumber):
        self.d = [0.0] * self.mNumberOfClusters

        for i in range(self.mNumberOfClusters):
            for j in range(self.mVectorLength):
                self.d[i] += (self.w[i][j] - vectorArray[vectorNumber][j])**2

        return

    def get_cluster(self, inputPattern):
        # Compute input for all nodes.
        self.compute_input(inputPattern)

        return self.get_minimum(self.d)

    def get_minimum(self, nodeArray):
        minimum = 0;
        foundNewMinimum = False
        done = False

        while not done:
            foundNewMinimum = False
            for i in range(self.mNumberOfClusters):
                if i != minimum:
                    if nodeArray[i] < nodeArray[minimum]:
                        minimum = i
                        foundNewMinimum = True

            if foundNewMinimum == False:
                done = True

        return minimum

    def update_weights(self, vectorNumber, dMin):
        for i in range(self.mVectorLength):
            # Update the winner.
            if dMin == self.mTarget[0][vectorNumber]:
                self.w[dMin][i] += (self.alpha * (self.mPatterns[vectorNumber][i] - self.w[dMin][i]))
            else:
                self.w[dMin][i] -= (self.alpha * (self.mPatterns[vectorNumber][i] - self.w[dMin][i]))

        return

    def cluster(self,target):
        clus = 0;
        for i in range (target.size):
            if clus<target[0][i]:
                clus = target[0][i]
        return clus+1

    def training(self):
        dMin = 0

        while self.alpha > self.mMinimumAlpha:
            for i in range(self.mTrainingPatterns):
                # Compute input for all nodes.
                self.compute_input_2D(self.mPatterns, i)

                # See which is smaller?
                dMin = self.get_minimum(self.d)

                # Update the weights on the winning unit.
                self.update_weights(i, dMin)

            # Reduce the learning rate.
            self.alpha = self.mDecayRate * self.alpha

        return

