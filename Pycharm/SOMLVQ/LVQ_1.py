import numpy as np
import sys
from LVQ import LVQ
# animal name = [hair;feathers;eggs;milk;airborne;aquatic;predator;toothed;backbone;breathes;venomous;fins;legs;tail;domestic;catsize] target = type
aardvark =  [1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1] #type 1

crow =      [0,1,1,0,1,0,1,0,1,1,0,0,2,1,0,0] #type 2

pitviper =  [0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0] #type 3

piranha =   [0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0] #type 4

newt =      [0,0,1,0,0,1,1,1,1,1,0,0,4,1,0,0] #type 5

termite =   [0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0] #type 6

crayfish =  [0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0] #type 7

calf =      [1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1] #type 1

chicken =   [0,1,1,0,1,0,0,0,1,1,0,0,2,1,1,0] #type 2

seasnake =  [0,0,0,0,0,1,1,1,1,0,1,0,0,1,0,0] #type 3

sole =      [0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0] #type 4

toad =      [0,0,1,0,0,1,0,1,1,1,0,0,4,0,0,0] #type 5

wasp =      [1,0,1,0,1,0,0,0,0,1,1,0,6,0,0,0] #type 6

starfish =  [0,0,1,0,0,1,1,0,0,0,0,0,5,0,0,0] #type 7

wolf =      [1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1] #type 1

swan =      [0,1,1,0,1,1,0,0,1,1,0,0,2,1,0,1] #type 2

tortoise =  [0,0,1,0,0,0,0,0,1,1,0,0,4,1,0,1] #type 3

tuna =      [0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,1] #type 4

frog =      [0,0,1,0,0,1,1,1,1,1,0,0,4,0,0,0] #type 5

flea    =   [0,0,1,0,0,0,0,0,0,1,0,0,6,0,0,0] #type 6

lobster =   [0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0] #type 7

TARGET = np.array([[0,1,2,3,4,5,6,0,1,2,3 ,4 ,5 ,6, 0, 1, 2, 3, 4, 5, 6]])
NAME = ["aardvark","crow","piranha","newt","termite","crayfish","calf","chicken","seasnake","sole","toad","wasp","starfish","wolf","swan",
        "tortoise","tuna","frog","flea","lobster"]

VEC_LEN = 16
TRAINING_PATTERNS = 21

DECAY_RATE = 0.96  # About 100 iterations.
MIN_ALPHA = 0.01

if __name__ == '__main__':
    pattern = []
    pattern.append(aardvark)
    pattern.append(crow)
    pattern.append(pitviper)
    pattern.append(piranha)
    pattern.append(newt)
    pattern.append(termite)
    pattern.append(crayfish)
    pattern.append(calf)
    pattern.append(chicken)
    pattern.append(seasnake)
    pattern.append(sole)
    pattern.append(toad)
    pattern.append(wasp)
    pattern.append(starfish)
    pattern.append(wolf)
    pattern.append(swan)
    pattern.append(tortoise)
    pattern.append(tuna)
    pattern.append(frog)
    pattern.append(flea)
    pattern.append(lobster)

    learningVQ1 = LVQ( VEC_LEN, TRAINING_PATTERNS, DECAY_RATE, MIN_ALPHA, pattern,
                               TARGET)

    learningVQ1.initialize_arrays()

    learningVQ1.training()


wolf = [1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1] #type 1
sys.stdout.write(
            "Pattern is in " + str(learningVQ1.get_cluster(wolf)+1) + "\n")
