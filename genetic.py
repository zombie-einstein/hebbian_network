import network as net
import numpy as np

INPUT_SIZE = 2
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2
POP_SIZE = 100
LEARN_RATE = 0.01
CROSSOVER_RATE = 0.4
MUTATION_RATE = 0.01


class netParams:
    "Initial parameters we need to make a network"
    def __init__(
            self,
            learn_rate,
            input_structure,
            hidden_structure,
            output_structure):
        self.lr = learn_rate
        self.input = input_structure
        self.hidden = hidden_structure
        self.output = output_structure


def crossover(arrA,arrB):
    "Randomly swap rows in matrices"
    rands = [x > CROSSOVER_RATE for x in np.random.random(arrA.shape[0])]
    print rands
    for i in xrange(arrA.shape[0]):
        if rands[i]: arrA[i],arrB[i] = arrB[i].copy(),arrA[i].copy()


def mutate(arr):
    "Randomly flip elements of array"
    for i in xrange(arr.shape[0]):
        rands = [y > MUTATION_RATE for y in np.random.random(arr.shape[1])]
        arr[i] = [x if y else x^1 for x, y in zip(arr[i], rands)]
        
        
population = [] # Initial population

for i in xrange(POP_SIZE):
    p = netParams(
        LEARN_RATE,
        np.random.randint(2,size=(HIDDEN_SIZE,INPUT_SIZE)),
        np.random.randint(2, size=(HIDDEN_SIZE, INPUT_SIZE)),
        np.random.randint(2, size=(OUTPUT_SIZE, HIDDEN_SIZE)))
    population.append({'params': p, 'fit': 0.0}) # Parameters and fitness

for p in population:
    p['fit'] = np.random.random()
    #train
    #test and assign fitness


# Sort by fitness and norm
population.sort(key=lambda p:p['fit'])
s = sum([x['fit'] for x in population])
#map(lambda x: x['fit']/=s,population)



