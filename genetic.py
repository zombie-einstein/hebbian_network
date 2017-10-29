import network as net
import numpy as np
from copy import deepcopy

# PARAMETERS
INPUT_SIZE = 2
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2
POP_SIZE = 10
LEARN_RATE = 0.01
CROSSOVER_RATE = 0.4
MUTATION_RATE = 0.01
UPDATE_STEPS = 100


def mutate(arr):
    """Randomly flip elements of structural arrays (i.e 0 <-> 1)"""
    for i in xrange(arr.shape[0]):
        rands = [y > MUTATION_RATE for y in np.random.random(arr.shape[1])]
        arr[i] = [x if y else x^1 for x, y in zip(arr[i], rands)]


class Parameters:
    """Structural parameters needed for n-net, and attached fitness parameter"""
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
        self.fitness = 0.0
    
    def mutate(self):
        mutate(self.input)
        mutate(self.hidden)
        mutate(self.output)


def cross_arrays(arr1,arr2):
    """Randomly swap rows in two matrices"""
    rands = [x > CROSSOVER_RATE for x in np.random.random(arr1.shape[0])]
    for i in xrange(arr1.shape[0]):
        if rands[i]: arr1[i], arr2[i] = arr2[i].copy(), arr1[i].copy()


def cross_params(params1, params2):
    """Perform crossover for all arrays in parameter set"""
    cross_arrays(params1.input, params2.input)
    cross_arrays(params1.hidden, params1.hidden)
    cross_arrays(params1.output, params2.output)
    
    
population = []  # Initial population

for i in xrange(POP_SIZE):
    p = Parameters(
        LEARN_RATE,
        np.random.randint(2,size=(HIDDEN_SIZE,INPUT_SIZE)),
        np.random.randint(2, size=(HIDDEN_SIZE, INPUT_SIZE)),
        np.random.randint(2, size=(OUTPUT_SIZE, HIDDEN_SIZE)))
    population.append(p)


def get_params(x):
    """Return parameters from population that bounds x"""
    i = 0
    while population[i].fit < x: i += 1
    return population[i]


for n in xrange(UPDATE_STEPS):
    for p in population:
        p.fit = np.random.random()
        # TODO: train
        # TODO: test and assign fitness
    
    # Sort by fitness, normalize values, then assign cumulative value
    population.sort(key=lambda p: p.fit)
    s = sum([x.fit for x in population])
    for i in population: i.fit /= s
    cumsum = 0
    for i in population:
        carry = i.fit
        i.fit += cumsum
        cumsum += carry
        
    newPop = []
    
    for i in xrange(POP_SIZE/2):
        a = deepcopy(get_params(np.random.random()))
        a.fit = 0.0
        b = deepcopy(get_params(np.random.random()))
        b.fit = 0.0
        cross_params(a, b)
        a.mutate()
        a.mutate()
        newPop.append(a)
        newPop.append(b)
        
    population = newPop