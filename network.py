import numpy as np
from scipy.special import expit as sig


class NeuralNet:
    def __init__(self, learn_rate, input_structure, hidden_structure, output_structure):
        self.learn_rate = learn_rate
        assert type(input_structure) == np.ndarray
        hidden_size = input_structure.shape[0]
        input_size = input_structure.shape[1]
        # Time independent structural weights
        self.S_i = input_structure
        assert type(hidden_structure) == np.ndarray and hidden_structure.shape == (hidden_size, hidden_size)
        self.S_h = hidden_structure
        assert type(output_structure) == np.ndarray and output_structure.shape[1] == hidden_size
        self.S_o = output_structure
        # Time dependent weights
        self.W_i = np.random.randint(2, size=(hidden_size, input_size))
        self.W_h = np.random.rand(hidden_size, hidden_size)
        # Neuron vectors
        self.v_i = np.random.rand(input_size, 1)
        self.v_h = np.random.rand(hidden_size, 1)
        
    def update(self):
        """One update step of network, update outputs and weightings"""
        # Updated Neuron activity based on current state
        hidden_update = np.tanh(np.multiply(np.concatenate((self.W_i, self.W_h), axis=1), np.concatenate((self.S_i, self.S_h), axis=1)).dot(np.concatenate((self.v_i, self.v_h), axis=0)))
        # Change in neuron activity between time-steps
        dt = hidden_update - self.v_h
        # Update weights of hidden layer based on change
        self.W_h = self.W_h + np.tanh(self.learn_rate * np.tensordot(dt, dt))
        # Update current state
        self.v_h = hidden_update

    def read_outputs(self):
        """Read outputs as read by hidden layer"""
        return sig(self.S_o.dot(self.v_h))
    
    def evaluate(self, training_set):
        """Given a labelled training set of the form [inputs...,label] run the network and evaluate the label"""
        score = 0.0
        for i in training_set:
            self.v_i = np.array(i[:-1]).reshape(len(i) - 1, 1)
            for j in xrange(100):
                print self.read_outputs()
                self.update()
            if np.argmax(self.read_outputs()) == i[-1]:
                score += 1.0
        return score/len(training_set)
