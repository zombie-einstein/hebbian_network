import numpy as np
from scipy.special import expit as sig # Sigmoid function

class nnet:
    def __init__(self,input_size,hidden_size,output_size,learn_rate,input_structure,hidden_structure,output_structure):
        self.vi = np.random.rand(input_size, 1)  # Input vector
        self.vn = np.random.rand(hidden_size, 1)  # Hidden layer neurons
        self.Wi = np.random.randint(2, size=(hidden_size, input_size))  # Input layer to hidden layer weights
        self.Wh = np.random.rand(hidden_size,hidden_size) # Hidden layer internal weights
        self.lr = learn_rate
        assert type(input_structure) == np.ndarray and input_structure.shape == (hidden_size,input_size)
        self.Si = input_structure  # Input to hidden structure (Time independent)
        assert type(hidden_structure) == np.ndarray and hidden_structure.shape == (hidden_size,hidden_size)
        self.Sh = hidden_structure # Hidden layer structure (Time independent)
        assert type(output_structure) == np.ndarray and output_structure.shape == (output_size,hidden_size)
        self.So = output_structure

    def update(self):
        "One update step of network, update outputs and weightings"
        o = sig(np.multiply(np.concatenate((self.Wi,self.Wh),axis=1),np.concatenate((self.Si,self.Sh),axis=1)).dot(np.concatenate((self.vi,self.vn),axis=0)))-0.5
        dt = self.vn-o  # Change in neuron output
        self.Wh = sig(self.Wh+self.lr*dt.T*dt)-0.5  # Update weights of hidden layer
        self.vn = o

    def read_outputs(self):
        return sig(self.So.dot(self.vn))