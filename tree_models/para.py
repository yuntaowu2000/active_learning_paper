"""
Author: Goutham Gopalakrishna
This block contains parameters, basic functions used in the model
"""
import torch
import numpy as np
from pyDOE import lhs

    
## Economic Parameters
# Save everything in a dictionary, for clarity
params = {
    "gamma"             : 5.0,      # Household risk aversion
    "rho"               : 0.05,     # Fund discount rate
    "nn_width"          : 80,       # Neural network width
    "nn_num_layers"     : 4,        # Neural network layers
    "n_trees"           : 3,       # Number of trees
    "mu_y1"             : 0.02,     # Mean of the first state variable
    "sig_y1"            : 0.02,     # Std of the first state variable
    "mu_y2"             : 0.05,     # Mean of the second state variable
    "sig_y2"            : 0.05,     # Std of the second state variable
    "mu_ys": [0.02, 0.05, 0.08],     # Mean of the state variables
    "sig_ys": [0.02, 0.05, 0.08],    # Std of the state variables
    "lr"                : 0.0005,    # Learning rate
    "batch_size"        : 500,      # Batch size
}



class Params():
    def __init__(self):
        self.params = params

class Environments():
    def __init__(self,para_dict):
        self.params = params
    
    
    
    
    