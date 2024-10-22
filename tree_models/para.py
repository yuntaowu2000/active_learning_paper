"""
Author: Goutham Gopalakrishna
This block contains parameters, basic functions used in the model
"""
import torch
import numpy as np
from pyDOE import lhs

    
## Economic Parameters
# Save everything in a dictionary, for clarity
# params = {
#     "gamma"             : 5.0,      # Household risk aversion
#     "rho"               : 0.05,     # Fund discount rate
#     "nn_width"          : 80,       # Neural network width
#     "nn_num_layers"     : 4,        # Neural network layers
#     "n_trees"           : 5,       # Number of trees
#     "mu_y1"             : 0.02,     # Mean of the first state variable
#     "sig_y1"            : 0.02,     # Std of the first state variable
#     "mu_y2"             : 0.05,     # Mean of the second state variable
#     "sig_y2"            : 0.05,     # Std of the second state variable
#     "mu_ys": [0.02, 0.05, 0.08, 0.11, 0.14],     # Mean of the state variables , 0.08, 0.11, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3
#     "sig_ys": [0.02, 0.05, 0.08, 0.11, 0.14],    # Std of the state variables , 0.08, 0.11, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3
#     "lr"                : 0.0005,    # Learning rate
#     "batch_size"        : 100,      # Batch size
#     "sample_method": "log_normal", # uniform, log_normal
#     "output_dir": "./models/tree5_log_normal"
# }

params_base = {
    "gamma"             : 5.0,      # Household risk aversion
    "rho"               : 0.05,     # Fund discount rate
    "nn_width"          : 80,       # Neural network width
    "nn_num_layers"     : 4,        # Neural network layers
    "lr"                : 0.0005,    # Learning rate
}
sample_methods = ["uniform", "log_normal"]

num_tree_mu_sig = [ # for now, we just assume mu=sig
    (2, [0.02, 0.05]),
    (3, [0.02, 0.05, 0.08]),
    (5, [0.02, 0.05, 0.08, 0.11, 0.14]), 
    (10, [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.2, 0.23, 0.26, 0.3]),
    (20, [0.01 * i for i in range(1, 21)]),
    (50, [0.01 * i for i in range(1, 51)]),
    (100, [0.01 * i for i in range(1, 101)]),
]



class Params():
    def __init__(self):
        self.params = params_base

class Environments():
    def __init__(self, params):
        self.params = params
    
    
    
    
    