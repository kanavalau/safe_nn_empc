import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import numpy as np
from dyn_systems import dyn_example1
from explicit_mpc_learning import nominal_training, primal_dual_training
from explicit_mpc_testing import test_empc

class constr_policy_loc(nn.Module):
    def __init__(self,input_bounds):
        super().__init__()
        self.layer1 = nn.Linear(2, 6)
        self.layer2 = nn.Linear(6, 6)
        self.output_layer = nn.Linear(6, 1)
        self.input_bounds = input_bounds

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.clip(self.output_layer(x),min=torch.tensor(self.input_bounds[:,0]),max=torch.tensor(self.input_bounds[:,1]))

system = dyn_example1(nn_model=constr_policy_loc)

# Data generation
# Running system.generate_data is going to save/replace 
# the data files present in example1/
system.n_data = {'training':(300,5),
                'validation':(300,5),
                'testing':(3*10**3,5),
                'boundary':(5000,1),
                'long_traj':(200,200)}
# system.generate_data('training')
# system.generate_data('validation')
# system.generate_data('boundary')
# system.generate_data('testing')

# Unconstrained model training
# Running the training is going to save/replace
# the model parameters stored in example1/unconstrained_model
# system.epochs = 200000
# system.lr_primal = 10**(-4)
# nominal_training(system,resume=False)

# Constrained model training
# Running the training is going to save/replace
# the model parameters stored in example1/primal_dual_model
# system.epochs = 200000
# system.lr_primal = 10**(-4)
# system.lr_dual = 10**(-5)
# system.lr_slack = 10**(-4)
# system.rho = 100
# primal_dual_training(system,resume=False)

# NN policy testing
# Running test_empc prints out relevant numerical results
# and saves the plots in example1/
x0 = system.cis_vertices[2]
test_empc(system,x0=x0,n_steps = 400)