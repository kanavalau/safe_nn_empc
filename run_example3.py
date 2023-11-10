import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import numpy as np
from dyn_systems import dyn_example3
from explicit_mpc_learning import nominal_training, primal_dual_training
from explicit_mpc_testing import test_empc

class constr_policy_loc(nn.Module):
    def __init__(self,input_bounds):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 1)
        self.input_bounds = input_bounds

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.clip(self.output_layer(x),min=torch.tensor(self.input_bounds[:,0]),max=torch.tensor(self.input_bounds[:,1]))

system = dyn_example3(nn_model=constr_policy_loc)

# Data generation
# Running system.generate_data with the different inputs is going to replace 
# the data files present in example3/
system.n_data = {'training':(300,5),
                'validation':(300,5),
                'testing':(3*10**3,5),
                'boundary':(2000,1),
                'long_traj':(200,200)}
# system.generate_data('training')
# system.generate_data('validation')
# system.generate_data('boundary')
# system.generate_data('testing')

# Unconstrained training
# Running the training is going to save/replace
# the model parameters stored in example3/unconstrained_model
# system.epochs = 10000
# system.lr_primal = 10**(-4)
# nominal_training(system,resume=False)

# Constrained training
# Running the training is going to save/replace
# the model parameters stored in example3/primal_dual_model
# system.epochs = 10000
# system.lr_primal = 10**(-3)
# system.lr_dual = 10**(-5)
# system.lr_slack = 10**(-2)
# system.rho = 10.0
# primal_dual_training(system,resume=False)

# NN policy testing
# Running test_empc prints out relevant numerical results
# and saves the plots in example3/
x0 = np.array([0.9,0])
test_empc(system,x0=x0,n_steps = 200)