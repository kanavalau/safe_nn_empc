import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class Augmented_Lagrangian(nn.Module):
    # Construct and augmented lagrangian for given objective, constraints, lagrange multipliers, and penalty constants

    def __init__(self,model,objective,constraint,lag_mult,rho=1):
        super(Augmented_Lagrangian, self).__init__()
        self.model = model
        self.objective = objective
        self.constraint = constraint
        self.rho = rho
        self.lag_mult = lag_mult
        self.lag_mult.requires_grad = True

    def forward(self, x, u_target):
        obj = self.objective(x,u_target)
        return obj + torch.dot(self.lag_mult,self.constraint()) + self.rho/2*torch.norm(self.constraint(),p = 2)**2

class MSE_Objective(nn.Module):
    def __init__(self,model):
        super(MSE_Objective, self).__init__()
        self.model = model

    def forward(self, x, u_target):
        return torch.mean(torch.norm(self.model(x) - u_target,p=2,dim=1) ** 2)

class Polytope_Constraint(nn.Module):
    # Closed loop polytope constraint with polytope described as intersection of halfspaces
    def __init__(self,model,system,x_vals,slack_init):
        super().__init__()
        self.model = model
        self.x_vals = x_vals
        self.n_constr = x_vals.shape[0]
        self.slack_vars = slack_init
        self.slack_vars.requires_grad = True
        self.system = system
        self.A = torch.tensor(self.system.cis_ineq['A'])
        self.b = torch.tensor(self.system.cis_ineq['b'])
        self.satisfied = False

    def forward(self):
        constr_val = self.ineq_constraint()
        self.satisfied = (constr_val <= 0).all()
        return constr_val + self.slack_vars
    
    def ineq_constraint(self):
        u = self.model(self.x_vals)
        xp1 = self.system.dynamics_step(self.x_vals,u)
        return torch.max(torch.matmul(xp1,self.A.T) + self.b,axis=1)[0]/self.n_constr

    def clip_slack(self):
        with torch.no_grad():
            self.slack_vars.copy_(torch.relu(self.slack_vars))

class Val_Func_Constraint(nn.Module):
    # Closed loop safety constraint based on an arbitrary value function for the system
    def __init__(self,model,system,x_vals,slack_init):
        super().__init__()
        self.model = model
        self.x_vals = x_vals
        self.n_constr = x_vals.shape[0]
        self.slack_vars = slack_init
        self.slack_vars.requires_grad = True
        self.system = system
        self.satisfied = False

    def forward(self):
        constr_val = self.ineq_constraint()
        self.satisfied = (constr_val <= 0).all()
        return constr_val + self.slack_vars
    
    def ineq_constraint(self):
        u = self.model(self.x_vals)
        xp1 = self.system.dynamics_step(self.x_vals,u)
        return self.system.cis_val_func(xp1)

    def clip_slack(self):
        with torch.no_grad():
            self.slack_vars.copy_(torch.relu(self.slack_vars))