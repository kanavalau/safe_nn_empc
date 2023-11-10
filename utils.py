import torch
import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull
import scipy as sp
import time

class dynamics_data():
    # More convenient and consistent way of storing the state action pairs
    def add_data(self,x,u):
        if hasattr(self,'X'):
            self.X = torch.cat((self.X,torch.tensor(x)))
            self.U = torch.cat((self.U,torch.tensor(u)))
        else:
            self.X = torch.tensor(x)
            self.U = torch.tensor(u)
            
    
class mpc_linear_cvx():
    # Linear mpc using cvxpy
    def __init__(self,system):
        self.system = system
        self.N = system.mpc_pred_hor
        if self.system.cis_type == 'polytope':
            self.cis_constraint = self.polytope_constraint
        else:
            raise('Unknown cis type')
        self.init_problem()
        
    def evaluate(self,x0):
        if len(x0.shape) > 1:
            u = np.zeros((x0.shape[0],self.system.n_inputs))
            for i,x in enumerate(x0):
                u[i] = self.solve_mpc(x)
        else:
            u = self.solve_mpc(x0)
        
        return u
    
    def polytope_constraint(self,point):
        return [self.system.cis_ineq['A']@point + self.system.cis_ineq['b'] <= 0]

    def init_problem(self):
        x_mpc = cp.Variable((self.system.n_states, self.N+1))
        u_mpc = cp.Variable((self.system.n_inputs, self.N))
        x0 = cp.Parameter(self.system.n_states,'x0')

        cost = 0
        constraints = []

        constraints += [x_mpc[:, 0] == x0]

        for k in range(self.N):
            cost += cp.quad_form(x_mpc[:, k], self.system.Q)
            cost += cp.quad_form(u_mpc[:, k], self.system.R)

            constraints += [x_mpc[:, k+1] == self.system.A @ x_mpc[:, k] + self.system.B @ u_mpc[:, k]]

            constraints += [u_mpc[:, k] >= self.system.control_bounds[:,0],
                            u_mpc[:, k] <= self.system.control_bounds[:,1]]
            
            constraints += self.cis_constraint(x_mpc[:, k+1])
            
        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve_mpc(self,x0):
        self.prob.parameters()[0].value = x0
        self.prob.solve()
        
        u_opt = self.prob.variables()[1].value[:, 0]

        return u_opt
    
class mpc_nonlinear():
    # Nonlinear mpc using scipy optimize minimize and slsqp
    def __init__(self,system):
        self.system = system
        self.N = system.mpc_pred_hor
        
    def evaluate(self,x0):
        if len(x0.shape) > 1:
            u = np.zeros((x0.shape[0],self.system.n_inputs))
            for i,x in enumerate(x0):
                u[i] = self.solve_mpc(x)
        else:
            u = self.solve_mpc(x0)
        
        return u
    
    def solve_mpc(self,x0):
        # cons = sp.optimize.NonlinearConstraint(self.cis_constraint, -np.inf, 0)
        cons = {'type': 'ineq', 'fun': self.cis_constraint}
        bounds = sp.optimize.Bounds(np.repeat(self.system.control_bounds[:,0],self.N),np.repeat(self.system.control_bounds[:,1],self.N))
        self.x_current = x0
        # u0 = self.clf_policy(x0)
        u0 = np.zeros((self.system.n_inputs, self.N))
        if hasattr(self,'u_prev'):
            u0[:,:-1] = self.u_prev[:,1:]
        opt_res = sp.optimize.minimize(self.objective_function, u0.flatten
        (),bounds = bounds,constraints = cons,options={'maxiter':250,'disp':False})

        # if not opt_res.message == 'Optimization terminated successfully':
        #     u0 = np.ones_like(self.u0)*self.system.control_bounds[:,0]
        #     opt_res = sp.optimize.minimize(self.objective_function, u0.flatten(),bounds = bounds,constraints = cons,options={'maxiter':250,'disp':False})

        # if not opt_res.message == 'Optimization terminated successfully':
        #     u0 = np.ones_like(self.u0)*self.system.control_bounds[:,1]
        #     opt_res = sp.optimize.minimize(self.objective_function, u0.flatten(),bounds = bounds,constraints = cons,options={'maxiter':250,'disp':False})

        # if not opt_res.message == 'Optimization terminated successfully':
        #     u0 = np.random.uniform(self.system.control_bounds[:,0], self.system.control_bounds[:,0], self.N)
        #     opt_res = sp.optimize.minimize(self.objective_function, u0.flatten(),bounds = bounds,constraints = cons,options={'maxiter':250,'disp':False})

        self.u_prev = opt_res.x.reshape((self.system.n_inputs,-1))

        return self.u_prev[:,0]
    
    def prediction_horizon_eval(self,x_current,u):
        x = np.zeros((self.system.n_states, self.N+1))
        x[:, 0] = x_current
        for k in range(self.N):
            x[:, k+1] = self.system.dynamics_step(x[:, k],u[:,k])

        return x

    def objective_function(self,u):
        u = u.reshape((self.system.n_inputs,-1))
        x = self.prediction_horizon_eval(self.x_current,u)

        return self.system.cost(x,u)
    
    def cis_constraint(self,u):
        u = u.reshape((self.system.n_inputs,-1))
        x = self.prediction_horizon_eval(self.x_current,u)
        return -self.system.cis_val_func(x)
    
class nn_policy():
    # Neural network base policy (EMPC)
    def __init__(self,model) -> None:
        self.model = model

    def evaluate(self,x0):
        if not torch.is_tensor(x0):
            x0 = torch.tensor(x0)
            return self.model(x0).detach().numpy()
        else:
            return self.model(x0).detach()
    
def unroll_policy(x0,system,policy,N_total=100):
    # Simulate a given policy for a given system from x0 for a given number of steps
    dt = system.dt
    T = dt*N_total

    n_states = system.n_states
    n_inputs = system.n_inputs

    t = np.arange(0, T, dt)
    x = np.zeros((len(t)+1,n_states))
    x[0] = x0.squeeze()
    u = np.zeros((len(t),n_inputs))
    time_taken = np.zeros(len(t))

    for i in range(1, N_total+1):
        start_time = time.time()
        u_opt = policy.evaluate(x[i-1])
        end_time = time.time()
        time_taken[i-1] = end_time - start_time
        x[i] = system.dynamics_step(x[i-1],u_opt)
        u[i-1] = u_opt

    return x,u,t,time_taken

def sample_polytope_surf_2d(vert,num_samples):
    # Sample surface of a polytope described by its vertices
    ch = ConvexHull(vert)
    polytope_vert = ch.points[ch.vertices]
    polytope_vert = np.vstack((polytope_vert,polytope_vert[0]))
    edges = np.diff(polytope_vert,axis=0)
    lengths_edges = np.linalg.norm(edges,axis=1)
    lengths_normed = lengths_edges/np.sum(lengths_edges)
    lengths_cumulative = np.cumsum(lengths_normed)

    samples = []
    edge_number_samp = []
    normed_length = []
    for i in range(num_samples):
        t = np.random.uniform()
        normed_length.append(t)
        edge = np.where(lengths_cumulative > t)[0][0]
        length_along_the_edge = lengths_cumulative[edge] - t
        samples.append(polytope_vert[edge] + length_along_the_edge*edges[edge]/lengths_normed[edge])
        edge_number_samp.append(edge)
    edge_number_samp = np.array(edge_number_samp)
    sorted_length = np.sort(normed_length)
    adjacent_length = np.diff(sorted_length,append=1+sorted_length[0])
    max_length = np.max(adjacent_length)*np.sum(lengths_edges)
    samples = np.vstack(samples)
    return samples,max_length

def sample_within_bounds(bounds, num_samples):
    # Generate uniform samples within the given given variable bounds 
    samples = []
    for i in range(num_samples):
        t = np.random.uniform(bounds[:,0],bounds[:,1])
        samples.append(t)
    samples = np.vstack(samples)
    return samples

def sample_polytope_hit_and_run(polytope, num_samples, burn_in=1000):
    # Use hit and run algorithm to sample inside a polytope
    samples = []

    A = polytope['A']
    b = -polytope['b']

    dim = A.shape[1]
    n_planes = b.shape[0]

    aux_points = [find_auxiliar_point(A[i], b[i]) for i in range(n_planes)]

    point = np.random.rand(dim)

    for i in range(num_samples + burn_in):
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)

        ts = []

        for j in range(n_planes):
            if np.isclose(direction @ A[j], 0):
                ts.append(np.nan)
            else:
                t = ((aux_points[j] - point) @ A[j]) / (direction @ A[j])
                ts.append(t)

        ts = np.array(ts)
        max_t = np.min(ts[ts > 0])
        min_t = np.max(ts[ts < 0])
        t_samp = np.random.uniform(min_t, max_t)
        point = point + t_samp * direction

        if i >= burn_in:
            samples.append(point)

    return np.array(samples)

def find_auxiliar_point(Ai, bi):
    p = np.zeros(Ai.shape[0])
    j = np.argmax(Ai != 0)
    p[j] = bi / Ai[j]
    return p