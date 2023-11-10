import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spt
import os
import torch
torch.set_default_dtype(torch.float64)
from utils import mpc_linear_cvx,mpc_nonlinear,dynamics_data, unroll_policy,sample_within_bounds,sample_polytope_surf_2d,sample_polytope_hit_and_run
from compute_cis_polytope import estimate_cis,generate_vertices_from_box_constraints

class dyn_example1():
    # Class for the system in example 1: linear dynamics, convex safe set
    def __init__(self,nn_model,
                 mpc_pred_hor = 10,
                 training_points = (300,5),
                 validation_points = (300,5),
                 testing_points = (3000,5),
                 long_traj_points = (200,200),
                 unsupervised_points = 5000,
                 epochs = 50000,
                 lr_primal = 10**(-4),
                 lr_dual = 10**(-4),
                 lr_slack = 10**(-4),
                 rho = 100.0):
        
        self.name = 'example1'
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        self.nn_model = nn_model
        self.mpc_policy = mpc_linear_cvx
        self.mpc_pred_hor = mpc_pred_hor
        self.dt = 0.05
        self.n_states = 2
        self.n_inputs = 1

        self.n_data = {'training':training_points,
                       'validation':validation_points,
                       'testing':testing_points,
                       'boundary':(unsupervised_points,1),
                       'long_traj':long_traj_points}

        self.epochs = epochs
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.lr_slack = lr_slack
        self.rho = rho
        self.n_unsup = unsupervised_points

        self.A = np.array([[1,self.dt],[0,1]])
        self.B = np.array([[self.dt**2 / 2],[self.dt]])

        self.R = np.eye(self.n_inputs)
        self.Q = np.eye(self.n_states)*10

        self.state_bounds = np.array([[-6.0,6.0],[-1.0,1.0]])
        self.control_bounds = np.array([[-2.5,2.5]])

        self.cis_type = 'polytope'
        cis_path = self.name + '/cis_' + self.cis_type + '.npy'
        if os.path.isfile(cis_path):
            self.cis_vertices = np.load(cis_path)
        else:
            initial_cis = generate_vertices_from_box_constraints(self.state_bounds)
            self.cis_vertices = estimate_cis(self,initial_cis,False)
            polytope_path = self.name + '/cis_polytope'
            np.save(polytope_path,self.cis_vertices)
        
        cis_ch = spt.ConvexHull(self.cis_vertices)
        self.cis_ineq = {'A':cis_ch.equations[:,:-1],
                         'b':cis_ch.equations[:,-1]}
        self.cis_vertices = cis_ch.points[cis_ch.vertices]

    def cis_val_func(self,x):
        # Computes the value of the value function that encodes the control invariant set as its zero sublevel set
        if torch.is_tensor(x):
            return torch.matmul(x,self.cis_ineq['A'].T) + self.cis_ineq['b']
        else:
            if len(x.shape) > 1:
                return self.cis_ineq['A']@x + np.expand_dims(self.cis_ineq['b'],1)
            else:
                return self.cis_ineq['A']@x + self.cis_ineq['b']

    def dynamics_step(self,x,u):
        # Performs a step of system dynamics given state and control action
        if torch.is_tensor(x):
            return torch.matmul(x,torch.tensor(self.A.T)) + torch.matmul(u,torch.tensor(self.B.T))
        else:
            return self.A@x + self.B@u
    
    def generate_data(self,ds_type):
        # Generates IMPC data for a given data set type (training, validation, testing, boundary)

        mpc = self.mpc_policy(self)

        data = dynamics_data()
        n_points, n_unroll = self.n_data[ds_type]
        save_path = self.name + '/implicit_mpc_data_' + ds_type
        if ds_type == 'boundary':
            x0s,_ = sample_polytope_surf_2d(self.cis_vertices,n_points)
            x0s = torch.tensor(x0s)
            u = np.zeros((n_points,self.n_inputs))
            u[:] = np.nan
            data.add_data(x0s,u)
        else:
            x0s = sample_polytope_hit_and_run(self.cis_ineq,n_points)
            for x0 in x0s:
                x,u,_,_ = unroll_policy(x0,self,mpc,N_total=n_unroll)
                data.add_data(x[:-1],u)

        torch.save(data,save_path)
    
    def cost(self,x,u):
        # Returns the value of ocp cost function

        if torch.is_tensor(x):
            return torch.sum(torch.matmul(x[1:],torch.tensor(self.Q.T))*x[1:] + torch.matmul(u,torch.tensor(self.R.T))*u)
        else:
            if x.shape[0] != self.n_states:
                x = x.reshape((self.n_states,-1))
                u = u.reshape((self.n_inputs,-1))

            return np.sum((self.Q@x[:,1:])*x[:,1:] + (self.R@u)*u)

    def phase_portrait(self,axis,x,label='',traj=False,color='k',linestyle = 'solid'):
        # Plot the phase portrait for given points/trajectory
        if traj:
            axis.plot(x[:, 0],x[:, 1],color=color,linestyle=linestyle,label=label,linewidth=2.5)
        else:
            axis.plot(x[:, 0],x[:, 1],'rx',zorder=2.001)

    def plot_cis(self, axis):
        # Plots the control invariant set

        poly_x = np.append(self.cis_vertices[:,0],self.cis_vertices[0,0])
        poly_y = np.append(self.cis_vertices[:,1],self.cis_vertices[0,1])
        axis.fill_between([self.state_bounds[0,0]-0.5, self.state_bounds[0,1]+0.5], self.state_bounds[1,0]-0.1, self.state_bounds[1,1]+0.1, color='k', alpha=0.5,label = 'Unsafe region')
        axis.fill(poly_x, poly_y, color='w')
        axis.set_xlim([-0.5,6.5])
        axis.set_ylim([-1.1,0.5])

class dyn_example2(dyn_example1):
    # Class for the system in example 2: linear dynamics, nonconvex safe set
    def __init__(self,nn_model,
                 mpc_pred_hor = 10,
                 training_points = (300,5),
                 validation_points = (300,5),
                 testing_points = (3000,5),
                 long_traj_points = (200,200),
                 unsupervised_points = 5000,
                 epochs = 50000,
                 lr_primal = 10**(-3),
                 lr_dual = 10**(-3),
                 lr_slack = 10**(-3),
                 rho = 1000.0):
        self.name = 'example2'
        
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        self.nn_model = nn_model
        self.mpc_policy = mpc_nonlinear
        self.mpc_pred_hor = mpc_pred_hor
        self.dt = 0.05
        self.n_states = 2
        self.n_inputs = 1

        self.n_data = {'training':training_points,
                       'validation':validation_points,
                       'testing':testing_points,
                       'boundary':(unsupervised_points,1),
                       'long_traj':long_traj_points}

        self.epochs = epochs
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.lr_slack = lr_slack
        self.rho = rho
        self.n_unsup = unsupervised_points

        self.obstacle = {'center':np.array([[1/2],[1/2]]),'radius':1/4}

        self.state_bounds = np.array([[0.0,1.0],[0.0,1.0]])
        self.control_bounds = np.array([[-2.5,2.5]])

        self.cis_type = 'V'

        self.Q = 1.0
        self.R = 0

    def dynamics_step(self,x,u):
        # Performs a step of system dynamics given state and control action
        if torch.is_tensor(x):
            xp1_1 = x[:,0] + self.dt*x[:,1] + self.dt**2/2*u[:,0]
            xp1_2 = x[:,1] + self.dt*u[:,0]
            xp1 = torch.stack((xp1_1,xp1_2)).T
        else:
            xp1_1 = x[0] + self.dt*x[1] + self.dt**2/2*u[0]
            xp1_2 = x[1] + self.dt*u[0]
            xp1 = np.stack((xp1_1,xp1_2))
        return xp1
        
    def cis_val_func(self,x):
        # Computes the value of the value function that encodes the control invariant set as its zero sublevel set
        if torch.is_tensor(x):
            return self.obstacle['radius'] - torch.norm(x - torch.tensor(self.obstacle['center']).T,dim=1)
        else:
            if len(x.shape) > 1:
                return self.obstacle['radius'] - np.linalg.norm(x - self.obstacle['center'],axis=0)
            else:
                return self.obstacle['radius'] - np.linalg.norm(x - self.obstacle['center'])
            
    def cost(self,x,u):
        # Returns the value of ocp cost function
        if torch.is_tensor(x):
            return torch.sum(self.Q*torch.norm(x[1:],dim=1)**2 + self.R*torch.norm(u,dim=1)**2)
        else:
            if x.shape[0] != self.n_states:
                x = x.reshape((self.n_states,-1))
                u = u.reshape((self.n_inputs,-1))

            return np.sum(self.Q*np.linalg.norm(x[:,1:],axis=0)**2 + self.R*np.linalg.norm(u,axis=0)**2)
            
    def generate_data(self,ds_type):
        # Generates IMPC data for a given data set type (training, validation, testing, boundary)
        mpc = self.mpc_policy(self)
        data = dynamics_data()
        n_points, n_unroll = self.n_data[ds_type]
        save_path = self.name + '/implicit_mpc_data_' + ds_type
        
        if ds_type == 'boundary':
            theta = np.random.uniform(0, 2*np.pi, n_points)
            d = np.abs(np.random.normal(0,0.03,n_points))
            x_circ = self.obstacle['center'][0] + (self.obstacle['radius'] + d) * np.cos(theta)
            y_circ = self.obstacle['center'][1] + (self.obstacle['radius'] + d) * np.sin(theta)
            x0s = np.vstack((x_circ,y_circ)).T

            u = np.zeros((n_points,self.n_inputs))
            u[:] = np.nan
            data.add_data(x0s,u)
        else:
            x0s = sample_within_bounds(self.state_bounds,n_points)
            x0s = x0s[self.cis_val_func(x0s.T)<=0]
            for x0 in x0s:
                x,u,_,_ = unroll_policy(x0,self,mpc,N_total=n_unroll)
                if np.all(self.cis_val_func(x.T)<=0):
                    data.add_data(x[:-1],u)
        
        torch.save(data,save_path)

    def plot_cis(self, axis):
        # Plots the control invariant set
        theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
        x_circ = self.obstacle['center'][0] + self.obstacle['radius'] * np.cos(theta)
        y_circ = self.obstacle['center'][1] + self.obstacle['radius'] * np.sin(theta)
        axis.fill(x_circ, y_circ, 'k', alpha=0.5,label = 'Unsafe region')

class dyn_example3(dyn_example1):
    # Class for the system in example 1: linear dynamics, convex safe set
    def __init__(self,nn_model,
                 mpc_pred_hor = 10,
                 training_points = (300,5),
                 validation_points = (300,5),
                 testing_points = (3000,5),
                 long_traj_points = (200,200),
                 unsupervised_points = 2000,
                 epochs = 50000,
                 lr_primal = 10**(-3),
                 lr_dual = 10**(-5),
                 lr_slack = 10**(-2),
                 rho = 10.0):
        self.name = 'example3'
        
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        self.nn_model = nn_model
        self.mpc_policy = mpc_nonlinear
        self.mpc_pred_hor = mpc_pred_hor
        self.dt = 1
        self.n_states = 2
        self.n_inputs = 1

        self.n_data = {'training':training_points,
                       'validation':validation_points,
                       'testing':testing_points,
                       'boundary':(unsupervised_points,1),
                       'long_traj':long_traj_points}

        self.epochs = epochs
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.lr_slack = lr_slack
        self.rho = rho
        self.n_unsup = unsupervised_points

        self.obstacle = {'center':np.array([[0.75],[0.2]]),'radius':1/4}

        self.cis_vertices = np.array([[-2.60513715,  1.28422571],
                                        [ 0.34007449, -1.89550454],
                                        [ 2.60513715, -1.28422571],
                                        [-0.34007449,  1.89550454]])

        cis_ch = spt.ConvexHull(self.cis_vertices)
        self.cis_vertices = cis_ch.points[cis_ch.vertices]
        self.cis_ineq = {'A':cis_ch.equations[:,:-1],
                         'b':cis_ch.equations[:,-1]}
        
        self.Q = 0.05
        self.R = 0.1

        self.state_bounds = np.array([[-4.0,4.0],[-4.0,4.0]])
        self.control_bounds = np.array([[-2.0,2.0]])

        self.cis_type = 'V'

    def dynamics_step(self,x,u):
        # Performs a step of system dynamics given state and control action
        if torch.is_tensor(x):
            xp1_1 = x[:,0] + 0.1*x[:,1] + 0.09*u[:,0] + 0.01*x[:,0]*u[:,0]
            xp1_2 = x[:,1] + 0.1*x[:,0] + 0.09*u[:,0] - 0.04*x[:,1]*u[:,0]
            xp1 = torch.stack((xp1_1,xp1_2)).T
        else:
            xp1_1 = x[0] + 0.1*x[1] + 0.09*u[0] + 0.01*x[0]*u[0]
            xp1_2 = x[1] + 0.1*x[0] + 0.09*u[0] - 0.04*x[1]*u[0]
            xp1 = np.stack((xp1_1,xp1_2))
        return xp1
        
    def cis_val_func(self,x):
        # Computes the value of the value function that encodes the control invariant set as its zero sublevel set
        if torch.is_tensor(x):
            cis = torch.matmul(x,torch.tensor(self.cis_ineq['A'].T)) + torch.tensor(self.cis_ineq['b'].T)
            obst = self.obstacle['radius'] - torch.norm(x - torch.tensor(self.obstacle['center']).T,dim=1)
            return torch.max(torch.hstack((cis,obst.unsqueeze(1))),dim=1).values
        else:
            cis = self.cis_ineq['A']@x + self.cis_ineq['b'].reshape((-1,1))
            obst = self.obstacle['radius'] - np.linalg.norm(x - self.obstacle['center'],axis=0)
            safe = np.vstack((cis,obst.reshape((1,-1))))
            return np.max(safe,axis=0)
            
    def cost(self,x,u):
        # Returns the value of ocp cost function
        if torch.is_tensor(x):
            return torch.sum(self.Q*torch.norm(x[1:],dim=1)**2 + self.R*torch.norm(u,dim=1)**2)
        else:
            if x.shape[0] != self.n_states:
                x = x.reshape((self.n_states,-1))
                u = u.reshape((self.n_inputs,-1))

            return np.sum(self.Q*np.linalg.norm(x[:,1:],axis=0)**2 + self.R*np.linalg.norm(u,axis=0)**2)
            
    def generate_data(self,ds_type):
        # Generates IMPC data for a given data set type (training, validation, testing, boundary)
        mpc = self.mpc_policy(self)
        data = dynamics_data()
        n_points, n_unroll = self.n_data[ds_type]
        save_path = self.name + '/implicit_mpc_data_' + ds_type
        
        if ds_type == 'boundary':
            import math
            n_points = math.ceil(n_points/2)
            x0s,_ = sample_polytope_surf_2d(self.cis_vertices,n_points)

            theta = np.random.uniform(0, 2*np.pi, n_points)
            d = np.abs(np.random.normal(0,0.015*self.obstacle['radius'],n_points))
            x_circ = self.obstacle['center'][0] + (self.obstacle['radius'] + d) * np.cos(theta)
            y_circ = self.obstacle['center'][1] + (self.obstacle['radius'] + d) * np.sin(theta)
            circ_bound = np.vstack((x_circ,y_circ)).T
            x0s = np.vstack((circ_bound,x0s))
            x0s = x0s[self.cis_val_func(x0s.T) <= 0]

            u = np.zeros((n_points,self.n_inputs))
            u[:] = np.nan
            data.add_data(x0s,u)
        else:
            x0s = sample_polytope_hit_and_run(self.cis_ineq,n_points)
            x0s = x0s[self.cis_val_func(x0s.T) <= 0]
            for x0 in x0s:
                x,u,_,_ = unroll_policy(x0,self,mpc,N_total=n_unroll)
                if np.all(self.cis_val_func(x.T)<=0):
                    data.add_data(x[:-1],u)
                    
        torch.save(data,save_path)

    def plot_cis(self, axis):
        # Plots the control invariant set
        poly_x = np.append(self.cis_vertices[:,0],self.cis_vertices[0,0])
        poly_y = np.append(self.cis_vertices[:,1],self.cis_vertices[0,1])
        axis.fill_between([np.min(self.cis_vertices[:,0])-0.1,np.max(self.cis_vertices[:,0])+0.1], np.min(self.cis_vertices[:,1])-0.1, np.max(self.cis_vertices[:,1])+0.1, color='k', alpha=0.5,label = 'Unsafe region')
        axis.fill(poly_x, poly_y, color='w')

        theta = np.linspace(0, 2*np.pi, 100, endpoint=True)
        x_circ = self.obstacle['center'][0] + self.obstacle['radius'] * np.cos(theta)
        y_circ = self.obstacle['center'][1] + self.obstacle['radius'] * np.sin(theta)
        
        axis.fill(x_circ, y_circ, 'k', alpha=0.5)
        axis.set_xlim([-0.5,1.5])
        axis.set_ylim([-0.5,1.0])