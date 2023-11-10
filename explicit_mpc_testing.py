import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from utils import nn_policy,unroll_policy
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import itertools

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "font.family": "serif",
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 10,
    "axes.titlesize": 13,           # Title size when one figure
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 13,         # Overall figure title
    "pgf.rcfonts": False,
    "text.latex.preamble": 
        r'\usepackage{xcolor}',
    "pgf.preamble": 
        r'\usepackage{xcolor}'
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
MATHFONTSIZE = 14

def eval_nn_model(system,model_name,zero_thresh = 10**(-6)):
    # Evaluate errors and safety of a specified model for a given system

    model_data_path = system.name + '/' + model_name + '/state_dict'
    model = system.nn_model(system.control_bounds)
    model.load_state_dict(torch.load(model_data_path))
    model.eval()

    nn_control = nn_policy(model)
    
    mpc_data_path = system.name + '/implicit_mpc_data_testing'
    data = torch.load(mpc_data_path)
    X = data.X
    U_mpc = data.U
    U_nn = nn_control.evaluate(X)

    Xp1 = system.dynamics_step(X,U_nn)
    constraint_val = compute_constraint_value(system,Xp1)

    constraint_violation = constraint_val <= 0
    sort_ind = np.argsort(constraint_val)
    constraint_val_sorted = constraint_val[sort_ind]
    points_constr_violation = X[sort_ind][constraint_val_sorted > 0]
    
    frac_safe = torch.sum(constraint_violation)/constraint_violation.shape[0]

    diff = U_mpc-U_nn
    diff[diff < zero_thresh] = 0
    
    data_boundary = torch.load(system.name + '/implicit_mpc_data_boundary')
    X_boundary = data_boundary.X
    U_nn_boundary = nn_control.evaluate(X_boundary)
    Xp1_boundary = system.dynamics_step(X_boundary,U_nn_boundary)
    constraint_val = compute_constraint_value(system,Xp1_boundary)

    sort_ind = np.argsort(constraint_val)
    constraint_val_sorted = constraint_val[sort_ind]

    points_constr_violation = np.vstack((points_constr_violation,X_boundary[sort_ind][constraint_val_sorted > 0]))
    constraint_violation = constraint_val <= 0
    frac_safe_boundary = torch.sum(constraint_violation)/constraint_violation.shape[0]

    out_dict = {'X':X,
                'U':U_nn,
                'diff':diff,
                'frac_safe':frac_safe,
                'points_constr_violation':points_constr_violation,
                'X_boundary':X_boundary,
                'U_boundary':U_nn_boundary,
                'frac_safe_boundary':frac_safe_boundary}
    x = X_boundary[sort_ind][constraint_val_sorted > 0]

    print(model_name)
    for i in range(system.n_inputs):
        print(f"Input dimension {i}")
        print(f"Mean error {np.mean(np.abs(out_dict['diff'].numpy()[:,i])):{0}.{3}}")
        print(f"Median error {np.median(np.abs(out_dict['diff'].numpy()[:,i])):{0}.{8}}")
        print(f"Max error {np.max(np.abs(out_dict['diff'].numpy()[:,i])):{0}.{3}}")
    
    print(f"Fraction safe {out_dict['frac_safe']:{0}.{3}}")
    print(f"Fraction safe on the boundary {out_dict['frac_safe_boundary']:{0}.{3}}")
    print()

    return out_dict

def long_traj_run(system,X_init,policy):
    # Unroll a given policy for an extended number of steps and compute the cost

    traj_length = system.n_data['long_traj'][1]

    cost = []
    for X0 in X_init:
        X_,U_,_,_ = unroll_policy(X0,system,policy,N_total=traj_length)
        if np.all(system.cis_val_func(X_.T)<=0):
            cost.append(system.cost(X_,U_))
        else:
            cost.append(-1)

    return np.array(cost)

def test_empc(system,x0 = None, n_steps = 20):
    # Test the trained EMPC models

    unconstrained_res = eval_nn_model(system,'unconstrained_model')
    constrained_res = eval_nn_model(system,'primal_dual_model')

    mpc_data_path = system.name + '/implicit_mpc_data_testing'
    data = torch.load(mpc_data_path)
    X = data.X
    U_mpc = data.U

    traj_number = system.n_data['long_traj'][0]
    X_start = X[::system.n_data['testing'][1]]
    X_long_traj = X_start[:traj_number]

    unc_model = system.nn_model(system.control_bounds)
    unc_model.load_state_dict(torch.load(system.name + '/' + 'unconstrained_model' + '/state_dict'))
    unc_model.eval()
    unc_nn_control = nn_policy(unc_model)
    unconstrained_costs = long_traj_run(system,X_long_traj,unc_nn_control)

    con_model = system.nn_model(system.control_bounds)
    con_model.load_state_dict(torch.load(system.name + '/' + 'primal_dual_model' + '/state_dict'))
    con_model.eval()
    con_nn_control = nn_policy(con_model)
    constrained_costs = long_traj_run(system,X_long_traj,con_nn_control)

    mpc_control = system.mpc_policy(system)
    mpc_costs = long_traj_run(system,X_long_traj,mpc_control)
    
    valid_inds = np.where((unconstrained_costs != -1)*(mpc_costs != -1)*(constrained_costs != -1))[0]

    unconstrained_costs = unconstrained_costs[valid_inds]
    constrained_costs = constrained_costs[valid_inds]
    mpc_costs = mpc_costs[valid_inds]

    unconstrained_subopt = np.mean((unconstrained_costs - mpc_costs)/mpc_costs)*100
    constrained_subopt = np.mean((constrained_costs - mpc_costs)/mpc_costs)*100
    print(f"Percent suboptimal unconstrained {unconstrained_subopt:{0}.{3}}")
    print(f"Percent suboptimal constrained {constrained_subopt:{0}.{3}}")
    print()

    mpc = system.mpc_policy(system)

    if not np.any(x0):
        if np.any(unconstrained_res['points_constr_violation']):
            x0 = unconstrained_res['points_constr_violation'][-1]
        else:
            ind = np.random.randint(0,X.shape[0]-1)
            x0 = X[ind]
    x_mpc,u_mpc,t,time_mpc = unroll_policy(x0,system,mpc,N_total=n_steps)

    model_data_path = system.name + '/' + 'unconstrained_model' + '/state_dict'
    model = system.nn_model(system.control_bounds)
    model.load_state_dict(torch.load(model_data_path))
    model.eval()

    nn_control_unconstrained = nn_policy(model)

    x_unconstrained,u_unconstrained,_,time_unconstrained = unroll_policy(x0,system,nn_control_unconstrained,N_total=n_steps)

    model_data_path = system.name + '/' + 'primal_dual_model' + '/state_dict'
    model = system.nn_model(system.control_bounds)
    model.load_state_dict(torch.load(model_data_path))
    model.eval()

    nn_control_constrained = nn_policy(model)

    x_constrained,u_constrained,_,time_constrained = unroll_policy(x0,system,nn_control_constrained,N_total=n_steps)

    fig,ax = plt.subplots(system.n_states)
    for i in range(system.n_states):
        ax[i].plot(t,x_mpc[:-1,i],label='MPC')
        ax[i].plot(t,x_unconstrained[:-1,i],label='Unconstrained')
        ax[i].plot(t,x_constrained[:-1,i],label='Constrained')
        ax[i].set_xlabel('Time/s')
        ax[i].set_ylabel(f"$x_{i+1}$",fontsize=MATHFONTSIZE)
        ax[i].legend()

    plt.savefig(system.name + '/trajectory_state_tracking')

    if system.n_inputs > 1:
        fig,ax = plt.subplots(system.n_inputs)
        for i in range(system.n_inputs):
            ax[i].plot(t,u_mpc[:,i],label='IMPC')
            ax[i].plot(t,u_unconstrained[:,i],label='Unconstrained')
            ax[i].plot(t,u_constrained[:,i],label='Constrained')
            ax[i].set_xlabel('Time/s')
            ax[i].set_ylabel(f"$u_{i+1}$",fontsize=MATHFONTSIZE)
            ax[i].legend()
    else:
        fig,ax = plt.subplots()
        ax.plot(t,u_mpc,label='IMPC')
        ax.plot(t,u_unconstrained,label='Nominal EMPC')
        ax.plot(t,u_constrained,label='SST EMPC')
        ax.set_xlabel('Time/s')
        ax.set_ylabel("$u$",fontsize=MATHFONTSIZE)
        ax.legend()

    plt.savefig(system.name + '/trajectory_control_tracking')

    fig,ax = plt.subplots(figsize=(3.4, 4.0))
    
    ax.plot(x0[0],x0[1],'ks',label = 'Initial point',fillstyle='none',markeredgewidth=1)
    ax.plot(0,0,'ko', label='Origin',fillstyle='none',markeredgewidth=1)
    system.plot_cis(ax)
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]
    ax.text(x0[0]-0.25*x_range,x0[1]-0.025*y_range, f'[{x0[0]:.1f},{x0[1]:.1f}]',fontsize=11)
    plot_trajectory(ax,system,x_mpc,'k','-','IMPC')
    plot_trajectory(ax,system,x_constrained,'g','--','SST EMPC')
    plot_trajectory(ax,system,x_unconstrained,'b',':','Nominal EMPC')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    ax.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.45, 1.4))
    fig.tight_layout()
    fig.subplots_adjust(top=0.75)

    plt.savefig(system.name + '/phase_portrait.pdf')

    print(f"Median time MPC {np.median(time_mpc):{0}.{3}}")
    print(f"Max time MPC {np.max(time_mpc):{0}.{3}}")

    print(f"Median time unconstrained {np.median(time_unconstrained):{0}.{3}}")
    print(f"Max time unconstrained {np.max(time_unconstrained):{0}.{3}}")

    print(f"Median time constrained {np.median(time_constrained):{0}.{3}}")
    print(f"Max time constrained {np.median(time_constrained):{0}.{3}}")

def compute_constraint_value(sys,x):
    # Compute constraint value for a given system and state
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if sys.cis_type == 'V':
        constraint_val = sys.cis_val_func(x)
    elif sys.cis_type == 'polytope':
        constraint_val = torch.max(torch.matmul(x,torch.tensor(sys.cis_ineq['A']).T) + torch.tensor(sys.cis_ineq['b']),axis=1)[0]

    return constraint_val

def plot_trajectory(ax,sys,x,color,linestyle,label):
    # Plot trajectory for a given system
    constraint_value = compute_constraint_value(sys,x)
    constraint_viols = np.where(constraint_value > 0)[0]
    if constraint_viols.shape[0] > 0 and label != 'IMPC':
        first_violation = constraint_viols[0]
        x_viol = x[first_violation,:]
        ax.plot(x_viol[0],x_viol[1],'rx',zorder=2.001,markeredgewidth=1)
        label = label + '\n' + r'(\textcolor{red}{$\times$} first unsafe)'
    sys.phase_portrait(ax,x,color=color,linestyle=linestyle,label=label,traj=True)
