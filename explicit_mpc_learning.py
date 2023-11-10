import torch
torch.set_default_dtype(torch.float64)
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from nn_training_utils import MSE_Objective, Val_Func_Constraint, Polytope_Constraint, Augmented_Lagrangian
import os
import torch.optim.lr_scheduler as scheduler
import copy

def train(x,u,aug_lag,optimizer,epoch):
    # Performs optimizer step on the aumented Lagrangian
    optimizer.zero_grad()
    L = aug_lag.objective(x, u)
    L.backward()
    optimizer.step()

def train_constrained(x, u, aug_lag, optimizer, optimizer_slack, optimizer_dual, epoch):
    # Performs primal-dual iteration step
    optimizer.zero_grad()
    optimizer_slack.zero_grad()
    optimizer_dual.zero_grad()

    loss = aug_lag(x, u)
    loss.backward()

    optimizer.step()
    optimizer_slack.step()
    aug_lag.constraint.clip_slack()
    optimizer_dual.step()

def record_history(hist_dict,epoch,x_train,u_train,x_validate,u_validate,aug_lag):
    # Appends the data from the most recent iteration to a dictionary
    with torch.no_grad():
        hist_dict['epoch'].append(epoch)
        hist_dict['training_loss'].append(aug_lag.objective(x_train, u_train).item())
        hist_dict['validation_loss'].append(aug_lag.objective(x_validate, u_validate).item())
        if torch.is_tensor(aug_lag.constraint.ineq_constraint()):
            hist_dict['constraint'].append(aug_lag.constraint.ineq_constraint().numpy().copy())
        else:
            hist_dict['constraint'].append(aug_lag.constraint.ineq_constraint().values.numpy().copy())
        hist_dict['lag_mult'].append(aug_lag.lag_mult.cpu().numpy().copy())
        hist_dict['slacks'].append(aug_lag.constraint.slack_vars.cpu().numpy().copy())

        print(f"Epoch {hist_dict['epoch'][-1]}:")
        print(f"training loss = {hist_dict['training_loss'][-1]}")
        print(f"validation loss = {hist_dict['validation_loss'][-1]}")
        
def plot_history(history_dict,plot_step=100,save_plot=False,save_path=''):
    # Plots training and validation losses
    fig,axs = plt.subplots(4,sharex=True)

    axs[0].set_title('log(loss)')
    axs[0].plot(np.log(history_dict['training_loss'][::plot_step]),label='training_loss')
    axs[0].plot(np.log(history_dict['validation_loss'][::plot_step]),label='validation_loss')
    # axs[0].legend()

    constraint = np.vstack(history_dict['constraint'])
    idx = np.argmax(np.clip(constraint,a_min=0,a_max=None), axis=1)
    rows = np.arange(len(idx))
    constraint_max_vio = constraint[rows, idx]
    
    axs[1].set_title('Constraint')
    axs[1].plot(constraint_max_vio[::plot_step])

    lag_mult = np.vstack(history_dict['lag_mult'])
    corr_lag_mult = lag_mult[rows, idx]
    axs[2].set_title('Lagrange Multiplier')
    axs[2].plot(corr_lag_mult[::plot_step])

    # slacks = np.vstack(history_dict['slacks'])
    # corr_slacks = slacks[rows, idx]
    # axs[3].set_title('Lagrange Multiplier')
    # axs[3].plot(slacks[:,5])
    num_constr_viol = np.sum(constraint > 0,axis=1)
    axs[3].set_title('Number of constraints violated')
    axs[3].plot(num_constr_viol)
    if save_plot:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def nominal_training(system,pretraining_for_pd = False,resume=False):
    # Trains nominal EMPC for a given system
    model_name = 'unconstrained_model'

    mpc_data_path = system.name + '/implicit_mpc_data_training'
    data_train = torch.load(mpc_data_path)
    x_train = data_train.X
    u_train = data_train.U

    mpc_data_path = system.name + '/implicit_mpc_data_validation'
    data_validate = torch.load(mpc_data_path)
    x_validate = data_validate.X
    u_validate = data_validate.U

    epochs = system.epochs
    lr_primal = system.lr_primal
    rho = system.rho

    model = system.nn_model(system.control_bounds)
    
    obj = MSE_Objective(model)

    data_unsup = torch.load(system.name + '/implicit_mpc_data_boundary')
    x_unsup = torch.vstack((x_train,data_unsup.X))
    slack_init = torch.zeros(x_unsup.shape[0])
    if system.cis_type == 'V':
        constr = Val_Func_Constraint(model,system,x_unsup,slack_init)
    elif system.cis_type == 'polytope':
        constr = Polytope_Constraint(model,system,x_unsup,slack_init)

    lag_mult = torch.zeros(len(slack_init))
    aug_lag = Augmented_Lagrangian(model, obj, constr, lag_mult, rho)

    model_folder = system.name + '/' + model_name
    if resume:
        model_data_path = system.name + '/' + model_name + '/state_dict'
        model.load_state_dict(torch.load(model_data_path))

    optimizer_objective = optim.Adam(model.parameters(), lr=lr_primal)

    history_dict = {'epoch':[],
                    'training_loss': [],
                    'validation_loss':[],
                    'constraint': [],
                    'lag_mult':[],
                    'slacks':[]}

    for epoch in range(1, epochs + 1):
        train(x_train,u_train,aug_lag,optimizer_objective,epoch)
        if epoch % 100 == 0 or epoch == epochs:
            record_history(history_dict,epoch,
                       x_train,u_train,
                       x_validate,u_validate,
                       aug_lag)

    model_folder = system.name + '/' + model_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plot_step = 1
    plot_save_path = model_folder + '/training_history'
    plot_history(history_dict,plot_step,True,plot_save_path)

    if pretraining_for_pd:
        model_path = model_folder + '/state_dict_pre'
    else:
        model_path = model_folder +  '/state_dict'
    torch.save(model.state_dict(),model_path)

def primal_dual_training(system,resume = False, pretrain_epochs = 0):
    # Trains SST EMPC for a given system using primal-dual method

    # model_name = 'primal_dual_model' + 'rho' + str(system.rho) + 'lr_dual' + str(system.lr_dual) + 'lr_primal' + str(system.lr_primal)
    model_name = 'primal_dual_model'

    mpc_data_path = system.name + '/implicit_mpc_data_training'
    data_train = torch.load(mpc_data_path)
    x_train = data_train.X
    u_train = data_train.U

    mpc_data_path = system.name + '/implicit_mpc_data_validation'
    data_validate = torch.load(mpc_data_path)
    x_validate = data_validate.X
    u_validate = data_validate.U

    epochs = copy.deepcopy(system.epochs)
    lr_primal = system.lr_primal
    lr_dual = system.lr_dual
    lr_slack = system.lr_slack
    rho = system.rho

    model = system.nn_model(system.control_bounds)

    # if pretrain_epochs > 0:
    #     system.epochs = pretrain_epochs
    #     nominal_training(system,pretraining_for_pd=True)
    #     system.epochs = epochs
    #     model_data_path = system.name + '/' + 'unconstrained_model' + '/state_dict_pre'
    #     model.load_state_dict(torch.load(model_data_path))
    
    obj = MSE_Objective(model)
        
    data_unsup = torch.load(system.name + '/implicit_mpc_data_boundary')
    x_unsup = torch.vstack((x_train,data_unsup.X))
    slack_init = torch.zeros(x_unsup.shape[0])
    if system.cis_type == 'V':
        constr = Val_Func_Constraint(model,system,x_unsup,slack_init)
    elif system.cis_type == 'polytope':
        constr = Polytope_Constraint(model,system,x_unsup,slack_init)

    lag_mult = torch.zeros(len(slack_init))
    aug_lag = Augmented_Lagrangian(model, obj, constr, lag_mult, rho)
    
    model_folder = system.name + '/' + model_name
    if resume:
        model_data_path = system.name + '/' + model_name + '/state_dict'
        model.load_state_dict(torch.load(model_data_path))
        aug_lag.lag_mult = torch.load(model_folder+'/lag_mult')
        aug_lag.constraint.slack_vars = torch.load(model_folder+'/slack_vars')

    optimizer_objective = optim.Adam(model.parameters(), lr=lr_primal)
    scheduler_objective = scheduler.ExponentialLR(optimizer_objective, gamma=0.975)
    optimizer_slack = optim.Adam([aug_lag.constraint.slack_vars], lr=lr_slack)
    scheduler_slack = scheduler.ExponentialLR(optimizer_slack, gamma=0.975)
    optimizer_dual = optim.Adam([aug_lag.lag_mult], lr=lr_dual, maximize = True)
    scheduler_dual = scheduler.ExponentialLR(optimizer_dual, gamma=0.975)

    history_dict = {'epoch':[],
                    'training_loss': [],
                    'validation_loss':[],
                    'constraint': [],
                    'lag_mult':[],
                    'slacks':[]}
    
    model_constr_sat = None
    for epoch in range(1, epochs + 1):
        train_constrained(x_train, u_train, aug_lag, optimizer_objective, optimizer_slack, optimizer_dual, epoch)
        if epoch % 100 == 0 or epoch == epochs:
            record_history(history_dict,epoch,
                       x_train,u_train,
                       x_validate,u_validate,
                       aug_lag)
        aug_lag.constraint()
        if aug_lag.constraint.satisfied:
            model_constr_sat = copy.deepcopy(model)
        # if epoch % 1000 == 0:
        #     scheduler_objective.step()
        #     scheduler_slack.step()
        #     scheduler_dual.step()

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plot_save_path = model_folder + '/training_history'
    plot_history(history_dict,1,True,plot_save_path)

    # fig_hist,axs_hist = plt.subplots(2,sharex=True)
    # lag_mult_diff = np.linalg.norm(np.diff(lag_mult,axis=0),ord=np.inf,axis=1)
    # slack_diff = np.linalg.norm(np.diff(constraint,axis=0),ord=np.inf,axis=1)

    # axs_hist[0].plot(lag_mult_diff)
    # axs_hist[0].set_title('Max absolute change in lag mult')
    # axs_hist[1].plot(slack_diff)
    # axs_hist[1].set_title('Max absolute change in slack vars')
    # plt.show()

    if not model_constr_sat:
        print('No model satisfied the constraints')
        # model_path = model_folder + '/state_dict_constraints_not_sat'
        # torch.save(model.state_dict(),model_path)
        model_path = model_folder + '/state_dict'
        torch.save(model.state_dict(),model_path)
        
    else:
        model_path = model_folder + '/state_dict'
        torch.save(model_constr_sat.state_dict(),model_path)
    torch.save(aug_lag.lag_mult,model_folder+'/lag_mult')
    torch.save(aug_lag.constraint.slack_vars,model_folder+'/slack_vars')