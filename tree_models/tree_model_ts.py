"""
Author: Goutham G. 
"""
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyDOE import lhs
from tqdm import tqdm

# Import required user-defined libraries
from para import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

plot_directory = "./output_ts/plots/"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
    
class Net1(torch.nn.Module):
    def __init__(self, nn_width, nn_num_layers,positive=False,sigmoid=False):
        super(Net1, self).__init__()
        # Initialize the first layer
        layers = [torch.nn.Linear(params['n_trees'], params['nn_width']), torch.nn.Tanh()]
        # Add the rest of the layers
            
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        # Final output layer        
        layers.append(torch.nn.Linear(nn_width, 1))
        
        # Define functions
        if positive:
            layers.append(torch.nn.Softplus())
        elif sigmoid:
            layers.append(torch.nn.Sigmoid())
        self.positive   = positive  # Define positive function
        self.sigmoid    = sigmoid   # Define sigmoid function
        self.net        = torch.nn.Sequential(*layers) # Construct neural net
        
    def forward(self, Z):
        ''' Define the forward pass of the neural network
                - sigmoid is used when parameterizing c/a ratio
                - positive is used for ∂V/∂a > 0 and q > 0
        '''
        output = self.net(Z)
        return output

class Training_Sampler():
    def __init__(self, params):
        self.params = params
        self.sv_count = params['n_trees'] - 1
        self.batch_size = params['batch_size']

        SV = np.random.uniform(low=[0] * self.sv_count, 
                         high=[1] * self.sv_count, 
                         size=(self.batch_size, self.sv_count))
        self.boundary_uniform_points = torch.Tensor(SV)

    # def sample(self,N):
        
    #     # Construct a by latin hypercube sampling. If there are
    #     # N trees, then construct N-1 shares 
    #     #Z       = 0.05+ (0.95-0.05)*np.random.rand(N)
    #     Z      = np.random.rand(N)
        
    #     return Z.reshape((N, self.sv_count))
    
    def sample(self):
        '''
            - No active sampling at the moment.
        '''
        SV = np.random.uniform(low=[0] * self.sv_count, 
                         high=[1] * self.sv_count, 
                         size=(self.batch_size, self.sv_count))
        SV = torch.Tensor(SV)
        # 10 equally spaced timesteps in [0,1]
        T = torch.linspace(0, 1, 10)

        SV_repeated = SV.repeat(1, T.shape[0]).view(-1, SV.shape[1])
        T_repeated = T.repeat(1, SV.shape[0]).view(-1, 1)
        return torch.cat((SV_repeated, T_repeated), dim=1)
    
    def sample_boundary_cond(self, time_val: float):
        time_dim = torch.ones((self.boundary_uniform_points.shape[0], 1)) * time_val
        return torch.cat([self.boundary_uniform_points, time_dim], dim=-1)
    
    def sample_fixed_grid_boundary_cond(self, time_val: float):
        '''
        This is only used for boundary conditions
        '''
        sv_ls = [0] * (self.sv_count)
        for i in range(self.sv_count):
            sv_ls[i] = torch.linspace(0, 1, steps=self.batch_size)
        sv = torch.cartesian_prod(*sv_ls)
        if len(sv.shape) == 1:
            sv = sv.unsqueeze(-1)
        time_dim = torch.ones((sv.shape[0], 1)) * time_val
        return torch.cat([sv, time_dim], dim=-1)

class Training_pde(Environments):

    def __init__(self, params):
        super().__init__(params)
    
    def get_derivs_1order(self,y_pred,x):
        """ Returns the first order derivatives,
            Automatic differentiation used
        """
        dy_dx = torch.autograd.grad(y_pred, x,
                            create_graph=True,
                            grad_outputs=torch.ones_like(y_pred))[0]
        return dy_dx ## Return 'automatic' gradient.
    
    def batch_jacobian(func, x, create_graph=False):
        ''' Compute the jacobian of a function for a batch of inputs
        '''
        def _func_sum(x):
            return func(x).sum(dim=0)
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)
    
    def loss_fun_Net1(self, kappa1_nn, kappa2_nn, SV: torch.Tensor,return_=True):
        ''' Loss function for the neural network
        Remark: two trees is a one state variable problem. 
            - kappa1: kappa(SV)
            - kappa2: kappa(SV)
            - SV: (share, t)
        '''
        SV   = SV.clone()
        SV.requires_grad_(True)
        
        z = SV[:, :1]
        kappa1 = kappa1_nn(SV)
        kappa2 = kappa2_nn(SV)
        q1 = z / kappa1
        q2 = (1-z)/(kappa2)

        # Automatic differentiation for q1 and q2
        q1_z        = self.get_derivs_1order(q1,SV)[:, :1]
        q2_z        = self.get_derivs_1order(q2,SV)[:, :1]
        q1_zz       = self.get_derivs_1order(q1_z,SV)[:, :1]
        q2_zz       = self.get_derivs_1order(q2_z,SV)[:, :1]

        kappa1_z    = self.get_derivs_1order(kappa1,SV)[:, :1]
        kappa2_z    = self.get_derivs_1order(kappa2,SV)[:, :1]
        kappa1_zz   = self.get_derivs_1order(kappa1_z,SV)[:, :1]
        kappa2_zz   = self.get_derivs_1order(kappa2_z,SV)[:, :1]

        kappa1_t = self.get_derivs_1order(kappa1,SV)[:, 1:2]
        kappa2_t = self.get_derivs_1order(kappa2,SV)[:, 1:2]

        
        # Compute dynamics of z
        mu_z_geo  = self.params['mu_y1'] - (z*self.params['mu_y1'] + (1-z)*self.params['mu_y2']) +\
                    (z*self.params['sig_y1'] + (1-z)*self.params['sig_y2']) * ((z*self.params['sig_y1'] + (1-z)*self.params['sig_y2'])  - self.params['sig_y1'])
        
        sig_z_geo =  self.params['sig_y1'] - (z*self.params['sig_y1'] + (1-z)*self.params['sig_y2']) 
        
        mu_z_ari    = mu_z_geo * z 
        sig_z_ari   = sig_z_geo * z

        # Compute dynamics of 1-z (required for computation of kappa_2 dynamics)
        mu_1minusz_ari  = -mu_z_ari
        sig_1minusz_ari = -sig_z_ari
        mu_1minusz_geo  = mu_1minusz_ari/(1-z)
        sig_1minusz_geo = sig_1minusz_ari/(1-z)

        mu_q1       = (q1_z * mu_z_ari + 0.5 * q1_zz * sig_z_ari**2)/q1
        mu_q2       = (q2_z * mu_z_ari + 0.5 * q2_zz * sig_z_ari**2)/q2
        sig_q1      = (q1_z * sig_z_ari)/q1
        sig_q2      = (q2_z * sig_z_ari)/q2

        r           = self.params['rho'] + self.params['gamma'] * (z*self.params['sig_y1']  + (1-z)*self.params['sig_y2']) -\
                        self.params['gamma'] * (1+self.params['gamma'])/2*(z**2 *self.params['sig_y1']**2  + (1-z)**2 *self.params['sig_y2']**2)
        zeta_1      = self.params['gamma'] * z * self.params['sig_y1']
        zeta_2      = self.params['gamma'] * (1-z) * self.params['sig_y2']
        
        mu_kappa1   = mu_z_geo - mu_q1 + sig_q1 *(sig_q1 - sig_z_geo) 
        mu_kappa2   = mu_1minusz_geo - mu_q2 + sig_q2 *(sig_q2 - sig_1minusz_geo)
        sig_kappa1  = sig_z_geo - sig_q1
        sig_kappa2  = sig_1minusz_geo - sig_q2

        hjb_kappa1          =  kappa1_t + kappa1_z * mu_z_ari + 0.5 * kappa1_zz * sig_z_ari**2 - mu_kappa1 * kappa1
        hjb_kappa2          =  kappa2_t + kappa2_z * mu_z_ari + 0.5 * kappa2_zz * sig_z_ari**2 - mu_kappa2 * kappa2
        consistency_kappa1  =  kappa1_t + kappa1_z * sig_z_ari - sig_kappa1 * kappa1
        consistency_kappa2  =  kappa2_t + kappa2_z * sig_z_ari - sig_kappa2 * kappa2
        l2_regularization   = torch.mean(kappa1**2 + kappa2**2)

        # Store variables
        self.kappa1, self.kappa2, self.q1, self.q2, self.zeta1, self.zeta2, self.r = kappa1, kappa2, q1, q2, zeta_1, zeta_2, r
        self.mu_z_geo, self.sig_z_geo, self.mu_z_ari, self.sig_z_ari = mu_z_geo, sig_z_geo, mu_z_ari, sig_z_ari
        self.mu_q1, self.mu_q2, self.sig_q1, self.sig_q2 = mu_q1, mu_q2, sig_q1, sig_q2
        self.mu_kappa1, self.mu_kappa2, self.sig_kappa1, self.sig_kappa2 = mu_kappa1, mu_kappa2, sig_kappa1, sig_kappa2
        

        lhjb_kappa1                 = torch.sum(torch.square(hjb_kappa1))
        lhjb_kappa2                 = torch.sum(torch.square(hjb_kappa2))
        lconsistency_kappa1         = torch.sum(torch.square(consistency_kappa1))
        lconsistency_kappa2         = torch.sum(torch.square(consistency_kappa2))
        lreg                        = torch.sum(torch.square(l2_regularization))

        total_loss = lhjb_kappa1 + lhjb_kappa2 + lconsistency_kappa1 + lconsistency_kappa2

        return total_loss, hjb_kappa1, hjb_kappa2, consistency_kappa1, consistency_kappa2, l2_regularization
    
if __name__ == '__main__':
    TP = Training_pde(params)
    TS = Training_Sampler(params)

    # Initialize neural networks
    kappa_nns: Dict[str, nn.Module] = {}
    for i in range(1, params['n_trees']+1):
        kappa_nns[f"kappa{i}"] = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    for i in range(2, params['n_trees']+1):
        kappa_nns[f"kappa{i}"].load_state_dict(kappa_nns["kappa1"].state_dict())
    # They should start with the same weights

    # Initialize neural nets to store last best model
    best_model_kappas: Dict[str, nn.Module] = {}
    for i in range(1, params['n_trees']+1):
        best_model_kappas[f"kappa{i}"] = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    for i in range(2, params['n_trees']+1):
        best_model_kappas[f"kappa{i}"].load_state_dict(best_model_kappas["kappa1"].state_dict())

    para_nn = []
    for kappa_nn in kappa_nns.values():
        para_nn += list(kappa_nn.parameters())

    # Set hyperparameters
    optimizer       = optim.Adam(para_nn, lr=params['lr'])
    outer_loop_size = 10
    outer_loop_convergence_thres = 1e-4
    epochs          = 1000
    outer_loop_min_loss = torch.inf

    SV_T0 = TS.sample_boundary_cond(0.0).to(device)
    SV_T0.requires_grad_(True)
    SV_T1 = TS.sample_boundary_cond(1.0).to(device)
    SV_T1.requires_grad_(True)

    prev_vals: Dict[str, torch.Tensor] = {}
    for i in range(1, params['n_trees']+1):
        prev_vals[f"kappa{i}"] = torch.ones_like(SV_T0[:, 0:1], device=device)

    for outer_loop in range(outer_loop_size):
        # For now, make sure the sampling is stable for each outer iteration
        min_loss = torch.inf
        SV = TS.sample().to(device)
        SV.requires_grad_(True)
        pbar = tqdm(range(int(epochs / (np.sqrt(outer_loop + 1)))))
        for epoch in pbar:
            total_loss, lhjb_kappa1_, lhjb_kappa2_, lconsistency_kappa1_, lconsistency_kappa2_,lreg_ = TP.loss_fun_Net1(kappa_nns["kappa1"],kappa_nns["kappa2"], SV)

            # match the time boundary condition
            loss_time_boundary = 0.
            for name, model in kappa_nns.items():
                loss_time_boundary += torch.sum(torch.square(model(SV_T1) - prev_vals[name]))

            total_loss += loss_time_boundary
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(para_nn, 1)
    
            optimizer.step()
            loss_val = total_loss.item()
            if (loss_val < min_loss):
                min_loss = loss_val
                for k, best_model in best_model_kappas.items():
                    best_model.load_state_dict(kappa_nns[k].state_dict())
                pbar.set_description(f"Total loss: {total_loss:.4f}")
                # print( (epoch, lhjb_kappa1.item(), lhjb_kappa2.item(), lconsistency_kappa1.item(), lconsistency_kappa2.item()))

        # check convergence and update the time boundary condition
        for name, model in kappa_nns.items():
            model.load_state_dict(best_model_kappas[name].state_dict())

        total_loss, *_ = TP.loss_fun_Net1(kappa_nns["kappa1"],kappa_nns["kappa2"], SV)

            # match the time boundary condition
        loss_time_boundary = 0.
        for name, model in kappa_nns.items():
            loss_time_boundary += torch.sum(torch.square(model(SV_T1) - prev_vals[name]))

        total_loss += loss_time_boundary
        if total_loss < outer_loop_min_loss:
            print(f"Updating min loss from {outer_loop_min_loss:.4f} to {total_loss:.4f}")
            outer_loop_min_loss = total_loss

        new_vals: Dict[str, torch.Tensor] = {}
        for name, model in kappa_nns.items():
            new_vals[name] = model(SV_T0).detach()

        max_abs_change = 0.
        max_rel_change = 0.
        all_changes = {}
        for k in prev_vals:
            mean_new_val = torch.mean(new_vals[k]).item()
            abs_change = torch.mean(torch.abs(new_vals[k] - prev_vals[k])).item()
            rel_change = torch.mean(torch.abs((new_vals[k] - prev_vals[k]) / prev_vals[k])).item()
            print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}")
            all_changes[f"{k}_mean_val"] = mean_new_val
            all_changes[f"{k}_abs"] = abs_change
            all_changes[f"{k}_rel"] = rel_change
            max_abs_change = max(max_abs_change, abs_change)
            max_rel_change = max(max_rel_change, rel_change)
        
        for k in prev_vals:
            prev_vals[k] = new_vals[k]

        total_rel_change = min(max_abs_change, max_rel_change)
        all_changes["total"] = total_rel_change

        if all_changes["total"] < outer_loop_convergence_thres:
            break
        
    # Load last best model as the final neural network model
    for name, model in kappa_nns.items():
        model.load_state_dict(best_model_kappas[name].state_dict())
    
    # Save data
    SV_T0 = TS.sample_fixed_grid_boundary_cond(0.0).to(device)
    SV_T0.requires_grad_(True)
    
    TP  = Training_pde(params)
    TP.loss_fun_Net1(kappa_nns["kappa1"],kappa_nns["kappa2"],SV_T0,False)
    kappa1, kappa2, q1, q2, zeta1, zeta2, r = TP.kappa1, TP.kappa2, TP.q1, TP.q2, TP.zeta1, TP.zeta2, TP.r
    mu_z_geo, sig_z_geo, mu_z_ari, sig_z_ari = TP.mu_z_geo, TP.sig_z_geo, TP.mu_z_ari, TP.sig_z_ari
    mu_q1, mu_q2, sig_q1, sig_q2 = TP.mu_q1, TP.mu_q2, TP.sig_q1, TP.sig_q2
    mu_kappa1, mu_kappa2, sig_kappa1, sig_kappa2 = TP.mu_kappa1, TP.mu_kappa2, TP.sig_kappa1, TP.sig_kappa2
    
    Z = SV_T0.detach().cpu().numpy()[:, :1].reshape(-1)
    fig, ax = plt.subplots(3,2,figsize=(16,9), num=1)
    ax[0,0].plot(Z,kappa1.detach().cpu().numpy(),label=r'$\kappa_1$')
    ax[0,0].plot(Z,kappa2.detach().cpu().numpy(),label=r'$\kappa_2$')
    ax[0,0].legend()
    
    
    ax[0,1].plot(Z,q1.detach().cpu().numpy(),label=r'$q_1$')
    ax[0,1].plot(Z,q2.detach().cpu().numpy(),label=r'$q_2$')
    ax[0,1].legend()
    

    ax[1,0].plot(Z,zeta1.detach().cpu().numpy(),label=r'$\zeta_1$')
    ax[1,0].plot(Z,zeta2.detach().cpu().numpy(),label=r'$\zeta_2$')
    ax[1,0].legend()
    

    ax[1,1].plot(Z,r.detach().cpu().numpy(),label=r'$r$')
    ax[1,1].set_title('r')

    ax[2,0].plot(Z,mu_z_geo.detach().cpu().numpy(),label=r'$\mu^z$')
    ax[2,0].plot(Z,sig_z_geo.detach().cpu().numpy(),label=r'$\sigma^z$')
    ax[2,0].legend()
    ax[2,0].set_xlabel('z')

    ax[2,1].plot(Z,mu_z_ari.detach().cpu().numpy(),label=r'$\mu_z$')
    ax[2,1].plot(Z,sig_z_ari.detach().cpu().numpy(),label=r'$\sigma_z$')
    ax[2,1].legend()
    ax[2,1].set_xlabel('z')

    
    name = 'equilibrium.jpg'
    plt.savefig(os.path.join(plot_directory, name),bbox_inches='tight',dpi=300)
    plt.show()
    plt.close('all')

    print('Training complete')



