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

plot_directory = "./output_ts_hd_single_output/plots/"
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
    
    def loss_fun_Net1(self, kappa_nn, SV: torch.Tensor):
        ''' Loss function for the neural network
            - kappann: neural network for all kappas
            - SV: (share, t)
        '''
        SV   = SV.clone()
        SV.requires_grad_(True)
        
        z = SV[:, :-1]
        B = z.shape[0] # batch size
        N = z.shape[1] + 1 # population size
        z_last    = 1 - torch.sum(z, dim=1).unsqueeze(1)

        kappa_vec: List[torch.Tensor] = [0] * N
        q_vec: List[torch.Tensor] = [0] * N
        dq_dz: List[torch.Tensor] = [0] * N # each element is (batch, N-1)
        dq_dzz: List[torch.Tensor] = [0] * N # (batch, N-1, N-1)
        dkappa_dz: List[torch.Tensor] = [0] * N # (batch, N-1)
        dkappa_dzz: List[torch.Tensor] = [0] * N # (batch, N-1, N-1)
        dkappa_dt: List[torch.Tensor] = [0] * N # (batch, t)

        # Compute kappa and q by exploiting symmetry
        kappa_vec[0] = kappa_nn(SV) # [:, 0:1]
        q_vec[0] = z[:, 0:1] / kappa_vec[0]
        for i in range(1, N-1):
            ind         = (0,i)
            indx        = (i,0)
            SV_swaped = SV.clone()
            SV_swaped[:,ind] = SV[:,indx].clone()
            kappa_vec[i]= kappa_nn(SV_swaped) # [:, 0:1]
            q_vec[i] = SV_swaped[:, 0:1] / kappa_vec[i]
        kappa_vec[-1] = kappa_nn(SV) # [:, 1:]
        q_vec[-1] = z_last / kappa_vec[-1]

        # compute derivatives
        for i in range(N):
            dkappa_dz[i] = self.get_derivs_1order(kappa_vec[i], SV)[:, :-1] # dki/dzj 
            dkappa_dt[i] = self.get_derivs_1order(kappa_vec[i], SV)[:, -1:] # last column is time
            
            dq_dz[i] = self.get_derivs_1order(q_vec[i], SV)[:, :-1]
            curr_dkappa_dzz = [0] * (N-1) 
            curr_dq_dzz = [0] * (N-1) 
            for j in range(N - 1):
                curr_dkappa_dzz[j] = self.get_derivs_1order(dkappa_dz[i][:, j:j+1], SV)[:, :-1] # d^ki/dzjdzk
                curr_dq_dzz[j] = self.get_derivs_1order(dq_dz[i][:, j:j+1], SV)[:, :-1] 
            # Hessians should be symmetric, so the order of j and k should not matter
            dkappa_dzz[i] = torch.einsum("kbj -> bjk", torch.stack(curr_dkappa_dzz))
            dq_dzz[i] = torch.einsum("kbj -> bjk", torch.stack(curr_dq_dzz))
        
        # Compute dynamics of z
        mu_z_geos: List[torch.Tensor] = [0] * (N-1)
        sig_z_geos: List[torch.Tensor] = [0] * (N-1)
        mu_z_aris: List[torch.Tensor] = [0] * (N-1)
        sig_z_aris: List[torch.Tensor] = [0] * (N-1)
        mu_ys = torch.tensor(self.params["mu_ys"], device=device).unsqueeze(0)
        sig_ys = torch.tensor(self.params["sig_ys"], device=device).unsqueeze(0)
        for i in range(N-1):
            # z has shape (batch, N-1)
            # mu_ys has shape (1, N)
            mu_z_geos[i] = (
                mu_ys[:, i:i+1] 
                - (torch.sum(mu_ys[:, :-1] * z, dim=1, keepdim=True) + mu_ys[:, -1:] * z_last)
                + (torch.sum(sig_ys[:, :-1] * z, dim=1, keepdim=True) + sig_ys[:, -1:] * z_last)
                    * (torch.sum(sig_ys[:, :-1] * z, dim=1, keepdim=True) + sig_ys[:, -1:] * z_last - sig_ys[:, i:i+1])
            )
            sig_z_geos[i] = (
                sig_ys[:, i:i+1] 
                - (torch.sum(sig_ys[:, :-1] * z, dim=1, keepdim=True)  + sig_ys[:, -1:] * z_last)
            )
            mu_z_aris[i] = mu_z_geos[i] * z[:, i:i+1]
            sig_z_aris[i] = sig_z_geos[i] * z[:, i:i+1]
        
        # Compute dynamics of 1-z (required for computation of kappa_n dynamics)
        mu_1minusz_ari  = -torch.sum(torch.stack(mu_z_aris), axis=0)
        sig_1minusz_ari = -torch.sum(torch.stack(sig_z_aris), axis=0)
        mu_1minusz_geo  = mu_1minusz_ari/z_last
        sig_1minusz_geo = sig_1minusz_ari/z_last

        mu_z_aris_tensor = torch.einsum("nbj -> bn", torch.stack(mu_z_aris)) # j=1, so we can simply stack them together
        sig_z_aris_tensor = torch.einsum("nbj -> bn", torch.stack(sig_z_aris))
        mu_z_geos_tensor = torch.einsum("nbj -> bn", torch.stack(mu_z_geos))
        sig_z_geos_tensor = torch.einsum("nbj -> bn", torch.stack(sig_z_geos))

        mu_qs: List[torch.Tensor] = [0] * N
        sig_qs: List[torch.Tensor] = [0] * N
        for i in range(N):
            # dq_dz has shape (batch, N-1)
            # dq_dzz has shape (batch, N-1, N-1)
            # mu_z_ari has shape (batch, N-1)
            # sig_z_ari has shape (batch, N-1)
            mu_qs[i] = (torch.sum(dq_dz[i] * mu_z_aris_tensor, dim=1, keepdim=True) 
                        + 0.5 * torch.einsum("bi, bij, bj -> b", sig_z_aris_tensor, dq_dzz[i], sig_z_aris_tensor).unsqueeze(-1)) / q_vec[i]
            sig_qs[i] = torch.sum(dq_dz[i] * sig_z_aris_tensor, dim=1, keepdim=True) / q_vec[i]

        r = (self.params["rho"] 
        + self.params["gamma"] * (torch.sum(mu_ys[:, :-1] * z, dim=1, keepdim=True) + mu_ys[:, -1:] * z_last)
        + 0.5 * self.params["gamma"] * (self.params["gamma"] + 1) * (torch.sum(sig_ys[:, :-1]**2 * z**2, dim=1, keepdim=True) + sig_ys[:, -1:]**2 * z_last**2)
        )

        zetas = [0] * N
        mu_kappas = [0] * N
        sig_kappas = [0] * N
        hjb_kappas = [0] * N
        consistency_kappas = [0] * N
        for i in range(N-1):
            zetas[i] = self.params["gamma"] * z * sig_ys[:, i:i+1]
            mu_kappas[i] = mu_z_geos[i] - mu_qs[i] + sig_qs[i] * (sig_qs[i] - sig_z_geos[i])
            sig_kappas[i] = sig_z_geos[i] - sig_qs[i]
        zetas[-1]  = self.params["gamma"] * z_last * sig_ys[:, -1:]
        mu_kappas[-1] = mu_1minusz_geo - mu_qs[-1] + sig_qs[-1] * (sig_qs[-1] - sig_1minusz_geo)
        sig_kappas[-1] = sig_1minusz_geo - sig_qs[-1]

        for i in range(N):
            hjb_kappas[i] = (dkappa_dt[i] 
                + torch.sum(dkappa_dz[i] * mu_z_aris_tensor, dim=1, keepdim=True) 
                + 0.5 * torch.einsum("bi, bij, bj -> b", sig_z_aris_tensor, dkappa_dzz[i], sig_z_aris_tensor).unsqueeze(-1) 
                - mu_kappas[i] * kappa_vec[i]
            )
            consistency_kappas[i] = (dkappa_dt[i] 
                + torch.sum(dkappa_dz[i] * sig_z_aris_tensor, dim=1, keepdim=True) 
                - sig_kappas[i] * kappa_vec[i]
            )

        # Store variables
        self.kappas = torch.einsum("nbj -> bn", torch.stack(kappa_vec))
        self.qs = torch.einsum("nbj -> bn", torch.stack(q_vec))
        self.zetas = torch.einsum("nbj -> bn", torch.stack(zetas))
        self.r = r
        self.mu_z_geos = mu_z_geos_tensor
        self.sig_z_geos = sig_z_geos_tensor
        self.mu_z_aris = mu_z_aris_tensor
        self.sig_z_aris = sig_z_aris_tensor
        self.mu_qs = torch.einsum("nbj -> bn", torch.stack(mu_qs))
        self.sig_qs = torch.einsum("nbj -> bn", torch.stack(sig_qs))
        self.mu_kappas = torch.einsum("nbj -> bn", torch.stack(mu_kappas))
        self.sig_kappas = torch.einsum("nbj -> bn", torch.stack(sig_kappas))
        
        # Compute loss function
        lhjbs = 0.
        for i in range(N):
            lhjbs += torch.sum(torch.square(hjb_kappas[i]))
        lconsistency = 0.
        for i in range(N):
            lconsistency += torch.sum(torch.square(consistency_kappas[i]))

        total_loss = lhjbs + lconsistency
        return total_loss, hjb_kappas, consistency_kappas
    
if __name__ == '__main__':
    TP = Training_pde(params)
    TS = Training_Sampler(params)

    # Initialize neural networks
    kappa_nn = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    # create a dictionary for easier tracking
    nn_dict = {"kappa": kappa_nn}

    # Initialize neural nets to store last best model
    best_model_kappa = Net1(params['nn_width'], params['nn_num_layers'],positive=True,sigmoid=False).to(device)
    best_model_kappa.load_state_dict(kappa_nn.state_dict())

    para_nn = list(kappa_nn.parameters())

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
    for k in nn_dict:
        prev_vals[k] = torch.ones_like(SV_T0[:, 0:1], device=device)

    for outer_loop in range(outer_loop_size):
        # For now, make sure the sampling is stable for each outer iteration
        min_loss = torch.inf
        SV = TS.sample().to(device)
        SV.requires_grad_(True)
        pbar = tqdm(range(int(epochs / (np.sqrt(outer_loop + 1)))))
        for epoch in pbar:
            total_loss, hjb_kappas_, consistency_kappas_ = TP.loss_fun_Net1(kappa_nn, SV)

            # match the time boundary condition
            loss_time_boundary = 0.
            for name, model in nn_dict.items():
                loss_time_boundary += torch.sum(torch.square(model(SV_T1) - prev_vals[name]))

            total_loss += loss_time_boundary
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(para_nn, 1)
    
            optimizer.step()
            loss_val = total_loss.item()
            if (loss_val < min_loss):
                min_loss = loss_val
                best_model_kappa.load_state_dict(kappa_nn.state_dict())
                pbar.set_description(f"Total loss: {total_loss:.4f}")
                # print( (epoch, lhjb_kappa1.item(), lhjb_kappa2.item(), lconsistency_kappa1.item(), lconsistency_kappa2.item()))

        # check convergence and update the time boundary condition
        kappa_nn.load_state_dict(best_model_kappa.state_dict())

        total_loss, *_ = TP.loss_fun_Net1(kappa_nn, SV)

        # match the time boundary condition
        loss_time_boundary = 0.
        for name, model in nn_dict.items():
            loss_time_boundary += torch.sum(torch.square(model(SV_T1) - prev_vals[name]))

        total_loss += loss_time_boundary
        if total_loss < outer_loop_min_loss:
            print(f"Updating min loss from {outer_loop_min_loss:.4f} to {total_loss:.4f}")
            outer_loop_min_loss = total_loss

        new_vals: Dict[str, torch.Tensor] = {}
        for name, model in nn_dict.items():
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
    kappa_nn.load_state_dict(best_model_kappa.state_dict())
    
    # Save data
    SV_T0 = TS.sample_fixed_grid_boundary_cond(0.0).to(device)
    SV_T0.requires_grad_(True)
    
    TP  = Training_pde(params)
    TP.loss_fun_Net1(kappa_nn, SV_T0)

    kappas, qs, zetas, r = TP.kappas, TP.qs, TP.zetas, TP.r
    mu_z_geos, sig_z_geos, mu_z_aris, sig_z_aris = TP.mu_z_geos, TP.sig_z_geos, TP.mu_z_aris, TP.sig_z_aris
    mu_qs, sig_qs, mu_kappas, sig_kappas = TP.mu_qs, TP.sig_qs, TP.mu_kappas, TP.sig_kappas
    
    Z = SV_T0.detach().cpu().numpy()[:, :1].reshape(-1)
    fig, ax = plt.subplots(3,2,figsize=(16,9), num=1)
    ax[0,0].plot(Z,kappas[:, 0].detach().cpu().numpy(),label=r'$\kappa_1$')
    ax[0,0].plot(Z,kappas[:, 1].detach().cpu().numpy(),label=r'$\kappa_2$')
    ax[0,0].legend()

    ax[0,1].plot(Z,qs[:,0].detach().cpu().numpy(),label=r'$q_1$')
    ax[0,1].plot(Z,qs[:,1].detach().cpu().numpy(),label=r'$q_2$')
    ax[0,1].legend()
    

    ax[1,0].plot(Z,zetas[:,0].detach().cpu().numpy(),label=r'$\zeta_1$')
    ax[1,0].plot(Z,zetas[:,1].detach().cpu().numpy(),label=r'$\zeta_2$')
    ax[1,0].legend()
    

    ax[1,1].plot(Z,r.detach().cpu().numpy(),label=r'$r$')
    ax[1,1].set_title('r')

    ax[2,0].plot(Z,mu_z_geos[:, 0].detach().cpu().numpy(),label=r'$\mu^z$')
    ax[2,0].plot(Z,sig_z_geos[:, 0].detach().cpu().numpy(),label=r'$\sigma^z$')
    ax[2,0].legend()
    ax[2,0].set_xlabel('z')

    ax[2,1].plot(Z,mu_z_aris[:, 0].detach().cpu().numpy(),label=r'$\mu_z$')
    ax[2,1].plot(Z,sig_z_aris[:, 0].detach().cpu().numpy(),label=r'$\sigma_z$')
    ax[2,1].legend()
    ax[2,1].set_xlabel('z')

    
    name = 'equilibrium.jpg'
    plt.savefig(os.path.join(plot_directory, name),bbox_inches='tight',dpi=300)
    plt.show()
    plt.close('all')

    print('Training complete')



