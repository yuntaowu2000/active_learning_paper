"""
Author: Goutham G. 
"""
import gc
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pyDOE import lhs
from sklearn.model_selection import ParameterGrid
from torch.func import jacrev, hessian, vmap
from tqdm import tqdm

# Import required user-defined libraries
from para import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class Net1(torch.nn.Module):
    def __init__(self, params, positive=False, sigmoid=False):
        nn_num_layers = params['nn_num_layers']
        nn_width = params['nn_width']
        super(Net1, self).__init__()
        # Initialize the first layer
        layers = [torch.nn.Linear(params['n_trees']-1, params['nn_width']), torch.nn.Tanh()]
        # Add the rest of the layers
            
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        # Final output layer        
        layers.append(torch.nn.Linear(nn_width, params['n_trees']))
        
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
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

class Training_Sampler():
    def __init__(self, params):
        self.params = params
        self.sv_count = params['n_trees'] - 1
        self.batch_size = params['batch_size']

        if params["sample_method"] == "uniform":
            self.sample = self.sample_
        elif params["sample_method"] == "log_normal":
            self.sample = self.sample_log_normal

        self.anchor_points = torch.empty((0, self.sv_count), device=device)

    # def sample(self,N):
    #     ''' Construct share by renormalization
    #         - No active sampling at the moment.
    #     '''
    #     # Construct a by latin hypercube sampling. If there are
    #     # N trees, then construct N-1 shares 
    #     #Z       = 0.05+ (0.95-0.05)*np.random.rand(N)
    #     Z      = np.random.rand(N)
        
    #     return Z.reshape((N, 1))

    def sample_(self):
        '''
            - No active sampling at the moment.
        '''
        SV = np.random.uniform(low=[0] * self.sv_count, 
                         high=[1] * self.sv_count, 
                         size=(self.batch_size, self.sv_count))
        return torch.vstack([torch.Tensor(SV).to(device), self.anchor_points])
    
    def sample_log_normal(self):
        '''
            - No active sampling at the moment.
        '''
        ys = [0] * len(self.params["mu_ys"])
        for i in range(len(self.params["mu_ys"])):
            ys[i] = torch.distributions.log_normal.LogNormal(self.params["mu_ys"][i], self.params["sig_ys"][i]).sample((self.batch_size, 1))
        ys = torch.einsum("jbi -> bj", torch.stack(ys))
        zs = ys[:, :-1] / torch.sum(ys, dim=1, keepdim=True) # the last dimension should be dropped
        return torch.vstack([zs.clone().to(device), self.anchor_points])
    
    def sample_fixed_grid_single_dim(self, non_zero_dim: int):
        sv_ls = [0] * (self.sv_count)
        for i in range(self.sv_count):
            if i == non_zero_dim:
                sv_ls[i] = torch.linspace(0, 1, steps=self.batch_size)
            else:
                sv_ls[i] = torch.zeros(1)
        sv = torch.cartesian_prod(*sv_ls)
        if len(sv.shape) == 1:
            sv = sv.unsqueeze(-1)
        return sv
    
    def sample_rar_distribution(self, kappa_nn, TP):
        # random sample 1000 points for validation, choose the highest ones
        if self.params["sample_method"] == "uniform":
            Z = np.random.uniform(low=[0] * self.sv_count, 
                         high=[1] * self.sv_count, 
                         size=(1000, self.sv_count))
            Z = torch.Tensor(Z)
        else:
            ys = [0] * len(self.params["mu_ys"])
            for i in range(len(self.params["mu_ys"])):
                ys[i] = torch.distributions.log_normal.LogNormal(self.params["mu_ys"][i], self.params["sig_ys"][i]).sample((1000, 1))
            ys = torch.einsum("jbi -> bj", torch.stack(ys))
            zs = ys[:, :-1] / torch.sum(ys, dim=1, keepdim=True) # the last dimension should be dropped
            Z = zs.clone()
        Z = Z.to(device)
        total_loss, hjb_kappas_, consistency_kappas_ = TP.loss_fun_Net1(kappa_nn, Z)
        all_losses = torch.sum(torch.square(hjb_kappas_) + torch.square(consistency_kappas_), axis=1)
        X_ids = torch.topk(all_losses, self.batch_size//self.params["resample_times"], dim=0)[1].squeeze(-1)
        self.anchor_points = torch.vstack((self.anchor_points, Z[X_ids]))
    

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
    
    def loss_fun_Net1(self, kappa_nn, Z: torch.Tensor):
        ''' Loss function for the neural network
        Remark: two trees is a one state variable problem. 
            - kappa_nn: neural network for all kappas
            - Z: share
        '''
        z   = Z.clone()
        z.requires_grad_(True)

        N = z.shape[1] + 1 # population size
        z_last    = 1 - torch.sum(z, dim=1).unsqueeze(1)
        z_all = torch.cat([z, z_last], dim=1) # (B, N)

        def compute_kappa(z):
            return kappa_nn(z)
        def compute_q(z):
            z_last = 1 - torch.sum(z, dim=-1).unsqueeze(-1)
            z_all = torch.cat([z, z_last], dim=-1) # (B, N)
            return z_all / compute_kappa(z)
        kappa_vec = compute_kappa(z)  # (B, N)
        q_vec = compute_q(z)  # (B, N)
        dkappa_dz = vmap(jacrev(compute_kappa), chunk_size=self.params["batch_size"])(z)
        dq_dz = vmap(jacrev(compute_q), chunk_size=self.params["batch_size"])(z)
        dkappa_dzz = vmap(hessian(compute_kappa), chunk_size=self.params["batch_size"])(z)
        dq_dzz = vmap(hessian(compute_q), chunk_size=self.params["batch_size"])(z)
        
         # Compute dynamics of z
        mu_ys = torch.tensor(self.params["mu_ys"], device=device).unsqueeze(0)
        sig_ys = torch.tensor(self.params["sig_ys"], device=device).unsqueeze(0)

        mu_z_geos = (
            mu_ys[:, :-1] 
            - torch.sum(mu_ys * z_all, dim=1, keepdim=True) 
            + torch.sum(sig_ys * z_all, dim=1, keepdim=True) 
                * (torch.sum(sig_ys * z_all, dim=1, keepdim=True) - sig_ys[:, :-1])
        ) # (batch, N-1)
        sig_z_geos = (
            sig_ys[:, :-1]
            - torch.sum(sig_ys * z_all, dim=1, keepdim=True)
        ) # (batch, N-1)
        mu_z_aris = mu_z_geos * z # (batch, N-1)
        sig_z_aris = sig_z_geos * z # (batch, N-1)

        mu_1minusz_ari  = -torch.sum(mu_z_aris, axis=1, keepdim=True)
        sig_1minusz_ari = -torch.sum(sig_z_aris, axis=1, keepdim=True)
        mu_1minusz_geo  = mu_1minusz_ari/z_last
        sig_1minusz_geo = sig_1minusz_ari/z_last

        # mu_z_aris, sig_z_aris (batch, N-1)
        mu_qs = (torch.einsum("bnj, bj -> bn", dq_dz, mu_z_aris)
            + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dq_dzz, sig_z_aris)
        ) / q_vec
        sig_qs = torch.einsum("bnj, bj -> bn", dq_dz, sig_z_aris) / q_vec

        r = (self.params["rho"] 
        + self.params["gamma"] * (torch.sum(mu_ys[:, :-1] * z, dim=1, keepdim=True) + mu_ys[:, -1:] * z_last)
        - 0.5 * self.params["gamma"] * (self.params["gamma"] + 1) * (torch.sum(sig_ys[:, :-1]**2 * z**2, dim=1, keepdim=True) + sig_ys[:, -1:]**2 * z_last**2)
        )

        mu_z_geos_all = torch.cat([mu_z_geos, mu_1minusz_geo], axis=1)
        sig_z_geos_all = torch.cat([sig_z_geos, sig_1minusz_geo], axis=1)
        zetas = self.params["gamma"] * z_all * sig_ys
        mu_kappas = mu_z_geos_all - mu_qs + sig_qs * (sig_qs - sig_z_geos_all)
        sig_kappas = sig_z_geos_all - sig_qs

        hjb_kappas = (torch.einsum("bnj, bj -> bn", dkappa_dz, mu_z_aris)
            + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dkappa_dzz, sig_z_aris)
            - torch.einsum("bn, bn -> bn", mu_kappas, kappa_vec)
        )
        consistency_kappas = (torch.einsum("bnj, bj -> bn", dkappa_dz, sig_z_aris)
            - torch.einsum("bn, bn -> bn", sig_kappas, kappa_vec)
        )

        lhjbs = torch.sum(torch.mean(torch.square(hjb_kappas), dim=0))
        lconsistency = torch.sum(torch.mean(torch.square(consistency_kappas), dim=0))

        # Store variables
        self.kappas = kappa_vec
        self.qs = q_vec
        self.zetas = zetas
        self.r = r
        self.mu_z_geos = mu_z_geos
        self.sig_z_geos = sig_z_geos
        self.mu_z_aris = mu_z_aris
        self.sig_z_aris = sig_z_aris
        self.mu_qs = mu_qs
        self.sig_qs = sig_qs
        self.mu_kappas = mu_kappas
        self.sig_kappas = sig_kappas

        total_loss = lhjbs + lconsistency
        return total_loss, hjb_kappas, consistency_kappas

def param_dump(params):
    params_ = params.copy()
    for k, v in params_.items():
        if isinstance(v, list):
            params_[k] = json.dumps(v)
    return json.dumps(params_, indent=4)

def train_loop(params):
    np.random.seed(42)
    torch.manual_seed(42)

    output_dir = params["output_dir"]
    plot_directory = os.path.join(output_dir, "plots")
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    with open(os.path.join(output_dir, "params.txt"), "w") as f:
        f.write(param_dump(params))
    TP = Training_pde(params)
    TS = Training_Sampler(params)

    # Initialize neural networks
    kappa_nn = Net1(params, positive=True, sigmoid=False).to(device)

    # Initialize neural nets to store last best model
    best_model_kappa = Net1(params, positive=True, sigmoid=False).to(device)
    best_model_kappa.load_state_dict(kappa_nn.state_dict())

    para_nn = list(kappa_nn.parameters())


    # Set hyperparameters
    optimizer       = optim.Adam(para_nn, lr=params['lr'])
    epochs          = params["epoch"]
    min_loss        = torch.inf

    min_loss_dict = defaultdict(list)
    kappa_val_dict = defaultdict(list)

    start_time = time.time()
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        Z = TS.sample().to(device)
        Z.requires_grad_(True)
        total_loss, hjb_kappas_, consistency_kappas_ = TP.loss_fun_Net1(kappa_nn, Z)

        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(para_nn, 1)
  
        optimizer.step()
        loss_val = total_loss.item()
        if (loss_val < min_loss):
            min_loss = loss_val
            best_model_kappa.load_state_dict(kappa_nn.state_dict())
            pbar.set_description(f"Total loss: {total_loss:.4f}")
            min_loss_dict["epoch"].append(len(min_loss_dict["epoch"]))
            min_loss_dict["total_loss"].append(loss_val)
        kappa_val_dict["epoch"].append(len(kappa_val_dict["epoch"]))
        for i in range(params["n_trees"]):
            kappa_val_dict[f"kappa_{i+1}"].append(torch.mean(TP.kappas[:,i]).item())
            kappa_val_dict[f"q_{i+1}"].append(torch.mean(TP.qs[:,i]).item())
        if (epoch + 1) % (epochs // params["resample_times"]) == 0:
            TS.sample_rar_distribution(kappa_nn, TP)
    end_time = time.time()
    summary_to_write = "Model Architecture:\n"
    summary_to_write += str(kappa_nn) + "\n"
    summary_to_write += f"Num Params: {kappa_nn.get_num_params()}\n"
    summary_to_write += f"Time taken: {end_time - start_time:.2f} seconds\n"
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_to_write)


    pd.DataFrame(min_loss_dict).to_csv(f"{output_dir}/min_loss.csv", index=False)
    pd.DataFrame(kappa_val_dict).to_csv(f"{output_dir}/kappa_val.csv", index=False)
    torch.save({"model": best_model_kappa.state_dict(), "params": params}, os.path.join(output_dir, "model.pt"))

    added_anchor_points = TS.anchor_points.detach().cpu().numpy()
    np.save(os.path.join(output_dir, f"anchor_points.npy"), added_anchor_points)
    
    # Load last best model as the final neural network model
    kappa_nn.load_state_dict(best_model_kappa.state_dict())
    # Save data
    Z = TS.sample_fixed_grid_single_dim(0).to(device)
    Z.requires_grad_(True)
    
    TP  = Training_pde(params)
    TP.loss_fun_Net1(kappa_nn, Z)
    kappas, qs, zetas, r = TP.kappas, TP.qs, TP.zetas, TP.r
    mu_z_geos, sig_z_geos, mu_z_aris, sig_z_aris = TP.mu_z_geos, TP.sig_z_geos, TP.mu_z_aris, TP.sig_z_aris
    mu_qs, sig_qs, mu_kappas, sig_kappas = TP.mu_qs, TP.sig_qs, TP.mu_kappas, TP.sig_kappas
    
    Z = Z.detach().cpu().numpy()[:, :1].reshape(-1)
    fig, ax = plt.subplots(3,2,figsize=(16,9), num=1)
    ax[0,0].plot(Z,kappas[:, 0].detach().cpu().numpy(),label=r'$\kappa_1$')
    ax[0,0].plot(Z,kappas[:, -1].detach().cpu().numpy(),label=r'$\kappa_{' + str(params["n_trees"]) + "}$")
    ax[0,0].legend()

    ax[0,1].plot(Z,qs[:,0].detach().cpu().numpy(),label=r'$q_1$')
    ax[0,1].plot(Z,qs[:,-1].detach().cpu().numpy(),label=r'$q_{' + str(params["n_trees"]) + "}$")
    ax[0,1].legend()
    

    ax[1,0].plot(Z,zetas[:,0].detach().cpu().numpy(),label=r'$\zeta_1$')
    ax[1,0].plot(Z,zetas[:,-1].detach().cpu().numpy(),label=r'$\zeta_{' + str(params["n_trees"]) + "}$")
    ax[1,0].legend()
    

    ax[1,1].plot(Z,r.detach().cpu().numpy(),label=r'$r$')
    ax[1,1].set_title('r')

    ax[2,0].plot(Z,mu_z_geos[:, 0].detach().cpu().numpy(),label=r'$\mu^{z_1}$')
    ax[2,0].plot(Z,sig_z_geos[:, 0].detach().cpu().numpy(),label=r'$\sigma^{z_1}$')
    ax[2,0].legend()
    ax[2,0].set_xlabel('z')

    ax[2,1].plot(Z,mu_z_aris[:, 0].detach().cpu().numpy(),label=r'$\mu_{z_1}$')
    ax[2,1].plot(Z,sig_z_aris[:, 0].detach().cpu().numpy(),label=r'$\sigma_{z_1}$')
    ax[2,1].legend()
    ax[2,1].set_xlabel('z')

    
    name = 'equilibrium.jpg'
    plt.savefig(os.path.join(plot_directory, name),bbox_inches='tight',dpi=300)
    # plt.show()
    plt.close()

    print('Training complete')

def distribution_plot(params, batch_size=5000):
    np.random.seed(42)
    torch.manual_seed(42)

    params_ = params.copy()
    params_["batch_size"] = batch_size
    TP = Training_pde(params_)
    TS = Training_Sampler(params_)

    kappa_nn = Net1(params, positive=True, sigmoid=False).to(device)
    kappa_nn.load_state_dict(torch.load(os.path.join(params["output_dir"], "model.pt"))["model"])

    Z = TS.sample().to(device)
    Z.requires_grad_(True)
    TP.loss_fun_Net1(kappa_nn, Z)

    kappas, qs, zetas, r = TP.kappas, TP.qs, TP.zetas, TP.r
    mu_z_geos, sig_z_geos, mu_z_aris, sig_z_aris = TP.mu_z_geos, TP.sig_z_geos, TP.mu_z_aris, TP.sig_z_aris
    mu_qs, sig_qs, mu_kappas, sig_kappas = TP.mu_qs, TP.sig_qs, TP.mu_kappas, TP.sig_kappas

    # mu_qs have shape (batch_size, N)
    mu_qs_mean = torch.mean(mu_qs, dim=0) # shape (N,)
    mu_qs_max_idx = torch.argmax(mu_qs_mean[:-1], dim=0)
    # mu_kappas_mean = torch.mean(mu_kappas, dim=0) # shape (N,)
    # mu_kappas_max_idx = torch.argmax(mu_kappas[:-1], dim=0)

    # plot the kappa and q histograms with max mu_q
    kappa_to_plot = kappas[:, mu_qs_max_idx].detach().cpu().numpy()
    q_to_plot = qs[:, mu_qs_max_idx].detach().cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(kappa_to_plot, bins=20)
    ax[0].set_xlabel(r'$\kappa$')
    ax[0].set_title(r"$\kappa$ distribution at max $\mu^q$")

    ax[1].hist(q_to_plot, bins=20, label=r'$q$')
    ax[1].set_xlabel(r'$q$')
    ax[1].set_title(r"$q$ distribution at max $\mu^q$")

    name = 'distribution.jpg'
    plt.savefig(os.path.join(params["output_dir"], "plots", name),bbox_inches='tight',dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    param_grid = ParameterGrid({
        "sample_method": sample_methods, 
        "num_tree_mu_sig": num_tree_mu_sig,
    })
    for param_set in param_grid:
        sample_method = param_set["sample_method"]
        n_trees, mu_sig = param_set["num_tree_mu_sig"]
        curr_params = params_base.copy()
        curr_params["sample_method"] =sample_method
        curr_params["n_trees"] = n_trees
        curr_params["mu_ys"] = mu_sig
        curr_params["sig_ys"] = mu_sig 
        curr_params["output_dir"] = f"./models_multioutput_rar/tree{n_trees}_{sample_method}"
        if n_trees > 2:
            curr_params["batch_size"] = 100
        else:
            curr_params["batch_size"] = 100
        if n_trees > 20:
            curr_params["epoch"] = 20
            device = "cpu"
        else:
            curr_params["epoch"] = 200
        curr_params["resample_times"] = 5
        print(curr_params)
        gc.collect()
        torch.cuda.empty_cache()
        train_loop(curr_params)
        if n_trees >= 2:
            distribution_plot(curr_params, 1000 if n_trees > 20 else 5000)

    gc.collect()
    torch.cuda.empty_cache()