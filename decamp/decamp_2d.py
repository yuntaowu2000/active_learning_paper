import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.func import hessian, jacrev, vmap
from tqdm import tqdm

from deep_macrofin import Comparator, OptimizerType, PDEModel, set_seeds

class PDEModelActiveSample(PDEModel): 
    def __init__(self, name, config = ..., latex_var_mapping = {}): 
        super().__init__(name, config, latex_var_mapping) 
        self.sample = self.active_sample 
    
    def active_sample(self, epoch): 
        if epoch % 100 == 0 and epoch > 0: # 
            band_size = 10 
            mu_eval = self.variable_val_dict["mu_eval"] 
            cstar = self.endog_vars["cstar"](mu_eval).detach() # (B, 1) 
            band_offsets = torch.randn((1, band_size), device=self.device) * 0.05 
            band = cstar + band_offsets # shape (B, 10) 
            band_flat = band.reshape(-1, 1) # shape (10*B, 1) 
            mu_repeated = mu_eval.repeat_interleave(band_size, dim=0) # shape (10*B, 1) 
            full_sv = torch.cat([band_flat, mu_repeated], dim=1) # shape (10*B, 2) 
            # check epoch > 0 so we don't need to resample in the first epoch 
            self.anchor_points = full_sv 
        if epoch == self.num_epochs - 1: 
            self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device) 
            # reset so we don't need to save anything 
        sv = self.sample_uniform(epoch)
        return torch.vstack((sv, self.anchor_points))
 
def compute_hjb(SV, F, F_Jac, F_Hess, r, alpha, carry_cost, siga, sigx, rho, a, b, s): 
    c = SV[:, :1] 
    mu = SV[:, 1:] 
    F_c = F_Jac[:, 0, :1] 
    F_mu = F_Jac[:, 0, 1:] 
    F_cc = F_Hess[:, 0, 0, :1] 
    F_mm = F_Hess[:, 0, 1, 1:] 
    first_order = (alpha + c * (r - carry_cost - mu)) * F_c + a * (b - mu) * F_mu 
    second_order = 0.5 * (siga**2 * c**2 - 2 * rho * siga * sigx * c + sigx**2) * F_cc + 0.5 * s**2 * F_mm 
    value_term = (r - mu) * F 
    return first_order + second_order - value_term 

def compute_bc(compute_F, c_band, mu, p, phi, omega, alpha, r): 
    # Flatten for compute_F 
    full_input = torch.cartesian_prod(c_band, mu) 

    # Compute F(c, mu) at all points 
    F_vals = compute_F(full_input) # shape (B*K, 1) 
    F_vals = F_vals.reshape(c_band.shape[0], mu.shape[0]).T 
    
    # Max over c for each mu 
    first_term = torch.max(F_vals - p * (c_band + phi), dim=1).values # shape (B,) 
    
    # Liquidation term 
    second_term = omega * alpha / (r - mu.squeeze(-1)) # shape (B,) 
    
    # Combine 
    F0 = torch.maximum(first_term, second_term).reshape(-1, 1) # shape (B, 1) 
    return F0 

def compute_opt_simple(compute_F, compute_cstar, mu): 
    ''' 
    In this function, we try to enforce F_c(c*(mu), mu)=1 and F_cc(c*(mu), mu)=0 
    ''' 
    cstar = compute_cstar(mu) 
    full_sv = torch.cat([cstar, mu], dim=1) 
    f_cstar = compute_F(full_sv) 
    f_c_cstar = vmap(jacrev(compute_F))(full_sv)[:, 0, :1] 
    f_cc_cstar = vmap(hessian(compute_F))(full_sv)[:, 0, 0, :1] 
    
    error1 = torch.mean((f_c_cstar - 1.) ** 2) 
    error2 = torch.mean(torch.relu(f_cc_cstar)) 
    return error1 + error2 

def compute_opt_active(compute_F, compute_cstar, mu): 
    ''' 
    In this function, we try to enforce F_c(c*(mu), mu)=1 and F_cc(c*(mu), mu)=0 
    ''' 
    delta = 0.02 
    cstar = compute_cstar(mu) # (B, 1) 
    band_offsets = torch.linspace(-delta, delta, 10, device=cstar.device).reshape(1, -1) # shape (1, 10) 
    band = cstar + band_offsets # shape (B, 10) 
    band_flat = band.reshape(-1, 1) # shape (10*B, 1) 
    mu_repeated = mu.repeat_interleave(10, dim=0) # shape (10*B, 1) 
    full_sv = torch.cat([band_flat, mu_repeated], dim=1) # shape (10*B, 2) 
    f_cstar = compute_F(full_sv) 
    f_c_cstar = vmap(jacrev(compute_F))(full_sv)[:, 0, :1] 
    f_cc_cstar = vmap(hessian(compute_F))(full_sv)[:, 0, 0, :1] 
    
    # Triangular weighting: peak at c_star 
    weights = 1 - torch.abs(band_offsets) / delta 
    weights_expanded = weights.repeat(cstar.shape[0], 1) # shape (B, 10)
    weights_flat = weights_expanded.reshape(-1, 1) # shape (10*B, 1) 
    error1 = torch.mean(weights_flat * (f_c_cstar - 1.) ** 2) 
    error2 = torch.mean(torch.relu(f_cc_cstar))
    return error1 + error2

def get_model(
    model_name,
    c_grid,
    mu_grid,
    params,
    active=False,
):
    set_seeds(0) 
    if active: 
        model_fn = "model_active" 
        model = PDEModelActiveSample(model_name, config={"num_epochs": 20000, "optimizer_type": OptimizerType.Adam}) 
    else: 
        model_fn = "model" 
        model = PDEModel(model_name, config={"num_epochs": 20000, "optimizer_type": OptimizerType.Adam}) 
    model.set_state(["c", "mu"], {"c": c_grid, "mu": mu_grid}) 
    model.add_params(params) 
    if active: 
        model.register_functions([compute_hjb, compute_bc, compute_opt_active]) 
    else: 
        model.register_functions([compute_hjb, compute_bc, compute_opt_simple]) 
    model.add_endog("F", config={"batch_jac_hes": True}) 
    model.add_endog("cstar", config={"batch_jac_hes": True, "sv_subset": [-1], "hidden_units": [8], "sigmoid": True}) 
    if active: 
        mu_eval = torch.linspace(mu_grid[0], mu_grid[1], 10, device=model.device).reshape(-1, 1) 
        model.add_params({"mu_eval": mu_eval}) 
    model.add_hjb_equation("compute_hjb(SV, F, F_Jac, F_Hess, r, alpha, carry_cost, siga, sigx, rho, a, b, s)") 
    if active: 
        model.add_hjb_equation("compute_opt_active(compute_F, compute_cstar, SV)", loss_reduction="None") 
    else: 
        model.add_hjb_equation("compute_opt_simple(compute_F, compute_cstar, SV)", loss_reduction="None") 
    mu_grid_small = torch.linspace(mu_grid[0], mu_grid[1], 3, device=model.device) 
    zeros = torch.stack([torch.zeros_like(mu_grid_small, device=model.device), mu_grid_small], dim=1) 
    model.add_endog_condition( "F", "F(zero)", {"zero": zeros}, 
                              Comparator.EQ, 
                              "compute_bc(compute_F, c_lin, mu_eval, prop, phi, omega, alpha, r)", 
                              params | { "c_lin": torch.linspace(c_grid[0], c_grid[1], steps=100, device=model.device), 
                                        "mu_eval": mu_grid_small, } )
    if not os.path.exists(f"./models/{model_name}/{model_fn}.pt"): 
        model.train_model(f"./models/{model_name}", f"{model_fn}.pt", True) 
    model.load_model(torch.load(f"./models/{model_name}/{model_fn}_best.pt", weights_only=False)) 
    model.eval_model(True) 
    return model

def plot_solution(model_active: PDEModel, plot_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for mu, marker in [(0.01, "^")]: # (0.0, "o"), (0.02, "s")
        SV = torch.zeros((100, 2), device=model_active.device)
        SV[:, 0] = torch.linspace(0.0, 0.6, 100, device=model_active.device)
        SV[:, 1] = mu
        cstar = model_active.endog_vars["cstar"](SV[0:1]).item()
        f = model_active.endog_vars["F"].model(SV).detach().cpu().numpy().reshape(-1)
        ax.plot(SV[:, 0].detach().cpu().numpy(), f, label=f"mu={mu}", linestyle="-", color="#5492ab", marker=marker, markevery=10)
        ax.plot([cstar, cstar, cstar], [9, 10.5, 12], linestyle="-.", color="red", marker=marker)
    ax.set_xlabel("c", fontsize=16)
    ax.set_ylabel("F(c)", fontsize=16)
    ax.set_ylim(9, 12)
    ax.tick_params(axis="both", which="major", labelsize=18)
    # ax.legend(loc="upper left", frameon=False, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_fit.pdf")
    plt.close()

def plot_loss(model_dir, plot_dir):
    model_loss = pd.read_csv(f"{model_dir}/model_loss.csv")
    model_active_loss = pd.read_csv(f"{model_dir}/model_active_loss.csv")

    x_ticks = [0, 5000, 10000, 15000, 20000]
    # HJB loss
    idx_max = model_loss["hjbeq_1"].idxmax()
    active_idx_max = model_active_loss["hjbeq_1"].idxmax()
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"].values[idx_max:], model_loss["hjbeq_1"].values[idx_max:], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"].values[active_idx_max:], model_active_loss["hjbeq_1"].values[active_idx_max:], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xticks(x_ticks, x_ticks, fontsize=18)
    ax.legend(loc="upper right", frameon=False, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_hjb.pdf")
    plt.close()

    # loss at cstar
    idx_max = model_loss["hjbeq_2"].idxmax()
    active_idx_max = model_active_loss["hjbeq_2"].idxmax()
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"].values[idx_max:], model_loss["hjbeq_2"].values[idx_max:], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"].values[active_idx_max:], model_active_loss["hjbeq_2"].values[active_idx_max:], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xticks(x_ticks, x_ticks, fontsize=18)
    # ax.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_cstar.pdf")
    plt.close()

    # volatility of loss at cstar
    window = 100

    model_loss["rolling_ratio"] = (
        model_loss["hjbeq_2"].rolling(window=window, min_periods=1).std()
        / (model_loss["hjbeq_2"].rolling(window=window, min_periods=1).mean() + 1e-8)
    )
    model_active_loss["rolling_ratio"] = (
        model_active_loss["hjbeq_2"].rolling(window=window, min_periods=1).std()
        / (model_active_loss["hjbeq_2"].rolling(window=window, min_periods=1).mean() + 1e-8)
    )
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"], model_loss["rolling_ratio"], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"], model_active_loss["rolling_ratio"], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xticks(x_ticks, x_ticks, fontsize=18)
    # ax.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_cstar_stability.pdf")
    plt.close()

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    plot_dir = "models/plots_2d"
    os.makedirs(plot_dir, exist_ok=True)
    c_grid = [0.01, 1.0]
    mu_grid = [0.0, 0.02]
    params = {
        "a": 1,
        "b": 0.01,
        "s": 0.01,
        "siga": 0.25,
        "sigx": 0.12,
        "rho": -0.2,
        "carry_cost": 0.02,
        "alpha": 0.18,
        "phi": 1.002,
        "prop": 1.06,
        "liquidation": 100,
        "omega": 0.55,
        "r": 0.03,
        "I": 10,
        "xi": 0.015,
        "eps": 1,
    }

    model = get_model("decamp_2d_liquidation", c_grid, mu_grid, params, False)
    model_active = get_model("decamp_2d_liquidation", c_grid, mu_grid, params, True)

    plot_solution(model_active, plot_dir)
    plot_loss("models/decamp_2d_liquidation", plot_dir)