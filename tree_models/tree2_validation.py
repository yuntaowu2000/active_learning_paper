# Compare the 1D model results with finite difference methods

import gc
import os
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, plot_loss_df,
                           set_seeds)

plt.rcParams["font.size"] = 15

BASE_DIR = "./models/Tree2"
os.makedirs(BASE_DIR, exist_ok=True)

LATEX_VAR_MAPPING = {
    r"\gamma": "gamma",
    r"\rho": "rho",
    r"\mu^{y_1}": "mu_y1",
    r"\mu^{y_2}": "mu_y2",
    r"\sigma^{y_1}": "sig_y1",
    r"\sigma^{y_2}": "sig_y2",

    r"\mu^z": "muz",
    r"\sigma^z": "sigz",
    r"\mu_a^z": "muz_ari",
    r"\sigma_a^z": "sigz_ari",
    r"\mu_a^{z_2}": "muz2_ari",
    r"\sigma_a^{z_2}": "sigz2_ari",
    r"\mu^{z_2}": "muz2",
    r"\sigma^{z_2}": "sigz2",
    r"\mu^{q_1}": "mu_q1",
    r"\mu^{q_2}": "mu_q2",
    r"\sigma^{q_1}": "sig_q1",
    r"\sigma^{q_2}": "sig_q2",
    r"\zeta^1": "zeta1",
    r"\zeta^2": "zeta2",
    r"\mu^{k_1}": "mu_k1",
    r"\mu^{k_2}": "mu_k2",
    r"\sigma^{k_1}": "sig_k1",
    r"\sigma^{k_2}": "sig_k2",

    r"k_1": "k1",
    r"k_2": "k2",
    r"q_1": "q1",
    r"q_2": "q2",
}

PARAMS = {
    "gamma": 5,
    "rho": 0.05,
    "mu_y1": 0.02,
    "sig_y1": 0.02,
    "mu_y2": 0.05,
    "sig_y2": 0.05
}

STATES = ["z"]
PROBLEM_DOMAIN = {"z": [0.01, 0.99]}

VARS_TO_PLOT = ["k1", "k2", "q1", "q2", "r", "zeta1", "zeta2"]
PLOT_ARGS = {
    "k1": {"ylabel": r"$\kappa_1$", "title": r"$\kappa_1$"},
    "k2": {"ylabel": r"$\kappa_2$", "title": r"$\kappa_2$"},
    "q1": {"ylabel": r"$q_1$", "title": r"Price of Asset 1 ($q_1$)"},
    "q2": {"ylabel": r"$q_2$", "title": r"Price of Asset 2 ($q_2$)"},
    "r": {"ylabel": r"$r$", "title": r"Risk-Free Rate"},
    "zeta1": {"ylabel": r"$\zeta_1$", "title": r"Price of Risk ($\zeta_1$)"},
    "zeta2": {"ylabel": r"$\zeta_2$", "title": r"Price of Risk ($\zeta_2$)"},
}

MODEL_CONFIGS = {
    "k1": {"positive": True, "hidden_units": [80] * 4},
    "k2": {"positive": True, "hidden_units": [80] * 4},
}

TRAINING_CONFIGS = {
    "basic": {
        "batch_size": 200, 
        "num_epochs": 200, 
        "lr": 0.0005, 
        "optimizer_type": OptimizerType.Adam
    },
    "timestep": {
        "batch_size": 200, 
        "num_outer_iterations": 10, 
        "num_inner_iterations": 200,
        "sampling_method": SamplingMethod.UniformRandom, 
        "lr": 0.0005, 
        "time_batch_size": 1,
    },
    # "timestep_rar": {
    #     "batch_size": 200, 
    #     "num_outer_iterations": 10, 
    #     "num_inner_iterations": 200,
    #     "sampling_method": SamplingMethod.RARG,  
    #     "lr": 0.0005, 
    #     "time_batch_size": 1,
    #     "refinement_rounds": 5
    # }
}

EQUATIONS = [
    r"$q_1 &= \frac{z}{k_1}$",
    r"$q_2 &= \frac{1-z}{k_2}$",
    "q1_z=deriv(q1, SV)[:,:1]",
    "q1_zz=deriv(q1_z, SV)[:,:1]",
    "q2_z=deriv(q2, SV)[:,:1]",
    "q2_zz=deriv(q2_z, SV)[:,:1]",
    r"$\mu^z &= \mu^{y_1} - (z * \mu^{y_1} + (1-z) * \mu^{y_2}) + (z * \sigma^{y_1} + (1-z) * \sigma^{y_2}) * (z * \sigma^{y_1} + (1-z) * \sigma^{y_2} - \sigma^{y_1})$",
    r"$\sigma^z &= \sigma^{y_1} - (z * \sigma^{y_1} + (1-z) * \sigma^{y_2})$",
    r"$\mu_a^z &= \mu^z * z$",
    r"$\sigma_a^z &= \sigma^z * z$",
    r"$\mu_a^{z_2} &= -\mu_a^z$",
    r"$\sigma_a^{z_2} &= -\sigma_a^z$",
    r"$\mu^{z_2} &= \frac{\mu_a^{z_2}}{1-z}$",
    r"$\sigma^{z_2} &= \frac{\sigma_a^{z_2}}{1-z}$",
    r"$\mu^{q_1} &= \frac{1}{q_1} * \left(\frac{\partial q_1}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 q_1}{\partial z^2} * (\sigma_a^z)^2\right)$",
    r"$\mu^{q_2} &= \frac{1}{q_2} * \left(\frac{\partial q_2}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 q_2}{\partial z^2} * (\sigma_a^z)^2\right)$",
    r"$\sigma^{q_1} &= \frac{1}{q_1} * \frac{\partial q_1}{\partial z} * \sigma_a^z$",
    r"$\sigma^{q_2} &= \frac{1}{q_2} * \frac{\partial q_2}{\partial z} * \sigma_a^z$",
    r"$r &= \rho + \gamma * (z * \sigma^{y_1} + (1-z) * \sigma^{y_2}) - \frac{1}{2} * \gamma * (\gamma + 1) * ((z*\sigma^{y_1})^2 + ((1-z)*\sigma^{y_2})^2)$",
    r"$\zeta^1 &= \gamma * z * \sigma^{y_1}$",
    r"$\zeta^2 &= \gamma * (1-z) * \sigma^{y_2}$",
    r"$\mu^{k_1} &= \mu^z - \mu^{q_1} + \sigma^{q_1} * (\sigma^{q_1} - \sigma^z)$",
    r"$\mu^{k_2} &= \mu^{z_2} - \mu^{q_2} + \sigma^{q_2} * (\sigma^{q_2} - \sigma^{z_2})$",
    r"$\sigma^{k_1} &= \sigma^z - \sigma^{q_1}$",
    r"$\sigma^{k_2} &= \sigma^{z_2} - \sigma^{q_2}$",
]

def get_hjb_equations(timestepping=False):
    if timestepping:
        return [
            r"$\frac{\partial k_1}{\partial t} + \frac{\partial k_1}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 k_1}{\partial z^2} * (\sigma_a^z)^2 - \mu^{k_1} * k_1$",
            r"$\frac{\partial k_2}{\partial t} + \frac{\partial k_2}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 k_2}{\partial z^2} * (\sigma_a^z)^2 - \mu^{k_2} * k_2$",
            r"$\frac{\partial k_1}{\partial z} * \sigma_a^z - \sigma^{k_1} * k_1$",
            r"$\frac{\partial k_2}{\partial z} * \sigma_a^z - \sigma^{k_2} * k_2$",
        ]
    else:
        return [
            r"$\frac{\partial k_1}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 k_1}{\partial z^2} * (\sigma_a^z)^2 - \mu^{k_1} * k_1$",
            r"$\frac{\partial k_2}{\partial z} * \mu_a^z + \frac{1}{2} * \frac{\partial^2 k_2}{\partial z^2} * (\sigma_a^z)^2 - \mu^{k_2} * k_2$",
            r"$\frac{\partial k_1}{\partial z} * \sigma_a^z - \sigma^{k_1} * k_1$",
            r"$\frac{\partial k_2}{\partial z} * \sigma_a^z - \sigma^{k_2} * k_2$",
        ]
    
def setup_model(timestepping=False, rar=False) -> Union[PDEModel, PDEModelTimeStep]:
    hjb_equations = get_hjb_equations(timestepping=timestepping)
    set_seeds(42)
    if timestepping:
        if rar:
            model = PDEModelTimeStep("tree2", TRAINING_CONFIGS["timestep_rar"], LATEX_VAR_MAPPING)
        else:
            model = PDEModelTimeStep("tree2", TRAINING_CONFIGS["timestep"], LATEX_VAR_MAPPING)
    else:
        model = PDEModel("tree2", TRAINING_CONFIGS["basic"], LATEX_VAR_MAPPING)
    model.set_state(STATES.copy(), PROBLEM_DOMAIN)
    model.add_endogs(MODEL_CONFIGS.keys(), MODEL_CONFIGS)
    model.endog_vars["k2"].load_state_dict(model.endog_vars["k1"].state_dict())
    model.add_params(PARAMS)
    for eq in EQUATIONS:
        model.add_equation(eq)
    for hjb in hjb_equations:
        model.add_hjb_equation(hjb)
    return model

def compute_func(pde_model: Union[PDEModel, PDEModelTimeStep], vars_to_plot):
    N = 100
    res_dict = {}
    if isinstance(pde_model, PDEModelTimeStep):
        SV = torch.zeros((N, 2), device=pde_model.device)
    else:
        SV = torch.zeros((N, 1), device=pde_model.device)
    SV[:, 0] = torch.linspace(PROBLEM_DOMAIN["z"][0], PROBLEM_DOMAIN["z"][1], N)
    SV = SV.detach().requires_grad_(True)
    x_plot = SV[:, 0].detach().cpu().numpy().reshape(-1)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
    pde_model.variable_val_dict["SV"] = SV
    pde_model.update_variables(SV)
    res_dict["x_plot"] = x_plot
    for var in vars_to_plot:
        res_dict[var] = pde_model.variable_val_dict[var].detach().cpu().numpy().reshape(-1)
    return res_dict

def plot_res(res_dicts: Dict[str, Dict[str, Any]], plot_args: Dict[str, Any]):
    x_label = "Wealth share (z)"
    
    for i, (func_name, plot_arg) in enumerate(plot_args.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for k, l, ls in [("fd", "Finite Difference", "-."), ("basic", "Basic", "--"), ("timestep", "Time-stepping", "-")]:
            res_dict = res_dicts[k].copy()
            x_plot = res_dict.pop("x_plot")
            y_vals = res_dict[f"{func_name}"]
            ax.plot(x_plot, y_vals, label=rf"{l}", linestyle=ls)
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_arg["ylabel"])
        ax.set_title(plot_arg["title"])
        ax.legend()
        plt.tight_layout()
        fn = os.path.join(BASE_DIR, "plots", f"{func_name}.jpg")
        plt.savefig(fn)
        plt.close()

def plot_loss(fn):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"Total Loss across Epochs")
    # ("basic", "Basic", "--"), 
    for k, l, ls in [("timestep", "Time-stepping", "-")]:
        curr_dir = os.path.join(BASE_DIR, k)
        loss_file = os.path.join(curr_dir, f"model_min_loss.csv")
        loss_df = pd.read_csv(loss_file)
        ax.plot(loss_df["epoch"], loss_df["total_loss"], label=l, linestyle=ls)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

if __name__ == "__main__":
    final_plot_dicts = {}
    fd_res_df = pd.read_csv("./models/2trees_solution-raw.csv")
    fd_res = {"x_plot": fd_res_df["z"].values}
    for var in VARS_TO_PLOT:
        fd_res[var] = fd_res_df[var].values
    final_plot_dicts["fd"] = fd_res
    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
    for k in TRAINING_CONFIGS.keys():
        print(f"{k:=^80}")
        timestepping = "timestep" in k
        rar = "rar" in k
        curr_dir = os.path.join(BASE_DIR, k)
        model: Union[PDEModel, PDEModelTimeStep] = setup_model(timestepping=timestepping, rar=rar)
        if not os.path.exists(os.path.join(curr_dir, f"model_best.pt")):
            if timestepping:
                model.train_model(curr_dir, "model.pt", full_log=True)
            else:
                model.train_model(curr_dir, "model.pt", full_log=True)
        model.load_model(torch.load(os.path.join(curr_dir, "model_best.pt"), weights_only=False))
        final_plot_dicts[k] = compute_func(model, VARS_TO_PLOT)
    plot_res(final_plot_dicts, PLOT_ARGS)
    plot_loss(os.path.join(BASE_DIR, "plots", "loss.jpg"))