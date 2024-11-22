import gc
import os
from copy import deepcopy
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, Comparator, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, set_seeds)
from parse_ditella_sol import *

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

BASE_DIR = "./models/StochasticVolatility"
os.makedirs(BASE_DIR, exist_ok=True)

LATEX_VAR_MAPPING = {
    # variables
    r"\iota": "iota",
    r"\hat{e}": "e_hat",
    r"\hat{c}": "c_hat",
    r"\sigma_{x,1}": "sigxtop",
    r"\sigma_{x,2}": "sigxbot",
    r"\sigma_x": "sigx",
    r"\sigma_p": "sigp",
    r"\sigma_\xi": "sigxi",
    r"\sigma_\zeta": "sigzeta",
    r"\tilde{\sigma_n}": "signtilde",
    r"\sigma_n": "sign",
    r"\pi": "signxi",
    r"\sigma_w": "sigw",
    r"\mu_n": "mun",
    r"\mu_x": "mux",
    r"\mu_p": "mup",
    r"\mu_\xi": "muxi",
    r"\mu_\zeta": "muzeta",
    r"\mu_w": "muw",

    # agents
    r"\xi": "xi",
    r"\zeta": "zeta",

    # constants
    r"\bar{\sigma_v}": "sigv_mean",
    r"\sigma_v": "sigv",
    r"\mu_v": "muv",
    r"\sigma": "sigma",
    r"\lambda": "lbd",
    r"\bar{v}": "v_mean",
    r"\rho": "rho",
    r"\gamma": "gamma",
    r"\psi": "psi",
    r"\tau": "tau",
    r"\delta": "delta",
    r"\phi": "phi",
}

PARAMS = {
    "a": 1,
    "sigma": 0.0125,
    "lbd": 1.38,
    "v_mean": 0.25,
    "sigv_mean": -0.17,
    "rho": 0.0665,
    "gamma": 5,
    "psi": 0.5,
    "tau": 1.15,
    "phi": 0.2,

    "A": 53.2,
    "B": -0.8668571428571438,
    "delta": 0.05,
}
STATES = ["x", "v"] 
PROBLEM_DOMAIN = {"x": [0.05, 0.95], "v": [0.05, 0.95]}

VARS_TO_PLOT = ["p", "sigx", "omega", "sigsigp", "signxi", "r"]
PLOT_ARGS = {
    # "p": {"ylabel": r"$p$", "title": r"Price"},
    # "sigx": {"ylabel": r"$\sigma_x$", "title": r"Diffusion of Wealth Share"},
    "omega": {"ylabel": r"$\Omega=\xi/\zeta$", "title": r"Ratio of Value Functions ($\Omega=\xi/\zeta$)", "show_legend": True},
    "sigsigp": {"ylabel": r"$\sigma+\sigma_p$", "title": r"Price Return Diffusion", "show_legend": False},
    "signxi": {"ylabel": r"$\pi$", "title": r"Price of Risk", "show_legend": False},
    # "r": {"ylabel": r"$r$", "title": r"Risk-Free Rate"},
}
v_list = [0.1, 0.25, 0.6]
COLORS = ["red", "orange", "blue"]

MODEL_CONFIGS = {
    "Agents": {
        "xi": {"positive": True},
        "zeta": {"positive": True},
    },
    "Endogs": {
        "p": {"positive": True},
        "r": {},
    }
}

TRAINING_CONFIGS = {
    "basic": {
        "sampling_method": SamplingMethod.UniformRandom, 
        "batch_size": 500,
        "num_epochs": 20000,
        "optimizer_type": OptimizerType.Adam,
    },
    "timestep": {
        "batch_size": 500, 
        "num_outer_iterations": 50, 
        "num_inner_iterations": 5000,
        "sampling_method": SamplingMethod.UniformRandom, 
        "time_batch_size": 1,
    },
    "timestep_lb": {
        "batch_size": 500, 
        "num_outer_iterations": 50, 
        "num_inner_iterations": 5000,
        "sampling_method": SamplingMethod.UniformRandom, 
        "time_batch_size": 1,
        "loss_balancing": True
    }
}

def get_equations(timestepping=False):
    BASE_EQUATIONS = [
        r"$g &= \frac{1}{2*A} * (p - B) - \delta$", 
        r"$\iota &= A * (g+\delta)^2 + B * (g+\delta)$",
        r"$\mu_v &= \lambda * (\bar{v} - v)$",
        r"$\sigma_v &= \bar{\sigma_v} * \sqrt{v}$",
        r"$\hat{e} &= \rho^{1/\psi} * \xi^{(\psi-1)/\psi}$",
        r"$\hat{c} &= \rho^{1/\psi} * \zeta^{(\psi-1)/\psi}$",
        r"$\sigma_{x,1} &= (1-x) * x * \frac{1-\gamma}{\gamma} * \left( \frac{1}{\xi} * \frac{\partial \xi}{\partial v} - \frac{1}{\zeta} * \frac{\partial \zeta}{\partial v} \right)$",
        r"$\sigma_{x,2} &= 1 - (1-x) * x * \frac{1-\gamma}{\gamma} * \left( \frac{1}{\xi} * \frac{\partial \xi}{\partial x} - \frac{1}{\zeta} * \frac{\partial \zeta}{\partial x} \right)$",
        r"$\sigma_x &= \frac{\sigma_{x,1}}{\sigma_{x,2}} * \sigma_v$",
        r"$\sigma_p &= \frac{1}{p} * \left( \frac{\partial p}{\partial v} * \sigma_v + \frac{\partial p}{\partial x} * \sigma_x \right)$",
        r"$\sigma_\xi &= \frac{1}{\xi} * \left( \frac{\partial \xi}{\partial v} * \sigma_v + \frac{\partial \xi}{\partial x} * \sigma_x \right)$",
        r"$\sigma_\zeta &= \frac{1}{\zeta} * \left( \frac{\partial \zeta}{\partial v} * \sigma_v + \frac{\partial \zeta}{\partial x} * \sigma_x \right)$",
        r"$\sigma_n &= \sigma + \sigma_p + \frac{\sigma_x}{x}$",
        r"$\pi &= \gamma * \sigma_n + (\gamma-1) * \sigma_\xi$",
        r"$\sigma_w &= \frac{\pi}{\gamma} - \frac{\gamma-1}{\gamma} *  \sigma_\zeta$",
        r"$\mu_w &= r + \pi * \sigma_w$",
        r"$\mu_n &= r + \frac{\gamma}{x^2} * (\phi * v)^2 + \pi * \sigma_n$",
        r"$\tilde{\sigma_n} &= \frac{\phi}{x} * v$",
        r"$\mu_x &= x * \left(\mu_n - \hat{e} - \tau + \frac{a-\iota}{p} - r - \pi * (\sigma+\sigma_p) - \frac{\gamma}{x} * (\phi * v)^2 + (\sigma + \sigma_p)^2 - \sigma_n * (\sigma + \sigma_p)\right)$",
    ]
    equations = BASE_EQUATIONS.copy()
    if timestepping:
        equations += [
            r"$\mu_p &= \frac{1}{p} * \left( \frac{\partial p}{\partial t} + \mu_v * \frac{\partial p}{\partial v} + \mu_x * \frac{\partial p}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 p}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 p}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 p}{\partial x^2} \right)\right)$",
            r"$\mu_\xi &= \frac{1}{\xi} * \left( \frac{\partial \xi}{\partial t} + \mu_v * \frac{\partial \xi}{\partial v} + \mu_x * \frac{\partial \xi}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \xi}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \xi}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \xi}{\partial x^2} \right)\right)$",
            r"$\mu_\zeta &= \frac{1}{\zeta} * \left( \frac{\partial \zeta}{\partial t} + \mu_v * \frac{\partial \zeta}{\partial v} + \mu_x * \frac{\partial \zeta}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \zeta}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \zeta}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \zeta}{\partial x^2} \right)\right)$",
        ]
    else:
        equations += [
            r"$\mu_p &= \frac{1}{p} * \left( \mu_v * \frac{\partial p}{\partial v} + \mu_x * \frac{\partial p}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 p}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 p}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 p}{\partial x^2} \right)\right)$",
            r"$\mu_\xi &= \frac{1}{\xi} * \left( \mu_v * \frac{\partial \xi}{\partial v} + \mu_x * \frac{\partial \xi}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \xi}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \xi}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \xi}{\partial x^2} \right)\right)$",
            r"$\mu_\zeta &= \frac{1}{\zeta} * \left( \mu_v * \frac{\partial \zeta}{\partial v} + \mu_x * \frac{\partial \zeta}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \zeta}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \zeta}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \zeta}{\partial x^2} \right)\right)$",
        ]
    
    equations += [
        "omega=xi/zeta", 
        "sigsigp=sigma+sigp",
    ]

    return equations

ENDOG_EQUATIONS = [
    r"$a - \iota &= p * (\hat{e} * x + \hat{c} * (1-x))$",
    r"$\sigma + \sigma_p &= \sigma_n * x + \sigma_w * (1-x)$",
    r"$\frac{a-\iota}{p} + g + \mu_p + \sigma * \sigma_p - r &= (\sigma + \sigma_p) * \pi + \gamma * \frac{1}{x} * (\phi * v)^2$",
]

HJB_EQUATIONS = [
    r"$\frac{\hat{e}^{1-\psi}}{1-\psi} * \rho * \xi^{\psi-1} + \frac{\tau}{1-\gamma} * \left(\left(\frac{\zeta}{\xi} \right)^{1-\gamma}-1 \right) + \mu_n - \hat{e} + \mu_\xi - \frac{\gamma}{2} * \left( \sigma_n^2 + \sigma_\xi^2 - 2 * \frac{1-\gamma}{\gamma} * \sigma_n * \sigma_\xi + \tilde{\sigma_n}^2 \right) - \frac{\rho}{1-\psi}$",
    r"$\frac{\hat{c}^{1-\psi}}{1-\psi} * \rho * \zeta^{\psi-1} + \mu_w - \hat{c} + \mu_\zeta - \frac{\gamma}{2} * \left( \sigma_w^2 + \sigma_\zeta^2 - 2 * \frac{1-\gamma}{\gamma} * \sigma_w * \sigma_\zeta \right) - \frac{\rho}{1-\psi}$"
]

def setup_model(timestepping=False, loss_balancing=False) -> Union[PDEModel, PDEModelTimeStep]:
    equations = get_equations(timestepping=timestepping)
    set_seeds(0)
    if timestepping:
        if loss_balancing:
            model = PDEModelTimeStep("stochastic_volatility", TRAINING_CONFIGS["timestep_lb"], LATEX_VAR_MAPPING)
        else:
            model = PDEModelTimeStep("stochastic_volatility", TRAINING_CONFIGS["timestep"], LATEX_VAR_MAPPING)
    else:
        model = PDEModel("stochastic_volatility", TRAINING_CONFIGS["basic"], LATEX_VAR_MAPPING)
    model.set_state(STATES.copy(), PROBLEM_DOMAIN)
    model.add_agents(list(MODEL_CONFIGS["Agents"].keys()), MODEL_CONFIGS["Agents"])
    model.add_endogs(list(MODEL_CONFIGS["Endogs"].keys()), MODEL_CONFIGS["Endogs"])
    model.add_params(PARAMS)
    for eq in equations:
        model.add_equation(eq)
    for endog in ENDOG_EQUATIONS:
        model.add_endog_equation(endog)
    for hjb in HJB_EQUATIONS:
        model.add_hjb_equation(hjb)
    return model

def compute_func(pde_model: Union[PDEModel, PDEModelTimeStep], v_list, vars_to_plot):
    N = 100
    res_dict = {}
    for v in v_list:
        if isinstance(pde_model, PDEModelTimeStep):
            SV = torch.zeros((N, 3), device=pde_model.device)
        else:
            SV = torch.zeros((N, 2), device=pde_model.device)
        SV[:, 0] = torch.linspace(PROBLEM_DOMAIN["x"][0], PROBLEM_DOMAIN["x"][1], N)
        SV[:, 1] = torch.ones((N,)) * v
        x_plot = SV[:, 0].detach().cpu().numpy().reshape(-1)
        for i, sv_name in enumerate(pde_model.state_variables):
            pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
        pde_model.update_variables(SV)
        res_dict["x_plot"] = x_plot
        for var in vars_to_plot:
            res_dict[f"{var}_{v}"] = pde_model.variable_val_dict[var].detach().cpu().numpy().reshape(-1)
    return res_dict

def plot_res(res_dicts: Dict[str, Dict[str, Any]], plot_args: Dict[str, Any], v_list: List[float]):
    x_label = "Wealth share of experts (x)"

    for i, (func_name, plot_arg) in enumerate(plot_args.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for k, l, ls, marker in [("fd", "Finite Difference", "-.", "x"), ("timestep_lb", "Our Method", "-", "")]:
            res_dict = res_dicts[k].copy()
            x_plot = res_dict.pop("x_plot")
            for i in range(len(v_list)):
                v = v_list[i]
                color = COLORS[i]
                y_vals = res_dict[f"{func_name}_{v}"]
                ax.plot(x_plot, y_vals, label=r"$v$={i} ({l})".format(i=round(v,2), l=l), linestyle=ls, color=color, marker=marker)
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_arg["ylabel"])
        # ax.set_title(plot_arg["title"])
        if plot_arg["show_legend"]:
            ax.legend()
        plt.tight_layout()
        fn = os.path.join(BASE_DIR, "plots", f"{func_name}.jpg")
        plt.savefig(fn)
        plt.close()
    
    for i, (func_name, plot_arg) in enumerate(plot_args.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for k, l, ls, marker in [("fd", "Finite Difference", "-.", "x"), ("basic", "Basic Neural Network", "--", ""), ("timestep_lb", "Our Method", "-", "")]:
            res_dict = res_dicts[k].copy()
            x_plot = res_dict.pop("x_plot")
            for i in range(len(v_list)):
                v = v_list[i]
                color = COLORS[i]
                y_vals = res_dict[f"{func_name}_{v}"]
                ax.plot(x_plot, y_vals, label=r"$v$={i} ({l})".format(i=round(v,2), l=l), linestyle=ls, color=color, marker=marker)
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_arg["ylabel"])
        # ax.set_title(plot_arg["title"])
        if plot_arg["show_legend"]:
            ax.legend()
        plt.tight_layout()
        fn = os.path.join(BASE_DIR, "plots", f"{func_name}_compare.jpg")
        plt.savefig(fn)
        plt.close()

def compute_mse(pde_model: PDEModelTimeStep, v_list, vars_to_plot, output_folder: str):
    ## Finite Difference Solution
    N = x_grid.shape[0]

    res_dict = {}
    for v in v_list:
        SV = torch.zeros((N, 3), device=pde_model.device)
        SV[:, 0] = torch.tensor(x_grid, dtype=torch.float32, device=pde_model.device)
        SV[:, 1] = torch.ones((N,)) * v
        for i, sv_name in enumerate(pde_model.state_variables):
            pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
        pde_model.update_variables(SV)
        for var in vars_to_plot:
            res_dict[f"{var}_{v}"] = pde_model.variable_val_dict[var].detach().cpu().numpy().reshape(-1)

    with open(os.path.join(output_folder, "mse.txt"), "w") as f:
        for i, var in enumerate(VARS_TO_PLOT):
            total_squares = 0.
            for v in v_list:
                curr_square = (res_dict[f"{var}_{v}"] - ditella_res_dict[f"{var}_{v}"]) ** 2
                total_squares += curr_square
            curr_mse = np.mean(total_squares) / len(v_list) # average across the slices
            print(f"{var} MSE: {curr_mse}", file=f)

def compute_consumption_mse(pde_model: PDEModelTimeStep, output_folder: str):
    N = x_grid.shape[0]
    x_tensor, v_tensor = torch.meshgrid(torch.tensor(x_grid), torch.tensor(v_grid))
    SV = torch.zeros((N**2, 3), device=pde_model.device)
    SV[:, 0] = x_tensor.reshape(-1)
    SV[:, 1] = v_tensor.reshape(-1)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
    pde_model.update_variables(SV)
    e_hat = pde_model.variable_val_dict["e_hat"].detach().cpu().numpy().reshape(N, N)
    c_hat = pde_model.variable_val_dict["c_hat"].detach().cpu().numpy().reshape(N, N)

    e_original = np.array(needed_eq["intere"]["original"])
    e_grid = e_original.reshape((len(v_grid), len(x_grid)))
    c_original = np.array(needed_eq["intere"]["original"])
    c_grid = c_original.reshape((len(v_grid), len(x_grid)))
    
    mse_e = np.mean((e_grid - e_hat) ** 2)
    mse_c = np.mean((c_grid - c_hat) ** 2)
    with open(os.path.join(output_folder, "mse_consumption.txt"), "w") as f:
        print(f"e MSE: {mse_e}", file=f)
        print(f"c MSE: {mse_c}", file=f)

def plot_loss(fn):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    # ax.set_title(f"Total Loss across Epochs")
    for k, l, ls in [("basic", "Basic Neural Network", "--"), ("timestep", "Time-stepping", "-."), ("timestep_lb", "Our Method", "-")]:
        curr_dir = os.path.join(BASE_DIR, k)
        loss_file = os.path.join(curr_dir, f"model_min_loss.csv")
        loss_df = pd.read_csv(loss_file)
        ax.plot(loss_df["epoch"], loss_df["total_loss"], label=l, linestyle=ls)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def plot_loss_weight(fn):
    loss_name_map = {
        "endogeq_1": "Consumption FOC",
        "endogeq_2": "Market Clearing",
        "endogeq_3": "Portfolio FOC",
        "hjbeq_1": "Experts HJB",
        "hjbeq_2": "Households HJB",
    }
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Weight")
    # ax.set_title(f"Loss Weight across Epochs (First Time Step)")
    curr_dir = os.path.join(BASE_DIR, "timestep_lb")
    loss_weight_file = os.path.join(curr_dir, "loss_weight_logs", f"model_loss_weight_0.csv")
    loss_weight_df = pd.read_csv(loss_weight_file)
    for k, label in loss_name_map.items():
        ax.plot(loss_weight_df["epoch"], loss_weight_df[k], label=label)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def plot_consumption_convergence(change_target_var={"e_hat": r"$\hat{e}$", "c_hat": r"$\hat{c}$"}):
    change_dicts = {}
    for k in ["timestep", "timestep_lb"]:
        change_dicts[k] = pd.read_csv(os.path.join(BASE_DIR, k, "model_change_dict.csv"))
    
    for var in change_target_var:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.set_xlabel("Time Step Iteration")
        ax.set_ylabel(change_target_var[var])
        # ax.set_title(f"Convergence of {change_target_var[var]} across Time Steps")
        for k, l, ls in [("timestep", "Time-stepping", "-."), ("timestep_lb", "Our Method", "-")]:
            change_df = change_dicts[k]
            ax.plot(change_df["outer_loop_iter"], change_df[f"{var}_mean_val"], label=l, linestyle=ls)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "plots", f"convergence_{var}.jpg"))
        plt.close()

def plot_abs_changes(plot_dir):
    change_dict = pd.read_csv(os.path.join(BASE_DIR, "timestep_lb", "model_change_dict.csv"))
    for var, label in [("xi_abs", r"$|\xi_{t+1}-\xi_{t}|$"), ("zeta_abs", r"$|\zeta_{t+1}-\zeta_{t}|$"),
                       ("e_hat_abs", r"$|\hat{e}_{t+1}-\hat{e}_{t}|$"), ("c_hat_abs", r"$|\hat{c}_{t+1}-\hat{c}_{t}|$"),]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(change_dict["outer_loop_iter"], change_dict[var])
        ax.set_yscale("log")
        ax.set_xlabel("Time Step Iteration")
        ax.set_ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{var}_change.jpg"))
        plt.close()

if __name__ == "__main__":
    final_plot_dicts = {}
    final_plot_dicts["fd"] = ditella_res_dict
    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
    for k in TRAINING_CONFIGS.keys():
        print(f"{k:=^80}")
        timestepping = "timestep" in k
        loss_balancing = "lb" in k
        curr_dir = os.path.join(BASE_DIR, k)
        model: Union[PDEModel, PDEModelTimeStep] = setup_model(timestepping=timestepping, loss_balancing=loss_balancing)
        if not os.path.exists(os.path.join(curr_dir, f"model_best.pt")):
            if timestepping:
                model.train_model(curr_dir, "model.pt", full_log=True, variables_to_track=["e_hat", "c_hat"])
            else:
                model.train_model(curr_dir, "model.pt", full_log=True)
        model.load_model(torch.load(os.path.join(curr_dir, "model_best.pt"), weights_only=False))
        final_plot_dicts[k] = compute_func(model, v_list, VARS_TO_PLOT)
        if timestepping and loss_balancing:
            compute_mse(model, v_list, VARS_TO_PLOT, curr_dir)
            compute_consumption_mse(model, curr_dir)
        gc.collect()
        torch.cuda.empty_cache()
    plot_res(final_plot_dicts, PLOT_ARGS, v_list)
    plot_loss(os.path.join(BASE_DIR, "plots", "loss.jpg"))
    plot_loss_weight(os.path.join(BASE_DIR, "plots", "loss_weight.jpg"))
    plot_consumption_convergence()
    plot_abs_changes(os.path.join(BASE_DIR, "plots"))