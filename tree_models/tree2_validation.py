# Compare the 1D model results with finite difference methods

import gc
import glob
import os
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import para
import seaborn as sns
import torch
import tree_model_hd_multioutput_rar as base_model
import tree_model_ts_hd_multioutput_rar as ts_model

plt.rcParams["font.size"] = 15

BASE_DIR = "./models/Tree2"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

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

PARAMS = para.params_base.copy()
PARAMS["sample_method"] ="uniform"
PARAMS["n_trees"] = 2
PARAMS["mu_ys"] = [0.02, 0.05]
PARAMS["sig_ys"] = [0.02, 0.05] 
PARAMS["epoch"] = para.EPOCHS
PARAMS["outer_loop_size"] = para.OUTER_LOOPS
PARAMS["resample_times"] = -1

TIMESTEP_PARAMS = PARAMS.copy()
TIMESTEP_PARAMS["output_dir"] = os.path.join(BASE_DIR, "timestep")
TIMESTEP_RAR_PARAMS = PARAMS.copy()
TIMESTEP_RAR_PARAMS["output_dir"] = os.path.join(BASE_DIR, "timestep_rar")
TIMESTEP_RAR_PARAMS["resample_times"] = para.RESAMPLE_TIMES

BASE_PARAMS = PARAMS.copy()
BASE_PARAMS["output_dir"] = os.path.join(BASE_DIR, "basic")
RAR_PARAMS = PARAMS.copy()
RAR_PARAMS["output_dir"] = os.path.join(BASE_DIR, "basic_rar")
RAR_PARAMS["resample_times"] = para.RESAMPLE_TIMES

ALL_PARAMS = {
    "basic": BASE_PARAMS,
    "basic_rar": RAR_PARAMS,
    "timestep": TIMESTEP_PARAMS,
    "timestep_rar": TIMESTEP_RAR_PARAMS
}

def compute_vars(model_lib, params: dict[Any]):
    if model_lib == base_model:
        z = torch.linspace(0.01, 0.99, 100).reshape(-1, 1).to(model_lib.device)
    else:
        z = torch.zeros((100, 2))
        z[:, 0] = torch.linspace(0.01, 0.99, 100)
        z = z.detach().to(model_lib.device)
    z.requires_grad_(True)
    model: ts_model.Net1 = model_lib.Net1(params, positive=True, sigmoid=False).to(model_lib.device)
    model.load_state_dict(torch.load(os.path.join(params["output_dir"], "model.pt"))["model"])
    TP: ts_model.Training_pde = model_lib.Training_pde(params)
    TP.loss_fun_Net1(model, z)

    kappas, qs, zetas, r = TP.kappas, TP.qs, TP.zetas, TP.r
    mu_z_geos, sig_z_geos, mu_z_aris, sig_z_aris = TP.mu_z_geos, TP.sig_z_geos, TP.mu_z_aris, TP.sig_z_aris
    mu_qs, sig_qs, mu_kappas, sig_kappas = TP.mu_qs, TP.sig_qs, TP.mu_kappas, TP.sig_kappas

    res_dict = {}
    for i in range(2):
        res_dict[f"k{i+1}"] = kappas[:, i].detach().cpu().numpy()
        res_dict[f"q{i+1}"] = qs[:, i].detach().cpu().numpy()
        res_dict[f"zeta{i+1}"] = zetas[:, i].detach().cpu().numpy()
    res_dict["r"] = r.detach().cpu().numpy()
    res_dict["x_plot"] = z[:, 0].detach().cpu().numpy()
    return res_dict

def plot_res(res_dicts: Dict[str, Dict[str, Any]], plot_args: Dict[str, Any]):
    x_label = "Wealth share (z)"
    
    for i, (func_name, plot_arg) in enumerate(plot_args.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for k, l, ls in [("fd", "Finite Difference", "-."), ("basic", "Basic", "--"), ("basic_rar", "Basic (RAR)", ":"), ("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
            res_dict = res_dicts[k].copy()
            x_plot = res_dict.pop("x_plot")
            y_vals = res_dict[f"{func_name}"]
            ax.plot(x_plot, y_vals, label=rf"{l}", linestyle=ls)
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_arg["ylabel"])
        ax.set_title(plot_arg["title"])
        ax.legend()
        plt.tight_layout()
        fn = os.path.join(PLOT_DIR, f"{func_name}.jpg")
        plt.savefig(fn)
        plt.close()

def plot_loss(fn):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"Total Loss across Epochs")
    # ("basic", "Basic", "--"), 
    for k, l, ls in [("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
        curr_dir = os.path.join(BASE_DIR, k)
        loss_file = os.path.join(curr_dir, "min_loss.csv")
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
    for k in ALL_PARAMS:
        print(f"{k:=^80}")
        curr_params = ALL_PARAMS[k]
        gc.collect()
        torch.cuda.empty_cache()
        if "basic" in k:
            model_lib = base_model
        elif "timestep" in k:
            model_lib = ts_model
        if not os.path.exists(os.path.join(curr_params["output_dir"], "model.pt")):
            model_lib.train_loop(curr_params)
        final_plot_dicts[k] = compute_vars(model_lib, curr_params)
    plot_loss(os.path.join(PLOT_DIR, "min_loss.jpg"))
    plot_res(final_plot_dicts, PLOT_ARGS)