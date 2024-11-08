# Compare the results of residual based learning with non-residual based learning for 2D models

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

BASE_DIR = "./models/Tree3"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

VARS_TO_PLOT = ["k1", "k2", "k3", "q1", "q2", "q3", "r", "zeta1", "zeta2", "zeta3"]
PLOT_ARGS = {
    "k1": {"ylabel": r"$\kappa_1$", "title": r"$\kappa_1$"},
    "k2": {"ylabel": r"$\kappa_2$", "title": r"$\kappa_2$"},
    "k3": {"ylabel": r"$\kappa_3$", "title": r"$\kappa_3$"},
    "q1": {"ylabel": r"$q_1$", "title": r"Price of Asset 1 ($q_1$)"},
    "q2": {"ylabel": r"$q_2$", "title": r"Price of Asset 2 ($q_2$)"},
    "q3": {"ylabel": r"$q_3$", "title": r"Price of Asset 3 ($q_3$)"},
    "r": {"ylabel": r"$r$", "title": r"Risk-Free Rate"},
    "zeta1": {"ylabel": r"$\zeta_1$", "title": r"Price of Risk ($\zeta_1$)"},
    "zeta2": {"ylabel": r"$\zeta_2$", "title": r"Price of Risk ($\zeta_2$)"},
    "zeta3": {"ylabel": r"$\zeta_3$", "title": r"Price of Risk ($\zeta_3$)"},
}

PARAMS = para.params_base.copy()
PARAMS["sample_method"] ="uniform"
PARAMS["n_trees"] = 3
PARAMS["mu_ys"] = [0.02, 0.05, 0.08]
PARAMS["sig_ys"] = [0.02, 0.05, 0.08] 
PARAMS["epoch"] = para.EPOCHS
PARAMS["outer_loop_size"] = para.OUTER_LOOPS
PARAMS["resample_times"] = -1

TIMESTEP_PARAMS = PARAMS.copy()
TIMESTEP_PARAMS["output_dir"] = os.path.join(BASE_DIR, "timestep")
TIMESTEP_RAR_PARAMS = PARAMS.copy()
TIMESTEP_RAR_PARAMS["output_dir"] = os.path.join(BASE_DIR, "timestep_rar")
TIMESTEP_RAR_PARAMS["resample_times"] = para.RESAMPLE_TIMES

BASE_PARAMS = PARAMS.copy()
BASE_PARAMS["output_dir"] = os.path.join(BASE_DIR, "base")
RAR_PARAMS = PARAMS.copy()
RAR_PARAMS["output_dir"] = os.path.join(BASE_DIR, "rar")
RAR_PARAMS["resample_times"] = para.RESAMPLE_TIMES

ALL_PARAMS = {
    "base": BASE_PARAMS,
    "base_rar": RAR_PARAMS,
    "timestep": TIMESTEP_PARAMS,
    "timestep_rar": TIMESTEP_RAR_PARAMS
}

def plot_min_loss(fn):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"Total Loss across Epochs")
    for k, l, ls in [("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
        curr_dir = os.path.join(BASE_DIR, k)
        loss_file = os.path.join(curr_dir, f"min_loss.csv")
        loss_df = pd.read_csv(loss_file)
        ax.plot(loss_df["epoch"], loss_df["total_loss"], label=l, linestyle=ls)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def plot_residual_points(fn):
    anchor_point_files = glob.glob(os.path.join(BASE_DIR, "timestep_rar", "anchor_points", "anchor_points_*.npy"))
    curr_rar_sampled_points = []
    for anchor_point_file in anchor_point_files:
        curr_rar_sampled_points.append(np.load(anchor_point_file))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(3):
        ax.scatter(curr_rar_sampled_points[i][:, 0], curr_rar_sampled_points[i][:, 1], label=f"Outer Loop {i+1}")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend()
    ax.set_title(f"RARG Sampled Points")
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def compute_vars(model_lib, params: dict[Any], z: torch.Tensor):
    z.requires_grad_(True)
    model: ts_model.Net1 = model_lib.Net1(params, positive=True, sigmoid=False).to(model_lib.device)
    model.load_state_dict(torch.load(os.path.join(params["output_dir"], "model.pt"))["model"])
    TP: ts_model.Training_pde = model_lib.Training_pde(params)
    TP.loss_fun_Net1(model, z)

    kappas, qs, zetas, r = TP.kappas, TP.qs, TP.zetas, TP.r
    mu_z_geos, sig_z_geos, mu_z_aris, sig_z_aris = TP.mu_z_geos, TP.sig_z_geos, TP.mu_z_aris, TP.sig_z_aris
    mu_qs, sig_qs, mu_kappas, sig_kappas = TP.mu_qs, TP.sig_qs, TP.mu_kappas, TP.sig_kappas

    res_dict = {}
    for i in range(3):
        res_dict[f"k{i+1}"] = kappas[:, i].detach().cpu().numpy()
        res_dict[f"q{i+1}"] = qs[:, i].detach().cpu().numpy()
        res_dict[f"zeta{i+1}"] = zetas[:, i].detach().cpu().numpy()
    res_dict["r"] = r.detach().cpu().numpy()
    return res_dict


def plot_variables_on_high_residuals(param_base, param_rar):
    batch_size = PARAMS["batch_size"]

    zplot1 = torch.zeros((batch_size, 3))
    zplot1[:, 0] = torch.linspace(0, 0.5, batch_size)
    zplot1[:, 1] = torch.linspace(0, 0.5, batch_size)
    zplot1 = zplot1.detach().to(ts_model.device)
    x_plot1 = zplot1[:, 0].detach().cpu().numpy()
    plot1_dict_base = compute_vars(ts_model, param_base, zplot1)
    plot1_dict_rar = compute_vars(ts_model, param_rar, zplot1)

    zplot2 = torch.zeros((batch_size, 3))
    zplot2[:, 1] = torch.linspace(0, 1.0, batch_size)
    zplot2 = zplot2.detach().to(ts_model.device)
    x_plot = zplot2[:, 1].detach().cpu().numpy()
    plot2_dict_base = compute_vars(ts_model, param_base, zplot2)
    plot2_dict_rar = compute_vars(ts_model, param_rar, zplot2)

    zplot3 = torch.zeros((batch_size, 3))
    zplot3[:, 0] = torch.linspace(0, 1.0, batch_size)
    zplot3 = zplot3.detach().to(ts_model.device)
    plot3_dict_base = compute_vars(ts_model, param_base, zplot3)
    plot3_dict_rar = compute_vars(ts_model, param_rar, zplot3)

    for var in VARS_TO_PLOT:
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))

        # plot on z_1=z_2,
        ax[0].plot(x_plot1, plot1_dict_base[var], label="Time-stepping")
        ax[0].plot(x_plot1, plot1_dict_rar[var], label="Time-stepping (RAR)")
        ax[0].set_xlabel(r"$z_1/z_2$")
        ax[0].set_title(PLOT_ARGS[var]["title"] + " on " + r"$z_1=z_2$")

        # plot on z_1=0, z_2
        ax[1].plot(x_plot, plot2_dict_base[var], label="Time-stepping")
        ax[1].plot(x_plot, plot2_dict_rar[var], label="Time-stepping (RAR)")
        ax[1].set_xlabel(r"$z_2$")
        ax[1].set_title(PLOT_ARGS[var]["title"] + " on " + r"$z_1=0$")

        # plot on z_2=0, z_1
        ax[2].plot(x_plot, plot3_dict_base[var], label="Time-stepping")
        ax[2].plot(x_plot, plot3_dict_rar[var], label="Time-stepping (RAR)")
        ax[2].set_xlabel(r"$z_1$")
        ax[2].set_title(PLOT_ARGS[var]["title"] + " on " + r"$z_2=0$")

        for i in range(3):
            ax[i].set_ylabel(PLOT_ARGS[var]["ylabel"])
            ax[i].legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{var}_high_residuals.png"))
        plt.close()


if __name__ == "__main__":
    for k in ALL_PARAMS:
        print(f"{k:=^80}")
        curr_params = ALL_PARAMS[k]
        gc.collect()
        torch.cuda.empty_cache()
        if "base" in k:
            model_lib = base_model
        elif "timestep" in k:
            model_lib = ts_model
        if not os.path.exists(os.path.join(curr_params["output_dir"], "model.pt")):
            model_lib.train_loop(curr_params)
    plot_min_loss(os.path.join(PLOT_DIR, "min_loss.png"))
    plot_residual_points(os.path.join(PLOT_DIR, "residual_points.png"))
    plot_variables_on_high_residuals(TIMESTEP_PARAMS, TIMESTEP_RAR_PARAMS)