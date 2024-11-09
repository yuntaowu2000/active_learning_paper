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

BASE_DIR = "./models/Tree50"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

PARAMS = para.params_base.copy()
PARAMS["sample_method"] ="log_normal"
PARAMS["n_trees"] = 50
PARAMS["mu_ys"] = [0.01 * i for i in range(1, 51)]
PARAMS["sig_ys"] = [0.01 * i for i in range(1, 51)]
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

def plot_kappas(fn: str):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for k, l, ls in [("basic", "Basic", "--"), ("basic_rar", "Basic (RAR)", "-."), ("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
        curr_params = ALL_PARAMS[k]
        kappa_df = pd.read_csv(os.path.join(BASE_DIR, k, "kappa_val.csv"))
        kappa_cols = [f"kappa_{i+1}" for i in range(curr_params["n_trees"])]
        kappa_df["kappa"] = kappa_df[kappa_cols].mean(axis=1)
        kappa_df = kappa_df[kappa_df["epoch"] < 200]
        ax.plot(kappa_df["epoch"], kappa_df["kappa"], label=l, linestyle=ls)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kappa")
    ax.set_title(r"$\kappa$ across First 200 Epochs")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fn)

if __name__ == "__main__":
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
            model_lib.distribution_plot(curr_params, 1000)
    plot_min_loss(os.path.join(PLOT_DIR, "min_loss.png"))
    plot_kappas(os.path.join(PLOT_DIR, "kappa.png"))
    