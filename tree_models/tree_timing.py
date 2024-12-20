import gc
import glob
import os
import shutil
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import para
import seaborn as sns
import torch
import tree_model_hd_multioutput_rar as base_model
import tree_model_ts_hd_multioutput_rar as ts_model

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

BASE_DIR = "./models/TreeTiming"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

TRIALS = 100
N_TREES = []

ALL_PARAMS = {
    "basic": [],
    "basic_rar": [],
    "timestep": [],
    "timestep_rar": [],
}

for n_tree, mu_sig, sampling_method in para.num_tree_mu_sig:
    curr_para = para.params_base.copy()
    curr_para["sample_method"] = sampling_method
    curr_para["n_trees"] = n_tree
    curr_para["mu_ys"] = mu_sig
    curr_para["sig_ys"] = mu_sig
    curr_para["epoch"] = para.EPOCHS
    curr_para["outer_loop_size"] = para.OUTER_LOOPS
    curr_para["resample_times"] = -1
    # the models don't have to be saved anyways
    curr_para["output_dir"] = os.path.join(BASE_DIR, "temp")

    rar_para = curr_para.copy()
    rar_para["resample_times"] = para.RESAMPLE_TIMES

    ALL_PARAMS["basic"].append(curr_para)
    ALL_PARAMS["basic_rar"].append(rar_para)
    ALL_PARAMS["timestep"].append(curr_para)
    ALL_PARAMS["timestep_rar"].append(rar_para)
    N_TREES.append(n_tree)

def plot_timing():
    dfs: Dict[str, pd.DataFrame] = {}
    for k in ALL_PARAMS:
        dfs[k] = pd.read_csv(os.path.join(BASE_DIR, f"{k}_timing.csv"))
    
    TOTAL_LABELS = [f"n_{n_tree}_total_time" for n_tree in N_TREES]
    EPOCH_LABELS = [f"n_{n_tree}_epoch_time" for n_tree in N_TREES]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for k, l, ls, color in [("timestep", "Time-stepping", ":", "blue"), ("timestep_rar", "Our Method", "-", "green")]:
        df = dfs[k]
        df_mean = df.mean(axis=0)
        df_5tile = df.quantile(q=0.05, axis=0)
        df_95tile = df.quantile(q=0.95, axis=0)
        ax.plot(N_TREES, df_mean[TOTAL_LABELS], label=l, linestyle=ls, color=color)
        ax.fill_between(N_TREES, df_5tile[TOTAL_LABELS], df_95tile[TOTAL_LABELS], alpha=0.2, color="gray")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Time (s)")
    # ax.set_xticks(N_TREES)
    ax.vlines(x=10, ymin=0, ymax=dfs["timestep"]["n_100_total_time"].max(), color="black", linestyles="--")
    ax.legend(loc="upper left")
    # ax.set_title("Training Time (Total)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_time_total.jpg"))


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # ("basic_rar", "Basic (RAR)", "-.", "orange"), 
    for k, l, ls, color in [("basic", "Basic Neural Network", "--", "red"), ("timestep", "Time-stepping", ":", "blue"), ("timestep_rar", "Our Method", "-", "green")]:
        df = dfs[k]
        df_mean = df.mean(axis=0)
        df_5tile = df.quantile(q=0.05, axis=0)
        df_95tile = df.quantile(q=0.95, axis=0)
        ax.plot(N_TREES, df_mean[EPOCH_LABELS], label=l, linestyle=ls, color=color)
        ax.fill_between(N_TREES, df_5tile[EPOCH_LABELS], df_95tile[EPOCH_LABELS], alpha=0.2, color="gray")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Time (s)")
    # ax.set_xticks(N_TREES)
    ax.vlines(x=10, ymin=0, ymax=dfs["timestep"]["n_100_epoch_time"].max(), color="black", linestyles="--")
    ax.legend(loc="upper left")
    # ax.set_title("Training Time (Per Epoch)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_time_per_epoch.jpg"))



if __name__ == "__main__":
    for k in ALL_PARAMS:
        result_df_path = os.path.join(BASE_DIR, f"{k}_timing.csv")
        if os.path.exists(result_df_path):
            continue
        time_df = pd.DataFrame(columns=[f"n_{n_tree}_total_time" for n_tree in N_TREES] + [f"n_{n_tree}_epoch_time" for n_tree in N_TREES])
        for curr_params in ALL_PARAMS[k]:
            n_tree = curr_params["n_trees"]
            if n_tree == 100 and "rar" in k:
                # A100 also gets OOM, so skip
                continue
            print("{0:=^80}".format(f"{k} {n_tree}"))
            gc.collect()
            torch.cuda.empty_cache()
            if "basic" in k:
                model_lib = base_model
            elif "timestep" in k:
                model_lib = ts_model
            for i in range(TRIALS):
                total_time, epoch_time =model_lib.train_loop(curr_params)
                time_df.loc[i, f"n_{n_tree}_total_time"] = total_time
                time_df.loc[i, f"n_{n_tree}_epoch_time"] = epoch_time
        time_df.to_csv(os.path.join(BASE_DIR, f"{k}_timing.csv"))
    plot_timing()
    shutil.rmtree(os.path.join(BASE_DIR, "temp"), ignore_errors=True)

