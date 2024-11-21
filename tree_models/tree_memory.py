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
import torch.optim as optim
import tree_model_hd_multioutput_rar as base_model
import tree_model_ts_hd_multioutput_rar as ts_model
from torch.profiler import ProfilerActivity, profile, record_function

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

BASE_DIR = "./models/TreeMemory"
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
    curr_para["epoch"] = 10
    curr_para["outer_loop_size"] = 2
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

def plot_mem_usage_flops():
    dfs = {}
    for k in ALL_PARAMS:
        dfs[k] = pd.read_csv(os.path.join(BASE_DIR, f"{k}_memory.csv"))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for k, l, ls in [("basic", "Basic", "--"), ("basic_rar", "Basic (RAR)", ":"), ("timestep", "Time-stepping", "-."), ("timestep_rar", "Time-stepping (RAR)", "-")]:
        df = dfs[k]
        ax.plot(df["n_trees"], df["cuda_memory_total"], label=l, linestyle=ls)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("CUDA Memory (MB)")
    ax.legend(loc="upper left")
    # ax.set_title("CUDA Memory Usage (Total)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "total_cuda_memory_usage.jpg"))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for k, l, ls in [("basic", "Basic", "--"), ("basic_rar", "Basic (RAR)", ":"), ("timestep", "Time-stepping", "-."), ("timestep_rar", "Time-stepping (RAR)", "-")]:
        df = dfs[k]
        ax.plot(df["n_trees"], df["flops_total"], label=l, linestyle=ls)
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("GFLOPS")
    ax.legend(loc="upper left")
    # ax.set_title("Number of FLOPS (Total)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "total_flops.jpg"))


if __name__ == "__main__":
    for k in ALL_PARAMS:
        if os.path.exists(os.path.join(BASE_DIR, f"{k}_memory.csv")):
            continue
        res_df = pd.DataFrame(columns=["n_trees", "cuda_memory_total", "flops_total"])
        for i_param, curr_params in enumerate(ALL_PARAMS[k]):
            n_tree = curr_params["n_trees"]
            if n_tree >= 50 and "rar" in k:
                # A100 also gets OOM when profiling, so skip
                continue
            elif n_tree == 100:
                continue
            print("{0:=^80}".format(f"{k} {n_tree}"))
            gc.collect()
            torch.cuda.empty_cache()
            if "basic" in k:
                model_lib = base_model
            elif "timestep" in k:
                model_lib = ts_model
            gc.collect()
            torch.cuda.empty_cache()

            # profile a single forward + backward pass to get the total FLOPS
            # Note that profiler will significantly slow down the training process, so the timing is not accurate anymore.
            np.random.seed(42)
            torch.manual_seed(42)
            TP = model_lib.Training_pde(curr_params)
            TS = model_lib.Training_Sampler(curr_params)
            kappa_nn = model_lib.Net1(curr_params, positive=True, sigmoid=False).to(model_lib.device)
            para_nn = list(kappa_nn.parameters())
            optimizer = optim.Adam(para_nn, lr=curr_params['lr'])
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=True, with_flops=True) as prof:
                with record_function("single_step"):
                    if "timestep" in k:
                        nn_dict = {"kappa": kappa_nn}
                        SV_T0 = TS.sample_boundary_cond(0.0).to(model_lib.device)
                        SV_T0.requires_grad_(True)
                        SV_T1 = TS.sample_boundary_cond(1.0).to(model_lib.device)
                        SV_T1.requires_grad_(True)
                        prev_vals: Dict[str, torch.Tensor] = {}
                        for nn in nn_dict:
                            prev_vals[nn] = torch.ones_like(SV_T0[:, 0:1], device=model_lib.device)
                        Z = TS.sample().to(model_lib.device)
                        if "rar" in k:
                            anchor_points = TS.sample_rar(kappa_nn, TP).to(model_lib.device)
                            Z = torch.vstack([Z, anchor_points])
                        Z.requires_grad_(True)
                    else:
                        if "rar" in k:
                            TS.sample_rar(kappa_nn, TP)
                        Z = TS.sample().to(model_lib.device)
                        Z.requires_grad_(True)
                    total_loss, hjb_kappas_, consistency_kappas_ = TP.loss_fun_Net1(kappa_nn, Z)
                    if "timestep" in k:
                        loss_time_boundary = 0.
                        for name, model in nn_dict.items():
                            loss_time_boundary += torch.mean(torch.square(model(SV_T1) - prev_vals[name]))
                        total_loss += loss_time_boundary
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
            key_avgs = prof.key_averages()
            main_loop_res = None
            total_flops = 0
            for i in range(len(key_avgs)):
                total_flops += key_avgs[i].flops
                if main_loop_res is None and "single_step" in key_avgs[i].key:
                    main_loop_res = key_avgs[i]
            
            res_df.loc[i_param, "n_trees"] = n_tree
            if hasattr(main_loop_res, "self_cuda_memory_usage"):
                mem_usage = main_loop_res.self_cuda_memory_usage / 1024**2
            elif hasattr(main_loop_res, "self_device_memory_usage"):
                mem_usage = main_loop_res.self_device_memory_usage / 1024**2
            res_df.loc[i_param, "cuda_memory_total"] = f"{mem_usage:.2f}"
            res_df.loc[i_param, "flops_total"] = f"{total_flops / 10**9:.2f}"

            gc.collect()
            torch.cuda.empty_cache()
        res_df.to_csv(os.path.join(BASE_DIR, f"{k}_memory.csv"), index=False)
    shutil.rmtree(os.path.join(BASE_DIR, "temp"), ignore_errors=True)
    
    N_TREES.remove(100)
    final_df = pd.DataFrame(index=["basic", "basic_rar", "timestep", "timestep_rar"], columns=[f"CUDA Memory {k}" for k in N_TREES] + [f"FLOPS {k}" for k in N_TREES])
    for k in ALL_PARAMS:
        res_df = pd.read_csv(os.path.join(BASE_DIR, f"{k}_memory.csv"))
        res_df["model"] = [k for _ in range(len(res_df))]
        for n_tree in N_TREES:
            if n_tree >= 50 and "rar" in k:
                # A100 also gets OOM when profiling, so skip
                continue
            elif n_tree == 100:
                continue
            idx = res_df[res_df["n_trees"] == n_tree].index
            final_df.loc[k, f"CUDA Memory {n_tree}"] = "{0:.2f}".format(res_df.loc[idx, "cuda_memory_total"].values[0])
            final_df.loc[k, f"FLOPS {n_tree}"] = "{0:.2f}".format(res_df.loc[idx, "flops_total"].values[0])

    ltx = final_df.style.to_latex(column_format="l" + "c" * len(final_df.columns), hrules=True)
    ltx = ltx.replace(" & CUDA Memory 2 & CUDA Memory 3 & CUDA Memory 5 & CUDA Memory 10 & CUDA Memory 20 & CUDA Memory 50 & FLOPS 2 & FLOPS 3 & FLOPS 5 & FLOPS 10 & FLOPS 20 & FLOPS 50 \\\\", 
r""" & \multicolumn{6}{c}{CUDA Memory (MB)} & \multicolumn{6}{c}{FLOPS ($\times 10^9$)} \\
 & 2-Tree & 3-Tree & 5-Tree & 10-Tree & 20-Tree & 50-Tree & 2-Tree & 3-Tree & 5-Tree & 10-Tree & 20-Tree & 50-Tree \\""")
    ltx = ltx.replace(r"\midrule", r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}")
    for k, v in [("basic_rar", "Basic (RAR)"), ("timestep_rar", "Our Method"), ("basic", "Basic"), ("timestep", "Time-stepping")]:
        ltx = ltx.replace(k, v)
    ltx = ltx.replace("nan", "")
    with open(os.path.join(BASE_DIR, "memory_usage.tex"), "w") as f:
        f.write(ltx)
    # plot_mem_usage_flops()

