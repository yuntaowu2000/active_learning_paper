import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (OptimizerType, PDEModelTimeStep, SamplingMethod,
                           set_seeds)

OUTER_ITERATIONS = 10
INNER_ITERATIONS = 500

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

def compute_q(SV, compute_k):
    z = SV[..., :-1]
    z_last = 1 - torch.sum(z, dim=-1).unsqueeze(-1)
    z_all = torch.cat([z, z_last], dim=-1) # (B, N)
    return z_all / compute_k(SV)

def compute_qz(SV, compute_k):
    return torch.vmap(torch.func.jacrev(lambda SV: compute_q(SV, compute_k)))(SV)[:, :, :-1]

def compute_qzz(SV, compute_k):
    return torch.vmap(torch.func.hessian(lambda SV: compute_q(SV, compute_k)))(SV)[:,:,:-1,:-1]

def compute_mu_z_geos(z_all, mu_ys, sig_ys):
    return (
            mu_ys[:, :-1] 
            - torch.sum(mu_ys * z_all, dim=1, keepdim=True) 
            + torch.sum(sig_ys * z_all, dim=1, keepdim=True) 
                * (torch.sum(sig_ys * z_all, dim=1, keepdim=True) - sig_ys[:, :-1])
        )
def compute_sig_z_geos(z_all, sig_ys):
    return (
            sig_ys[:, :-1]
            - torch.sum(sig_ys * z_all, dim=1, keepdim=True)
        )

def compute_mu_qs(q, dq_dz, dq_dzz, mu_z_aris, sig_z_aris):
    return (torch.einsum("bnj, bj -> bn", dq_dz, mu_z_aris)
            + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dq_dzz, sig_z_aris)
        ) / q

def compute_sig_qs(q, dq_dz, sig_z_aris):
    return torch.einsum("bnj, bj -> bn", dq_dz, sig_z_aris) / q

def compute_r(rho, gamma, mu_ys, sig_ys, z, z_last):
    return (rho + gamma * (torch.sum(mu_ys[:, :-1] * z, dim=1, keepdim=True) + mu_ys[:, -1:] * z_last)
        - 0.5 * gamma * (gamma + 1) * (torch.sum(sig_ys[:, :-1]**2 * z**2, dim=1, keepdim=True) + sig_ys[:, -1:]**2 * z_last**2)
        )

def compute_hjb_kappa(kappa, dkappa_dt, dkappa_dz, dkappa_dzz, mu_z_aris, sig_z_aris, mu_kappas):
    return torch.sum(dkappa_dt + torch.einsum("bnj, bj -> bn", dkappa_dz, mu_z_aris)
            + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dkappa_dzz, sig_z_aris)
            - torch.einsum("bn, bn -> bn", mu_kappas, kappa), dim=1, keepdim=True
        )

def compute_kappa_penalization(kappas):
    return torch.sum(torch.square(kappas - kappas[:, 0:1]), dim=1, keepdim=True)

def compute_consistency_kappa(kappa, dkappa_dz, sig_z_aris, sig_kappas):
    return torch.sum(torch.einsum("bnj, bj -> bn", dkappa_dz, sig_z_aris)
            - torch.einsum("bn, bn -> bn", sig_kappas, kappa), dim=1, keepdim=True
        )

def get_model(n_tree, params, seed=0, rar=False):
    set_seeds(seed)
    model = PDEModelTimeStep(f"tree{n_tree}", 
                 config={"batch_size": 200, "time_batch_size": -1,
                         "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
                         "num_outer_iterations": OUTER_ITERATIONS,
                         "num_inner_iterations": INNER_ITERATIONS,
                         "lr": 0.0005, 
                         "optimizer_type": OptimizerType.Adam})
    model.set_state([f"z{i+1}" for i in range(n_tree-1)], {f"z{i+1}": [0.01, 0.99] for i in range(n_tree-1)})
    model.add_params(params)
    model.add_endogs(["k"], configs={
        "k": {"positive": True, "hidden_units": [80] * 4, "output_size": n_tree},
    })
    model.register_function(compute_q)
    model.register_function(compute_qz)
    model.register_function(compute_qzz)
    model.register_function(compute_mu_z_geos)
    model.register_function(compute_sig_z_geos)
    model.register_function(compute_mu_qs)
    model.register_function(compute_sig_qs)
    model.register_function(compute_r)
    model.register_function(compute_hjb_kappa)
    model.register_function(compute_consistency_kappa)
    model.register_function(compute_kappa_penalization)
    model.add_equations([
        "z = SV[:, :-1]",
        "z_last = 1 - torch.sum(z, dim=1).unsqueeze(1)",
        "z_all = torch.cat([z, z_last], dim=1)",
        "dk_dz = k_Jac[:, :, :-1]",
        "dk_dt = k_Jac[:, :, -1]",
        "dk_dzz = k_Hess[:,:,:-1,:-1]",
        "q = compute_q(SV, compute_k)",
        "dq_dz = compute_qz(SV, compute_k)",
        "dq_dzz = compute_qzz(SV, compute_k)",
        "mu_z_geos = compute_mu_z_geos(z_all, mu_ys, sig_ys)",
        "sig_z_geos = compute_sig_z_geos(z_all, sig_ys)",
        "mu_z_aris = mu_z_geos * z",
        "sig_z_aris = sig_z_geos * z",
        "mu_1minusz_ari  = -torch.sum(mu_z_aris, axis=1, keepdim=True)",
        "sig_1minusz_ari = -torch.sum(sig_z_aris, axis=1, keepdim=True)",
        "mu_1minusz_geo  = mu_1minusz_ari/z_last",
        "sig_1minusz_geo = sig_1minusz_ari/z_last",
        "mu_qs  = compute_mu_qs(q, dq_dz, dq_dzz, mu_z_aris, sig_z_aris)",
        "sig_qs = compute_sig_qs(q, dq_dz, sig_z_aris)",
        "r = compute_r(rho, gamma, mu_ys, sig_ys, z, z_last)",
        "mu_z_geos_all = torch.cat([mu_z_geos, mu_1minusz_geo], axis=1)",
        "sig_z_geos_all = torch.cat([sig_z_geos, sig_1minusz_geo], axis=1)",
        "zetas = gamma * z_all * sig_ys",
        "mu_kappas = mu_z_geos_all - mu_qs + sig_qs * (sig_qs - sig_z_geos_all)",
        "sig_kappas = sig_z_geos_all - sig_qs",
    ])

    model.add_endog_equation("0=compute_kappa_penalization(k)")
    model.add_hjb_equation("compute_hjb_kappa(k, dk_dt, dk_dz, dk_dzz, mu_z_aris, sig_z_aris, mu_kappas)")
    model.add_hjb_equation("compute_consistency_kappa(k, dk_dz, sig_z_aris, sig_kappas)")
    return model

def format_sci(x):
    sci_str = f"{x:.2e}"  # Convert to scientific notation
    base, exp = sci_str.split("e")  # Split into base and exponent
    exp = int(exp)  # Convert exponent to integer to remove leading zeros and '+'
    if exp == 0:
        return f"{base}"
    return f"${base} \\times 10^{{{exp}}}$"

def eval_1d(model: PDEModelTimeStep, model_rar: PDEModelTimeStep, out_dir: str):
    df = pd.read_csv("models/2trees_solution-raw.csv")
    x_plot_base = df["z"]
    SV = torch.zeros((len(x_plot_base), 2), device=model.device)
    SV[:, 0] = torch.tensor(x_plot_base, dtype=torch.float32, device=model.device)
    SV.requires_grad_(True)
    for i, sv_name in enumerate(model.state_variables):
        model.variable_val_dict[sv_name] = SV[:, i:i+1]
    model.variable_val_dict["SV"] = SV
    model.update_variables(SV)
    ks_nn = model.variable_val_dict["k"].detach().cpu().numpy()
    qs_nn = model.variable_val_dict["q"].detach().cpu().numpy()

    for i, sv_name in enumerate(model_rar.state_variables):
        model_rar.variable_val_dict[sv_name] = SV[:, i:i+1]
    model_rar.variable_val_dict["SV"] = SV
    model_rar.update_variables(SV)
    ks_rar = model_rar.variable_val_dict["k"].detach().cpu().numpy()
    qs_rar = model_rar.variable_val_dict["q"].detach().cpu().numpy()

    fd_res = {
        k: np.array(df[k]) for k in ["k1", "k2", "q1", "q2"]
    }
    nn_res = {
        "k1": ks_nn[:, 0], "k2": ks_nn[:, 1],
        "q1": qs_nn[:, 0], "q2": qs_nn[:, 1],
    }
    rar_res = {
        "k1": ks_rar[:, 0], "k2": ks_rar[:, 1],
        "q1": qs_rar[:, 0], "q2": qs_rar[:, 1],
    }

    index = ["Time-stepping", "Our Method"]
    columns = pd.MultiIndex.from_tuples([("MSE", r"$\kappa_1$"), ("MSE", r"$\kappa_2$"), ("MSE", "$q_1$"), ("MSE", "$q_2$"),
                                         (r"$L^{\infty}$", r"$\kappa_1$"), (r"$L^{\infty}$", r"$\kappa_2$"), (r"$L^{\infty}$", "$q_1$"), (r"$L^{\infty}$", "$q_2$"),])
    var_maps = {"k1": r"$\kappa_1$", "k2": r"$\kappa_2$", "q1": r"$q_1$", "q2": r"$q_2$", }
    res_df = pd.DataFrame(index=index, columns=columns)
    for idx, res_dict in [("Time-stepping", nn_res), ("Our Method", rar_res)]:
        for var in ["k1", "k2", "q1", "q2"]:
            mse = np.mean((fd_res[var] - res_dict[var])**2)
            linf = np.linalg.norm((fd_res[var] - res_dict[var]), ord=np.inf)
            res_df.loc[idx, ("MSE", var_maps[var])] = format_sci(mse)
            res_df.loc[idx, (r"$L^{\infty}$", var_maps[var])] = format_sci(linf)

    ltx = res_df.style.to_latex(column_format="l" + "c"*len(columns), hrules=True, multicol_align="c")
    with open(f"{out_dir}/loss_summary.tex", "w") as f:
        f.write(ltx)
    

def plot_1d(model: PDEModelTimeStep, model_rar: PDEModelTimeStep, plot_dir: str):
    N = 100
    SV = torch.zeros((N, 2), device=model.device)
    SV[:, 0] = torch.linspace(0.01, 0.99, N, device=model.device)
    SV.requires_grad_(True)
    x_plot = SV[:, 0].detach().cpu().numpy().reshape(-1)
    for i, sv_name in enumerate(model.state_variables):
        model.variable_val_dict[sv_name] = SV[:, i:i+1]
    model.variable_val_dict["SV"] = SV
    model.update_variables(SV)
    ks_nn = model.variable_val_dict["k"].detach().cpu().numpy()
    qs_nn = model.variable_val_dict["q"].detach().cpu().numpy()

    xlabel = "$z$"
    plot_args_timestep = [
        {"y": ks_nn[:, 0], "ylabel": r"$k_1$", "title": r"$k_1$ vs. $z$"},
        {"y": ks_nn[:, 1], "ylabel": r"$k_2$", "title": r"$k_2$ vs. $z$"},
        {"y": qs_nn[:, 0], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z$"},
        {"y": qs_nn[:, 1], "ylabel": r"$q_2$", "title": r"$q_2$ vs. $z$"},
    ]


    for i, sv_name in enumerate(model_rar.state_variables):
        model_rar.variable_val_dict[sv_name] = SV[:, i:i+1]
    model_rar.variable_val_dict["SV"] = SV
    model_rar.update_variables(SV)
    ks_rar = model_rar.variable_val_dict["k"].detach().cpu().numpy()
    qs_rar = model_rar.variable_val_dict["q"].detach().cpu().numpy()

    plot_args_rar = [
        {"y": ks_rar[:, 0], "ylabel": r"$k_1$", "title": r"$k_1$ vs. $z$"},
        {"y": ks_rar[:, 1], "ylabel": r"$k_2$", "title": r"$k_2$ vs. $z$"},
        {"y": qs_rar[:, 0], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z$"},
        {"y": qs_rar[:, 1], "ylabel": r"$q_2$", "title": r"$q_2$ vs. $z$"},
    ]

    df = pd.read_csv("models/2trees_solution-raw.csv")
    plot_args_base = [
        {"y": df["k1"], "ylabel": r"$k_1$", "title": r"$k_1$ vs. $z$"},
        {"y": df["k2"], "ylabel": r"$k_2$", "title": r"$k_2$ vs. $z$"},
        {"y": df["q1"], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z$"},
        {"y": df["q2"], "ylabel": r"$q_2$", "title": r"$q_2$ vs. $z$"},
    ]
    x_plot_base = df["z"]

    vars = ["k1", "k2", "q1", "q2"]

    for i in range(len(plot_args_timestep)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(x_plot_base, plot_args_base[i]["y"], label="PyMacroFin", color="black", marker="x", markevery=10)
        ax.plot(x_plot, plot_args_timestep[i]["y"], label="Timestep", linestyle="--", color="blue")
        ax.plot(x_plot, plot_args_rar[i]["y"], label="Our Method", color="red")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_args_timestep[i]["ylabel"])
        ax.legend()
        plt.savefig(f"{plot_dir}/{vars[i]}.jpg")
        plt.close()

def plot_loss(curr_base_dir, plot_dir):
    for loss_name, plot_name in [("total_loss", "loss.jpg"), ("hjbeq_1", "loss_hjb.jpg")]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        # ax.set_title(f"Total Loss across Epochs")
        for k, l, ls in [("timestep", "Time-stepping", "-."), ("timestep_rar", "Our Method", "-")]:
            curr_dir = os.path.join(curr_base_dir, k)
            loss_file = os.path.join(curr_dir, f"model_global_min_loss.csv")
            loss_df = pd.read_csv(loss_file)
            ax.plot(loss_df["epoch"], loss_df[loss_name], label=l, linestyle=ls)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{plot_name}")
        plt.close()

def plot_residual_points(curr_base_dir, plot_dir):
    anchor_point_files = glob.glob(os.path.join(curr_base_dir, "timestep_rar", "anchor_points", "model_anchor_points_*.npy"))
    curr_rar_sampled_points = []
    for anchor_point_file in anchor_point_files:
        curr_rar_sampled_points.append(np.load(anchor_point_file))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(5):
        ax.scatter(curr_rar_sampled_points[i][:, 0], curr_rar_sampled_points[i][:, 1], label=f"Outer Loop {i+1}")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend(loc="upper right")
    # ax.set_title(f"RARG Sampled Points")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/residual_points.jpg")
    plt.close()

def plot_distribution(model: PDEModelTimeStep, n_eval=1000, fn_prefix=""):
    kappa_vals = np.zeros((n_eval,))
    q_vals = np.zeros((n_eval,))

    model_batch_size = model.batch_size
    for batch_i in range(n_eval//model_batch_size + 1):
        SV = model.sample()
        SV[:, -1] = 0
        SV.requires_grad_(True)
        for i, sv_name in enumerate(model.state_variables):
            model.variable_val_dict[sv_name] = SV[:, i:i+1]
        model.variable_val_dict["SV"] = SV
        model.update_variables(SV)
        mu_qs = model.variable_val_dict["mu_qs"] # (B, N)
        kappas = model.variable_val_dict["k"] # (B, N)
        qs = model.variable_val_dict["q"] # (B, N)
        mu_qs_mean = torch.mean(mu_qs, dim=0) # shape (N,)
        mu_qs_max_idx = torch.argmax(mu_qs_mean[:-1], dim=0)

        # plot the kappa and q histograms with max mu_q
        kappa_to_plot = kappas[:, mu_qs_max_idx].detach().cpu().numpy()
        q_to_plot = qs[:, mu_qs_max_idx].detach().cpu().numpy()
        lidx_min = batch_i*model_batch_size
        lidx_max = min((batch_i+1)*model_batch_size, n_eval)
        ridx_max = min(model_batch_size, (lidx_max-lidx_min))
        kappa_vals[lidx_min:lidx_max] = kappa_to_plot[:ridx_max]
        q_vals[lidx_min:lidx_max] = q_to_plot[:ridx_max]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(kappa_vals, bins=20)
    ax.set_xlabel(r'$\kappa$')
    plt.savefig(f"{fn_prefix}_kappa.jpg")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(q_vals, bins=20)
    ax.set_xlabel(r'$q$')
    plt.savefig(f"{fn_prefix}_q.jpg")
    plt.close()

BASE_PARAMS = {
    "gamma" : 5.0, # Household risk aversion
    "rho" : 0.05, # Fund discount rate
}
MU_SIGS = {
    k: [0.01 * i for i in range(1, k+1)] for k in [2, 3] # , 5, 10, 20, 40, 50
}
# N_MODELS = 50
BASE_DIR = "./models/"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n_tree in MU_SIGS:
        curr_base_dir = os.path.join(BASE_DIR, f"tree_{n_tree}")
        plot_dir = os.path.join(curr_base_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        mu_sig = MU_SIGS[n_tree]
        mu_sig_tensor = torch.tensor(mu_sig, dtype=torch.float32, device=device).reshape(1, -1)
        curr_params = BASE_PARAMS | {"mu_ys": mu_sig_tensor, "sig_ys": mu_sig_tensor}

        print("{0:=^80}".format(f"Tree {n_tree} base timestep"))
        curr_dir = os.path.join(curr_base_dir, "timestep")
        
        model = get_model(n_tree, curr_params, seed=0, rar=False)
        if not os.path.exists(f"{curr_dir}/model.pt"):
            model.train_model(curr_dir, f"model.pt", True)
        model.load_model(torch.load(f"{curr_dir}/model_best.pt", weights_only=False, map_location=model.device))
        model.eval_model(True)
        gc.collect()
        torch.cuda.empty_cache()

        print("{0:=^80}".format(f"Tree {n_tree} rar timestep"))
        curr_dir_rar = os.path.join(curr_base_dir, "timestep_rar")
        
        model_rar = get_model(n_tree, curr_params, seed=0, rar=True)
        if not os.path.exists(f"{curr_dir_rar}/model.pt"):
            model_rar.train_model(curr_dir_rar, f"model.pt", True)
        model_rar.load_model(torch.load(f"{curr_dir_rar}/model_best.pt", weights_only=False, map_location=model_rar.device))
        model_rar.eval_model(True)
        gc.collect()
        torch.cuda.empty_cache()

        if n_tree == 2:
            plot_1d(model, model_rar, plot_dir)
            eval_1d(model, model_rar, plot_dir)
        elif n_tree == 3:
            plot_residual_points(curr_base_dir, plot_dir)
        plot_loss(curr_base_dir, plot_dir)
