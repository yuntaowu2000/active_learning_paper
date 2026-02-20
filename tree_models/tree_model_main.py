import gc
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deep_macrofin import (OptimizerType, PDEModelTimeStep, SamplingMethod,
                           set_seeds)

OUTER_ITERATIONS = 10
INNER_ITERATIONS = 500

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

class PDEModelTimeStepCustomSample(PDEModelTimeStep):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self.sample = self.sample_custom
        self.boundary_uniform_points = self.sample_simplex()
    
    def sample_simplex(self):
        alpha = torch.ones(len(self.state_variables), device=self.device)
        samples = torch.distributions.Dirichlet(alpha).sample((self.batch_size,))
        return samples[:, :-1]

    def sample_custom(self):
        simplex = self.sample_simplex()
        T = torch.rand((self.batch_size, 1), device=self.device)
        return torch.cat([simplex, T], dim=1)
    
    def __get_refinement_loss_dict(self):
        '''
        Sample a dense subset of the problem domain, compute the loss and return total loss for each point sampled. Used for Residual-based Adaptive Refinement and Active Learning

        Returns:
            {
                "SV": sampled state variables, shape (1000, len(self.state_variables))
                "loss": total loss computed at each sv, shape (1000, 1)
            }
        '''
        # because we need a set of dense points to compute residual for adaptive sampling
        # we set all models to evaluation models so that gradients won't be computed.
        # it speeds up the computation and reduces memory usages
        self.set_all_model_eval()

        # Temporarily set a large batch size
        self.batch_size = 1000
        SV = self.sample_custom()
        SV.requires_grad_(True)
        # make a copy of variable value mapping
        # so that we don't break the top level training routine
        variable_val_dict_ = self.variable_val_dict.copy()
        total_loss = torch.zeros((self.batch_size, 1), device=self.device)

        # forward pass
        for i, sv_name in enumerate(self.state_variables):
            variable_val_dict_[sv_name] = SV[:, i:i+1]
        variable_val_dict_["SV"] = SV

        # update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            variable_val_dict_[func_name] = self.local_function_dict[func_name](SV)

        # update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            variable_val_dict_[lhs] = self.equations[eq_name].eval(self.custom_function_dict, variable_val_dict_)

        # compute total losses, without reducing to a single value, keep the original dimension, but summing up using abs values
        # Note that the conditions (IC/BC, or user pre-defined sampling regions) are not considered
        # Systems are not considered
        for label in self.endog_equations:
            total_loss += torch.mean(torch.abs(self.endog_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.constraints:
            total_loss += torch.mean(torch.abs(self.constraints[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.hjb_equations:
            total_loss += torch.mean(torch.abs(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.systems:
            total_loss += torch.mean(torch.abs(self.systems[label].eval_no_loss(self.custom_function_dict, variable_val_dict_, self.batch_size)), dim=1, keepdim=True)

        self.batch_size = self.config.get("batch_size", 100) # reset the batch size for normal computation
        self.set_all_model_training() # reset the model for training stage

        return {
            "SV": SV.detach(),
            "loss": total_loss,
        }
    
    def sample_rar_greedy(self):
        refinement_loss_dict = self.__get_refinement_loss_dict()
        SV = refinement_loss_dict["SV"]
        all_losses = refinement_loss_dict["loss"]
        X_ids = torch.topk(all_losses, self.batch_size//self.refinement_rounds, dim=0)[1].squeeze(-1)
        self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))


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
    if n_tree == 2:
        # in one dimension, it is fine to sample uniformly
        model = PDEModelTimeStep(f"tree{n_tree}", 
            config={"batch_size": 200, "time_batch_size": -1,
                    "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
                    "num_outer_iterations": OUTER_ITERATIONS,
                    "num_inner_iterations": INNER_ITERATIONS,
                    "lr": 0.0005, 
                    "optimizer_type": OptimizerType.Adam})
    else:
        model = PDEModelTimeStepCustomSample(f"tree{n_tree}", 
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
    ("MSE", r"$\kappa$"), ("MSE", r"$q$"),
    (r"$L^{\infty}$", r"$\kappa_1$"), (r"$L^{\infty}$", r"$\kappa_2$"), (r"$L^{\infty}$", "$q_1$"), (r"$L^{\infty}$", "$q_2$"),])
    var_maps = {"k1": r"$\kappa_1$", "k2": r"$\kappa_2$", "q1": r"$q_1$", "q2": r"$q_2$", }
    res_df = pd.DataFrame(index=index, columns=columns)
    for idx, res_dict in [("Time-stepping", nn_res), ("Our Method", rar_res)]:
        for var in ["k1", "k2", "q1", "q2"]:
            mse = np.mean((fd_res[var] - res_dict[var])**2)
            linf = np.linalg.norm((fd_res[var] - res_dict[var]), ord=np.inf)
            res_df.loc[idx, ("MSE", var_maps[var])] = format_sci(mse)
            res_df.loc[idx, (r"$L^{\infty}$", var_maps[var])] = format_sci(linf)
        k_err = np.stack([
            fd_res["k1"] - res_dict["k1"],
            fd_res["k2"] - res_dict["k2"],
        ], axis=0)
        mse_k = np.mean(k_err**2)
        q_err = np.stack([
            fd_res["q1"] - res_dict["q1"],
            fd_res["q2"] - res_dict["q2"],
        ], axis=0)
        mse_q = np.mean(q_err**2)
        res_df.loc[idx, ("MSE", r"$\kappa$")] = format_sci(mse_k)
        res_df.loc[idx, ("MSE", r"$q$")] = format_sci(mse_q)

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
        {"y": ks_rar[:, 0], "ylabel": r"$\kappa_1$", "title": r"$k_1$ vs. $z$"},
        {"y": ks_rar[:, 1], "ylabel": r"$\kappa_2$", "title": r"$k_2$ vs. $z$"},
        {"y": qs_rar[:, 0], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z$"},
        {"y": qs_rar[:, 1], "ylabel": r"$q_2$", "title": r"$q_2$ vs. $z$"},
    ]

    df = pd.read_csv("models/2trees_solution-raw.csv")
    plot_args_base = [
        {"y": df["k1"], "ylabel": r"$\kappa_1$", "title": r"$k_1$ vs. $z$"},
        {"y": df["k2"], "ylabel": r"$\kappa_2$", "title": r"$k_2$ vs. $z$"},
        {"y": df["q1"], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z$"},
        {"y": df["q2"], "ylabel": r"$q_2$", "title": r"$q_2$ vs. $z$"},
    ]
    x_plot_base = df["z"]

    vars = ["k1", "k2", "q1", "q2"]

    for i in range(len(plot_args_timestep)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(x_plot_base, plot_args_base[i]["y"], label="PyMacroFin", color="#000000", linestyle="-.", marker="x", markevery=10)
        ax.plot(x_plot, plot_args_timestep[i]["y"], label="Timestep", color="#317e46", linestyle="-.")
        ax.plot(x_plot, plot_args_rar[i]["y"], label="Our Method", color="#5492ab")
        ax.set_xlabel("", fontsize=20)
        ax.set_ylabel("", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.legend(frameon=False, fontsize=25)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{vars[i]}.pdf")
        plt.close()

def plot_2d(model: PDEModelTimeStepCustomSample, plot_dir: str):
    N = 100
    SV = torch.zeros((N, 3), device=model.device)
    SV[:, 0] = torch.linspace(0.01, 0.8, N, device=model.device)
    SV[:, 1] = 0.2
    SV.requires_grad_(True)
    x_plot = SV[:, 0].detach().cpu().numpy().reshape(-1)
    for i, sv_name in enumerate(model.state_variables):
        model.variable_val_dict[sv_name] = SV[:, i:i+1]
    model.variable_val_dict["SV"] = SV
    model.update_variables(SV)
    ks_nn = model.variable_val_dict["k"].detach().cpu().numpy()
    qs_nn = model.variable_val_dict["q"].detach().cpu().numpy()

    xlabel = "$z$"
    plot_args_rar = [
        {"y": ks_nn[:, 0], "ylabel": r"$\kappa_1$", "title": r"$k_1$ vs. $z_1$"},
        {"y": qs_nn[:, 0], "ylabel": r"$q_1$", "title": r"$q_1$ vs. $z_1$"},
    ]

    vars = ["k1", "q1"]

    for i in range(len(plot_args_rar)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(x_plot, plot_args_rar[i]["y"], label="Our Method", color="#5492ab")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(plot_args_rar[i]["ylabel"], fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        # ax.legend(frameon=False, fontsize=14)
        plt.savefig(f"{plot_dir}/{vars[i]}.pdf")
        plt.close()

def plot_loss(curr_base_dir, plot_dir, fontsize=20):
    for loss_name, plot_name in [("total_loss", "loss.pdf"), ("hjbeq_1", "loss_hjb.pdf")]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        # ax.set_xlabel("Epochs")
        # ax.set_ylabel("Loss")
        ax.set_yscale("log")
        # ax.set_title(f"Total Loss across Epochs")
        for k, l, ls, color in [("timestep", "Time-stepping", "-.", "#317e46"), ("timestep_rar", "Our Method", "-", "#5492ab")]:
            curr_dir = os.path.join(curr_base_dir, k)
            loss_file = os.path.join(curr_dir, f"model_global_min_loss.csv")
            loss_df = pd.read_csv(loss_file)
            ax.plot(loss_df["epoch"], loss_df[loss_name], label=l, linestyle=ls, color=color)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        ax.legend(loc="upper right", frameon=False, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{plot_name}")
        plt.close()

def plot_residual_points(curr_base_dir, plot_dir):
    anchor_point_files = glob.glob(os.path.join(curr_base_dir, "timestep_rar", "anchor_points", "model_anchor_points_*.npy"))
    curr_rar_sampled_points = []
    for anchor_point_file in anchor_point_files:
        curr_rar_sampled_points.append(np.load(anchor_point_file))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(4):
        ax.scatter(curr_rar_sampled_points[i][:, 0], curr_rar_sampled_points[i][:, 1], label=f"Outer Loop {i+1}", s=40)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=20
    )
    ax.set_xlabel("$z_1$", fontsize=20)
    ax.set_ylabel("$z_2$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/residual_points.pdf")
    plt.close()

def compute_hjb_error(model: PDEModelTimeStep, n_eval=5000):
    hjb_errs = np.zeros(n_eval)
    # temporarily add a new equation for analysis
    if "eq_test" not in model.equations:
        model.add_equation("hjb_err=compute_hjb_kappa(k, dk_dt, dk_dz, dk_dzz, mu_z_aris, sig_z_aris, mu_kappas)", label="test")
    model_batch_size = model.batch_size
    for batch_i in tqdm(range(n_eval//model_batch_size + 1), desc="computing HJB residual distribution"):
        torch.cuda.empty_cache()
        SV = model.sample()
        SV[:, -1] = 0
        SV.requires_grad_(True)
        for i, sv_name in enumerate(model.state_variables):
            model.variable_val_dict[sv_name] = SV[:, i:i+1]
        model.variable_val_dict["SV"] = SV
        model.update_variables(SV)
        lidx_min = batch_i*model_batch_size
        lidx_max = min((batch_i+1)*model_batch_size, n_eval)
        ridx_max = min(model_batch_size, (lidx_max-lidx_min))
        curr_hjb_errs = model.variable_val_dict["hjb_err"].detach().cpu().numpy().reshape(-1)
        hjb_errs[lidx_min:lidx_max] = np.abs(curr_hjb_errs[:ridx_max]) ** 2 # get the squared error
        del SV
        gc.collect()
        torch.cuda.empty_cache()
    return hjb_errs

def plot_hjb_error_distribution(model: PDEModelTimeStep, model_rar: PDEModelTimeStep, plot_dir, n_eval=5000):
    hjb_errs = compute_hjb_error(model, n_eval)
    hjb_rar_errs = compute_hjb_error(model_rar, n_eval)
    weights_ts  = np.ones_like(hjb_errs) / len(hjb_errs)
    weights_rar = np.ones_like(hjb_rar_errs) / len(hjb_rar_errs)
    
    # clip the errors that are too small
    eps = 1e-10
    hjb_errs = np.clip(hjb_errs, eps, None)
    hjb_rar_errs = np.clip(hjb_rar_errs, eps, None)
    all_errs = np.concatenate([hjb_errs, hjb_rar_errs])
    # compute base-10 exponents
    min_exp = np.log10(all_errs.min())
    max_exp = np.log10(all_errs.max())

    # clip to desired range [-8, 0]
    min_exp = max(min_exp, -8)
    max_exp = min(max_exp, 0)

    bins = np.logspace(min_exp, max_exp, 50)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.hist(
        hjb_rar_errs, bins=bins, weights=weights_rar,
        alpha=1.0, color="#5492ab", edgecolor="black", label="Our Method",
    )

    ax.hist(
        hjb_errs, bins=bins, weights=weights_ts,
        alpha=0.5, color="#D9D9D9", edgecolor="black", label="Time-stepping",
    )

    ax.set_xscale("log")
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=20)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.legend(loc="upper left", frameon=False, fontsize=20)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/hjb_residuals.pdf")
    plt.close()

BASE_PARAMS = {
    "gamma" : 5.0, # Household risk aversion
    "rho" : 0.05, # Fund discount rate
}
MU_SIGS = {
    k: [0.01 * i for i in range(1, k+1)] for k in [2, 3, 50] # , 5, 10, 20, 40, 50
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

        fontsize = 20
        if n_tree == 2:
            plot_1d(model, model_rar, plot_dir)
            eval_1d(model, model_rar, plot_dir)
            fontsize = 25
        elif n_tree == 3:
            plot_residual_points(curr_base_dir, plot_dir)
            plot_2d(model_rar, plot_dir)
        plot_loss(curr_base_dir, plot_dir, fontsize)
        plot_hjb_error_distribution(model, model_rar, plot_dir)
