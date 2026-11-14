"""2-tree (1D) check: Basic vs. time-stepping + RAR at the same training budget.

* ``basic``        -- stationary ``PDEModel`` on z (no time dimension), trained
                      for the TOTAL number of epochs the time-stepping run uses
                      (sum over outer iterations of the library schedule
                      max(int(num_inner / sqrt(i + 1)), min_inner)).
* ``timestep_rar`` -- ``tree_model_main.get_model(2, ..., rar=True)``.

Both use the same network, batch size, learning rate, optimizer and seed, and
the economic parameters of the finite-difference reference
``models/2trees_solution-raw.csv`` (read from the file itself).

Outputs (``models/tree_2_basic_comparison``):
  * ``basic/``, ``timestep_rar/``  -- checkpoints (trained only if missing);
  * ``tree2_basic_vs_timestep.{csv,tex}`` -- final validation loss (PDE
    residuals only, time-boundary terms excluded) and MSE vs. the FD solution;
  * ``plots/{k1,k2,q1,q2,r,zeta1,zeta2}.pdf`` -- FD vs. Basic vs. Our Method.

Run from this folder:  uv run python tree2_basic_vs_timestep.py
"""

import gc
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import OptimizerType, PDEModel, SamplingMethod, set_seeds

import tree_model_main as tm

N_TREE = 2
SEED = 0
FD_PATH = "models/2trees_solution-raw.csv"
OUT_DIR = "models/tree_2_basic_comparison"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
N_VALIDATION = 5_000

VARS = ("k1", "k2", "q1", "q2", "r", "zeta1", "zeta2")
Y_LABELS = {
    "k1": r"$\kappa_1$", "k2": r"$\kappa_2$",
    "q1": r"$q_1$", "q2": r"$q_2$",
    "r": r"$r$",
    "zeta1": r"$\zeta_1$", "zeta2": r"$\zeta_2$",
}

# Paper style shared with stochastic_volatility_hd (common.py / analysis.py).
PAPER_FIGSIZE = (8, 6)
PAPER_FONT_SIZE = 20
FD_STYLE = {"label": "FD", "color": "#000000", "linestyle": "-.",
            "marker": "x", "markevery": 10}
METHOD_STYLES = {
    "basic": {"label": "Basic", "color": "#D9D9D9", "linestyle": "--",
              "marker": "o", "markevery": 8},
    "timestep_rar": {"label": "Time-stepping + RAR", "color": "#5492ab",
                     "linestyle": "-"},
}
METHOD_DISPLAY = {name: style["label"] for name, style in METHOD_STYLES.items()}

plt.rcParams.update({
    "font.size": PAPER_FONT_SIZE,
    "axes.labelsize": PAPER_FONT_SIZE,
    "xtick.labelsize": PAPER_FONT_SIZE,
    "ytick.labelsize": PAPER_FONT_SIZE,
    "legend.fontsize": PAPER_FONT_SIZE,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Stationary (no time column) counterparts of tree_model_main's q functions,
# which drop the last SV column as time.
# ---------------------------------------------------------------------------
def compute_q_stationary(SV, compute_k):
    z_last = 1 - torch.sum(SV, dim=-1).unsqueeze(-1)
    z_all = torch.cat([SV, z_last], dim=-1)
    return z_all / compute_k(SV)


def compute_qz_stationary(SV, compute_k):
    return torch.vmap(torch.func.jacrev(lambda SV: compute_q_stationary(SV, compute_k)))(SV)


def compute_qzz_stationary(SV, compute_k):
    return torch.vmap(torch.func.hessian(lambda SV: compute_q_stationary(SV, compute_k)))(SV)


def load_fd():
    df = pd.read_csv(FD_PATH).drop_duplicates(subset=["z"], keep="first")
    return df.sort_values("z").reset_index(drop=True)


def fd_params(fd, device):
    row = fd.iloc[0]
    mu = torch.tensor([[row["mu_y1"], row["mu_y2"]]], dtype=torch.float32, device=device)
    sig = torch.tensor([[row["sig_y1"], row["sig_y2"]]], dtype=torch.float32, device=device)
    return {"gamma": float(row["gamma"]), "rho": float(row["rho"]),
            "mu_ys": mu, "sig_ys": sig}


def timestep_total_epochs(model):
    return sum(
        max(int(model.num_inner_iterations / np.sqrt(i + 1)), model.min_inner_iterations)
        for i in range(model.config["num_outer_iterations"])
    )


def get_basic_model(params, n_epochs, batch_size, lr, seed=SEED):
    """``tree_model_main.get_model`` for n_tree=2 without the time dimension."""
    set_seeds(seed)
    model = PDEModel("tree2_basic", config={
        "batch_size": batch_size, "num_epochs": n_epochs,
        "sampling_method": SamplingMethod.UniformRandom,
        "lr": lr, "optimizer_type": OptimizerType.Adam,
    })
    model.set_state(["z1"], {"z1": [0.01, 0.99]})
    model.add_params(params)
    model.add_endogs(["k"], configs={
        "k": {"positive": True, "hidden_units": [80] * 4, "output_size": N_TREE},
    })
    for fn in (compute_q_stationary, compute_qz_stationary, compute_qzz_stationary,
               tm.compute_mu_z_geos, tm.compute_sig_z_geos, tm.compute_mu_qs,
               tm.compute_sig_qs, tm.compute_r, tm.compute_hjb_kappa,
               tm.compute_consistency_kappa, tm.compute_kappa_penalization):
        model.register_function(fn)
    model.add_equations([
        "z = SV",
        "z_last = 1 - torch.sum(z, dim=1).unsqueeze(1)",
        "z_all = torch.cat([z, z_last], dim=1)",
        "dk_dz = k_Jac",
        "dk_dt = torch.zeros_like(k)",
        "dk_dzz = k_Hess",
        "q = compute_q_stationary(SV, compute_k)",
        "dq_dz = compute_qz_stationary(SV, compute_k)",
        "dq_dzz = compute_qzz_stationary(SV, compute_k)",
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


def train_or_load(model, out_dir):
    if not os.path.exists(os.path.join(out_dir, "model.pt")):
        model.train_model(out_dir, "model.pt", True)
    model.load_model(torch.load(os.path.join(out_dir, "model_best.pt"),
                                weights_only=False, map_location=model.device))
    gc.collect()
    torch.cuda.empty_cache()
    return model


def _with_time(model, z, timestep):
    if not timestep:
        return z
    return torch.cat([z, torch.zeros((z.shape[0], 1), device=model.device)], dim=1)


def validation_loss(model, timestep, n_samples=N_VALIDATION, seed=SEED):
    """Weighted PDE-residual loss (HJB + consistency + kappa penalization) on a
    seeded sample, scored at t = 0 for time-stepping.  The time-boundary
    matching terms only exist for time-stepping, so they are excluded."""
    set_seeds(seed)
    eps = 0.01
    shares = torch.distributions.Dirichlet(
        torch.ones(N_TREE, device=model.device)).sample((n_samples,))
    shares = eps + (1.0 - N_TREE * eps) * shares
    SV_all = _with_time(model, shares[:, :-1], timestep)
    validate = (model._PDEModelTimeStep__validation if timestep
                else model._PDEModel__validation)
    chunk = model.batch_size
    sums = {}
    for c in range(0, n_samples, chunk):
        SV = SV_all[c:c + chunk]
        loss_dict = validate(SV)
        for label, loss in loss_dict.items():
            if label == "total_loss" or "time_boundary" in label:
                continue
            val = float(torch.nanmean(model.loss_weight_dict[label] * loss).detach().cpu())
            sums[label] = sums.get(label, 0.0) + val * SV.shape[0]
        del SV, loss_dict
    comps = {label: total / n_samples for label, total in sums.items()}
    comps["total"] = sum(comps.values())
    return comps


def evaluate_on_grid(model, z_grid, timestep):
    z = torch.tensor(z_grid, dtype=torch.float32, device=model.device).reshape(-1, 1)
    SV = _with_time(model, z, timestep)
    SV.requires_grad_(True)
    for i, sv_name in enumerate(model.state_variables):
        model.variable_val_dict[sv_name] = SV[:, i:i + 1]
    model.variable_val_dict["SV"] = SV
    model.update_variables(SV)
    vd = model.variable_val_dict
    to_np = lambda name: vd[name].detach().cpu().numpy()
    k, q, zetas = to_np("k"), to_np("q"), to_np("zetas")
    return {
        "k1": k[:, 0], "k2": k[:, 1], "q1": q[:, 0], "q2": q[:, 1],
        "r": to_np("r").reshape(-1), "zeta1": zetas[:, 0], "zeta2": zetas[:, 1],
    }


def write_table(rows, out_dir):
    raw = pd.DataFrame.from_dict(rows, orient="index")
    raw.to_csv(os.path.join(out_dir, "tree2_basic_vs_timestep.csv"))

    disp = pd.DataFrame(index=raw.index)
    disp["Epochs"] = raw["epochs"].map(lambda x: f"{int(x):,}")
    disp["Validation loss"] = raw["validation_loss"].map(tm.format_sci)
    for var in VARS:
        disp[f"MSE {Y_LABELS[var]}"] = raw[f"mse_{var}"].map(tm.format_sci)
    disp["MSE total"] = raw["mse_total"].map(tm.format_sci)
    with open(os.path.join(out_dir, "tree2_basic_vs_timestep.tex"), "w") as f:
        f.write(disp.style.to_latex(column_format="l" + "c" * len(disp.columns), hrules=True))
    print(raw.to_string(float_format=lambda x: f"{x:.3e}"))


def plot_solutions(fd, solutions, plot_dir):
    x_fd = fd["z"].values
    for var in VARS:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        ax.plot(x_fd, fd[var].values, **FD_STYLE)
        for name, style in METHOD_STYLES.items():
            ax.plot(x_fd, solutions[name][var], **style)
        ax.set_xlabel("Wealth share $z$")
        ax.set_ylabel(Y_LABELS[var])
        ax.tick_params(axis="both", which="major", labelsize=PAPER_FONT_SIZE)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{var}.pdf"))
        plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(PLOT_DIR, exist_ok=True)
    fd = load_fd()
    params = fd_params(fd, device)

    ts_model = tm.get_model(N_TREE, params, seed=SEED, rar=True)
    epochs = timestep_total_epochs(ts_model)
    print(f"time-stepping total epochs = {epochs}; Basic uses the same")
    ts_model = train_or_load(ts_model, os.path.join(OUT_DIR, "timestep_rar"))

    basic = get_basic_model(params, epochs, ts_model.batch_size, ts_model.lr)
    basic = train_or_load(basic, os.path.join(OUT_DIR, "basic"))

    with open(os.path.join(OUT_DIR, "setup.json"), "w", encoding="utf-8") as f:
        json.dump({
            "epochs": epochs, "batch_size": ts_model.batch_size, "lr": ts_model.lr,
            "gamma": params["gamma"], "rho": params["rho"],
            "mu_ys": params["mu_ys"].reshape(-1).tolist(),
            "sig_ys": params["sig_ys"].reshape(-1).tolist(),
            "fd_path": FD_PATH,
        }, f, indent=2)

    models = {"basic": (basic, False), "timestep_rar": (ts_model, True)}
    solutions, rows = {}, {}
    for name, (model, timestep) in models.items():
        val = validation_loss(model, timestep)
        sol = evaluate_on_grid(model, fd["z"].values, timestep)
        solutions[name] = sol
        sq = {var: (sol[var] - fd[var].values) ** 2 for var in VARS}
        row = {"epochs": epochs, "validation_loss": val["total"]}
        row.update({f"val_{k}": v for k, v in val.items() if k != "total"})
        row.update({f"mse_{var}": float(np.mean(sq[var])) for var in VARS})
        row["mse_total"] = float(np.mean([sq[var] for var in VARS]))
        rows[METHOD_DISPLAY[name]] = row

    write_table(rows, OUT_DIR)
    plot_solutions(fd, solutions, PLOT_DIR)
    print(f"outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
