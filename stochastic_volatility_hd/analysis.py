import os
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as smi

from common import *


# ===========================================================================
# Evaluation helpers
# ===========================================================================
def _forward_states(model, SV_np, chunk=2000):
    """Run the fused forward on (B, n_state) numpy points, return the collected
    variable_val_dict as numpy arrays (concatenated over chunks)."""
    has_t = model.statics["has_t"]
    D_state = len(model.state_variables)
    SV_np = np.asarray(SV_np, dtype=np.float64)
    if SV_np.shape[1] < D_state:                       # append t=min_t for timestep eval
        pad = np.full((SV_np.shape[0], D_state - SV_np.shape[1]),
                      model.config.get("min_t", 0.0))
        SV_np = np.concatenate([SV_np, pad], axis=1)
    keys = ["p", "sigx_full", "sig_agg", "pi", "r", "chat", "xi_active",
            "theta_full", "hjb_expert", "hjb_household", "hjb_k",
            "goods_resid", "asset_pricing_resid", "sigtilde_full",
            "sign_k", "sigxi", "chi", "risk_premium"]
    acc = {k: [] for k in keys}
    n = SV_np.shape[0]
    for c in range(0, n, chunk):
        SV = torch.tensor(SV_np[c:c + chunk], device=model.device, dtype=torch.get_default_dtype())
        # SV.requires_grad_(True)
        for i, nm in enumerate(model.state_variables):
            model.variable_val_dict[nm] = SV[:, i:i + 1]
        model.variable_val_dict["SV"] = SV
        model.update_variables(SV)
        for k in keys:
            acc[k].append(model.variable_val_dict[k].detach().cpu().numpy())
        del SV
        gc.collect(); torch.cuda.empty_cache()
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def evaluate_slices(model, v_list, n=100, x_lo=0.05, x_hi=0.95):
    """For the 2-agent case: evaluate along x_1 in [x_lo,x_hi] at fixed v.
    Returns a dict shaped like the original ``compute_func`` output."""
    res = {"x_plot": np.linspace(x_lo, x_hi, n)}
    with model_on(model):
        for v in v_list:
            SV = np.zeros((n, 2))
            SV[:, 0] = res["x_plot"]
            SV[:, 1] = v
            out = _forward_states(model, SV)
            res[f"p_{v}"] = out["p"].reshape(-1)
            res[f"sigx_{v}"] = out["sigx_full"][:, 0].reshape(-1)
            res[f"sigsigp_{v}"] = out["sig_agg"].reshape(-1)
            res[f"signxi_{v}"] = out["pi"].reshape(-1)
            res[f"r_{v}"] = out["r"].reshape(-1)
            res[f"omega_{v}"] = (out["xi_active"][:, 0] / out["xi_active"][:, 1]).reshape(-1)
            res[f"e_hat_{v}"] = out["chat"][:, 0].reshape(-1)
            res[f"c_hat_{v}"] = out["chat"][:, 1].reshape(-1)
            res[f"risk_premium_{v}"] = out["risk_premium"].reshape(-1)
    return res


def _validation_states(model, n_samples=10000, seed=0):
    K = model.statics["K"]
    eps = 0.1 / K
    rng = np.random.default_rng(seed)
    alpha_lo = getattr(model, "share_alpha_lo", SHARE_ALPHA_LO)
    alpha_hi = getattr(model, "share_alpha_hi", SHARE_ALPHA_HI)
    shares = mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
    x_states = shares[:, :K - 1]
    vlo, vhi = V_DOMAIN
    v = vlo + (vhi - vlo) * rng.random((n_samples, 1))
    return np.concatenate([x_states, v], axis=1)


def compute_validation_losses(model, SV_val, chunk_size=2000):
    """Per-component validation losses + per-type HJB + total, on common states.

    Residual components use the same reductions as training: HJB groups are the
    mean of sum_k hjb_k**2 (== MSE of the raw residual), and the equilibrium
    constraints (goods, asset pricing) are MSE.  Expert capital shares are pinned
    analytically by the interior FOC, so capital clearing holds by construction
    and contributes no residual.
    """
    with model_on(model):
        out = _forward_states(model, SV_val, chunk_size)
    res = {
        "HJB expert": float(np.mean(np.abs(out["hjb_expert"]))),
        "HJB household": float(np.mean(np.abs(out["hjb_household"]))),
        "Goods clearing": float(np.mean(out["goods_resid"] ** 2)),
        "Asset pricing": float(np.mean(out["asset_pricing_resid"] ** 2)),
    }
    res["Total"] = sum(res.values())
    return res


def _append_random_t(model, SV_val, seed=0):
    """For a time-stepping model, append a per-point RANDOM ``t`` drawn uniformly
    on ``[min_t, max_t]`` so the validation sample covers the whole horizon
    rather than only the ``t = min_t`` slice (which ``_forward_states`` pads by
    default).  For a stationary model (no ``t`` dimension) ``SV_val`` is returned
    unchanged, so mixed tables stay consistent."""
    SV_val = np.asarray(SV_val, dtype=np.float64)
    if not model.statics.get("has_t", False):
        return SV_val
    D_state = len(model.state_variables)
    if SV_val.shape[1] >= D_state:                     # already carries t
        return SV_val
    rng = np.random.default_rng(seed)
    min_t = model.config.get("min_t", 0.0)
    max_t = model.config.get("max_t", 1.0)
    t = min_t + (max_t - min_t) * rng.random((SV_val.shape[0], 1))
    return np.concatenate([SV_val, t], axis=1)


def compute_validation_losses_random_t(model, SV_val, chunk_size=2000, seed=0):
    """Random-``t`` counterpart of ``compute_validation_losses``.

    For a time-stepping model the shared wealth-simplex+v validation states are
    given a RANDOM time coordinate ``t ~ U[min_t, max_t]`` (rather than pinned to
    ``t = min_t``), so the reported residuals reflect the solution over the full
    time horizon it was trained on.  For a stationary model this is identical to
    ``compute_validation_losses`` (no ``t`` dimension is appended).
    """
    SV_full = _append_random_t(model, SV_val, seed=seed)
    with model_on(model):
        out = _forward_states(model, SV_full, chunk_size)
    res = {
        "HJB expert": float(np.mean(np.abs(out["hjb_expert"]))),
        "HJB household": float(np.mean(np.abs(out["hjb_household"]))),
        "Goods clearing": float(np.mean(out["goods_resid"] ** 2)),
        "Asset pricing": float(np.mean(out["asset_pricing_resid"] ** 2)),
    }
    res["Total"] = sum(res.values())
    return res


def compute_theta_chat_distributions(model, SV_val, chunk_size=2000):
    with model_on(model):
        out = _forward_states(model, SV_val, chunk_size)
    gamma_vec = model.statics["gamma"].cpu().reshape(-1).numpy()
    theta_full = out["theta_full"]
    chat_full = out["chat"]
    iter_dict = {}
    for i in range(len(gamma_vec)):
        iter_dict[f"theta_{i+1}"] = theta_full[:, i]
        iter_dict[f"chat_{i+1}"] = chat_full[:, i]

    idx = pd.Index(list(range(1, len(gamma_vec)+1)), name="agent_idx")
    df = pd.DataFrame(index=idx, columns=["theta_mean", "theta_low", "theta_high", "theta_std", "chat_mean", "chat_low", "chat_high", "chat_std"])

    for i in range(1, len(gamma_vec)+1):
        for var in ["theta", "chat"]:
            y = iter_dict[f"{var}_{i}"]
            X = np.ones((len(y), 1))
            model = smi.OLS(y, X).fit()
            conf_int = model.conf_int(alpha=0.05)[0]
            df.loc[i, f"{var}_mean"] = model.params[0]
            df.loc[i, f"{var}_low"] = conf_int[0]
            df.loc[i, f"{var}_high"] = conf_int[1]
            df.loc[i, f"{var}_std"] = np.std(y)

    df = df.fillna(0.0, inplace=False)
    return df

def compare_loss_table(models, baseline_key="basic", n_samples=10000, chunk_size=2000, seed=0, cols=None,
                       compute_fn=compute_validation_losses):
    """Per-component validation-loss comparison table over ``models``.

    ``compute_fn`` selects how each model's validation loss is scored on the
    shared wealth-simplex+v states: the default ``compute_validation_losses``
    evaluates time-stepping models at ``t = min_t``; pass
    ``compute_validation_losses_random_t`` to instead score them at a random
    ``t ~ U[min_t, max_t]`` (stationary models are unaffected either way).
    """
    SV_val = _validation_states(next(iter(models.values())), n_samples=n_samples, seed=seed)
    rows = {name: compute_fn(m, SV_val, chunk_size) for name, m in models.items()}
    base = rows[baseline_key]
    # default: every reported component (Total kept last), so the columns adapt
    # to whatever residuals the case reports.
    if cols is None:
        keys = list(next(iter(rows.values())).keys())
        cols = [k for k in keys if k != "Total"] + ["Total"]
    for name, row in rows.items():
        for c in cols:
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    ordered = list(cols) + [f"{c} impr." for c in cols]
    renamed_rows = {METHOD_DISPLAY[name]: row for name, row in rows.items()}
    return pd.DataFrame.from_dict(renamed_rows, orient="index")[ordered]


def compute_welfare_equivalent_losses(models, baseline_key="basic", n_samples=10000, chunk_size=2000, seed=0):
    """Map each residual to a certainty-equivalent consumption-wealth (c/W)
    cost (units 1/time), averaged over a validation sample.

      HJB residual h_k  ->  rho * |h_k|                          (first order)

    Expert capital shares are pinned analytically by the interior FOC, so there
    is no capital-FOC residual to price.
    """
    SV_val = _validation_states(next(iter(models.values())), n_samples=n_samples, seed=seed)
    rows = {}
    for name, model in models.items():
        st = model.statics
        rho = st["rho"]
        with model_on(model):
            out = _forward_states(model, SV_val, chunk_size)
        hjb_k = out["hjb_k"]
        hjb_we = float(np.mean(rho * np.abs(hjb_k)))
        rows[name] = {"HJB (c/W)": hjb_we, "total (c/W)": hjb_we}
    base = rows[baseline_key]
    for name, row in rows.items():
        for c in list(base.keys()):
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    abs_cols = ["HJB (c/W)", "total (c/W)"]
    ordered = abs_cols + [f"{c} impr." for c in abs_cols]
    renamed_rows = {METHOD_DISPLAY[name]: row for name, row in rows.items()}
    return pd.DataFrame.from_dict(renamed_rows, orient="index")[ordered]



def fd_v_slices(fd_dict, var="omega"):
    """v-values for which the FD solution stored a slice of ``var``."""
    keys = fd_dict.files if hasattr(fd_dict, "files") else list(fd_dict.keys())
    prefix = f"{var}_"
    vs = []
    for k in keys:
        if k.startswith(prefix):
            try:
                vs.append(float(k[len(prefix):]))
            except ValueError:
                pass
    return sorted(vs)


def compute_fd_errors(method_dict, fd_dict, v_list, vars):
    """MSE and relative-MAE of one method's slices vs the FD solution, averaged
    over the ``v_list`` slices (mirrors ``stochastic_volatility_model``)."""
    x_nn = np.asarray(method_dict["x_plot"])
    x_fd = np.asarray(fd_dict["x_plot"])
    same_grid = len(x_nn) == len(x_fd) and np.allclose(x_nn, x_fd)
    mses, rel_maes = {}, {}
    for var in vars:
        tot_sq = tot_abs = tot_ref = 0.0
        for v in v_list:
            nn = np.asarray(method_dict[f"{var}_{v}"])
            fd = np.asarray(fd_dict[f"{var}_{v}"])
            if not same_grid:                       # align FD onto the NN x-grid
                fd = np.interp(x_nn, x_fd, fd)
            tot_sq = tot_sq + (fd - nn) ** 2
            tot_abs = tot_abs + np.abs(fd - nn)
            tot_ref = tot_ref + np.abs(fd)
        mses[var] = float(np.mean(tot_sq) / len(v_list))
        rel_maes[var] = float(np.mean(tot_abs) / np.mean(tot_ref))
    return mses, rel_maes


def compare_fd_table(method_dicts, fd_dict, v_list, out_dir,
                     vars=tuple(FD_TABLE_VARS), prefix="fd_error"):
    """Per-method error tables (MSE and relative-MAE) vs the Di Tella FD
    solution for ``vars`` (default: omega, e_hat, c_hat, risk_premium).  Rows are
    the trained methods, columns the objects.  Writes ``{prefix}_mse.{csv,tex}``
    and ``{prefix}_rel_mae.{csv,tex}``; returns ``(mse_df, mae_df)``."""
    rows_mse, rows_mae = {}, {}
    for name, md in method_dicts.items():
        mses, maes = compute_fd_errors(md, fd_dict, v_list, vars)
        label = METHOD_DISPLAY.get(name, name)
        rows_mse[label] = mses
        rows_mae[label] = maes
    cols = list(vars)
    mse_df = pd.DataFrame.from_dict(rows_mse, orient="index")[cols]
    mae_df = pd.DataFrame.from_dict(rows_mae, orient="index")[cols]
    col_rename = {v: FD_TABLE_VARS.get(v, v) for v in cols}
    col_fmt = "l" + "c" * len(cols)
    for df, kind, fmt, scale in [(mse_df, "mse", format_sci, 1.0),
                                 (mae_df, "rel_mae", format_pct, 100.0)]:
        df.to_csv(os.path.join(out_dir, f"{prefix}_{kind}.csv"))
        disp = (df * scale).rename(columns=col_rename)
        for c in disp.columns:
            disp[c] = disp[c].apply(fmt)
        with open(os.path.join(out_dir, f"{prefix}_{kind}.tex"), "w") as f:
            f.write(disp.style.to_latex(column_format=col_fmt, hrules=True))
    return mse_df, mae_df



def plot_slice_comparison(method_dicts, fd_dict, v_list, out_dir):
    """2-D slice overlay: each method vs the Di Tella finite-difference solution."""
    os.makedirs(out_dir, exist_ok=True)
    for var, parg in SLICE_PLOT_ARGS.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        if fd_dict is not None:
            xfd = fd_dict["x_plot"]
            for i, v in enumerate(v_list):
                key = f"{var}_{v}"
                if key in fd_dict:
                    ax.plot(xfd, fd_dict[key], ls="-.", color=SLICE_COLORS[i], marker="x", markevery=3, label=f"FD v={v}")
        for mi, (name, md) in enumerate(method_dicts.items()):
            xp = md["x_plot"]
            ls, mk = METHOD_PLOT_STYLES[mi % len(METHOD_PLOT_STYLES)]
            for i, v in enumerate(v_list):
                ax.plot(xp, md[f"{var}_{v}"], ls=ls, marker=mk, markevery=8,
                        markersize=6, color=SLICE_COLORS[i],
                        alpha=0.85, label=f"{name} v={v}")
        ax.set_xlabel("Expert wealth share x"); ax.set_ylabel(parg["ylabel"])
        ax.legend(fontsize=9, ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"slice_{var}.pdf"))
        plt.close(fig)


def plot_loss_decay(model_paths, out_dir, timestepping_map,
                    targets=(("hjbeq_expert", "HJB (experts)"),
                             ("hjbeq_household", "HJB (households)"),
                             ("total_loss", "Total"))):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(targets), figsize=(6.2 * len(targets), 4.8), squeeze=False)
    axes = axes[0]
    colors = plt.get_cmap("tab10")
    for c, (name, path) in enumerate(model_paths.items()):
        ts = timestepping_map.get(name, False)
        fname = "model_global_min_loss.csv" if ts else "model_min_loss.csv"
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df = df.iloc[1:] if len(df) > 1 else df
        xcol = "epoch" if "epoch" in df.columns else df.columns[0]
        if "hjbeq_expert" not in df.columns:
            df["hjbeq_expert"] = df[[col for col in df.columns if col.startswith("hjbeq_expert_")]].sum(axis=1)
            df["hjbeq_household"] = df[[col for col in df.columns if col.startswith("hjbeq_household_")]].sum(axis=1)
        for j, (col, _) in enumerate(targets):
            if col not in df.columns:
                continue
            axes[j].semilogy(df[xcol], df[col].cummin(), color=colors(c), lw=2, label=name)
    for j, (_, label) in enumerate(targets):
        axes[j].set_xlabel("epoch"); axes[j].set_ylabel(f"{label} loss")
        axes[j].legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_decay.pdf"))
    plt.close(fig)


def plot_loss_weights(model_path, out_dir, file_name="loss_weight.pdf", timestepping=False):
    mapping = {"endogeq_goods": "Goods clearing", "endogeq_capital": "Capital clearing",
               "hjbeq_expert": "HJB experts", "hjbeq_expert_0": "HJB experts", 
               "hjbeq_household": "HJB households", "hjbeq_household_1": "HJB households",
               "endogeq_asset_pricing": "Asset pricing",
               "endogeq_sig_clearning": "Market Clearing",}
    if timestepping:
        wdir = os.path.join(model_path, "loss_weight_logs")
        if not os.path.isdir(wdir):
            return
        files = sorted(f for f in os.listdir(wdir) if f.endswith(".csv"))
        if not files:
            return
        df = pd.read_csv(os.path.join(wdir, files[0]))
    else:
        fpath = os.path.join(model_path, "model_loss_weight.csv")
        if not os.path.exists(fpath):
            return
        df = pd.read_csv(fpath)
    expert_cols = sorted([col for col in df.columns if col.startswith("hjbeq_expert")])
    if len(expert_cols) >= 2:
        mapping[expert_cols[0]] = "HJB expert (low RA)"
        mapping[expert_cols[-1]] = "HJB expert (high RA)"
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.get_cmap("tab10")
    for i, (col, lab) in enumerate(mapping.items()):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=colors(i), label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name), bbox_inches="tight")
    plt.close(fig)


def plot_rar_anchors(model_path, K, out_dir, file_name="rar_anchors.pdf", timestepping=False):
    """Scatter of RAR anchor points in (sum(xi), v)."""
    if timestepping:
        adir = os.path.join(model_path, "anchor_points")
        if not os.path.isdir(adir):
            return
        files = sorted(f for f in os.listdir(adir) if f.endswith(".npy"))
        if not files:
            return
        n = len(files)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        cmap = plt.get_cmap("viridis")
        # map outer-loop index (1..n) onto the colormap; earliest = dark, latest = bright
        norm = mpl.colors.Normalize(vmin=1, vmax=max(n, 2))
        for i, file in enumerate(files):
            anchors = np.load(os.path.join(adir, file))
            x_sum = np.sum(anchors[:, :K-1], axis=1)
            sc = ax.scatter(x_sum, anchors[:, K-1], c=cmap(norm(i + 1)), cmap="viridis", s=12, alpha=0.7)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Outer Loop (earliest \u2192 latest)", fontsize=20)
        cbar.locator = mpl.ticker.MaxNLocator(integer=True)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=16)
    
        ax.set_xlabel("$x$ sum", fontsize=20)
        ax.set_ylabel("$v$", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, file_name))
        plt.close()
    else:
        apath = os.path.join(model_path, "model_anchor_points.npy")
        if not os.path.exists(apath):
            return
        anchors = np.load(apath)
        if anchors.ndim != 2 or anchors.shape[1] < 2:
            return
        x_sum = np.sum(anchors[:, :K-1], axis=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x_sum, anchors[:, K-1], c=np.arange(len(anchors)),
                        cmap="viridis", s=12, alpha=0.7)
        fig.colorbar(sc, ax=ax, label="anchor index (old -> new)")
        ax.set_xlabel("$x_1$ (expert share)"); ax.set_ylabel("$v$")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, file_name), bbox_inches="tight")
        plt.close(fig)


def plot_aggregate_scatter(model, out_dir, file_name="aggregate_scatter.pdf", n_samples=4000, chunk_size=2000, v_fixed=0.25, seed=0):
    """For K>2 cases, scatter p / risk premium / omega against two summaries of
    the wealth distribution at fixed v:

      * row 1 -- the TOTAL expert wealth share (sum_{i in experts} x_i);
      * row 2 -- the Herfindahl concentration index H = sum_k x_k^2 (1/K when
        wealth is split equally, ->1 when one agent owns everything).

    A single x-value maps to many states (the wealth can be distributed
    differently among the agents), so each panel is a genuine scatter rather
    than a 1-D slice.  Shares are drawn with the same Dirichlet-alpha mixture as
    training, so concentrated states (large H, small expert share) are covered.
    omega is the analogue of the 2-agent xi_E/xi_H: the mean expert
    value-multiplier over the mean household value-multiplier.
    """
    K = model.statics["K"]
    e_idx = model.statics["expert_idx"]
    h_idx = model.statics["household_idx"]

    rng = np.random.default_rng(seed)
    eps = 0.1 / K
    alpha_lo = getattr(model, "share_alpha_lo", SHARE_ALPHA_LO)
    alpha_hi = getattr(model, "share_alpha_hi", SHARE_ALPHA_HI)
    shares = mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
    SV_np = np.concatenate([shares[:, :K - 1], np.full((n_samples, 1), v_fixed)], axis=1)

    with model_on(model):
        out = _forward_states(model, SV_np, chunk_size)
    expert_share = shares[:, e_idx].sum(axis=1)
    herfindahl = (shares ** 2).sum(axis=1)
    xi = out["xi_active"]
    omega = xi[:, e_idx].mean(axis=1) / xi[:, h_idx].mean(axis=1)

    panels = [("p", out["p"].reshape(-1), "Capital price $p$"),
              ("risk_premium", out["risk_premium"].reshape(-1), "Risk premium $\\pi(\\sigma+\\sigma_p)$"),
              ("omega", omega, "$\\Omega=\\bar\\xi_E/\\bar\\xi_H$")]
    x_axes = [(expert_share, "Total expert wealth share $\\sum_{i\\in E} x_i$"),
              (herfindahl, "Wealth concentration $H=\\sum_k x_k^2$")]

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (xv, xlabel) in enumerate(x_axes):
        for col, (_, yv, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(xv, yv, s=6, alpha=0.35, edgecolors="none")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    fig.suptitle(f"v={v_fixed}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)


def plot_theta_chat_histogram(model, out_dir, seed=0, chunk_size=2000):
    SV_val = _validation_states(model, n_samples=10000, seed=seed)
    generated_df = compute_theta_chat_distributions(model, SV_val, chunk_size=chunk_size)
    os.makedirs(out_dir, exist_ok=True)
    for var in ["theta", "chat"]:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8))
        yerr = [
            generated_df[f"{var}_mean"] - generated_df[f"{var}_low"],    # lower
            generated_df[f"{var}_high"] - generated_df[f"{var}_mean"]    # upper
        ]
        ax.bar(
            generated_df.index,
            generated_df[f"{var}_mean"],
            yerr=yerr,
            capsize=6,
            width=0.6,
            linewidth=1.5,
            edgecolor="black",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{var}_histogram.pdf"))
        plt.close(fig)


def select_plot_methods(models, loss_df=None, welfare_df=None,
                        baseline_key="basic",
                        val_improvement_col="Total impr.",
                        welfare_improvement_col="total (c/W) impr."):
    """Return ``{name: model}`` with the baseline plus the method(s) showing the
    biggest improvement over baseline -- by validation total loss and by total
    welfare-equivalent loss (the two may coincide)."""
    mapping = {v: k for k, v in METHOD_DISPLAY.items()}
    ordered = [baseline_key] if baseline_key in models else list(models)[:1]
    for df, col in [(loss_df, val_improvement_col),
                    (welfare_df, welfare_improvement_col)]:
        if df is None or col not in df.columns:
            continue
        cand = df.drop(index=baseline_key, errors="ignore")[col].dropna()
        cand = cand[np.isfinite(cand)]
        if len(cand) == 0:
            continue
        best = cand.idxmax()
        best = mapping.get(best, best)
        if best in models and best not in ordered:
            ordered.append(best)
    return {k: models[k] for k in ordered if k in models}