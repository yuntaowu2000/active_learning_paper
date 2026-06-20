"""Forward-simulate the N-agent complete-market (GP) economy.

The model solved in ``gp_n_agents_NN.py`` lives on the wealth-share simplex
``sum_i x_i = 1`` with N active agents.  The state is the ``N-1`` FREE shares
``x_1, ..., x_{N-1}`` (the dependent share is ``x_N = 1 - sum_{j<N} x_j``).
Each free share follows

    dx_j = mux_j dt + sigx_j . dW          (j = 1..N-1)

where ``dW`` is the *aggregate* Brownian shock (the model's ``sigma`` has two
components, so ``sigx_j`` is a 2-vector loading on a 2-D shock; the GP baseline
uses ``sigma = (0.0357, 0)`` so the economy is effectively single-shock, but we
step the full 2-D structure for generality).  The trained model already exposes
the absolute share drift ``mux`` (B, N-1) and diffusion ``sigx_active``
(B, N-1, 2), so we step them directly with Euler-Maruyama.

Risk premium
------------
The (aggregate) equity risk premium is ``pi = q |sigR|^2`` -- already an
equilibrium object exposed by the model (variable ``pi``).  This *is* the risk
premium; ``eta = q |sigR|`` is the associated price of risk (Sharpe ratio).
Each agent's wealth risk premium is its portfolio share times the market
premium, ``alpha_i * pi``.

The same simulation runs on a trained neural-network model (``NNEconomy``) or on
the Chebyshev finite-difference solution (``NumericalEconomy``, 2-agent) read
from ``cheb_solution.csv`` -- both expose a common
``drift_diffusion`` / ``premium`` interface.

Usage::

    # trained NN
    python gp_n_agents_simulate.py --source nn --case asym2 --config timestep \
        --base-dir ./models/GP_NN_NAgents_faster_64bits --float64

    # Chebyshev numerical solution (2-agent cases)
    python gp_n_agents_simulate.py --source numerical --case asym2
"""

import argparse
import os
from typing import Union

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from gp_n_agents_NN import (
    CONFIGS, get_model, make_params, make_case,
    PDEModelNAgents, PDEModelTimeStepNAgents,
)

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(base_dir, case, config="timestep", width=30, layers=4):
    """Reconstruct and load a trained checkpoint (``model_best.pt``)."""
    ts, rar, lb = CONFIGS[config]
    n, gamma_active, psi_active, alpha_caps, suffix = make_case(case)
    mpath = os.path.join(base_dir, suffix, config)
    if not os.path.exists(os.path.join(mpath, "model_best.pt")):
        raise FileNotFoundError(
            f"no trained checkpoint at {mpath}/model_best.pt -- train first.")
    params = make_params(gamma_active, psi_active_vec=psi_active)
    model = get_model(
        mpath, n, params,
        model_size=[width] * layers,
        alpha_caps=alpha_caps,
        timestepping=ts, rar=rar, loss_balancing=lb,
        train=False,
    )
    return model


# ---------------------------------------------------------------------------
# Economy adapters: a common (drift_diffusion, premium) interface backed either
# by the trained NN or by the Chebyshev numerical solution.
# ---------------------------------------------------------------------------
class NNEconomy:
    """Drift/diffusion + risk premia from a trained ``gp_n_agents_NN`` model."""

    source = "nn"

    def __init__(self, model):
        self.model: Union[PDEModelNAgents, PDEModelTimeStepNAgents] = model
        self.K = int(getattr(model, "n_agents", None) or (model.n_share + 1))
        self.n_share = int(getattr(model, "n_share", self.K - 1))
        self.has_t = isinstance(model, PDEModelTimeStepNAgents) or ("t" in model.state_variables)
        self.min_t = float(model.config.get("min_t", 0.0))
        self.n_shock = 2                         # sigma is a 2-vector
        self.share_lo, self.share_hi = 0.02, 0.98
        model.set_all_model_eval()

    def _forward(self, x_states):
        model = self.model
        P = x_states.shape[0]
        dtype = torch.get_default_dtype()
        SV_np = np.asarray(x_states, dtype=np.float64)
        if self.has_t:                            # pad pseudo-time t = min_t
            SV_np = np.concatenate([SV_np, np.full((P, 1), self.min_t)], axis=1)
        SV = torch.tensor(SV_np, device=model.device, dtype=dtype)
        SV.requires_grad_(True)                   # update_variables autodiffs
        for i, nm in enumerate(model.state_variables):
            model.variable_val_dict[nm] = SV[:, i:i + 1]
        model.variable_val_dict["SV"] = SV
        model.update_variables(SV)
        return model.variable_val_dict

    def drift_diffusion(self, x_states):
        """(mu_x [P, N-1], sig_x [P, N-1, 2]) at the given free shares."""
        vd = self._forward(x_states)
        mu_x = vd["mux"].detach().cpu().numpy()                     # (P, N-1)
        sig_x = vd["sigx_active"].detach().cpu().numpy()            # (P, N-1, 2)
        return mu_x, sig_x

    def premium(self, x_states):
        """Market risk premium pi (+ price of risk, riskfree, per-agent rp)."""
        vd = self._forward(x_states)
        pi = vd["pi"].detach().cpu().numpy().reshape(-1)
        eta = vd["eta"].detach().cpu().numpy().reshape(-1)
        r = vd["r"].detach().cpu().numpy().reshape(-1)
        sigR = vd["sigR_norm"].detach().cpu().numpy().reshape(-1)
        alpha = vd["alpha_active"].detach().cpu().numpy()           # (P, N)
        return dict(pi=pi, eta=eta, r=r, sigR=sigR, alpha=alpha,
                    agent_rp=alpha * pi[:, None], risk_premium=pi)


class NumericalEconomy:
    """Same interface, backed by the Chebyshev 2-agent solution
    (``cheb_solution.csv``) through 1-D interpolation in the free share x_1."""

    source = "numerical"

    def __init__(self, df):
        import pandas as pd  # noqa: F401  (df is already a DataFrame)
        df = df.sort_values("x").reset_index(drop=True)
        self.df = df
        self.K = 2
        self.n_share = 1
        self.n_shock = 1
        self._x = df["x"].to_numpy()
        self.share_lo, self.share_hi = float(self._x.min()), float(self._x.max())
        required = ["mux", "sigma_x", "pi", "eta", "r", "sigma_R",
                    "alpha_1", "alpha_2"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"cheb_solution.csv is missing columns {missing}; re-run "
                f"gp_n_agents_numerical.py to regenerate it with mux/sigma_x.")

    def _interp(self, col, xq):
        return np.interp(xq, self._x, self.df[col].to_numpy())

    def drift_diffusion(self, x_states):
        x = x_states[:, 0]
        mu_x = self._interp("mux", x)[:, None]                       # (P, 1)
        sig_x = self._interp("sigma_x", x)[:, None, None]            # (P, 1, 1)
        return mu_x, sig_x

    def premium(self, x_states):
        x = x_states[:, 0]
        pi = self._interp("pi", x)
        eta = self._interp("eta", x)
        r = self._interp("r", x)
        sigR = np.abs(self._interp("sigma_R", x))
        alpha = np.column_stack([self._interp("alpha_1", x),
                                 self._interp("alpha_2", x)])         # (P, 2)
        return dict(pi=pi, eta=eta, r=r, sigR=sigR, alpha=alpha,
                    agent_rp=alpha * pi[:, None], risk_premium=pi)


def build_numerical_economy(case, out_base="./models/gp_numerical_cheb",
                            csv_path=None):
    """Load (or solve) the Chebyshev solution for ``case`` and wrap it."""
    import pandas as pd
    n, *_ , suffix = make_case(case)
    if n != 2:
        raise ValueError(f"numerical (Chebyshev) solution is 2-agent only; "
                         f"case {case!r} has N={n}.")
    csv_path = csv_path or os.path.join(out_base, suffix, "cheb_solution.csv")

    def _needs_solve():
        if not os.path.exists(csv_path):
            return True
        cols = pd.read_csv(csv_path, nrows=1).columns
        return "mux" not in cols or "sigma_x" not in cols   # stale (pre-export) file

    if _needs_solve():
        # solve on demand (also regenerates stale CSVs lacking mux/sigma_x)
        from gp_n_agents_numerical import run_case
        run_case(case, out_base=out_base)
    df = pd.read_csv(csv_path)
    return NumericalEconomy(df)


# ---------------------------------------------------------------------------
# Domain projection (keep states where the solution is valid)
# ---------------------------------------------------------------------------
def _project(economy, x_states):
    """Clip free shares into the valid box, keeping the anchor share feasible."""
    lo, hi = economy.share_lo, economy.share_hi
    x = np.clip(x_states, lo, hi)
    s = x.sum(axis=1, keepdims=True)
    too_big = (s > 1.0 - lo).squeeze(-1)
    if np.any(too_big):
        scale = (1.0 - lo) / s[too_big]
        x[too_big] = np.maximum(x[too_big] * scale, lo)
    return x


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(economy, n_paths=100, years=500.0, dt=0.05,
             x0=None, seed=0, store_every=1):
    """Euler-Maruyama forward simulation of the free wealth shares.

    Returns dict with ``x_hist`` (T, P, N-1) and ``t`` (T,).  One shared
    Gaussian shock vector per (path, step) drives all shares.
    """
    rng = np.random.default_rng(seed)
    K, n_share = economy.K, economy.n_share
    n_steps = int(round(years / dt))
    sqrt_dt = np.sqrt(dt)

    if x0 is None:
        x0_vec = np.full(n_share, 1.0 / K)        # equal split
    else:
        x0_vec = np.full(n_share, float(x0))
        if x0_vec.sum() >= 1.0 - economy.share_lo:
            x0_vec = np.full(n_share, 1.0 / K)
            print(f"[simulate] x0={x0} infeasible for K={K}; using equal 1/K.")
    x = np.tile(x0_vec, (n_paths, 1)).astype(np.float64)             # (P, N-1)
    x = _project(economy, x)

    n_store = n_steps // store_every + 1
    x_hist = np.empty((n_store, n_paths, n_share))
    t_hist = np.empty(n_store)
    x_hist[0], t_hist[0] = x, 0.0
    si = 1

    for step in range(1, n_steps + 1):
        mu_x, sig_x = economy.drift_diffusion(x)
        dW = rng.standard_normal((n_paths, economy.n_shock)) * sqrt_dt
        x = x + mu_x * dt + np.einsum("pij,pj->pi", sig_x, dW)
        x = _project(economy, x)
        if step % store_every == 0:
            x_hist[si], t_hist[si] = x, step * dt
            si += 1
        if step % max(1, n_steps // 20) == 0:
            print(f"[simulate] step {step:>6d}/{n_steps}  t={step*dt:7.1f}y  "
                  f"mean sum(free shares)={x.sum(axis=1).mean():.3f}")

    return {"x_hist": x_hist[:si], "t": t_hist[:si]}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _full_shares(x_states):
    """Append the dependent share x_N = 1 - sum(free); (..., N-1) -> (..., N)."""
    dep = 1.0 - x_states.sum(axis=-1, keepdims=True)
    return np.concatenate([x_states, dep], axis=-1)


def analyze(economy, sim, out_dir, burn_in_frac=0.2):
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd

    x_hist, t = sim["x_hist"], sim["t"]
    burn = int(len(t) * burn_in_frac)
    x_pool = x_hist[burn:]                                  # (Tb, P, N-1)
    flat = x_pool.reshape(-1, x_pool.shape[-1])             # (M, N-1)
    full = _full_shares(flat)                               # (M, N)
    K = economy.K

    # ---- marginal distributions of every agent share ----------------------
    ncol = min(K, 5)
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i in range(K):
        ax = axes[i // ncol][i % ncol]
        ax.hist(full[:, i], bins=60, density=True, color=f"C{i % 10}", alpha=0.8)
        ax.set_title(f"$x_{{{i+1}}}$  (mean={full[:, i].mean():.3f})")
        ax.set_xlabel(f"$x_{{{i+1}}}$")
    for j in range(K, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Stationary marginal wealth-share distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distribution.pdf"))
    plt.close(fig)

    # ---- joint (x_1, x_2) when there are >= 2 free shares -----------------
    if economy.n_share >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        h = ax.hist2d(flat[:, 0], flat[:, 1], bins=60, density=True, cmap="viridis")
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title("Joint stationary distribution $(x_1, x_2)$")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "joint_x1_x2.pdf"))
        plt.close(fig)

    # ---- a few sample paths of x_1 ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for pth in range(min(10, x_hist.shape[1])):
        ax.plot(t, x_hist[:, pth, 0], lw=0.6, alpha=0.7)
    ax.set_title("$x_1$ sample paths"); ax.set_xlabel("years"); ax.set_ylabel("$x_1$")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_paths.pdf"))
    plt.close(fig)

    # ---- risk premia at the simulated stationary mean state ---------------
    x_mean = flat.mean(axis=0)[None, :]                    # (1, N-1)
    x_mean = _project(economy, x_mean)
    prem = economy.premium(x_mean)
    alpha = np.asarray(prem["alpha"]).reshape(-1)
    agent_rp = np.asarray(prem["agent_rp"]).reshape(-1)

    summary = {
        "source": economy.source,
        "mean_free_shares": x_mean.reshape(-1).tolist(),
        "mean_full_shares": _full_shares(x_mean).reshape(-1).tolist(),
        "std_x_1": float(flat[:, 0].std()),
        "riskfree_r_at_mean": float(prem["r"][0]),
        "risk_premium_pi_at_mean": float(prem["pi"][0]),
        "price_of_risk_eta_at_mean": float(prem["eta"][0]),
        "return_vol_sigR_at_mean": float(prem["sigR"][0]),
        "portfolio_alpha_at_mean": alpha.tolist(),
        "agent_risk_premium_alpha_i*pi": agent_rp.tolist(),
    }
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, val in summary.items():
            f.write(f"{k}: {val}\n")

    # also a tidy per-agent CSV
    pd.DataFrame({
        "agent": [f"x_{i+1}" for i in range(K)],
        "mean_share": _full_shares(x_mean).reshape(-1),
        "alpha": alpha,
        "agent_risk_premium": agent_rp,
    }).to_csv(os.path.join(out_dir, "risk_premia.csv"), index=False)

    print("\n[simulate] summary:")
    for k, val in summary.items():
        print(f"  {k}: {val}")
    return summary


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nn", choices=["nn", "numerical"])
    parser.add_argument("--case", default="asym2")
    parser.add_argument("--config", default="timestep", choices=list(CONFIGS))
    parser.add_argument("--base-dir", default="./models/GP_NN_NAgents_faster_64bits")
    parser.add_argument("--num-base", default="./models/gp_numerical_cheb",
                        help="base dir of the Chebyshev cheb_solution.csv files")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--layers", type=int, default=4)
    # simulation options
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--years", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--x0", type=float, default=None,
                        help="initial value for every free share (default: 1/K)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.float64:
        torch.set_default_dtype(torch.float64)

    if args.source == "nn":
        model = load_model(args.base_dir, args.case, args.config,
                           width=args.width, layers=args.layers)
        economy = NNEconomy(model)
        out_dir = args.out or os.path.join(args.base_dir, args.case, args.config, "simulation")
    else:
        economy = build_numerical_economy(args.case, out_base=args.num_base)
        out_dir = args.out or os.path.join(args.num_base, args.case, "simulation")

    sim = simulate(economy, n_paths=args.paths, years=args.years, dt=args.dt, x0=args.x0, seed=args.seed)
    analyze(economy, sim, out_dir)


if __name__ == "__main__":
    main()
