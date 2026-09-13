"""Forward-simulate the N-agent stochastic-volatility economy.

Implements the simulation described in ``spec.md`` (lines 13-23), generalised to
the K-agent model solved in ``sv_n_agents_NN.py``:

    dx_i = mu_{x_i} dt + sigma_{x_i} dW            (wealth shares, i = 1..K-1)
    dv   = mu_v       dt + sigma_v       dW        (idiosyncratic-risk state)

This is a *single-shock* economy: the same aggregate Brownian increment dW
drives v and every wealth share (the model's ``sigx_full`` are loadings on that
one shock; see the Ito terms in ``sv_n_agents_NN.compute_sv_equilibrium``).  The
model already exposes the *absolute* drift / diffusion of each share
(``mu_x_full``, ``sigx_full``), so we step them directly with Euler-Maruyama.

The same simulation can be driven either by a trained neural-network model
(``NNEconomy``) or by the finite-difference numerical solution
(``NumericalEconomy``, 2-agent Di Tella) via grid interpolation -- both expose a
common ``drift_diffusion`` / ``premium`` interface.

Outputs (saved to ``--out``):
  * marginal histograms of the aggregate expert wealth share X_E and of v;
  * the joint (X_E, v) distribution;
  * at the simulated mean state, the two risk-premium components:
      - aggregate    : pi * (sigma + sigma_p)
      - idiosyncratic: gamma * (phi v)^2 / x^2   (== chi / x_E[:, 0])
    and their sum.

Usage (examples)::

    # trained NN
    python sv_n_agents_simulate.py --source nn --case agents2 --config timestep \
        --base-dir ./models/SV_NAgents_64bit_analytic_rp --float64

    # finite-difference numerical solution, with parameter overrides
    python sv_n_agents_simulate.py --source numerical --sigma 0.04 --gamma 10
"""

import argparse
import os
from typing import Union
from itertools import product

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from common import BASE_PARAMS, CONFIGS, V_DOMAIN, make_case
from model import get_model, PDEModelNAgentsSV, PDEModelTimeStepNAgentsSV
from numerical import DITELLA_PARAMS, solve_ditella


matplotlib.use("Agg")
# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_dir, case, config="timestep", width=64, layers=4,
               gamma=None, tau=None, sigma=None, a=None, vmean=0.25):
    """Load a trained checkpoint (``model_best.pt``) for ``case``/``config``.

    The economic parameters MUST match those used at training time -- the saved
    weights only encode the *functions* xi/zeta/p, while every equilibrium
    object (pi, sigma+sigma_p, the idiosyncratic term, mu_x, ...) is recomputed
    from ``a``, ``sigma``, ``gamma``, ``tau`` at evaluation time.  Loading with
    the wrong constants silently produces a different economy on the same
    weights.  Params are parsed from the ``free_pr_{gamma}_{tau}_{sigma}_{a}``
    directory name; explicit arguments override the parsed values.
    """
    ts, rar, lb = CONFIGS[config]
    K, eidx, hidx, gamma_vec = make_case(case, gamma)
    subpath_name = case if tau == 1.15 else f"{case}_{tau}"
    if vmean != 0.25:
        subpath_name = f"{case}_{tau}_{vmean}"
    mpath = os.path.join(base_dir, subpath_name, config)
    if not os.path.exists(os.path.join(mpath, "model_best.pt")):
        raise FileNotFoundError(f"no trained checkpoint at {mpath}/model_best.pt -- train first.")
    print(f"[load_model] params from '{os.path.basename(os.path.normpath(base_dir))}': "
          f"gamma={gamma} tau={tau} sigma={sigma} a={a}")
    model = get_model(
        mpath, K, eidx, hidx, gamma_vec,
        model_size=[width] * layers,
        timestepping=ts, rar=rar, loss_balancing=lb,
        params=BASE_PARAMS | {"tau": float(tau), "a": float(a), "sigma": float(sigma), "v_mean": vmean},
        train=False,
    )
    return model


# ---------------------------------------------------------------------------
# Economy adapters: a common (drift_diffusion, premium) interface that can be
# backed either by the trained NN or by the finite-difference numerical solution
# ---------------------------------------------------------------------------
class NNEconomy:
    """Drift/diffusion + risk premia from a trained ``sv_n_agents_NN`` model."""

    source = "nn"

    def __init__(self, model):
        self.model: Union[PDEModelNAgentsSV, PDEModelTimeStepNAgentsSV] = model
        self.K = model.statics["K"]
        self.expert_idx = list(model.statics["expert_idx"])
        self.household_idx = list(model.statics["household_idx"])
        self.gamma_vec = model.statics["gamma"].detach().cpu().numpy().reshape(-1)
        self.v_lo, self.v_hi = model.statics.get("v_domain", V_DOMAIN)
        self.share_lo, self.share_hi = 0.1 / self.K, 1.0 - 0.1 / self.K

    def portfolio(self, x_states, v):
        """Per-agent wealth share ``x_k`` (P, K) and the value of risky capital
        each agent holds AS A FRACTION OF TOTAL (aggregate) WEALTH ``theta_k``
        (P, K) at (x_states, v).

        Aggregate wealth is ``N = p * kappa`` and expert ``k`` holds capital
        worth ``theta_k * N``, so the risky value as a fraction of total wealth
        is simply ``theta_k`` (the share of aggregate capital held by ``k``).
        Households hold no capital (``theta = 0``) -> 0%.  (The agent's OWN-wealth
        portfolio weight would instead be ``theta_k / x_k``; we report the
        total-wealth fraction here.)
        """
        vd = self._forward(x_states, v)
        x_full = vd["x_full"].detach().cpu().numpy()          # (P, K)
        theta_full = vd["theta_full"].detach().cpu().numpy()  # (P, K)
        return x_full, theta_full

    def _forward(self, x_states, v):
        model = self.model
        has_t = model.statics["has_t"]
        P = x_states.shape[0]
        dtype = torch.get_default_dtype()
        SV_np = np.concatenate([x_states, v[:, None]], axis=1)        # (P, K)
        if has_t:                                                     # pad t = min_t
            SV_np = np.concatenate([SV_np, np.zeros((P, 1))], axis=1)
        SV = torch.tensor(SV_np, device=model.device, dtype=dtype)
        SV.requires_grad_(True)                      # update_variables autodiffs
        for i, nm in enumerate(model.state_variables):
            model.variable_val_dict[nm] = SV[:, i:i + 1]
        model.variable_val_dict["SV"] = SV
        model.update_variables(SV)
        return model.variable_val_dict

    def drift_diffusion(self, x_states, v):
        """(mu_x, sig_x [P, K-1]),  (mu_v, sig_v [P]) at (x_states, v)."""
        vd = self._forward(x_states, v)
        K = self.K
        mu_x = vd["mu_x_full"][:, :K - 1].detach().cpu().numpy()
        sig_x = vd["sigx_full"][:, :K - 1].detach().cpu().numpy()
        lbd = self.model.statics["lbd"]; v_mean = self.model.statics["v_mean"]
        sigv_mean = self.model.statics["sigv_mean"]
        mu_v = lbd * (v_mean - v)
        sig_v = sigv_mean * np.sqrt(v)
        return mu_x, sig_x, mu_v, sig_v

    def premium(self, x_states, v):
        """Aggregate + idiosyncratic risk premia (each shape (P,))."""
        vd = self._forward(x_states, v)
        pi = vd["pi"].detach().cpu().numpy().reshape(-1)
        sig_agg = vd["sig_agg"].detach().cpu().numpy().reshape(-1)
        chi = vd["chi"].detach().cpu().numpy().reshape(-1)
        x_full = vd["x_full"].detach().cpu().numpy()
        theta_full = vd["theta_full"].detach().cpu().numpy()
        i0 = self.expert_idx[0]
        x_E0 = x_full[:, i0]                          # anchor expert wealth share
        theta_E0 = theta_full[:, i0]                  # anchor expert capital share
        agg = pi * sig_agg
        # anchor's idiosyncratic premium = chi * theta_0/x_0 = gamma_0 (phi v theta_0/x_0)^2
        # (the term that enters its return).  Reduces to chi/x = gamma*(phi v)^2/x^2
        # in the 2-agent case where the lone expert holds all capital (theta_0 = 1).
        idio = chi * theta_E0 / x_E0
        return dict(pi=pi, sig_agg=sig_agg, agg_rp=agg, idio_rp=idio, total_rp=agg + idio)

    def price(self, x_states, v):
        """Capital price ``p`` (P,) at (x_states, v)."""
        vd = self._forward(x_states, v)
        return vd["p"].detach().cpu().numpy().reshape(-1)


class NumericalEconomy:
    """Same interface, backed by a solved ``DiTellaNumerical`` (2-agent) model
    through grid interpolation of mu_x / sigma_x / pi / (sigma+sigma_p)."""

    source = "numerical"

    def __init__(self, solver):
        if solver.xi is None:
            raise RuntimeError("solve the numerical model first (solver.solve()).")
        self.solver = solver
        self.K = 2
        self.expert_idx = [0]
        self.v_lo, self.v_hi = float(solver.gridv[0]), float(solver.gridv[-1])
        self.share_lo, self.share_hi = float(solver.gridx[0]), float(solver.gridx[-1])
        eq = solver.equilibrium(solver.price, solver.xi, solver.zeta)
        self._mux = solver._interp(eq["mux"])
        self._sigx = solver._interp(eq["sigx"])
        self._pi = solver._interp(eq["pi"])
        self._sig_agg = solver._interp(eq["sig_agg"])
        P = solver.p
        self.gamma = float(P["gamma"]); self.phi = float(P["phi"])
        self.lbd = float(P["lbd"]); self.v_mean = float(P["v_mean"])
        self.sigv_mean = float(P["sigv_mean"])

    def drift_diffusion(self, x_states, v):
        x = x_states[:, 0]
        pts = np.column_stack([v, x])                # interpolation order is (v, x)
        mu_x = self._mux(pts)[:, None]
        sig_x = self._sigx(pts)[:, None]
        mu_v = self.lbd * (self.v_mean - v)
        sig_v = self.sigv_mean * np.sqrt(v)
        return mu_x, sig_x, mu_v, sig_v

    def premium(self, x_states, v):
        x = x_states[:, 0]
        pts = np.column_stack([v, x])
        pi = self._pi(pts)
        sig_agg = self._sig_agg(pts)
        agg = pi * sig_agg
        idio = self.gamma * (self.phi * v) ** 2 / x ** 2     # = chi / x_E0
        return dict(pi=pi, sig_agg=sig_agg, agg_rp=agg, idio_rp=idio, total_rp=agg + idio)


# ---------------------------------------------------------------------------
# Domain projection (keep states where the trained model is valid)
# ---------------------------------------------------------------------------
def _project(economy, x_states, v):
    """Clip free shares and v into the model's valid box, keeping the anchor
    share (1 - sum) feasible."""
    lo, hi = economy.share_lo, economy.share_hi
    x = np.clip(x_states, lo, hi)
    s = x.sum(axis=1, keepdims=True)
    too_big = (s > 1.0 - lo).squeeze(-1)
    if np.any(too_big):
        scale = (1.0 - lo) / s[too_big]
        x[too_big] = np.maximum(x[too_big] * scale, lo)
    v = np.clip(v, economy.v_lo, economy.v_hi)
    return x, v


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(economy, n_paths=100, years=500.0, dt=0.08,
             x0=0.2, v0=0.1, seed=0, store_every=1):
    """Euler-Maruyama forward simulation of the K-agent economy.

    Returns dict with ``x_hist`` (T, P, K-1), ``v_hist`` (T, P), ``t`` (T,).
    A single shared Gaussian shock per (path, step) drives all states.
    """
    rng = np.random.default_rng(seed)
    K = economy.K
    n_steps = int(round(years / dt))
    sqrt_dt = np.sqrt(dt)

    # initial state: each free share at x0 (fall back to equal split if x0 is
    # infeasible for this K, e.g. (K-1)*x0 >= 1).
    if (K - 1) * x0 >= 1.0 - economy.share_lo:
        x0_vec = np.full(K - 1, 1.0 / K)
        print(f"[simulate] x0={x0} infeasible for K={K}; using equal shares 1/K={1.0/K:.3f}")
    else:
        x0_vec = np.full(K - 1, x0)
    x = np.tile(x0_vec, (n_paths, 1)).astype(np.float64)             # (P, K-1)
    v = np.full(n_paths, float(v0))                                  # (P,)
    x, v = _project(economy, x, v)

    n_store = n_steps // store_every + 1
    x_hist = np.empty((n_store, n_paths, K - 1))
    v_hist = np.empty((n_store, n_paths))
    t_hist = np.empty(n_store)
    x_hist[0], v_hist[0], t_hist[0] = x, v, 0.0
    si = 1

    for step in range(1, n_steps + 1):
        mu_x, sig_x, mu_v, sig_v = economy.drift_diffusion(x, v)
        dW = rng.standard_normal(n_paths) * sqrt_dt                  # shared shock
        x = x + mu_x * dt + sig_x * dW[:, None]
        v = v + mu_v * dt + sig_v * dW
        x, v = _project(economy, x, v)
        if step % store_every == 0:
            x_hist[si], v_hist[si], t_hist[si] = x, v, step * dt
            si += 1
        if step % max(1, n_steps // 20) == 0:
            print(f"[simulate] step {step:>6d}/{n_steps}  t={step*dt:7.1f}y  "
                  f"mean sum(free shares)={x.sum(axis=1).mean():.3f}  "
                  f"mean v={v.mean():.3f}")

    return {"x_hist": x_hist[:si], "v_hist": v_hist[:si], "t": t_hist[:si]}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _expert_total_share(economy, x_states):
    """Aggregate expert wealth share X_E from the free-share array.

    x_states : (..., K-1).  The anchor (K-th) agent share = 1 - sum(free).
    """
    full = np.concatenate([x_states, 1.0 - x_states.sum(axis=-1, keepdims=True)], axis=-1)  # (...,K)
    return full[..., economy.expert_idx].sum(axis=-1)


def analyze(economy, sim, out_dir, burn_in_frac=0.2):
    os.makedirs(out_dir, exist_ok=True)

    x_hist, v_hist, t = sim["x_hist"], sim["v_hist"], sim["t"]
    burn = int(len(t) * burn_in_frac)
    x_pool = x_hist[burn:]                          # (Tb, P, K-1)
    v_pool = v_hist[burn:].reshape(-1)              # (Tb*P,)
    XE_pool = _expert_total_share(economy, x_pool).reshape(-1)  # (Tb*P,)

    # ---- marginal + joint distributions -----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist(XE_pool, bins=60, density=True, color="C0", alpha=0.8)
    ax[0].set_title("Marginal: aggregate expert share $X_E$")
    ax[0].set_xlabel("$X_E$")
    ax[1].hist(v_pool, bins=60, density=True, color="C1", alpha=0.8)
    ax[1].set_title("Marginal: volatility state $v$")
    ax[1].set_xlabel("$v$")
    h = ax[2].hist2d(XE_pool, v_pool, bins=60, density=True, cmap="viridis")
    fig.colorbar(h[3], ax=ax[2])
    ax[2].set_title("Joint distribution $(X_E, v)$")
    ax[2].set_xlabel("$X_E$"); ax[2].set_ylabel("$v$")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distribution.pdf"))
    plt.close(fig)

    # ---- a few sample paths of X_E and v ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    XE_path = _expert_total_share(economy, x_hist)    # (T, P)
    for pth in range(min(10, XE_path.shape[1])):
        ax[0].plot(t, XE_path[:, pth], lw=0.6, alpha=0.7)
        ax[1].plot(t, v_hist[:, pth], lw=0.6, alpha=0.7)
    ax[0].set_title("$X_E$ sample paths"); ax[0].set_xlabel("years")
    ax[1].set_title("$v$ sample paths"); ax[1].set_xlabel("years")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_paths.pdf"))
    plt.close(fig)

    # ---- risk premia at the simulated mean state --------------------------
    x_mean = x_pool.reshape(-1, x_pool.shape[-1]).mean(axis=0)[None, :]  # (1, K-1)
    v_mean = np.array([v_pool.mean()])
    x_mean, v_mean = _project(economy, x_mean, v_mean)
    prem = economy.premium(x_mean, v_mean)
    pi = float(prem["pi"][0]); sig_agg = float(prem["sig_agg"][0])
    agg_rp = float(prem["agg_rp"][0])
    idio_rp = float(prem["idio_rp"][0])
    total_rp = float(prem["total_rp"][0])

    summary = {
        "source": economy.source,
        "mean_X_E": float(XE_pool.mean()),
        "mean_v": float(v_pool.mean()),
        "std_X_E": float(XE_pool.std()),
        "std_v": float(v_pool.std()),
        "x_mean_free_shares": x_mean.reshape(-1).tolist(),
        "pi_at_mean": pi,
        "sig_plus_sigp_at_mean": sig_agg,
        "aggregate_risk_premium_pi*(sig+sigp)": agg_rp,
        "idiosyncratic_risk_premium_gamma*(phi*v)^2/x^2": idio_rp,
        "total_risk_premium": total_rp,
    }
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, val in summary.items():
            f.write(f"{k}: {val}\n")
    print("\n[simulate] summary:")
    for k, val in summary.items():
        print(f"  {k}: {val}")
    return summary


# ---------------------------------------------------------------------------
# Portfolio composition by wealth decile
# ---------------------------------------------------------------------------
def plot_portfolio_deciles(economy, sim, out_dir, n_deciles=10, burn_in_frac=0.2,
                           max_states=20000, chunk=2000, seed=0,
                           file_name="portfolio_deciles.pdf"):
    """Risky-asset holdings as a share of TOTAL wealth (``theta_k``) by wealth decile.

    Pools every ``(path, time, agent-type)`` point of the ergodic simulation
    (after burn-in) into ``(wealth x_k, risky-share-of-total-wealth theta_k)``
    pairs (households contribute theta = 0), sorts by wealth, splits into
    ``n_deciles`` equal-count bins, and plots the mean ``theta_k`` per decile.
    Returns the per-decile means and prints them.
    """
    os.makedirs(out_dir, exist_ok=True)
    x_hist, v_hist, t = sim["x_hist"], sim["v_hist"], sim["t"]
    burn = int(len(t) * burn_in_frac)
    x_pool = x_hist[burn:].reshape(-1, x_hist.shape[-1])       # (M, K-1)
    v_pool = v_hist[burn:].reshape(-1)                         # (M,)
    M = x_pool.shape[0]
    if M > max_states:                                        # subsample for cost
        rng = np.random.default_rng(seed)
        sel = rng.choice(M, size=max_states, replace=False)
        x_pool, v_pool = x_pool[sel], v_pool[sel]

    wealth_list, risky_list = [], []
    for c in range(0, x_pool.shape[0], chunk):
        xf, rf = economy.portfolio(x_pool[c:c + chunk], v_pool[c:c + chunk])
        wealth_list.append(xf)
        risky_list.append(rf)
    wealth = np.concatenate(wealth_list, 0).reshape(-1)        # (M*K,)
    risky = np.concatenate(risky_list, 0).reshape(-1)         # (M*K,)  theta_k

    order = np.argsort(wealth)
    wealth_s, risky_s = wealth[order], risky[order]
    edges = np.linspace(0, len(wealth_s), n_deciles + 1).astype(int)
    dec_risky = np.array([risky_s[edges[i]:edges[i + 1]].mean() for i in range(n_deciles)])
    dec_wealth = np.array([wealth_s[edges[i]:edges[i + 1]].mean() for i in range(n_deciles)])

    deciles = np.arange(1, n_deciles + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(deciles, dec_risky, color="0.35")
    ax.set_xlabel("Wealth decile (1 = poorest, 10 = richest)")
    ax.set_ylabel(r"Risky assets / total wealth ($\theta_k$)")
    ax.set_xticks(deciles)
    ax.set_ylim(bottom=0)
    ax.set_title("Risky-asset holdings (share of total wealth) by wealth decile")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)

    print("\n[portfolio] decile | mean wealth x_k | mean risky theta_k (share of total wealth)")
    for i in range(n_deciles):
        print(f"  {i + 1:>2d}   {dec_wealth[i]:.5f}    {dec_risky[i]:.4f}")
    return dict(decile=deciles, wealth=dec_wealth, risky=dec_risky)


def plot_portfolio_terciles(economy, sim, out_dir, burn_in_frac=0.2,
                            max_states=20000, chunk=2000, seed=0,
                            file_prefix="portfolio_terciles"):
    """Three IDENTITY-based groups of how capital-intensive each balance sheet is.

    Groups are defined by agent identity (NOT by ranking pooled points), so a
    household is always a household:

    1. ``Households``            -- every household agent (hold no capital -> 0).
    2. ``Poorer experts``        -- the lower-wealth half of the experts.
    3. ``Richer experts``        -- the upper-wealth half of the experts.

    Experts are split by their time-averaged wealth share ``mean_t x_k`` (the
    split need not be even in count).  For each group we pool every
    ``(path, time, agent-in-group)`` point (after burn-in) and aggregate.

    Writes TWO figures (user picks later):

    * ``*_balance_sheet.pdf`` -- "proportion of the group's OWN wealth in
      capital", i.e. the wealth-weighted ``Sum(theta)/Sum(x)`` per group.  Net
      worth is 1 (dashed line); the risk-free (bond) piece is ``1 - theta/x``.
      For experts ``theta/x > 1`` (they borrow from households), so the capital
      bar overshoots 1 -- the overshoot is drawn as a separate hatched
      "levered / borrowed" segment.  Households: 0 capital, all risk-free.
    * ``*_capital_share.pdf`` -- bounded [0,1] alternative: each group's share of
      the economy's AGGREGATE capital, ``Sum(theta)/Sum_all(theta)`` (bars sum
      to 1 across groups).
    """
    os.makedirs(out_dir, exist_ok=True)
    x_hist, v_hist, t = sim["x_hist"], sim["v_hist"], sim["t"]
    burn = int(len(t) * burn_in_frac)
    x_pool = x_hist[burn:].reshape(-1, x_hist.shape[-1])       # (M, K-1)
    v_pool = v_hist[burn:].reshape(-1)                         # (M,)
    M = x_pool.shape[0]
    if M > max_states:
        rng = np.random.default_rng(seed)
        sel = rng.choice(M, size=max_states, replace=False)
        x_pool, v_pool = x_pool[sel], v_pool[sel]

    wealth_list, theta_list = [], []
    for c in range(0, x_pool.shape[0], chunk):
        xf, tf = economy.portfolio(x_pool[c:c + chunk], v_pool[c:c + chunk])
        wealth_list.append(xf)
        theta_list.append(tf)
    wealth = np.concatenate(wealth_list, 0)                    # (M, K)  x_k
    theta = np.concatenate(theta_list, 0)                      # (M, K)  theta_k

    eps = 1e-12
    ratio = np.zeros_like(wealth)                              # theta_k / x_k
    np.divide(theta, wealth, out=ratio, where=wealth > eps)

    # ---- identity-based groups (columns are agents) -----------------------
    eidx = np.asarray(economy.expert_idx, dtype=int)
    hidx = np.asarray(economy.household_idx, dtype=int)
    expert_meanw = wealth[:, eidx].mean(0)                     # per-expert mean x_k
    order_e = eidx[np.argsort(expert_meanw)]                   # ascending wealth
    n_e = len(eidx)
    split = n_e // 2                                           # lower half = poorer
    poor_cols = order_e[:split]
    rich_cols = order_e[split:]
    groups = [("Households", hidx),
              ("Poorer experts\n(lower half)", poor_cols),
              ("Richer experts\n(upper half)", rich_cols)]

    total_theta = theta.sum()
    cap_frac, cap_share, mean_ratio, mean_wealth = [], [], [], []
    for _, cols in groups:
        if len(cols) == 0:
            cap_frac.append(0.0); cap_share.append(0.0)
            mean_ratio.append(0.0); mean_wealth.append(0.0)
            continue
        sx = wealth[:, cols].sum(); st = theta[:, cols].sum()
        cap_frac.append(st / max(sx, eps))                    # Sum(theta)/Sum(x)
        cap_share.append(st / max(total_theta, eps))          # share of aggregate capital
        mean_ratio.append(float(ratio[:, cols].mean()))
        mean_wealth.append(float(wealth[:, cols].mean()))
    cap_frac = np.array(cap_frac); cap_share = np.array(cap_share)
    mean_ratio = np.array(mean_ratio); mean_wealth = np.array(mean_wealth)

    n_bins = len(groups)
    labels = [g[0] for g in groups]
    xpos = np.arange(n_bins)

    # ---- figure 1: balance-sheet composition (own wealth = 1) --------------
    own = np.minimum(cap_frac, 1.0)                 # own-funded capital
    lev = np.maximum(cap_frac - 1.0, 0.0)           # borrowed (levered) capital > 1
    rf = np.maximum(1.0 - cap_frac, 0.0)            # positive risk-free (households)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xpos, own, color="#4C72B0", label="Capital, own-funded")
    ax.bar(xpos, rf, bottom=own, color="#DD8452", label="Risk-free (bonds)")
    ax.bar(xpos, lev, bottom=1.0, color="#C44E52", hatch="//",
           label="Capital, levered (borrowed)")
    ax.axhline(1.0, ls="--", color="k", lw=1.2)
    ax.text(n_bins - 0.5, 1.02, "own wealth = 1", ha="right", va="bottom", fontsize=9)
    for i in range(n_bins):
        ax.text(xpos[i], cap_frac[i] + 0.03 * max(1.0, cap_frac.max()),
                f"{cap_frac[i]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel(r"Portfolio as fraction of own wealth ($\theta_k/x_k$)")
    ax.set_title("Balance sheet by group (net worth = 1)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{file_prefix}_balance_sheet.pdf"))
    plt.close(fig)

    # ---- figure 2: bounded share of aggregate capital ----------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xpos, cap_share, color="0.35")
    for i in range(n_bins):
        ax.text(xpos[i], cap_share[i] + 0.01, f"{cap_share[i]:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("Share of aggregate capital held")
    ax.set_title("Share of the economy's capital by group (sums to 1)")
    ax.set_ylim(0, min(1.0, cap_share.max() * 1.2 + 0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{file_prefix}_capital_share.pdf"))
    plt.close(fig)

    print("\n[portfolio-terciles] group | mean x_k | mean theta/x | Sum(theta)/Sum(x) | capital share")
    for i in range(n_bins):
        lbl = labels[i].replace("\n", " ")
        print(f"  {lbl:>22s}  {mean_wealth[i]:.5f}   {mean_ratio[i]:8.3f}   "
              f"{cap_frac[i]:8.3f}   {cap_share[i]:.4f}")
    return dict(labels=labels, cap_frac=cap_frac, cap_share=cap_share,
                mean_ratio=mean_ratio, mean_wealth=mean_wealth)


def _gini_rows(w):
    """Row-wise Gini of a (M, K) array of NON-negative weights.

    For each row, ``G = (2 * sum_i i * w_(i)) / (K * sum w) - (K + 1) / K`` on the
    ascending-sorted row (``i = 1..K``).  Returns a length-M array; a row that
    sums to <= 0 gets NaN.  With shares that sum to 1, this is the standard
    cross-sectional wealth Gini for that snapshot.
    """
    w = np.sort(np.clip(w, 0.0, None), axis=1)                 # (M, K) ascending
    M, K = w.shape
    s = w.sum(axis=1)
    idx = np.arange(1, K + 1)
    g = (2.0 * (w * idx).sum(axis=1)) / (K * s) - (K + 1.0) / K
    g[s <= 0] = np.nan
    return g


def _gini(w):
    """Gini of a 1-D array of non-negative weights (pooled version)."""
    return float(_gini_rows(np.asarray(w, float).reshape(1, -1))[0])


def plot_wealth_distribution(economy, sim, out_dir, burn_in_frac=0.2,
                             last_steps=None, file_name="wealth_distribution.pdf"):
    """Cross-sectional distribution of individual wealth shares ``x_k``.

    Pools every ``(path, time, agent)`` wealth share into one population, then
    plots the distribution and reports the ratios discussed for the empirical
    comparison -- headline is ``p95 / median``.

    Sampling window:

    * ``last_steps`` (int)  -- keep only the last ``last_steps`` simulation steps
      of every path (a fixed ergodic window, e.g. 500).  Use a longer/denser sim
      (``--years`` / ``--paths``) to pool more values.
    * otherwise the first ``burn_in_frac`` of the horizon is dropped.

    Two panels: a linear histogram and a ``log10 x_k`` histogram (mass piles up at
    the share floor ``0.1/K`` with a long right tail, so the log axis is what
    makes the shape legible).  Each panel overlays a Gaussian-KDE density curve
    (fit in linear / log10 space respectively) so the shape is readable without
    reading bar heights.  Median (green) and p95 (red) are marked; the ratio
    table is annotated, printed, and written to
    ``wealth_distribution_summary.txt``.
    """
    os.makedirs(out_dir, exist_ok=True)
    x_hist, t = sim["x_hist"], sim["t"]
    if last_steps is not None and last_steps > 0:
        w = min(int(last_steps), x_hist.shape[0])
        x_free = x_hist[-w:].reshape(-1, x_hist.shape[-1])     # (M, K-1)
        window = f"last {w} steps"
    else:
        burn = int(len(t) * burn_in_frac)
        x_free = x_hist[burn:].reshape(-1, x_hist.shape[-1])   # (M, K-1)
        window = f"post burn-in ({int(burn_in_frac * 100)}%)"
    x_full = _full_shares(economy, x_free)                     # (M, K)
    x = x_full.reshape(-1)                                     # (M*K,)  one point per (agent, obs)
    x = x[np.isfinite(x)]
    x = x[x > 0]

    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pv = {q: float(np.percentile(x, q)) for q in qs}
    med = pv[50]
    ratios = {
        "p75/p25": pv[75] / pv[25],
        "p90/median": pv[90] / med,
        "p95/median": pv[95] / med,
        "p99/median": pv[99] / med,
    }

    # Gini: per-snapshot cross-sectional inequality (each row of x_full sums to
    # 1 over the K agents), then averaged over snapshots -- this is the "typical"
    # wealth Gini and does NOT mix in time variation.  The pooled Gini (over all
    # (agent, obs) points at once) is reported for reference only.
    gini_cs = _gini_rows(x_full)                               # (M,)
    gini_cs = gini_cs[np.isfinite(gini_cs)]
    gini = {
        "gini_cross_section_mean": float(np.mean(gini_cs)),
        "gini_cross_section_median": float(np.median(gini_cs)),
        "gini_cross_section_p5": float(np.percentile(gini_cs, 5)),
        "gini_cross_section_p95": float(np.percentile(gini_cs, 95)),
        "gini_pooled": _gini(x),
    }

    try:
        from scipy.stats import gaussian_kde
    except Exception:                                          # scipy optional
        gaussian_kde = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, logx in zip(axes, (False, True)):
        if logx:
            data = np.log10(x)
            bins = np.linspace(data.min(), data.max(), 60)
            med_m, p95_m = np.log10(med), np.log10(pv[95])
            xlabel = r"$\log_{10}$ wealth share $x_k$"
        else:
            data = x
            bins = 60
            med_m, p95_m = med, pv[95]
            xlabel = r"wealth share $x_k$"
        ax.hist(data, bins=bins, density=True, color="C0", alpha=0.55)
        if gaussian_kde is not None and data.size > 2 and data.std() > 0:
            kde = gaussian_kde(data)
            grid = np.linspace(data.min(), data.max(), 400)
            ax.plot(grid, kde(grid), color="navy", lw=1.8, label="KDE density")
        ax.axvline(med_m, color="green", ls="--", lw=1.5, label=f"median = {med:.4f}")
        ax.axvline(p95_m, color="red", ls="--", lw=1.5, label=f"p95 = {pv[95]:.4f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
        ax.set_title(("log10 scale" if logx else "linear scale"))
    axes[1].text(0.98, 0.95,
                 f"p95/median = {ratios['p95/median']:.2f}\n"
                 f"Gini = {gini['gini_cross_section_mean']:.3f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=11,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.suptitle(f"Cross-sectional wealth-share distribution (pooled agents x sim, {window})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)

    lines = [f"window: {window}", f"n_obs: {x.size}", f"n_agents(K): {x_full.shape[1]}"]
    lines += [f"p{q}: {pv[q]:.6f}" for q in qs]
    lines += [f"{k}: {v:.4f}" for k, v in ratios.items()]
    lines += [f"{k}: {v:.4f}" for k, v in gini.items()]
    with open(os.path.join(out_dir, "wealth_distribution_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[wealth-dist] percentiles of individual wealth share x_k:")
    for q in qs:
        print(f"  p{q:>2d} = {pv[q]:.6f}")
    print("[wealth-dist] ratios:")
    for k, v in ratios.items():
        print(f"  {k:>12s} = {v:.3f}")
    print("[wealth-dist] Gini (cross-sectional, per-snapshot then averaged):")
    print(f"  mean   = {gini['gini_cross_section_mean']:.4f}  "
          f"(median {gini['gini_cross_section_median']:.4f}, "
          f"p5-p95 {gini['gini_cross_section_p5']:.4f}-{gini['gini_cross_section_p95']:.4f})")
    print(f"  pooled = {gini['gini_pooled']:.4f}  (reference, mixes time variation)")
    return dict(percentiles=pv, ratios=ratios, gini=gini)


# ---------------------------------------------------------------------------
# Impulse response: deterministic -k*sigma shock on the aggregate (capital) shock
# ---------------------------------------------------------------------------
def _full_shares(economy, x_free):
    """(T, K-1) free shares -> (T, K) full shares (anchor = 1 - sum)."""
    anchor = 1.0 - x_free.sum(axis=-1, keepdims=True)
    return np.concatenate([x_free, anchor], axis=-1)


def _irf_paths(economy, x0_free, v0, n_steps, dt, shock_step, specs,
               track_price=True):
    """Step several deterministic paths TOGETHER (one batched forward per step).

    All paths start at (x0_free, v0) and evolve drift-only.  ``specs`` is a list
    of dicts (one per path); at ``shock_step`` each path adds Brownian increments
    ``dW_x`` (to the wealth-share diffusion) and ``dW_v`` (to ``v``).  A path with
    ``freeze_v=True`` holds ``v`` fixed at ``v0`` for the whole horizon (shutting
    down the uncertainty channel).

    Returns a list of dicts (one per spec), each with ``x_full`` (T, K),
    ``X_E`` (T,), ``v`` (T,), ``p`` (T,).
    """
    K = economy.K
    P = len(specs)
    x = np.tile(np.asarray(x0_free, dtype=np.float64).reshape(1, K - 1), (P, 1))
    v = np.full(P, float(v0))
    x, v = _project(economy, x, v)
    dWx = np.array([s.get("dW_x", 0.0) for s in specs])
    dWv = np.array([s.get("dW_v", 0.0) for s in specs])
    freeze_v = np.array([bool(s.get("freeze_v", False)) for s in specs])
    T = n_steps + 1
    x_hist = np.empty((P, T, K)); v_hist = np.empty((P, T)); p_hist = np.full((P, T), np.nan)

    def _record(i):
        x_hist[:, i, :] = _full_shares(economy, x)
        v_hist[:, i] = v
        if track_price and hasattr(economy, "price"):
            p_hist[:, i] = economy.price(x, v)

    _record(0)
    for step in range(1, n_steps + 1):
        mu_x, sig_x, mu_v, sig_v = economy.drift_diffusion(x, v)
        ax = dWx if step == shock_step else np.zeros(P)
        av = dWv if step == shock_step else np.zeros(P)
        x = x + mu_x * dt + sig_x * ax[:, None]
        v_new = v + mu_v * dt + sig_v * av
        v = np.where(freeze_v, v, v_new)              # frozen paths keep v == v0
        x, v = _project(economy, x, v)
        _record(step)

    out = []
    for j in range(P):
        xf = x_hist[j]
        out.append(dict(x_full=xf, X_E=xf[:, economy.expert_idx].sum(axis=1),
                        v=v_hist[j], p=p_hist[j]))
    return out


def impulse_response(economy, years_burn=250.0, years_irf=30.0, dt=0.02,
                     shock_sd=2.0, t_shock=1.0, x0=0.2, v0=0.1, seed=0,
                     n_paths_stat=40, burn_dt=0.1, shock_horizon=1.0,
                     include_vfixed=False, v_start=None):
    """Deterministic impulse response to a ``-shock_sd``-sigma aggregate shock.

    (1) Runs a long stochastic simulation and takes the post-burn-in mean state
    ``(x*, v*)`` as the stationary starting point.  (2) Evolves drift-only
    (deterministic) paths from ``(x*, v*)``: a *baseline* with no shock and a
    *shocked* path that injects a single Brownian increment at ``t = t_shock``,
    then lets the system mean-revert.  The difference is the IRF.

    If ``v_start`` is given it OVERRIDES the ergodic ``v*`` as the initial (and
    baseline) volatility level -- useful to start exactly at ``v_mean`` rather
    than the clip-biased simulated mean (``x*`` is still the ergodic mean).

    The shock is a ``shock_sd``-standard-deviation ANNUAL innovation:
    ``dW = -shock_sd * sqrt(shock_horizon)`` with ``shock_horizon = 1`` year, so
    it injects a full year's worth of a 2-sigma negative capital shock at once
    (log-capital drops ~ sigma * shock_sd).

    If ``include_vfixed`` is True, a third counterfactual path is computed in
    which the volatility state ``v`` is HELD CONSTANT at ``v*`` (the uncertainty
    channel is shut down), isolating the pure capital/wealth-redistribution
    response -- this removes the ``v``-driven overshoot.

    Returns dict with ``t`` (T,), ``base``/``shock``/``vfix`` sub-dicts (``vfix``
    is None unless ``include_vfixed``), and the stationary ``x_star``/``v_star``.
    """
    print(f"[irf] finding stationary state via {n_paths_stat}-path, {years_burn:.0f}y simulation ...")
    sim = simulate(economy, n_paths=n_paths_stat, years=years_burn, dt=burn_dt,
                   x0=x0, v0=v0, seed=seed)
    burn = int(len(sim["t"]) * 0.5)
    x_star = sim["x_hist"][burn:].reshape(-1, economy.K - 1).mean(axis=0)
    v_ergodic = float(sim["v_hist"][burn:].mean())
    if v_start is not None:
        v_star = float(np.clip(v_start, economy.v_lo, economy.v_hi))
        print(f"[irf] stationary X_E*={_expert_total_share(economy, x_star[None,:])[0]:.4f}; "
              f"v ergodic mean={v_ergodic:.4f}, overriding start v*={v_star:.4f}")
    else:
        v_star = v_ergodic
        print(f"[irf] stationary v*={v_star:.4f}  X_E*={_expert_total_share(economy, x_star[None,:])[0]:.4f}")

    n_steps = int(round(years_irf / dt))
    shock_step = max(1, int(round(t_shock / dt)))
    shock_dW = -float(shock_sd) * np.sqrt(shock_horizon)
    print(f"[irf] injecting dW={shock_dW:.4f} (-{shock_sd:g} sd over {shock_horizon:g}y) at t={t_shock:g}y"
          + ("  (+ v-held-constant counterfactual)" if include_vfixed else ""))

    specs = [dict(dW_x=0.0, dW_v=0.0, freeze_v=False),          # baseline
             dict(dW_x=shock_dW, dW_v=shock_dW, freeze_v=False)]  # full shock
    if include_vfixed:
        specs.append(dict(dW_x=shock_dW, dW_v=0.0, freeze_v=True))  # v held fixed

    paths = _irf_paths(economy, x_star, v_star, n_steps, dt, shock_step, specs)
    base, shock = paths[0], paths[1]
    vfix = paths[2] if include_vfixed else None
    t = np.arange(n_steps + 1) * dt - shock_step * dt      # 0 at the shock

    i0 = shock_step
    print("[irf] on-impact response (shock - baseline) at t=0+:")
    print(f"      X_E: {base['X_E'][i0]:.4f} -> {shock['X_E'][i0]:.4f} "
          f"(delta {shock['X_E'][i0]-base['X_E'][i0]:+.4f})")
    print(f"      v  : {base['v'][i0]:.4f} -> {shock['v'][i0]:.4f} "
          f"(delta {shock['v'][i0]-base['v'][i0]:+.4f})")
    print(f"      p  : {base['p'][i0]:.4f} -> {shock['p'][i0]:.4f} "
          f"(delta {shock['p'][i0]-base['p'][i0]:+.4f})")
    dev0 = shock['x_full'][i0] - base['x_full'][i0]
    for k in economy.household_idx:
        verb = "GAINS" if dev0[k] > 0 else "loses"
        print(f"      household {k}: wealth-share {verb} {dev0[k]:+.5f} on impact")

    hdr = "[irf]  t(y) |   dX_E    |    dv     |    dp    "
    hdr += "|  dp(v-fix)" if include_vfixed else ""
    print(hdr)
    for ty in [0.0, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, float(t[-1])]:
        j = int(np.clip(i0 + round(ty / dt), 0, len(t) - 1))
        row = (f"      {t[j]:5.2f} | {shock['X_E'][j]-base['X_E'][j]:+.5f} | "
               f"{shock['v'][j]-base['v'][j]:+.5f} | {shock['p'][j]-base['p'][j]:+.5f}")
        if include_vfixed:
            row += f" | {vfix['p'][j]-base['p'][j]:+.5f}"
        print(row)
    return dict(t=t, base=base, shock=shock, vfix=vfix, x_star=x_star,
                v_star=v_star, shock_sd=shock_sd, shock_time_idx=shock_step)


def plot_impulse_response(economy, irf, out_dir, file_name="impulse_response.pdf",
                          xmax=10.0):
    """Plot per-agent wealth-share IMPULSE RESPONSES and aggregates for the IRF.

    Top-left: every agent's wealth-share response ``x_k(t) - baseline`` so the
    reaction of each agent is visible on a comparable scale (levels are dominated
    by the big saver household).  Experts are colour-graded by risk aversion;
    each household gets its OWN distinct colour (red/magenta/brown/black) and a
    legend entry so it's clear which household gains vs loses.  Others: aggregate
    expert share ``X_E``, volatility state ``v``, price ``p``.

    ``xmax`` caps the x-axis (years since shock) since the response has decayed
    by then.  If ``irf['vfix']`` is present, its curves are overlaid (dashed) so
    the full IRF can be compared against the ``v``-held-constant counterfactual.
    """
    os.makedirs(out_dir, exist_ok=True)
    t = irf["t"]; base = irf["base"]; shock = irf["shock"]; vfix = irf.get("vfix")
    K = economy.K
    eidx = list(economy.expert_idx); hidx = list(economy.household_idx)
    gam = np.asarray(economy.gamma_vec).reshape(-1)
    dev = shock["x_full"] - base["x_full"]                 # (T, K) wealth-share IRF
    dev_vf = (vfix["x_full"] - base["x_full"]) if vfix is not None else None
    xlim = (float(t[0]), float(xmax))
    hh_colors = ["tab:red", "tab:purple", "saddlebrown", "black", "deeppink", "olive"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ge = gam[eidx]
    gmin, gmax = float(ge.min()), float(ge.max())
    cmap = plt.get_cmap("viridis")
    for k in eidx:
        c = cmap((gam[k] - gmin) / (gmax - gmin + 1e-12))
        ax.plot(t, dev[:, k], color=c, lw=1.3)
        if dev_vf is not None:
            ax.plot(t, dev_vf[:, k], color=c, lw=0.9, ls="--", alpha=0.7)
    for j, k in enumerate(hidx):
        col = hh_colors[j % len(hh_colors)]
        ax.plot(t, dev[:, k], color=col, lw=2.0,
                label=f"household {k} ($\\gamma$={gam[k]:g})")
        if dev_vf is not None:
            ax.plot(t, dev_vf[:, k], color=col, lw=1.1, ls="--", alpha=0.7)
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(0.0, color="red", lw=0.8, alpha=0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("years since shock")
    ax.set_ylabel(r"wealth-share response $x_k - x_k^{\rm base}$")
    ax.set_title(f"Per-agent wealth-share response (-{irf['shock_sd']:g}$\\sigma$ shock)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(gmin, gmax))
    cb = fig.colorbar(sm, ax=ax); cb.set_label(r"expert risk aversion $\gamma$")
    if hidx:
        ax.legend(frameon=False, fontsize=8, loc="best", title="households")

    def _panel(ax, key, color, title, ylabel):
        ax.plot(t, shock[key], color=color, lw=1.6, label="shocked (full)")
        ax.plot(t, base[key], color=color, lw=0.9, ls=":", label="baseline")
        if vfix is not None:
            ax.plot(t, vfix[key], color="0.35", lw=1.4, ls="--", label="shocked (v-fixed)")
        ax.axvline(0.0, color="red", lw=0.8, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel("years since shock"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(frameon=False)

    _panel(axes[0, 1], "X_E", "C0", "Aggregate expert wealth share", r"$X_E$")
    _panel(axes[1, 0], "v", "C3", "Idiosyncratic-risk state $v$", r"$v$")
    _panel(axes[1, 1], "p", "C2", "Capital price $p$", r"$p$")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)
    print(f"[irf] wrote {os.path.join(out_dir, file_name)}")


# ---------------------------------------------------------------------------
def build_numerical_economy(param_overrides=None, h=2e-4, max_iters=300_000,
                            tol=1e-7):
    """Solve the finite-difference Di Tella model (optionally with parameter
    overrides for exploration) and wrap it as an economy."""
    params = dict(DITELLA_PARAMS)
    if param_overrides:
        params.update({k: v for k, v in param_overrides.items() if v is not None})
    solver = solve_ditella(params=params, h=h, max_iters=max_iters, tol=tol)
    return NumericalEconomy(solver)


def sweep_risk_premium(a_list, sigmas, gammas, taus=None, base_overrides=None, h=2e-4,
                       max_iters=300_000, tol=1e-7, eval_x=0.5, eval_v=None,
                       use_simulation=True, sim_kwargs=None,
                       csv_path=None, verbose=True):
    """Solve the FD Di Tella model over a ``sigma`` x ``gamma`` x ``tau`` grid and
    tabulate the risk premium, so a parameter set hitting a target aggregate
    premium can be read off a single table.

    ``taus`` is the Poisson expert-retirement rate; if ``None`` the baseline
    ``DITELLA_PARAMS["tau"]`` is used (so the sweep stays 2-D).

    For each (sigma, gamma, tau):
      * solve the model with those overrides (plus any ``base_overrides``);
      * report the *aggregate* premium  pi*(sigma+sigma_p)  and the
        *idiosyncratic* premium  gamma*(phi v)^2/x^2  at an evaluation state, and
        the grid-averaged aggregate premium for a state-independent summary.

    The evaluation state is a fixed representative point ``(eval_x, eval_v)``
    (default x=0.5, v=long-run mean) -- cheap, no simulation.  Set
    ``use_simulation=True`` to instead evaluate at the *stationary mean* of a
    forward simulation (slower; ``sim_kwargs`` overrides its settings).

    Returns a ``pandas.DataFrame`` (also written to ``csv_path`` if given).
    """
    import pandas as pd

    if taus is None:
        taus = [DITELLA_PARAMS["tau"]]

    rows = []
    for a, sigma, gamma, tau in product(a_list, sigmas, gammas, taus):
        print(f"Processing {a, sigma, gamma, tau}")
        params = dict(DITELLA_PARAMS)
        if base_overrides:
            params.update({k: v for k, v in base_overrides.items() if v is not None})
        params.update(a=float(a), sigma=float(sigma), gamma=float(gamma), tau=float(tau))
        solver = solve_ditella(params=params, h=h, max_iters=max_iters,
                                tol=tol, verbose=False)
        econ = NumericalEconomy(solver)

        if use_simulation:
            sim = simulate(econ, **(sim_kwargs or dict(
                n_paths=50, years=300.0, dt=0.08)))
            burn = int(len(sim["t"]) * 0.2)
            xq = sim["x_hist"][burn:].reshape(-1, 1).mean(axis=0)[None, :]
            vq = np.array([sim["v_hist"][burn:].mean()])
            xq, vq = _project(econ, xq, vq)
        else:
            vq = np.array([eval_v if eval_v is not None else params["v_mean"]])
            xq = np.array([[eval_x]])
            xq, vq = _project(econ, xq, vq)

        prem = econ.premium(xq, vq)
        grid_rp = float(np.mean(solver.grid_solution()["risk_premium"]))
        row = dict(a=float(a), sigma=float(sigma), gamma=float(gamma), tau=float(tau),
                    x_eval=float(xq[0, 0]), v_eval=float(vq[0]),
                    pi=float(prem["pi"][0]), sig_agg=float(prem["sig_agg"][0]),
                    agg_rp=float(prem["agg_rp"][0]),
                    idio_rp=float(prem["idio_rp"][0]),
                    total_rp=float(prem["total_rp"][0]),
                    grid_mean_agg_rp=grid_rp)
        rows.append(row)
        if verbose:
            print(f"a={a:.4f} sigma={sigma:.4f} gamma={gamma:4.1f} tau={tau:5.2f}  "
                    f"agg_rp={row['agg_rp']:.4f}  idio_rp={row['idio_rp']:.4f}  "
                    f"grid_mean_agg_rp={grid_rp:.4f}")

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"[sweep] wrote {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nn", choices=["nn", "numerical"])
    # NN options
    parser.add_argument("--case", default="agents2")
    parser.add_argument("--config", default="timestep", choices=list(CONFIGS))
    parser.add_argument("--base-dir", default="./models/SV_NAgents_64bit_baseline_6.0_1.15_0.06_0.1")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    # numerical-solver parameter overrides (for risk-premium exploration)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None,
                        help="Poisson expert-retirement rate override")
    parser.add_argument("--sigv-mean", type=float, default=None)
    parser.add_argument("--vmean", type=float, default=0.25)
    parser.add_argument("--h", type=float, default=2e-4)
    parser.add_argument("--iters", type=int, default=300_000)
    # parameter-sweep mode (numerical solver): tabulate risk premium
    parser.add_argument("--sweep", action="store_true",
                        help="sweep sigma x gamma x tau and tabulate the risk premium")
    parser.add_argument("--a", default="0.1,0.2,0.5,1")
    parser.add_argument("--sigmas", default="0.0125,0.02,0.025,0.028,0.04,0.06")
    parser.add_argument("--gammas", default="5,6,8,10,15")
    parser.add_argument("--taus", default="1.15,1.5,2.0", help="comma list of tau values to sweep (default: baseline tau only)")
    parser.add_argument("--sweep-fixed", action="store_true",
                        help="evaluate the sweep at a fixed representative state "
                             "(x=0.5, v=mean) instead of the simulated stationary mean")
    parser.add_argument("--sweep-paths", type=int, default=50,
                        help="paths for the per-combination sweep simulation")
    parser.add_argument("--sweep-years", type=float, default=300.0,
                        help="horizon (years) for the per-combination sweep simulation")
    parser.add_argument("--sweep-out", default="./ditella_rp_sweep.csv")
    # simulation options
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--years", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--x0", type=float, default=0.2)
    parser.add_argument("--v0", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--portfolio", action="store_true",
                        help="plot risky-asset share (theta_k) of total wealth by wealth decile")
    parser.add_argument("--portfolio-terciles", action="store_true",
                        help="plot BOTH tercile views (balance-sheet theta/x + bounded "
                             "share-of-aggregate-capital), grouped by own-wealth capital share")
    parser.add_argument("--wealth-dist", action="store_true",
                        help="plot the cross-sectional wealth-share distribution and report "
                             "p95/median (+ p75/p25, p90/median, p99/median)")
    parser.add_argument("--wealth-dist-last", type=int, default=0,
                        help="pool only the last N simulation steps for --wealth-dist "
                             "(0 = use burn-in fraction; e.g. 500)")
    parser.add_argument("--irf", action="store_true",
                        help="deterministic impulse response to a -N*sigma aggregate/capital shock")
    parser.add_argument("--irf-sd", type=float, default=2.0,
                        help="shock size in std devs (default 2)")
    parser.add_argument("--irf-years", type=float, default=30.0,
                        help="IRF horizon in years after the shock")
    parser.add_argument("--irf-dt", type=float, default=0.05,
                        help="IRF deterministic-path time step")
    parser.add_argument("--irf-hold-v", action="store_true",
                        help="also compute/overlay the v-held-constant counterfactual "
                             "(shuts down the uncertainty channel; removes the overshoot)")
    parser.add_argument("--irf-plot-years", type=float, default=7.0,
                        help="cap the IRF plot x-axis at this many years since the shock")
    parser.add_argument("--irf-v-start", type=float, default=None,
                        help="override the ergodic v* as the IRF start/baseline v "
                             "(e.g. set to v_mean; clipped into the v-domain)")
    args = parser.parse_args()

    if args.float64:
        torch.set_default_dtype(torch.float64)

    if args.sweep:
        print("Computing sweeps")
        a_list = [float(a) for a in args.a.split(",")]
        sigmas = [float(s) for s in args.sigmas.split(",")]
        gammas = [float(g) for g in args.gammas.split(",")]
        taus = [float(t) for t in args.taus.split(",")] if args.taus else None
        df = sweep_risk_premium(
            a_list, sigmas, gammas, taus=taus, h=args.h, max_iters=args.iters,
            use_simulation=True,
            sim_kwargs=dict(n_paths=args.sweep_paths, years=args.sweep_years,
                            dt=args.dt, x0=args.x0, v0=args.v0, seed=args.seed),
            csv_path=args.sweep_out)
        print("\n[sweep] risk-premium table:")
        print(df.to_string(index=False))
        return

    if args.source == "nn":
        model = load_model(args.base_dir, args.case, args.config, width=args.width,
                           layers=args.layers, gamma=args.gamma, tau=args.tau,
                           sigma=args.sigma, a=args.a, vmean=args.vmean)
        economy = NNEconomy(model)
        subpath_name = args.case if args.tau == 1.15 else f"{args.case}_{args.tau}"
        if args.vmean != 0.25:
            subpath_name = f"{args.case}_{args.tau}_{args.vmean}"
        out_dir = os.path.join(args.base_dir, subpath_name, args.config, "simulation")
    else:
        overrides = {"sigma": args.sigma, "gamma": args.gamma, "phi": args.phi,
                     "tau": args.tau, "sigv_mean": args.sigv_mean}
        economy = build_numerical_economy(overrides, h=args.h, max_iters=args.iters)
        out_dir = "./ditella_numerical_simulation"
    sim = simulate(economy, n_paths=args.paths, years=args.years, dt=args.dt,
                   x0=args.x0, v0=args.v0, seed=args.seed)
    analyze(economy, sim, out_dir)
    if args.portfolio:
        plot_portfolio_deciles(economy, sim, out_dir)
    if args.portfolio_terciles:
        plot_portfolio_terciles(economy, sim, out_dir)
    if args.wealth_dist:
        plot_wealth_distribution(economy, sim, out_dir,
                                 last_steps=(args.wealth_dist_last or None))
    if args.irf:
        irf = impulse_response(economy, years_irf=args.irf_years, dt=args.irf_dt,
                               shock_sd=args.irf_sd, x0=args.x0, v0=args.v0,
                               seed=args.seed, include_vfixed=args.irf_hold_v,
                               v_start=args.irf_v_start)
        plot_impulse_response(economy, irf, out_dir, xmax=args.irf_plot_years)


if __name__ == "__main__":
    main()

'''
Example invocations (base-dir matches main.py's tagged output directory):

# comparative-statics sweep via the finite-difference solver (section 4.4)
python simulate.py --sweep --sigmas 0.0125,0.028,0.04 --gammas 5,10 --taus 0.5,1.15,2.0 --sweep-out ./ditella_rp_sweep.csv

# simulate + portfolio deciles + impulse response for the best 2-D model
python simulate.py --float64 --base-dir ./models/SV_NAgents_64bit_260713_t0frac0.4_6.0_1.15_0.06_0.1 --case agents2 --config timestep_rar --gamma 6.0 --a 0.1 --sigma 0.06 --tau 1.15 --portfolio --irf --irf-hold-v

# high-dimensional portfolio choice (section 4.2/4.4)
python simulate.py --float64 --base-dir ./models/SV_NAgents_64bit_260713_t0frac0.4_6.0_1.15_0.06_0.1 --case agents20 --config timestep_rar --gamma 6.0 --a 0.1 --sigma 0.06 --tau 1.15 --portfolio
'''