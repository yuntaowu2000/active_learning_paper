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
               gamma=None, tau=None, sigma=None, a=None):
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
    K, eidx, hidx, gamma_vec, caps_E = make_case(case, gamma)
    mpath = os.path.join(base_dir, case, config)
    if not os.path.exists(os.path.join(mpath, "model_best.pt")):
        raise FileNotFoundError(f"no trained checkpoint at {mpath}/model_best.pt -- train first.")
    print(f"[load_model] params from '{os.path.basename(os.path.normpath(base_dir))}': "
          f"gamma={gamma} tau={tau} sigma={sigma} a={a}")
    model = get_model(
        mpath, K, eidx, hidx, gamma_vec, caps_E,
        model_size=[width] * layers,
        timestepping=ts, rar=rar, loss_balancing=lb,
        params=BASE_PARAMS | {"tau": float(tau), "a": float(a), "sigma": float(sigma)},
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
                     include_vfixed=False):
    """Deterministic impulse response to a ``-shock_sd``-sigma aggregate shock.

    (1) Runs a long stochastic simulation and takes the post-burn-in mean state
    ``(x*, v*)`` as the stationary starting point.  (2) Evolves drift-only
    (deterministic) paths from ``(x*, v*)``: a *baseline* with no shock and a
    *shocked* path that injects a single Brownian increment at ``t = t_shock``,
    then lets the system mean-revert.  The difference is the IRF.

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
    v_star = float(sim["v_hist"][burn:].mean())
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
                           sigma=args.sigma, a=args.a, foc=args.foc)
        economy = NNEconomy(model)
        out_dir = os.path.join(args.base_dir, args.case, args.config, "simulation")
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
    if args.irf:
        irf = impulse_response(economy, years_irf=args.irf_years, dt=args.irf_dt,
                               shock_sd=args.irf_sd, x0=args.x0, v0=args.v0,
                               seed=args.seed, include_vfixed=args.irf_hold_v)
        plot_impulse_response(economy, irf, out_dir, xmax=args.irf_plot_years)


if __name__ == "__main__":
    main()

'''
python sv_n_agents_simulate.py --sweep  --sigmas 0.0125,0.028,0.04 --gammas 5,10 --taus 0.5,1.15,2.0  --sweep-out ./ditella_rp_sweep.csv
python sv_n_agents_simulate.py --float64 --base-dir ./models/SV_NAgents_64bit_6.0_1.15_0.06_0.1 --case agents2 --sigma 0.06 --a 0.1 --tau 1.15 --gamma 6.0 --config timestep_lb
python sv_n_agents_simulate.py --float64 --base-dir ./models/SV_NAgents_64bit_improved2_FOC_5.0_1.15_0.06_0.1 --case agents20 --sigma 0.06 --a 0.1 --tau 1.15 --gamma 6.0 --config timestep
python sv_n_agents_simulate.py --float64 --base-dir ./models/SV_NAgents_64bit_improved2_FOC_5.0_1.15_0.06_0.1 --case agents20 --sigma 0.06 --a 0.1 --tau 1.15 --gamma 6.0 --config timestep_rar --foc
'''