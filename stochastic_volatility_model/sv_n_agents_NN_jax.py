"""
sv_n_agents_NN_jax.py
=====================

**JAX port** of ``sv_n_agents_NN.py`` (the heterogeneous N-agent Di Tella (2017)
stochastic-volatility model).  This is a faithful re-implementation of the
*modelling and training* core in JAX/Optax, written so it can be read and
checked side-by-side with the PyTorch original.  It is **not meant to be run on
this Windows box** (no GPU speed-up there); it exists as a reference port.

What is reproduced (1-to-1 with the torch version)
--------------------------------------------------
* ``compute_sv_equilibrium`` -- the entire differentiable equilibrium forward
  pass (share-diffusion linear solve, price of risk, idiosyncratic risk / free
  boundary, HJBs per type, goods / asset-pricing / capital-FOC residuals).  Each
  line maps directly onto the torch version.
* The fused multi-network derivative trick (``StackedAgent``): here it is just a
  ``vmap`` over a stacked-parameter pytree composed with ``jax.grad`` /
  ``jax.hessian`` over the state input -- JAX makes this trivial.
* The wealth-share Dirichlet-alpha *mixture* sampler (log-uniform alpha so one
  batch spans egalitarian and concentrated wealth states).
* Both training modes: the stationary ``PDEModel`` (basic / RAR) and the
  backward ``PDEModelTimeStep`` march (time boundary condition + prev_vals
  update + per-outer-loop LR step decay + optimizer reset, exactly as the
  library does).
* Interior-FOC mode (``foc=True``) that hard-wires expert capital shares
  ``theta_k = (x_k/gamma_k)/sum_j(x_j/gamma_j)`` instead of using free theta
  networks.
* The loss reductions: equilibrium constraints use MSE; the HJB groups are the
  *mean* of ``sum_k hjb_k**2`` (i.e. MAE applied to an already-squared residual,
  == MSE of the raw residual), matching the original.
* ReLoBRaLo relative loss balancing with random lookback.

Assumptions / simplifications (documented, not silent)
------------------------------------------------------
* **Only MLPs** (Tanh hidden activations, Softplus on positive outputs), per the
  request.  KAN/DGM/ResNet layer types are dropped.
* Plotting, FD-comparison tables, welfare-equivalent tables and the simulator
  are NOT ported -- they only consume the forward pass, which is provided by
  ``forward_equilibrium`` and can be wrapped trivially (convert outputs with
  ``np.asarray``).
* The library's per-outer-loop "reload best checkpoint" bookkeeping is
  simplified: we keep the optimizer state marching continuously through the
  inner epochs of a slab and track the best parameters by total loss for the
  final return.  prev_vals are updated from the end-of-slab parameters.
* x64 is enabled globally (the torch models train in float64).

Dependencies: ``jax``, ``jaxlib``, ``optax``, ``numpy``.
"""
import argparse
import math
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

# The torch SV models run in double precision; mirror that here.
jax.config.update("jax_enable_x64", True)


# ===========================================================================
# Default economic parameters (original Di Tella calibration).
# ===========================================================================
BASE_PARAMS = {
    "a": 1.0, "sigma": 0.0125, "lbd": 1.38, "v_mean": 0.25, "sigv_mean": -0.17,
    "rho": 0.0665, "psi": 0.5, "tau": 1.15, "phi": 0.2,
    "A": 53.2, "B": -0.8668571428571438, "delta": 0.05,
}
V_DOMAIN = (0.05, 0.5)

# Dirichlet-alpha mixture range for wealth-share sampling (see _mixture_shares).
SHARE_ALPHA_LO = 0.05
SHARE_ALPHA_HI = 1.0


# ===========================================================================
# Plain-JAX MLP (Tanh hidden, optional Softplus output), as pytrees of (W, b).
# ===========================================================================
def init_net(key, in_dim: int, hidden_units: List[int], out_dim: int = 1):
    """Glorot-uniform MLP parameters: a list of (W, b) tuples."""
    sizes = [in_dim] + list(hidden_units) + [out_dim]
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i in range(len(sizes) - 1):
        din, dout = sizes[i], sizes[i + 1]
        lim = math.sqrt(6.0 / (din + dout))
        W = jax.random.uniform(keys[i], (din, dout), minval=-lim, maxval=lim)
        b = jnp.zeros((dout,))
        params.append((W, b))
    return params


def stack_nets(net_list):
    """Stack a list of identical-architecture net pytrees along a new leading
    axis (the JAX analogue of StackedAgent's _stack_module_state)."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *net_list)


def mlp_apply(params, x, positive: bool):
    """Forward a SINGLE sample ``x`` (D,) through one MLP -> scalar."""
    h = x
    for (W, b) in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    out = h @ W + b                      # (out_dim,) == (1,)
    if positive:
        out = jax.nn.softplus(out)
    return out[0]                        # scalar


# ---- single-network batched value / jacobian / hessian over the state --------
def net_value(params, SV, positive):
    return jax.vmap(lambda x: mlp_apply(params, x, positive))(SV)            # (B,)


def net_value_jac_hess(params, SV, positive):
    f = lambda x: mlp_apply(params, x, positive)
    val = jax.vmap(f)(SV)                                                     # (B,)
    jac = jax.vmap(jax.grad(f))(SV)                                          # (B,D)
    hess = jax.vmap(jax.hessian(f))(SV)                                      # (B,D,D)
    return val, jac, hess


# ---- stacked (multi-network) versions: vmap over the stacked leading axis -----
def stacked_value(stacked_params, SV, positive):
    """(B, N) values from N stacked scalar nets."""
    return jax.vmap(lambda p: net_value(p, SV, positive))(stacked_params).T


def stacked_value_jac_hess(stacked_params, SV, positive):
    """(B,N), (B,N,D), (B,N,D,D) from N stacked scalar nets."""
    def per_net(p):
        return net_value_jac_hess(p, SV, positive)
    vals, jacs, hess = jax.vmap(per_net)(stacked_params)        # (N,B),(N,B,D),(N,B,D,D)
    return (vals.T,
            jnp.transpose(jacs, (1, 0, 2)),
            jnp.transpose(hess, (1, 0, 2, 3)))


# ===========================================================================
# Parameter container: one stacked group of xi nets, optional stacked theta
# nets, and single p / r nets.  (FOC mode -> no theta nets.)
# ===========================================================================
def init_params(key, K: int, n_theta: int, D: int, hidden_units: List[int]):
    """Initialise all networks.  Returns a dict pytree:
        {"xi": stacked (K nets), "theta": stacked (n_theta nets) or None,
         "p": single net, "r": single net}.
    """
    keys = jax.random.split(key, K + n_theta + 2)
    xi_nets = [init_net(keys[i], D, hidden_units) for i in range(K)]
    params = {"xi": stack_nets(xi_nets)}
    if n_theta > 0:
        theta_nets = [init_net(keys[K + i], D, hidden_units) for i in range(n_theta)]
        params["theta"] = stack_nets(theta_nets)
    else:
        params["theta"] = None
    params["p"] = init_net(keys[K + n_theta], D, hidden_units)
    params["r"] = init_net(keys[K + n_theta + 1], D, hidden_units)
    return params


# ===========================================================================
# Statics (economic constants + indices).  Indices kept as concrete numpy /
# python for static control flow; gamma/caps as jnp arrays for the math.
# ===========================================================================
def build_statics(K, expert_idx, household_idx, gamma_vec, caps_E, has_t,
                  params=BASE_PARAMS):
    n_E = len(expert_idx)
    D = K + 1 if has_t else K
    statics = {
        "K": K, "D": D, "n_E": n_E,
        "expert_idx": jnp.asarray(expert_idx, dtype=jnp.int32),
        "household_idx": jnp.asarray(household_idx, dtype=jnp.int32),
        "expert_idx_py": list(expert_idx),
        "household_idx_py": list(household_idx),
        "v_index": K - 1, "has_t": has_t,
        "gamma": jnp.asarray(gamma_vec, dtype=jnp.float64).reshape(1, K),
        "caps_E": jnp.asarray(caps_E, dtype=jnp.float64).reshape(1, n_E),
        "v_domain": V_DOMAIN,
    }
    for k, val in params.items():
        statics[k] = val
    return statics


# ===========================================================================
# Economic core (verbatim translation of the torch compute_sv_equilibrium).
# ===========================================================================
def compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E,
                           r, statics):
    """All shapes batched over B.  See the torch original for full annotation.

    SV (B,D); xi (B,K); xi_Jac (B,K,D); xi_Hess (B,K,D,D);
    p (B,1); p_Jac (B,1,D); p_Hess (B,1,D,D); theta_E (B,n_E); r (B,1).
    """
    K = statics["K"]
    D = statics["D"]
    v_index = statics["v_index"]
    has_t = statics["has_t"]
    eidx = statics["expert_idx"]
    hidx = statics["household_idx"]
    gamma = statics["gamma"]               # (1, K)
    caps_E = statics["caps_E"]             # (1, n_E)

    rho = statics["rho"]; psi = statics["psi"]; tau = statics["tau"]
    phi = statics["phi"]; sigma = statics["sigma"]
    lbd = statics["lbd"]; v_mean = statics["v_mean"]; sigv_mean = statics["sigv_mean"]
    A = statics["A"]; Bc = statics["B"]; delta = statics["delta"]; a = statics["a"]

    B_ = SV.shape[0]

    # ---- shares (full vector incl. residual x_K) and v --------------------
    x_states = SV[:, :K - 1]                                  # (B, K-1)
    x_K = 1.0 - x_states.sum(axis=1, keepdims=True)           # (B, 1)
    x_full = jnp.concatenate([x_states, x_K], axis=1)         # (B, K)
    v = SV[:, v_index:v_index + 1]                            # (B, 1)

    # ---- aggregate (capital) block ----------------------------------------
    g = (p - Bc) / (2.0 * A) - delta
    iota = A * (g + delta) ** 2 + Bc * (g + delta)
    mu_v = lbd * (v_mean - v)
    sig_v = sigv_mean * jnp.sqrt(v)
    chat = rho ** (1.0 / psi) * xi ** ((psi - 1.0) / psi)     # (B, K)

    # ---- share-diffusion linear system ------------------------------------
    g_arr = gamma                                            # (1, K)
    gm1_over_g = (g_arr - 1.0) / g_arr                       # (1, K)
    inv_g = 1.0 / g_arr                                      # (1, K)

    xi_v = xi_Jac[:, :, v_index]                             # (B, K)
    xi_x = xi_Jac[:, :, :K - 1]                              # (B, K, K-1)
    a_k = xi_v * sig_v / xi                                  # (B, K)
    b_k = xi_x / xi[:, :, None]                              # (B, K, K-1)

    p_v = p_Jac[:, 0, v_index:v_index + 1]                   # (B, 1)
    p_x = p_Jac[:, 0, :K - 1]                                # (B, K-1)
    a_p = p_v * sig_v / p                                    # (B, 1)
    b_p = p_x / p                                            # (B, K-1)

    P0 = sigma + a_p                                         # (B, 1)
    coeff = x_full * gm1_over_g                              # (B, K)
    S0 = (coeff * a_k).sum(axis=1, keepdims=True)           # (B, 1)
    S_m = jnp.einsum("bk,bkm->bm", coeff, b_k)             # (B, K-1)
    T = (x_full * inv_g).sum(axis=1, keepdims=True)         # (B, 1)
    pi0 = (P0 + S0) / T                                     # (B, 1)
    pi_m = (b_p + S_m) / T                                  # (B, K-1)

    xs = x_states                                           # (B, K-1)
    gs = g_arr[:, :K - 1]                                   # (1, K-1)
    b_ks = b_k[:, :K - 1, :]                                # (B, K-1, K-1)
    term1 = (xs / gs)[:, :, None] * pi_m[:, None, :]               # (B,K-1,K-1)
    term2 = (xs * (gs - 1.0) / gs)[:, :, None] * b_ks             # (B,K-1,K-1)
    term3 = xs[:, :, None] * b_p[:, None, :]                      # (B,K-1,K-1)
    M = term1 - term2 - term3
    c = (xs / gs) * pi0 - (xs * (gs - 1.0) / gs) * a_k[:, :K - 1] - xs * P0   # (B,K-1)

    Imat = jnp.eye(K - 1, dtype=SV.dtype)[None, :, :]
    u = jnp.linalg.solve(Imat - M, c[:, :, None])[:, :, 0]        # (B, K-1)
    sigx_full = jnp.concatenate([u, -u.sum(axis=1, keepdims=True)], axis=1)   # (B, K)

    # ---- recompute diffusions from the solved u ---------------------------
    sigp = a_p + (b_p * u).sum(axis=1, keepdims=True)             # (B, 1)
    sig_agg = sigma + sigp                                        # (B, 1)
    sigxi = a_k + jnp.einsum("bkm,bm->bk", b_k, u)              # (B, K)
    S_full = (coeff * sigxi).sum(axis=1, keepdims=True)          # (B, 1)
    pi = (sig_agg + S_full) / T                                  # (B, 1)
    sign_k = pi * inv_g - gm1_over_g * sigxi                     # (B, K)

    # ---- capital allocation, idiosyncratic risk, free boundary ------------
    theta_full = jnp.zeros((B_, K), dtype=SV.dtype).at[:, eidx].set(theta_E)
    x_E = x_full[:, eidx]                                        # (B, n_E)
    g_E = g_arr[:, eidx]                                         # (1, n_E)
    phiv = phi * v                                               # (B, 1)
    phiv2 = phiv ** 2                                            # (B, 1)
    sigtilde_E = phiv * theta_E / x_E                            # (B, n_E)

    chi = g_E[:, 0:1] * phiv2 * theta_E[:, 0:1] / x_E[:, 0:1]    # (B, 1)
    theta_star_E = chi * x_E / (g_E * phiv2)                     # (B, n_E)
    vi_expert_resid = jnp.minimum(caps_E * x_E - theta_E, theta_star_E - theta_E)

    sigtilde_full = jnp.zeros((B_, K), dtype=SV.dtype).at[:, eidx].set(sigtilde_E)
    chi_theta_over_x_full = jnp.zeros((B_, K), dtype=SV.dtype).at[:, eidx].set(
        chi * theta_E / x_E)

    # ---- goods-market clearing residual -----------------------------------
    goods_resid = (a - iota) - p * (x_full * chat).sum(axis=1, keepdims=True)

    # ---- share drifts (r-independent) -------------------------------------
    mu_net0 = pi * sign_k + chi_theta_over_x_full               # (B, K)
    agg_cons = (a - iota) / p                                   # (B, 1)
    mu_N0_ = pi * sig_agg + chi - agg_cons                      # (B, 1)
    mu_x_full = x_full * ((mu_net0 - chat) - mu_N0_ - (sign_k - sig_agg) * sig_agg)

    # retirement transfers: experts -> households (pro-rata by household share)
    X_E = x_full[:, eidx].sum(axis=1, keepdims=True)           # (B, 1)
    X_H = x_full[:, hidx].sum(axis=1, keepdims=True)           # (B, 1)
    retire = jnp.zeros((B_, K), dtype=SV.dtype)
    retire = retire.at[:, eidx].set(-tau * x_full[:, eidx])
    retire = retire.at[:, hidx].set(tau * X_E * (x_full[:, hidx] / X_H))
    mu_x_full = mu_x_full + retire
    mu_x_states = mu_x_full[:, :K - 1]                          # (B, K-1)

    # ---- state drift / diffusion vectors ----------------------------------
    mu_s = jnp.zeros((B_, D), dtype=SV.dtype)
    mu_s = mu_s.at[:, :K - 1].set(mu_x_states)
    mu_s = mu_s.at[:, v_index].set(mu_v[:, 0])
    if has_t:
        mu_s = mu_s.at[:, D - 1].set(1.0)                       # d/dt coefficient
    sig_s = jnp.zeros((B_, D), dtype=SV.dtype)
    sig_s = sig_s.at[:, :K - 1].set(u)
    sig_s = sig_s.at[:, v_index].set(sig_v[:, 0])

    # ---- mu_xi (Ito) and mu_P --------------------------------------------
    mu_xi = (jnp.einsum("bd,bkd->bk", mu_s, xi_Jac)
             + 0.5 * jnp.einsum("bd,bkde,be->bk", sig_s, xi_Hess, sig_s)) / xi
    mu_P = (jnp.einsum("bd,bd->b", mu_s, p_Jac[:, 0, :])[:, None]
            + 0.5 * jnp.einsum("bd,bde,be->b", sig_s, p_Hess[:, 0], sig_s)[:, None]) / p

    # ---- risk-free rate: free network + asset-pricing residual ------------
    r_implied = ((a - iota) / p + g + mu_P + sigma * sigp - sig_agg * pi - chi)
    asset_pricing_resid = r_implied - r                         # (B, 1)
    sig_clearing_resid = (sigma + sigp) - (x_full * sign_k).sum(axis=1, keepdims=True)

    # ---- HJB per type -----------------------------------------------------
    mu_net = r + pi * sign_k + chi_theta_over_x_full           # (B, K)
    xi_H = xi[:, hidx]                                          # (B, n_H)
    x_H = x_full[:, hidx]                                       # (B, n_H)
    xi_ret = (x_H * xi_H).sum(axis=1, keepdims=True) / X_H      # (B, 1)

    hjb_common = (chat ** (1.0 - psi) / (1.0 - psi) * rho * xi ** (psi - 1.0)
                  + mu_net - chat + mu_xi
                  - g_arr / 2.0 * (sign_k ** 2 + sigxi ** 2 + 2.0 * gm1_over_g * sign_k * sigxi)
                  - rho / (1.0 - psi))                          # (B, K)
    retire_term = tau / (1.0 - g_arr) * ((xi_ret / xi) ** (1.0 - g_arr) - 1.0)
    idio_pen = -g_arr / 2.0 * sigtilde_full ** 2               # (B, K)
    is_expert = jnp.zeros((1, K), dtype=SV.dtype).at[0, eidx].set(1.0)
    hjb_k = hjb_common + is_expert * (retire_term + idio_pen)   # (B, K)

    hjb_expert = (hjb_k[:, eidx] ** 2).sum(axis=1, keepdims=True)        # (B, 1)
    hjb_household = (hjb_k[:, hidx] ** 2).sum(axis=1, keepdims=True)     # (B, 1)

    risk_premium = sig_agg * pi

    return {
        "x_full": x_full, "theta_full": theta_full, "chat": chat,
        "xi_active": xi,
        "sigx_full": sigx_full, "sigp": sigp, "sig_agg": sig_agg,
        "sigxi": sigxi, "pi": pi, "sign_k": sign_k, "chi": chi,
        "mu_x_full": mu_x_full, "mu_xi": mu_xi, "mu_P": mu_P, "r": r, "mu_net": mu_net,
        "r_implied": r_implied, "asset_pricing_resid": asset_pricing_resid,
        "hjb_k": hjb_k, "hjb_expert": hjb_expert, "hjb_household": hjb_household,
        "goods_resid": goods_resid, "sig_clearing_resid": sig_clearing_resid,
        "vi_expert_resid": vi_expert_resid, "xi_ret": xi_ret,
        "g": g, "iota": iota, "sigtilde_full": sigtilde_full,
        "risk_premium": risk_premium,
    }


# ===========================================================================
# Expert capital shares (FOC or anchor-residual parameterisation).
# ===========================================================================
def compute_theta_E(params, SV, x_full, statics, foc: bool):
    if foc:
        eidx = statics["expert_idx"]
        g_E = jnp.take(statics["gamma"][0], eidx)[None, :]      # (1, n_E)
        w = x_full[:, eidx] / g_E                              # (B, n_E)
        return w / w.sum(axis=1, keepdims=True)

    if params["theta"] is not None:
        theta_others = stacked_value(params["theta"], SV, True)        # (B, n_E-1)
        theta_anchor = 1.0 - theta_others.sum(axis=1, keepdims=True)   # (B, 1)
        return jnp.concatenate([theta_anchor, theta_others], axis=1)   # (B, n_E)
    # single expert -> holds all capital
    return jnp.ones((SV.shape[0], 1), dtype=SV.dtype)


def forward_equilibrium(params, SV, statics, foc: bool):
    """The JAX analogue of update_variables: stacked forward -> equilibrium dict."""
    K = statics["K"]
    xi, xi_Jac, xi_Hess = stacked_value_jac_hess(params["xi"], SV, True)
    p, p_Jac, p_Hess = net_value_jac_hess(params["p"], SV, True)
    p = p[:, None]; p_Jac = p_Jac[:, None, :]; p_Hess = p_Hess[:, None, :, :]
    r = net_value(params["r"], SV, False)[:, None]
    x_states = SV[:, :K - 1]
    x_full = jnp.concatenate([x_states, 1.0 - x_states.sum(axis=1, keepdims=True)], axis=1)
    theta_E = compute_theta_E(params, SV, x_full, statics, foc)
    return compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess,
                                  theta_E, r, statics)


# ===========================================================================
# Loss components.  Constraints use MSE; HJB groups use mean(sum_k hjb_k**2)
# (== MAE applied to a pre-squared residual == MSE of the raw residual).
# ===========================================================================
def loss_components(params, SV, statics, foc: bool):
    out = forward_equilibrium(params, SV, statics, foc)
    comps = {
        "goods": jnp.mean(out["goods_resid"] ** 2),
        "asset_pricing": jnp.mean(out["asset_pricing_resid"] ** 2),
        "sig_clearning": jnp.mean(out["sig_clearing_resid"] ** 2),
        "expert": jnp.mean(out["hjb_expert"]),
        "household": jnp.mean(out["hjb_household"]),
    }
    if statics["K"] > 2:
        comps["vi_expert"] = jnp.mean(out["vi_expert_resid"] ** 2)
    return comps


def residual_score(params, SV, statics, foc: bool):
    """Per-point residual magnitude (for RAR ranking): sum of |endog| + |hjb|."""
    out = forward_equilibrium(params, SV, statics, foc)
    Bn = SV.shape[0]
    total = jnp.zeros((Bn, 1))
    total = total + jnp.abs(out["goods_resid"])
    total = total + jnp.abs(out["asset_pricing_resid"])
    total = total + jnp.abs(out["sig_clearing_resid"])
    if statics["K"] > 2:
        total = total + jnp.abs(out["vi_expert_resid"]).mean(axis=1, keepdims=True)
    total = total + jnp.abs(out["hjb_expert"])
    total = total + jnp.abs(out["hjb_household"])
    return total[:, 0]


# ===========================================================================
# Samplers (wealth-share Dirichlet-alpha mixture + uniform v [+ uniform t]).
# ===========================================================================
def _mixture_shares_jax(key, n, K, eps, alpha_lo=SHARE_ALPHA_LO,
                        alpha_hi=SHARE_ALPHA_HI):
    """(n, K) eps-floored simplex shares; per-row log-uniform Dirichlet alpha."""
    ku, kg = jax.random.split(key)
    u = jax.random.uniform(ku, (n, 1))
    alpha = jnp.exp(math.log(alpha_lo) + (math.log(alpha_hi) - math.log(alpha_lo)) * u)
    conc = jnp.broadcast_to(alpha, (n, K))
    gam = jax.random.gamma(kg, conc)                            # Gamma(alpha, 1)
    shares = gam / gam.sum(axis=1, keepdims=True)
    return eps + (1.0 - K * eps) * shares


def _mixture_shares_np(n, K, eps, rng, alpha_lo=SHARE_ALPHA_LO, alpha_hi=SHARE_ALPHA_HI):
    """NumPy counterpart for reproducible (seeded) evaluation sampling."""
    u = rng.random((n, 1))
    alpha = np.exp(np.log(alpha_lo) + (np.log(alpha_hi) - np.log(alpha_lo)) * u)
    gam = rng.gamma(np.broadcast_to(alpha, (n, K)), 1.0)
    shares = gam / gam.sum(axis=1, keepdims=True)
    return eps + (1.0 - K * eps) * shares


def sample_simplex_v(key, batch, statics, alpha_lo, alpha_hi):
    K = statics["K"]
    eps = 0.1 / K
    ks, kv = jax.random.split(key)
    shares = _mixture_shares_jax(ks, batch, K, eps, alpha_lo, alpha_hi)
    x_states = shares[:, :K - 1]
    vlo, vhi = statics["v_domain"]
    v = vlo + (vhi - vlo) * jax.random.uniform(kv, (batch, 1))
    return jnp.concatenate([x_states, v], axis=1)


def sample_simplex_v_t(key, batch, statics, alpha_lo, alpha_hi, min_t, max_t):
    kb, kt = jax.random.split(key)
    base = sample_simplex_v(kb, batch, statics, alpha_lo, alpha_hi)
    t = min_t + (max_t - min_t) * jax.random.uniform(kt, (base.shape[0], 1))
    return jnp.concatenate([base, t], axis=1)


# ===========================================================================
# ReLoBRaLo loss balancing (relative loss balancing with random lookback).
# State is kept host-side (small) and updated every loss_log_interval steps.
# ===========================================================================
class ReLoBRaLo:
    def __init__(self, labels, alpha=0.999, temp=0.1, bernoulli_prob=0.9999,
                 log_interval=50, seed=0):
        self.labels = list(labels)
        self.n = len(self.labels)
        self.alpha = alpha
        self.temp = temp
        self.bernoulli_prob = bernoulli_prob
        self.log_interval = log_interval
        self.weights = {k: 1.0 for k in self.labels}
        self.rng = np.random.default_rng(seed)
        self.init_loss = None
        self.prev_loss = None

    @staticmethod
    def _softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def weight_vector(self):
        return np.array([self.weights[k] for k in self.labels])

    def step(self, epoch, comp_dict):
        curr = np.array([float(comp_dict[k]) for k in self.labels])
        curr = np.where(np.isnan(curr), np.finfo(np.float64).eps, curr)
        if epoch == 0:
            self.init_loss = curr.copy()
            self.prev_loss = curr.copy()
            return
        if epoch % self.log_interval != 0:
            return
        ratio_prev = curr / (self.temp * self.prev_loss + 1e-8)
        ratio_zero = curr / (self.temp * self.init_loss + 1e-8)
        bal_prev = self.n * self._softmax(ratio_prev)
        bal_zero = self.n * self._softmax(ratio_zero)
        rho = float(self.rng.random() < self.bernoulli_prob)
        prev_w = self.weight_vector()
        weight_hist = rho * prev_w + (1.0 - rho) * bal_zero
        new_w = self.alpha * weight_hist + (1.0 - self.alpha) * bal_prev
        for i, k in enumerate(self.labels):
            self.weights[k] = float(new_w[i])
        self.prev_loss = curr.copy()


# ===========================================================================
# Incremental CSV logger (writes a header lazily, flushes every row so the loss
# history survives an interrupted run).
# ===========================================================================
class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.f = None
        self.cols = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, row: dict):
        if not self.path:
            return
        if self.f is None:
            self.cols = list(row.keys())
            self.f = open(self.path, "w")
            self.f.write(",".join(self.cols) + "\n")
        self.f.write(",".join(str(row.get(c, "")) for c in self.cols) + "\n")
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


# ===========================================================================
# Optimizer (global-norm grad clipping + Adam, matching the mixin closure).
# ===========================================================================
def make_optimizer(lr):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))


def _build_step(statics, foc, labels, optimizer):
    """Return a jitted (params, opt_state, SV, weights)-> step for the stationary
    model.  ``labels`` fixes the order of the weight vector."""

    def total_and_comps(params, SV, weights):
        comps = loss_components(params, SV, statics, foc)
        comp_vec = jnp.stack([comps[l] for l in labels])
        return jnp.sum(weights * comp_vec), comp_vec

    @jax.jit
    def step(params, opt_state, SV, weights):
        (total, comp_vec), grads = jax.value_and_grad(
            total_and_comps, has_aux=True)(params, SV, weights)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total, comp_vec

    return step


# ===========================================================================
# Stationary training (basic / RAR).
# ===========================================================================
def train_basic(params, statics, foc, cfg, key, history_csv=None):
    labels = list(loss_components(
        params, sample_simplex_v(key, 2, statics, cfg["alpha_lo"], cfg["alpha_hi"]),
        statics, foc).keys())
    optimizer = make_optimizer(cfg["lr"])
    opt_state = optimizer.init(params)
    step = _build_step(statics, foc, labels, optimizer)

    lb = ReLoBRaLo(labels, cfg["lb_alpha"], cfg["lb_temp"], cfg["bernoulli_prob"],
                   cfg["loss_log_interval"]) if cfg["loss_balancing"] else None

    n_epochs = cfg["num_epochs"]
    batch = cfg["batch_size"]
    rar = cfg["rar"]
    refinement_rounds = cfg["refinement_rounds"]
    log_interval = cfg["loss_log_interval"]
    anchors = None
    weights = jnp.ones(len(labels))
    best = (math.inf, params)
    history = []
    logger = CSVLogger(history_csv)

    pbar = tqdm(range(n_epochs), desc="basic", dynamic_ncols=True)
    for epoch in pbar:
        key, ks = jax.random.split(key)
        SV = sample_simplex_v(ks, batch, statics, cfg["alpha_lo"], cfg["alpha_hi"])

        if rar and epoch > 0 and epoch % max(1, n_epochs // refinement_rounds) == 0:
            key, kp = jax.random.split(key)
            pool = sample_simplex_v(kp, batch * refinement_rounds, statics,
                                    cfg["alpha_lo"], cfg["alpha_hi"])
            scores = residual_score(jax.lax.stop_gradient(params), pool, statics, foc)
            k_keep = batch // refinement_rounds
            top = jnp.argsort(scores)[-k_keep:]            # stays on device (no host sync)
            anchors = pool[top] if anchors is None else jnp.concatenate([anchors, pool[top]], 0)

        SV_train = SV if anchors is None else jnp.concatenate([SV, anchors], 0)
        params, opt_state, total, comp_vec = step(params, opt_state, SV_train, weights)

        # The ONLY host sync: at log steps we pull the scalars once, then do
        # best-tracking / loss-balancing / history / progress-bar.  Everything
        # else stays on device so JAX can pipeline the steps in between.
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            total_f = float(total)
            comps_f = {l: float(comp_vec[i]) for i, l in enumerate(labels)}
            if lb is not None:
                lb.step(epoch, comps_f)
                weights = jnp.asarray(lb.weight_vector())
            if total_f < best[0]:
                best = (total_f, params)
            row = {"epoch": epoch, "total_loss": total_f, **comps_f}
            history.append(row)
            logger.log(row)
            pbar.set_postfix(total=f"{total_f:.2e}",
                             E=f"{comps_f.get('expert', float('nan')):.2e}",
                             H=f"{comps_f.get('household', float('nan')):.2e}")

    logger.close()
    return best[1], history


# ===========================================================================
# Time-stepping training (backward parabolic march).
# ===========================================================================
def _ts_boundary_loss(params, bd_pts, prev_vals, statics):
    """Sum of MSE time-boundary residuals over xi / p / r (+ theta) networks.
    Pins network(bd_pts at t=max_t) to prev_vals (the previous slab's t=min_t)."""
    xi_bd = stacked_value(params["xi"], bd_pts, True)              # (Bd, K)
    p_bd = net_value(params["p"], bd_pts, True)[:, None]          # (Bd, 1)
    r_bd = net_value(params["r"], bd_pts, False)[:, None]         # (Bd, 1)
    loss = (jnp.mean((xi_bd - prev_vals["xi"]) ** 2)
            + jnp.mean((p_bd - prev_vals["p"]) ** 2)
            + jnp.mean((r_bd - prev_vals["r"]) ** 2))
    if params["theta"] is not None:
        theta_bd = stacked_value(params["theta"], bd_pts, True)
        loss = loss + jnp.mean((theta_bd - prev_vals["theta"]) ** 2)
    return loss


def _boundary_network_values(params, bd_pts):
    """Evaluate the tracked networks at bd_pts -> prev_vals dict."""
    vals = {
        "xi": stacked_value(params["xi"], bd_pts, True),
        "p": net_value(params["p"], bd_pts, True)[:, None],
        "r": net_value(params["r"], bd_pts, False)[:, None],
    }
    if params["theta"] is not None:
        vals["theta"] = stacked_value(params["theta"], bd_pts, True)
    return vals


def _build_ts_step(statics, foc, labels, optimizer):
    """Jitted time-step inner step including the time-boundary loss term."""

    def total_and_comps(params, SV, weights, bd_pts, prev_vals):
        comps = loss_components(params, SV, statics, foc)
        comps["time_boundary"] = _ts_boundary_loss(params, bd_pts, prev_vals, statics)
        comp_vec = jnp.stack([comps[l] for l in labels])
        return jnp.sum(weights * comp_vec), comp_vec

    @jax.jit
    def step(params, opt_state, SV, weights, bd_pts, prev_vals):
        (total, comp_vec), grads = jax.value_and_grad(
            total_and_comps, has_aux=True)(params, SV, weights, bd_pts, prev_vals)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total, comp_vec

    return step


def train_timestep(params, statics, foc, cfg, key, init_guess=None, history_csv=None):
    """Backward time-march: each outer iteration solves a [min_t, max_t] slab
    with the terminal (t=max_t) value pinned to the previous slab's t=min_t
    solution, then reads off the new t=min_t solution.  LR step-decays and the
    Adam state is reset at the start of each outer iteration (as in the library).
    """
    min_t, max_t = cfg["min_t"], cfg["max_t"]
    batch = cfg["batch_size"]

    # fixed boundary grid (shares + v), reused every outer loop
    key, kb = jax.random.split(key)
    bd_base = sample_simplex_v(kb, batch, statics, cfg["alpha_lo"], cfg["alpha_hi"])
    bd_pts_max = jnp.concatenate([bd_base, jnp.full((batch, 1), max_t)], axis=1)
    bd_pts_min = jnp.concatenate([bd_base, jnp.full((batch, 1), min_t)], axis=1)

    # prev_vals: constant init guess at the boundary grid
    ig = {"xi": 1.0, "p": 1.0, "r": 1.0, "theta": 1.0}
    if init_guess:
        ig.update(init_guess)
    K, n_E = statics["K"], statics["n_E"]
    prev_vals = {
        "xi": jnp.full((batch, K), ig["xi"]),
        "p": jnp.full((batch, 1), ig["p"]),
        "r": jnp.full((batch, 1), ig["r"]),
    }
    if params["theta"] is not None:
        prev_vals["theta"] = jnp.full((batch, n_E - 1), ig["theta"])

    labels = list(loss_components(params, bd_pts_max, statics, foc).keys()) + ["time_boundary"]

    best = (math.inf, params)
    history = []
    logger = CSVLogger(history_csv)

    for outer in range(cfg["num_outer"]):
        # LR step decay + fresh Adam state (matches library outer-loop reset)
        factor = cfg["lr_decay_gamma"] ** (outer // cfg["lr_decay_every"]) \
            if cfg["lr_decay_every"] > 0 else 1.0
        lr = cfg["lr"] * factor
        optimizer = make_optimizer(lr)
        opt_state = optimizer.init(params)
        step = _build_ts_step(statics, foc, labels, optimizer)

        lb = ReLoBRaLo(labels, cfg["lb_alpha"], cfg["lb_temp"], cfg["bernoulli_prob"],
                       cfg["loss_log_interval"]) if cfg["loss_balancing"] else None
        weights = jnp.ones(len(labels))

        # decaying inner-iteration budget, as in the library
        num_inner = max(int(cfg["num_inner"] / math.sqrt(outer + 1)), cfg["min_inner"])
        log_interval = cfg["loss_log_interval"]
        anchors = None

        pbar = tqdm(range(num_inner), desc=f"ts outer {outer:3d} (lr={lr:.1e})", leave=False, dynamic_ncols=True)
        for epoch in pbar:
            key, ks = jax.random.split(key)
            SV = sample_simplex_v_t(ks, batch, statics, cfg["alpha_lo"], cfg["alpha_hi"],
                                    min_t, max_t)

            if cfg["rar"] and epoch > 0 and epoch % max(1, num_inner // cfg["refinement_rounds"]) == 0:
                key, kp = jax.random.split(key)
                pool = sample_simplex_v_t(kp, batch * cfg["refinement_rounds"], statics,
                                          cfg["alpha_lo"], cfg["alpha_hi"], min_t, max_t)
                scores = residual_score(jax.lax.stop_gradient(params), pool, statics, foc)
                k_keep = batch // cfg["refinement_rounds"]
                top = jnp.argsort(scores)[-k_keep:]
                anchors = pool[top] if anchors is None else jnp.concatenate([anchors, pool[top]], 0)

            SV_train = SV if anchors is None else jnp.concatenate([SV, anchors], 0)
            params, opt_state, total, comp_vec = step(
                params, opt_state, SV_train, weights, bd_pts_max, prev_vals)

            # single host sync per log step (see train_basic note)
            if epoch % log_interval == 0 or epoch == num_inner - 1:
                total_f = float(total)
                comps_f = {l: float(comp_vec[i]) for i, l in enumerate(labels)}
                if lb is not None:
                    lb.step(epoch, comps_f)
                    weights = jnp.asarray(lb.weight_vector())
                if total_f < best[0]:
                    best = (total_f, params)
                history.append({"outer": outer, "epoch": epoch, "total_loss": total_f, **comps_f})
                logger.log({"outer": outer, "epoch": epoch, "total_loss": total_f, **comps_f})
                pbar.set_postfix(total=f"{total_f:.2e}",
                                 E=f"{comps_f.get('expert', float('nan')):.2e}",
                                 bc=f"{comps_f.get('time_boundary', float('nan')):.2e}")

        # march: next slab's terminal (t=max_t) target = this slab's t=min_t value
        prev_vals = _boundary_network_values(params, bd_pts_min)
        # convergence diagnostic (mean p at t=0), printed above the bars
        tqdm.write(f"[ts outer {outer:3d}] done  mean p(t=0)={float(jnp.mean(prev_vals['p'])):.4f}  "
                   f"best total={best[0]:.3e}")

    logger.close()
    return best[1], history


# ===========================================================================
# Cases (identical calibration to the torch version).
# ===========================================================================
def make_case(case: str, gamma):
    """Return (K, expert_idx, household_idx, gamma_vec, caps_E)."""
    if case == "agents2":
        K = 2
        expert_idx = [0]; household_idx = [1]
        gamma_vec = [gamma, gamma]
        caps_E = [1e6]
    elif case == "agents5":
        K = 5
        expert_idx = [0, 1, 2, 3]; household_idx = [4]
        gamma_vec = [5.5, 6.0, 6.5, 7.0] + [8.0]
        caps_E = [1e6, 1e6, 1e6, 1e6]
    elif case == "agents5_cap":
        K = 5
        expert_idx = [0, 1, 2, 3]; household_idx = [4]
        gamma_vec = [5.5, 6.0, 6.5, 7.0] + [8.0]
        caps_E = [1e6, 0.1, 0.08, 0.05]
    elif case == "agents5_cap2":
        K = 5
        expert_idx = [0, 1, 2, 3]; household_idx = [4]
        gamma_vec = [5.5, 6.0, 6.5, 7.0] + [8.0]
        caps_E = [1e6, 0.08, 1e6, 1e6]
    elif case == "agents20":
        K = 20
        n_E = 18
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 7.0 * i / (n_E - 1) for i in range(n_E)] + [12.0, 14.0]
        caps_E = [1e6] * n_E
    elif case == "agents50":
        K = 50
        n_E = 45
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 8.0 * i / (n_E - 1) for i in range(n_E)] + [12.0, 13.0, 14.0, 15.0, 16.0]
        caps_E = [1e6] * n_E
    else:
        raise ValueError(f"unknown case {case!r}")
    return K, expert_idx, household_idx, gamma_vec, caps_E


# config name -> (timestepping, rar, loss_balancing)
CONFIGS = {
    "basic": (False, False, False),
    "basic_rar": (False, True, False),
    "timestep": (True, False, False),
    "timestep_rar": (True, True, False),
    "timestep_lb": (True, False, True),
}


# ===========================================================================
# Model assembly + train/save/load.
# ===========================================================================
def save_params(path, params, statics_meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"params": jax.device_get(params), "meta": statics_meta}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_params(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return params, payload["meta"]


def get_model(model_path, K, expert_idx, household_idx, gamma_vec, caps_E,
              model_size, n_epochs=20000, batch_size=500, lr=1e-3,
              timestepping=False, rar=False, loss_balancing=False,
              params=BASE_PARAMS, train=True, num_outer=70, num_inner=5000,
              min_inner=1000, loss_log_interval=50, max_t=1.0, init_guess=None,
              share_alpha_lo=SHARE_ALPHA_LO, share_alpha_hi=SHARE_ALPHA_HI,
              lr_decay_every=20, lr_decay_gamma=0.5, loss_balancing_alpha=0.9,
              loss_balancing_temp=0.1, bernoulli_prob=0.99, foc=False, seed=0):
    """Assemble (and train if no checkpoint) the heterogeneous N-agent SV model.

    Mirrors the torch ``get_model`` signature; returns ``(params, statics, foc)``.
    """
    assert caps_E[0] >= 1e3, "First expert anchors chi and must be unconstrained."

    if rar:
        batch_size = batch_size // 2

    statics = build_statics(K, expert_idx, household_idx, gamma_vec, caps_E,
                            has_t=timestepping, params=params)
    D = statics["D"]
    n_theta = 0 if foc else (len(expert_idx) - 1)

    key = jax.random.PRNGKey(seed)
    key, kinit = jax.random.split(key)
    model_params = init_params(kinit, K, n_theta, D, model_size)

    ckpt = f"{model_path}/model_best.pkl"
    if os.path.exists(ckpt):
        model_params, _ = load_params(ckpt)
        return model_params, statics, foc

    if not train:
        return model_params, statics, foc

    cfg = {
        "batch_size": batch_size, "num_epochs": n_epochs, "lr": lr,
        "rar": rar, "loss_balancing": loss_balancing, "refinement_rounds": 10,
        "alpha_lo": share_alpha_lo, "alpha_hi": share_alpha_hi,
        "loss_log_interval": loss_log_interval,
        "lb_alpha": loss_balancing_alpha, "lb_temp": loss_balancing_temp,
        "bernoulli_prob": bernoulli_prob,
        "num_outer": num_outer, "num_inner": num_inner, "min_inner": min_inner,
        "min_t": 0.0, "max_t": max_t,
        "lr_decay_every": lr_decay_every, "lr_decay_gamma": lr_decay_gamma,
    }

    history_csv = f"{model_path}/loss_history.csv"
    if timestepping:
        model_params, history = train_timestep(model_params, statics, foc, cfg, key,
                                               init_guess, history_csv=history_csv)
    else:
        model_params, history = train_basic(model_params, statics, foc, cfg, key,
                                            history_csv=history_csv)

    save_params(ckpt, model_params, {"K": K, "expert_idx": list(expert_idx),
                                     "household_idx": list(household_idx),
                                     "gamma_vec": list(gamma_vec), "caps_E": list(caps_E),
                                     "foc": foc, "timestepping": timestepping})
    return model_params, statics, foc


# ===========================================================================
# Lightweight evaluation (numpy outputs); plotting/tables intentionally omitted.
# ===========================================================================
def forward_states(params, SV_np, statics, foc, chunk=2000):
    """Run forward_equilibrium on (B, n_state) numpy points; pad t=min_t if the
    model is a time-step model.  Returns a dict of numpy arrays."""
    has_t = statics["has_t"]
    D = statics["D"]
    SV_np = np.asarray(SV_np, dtype=np.float64)
    if SV_np.shape[1] < D:
        SV_np = np.concatenate([SV_np, np.zeros((SV_np.shape[0], D - SV_np.shape[1]))], axis=1)
    keys = ["p", "sigx_full", "sig_agg", "pi", "r", "chat", "xi_active",
            "hjb_expert", "hjb_household", "hjb_k", "theta_full",
            "vi_expert_resid", "goods_resid", "asset_pricing_resid",
            "risk_premium", "sign_k", "chi"]
    acc = {k: [] for k in keys}
    for c in range(0, SV_np.shape[0], chunk):
        SV = jnp.asarray(SV_np[c:c + chunk])
        out = forward_equilibrium(params, SV, statics, foc)
        for k in keys:
            acc[k].append(np.asarray(out[k]))
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def _validation_states(statics, n_samples=10000, seed=0,
                       alpha_lo=SHARE_ALPHA_LO, alpha_hi=SHARE_ALPHA_HI):
    """Common (seeded) (n, K) validation states: x_states + v (no t)."""
    K = statics["K"]
    eps = 0.1 / K
    rng = np.random.default_rng(seed)
    shares = _mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
    vlo, vhi = V_DOMAIN
    v = vlo + (vhi - vlo) * rng.random((n_samples, 1))
    return np.concatenate([shares[:, :K - 1], v], axis=1)


def validation_losses(params, statics, foc, n_samples=10000, seed=0):
    """Mean per-component (training-style) losses on common (seeded) states."""
    SV_state = _validation_states(statics, n_samples, seed)
    if statics["has_t"]:
        SV_state = np.concatenate([SV_state, np.zeros((n_samples, 1))], axis=1)
    comps = loss_components(params, jnp.asarray(SV_state), statics, foc)
    return {k: float(v) for k, v in comps.items()}


# ===========================================================================
# Comparison tables (validation-loss and welfare-equivalent), ported from the
# torch version.  A "model" here is the (params, statics, foc) triple; pass a
# ``models`` dict {name: (params, statics, foc)} sharing the same K / case.
# ===========================================================================
METHOD_DISPLAY = {
    "basic":            "Basic",
    "basic_rar":        "Basic + RAR",
    "basic_lb":         "Basic + LB",
    "basic_rar_lb":     "Basic + RAR + LB",
    "timestep":         "Time-stepping",
    "timestep_rar":     "Time-stepping + RAR",
    "timestep_lb":      "Time-stepping + LB",
    "timestep_rar_lb":  "Time-stepping + RAR + LB",
}


def format_sci(x):
    if not np.isfinite(x):
        return "--"
    base, exp = f"{x:.2e}".split("e"); exp = int(exp)
    return f"{base}" if exp == 0 else f"${base} \\times 10^{{{exp}}}$"


def format_pct(x):
    return "--" if not np.isfinite(x) else f"{x:.2f}\\%"


def df_to_latex(df, path):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(format_pct if "impr" in col else format_sci)
    with open(path, "w") as f:
        f.write(out.style.to_latex(hrules=True))


def compute_validation_losses(params, statics, foc, SV_val, chunk_size=2000):
    """Per-component validation losses + total, on common states, using the same
    reductions as training: HJB groups are mean(sum_k hjb_k**2) (== MSE of the
    raw residual); goods / asset pricing / capital FOC are MSE.  Capital FOC
    only enters for K > 2."""
    out = forward_states(params, SV_val, statics, foc, chunk=chunk_size)
    res = {
        "HJB expert": float(np.mean(np.abs(out["hjb_expert"]))),
        "HJB household": float(np.mean(np.abs(out["hjb_household"]))),
        "Goods clearing": float(np.mean(out["goods_resid"] ** 2)),
        "Asset pricing": float(np.mean(out["asset_pricing_resid"] ** 2)),
    }
    if statics["K"] > 2:
        res["Capital FOC"] = float(np.mean(out["vi_expert_resid"] ** 2))
    res["Total"] = sum(res.values())
    return res


def compare_loss_table(models, baseline_key="basic", n_samples=10000,
                       chunk_size=2000, seed=0, cols=None):
    """Per-method validation losses + % improvement over ``baseline_key``.

    ``models`` : {name: (params, statics, foc)}.  Returns a DataFrame whose rows
    are the methods (display names) and columns are each loss component, the
    Total, and the corresponding "<col> impr." percentages vs the baseline.
    """
    first_statics = next(iter(models.values()))[1]
    SV_val = _validation_states(first_statics, n_samples=n_samples, seed=seed)
    rows = {name: compute_validation_losses(p, st, fc, SV_val, chunk_size)
            for name, (p, st, fc) in models.items()}
    base = rows[baseline_key]
    if cols is None:
        keys = list(next(iter(rows.values())).keys())
        cols = [k for k in keys if k != "Total"] + ["Total"]
    for name, row in rows.items():
        for c in cols:
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    ordered = list(cols) + [f"{c} impr." for c in cols]
    renamed = {METHOD_DISPLAY.get(name, name): row for name, row in rows.items()}
    return pd.DataFrame.from_dict(renamed, orient="index")[ordered]


def compute_welfare_equivalent_losses(models, baseline_key="basic", n_samples=10000,
                                      chunk_size=2000, seed=0):
    """Map each residual to a certainty-equivalent consumption-wealth (c/W) cost
    (units 1/time), averaged over a validation sample.

      HJB residual h_k  ->  rho * |h_k|                              (first order)
      capital FOC (vi)  ->  1/2 gamma_k (phi v)^2 (d theta_k/x_k)^2  (second order)
    """
    first_statics = next(iter(models.values()))[1]
    SV_val = _validation_states(first_statics, n_samples=n_samples, seed=seed)
    rows = {}
    for name, (params, st, foc) in models.items():
        rho = st["rho"]; phi = st["phi"]
        gamma = np.asarray(st["gamma"]).reshape(-1)
        e_idx = st["expert_idx_py"]; v_index = st["v_index"]; K = st["K"]
        out = forward_states(params, SV_val, st, foc, chunk=chunk_size)
        v = SV_val[:, v_index]
        hjb_k = out["hjb_k"]
        x_full = np.concatenate([SV_val[:, :K - 1],
                                 1.0 - SV_val[:, :K - 1].sum(axis=1, keepdims=True)], axis=1)
        hjb_we = float(np.mean(rho * np.abs(hjb_k)))
        x_E = x_full[:, e_idx]
        g_E = gamma[e_idx].reshape(1, -1)
        vi = out["vi_expert_resid"]
        vi_we = float(np.mean(0.5 * g_E * (phi * v[:, None]) ** 2 * (vi / (x_E + 1e-8)) ** 2))
        rows[name] = {"HJB (c/W)": hjb_we, "Capital FOC (c/W)": vi_we,
                      "total (c/W)": hjb_we + vi_we}
    base = rows[baseline_key]
    for name, row in rows.items():
        for c in list(base.keys()):
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    abs_cols = ["HJB (c/W)", "Capital FOC (c/W)", "total (c/W)"]
    ordered = abs_cols + [f"{c} impr." for c in abs_cols]
    return pd.DataFrame.from_dict(rows, orient="index")[ordered]


# ===========================================================================
# Plotting (ported from the torch version; verifies training quality).
# ===========================================================================
def evaluate_slices(params, statics, foc, v_list, n=100, x_lo=0.05, x_hi=0.95):
    """2-agent case: evaluate along x_1 in [x_lo, x_hi] at each fixed v.
    Returns the same dict layout as the torch ``evaluate_slices``."""
    res = {"x_plot": np.linspace(x_lo, x_hi, n)}
    for v in v_list:
        SV = np.zeros((n, 2))
        SV[:, 0] = res["x_plot"]
        SV[:, 1] = v
        out = forward_states(params, SV, statics, foc)
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


SLICE_PLOT_ARGS = {
    "p": r"$p$", "sigx": r"$\sigma_x$", "omega": r"$\Omega=\xi/\zeta$",
    "sigsigp": r"$\sigma+\sigma_p$", "signxi": r"$\pi$", "r": r"$r$",
    "risk_premium": r"$\pi(\sigma+\sigma_p)$",
}
SLICE_COLORS = ["red", "orange", "blue"]


def plot_slice_comparison(method_dict, fd_dict, v_list, out_dir,
                          file_name="slice_comparison.pdf"):
    """Overlay the NN slices (solid) on the Di Tella FD solution (dash-dot) for
    each plotted object -- the headline 'is it right?' plot for agents2."""
    os.makedirs(out_dir, exist_ok=True)
    variables = list(SLICE_PLOT_ARGS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for ai, var in enumerate(variables):
        ax = axes[ai]
        if fd_dict is not None:
            xfd = fd_dict["x_plot"]
            for i, v in enumerate(v_list):
                key = f"{var}_{v}"
                if key in fd_dict:
                    ax.plot(xfd, fd_dict[key], ls="-.", color=SLICE_COLORS[i % len(SLICE_COLORS)],
                            marker="x", markevery=8, label=f"FD v={v}")
        xp = method_dict["x_plot"]
        for i, v in enumerate(v_list):
            ax.plot(xp, method_dict[f"{var}_{v}"], ls="-",
                    color=SLICE_COLORS[i % len(SLICE_COLORS)], alpha=0.85,
                    label=f"NN v={v}")
        ax.set_xlabel("Expert wealth share $x$")
        ax.set_ylabel(SLICE_PLOT_ARGS[var])
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[len(variables):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)


FD_TABLE_VARS = ("omega", "e_hat", "c_hat", "risk_premium")


def compute_fd_errors(method_dict, fd_dict, v_list, vars=FD_TABLE_VARS):
    """MSE and relative-MAE of the NN slices vs the FD solution."""
    x_nn = np.asarray(method_dict["x_plot"])
    x_fd = np.asarray(fd_dict["x_plot"])
    same_grid = len(x_nn) == len(x_fd) and np.allclose(x_nn, x_fd)
    mses, rel_maes = {}, {}
    for var in vars:
        tot_sq = tot_abs = tot_ref = 0.0
        for v in v_list:
            nn = np.asarray(method_dict[f"{var}_{v}"])
            fd = np.asarray(fd_dict[f"{var}_{v}"])
            if not same_grid:
                fd = np.interp(x_nn, x_fd, fd)
            tot_sq = tot_sq + (fd - nn) ** 2
            tot_abs = tot_abs + np.abs(fd - nn)
            tot_ref = tot_ref + np.abs(fd)
        mses[var] = float(np.mean(tot_sq) / len(v_list))
        rel_maes[var] = float(np.mean(tot_abs) / np.mean(tot_ref))
    return mses, rel_maes


def _fd_v_slices(fd_dict, var="omega"):
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


def plot_aggregate_scatter(params, statics, foc, out_dir,
                           file_name="aggregate_scatter.pdf",
                           n_samples=4000, v_fixed=0.25, seed=0,
                           alpha_lo=SHARE_ALPHA_LO, alpha_hi=SHARE_ALPHA_HI):
    """K>2: scatter p / risk premium / omega against (row 1) the total expert
    wealth share and (row 2) the Herfindahl concentration index, at fixed v."""
    K = statics["K"]
    e_idx = statics["expert_idx_py"]
    h_idx = statics["household_idx_py"]
    rng = np.random.default_rng(seed)
    eps = 0.1 / K
    shares = _mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
    SV_np = np.concatenate([shares[:, :K - 1], np.full((n_samples, 1), v_fixed)], axis=1)
    out = forward_states(params, SV_np, statics, foc)

    expert_share = shares[:, e_idx].sum(axis=1)
    herfindahl = (shares ** 2).sum(axis=1)
    xi = out["xi_active"]
    omega = xi[:, e_idx].mean(axis=1) / xi[:, h_idx].mean(axis=1)

    panels = [(out["p"].reshape(-1), "Capital price $p$"),
              (out["risk_premium"].reshape(-1), r"Risk premium $\pi(\sigma+\sigma_p)$"),
              (omega, r"$\Omega=\bar\xi_E/\bar\xi_H$")]
    x_axes = [(expert_share, r"Total expert wealth share $\sum_{i\in E} x_i$"),
              (herfindahl, r"Wealth concentration $H=\sum_k x_k^2$")]

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (xv, xlabel) in enumerate(x_axes):
        for col, (yv, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(xv, yv, s=6, alpha=0.35, edgecolors="none")
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.suptitle(f"v={v_fixed}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)


def plot_theta_chat_histogram(params, statics, foc, out_dir, seed=0, n_samples=10000):
    """Per-agent theta / chat mean +- 95% CI bars (K>2 sanity check: theta should
    decay with gamma, chat should be ordered sensibly)."""
    K = statics["K"]
    eps = 0.1 / K
    rng = np.random.default_rng(seed)
    shares = _mixture_shares_np(n_samples, K, eps, rng)
    vlo, vhi = V_DOMAIN
    v = vlo + (vhi - vlo) * rng.random((n_samples, 1))
    SV_np = np.concatenate([shares[:, :K - 1], v], axis=1)
    out = forward_states(params, SV_np, statics, foc)
    theta = out["theta_full"]; chat = out["chat"]
    idx = np.arange(1, K + 1)

    os.makedirs(out_dir, exist_ok=True)
    for var, arr in [("theta", theta), ("chat", chat)]:
        mean = arr.mean(axis=0)
        se = arr.std(axis=0) / math.sqrt(n_samples)
        yerr = 1.96 * se
        fig, ax = plt.subplots(figsize=(max(6.2, 0.3 * K), 4.8))
        ax.bar(idx, mean, yerr=yerr, capsize=4, width=0.6,
               linewidth=1.0, edgecolor="black")
        ax.set_xlabel("agent index"); ax.set_ylabel(f"{var} mean (95% CI)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{var}_histogram.pdf"))
        plt.close(fig)


def plot_loss_history(csv_path, out_dir, file_name="loss_history.pdf"):
    """Semilog convergence of total / expert-HJB / household-HJB loss from the
    CSV logged during training."""
    if not os.path.exists(csv_path):
        return
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    if rows.size == 0:
        return
    x = np.arange(len(np.atleast_1d(rows["total_loss"])))
    targets = [("total_loss", "Total"), ("expert", "HJB experts"),
               ("household", "HJB households")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for j, (col, label) in enumerate(targets):
        if col not in rows.dtype.names:
            continue
        y = np.atleast_1d(rows[col])
        ymin = np.minimum.accumulate(np.where(np.isfinite(y), y, np.inf))
        axes[j].semilogy(x, ymin, lw=2)
        axes[j].set_xlabel("logged step"); axes[j].set_ylabel(f"{label} loss")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, file_name))
    plt.close(fig)


def make_plots(params, statics, foc, case, model_path, gamma, tau, sigma, a,
               alpha_lo=SHARE_ALPHA_LO, alpha_hi=SHARE_ALPHA_HI):
    """Emit the verification plots for a single trained config."""
    out_dir = os.path.join(model_path, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_loss_history(os.path.join(model_path, "loss_history.csv"), out_dir)

    if case == "agents2":
        v_list = [0.25]
        md = evaluate_slices(params, statics, foc, v_list)
        fd_path = f"./models/numerical/numerical_{gamma}_{tau}_{sigma}_{a}.npz"
        fd_dict = None
        try:
            fd_dict = np.load(fd_path)
        except Exception as e:
            print(f"[plots] no FD solution at {fd_path}: {e}")
        plot_slice_comparison(md, fd_dict, v_list, out_dir)
        if fd_dict is not None:
            v_tab = _fd_v_slices(fd_dict) or v_list
            md_tab = evaluate_slices(params, statics, foc, v_tab)
            mse, mae = compute_fd_errors(md_tab, fd_dict, v_tab)
            print("[plots] FD MSE :", {k: f"{v:.3e}" for k, v in mse.items()})
            print("[plots] FD relMAE:", {k: f"{v*100:.2f}%" for k, v in mae.items()})
    else:
        plot_aggregate_scatter(params, statics, foc, out_dir,
                               alpha_lo=alpha_lo, alpha_hi=alpha_hi)
        plot_theta_chat_histogram(params, statics, foc, out_dir)
    print(f"[plots] written under {out_dir}")

def format_sci(x):
    if not np.isfinite(x):
        return "--"
    base, exp = f"{x:.2e}".split("e"); exp = int(exp)
    return f"{base}" if exp == 0 else f"${base} \\times 10^{{{exp}}}$"


def format_pct(x):
    return "--" if not np.isfinite(x) else f"{x:.2f}\\%"

# Display names for the 8 training configs (rows of the FD-error table) and the
# LaTeX labels for the objects compared against the finite-difference solution.
METHOD_DISPLAY = {
    "basic":           "Basic",
    "basic_rar":       "Basic + RAR",
    "basic_lb":        "Basic + LB",
    "basic_rar_lb":    "Basic + RAR + LB",
    "timestep":        "Time-stepping",
    "timestep_rar":    "Time-stepping + RAR",
    "timestep_lb":     "Time-stepping + LB",
    "timestep_rar_lb": "Time-stepping + RAR + LB",
}


def df_to_latex(df, path):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(format_pct if "impr" in col else format_sci)
    with open(path, "w") as f:
        f.write(out.style.to_latex(hrules=True))

# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="JAX N-agent SV model (no GPU on Windows).")
    parser.add_argument("--case", default="agents2",
                        choices=["agents2", "agents5", "agents5_cap", "agents5_cap2",
                                 "agents20", "agents50"])
    parser.add_argument("--config", default="timestep", choices=list(CONFIGS))
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--tau", type=float, default=1.15)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--outer", type=int, default=70)
    parser.add_argument("--num-inner", type=int, default=5000)
    parser.add_argument("--min-inner", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lo", type=float, default=SHARE_ALPHA_LO)
    parser.add_argument("--alpha-hi", type=float, default=SHARE_ALPHA_HI)
    parser.add_argument("--lr-decay-every", type=int, default=20)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=50,
                        help="log/print + host-sync cadence (in inner epochs)")
    parser.add_argument("--foc", action="store_true")
    args = parser.parse_args()

    ts, rar, lb = CONFIGS[args.config]
    gamma, tau, sigma, a = args.gamma, args.tau, args.sigma, args.a

    dir_tag = "_FOC" if args.foc else ""
    base_dir = f"./models/SV_NAgents_jax{dir_tag}_{gamma}_{tau}_{sigma}_{a}/{args.case}"
    model_path = os.path.join(base_dir, args.config)

    K, eidx, hidx, gamma_vec, caps_E = make_case(args.case, gamma)
    bp = BASE_PARAMS | {"tau": tau, "a": a, "sigma": sigma}

    # init guess for the time-step boundary (helps reach the correct p basin)
    ts_init_guess = {"xi": BASE_PARAMS["rho"]}
    ts_init_guess["r"] = 0.01

    # Train every config; ``models`` maps name -> (params, statics, foc) so the
    # comparison tables can iterate over the methods.
    models, model_paths = {}, {}
    for name in list(CONFIGS.keys()):
        ts, rar, lb = CONFIGS[name]
        mpath = os.path.join(base_dir, name)
        print(f"\n{('=== ' + name + ' ==='):=^80}")
        params, statics, foc = get_model(
            mpath, K, eidx, hidx, gamma_vec, caps_E,
            model_size=[args.width] * args.layers,
            n_epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            timestepping=ts, rar=rar, loss_balancing=lb,
            num_outer=args.outer, num_inner=args.num_inner, min_inner=args.min_inner,
            lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
            loss_log_interval=args.log_interval,
            foc=args.foc, params=bp, init_guess=ts_init_guess,
            share_alpha_lo=args.alpha_lo, share_alpha_hi=args.alpha_hi,
        )
        models[name] = (params, statics, foc)
        model_paths[name] = mpath

    cmp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)
    chunk_size = 500 if K >= 10 else 2000

    # ---- Loss tables (computed over ALL trained methods) --------------------
    for baseline in ("basic", "timestep"):
        if baseline not in models:
            continue
        suffix = "" if baseline == "basic" else "_timestep"
        loss_df = compare_loss_table(models, baseline_key=baseline, chunk_size=chunk_size)
        loss_df.to_csv(os.path.join(cmp_dir, f"comparative_losses{suffix}.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, f"comparative_losses{suffix}.tex"))
        print(f"\n[sv_n_agents] comparative loss table (vs {baseline}):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key=baseline, chunk_size=chunk_size)
        welfare_df.to_csv(os.path.join(cmp_dir, f"welfare_equivalent_losses{suffix}.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, f"welfare_equivalent_losses{suffix}.tex"))
        print(f"\n[sv_n_agents] welfare-equivalent loss table (c/W, vs {baseline}):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    # ---- Per-config verification plots --------------------------------------
    for name, (params, statics, foc) in models.items():
        make_plots(params, statics, foc, args.case, model_paths[name],
                   gamma, tau, sigma, a, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi)


if __name__ == "__main__":
    main()
