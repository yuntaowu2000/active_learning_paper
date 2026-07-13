"""
sv_n_agents_NN.py
=================

Heterogeneous **N-agent** generalisation of the Di Tella (2017) stochastic-
volatility model (the 2-agent version lives in
``stochastic_volatility_model.py``).  See ``sv_n_agents_spec.md`` for the full
math; the headline points:

* State ``s = (x_1, ..., x_{K-1}, v)``: wealth shares of ``K`` agent types on the
  open simplex (``x_K = 1 - sum``) plus the idiosyncratic-risk volatility ``v``.
* Types split into **experts** (bear idiosyncratic ``phi*v`` risk, manage
  capital, retire at rate ``tau``) and **households** (diversified).
* Heterogeneous risk aversion ``gamma_k``; one positive MLP per value-function
  multiplier ``xi_k`` and one MLP per expert capital share ``theta_k``, all fused
  into single ``vmap`` calls (``StackedAgent``) for efficiency at K=20, 50.
* Di Tella is a *single aggregate-shock* model, so the share diffusions solve
  one differentiable batched linear system and everything reduces **exactly** to
  the original 2-agent equations when ``K = 2``.
* A leverage cap on experts (``theta_k/x_k <= cap_k``) is imposed as a
  variational-inequality (NCP) free boundary, so residual-based sampling stands
  out.

The script mirrors ``complete_market_model/gp_n_agents_NN.py`` in structure
(StackedAgent, simplex sampler, RAR override, loss balancing, time stepping,
the comparison plots/tables).
"""

import contextlib
import gc
import math
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, hessian, jacrev, vmap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as smi

from deep_macrofin import (LossReductionMethod, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, set_seeds)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Device used for evaluation/plotting forwards.  Models live on CPU between uses
# (see move_model / model_on) so that training/evaluating many configs does not
# pin every network's VRAM at once; each model is moved to EVAL_DEVICE only for
# the duration of its own forward pass.
EVAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _module_of(agent):
    """The underlying nn.Module of an Agent / EndogVar wrapper."""
    return agent.model if hasattr(agent, "model") else agent


def move_model(model, dev):
    """Move all of a model's device-resident state to ``dev`` (in place).

    Covers the agent/endog networks, the static economic tensors (gamma,
    caps_E), the cached variable_val_dict placeholders, and any leftover
    time-stepping training buffers, then updates ``model.device`` so subsequent
    forwards (which read it) land on the right device.  The StackedAgents read
    the live modules each call, so no re-attachment is needed.
    """
    for d in (getattr(model, "agents", {}), getattr(model, "endog_vars", {})):
        for name in d:
            _module_of(d[name]).to(dev)
    st = getattr(model, "statics", None)
    if st:
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    vd = getattr(model, "variable_val_dict", None)
    if vd:
        for k, v in vd.items():
            if torch.is_tensor(v):
                vd[k] = v.to(dev)
    # drop / move training-only buffers so they stop pinning VRAM on CPU moves
    for attr in ("anchor_points", "boundary_uniform_points",
                 "init_loss_tensor", "prev_loss_tensor"):
        t = getattr(model, attr, None)
        if torch.is_tensor(t):
            setattr(model, attr, t.to(dev))
    pv = getattr(model, "prev_vals", None)
    if isinstance(pv, dict):
        for k, v in pv.items():
            if torch.is_tensor(v):
                pv[k] = v.to(dev)
    model.device = dev
    if dev == "cpu":
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


@contextlib.contextmanager
def model_on(model, dev=EVAL_DEVICE):
    """Temporarily move ``model`` to ``dev`` for an evaluation, then restore it
    to its previous device (typically CPU).  Ensures only one model occupies the
    GPU at a time."""
    prev = getattr(model, "device", dev)
    if prev != dev:
        move_model(model, dev)
    try:
        yield model
    finally:
        if prev != dev:
            move_model(model, prev)


# ===========================================================================
# Wealth-share sampling
# ===========================================================================
# A plain Dirichlet(1) on the K-simplex concentrates ALL its mass near the
# centroid (every share ~ 1/K) once K is large, so for K=20/50 the economy is
# always near-egalitarian and the concentrated states -- a few low-gamma experts
# holding most capital, where the model is stiff and interesting -- are never
# sampled.  We instead draw the Dirichlet concentration alpha per sample,
# log-uniform on [SHARE_ALPHA_LO, SHARE_ALPHA_HI]: alpha<1 gives sparse
# (concentrated) draws, alpha~1 gives egalitarian draws, so one batch spans the
# full range of wealth concentration.  Shares are then floored at eps.
SHARE_ALPHA_LO = 0.05
SHARE_ALPHA_HI = 1.0


def _mixture_shares_torch(n, K, eps, device, alpha_lo=SHARE_ALPHA_LO,
                          alpha_hi=SHARE_ALPHA_HI):
    """(n, K) eps-floored simplex shares; per-row log-uniform Dirichlet alpha.
    Uses the global torch RNG (so set_seeds controls it)."""
    u = torch.rand((n, 1), device=device)
    alpha = torch.exp(math.log(alpha_lo) + (math.log(alpha_hi) - math.log(alpha_lo)) * u)
    conc = alpha.expand(n, K).contiguous()
    shares = torch.distributions.Dirichlet(conc).sample()      # (n, K)
    return eps + (1.0 - K * eps) * shares


def _mixture_shares_np(n, K, eps, rng, alpha_lo=SHARE_ALPHA_LO,
                       alpha_hi=SHARE_ALPHA_HI):
    """NumPy counterpart of _mixture_shares_torch for reproducible (seeded)
    evaluation/plotting sampling."""
    u = rng.random((n, 1))
    alpha = np.exp(np.log(alpha_lo) + (np.log(alpha_hi) - np.log(alpha_lo)) * u)
    gam = rng.gamma(np.broadcast_to(alpha, (n, K)), 1.0)        # (n, K)
    shares = gam / gam.sum(axis=1, keepdims=True)
    return eps + (1.0 - K * eps) * shares


# ===========================================================================
# Fused multi-network derivative helper (verbatim idea from gp_n_agents_NN.py).
# One vmap-batched (forward, jacrev, hessian) call across a list of identical-
# architecture scalar networks AND the batch dimension.
# ===========================================================================
def _stack_module_state_with_grad(modules):
    if len(modules) == 0:
        raise ValueError("Need at least one module.")
    all_params = [dict(m.named_parameters()) for m in modules]
    all_buffers = [dict(m.named_buffers()) for m in modules]
    params = {k: torch.stack([p[k] for p in all_params]) for k in all_params[0]}
    buffers = {k: torch.stack([b[k] for b in all_buffers]) for k in all_buffers[0]} if all_buffers[0] else {}
    return params, buffers


class StackedAgent:
    """One vmap-batched call across identical-architecture scalar networks.

    ``compute(SV) -> (value (B,N), jac (B,N,D), hess (B,N,D,D))``;
    ``value_only=True`` returns ``(value, None, None)``.
    """

    def __init__(self, agents, value_only: bool = False):
        if len(agents) == 0:
            raise ValueError("StackedAgent needs at least one agent.")
        self._agents = agents
        self._template = agents[0].model
        self._value_only = value_only

    def compute(self, SV: torch.Tensor):
        params, buffers = _stack_module_state_with_grad([a.model for a in self._agents])
        template = self._template

        def value_scalar(p, b, x):
            return functional_call(template, (p, b), (x,)).squeeze(-1)

        def fwd_per_net(p, b):
            return vmap(lambda x: value_scalar(p, b, x))(SV)               # (B,)

        val = vmap(fwd_per_net)(params, buffers).transpose(0, 1).contiguous()   # (B,N)
        if self._value_only:
            return val, None, None

        def jac_per_net(p, b):
            return vmap(jacrev(lambda x: value_scalar(p, b, x)))(SV)        # (B,D)

        def hess_per_net(p, b):
            return vmap(hessian(lambda x: value_scalar(p, b, x)))(SV)       # (B,D,D)

        jac = vmap(jac_per_net)(params, buffers).transpose(0, 1).contiguous()    # (B,N,D)
        hess = vmap(hess_per_net)(params, buffers).transpose(0, 1).contiguous()  # (B,N,D,D)
        return val, jac, hess

    def value(self, SV: torch.Tensor):
        return self.compute(SV)[0]

    # def analytic_p(self, SV: torch.Tensor, statics):
    #     """Capital price solved from goods-market clearing, treating these
    #     stacked networks as the ``xi`` value functions.

    #     Goods clearing  a - iota(p) = p * sum_k x_k chat_k  with
    #     chat_k = rho^(1/psi) xi_k^((psi-1)/psi)  (no p-dependence) and
    #     iota(p) = A (g+delta)^2 + B(g+delta), g+delta = (p-B)/(2A) is a
    #     QUADRATIC in p whose positive root is

    #         p = sqrt(4 A^2 C^2 + 4 A a + B^2) - 2 A C ,   C = sum_k x_k chat_k .

    #     First/second state-derivatives of p are obtained by autodiff *through*
    #     the xi networks (so p_Hess needs only xi's 2nd derivatives -- same
    #     differentiation order as a free p-network).  Returns
    #     ``(p (B,1), p_Jac (B,1,D), p_Hess (B,1,D,D))``.
    #     """
    #     models = [a.model for a in self._agents]
    #     per_p = [dict(m.named_parameters()) for m in models]
    #     per_b = [dict(m.named_buffers()) for m in models]
    #     K = statics["K"]
    #     A = statics["A"]; Bc = statics["B"]; a = statics["a"]
    #     rho = statics["rho"]; psi = statics["psi"]
    #     alpha = (psi - 1.0) / psi
    #     coef = rho ** (1.0 / psi)

    #     def p_scalar(x):                                   # x: (D,) single sample
    #         xis = [functional_call(m, (pa, bu), (x,)).squeeze(-1)
    #                for m, pa, bu in zip(models, per_p, per_b)]
    #         xi_vec = torch.stack(xis)                      # (K,)
    #         x_states = x[:K - 1]
    #         x_K = 1.0 - x_states.sum()
    #         x_full = torch.cat([x_states, x_K.reshape(1)])  # (K,)
    #         chat = coef * xi_vec ** alpha
    #         C = (x_full * chat).sum()
    #         return torch.sqrt(4.0 * A * A * C * C + 4.0 * A * a + Bc * Bc) - 2.0 * A * C

    #     B0, D = SV.shape[0], SV.shape[1]
    #     p_val = vmap(p_scalar)(SV).reshape(B0, 1)
    #     # p_jac = vmap(jacrev(p_scalar))(SV).reshape(B0, 1, D)
    #     # p_hess = vmap(hessian(p_scalar))(SV).reshape(B0, 1, D, D)
    #     return p_val # , p_jac, p_hess


# ===========================================================================
# Economic core: one differentiable forward pass computing every equilibrium
# object from the raw network outputs.  See spec sections 2-6.
# ===========================================================================
def compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, r=None, statics=None):
    """All shapes batched over B.

    Inputs
    ------
    SV        : (B, D)         state; D = K-1 shares + 1 (v) [+ 1 (t) in timestep]
    xi        : (B, K)         value multipliers
    xi_Jac    : (B, K, D)
    xi_Hess   : (B, K, D, D)
    p         : (B, 1)         capital price
    p_Jac     : (B, 1, D)
    p_Hess    : (B, 1, D, D)
    theta_E   : (B, n_E)       expert capital shares (NN outputs, experts only)
    statics   : dict with K, D, n_E, expert_idx (LongTensor), household_idx,
                v_index, has_t, gamma (1,K), caps_E (1,n_E), and scalar params.

    Returns a dict of named (B,*) tensors used by the registered losses & plots.
    """
    K = statics["K"]
    D = statics["D"]
    v_index = statics["v_index"]
    has_t = statics["has_t"]
    expert_idx = statics["expert_idx"]
    household_idx = statics["household_idx"]
    gamma = statics["gamma"]               # (1, K)
    caps_E = statics["caps_E"]             # (1, n_E)

    rho = statics["rho"]; psi = statics["psi"]; tau = statics["tau"]
    phi = statics["phi"]; sigma = statics["sigma"]
    lbd = statics["lbd"]; v_mean = statics["v_mean"]; sigv_mean = statics["sigv_mean"]
    A = statics["A"]; B = statics["B"]; delta = statics["delta"]; a = statics["a"]

    B_ = SV.shape[0]

    # ---- shares (full vector incl. residual x_K) and v --------------------
    x_states = SV[:, :K - 1]                                  # (B, K-1)
    x_K = 1.0 - x_states.sum(dim=1, keepdim=True)             # (B, 1)
    x_full = torch.cat([x_states, x_K], dim=1)                # (B, K)
    v = SV[:, v_index:v_index + 1]                            # (B, 1)

    # ---- aggregate (capital) block ----------------------------------------
    g = (p - B) / (2.0 * A) - delta
    iota = A * (g + delta) ** 2 + B * (g + delta)
    mu_v = lbd * (v_mean - v)
    sig_v = sigv_mean * torch.sqrt(v)
    chat = rho ** (1.0 / psi) * xi ** ((psi - 1.0) / psi)     # (B, K)

    # ---- share-diffusion linear system (spec section 3) -------------------
    g_arr = gamma                                            # (1, K)
    gm1_over_g = (g_arr - 1.0) / g_arr                       # (1, K)
    inv_g = 1.0 / g_arr                                      # (1, K)

    xi_v = xi_Jac[:, :, v_index]                             # (B, K)
    xi_x = xi_Jac[:, :, :K - 1]                              # (B, K, K-1)
    a_k = xi_v * sig_v / xi                         # (B, K)
    b_k = xi_x / xi.unsqueeze(-1)                    # (B, K, K-1)

    p_v = p_Jac[:, 0, v_index:v_index + 1]                   # (B, 1)
    p_x = p_Jac[:, 0, :K - 1]                                # (B, K-1)
    a_p = p_v * sig_v / p                            # (B, 1)
    b_p = p_x / p                                    # (B, K-1)

    P0 = sigma + a_p                                         # (B, 1)
    coeff = x_full * gm1_over_g                              # (B, K)  x_k (g_k-1)/g_k
    S0 = (coeff * a_k).sum(dim=1, keepdim=True)              # (B, 1)
    S_m = torch.einsum("bk,bkm->bm", coeff, b_k)            # (B, K-1)
    T = (x_full * inv_g).sum(dim=1, keepdim=True)            # (B, 1)
    pi0 = (P0 + S0) / T                              # (B, 1)
    pi_m = (b_p + S_m) / T                           # (B, K-1)

    xs = x_states                                            # (B, K-1)
    gs = g_arr[:, :K - 1]                                    # (1, K-1)
    b_ks = b_k[:, :K - 1, :]                                 # (B, K-1, K-1)
    # M[b,k,m] = x_k/g_k * pi_m - x_k (g_k-1)/g_k b_{k,m} - x_k b_{p,m}
    term1 = (xs / gs).unsqueeze(-1) * pi_m.unsqueeze(1)              # (B,K-1,K-1)
    term2 = (xs * (gs - 1.0) / gs).unsqueeze(-1) * b_ks             # (B,K-1,K-1)
    term3 = xs.unsqueeze(-1) * b_p.unsqueeze(1)                     # (B,K-1,K-1)
    M = term1 - term2 - term3
    c = (xs / gs) * pi0 - (xs * (gs - 1.0) / gs) * a_k[:, :K - 1] - xs * P0   # (B,K-1)

    Imat = torch.eye(K - 1, device=SV.device, dtype=SV.dtype).unsqueeze(0)
    u = torch.linalg.solve(Imat - M, c.unsqueeze(-1)).squeeze(-1)   # (B, K-1) = sigma_x for states
    sigx_full = torch.cat([u, -u.sum(dim=1, keepdim=True)], dim=1)  # (B, K)

    # ---- recompute diffusions from the solved u ---------------------------
    sigp = a_p + (b_p * u).sum(dim=1, keepdim=True)                 # (B, 1)
    sig_agg = sigma + sigp                                         # (B, 1) = sigma + sigma_p
    sigxi = a_k + torch.einsum("bkm,bm->bk", b_k, u)              # (B, K)
    S_full = (coeff * sigxi).sum(dim=1, keepdim=True)             # (B, 1)
    pi = (sig_agg + S_full) / T                          # (B, 1) price of risk
    sign_k = pi * inv_g - gm1_over_g * sigxi                     # (B, K) sigma_{n,k}

    # ---- capital allocation, idiosyncratic risk, free boundary ------------
    theta_full = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    theta_full[:, expert_idx] = theta_E
    x_E = x_full[:, expert_idx]                                  # (B, n_E)
    g_E = g_arr[:, expert_idx]                                   # (1, n_E)
    phiv = phi * v                                               # (B, 1)
    phiv2 = phiv ** 2                                            # (B, 1)
    sigtilde_E = phiv * theta_E / x_E                            # (B, n_E)

    # chi anchored on the first expert (must be unconstrained)
    chi = g_E[:, 0:1] * phiv2 * theta_E[:, 0:1] / x_E[:, 0:1]    # (B, 1)
    theta_star_E = chi * x_E / (g_E * phiv2)               # (B, n_E)
    vi_expert_resid = torch.minimum(caps_E * x_E - theta_E, theta_star_E - theta_E)      # (B, n_E)

    # full-K idiosyncratic exposure (households 0)
    sigtilde_full = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    sigtilde_full[:, expert_idx] = sigtilde_E
    chi_theta_over_x_full = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    chi_theta_over_x_full[:, expert_idx] = chi * theta_E / x_E

    # ---- goods-market clearing residual -----------------------------------
    goods_resid = (a - iota) - p * (x_full * chat).sum(dim=1, keepdim=True)   # (B, 1)

    # ---- share drifts (r-independent, see spec section 5) -----------------
    # net-worth drift with r = 0 (r cancels in mu_x); + chi theta/x for experts
    mu_net0 = pi * sign_k + chi_theta_over_x_full                # (B, K)
    agg_cons = (a - iota) / p                                  # (B, 1) = C/N
    # mu_N0 = (x_full * mu_net0).sum(dim=1, keepdim=True) - agg_cons  # (B, 1)
    mu_N0_ = pi * sig_agg + chi - agg_cons
    mu_x_full = x_full * ((mu_net0 - chat) - mu_N0_ - (sign_k - sig_agg) * sig_agg)        # (B, K)

    # retirement transfers: experts -> households (pro-rata by household share)
    retire = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    X_E = x_full[:, expert_idx].sum(dim=1, keepdim=True)        # (B, 1)
    X_H = x_full[:, household_idx].sum(dim=1, keepdim=True)     # (B, 1)
    retire[:, expert_idx] = -tau * x_full[:, expert_idx]
    retire[:, household_idx] = tau * X_E * (x_full[:, household_idx] / X_H)
    mu_x_full = mu_x_full + retire
    mu_x_states = mu_x_full[:, :K - 1]                          # (B, K-1)

    # ---- state drift / diffusion vectors ----------------------------------
    mu_s = torch.zeros((B_, D), device=SV.device, dtype=SV.dtype)
    mu_s[:, :K - 1] = mu_x_states
    mu_s[:, v_index] = mu_v.squeeze(-1)
    if has_t:
        mu_s[:, D - 1] = 1.0                                    # d/dt coefficient
    sig_s = torch.zeros((B_, D), device=SV.device, dtype=SV.dtype)
    sig_s[:, :K - 1] = u
    sig_s[:, v_index] = sig_v.squeeze(-1)

    # ---- mu_xi (Ito) and mu_P --------------------------------------------
    # drift term: sum_d mu_s_d f_{,d};   diffusion: 0.5 sig_s^T H sig_s (single shock)
    mu_xi = (torch.einsum("bd,bkd->bk", mu_s, xi_Jac)
             + 0.5 * torch.einsum("bd,bkde,be->bk", sig_s, xi_Hess, sig_s)) / xi
    mu_P = (torch.einsum("bd,bd->b", mu_s, p_Jac[:, 0, :]).unsqueeze(-1)
            + 0.5 * torch.einsum("bd,bde,be->b", sig_s, p_Hess[:, 0], sig_s).unsqueeze(-1)) / p

    # ---- risk-free rate: FREE network + asset-pricing residual ------------
    # Asset pricing for the aggregate capital claim (the anchor expert holds it):
    #   (a-iota)/p + g + mu_P + sigma*sigp - r = sig_agg*pi + chi
    # The price *curvature* mu_P enters the model ONLY through this equation, so
    # if r is back-solved here (asset_pricing_resid == 0) the curvature of p is
    # left UNCONSTRAINED -- the HJBs see only mu_xi.  We therefore keep r as a
    # free network and enforce the asset-pricing relation as a soft loss, exactly
    # as in the original 2-agent model.  r_implied is the level the equation
    # demands; the residual pins both r's level and p's curvature.
    r_implied = ((a - iota) / p + g + mu_P + sigma * sigp - sig_agg * pi - chi)   # (B, 1)
    if r is None:                                   # fallback: analytic r (no asset-pricing loss)
        r = r_implied
        asset_pricing_resid = torch.zeros_like(r)
    else:
        asset_pricing_resid = r_implied - r         # (B, 1)
    sig_clearing_resid = (sigma + sigp) - (x_full * sign_k).sum(dim=1, keepdim=True)
    
    # ---- HJB per type (spec section 6) ------------------------------------
    mu_net = r + pi * sign_k + chi_theta_over_x_full           # (B, K)
    # retirement target value (wealth-weighted household xi)
    xi_H = xi[:, household_idx]                                 # (B, n_H)
    x_H = x_full[:, household_idx]                              # (B, n_H)
    xi_ret = (x_H * xi_H).sum(dim=1, keepdim=True) / X_H   # (B, 1)

    # NOTE the HJB risk-penalty cross term uses (1-gamma)/gamma = -gm1_over_g
    # (the original hjbeq_1/2), which is the OPPOSITE sign of the (gamma-1)/gamma
    # that appears in sign_k/sigw.  So the cross term is +2*gm1_over_g*... here.
    hjb_common = (chat ** (1.0 - psi) / (1.0 - psi) * rho * xi ** (psi - 1.0)
                  + mu_net - chat + mu_xi
                  - g_arr / 2.0 * (sign_k ** 2 + sigxi ** 2 + 2.0 * gm1_over_g * sign_k * sigxi)
                  - rho / (1.0 - psi))                          # (B, K)
    # expert-only additions
    retire_term = tau / (1.0 - g_arr) * ((xi_ret / xi) ** (1.0 - g_arr) - 1.0)   # (B,K)
    idio_pen = -g_arr / 2.0 * sigtilde_full ** 2               # (B, K)
    is_expert = torch.zeros((1, K), device=SV.device, dtype=SV.dtype)
    is_expert[0, expert_idx] = 1.0
    hjb_k = hjb_common + is_expert * (retire_term + idio_pen)   # (B, K)

    hjb_expert = (hjb_k[:, expert_idx] ** 2).sum(dim=1, keepdim=True)       # (B, 1)
    hjb_household = (hjb_k[:, household_idx] ** 2).sum(dim=1, keepdim=True)  # (B, 1)

    risk_premium = sig_agg * pi

    out = {
        "x_full": x_full, "theta_full": theta_full, "chat": chat,
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
    for idx in expert_idx:
        out[f"hjb_expert_{idx}"] = hjb_k[:, idx:idx+1] ** 2
    for idx in household_idx:
        out[f"hjb_household_{idx}"] = hjb_k[:, idx:idx+1] ** 2
    return out


# ===========================================================================
# PDEModel subclass: simplex+v sampler, RAR override, fused forward.
# ===========================================================================
class _SVNAgentMixin:
    """Shared logic for the stationary and time-stepping N-agent SV models."""

    def _sv_init(self, config):
        self.rar = config.get("rar", False)
        # interior-FOC mode: hard-wire expert capital theta from the FOC instead
        # of using free theta networks (see _compute_theta_E).
        self.foc = config.get("foc", False)
        # Dirichlet-alpha mixture range for wealth-share sampling (see
        # _mixture_shares_torch); configurable so the concentration can be swept.
        self.share_alpha_lo = config.get("share_alpha_lo", SHARE_ALPHA_LO)
        self.share_alpha_hi = config.get("share_alpha_hi", SHARE_ALPHA_HI)
        self.statics = None
        self._xi_stack = None
        self._theta_stack = None
        self._p_stack = None
        self._r_stack = None
        self._skip_local_keys = set()
        # parameter-homotopy state (see enable_homotopy on the time-step model);
        # _block_best gates the *final* model_best.pt while still ramping params.
        # self._homotopy = None
        # self._block_best = False
        # disable the per-epoch diagnostic (expensive, unused here)
        try:
            self._PDEModel__compute_changes = lambda SV: {"total": 0.0}
        except Exception:
            pass

    # -- checkpoint gating during parameter homotopy ------------------------
    def save_model(self, model_dir="./", filename=None, verbose=False):
        # While the homotopy is ramping (gamma, tau) from the easy regime to the
        # target, the easy-phase outer iterations have artificially LOW residuals
        # and would otherwise win the "best" checkpoint that get_model reloads.
        # Block ONLY the final f"{prefix}_best.pt" during the ramp -- crucially
        # NOT f"{prefix}_temp_best.pt", which the outer loop reloads each
        # iteration to continue the backward march.
        if (getattr(self, "_block_best", False) and filename is not None
                and filename.endswith("_best.pt") and not filename.endswith("_temp_best.pt")):
            return
        return super().save_model(model_dir, filename, verbose)

    # -- stack attachment ----------------------------------------------------
    def attach_stacks(self, xi_names, theta_names, p_name="p", r_name="r"):
        # `theta_names` are the NON-anchor experts only; the anchor expert's
        # capital share is pinned by clearing (theta_anchor = 1 - sum(others)),
        # so capital-market clearing holds *by construction*.
        self._xi_stack = (StackedAgent([self.agents[n] for n in xi_names], value_only=False), list(xi_names))
        if theta_names:
            self._theta_stack = (StackedAgent([self.endog_vars[n] for n in theta_names], value_only=True), list(theta_names))
        else:
            self._theta_stack = (None, [])
        self._p_stack = (StackedAgent([self.endog_vars[p_name]], value_only=False), [p_name])
        # r is a free network; only its VALUE is used (no r-derivatives appear).
        self._r_stack = (StackedAgent([self.endog_vars[r_name]], value_only=True), [r_name])
        self._skip_local_keys.update(xi_names)
        self._skip_local_keys.update(theta_names)
        self._skip_local_keys.update([p_name, r_name])
        # remember the names so we can re-bind the stacks after load_model (which
        # recreates the underlying Agent/EndogVar objects -- see load_model).
        self._attached_names = (list(xi_names), list(theta_names))

    def load_model(self, dict_to_load):
        # CRITICAL for the stacked model + time-stepping outer loop:
        # the library's load_model (called at the end of every outer loop)
        # rebuilds each agent/endog var via add_agent(overwrite=True), which
        # constructs BRAND-NEW module objects and rebinds self.agents[name].
        # Our StackedAgents hold direct references to the *previous* objects, so
        # without re-attaching, the reinitialized optimizer would train the new
        # modules while update_variables still evaluates the old (frozen) ones
        # -- the model then stops changing after the first time iteration.
        super().load_model(dict_to_load)
        names = getattr(self, "_attached_names", None)
        if names is not None:
            self.attach_stacks(*names)

    # -- expert capital allocation -----------------------------------------
    def _compute_theta_E(self, SV, x_full):
        """Expert capital shares ``theta_E`` (B, n_E), ordered as ``expert_idx``.

        Two modes, selected by ``self.foc``:

        * free (default): the anchor expert (column 0) holds the residual capital
          so capital clearing (sum_k theta_k = 1) is exact, and the non-anchor
          experts come from their free networks (``self._theta_stack``).
        * interior-FOC (``self.foc``): every expert is at its interior optimum,
          theta_k/x_k = chi/(gamma_k (phi v)^2); imposing clearing pins chi
          analytically, so theta_k = (x_k/gamma_k) / sum_j(x_j/gamma_j).  This is
          a closed form of the state -- leverage monotone in gamma by
          construction, the capital-FOC residual identically 0, and NO theta
          networks needed.  (Assumes all experts uncapped, which is the case for
          the multi-expert cases that use this mode.)
        """
        if self.foc:
            eidx = self.statics["expert_idx"]
            g_E = self.statics["gamma"][:, eidx]              # (1, n_E)
            w = x_full[:, eidx] / g_E                         # (B, n_E) = x_k/gamma_k
            return w / w.sum(dim=1, keepdim=True)

        B_ = SV.shape[0]
        if self._theta_stack[0] is not None:
            theta_others = self._theta_stack[0].value(SV)              # (B, n_E-1)
            theta_anchor = 1.0 - theta_others.sum(dim=1, keepdim=True)  # (B, 1)
            return torch.cat([theta_anchor, theta_others], dim=1)      # (B, n_E)
        return torch.ones((B_, 1), device=SV.device, dtype=SV.dtype)

    # -- forward / equation evaluation --------------------------------------
    def update_variables(self, SV, vd=None):
        if vd is None:
            vd = self.variable_val_dict
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            vd[sv_name] = SV[:, i:i+1]
        vd["SV"] = SV

        xi, xi_Jac, xi_Hess = self._xi_stack[0].compute(SV)
        p, p_Jac, p_Hess = self._p_stack[0].compute(SV)
        r = self._r_stack[0].value(SV)                                 # (B, 1) free rate

        # expert capital shares (the FOC variant overrides _compute_theta_E to
        # impose the interior FOC instead of the anchor-residual parameterization)
        K = self.statics["K"]
        x_states = SV[:, :K - 1]
        x_full = torch.cat([x_states, 1.0 - x_states.sum(dim=1, keepdim=True)], dim=1)
        theta_E = self._compute_theta_E(SV, x_full)

        out = compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, r=r, statics=self.statics)

        # expose per-agent names (cheap slices) for any direct references
        for i, n in enumerate(self._xi_stack[1]):
            vd[n] = xi[:, i:i + 1]
        # non-anchor experts occupy columns 1.. of theta_E (col 0 = anchor)
        for i, n in enumerate(self._theta_stack[1]):
            vd[n] = theta_E[:, i + 1:i + 2]
        vd["p"] = p
        vd["r"] = r                                # free network; out["r"] == r too
        vd["xi_active"] = xi
        vd.update(out)

        # remaining library local functions (skip the ones we computed)
        for func_name in self.local_function_dict:
            if func_name in self._skip_local_keys:
                continue
            vd[func_name] = self.local_function_dict[func_name](SV)

        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            vd[lhs] = self.equations[eq_name].eval(self.custom_function_dict, vd)
    
    def closure(self, SV):
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV
        self.update_variables(SV)
        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
        return total_loss

    def sample_simplex_v(self, epoch):
        """
        Dirichlet-alpha MIXTURE over the K wealth shares (drop residual) +
        uniform v.  The per-row log-uniform alpha spans egalitarian (alpha~1)
        and concentrated (alpha<1) wealth distributions -- essential at large K,
        where Dirichlet(1) would pin every share near 1/K (see
        _mixture_shares_torch).
        """
        K = self.statics["K"]
        eps = 0.1 / K
        shares = _mixture_shares_torch(self.batch_size, K, eps, self.device, self.share_alpha_lo, self.share_alpha_hi)
        x_states = shares[:, :K - 1]                          # (B, K-1)
        vlo, vhi = self.statics["v_domain"]
        v = vlo + (vhi - vlo) * torch.rand((self.batch_size, 1), device=self.device)
        return torch.cat([x_states, v], dim=1)


class PDEModelNAgentsSV(_SVNAgentMixin, PDEModel):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self._sv_init(config)
        if self.rar:
            self.sample = self.sample_rar_greedy
            self.sampling_method = SamplingMethod.RARG
        else:
            self.sample = self.sample_simplex_v

    # NOTE: we deliberately do NOT override `closure`; the library default
    # (no gradient clipping) is used, matching the original 2-agent model.
    # Our `update_variables` override is invoked by that default closure.

    # -- residual-adaptive refinement ---------------------------------------
    def _refinement_loss_dict(self, epoch):
        self.set_all_model_eval()
        all_SVs, all_loss = [], []
        saved_bs = self.batch_size
        if self.statics["K"] > 20:
            self.batch_size = 500
            sample_times = 20
        else:
            self.batch_size = 1000
            sample_times = 10
        for _ in range(sample_times):
            torch.cuda.empty_cache()
            # values-only scoring -> run under no_grad so no autograd graph (incl.
            # the (B,N,D,D) Hessians) is retained.  torch.func still differentiates
            # the state-derivatives internally (see _score_pool note).
            with torch.no_grad():
                SV = self.sample_simplex_v(epoch)
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)
                total = torch.zeros((SV.shape[0], 1), device=self.device)
                Bn = SV.shape[0]

                def per_sample(res):
                    aa = torch.abs(res)
                    if aa.dim() == 0:
                        return aa.expand(Bn, 1).reshape(Bn, 1)
                    if aa.dim() == 1:
                        return aa.reshape(Bn, 1)
                    return aa.reshape(Bn, -1).mean(dim=-1, keepdim=True)

                for label in self.endog_equations:
                    total = total + per_sample(self.endog_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                for label in self.hjb_equations:
                    total = total + per_sample(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                all_SVs.append(SV.detach().cpu())
                all_loss.append(total.detach().cpu())
            del SV, total, vd_
            gc.collect(); torch.cuda.empty_cache()
        self.batch_size = saved_bs
        self.set_all_model_training()
        return {"SV": torch.cat(all_SVs, 0), "loss": torch.cat(all_loss, 0)}

    def sample_rar_greedy(self, epoch):
        if self.num_epochs and epoch % max(1, self.num_epochs // self.refinement_rounds) == 0 and epoch > 0:
            rd = self._refinement_loss_dict(epoch)
            ids = torch.topk(rd["loss"], self.batch_size // self.refinement_rounds, dim=0)[1].squeeze(-1)
            self.anchor_points = torch.vstack((self.anchor_points, rd["SV"][ids].to(self.device)))
        sv = self.sample_simplex_v(epoch)
        if self.anchor_points is not None and len(self.anchor_points) > 0:
            return torch.vstack((sv, self.anchor_points))
        return sv


class PDEModelTimeStepNAgentsSV(_SVNAgentMixin, PDEModelTimeStep):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self._sv_init(config)
        self.sample = self.sample_simplex_v_ts

        self.sample_boundary_cond = self.__sample_custom_boundary_cond
        self.boundary_uniform_points = None
        self._outer_iter = 0

        # --- learning-rate step decay along the backward (outer) march ---------
        # The library rebuilds the optimizer from self.lr at the START of every
        # outer loop and then fires OnInnerLoopStart once (with no args) right
        # after.  We hook that to (a) recompute self.lr from the base lr on a step
        # schedule and (b) patch the just-built optimizer's param groups, so the
        # decay takes effect on the current outer loop too.  Decay multiplies lr
        # by lr_decay_gamma every lr_decay_every outer iterations.
        self._lr_base = config.get("lr", self.lr)
        self.lr_decay_every = int(config.get("lr_decay_every", 20))
        self.lr_decay_gamma = float(config.get("lr_decay_gamma", 0.5))
        self._lr_decay_outer = 0
        self.OnInnerLoopStart += self._lr_decay_step
        # outer iterations to hold each gamma level before the next ramp step
        # (1 = advance every outer iteration).  Safe default so non-homotopy
        # runs don't need the key.
    #     self.homotopy_steps = config.get("homotopy_step", 1)

    def _lr_decay_step(self):
        """Step the LR every ``lr_decay_every`` outer iterations.  Fired once per
        outer loop (no args) via OnInnerLoopStart, so the call count equals the
        outer-iteration index.  Idempotent: lr is always recomputed from the base
        lr, so re-firing never compounds."""
        k = self._lr_decay_outer
        self._lr_decay_outer += 1
        if self.lr_decay_every and self.lr_decay_every > 0:
            factor = self.lr_decay_gamma ** (k // self.lr_decay_every)
        else:
            factor = 1.0
        new_lr = self._lr_base * factor
        self.lr = new_lr
        opt = getattr(self, "optimizer", None)
        if opt is not None:
            for pg in opt.param_groups:
                pg["lr"] = new_lr

    def sample_simplex_v_ts(self, epoch=0):
        """Simplex over shares + uniform v + uniform t in [min_t, max_t]."""
        base = self.sample_simplex_v(epoch)                   # (B, K) -> shares + v
        B = base.shape[0]
        min_t = self.config.get("min_t", 0.0)
        max_t = self.config.get("max_t", 1.0)
        t0_frac = float(self.config.get("t0_frac", 0.4))
        n_t0 = int(round(t0_frac * B))
        t = min_t + (max_t - min_t) * torch.rand((B, 1), device=self.device)
        t[:n_t0] = min_t
        return torch.cat([base, t], dim=1)
    
    def sample_simplex_v_random_t(self, epoch=0):
        """Wealth-simplex + v + one uniform-random ``t`` in ``[min_t, max_t]`` per
        point.  Used to score the interior ``(x, v, t)`` domain in
        ``sample_rar_greedy`` (distinct from the training sampler, which
        over-weights ``t=0``)."""
        base = self.sample_simplex_v(epoch)                   # (B, K) -> shares + v
        min_t = self.config.get("min_t", 0.0)
        max_t = self.config.get("max_t", 1.0)
        t = min_t + (max_t - min_t) * torch.rand((base.shape[0], 1), device=self.device)
        return torch.cat([base, t], dim=1)
    
    def __sample_custom_boundary_cond(self, time_val: float):
        if self.boundary_uniform_points is None:
            self.boundary_uniform_points = self.sample_simplex_v(0)
        time_dim = torch.ones((self.boundary_uniform_points.shape[0], 1), device=self.device) * time_val
        return torch.cat([self.boundary_uniform_points, time_dim], dim=-1)

    def _score_pool(self, sampler, rounds=5):
        """Sample ``rounds`` dense pools with ``sampler`` and return
        ``(SV_cpu, residual_cpu)`` where the residual is the per-point sum of
        |endog| + |hjb| equation residuals (the same equilibrium objects the
        training loss uses).  Heavy tensors are moved to CPU between pools to
        keep peak memory bounded."""
        SVs, losses = [], []
        for _ in range(rounds):
            # Scoring only needs residual VALUES for the topk ranking -- never a
            # backward pass.  The state-derivatives (xi_Jac/xi_Hess) come from
            # torch.func transforms, which differentiate independently of the
            # outer grad mode, so we run the whole forward under no_grad.  Without
            # this, an autograd graph (incl. the (B,N,D,D) Hessians) is built and
            # held for every endog/HJB residual -> the VRAM growth seen in RAR.
            with torch.no_grad():
                SV = sampler()
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)

                Bn = SV.shape[0]
                total = torch.zeros((Bn, 1), device=self.device)

                def _per_sample(res):
                    a = torch.abs(res)
                    return a.reshape(Bn, -1).mean(dim=-1, keepdim=True)

                for label in self.endog_equations:
                    total = total + _per_sample(self.endog_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                for label in self.hjb_equations:
                    total = total + _per_sample(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))

                SVs.append(SV.detach().cpu()); losses.append(total.detach().cpu())
            del SV, total, vd_
            gc.collect(); torch.cuda.empty_cache()
        return torch.cat(SVs, 0), torch.cat(losses, 0)

    def _sample_simplex_at_t0(self):
        """Wealth-simplex pool pinned to ``t = min_t`` -- the slice we actually
        extract as the stationary solution."""
        base = self.sample_simplex_v(0)
        min_t = self.config.get("min_t", 0.0)
        t = torch.full((base.shape[0], 1), min_t, device=self.device, dtype=base.dtype)
        return torch.cat([base, t], dim=1)

    def sample_rar_greedy(self):
        """Residual-based anchor accumulation, mirroring the library's
        ``PDEModelTimeStep.sample_rar_greedy`` (which calls
        ``__get_refinement_loss_dict`` + topk + vstack onto anchor_points).

        IMPORTANT contract (matches the library): this method takes no epoch
        argument and does NOT resample/return a training batch.  It only GROWS
        ``self.anchor_points`` with the highest-residual points from a dense
        pool.  The library inner loop decides *when* to call it (a few epochs
        per outer loop, gated on epoch>0) and then vstacks anchor_points onto
        the current SV batch itself.

        We use the stacked forward (``update_variables``) and the simplex+t
        sampler, because the equilibrium residual is not reachable through the
        library's local_function_dict path, and the domain is the wealth simplex
        (uniform sampling would be invalid).
        """
        self.set_all_model_eval()
        saved_bs = self.batch_size
        if self.statics["K"] > 20:
            self.batch_size = 500
            sample_times = 10
        else:
            self.batch_size = 1000
            sample_times = 5

        n_keep = min(max(2, self.batch_size // self.refinement_rounds), 5000)
        k_t0 = max(1, n_keep // 2)            # budget pinned to the t=min_t slice
        k_int = max(1, n_keep - k_t0)         # budget on the interior (x, t) domain

        SV_int, l_int = self._score_pool(self.sample_simplex_v_random_t, sample_times)
        SV_t0, l_t0 = self._score_pool(self._sample_simplex_at_t0, sample_times)

        self.batch_size = saved_bs
        self.set_all_model_training()
        
        ids_int = torch.topk(l_int, min(k_int, SV_int.shape[0]), dim=0)[1].squeeze(-1)
        ids_t0 = torch.topk(l_t0, min(k_t0, SV_t0.shape[0]), dim=0)[1].squeeze(-1)
        new_anchors = torch.vstack((SV_int[ids_int], SV_t0[ids_t0])).detach().to(self.device)

        if self.anchor_points is None or self.anchor_points.numel() == 0:
            self.anchor_points = new_anchors
        else:
            self.anchor_points = torch.vstack((self.anchor_points, new_anchors))
        del SV_int, l_int, SV_t0, l_t0
        gc.collect(); torch.cuda.empty_cache()
        return new_anchors

    # -- outer-loop convergence / variable tracking -------------------------
    # The library's __check_outer_loop_converge rebuilds every tracked variable
    # from self.local_function_dict + self.equations.  Our equilibrium (chat, r,
    # mu_P, ...) is produced by the custom `update_variables` forward instead --
    # and r is now ANALYTIC (no network, no equation), so the library version
    # would KeyError on it.  We override (matching the mangled name so the call
    # at train_model resolves here) to run our forward, then measure the change
    # against prev_vals exactly as the base class does.
    def _PDEModelTimeStep__check_outer_loop_converge(self, SV_T0):
        temp_dict = {}
        self.update_variables(SV_T0, vd=temp_dict)

        new_vals = {k: temp_dict[k].detach() for k in self.prev_vals}

        max_abs_change = 0.0
        max_rel_change = 0.0
        all_changes = {}
        for k in self.prev_vals:
            mean_new_val = torch.mean(new_vals[k]).item()
            abs_change = torch.mean(torch.abs(new_vals[k] - self.prev_vals[k])).item()
            rel_change = torch.mean(torch.abs((new_vals[k] - self.prev_vals[k]) / self.prev_vals[k])).item()
            print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}")
            all_changes[f"{k}_mean_val"] = mean_new_val
            all_changes[f"{k}_abs"] = abs_change
            all_changes[f"{k}_rel"] = rel_change
            max_abs_change = max(max_abs_change, abs_change)
            max_rel_change = max(max_rel_change, rel_change)

        for k in self.prev_vals:
            self.prev_vals[k] = new_vals[k]

        total_rel_change = min(max_abs_change, max_rel_change)
        all_changes["total"] = total_rel_change
        return all_changes


# ===========================================================================
# Default economic parameters (original Di Tella calibration).
# ===========================================================================
BASE_PARAMS = {
    "a": 1.0, "sigma": 0.0125, "lbd": 1.38, "v_mean": 0.25, "sigv_mean": -0.17,
    "rho": 0.0665, "psi": 0.5, "tau": 1.15, "phi": 0.2,
    "A": 53.2, "B": -0.8668571428571438, "delta": 0.05,
}
V_DOMAIN = (0.05, 0.5)


def build_statics(K, expert_idx, household_idx, gamma_vec, caps_E, has_t, params=BASE_PARAMS):
    n_E = len(expert_idx)
    D = K + 1 if has_t else K
    statics = {
        "K": K, "D": D, "n_E": n_E,
        "expert_idx": list(expert_idx), "household_idx": list(household_idx),
        "v_index": K - 1, "has_t": has_t,
        "gamma": torch.tensor(gamma_vec, device=device, dtype=torch.get_default_dtype()).reshape(1, K),
        "caps_E": torch.tensor(caps_E, device=device, dtype=torch.get_default_dtype()).reshape(1, n_E),
        "v_domain": V_DOMAIN,
    }
    for k, val in params.items():
        statics[k] = val
    return statics


# ===========================================================================
# Model assembly
# ===========================================================================
def get_model(model_path, K, expert_idx, household_idx, gamma_vec, caps_E,
              model_size, n_epochs=20000, batch_size=500, lr=1e-3,
              timestepping=False, rar=False, loss_balancing=False,
              params=BASE_PARAMS, train=True, num_outer=70, num_inner=5000,
              min_inner=1000, loss_log_interval=50, max_t=1.0, init_guess=None,
              share_alpha_lo=SHARE_ALPHA_LO, share_alpha_hi=SHARE_ALPHA_HI,
              lr_decay_every=20, lr_decay_gamma=0.5, loss_balancing_alpha=0.9, loss_balancing_temp=0.1, bernoulli_prob=0.99, foc=False,
              t0_frac=0.4):
    """Assemble (and train if no checkpoint) the heterogeneous N-agent SV model.

    expert_idx / household_idx : 0-based agent indices (their union is 0..K-1).
    gamma_vec                  : length-K risk-aversion vector.
    caps_E                     : length-len(expert_idx) leverage caps (>=1e3 = unconstrained);
                                 the first expert MUST be unconstrained (anchors chi).
    init_guess                 : optional {name: value} seed for the time-boundary in
                                 time-stepping.
    """
    set_seeds(0)
    assert caps_E[0] >= 1e3, "First expert anchors chi and must be unconstrained (caps_E[0] >= 1e3)."

    # TODO: set the initial batch size to be half for RAR as an experiment
    if rar:
        batch_size = batch_size // 2
    
    if timestepping:
        cfg = {"batch_size": batch_size, "time_batch_size": 1,
               "min_t": 0.0, "max_t": max_t,
               # RARG enables the library's inner-loop residual-refinement hook,
               # which calls self.sample_rar_greedy() and vstacks anchor_points.
               "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
               "num_outer_iterations": num_outer, "num_inner_iterations": num_inner,
               "min_inner_iterations": min_inner, "loss_log_interval": loss_log_interval,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10,
               "share_alpha_lo": share_alpha_lo, "share_alpha_hi": share_alpha_hi,
               "lr_decay_every": lr_decay_every, "lr_decay_gamma": lr_decay_gamma,
               "foc": foc, "loss_balancing_alpha": loss_balancing_alpha, "loss_balancing_temp": loss_balancing_temp, "bernoulli_prob": bernoulli_prob,
               # t0-mix training sampler: fraction of each batch pinned to t=min_t
               "t0_frac": t0_frac,
        }
        model = PDEModelTimeStepNAgentsSV("sv_n_agents", cfg)
    else:
        cfg = {"batch_size": batch_size, "num_epochs": n_epochs,
               "sampling_method": SamplingMethod.UniformRandom,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10,
               "share_alpha_lo": share_alpha_lo, "share_alpha_hi": share_alpha_hi,
               "foc": foc, "loss_balancing_alpha": loss_balancing_alpha, "loss_balancing_temp": loss_balancing_temp, "bernoulli_prob": bernoulli_prob,
            }
        model = PDEModelNAgentsSV("sv_n_agents", cfg)

    state_names = [f"x_{i+1}" for i in range(K - 1)] + ["v"]
    domain = {f"x_{i+1}": [0.1 / K, 1 - 0.1 / K] for i in range(K - 1)}
    domain["v"] = list(V_DOMAIN)
    model.set_state(state_names, domain)
    model.add_params(params)
    model.statics = build_statics(K, expert_idx, household_idx, gamma_vec, caps_E,
                                  has_t=timestepping, params=params)

    net_cfg = {"hidden_units": model_size, "derivative_order": 0, "batch_jac_hes": False}
    for k in range(1, K + 1):
        model.add_agent(f"xi_{k}", config={**net_cfg, "positive": True})
    # p is a FREE network (capital price), pinned by the goods-market clearing
    # soft-loss + asset-pricing residual -- exactly as in the original 2-agent
    # model.  (Deriving p analytically from xi instead slaves dp/dt to dxi/dt,
    # which destabilises the backward time-stepping march to the low-p branch.)
    model.add_endog("p", config={**net_cfg, "positive": True})
    # r is a FREE network (can be negative -> NOT positive).  The asset-pricing
    # residual (added below) is the ONLY equation containing mu_P, so without a
    # free r + this residual the curvature of p is unconstrained.
    model.add_endog("r", config={**net_cfg})
    # only NON-anchor experts get a theta network; the anchor expert holds the
    # residual capital (theta_anchor = 1 - sum others) so clearing is exact.
    # In interior-FOC mode theta is hard-wired from the FOC (see _compute_theta_E),
    # so NO theta networks are created.
    theta_names = [] if foc else [f"theta_{expert_idx[i]+1}" for i in range(1, len(expert_idx))]
    for nm in theta_names:
        model.add_endog(nm, config={**net_cfg, "positive": True})

    xi_names = [f"xi_{k}" for k in range(1, K + 1)]
    model.attach_stacks(xi_names, theta_names)

    # placeholder entries so equation registration / validation has shapes
    bsz = batch_size
    n_E = len(expert_idx)
    nm_shp_list = [("goods_resid", (bsz, 1)), ("asset_pricing_resid", (bsz, 1)),
                    ("vi_expert_resid", (bsz, n_E)), ("sig_clearing_resid", (bsz, 1)),
                    ("hjb_expert", (bsz, 1)), ("hjb_household", (bsz, 1)),
                    # extra quantities exposed only for variables_to_track (see
                    # the __check_outer_loop_converge override on the time-step
                    # model); chat/mu_P/r_implied are intermediates.
                    ("chat", (bsz, K)), ("r", (bsz, 1)), ("r_implied", (bsz, 1)),
                    ("mu_P", (bsz, 1)), ("p", (bsz, 1)),
        ]
    for idx in expert_idx:
        nm_shp_list.append((f"hjb_expert_{idx}", (bsz, 1)))
    for idx in household_idx:
        nm_shp_list.append((f"hjb_household_{idx}", (bsz, 1)))
    for nm, shp in nm_shp_list:
        model.variable_val_dict[nm] = torch.zeros(shp, device=model.device)

    # ---- equilibrium-residual losses (spec section 6) ---------------------
    model.add_endog_equation("goods_resid = 0", label="goods")
    model.add_endog_equation("asset_pricing_resid = 0", label="asset_pricing")
    if K > 2:
        model.add_endog_equation("vi_expert_resid = 0", label="vi_expert")
    model.add_endog_equation("sig_clearing_resid = 0", label="sig_clearning")
    # hjb_expert/_household are ALREADY sum_k hjb_k**2, so MAE reduction gives
    # mean(sum_k hjb_k**2) == MSE of the raw residual (matching the original).
    # (MSE here would square again -> mean(hjb_k**4), which is wrong.)
    # model.add_hjb_equation("hjb_expert", label="expert", loss_reduction=LossReductionMethod.MAE)
    # model.add_hjb_equation("hjb_household", label="household", loss_reduction=LossReductionMethod.MAE)
    for idx in expert_idx:
        model.add_hjb_equation(f"hjb_expert_{idx}", label=f"expert_{idx}", loss_reduction=LossReductionMethod.MAE)
    for idx in household_idx:
        model.add_hjb_equation(f"hjb_household_{idx}", label=f"household_{idx}", loss_reduction=LossReductionMethod.MAE)

    if train and not os.path.exists(f"{model_path}/model.pt"):
        os.makedirs(model_path, exist_ok=True)
        if timestepping and init_guess:
            # seed the backward march in the correct basin (default guess of 1
            # leaves p pinned ~1 and goods_resid stuck ~1; see get_model docstring)
            model.set_initial_guess(init_guess)
        # if timestepping and homotopy:
        #     # ramp gamma from an easy regime to the target ALONG the backward
        #     # march -- continuation for free (one march, no per-step retrain).
        #     # Target gamma is already in model.statics; pass only the start,
        #     # delay and ramp fraction (stepping cadence comes from homotopy_step).
        #     model.enable_homotopy(**homotopy)
        model.train_model(model_path, "model.pt", full_log=True,
                          variables_to_track=["chat", "r", "r_implied", "p", "mu_P"])
    if os.path.exists(f"{model_path}/model_best.pt"):
        model.load_model(torch.load(f"{model_path}/model_best.pt", weights_only=False,
                                    map_location=device))
        model.attach_stacks(xi_names, theta_names)
    return model


# ===========================================================================
# Cases
# ===========================================================================
def make_case(case: str, gamma):
    """Return (K, expert_idx, household_idx, gamma_vec, caps_E)."""
    if case == "agents2":
        # 1 expert + 1 household, original Di Tella gamma=5, no cap -> validation
        K = 2
        expert_idx = [0]; household_idx = [1]
        gamma_vec = [gamma, gamma]
        caps_E = [1e6]
    elif case == "agents5":
        # 4 experts + 1 household.  Match the 2-agent aggregate risk premium
        # (gamma=6): keep the wealth-weighted harmonic mean of gamma ~ 6.
        # Experts straddle 6 (slightly tolerant -> they manage capital), the
        # household is a bit more averse but holds little wealth, so it barely
        # shifts the aggregate.  Centred just above 6 to offset the wealth
        # concentration on the low-gamma experts.  Max gamma kept <= 8 (trainable).
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
        # 18 experts + 2 households.  HARD calibration: a WIDE risk-aversion
        # spread so capital concentrates on the low-gamma experts, which then
        # take large leverage theta_k/x_k -- restoring the 1/x stiffness that
        # diversification otherwise washes out (see the difficulty analysis),
        # while staying a genuine multi-agent economy.  The anchor (index 0) is
        # the LEAST averse, so it endogenously holds the most capital.  Experts
        # span [3, 10] (aggressive low end drives the stiffness); households are
        # more averse (12, 14) and hold little wealth.  NOTE: the harmonic mean
        # is now well below 6, so the aggregate risk premium will differ from
        # the 2-agent calibration -- that is intended (a harder, separating case).
        K = 20
        n_E = 18
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 7.0 * i / (n_E - 1) for i in range(n_E)] + [12.0, 14.0]
        caps_E = [1e6] * n_E
    elif case == "agents40":
        # 36 experts + 4 households.  Same HARD design as agents20/agents50:
        # wide expert spread [3, 10.5] (anchor index 0 = least averse, so capital
        # concentrates there and takes large leverage), households more averse
        # [12..16.5] holding little wealth.  gamma_vec length K (experts then hh).
        K = 40
        n_E = 36
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 7.5 * i / (n_E - 1) for i in range(n_E)] + [12.0, 13.5, 15.0, 16.5]
        caps_E = [1e6] * n_E
    elif case == "agents50":
        # 45 experts + 5 households.  Same HARD design as agents20: wide expert
        # spread [3, 11] (anchor = least averse -> capital concentrates there),
        # households [12..16].  gamma_vec has length K (experts then households).
        K = 50
        n_E = 45
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 8.0 * i / (n_E - 1) for i in range(n_E)] + [12.0, 13.0, 14.0, 15.0, 16.0]
        caps_E = [1e6] * n_E
    else:
        raise ValueError(f"unknown case {case!r}")
    return K, expert_idx, household_idx, gamma_vec, caps_E


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
            "vi_expert_resid", "goods_resid", "asset_pricing_resid", "sigtilde_full",
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
    shares = _mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
    x_states = shares[:, :K - 1]
    vlo, vhi = V_DOMAIN
    v = vlo + (vhi - vlo) * rng.random((n_samples, 1))
    return np.concatenate([x_states, v], axis=1)


def compute_validation_losses(model, SV_val, chunk_size=2000):
    """Per-component validation losses + per-type HJB + total, on common states.

    Residual components use the same reductions as training: HJB groups are the
    mean of sum_k hjb_k**2 (== MSE of the raw residual), and the equilibrium
    constraints (goods, asset pricing, capital FOC) are MSE.  The capital FOC
    (vi_expert) only enters for K > 2 (for K=2 the lone expert holds all
    capital, so there is no free theta and no FOC residual to satisfy).
    """
    with model_on(model):
        out = _forward_states(model, SV_val, chunk_size)
    res = {
        "HJB expert": float(np.mean(np.abs(out["hjb_expert"]))),
        "HJB household": float(np.mean(np.abs(out["hjb_household"]))),
        "Goods clearing": float(np.mean(out["goods_resid"] ** 2)),
        "Asset pricing": float(np.mean(out["asset_pricing_resid"] ** 2)),
    }
    if model.statics["K"] > 2:
        res["Capital FOC"] = float(np.mean(out["vi_expert_resid"] ** 2))
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
    if model.statics["K"] > 2:
        res["Capital FOC"] = float(np.mean(out["vi_expert_resid"] ** 2))
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
            

# ===========================================================================
# Tables
# ===========================================================================
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
    # to the case (e.g. Capital FOC only appears for K > 2).
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
      capital FOC (vi)  ->  1/2 gamma_k (phi v)^2 (d theta_k/x_k)^2  (second order)
    """
    SV_val = _validation_states(next(iter(models.values())), n_samples=n_samples, seed=seed)
    rows = {}
    for name, model in models.items():
        st = model.statics
        rho = st["rho"]; phi = st["phi"]
        gamma = st["gamma"].detach().cpu().numpy().reshape(-1)
        e_idx = st["expert_idx"]; v_index = st["v_index"]
        with model_on(model):
            out = _forward_states(model, SV_val, chunk_size)
        v = SV_val[:, v_index]
        hjb_k = out["hjb_k"]
        x_full = np.concatenate([SV_val[:, :st["K"] - 1],
                                 1.0 - SV_val[:, :st["K"] - 1].sum(axis=1, keepdims=True)], axis=1)
        hjb_we = float(np.mean(rho * np.abs(hjb_k)))
        # capital FOC welfare (experts only)
        x_E = x_full[:, e_idx]
        g_E = gamma[e_idx].reshape(1, -1)
        vi = out["vi_expert_resid"]
        vi_we = float(np.mean(0.5 * g_E * (phi * v[:, None]) ** 2 * (vi / (x_E + 1e-8)) ** 2))
        total = hjb_we + vi_we
        rows[name] = {"HJB (c/W)": hjb_we, "Capital FOC (c/W)": vi_we, "total (c/W)": total}
    base = rows[baseline_key]
    for name, row in rows.items():
        for c in list(base.keys()):
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    abs_cols = ["HJB (c/W)", "Capital FOC (c/W)", "total (c/W)"]
    ordered = abs_cols + [f"{c} impr." for c in abs_cols]
    renamed_rows = {METHOD_DISPLAY[name]: row for name, row in rows.items()}
    return pd.DataFrame.from_dict(renamed_rows, orient="index")[ordered]


def df_to_latex(df, path):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(format_pct if "impr" in col else format_sci)
    with open(path, "w") as f:
        f.write(out.style.to_latex(hrules=True))


FD_TABLE_VARS = {
    "omega":        r"$\Omega=\xi/\zeta$",
    "e_hat":        r"$\hat{e}$",
    "c_hat":        r"$\hat{c}$",
    "risk_premium": r"$\pi(\sigma+\sigma_p)$",
}


def _fd_v_slices(fd_dict, var="omega"):
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


# ===========================================================================
# Plots
# ===========================================================================
SLICE_PLOT_ARGS = {
    "p": {"ylabel": r"$p$"},
    "sigx": {"ylabel": r"$\sigma_x$"},
    "omega": {"ylabel": r"$\Omega=\xi/\zeta$"},
    "sigsigp": {"ylabel": r"$\sigma+\sigma_p$"},
    "signxi": {"ylabel": r"$\pi$"},
    "r": {"ylabel": r"$r$"},
    "risk_premium": {"ylabel": r"$\pi(\sigma+\sigma_p)$"}
}
SLICE_COLORS = ["red", "orange", "blue"]


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
               "endogeq_vi_expert": "Capital FOC", "hjbeq_expert": "HJB experts",
               "hjbeq_household": "HJB households"}
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
        anchors = np.load(os.path.join(adir, files[-1]))
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
    shares = _mixture_shares_np(n_samples, K, eps, rng, alpha_lo, alpha_hi)
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

# ===========================================================================
# The 8 training configurations
# ===========================================================================
# name -> (timestepping, rar, loss_balancing)
CONFIGS = {
    "basic":            (False, False, False),
    "basic_rar":        (False, True,  False),
    # "basic_lb":         (False, False, True),
    # "basic_rar_lb":     (False, True,  True),
    "timestep":         (True,  False, False),
    "timestep_rar":     (True,  True,  False),
    # "timestep_lb":      (True,  False, True),
    # "timestep_rar_lb":  (True,  True,  True),
}


# Per-method plot styling so basic vs. the best method are separable in B&W.
METHOD_PLOT_STYLES = [
    ("-",  ""),
    ("--", "o"),
    ("-.", "s"),
    (":",  "^"),
]


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


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    """Train all configs for one case + emit the comparison artifacts."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["agents2", "agents5", "agents5_cap", "agents5_cap2", "agents20", "agents40", "agents50"], default="agents2")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--outer", type=int, default=100, help="num_outer_iterations for time-stepping configs")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_balancing_temp", type=float, default=0.1)
    parser.add_argument("--loss_balancing_alpha", type=float, default=0.9)
    parser.add_argument("--bernoulli_prob", type=float, default=0.99)
    parser.add_argument("--num-inner", type=int, default=5000,
                        help="num_inner_iterations at outer 0 (the library decays it ~1/sqrt(outer) down to --min-inner)")
    parser.add_argument("--min-inner", type=int, default=1000,
                        help="floor for the per-outer inner iterations")
    parser.add_argument("--lr-decay-every", type=int, default=20,
                        help="multiply the LR by --lr-decay-gamma every N outer (time-stepping) iterations")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5,
                        help="LR step-decay factor applied every --lr-decay-every outer iterations")
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=1.15)
    parser.add_argument("--a", type=float, default=1)
    parser.add_argument("--sigma", type=float, default=0.0125)
    parser.add_argument("--alpha-lo", type=float, default=SHARE_ALPHA_LO,
                        help="lower bound of the log-uniform Dirichlet-alpha mixture for wealth-share sampling (alpha<1 => concentrated draws)")
    parser.add_argument("--alpha-hi", type=float, default=SHARE_ALPHA_HI,
                        help="upper bound of the log-uniform Dirichlet-alpha mixture for wealth-share sampling")
    parser.add_argument("--foc", action="store_true",
                        help="interior-FOC mode: hard-wire expert theta from the FOC (no theta networks); requires all experts uncapped")
    parser.add_argument("--t0-frac", type=float, default=0.4,
                        help="fraction of each time-stepping training batch pinned to t=min_t (t0-mix sampler)")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    gamma = args.gamma
    tau = args.tau
    a = args.a
    sigma = args.sigma

    # homotopy = None
    # if args.homotopy:
    #     homotopy = dict(gamma_start=args.gamma_start, ramp_frac=args.ramp_frac,
    #                     delay=args.homotopy_delay)

    # free p AND free r (goods + asset-pricing residuals) -> "free_pr".  New dir
    # so we don't reload incompatible analytic-r checkpoints (no "r" endog).
    dir_tag = "_FOC" if args.foc else ""
    if args.t0_frac > 0:
        dir_tag += f"_t0frac{args.t0_frac}"
    if args.float64:
        torch.set_default_dtype(torch.float64)
        base_dir = f"./models/SV_NAgents_64bit_260713{dir_tag}_{gamma}_{tau}_{sigma}_{a}/{args.case}"
    else:
        torch.set_default_dtype(torch.float32)
        base_dir = f"./models/SV_NAgents_260713{dir_tag}_{gamma}_{tau}_{sigma}_{a}/{args.case}"

    K, eidx, hidx, gamma_vec, caps_E = make_case(args.case, gamma)
    print(f"[sv_n_agents] case={args.case} K={K} experts={eidx} households={hidx}")
    print(f"             gamma={gamma_vec}  caps_E={caps_E}")

    ts_init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    ts_init_guess["r"] = 0.01

    models, model_paths, ts_map = {}, {}, {}
    for name in list(CONFIGS.keys()):
        ts, rar, lb = CONFIGS[name]
        mpath = os.path.join(base_dir, name)
        print(f"\n{('=== ' + name + ' ==='):=^80}")
        model = get_model(
            mpath, K, eidx, hidx, gamma_vec, caps_E,
            model_size=[args.width] * args.layers,
            n_epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            timestepping=ts, rar=rar, loss_balancing=lb, num_outer=args.outer,
            num_inner=args.num_inner, min_inner=args.min_inner,
            lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
            loss_balancing_alpha=args.loss_balancing_alpha, loss_balancing_temp=args.loss_balancing_temp, bernoulli_prob=args.bernoulli_prob,
            foc=args.foc, t0_frac=args.t0_frac,
            init_guess=ts_init_guess, params=BASE_PARAMS | {"tau": tau, "a": a, "sigma": sigma},
            share_alpha_lo=args.alpha_lo, share_alpha_hi=args.alpha_hi,
            # homotopy=homotopy, homotopy_step=args.homotopy_steps,
        )
        # park the trained model on CPU so the next config's training (and the
        # later evaluation) doesn't have every config's networks pinned in VRAM.
        move_model(model, "cpu")
        models[name] = model
        model_paths[name] = mpath
        ts_map[name] = ts
        gc.collect(); torch.cuda.empty_cache()

    cmp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    if K >= 10:
        chunk_size = 500
    else:
        chunk_size = 2000

    # ---- 1) Loss tables (computed over ALL trained methods) -----------------
    # Two flavours of validation sampling for the time-stepping rows: the
    # default scores them at t=min_t (the stationary slice we extract); the
    # "_random_t" variant scores them at a random t over [min_t, max_t] so the
    # whole trained horizon is validated (see compute_validation_losses_random_t).
    loss_df = welfare_df = None
    if "basic" in models:
        loss_df = compare_loss_table(models, baseline_key="basic", chunk_size=chunk_size)
        loss_df.to_csv(os.path.join(cmp_dir, "comparative_losses.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, "comparative_losses.tex"))
        print("\n[sv_n_agents] comparative loss table (vs basic):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        loss_df_rt = compare_loss_table(models, baseline_key="basic", chunk_size=chunk_size,
                                        compute_fn=compute_validation_losses_random_t)
        loss_df_rt.to_csv(os.path.join(cmp_dir, "comparative_losses_random_t.csv"))
        df_to_latex(loss_df_rt, os.path.join(cmp_dir, "comparative_losses_random_t.tex"))
        print("\n[sv_n_agents] comparative loss table (random t, vs basic):")
        print(loss_df_rt.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key="basic", chunk_size=chunk_size)
        welfare_df.to_csv(os.path.join(cmp_dir, "welfare_equivalent_losses.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, "welfare_equivalent_losses.tex"))
        print("\n[sv_n_agents] welfare-equivalent loss table (c/W, vs basic):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    if "timestep" in models:
        loss_df = compare_loss_table(models, baseline_key="timestep", chunk_size=chunk_size)
        loss_df.to_csv(os.path.join(cmp_dir, "comparative_losses_timestep.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, "comparative_losses_timestep.tex"))
        print("\n[sv_n_agents] comparative loss table (vs timestep):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        loss_df_rt = compare_loss_table(models, baseline_key="timestep", chunk_size=chunk_size,
                                        compute_fn=compute_validation_losses_random_t)
        loss_df_rt.to_csv(os.path.join(cmp_dir, "comparative_losses_timestep_random_t.csv"))
        df_to_latex(loss_df_rt, os.path.join(cmp_dir, "comparative_losses_timestep_random_t.tex"))
        print("\n[sv_n_agents] comparative loss table (random t, vs timestep):")
        print(loss_df_rt.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key="timestep", chunk_size=chunk_size)
        welfare_df.to_csv(os.path.join(cmp_dir, "welfare_equivalent_losses_timestep.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, "welfare_equivalent_losses_timestep.tex"))
        print("\n[sv_n_agents] welfare-equivalent loss table (c/W, vs timestep):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    # ---- pick basic + the best-improving method(s) for the overlay plots ----
    plot_models = select_plot_methods(models, loss_df=loss_df, welfare_df=welfare_df, baseline_key="basic")
    plot_paths = {k: model_paths[k] for k in plot_models}
    plot_ts = {k: ts_map[k] for k in plot_models}
    print(f"[sv_n_agents] plotting methods: {list(plot_models)}")

    # ---- 2) 2-D slice comparison vs the Di Tella FD solution (agents2 only) --
    if args.case == "agents2":
        try:
            fd_dict = np.load(f"./models/numerical/numerical_{gamma}_{tau}_{sigma}_{a}.npz")
        except Exception as e:
            print(f"[sv_n_agents] could not load FD solution: {e}")
            fd_dict = None
        v_list = [0.25]
        method_dicts = {name: evaluate_slices(m, v_list) for name, m in plot_models.items()}
        plot_slice_comparison(method_dicts, fd_dict, v_list, cmp_dir)
        print(f"[sv_n_agents] slice comparison plots saved to {cmp_dir}")

        # FD-error table (omega, e_hat, c_hat, risk_premium) over ALL methods,
        # averaged across every v-slice the FD solution provides.
        if fd_dict is not None:
            v_tab = _fd_v_slices(fd_dict) or v_list
            all_method_dicts = {name: evaluate_slices(m, v_tab) for name, m in models.items()}
            mse_df, mae_df = compare_fd_table(all_method_dicts, fd_dict, v_tab, cmp_dir)
            print(f"\n[sv_n_agents] FD-error table (MSE vs FD, v={v_tab}):")
            print(mse_df.to_string(float_format=lambda x: f"{x:.3e}"))
    else:
        # plot histograms + aggregate scatter (p / risk premium / omega vs the
        # total expert wealth share at v=0.25) for the high-dimensional cases.
        for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
            plot_theta_chat_histogram(models[name], model_paths[name], chunk_size=chunk_size)
            plot_aggregate_scatter(models[name], cmp_dir, file_name=f"aggregate_scatter_{name}.pdf", chunk_size=chunk_size)

    # ---- 3) RAR anchor scatter (rar configs, K=2) ---------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if rar:
            plot_rar_anchors(model_paths[name], K, cmp_dir, file_name=f"rar_anchors_{name}.pdf", timestepping=ts)

    # ---- 4) Loss-weight evolution (lb configs) ------------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if lb:
            plot_loss_weights(model_paths[name], cmp_dir, file_name=f"loss_weight_{name}.pdf", timestepping=ts)

    # ---- 5) HJB / total loss convergence (basic + best method) --------------
    plot_paths = {k: model_paths[k] for k in ["basic", "timestep_rar", "timestep_rar_lb"]}
    plot_ts = {k: ts_map[k] for k in ["basic", "timestep_rar", "timestep_rar_lb"]}
    plot_loss_decay(plot_paths, cmp_dir, plot_ts)
    print(f"[sv_n_agents] loss-decay plot saved to {cmp_dir}")

    print(f"\n[sv_n_agents] all artifacts written under {cmp_dir}")


if __name__ == "__main__":
    main()

    # --case agents2 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 6.0
    # --case agents2 --float64 --a 0.2 --sigma 0.06 --tau 1.15 --gamma 6.0
    # --case agents5 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 6.0
    # --case agents5_cap --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 6.0
    # --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 5.0
    # --case agents20 --float64 --a 0.1 --sigma 0.02 --tau 1.15 --gamma 5.0
    # --case agents50 --float64 --a 0.1 --sigma 0.02 --tau 1.15 --gamma 5.0
    # --case agents5 --float64 --a 0.1 --sigma 0.02 --tau 1.15 --gamma 5.0
    # --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 5.0
    # --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 5.0 --foc --loss_balancing_temp 1.0
    # --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 5.0 --foc --min-inner 2000