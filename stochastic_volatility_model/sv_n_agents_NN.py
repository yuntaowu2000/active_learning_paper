from __future__ import annotations

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

import gc
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, hessian, jacrev, vmap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deep_macrofin import (LossReductionMethod, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, set_seeds)

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    def analytic_p(self, SV: torch.Tensor, statics):
        """Capital price solved from goods-market clearing, treating these
        stacked networks as the ``xi`` value functions.

        Goods clearing  a - iota(p) = p * sum_k x_k chat_k  with
        chat_k = rho^(1/psi) xi_k^((psi-1)/psi)  (no p-dependence) and
        iota(p) = A (g+delta)^2 + B(g+delta), g+delta = (p-B)/(2A) is a
        QUADRATIC in p whose positive root is

            p = sqrt(4 A^2 C^2 + 4 A a + B^2) - 2 A C ,   C = sum_k x_k chat_k .

        First/second state-derivatives of p are obtained by autodiff *through*
        the xi networks (so p_Hess needs only xi's 2nd derivatives -- same
        differentiation order as a free p-network).  Returns
        ``(p (B,1), p_Jac (B,1,D), p_Hess (B,1,D,D))``.
        """
        models = [a.model for a in self._agents]
        per_p = [dict(m.named_parameters()) for m in models]
        per_b = [dict(m.named_buffers()) for m in models]
        K = statics["K"]
        A = statics["A"]; Bc = statics["B"]; a = statics["a"]
        rho = statics["rho"]; psi = statics["psi"]
        alpha = (psi - 1.0) / psi
        coef = rho ** (1.0 / psi)

        def p_scalar(x):                                   # x: (D,) single sample
            xis = [functional_call(m, (pa, bu), (x,)).squeeze(-1)
                   for m, pa, bu in zip(models, per_p, per_b)]
            xi_vec = torch.stack(xis)                      # (K,)
            x_states = x[:K - 1]
            x_K = 1.0 - x_states.sum()
            x_full = torch.cat([x_states, x_K.reshape(1)])  # (K,)
            chat = coef * xi_vec ** alpha
            C = (x_full * chat).sum()
            return torch.sqrt(4.0 * A * A * C * C + 4.0 * A * a + Bc * Bc) - 2.0 * A * C

        B0, D = SV.shape[0], SV.shape[1]
        p_val = vmap(p_scalar)(SV).reshape(B0, 1)
        p_jac = vmap(jacrev(p_scalar))(SV).reshape(B0, 1, D)
        p_hess = vmap(hessian(p_scalar))(SV).reshape(B0, 1, D, D)
        return p_val, p_jac, p_hess


# ===========================================================================
# Economic core: one differentiable forward pass computing every equilibrium
# object from the raw network outputs.  See spec sections 2-6.
# ===========================================================================
def compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, statics=None):
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
    capital_resid = theta_E.sum(dim=1, keepdim=True) - 1.0       # (B, 1)

    # full-K idiosyncratic exposure (households 0)
    sigtilde_full = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    sigtilde_full[:, expert_idx] = sigtilde_E
    chi_theta_over_x_full = torch.zeros((B_, K), device=SV.device, dtype=SV.dtype)
    chi_theta_over_x_full[:, expert_idx] = chi * theta_E / x_E

    # ---- share drifts (r-independent, see spec section 5) -----------------
    # net-worth drift with r = 0 (r cancels in mu_x); + chi theta/x for experts
    mu_net0 = pi * sign_k + chi_theta_over_x_full                # (B, K)
    agg_cons = (a - iota) / p                                  # (B, 1) = C/N
    # mu_N0 = (x_full * mu_net0).sum(dim=1, keepdim=True) - agg_cons  # (B, 1)
    mu_N0_ = pi * sig_agg + g_E * phiv2 / x_E - agg_cons
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

    # ---- risk-free rate: solved ANALYTICALLY from the asset-pricing eq -----
    # asset pricing for the aggregate capital claim (the anchor expert holds it):
    #   (a-iota)/p + g + mu_P + sigma*sigp - r = sig_agg*pi + chi
    # Instead of making r a free network (which leaves the combination (mu_P - r)
    # under-identified by this single equation and lets r ratchet away across the
    # time-stepping boundary), we SOLVE the equation for r:
    #   r = (a-iota)/p + g + mu_P + sigma*sigp - sig_agg*pi - chi
    # so asset_pricing_resid == 0 by construction (dropped as a loss).  r then
    # enters the HJBs via mu_net, so the curvature of p (through mu_P) is pinned
    # by the HJB residuals -- there is no free r variable left to diverge.
    r = ((a - iota) / p + g + mu_P + sigma * sigp - sig_agg * pi - chi)   # (B, 1)
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

    out = {
        "x_full": x_full, "theta_full": theta_full, "chat": chat,
        "sigx_full": sigx_full, "sigp": sigp, "sig_agg": sig_agg,
        "sigxi": sigxi, "pi": pi, "sign_k": sign_k, "chi": chi,
        "mu_x_full": mu_x_full, "mu_xi": mu_xi, "mu_P": mu_P, "r": r, "mu_net": mu_net,
        "hjb_k": hjb_k, "hjb_expert": hjb_expert, "hjb_household": hjb_household,
        "capital_resid": capital_resid, "sig_clearing_resid": sig_clearing_resid,
        "vi_expert_resid": vi_expert_resid, "xi_ret": xi_ret,
        "g": g, "iota": iota, "sigtilde_full": sigtilde_full,
    }
    return out


# ===========================================================================
# PDEModel subclass: simplex+v sampler, RAR override, fused forward.
# ===========================================================================
class _SVNAgentMixin:
    """Shared logic for the stationary and time-stepping N-agent SV models."""

    def _sv_init(self, config):
        self.rar = config.get("rar", False)
        self.statics = None
        self._xi_stack = None
        self._theta_stack = None
        self._p_stack = None
        self._skip_local_keys = set()
        # disable the per-epoch diagnostic (expensive, unused here)
        try:
            self._PDEModel__compute_changes = lambda SV: {"total": 0.0}
        except Exception:
            pass

    # -- stack attachment ----------------------------------------------------
    def attach_stacks(self, xi_names, theta_names):
        # `theta_names` are the NON-anchor experts only; the anchor expert's
        # capital share is pinned by clearing (theta_anchor = 1 - sum(others)),
        # so capital-market clearing holds *by construction*.
        self._xi_stack = (StackedAgent([self.agents[n] for n in xi_names], value_only=False), list(xi_names))
        if theta_names:
            self._theta_stack = (StackedAgent([self.endog_vars[n] for n in theta_names], value_only=True), list(theta_names))
        else:
            self._theta_stack = (None, [])
        self._skip_local_keys.update(xi_names)
        self._skip_local_keys.update(theta_names)
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

    # -- forward / equation evaluation --------------------------------------
    def update_variables(self, SV, vd=None):
        if vd is None:
            vd = self.variable_val_dict
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            vd[sv_name] = SV[:, i:i+1]
        vd["SV"] = SV

        xi, xi_Jac, xi_Hess = self._xi_stack[0].compute(SV)
        # p (and its state-derivatives) solved analytically from goods clearing,
        # differentiated through the xi networks -- no p network.
        p, p_Jac, p_Hess = self._xi_stack[0].analytic_p(SV, self.statics)

        # anchor expert holds residual capital: theta_anchor = 1 - sum(others)
        B_ = SV.shape[0]
        if self._theta_stack[0] is not None:
            theta_others = self._theta_stack[0].value(SV)              # (B, n_E-1)
            theta_anchor = 1.0 - theta_others.sum(dim=1, keepdim=True)  # (B, 1)
            theta_E = torch.cat([theta_anchor, theta_others], dim=1)   # (B, n_E)
        else:
            theta_E = torch.ones((B_, 1), device=SV.device, dtype=SV.dtype)

        out = compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, statics=self.statics)

        # expose per-agent names (cheap slices) for any direct references
        for i, n in enumerate(self._xi_stack[1]):
            vd[n] = xi[:, i:i + 1]
        # non-anchor experts occupy columns 1.. of theta_E (col 0 = anchor)
        for i, n in enumerate(self._theta_stack[1]):
            vd[n] = theta_E[:, i + 1:i + 2]
        vd["p"] = p
        # r comes from out (analytic); vd.update(out) below sets vd["r"].
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

    def sample_simplex_v(self, epoch):
        """Dirichlet over the K wealth shares (drop residual) + uniform v.

        Shares are floored at ``eps`` (each in [eps, 1-(K-1)eps]) so the full
        simplex sums to 1 exactly; eps=0.05 matches the original 2-agent domain
        x in [0.05, 0.95] and avoids the 1/x blow-up in the expert HJB.
        """
        K = self.statics["K"]
        eps = 0.05
        alpha = torch.ones(K, device=self.device)
        shares = torch.distributions.Dirichlet(alpha).sample((self.batch_size,))
        shares = eps + (1.0 - K * eps) * shares               # sums to 1
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
        self.batch_size = 1000
        for _ in range(10):
            torch.cuda.empty_cache()
            SV = self.sample_simplex_v(epoch)
            SV.requires_grad_(True)
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
            del SV, total
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
        if self.rar:
            self.sample = self.sample_rar_greedy
            self.sampling_method = SamplingMethod.RARG
        else:
            self.sample = self.sample_simplex_v_ts

        self.sample_boundary_cond = self.__sample_custom_boundary_cond
        self.boundary_uniform_points = None

    def sample_simplex_v_ts(self, epoch=0):
        """Simplex over shares + uniform v + uniform t in [min_t, max_t]."""
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

    def sample_rar_greedy(self, epoch=0):
        # mirror the library: only accumulate anchors, vstack onto a fresh batch
        if epoch % max(1, self.num_inner_iterations // self.refinement_rounds) == 0 and epoch > 0:
            self.set_all_model_eval()
            SVs, losses = [], []
            saved_bs = self.batch_size
            self.batch_size = 1000
            for _ in range(5):
                SV = self.sample_simplex_v_ts(epoch)
                SV.requires_grad_(True)
                vd_ = self.variable_val_dict.copy()
                for i, sv_name in enumerate(self.state_variables):
                    vd_[sv_name] = SV[:, i:i + 1]
                vd_["SV"] = SV
                self.update_variables(SV, vd=vd_)
                total = torch.zeros((SV.shape[0], 1), device=self.device)
                Bn = SV.shape[0]
                for label in self.hjb_equations:
                    res = torch.abs(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, vd_))
                    total = total + (res.reshape(Bn, -1).mean(dim=-1, keepdim=True) if res.dim() > 1 else res.reshape(Bn, 1))
                SVs.append(SV.detach().cpu()); losses.append(total.detach().cpu())
                del SV, total
            self.batch_size = saved_bs
            self.set_all_model_training()
            SVall = torch.cat(SVs, 0); lall = torch.cat(losses, 0)
            ids = torch.topk(lall, self.batch_size // self.refinement_rounds, dim=0)[1].squeeze(-1)
            if self.anchor_points is None:
                self.anchor_points = SVall[ids].to(self.device)
            else:
                self.anchor_points = torch.vstack((self.anchor_points, SVall[ids].to(self.device)))
        sv = self.sample_simplex_v_ts(epoch)
        if self.anchor_points is not None and len(self.anchor_points) > 0:
            return torch.vstack((sv, self.anchor_points))
        return sv

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
V_DOMAIN = (0.05, 0.95)


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
              min_inner=1000, loss_log_interval=50, max_t=1.0, init_guess=None):
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

    if timestepping:
        cfg = {"batch_size": batch_size, "time_batch_size": 1,
               "min_t": 0.0, "max_t": max_t,
               "sampling_method": SamplingMethod.UniformRandom,
               "num_outer_iterations": num_outer, "num_inner_iterations": num_inner,
               "min_inner_iterations": min_inner, "loss_log_interval": loss_log_interval,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10}
        model = PDEModelTimeStepNAgentsSV("sv_n_agents", cfg)
    else:
        cfg = {"batch_size": batch_size, "num_epochs": n_epochs,
               "sampling_method": SamplingMethod.UniformRandom,
               "optimizer_type": OptimizerType.Adam, "lr": lr,
               "loss_balancing": loss_balancing, "rar": rar, "refinement_rounds": 10}
        model = PDEModelNAgentsSV("sv_n_agents", cfg)

    state_names = [f"x_{i+1}" for i in range(K - 1)] + ["v"]
    domain = {f"x_{i+1}": [0.05, 0.95] for i in range(K - 1)}
    domain["v"] = list(V_DOMAIN)
    model.set_state(state_names, domain)

    model.statics = build_statics(K, expert_idx, household_idx, gamma_vec, caps_E,
                                  has_t=timestepping, params=params)

    net_cfg = {"hidden_units": model_size, "derivative_order": 0, "batch_jac_hes": False}
    for k in range(1, K + 1):
        model.add_agent(f"xi_{k}", config={**net_cfg, "positive": True})
    # p is NOT a network in this variant: it is solved analytically from goods-
    # market clearing (a quadratic in p; see StackedAgent.analytic_p), with its
    # state-derivatives obtained by autodiff through the xi networks.  Goods
    # clearing then holds by construction, so there is no "p" endog and no goods
    # loss.  (Caveat: this slaves dp/dt to dxi/dt, which previously destabilised
    # the backward time-stepping march -- this file is the experiment to test it.)
    # r is likewise solved analytically from the asset-pricing eq inside
    # compute_sv_equilibrium.  So only the xi (and theta) networks remain free.
    # only NON-anchor experts get a theta network; the anchor expert holds the
    # residual capital (theta_anchor = 1 - sum others) so clearing is exact.
    theta_names = [f"theta_{expert_idx[i]+1}" for i in range(1, len(expert_idx))]
    for nm in theta_names:
        model.add_endog(nm, config={**net_cfg, "positive": True})

    xi_names = [f"xi_{k}" for k in range(1, K + 1)]
    model.attach_stacks(xi_names, theta_names)

    # placeholder entries so equation registration / validation has shapes
    bsz = batch_size
    n_E = len(expert_idx)
    for nm, shp in [("capital_resid", (bsz, 1)), ("vi_expert_resid", (bsz, n_E)),
                    ("sig_clearing_resid", (bsz, 1)),
                    ("hjb_expert", (bsz, 1)), ("hjb_household", (bsz, 1)),
                    # extra quantities exposed only for variables_to_track (see
                    # the __check_outer_loop_converge override on the time-step
                    # model); r is analytic, chat/mu_P are intermediates.
                    ("chat", (bsz, K)), ("r", (bsz, 1)),
                    ("mu_P", (bsz, 1)), ("p", (bsz, 1)),
        ]:
        model.variable_val_dict[nm] = torch.zeros(shp, device=model.device)

    # ---- equilibrium-residual losses (spec section 6) ---------------------
    # goods clearing is satisfied by construction (p solved analytically) -> no loss
    # asset_pricing is satisfied by construction (r solved analytically) -> no loss
    # model.add_endog_equation("capital_resid = 0", label="capital")
    model.add_endog_equation("vi_expert_resid = 0", label="vi_expert")
    model.add_endog_equation("sig_clearing_resid = 0", label="sig_clearning")
    # hjb_expert/_household are ALREADY sum_k hjb_k**2, so MAE reduction gives
    # mean(sum_k hjb_k**2) == MSE of the raw residual (matching the original).
    # (MSE here would square again -> mean(hjb_k**4), which is wrong.)
    model.add_hjb_equation("hjb_expert", label="expert", loss_reduction=LossReductionMethod.MAE)
    model.add_hjb_equation("hjb_household", label="household", loss_reduction=LossReductionMethod.MAE)

    if train and not os.path.exists(f"{model_path}/model.pt"):
        os.makedirs(model_path, exist_ok=True)
        if timestepping and init_guess:
            # seed the backward march in the correct basin (default guess of 1
            # leaves p pinned ~1 and goods_resid stuck ~1; see get_model docstring)
            model.set_initial_guess(init_guess)
        model.train_model(model_path, "model.pt", full_log=True, variables_to_track=["chat", "r", "p", "mu_P"])
    if os.path.exists(f"{model_path}/model_best.pt"):
        model.load_model(torch.load(f"{model_path}/model_best.pt", weights_only=False))
        model.attach_stacks(xi_names, theta_names)
    return model


# ===========================================================================
# Cases
# ===========================================================================
def make_case(case: str):
    """Return (K, expert_idx, household_idx, gamma_vec, caps_E)."""
    if case == "agents2":
        # 1 expert + 1 household, original Di Tella gamma=5, no cap -> validation
        K = 2
        expert_idx = [0]; household_idx = [1]
        gamma_vec = [5.0, 5.0]
        caps_E = [1e6]
    if case == "agents5":
        # 4 experts + 1 household
        K = 5
        expert_idx = [0, 1, 2, 3]; household_idx = [4]
        gamma_vec = [3.0, 4.0, 5.0, 6.0, 12.0]
        caps_E = [1e6, 1e6, 1e6, 1e6]
    elif case == "agents20":
        K = 20
        n_E = 18
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        # experts more risk-tolerant (manage capital), households more averse
        gamma_vec = [3.0 +  i for i in range(n_E)] + [12.0 + i for i in range(K-n_E)]
        caps_E = [1e6] * n_E
    elif case == "agents50":
        K = 50
        expert_idx = list(range(25)); household_idx = list(range(25, 50))
        gamma_vec = [3.0 + 0.15 * i for i in range(25)] + [7.0 + 0.2 * i for i in range(25)]
        caps_E = [1e6] + [4.0 + 0.15 * i for i in range(24)]
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
            "goods_resid", "capital_resid", "vi_expert_resid", "sigtilde_full",
            "sign_k", "sigxi", "chi"]
    acc = {k: [] for k in keys}
    n = SV_np.shape[0]
    for c in range(0, n, chunk):
        SV = torch.tensor(SV_np[c:c + chunk], device=model.device, dtype=torch.get_default_dtype())
        SV.requires_grad_(True)
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
    return res


def _validation_states(model, n_samples=10000, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = model.statics["K"]
    eps = 0.05
    e = -torch.log(torch.rand((n_samples, K), generator=g) + 1e-30)
    shares = e / e.sum(dim=1, keepdim=True)
    shares = eps + (1.0 - K * eps) * shares               # sums to 1
    x_states = shares[:, :K - 1]
    vlo, vhi = V_DOMAIN
    v = vlo + (vhi - vlo) * torch.rand((n_samples, 1), generator=g)
    return torch.cat([x_states, v], dim=1).numpy()


def compute_validation_losses(model, SV_val):
    """Per-component validation losses + per-type HJB + total, on common states."""
    out = _forward_states(model, SV_val)
    res = {
        "HJB expert": float(np.mean(np.abs(out["hjb_expert"]))),
        "HJB household": float(np.mean(np.abs(out["hjb_household"]))),
        "goods": float(np.mean(out["goods_resid"] ** 2)),
        "capital": float(np.mean(out["capital_resid"] ** 2)),
        "vi_expert": float(np.mean(out["vi_expert_resid"] ** 2)),
    }
    res["Total"] = sum(res.values())
    return res


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


def compare_loss_table(models, baseline_key="basic", n_samples=10000, seed=0,
                       cols=("HJB expert", "HJB household", "Total")):
    SV_val = _validation_states(next(iter(models.values())), n_samples=n_samples, seed=seed)
    rows = {name: compute_validation_losses(m, SV_val) for name, m in models.items()}
    base = rows[baseline_key]
    for name, row in rows.items():
        for c in cols:
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    ordered = list(cols) + [f"{c} impr." for c in cols]
    return pd.DataFrame.from_dict(rows, orient="index")[ordered]


def compute_welfare_equivalent_losses(models, baseline_key="basic",
                                      n_samples=10000, seed=0):
    """Map each residual to a certainty-equivalent consumption-wealth (c/W)
    cost (units 1/time), averaged over a validation sample.

      HJB residual h_k  ->  rho * |h_k|                          (first order)
      capital FOC (vi)  ->  1/2 gamma_k (phi v)^2 (d theta_k/x_k)^2  (second order)
      goods clearing    ->  |goods_resid| / p                    (consumption units)
    """
    SV_val = _validation_states(next(iter(models.values())), n_samples=n_samples, seed=seed)
    rows = {}
    for name, model in models.items():
        st = model.statics
        rho = st["rho"]; phi = st["phi"]
        gamma = st["gamma"].detach().cpu().numpy().reshape(-1)
        e_idx = st["expert_idx"]; v_index = st["v_index"]
        out = _forward_states(model, SV_val)
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
        goods_we = float(np.mean(np.abs(out["goods_resid"][:, 0]) / (out["p"][:, 0] + 1e-8)))
        total = hjb_we + vi_we + goods_we
        rows[name] = {"HJB (c/W)": hjb_we, "Capital FOC (c/W)": vi_we,
                      "Goods (c/W)": goods_we, "total (c/W)": total}
    base = rows[baseline_key]
    for name, row in rows.items():
        for c in list(base.keys()):
            row[f"{c} impr."] = 0.0 if name == baseline_key else \
                100.0 * (base[c] - row[c]) / (abs(base[c]) + 1e-30)
    abs_cols = ["HJB (c/W)", "Capital FOC (c/W)", "Goods (c/W)", "total (c/W)"]
    ordered = abs_cols + [f"{c} impr." for c in abs_cols]
    return pd.DataFrame.from_dict(rows, orient="index")[ordered]


def df_to_latex(df, path):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(format_pct if "impr" in col else format_sci)
    with open(path, "w") as f:
        f.write(out.style.to_latex(hrules=True))


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
                    ax.plot(xfd, fd_dict[key], ls="-.", color=SLICE_COLORS[i],
                            marker="x", markevery=3, label=f"FD v={v}")
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
        fname = "model_global_min_loss.csv" if ts else "model_loss.csv"
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        df = df.iloc[1:] if len(df) > 1 else df
        xcol = "epoch" if "epoch" in df.columns else df.columns[0]
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


def plot_loss_weights(model_path, out_dir, file_name="loss_weight.pdf",
                      timestepping=False):
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


def plot_rar_anchors(model_path, K, out_dir, file_name="rar_anchors.pdf",
                     timestepping=False):
    """Scatter of RAR anchor points in (x_1, v).  Meaningful for K=2."""
    if K != 2:
        return
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
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(anchors[:, 0], anchors[:, 1], c=np.arange(len(anchors)),
                    cmap="viridis", s=12, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="anchor index (old -> new)")
    ax.set_xlabel("$x_1$ (expert share)"); ax.set_ylabel("$v$")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name), bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# The 8 training configurations
# ===========================================================================
# name -> (timestepping, rar, loss_balancing)
CONFIGS = {
    "basic":            (False, False, False),
    "basic_rar":        (False, True,  False),
    "basic_lb":         (False, False, True),
    "basic_rar_lb":     (False, True,  True),
    "timestep":         (True,  False, False),
    "timestep_rar":     (True,  True,  False),
    "timestep_lb":      (True,  False, True),
    "timestep_rar_lb":  (True,  True,  True),
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
        if best in models and best not in ordered:
            ordered.append(best)
    return {k: models[k] for k in ordered if k in models}


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["agents2", "agents20", "agents50"], default="agents2")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--outer", type=int, default=50, help="num_outer_iterations for time-stepping configs")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    if args.float64:
        torch.set_default_dtype(torch.float64)
        base_dir = f"./models/SV_NAgents_64bit_analytic_rp/{args.case}"
    else:
        torch.set_default_dtype(torch.float32)
        base_dir = f"./models/SV_NAgents_analytic_rp/{args.case}"

    K, eidx, hidx, gamma_vec, caps_E = make_case(args.case)
    print(f"[sv_n_agents] case={args.case} K={K} experts={eidx} households={hidx}")
    print(f"             gamma={gamma_vec}  caps_E={caps_E}")

    # Seed the time-stepping backward march in the correct basin (p >> 1).  From
    # the library default guess of 1 the march cannot escape the goods-market
    # plateau (p pinned ~1, goods_resid ~1).  Rough values suffice; the march
    # refines them to the true equilibrium.
    ts_init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    # ts_init_guess["p"] = 6.0
    # no "r" seed: r is solved analytically (not a network) in this variant.
    # ts_init_guess = None

    models, model_paths, ts_map = {}, {}, {}
    for name in list(CONFIGS.keys()):
        ts, rar, lb = CONFIGS[name]
        mpath = os.path.join(base_dir, name)
        print(f"\n{('=== ' + name + ' ==='):=^80}")
        model = get_model(
            mpath, K, eidx, hidx, gamma_vec, caps_E,
            model_size=[args.width] * args.layers,
            n_epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            timestepping=ts, rar=rar, loss_balancing=lb, num_outer=args.outer, num_inner=5000, min_inner=1000,
            init_guess=ts_init_guess,
        )
        models[name] = model
        model_paths[name] = mpath
        ts_map[name] = ts
        gc.collect(); torch.cuda.empty_cache()

    cmp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    # ---- 1) Loss tables (computed over ALL trained methods) -----------------
    loss_df = welfare_df = None
    if "basic" in models:
        loss_df = compare_loss_table(models, baseline_key="basic")
        loss_df.to_csv(os.path.join(cmp_dir, "comparative_losses.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, "comparative_losses.tex"))
        print("\n[sv_n_agents] comparative loss table (vs basic):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key="basic")
        welfare_df.to_csv(os.path.join(cmp_dir, "welfare_equivalent_losses.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, "welfare_equivalent_losses.tex"))
        print("\n[sv_n_agents] welfare-equivalent loss table (c/W, vs basic):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    # ---- pick basic + the best-improving method(s) for the overlay plots ----
    plot_models = select_plot_methods(models, loss_df=loss_df, welfare_df=welfare_df,
                                       baseline_key="basic")
    plot_paths = {k: model_paths[k] for k in plot_models}
    plot_ts = {k: ts_map[k] for k in plot_models}
    print(f"[sv_n_agents] plotting methods: {list(plot_models)}")

    # ---- 2) 2-D slice comparison vs the Di Tella FD solution (agents2 only) --
    if args.case == "agents2":
        try:
            from parse_ditella_sol import ditella_res_dict
            fd_dict = ditella_res_dict
        except Exception as e:
            print(f"[sv_n_agents] could not load FD solution: {e}")
            fd_dict = None
        v_list = [0.1, 0.25, 0.6]
        method_dicts = {name: evaluate_slices(m, v_list) for name, m in plot_models.items()}
        plot_slice_comparison(method_dicts, fd_dict, v_list, cmp_dir)
        print(f"[sv_n_agents] slice comparison plots saved to {cmp_dir}")

    # ---- 3) RAR anchor scatter (rar configs, K=2) ---------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if rar:
            plot_rar_anchors(model_paths[name], K, cmp_dir,
                             file_name=f"rar_anchors_{name}.pdf", timestepping=ts)

    # ---- 4) Loss-weight evolution (lb configs) ------------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if lb:
            plot_loss_weights(model_paths[name], cmp_dir,
                              file_name=f"loss_weight_{name}.pdf", timestepping=ts)

    # ---- 5) HJB / total loss convergence (basic + best method) --------------
    plot_loss_decay(plot_paths, cmp_dir, plot_ts)
    print(f"[sv_n_agents] loss-decay plot saved to {cmp_dir}")

    print(f"\n[sv_n_agents] all artifacts written under {cmp_dir}")

