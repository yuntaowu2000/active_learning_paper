import torch

# ===========================================================================
# Economic core: one differentiable forward pass computing every equilibrium
# object from the raw network outputs.  See spec sections 2-6.
# ===========================================================================
def compute_sv_equilibrium(SV, xi, xi_Jac, xi_Hess, p, p_Jac, p_Hess, theta_E, r, statics):
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
    theta_E   : (B, n_E)       expert capital shares (analytic interior FOC)
    statics   : dict with K, D, n_E, expert_idx (LongTensor), household_idx,
                v_index, has_t, gamma (1,K), and scalar params.

    Returns a dict of named (B,*) tensors used by the registered losses & plots.
    """
    K = statics["K"]
    D = statics["D"]
    v_index = statics["v_index"]
    has_t = statics["has_t"]
    expert_idx = statics["expert_idx"]
    household_idx = statics["household_idx"]
    gamma = statics["gamma"]               # (1, K)

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

    # chi: common across experts under the interior FOC (computed from expert 0)
    chi = g_E[:, 0:1] * phiv2 * theta_E[:, 0:1] / x_E[:, 0:1]    # (B, 1)

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
        "xi_ret": xi_ret,
        "g": g, "iota": iota, "sigtilde_full": sigtilde_full,
        "risk_premium": risk_premium, 
    }
    for idx in expert_idx:
        out[f"hjb_expert_{idx}"] = hjb_k[:, idx:idx+1] ** 2
    for idx in household_idx:
        out[f"hjb_household_{idx}"] = hjb_k[:, idx:idx+1] ** 2
    return out
