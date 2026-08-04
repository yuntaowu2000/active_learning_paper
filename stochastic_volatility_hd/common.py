import contextlib
import gc
import math

import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
# Device used for evaluation/plotting forwards.  Models live on CPU between uses
# (see move_model / model_on) so that training/evaluating many configs does not
# pin every network's VRAM at once; each model is moved to EVAL_DEVICE only for
# the duration of its own forward pass.
EVAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

FD_TABLE_VARS = {
    "omega":        r"$\Omega=\xi/\zeta$",
    "e_hat":        r"$\hat{e}$",
    "c_hat":        r"$\hat{c}$",
    "risk_premium": r"$\pi(\sigma+\sigma_p)$",
}

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

# ===========================================================================
# Default economic parameters (original Di Tella calibration).
# ===========================================================================
BASE_PARAMS = {
    "a": 1.0, "sigma": 0.0125, "lbd": 1.38, "v_mean": 0.25, "sigv_mean": -0.17,
    "rho": 0.0665, "psi": 0.5, "tau": 1.15, "phi": 0.2,
    "A": 53.2, "B": -0.8668571428571438, "delta": 0.05,
}
V_DOMAIN = (0.05, 0.5)


def _module_of(agent):
    """The underlying nn.Module of an Agent / EndogVar wrapper."""
    return agent.model if hasattr(agent, "model") else agent


def move_model(model, dev):
    """Move all of a model's device-resident state to ``dev`` (in place).

    Covers the agent/endog networks, the static economic tensors (gamma),
    the cached variable_val_dict placeholders, and any leftover
    time-stepping training buffers, then updates ``model.device`` so subsequent
    forwards (which read it) land on the right device.  The library's stacked
    evaluator reads the live modules each call (and is invalidated on
    load_model), so no re-attachment is needed.
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


def mixture_shares_torch(n, K, eps, device, alpha_lo=SHARE_ALPHA_LO,
                          alpha_hi=SHARE_ALPHA_HI):
    """(n, K) eps-floored simplex shares; per-row log-uniform Dirichlet alpha.
    Uses the global torch RNG (so set_seeds controls it)."""
    u = torch.rand((n, 1), device=device)
    alpha = torch.exp(math.log(alpha_lo) + (math.log(alpha_hi) - math.log(alpha_lo)) * u)
    conc = alpha.expand(n, K).contiguous()
    shares = torch.distributions.Dirichlet(conc).sample()      # (n, K)
    return eps + (1.0 - K * eps) * shares


def mixture_shares_np(n, K, eps, rng, alpha_lo=SHARE_ALPHA_LO,
                       alpha_hi=SHARE_ALPHA_HI):
    """NumPy counterpart of _mixture_shares_torch for reproducible (seeded)
    evaluation/plotting sampling."""
    u = rng.random((n, 1))
    alpha = np.exp(np.log(alpha_lo) + (np.log(alpha_hi) - np.log(alpha_lo)) * u)
    gam = rng.gamma(np.broadcast_to(alpha, (n, K)), 1.0)        # (n, K)
    shares = gam / gam.sum(axis=1, keepdims=True)
    return eps + (1.0 - K * eps) * shares


def build_statics(K, expert_idx, household_idx, gamma_vec, has_t, params=BASE_PARAMS):
    n_E = len(expert_idx)
    D = K + 1 if has_t else K
    statics = {
        "K": K, "D": D, "n_E": n_E,
        "expert_idx": list(expert_idx), "household_idx": list(household_idx),
        "v_index": K - 1, "has_t": has_t,
        "gamma": torch.tensor(gamma_vec, device=device, dtype=torch.get_default_dtype()).reshape(1, K),
        "v_domain": V_DOMAIN,
    }
    for k, val in params.items():
        statics[k] = val
    return statics

# ===========================================================================
# Cases
# ===========================================================================
def make_case(case: str, gamma):
    """Return (K, expert_idx, household_idx, gamma_vec)."""
    if case == "agents2":
        # 1 expert + 1 household, original Di Tella gamma=5 -> validation
        K = 2
        expert_idx = [0]; household_idx = [1]
        gamma_vec = [gamma, gamma]
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
    elif case == "agents40":
        # 36 experts + 4 households.  Same HARD design as agents20/agents50:
        # wide expert spread [3, 10.5] (anchor index 0 = least averse, so capital
        # concentrates there and takes large leverage), households more averse
        # [12..16.5] holding little wealth.  gamma_vec length K (experts then hh).
        K = 40
        n_E = 36
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 7.5 * i / (n_E - 1) for i in range(n_E)] + [12.0, 13.5, 15.0, 16.5]
    elif case == "agents50":
        # 45 experts + 5 households.  Same HARD design as agents20: wide expert
        # spread [3, 11] (anchor = least averse -> capital concentrates there),
        # households [12..16].  gamma_vec has length K (experts then households).
        K = 50
        n_E = 45
        expert_idx = list(range(n_E)); household_idx = list(range(n_E, K))
        gamma_vec = [3.0 + 8.0 * i / (n_E - 1) for i in range(n_E)] + [12.0, 13.0, 14.0, 15.0, 16.0]
    else:
        raise ValueError(f"unknown case {case!r}")
    return K, expert_idx, household_idx, gamma_vec


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

def df_to_latex(df, path):
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(format_pct if "impr" in col else format_sci)
    with open(path, "w") as f:
        f.write(out.style.to_latex(hrules=True))