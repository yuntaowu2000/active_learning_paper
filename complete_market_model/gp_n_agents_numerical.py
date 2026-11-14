"""
gp_n_agents_numerical.py
========================

Chebyshev-collocation benchmark solver for the **active-only** N-agent
Gârleanu–Panageas economy implemented in ``gp_n_agents_NN.py`` (see
``gp_n_agents_spec.md``).

Closure
-------
There is **no passive agent**.  The N active agents hold the entire economy and
their wealth shares **sum to one** (``sum_i x_i = 1``), so the state space is the
(N-1)-simplex.  Market clearing ``sum_i x_i alpha_i = 1`` then implies a zero
net-supply bond market; the aggregate consumption-wealth ratio is
``y = sum_i x_i xi_i`` over all N agents.

This file provides reference solutions for the **2-agent** validation cases.
With N=2 the state is the single free share ``x = x_1`` and the dependent share
is ``x_2 = 1 - x``.  We solve a 1-D Chebyshev BVP for the four fields

    H_1(x) = xi_1,   H_2(x) = xi_2,   alpha_1(x),   alpha_2(x)

against the *exact* equation system of the NN (closed-form y-derivatives of
spec section 2 and GP primitives of sections 3-5).  The single-shock
specialisation ``sigma = (sigma1, 0)`` collapses every vector to its first
component, so all objects below are scalars per node.

Agent 2 may carry a leverage cap ``alpha_2 <= cap``.  The cap is imposed as the
variational inequality ``min(cap - alpha_2, q/gamma_2 - varsigma_2 - alpha_2)=0``
smoothed by the Fischer-Burmeister function; ``cap = +inf`` recovers the plain
Merton FOC for agent 2.  A homotopy ramps gamma_1 from gamma_2 (the trivial
representative-agent start, alpha == 1) up to its target, then ramps the cap
down to its target, so ``scipy.optimize.root`` always starts near a root.

Notes on the symmetric cases
----------------------------
On ``sum_i x_i = 1`` with identical agents (gamma_1 = gamma_2) the equilibrium
is the representative agent: ``alpha_1 = alpha_2 = 1`` and constant prices.  A
leverage cap >= 1 therefore never binds, so ``sym2`` and ``sym2_const`` are both
trivial; the genuinely non-trivial cases are the heterogeneous ones
(``asym2``, ``asym2_const``).

Outputs (per case) under ``./models/gp_numerical_cheb/<case>/``:
    cheb_solution.csv      -- equilibrium objects vs x = x_1, columns:
                              x, H_1, H_2, alpha_1, alpha_2, y, pi, r, eta,
                              sigma_R, q  (consumed by ``gp_n_agents_NN.py``).
    cheb_equilibrium.png   -- diagnostic panel of r, pi, y, alpha_1, alpha_2.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root

EPS = 1e-12


# ---------------------------------------------------------------------------
# Chebyshev-Lobatto nodes + differentiation matrix (Trefethen cheb.m),
# affinely mapped to [a, b] and reordered ascending in x.
# ---------------------------------------------------------------------------
def cheb_diff_matrix(N: int, a: float = 0.0, b: float = 1.0):
    if N == 0:
        return np.array([a]), np.zeros((1, 1))
    k = np.arange(N + 1)
    xi = np.cos(np.pi * k / N)                  # nodes on [-1, 1], descending
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** k
    X = np.tile(xi, (N + 1, 1)).T
    dX = X - X.T + np.eye(N + 1)
    D = (c[:, None] / c[None, :]) / dX
    D -= np.diag(np.sum(D, axis=1))
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)      # map to [a, b]
    D = (2.0 / (b - a)) * D
    order = np.argsort(x)
    return x[order], D[order][:, order]


def _fischer_burmeister(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """phi(a, b) = 0  iff  a >= 0, b >= 0, a*b = 0."""
    return a + b - np.sqrt(a ** 2 + b ** 2 + 1e-30)


# ===========================================================================
# 1-D BVP for the 2-agent economy on  x_1 + x_2 = 1.
#
# State x = x_1 (free); x_2 = 1 - x (dependent).  All four fields H_1, H_2,
# alpha_1, alpha_2 are functions of x.  The only state diffusion is that of
# x_1:  sigma_x = x (alpha_1 - 1) sigma_R   (and sigma_{x_2} = -sigma_x).
#
#   y    = x H_1 + (1-x) H_2
#   y_x  = H_1 - H_2 + x H_1x + (1-x) H_2x
#   y_xx = 2 H_1x - 2 H_2x + x H_1xx + (1-x) H_2xx
#   A    = y_x * x (alpha_1 - 1)              (only the x_1 channel)
# ===========================================================================
def _residual_2agent(state, x, D, D2, g1, g2, p1, p2, sigma1, rho, mu, cap2):
    H1, H2, alpha1, alpha2 = np.split(state, 4)

    H1x = D @ H1;  H1xx = D2 @ H1
    H2x = D @ H2;  H2xx = D2 @ H2

    y   = x * H1 + (1.0 - x) * H2
    yx  = (H1 - H2) + x * H1x + (1.0 - x) * H2x
    yxx = 2.0 * (H1x - H2x) + x * H1xx + (1.0 - x) * H2xx

    A = yx * x * (alpha1 - 1.0)
    sigma_y = (A * sigma1) / (y + A + EPS)
    sigma_R = sigma1 - sigma_y
    sigma_R_sq = sigma_R ** 2
    sigma_R_norm = np.abs(sigma_R)

    sigma_x = x * (alpha1 - 1.0) * sigma_R
    sig_xi_1 = (H1x / (H1 + EPS)) * sigma_x
    sig_xi_2 = (H2x / (H2 + EPS)) * sigma_x

    varsigma1 = (1.0 - 1.0 / g1) / (1.0 - p1) * sig_xi_1 / (sigma_R + EPS)
    varsigma2 = (1.0 - 1.0 / g2) / (1.0 - p2) * sig_xi_2 / (sigma_R + EPS)

    q = g1 * (alpha1 + varsigma1)                # FOC of agent 1 defines q
    pi = q * sigma_R_sq
    eta = q * sigma_R_norm
    foc2 = q / g2 - varsigma2                     # Merton target for agent 2

    # drift of the free state x_1
    mu_x = x * (y - H1 + (1.0 - alpha1) * (1.0 - q) * sigma_R_sq)

    mu_xi_1 = H1x * mu_x / (H1 + EPS) + 0.5 * H1xx * sigma_x ** 2 / (H1 + EPS)
    mu_xi_2 = H2x * mu_x / (H2 + EPS) + 0.5 * H2xx * sigma_x ** 2 / (H2 + EPS)
    mu_y = yx * mu_x / (y + EPS) + 0.5 * yxx * sigma_x ** 2 / (y + EPS)
    mu_P = mu - mu_y + sigma_y * (sigma_y - sigma1)
    r = y + mu_P - pi

    def hjb(H_, g_, p_, a_, mu_xi_, sig_xi_):
        return (
            rho * p_
            + (1.0 - p_) * (r + eta * a_ * sigma_R_norm
                            - 0.5 * g_ * (a_ * sigma_R_norm) ** 2)
            + mu_xi_
            + (1.0 - g_) * sig_xi_ * sigma_R * a_
            + 0.5 * (p_ - g_) / (1.0 - p_) * sig_xi_ ** 2
            - H_
        ) / rho

    res_hjb1 = hjb(H1, g1, p1, alpha1, mu_xi_1, sig_xi_1)
    res_hjb2 = hjb(H2, g2, p2, alpha2, mu_xi_2, sig_xi_2)
    res_mc = x * alpha1 + (1.0 - x) * alpha2 - 1.0
    if np.isfinite(cap2):
        res_vi = _fischer_burmeister(cap2 - alpha2, foc2 - alpha2)
    else:
        res_vi = foc2 - alpha2                    # plain Merton FOC for agent 2
    return np.concatenate([res_hjb1, res_hjb2, res_mc, res_vi])


def _try_root(state0, x, D, D2, g1, g2, p1, p2, sigma1, rho, mu, cap2,
              methods=("lm", "hybr")):
    last = None
    for method in methods:
        sol = root(_residual_2agent, state0,
                   args=(x, D, D2, g1, g2, p1, p2, sigma1, rho, mu, cap2),
                   method=method, tol=1e-11,
                   options={"xtol": 1e-13, "maxiter": 5000} if method != "hybr"
                       else {"xtol": 1e-13, "maxfev": 1_000_000})
        last = sol
        state0 = sol.x
        if sol.success and np.max(np.abs(sol.fun)) < 1e-7:
            return sol
    return last


def solve_2agent_1d(params: dict, cap2: float = np.inf, N: int = 80,
                    x_lo: float = 0.02, x_hi: float = 0.98,
                    verbose: bool = True):
    """Solve the 2-agent economy on x_1 + x_2 = 1 over x = x_1 in [x_lo, x_hi].

    A homotopy starts from the representative-agent solution (gamma_1 = gamma_2,
    alpha == 1, cap = +inf), ramps gamma_1 to its target, then ramps the cap
    down to ``cap2``.  Returns ``(x, H1, H2, alpha1, alpha2, sol)``.
    """
    g1, g2 = params["gamma"]
    p1, p2 = params["psi"]
    sigma1 = params["sigma"]
    rho = params["rho"]
    mu = params["mu"]

    x, D = cheb_diff_matrix(N, a=x_lo, b=x_hi)
    D2 = D @ D

    # representative-agent seed: alpha == 1, H == rho (consumption-wealth ~ rho)
    H0 = np.full_like(x, rho)
    a0 = np.ones_like(x)
    state0 = np.concatenate([H0, H0, a0, a0])

    # cap used during the gamma ramp: exact FOC (inf) when unconstrained, else a
    # large finite value that the step-2 cap homotopy continues down from.
    ramp_cap = np.inf if not np.isfinite(cap2) else 1e6

    # step 0: symmetric, unconstrained (g1 := g2)
    sol = _try_root(state0, x, D, D2, g2, g2, p2, p2, sigma1, rho, mu, ramp_cap)
    state0 = sol.x
    if verbose:
        print(f"[2agent] sym start g={g2}  |res|_inf={np.max(np.abs(sol.fun)):.2e}  "
              f"success={sol.success}")

    # step 1: ramp gamma_1 (and psi_1) from g2 to target
    if not np.isclose(g1, g2) or not np.isclose(p1, p2):
        for t in np.linspace(0.0, 1.0, 8)[1:]:
            g1_t = g2 + t * (g1 - g2)
            p1_t = p2 + t * (p1 - p2)
            sol = _try_root(state0, x, D, D2, g1_t, g2, p1_t, p2, sigma1, rho, mu, ramp_cap)
            state0 = sol.x
            if verbose:
                print(f"[2agent] gamma_1={g1_t:.3g}  "
                      f"|res|_inf={np.max(np.abs(sol.fun)):.2e}  success={sol.success}")

    # step 2: ramp the cap down to target
    if np.isfinite(cap2):
        for cap_step in np.geomspace(1e6, cap2, 10):
            sol = _try_root(state0, x, D, D2, g1, g2, p1, p2, sigma1, rho, mu, cap_step)
            state0 = sol.x
            if verbose:
                print(f"[2agent] cap={cap_step:.3g}  "
                      f"|res|_inf={np.max(np.abs(sol.fun)):.2e}  success={sol.success}")

    if verbose:
        print(f"[2agent] final: success={sol.success}  "
              f"|res|_inf={np.max(np.abs(sol.fun)):.2e}")
    H1, H2, alpha1, alpha2 = np.split(sol.x, 4)
    return x, H1, H2, alpha1, alpha2, sol


def post_process(x, H1, H2, alpha1, alpha2, params):
    """Reconstruct equilibrium objects and return them in the NN-comparison
    CSV format (one row per node, x = x_1)."""
    _, D = cheb_diff_matrix(len(x) - 1, a=x[0], b=x[-1])
    D2 = D @ D
    g1, g2 = params["gamma"]
    p1, p2 = params["psi"]
    sigma1 = params["sigma"]
    rho = params["rho"]
    mu = params["mu"]

    H1x = D @ H1;  H1xx = D2 @ H1
    H2x = D @ H2;  H2xx = D2 @ H2

    y   = x * H1 + (1.0 - x) * H2
    yx  = (H1 - H2) + x * H1x + (1.0 - x) * H2x
    yxx = 2.0 * (H1x - H2x) + x * H1xx + (1.0 - x) * H2xx
    A = yx * x * (alpha1 - 1.0)
    sigma_y = (A * sigma1) / (y + A + EPS)
    sigma_R = sigma1 - sigma_y
    sigma_R_sq = sigma_R ** 2
    sigma_R_norm = np.abs(sigma_R)
    sigma_x = x * (alpha1 - 1.0) * sigma_R
    sig_xi_1 = (H1x / (H1 + EPS)) * sigma_x
    varsigma1 = (1.0 - 1.0 / g1) / (1.0 - p1) * sig_xi_1 / (sigma_R + EPS)
    q = g1 * (alpha1 + varsigma1)
    pi = q * sigma_R_sq
    eta = q * sigma_R_norm
    mu_x = x * (y - H1 + (1.0 - alpha1) * (1.0 - q) * sigma_R_sq)
    mu_y = yx * mu_x / (y + EPS) + 0.5 * yxx * sigma_x ** 2 / (y + EPS)
    mu_P = mu - mu_y + sigma_y * (sigma_y - sigma1)
    r = y + mu_P - pi
    return pd.DataFrame({
        "x": x, "H_1": H1, "H_2": H2,
        "alpha_1": alpha1, "alpha_2": alpha2,
        "y": y, "pi": pi, "r": r, "eta": eta,
        "sigma_R": sigma_R, "q": q,
        # drift / diffusion of the free wealth share x_1 -- needed to forward
        # simulate the stationary distribution (see gp_n_agents_simulate.py).
        # ``pi`` above is the (aggregate) risk premium.
        "mux": mu_x, "sigma_x": sigma_x,
    })


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_solution(df: pd.DataFrame, output_dir: str, cap: float = np.inf):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))

    ax[0].plot(df["x"], df["r"], color="C0", lw=2)
    ax[0].set_xlabel("$x_1$"); ax[0].set_ylabel("$r$"); ax[0].grid(alpha=.3)

    ax[1].plot(df["x"], df["pi"], color="C0", lw=2)
    ax[1].set_xlabel("$x_1$"); ax[1].set_ylabel("$\\pi$"); ax[1].grid(alpha=.3)

    ax[2].plot(df["x"], df["y"], color="C0", lw=2)
    ax[2].set_xlabel("$x_1$"); ax[2].set_ylabel("$y$"); ax[2].grid(alpha=.3)

    ax[3].plot(df["x"], df["alpha_1"], color="C0", lw=2, label="$\\alpha_1$")
    ax[3].plot(df["x"], df["alpha_2"], color="C1", lw=2, ls="--", label="$\\alpha_2$")
    if np.isfinite(cap):
        ax[3].axhline(cap, color="gray", lw=0.8, ls=":", label=f"cap = {cap}")
    ax[3].set_xlabel("$x_1$"); ax[3].set_ylabel("$\\alpha$")
    ax[3].grid(alpha=.3); ax[3].legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "cheb_equilibrium.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def make_params(gamma=(4.0, 4.0), psi=(1.5, 1.5), rho=0.05, sigma=0.0357,
                mu=0.0183, kappa=0.0):
    return {
        "gamma": np.asarray(gamma, dtype=float),
        "psi": np.asarray(psi, dtype=float),
        "rho": rho, "sigma": sigma, "mu": mu, "kappa": kappa,
    }


# Case table matches the validation cases in gp_n_agents_NN.py __main__.
_CASES = {
    "sym2":        dict(gamma=(4.0, 4.0), cap=np.inf),
    "asym2":       dict(gamma=(8.0, 4.0), cap=np.inf),
    "sym2_const":  dict(gamma=(4.0, 4.0), cap=1.8),
    "asym2_const": dict(gamma=(8.0, 4.0), cap=1.8),
    "asym2_const2": dict(gamma=(8.0, 4.0), cap=1.3),
}


def run_case(case: str, out_base: str = "./models/gp_numerical_cheb", N: int = 80):
    if case not in _CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {list(_CASES)}")
    spec = _CASES[case]
    params = make_params(gamma=spec["gamma"], psi=(1.5, 1.5))
    cap = spec["cap"]
    out_dir = os.path.join(out_base, case)
    os.makedirs(out_dir, exist_ok=True)

    x, H1, H2, a1, a2, sol = solve_2agent_1d(params, cap2=cap, N=N,
                                             x_lo=0.02, x_hi=0.98)
    df = post_process(x, H1, H2, a1, a2, params)

    csv_path = os.path.join(out_dir, "cheb_solution.csv")
    df.to_csv(csv_path, index=False)
    print(f"[gp_n_agents_numerical] {case}: solution saved to {csv_path}")

    png_path = plot_solution(df, out_dir, cap=cap)
    print(f"[gp_n_agents_numerical] {case}: plot saved to {png_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=80)
    args = parser.parse_args()

    cases = ["asym2_const2"]
    for c in cases:
        run_case(c, N=args.N)
