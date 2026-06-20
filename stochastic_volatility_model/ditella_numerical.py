"""Finite-difference (false-transient RK4) numerical solution of the 2-agent
Di Tella (2017) stochastic-volatility model.

This is a faithful Python re-implementation of the Mathematica notebook
``CE.nb`` whose output is stored in ``ditella_sol``.  It is meant as a
*reference / validation* solver: the converged grid solution should match
``ditella_sol`` (and hence the neural-network solution evaluated in
``sv_n_agents_NN.evaluate_slices``) up to discretisation error.

Method (mirrors CE.nb)
----------------------
State variables (x, v); unknown fields are the two value-function levels
xi (experts) and zeta (households) plus the capital price p.

* Grids are Chebyshev-Gauss-Lobatto (CGL), clustered at both ends, exactly the
  grids dumped in ``parse_ditella_sol.py`` (v in [0.05, 2.0], x in [0.05, 0.95],
  31x31 by default).
* Spatial derivatives use finite-difference weights (Fornberg) on the
  non-uniform grid, matching ``NDSolve`FiniteDifferenceDerivative``.
* p is slaved to goods-market clearing (``MC = 0``), a quadratic in p with the
  *same* analytic root used in the NN ``analytic_p``:
      p = sqrt(4 A^2 C^2 + 4 A a + B^2) - 2 A C,   C = e_hat*x + c_hat*(1-x).
* r is solved from the household HJB (``HJBc = 0``).
* The three evolution residuals {expert HJB, asset-pricing FOC, differentiated
  market clearing DMC} are solved for (dpdt, dxidt, dzetadt).  Because r and all
  three time-derivatives enter *linearly*, the per-grid-point closure reduces to
  one scalar solve for r and explicit back-substitution.
* Pseudo-time integration is RK4 with ``field -= (1/6)(k1+2k2+2k3+k4)``,
  ``k = h * delta`` (the exact CE.nb update convention).
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters (identical to di_tella.md / CE.nb)
# ---------------------------------------------------------------------------
DITELLA_PARAMS = dict(
    a=1.0,
    sigma=0.0125,
    lbd=1.38,        # lambda: mean reversion of v        (CE.nb L)
    v_mean=0.25,     # long-run mean of v                  (CE.nb LRv)
    sigv_mean=-0.17, # idiosyncratic vol loading on v      (CE.nb -S)
    rho=0.0665,
    gamma=5.0,
    psi=0.5,
    tau=1.15,
    phi=0.2,
    A=53.2,
    B=-0.8668571428571438,
    delta=0.05,
)


# ---------------------------------------------------------------------------
# Grids and finite-difference operators
# ---------------------------------------------------------------------------
def cgl_grid(x0: float, x1: float, n_points: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto grid on [x0, x1] with ``n_points`` nodes.

    Matches CE.nb ``CGLGrid[x0, L, n] = x0 + L/2 (1 - Cos[pi k/(n-1)])``.
    """
    k = np.arange(n_points)
    return x0 + 0.5 * (x1 - x0) * (1.0 - np.cos(np.pi * k / (n_points - 1)))


def _fornberg_weights(z: float, nodes: np.ndarray, m: int) -> np.ndarray:
    """Fornberg (1988) finite-difference weights.

    Returns array ``c`` of shape ``(m + 1, len(nodes))`` such that the k-th
    derivative at ``z`` is ``sum_j c[k, j] * f(nodes[j])`` (2 <= len(nodes)).
    """
    n = len(nodes)
    c = np.zeros((m + 1, n))
    c1 = 1.0
    c4 = nodes[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = nodes[i] - z
        for j in range(i):
            c3 = nodes[i] - nodes[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def build_diff_matrices(grid: np.ndarray, halfwidth: int = 2):
    """Dense 1st/2nd-derivative matrices on a (possibly non-uniform) grid.

    For each node a symmetric stencil of up to ``2*halfwidth + 1`` nearest nodes
    is used, shifted to stay in range near the boundaries (one-sided).
    """
    n = len(grid)
    D1 = np.zeros((n, n))
    D2 = np.zeros((n, n))
    for i in range(n):
        lo = max(0, i - halfwidth)
        hi = min(n, i + halfwidth + 1)
        # keep a full-width stencil near the boundaries
        if hi - lo < 2 * halfwidth + 1:
            if lo == 0:
                hi = min(n, lo + 2 * halfwidth + 1)
            else:
                lo = max(0, hi - (2 * halfwidth + 1))
        idx = np.arange(lo, hi)
        w = _fornberg_weights(grid[i], grid[idx], 2)
        D1[i, idx] = w[1]
        D2[i, idx] = w[2]
    return D1, D2


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class DiTellaNumerical:
    """False-transient finite-difference solver for the 2-agent Di Tella model."""

    def __init__(self, params: dict = None, Nx: int = 30, Nv: int = 30,
                 minx: float = 0.05, maxx: float = 0.95,
                 minv: float = 0.05, maxv: float = 2.0,
                 halfwidth: int = 2):
        self.p = dict(DITELLA_PARAMS)
        if params:
            self.p.update(params)
        self.gridx = cgl_grid(minx, maxx, Nx + 1)
        self.gridv = cgl_grid(minv, maxv, Nv + 1)
        # 2D state fields shaped (Nv+1, Nx+1): rows index v, cols index x.
        self.X = self.gridx[None, :] * np.ones((Nv + 1, 1))
        self.V = self.gridv[:, None] * np.ones((1, Nx + 1))
        self.D1v, self.D2v = build_diff_matrices(self.gridv, halfwidth)
        self.D1x, self.D2x = build_diff_matrices(self.gridx, halfwidth)
        self.xi = None
        self.zeta = None
        self.price = None

    # --- spatial derivative helpers (axis 0 = v, axis 1 = x) ----------------
    def _dv(self, F):
        return self.D1v @ F

    def _dx(self, F):
        return F @ self.D1x.T

    def _dvv(self, F):
        return self.D2v @ F

    def _dxx(self, F):
        return F @ self.D2x.T

    def _dvx(self, F):
        return self.D1v @ F @ self.D1x.T

    # --- analytic goods-clearing price --------------------------------------
    def _capital_index(self, xi, zeta):
        """C = e_hat * x + c_hat * (1-x) (consumption-to-wealth aggregate)."""
        P = self.p
        ae = (P["psi"] - 1.0) / P["psi"]
        coef = P["rho"] ** (1.0 / P["psi"])
        e_hat = coef * xi ** ae
        c_hat = coef * zeta ** ae
        C = e_hat * self.X + c_hat * (1.0 - self.X)
        return C, e_hat, c_hat

    def _price_from_clearing(self, xi, zeta):
        P = self.p
        C, _, _ = self._capital_index(xi, zeta)
        disc = 4.0 * P["A"] ** 2 * C ** 2 + 4.0 * P["A"] * P["a"] + P["B"] ** 2
        return np.sqrt(disc) - 2.0 * P["A"] * C

    def _dprice_dxi_dzeta(self, xi, zeta):
        """Partials of the clearing price w.r.t. xi and zeta (for the DMC term)."""
        P = self.p
        ae = (P["psi"] - 1.0) / P["psi"]
        coef = P["rho"] ** (1.0 / P["psi"])
        C, _, _ = self._capital_index(xi, zeta)
        disc = 4.0 * P["A"] ** 2 * C ** 2 + 4.0 * P["A"] * P["a"] + P["B"] ** 2
        dp_dC = 4.0 * P["A"] ** 2 * C / np.sqrt(disc) - 2.0 * P["A"]
        dC_dxi = coef * ae * xi ** (ae - 1.0) * self.X
        dC_dzeta = coef * ae * zeta ** (ae - 1.0) * (1.0 - self.X)
        return dp_dC * dC_dxi, dp_dC * dC_dzeta

    # --- core equilibrium closure -------------------------------------------
    def equilibrium(self, price, xi, zeta):
        """Return a dict of all equilibrium objects (incl. r and the pseudo-time
        derivatives dpdt, dxidt, dzetadt) for the given fields.

        All time-derivatives and r enter the static equations linearly, so the
        closure is: solve {HJBc=0} for r, {HJB,FOC,DMC} for the time-derivatives.
        """
        P = self.p
        x, v = self.X, self.V
        gamma, psi, rho = P["gamma"], P["psi"], P["rho"]
        sigma, tau, phi = P["sigma"], P["tau"], P["phi"]
        A, B, delta, a = P["A"], P["B"], P["delta"], P["a"]
        ae = (psi - 1.0) / psi
        coef = rho ** (1.0 / psi)
        igam = (1.0 - gamma) / gamma

        # exogenous v dynamics
        sigv = P["sigv_mean"] * np.sqrt(v)
        muv = P["lbd"] * (P["v_mean"] - v)

        # production / investment
        g = (price - B - 2.0 * A * delta) / (2.0 * A)
        iota = A * (g + delta) ** 2 + B * (g + delta)
        e_hat = coef * xi ** ae
        c_hat = coef * zeta ** ae

        # spatial derivatives
        dpdv, dpdx = self._dv(price), self._dx(price)
        dpdvv, dpdvx, dpdxx = self._dvv(price), self._dvx(price), self._dxx(price)
        dxidv, dxidx = self._dv(xi), self._dx(xi)
        dxidvv, dxidvx, dxidxx = self._dvv(xi), self._dvx(xi), self._dxx(xi)
        dzdv, dzdx = self._dv(zeta), self._dx(zeta)
        dzdvv, dzdvx, dzdxx = self._dvv(zeta), self._dvx(zeta), self._dxx(zeta)

        # endogenous risk of the wealth share x
        num = (1.0 - x) * x * igam * (dxidv / xi - dzdv / zeta)
        den = 1.0 - (1.0 - x) * x * igam * (dxidx / xi - dzdx / zeta)
        sigx = num / den * sigv

        # diffusions
        sigp = (dpdv / price) * sigv + (dpdx / price) * sigx
        sigxi = (dxidv / xi) * sigv + (dxidx / xi) * sigx
        sigzeta = (dzdv / zeta) * sigv + (dzdx / zeta) * sigx
        sign = sigma + sigp + sigx / x
        pi = gamma * sign - (1.0 - gamma) * sigxi
        sigw = pi / gamma + igam * sigzeta
        sig_agg = sigma + sigp
        phiv2 = (phi * v) ** 2

        # drift of x (r cancels: mu_n - r = gamma/x^2 phiv2 + pi*sign)
        mux = x * (
            (gamma / x ** 2) * phiv2 + pi * sign
            - e_hat - tau + (a - iota) / price
            - pi * sig_agg - gamma / x * phiv2
            + sig_agg ** 2 - sign * sig_agg
        )

        # "known" (time-derivative-free) parts of the generators
        def generator_known(F, dFv, dFx, dFvv, dFvx, dFxx):
            return (dFv / F) * muv + (dFx / F) * mux + (1.0 / (2.0 * F)) * (
                dFvv * sigv ** 2 + 2.0 * dFvx * sigv * sigx + dFxx * sigx ** 2)

        L_p = generator_known(price, dpdv, dpdx, dpdvv, dpdvx, dpdxx)
        L_xi = generator_known(xi, dxidv, dxidx, dxidvv, dxidvx, dxidxx)
        L_zeta = generator_known(zeta, dzdv, dzdx, dzdvv, dzdvx, dzdxx)

        # ---- linear closure constants -------------------------------------
        # HJBc = 0  =>  r + dzetadt/zeta = C_HJBc
        C_HJBc = (
            rho / (1.0 - psi)
            - (psi / (1.0 - psi)) * coef * zeta ** ae
            - pi * sigw - L_zeta
            + (gamma / 2.0) * (sigw ** 2 + sigzeta ** 2 - 2.0 * igam * sigw * sigzeta)
        )
        # expert HJB = 0  =>  r + dxidt/xi = C_HJB
        C_HJB = (
            rho / (1.0 - psi)
            + tau / (1.0 - gamma)
            - (tau / (1.0 - gamma)) * (zeta / xi) ** (1.0 - gamma)
            - (psi / (1.0 - psi)) * coef * xi ** ae
            - (gamma / x ** 2) * phiv2 - pi * sign - L_xi
            + (gamma / 2.0) * (sign ** 2 + sigxi ** 2 - 2.0 * igam * sign * sigxi
                               + phiv2 / x ** 2)
        )
        # FOC = 0  =>  dpdt - p*r = C_FOC
        C_FOC = -(price * L_p + price * g + price * sigma * sigp + a - iota
                  - price * pi * sig_agg - price * (gamma / x) * phiv2)

        # DMC: dpdt = pMC_xi*dxidt + pMC_zeta*dzetadt
        pMC_xi, pMC_zeta = self._dprice_dxi_dzeta(xi, zeta)

        # Solve the 4x4 linear system (r, dpdt, dxidt, dzetadt) in closed form:
        #   dzetadt = zeta*(C_HJBc - r),  dxidt = xi*(C_HJB - r),  dpdt = C_FOC + p*r
        # substitute into DMC:
        denom = price + pMC_xi * xi + pMC_zeta * zeta
        rhs = pMC_xi * xi * C_HJB + pMC_zeta * zeta * C_HJBc - C_FOC
        r = rhs / denom

        dxidt = xi * (C_HJB - r)
        dzetadt = zeta * (C_HJBc - r)
        dpdt = C_FOC + price * r

        return dict(
            price=price, xi=xi, zeta=zeta, r=r,
            dpdt=dpdt, dxidt=dxidt, dzetadt=dzetadt,
            sigx=sigx, sigp=sigp, sig_agg=sig_agg, pi=pi,
            sigxi=sigxi, sigzeta=sigzeta, sign=sign, sigw=sigw,
            e_hat=e_hat, c_hat=c_hat, mux=mux, muv=muv,
        )

    def _deltas(self, price, xi, zeta):
        eq = self.equilibrium(price, xi, zeta)
        return eq["dpdt"], eq["dxidt"], eq["dzetadt"]

    # --- pseudo-time integration --------------------------------------------
    def solve(self, h: float = 2e-4, max_iters: int = 300_000,
              tol: float = 1e-7, report_every: int = 5_000,
              xi0: float = 0.015, zeta0: float = 0.01, verbose: bool = True,
              loss_csv: str = None):
        """Run the RK4 false transient until the relative change rate drops
        below ``tol`` (or ``max_iters`` is reached).

        The "loss" of the false transient is the (relative) pseudo-time
        change rate ``max|dfield/dt| / field`` -- it -> 0 as the HJB / FOC /
        market-clearing residuals are driven to zero.  The per-field components
        are logged to ``self.loss_history`` (every ``report_every`` iterations,
        plus the first and final step) and, if ``loss_csv`` is given, written to
        that path via :meth:`save_loss_csv`.
        """
        shape = self.X.shape
        xi = np.full(shape, xi0)
        zeta = np.full(shape, zeta0)
        price = self._price_from_clearing(xi, zeta)

        self.loss_history = []
        floor = 1e-8
        n = 0
        for n in range(max_iters):
            k1p, k1xi, k1z = self._deltas(price, xi, zeta)
            k1p, k1xi, k1z = h * k1p, h * k1xi, h * k1z

            k2p, k2xi, k2z = self._deltas(price - k1p / 2, xi - k1xi / 2, zeta - k1z / 2)
            k2p, k2xi, k2z = h * k2p, h * k2xi, h * k2z

            k3p, k3xi, k3z = self._deltas(price - k2p / 2, xi - k2xi / 2, zeta - k2z / 2)
            k3p, k3xi, k3z = h * k3p, h * k3xi, h * k3z

            k4p, k4xi, k4z = self._deltas(price - k3p, xi - k3xi, zeta - k3z)
            k4p, k4xi, k4z = h * k4p, h * k4xi, h * k4z

            dp = (k1p + 2 * k2p + 2 * k3p + k4p) / 6.0
            dxi = (k1xi + 2 * k2xi + 2 * k3xi + k4xi) / 6.0
            dz = (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0

            price = np.maximum(price - dp, floor)
            xi = np.maximum(xi - dxi, floor)
            zeta = np.maximum(zeta - dz, floor)

            rate_p = float(np.max(np.abs(dp / price))) / h
            rate_xi = float(np.max(np.abs(dxi / xi))) / h
            rate_zeta = float(np.max(np.abs(dz / zeta))) / h
            rate = max(rate_p, rate_xi, rate_zeta)
            if not np.isfinite(rate):
                raise FloatingPointError(
                    f"diverged at iter {n}; reduce h (current h={h}).")
            converged = rate < tol
            if n == 0 or n % report_every == 0 or converged:
                self.loss_history.append(
                    {"iter": n, "loss": rate, "dpdt_over_p": rate_p,
                     "dxidt_over_xi": rate_xi, "dzetadt_over_zeta": rate_zeta})
                if verbose:
                    print(f"iter {n:>7d}  max|dfield/dt|/field = {rate:.3e}")
            if converged:
                if verbose:
                    print(f"converged at iter {n}  rate={rate:.3e}")
                break

        self.xi, self.zeta, self.price = xi, zeta, price
        if loss_csv is not None:
            self.save_loss_csv(loss_csv)
        return self

    def save_loss_csv(self, path: str):
        """Write the false-transient loss history to ``path`` (one row per
        logged iteration; columns: iter, loss, dpdt_over_p, dxidt_over_xi,
        dzetadt_over_zeta)."""
        if not getattr(self, "loss_history", None):
            raise RuntimeError("no loss history; call solve() first")
        df = pd.DataFrame(self.loss_history)
        df.to_csv(path, index=False)

    # --- output -------------------------------------------------------------
    def grid_solution(self) -> dict:
        """All output variables on the full (v, x) grid."""
        if self.xi is None:
            raise RuntimeError("call solve() first")
        eq = self.equilibrium(self.price, self.xi, self.zeta)
        return dict(
            p=eq["price"], sigx=eq["sigx"], sigsigp=eq["sig_agg"],
            signxi=eq["pi"], r=eq["r"], omega=self.xi / self.zeta,
            e_hat=eq["e_hat"], c_hat=eq["c_hat"],
            risk_premium=eq["sig_agg"] * eq["pi"],
        )

    def _interp(self, field):
        return RegularGridInterpolator((self.gridv, self.gridx), field,
                                       bounds_error=False, fill_value=None)

    def evaluate_slices(self, v_list, n: int = 100,
                        x_lo: float = 0.05, x_hi: float = 0.95) -> dict:
        """Mirror ``sv_n_agents_NN.evaluate_slices``: for each v in ``v_list``,
        evaluate the model along x in [x_lo, x_hi] and return a dict keyed
        identically (``p_{v}``, ``sigx_{v}``, ``sigsigp_{v}``, ``signxi_{v}``,
        ``r_{v}``, ``omega_{v}``, ``e_hat_{v}``, ``c_hat_{v}``,
        ``risk_premium_{v}``)."""
        grids = self.grid_solution()
        interps = {k: self._interp(val) for k, val in grids.items()}
        x_plot = np.linspace(x_lo, x_hi, n)
        res = {"x_plot": x_plot}
        for v in v_list:
            pts = np.column_stack([np.full(n, v), x_plot])
            for name in ("p", "sigx", "sigsigp", "signxi", "r",
                         "omega", "e_hat", "c_hat", "risk_premium"):
                res[f"{name}_{v}"] = interps[name](pts).reshape(-1)
        return res


def solve_ditella(params: dict = None, **kwargs) -> DiTellaNumerical:
    """Convenience: build and solve a Di Tella model, returning the solver."""
    solver_keys = {"Nx", "Nv", "minx", "maxx", "minv", "maxv", "halfwidth"}
    init_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in solver_keys}
    solver = DiTellaNumerical(params=params, **init_kw)
    solver.solve(**kwargs)
    return solver

SLICE_COLORS = ["red", "orange", "blue"]

def plot_slices(out_dict, out_fn, v_list: list[str]=[0.1, 0.25, 0.6]):
    fig, ax = plt.subplots(1, 5, figsize=(32, 6))
    ax = ax.flatten()
    for i, var in enumerate(["p", "r", "omega", "sigx", "risk_premium"]):
        for j, v in enumerate(v_list):
            key = f"{var}_{v}"
            ax[i].plot(out_dict["x_plot"], out_dict[key], ls="-.", color=SLICE_COLORS[j], marker="x", markevery=3, label=f"FD v={v}")
        ax[i].set_xlabel("x", fontsize=14)
        ax[i].set_ylabel(var, fontsize=14)
        ax[i].legend(fontsize=16, frameon=False)
        ax[i].tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(out_fn)
    plt.close()

if __name__ == "__main__":
    base_dir = "models/numerical"
    os.makedirs(base_dir, exist_ok=True)
    for a, sigma, tau, gamma in [(0.2, 0.06, 1.15, 6.0)]:
        params = {"a": a, "sigma": sigma, "gamma": gamma, "tau": tau}
        print(f"solving {a, sigma, gamma, tau}")
        model = solve_ditella(params=params, h=2e-4, max_iters=300_000, tol=1e-7, loss_csv=f"{base_dir}/ditella_numerical_loss_{gamma}_{tau}_{sigma}_{a}.csv")
        out = model.evaluate_slices([0.1, 0.25, 0.6])
        x = out["x_plot"]
        print("\n x      p(v=.25)   r(v=.25)   pi(v=.25)   omega(v=.25)")
        for i in range(0, len(x), 10):
            print(f"{x[i]:5.3f}  {out['p_0.25'][i]:9.4f}  {out['r_0.25'][i]:9.4f}  "
                f"{out['signxi_0.25'][i]:9.4f}  {out['omega_0.25'][i]:9.4f}")
        plot_slices(out, f"{base_dir}/numerical_{gamma}_{tau}_{sigma}_{a}.png")
        np.savez(f"{base_dir}/numerical_{gamma}_{tau}_{sigma}_{a}.npz", **out)
