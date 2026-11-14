# Heterogeneous N-agent Gârleanu–Panageas model — design spec

This file is the design document for `gp_n_agents_NN.py`.  It pins down the
*math first*, before any code, so that every loss component / NN output /
tensor shape can be traced back to a specific equilibrium condition.

The model is a *heterogeneous* version of the Gârleanu–Panageas (GP) economy
already implemented in `gp_base.py`.  The base file has **three** agents — an
unconstrained active agent `u`, a constrained active agent `c`, and a third
agent `p` that holds the residual wealth share $x_p = 1 - x_u - x_c$.  The model
in this directory keeps **all** agents active and generalises to arbitrary $N$:

> **$N$ active agents whose wealth shares sum to one.**  No passive optimiser.
> No rotational / permutation symmetry — every agent has its own preference
> parameters $(\gamma_i, \psi_i, \omega_i)$.

### The closure: $\sum_i x_i = 1$

This is the standard GP closure.  The $N$ active agents **are** the whole
economy: they collectively own all wealth, so their shares satisfy
$$
\sum_{i=1}^N x_i = 1.
$$
Combined with market clearing $\sum_i x_i\alpha_i = 1$ this means the bond is in
**zero net supply** (no outside claim) and the agents' average risky share is
$\approx 1$ everywhere — there is no leverage blow-up.

Because the shares sum to one, the state space is the **$(N-1)$-simplex**.  We
parametrise it by the $N-1$ **free** shares $x_1, \dots, x_{N-1}$ and treat
agent $N$ as the **dependent** coordinate
$$
x_N = 1 - \sum_{j=1}^{N-1} x_j .
$$
This is exactly the role the third agent $x_p = 1 - x_u - x_c$ played in
`gp_base.py`, except agent $N$ is now a full optimiser (its own HJB and FOC).
Since $x_N$ depends on every free coordinate ($\partial x_N/\partial x_j = -1$),
the closed-form $y$-derivatives carry extra $-\xi_N$ / $-J_{N,\cdot}$ terms (§2).

### What changed relative to the 3-agent base (`gp_base.py`)

| Object | 3-agent base | This model ($N$ active) |
|--------|--------------|--------------------------|
| Agents | $u, c, p$ | $1, \dots, N$ (all active optimisers) |
| Wealth multipliers | $\xi_u, \xi_c, \xi_p$ | $\xi_1, \dots, \xi_N$ |
| Dependent share | $x_p = 1 - x_u - x_c$ | $x_N = 1 - \sum_{j<N} x_j$ |
| Aggregate $y$ | $x_u\xi_u + x_c\xi_c + x_p\xi_p$ | $\sum_{i=1}^N x_i\xi_i$ |
| Risky shares | $\alpha_u, \alpha_c, \alpha_p$ | $\alpha_1, \dots, \alpha_N$ |
| Market clearing | $\sum x_i\alpha_i = 1$ | $\sum_{i=1}^N x_i\alpha_i = 1$ |
| Agent-$p$ HJB / FOC | special (passive) | agent $N$ is a normal active agent |

> **Why not just drop an agent and let $\sum_i x_i < 1$?**  If the $N$ agents
> hold less than the whole economy, the residual wealth $1 - \sum_i x_i$ has no
> owner: the aggregate consumption-wealth ratio $y = \sum_i x_i\xi_i$ then
> *understates* aggregate consumption, the risk-free rate $r = y + \mu_P - \pi$
> built on it is inconsistent with the agents' HJBs, and market clearing forces
> the surviving agents to hold the entire risky asset themselves
> ($\bar\alpha = 1/\sum_i x_i$, which blows up to $20$–$50\times$ near the
> corner).  Both effects floor the training loss around $0.2$.  Enforcing
> $\sum_i x_i = 1$ removes both pathologies.

Three consequences for the implementation:

1. We **cannot** use a DeepSet, because the agents are not exchangeable.
2. For separability at large $N$ we use **one MLP per agent** (`output_size=1`):
   $N$ networks for $\xi_i$ and $N$ for $\alpha_i$.  This matches `gp_base.py`
   (separate `add_agent("xiu"|"xic"|...)`) extended to arbitrary $N$.
3. Every network takes the **$N-1$ free shares** as input; the dependent share
   $x_N$ is reconstructed inside the closed-form blocks.

---

## 1.  State, agents, and notation

| Object | Symbol | Dim | Description |
|--------|--------|-----|-------------|
| Free state variables | $x = (x_1, \dots, x_{N-1})$ | $N-1$ | wealth shares of agents $1, \dots, N-1$ |
| Dependent share | $x_N = 1 - \sum_{j<N} x_j$ | 1 | agent $N$'s share (not a free state) |
| Sampling region | $x_i \ge \varepsilon,\ \sum_{j<N} x_j \le 1 - \varepsilon$ | — | interior of the $(N-1)$-simplex, $\varepsilon \approx 0.02$ |
| Active wealth multipliers | $\xi_i(x)$, $i = 1, \dots, N$ | $N$ | one NN output per agent |
| Active risky shares | $\alpha_i(x)$, $i = 1, \dots, N$ | $N$ | one NN output per agent |
| Preference parameters | $\gamma_i, \psi_i, \omega_i$, $i = 1, \dots, N$ | $N$ each | heterogeneous |
| Aggregate-endowment vol | $\sigma = (\sigma^1, 0) \in \mathbb{R}^2$ | 2 | 1-shock economy ($\sigma^2 = 0$) |
| Discount rate / drift | $\rho, \mu$ | scalar | shared |
| Death intensity | $\kappa$ | scalar | shared (0 by default) |

We follow the **single-shock** specialisation used in `gp_base.py`:
$\sigma^2 = 0$.  All vector dot-products below collapse to 1-D, but we keep the
2-D shock dimension for parity with the base code.

Throughout, $i$ indexes **all $N$ agents**, while $j, k$ index the **$N-1$ free
states**.

---

## 2.  Aggregate consumption-wealth ratio and its closed-form derivatives

By definition (all $N$ agents, with $x_N = 1 - \sum_{j<N} x_j$),
$$
y(x) = \sum_{i=1}^N x_i\, \xi_i(x).
$$
$y$ is linear in the NN outputs, so its derivatives w.r.t. the **free** states
can be written analytically in terms of the network outputs and their batched
Jacobians/Hessians (no redundant `vmap(jacrev/hessian)` pass through $y$).

Let
$$
J_{i,j} := \frac{\partial \xi_i}{\partial x_j},
\qquad
H_{i,j,k} := \frac{\partial^2 \xi_i}{\partial x_j \partial x_k},
\qquad i \in \{1,\dots,N\},\ j,k \in \{1,\dots,N-1\}.
$$

Using $\partial x_N/\partial x_j = -1$ (the dependent-coordinate correction):

$$
\boxed{\;
\frac{\partial y}{\partial x_j}
=
(\xi_j - \xi_N)
+ \sum_{i=1}^N x_i\, J_{i,j}
\;}
\qquad (j = 1, \dots, N-1)
$$

$$
\boxed{\;
\frac{\partial^2 y}{\partial x_j \partial x_k}
=
\bigl(J_{j,k} + J_{k,j}\bigr)
-\bigl(J_{N,k} + J_{N,j}\bigr)
+\sum_{i=1}^N x_i\, H_{i,j,k}
\;}
$$

The $-\xi_N$ and $-J_{N,\cdot}$ terms are precisely what was missing when agents
were treated as independent with $\sum_i x_i < 1$.

Implementation (`compute_x_full` / `compute_y_closed` / `compute_dy_dx_closed`
/ `compute_d2y_dx2_closed`):

```python
# Inputs:
#   SV:           (B, N-1)        free shares x_1, ..., x_{N-1}
#   xi_active:    (B, N)          ξ_i (all N agents)
#   xi_active_J:  (B, N, N-1)     ∂ξ_i / ∂x_j   (j over free states)
#   xi_active_H:  (B, N, N-1, N-1)
# Outputs:
#   y:            (B, 1)
#   dy_dx:        (B, 1, N-1)
#   d2y_dx2:      (B, 1, N-1, N-1)
# x_N = 1 - sum_{j<N} x_j is appended internally by compute_x_full.
```

---

## 3.  Vol and drift coefficients (in 2-D shock space, sigma2 = 0)

Diffusion of each **free** state variable $x_j$ ($j = 1, \dots, N-1$):
$$
\sigma_{x_j} = x_j (\alpha_j - 1) \sigma_R \in \mathbb{R}^2,
$$
stacked as $\sigma_x \in \mathbb{R}^{B \times (N-1) \times 2}$.  (The dependent
share carries $\sigma_{x_N} = -\sum_{j<N}\sigma_{x_j}$, consistent with
$\sum_i \sigma_{x_i} = 0$ at market clearing.)

Aggregate $y$-vol (sum over **free** states):
$$
A \;=\; \sum_{j=1}^{N-1} \frac{\partial y}{\partial x_j}\, x_j (\alpha_j - 1),
\qquad
\sigma_y \;=\; \frac{A\,\sigma}{y + A} \in \mathbb{R}^2.
$$

Return vol:
$$
\sigma_R = \sigma - \sigma_y, \qquad \|\sigma_R\|^2 = (\sigma_R^1)^2 + (\sigma_R^2)^2.
$$

Wealth-multiplier vol (for each agent $i = 1, \dots, N$, contracting over the
$N-1$ free states):
$$
\sigma_{\xi_i} \;=\; \frac{1}{\xi_i} \sum_{j=1}^{N-1} J_{i,j}\, \sigma_{x_j}
\in \mathbb{R}^2,
\qquad
\sigma_{\xi} = (J/\xi)\cdot\sigma_x \in \mathbb{R}^{B \times N \times 2}.
$$

Hedging-demand object (each agent $i$):
$$
\varsigma_i \;=\; \frac{1 - 1/\gamma_i}{1 - \psi_i}
\;\frac{\sigma_{\xi_i}\cdot\sigma_R}{\|\sigma_R\|^2}.
$$

Drift of each **free** state ($j = 1, \dots, N-1$):
$$
\mu_{x_j} = x_j \bigl(y - \xi_j + (1-\alpha_j)(1-q)\|\sigma_R\|^2 \bigr)
           + \kappa(\omega_j - x_j).
$$

Covariance of the free states:
$$
\Sigma_{j,k} = \sigma_{x_j}\cdot\sigma_{x_k}
            = x_j x_k (\alpha_j - 1)(\alpha_k - 1)\|\sigma_R\|^2,
\qquad j,k = 1,\dots,N-1.
$$

Drift of $\xi_i$ (Itô on the NN output, contracted over free states):
$$
\mu_{\xi_i}
= \frac{1}{\xi_i} \sum_{j=1}^{N-1} J_{i,j}\, \mu_{x_j}
+ \frac{1}{2\xi_i}\sum_{j,k=1}^{N-1} H_{i,j,k}\,\Sigma_{k,j}.
$$
Drift of $y$ is the same formula with $\partial y/\partial x_j$, $\partial^2
y/\partial x_j\partial x_k$ and $y$ in place of $\xi_i$.

Asset-price drift and risk-free rate (unchanged from `gp_base.py`):
$$
\mu_P = \mu - \mu_y + \sigma_y\cdot(\sigma_y - \sigma),
\qquad
r = y + \mu_P - \pi.
$$

---

## 4.  Definition of $q$ (state-price ratio anchor)

The portfolio FOC for **every** agent $i$ reads $\alpha_i = q/\gamma_i -
\varsigma_i$, a single number $q$ in equilibrium.  We *anchor* on agent 1:
$$
q := \gamma_1 \bigl(\alpha_1 + \varsigma_1\bigr),
\qquad
\pi = q\|\sigma_R\|^2,\quad \eta = q\|\sigma_R\|.
$$
With this definition agent 1's FOC is identically satisfied; for all other
agents it becomes a non-trivial loss component (§5).

---

## 5.  Loss components (equilibrium residuals)

| # | Name in code | Equation | Tensor shape |
|---|--------------|----------|--------------|
| L1 | `hjbeq_1` (`hjb`) | $\text{HJB}_i(x) = 0$, $i = 1,\dots,N$, scaled by $1/\rho$ | $(B, 1)$, $=\sum_i \text{HJB}_i^2$ per sample |
| L2 | `endogeq_mc` | $\sum_{i=1}^N x_i \alpha_i = 1$ (with $x_N$ dependent) | $(B, 1)$ |
| L3 | `endogeq_vi_active` | $\min(\bar\alpha_i - \alpha_i,\; q/\gamma_i - \varsigma_i - \alpha_i) = 0$, $i = 1,\dots,N$ | $(B, N)$ (agent 1 residual $\equiv 0$ when unconstrained) |
| L4 | `endogeq_pricing` | $\pi = \dfrac{\bigl(1 + \sum_i x_i \varsigma_i\bigr)\|\sigma_R\|^2}{\sum_i x_i/\gamma_i}$ | $(B, 1)$ |

The HJB for agent $i$ is the usual GP residual:
$$
\text{HJB}_i = \rho\psi_i + (1-\psi_i)\!\left(r + \eta\alpha_i\|\sigma_R\| - \tfrac{\gamma_i}{2}(\alpha_i\|\sigma_R\|)^2\right)
+ \mu_{\xi_i} + (1-\gamma_i)\,\sigma_{\xi_i}\!\cdot\sigma_R\,\alpha_i
+ \tfrac{1}{2}\tfrac{\psi_i - \gamma_i}{1-\psi_i}\|\sigma_{\xi_i}\|^2 - \xi_i.
$$

### Leverage constraints (the variational inequality L3)

`alpha_caps` is a length-$N$ vector.  Entry $\bar\alpha_i = +\infty$ (coded as
$\ge 10^3$) marks agent $i$ **unconstrained**; a finite $\bar\alpha_i$ imposes
$\alpha_i \le \bar\alpha_i$ via the NCP residual
$$
\min\bigl(\bar\alpha_i - \alpha_i,\; \tfrac{q}{\gamma_i} - \varsigma_i - \alpha_i\bigr) = 0,
$$
which reduces to the plain Merton FOC when the cap is inactive and to
$\alpha_i = \bar\alpha_i$ otherwise.

> **Anchor caveat.** Agent 1 anchors $q$, so its FOC residual is identically
> zero; it *cannot* be a constrained agent.  Put capped agents at indices
> $2..N$ (`alpha_caps[0]` must be $\ge 10^3$).

> **Redundancy of pricing (L4).** Substituting the FOCs into market clearing
> gives exactly $q = (1 + \sum_i x_i\varsigma_i)/(\sum_i x_i/\gamma_i)$, so L4 is
> implied by L2 + L3 and is kept only as an over-identifying training signal.

---

## 6.  Neural-network architecture

There are **$2N$** NNs, all taking the $N-1$ free shares as input, all with
`output_size = 1` and private parameters (no weight sharing).

| name | role | input dim | output dim | extra config |
|------|------|-----------|-----------|--------------|
| `xi_1, ..., xi_N` | one per agent | $N-1$ | each **1** | positive output (softplus) |
| `alpha_1, ..., alpha_N` | one per agent | $N-1$ | each **1** | `derivative_order=0` |

The active outputs and their derivatives are stacked in one fused `vmap` call
(`StackedAgent`):

```python
xi_active      = stack([xi_1, ..., xi_N])           # (B, N)
xi_active_Jac  = stack([xi_1_Jac, ..., xi_N_Jac])   # (B, N, N-1)
xi_active_Hess = stack([xi_1_Hess, ..., xi_N_Hess]) # (B, N, N-1, N-1)
alpha_active   = stack([alpha_1, ..., alpha_N])     # (B, N)
```

Closed-form `y`, `dy_dx`, `d2y_dx2` follow from §2.  These are the ONLY
differentiations the framework needs — no separate `vmap(jacrev)` through $y$.

---

## 7.  Calibration template & validation cases

The script accepts $\gamma_{\text{active}} \in \mathbb{R}^N$ (no symmetry
imposed).  Defaults: $\psi_i = 1.5$; $\rho = 0.05$, $\mu = 0.0183$,
$\sigma = (0.0357, 0)$, $\kappa = 0$.

### Validation cases

The reference solver `gp_n_agents_numerical.py` solves the **2-agent** cases as
a 1-D Chebyshev BVP over $x = x_1 \in (0,1)$ (with $x_2 = 1 - x_1$), using all
four fields $\xi_1(x), \xi_2(x), \alpha_1(x), \alpha_2(x)$ and a $\gamma$/cap
homotopy from the trivial representative-agent start.

| case | $N$ | $\gamma$ | caps | sanity check |
|------|-----|----------|------|--------------|
| **sym2**        | 2 | $(4, 4)$   | none            | Identical agents on $\sum x_i = 1$ ⇒ representative agent: $\alpha_1 = \alpha_2 = 1$, flat prices.  Degenerate sanity check. |
| **sym2_const**  | 2 | $(4, 4)$   | $(\infty, 1.8)$ | Cap $\ge 1$ never binds at $\alpha = 1$, so this coincides with `sym2`. |
| **asym2**       | 2 | $(8, 4)$   | none            | Non-trivial 1-D problem.  $\alpha_1 < 1 < \alpha_2$ (the more risk-averse agent 1 holds less risky), market clearing $x_1\alpha_1 + x_2\alpha_2 = 1$.  NN matches Chebyshev to $\sim$3 sig. figs. |
| **asym2_const** | 2 | $(8, 4)$   | $(\infty, 1.8)$ | Agent 2 wants $\alpha_2 > 1$; the cap binds near $x_1 \to 1$.  NN matches the Fischer–Burmeister Chebyshev reference. |

Larger benchmarks (`mix20`, `mix50`) keep the same structure with dispersed
$\gamma_i$ and per-agent caps; they are evaluated along the $x_1$ cut below.

### The Chebyshev reference (N = 2, $\sum x_i = 1$)

State $x = x_1$, dependent $x_2 = 1 - x_1$.  With
$$
y = x\xi_1 + (1-x)\xi_2,\quad
y_x = (\xi_1 - \xi_2) + x\xi_{1,x} + (1-x)\xi_{2,x},
$$
the only state diffusion is $\sigma_{x} = x(\alpha_1 - 1)\sigma_R$, and the
system is HJB$_1$, HJB$_2$, market clearing $x\alpha_1 + (1-x)\alpha_2 = 1$, and
agent 2's FOC / capped VI.  For $\gamma_1 = \gamma_2$ this collapses to the
representative agent ($\alpha \equiv 1$); heterogeneity or a binding cap makes it
genuinely non-trivial.  All four cases solve to $\|\text{res}\|_\infty \sim
10^{-14}$.

---

## 8.  Sampling and training notes

- **Sampler**: Dirichlet on the $N$-simplex with $\varepsilon$ truncation, then
  drop the last coordinate to obtain the $N-1$ free shares.  This keeps
  $\sum_{j<N} x_j \le 1-\varepsilon$, i.e. the dependent share $x_N \ge
  \varepsilon$ stays strictly positive.  (Identical code to the base sampler,
  now interpreted as the $(N-1)$-simplex of free coordinates.)
- **Optimizer**: Adam at lr = 5e-4 – 1e-3, gradient clipping at norm 1.
- **Epochs / batch**: 2000–20000 epochs, batch 200–500.  On `asym2` the loss
  drops below $10^{-3}$ within ~2000 epochs (cf. the $\sim 0.2$ floor under the
  inconsistent $\sum x_i < 1$ setup).  Cost scales linearly in $N$ for the NN
  forward and roughly cubically in $N$ for the Hessian over the $N-1$ free
  states.

---

## 9.  Diagnostics

The model is evaluated and plotted along the **$x_1$ cut** of the simplex:
$x_1$ varies over $(0,1)$ and the remaining mass $1 - x_1$ is split equally
among the other $N-1$ agents ($x_2 = \dots = x_N = (1-x_1)/(N-1)$).  For $N = 2$
this is simply $x_1 \in (0,1)$ with $x_2 = 1 - x_1$.

- **Equilibrium slice**: $r, \pi, y$ and every $\alpha_i$ plotted vs $x_1$, with
  the Chebyshev reference (`alpha_1`, `alpha_2`, …) overlaid for the 2-agent
  cases.
- **Cross-method comparison**: validation-loss table, RAR anchor scatter
  (2-D simplex for $N = 2$), and $y$ / $\alpha$-sum panels — all on the $x_1$
  cut and the full $(N-1)$-simplex validation sample.
