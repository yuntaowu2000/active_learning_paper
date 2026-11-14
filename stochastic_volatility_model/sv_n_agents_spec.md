# Heterogeneous N-agent Di Tella (2017) stochastic-volatility model — design spec

This file pins down the *math first* for `sv_n_agents_NN.py`, the N-agent
generalisation of the two-agent Di Tella stochastic-volatility model in
`stochastic_volatility_model.py`.  Every loss component / NN output / tensor
shape traces back to a specific equilibrium condition, and the whole system
reduces **exactly** to the original 2-agent model when `K = 2` (one expert,
one household) so the existing finite-difference solution validates it.

The design deliberately mirrors `complete_market_model/gp_n_agents_NN.py` for
efficient computation: one MLP per agent value function, fused into a single
`vmap` call (`StackedAgent`), an anchored state-price (`χ` here, `q` there),
per-agent capital shares (`θ_k` here, `α_i` there), a market-clearing loss, a
variational-inequality (NCP) free boundary, and a vectorised HJB.

---

## 0. Why Di Tella is a *single aggregate shock* model

In the original equations every aggregate diffusion (`σ_n, σ_p, σ_x, σ_ξ,
σ_ζ, σ_v`) is a **scalar** loading on one aggregate Brownian `Z`.  `v` (the
*idiosyncratic-risk volatility* state) loads on the same `Z` via
`σ_v = σ̄_v √v`.  Experts additionally bear **idiosyncratic** risk of
magnitude `σ̃_n = φ v / x` (separate per-expert Brownians that wash out in the
aggregate).  This is what makes the multi-agent generalisation tractable: the
share diffusions solve one differentiable linear system (§3), not a 2-shock
vector system.

---

## 1. State, agents, notation

| Object | Symbol | Dim | Description |
|--------|--------|-----|-------------|
| Wealth shares | `x = (x_1,…,x_{K-1})` | `K-1` | shares of the first `K-1` agent types |
| Residual share | `x_K = 1 - Σ_{k<K} x_k` | 1 | last type's share (not a state input) |
| Volatility state | `v` | 1 | idiosyncratic-risk volatility |
| **State vector** | `s = (x_1,…,x_{K-1}, v)` | `D = K` | NN inputs |
| Value multipliers | `ξ_k(s)`, `k=1..K` | `K` | one positive-output MLP per type |
| Expert capital shares | `θ_k(s)`, `k ∈ E` | `|E|` | one MLP per expert (value-only) |
| Risk aversion | `γ_k` | `K` | **heterogeneous** |
| Types | `E` (experts), `H` (households) | — | `E ∪ H = {1,…,K}` |

Shared params: `ρ` (discount), `ψ` (inverse EIS), `τ` (expert retirement
intensity), `φ` (moral hazard / skin-in-the-game), `σ` (TFP vol), `λ, v̄, σ̄_v`
(vol process), `A, B, δ` (investment), `a` (productivity).

Sampling region: `x_k ≥ ε`, `Σ_k x_k ≤ 1 - ε` (open simplex, `ε≈0.02`),
`v ∈ [0.05, 0.95]`.

Validation reduction (`K=2`, `E={1}`, `H={2}`): `x_1 = x` (expert share),
`x_2 = 1-x` (household share), state `(x, v)` — identical to the original.

---

## 2. Aggregate (capital) block — unchanged, depends only on `p`

```
g   = (p - B)/(2A) - δ
ι   = A (g+δ)^2 + B (g+δ)
μ_v = λ (v̄ - v)
σ_v = σ̄_v √v
ĉ_k = ρ^{1/ψ} ξ_k^{(ψ-1)/ψ}        (consumption-wealth ratio of type k)
```
`ĉ_k` generalises `ê` (expert) and `ĉ` (household).

---

## 3. Aggregate-risk diffusions — the share-diffusion linear system

All scalar loadings on `Z`.  Let `m, l` range over `1..K-1` (state shares).

```
σ_p     = (1/p)  ( p_v σ_v   + Σ_m p_{x_m}  σ_{x_m} )
σ_{ξ_k} = (1/ξ_k)( ξ_{k,v} σ_v + Σ_m ξ_{k,x_m} σ_{x_m} )
```
Each type's aggregate-risk loading from its **portfolio FOC**:
```
σ_{n,k} = π/γ_k - (γ_k-1)/γ_k σ_{ξ_k}
```
Price of aggregate risk from **risk-market clearing** `Σ_k x_k σ_{n,k} = σ+σ_p`:
```
π = [ (σ+σ_p) + Σ_k x_k (γ_k-1)/γ_k σ_{ξ_k} ] / [ Σ_k x_k/γ_k ]
```
Wealth-share diffusions:
```
σ_{x_k} = x_k ( σ_{n,k} - (σ+σ_p) )      ⇒  Σ_k σ_{x_k} = 0 (identically)
```
Because `σ_{ξ_k}`, `σ_p`, `π` are all affine in the unknowns
`u = (σ_{x_1},…,σ_{x_{K-1}})`, these collapse to one linear system
`(I - M) u = c`, solved with a **differentiable batched** `torch.linalg.solve`:

```
a_k    = ξ_{k,v} σ_v / ξ_k ,           b_{k,m} = ξ_{k,x_m} / ξ_k
a_p    = p_v σ_v / p ,                 b_{p,m} = p_{x_m} / p
P0     = σ + a_p
S0     = Σ_k x_k (γ_k-1)/γ_k a_k ,     S_m   = Σ_k x_k (γ_k-1)/γ_k b_{k,m}
T      = Σ_k x_k/γ_k
π0     = (P0 + S0)/T ,                 π_m   = (b_{p,m} + S_m)/T
M_{k,m}= x_k/γ_k π_m - x_k (γ_k-1)/γ_k b_{k,m} - x_k b_{p,m}
c_k    = x_k/γ_k π0   - x_k (γ_k-1)/γ_k a_k   - x_k P0
```
(`k, m = 1..K-1`).  After solving `u`, set `σ_{x_K} = -Σ_m u_m`, then recompute
`σ_p, σ_{ξ_k}, π, σ_{n,k}` from the now-known `u`.

**`K=2` check.**  With `γ_1=γ_2=γ` the single equation reproduces
`σ_x = (1-x)x (1-γ)/γ (ξ_v/ξ-ζ_v/ζ) σ_v / [1 - (1-x)x (1-γ)/γ (ξ_x/ξ-ζ_x/ζ)]`,
i.e. the original `σ_{x,1}/σ_{x,2}·σ_v`.

---

## 4. Idiosyncratic risk, capital allocation, and the free boundary

Experts manage capital shares `θ_k ≥ 0`, `Σ_{k∈E} θ_k = 1` (households hold no
capital directly, only the diversified equity claim priced by `π`).  Skin-in-
the-game idiosyncratic exposure per unit wealth:
```
σ̃_{n,k} = φ v θ_k / x_k          (k ∈ E;  σ̃ = 0 for households)
```
The capital-management FOC equalises the (common) excess return on capital,
`χ`, with each active expert's marginal idiosyncratic-risk cost:
```
χ = γ_k (φ v)^2 θ_k / x_k          (interior, k ∈ E)
```
**Anchor (mirrors gp's `q`).**  Define `χ := γ_1 (φv)^2 θ_1 / x_1` from expert
1 (which must therefore be unconstrained).  The FOC target for expert `k` is
`θ_k^* = χ x_k / (γ_k (φv)^2)`.  The leverage cap `θ_k/x_k ≤ ℓ_k` enters as the
NCP residual (the variational free boundary):
```
min( ℓ_k x_k - θ_k ,  θ_k^* - θ_k ) = 0      (k ∈ E)
```
Off the cap this is the Merton/skin-in-the-game FOC residual `θ_k^* - θ_k`; on
the cap it pins `θ_k = ℓ_k x_k`.  Capped experts must be at indices `≥ 2`.

`K=2` check: a single expert holds `θ_1 = 1` by capital-market clearing, and
`χ = γ (φv)^2 / x` reproduces the original idiosyncratic premium `γ/x (φv)^2`.

---

## 5. Drifts, risk-free rate

Net-worth drift per unit wealth (before consumption):
```
μ_net,k = r + π σ_{n,k} + 1_{k∈E} · χ θ_k / x_k
```
Aggregate-wealth drift and aggregate risk:
```
σ_agg = σ + σ_p = Σ_j x_j σ_{n,j}
μ_N   = Σ_j x_j ( μ_net,j - ĉ_j )            (retirement nets to zero)
```
Share drifts (Itô on `x_k = n_k/N`) with expert retirement at rate `τ`
(retiring expert wealth flows to households pro-rata by household share):
```
μ_{x_k} = x_k [ (μ_net,k - ĉ_k) - μ_N - (σ_{n,k} - σ_agg) σ_agg ] + retire_k
retire_k = -τ x_k                                   (k ∈ E)
retire_k = +τ (Σ_{e∈E} x_e)(x_k / Σ_{h∈H} x_h)      (k ∈ H)
```
(`Σ_k μ_{x_k}=0`, `Σ_k retire_k=0`.)  Only `μ_{x_1..K-1}` are used.

Second-order drifts via the state covariance `Σ_state` (D×D) with
`Cov(x_m,x_l)=σ_{x_m}σ_{x_l}`, `Cov(x_m,v)=σ_{x_m}σ_v`, `Cov(v,v)=σ_v²`, and the
state drift vector `μ_s = (μ_{x_1},…,μ_{x_{K-1}}, μ_v)`:
```
μ_{ξ_k} = (1/ξ_k)[ Σ_d μ_{s_d} ξ_{k,s_d} + ½ tr( H[ξ_k] · Σ_state ) ]      (+ ξ_{k,t}/ξ_k in time-stepping)
μ_P     = (1/p)  [ Σ_d μ_{s_d} p_{s_d}    + ½ tr( H[p]   · Σ_state ) ]      (+ p_t/p   in time-stepping)
```
Risk-free rate from the **asset-pricing** condition (generalises the original
3rd endogenous equation; `r` is *computed*, not a network):
```
r = (a-ι)/p + g + μ_P + σ σ_p - (σ+σ_p) π - χ
```

---

## 6. Loss components (equilibrium residuals)

| # | label | equation | shape | reduction |
|---|-------|----------|-------|-----------|
| L1 | `endogeq_goods` | `a - ι = p Σ_k x_k ĉ_k` | (B,1) | MSE |
| L2 | `endogeq_capital` | `Σ_{k∈E} θ_k = 1` | (B,1) | MSE |
| L3 | `endogeq_vi_expert` | `min(ℓ_k x_k - θ_k, θ_k^* - θ_k) = 0`, `k∈E` | (B,\|E\|) | MSE |
| L4 | `hjbeq_expert` | mean over `k∈E` of `HJB_k²` (per sample) | (B,1) | MAE |
| L5 | `hjbeq_household` | mean over `k∈H` of `HJB_k²` (per sample) | (B,1) | MAE |

Risk-market clearing and asset pricing are satisfied **by construction**
(`π` and `r` are solved/derived in §3, §5), so they are not separate losses.

**Vectorised, per-type HJB** ("one value per agent type").  For each type:
```
HJB_k = ĉ_k^{1-ψ}/(1-ψ) ρ ξ_k^{ψ-1}
      + 1_{k∈E} · τ/(1-γ_k) ( (ξ_ret/ξ_k)^{1-γ_k} - 1 )
      + μ_net,k - ĉ_k + μ_{ξ_k}
      - γ_k/2 ( σ_{n,k}² + σ_{ξ_k}² - 2 (1-γ_k)/γ_k σ_{n,k} σ_{ξ_k} + 1_{k∈E} σ̃_{n,k}² )
      - ρ/(1-ψ)
```
`ξ_ret = Σ_{j∈H} x_j ξ_j / Σ_{j∈H} x_j` (wealth-weighted household value; `=ζ`
when `K=2`).  Registering `hjbeq_expert`/`hjbeq_household` separately gives one
average-HJB number per type, exactly as requested, and keeps the per-sample
residual non-trivial so residual-adaptive sampling (RAR) has signal.

---

## 7. Neural-network architecture (efficient computation)

`K` value-function MLPs `ξ_1..ξ_K` (positive output) + `|E|` capital-share MLPs
`θ_k` (`k∈E`).  All `ξ_k` share an architecture and are fused into one
`vmap` call producing `ξ`(B,K), `ξ_Jac`(B,K,D), `ξ_Hess`(B,K,D,D); the `θ_k`
are fused value-only into `θ`(B,|E|).  The single price network `p` uses the
library's `batch_jac_hes` to expose `p_Jac`(B,1,D), `p_Hess`(B,1,D,D).  Cost
scales linearly in `K` for the forward and ~cubically for the fused Hessian.

---

## 8. Configurations (8, run together by `__main__`)

Stationary `PDEModel`: `basic`, `basic+RAR`, `basic+LB`, `basic+RAR+LB`.
`PDEModelTimeStep`: `timestep`, `timestep+RAR`, `timestep+LB`, `timestep+RAR+LB`.
(`RAR` = residual-adaptive greedy sampling; `LB` = ReLoBRaLo loss balancing.)
The 7 non-basic methods are compared against `basic` in the tables.

Cases: `agents2` (1 expert + 1 household, original params, no cap — validation),
`agents20`, `agents50` (mixed experts/households, heterogeneous `γ`, leverage
caps on a subset of experts to activate the free boundary).

---

## 9. Diagnostics (plots & tables)

- **2-D slice vs numerical** (`agents2` only): `p, σ_x, Ω=ξ/ζ, σ+σ_p, π, r` on
  the slices `v ∈ {0.1, 0.25, 0.6}` vs `x`, overlaid on the Di Tella FD solution.
- **RAR sampled points**: scatter of anchor points in `(x_1, v)` (2-D cases).
- **Loss-weight evolution**: ReLoBRaLo weights per loss label across epochs.
- **Convergence**: per-type average HJB loss and total loss vs epoch.
- **Comparative error table**: HJB(expert), HJB(household), total for all 7
  methods + % improvement over `basic`.
- **Welfare-equivalent error table**: each residual mapped to a certainty-
  equivalent consumption-wealth-ratio cost, + % improvement over `basic`.
