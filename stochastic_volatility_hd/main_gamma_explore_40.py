"""40-agent calibration: gamma pack x (rho, delta), timestep_rar only.

Targets (ergodic-mean summary): r ~ 2%, g ~ 2%, aggregate RP ~ 4-5%.

delta only enters through g = (p - B)/(2A) - delta, and iota depends on
g + delta = (p - B)/(2A), so delta shifts g one-for-one without touching the
goods market.  With a = 0.1, goods clearing pins g + delta to roughly
0.029-0.032 for consumption-wealth ratios of 0.03-0.037, so g = 2% needs
delta ~ 0.009.  rho mostly moves the consumption-wealth ratio, hence r - g,
and barely moves g (dg/dC ~ -0.36).
"""

import gc
import json
import os
import torch

from common import (
    BASE_PARAMS,
    BASE_TAU,
    BASE_V_MEAN,
    CONFIGS,
    MODEL_ROOT,
    PAPER_A,
    PAPER_SIGMA,
    SHARE_ALPHA_HI,
    SHARE_ALPHA_LO,
    SIMULATION_SEED,
    TRAINING_SEED,
    format_parameter,
    move_model,
)
from main import TRAINING
from model import get_model
from simulate import (
    NNEconomy,
    analyze,
    impulse_response,
    plot_impulse_response,
    plot_portfolio_deciles,
    plot_portfolio_terciles,
    plot_wealth_distribution,
    simulate,
)


K = 40
N_E = 36
N_H = K - N_E
EXPERT_IDX = list(range(N_E))
HOUSEHOLD_IDX = list(range(N_E, K))
CONFIG = "timestep_rar"

BASE_RHO = BASE_PARAMS["rho"]      # 0.0665
BASE_DELTA = BASE_PARAMS["delta"]  # 0.05

# Targets used only for printed diagnostics (not hard constraints).
TARGET_R = 0.02
TARGET_G = 0.02
TARGET_RP_LO, TARGET_RP_HI = 0.04, 0.05


def _lin(lo, hi, n):
    if n == 1:
        return [float(lo)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _pack(expert_gammas, household_gammas):
    assert len(expert_gammas) == N_E and len(household_gammas) == N_H
    return list(expert_gammas) + list(household_gammas)


GAMMA_EXPERIMENTS = {
    # Same expert band as agents20 [3,10]; graded households.
    "match20_band": _pack(_lin(3.0, 10.0, N_E), _lin(12.0, 16.5, N_H)),
    # Higher floor / higher harmonic-mean gamma -> typically higher pi.
    "higher_floor": _pack(_lin(4.0, 11.0, N_E), _lin(13.0, 17.0, N_H)),
}

# (gamma_name, rho, delta)
CANDIDATES = (
    ("match20_band", BASE_RHO, 0.009),
    ("higher_floor", BASE_RHO, 0.009),
)


def economic_params(tau=BASE_TAU, v_mean=BASE_V_MEAN, rho=BASE_RHO,
                    delta=BASE_DELTA):
    return BASE_PARAMS | {
        "a": PAPER_A,
        "sigma": PAPER_SIGMA,
        "tau": tau,
        "v_mean": v_mean,
        "rho": rho,
        "delta": delta,
    }


def run_name(gamma_name, rho, delta):
    name = f"agents40_gamma_{gamma_name}"
    if abs(rho - BASE_RHO) > 1e-12:
        name += f"_rho{format_parameter(rho)}"
    if abs(delta - BASE_DELTA) > 1e-12:
        name += f"_delta{format_parameter(delta)}"
    return name


def train_and_simulate(gamma_name, gamma_vec, rho, delta,
                       tau=BASE_TAU, v_mean=BASE_V_MEAN, tag=""):
    folder = run_name(gamma_name, rho, delta)
    root = os.path.join(MODEL_ROOT, folder)
    path = os.path.join(root, CONFIG)
    os.makedirs(path, exist_ok=True)
    params = economic_params(tau, v_mean, rho=rho, delta=delta)

    metadata = {
        "case": "agents40_gamma_explore",
        "experiment": gamma_name,
        "tag": tag,
        "K": K,
        "expert_idx": EXPERT_IDX,
        "household_idx": HOUSEHOLD_IDX,
        "gamma": gamma_vec,
        "parameters": params,
        "targets": {
            "risk_free_rate": TARGET_R,
            "output_growth_g": TARGET_G,
            "aggregate_risk_premium": [TARGET_RP_LO, TARGET_RP_HI],
            "investment_capital_ratio_iota": [0.10, 0.25],
        },
        "configs": [CONFIG],
        "training_seed": TRAINING_SEED,
        "training": TRAINING,
    }
    with open(os.path.join(root, "experiment.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    ts, rar, lb = CONFIGS[CONFIG]
    init_guess = {f"xi_{k}": params["rho"] for k in range(1, K + 1)}
    init_guess["r"] = TARGET_R

    print(f"\n{(' ' + folder + ' / ' + CONFIG + ' '):=^80}")
    print(f"tag={tag}  rho={rho:g}  delta={delta:g}")
    print(f"gamma experts: [{gamma_vec[0]:.3g}, ..., {gamma_vec[N_E - 1]:.3g}]")
    print(f"gamma households: {gamma_vec[N_E:]}")

    model = get_model(
        path,
        K,
        EXPERT_IDX,
        HOUSEHOLD_IDX,
        gamma_vec,
        model_size=[TRAINING["width"]] * TRAINING["layers"],
        n_epochs=TRAINING["epochs"],
        batch_size=TRAINING["batch"],
        lr=TRAINING["lr"],
        timestepping=ts,
        rar=rar,
        loss_balancing=lb,
        num_outer=TRAINING["outer"],
        num_inner=TRAINING["num_inner"],
        min_inner=TRAINING["min_inner"],
        lr_decay_every=TRAINING["lr_decay_every"],
        lr_decay_gamma=TRAINING["lr_decay_gamma"],
        loss_balancing_alpha=TRAINING["loss_balancing_alpha"],
        loss_balancing_temp=TRAINING["loss_balancing_temp"],
        bernoulli_prob=TRAINING["bernoulli_prob"],
        t0_frac=TRAINING["t0_frac"],
        init_guess=init_guess,
        params=params,
        share_alpha_lo=SHARE_ALPHA_LO,
        share_alpha_hi=SHARE_ALPHA_HI,
        seed=TRAINING_SEED,
    )

    economy = NNEconomy(model)
    out_dir = os.path.join(path, "simulation")
    sim = simulate(
        economy,
        n_paths=100,
        years=500.0,
        dt=0.08,
        x0=0.2,
        v0=0.1,
        seed=SIMULATION_SEED,
    )
    summary = analyze(economy, sim, out_dir)
    plot_portfolio_deciles(economy, sim, out_dir, seed=SIMULATION_SEED)
    plot_portfolio_terciles(economy, sim, out_dir, seed=SIMULATION_SEED)
    plot_wealth_distribution(economy, sim, out_dir)
    irf = impulse_response(
        economy, x0=0.2, v0=0.1, seed=SIMULATION_SEED, include_vfixed=True,
    )
    plot_impulse_response(economy, irf, out_dir, xmax=7.0)

    r = summary.get("r_free", float("nan"))
    r_implied = summary.get("r_implied", float("nan"))
    g = summary.get("output_growth_g", float("nan"))
    iota = summary.get("investment_capital_ratio_iota", float("nan"))
    rp = summary.get("risk_premium_aggregate", float("nan"))
    print(
        f"[explore] {folder}\n"
        f"  r_free={r:.4f}  r_implied={r_implied:.4f} (target {TARGET_R:.2f})  "
        f"g={g:.4f} (target {TARGET_G:.2f})  "
        f"iota={iota:.4f} (target 0.10-0.25)\n"
        f"  agg_rp={rp:.4f} (target {TARGET_RP_LO:.2f}-{TARGET_RP_HI:.2f})  "
        f"X_E={summary.get('expert_wealth_share_X_E', float('nan')):.4f}"
    )

    move_model(model, "cpu")
    del model, economy, sim, irf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    torch.set_default_dtype(torch.float64)
    for gamma_name, rho, delta in CANDIDATES:
        train_and_simulate(
            gamma_name,
            GAMMA_EXPERIMENTS[gamma_name],
            rho=rho,
            delta=delta,
            tag="calibration",
        )


if __name__ == "__main__":
    main()
