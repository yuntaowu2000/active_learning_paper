"""40-agent gamma calibrations: train/simulate timestep_rar only.

Suggested gamma vectors target the gap vs agents20 aggregate risk premium
(~5.2% vs ~3.7%).  At fixed X_E, more expert diversification lowers pi and
sig_p; these designs either (i) restore low-gamma concentration or (ii) raise
the wealth-weighted harmonic mean of gamma.

Edit ACTIVE_EXPERIMENTS to choose which packs to run.
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
    SHARE_ALPHA_HI,
    SHARE_ALPHA_LO,
    SIMULATION_SEED,
    TRAINING_SEED,
    move_model,
)
from main import TRAINING, economic_params
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


def _lin(lo, hi, n):
    if n == 1:
        return [float(lo)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _pack(expert_gammas, household_gammas):
    assert len(expert_gammas) == N_E and len(household_gammas) == N_H
    return list(expert_gammas) + list(household_gammas)


# ---------------------------------------------------------------------------
# Suggested gamma packs (36 experts + 4 households)
# ---------------------------------------------------------------------------
# baseline_hard     : current agents40 HARD (reference)
# match20_band      : same expert band as agents20 [3,10]
# narrow_hard       : experts [3,8] -- less high-gamma risk capacity
# higher_floor      : experts [4,11] -- higher harmonic-mean gamma
# ---------------------------------------------------------------------------
GAMMA_EXPERIMENTS = {
    "baseline_hard": _pack(_lin(3.0, 10.5, N_E), _lin(12.0, 16.5, N_H)),
    "match20_band": _pack(_lin(3.0, 10.0, N_E), _lin(12.0, 16.5, N_H)),
    "narrow_hard": _pack(_lin(3.0, 8.0, N_E), _lin(12.0, 16.5, N_H)),
    "higher_floor": _pack(_lin(4.0, 11.0, N_E), _lin(13.0, 17.0, N_H)),
    "wider_band": _pack(_lin(3.0, 15.0, N_E), _lin(16.0, 20.0, N_H)),
}

ACTIVE_EXPERIMENTS = (
    # "baseline_hard",
    "match20_band",
    "narrow_hard",
    "higher_floor",
    "wider_band",
)


def run_name(experiment):
    return f"agents40_gamma_{experiment}"


def train_and_simulate(experiment, gamma_vec, tau=BASE_TAU, v_mean=BASE_V_MEAN):
    root = os.path.join(MODEL_ROOT, run_name(experiment))
    path = os.path.join(root, CONFIG)
    os.makedirs(path, exist_ok=True)

    metadata = {
        "case": "agents40_gamma_explore",
        "experiment": experiment,
        "K": K,
        "expert_idx": EXPERT_IDX,
        "household_idx": HOUSEHOLD_IDX,
        "gamma": gamma_vec,
        "parameters": economic_params(tau, v_mean),
        "configs": [CONFIG],
        "training_seed": TRAINING_SEED,
        "training": TRAINING,
    }
    with open(os.path.join(root, "experiment.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    ts, rar, lb = CONFIGS[CONFIG]
    init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    init_guess["r"] = 0.01
    params = economic_params(tau, v_mean)

    print(f"\n{(' ' + run_name(experiment) + ' / ' + CONFIG + ' '):=^80}")
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

    print(
        f"[gamma explore] {experiment}: "
        f"agg_rp={summary.get('risk_premium_aggregate', float('nan')):.4f}  "
        f"X_E={summary.get('expert_wealth_share_X_E', float('nan')):.4f}"
    )

    move_model(model, "cpu")
    del model, economy, sim, irf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    torch.set_default_dtype(torch.float64)
    for experiment in ACTIVE_EXPERIMENTS:
        train_and_simulate(experiment, GAMMA_EXPERIMENTS[experiment])


if __name__ == "__main__":
    main()
