"""Compare agents20 Gini with stochastic v versus zero volatility of v.

The counterfactual sets ``model.statics["sigv_mean"] = 0`` before evaluating
the equilibrium equations.  Thus sigma_v and every wealth drift/diffusion term
that depends on sigma_v are recomputed with zero volatility of v.  By default
each economy starts at its own v_mean, so mu_v is also zero and v is constant.

The neural value/price functions remain those of the original checkpoint.  This
is therefore a fixed-function channel counterfactual, not a newly solved
sigv_mean=0 equilibrium.
"""

import gc
import json
import os

import numpy as np
import torch

from common import (BASE_TAU, MODEL_ROOT, PAPER_A, PAPER_GAMMA, PAPER_SIGMA,
                    SIMULATION_SEED, move_model, simulation_dir)
from gini_significance import paired_inference, path_mean_ginis
from simulate import NNEconomy, load_model, simulate

CASE = "agents20"
CONFIG = "timestep_rar"
ECONOMIES = ((0.2, 0.7), (BASE_TAU, 0.5))
PATHS = 100
YEARS = 500.0
DT = 0.08
X0 = 0.2
V0 = None
BURN_IN_FRAC = 0.2
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 54_321
OUTPUT_NAME = "gini_zero_v_volatility"


def compare_one_economy(
    base_dir,
    case,
    config,
    tau,
    vmean,
    a,
    sigma,
    gamma,
    paths,
    years,
    dt,
    x0,
    v0,
    sim_seed,
    burn_in_frac,
    bootstrap_samples,
    bootstrap_seed,
):
    model = load_model(
        base_dir,
        case,
        config=config,
        gamma=gamma,
        tau=tau,
        sigma=sigma,
        a=a,
        vmean=vmean,
    )
    baseline = NNEconomy(model)
    initial_v = vmean if v0 is None else v0

    baseline_result = simulate(
        baseline,
        n_paths=paths,
        years=years,
        dt=dt,
        x0=x0,
        v0=initial_v,
        seed=sim_seed,
    )
    baseline_gini = path_mean_ginis(baseline, baseline_result, burn_in_frac)
    baseline_v_std = float(
        baseline_result["v_hist"][
            int(baseline_result["v_hist"].shape[0] * burn_in_frac):
        ].std()
    )
    del baseline_result

    # Recompute all equilibrium terms that depend on sigma_v with sigma_v=0.
    original_sigv_mean = model.statics["sigv_mean"]
    model.statics["sigv_mean"] = 0.0
    zero_v_vol = NNEconomy(model)
    zero_result = simulate(
        zero_v_vol,
        n_paths=paths,
        years=years,
        dt=dt,
        x0=x0,
        v0=initial_v,
        seed=sim_seed,
    )
    zero_gini = path_mean_ginis(zero_v_vol, zero_result, burn_in_frac)
    zero_v_std = float(
        zero_result["v_hist"][
            int(zero_result["v_hist"].shape[0] * burn_in_frac):
        ].std()
    )
    del zero_result
    model.statics["sigv_mean"] = original_sigv_mean

    # Here "high" means the baseline stochastic-v value, so a positive
    # difference means stochastic v raises Gini relative to constant v.
    inference = paired_inference(
        baseline_gini,
        zero_gini,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    result = {
        "parameters": {"tau": tau, "vmean": vmean},
        "mean_gini_stochastic_v": float(baseline_gini.mean()),
        "mean_gini_zero_v_volatility": float(zero_gini.mean()),
        "mean_difference_stochastic_minus_zero": (
            inference["mean_difference_high_minus_low"]
        ),
        "post_burn_in_v_std_stochastic": baseline_v_std,
        "post_burn_in_v_std_zero_volatility": zero_v_std,
        "paired_inference": inference,
    }

    move_model(model, "cpu")
    del zero_v_vol, baseline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, baseline_gini, zero_gini


def main():
    torch.set_default_dtype(torch.float64)
    for index, (tau, vmean) in enumerate(ECONOMIES):
        result, baseline_gini, zero_gini = compare_one_economy(
            base_dir=MODEL_ROOT,
            case=CASE,
            config=CONFIG,
            tau=tau,
            vmean=vmean,
            a=PAPER_A,
            sigma=PAPER_SIGMA,
            gamma=PAPER_GAMMA,
            paths=PATHS,
            years=YEARS,
            dt=DT,
            x0=X0,
            v0=V0,
            sim_seed=SIMULATION_SEED,
            burn_in_frac=BURN_IN_FRAC,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED + index,
        )
        report = {
            "counterfactual": (
                "Set model statics sigv_mean=0, recompute equilibrium "
                "drift/diffusion terms, and initialize v at v_mean; retain "
                "the checkpoint functions."
            ),
            "shared_parameters": {
                "case": CASE,
                "config": CONFIG,
                "a": PAPER_A,
                "sigma": PAPER_SIGMA,
                "paths": PATHS,
                "years": YEARS,
                "dt": DT,
                "x0": X0,
                "v0": V0,
                "burn_in_frac": BURN_IN_FRAC,
                "sim_seed": SIMULATION_SEED,
            },
            "economy": result,
            "scope": (
                "Fixed-function channel counterfactual conditional on the "
                "checkpoint; not an equilibrium retrained with sigv_mean=0."
            ),
        }

        output_dir = simulation_dir(
            CASE, CONFIG, tau, vmean, MODEL_ROOT
        )
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, OUTPUT_NAME)

        with open(output_prefix + ".json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        np.savez(
            output_prefix + "_path_ginis.npz",
            stochastic=baseline_gini,
            zero_v_volatility=zero_gini,
            difference=baseline_gini - zero_gini,
        )

        params = result["parameters"]
        inference = result["paired_inference"]
        lines = [
            "Gini with volatility of v shut down",
            "======================================",
            report["counterfactual"],
            "",
            f"tau={params['tau']}, v_mean={params['vmean']}",
            f"stochastic-v Gini: {result['mean_gini_stochastic_v']:.8f}",
            "zero-v-volatility Gini: "
            f"{result['mean_gini_zero_v_volatility']:.8f}",
            "difference (stochastic-zero): "
            f"{result['mean_difference_stochastic_minus_zero']:.8f}",
            "paired t p-value: "
            f"{inference['paired_t_p_value_two_sided']:.6g}",
            "paired-path bootstrap 95% CI: "
            f"{inference['paired_path_bootstrap_ci95']}",
            "",
            "Inference is conditional on the original trained checkpoints.",
            "A structural sigv_mean=0 experiment requires retraining the PDE.",
        ]
        with open(output_prefix + ".txt", "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")

        print(json.dumps(report, indent=2))
        print(f"\nSaved {output_prefix}.txt/.json and path-level .npz")


if __name__ == "__main__":
    main()
