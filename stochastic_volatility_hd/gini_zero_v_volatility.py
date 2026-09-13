"""Compare agents20 Gini with stochastic v versus zero volatility of v.

The counterfactual sets ``model.statics["sigv_mean"] = 0`` before evaluating
the equilibrium equations.  Thus sigma_v and every wealth drift/diffusion term
that depends on sigma_v are recomputed with zero volatility of v.  By default
each economy starts at its own v_mean, so mu_v is also zero and v is constant.

The neural value/price functions remain those of the original checkpoint.  This
is therefore a fixed-function channel counterfactual, not a newly solved
sigv_mean=0 equilibrium.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

from common import move_model
from gini_significance import paired_inference, path_mean_ginis
from simulate import NNEconomy, load_model, simulate


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
    parser = argparse.ArgumentParser(
        description="Compare Gini under stochastic v and sigma_v=0."
    )
    parser.add_argument("--base-dir", default="./models")
    parser.add_argument("--case", default="agents20")
    parser.add_argument("--config", default="timestep_rar")
    parser.add_argument(
        "--economies",
        default="0.2:0.7,1.15:0.4",
        help="comma-separated tau:vmean pairs",
    )
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--years", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--x0", type=float, default=0.2)
    parser.add_argument(
        "--v0",
        type=float,
        default=None,
        help="initial v; default v_mean makes v constant when sigma_v=0",
    )
    parser.add_argument("--burn-in-frac", type=float, default=0.2)
    parser.add_argument("--sim-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=54_321)
    parser.add_argument(
        "--output-name",
        default="gini_zero_v_volatility",
        help="file basename within each model's config/simulation directory",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    specifications = []
    for item in args.economies.split(","):
        tau_text, vmean_text = item.split(":")
        specifications.append((float(tau_text), float(vmean_text)))

    for index, (tau, vmean) in enumerate(specifications):
        result, baseline_gini, zero_gini = compare_one_economy(
            base_dir=args.base_dir,
            case=args.case,
            config=args.config,
            tau=tau,
            vmean=vmean,
            a=args.a,
            sigma=args.sigma,
            gamma=args.gamma,
            paths=args.paths,
            years=args.years,
            dt=args.dt,
            x0=args.x0,
            v0=args.v0,
            sim_seed=args.sim_seed,
            burn_in_frac=args.burn_in_frac,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        report = {
            "counterfactual": (
                "Set model statics sigv_mean=0, recompute equilibrium "
                "drift/diffusion terms, and initialize v at v_mean; retain "
                "the checkpoint functions."
            ),
            "shared_parameters": {
                "case": args.case,
                "config": args.config,
                "a": args.a,
                "sigma": args.sigma,
                "paths": args.paths,
                "years": args.years,
                "dt": args.dt,
                "x0": args.x0,
                "v0": args.v0,
                "burn_in_frac": args.burn_in_frac,
                "sim_seed": args.sim_seed,
            },
            "economy": result,
            "scope": (
                "Fixed-function channel counterfactual conditional on the "
                "checkpoint; not an equilibrium retrained with sigv_mean=0."
            ),
        }

        subpath_name = (
            args.case if tau == 1.15 else f"{args.case}_{tau}"
        )
        if vmean != 0.25:
            subpath_name = f"{args.case}_{tau}_{vmean}"
        output_dir = os.path.join(
            args.base_dir, subpath_name, args.config, "simulation"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_prefix = os.path.join(output_dir, args.output_name)

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
