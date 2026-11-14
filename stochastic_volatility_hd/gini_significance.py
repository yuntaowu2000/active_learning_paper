"""Paired-path significance test for the clearest agents20 Gini contrast.

The default comparison is

    low-Gini economy:   tau=0.20, v_mean=0.70
    high-Gini economy:  tau=1.15, v_mean=0.50

Both economies are simulated with the same seed, so path p in one economy uses
the same Gaussian innovations as path p in the other.  Inference is performed
on post-burn-in *path-level mean Ginis*, not on autocorrelated agent-time rows.

This measures simulation uncertainty conditional on the two trained
checkpoints.  It does not include neural-network training uncertainty; repeat
training with multiple training seeds to measure that component.
"""

import gc
import json
import math
import os

import numpy as np
import torch

from common import (BASE_TAU, MODEL_ROOT, PAPER_A, PAPER_GAMMA, PAPER_SIGMA,
                    SIMULATION_SEED, move_model)
from simulate import NNEconomy, _gini_rows, load_model, simulate

CASE = "agents20"
CONFIG = "timestep_rar"
HIGH_TAU, HIGH_V_MEAN = BASE_TAU, 0.5
LOW_TAU, LOW_V_MEAN = 0.2, 0.7
PATHS = 100
YEARS = 500.0
DT = 0.08
X0 = 0.2
V0 = None
BURN_IN_FRAC = 0.2
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 12_345
OUTPUT_PREFIX = os.path.join(
    MODEL_ROOT, "gini_significance_tau1.15_v0.5_vs_tau0.2_v0.7"
)


def path_mean_ginis(economy, sim_result, burn_in_frac=0.2):
    """Return one post-burn-in mean cross-sectional Gini per simulated path."""
    x_hist = sim_result["x_hist"]
    burn = int(x_hist.shape[0] * burn_in_frac)
    x_free = x_hist[burn:]                                  # (T, P, K-1)
    x_last = 1.0 - x_free.sum(axis=2, keepdims=True)
    x_full = np.concatenate([x_free, x_last], axis=2)        # (T, P, K)
    n_time, n_paths, n_agents = x_full.shape
    snapshot_gini = _gini_rows(x_full.reshape(-1, n_agents))
    snapshot_gini = snapshot_gini.reshape(n_time, n_paths)
    return np.nanmean(snapshot_gini, axis=0)


def paired_inference(high, low, bootstrap_samples=20_000, seed=12_345):
    """Paired t inference and a path bootstrap CI for mean(high - low)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    valid = np.isfinite(high) & np.isfinite(low)
    diff = high[valid] - low[valid]
    n = diff.size
    if n < 2:
        raise ValueError("at least two finite paired paths are required")

    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se_diff = sd_diff / math.sqrt(n)
    t_stat = mean_diff / se_diff if se_diff > 0 else math.copysign(math.inf, mean_diff)

    try:
        from scipy.stats import t as student_t

        p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
        critical = float(student_t.ppf(0.975, df=n - 1))
    except ImportError:
        # Large-sample normal approximation if SciPy is unavailable.
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
        critical = 1.959963984540054

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    bootstrap_means = diff[indices].mean(axis=1)
    boot_lo, boot_hi = np.percentile(bootstrap_means, [2.5, 97.5])

    return {
        "n_paired_paths": int(n),
        "mean_gini_high": float(high[valid].mean()),
        "mean_gini_low": float(low[valid].mean()),
        "mean_difference_high_minus_low": mean_diff,
        "sd_paired_difference": sd_diff,
        "se_paired_difference": float(se_diff),
        "paired_t_statistic": float(t_stat),
        "paired_t_df": int(n - 1),
        "paired_t_p_value_two_sided": p_value,
        "paired_t_ci95": [
            float(mean_diff - critical * se_diff),
            float(mean_diff + critical * se_diff),
        ],
        "paired_path_bootstrap_ci95": [float(boot_lo), float(boot_hi)],
        "cohen_dz": float(mean_diff / sd_diff) if sd_diff > 0 else math.inf,
        "bootstrap_samples": int(bootstrap_samples),
    }


def load_economy(base_dir, case, config, tau, vmean, a, sigma, gamma):
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
    return model, NNEconomy(model)


def simulate_path_ginis(
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
):
    model, economy = load_economy(
        base_dir, case, config, tau, vmean, a, sigma, gamma
    )
    initial_v = vmean if v0 is None else v0
    result = simulate(
        economy,
        n_paths=paths,
        years=years,
        dt=dt,
        x0=x0,
        v0=initial_v,
        seed=sim_seed,
    )
    ginis = path_mean_ginis(economy, result, burn_in_frac)
    move_model(model, "cpu")
    del result, economy, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ginis


def write_report(output_prefix, report, high_path_ginis, low_path_ginis):
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    with open(output_prefix + ".json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    np.savez(
        output_prefix + "_path_ginis.npz",
        high=high_path_ginis,
        low=low_path_ginis,
        difference=high_path_ginis - low_path_ginis,
    )

    lines = [
        "Paired-path Gini significance test",
        "==================================",
        f"high economy: tau={report['high_parameters']['tau']}, "
        f"v_mean={report['high_parameters']['vmean']}",
        f"low economy: tau={report['low_parameters']['tau']}, "
        f"v_mean={report['low_parameters']['vmean']}",
        f"paired paths: {report['inference']['n_paired_paths']}",
        f"mean Gini (high): {report['inference']['mean_gini_high']:.8f}",
        f"mean Gini (low): {report['inference']['mean_gini_low']:.8f}",
        "mean difference (high-low): "
        f"{report['inference']['mean_difference_high_minus_low']:.8f}",
        f"paired t statistic: {report['inference']['paired_t_statistic']:.4f}",
        f"paired t p-value (two-sided): "
        f"{report['inference']['paired_t_p_value_two_sided']:.6g}",
        f"paired t 95% CI: {report['inference']['paired_t_ci95']}",
        "paired-path bootstrap 95% CI: "
        f"{report['inference']['paired_path_bootstrap_ci95']}",
        f"Cohen dz: {report['inference']['cohen_dz']:.4f}",
        "",
        "Interpretation: inference is conditional on the trained checkpoints.",
        "It does not include neural-network training uncertainty.",
    ]
    with open(output_prefix + ".txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    torch.set_default_dtype(torch.float64)

    common = dict(
        base_dir=MODEL_ROOT,
        case=CASE,
        config=CONFIG,
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
    )
    high = simulate_path_ginis(
        tau=HIGH_TAU, vmean=HIGH_V_MEAN, **common
    )
    low = simulate_path_ginis(
        tau=LOW_TAU, vmean=LOW_V_MEAN, **common
    )
    inference = paired_inference(
        high,
        low,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    report = {
        "high_parameters": {"tau": HIGH_TAU, "vmean": HIGH_V_MEAN},
        "low_parameters": {"tau": LOW_TAU, "vmean": LOW_V_MEAN},
        "shared_parameters": {
            "case": CASE,
            "config": CONFIG,
            "a": PAPER_A,
            "sigma": PAPER_SIGMA,
            "gamma_case_note": (
                "PAPER_GAMMA is ignored by make_case for agents20; retained "
                "for loader compatibility"
            ),
            "paths": PATHS,
            "years": YEARS,
            "dt": DT,
            "x0": X0,
            "v0": V0,
            "burn_in_frac": BURN_IN_FRAC,
            "sim_seed": SIMULATION_SEED,
        },
        "inference": inference,
        "scope": (
            "Simulation uncertainty conditional on trained checkpoints; "
            "training uncertainty excluded."
        ),
    }
    write_report(OUTPUT_PREFIX, report, high, low)
    print(json.dumps(report, indent=2))
    print(f"\nSaved {OUTPUT_PREFIX}.txt/.json and path-level .npz")


if __name__ == "__main__":
    main()
