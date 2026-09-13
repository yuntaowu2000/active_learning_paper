"""Reproduce the full calibrated 20/40-agent parameter grid.

The grid contains six case calibrations, three retirement rates, and three
long-run volatility means (54 runs).  Existing checkpoints are loaded and never
overwritten; simulations and the final LaTeX table are regenerated every run.
"""

import gc
import os
from itertools import product

import torch

from common import (
    CALIBRATED_CASES,
    CALIBRATED_TAUS,
    CALIBRATED_V_MEANS,
    MODEL_ROOT,
    SIMULATION_SEED,
    move_model,
)
from generate_gini_parameter_table import (
    collect_simulation_row,
    render_latex,
)
from main import train_case
from simulate import run_simulation


TABLE_PATH = os.path.join(MODEL_ROOT, "calibrated_parameter_outcomes.tex")


def main():
    torch.set_default_dtype(torch.float64)
    rows = []
    total = (
        len(CALIBRATED_CASES)
        * len(CALIBRATED_TAUS)
        * len(CALIBRATED_V_MEANS)
    )
    run_number = 0

    for case, tau, v_mean in product(CALIBRATED_CASES, CALIBRATED_TAUS, CALIBRATED_V_MEANS):
        run_number += 1
        print(
            f"\n{'=' * 80}\n"
            f"Calibrated run {run_number}/{total}: "
            f"{case}, tau={tau}, v_mean={v_mean}\n"
            f"{'=' * 80}"
        )
        models, _, _ = train_case(
            case,
            tau=tau,
            v_mean=v_mean,
            configs=("timestep_rar",),
        )
        del models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = run_simulation(
            case,
            tau=tau,
            vmean=v_mean,
            config="timestep_rar",
            seed=SIMULATION_SEED,
            include_irf=True,
        )
        rows.append(
            collect_simulation_row(
                case,
                tau,
                v_mean,
                result["economy"],
                result["simulation"],
                SIMULATION_SEED,
            )
        )
        move_model(result["model"], "cpu")
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latex = render_latex(
        rows,
        (
            "Wealth distribution, portfolios, and risk prices across "
            "calibrated 20- and 40-agent economies."
        ),
        "tab:calibrated_parameter_outcomes",
    )
    with open(TABLE_PATH, "w", encoding="utf-8") as file:
        file.write(latex)
    print(f"\nCalibrated table written to {TABLE_PATH}")


if __name__ == "__main__":
    main()
