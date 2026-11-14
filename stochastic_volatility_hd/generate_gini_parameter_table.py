"""Generate a LaTeX comparative-statics table from fresh simulations.

Each parameter set loads one checkpoint and is simulated exactly once.  Gini,
wealth percentiles, and theta buckets are computed from that simulation.
Prices, premia, and the risk-free rate are evaluated at every post-burn-in
simulated state (subsampled) and averaged, like the Gini.

The two reported prices of risk are:

* aggregate: pi;
* idiosyncratic: gamma_0 * sigmatilde_0 = chi / (phi * v), for the anchor expert.

The three theta columns are simulation averages of per-agent capital shares
for households, the poorer half of experts, and the richer half of experts.
The expert split uses agents' simulated post-burn-in mean wealth shares.
"""

import gc
import os
from itertools import product

import numpy as np
import torch

from common import (CALIBRATED_CASES_20, CALIBRATED_CASES_40, CALIBRATED_TAUS, CALIBRATED_V_MEANS,
                    MODEL_ROOT, PAPER_A, PAPER_GAMMA, PAPER_SIGMA,
                    SIMULATION_SEED, move_model)
from simulate import (NNEconomy, _gini_rows, equilibrium_series,
                      ergodic_states, load_model, simulate)


PERCENTILES = (1, 5, 50, 75, 90, 95, 99)

# Fixed paper-table settings; these match main_calibrated.py and simulate.py.
CONFIG = "timestep_rar"
PATHS = 100
YEARS = 500.0
DT = 0.08
X0 = 0.2
V0 = 0.1
BURN_IN_FRAC = 0.2
MAX_THETA_STATES = 20_000
THETA_CHUNK = 2_000


def collect_simulation_row(
    case,
    tau,
    vmean,
    economy,
    simulation,
    sim_seed,
    burn_in_frac=0.2,
    max_theta_states=20_000,
    theta_chunk=2_000,
):
    """Calculate one table row from an existing simulation."""
    burn = int(simulation["x_hist"].shape[0] * burn_in_frac)
    x_free = simulation["x_hist"][burn:]                       # (T, P, K-1)
    v_post = simulation["v_hist"][burn:]                        # (T, P)
    x_last = 1.0 - x_free.sum(axis=2, keepdims=True)
    x_full_sim = np.concatenate([x_free, x_last], axis=2)       # (T, P, K)
    n_time, n_paths, n_agents = x_full_sim.shape
    snapshot_gini = _gini_rows(x_full_sim.reshape(-1, n_agents))
    wealth = x_full_sim.reshape(-1)

    # Every equilibrium object is evaluated state by state on a reproducible
    # subsample of the same simulation and then averaged.
    theta_x, theta_v = ergodic_states(
        economy, simulation, burn_in_frac, max_theta_states, sim_seed
    )
    chunk = min(theta_chunk, 500) if economy.K >= 40 else theta_chunk
    series = equilibrium_series(economy, theta_x, theta_v, chunk=chunk)
    wealth_for_theta = series["x_full"]
    theta_for_buckets = series["theta_full"]

    expert_idx = np.asarray(economy.expert_idx, dtype=int)
    household_idx = np.asarray(economy.household_idx, dtype=int)
    expert_mean_wealth = wealth_for_theta[:, expert_idx].mean(axis=0)
    expert_order = expert_idx[np.argsort(expert_mean_wealth)]
    split = len(expert_order) // 2
    poor_experts = expert_order[:split]
    rich_experts = expert_order[split:]

    def group_mean(array, indices):
        return float(np.mean(array[:, indices])) if len(indices) else np.nan

    row = {
        "case": case,
        "tau": tau,
        "vmean": vmean,
        "gini": float(np.nanmean(snapshot_gini)),
        "gini_p5": float(np.nanpercentile(snapshot_gini, 5)),
        "gini_p95": float(np.nanpercentile(snapshot_gini, 95)),
        "theta_households": group_mean(
            theta_for_buckets, household_idx
        ),
        "theta_poor_experts": group_mean(
            theta_for_buckets, poor_experts
        ),
        "theta_rich_experts": group_mean(
            theta_for_buckets, rich_experts
        ),
        "risk_free_rate": float(np.mean(series["r"])),
        "aggregate_risk_premium": float(np.mean(series["agg_rp"])),
        "idiosyncratic_risk_premium": float(np.mean(series["idio_rp"])),
        "aggregate_price_of_risk": float(np.mean(series["pi"])),
        "idiosyncratic_price_of_risk": float(np.mean(series["idio_price"])),
    }
    for percentile in PERCENTILES:
        row[f"p{percentile}"] = float(
            np.percentile(wealth, percentile)
        )

    return row


def collect_row(
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
    max_theta_states,
    theta_chunk,
):
    """Load one checkpoint, simulate once, and return one table row."""
    model = load_model(
        base_dir, case, config=config, gamma=gamma, tau=tau,
        sigma=sigma, a=a, vmean=vmean,
    )
    economy = NNEconomy(model)
    simulation = simulate(
        economy, n_paths=paths, years=years, dt=dt,
        x0=x0, v0=v0, seed=sim_seed,
    )
    row = collect_simulation_row(
        case, tau, vmean, economy, simulation, sim_seed,
        burn_in_frac, max_theta_states, theta_chunk,
    )
    move_model(model, "cpu")
    del simulation, economy, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_number(value, digits=4):
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def format_gini(row):
    return (
        f"{format_number(row['gini'])} [{format_number(row['gini_p5'])}, {format_number(row['gini_p95'])}]"
    )


def render_latex(rows):
    percentile_headers = [rf"$p_{{{p}}}$" for p in PERCENTILES]
    headers = [
        r"$\gamma$ case",
        r"$\tau$",
        r"$\bar v$",
        "Gini",
        r"$\bar\theta_H$",
        r"$\bar\theta_{E^-}$",
        r"$\bar\theta_{E^+}$",
        *percentile_headers,
        r"$r$",
        r"$RP^{agg}$",
        r"$RP^{idio}_0$",
        r"$\lambda^{agg}$",
        r"$\lambda^{idio}_0$",
    ]
    column_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(row["case"]),
            format_number(row["tau"], 2),
            format_number(row["vmean"], 2),
            format_gini(row),
            format_number(row["theta_households"]),
            format_number(row["theta_poor_experts"]),
            format_number(row["theta_rich_experts"]),
            *[
                format_number(row[f"p{percentile}"])
                for percentile in PERCENTILES
            ],
            format_number(row["risk_free_rate"]),
            format_number(row["aggregate_risk_premium"]),
            format_number(row["idiosyncratic_risk_premium"]),
            format_number(row["aggregate_price_of_risk"]),
            format_number(row["idiosyncratic_price_of_risk"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines)


def main(
    cases,
    table_path,
):
    """Regenerate the fixed calibrated-grid table from fresh simulations."""
    torch.set_default_dtype(torch.float64)
    rows = []
    for case, tau, vmean in product(cases, CALIBRATED_TAUS, CALIBRATED_V_MEANS):
        row = collect_row(
            MODEL_ROOT,
            case,
            CONFIG,
            tau,
            vmean,
            PAPER_A,
            PAPER_SIGMA,
            PAPER_GAMMA,
            PATHS,
            YEARS,
            DT,
            X0,
            V0,
            SIMULATION_SEED,
            BURN_IN_FRAC,
            MAX_THETA_STATES,
            THETA_CHUNK,
        )
        rows.append(row)
        print(
            f"[row] {case} tau={tau} v_mean={vmean} "
            f"Gini={row['gini']:.4f}"
        )

    if not rows:
        raise RuntimeError("no requested checkpoints were found")

    latex = render_latex(rows)
    with open(table_path, "w", encoding="utf-8") as file:
        file.write(latex)
    print(f"\nSaved {len(rows)} rows to {table_path}")


if __name__ == "__main__":
    main(CALIBRATED_CASES_20, os.path.join(MODEL_ROOT, "agents20_calibrated_parameter_outcomes.tex"))
    # main(CALIBRATED_CASES_40, os.path.join(MODEL_ROOT, "agents40_calibrated_parameter_outcomes.tex"))
