"""Generate a LaTeX comparative-statics table from fresh simulations.

Each parameter set loads one checkpoint and is simulated exactly once.  Gini,
wealth percentiles, and theta buckets are computed from that simulation.
Prices, premia, and the risk-free rate are evaluated at the same simulation's
post-burn-in ergodic mean state.

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

from common import (CALIBRATED_CASES, CALIBRATED_TAUS, CALIBRATED_V_MEANS,
                    MODEL_ROOT, PAPER_A, PAPER_GAMMA, PAPER_SIGMA,
                    SIMULATION_SEED, move_model)
from simulate import NNEconomy, _gini_rows, _project, load_model, simulate


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
OUTPUT_PATH = os.path.join(MODEL_ROOT, "calibrated_parameter_outcomes.tex")
TABLE_CAPTION = (
    "Wealth distribution, portfolios, and risk prices across calibrated "
    "20- and 40-agent economies."
)
TABLE_LABEL = "tab:calibrated_parameter_outcomes"


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

    x_states = x_free.reshape(-1, n_agents - 1)
    v_states = v_post.reshape(-1)
    x_mean = x_states.mean(axis=0, keepdims=True)
    mean_v = np.asarray([v_states.mean()])
    x_mean, mean_v = _project(economy, x_mean, mean_v)
    values = economy._forward(x_mean, mean_v)
    x_full_mean = values["x_full"].detach().cpu().numpy()[0]
    theta_mean = values["theta_full"].detach().cpu().numpy()[0]

    # Evaluate theta over a reproducible subsample of the same simulation.
    n_states = x_states.shape[0]
    if n_states > max_theta_states:
        rng = np.random.default_rng(sim_seed)
        selected = rng.choice(
            n_states, size=max_theta_states, replace=False
        )
        theta_x = x_states[selected]
        theta_v = v_states[selected]
    else:
        theta_x = x_states
        theta_v = v_states
    simulated_wealth = []
    simulated_theta = []
    for start in range(0, theta_x.shape[0], theta_chunk):
        stop = start + theta_chunk
        wealth_chunk, theta_values = economy.portfolio(
            theta_x[start:stop], theta_v[start:stop]
        )
        simulated_wealth.append(wealth_chunk)
        simulated_theta.append(theta_values)
    wealth_for_theta = np.concatenate(simulated_wealth, axis=0)
    theta_for_buckets = np.concatenate(simulated_theta, axis=0)

    expert_idx = np.asarray(economy.expert_idx, dtype=int)
    household_idx = np.asarray(economy.household_idx, dtype=int)
    expert_mean_wealth = wealth_for_theta[:, expert_idx].mean(axis=0)
    expert_order = expert_idx[np.argsort(expert_mean_wealth)]
    split = len(expert_order) // 2
    poor_experts = expert_order[:split]
    rich_experts = expert_order[split:]

    def group_mean(array, indices):
        return float(np.mean(array[:, indices])) if len(indices) else np.nan

    anchor = expert_idx[0]
    pi = float(values["pi"].detach().cpu().numpy().reshape(-1)[0])
    sig_agg = float(
        values["sig_agg"].detach().cpu().numpy().reshape(-1)[0]
    )
    chi = float(values["chi"].detach().cpu().numpy().reshape(-1)[0])
    sigtilde = values["sigtilde_full"].detach().cpu().numpy()[0]
    gamma_vec = economy.gamma_vec
    idio_price = float(gamma_vec[anchor] * sigtilde[anchor])
    idio_premium = float(
        chi * theta_mean[anchor] / x_full_mean[anchor]
    )

    row = {
        "case": case,
        "tau": tau,
        "vmean": vmean,
        "gini": float(np.nanmean(snapshot_gini)),
        "theta_households": group_mean(
            theta_for_buckets, household_idx
        ),
        "theta_poor_experts": group_mean(
            theta_for_buckets, poor_experts
        ),
        "theta_rich_experts": group_mean(
            theta_for_buckets, rich_experts
        ),
        "risk_free_rate": float(
            values["r"].detach().cpu().numpy().reshape(-1)[0]
        ),
        "aggregate_risk_premium": pi * sig_agg,
        "idiosyncratic_risk_premium": idio_premium,
        "aggregate_price_of_risk": pi,
        "idiosyncratic_price_of_risk": idio_price,
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


def render_latex(rows, caption, label):
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
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
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
            format_number(row["gini"]),
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
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{\textwidth}",
            r"\vspace{2pt}\footnotesize",
            (
                r"\textit{Notes:} Gini and wealth-share percentiles are computed "
                r"from post-burn-in simulation snapshots. All remaining outcomes "
                r"except $\bar\theta$ are evaluated at the same simulation's "
                r"ergodic mean state. "
                r"$\bar\theta_H$, $\bar\theta_{E^-}$, and "
                r"$\bar\theta_{E^+}$ are simulated mean per-agent capital shares "
                r"for households, poorer experts, and richer experts. "
                r"$\lambda^{agg}=\pi$ and "
                r"$\lambda^{idio}_0=\gamma_0\widetilde{\sigma}_0$; "
                r"idiosyncratic quantities refer to the anchor expert."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    """Regenerate the fixed calibrated-grid table from fresh simulations."""
    torch.set_default_dtype(torch.float64)
    rows = []
    for case, tau, vmean in product(CALIBRATED_CASES, CALIBRATED_TAUS, CALIBRATED_V_MEANS):
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

    latex = render_latex(rows, TABLE_CAPTION, TABLE_LABEL)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(latex)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
