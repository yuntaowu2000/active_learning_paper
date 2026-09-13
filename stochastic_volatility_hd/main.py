"""Reproduce the baseline stochastic-volatility paper results.

One invocation trains (or loads) agents2, agents20, and agents40, writes the
method-comparison plots/tables, and simulates the timestep_rar checkpoint.
Edit the small constants below only when intentionally changing the paper
calibration.
"""

import gc
import json
import os

import numpy as np
import torch

from analysis import (
    compare_fd_table,
    compare_loss_table,
    compute_validation_losses_random_t,
    compute_welfare_equivalent_losses,
    evaluate_slices,
    plot_aggregate_scatter,
    plot_loss_decay,
    plot_loss_weights,
    plot_rar_anchors,
    plot_slice_comparison,
    plot_theta_chat_histogram,
    select_plot_methods,
)
from common import (
    BASE_CASES,
    BASE_PARAMS,
    BASE_TAU,
    BASE_V_MEAN,
    CONFIGS,
    MODEL_ROOT,
    PAPER_A,
    PAPER_GAMMA,
    PAPER_SIGMA,
    SHARE_ALPHA_HI,
    SHARE_ALPHA_LO,
    TRAINING_SEED,
    configs_for_case,
    df_to_latex,
    make_case,
    move_model,
    run_dir,
)
from model import get_model
from simulate import run_simulation


TRAINING = {
    "epochs": 50_000,
    "outer": 100,
    "batch": 500,
    "layers": 4,
    "width": 64,
    "lr": 1e-3,
    "num_inner": 5_000,
    "min_inner": 2_000,
    "lr_decay_every": 20,
    "lr_decay_gamma": 0.5,
    "loss_balancing_alpha": 0.999,
    "loss_balancing_temp": 0.1,
    "bernoulli_prob": 0.9999,
    "t0_frac": 0.4,
}


def economic_params(tau=BASE_TAU, v_mean=BASE_V_MEAN):
    return BASE_PARAMS | {
        "a": PAPER_A,
        "sigma": PAPER_SIGMA,
        "tau": tau,
        "v_mean": v_mean,
    }


def train_case(case, tau=BASE_TAU, v_mean=BASE_V_MEAN, configs=None):
    """Train missing checkpoints for one case and return CPU-resident models."""
    configs = list(configs or configs_for_case(case))
    root = run_dir(case, tau, v_mean, MODEL_ROOT)
    K, expert_idx, household_idx, gamma_vec = make_case(case, PAPER_GAMMA)
    init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    init_guess["r"] = 0.01

    os.makedirs(root, exist_ok=True)
    metadata = {
        "case": case,
        "K": K,
        "expert_idx": expert_idx,
        "household_idx": household_idx,
        "gamma": gamma_vec,
        "parameters": economic_params(tau, v_mean),
        "configs": configs,
        "training_seed": TRAINING_SEED,
        "training": TRAINING,
    }
    with open(os.path.join(root, "experiment.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    models, paths, timestepping = {}, {}, {}
    for name in configs:
        ts, rar, lb = CONFIGS[name]
        path = os.path.join(root, name)
        print(f"\n{(' ' + case + ' / ' + name + ' '):=^80}")
        model = get_model(
            path,
            K,
            expert_idx,
            household_idx,
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
            params=economic_params(tau, v_mean),
            share_alpha_lo=SHARE_ALPHA_LO,
            share_alpha_hi=SHARE_ALPHA_HI,
            seed=TRAINING_SEED,
        )
        move_model(model, "cpu")
        models[name] = model
        paths[name] = path
        timestepping[name] = ts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return models, paths, timestepping


def _write_method_tables(models, comparison_dir, baseline, chunk_size):
    loss = compare_loss_table(
        models, baseline_key=baseline, chunk_size=chunk_size
    )
    suffix = "" if baseline == "basic" else "_timestep"
    loss.to_csv(
        os.path.join(comparison_dir, f"comparative_losses{suffix}.csv")
    )
    df_to_latex(
        loss, os.path.join(comparison_dir, f"comparative_losses{suffix}.tex")
    )

    random_t = compare_loss_table(
        models,
        baseline_key=baseline,
        chunk_size=chunk_size,
        compute_fn=compute_validation_losses_random_t,
    )
    random_t.to_csv(os.path.join(
        comparison_dir, f"comparative_losses{suffix}_random_t.csv"
    ))
    df_to_latex(random_t, os.path.join(
        comparison_dir, f"comparative_losses{suffix}_random_t.tex"
    ))

    welfare = compute_welfare_equivalent_losses(
        models, baseline_key=baseline, chunk_size=chunk_size
    )
    welfare.to_csv(os.path.join(
        comparison_dir, f"welfare_equivalent_losses{suffix}.csv"
    ))
    df_to_latex(welfare, os.path.join(
        comparison_dir, f"welfare_equivalent_losses{suffix}.tex"
    ))
    return loss, welfare


def create_case_artifacts(case, models, paths, timestepping):
    """Write all method-comparison tables and one-panel paper figures."""
    K, _, _, _ = make_case(case, PAPER_GAMMA)
    comparison_dir = os.path.join(run_dir(case), "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    chunk_size = 500 if K >= 10 else 2_000

    loss_df = welfare_df = None
    if "basic" in models:
        loss_df, welfare_df = _write_method_tables(
            models, comparison_dir, "basic", chunk_size
        )
    if "timestep" in models:
        _write_method_tables(
            models, comparison_dir, "timestep", chunk_size
        )

    plot_models = select_plot_methods(
        models, loss_df, welfare_df, baseline_key="basic"
    )
    if case == "agents2":
        fd_path = os.path.join(
            MODEL_ROOT,
            "numerical",
            f"numerical_{PAPER_GAMMA}_{BASE_TAU}_{PAPER_SIGMA}_{PAPER_A}.npz",
        )
        fd = np.load(fd_path) if os.path.exists(fd_path) else None
        v_list = [BASE_V_MEAN]
        slices = {
            name: evaluate_slices(model, v_list)
            for name, model in plot_models.items()
        }
        plot_slice_comparison(slices, fd, v_list, comparison_dir)
        if fd is not None:
            all_slices = {
                name: evaluate_slices(model, v_list)
                for name, model in models.items()
            }
            compare_fd_table(all_slices, fd, v_list, comparison_dir)
    else:
        for name, model in models.items():
            plot_theta_chat_histogram(
                model, comparison_dir, file_suffix=name,
                chunk_size=chunk_size,
            )
            plot_aggregate_scatter(
                model,
                comparison_dir,
                file_prefix=f"aggregate_scatter_{name}",
                chunk_size=chunk_size,
            )

    for name in models:
        ts, rar, lb = CONFIGS[name]
        if rar:
            plot_rar_anchors(
                paths[name], K, comparison_dir,
                file_name=f"rar_anchors_{name}.pdf",
                timestepping=ts,
            )
        if lb:
            plot_loss_weights(
                paths[name], comparison_dir,
                file_name=f"loss_weight_{name}.pdf",
                timestepping=ts,
            )
    decay_names = [
        name for name in ("basic", "timestep", "timestep_rar")
        if name in paths
    ]
    plot_loss_decay(
        {name: paths[name] for name in decay_names},
        comparison_dir,
        {name: timestepping[name] for name in decay_names},
    )


def main():
    torch.set_default_dtype(torch.float64)
    for case in BASE_CASES:
        models, paths, timestepping = train_case(case)
        create_case_artifacts(case, models, paths, timestepping)
        run = run_simulation(case)
        del run, models, paths, timestepping
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
