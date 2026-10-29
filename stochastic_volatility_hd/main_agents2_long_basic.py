"""2-agent comparison with a Basic model trained for as many epochs as the
whole time-stepping march.

The existing checkpoints under ``models/agents2`` are loaded (never retrained).
Only ``basic_long`` -- the Basic configuration with ``n_epochs`` equal to the
total inner epochs of the 100 time-stepping outer iterations -- is trained, in
``models/agents2/basic_long``.  Tables and figures go to
``models/agents2/comparison_basic_long`` so the paper comparison is untouched.
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
    plot_loss_decay,
    plot_slice_comparison,
)
from common import (
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
    df_to_latex,
    make_case,
    move_model,
)
from main import TRAINING, economic_params
from model import get_model


LONG_BASIC = "basic_long"
COMPARISON_DIR = "comparison_basic_long"
# Methods drawn in the slice and loss-decay figures; tables use every model.
PLOT_METHODS = ("basic", LONG_BASIC, "timestep")
CHUNK_SIZE = 2_000


def timestep_total_epochs(training=TRAINING):
    """Inner epochs summed over all outer iterations, using the library's
    schedule max(int(num_inner / sqrt(i + 1)), min_inner)."""
    return sum(
        max(int(training["num_inner"] / np.sqrt(i + 1)), training["min_inner"])
        for i in range(training["outer"])
    )


def _check_metadata(root, gamma_vec, params):
    path = os.path.join(root, "experiment.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as file:
        metadata = json.load(file)
    trained = metadata["parameters"]
    mismatches = {
        key: (trained.get(key), params[key])
        for key in ("a", "sigma", "tau", "v_mean", "rho", "delta")
        if not np.isclose(trained.get(key, np.nan), params[key])
    }
    if not np.allclose(metadata.get("gamma", []), gamma_vec):
        mismatches["gamma"] = (metadata.get("gamma"), gamma_vec)
    if mismatches:
        raise ValueError(f"parameter mismatch with {path}: {mismatches}")


def _get_model(path, case, gamma, params, config, n_epochs, train):
    K, expert_idx, household_idx, gamma_vec = make_case(case, gamma)
    ts, rar, lb = CONFIGS["basic" if config == LONG_BASIC else config]
    init_guess = {f"xi_{k}": params["rho"] for k in range(1, K + 1)}
    init_guess["r"] = 0.01
    return get_model(
        path,
        K,
        expert_idx,
        household_idx,
        gamma_vec,
        model_size=[TRAINING["width"]] * TRAINING["layers"],
        n_epochs=n_epochs,
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
        train=train,
    )


def _write_tables(models, out_dir):
    loss = compare_loss_table(models, baseline_key=LONG_BASIC, chunk_size=CHUNK_SIZE)
    loss.to_csv(os.path.join(out_dir, "comparative_losses.csv"))
    df_to_latex(loss, os.path.join(out_dir, "comparative_losses.tex"))

    random_t = compare_loss_table(
        models,
        baseline_key=LONG_BASIC,
        chunk_size=CHUNK_SIZE,
        compute_fn=compute_validation_losses_random_t,
    )
    random_t.to_csv(os.path.join(out_dir, "comparative_losses_random_t.csv"))
    df_to_latex(random_t, os.path.join(out_dir, "comparative_losses_random_t.tex"))

    welfare = compute_welfare_equivalent_losses(
        models, baseline_key=LONG_BASIC, chunk_size=CHUNK_SIZE
    )
    welfare.to_csv(os.path.join(out_dir, "welfare_equivalent_losses.csv"))
    df_to_latex(welfare, os.path.join(out_dir, "welfare_equivalent_losses.tex"))
    print(loss)


def run(case_dir, gamma, params, fd_path, case="agents2"):
    """Load every trained config under ``models/<case_dir>``, train
    ``basic_long`` if missing, and write the comparison artifacts."""
    torch.set_default_dtype(torch.float64)
    root = os.path.join(MODEL_ROOT, case_dir)
    _, _, _, gamma_vec = make_case(case, gamma)
    _check_metadata(root, gamma_vec, params)
    epochs = timestep_total_epochs()
    print(f"[{case_dir}] basic_long epochs = {epochs} "
          f"(time-stepping total over {TRAINING['outer']} outer iterations)")

    names = ["basic", LONG_BASIC] + [n for n in CONFIGS if n != "basic"]
    models, paths = {}, {}
    for name in names:
        path = os.path.join(root, name)
        train = name == LONG_BASIC
        if not train and not os.path.exists(os.path.join(path, "model_best.pt")):
            print(f"[{case_dir}] skip {name}: no model_best.pt")
            continue
        if train:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "training.json"), "w", encoding="utf-8") as file:
                json.dump({"config": "basic", "epochs": epochs,
                           "matched_to": "timestep", "training": TRAINING,
                           "parameters": params, "gamma": gamma_vec}, file, indent=2)
        print(f"\n{(' ' + case_dir + ' / ' + name + ' '):=^80}")
        model = _get_model(path, case, gamma, params, name, epochs, train)
        move_model(model, "cpu")
        models[name] = model
        paths[name] = path
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = os.path.join(root, COMPARISON_DIR)
    os.makedirs(out_dir, exist_ok=True)
    _write_tables(models, out_dir)

    v_list = [BASE_V_MEAN]
    slices = {name: evaluate_slices(model, v_list) for name, model in models.items()}
    fd = np.load(fd_path) if os.path.exists(fd_path) else None
    if fd is None:
        print(f"[{case_dir}] no FD solution at {fd_path}; skipping FD comparison")
    else:
        compare_fd_table(slices, fd, v_list, out_dir)
    plot_slice_comparison(
        {name: slices[name] for name in PLOT_METHODS if name in slices},
        fd, v_list, out_dir,
    )

    decay = [name for name in PLOT_METHODS if name in paths]
    plot_loss_decay(
        {name: paths[name] for name in decay},
        out_dir,
        {name: CONFIGS["basic" if name == LONG_BASIC else name][0] for name in decay},
    )
    print(f"[{case_dir}] comparison written to {out_dir}")


if __name__ == "__main__":
    run(
        "agents2",
        PAPER_GAMMA,
        economic_params("agents2"),
        os.path.join(
            MODEL_ROOT,
            "numerical",
            f"numerical_{PAPER_GAMMA}_{BASE_TAU}_{PAPER_SIGMA}_{PAPER_A}.npz",
        ),
    )
