"""Optimiser / network-capacity ablation for agents40 timestep_rar.

Three variants of the base ``models/agents40/timestep_rar`` run, each written
to its own folder under ``models/agents40_capacity`` (nothing in
``models/agents40`` is modified):

  * ``lbfgs``      -- continue from ``agents40/timestep_rar/model_best.pt`` with
                      LBFGS for ``LBFGS["outer"]`` more time steps;
  * ``deep_8x64``  -- the base run with hidden size [64]*8;
  * ``wide_4x128`` -- the base run with hidden size [128]*4.

Deep / wide differ from the base only in ``model_size``; every other setting
comes from ``main.TRAINING`` / ``main.economic_params``.  Existing checkpoints
are loaded instead of retrained.  The script then writes
``models/agents40_capacity/capacity_comparison.{csv,tex}`` with the validation
loss, VRAM and time per epoch of the base run and the three variants.

Run with ``uv run python main_agents40_capacity.py``.
"""

import gc
import glob
import json
import os
import re
import time
import types

import numpy as np
import pandas as pd
import torch

from analysis import _validation_states, compute_validation_losses
from common import (
    MODEL_ROOT,
    PAPER_GAMMA,
    SHARE_ALPHA_HI,
    SHARE_ALPHA_LO,
    TRAINING_SEED,
    format_sci,
    make_case,
    move_model,
    run_dir,
)
from diagnostics import _count_params, _prepopulate_anchors
from main import TRAINING, economic_params
from model import get_model


CASE = "agents40"
METHOD = "timestep_rar"
BASE_DIR = os.path.join(run_dir(CASE), METHOD)
OUT_ROOT = os.path.join(MODEL_ROOT, f"{CASE}_capacity")
BASE_SIZE = [TRAINING["width"]] * TRAINING["layers"]

# Each LBFGS step runs up to ``max_iter`` closure evaluations, so 100 steps per
# time step cost about as many forward/backward passes as the base run's
# ``min_inner`` = 2000 Adam epochs per time step.
LBFGS = {
    "outer": 2,
    "inner": 100,
    "max_iter": 20,
    "history_size": 50,
    "lr": 1.0,
    "loss_log_interval": 10,
}

VARIANTS = {
    "baseline": {"label": "Time-stepping + RAR (base)", "size": BASE_SIZE,
                 "path": BASE_DIR, "optimizer": "Adam"},
    "lbfgs": {"label": "Base + LBFGS", "size": BASE_SIZE,
              "path": os.path.join(OUT_ROOT, "lbfgs"), "optimizer": "LBFGS"},
    "deep_8x64": {"label": "Deeper [64]x8", "size": [64] * 8,
                  "path": os.path.join(OUT_ROOT, "deep_8x64"), "optimizer": "Adam"},
    "wide_4x128": {"label": "Wider [128]x4", "size": [128] * 4,
                   "path": os.path.join(OUT_ROOT, "wide_4x128"), "optimizer": "Adam"},
}

N_VALIDATION = 10_000
VALIDATION_CHUNK = 500
BENCH_WARMUP = 2
BENCH_STEPS = 5


# ---------------------------------------------------------------------------
# Case / log helpers
# ---------------------------------------------------------------------------
def _case():
    K, expert_idx, household_idx, gamma_vec = make_case(CASE, PAPER_GAMMA)
    init_guess = {f"xi_{k}": economic_params(CASE)["rho"] for k in range(1, K + 1)}
    init_guess["r"] = 0.01
    return K, expert_idx, household_idx, gamma_vec, init_guess


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb():
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def _free(model):
    move_model(model, "cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_summary(path):
    """Config, parameters, completed outer loops, epochs and wall time of a
    time-stepping training log."""
    logs = sorted(glob.glob(os.path.join(path, "model-*-log.txt")))
    if not logs:
        return None
    with open(logs[-1], encoding="utf-8") as file:
        text = file.read()
    config = json.loads(re.search(r"Config:\s*(\{.*?\n\})", text, re.S).group(1))
    params = json.loads(
        re.search(r"User Defined Parameters:\s*(\{.*?\n\})", text, re.S).group(1)
    )
    n_outer = len(re.findall(r"^Outer Loop \d+ Finished", text, re.M))
    inner, min_inner = config["num_inner_iterations"], config["min_inner_iterations"]
    epochs = sum(max(int(inner / np.sqrt(i + 1)), min_inner) for i in range(n_outer))
    total = re.search(r"training finished, total time ::\s*([0-9.eE+-]+)", text)
    return {
        "config": config,
        "params": params,
        "n_outer": n_outer,
        "epochs": int(epochs),
        "train_seconds": float(total.group(1)) if total else float("nan"),
    }


def _check_baseline():
    """Refuse to compare against a base checkpoint trained on another setup."""
    info = _log_summary(BASE_DIR)
    if info is None or not os.path.exists(os.path.join(BASE_DIR, "model_best.pt")):
        raise SystemExit(f"No trained base run found in {BASE_DIR}.")
    expected = economic_params(CASE)
    bad = {
        k: (info["params"][k], v) for k, v in expected.items()
        if k in info["params"] and not np.isclose(info["params"][k], v)
    }
    cfg = info["config"]
    expected_cfg = {
        "num_outer_iterations": TRAINING["outer"],
        "num_inner_iterations": TRAINING["num_inner"],
        "min_inner_iterations": TRAINING["min_inner"],
        "lr": TRAINING["lr"],
        "batch_size": TRAINING["batch"] // 2,
    }
    bad.update({
        k: (cfg.get(k), v) for k, v in expected_cfg.items() if cfg.get(k) != v
    })
    if bad:
        detail = ", ".join(f"{k}: trained {a} vs current {b}" for k, (a, b) in bad.items())
        raise SystemExit(
            f"{BASE_DIR} was trained with a different setup ({detail}). "
            "Retrain the base agents40 timestep_rar run with main.py first."
        )
    return info


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _build(path, model_size, train, **overrides):
    K, expert_idx, household_idx, gamma_vec, init_guess = _case()
    kwargs = dict(
        model_size=model_size,
        n_epochs=TRAINING["epochs"],
        batch_size=TRAINING["batch"],
        lr=TRAINING["lr"],
        timestepping=True,
        rar=True,
        loss_balancing=False,
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
        params=economic_params(CASE),
        share_alpha_lo=SHARE_ALPHA_LO,
        share_alpha_hi=SHARE_ALPHA_HI,
        seed=TRAINING_SEED,
        train=train,
    )
    kwargs.update(overrides)
    return get_model(path, K, expert_idx, household_idx, gamma_vec, **kwargs)


def _write_run_info(path, info):
    with open(os.path.join(path, "run_info.json"), "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)


def _read_run_info(path):
    fn = os.path.join(path, "run_info.json")
    if not os.path.exists(fn):
        return {}
    with open(fn, encoding="utf-8") as file:
        return json.load(file)


def train_architecture(name):
    """Base run with only ``model_size`` changed."""
    spec = VARIANTS[name]
    path = spec["path"]
    fresh = not os.path.exists(os.path.join(path, "model.pt"))
    print(f"\n{(' ' + CASE + ' / ' + name + ' '):=^80}")
    _reset_peak()
    start = time.perf_counter()
    model = _build(path, spec["size"], train=True)
    _cuda_sync()
    if fresh:
        _write_run_info(path, {
            "model_size": spec["size"],
            "wall_seconds": time.perf_counter() - start,
            "peak_vram_gb": _peak_gb(),
            "peak_reserved_gb": (torch.cuda.max_memory_reserved() / 1024 ** 3
                                 if torch.cuda.is_available() else float("nan")),
        })
    _free(model)


# ---------------------------------------------------------------------------
# LBFGS continuation
# ---------------------------------------------------------------------------
def _closure_no_clip(self, SV):
    """The model's closure without gradient clipping: LBFGS's line search needs
    the gradient of the returned loss."""
    for i, sv_name in enumerate(self.state_variables):
        self.variable_val_dict[sv_name] = SV[:, i:i + 1]
    self.variable_val_dict["SV"] = SV
    self.update_variables(SV)
    self.loss_fn()
    total_loss = 0
    for loss_label, loss in self.loss_val_dict.items():
        total_loss += self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss)
    self.optimizer.zero_grad()
    total_loss.backward()
    self.n_closure_evals += 1
    return total_loss


def _make_lbfgs(model, lr):
    params = []
    for d in (model.agents, model.endog_vars):
        for name in d:
            params += list(d[name].parameters())
    return torch.optim.LBFGS(
        params, lr=lr, max_iter=LBFGS["max_iter"],
        history_size=LBFGS["history_size"], line_search_fn="strong_wolfe",
    )


def _init_lbfgs(self):
    self.optimizer = _make_lbfgs(self, self.lr)


def _reset_lbfgs_on_new_anchors(self, *args, **kwargs):
    """RAR appends anchors to the training batch, which changes the objective;
    drop the curvature pairs collected on the old batch."""
    n = 0 if self.anchor_points is None else int(self.anchor_points.shape[0])
    if n != self._lbfgs_n_anchor:
        self._lbfgs_n_anchor = n
        self.optimizer.state.clear()


def _use_lbfgs(model):
    model.n_closure_evals = 0
    model._lbfgs_n_anchor = 0
    model.closure = types.MethodType(_closure_no_clip, model)
    model._PDEModelTimeStep__init_optimizer = types.MethodType(_init_lbfgs, model)
    model.OnInnerLoopStep += types.MethodType(_reset_lbfgs_on_new_anchors, model)


class _NetworkGuess:
    """Initial time-boundary values = the loaded networks at (SV_T0, min_t), i.e.
    what the base run would have passed to its next time step."""

    def __init__(self, model):
        self.model = model
        self._key = None
        self._vals = None

    def __call__(self, name):
        def guess(SV_T0):
            if self._key is not SV_T0:
                self._vals = {}
                with torch.no_grad():
                    self.model.update_variables(SV_T0, vd=self._vals)
                self._key = SV_T0
            return self._vals[name].detach()
        return guess


def lbfgs_overrides():
    return dict(
        lr=LBFGS["lr"],
        num_outer=LBFGS["outer"],
        num_inner=LBFGS["inner"],
        min_inner=LBFGS["inner"],
        loss_log_interval=LBFGS["loss_log_interval"],
    )


def train_lbfgs():
    path = VARIANTS["lbfgs"]["path"]
    print(f"\n{(' ' + CASE + ' / lbfgs '):=^80}")
    if os.path.exists(os.path.join(path, "model.pt")):
        return
    model = _build(path, BASE_SIZE, train=False, **lbfgs_overrides())
    model.load_model(torch.load(os.path.join(BASE_DIR, "model_best.pt"),
                                weights_only=False, map_location=model.device))
    model.attach_stacks(list(model._xi_names))
    _use_lbfgs(model)
    guess = _NetworkGuess(model)
    model.set_initial_guess({
        name: guess(name) for name in [*model.agents, *model.endog_vars]
    })

    os.makedirs(path, exist_ok=True)
    _reset_peak()
    start = time.perf_counter()
    model.train_model(path, "model.pt", full_log=True,
                      variables_to_track=["chat", "r", "r_implied", "p", "mu_P"])
    _cuda_sync()
    _write_run_info(path, {
        "init_checkpoint": os.path.join(BASE_DIR, "model_best.pt"),
        "lbfgs": LBFGS,
        "wall_seconds": time.perf_counter() - start,
        "lbfgs_steps": LBFGS["outer"] * LBFGS["inner"],
        "closure_evals": model.n_closure_evals,
        "peak_vram_gb": _peak_gb(),
        "peak_reserved_gb": (torch.cuda.max_memory_reserved() / 1024 ** 3
                             if torch.cuda.is_available() else float("nan")),
    })
    _free(model)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _load_variant(name):
    spec = VARIANTS[name]
    overrides = lbfgs_overrides() if name == "lbfgs" else {}
    model = _build(spec["path"], spec["size"], train=False, **overrides)
    if not os.path.exists(os.path.join(spec["path"], "model_best.pt")):
        raise SystemExit(f"No checkpoint for {name} in {spec['path']}.")
    return model


def benchmark(model, optimizer):
    """Peak VRAM and median wall time of one training step on the steady-state
    RAR batch (base batch + anchors), plus one RAR scoring pass -- the same
    measurement as diagnostics.measure_config, on the trained weights."""
    move_model(model, "cuda" if torch.cuda.is_available() else "cpu")
    if optimizer == "LBFGS":
        _use_lbfgs(model)
        model.optimizer = _make_lbfgs(model, LBFGS["lr"])
    else:
        params = []
        for d in (model.agents, model.endog_vars):
            for name in d:
                params += list(d[name].parameters())
        model.optimizer = torch.optim.Adam(params, lr=TRAINING["lr"])
    _prepopulate_anchors(model)

    def step():
        SV = torch.vstack((model.sample(0), model.anchor_points))
        model.optimizer.step(lambda: model.closure(SV))

    _reset_peak()
    for _ in range(BENCH_WARMUP):
        step()
    _cuda_sync()
    times = []
    for _ in range(BENCH_STEPS):
        t0 = time.perf_counter()
        step()
        _cuda_sync()
        times.append(time.perf_counter() - t0)
    step_peak = _peak_gb()

    _reset_peak()
    model.sample_rar_greedy()
    _cuda_sync()
    refine_peak = _peak_gb()
    return {
        "bench_peak_vram_gb": float(np.nanmax([step_peak, refine_peak])),
        "bench_ms_per_step": 1e3 * float(np.median(times)),
    }


def evaluate(base_info):
    SV_val = None
    rows = []
    for name, spec in VARIANTS.items():
        print(f"\n{(' evaluating ' + name + ' '):=^80}")
        model = _load_variant(name)
        if SV_val is None:
            SV_val = _validation_states(model, n_samples=N_VALIDATION, seed=0)
        losses = compute_validation_losses(model, SV_val, chunk_size=VALIDATION_CHUNK)
        n_params = _count_params(model)

        log = base_info if name == "baseline" else _log_summary(spec["path"])
        run = _read_run_info(spec["path"])
        if name == "lbfgs":
            epochs = run.get("lbfgs_steps", float("nan"))
            evals = run.get("closure_evals", float("nan"))
        else:
            epochs = log["epochs"] if log else float("nan")
            evals = epochs
        seconds = log["train_seconds"] if log else float("nan")

        bench = benchmark(model, spec["optimizer"])
        _free(model)

        rows.append({
            "Model": spec["label"],
            "Hidden units": "[{}]x{}".format(spec["size"][0], len(spec["size"])),
            "Optimizer": spec["optimizer"],
            "Parameters": n_params,
            "Epochs": epochs,
            "Function evaluations": evals,
            "Training time (h)": seconds / 3600,
            "Time / epoch (s)": seconds / epochs,
            "Time / function evaluation (s)": seconds / evals,
            "Peak VRAM, training (GB)": run.get("peak_vram_gb", float("nan")),
            "Peak VRAM, benchmark (GB)": bench["bench_peak_vram_gb"],
            "Step time, benchmark (ms)": bench["bench_ms_per_step"],
            **{f"Validation {k}": v for k, v in losses.items() if k != "Total"},
            "Validation loss": losses["Total"],
        })
    return pd.DataFrame(rows).set_index("Model")


def _fmt(fmt):
    def f(x):
        return "--" if isinstance(x, float) and not np.isfinite(x) else fmt(x)
    return f


TABLE_FORMATS = {
    "Hidden units": str,
    "Optimizer": str,
    "Parameters": _fmt(lambda x: f"{int(x):,}"),
    "Epochs": _fmt(lambda x: f"{int(x):,}"),
    "Validation loss": _fmt(format_sci),
    "Peak VRAM, benchmark (GB)": _fmt(lambda x: f"{x:.2f}"),
    "Peak VRAM, training (GB)": _fmt(lambda x: f"{x:.2f}"),
    "Time / epoch (s)": _fmt(lambda x: f"{x:.3f}"),
    "Training time (h)": _fmt(lambda x: f"{x:.2f}"),
}


def write_table(df):
    os.makedirs(OUT_ROOT, exist_ok=True)
    df.to_csv(os.path.join(OUT_ROOT, "capacity_comparison.csv"))
    out = df[list(TABLE_FORMATS)].copy()
    for col, fmt in TABLE_FORMATS.items():
        out[col] = out[col].apply(fmt)
    with open(os.path.join(OUT_ROOT, "capacity_comparison.tex"), "w") as file:
        file.write(out.style.to_latex(hrules=True))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out)


def main():
    torch.set_default_dtype(torch.float64)
    base_info = _check_baseline()
    os.makedirs(OUT_ROOT, exist_ok=True)
    train_lbfgs()
    train_architecture("deep_8x64")
    train_architecture("wide_4x128")
    write_table(evaluate(base_info))


if __name__ == "__main__":
    main()
