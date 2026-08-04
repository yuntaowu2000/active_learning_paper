"""Compute / memory diagnostics for the N-agent stochastic-volatility model
(paper section 4.3).

For each requested case (dimension) and each of the core training configs this
script builds a *fresh, untrained* model and measures the per-training-step
cost of the equilibrium forward + backward:

  * number of trainable parameters (identical across configs of one case);
  * peak CUDA memory (MB) during a training step;
  * wall-clock time per training step (ms), median over several timed steps;
  * FLOPs per step (GFLOPs) via the torch profiler (best effort).

The point is a like-for-like scaling table (2D -> 20D -> 40D), so the numbers
are measured on freshly-initialised networks -- they do NOT require trained
checkpoints and are independent of the learned weights.

Outputs (under ``--out-dir``, default ``./models/diagnostics``):
  * ``compute_memory.csv`` -- raw measurements, one row per (case, config);
  * ``compute_memory.tex`` -- a formatted LaTeX table (cases x methods).

Usage::

    python diagnostics.py --cases agents2,agents20,agents40 --float64
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import torch

from common import (BASE_PARAMS, CONFIGS, CORE_CONFIGS, _module_of,
                    make_case, move_model)
from model import get_model


def _iter_param_modules(model):
    for d in (getattr(model, "agents", {}), getattr(model, "endog_vars", {})):
        for name in d:
            yield _module_of(d[name])


def _count_params(model):
    seen, total = set(), 0
    for mod in _iter_param_modules(model):
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def _make_optimizer(model, lr):
    seen, params = set(), []
    for mod in _iter_param_modules(model):
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return torch.optim.Adam(params, lr=lr)


def _training_step(model):
    """One forward+backward+optimizer step on a freshly sampled batch, using the
    same closure the trainer uses (``_SVNAgentMixin.closure``)."""
    SV = model.sample(0)
    loss = model.closure(SV)
    model.optimizer.step()
    return float(loss.detach().cpu())


def measure_config(case, config, args):
    """Build + measure a single (case, config).  Returns a dict of metrics."""
    ts, rar, lb = CONFIGS[config]
    gamma = args.gamma
    K, eidx, hidx, gamma_vec = make_case(case, gamma)
    params = BASE_PARAMS | {"tau": args.tau, "a": args.a, "sigma": args.sigma}

    # A throwaway path: train=False means get_model just assembles the (untrained)
    # networks; nothing is written and no checkpoint is required.
    mpath = os.path.join(args.out_dir, "_scratch", case, config)
    ts_init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    ts_init_guess["r"] = 0.01

    model = get_model(
        mpath, K, eidx, hidx, gamma_vec,
        model_size=[args.width] * args.layers,
        batch_size=args.batch, lr=args.lr,
        timestepping=ts, rar=rar, loss_balancing=lb,
        num_outer=1, num_inner=2, min_inner=2,
        init_guess=ts_init_guess, params=params,
        train=False,
    )
    # get_model assembles the networks on the default device already; make sure
    # every device-resident tensor (networks + static gamma) is on the compute
    # device before we measure.
    move_dev = "cuda" if torch.cuda.is_available() else "cpu"
    move_model(model, move_dev)

    n_params = _count_params(model)
    model.optimizer = _make_optimizer(model, args.lr)

    cuda = torch.cuda.is_available()
    if cuda:
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # warmup (build kernels / autograd graph shapes) then timed steps
    for _ in range(args.warmup):
        _training_step(model)
    if cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        _training_step(model)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if cuda else float("nan")

    gflops = float("nan")
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with profile(activities=acts, with_flops=True) as prof:
            with record_function("train_step"):
                _training_step(model)
        total_flops = sum(k.flops for k in prof.key_averages())
        gflops = total_flops / 1e9
    except Exception as e:
        print(f"  [flops] profiler failed ({e}); leaving GFLOPs blank")

    res = {
        "case": case,
        "K": K,
        "config": config,
        "n_params": n_params,
        "peak_mem_mb": peak_mem_mb,
        "ms_per_step": 1e3 * float(np.median(times)),
        "gflops": gflops,
    }

    del model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return res


def format_table(df, out_dir):
    """Cases (rows) x method (Time-stepping / Our Method) column groups, mirroring
    the tree-model compute/memory table."""
    method_map = {"timestep": "Time-stepping", "timestep_rar": "Our Method"}
    sub = df[df["config"].isin(method_map)].copy()
    if sub.empty:
        return
    cases = list(dict.fromkeys(sub["case"]))
    case_label = {c: f"{int(sub[sub['case'] == c]['K'].iloc[0])}D" for c in cases}

    metrics = [("n_params", "Params"),
               ("peak_mem_mb", "CUDA Memory (MB)"),
               ("gflops", r"FLOPs ($\times 10^9$)"),
               ("ms_per_step", "Time / step (ms)")]
    method_order = ["Time-stepping", "Our Method"]
    cols = pd.MultiIndex.from_tuples(
        [(disp, m) for _, disp in metrics for m in method_order])
    res = pd.DataFrame(index=[case_label[c] for c in cases], columns=cols)

    for c in cases:
        for cfg, disp_m in method_map.items():
            r = sub[(sub["case"] == c) & (sub["config"] == cfg)]
            if r.empty:
                continue
            r = r.iloc[0]
            for key, disp in metrics:
                val = r[key]
                if key == "n_params":
                    txt = f"{int(val):,}" if np.isfinite(val) else ""
                elif np.isfinite(val):
                    txt = f"{val:.2f}"
                else:
                    txt = ""
                res.loc[case_label[c], (disp, disp_m)] = txt

    ltx = res.style.to_latex(column_format="l" + "c" * len(cols),
                             hrules=True, multicol_align="c")
    with open(os.path.join(out_dir, "compute_memory.tex"), "w") as f:
        f.write(ltx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="agents2,agents20,agents40",
                        help="comma-separated cases to profile")
    parser.add_argument("--configs", default=",".join(CORE_CONFIGS),
                        help="comma-separated configs to profile (subset of CONFIGS)")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--tau", type=float, default=1.15)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--warmup", type=int, default=3, help="untimed warmup steps")
    parser.add_argument("--steps", type=int, default=10, help="timed steps (median reported)")
    parser.add_argument("--out-dir", default="./models/diagnostics")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    rows = []
    for case in cases:
        for config in configs:
            print("{0:=^80}".format(f" {case} / {config} "))
            try:
                res = measure_config(case, config, args)
                rows.append(res)
                print(f"  params={res['n_params']:,}  peak_mem={res['peak_mem_mb']:.1f} MB  "
                      f"{res['ms_per_step']:.2f} ms/step  {res['gflops']:.2f} GFLOPs")
            except Exception as e:
                print(f"  [error] {case}/{config}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        print("[diagnostics] no measurements collected")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "compute_memory.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[diagnostics] raw measurements -> {csv_path}")
    print(df.to_string(index=False))
    format_table(df, args.out_dir)
    print(f"[diagnostics] LaTeX table -> {os.path.join(args.out_dir, 'compute_memory.tex')}")


if __name__ == "__main__":
    main()
