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

RAR configs are measured representatively (not at the empty-buffer epoch 0):
before timing we (a) pre-populate ``anchor_points`` to the steady-state size a
real run reaches so the training step runs on ``base_batch + anchors`` (the same
nominal batch as ``basic``), and (b) separately measure the periodic
residual-scoring pass (the dense ``sample_times x batch`` forward incl. the
Hessians) -- the true RAR memory/FLOP peak.  ``peak_mem_mb`` is the max over the
training step and the refinement pass.  Without this, ``model.sample(0)`` at
epoch 0 samples a HALF-size batch with no anchors and never scores a pool, which
is why the naive table reports RAR as *cheaper* than the dense method.

Outputs (under ``--out-dir``, default ``./models/diagnostics``):
  * ``compute_memory.csv`` -- raw measurements, one row per (case, config);
    includes ``n_train_points``/``n_anchor`` and the separate refinement-pass
    cost (``refine_mem_mb``/``refine_gflops``) for transparency;
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
from model import (PDEModelNAgentsSV, PDEModelTimeStepNAgentsSV, get_model)


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


def _training_step(model, append_anchors=False):
    """One forward+backward+optimizer step on a freshly sampled batch, using the
    same closure the trainer uses (``_SVNAgentMixin.closure``).

    ``append_anchors`` reproduces what the time-stepping inner loop does: its
    training sampler (``sample_simplex_v_ts``) does NOT itself carry the RAR
    anchors -- the library vstacks ``anchor_points`` onto the batch each step --
    so for a representative measurement we append them here.  The stationary RAR
    sampler (``sample_rar_greedy``) already appends them, so it passes False.
    """
    SV = model.sample(0)
    if append_anchors:
        ap = getattr(model, "anchor_points", None)
        if torch.is_tensor(ap) and ap.numel() and ap.shape[1] == SV.shape[1]:
            SV = torch.vstack((SV, ap))
    loss = model.closure(SV)
    model.optimizer.step()
    return float(loss.detach().cpu())


def _profile_gflops(fn, cuda=None):
    """GFLOPs of a single call to ``fn`` via the torch profiler (best effort)."""
    if cuda is None:
        cuda = torch.cuda.is_available()
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with profile(activities=acts, with_flops=True) as prof:
            with record_function("step"):
                fn()
        return sum(k.flops for k in prof.key_averages()) / 1e9
    except Exception as e:
        print(f"  [flops] profiler failed ({e}); leaving GFLOPs blank")
        return float("nan")


def _prepopulate_anchors(model):
    """Fill ``model.anchor_points`` to the size a real RAR run reaches at steady
    state, so the measured training step reflects ``base_batch + anchors`` rather
    than the empty-buffer epoch-0 state.

    Both RAR variants add ``batch_size // refinement_rounds`` points per
    refinement over ``refinement_rounds`` rounds, i.e. ~``batch_size`` anchors in
    total (recall ``batch_size`` is already the RAR-halved value).  Restoring
    that many anchors brings the RAR training batch back up to the same nominal
    size as ``basic`` -- the honest per-step comparison.
    """
    rounds = int(getattr(model, "refinement_rounds", 10)) or 10
    per = max(1, model.batch_size // rounds)
    n_anchor = per * rounds
    if isinstance(model, PDEModelTimeStepNAgentsSV):
        pts = model._sample_simplex_at_t0()      # (B, K+1), carries the t column
    else:
        pts = model.sample_simplex_v(0)          # (B, K)
    if pts.shape[0] < n_anchor:
        reps = (n_anchor + pts.shape[0] - 1) // pts.shape[0]
        pts = pts.repeat(reps, 1)
    model.anchor_points = pts[:n_anchor].detach()
    return int(n_anchor)


def _refinement_step(model):
    """Run one periodic RAR residual-scoring pass -- the dense
    (``sample_times x batch``) forward over the wealth simplex (incl. the
    ``(B,N,D,D)`` Hessians) that ranks residuals and accumulates anchors.  This
    is the RAR-specific memory/FLOP peak that a single training step never
    exercises."""
    if isinstance(model, PDEModelTimeStepNAgentsSV):
        model.sample_rar_greedy()                # scores 2 pools, grows anchors
    else:
        model._refinement_loss_dict(0)           # scores the dense pool


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

    # --- make the RAR measurement representative -----------------------------
    # A real RAR run trains on ``base_batch + accumulated anchors`` and pays a
    # periodic dense residual-scoring pass; measuring ``model.sample(0)`` at
    # epoch 0 (empty anchor buffer, no refinement) instead reports a HALF-batch
    # step -- which is why the raw table shows RAR as *cheaper*.  Fill the anchor
    # buffer to its steady-state size, and (below) measure the refinement pass.
    is_rar = bool(CONFIGS[config][1])
    append_anchors = is_rar and isinstance(model, PDEModelTimeStepNAgentsSV)
    n_anchor = _prepopulate_anchors(model) if is_rar else 0
    n_train_points = int(model.batch_size + n_anchor)

    cuda = torch.cuda.is_available()
    if cuda:
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # warmup (build kernels / autograd graph shapes) then timed steps
    for _ in range(args.warmup):
        _training_step(model, append_anchors)
    if cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        _training_step(model, append_anchors)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    step_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if cuda else float("nan")
    step_gflops = _profile_gflops(lambda: _training_step(model, append_anchors), cuda)

    # --- periodic refinement pass (RAR only): the real memory/FLOP peak -------
    refine_mem_mb = float("nan")
    refine_gflops = float("nan")
    if is_rar:
        if cuda:
            gc.collect(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            _refinement_step(model)
            if cuda:
                torch.cuda.synchronize()
            refine_mem_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if cuda else float("nan")
            refine_gflops = _profile_gflops(lambda: _refinement_step(model), cuda)
        except Exception as e:
            print(f"  [refine] refinement-pass measurement failed ({e})")

    # headline peak = worst case a real RAR step actually hits
    peak_mem_mb = float(np.nanmax([step_mem_mb, refine_mem_mb])) if cuda else float("nan")

    res = {
        "case": case,
        "K": K,
        "config": config,
        "n_params": n_params,
        "peak_mem_mb": peak_mem_mb,
        "ms_per_step": 1e3 * float(np.median(times)),
        "gflops": step_gflops,
        "n_train_points": n_train_points,
        "n_anchor": int(n_anchor),
        "step_mem_mb": step_mem_mb,
        "refine_mem_mb": refine_mem_mb,
        "refine_gflops": refine_gflops,
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
                      f"{res['ms_per_step']:.2f} ms/step  {res['gflops']:.2f} GFLOPs  "
                      f"(train_pts={res['n_train_points']}, anchors={res['n_anchor']}, "
                      f"refine_mem={res['refine_mem_mb']:.1f} MB, refine={res['refine_gflops']:.2f} GFLOPs)")
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
