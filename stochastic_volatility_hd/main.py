import argparse
import gc
import os

import numpy as np
import torch

from analysis import *
from common import (BASE_PARAMS, CONFIGS, CORE_CONFIGS, SHARE_ALPHA_LO,
                    SHARE_ALPHA_HI, configs_for_case, df_to_latex, make_case,
                    move_model)
from model import get_model


def main():
    """Train all configs for one case + emit the comparison artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["agents2", "agents5", "agents20", "agents40", "agents50"], default="agents2")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--outer", type=int, default=100, help="num_outer_iterations for time-stepping configs")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_balancing_temp", type=float, default=0.1)
    parser.add_argument("--loss_balancing_alpha", type=float, default=0.999)
    parser.add_argument("--bernoulli_prob", type=float, default=0.9999)
    parser.add_argument("--num-inner", type=int, default=5000,
                        help="num_inner_iterations at outer 0 (the library decays it ~1/sqrt(outer) down to --min-inner)")
    parser.add_argument("--min-inner", type=int, default=2000,
                        help="floor for the per-outer inner iterations")
    parser.add_argument("--lr-decay-every", type=int, default=20,
                        help="multiply the LR by --lr-decay-gamma every N outer (time-stepping) iterations")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5,
                        help="LR step-decay factor applied every --lr-decay-every outer iterations")
    # Calibrated defaults for the 2-D validation case (paper section 4):
    # a=0.1, sigma=0.06, tau=1.15, gamma=6.  For higher-dimensional cases the
    # per-agent gamma vector is set inside make_case (common.py) and --gamma is
    # ignored; a/sigma/tau stay at these defaults.
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--tau", type=float, default=1.15)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--alpha-lo", type=float, default=SHARE_ALPHA_LO,
                        help="lower bound of the log-uniform Dirichlet-alpha mixture for wealth-share sampling (alpha<1 => concentrated draws)")
    parser.add_argument("--alpha-hi", type=float, default=SHARE_ALPHA_HI,
                        help="upper bound of the log-uniform Dirichlet-alpha mixture for wealth-share sampling")
    parser.add_argument("--t0-frac", type=float, default=0.4,
                        help="fraction of each time-stepping training batch pinned to t=min_t (t0-mix sampler)")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()

    gamma = args.gamma
    tau = args.tau
    a = args.a
    sigma = args.sigma

    torch.set_default_dtype(torch.float64)
    base_dir = f"./models/{args.case}"

    K, eidx, hidx, gamma_vec = make_case(args.case, gamma)
    print(f"[sv_n_agents] case={args.case} K={K} experts={eidx} households={hidx}")
    print(f"             gamma={gamma_vec}")

    ts_init_guess = {f"xi_{k}": BASE_PARAMS["rho"] for k in range(1, K + 1)}
    ts_init_guess["r"] = 0.01

    # agents2 -> full 8-method ladder (validation); higher-D -> 4 core methods.
    case_configs = configs_for_case(args.case)
    print(f"[sv_n_agents] training configs: {case_configs}")

    models, model_paths, ts_map = {}, {}, {}
    for name in case_configs:
        ts, rar, lb = CONFIGS[name]
        mpath = os.path.join(base_dir, name)
        print(f"\n{('=== ' + name + ' ==='):=^80}")
        model = get_model(
            mpath, K, eidx, hidx, gamma_vec,
            model_size=[args.width] * args.layers,
            n_epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            timestepping=ts, rar=rar, loss_balancing=lb, num_outer=args.outer,
            num_inner=args.num_inner, min_inner=args.min_inner,
            lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
            loss_balancing_alpha=args.loss_balancing_alpha, loss_balancing_temp=args.loss_balancing_temp, bernoulli_prob=args.bernoulli_prob,
            t0_frac=args.t0_frac,
            init_guess=ts_init_guess, params=BASE_PARAMS | {"tau": tau, "a": a, "sigma": sigma},
            share_alpha_lo=args.alpha_lo, share_alpha_hi=args.alpha_hi,
        )
        # park the trained model on CPU so the next config's training (and the
        # later evaluation) doesn't have every config's networks pinned in VRAM.
        move_model(model, "cpu")
        models[name] = model
        model_paths[name] = mpath
        ts_map[name] = ts
        gc.collect(); torch.cuda.empty_cache()

    cmp_dir = os.path.join(base_dir, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)

    if K >= 10:
        chunk_size = 500
    else:
        chunk_size = 2000

    # ---- 1) Loss tables (computed over ALL trained methods) -----------------
    # Two flavours of validation sampling for the time-stepping rows: the
    # default scores them at t=min_t (the stationary slice we extract); the
    # "_random_t" variant scores them at a random t over [min_t, max_t] so the
    # whole trained horizon is validated (see compute_validation_losses_random_t).
    loss_df = welfare_df = None
    if "basic" in models:
        loss_df = compare_loss_table(models, baseline_key="basic", chunk_size=chunk_size)
        loss_df.to_csv(os.path.join(cmp_dir, "comparative_losses.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, "comparative_losses.tex"))
        print("\n[sv_n_agents] comparative loss table (vs basic):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        loss_df_rt = compare_loss_table(models, baseline_key="basic", chunk_size=chunk_size,
                                        compute_fn=compute_validation_losses_random_t)
        loss_df_rt.to_csv(os.path.join(cmp_dir, "comparative_losses_random_t.csv"))
        df_to_latex(loss_df_rt, os.path.join(cmp_dir, "comparative_losses_random_t.tex"))
        print("\n[sv_n_agents] comparative loss table (random t, vs basic):")
        print(loss_df_rt.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key="basic", chunk_size=chunk_size)
        welfare_df.to_csv(os.path.join(cmp_dir, "welfare_equivalent_losses.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, "welfare_equivalent_losses.tex"))
        print("\n[sv_n_agents] welfare-equivalent loss table (c/W, vs basic):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    if "timestep" in models:
        loss_df = compare_loss_table(models, baseline_key="timestep", chunk_size=chunk_size)
        loss_df.to_csv(os.path.join(cmp_dir, "comparative_losses_timestep.csv"))
        df_to_latex(loss_df, os.path.join(cmp_dir, "comparative_losses_timestep.tex"))
        print("\n[sv_n_agents] comparative loss table (vs timestep):")
        print(loss_df.to_string(float_format=lambda x: f"{x:.3e}"))

        loss_df_rt = compare_loss_table(models, baseline_key="timestep", chunk_size=chunk_size,
                                        compute_fn=compute_validation_losses_random_t)
        loss_df_rt.to_csv(os.path.join(cmp_dir, "comparative_losses_timestep_random_t.csv"))
        df_to_latex(loss_df_rt, os.path.join(cmp_dir, "comparative_losses_timestep_random_t.tex"))
        print("\n[sv_n_agents] comparative loss table (random t, vs timestep):")
        print(loss_df_rt.to_string(float_format=lambda x: f"{x:.3e}"))

        welfare_df = compute_welfare_equivalent_losses(models, baseline_key="timestep", chunk_size=chunk_size)
        welfare_df.to_csv(os.path.join(cmp_dir, "welfare_equivalent_losses_timestep.csv"))
        df_to_latex(welfare_df, os.path.join(cmp_dir, "welfare_equivalent_losses_timestep.tex"))
        print("\n[sv_n_agents] welfare-equivalent loss table (c/W, vs timestep):")
        print(welfare_df.to_string(float_format=lambda x: f"{x:.3e}"))

    # ---- pick basic + the best-improving method(s) for the overlay plots ----
    plot_models = select_plot_methods(models, loss_df=loss_df, welfare_df=welfare_df, baseline_key="basic")
    plot_paths = {k: model_paths[k] for k in plot_models}
    plot_ts = {k: ts_map[k] for k in plot_models}
    print(f"[sv_n_agents] plotting methods: {list(plot_models)}")

    # ---- 2) 2-D slice comparison vs the Di Tella FD solution (agents2 only) --
    if args.case == "agents2":
        try:
            fd_dict = np.load(f"./models/numerical/numerical_{gamma}_{tau}_{sigma}_{a}.npz")
        except Exception as e:
            print(f"[sv_n_agents] could not load FD solution: {e}")
            fd_dict = None
        v_list = [0.25]
        method_dicts = {name: evaluate_slices(m, v_list) for name, m in plot_models.items()}
        plot_slice_comparison(method_dicts, fd_dict, v_list, cmp_dir)
        print(f"[sv_n_agents] slice comparison plots saved to {cmp_dir}")

        # FD-error table (omega, e_hat, c_hat, risk_premium) over ALL methods,
        # averaged across every v-slice the FD solution provides.
        if fd_dict is not None:
            v_tab = fd_v_slices(fd_dict) or v_list
            all_method_dicts = {name: evaluate_slices(m, v_tab) for name, m in models.items()}
            mse_df, mae_df = compare_fd_table(all_method_dicts, fd_dict, v_tab, cmp_dir)
            print(f"\n[sv_n_agents] FD-error table (MSE vs FD, v={v_tab}):")
            print(mse_df.to_string(float_format=lambda x: f"{x:.3e}"))
    else:
        # plot histograms + aggregate scatter (p / risk premium / omega vs the
        # total expert wealth share at v=0.25) for the high-dimensional cases.
        for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
            plot_theta_chat_histogram(models[name], model_paths[name], chunk_size=chunk_size)
            plot_aggregate_scatter(models[name], cmp_dir, file_name=f"aggregate_scatter_{name}.pdf", chunk_size=chunk_size)

    # ---- 3) RAR anchor scatter (rar configs, K=2) ---------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if rar:
            plot_rar_anchors(model_paths[name], K, cmp_dir, file_name=f"rar_anchors_{name}.pdf", timestepping=ts)

    # ---- 4) Loss-weight evolution (lb configs) ------------------------------
    for name, ts, rar, lb in [(n, *CONFIGS[n]) for n in models]:
        if lb:
            plot_loss_weights(model_paths[name], cmp_dir, file_name=f"loss_weight_{name}.pdf", timestepping=ts)

    # ---- 5) HJB / total loss convergence (basic + best method) --------------
    plot_paths = {k: model_paths[k] for k in ["basic", "timestep", "timestep_rar"]}
    plot_ts = {k: ts_map[k] for k in ["basic", "timestep", "timestep_rar"]}
    plot_loss_decay(plot_paths, cmp_dir, plot_ts)
    print(f"[sv_n_agents] loss-decay plot saved to {cmp_dir}")

    print(f"\n[sv_n_agents] all artifacts written under {cmp_dir}")


if __name__ == "__main__":
    main()

    # Paper runs (all float64; a=0.1, sigma=0.06, tau=1.15 throughout):
    #   2-D validation (section 4.1, full 8-method ladder, gamma=6):
    #     --case agents2  --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 6.0
    #   scaling study (section 4.2, 4 core methods; --gamma ignored, gamma_vec
    #   is set in make_case):
    #     --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15
    #     --case agents40 --float64 --a 0.1 --sigma 0.06 --tau 1.15