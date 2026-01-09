import gc
import re

from itertools import product

from torch.profiler import ProfilerActivity, profile, record_function
from tree_model_main import *

# over write the mu sigs 
MU_SIGS = {
    k: [0.01 * i for i in range(1, k+1)] for k in [2, 3, 5, 10, 20, 40, 50] #
}

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

def get_time(curr_dir, rar=False):
    # Outer Loop ([0-9]+) Finished in ([0-9\.]+)s
    all_epoch_times = np.zeros((OUTER_ITERATIONS,))
    if rar:
        fn = f"{curr_dir}/timestep_rar/model-{OUTER_ITERATIONS}-{INNER_ITERATIONS}-log.txt"
    else:
        fn = f"{curr_dir}/timestep/model-{OUTER_ITERATIONS}-{INNER_ITERATIONS}-log.txt"
    with open(fn, "r") as f:
        logs = f.read()
    matched_groups = re.findall("Outer Loop ([0-9]+) Finished in ([0-9\.]+)s", logs)
    for outer_iter, total_time in matched_groups:
        outer_iter = int(outer_iter)
        total_time = float(total_time)
        curr_inner_iter = max(1000, int(INNER_ITERATIONS / (np.sqrt(outer_iter + 1)))) # note that by default, we run for at least 1000 epochs in each out loop
        epoch_time = total_time / curr_inner_iter
        all_epoch_times[outer_iter] = epoch_time
    return {
        "mean": np.mean(all_epoch_times),
        "5percent": np.quantile(all_epoch_times, 0.05),
        "95percent": np.quantile(all_epoch_times, 0.95)
    }

def plot_timing(plot_dir):
    n_trees = list(MU_SIGS.keys())
    time_mean = [0] * len(n_trees)
    time_5percent = [0] * len(n_trees)
    time_95percent = [0] * len(n_trees)

    rar_time_mean = [0] * len(n_trees)
    rar_time_5percent = [0] * len(n_trees)
    rar_time_95percent = [0] * len(n_trees)
    for i, n_tree in enumerate(n_trees):
        curr_dir = os.path.join(BASE_DIR, f"tree_{n_tree}")
        curr_res = get_time(curr_dir)
        time_mean[i] = curr_res["mean"]
        time_5percent[i] = curr_res["5percent"]
        time_95percent[i] = curr_res["95percent"]

        rar_res = get_time(curr_dir, True)
        rar_time_mean[i] = rar_res["mean"]
        rar_time_5percent[i] = rar_res["5percent"]
        rar_time_95percent[i] = rar_res["95percent"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(n_trees, time_mean, linestyle="-.", color="#317e46", label="Time-stepping")
    ax.fill_between(n_trees, time_5percent, time_95percent, alpha=0.2, color="gray")

    ax.plot(n_trees, rar_time_mean, linestyle="-", color="#5492ab", label="Our Method")
    ax.fill_between(n_trees, rar_time_5percent, rar_time_95percent, alpha=0.2, color="gray")
    ax.set_xlabel("Number of Trees", fontsize=16)
    ax.set_ylabel("Time (s)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/tree_timing.pdf")
    plt.close()

def get_model_for_memory(n_tree, params, seed=0, rar=False):
    set_seeds(seed)
    if n_tree == 2:
        # in one dimension, it is fine to sample uniformly
        model = PDEModelTimeStep(f"tree{n_tree}", 
            config={"batch_size": 200, "time_batch_size": -1,
                    "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
                    "num_outer_iterations": 1,
                    "num_inner_iterations": 2,
                    "min_inner_iterations": 2,
                    "lr": 0.0005, 
                    "optimizer_type": OptimizerType.Adam,
                    "refinement_rounds": 2})
    else:
        model = PDEModelTimeStepCustomSample(f"tree{n_tree}", 
            config={"batch_size": 200, "time_batch_size": -1,
                    "sampling_method": SamplingMethod.RARG if rar else SamplingMethod.UniformRandom,
                    "num_outer_iterations": 1,
                    "num_inner_iterations": 2,
                    "min_inner_iterations": 2,
                    "lr": 0.0005, 
                    "optimizer_type": OptimizerType.Adam,
                    "refinement_rounds": 2})
    model.set_state([f"z{i+1}" for i in range(n_tree-1)], {f"z{i+1}": [0.01, 0.99] for i in range(n_tree-1)})
    model.add_params(params)
    model.add_endogs(["k"], configs={
        "k": {"positive": True, "hidden_units": [80] * 4, "output_size": n_tree},
    })
    model.register_function(compute_q)
    model.register_function(compute_qz)
    model.register_function(compute_qzz)
    model.register_function(compute_mu_z_geos)
    model.register_function(compute_sig_z_geos)
    model.register_function(compute_mu_qs)
    model.register_function(compute_sig_qs)
    model.register_function(compute_r)
    model.register_function(compute_hjb_kappa)
    model.register_function(compute_consistency_kappa)
    model.register_function(compute_kappa_penalization)
    model.add_equations([
        "z = SV[:, :-1]",
        "z_last = 1 - torch.sum(z, dim=1).unsqueeze(1)",
        "z_all = torch.cat([z, z_last], dim=1)",
        "dk_dz = k_Jac[:, :, :-1]",
        "dk_dt = k_Jac[:, :, -1]",
        "dk_dzz = k_Hess[:,:,:-1,:-1]",
        "q = compute_q(SV, compute_k)",
        "dq_dz = compute_qz(SV, compute_k)",
        "dq_dzz = compute_qzz(SV, compute_k)",
        "mu_z_geos = compute_mu_z_geos(z_all, mu_ys, sig_ys)",
        "sig_z_geos = compute_sig_z_geos(z_all, sig_ys)",
        "mu_z_aris = mu_z_geos * z",
        "sig_z_aris = sig_z_geos * z",
        "mu_1minusz_ari  = -torch.sum(mu_z_aris, axis=1, keepdim=True)",
        "sig_1minusz_ari = -torch.sum(sig_z_aris, axis=1, keepdim=True)",
        "mu_1minusz_geo  = mu_1minusz_ari/z_last",
        "sig_1minusz_geo = sig_1minusz_ari/z_last",
        "mu_qs  = compute_mu_qs(q, dq_dz, dq_dzz, mu_z_aris, sig_z_aris)",
        "sig_qs = compute_sig_qs(q, dq_dz, sig_z_aris)",
        "r = compute_r(rho, gamma, mu_ys, sig_ys, z, z_last)",
        "mu_z_geos_all = torch.cat([mu_z_geos, mu_1minusz_geo], axis=1)",
        "sig_z_geos_all = torch.cat([sig_z_geos, sig_1minusz_geo], axis=1)",
        "zetas = gamma * z_all * sig_ys",
        "mu_kappas = mu_z_geos_all - mu_qs + sig_qs * (sig_qs - sig_z_geos_all)",
        "sig_kappas = sig_z_geos_all - sig_qs",
    ])

    model.add_endog_equation("0=compute_kappa_penalization(k)")
    model.add_hjb_equation("compute_hjb_kappa(k, dk_dt, dk_dz, dk_dzz, mu_z_aris, sig_z_aris, mu_kappas)")
    model.add_hjb_equation("compute_consistency_kappa(k, dk_dz, sig_z_aris, sig_kappas)")
    return model

def compute_memory_logs(mem_log_fn):
    n_trees = MU_SIGS.keys()
    print("{0:=^80}".format("Memory"))
    df = pd.DataFrame(columns=["n_tree", "cuda_memory_total", "flops_total", "rar_cuda_memory_total", "rar_flops_total"])
    for idx, n_tree in enumerate(n_trees):
        print("{0:=^40}".format(f"Training {n_tree}"))
        try:
            gc.collect()
            torch.cuda.empty_cache()
            mu_sig = MU_SIGS[n_tree]
            mu_sig_tensor = torch.tensor(mu_sig, dtype=torch.float32, device=device).reshape(1, -1)
            curr_params = BASE_PARAMS | {"mu_ys": mu_sig_tensor, "sig_ys": mu_sig_tensor}
            curr_dir = os.path.join(BASE_DIR, f"temp")
            # get memory
            torch.cuda.reset_peak_memory_stats()
            model = get_model_for_memory(n_tree, curr_params)
            model.train_model(curr_dir, "model.pt", True)
            peak_mem_usage = torch.cuda.max_memory_allocated() / 1024**2
            gc.collect()
            torch.cuda.empty_cache()
            df.loc[idx, "n_tree"] = n_tree
            df.loc[idx, "cuda_memory_total"] = peak_mem_usage

            try:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_flops=True) as prof:
                    with record_function("single_step"):
                        model.train_model(curr_dir, "model.pt", True)
                key_avgs = prof.key_averages()
                total_flops = 0
                for i in range(len(key_avgs)):
                    total_flops += key_avgs[i].flops
                df.loc[idx, "flops_total"] = total_flops / 10**9
                del prof
            except:
                pass
            del model
            gc.collect()
            torch.cuda.empty_cache()


            gc.collect()
            torch.cuda.empty_cache()
            # get memory
            torch.cuda.reset_peak_memory_stats()
            model_rar = get_model_for_memory(n_tree, curr_params, rar=True)
            model_rar.train_model(curr_dir, "model_rar.pt", True)
            peak_mem_usage = torch.cuda.max_memory_allocated() / 1024**2
            gc.collect()
            torch.cuda.empty_cache()
            df.loc[idx, "rar_cuda_memory_total"] = peak_mem_usage

            try:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_flops=True) as prof:
                    with record_function("single_step"):
                        model_rar.train_model(curr_dir, "model_rar.pt", True)
                key_avgs = prof.key_averages()
                total_flops = 0
                for i in range(len(key_avgs)):
                    total_flops += key_avgs[i].flops
                df.loc[idx, "rar_flops_total"] = total_flops / 10**9
                del prof
            except:
                pass
            del model_rar
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print("Error", e)
            break
    df.to_csv(mem_log_fn, index=False)

def format_memory_flops(mem_log: pd.DataFrame, plot_dir):
    mem_log = mem_log.set_index("n_tree")
    cols_level1 = ["CUDA Memory (MB)", r"FLOPS ($\times 10^9$)"]
    cols_level2 = ["2-Tree", "3-Tree", "5-Tree", "10-Tree", "20-Tree", "50-Tree"]
    cols = pd.MultiIndex.from_tuples(product(cols_level1, cols_level2))
    idx = ["Time-stepping", "Our Method"]

    idx_col_map = {
        ("Time-stepping", "CUDA Memory (MB)"): "cuda_memory_total",
        ("Time-stepping", r"FLOPS ($\times 10^9$)"): "flops_total",
        ("Our Method", "CUDA Memory (MB)"): "rar_cuda_memory_total",
        ("Our Method", r"FLOPS ($\times 10^9$)"): "rar_flops_total",
    }

    res_df = pd.DataFrame(index=idx, columns=cols)
    for (row, c1), v in idx_col_map.items():
        for n_tree in [2, 3, 5, 10, 20, 50]:
            curr_val = mem_log.loc[n_tree, v]
            if pd.notna(curr_val):
                res_df.loc[row, (c1, f"{n_tree}-Tree")] = f"{curr_val:.2f}"
            else:
                res_df.loc[row, (c1, f"{n_tree}-Tree")] = ""
    
    ltx = res_df.style.to_latex(column_format="l" + "c" * len(cols), hrules=True, multicol_align="c")
    ltx = ltx.replace(r"\midrule", r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}")

    with open(f"{plot_dir}/tree_memory.tex", "w") as f:
        f.write(ltx)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training the models to get runtimes
    for n_tree in MU_SIGS:
        curr_base_dir = os.path.join(BASE_DIR, f"tree_{n_tree}")
        plot_dir = os.path.join(curr_base_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        mu_sig = MU_SIGS[n_tree]
        mu_sig_tensor = torch.tensor(mu_sig, dtype=torch.float32, device=device).reshape(1, -1)
        curr_params = BASE_PARAMS | {"mu_ys": mu_sig_tensor, "sig_ys": mu_sig_tensor}

        print("{0:=^80}".format(f"Tree {n_tree} base timestep"))
        curr_dir = os.path.join(curr_base_dir, "timestep")
        
        model = get_model(n_tree, curr_params, seed=0, rar=False)
        if not os.path.exists(f"{curr_dir}/model.pt"):
            model.train_model(curr_dir, f"model.pt", True)
        
        curr_dir = os.path.join(curr_base_dir, "timestep_rar")
        model = get_model(n_tree, curr_params, seed=0, rar=True)
        if not os.path.exists(f"{curr_dir}/model.pt"):
            model.train_model(curr_dir, f"model.pt", True)
        gc.collect()
        torch.cuda.empty_cache()
    
    print("{0:=^80}".format("timing memory loss"))
    plot_timing(BASE_DIR)
    mem_log_fn = f"{BASE_DIR}/tree_memory.csv"
    if not os.path.exists(mem_log_fn):
        compute_memory_logs(mem_log_fn)
    mem_log = pd.read_csv(mem_log_fn)
    format_memory_flops(mem_log, BASE_DIR)