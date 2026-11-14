import gc
import os

import deepxde as dde
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "./models"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

params_base = {
    "gamma"             : 5.0,      # Household risk aversion
    "rho"               : 0.05,     # Fund discount rate
    "nn_width"          : 80,       # Neural network width
    "nn_num_layers"     : 4,        # Neural network layers
    "lr"                : 0.0005,    # Learning rate
    "batch_size": 200
}
num_tree_mu_sig = {
    2: [0.02, 0.05],
    3: [0.02, 0.05, 0.08],
    5: [0.02, 0.05, 0.08, 0.11, 0.14],
    10: [0.01 * i for i in range(1, 11)],
    20: [0.01 * i for i in range(1, 21)],
    50: [0.01 * i for i in range(1, 51)],
    100: [0.01 * i for i in range(1, 101)],
} 

def deep_xde_pde(z, kappa_vec):
    '''
    z: (batch_size, dim-1)
    kappa_vec: (batch_size, dim)
    '''
    b = z.shape[0]
    z_dim = z.shape[1]
    o_dim = kappa_vec.shape[1]
    z_last = 1 - torch.sum(z, dim=1).unsqueeze(1)
    z_all = torch.cat([z, z_last], dim=1)
    q_vec = z_all / kappa_vec
    dkappa_dz = torch.zeros((b, o_dim, z_dim))
    dq_dz = torch.zeros((b, o_dim, z_dim))
    dkappa_dzz = torch.zeros((b, o_dim, z_dim, z_dim))
    dq_dzz = torch.zeros((b, o_dim, z_dim, z_dim))
    for i in range(o_dim):
        for j in range(z_dim):
            dkappa_dz[:, i, j] = dde.grad.jacobian(kappa_vec, z, i, j).reshape(-1)
            dq_dz[:, i, j] = dde.grad.jacobian(q_vec, z, i, j).reshape(-1)
            for k in range(z_dim):
                dkappa_dzz[:, i, j, k] = dde.grad.hessian(kappa_vec, z, i, j, k).reshape(-1)
                dq_dzz[:, i, j, k] = dde.grad.hessian(q_vec, z, i, j, k).reshape(-1)
    
    # Compute dynamics of z
    mu_ys = torch.tensor(curr_params["mu_ys"], device=device).unsqueeze(0)
    sig_ys = torch.tensor(curr_params["sig_ys"], device=device).unsqueeze(0)

    mu_z_geos = (
        mu_ys[:, :-1] 
        - torch.sum(mu_ys * z_all, dim=1, keepdim=True) 
        + torch.sum(sig_ys * z_all, dim=1, keepdim=True) 
            * (torch.sum(sig_ys * z_all, dim=1, keepdim=True) - sig_ys[:, :-1])
    ) # (batch, N-1)
    sig_z_geos = (
        sig_ys[:, :-1]
        - torch.sum(sig_ys * z_all, dim=1, keepdim=True)
    ) # (batch, N-1)
    mu_z_aris = mu_z_geos * z # (batch, N-1)
    sig_z_aris = sig_z_geos * z # (batch, N-1)

    mu_1minusz_ari  = -torch.sum(mu_z_aris, axis=1, keepdim=True)
    sig_1minusz_ari = -torch.sum(sig_z_aris, axis=1, keepdim=True)
    mu_1minusz_geo  = mu_1minusz_ari/z_last
    sig_1minusz_geo = sig_1minusz_ari/z_last

    # mu_z_aris, sig_z_aris (batch, N-1)
    mu_qs = (torch.einsum("bnj, bj -> bn", dq_dz, mu_z_aris)
        + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dq_dzz, sig_z_aris)
    ) / q_vec
    sig_qs = torch.einsum("bnj, bj -> bn", dq_dz, sig_z_aris) / q_vec

    r = (curr_params["rho"] 
    + curr_params["gamma"] * (torch.sum(mu_ys[:, :-1] * z, dim=1, keepdim=True) + mu_ys[:, -1:] * z_last)
    - 0.5 * curr_params["gamma"] * (curr_params["gamma"] + 1) * (torch.sum(sig_ys[:, :-1]**2 * z**2, dim=1, keepdim=True) + sig_ys[:, -1:]**2 * z_last**2)
    )

    mu_z_geos_all = torch.cat([mu_z_geos, mu_1minusz_geo], axis=1)
    sig_z_geos_all = torch.cat([sig_z_geos, sig_1minusz_geo], axis=1)
    zetas = curr_params["gamma"] * z_all * sig_ys
    mu_kappas = mu_z_geos_all - mu_qs + sig_qs * (sig_qs - sig_z_geos_all)
    sig_kappas = sig_z_geos_all - sig_qs

    hjb_kappas = (torch.einsum("bnj, bj -> bn", dkappa_dz, mu_z_aris)
        + 0.5 * torch.einsum("bj, bnjk, bk -> bn", sig_z_aris, dkappa_dzz, sig_z_aris)
        - torch.einsum("bn, bn -> bn", mu_kappas, kappa_vec)
    )
    consistency_kappas = (torch.einsum("bnj, bj -> bn", dkappa_dz, sig_z_aris)
        - torch.einsum("bn, bn -> bn", sig_kappas, kappa_vec)
    )
    return [
        hjb_kappas,
        consistency_kappas
    ]



def get_deepxde_result(dim: int):
    '''
    X: entire grid
    X1: boundary T=1
    '''
    geom = dde.geometry.Hypercube([0]*(dim-1), [1]*(dim-1))
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(
        geom,
        deep_xde_pde,
        [],
        num_domain=100,
        train_distribution="pseudo"
    )

    layer_size = [dim-1] + [80] * 4 + [dim]
    activation = "silu"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=True, with_flops=True) as prof:
        with record_function("single_step"):
            model.train(iterations=1)
    key_avgs = prof.key_averages()
    main_loop_res = None
    total_flops = 0
    for i in range(len(key_avgs)):
        total_flops += key_avgs[i].flops
        if main_loop_res is None and "single_step" in key_avgs[i].key:
            main_loop_res = key_avgs[i]
    if hasattr(main_loop_res, "self_cuda_memory_usage"):
        mem_usage = main_loop_res.self_cuda_memory_usage / 1024**2
    elif hasattr(main_loop_res, "self_device_memory_usage"):
        mem_usage = main_loop_res.self_device_memory_usage / 1024**2
    peak_mem_usage = torch.cuda.max_memory_allocated() / 1024**2
    res_dict = {
        "n_dim": dim,
        "cuda_memory_total": peak_mem_usage,
        "flops_total": total_flops / 10**9
    }
    return res_dict


if __name__ == "__main__":
    os.environ["DDE_BACKEND"] = "pytorch"
    dde.backend.set_default_backend("pytorch")
    mem_log_fn_deepxde = f"{BASE_DIR}/deep_xde_tree_memory.csv"
    if not os.path.exists(mem_log_fn_deepxde):
        df = pd.DataFrame(columns=["n_dim", "cuda_memory_total", "flops_total"])
        for idx, n_dim in enumerate(list(num_tree_mu_sig.keys())):
            print("{0:=^40}".format(f"Training {n_dim}"))
            curr_params = params_base.copy()
            curr_params["n_trees"] = n_dim
            curr_params["mu_ys"] = num_tree_mu_sig[n_dim]
            curr_params["sig_ys"] = num_tree_mu_sig[n_dim]
            curr_params["epoch"] = 10
            try:
                res = get_deepxde_result(n_dim)
            except Exception as e:
                print("Error", e)
                break
            for k in df.columns:
                df.loc[idx, k] = res[k]
            gc.collect()
            torch.cuda.empty_cache()
            break
        df.to_csv(mem_log_fn_deepxde, index=False)