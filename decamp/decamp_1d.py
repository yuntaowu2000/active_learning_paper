import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.func import hessian, jacrev, vmap
from tqdm import tqdm

from deep_macrofin import Comparator, OptimizerType, PDEModel, set_seeds


class PDEModelActiveSample(PDEModel): 
    def __init__(self, name, config = ..., latex_var_mapping = {}): 
        super().__init__(name, config, latex_var_mapping) 
        self.sample = self.active_sample 
    
    def active_sample(self, epoch): 
        if epoch % 100 == 0 and epoch > 0: # 
            band_size = 50 
            cstar = self.variable_val_dict["cstar"].detach() # (1, 1) 
            band_offsets = torch.randn((band_size, 1), device=self.device) * 0.05 
            band = cstar + band_offsets # shape (B, 50)
            # check epoch > 0 so we don't need to resample in the first epoch 
            self.anchor_points = band 
        if epoch == self.num_epochs - 1: 
            self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device) 
            # reset so we don't need to save anything 
        sv = self.sample_uniform(epoch) 
        return torch.vstack((sv, self.anchor_points)) 

def compute_hjb(c, F, F_Jac, F_Hess, r, mu, alpha, carry_cost, siga, sigx, rho):
    first_order = (alpha + c * (r - carry_cost - mu)) * F_Jac.reshape(-1, 1)
    # need to change this for higher dimension
    second_order = (siga**2 * c**2 - 2 * rho * siga * sigx * c + sigx**2) * F_Hess.reshape(-1, 1) 
    value_term = (r - mu) * F
    return first_order + 0.5 * second_order - value_term    

def compute_bc(compute_F, c, p, phi, omega, alpha, r, mu):
    F = compute_F(c)
    first_term = torch.max(F - p * (c + phi), dim=0).values
    second_term = torch.tensor(omega * alpha / (r - mu), device=first_term.device)
    return torch.maximum(first_term, second_term).reshape(-1,1)

def compute_opt(compute_F, cstar, alpha, r, mu, carry_cost):
    '''
    In this function, we try to find the place F(c*)=(alpha+c(r-carry_cost-mu))/(r-mu)

    and enforce that F'(c*)=1
    '''
    f_cstar = compute_F(cstar)
    f_c_cstar = vmap(jacrev(compute_F))(cstar).reshape(-1, 1)

    m = alpha + cstar * (r - carry_cost - mu)
    l = r - mu
    expected_f_cstar = m / l

    # check the position where F = m/l
    # enforce F(c*)=(alpha+c(r-carry_cost-mu))/(r-mu)
    # and F'(c*)=1
    error1 = torch.mean((f_cstar - expected_f_cstar) ** 2)
    error2 = torch.mean((f_c_cstar - 1.) ** 2)
    return error1 + error2

def compute_opt(compute_F, cstar, alpha, r, mu, carry_cost):
    '''
    In this function, we try to find the place F(c*)=(alpha+c(r-carry_cost-mu))/(r-mu)

    and enforce that F'(c*)=1
    '''
    f_cstar = compute_F(cstar)
    f_c_cstar = vmap(jacrev(compute_F))(cstar).reshape(-1, 1)

    m = alpha + cstar * (r - carry_cost - mu)
    l = r - mu
    expected_f_cstar = m / l

    # check the position where F = m/l
    # enforce F(c*)=(alpha+c(r-carry_cost-mu))/(r-mu)
    # and F'(c*)=1
    error1 = torch.mean((f_cstar - expected_f_cstar) ** 2)
    error2 = torch.mean((f_c_cstar - 1.) ** 2)
    return error1 + error2

def compute_opt_active(compute_F, cstar, alpha, r, mu, carry_cost):
    delta= 0.02
    steps = 10
    band_offsets = torch.linspace(-delta, delta, steps, device=cstar.device).reshape(-1,1)
    band = cstar + band_offsets
    f_band = compute_F(band) 
    f_band_jac = vmap(jacrev(compute_F))(band) 
    m = alpha + band * (r - carry_cost - mu)
    l = r - mu
    expected_f_band = m / l 

    weights = 1 - torch.abs(band_offsets) / delta
    
    error1 = torch.mean(weights * (f_band - expected_f_band) ** 2)
    error2 = torch.mean(weights * (f_band_jac - 1.) ** 2)
    return error1 + error2

def get_model(
    model_name,
    c_grid,
    params,
    active=False,
):
    set_seeds(0)
    model_fn = "model_active" if active else "model"
    if active:
        model = PDEModelActiveSample(model_name, config={"num_epochs": 20000, "optimizer_type": OptimizerType.Adam})
    else:
        model = PDEModel(model_name, config={"num_epochs": 20000, "optimizer_type": OptimizerType.Adam})
    model.set_state(["c"], {"c": c_grid})
    model.add_params(params)
    model.add_learnable_param("cstar", [[0.3]])
    if active:
        model.register_functions([compute_hjb, compute_bc, compute_opt_active])
    else:
        model.register_functions([compute_hjb, compute_bc, compute_opt])
    model.add_endog("F", config={"batch_jac_hes": True})
    model.add_hjb_equation("compute_hjb(c, F, F_Jac, F_Hess, r, mu, alpha, carry_cost, siga, sigx, rho)")
    if active:
        model.add_hjb_equation("compute_opt_active(compute_F, cstar, alpha, r, mu, carry_cost)", loss_reduction="None")
    else:
        model.add_hjb_equation("compute_opt(compute_F, cstar, alpha, r, mu, carry_cost)", loss_reduction="None")
    model.add_endog_condition("F", 
                            "F(zero)", {"zero": torch.zeros(1, 1, device=model.device)},
                            Comparator.EQ,
                            "compute_bc(compute_F, c_lin, prop, phi, omega, alpha, r, mu)",
                            params | {"c_lin": torch.linspace(c_grid[0], c_grid[1], steps=200, device=model.device).reshape(-1, 1)}
                            )
    if not os.path.exists(f"./models/{model_name}/{model_fn}.pt"):
        model.train_model(f"./models/{model_name}", f"{model_fn}.pt", True)
    model.load_model(torch.load(f"./models/{model_name}/{model_fn}_best.pt", weights_only=False))
    model.eval_model(True)
    return model

def plot_solution(model_active: PDEModel, numerical_sol, plot_dir):
    cstar_model = model_active.variable_val_dict["cstar"].item()
    c_grid_tensor = torch.tensor(numerical_sol["c"], dtype=torch.float32, device=model_active.device).reshape(-1, 1)
    f = model_active.endog_vars["F"].model(c_grid_tensor).detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(numerical_sol["c"], numerical_sol["F"], label="Numerical", linestyle="-.", color="#000000", marker="x", markevery=10)
    ax.plot(numerical_sol["c"], f, label="Our Method", linestyle="-", color="#5492ab")
    ax.vlines(cstar_model, 7, 10, colors="red", linestyles="-.")
    ax.set_xlabel("c", fontsize=16)
    ax.set_ylabel("F(c)", fontsize=16)
    ax.set_ylim(7, 10)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_fit.pdf")
    plt.close()

def plot_loss(model_dir, plot_dir):
    model_loss = pd.read_csv(f"{model_dir}/model_loss.csv")
    model_active_loss = pd.read_csv(f"{model_dir}/model_active_loss.csv")

    # HJB loss
    idx_max = model_loss["hjbeq_1"].idxmax()
    active_idx_max = model_active_loss["hjbeq_1"].idxmax()
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"].values[idx_max:], model_loss["hjbeq_1"].values[idx_max:], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"].values[active_idx_max:], model_active_loss["hjbeq_1"].values[active_idx_max:], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_hjb.pdf")
    plt.close()

    # loss at cstar
    idx_max = model_loss["hjbeq_2"].idxmax()
    active_idx_max = model_active_loss["hjbeq_2"].idxmax()
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"].values[idx_max:], model_loss["hjbeq_2"].values[idx_max:], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"].values[active_idx_max:], model_active_loss["hjbeq_2"].values[active_idx_max:], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=14)
    # ax.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_cstar.pdf")
    plt.close()

    # volatility of loss at cstar
    window = 100

    model_loss["rolling_ratio"] = (
        model_loss["hjbeq_2"].rolling(window=window, min_periods=1).std()
        / (model_loss["hjbeq_2"].rolling(window=window, min_periods=1).mean() + 1e-8)
    )
    model_active_loss["rolling_ratio"] = (
        model_active_loss["hjbeq_2"].rolling(window=window, min_periods=1).std()
        / (model_active_loss["hjbeq_2"].rolling(window=window, min_periods=1).mean() + 1e-8)
    )
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.plot(model_loss["epoch"], model_loss["rolling_ratio"], label="Basic Neural Network", linestyle="--", color="#D9D9D9")
    ax.plot(model_active_loss["epoch"], model_active_loss["rolling_ratio"], label="Our Method", linestyle="-", color="#5492ab")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=14)
    # ax.legend(loc="upper right", frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/decamp_loss_cstar_stability.pdf")
    plt.close()

def compute_errors_single(model: PDEModel, numerical_sol):
    cstar_model = model.variable_val_dict["cstar"].item()
    c_grid_tensor = torch.tensor(numerical_sol["c"], dtype=torch.float32, device=model.device).reshape(-1, 1)
    f = model.endog_vars["F"].model(c_grid_tensor).detach().cpu().numpy().reshape(-1)

    numerical_f = numerical_sol["F"]
    numerical_cstar = numerical_sol["cstar"]

    return {
        "f_mse": np.mean((f-numerical_f)**2),
        "c_mae": np.mean(np.abs(cstar_model - numerical_cstar)),
    }

def format_sci(x):
    sci_str = f"{x:.2e}"  # Convert to scientific notation
    base, exp = sci_str.split("e")  # Split into base and exponent
    exp = int(exp)  # Convert exponent to integer to remove leading zeros and '+'
    if exp == 0:
        return f"{base}"
    else:
        return f"${base} \\times 10^{{{exp}}}$"

def compute_errors(model: PDEModel, model_active: PDEModel, numerical_sol, plot_dir):
    model_errors = compute_errors_single(model, numerical_sol)
    model_active_errors = compute_errors_single(model_active, numerical_sol)

    res_df = pd.DataFrame(index=["Baseline", "Active"], columns=["MSE($F$)", "MAE($c^*$)"])
    for m, err in [("Baseline", model_errors), ("Active", model_active_errors)]:
        res_df.loc[m, "MSE($F$)"] = format_sci(err["f_mse"])
        res_df.loc[m, "MAE($c^*$)"] = format_sci(err["c_mae"])
    
    ltx = res_df.style.to_latex(column_format="l" + "c" * len(res_df.columns), hrules=True)
    with open(f"{plot_dir}/loss_liquidation.tex", "w") as f:
        f.write(ltx)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    plot_dir = "models/plots"
    os.makedirs(plot_dir, exist_ok=True)
    c_grid = [0.01, 1.0]
    params = {
        "mu": 0.01,
        "siga": 0.25,
        "sigx": 0.12,
        "rho": -0.2,
        "carry_cost": 0.02,
        "alpha": 0.18,
        "phi": 1.002,
        "prop": 1.06,
        "liquidation": 100,
        "omega": 0.55,
        "r": 0.03,
        "I": 10,
        "xi": 0.015,
        "eps": 1,
    }

    model = get_model("decamp_liquidation", c_grid, params, False)
    model_active = get_model("decamp_liquidation", c_grid, params, True)

    numerical_sol = np.load("models/liquidation.npz", allow_pickle=True)
    compare = numerical_sol["compare"].item()

    plot_solution(model_active, compare, plot_dir)
    plot_loss("models/decamp_liquidation", plot_dir)
    compute_errors(model, model_active, compare, plot_dir)


    
