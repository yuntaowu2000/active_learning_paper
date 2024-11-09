import gc
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, Comparator, Constraint, LayerType,
                           LossReductionMethod, OptimizerType, PDEModel,
                           SamplingMethod, System, set_seeds)

plt.rcParams["font.size"] = 15

BASE_DIR = "./models/FreeBoundary1D"
os.makedirs(BASE_DIR, exist_ok=True)

LATEX_VAR_MAPPING = {
    r"\sigma_t^q": "sigq",
    r"\sigma_t^\theta": "sigtheta",
    r"\sigma_t^\eta": "sige",
    r"\mu_t^\eta": "mue",
    r"\mu_t^q": "muq",
    r"\mu_t^\theta": "mutheta",


    r"\rho": "rho",
    r"\underline{a}": "ah",
    r"\underline{\delta}": "deltah",
    r"\delta": "deltae",
    r"\sigma": "sig",
    r"\kappa": "kappa",

    r"\eta": "e",

    r"\theta": "theta",
    r"\psi": "psi",
    r"\iota": "iota",
    r"\Phi": "phi",

}

PROBLEM_DOMAIN = {
    "e": [0.01, 0.99]
}

PARAMS = {
    "sig": .1,
    "deltae": .05,
    "deltah": .05,
    "rho": .06,
    "r": .05,
    "a": .11,
    "ah": .07,
    "kappa": 2,
}

MODEL_CONFIGS = {
    "MLP": {
        "q": {
            "positive": True,
            "activation_type": ActivationType.SiLU,
        },
        "psi": {
            "positive": True, 
            "activation_type": ActivationType.SiLU,
        }
    },
    "KAN": {
        "q": {
            "hidden_units": [1, 1],
            "layer_type": LayerType.KAN,
            "activation_type": ActivationType.SiLU,
        },
        "psi": {
            "hidden_units": [1, 1],
            "layer_type": LayerType.KAN,
            "activation_type": ActivationType.SiLU,
        }
    }
}

TRAINING_CONFIGS = {
    "MLP": {
        "sampling_method": SamplingMethod.FixedGrid, 
        "batch_size": 1000,
        "num_epochs": 20000,
        "optimizer_type": OptimizerType.Adam,
    },
    "KAN": {
        "sampling_method": SamplingMethod.FixedGrid, 
        "batch_size": 1000,
        "num_epochs": 200,
        "lr": 1,
    }
}

EQUATIONS = {
    "region1": [
        r"$\iota = \frac{q^2-1}{ 2 * \kappa}$",
        r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (\psi - \eta)} - \sigma$",
        r"$\sigma_t^\eta = \frac{\psi - \eta}{\eta} * (\sigma + \sigma_t^q)$",
        r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) * (\underline{\delta} - \delta) - \rho$",
    ],
    "region2": [
        r"$\iota = \frac{q^2-1}{ 2 * \kappa}$",
        r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (1 - \eta)} - \sigma$",
        r"$\sigma_t^\eta = \frac{1 - \eta}{\eta} * (\sigma + \sigma_t^q)$",
        r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} - \rho$"
    ]
}

ENDOG_EQUATIONS = {
    "region1": [
        (r"$(r*(1-\eta) + \rho * \eta) * q = \psi * a + (1-\psi) * \underline{a} - \iota$", 1.0),
        (r"$(\sigma + \sigma_t^q) ^2 * \frac{q * (\psi - \eta)}{\eta * (1-\eta)} = (a - \underline{a}) + (\underline{\delta} - \delta) * q$", 2.0)
    ],
    "region2": [
        (r"$(r*(1-\eta) + \rho * \eta) * q = a - \iota$", 1.0)
    ]
}

def setup_model_system(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type {model_type} not supported.")
    model_config = MODEL_CONFIGS[model_type]
    training_config = TRAINING_CONFIGS[model_type]
    set_seeds(0)
    model = PDEModel("BruSan14_log_utility", 
                     config=training_config,
                    latex_var_mapping=LATEX_VAR_MAPPING)
    model.set_state(["e"], PROBLEM_DOMAIN)
    model.add_endogs(["q", "psi"], configs=model_config)
    model.add_params(PARAMS)
    return model

def setup_model_split(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type {model_type} not supported.")
    model_config = MODEL_CONFIGS[model_type]
    training_config = TRAINING_CONFIGS[model_type]
    set_seeds(0)
    model_region1 = PDEModel("BruSan14_log_utility_region1", 
                     config=training_config,
                    latex_var_mapping=LATEX_VAR_MAPPING)
    model_region1.set_state(["e"], PROBLEM_DOMAIN)
    model_region1.add_endogs(["q", "psi"], configs=model_config)
    model_region1.add_params(PARAMS)

    set_seeds(0)
    model_region2 = PDEModel("BruSan14_log_utility_region2", 
                     config=training_config,
                    latex_var_mapping=LATEX_VAR_MAPPING)
    model_region2.set_state(["e"], PROBLEM_DOMAIN)
    model_region2.add_endogs(["q"], configs={"q": model_config["q"]})
    model_region2.add_params(PARAMS)
    return {"region1": model_region1, "region2": model_region2}
    

def add_boundary_conditions(model: PDEModel):
    model.add_endog_condition("q", 
                              "q(SV)", 
                              {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "-kappa*r + (kappa**2*r**2 + 1 + 2*ah*kappa)**0.5", {"r": 0.05, "ah": .07, "kappa": 2},
                              label="q_min", weight=100)
    model.add_endog_condition("q", 
                                "q(SV)", 
                                {"SV": torch.ones((1, 1))},
                                Comparator.EQ,
                                "-kappa*rho + (kappa**2*rho**2 + 1 + 2*a*kappa)**0.5", {"rho": 0.06, "a": .11, "kappa": 2},
                                label="q_max", weight=100)
    model.add_endog_condition("psi", 
                                "psi(SV)", 
                                {"SV": torch.zeros((1, 1))},
                                Comparator.EQ,
                                "0", {},
                                label="psi_min", weight=100)
    model.add_endog_condition("psi", 
                                "psi(SV)", 
                                {"SV": torch.ones((1, 1))},
                                Comparator.EQ,
                                "1", {},
                                label="psi_max", weight=100)
    return model

def add_model_equations(model: Union[PDEModel, Dict[str, PDEModel]]):
    if isinstance(model, PDEModel):
        # this is where we try to train a single model across the entire domain
        sys1 = System([Constraint("psi", Comparator.LT, "1", label="crisis_constraint")], label="crisis_region", latex_var_mapping=LATEX_VAR_MAPPING)
        for eq in EQUATIONS["region1"]:
            sys1.add_equation(eq)
        for endog_eq in ENDOG_EQUATIONS["region1"]:
            sys1.add_endog_equation(endog_eq[0], weight=endog_eq[1], loss_reduction=LossReductionMethod.SSE)

        sys2 = System([Constraint("psi", Comparator.GEQ, "1", label="crisis_constraint")], label="normal_region", latex_var_mapping=LATEX_VAR_MAPPING)
        for eq in EQUATIONS["region2"]:
            sys2.add_equation(eq)
        for endog_eq in ENDOG_EQUATIONS["region2"]:
            sys2.add_endog_equation(endog_eq[0], weight=endog_eq[1], loss_reduction=LossReductionMethod.SSE)
        
        model.add_system(sys1)
        model.add_system(sys2)
        return model
    else:
        for k in ["region1", "region2"]:
            for eq in EQUATIONS[k]:
                model[k].add_equation(eq)
            for endog_eq in ENDOG_EQUATIONS[k]:
                model[k].add_endog_equation(endog_eq[0], weight=endog_eq[1], loss_reduction=LossReductionMethod.SSE)
        return model

def plot_models(model_system: PDEModel, 
                model_split: Dict[str, PDEModel],
                output_folder: str,
                vars_to_plot_ltx: List[str],
                x_var_ltx = r"$\eta$",):
    
    ## Finite Difference Solution
    df = pd.read_csv("models/BruSan14_log_utility_solution-raw.csv")
    plot_args_base = []
    x_plot_base = df["e"]
    for var in vars_to_plot_ltx:
        plot_args_base.append({
            "y": df[LATEX_VAR_MAPPING.get(var, var)],
            "ylabel": rf"${var}$",
            "title": rf"${var}$ vs. ${x_var_ltx}$"
        })

    ## Baseline no split solution
    N = 1000
    SV = torch.linspace(PROBLEM_DOMAIN["e"][0], PROBLEM_DOMAIN["e"][1], N, device=model_system.device).reshape(-1, 1)
    x_plot = SV.detach().cpu().numpy().reshape(-1)

    plot_dict_model_system = []
    for i, sv_name in enumerate(model_system.state_variables):
        model_system.variable_val_dict[sv_name] = SV[:, i:i+1]
    model_system.update_variables(SV)
    for sys_name in model_system.systems:
        model_system.systems[sys_name].eval({}, model_system.variable_val_dict)
    psi_system = model_system.variable_val_dict["psi"].detach().cpu().numpy().reshape(-1)
    q = model_system.variable_val_dict["q"].detach().cpu().numpy().reshape(-1)
    index_unconstrain = (psi_system < 1)
    index_constrain = (psi_system >= 1)
    for var in vars_to_plot_ltx:
        if LATEX_VAR_MAPPING.get(var, var) == "q":
            res = q
        elif LATEX_VAR_MAPPING.get(var, var) == "psi":
            res = psi_system
        else:
            region1_sol = model_system.systems["system_crisis_region"].variable_val_dict[LATEX_VAR_MAPPING.get(var, var)].detach().cpu().numpy().reshape(-1)
            region2_sol =  model_system.systems["system_normal_region"].variable_val_dict[LATEX_VAR_MAPPING.get(var, var)].detach().cpu().numpy().reshape(-1)
            res = region1_sol * index_unconstrain + region2_sol * index_constrain
        plot_dict_model_system.append({
            "y": res,
            "ylabel": rf"${var}$",
            "title": rf"${var}$ vs. ${x_var_ltx}$"
        })

    ## Split Solution
    plot_dict_model_split = []
    for k in ["region1", "region2"]:
        for i, sv_name in enumerate(model_split[k].state_variables):
            model_split[k].variable_val_dict[sv_name] = SV[:, i:i+1]
        model_split[k].update_variables(SV)
    psi_region1 = model_split["region1"].variable_val_dict["psi"].detach().cpu().numpy().reshape(-1)
    index_unconstrain = (psi_region1 < 1)
    index_constrain = (psi_region1 >= 1)

    for var in vars_to_plot_ltx:
        region1_sol = model_split["region1"].variable_val_dict[LATEX_VAR_MAPPING.get(var, var)].detach().cpu().numpy().reshape(-1)
        if LATEX_VAR_MAPPING.get(var, var) == "psi":
            region2_sol = 1
        else:
            region2_sol =  model_split["region2"].variable_val_dict[LATEX_VAR_MAPPING.get(var, var)].detach().cpu().numpy().reshape(-1)
        res = region1_sol * index_unconstrain + region2_sol * index_constrain
        plot_dict_model_split.append({
            "y": res,
            "ylabel": rf"${var}$",
            "title": rf"${var}$ vs. ${x_var_ltx}$"
        })
    
    for i, var in enumerate(vars_to_plot_ltx):
        fn = os.path.join(output_folder, f"{LATEX_VAR_MAPPING.get(var, var)}.jpg")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.plot(x_plot_base, plot_args_base[i]["y"], linestyle="-.", label="Finite Difference")

        ax.plot(x_plot, plot_dict_model_system[i]["y"], linestyle="--", label="Our Method (Single Neural Network)")

        ax.plot(x_plot, plot_dict_model_split[i]["y"], linestyle="-", label="Our Method (Splitted)")
        ax.set_ylabel(plot_args_base[i]["ylabel"])
        ax.set_xlabel(x_var_ltx)
        ax.legend()
        ax.set_title(plot_args_base[i]["title"])
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()


if __name__ == "__main__":
    for model_type in ["MLP", "KAN"]:
        print(f"{model_type:=^80}")
        plot_dir = os.path.join(BASE_DIR, model_type, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        model_system = setup_model_system(model_type)
        model_system = add_boundary_conditions(model_system)
        model_system = add_model_equations(model_system)

        if not os.path.exists(os.path.join(BASE_DIR, model_type, "system", "model_best.pt")):
            model_system.train_model(os.path.join(BASE_DIR, model_type, "system"), "model.pt", full_log=True)
        model_system.load_model(torch.load(os.path.join(BASE_DIR, model_type, "system", "model_best.pt"), weights_only=False))

        model_split = setup_model_split(model_type)
        model_split["region1"] = add_boundary_conditions(model_split["region1"])
        model_split = add_model_equations(model_split)

        for k in ["region1", "region2"]:
            if not os.path.exists(os.path.join(BASE_DIR, model_type, "split", f"{k}_best.pt")):
                model_split[k].train_model(os.path.join(BASE_DIR, model_type, "split"), f"{k}.pt", full_log=True)
            model_split[k].load_model(torch.load(os.path.join(BASE_DIR, model_type, "split", f"{k}_best.pt"), weights_only=False))
        plot_models(model_system, model_split, 
                    plot_dir, ["q", r"\psi", r"\sigma_t^q"], 
                    r"\eta")
        gc.collect()
        torch.cuda.empty_cache()
        if model_type == "KAN":
            x = model_system.sample(0)
            set_seeds(0)
            model_system.endog_vars["q"].model(x)
            model_system.endog_vars["psi"].model(x)
            model_system.endog_vars["q"].model.auto_symbolic()
            model_system.endog_vars["psi"].model.auto_symbolic()
            q_formula = model_system.endog_vars["q"].model.symbolic_formula(floating_digit=4)[0][0]
            psi_formula = model_system.endog_vars["psi"].model.symbolic_formula(floating_digit=4)[0][0]
            with open(os.path.join(BASE_DIR, model_type, "system", "model_formula.txt"), "w") as f:
                f.write(f"q={q_formula}\n")
                f.write(f"psi={psi_formula}\n")

            x = model_split["region1"].sample(0)
            set_seeds(0)
            model_split["region1"].endog_vars["q"].model(x)
            model_split["region1"].endog_vars["psi"].model(x)
            model_split["region1"].endog_vars["q"].model.auto_symbolic()
            model_split["region1"].endog_vars["psi"].model.auto_symbolic()
            q_formula = model_split["region1"].endog_vars["q"].model.symbolic_formula(floating_digit=4)[0][0]
            psi_formula = model_split["region1"].endog_vars["psi"].model.symbolic_formula(floating_digit=4)[0][0]

            model_split["region2"].endog_vars["q"].model(x)
            model_split["region2"].endog_vars["q"].model.auto_symbolic(lib=["x"])
            q_formula_region2 = model_split["region2"].endog_vars["q"].model.symbolic_formula(floating_digit=4)[0][0]

            with open(os.path.join(BASE_DIR, model_type, "split", "model_formula.txt"), "w") as f:
                f.write("Region 1:\n")
                f.write(f"q={q_formula}\n")
                f.write(f"psi={psi_formula}\n")
                f.write("Region 2:\n")
                f.write(f"q={q_formula_region2}\n")
        gc.collect()
        torch.cuda.empty_cache()

        

        
        