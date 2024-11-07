import os
from copy import deepcopy
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, Comparator, OptimizerType, PDEModel,
                           PDEModelTimeStep, SamplingMethod, set_seeds)

plt.rcParams["font.size"] = 15

BASE_DIR = "./models/FreeBoundary2D"
os.makedirs(BASE_DIR, exist_ok=True)

LATEX_VAR_MAPPING = {
    r"J_e": "Je",
    r"J_h": "Jh",
    r"\psi": "psi",
    r"\iota": "iota",
    r"\Phi": "Phi",
    r"\hat{c}_e": "ce",
    r"\hat{c}_h": "ch",

    r"\mu_{ae}": "muae",
    r"\sigma_{ae}": "sigae",
    r"\theta_e": "thetae",
    r"\theta_h": "thetah",
    r"\sigma^{q,k}": "sigqk",
    r"\sigma^{q,a}": "sigqa",
    r"\sigma^{q,k,1}": "sigsigqk",
    r"\sigma^{z,k}": "sigzk",
    r"\sigma^{z,a}": "sigza",
    r"\sigma_e^{J,k}": "sigJe_k",
    r"\sigma_h^{J,k}": "sigJh_k",
    r"\sigma_e^{J,a}": "sigJe_a",
    r"\sigma_h^{J,a}": "sigJh_a",

    r"\zeta_e^k": "zetae_k",
    r"\zeta_e^a": "zetae_a",
    r"\zeta_h^k": "zetah_k",
    r"\zeta_h^a": "zetah_a",
    r"\zeta_e^1": "zetae_1",
    r"\zeta_e^2": "zetae_2",
    r"\zeta_h^1": "zetah_1",
    r"\zeta_h^2": "zetah_2",
    r"\epsilon_e": "epse",
    r"\epsilon_h": "epsh",

    r"\mu^z": "muz",
    r"\mu^q": "muq",
    r"\mu_e^J": "muJe",
    r"\mu_h^J": "muJh",
    r"\mu_e^R": "muRe",
    r"\mu_h^R": "muRh",

    r"\hat{a_e}": "aebar",
    r"a_e": "ae",
    r"a_h": "ah",
    r"\sigma": "sig",
    r"\delta": "delta",
    r"\kappa": "kappa",
    r"\phi": "phi",
    r"\gamma": "gamma",
    r"\rho": "rho",
    r"\lambda_d": "lambdad",
    r"\underline{\chi}": "chi_a",
    r"\chi": "chi",
    r"\bar{z}": "zbar",
}


z_min = 0.01
z_max = 0.99
a_min = 0.1
a_max = 0.2
ae_bar = (a_min + a_max) / 2
a_list = [a_min, ae_bar, a_max]
PARAMS = {
    "ah": 0.03,
    "sig": 0.1,
    "delta": 0.05,
    "kappa": 5,
    "phi": 0.5,
    "gamma": 5,
    "rho": torch.tensor(0.05),
    "lambdad": 0.03,
    "zbar": 0.1,
    "chi_a": 1,
    "v": 2.5,
    "p": 0.01,
    "fl": a_min,
    "fu": a_max,
    "aebar": ae_bar,
}
STATES = ["z", "ae"] 
PROBLEM_DOMAIN = {"z": [z_min, z_max], "ae": [a_min, a_max]}

VARS_TO_PLOT = ["q", "thetae", "thetah", "psi", "sigsigqk", "sigqa", "epse", "epsh",
                "zmuz", "zsigzk", "zsigza", "zetae_k", "zetae_a", "zetah_k", "zetah_a", "Je", "Jh"]
PLOT_ARGS = {
    "q": {"ylabel": r"$q$", "title": r"Price"},
    "thetae": {"ylabel": r"$\theta_e$", "title": r"Portfolio Choice: Experts"},
    "thetah": {"ylabel": r"$\theta_h$", "title": r"Portfolio Choice: Households"},
    "psi": {"ylabel": r"$\psi$", "title": r"Capital Share: Experts"},
    "sigsigqk": {"ylabel": r"$\sigma+\sigma^{q,k}$", "title": r"Price return diffusion (capital shock)"},
    "sigqa": {"ylabel": r"$\sigma^{q,a}$", "title": r"Price return diffusion (productivity shock)"},
    "zmuz": {"ylabel": r"$z\mu^z$", "title": r"Drift of wealth share: Experts"},
    "zsigzk": {"ylabel": r"$z\sigma^{z,k}$", "title": r"Diffusion of wealth share (capital shock)"},
    "zsigza": {"ylabel": r"$z\sigma^{z,a}$", "title": r"Diffusion of wealth share (productivity shock)"},
    "zetae_k": {"ylabel": r"$\zeta_e^k$", "title": r"Experts price of risk: capital shock"},
    "zetae_a": {"ylabel": r"$\zeta_e^a$", "title": r"Experts price of risk: productivity shock"},
    "zetah_k": {"ylabel": r"$\zeta_h^k$", "title": r"Households price of risk: capital shock"},
    "zetah_a": {"ylabel": r"$\zeta_h^a$", "title": r"Households price of risk: productivity shock"},
    "Je": {"ylabel": r"$J_e$", "title": r"Expert Value Function"},
    "Jh": {"ylabel": r"$J_h$", "title": r"Households Value Function"},
}

MODEL_CONFIGS = {
    "Agents": {
        "Je": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
        "Jh": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
    },
    "Endogs": {
        "region1": {
            "q": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
            "psi": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
        },
        "region2": {
            "q": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
        },
        "region3": {
            "q": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4},
            "chi": {"positive": True, "activation_type": ActivationType.SiLU, "hidden_units": [64] * 4}
        }   
    }
}

TRAINING_CONFIGS = {
    "basic": {
        "sampling_method": SamplingMethod.UniformRandom, 
        "batch_size": 500,
        "num_epochs": 20000,
        "optimizer_type": OptimizerType.Adam,
    },
    "timestep": {
        "batch_size": 500, 
        "num_outer_iterations": 50, 
        "num_inner_iterations": 5000,
        "sampling_method": SamplingMethod.UniformRandom, 
        "time_batch_size": 1,
    },
    "timestep_rar": {
        "batch_size": 500, 
        "num_outer_iterations": 50, 
        "num_inner_iterations": 5000,
        "sampling_method": SamplingMethod.RARG, 
        "time_batch_size": 1,
        "refinement_rounds": 5,
    }
}

def get_equations(timestepping=False):
    BASE_EQUATIONS = {
            "region1": [
                r"$\iota &= \frac{q - 1}{\kappa}$",
                r"$\Phi &= \frac{\log ( q)}{\kappa}$",
                r"$\hat{c}_e &= \rho$",
                r"$\hat{c}_h &= \rho$",
                r"$\mu_{ae} &= p * (\hat{a_e} - a_e)$",
                r"$\sigma_{ae} &= v * (fu - a_e) * (a_e - fl)$",
                r"$\theta_e &= \frac{\underline{\chi} * \psi}{z}$",
                r"$\theta_h &= \frac{1 - \underline{\chi} * \psi}{1 - z}$",
                r"$\sigma^{q,k} &= \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\underline{\chi} * \psi - z)} - \sigma$",
                r"$\sigma^{q,a} &= \frac{\frac{1}{q} * \frac{\partial q}{\partial a_e} * \sigma_{ae}}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\underline{\chi} * \psi - z)}$",
                r"$\sigma^{q,k,1} &= \sigma + \sigma^{q,k}$",
                r"$\sigma^{z,k} &= \left(\theta_e - 1\right) * \sigma^{q,k,1}$",
                r"$\sigma^{z,a} &= \left(\theta_e - 1\right) * \sigma^{q,a}$",
                r"$\sigma_e^{J,k} &= \frac{1}{J_e} * \frac{\partial J_e}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_h^{J,k} &= \frac{1}{J_h} * \frac{\partial J_h}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_e^{J,a} &= \frac{1}{J_e} * \left(\frac{\partial J_e}{\partial a_e} * \sigma_{ae} + \frac{\partial J_e}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\sigma_h^{J,a} &= \frac{1}{J_h} * \left(\frac{\partial J_h}{\partial a_e} * \sigma_{ae} + \frac{\partial J_h}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\zeta_e^k &= -(1-\gamma) * \sigma_e^{J,k} + \sigma^{z,k} + \sigma^{q,k} + \gamma * \sigma$",
                r"$\zeta_e^a &= -(1-\gamma) * \sigma_e^{J,a} + \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_h^k &= -(1-\gamma) * \sigma_h^{J,k} - \frac{z}{1-z} * \sigma^{z,k} + \sigma^{q,k,1} + \gamma * \sigma$",
                r"$\zeta_h^a &= -(1-\gamma) * \sigma_h^{J,a} - \frac{z}{1-z} * \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_e^1 &= \zeta_e^k + \phi * \zeta_e^a$",
                r"$\zeta_h^1 &= \zeta_h^k + \phi * \zeta_h^a$",
                r"$\zeta_e^2 &= \zeta_e^a + \phi * \zeta_e^k$",
                r"$\zeta_h^2 &= \zeta_h^a + \phi * \zeta_h^k$",
                r"$\epsilon_e &= \sigma^{q,k,1} * \zeta_e^1 + \sigma^{q,a} * \zeta_e^2$",
                r"$\epsilon_h &= \sigma^{q,k,1} * \zeta_h^1 + \sigma^{q,a} * \zeta_h^2$",
                r"$\mu^z &= \frac{a_e - \iota}{q} - \hat{c}_e+ (\theta_e - 1) * (\sigma^{q,k,1} * (\zeta_e^1 - \sigma^{q,k,1}) + \sigma^{q,a} * (\zeta_e^2 - \sigma^{q,a}) - 2 * \phi * \sigma^{q,k,1} * \sigma^{q,a})+ (1-\underline{\chi}) * (\sigma^{q,k,1} * (\zeta_e^1 - \zeta_h^1) + \sigma^{q,a} * (\zeta_e^2 - \zeta_h^2)) + \frac{\lambda_d}{z} * (\bar{z} - z)$",
                r"$\mu^q &= \frac{1}{q} * \left( \frac{\partial q}{\partial z} * \mu^z * z + \frac{\partial q}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2 q}{\partial z^2} * z^2 * ((\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k} * \sigma^{z,a})\right)+ \frac{1}{q} * \left( \frac{1}{2} * \frac{\partial^2 q}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 q}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a})  * \sigma_{ae} * z\right)$",
            ],
            "region2": [
                r"$\iota &= \frac{q - 1}{\kappa}$",
                r"$\Phi &= \frac{\log ( q)}{\kappa}$",
                r"$\hat{c}_e &= \rho$",
                r"$\hat{c}_h &= \rho$",
                r"$\mu_{ae} &= p * (\hat{a_e} - a_e)$",
                r"$\sigma_{ae} &= v * (fu - a_e) * (a_e - fl)$",
                r"$\theta_e &= \frac{\underline{\chi}}{z}$",
                r"$\theta_h &= \frac{1 - \underline{\chi}}{1 - z}$",
                r"$\sigma^{q,k} &= \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\underline{\chi} - z)} - \sigma$",
                r"$\sigma^{q,a} &= \frac{\frac{1}{q} * \frac{\partial q}{\partial a_e} * \sigma_{ae}}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\underline{\chi} - z)}$",
                r"$\sigma^{q,k,1} &= \sigma + \sigma^{q,k}$",
                r"$\sigma^{z,k} &= \left(\theta_e - 1\right) * \sigma^{q,k,1}$",
                r"$\sigma^{z,a} &= \left(\theta_e - 1\right) * \sigma^{q,a}$",
                r"$\sigma_e^{J,k} &= \frac{1}{J_e} * \frac{\partial J_e}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_h^{J,k} &= \frac{1}{J_h} * \frac{\partial J_h}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_e^{J,a} &= \frac{1}{J_e} * \left(\frac{\partial J_e}{\partial a_e} * \sigma_{ae} + \frac{\partial J_e}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\sigma_h^{J,a} &= \frac{1}{J_h} * \left(\frac{\partial J_h}{\partial a_e} * \sigma_{ae} + \frac{\partial J_h}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\zeta_e^k &= -(1-\gamma) * \sigma_e^{J,k} + \sigma^{z,k} + \sigma^{q,k} + \gamma * \sigma$",
                r"$\zeta_e^a &= -(1-\gamma) * \sigma_e^{J,a} + \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_h^k &= -(1-\gamma) * \sigma_h^{J,k} - \frac{z}{1-z} * \sigma^{z,k} + \sigma^{q,k,1} + \gamma * \sigma$",
                r"$\zeta_h^a &= -(1-\gamma) * \sigma_h^{J,a} - \frac{z}{1-z} * \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_e^1 &= \zeta_e^k + \phi * \zeta_e^a$",
                r"$\zeta_h^1 &= \zeta_h^k + \phi * \zeta_h^a$",
                r"$\zeta_e^2 &= \zeta_e^a + \phi * \zeta_e^k$",
                r"$\zeta_h^2 &= \zeta_h^a + \phi * \zeta_h^k$",
                r"$\epsilon_e &= \sigma^{q,k,1} * \zeta_e^1 + \sigma^{q,a} * \zeta_e^2$",
                r"$\epsilon_h &= \sigma^{q,k,1} * \zeta_h^1 + \sigma^{q,a} * \zeta_h^2$",
                r"$\mu^z &= \frac{a_e - \iota}{q} - \hat{c}_e+ (\theta_e - 1) * (\sigma^{q,k,1} * (\zeta_e^1 - \sigma^{q,k,1}) + \sigma^{q,a} * (\zeta_e^2 - \sigma^{q,a}) - 2 * \phi * \sigma^{q,k,1} * \sigma^{q,a})+ (1-\underline{\chi}) * (\sigma^{q,k,1} * (\zeta_e^1 - \zeta_h^1) + \sigma^{q,a} * (\zeta_e^2 - \zeta_h^2)) + \frac{\lambda_d}{z} * (\bar{z} - z)$",
                r"$\mu^q &= \frac{1}{q} * \left( \frac{\partial q}{\partial z} * \mu^z * z + \frac{\partial q}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2 q}{\partial z^2} * z^2 * ((\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k} * \sigma^{z,a})\right)+ \frac{1}{q} * \left( \frac{1}{2} * \frac{\partial^2 q}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 q}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a})  * \sigma_{ae} * z\right)$",
            ],
            "region3": [
                r"$\iota &= \frac{q - 1}{\kappa}$",
                r"$\Phi &= \frac{\log ( q)}{\kappa}$",
                r"$\hat{c}_e &= \rho$",
                r"$\hat{c}_h &= \rho$",
                r"$\mu_{ae} &= p * (\hat{a_e} - a_e)$",
                r"$\sigma_{ae} &= v * (fu - a_e) * (a_e - fl)$",
                r"$\theta_e &= \frac{\chi}{z}$",
                r"$\theta_h &= \frac{1 - \chi}{1 - z}$",
                r"$\sigma^{q,k} &= \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\chi - z)} - \sigma$",
                r"$\sigma^{q,a} &= \frac{\frac{1}{q} * \frac{\partial q}{\partial a_e} * \sigma_{ae}}{1 - \frac{1}{q} * \frac{\partial q}{\partial z} * (\chi - z)}$",
                r"$\sigma^{q,k,1} &= \sigma + \sigma^{q,k}$",
                r"$\sigma^{z,k} &= \left(\theta_e - 1\right) * \sigma^{q,k,1}$",
                r"$\sigma^{z,a} &= \left(\theta_e - 1\right) * \sigma^{q,a}$",
                r"$\sigma_e^{J,k} &= \frac{1}{J_e} * \frac{\partial J_e}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_h^{J,k} &= \frac{1}{J_h} * \frac{\partial J_h}{\partial z} * z * \sigma^{z,k}$",
                r"$\sigma_e^{J,a} &= \frac{1}{J_e} * \left(\frac{\partial J_e}{\partial a_e} * \sigma_{ae} + \frac{\partial J_e}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\sigma_h^{J,a} &= \frac{1}{J_h} * \left(\frac{\partial J_h}{\partial a_e} * \sigma_{ae} + \frac{\partial J_h}{\partial z} * z * \sigma^{z,a} \right)$",
                r"$\zeta_e^k &= -(1-\gamma) * \sigma_e^{J,k} + \sigma^{z,k} + \sigma^{q,k} + \gamma * \sigma$",
                r"$\zeta_e^a &= -(1-\gamma) * \sigma_e^{J,a} + \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_h^k &= -(1-\gamma) * \sigma_h^{J,k} - \frac{z}{1-z} * \sigma^{z,k} + \sigma^{q,k,1} + \gamma * \sigma$",
                r"$\zeta_h^a &= -(1-\gamma) * \sigma_h^{J,a} - \frac{z}{1-z} * \sigma^{z,a} + \sigma^{q,a}$",
                r"$\zeta_e^1 &= \zeta_e^k + \phi * \zeta_e^a$",
                r"$\zeta_h^1 &= \zeta_h^k + \phi * \zeta_h^a$",
                r"$\zeta_e^2 &= \zeta_e^a + \phi * \zeta_e^k$",
                r"$\zeta_h^2 &= \zeta_h^a + \phi * \zeta_h^k$",
                r"$\epsilon_e &= \sigma^{q,k,1} * \zeta_e^1 + \sigma^{q,a} * \zeta_e^2$",
                r"$\epsilon_h &= \sigma^{q,k,1} * \zeta_h^1 + \sigma^{q,a} * \zeta_h^2$",
                r"$\mu^z &= \frac{a_e - \iota}{q} - \hat{c}_e+ (\theta_e - 1) * (\sigma^{q,k,1} * (\zeta_e^1 - \sigma^{q,k,1}) + \sigma^{q,a} * (\zeta_e^2 - \sigma^{q,a}) - 2 * \phi * \sigma^{q,k,1} * \sigma^{q,a})+ (1-\chi) * (\sigma^{q,k,1} * (\zeta_e^1 - \zeta_h^1) + \sigma^{q,a} * (\zeta_e^2 - \zeta_h^2)) + \frac{\lambda_d}{z} * (\bar{z} - z)$",
                r"$\mu^q &= \frac{1}{q} * \left( \frac{\partial q}{\partial z} * \mu^z * z + \frac{\partial q}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2 q}{\partial z^2} * z^2 * ((\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k} * \sigma^{z,a})\right)+ \frac{1}{q} * \left( \frac{1}{2} * \frac{\partial^2 q}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 q}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a})  * \sigma_{ae} * z\right)$",
            ]
        }
    equations = deepcopy(BASE_EQUATIONS)
    if timestepping:
        for k in equations:
            equations[k] += [
                r"$\mu_e^J &= \frac{\gamma}{2} * ((\sigma_e^{J,k})^2 + (\sigma_e^{J,a})^2 + 2 * \phi * \sigma_e^{J,k} * \sigma_e^{J,a} + \sigma^2) - (\Phi - \delta) + (\gamma - 1) * ( \sigma_e^{J,k} * \sigma + \phi * \sigma * \sigma_e^{J,a}) - \rho * (\log(\rho) - \log(J_e) + \log(z*q))$",
                r"$\mu_h^J &= \frac{\gamma}{2} * ((\sigma_h^{J,k})^2 + (\sigma_h^{J,a})^2 + 2*\phi*\sigma_h^{J,k} * \sigma_h^{J,a} + \sigma^2) - (\Phi-\delta) + (\gamma - 1) * (\sigma_h^{J,k}*\sigma + \phi*\sigma*\sigma_h^{J,a})- \rho * (\log(\rho) - \log(J_h) + \log((1-z)*q))$",
            ]
    else:
        for k in equations:
            equations[k] += [
                r"$\mu_e^J &= \frac{1}{J_e} * \left(\frac{\partial J_e}{\partial z} * \mu^z * z + \frac{\partial J_e}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2  J_e}{\partial z^2} * z^2 * \left( (\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k}* \sigma^{z,a} \right) \right)+ \frac{1}{J_e} * \left(\frac{1}{2} * \frac{\partial^2 J_e}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 J_e}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a}) * \sigma_{ae} * z \right)$",
                r"$\mu_h^J &= \frac{1}{J_h} * \left(\frac{\partial J_h}{\partial z} * \mu^z * z + \frac{\partial J_h}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2  J_h}{\partial z^2} * z^2 * \left( (\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k}* \sigma^{z,a} \right) \right)+ \frac{1}{J_h} * \left(\frac{1}{2} * \frac{\partial^2 J_h}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 J_h}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a}) * \sigma_{ae} * z \right)$",
            ]
    for k in equations:
        equations[k] += [
            r"$\mu_e^R &= \frac{a_e - \iota}{q} + \Phi - \delta + \mu^q + \sigma * \sigma^{q,k} + \phi * \sigma * \sigma^{q,a}$",
            r"$\mu_h^R &= \frac{a_h - \iota}{q} + \Phi - \delta + \mu^q + \sigma * \sigma^{q,k} + \phi * \sigma * \sigma^{q,a}$"
        ]
    for k in ["region1", "region2"]:
        equations[k].append(r"$r &= \mu_e^R - \underline{\chi} * \epsilon_e - (1-\underline{\chi}) * \epsilon_h$")
    equations["region3"].append(r"$r &= \mu_e^R - \chi * \epsilon_e - (1-\chi) * \epsilon_h$")
    for k in equations:
        equations[k] += [
            r"zmuz = z * muz",
            r"zsigzk = z * sigzk",
            r"zsigza = z * sigza",
        ]
    return equations

def get_hjb_equations(timestepping=False):
    if timestepping:
        return [
            r"$\frac{\partial J_e}{\partial t} + \frac{\partial J_e}{\partial z} * \mu^z * z + \frac{\partial J_e}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2  J_e}{\partial z^2} * z^2 * \left( (\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k}* \sigma^{z,a} \right) +\frac{1}{2} * \frac{\partial^2 J_e}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 J_e}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a}) * \sigma_{ae} * z - \mu_e^J * J_e$",
            r"$\frac{\partial J_h}{\partial t} + \frac{\partial J_h}{\partial z} * \mu^z * z + \frac{\partial J_h}{\partial a_e} * \mu_{ae} + \frac{1}{2} * \frac{\partial^2  J_h}{\partial z^2} * z^2 * \left( (\sigma^{z,k})^2 + (\sigma^{z,a})^2 + 2 * \phi * \sigma^{z,k}* \sigma^{z,a} \right) + \frac{1}{2} * \frac{\partial^2 J_h}{\partial a_e^2} * \sigma_{ae}^2 + \frac{\partial^2 J_h}{\partial z \partial a_e} * (\phi * \sigma^{z,k} + \sigma^{z,a}) * \sigma_{ae} * z - \mu_h^J * J_h$"
        ]
    else:
        return [
            r"$\rho * (\log(\rho) - \log(J_e) + \log(q * z)) + \Phi - \delta - (\gamma - 1) * (\sigma_e^{J,k} * \sigma + \phi * \sigma * \sigma_e^{J,a})-\frac{\gamma}{2} * ((\sigma_e^{J,k})^2 + (\sigma_e^{J,a})^2 + 2*\phi * \sigma_e^{J,k} * \sigma_e^{J,a} + \sigma^2) - \mu_e^J$",
            r"$\rho * (\log(\rho) - \log(J_h) + \log(q * (1-z))) + \Phi - \delta - (\gamma - 1) * (\sigma_h^{J,k} * \sigma + \phi * \sigma * \sigma_h^{J,a})-\frac{\gamma}{2} * ((\sigma_h^{J,k})^2 + (\sigma_h^{J,a})^2 + 2*\phi * \sigma_h^{J,k} * \sigma_h^{J,a} + \sigma^2) - \mu_h^J$"
        ]

ENDOG_EQUATIONS = {
    "region1": [
        r"$\rho * q &= \psi * a_e + (1-\psi) * a_h - \iota$",
        r"$\frac{a_e - a_h}{q} &= \underline{\chi} * (\epsilon_e - \epsilon_h)$"
    ],
    "region2": [
        r"$\rho * q &= a_e - \iota$"
    ],
    "region3": [
        r"$\rho * q &= a_e - \iota$",
        r"$\epsilon_e &= \epsilon_h$"
    ]
}

def add_boundary_conditions(model: Union[PDEModel, PDEModelTimeStep]):
    if isinstance(model, PDEModelTimeStep):
        zero_z = torch.zeros((100, 3), device=model.device)
        one_z = torch.zeros((100, 3), device=model.device)
        zero_z[:, 1] = torch.linspace(a_min, a_max, steps=100, device=model.device)
        one_z[:, 0] = torch.ones((100,), device=model.device)
        one_z[:, 1] = torch.linspace(a_min, a_max, steps=100, device=model.device)
    else:
        zero_z = torch.zeros((100, 2), device=model.device)
        one_z = torch.zeros((100, 2), device=model.device)
        zero_z[:, 1] = torch.linspace(a_min, a_max, steps=100, device=model.device)
        one_z[:, 0] = torch.ones((100,), device=model.device)
        one_z[:, 1] = torch.linspace(a_min, a_max, steps=100, device=model.device)
    model.add_endog_condition("q",
                                "q(SV)",
                                {"SV": zero_z},
                                Comparator.EQ,
                                "(1+kappa*ah)/(1+kappa*rho)", {"rho": PARAMS["rho"], "ah": PARAMS["ah"], "kappa": PARAMS["kappa"]},
                                label="q_min", weight=0.01)
    model.add_endog_condition("q",
                                "q(SV)",
                                {"SV": one_z},
                                Comparator.EQ,
                                "(1+kappa*ae)/(1+kappa*rho)", {"rho": PARAMS["rho"], "ae": zero_z[:, 1:2], "kappa": PARAMS["kappa"]},
                                label="q_max", weight=0.01)
    model.add_endog_condition("psi",
                                "psi(SV)",
                                {"SV": zero_z},
                                Comparator.EQ,
                                "0", {},
                                label="psi_min", weight=0.01)
    model.add_endog_condition("psi",
                                "psi(SV)",
                                {"SV": one_z},
                                Comparator.EQ,
                                "1", {},
                                label="psi_max", weight=0.01)
    return model

def setup_model(timestepping=False, rar=False) -> Dict[str, Union[PDEModel, PDEModelTimeStep]]:
    models = {}
    equations = get_equations(timestepping=timestepping)
    hjb_equations = get_hjb_equations(timestepping=timestepping)
    for k in ["region1", "region2", "region3"]:
        set_seeds(0)
        if timestepping:
            if rar:
                model = PDEModelTimeStep(k, TRAINING_CONFIGS["timestep_rar"], LATEX_VAR_MAPPING)
            else:
                model = PDEModelTimeStep(k, TRAINING_CONFIGS["timestep"], LATEX_VAR_MAPPING)
        else:
            model = PDEModel(k, TRAINING_CONFIGS["basic"], LATEX_VAR_MAPPING)
        model.set_state(STATES.copy(), PROBLEM_DOMAIN)
        model.add_agents(list(MODEL_CONFIGS["Agents"].keys()), MODEL_CONFIGS["Agents"])
        model.add_endogs(list(MODEL_CONFIGS["Endogs"][k].keys()), MODEL_CONFIGS["Endogs"][k])
        model.add_params(PARAMS)
        if k == "region1":
            model = add_boundary_conditions(model)
        for eq in equations[k]:
            model.add_equation(eq)
        for endog_eq in ENDOG_EQUATIONS[k]:
            model.add_endog_equation(endog_eq)
        if k == "region2":
            model.add_constraint(r"$\frac{a_e - a_h}{q}$",
                         Comparator.GEQ,
                         r"$\underline{\chi} * (\epsilon_e - \epsilon_h)$")
        for hjb_eq in hjb_equations:
            model.add_hjb_equation(hjb_eq)

        models[k] = model
    return models


def compute_func(pde_model: Union[PDEModel, PDEModelTimeStep], a_list, z_min, z_max, vars_to_plot):
    N = 100
    res_dict = {}
    for a in a_list:
        if isinstance(pde_model, PDEModelTimeStep):
            SV = torch.zeros((N, 3), device=pde_model.device)
        else:
            SV = torch.zeros((N, 2), device=pde_model.device)
        SV[:, 0] = torch.linspace(z_min, z_max, N)
        SV[:, 1] = torch.ones((N,)) * a
        x_plot = SV[:, 0].detach().cpu().numpy().reshape(-1)
        for i, sv_name in enumerate(pde_model.state_variables):
            pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
        pde_model.update_variables(SV)
        res_dict["x_plot"] = x_plot
        for var in vars_to_plot:
            if var in pde_model.variable_val_dict:
                res_dict[f"{var}_{a}"] = pde_model.variable_val_dict[var].detach().cpu().numpy().reshape(-1)
            elif var == "psi":
                res_dict[f"psi_{a}"] = np.ones(N)
    return res_dict

def compute_final_plot_dict(res_dict1: Dict[str, Dict[str, Any]], 
                            res_dict2: Dict[str, Dict[str, Any]], 
                            res_dict3: Dict[str, Dict[str, Any]], 
                            plot_args: Dict[str, Any], 
                            a_list: List[float]):
    final_plot_dict = {}
    for a in a_list:
        index_region1 = (res_dict1[f"psi_{a}"] < 1)
        index_region2 = (res_dict1[f"psi_{a}"] >= 1) & (res_dict2[f"epse_{a}"] > res_dict2[f"epsh_{a}"])
        index_region3 = (res_dict1[f"psi_{a}"] >= 1) & (res_dict2[f"epse_{a}"] <= res_dict2[f"epsh_{a}"])
        for k in plot_args:
            final_plot_dict[f"{k}_{a}"] = res_dict1[f"{k}_{a}"] * index_region1 + res_dict2[f"{k}_{a}"] * index_region2 + res_dict3[f"{k}_{a}"] * index_region3
    final_plot_dict["x_plot"] = res_dict1["x_plot"]
    return final_plot_dict

def plot_res(res_dicts: Dict[str, Dict[str, Any]], plot_args: Dict[str, Any], a_list: List[float]):
    x_label = "Wealth share (z)"
    
    for i, (func_name, plot_arg) in enumerate(plot_args.items()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        for k, l, ls in [("basic", "Basic", "--"), ("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
            res_dict = res_dicts[k].copy()
            x_plot = res_dict.pop("x_plot")
            for a in a_list:
                y_vals = res_dict[f"{func_name}_{a}"]
                ax.plot(x_plot, y_vals, label=r"$a_e$={i} ({l})".format(i=round(a,2), l=l), linestyle=ls)
        ax.set_xlabel(x_label)
        ax.set_ylabel(plot_arg["ylabel"])
        ax.set_title(plot_arg["title"])
        ax.legend()
        plt.tight_layout()
        fn = os.path.join(BASE_DIR, "plots", f"{func_name}.jpg")
        plt.savefig(fn)
        plt.close()
        

def plot_loss(fn):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 30))
    for i, region in enumerate(["Region 1 ($\psi < 1$)", "Region 2 ($\psi = 1$, $\epsilon_e > \epsilon_h$)", "Region 3 ($\psi = 1$, $\epsilon_e = \epsilon_h$)"]):
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel("Loss")
        ax[i].set_yscale("log")
        ax[i].set_title(f"Total Loss across Epochs for {region}")
    for k, l, ls in [("basic", "Basic", "--"), ("timestep", "Time-stepping", "-"), ("timestep_rar", "Time-stepping (RAR)", ":")]:
        curr_dir = os.path.join(BASE_DIR, k)
        for i, region in enumerate(["region1", "region2", "region3"]):
            loss_file = os.path.join(curr_dir, f"{region}_min_loss.csv")
            loss_df = pd.read_csv(loss_file)
            ax[i].plot(loss_df["epoch"], loss_df["total_loss"], label=l, linestyle=ls)
    for i in range(3):
        ax[i].legend()
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

if __name__ == "__main__":
    final_plot_dicts = {}
    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
    for k in TRAINING_CONFIGS.keys():
        print(f"{k:=^80}")
        timestepping = "timestep" in k
        rar = "rar" in k
        curr_dir = os.path.join(BASE_DIR, k)
        res_dicts = [0] * 3
        model = setup_model(timestepping=timestepping, rar=rar)
        for i, region in enumerate(["region1", "region2", "region3"]):
            if not os.path.exists(os.path.join(curr_dir, f"{region}_best.pt")):
                model[region].train_model(curr_dir, f"{region}.pt", full_log=True)
            model[region].load_model(torch.load(os.path.join(curr_dir, f"{region}_best.pt"), weights_only=False))
            res_dict = compute_func(model[region], a_list, z_min, z_max, VARS_TO_PLOT)
            res_dicts[i] = res_dict
        final_plot_dict = compute_final_plot_dict(res_dicts[0], res_dicts[1], res_dicts[2], VARS_TO_PLOT, a_list)
        final_plot_dicts[k] = final_plot_dict
    plot_res(final_plot_dicts, PLOT_ARGS, a_list)
    plot_loss(os.path.join(BASE_DIR, "plots", "loss.jpg"))


