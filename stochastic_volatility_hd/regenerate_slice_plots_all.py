"""Redraw four-way slice figures (FD / Basic / Time-stepping / Our Method).

Loads existing checkpoints under ``models/agents2`` and
``models/agents2_ditella`` -- no training.  Writes ``slice_all_*.pdf`` into
each run's ``comparison/`` folder via ``plot_slice_comparison_all``.
"""

import os

import numpy as np
import torch

from analysis import evaluate_slices
from common import (BASE_PARAMS, BASE_V_MEAN, MODEL_ROOT, PAPER_GAMMA,
                    make_case, move_model)
from main import economic_params
from main_agents2_long_basic import _check_metadata, _get_model
from numerical import DITELLA_PARAMS
from plot_slice_comparison_all import PLOT_METHODS, plot_slice_comparison


def _fd_path(gamma, params):
    return os.path.join(
        MODEL_ROOT, "numerical",
        f"numerical_{gamma}_{params['tau']}_{params['sigma']}_{params['a']}.npz",
    )


def regenerate(case_dir, gamma, params, case="agents2"):
    root = os.path.join(MODEL_ROOT, case_dir)
    comparison_dir = os.path.join(root, "comparison")
    _check_metadata(root, make_case(case, gamma)[3], params)

    names = [
        name for name in PLOT_METHODS
        if os.path.exists(os.path.join(root, name, "model_best.pt"))
    ]
    missing = [name for name in PLOT_METHODS if name not in names]
    if missing:
        print(f"[{case_dir}] missing checkpoints: {missing}")
    print(f"[{case_dir}] plotting {names}")

    v_list = [BASE_V_MEAN]
    slices = {}
    for name in names:
        model = _get_model(
            os.path.join(root, name), case, gamma, params,
            name, n_epochs=0, train=False,
        )
        slices[name] = evaluate_slices(model, v_list)
        move_model(model, "cpu")
        del model

    fd_path = _fd_path(gamma, params)
    fd = np.load(fd_path) if os.path.exists(fd_path) else None
    if fd is None:
        print(f"[{case_dir}] no FD solution at {fd_path}")
    plot_slice_comparison(slices, fd, v_list, comparison_dir)
    print(f"[{case_dir}] slice_all_*.pdf written to {comparison_dir}")


def main():
    torch.set_default_dtype(torch.float64)
    regenerate("agents2", PAPER_GAMMA, economic_params("agents2"))
    ditella_params = BASE_PARAMS | {
        key: DITELLA_PARAMS[key] for key in BASE_PARAMS if key in DITELLA_PARAMS
    }
    regenerate("agents2_ditella", DITELLA_PARAMS["gamma"], ditella_params)


if __name__ == "__main__":
    main()
