"""Slice overlay with FD, Basic, Time-stepping, and Our Method (timestep_rar).

Separate from ``analysis.plot_slice_comparison`` so the paper figures stay
untouched.  Output files are named ``slice_all_{var}.pdf``.
"""

import os

import matplotlib.pyplot as plt

from common import (
    METHOD_COLORS,
    METHOD_DISPLAY,
    METHOD_PLOT_STYLES,
    PAPER_BLACK,
    PAPER_FIGSIZE,
    PAPER_FONT_SIZE,
    SLICE_PLOT_ARGS,
    configure_paper_plots,
)

configure_paper_plots()

# Fixed method order for the four-way comparison figure.
PLOT_METHODS = ("basic", "timestep", "timestep_rar")

# timestep_rar legend + style override; other methods keep common.py styling.
OUR_METHOD = "timestep_rar"
OUR_METHOD_LABEL = "Our Method"
OUR_METHOD_LS = "-"
OUR_METHOD_COLOR = "#5492ab"


def _style_for(name, index):
    """Linestyle / marker / color for ``name`` at plot position ``index``."""
    if name == OUR_METHOD:
        return OUR_METHOD_LS, "", OUR_METHOD_COLOR
    ls, mk = METHOD_PLOT_STYLES[index % len(METHOD_PLOT_STYLES)]
    return ls, mk, METHOD_COLORS.get(name, PAPER_BLACK)


def _label_for(name):
    if name == OUR_METHOD:
        return OUR_METHOD_LABEL
    return METHOD_DISPLAY.get(name, name)


def plot_slice_comparison(method_dicts, fd_dict, v_list, out_dir,
                          file_prefix="slice_all"):
    """2-D slice overlay: FD + Basic + Time-stepping + Our Method.

    Writes ``{file_prefix}_{var}.pdf`` into ``out_dir`` (defaults to
    ``slice_all_{var}.pdf`` so it does not clobber ``slice_{var}.pdf``).
    """
    os.makedirs(out_dir, exist_ok=True)
    ordered = [
        (name, method_dicts[name])
        for name in PLOT_METHODS
        if name in method_dicts
    ]
    for var, parg in SLICE_PLOT_ARGS.items():
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        if fd_dict is not None:
            xfd = fd_dict["x_plot"]
            for v in v_list:
                key = f"{var}_{v}"
                if key in fd_dict:
                    ax.plot(
                        xfd, fd_dict[key],
                        ls="-.", color=PAPER_BLACK,
                        marker="x", markevery=3, label="FD",
                    )
        for mi, (name, md) in enumerate(ordered):
            xp = md["x_plot"]
            ls, mk, color = _style_for(name, mi)
            for v in v_list:
                ax.plot(
                    xp, md[f"{var}_{v}"],
                    ls=ls, marker=mk, markevery=8 if mk else None,
                    color=color, label=_label_for(name),
                )
        ax.set_xlabel("Expert wealth share x")
        ax.set_ylabel(parg["ylabel"])
        ax.tick_params(axis="both", which="major", labelsize=PAPER_FONT_SIZE)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{file_prefix}_{var}.pdf"))
        plt.close(fig)
