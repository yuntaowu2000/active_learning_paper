#!/usr/bin/env bash
# =============================================================================
# Top-level driver: reproduce all paper results.  Each subfolder has its own
# run.sh; this script just invokes them in order.  Run from the repo root.
# The stochastic-volatility and tree runs are compute-heavy (GPU recommended).
# =============================================================================

set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"

# --- Appendix: free-boundary model (1-D frozen, 2-D + validation table) -------
( cd "$root/free_boundary_model" && python 1d_problem.py && python 2d_problem.py )

# --- Appendix: tree models (validation + compute/memory tables) ---------------
( cd "$root/tree_models" && bash run.sh )

# --- Main results: high-dimensional stochastic-volatility model (section 4) ---
( cd "$root/stochastic_volatility_hd" && bash run.sh )
