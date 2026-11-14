# =============================================================================
# Top-level driver: reproduce all paper results.  Each subfolder has its own
# run.ps1; this script just invokes them in order.  Run from the repo root.
# The stochastic-volatility and tree runs are compute-heavy (GPU recommended).
# =============================================================================

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# --- Appendix: free-boundary model (1-D frozen, 2-D + validation table) -------
Push-Location "$root/free_boundary_model"
python 1d_problem.py
python 2d_problem.py
Pop-Location

# --- Main results: high-dimensional stochastic-volatility model (section 4) ---
Push-Location "$root/stochastic_volatility_hd"
./run.ps1
Pop-Location

# --- Appendix: tree models (validation + compute/memory tables) ---------------
Push-Location "$root/tree_models"
./run.ps1
Pop-Location
