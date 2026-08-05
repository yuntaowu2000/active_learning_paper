#!/usr/bin/env bash
# =============================================================================
# stochastic_volatility_hd -- full paper pipeline (section 4).
# Run from this folder:  cd stochastic_volatility_hd && ./run.sh
# All runs use the calibrated point a=0.1, sigma=0.06, tau=1.15, gamma=6.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

base="./models"

# --- 0) Di Tella (2017) finite-difference reference for the 2-D validation ----
python numerical.py

# --- 4.1) 2-D validation vs FD + full 8-method component ladder ---------------
python main.py --case agents2  --float64 --a 0.1 --sigma 0.06 --tau 1.15 --gamma 6.0

# --- 4.2) Scaling to 20-D and 40-D (4 core methods each) ----------------------
python main.py --case agents20 --float64 --a 0.1 --sigma 0.06 --tau 1.15
python main.py --case agents40 --float64 --a 0.1 --sigma 0.06 --tau 1.15

# --- 4.3) Compute / memory diagnostics ----------------------------------------
python diagnostics.py --cases agents2,agents20,agents40 --float64

# --- 4.1/4.4) Simulate the best 2-D model: distribution, portfolio, IRF --------
python simulate.py --float64 --base-dir "$base" --case agents2 --config timestep_rar \
    --gamma 6.0 --a 0.1 --sigma 0.06 --tau 1.15 --portfolio --irf --irf-hold-v

# --- 4.2/4.4) Portfolio choice + economic insight for the high-D models -------
python simulate.py --float64 --base-dir "$base" --case agents20 --config timestep_rar \
    --gamma 6.0 --a 0.1 --sigma 0.06 --tau 1.15 --portfolio --irf --irf-hold-v
python simulate.py --float64 --base-dir "$base" --case agents40 --config timestep_rar \
    --gamma 6.0 --a 0.1 --sigma 0.06 --tau 1.15 --portfolio

# --- 4.4) Comparative statics: capital-shock -> risk-premium sweep (FD) --------
python simulate.py --sweep --a 0.1 --sigmas 0.06 --gammas 5,6,8 --taus 1.15 \
    --sweep-out ./models/ditella_rp_sweep.csv
