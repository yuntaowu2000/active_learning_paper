#!/usr/bin/env bash
# =============================================================================
# tree_models -- appendix pipeline.
# Run from this folder:  cd tree_models && ./run.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# Finite-difference reference for the 2-tree case (models/2trees_solution-raw.csv)
python tree_model_pymacrofin.py

# Train time-stepping + RAR for each tree count; emit plots, FD comparison,
# and the validation-loss comparison table (tree_validation_losses.{csv,tex}).
python tree_model_main.py

# Compute / memory table + per-epoch timing (tree_memory.{csv,tex}, tree_timing.pdf)
python tree_model_efficiency.py
