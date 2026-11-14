#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Full paper experiment pipeline.
python numerical.py
python main.py
python main_calibrated.py
python gini_significance.py
python gini_zero_v_volatility.py
python diagnostics.py
