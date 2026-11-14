$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Full paper experiment pipeline.
python numerical.py
python main.py
python main_calibrated.py
python gini_significance.py
python gini_zero_v_volatility.py
python diagnostics.py