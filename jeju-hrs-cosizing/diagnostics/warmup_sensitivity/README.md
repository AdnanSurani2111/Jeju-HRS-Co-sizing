# Warm-up sensitivity diagnostic

Run `python3 run_warmup_sensitivity.py` from this directory. The script uses the
locked selected design, the packaged renewable and demand files, and validation
seeds 500--519. It writes draw-level and summary CSV files for zero, one, three,
and five full-horizon warm-up passes plus an iterative fixed-point criterion.

The production setting is one warm-up pass. This diagnostic does not alter the
optimizer and does not make terminal equality a formal candidate constraint.
