# Jeju hydrogen supply co-sizing: revised analysis package

This package preserves the original model, data, and saved candidate designs and adds the revision analyses in `revision/`.

## Additional experiments

From this folder, compile the accelerated dispatcher:

```bash
g++ -O3 -fPIC -shared -fopenmp revision/fast_dispatch.cpp -o revision/fast_dispatch.so
```

Run the experiments in their documented order:

```bash
OMP_NUM_THREADS=4 python3 revision/run_revision.py
OMP_NUM_THREADS=4 python3 revision/confidence_followup.py
OMP_NUM_THREADS=4 python3 revision/margin_followup.py
python3 revision/verify_revision.py
python3 revision/make_revision_figure.py
```

A C++ compiler with OpenMP support is needed. The new Python analyses require NumPy, pandas, SciPy, and openpyxl. Matplotlib is used for plots. The original optimizer additionally requires pymoo and dill. Original requirements are retained; `revision/environment.json` records the actual revision runtime.

`run_revision.py` verifies several metrics against the original Python simulator, constructs 104 unique fixed capacity vectors, evaluates separate screening and audit banks, and repeats screening on overlapping subsets. Existing saved screen/audit matrices are reused; remove those two NPZ files to force their recomputation. Their input hashes are recorded in `revision/verification.json`.

The first analysis uses demand seeds 10000-10399 for screening and 20000-20399 for audit. The confidence follow-up uses unused audit seeds 40000-40399. The conservative-margin follow-up uses 50000-50399. Follow-up rules were introduced after preceding audit failures. All outcomes are retained; the final margin is exploratory.

`candidate_pool.npz` contains the exact designs and source labels. `screen.npz` and audit NPZ files contain per-design costs and metrics. Metric columns are: mass-loss fraction, annual hydrogen served, curtailed fraction, annual AWE degradation, annual PEM degradation, annual AWE starts, annual PEM starts, annual mass residual, annual hydrogen produced, terminal hydrogen-inventory imbalance fraction. All three-year demand scenarios use the same underlying measured daily-sales record and original seven-day bootstrap.

`repeated_selection.csv` records each screening decision. `repeated_summary.csv` contains conditional counts. These repetitions share scenario banks and are not independent field replications. `results.json`, `followup_results.json`, and `margin_results.json` contain rule-specific results. Selection records are saved before follow-up audit evaluation.

## Main new findings

The full-bank empirical rule selects a joint design that fails its independent audit. The confidence-rule follow-up also fails its next audit. The final conservative screen margin selects 1,500 kWh BESS and 2,000 kg HSS. On its unused 400-scenario bank the joint design has 7 exceedances of 0.1% three-year mass loss, with a one-sided 95% probability upper bound of 3.2618%. Its cost is 13.0423 USD/kg. The restricted-pool design has zero exceedances, a 0.7461% bound, and cost 20.1979 USD/kg on the same bank. Both satisfy the common 5% pointwise criterion. The 35.4273% cost difference is conditional on unequal finite pools, not global optimization or equal true reliability.

## Original results and reproduction limits

The original primary design remains in `checkpoints/FINAL.json`. Its 20-draw summary was reproduced in the revision to numerical tolerance using the checked accelerated model. Original full optimizer searches and historical supplementary S9 searches were not newly rerun. Original ablation scripts and checkpoints remain for provenance. Some intermediate artifacts are older than the revised paper; do not treat every historical JSON as a current headline result. In particular, `ablation_final.json` describes a superseded 1% protocol. The revised manuscript distinguishes that from the original equal-budget 0.1% ablation and the new finite-pool experiment.

The accelerated dispatcher is locked to the shipped configuration. Changing model parameters requires updating it and rerunning equivalence checks. It has no trajectory output; use `dispatch_v2.py` for detailed original-model trajectories. Data hashes, reference reproduction checks, and mass-balance verification are recorded in `revision/verification.json`.

Costs cover renewable generation, electrolysis, and electrical/hydrogen storage. Compression, pre-cooling, dispensing, land, and water treatment are excluded. Demand is daily sales spread uniformly within a day. Conclusions concern modeled hydrogen mass supplied, not individual refueling completion.
