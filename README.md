# Renewable hydrogen supply co-sizing

Research code for **Chronological Co-Sizing of Renewable Hydrogen Supply Systems: Reliability Screening and Independent Validation**.

## Start here

The principal evidence is the recorded `full_search/` campaign. The `revision/` directory and older root-level scripts/results preserve earlier exploratory work, including unsuccessful reliability tests. They are retained for provenance and must not be substituted for the latest results.

| Location | Purpose |
|---|---|
| `run.py` | Safe entry point for verification, compilation, audit reproduction and updated figures |
| `verify_release.py` | Checks model/input hashes, all stored runs, screening decisions, audit statistics and release integrity |
| `config.py`, `data_v2.py`, `dispatch_v2.py`, `objectives_v2.py` | Unchanged original Python model |
| `full_search/protocol.json` | Frozen complete-search protocol and model/input hashes |
| `full_search/F2_*`, `full_search/F3_*` | Ten full optimizer checkpoints, populations, traces and screening matrices |
| `full_search/selections_frozen.json` | All 30 decisions fixed before audit |
| `full_search/audit_metrics.npz`, `audit_results.json` | Shared independent 2000-scenario audit and derived statistics |
| `revision/fast_dispatch.cpp` | Exact numerical accelerator checked against original Python |
| `make_updated_figures.py`, `updated_figures/` | Current boundary, workflow, population and search-progress vector figures |
| `docs/` | Historical notes and GitHub update instructions |

## Environment and quick verification

Tested on Linux, Python 3.12.14 and g++ with OpenMP. Use Python 3.12 for the pinned environment; serialized `dill` checkpoints require trusted files and matching package versions. Read-only stored-result verification never loads those checkpoints.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py verify
python run.py build
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python run.py check-model
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python run.py recheck-audit
python run.py figures
```

`verify` reads the recorded results without compiling or simulating. `build` creates the local Linux shared library. `check-model` compares the accelerated and original Python implementations on three capacity vectors. `recheck-audit` recomputes every distinct frozen design on all 2000 recorded audit draws and compares outputs without replacing any historical file. `figures` regenerates PDF and editable SVG figures. Windows users can use WSL2; macOS requires a compiler with OpenMP or a suitable Linux environment.

## Experiment and result

- F3 frees all six capacities; F2 fixes PV=5500 kW and wind=1400 kW and frees the four downstream decisions.
- Five optimizer seeds per case, 182 individuals, 91 reference directions and 100 generations: 18,200 objective evaluations per run, 182,000 in total. Each evaluation averages eight training-demand scenarios.
- Every final population is screened separately on three banks of 400 scenarios. The cheapest candidate passes only when the one-sided 95% Clopper–Pearson upper bound on the probability of three-year hydrogen mass loss above 0.1% is at most 2.5%. No passing candidate means no selection.
- All 30 attempted decisions are frozen before the shared unused audit. Primary audit bound: at most 5%. Secondary bounds adjust over 30 planned decisions using alpha=0.05/30.
- Joint sizing qualifies in 3/5 optimizer searches and 9/15 crossed screening decisions; restricted sizing qualifies in 5/5 and 15/15. All 24 selected decisions, representing ten distinct vectors, pass both audit criteria. The nine qualified pairs have modeled cost reductions of 25.36–30.88% relative to F2. This is conditional on qualified pairs, rather than success in every run or equal true reliability.

The three screening banks crossed with five runs do not constitute fifteen independent optimizations. Equal budgets do not establish equal convergence or global optimality. No selection does not establish physical infeasibility.

## Reproducing searches versus verifying results

The historical scripts are retained byte-for-byte. After `verify` and `build`, running `python full_search/run_campaign.py` reuses completed populations and checks their frozen selections; it does **not** perform ten new searches when those records are present. `audit_campaign.py` likewise reuses the saved audit. Use `run.py recheck-audit` for actual independent recomputation of that bank.

For a complete clean rerun, first copy the entire repository into a separate experiment directory. In that copy only, move the ten F2/F3 run folders, `selections_frozen.*`, `audit_metrics.npz`, `audit_results.json`, `all_decisions.csv`, and `paired_results.csv` into a backup directory outside the code tree. Keep the original protocol and source files. Compile and run `full_search/run_campaign.py` followed by `full_search/audit_campaign.py`. The campaign then executes all ten searches and freezes the decisions before computing the audit. Do not run the optional cache worker unless you understand its completion dependencies. New output timestamps and a regenerated selections file require a separate manifest; they are not the distributed historical record. A changed model or selection rule is a new experiment and needs its own protocol, unused evaluation data, and report.

Screening NPZ rows may contain NaN tails: rejection after five observed exceedances cannot be reversed under the fixed 400-draw gate. Passing candidates always receive all 400 simulations. The release verifier checks these certificates and the exact cheapest passing selection.

## Scope and units

Demand draws resample the same 255 daily station records using the original seven-day block generator. Renewable data use fixed regional 2020–2022 hourly generation indices. This is internal demand-generator validation, not external validation at new stations or all possible weather conditions. Costs cover generation, electrolysis and storage; compression, pre-cooling, dispensing, queues, land and water treatment are excluded. Service means hydrogen mass supplied, not completed vehicle refuelings.

Capacity vector order: PV kW, wind kW, electrolysis kW, PEM fraction, battery kWh, hydrogen storage kg. Metric columns: mass-loss fraction, annual hydrogen served kg, curtailed fraction, annual AWE degradation, annual PEM degradation, annual AWE starts, annual PEM starts, annual mass-balance residual kg, annual hydrogen produced kg, terminal HSS imbalance fraction. Costs are separate in USD/kg. Fractions must be multiplied by 100 for percentages.

## Source integrity and known historical comments

The release manifest detects accidental mismatches; it is not a cryptographic signature or proof of scientific validity. The frozen accelerator contains fixed numerical constants matching the hashed configuration. The safe entry point checks all source/input hashes before numerical use. Do not edit configuration values and reuse that accelerator without updating it and revalidating.

Some old comments in `config.py` describe year-specific installed fleet capacities, while `data_v2.py` actually normalizes regional generation by each year's 99.9th percentile. The paper and this README describe the implemented method. That frozen comment is retained so the original source hashes remain valid. Earlier root scripts and checkpoints can use older scenario counts or thresholds; consult their provenance rather than inferring the latest protocol from their filenames.

See `docs/GITHUB_UPDATE.md` for updating your existing repository. No repository URL, release DOI or license is invented in this package; preserve your repository's existing license and add its actual URL to `CITATION.cff` when publishing.

## Three-objective diagnostics and complete capacity tables

Run `python campaign_diagnostics.py` after installing the pinned environment.
It verifies the release, then writes six-panel PDF/SVG/PNG Pareto-front approximations,
all 1,820 final-population objective vectors, all 30 decision/capacity records,
and search-level outcomes to `campaign_diagnostics/`. Nondominance uses all three
training objectives within each run. Stars mark frozen selections even if dominated
on training objectives. Neither plotted front membership nor mean training loss
establishes audit qualification. No selections are pooled or changed.

The historical `trace.json` files contain per-generation scalar minima, not population
objective matrices. The single `optimizer.dill` checkpoint was overwritten during
execution; checkpoint frequency does not mean intermediate states were retained.
Consequently this release does not fabricate historical hypervolume curves.

An optional, computationally expensive replay records every population separately:

```bash
python run.py build
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python replay_diagnostics.py --case F3 --seed 7
```

Repeat for both cases and all five protocol seeds if complete convergence diagnostics
are required. The replay writes only to `diagnostic_replays/`, checks the final
population, objective matrix, and evaluation budget against the archive exactly,
and emits a hypervolume figure only on PASS. The replay tool was syntax
checked in this release; ten complete replay optimizations have not been executed.
The common post hoc transformation is cost/40, raw mass-loss fraction, raw
curtailment fraction, with reference (1,1,1). This measures the region below
(40 USD/kg, 100% loss, 100% curtailment); outside-box points contribute zero.
The excluded-point count is recorded. It is a diagnostic choice, not a frozen
selection setting or proof of global convergence. No cumulative all-evaluations
archive is claimed.

The F2 restriction retains PV=5500 kW and wind=1400 kW from the exploratory F0
configuration. The cost comparison is conditional on this baseline, rather than
all possible fixed-generation configurations. Hashes identify content and do not
independently prove experiment timestamps. The 2.5% screening margin was informed
by exploratory results before the complete-search campaign. Existing protocol,
selection, source, and audit records remain unchanged.
