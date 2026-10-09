# Superseded scripts

These scripts are retained for completeness and provenance. **None of them
produced a result reported in the manuscript.** Several produced values that were
later withdrawn. Do not use them to reproduce the paper; use the scripts in the repository root.

| Script | Why it is superseded |
|---|---|
| `run_chunk.py` | Rebuilt a fresh NSGA-III object every chunk, discarding survival, niching and ideal/nadir state. Its runs are the "chunk-restarted" runs excluded from the paper. Replaced by `run_continuous.py`. |
| `archive.py` | Built an external archive that **includes** the chunk-restarted run (`X_gen.npy`). The manuscript excludes that run. Replaced by `final_select.py`, which is the archive and selection protocol actually used. |
| `reconcile.py` | Pooled-archive reconciliation from the restarted pipeline. Its pooled design was withdrawn. |
| `results_v2.py`, `run_v2.py`, `finalize.py` | Earlier production drivers built on the restarted pipeline. |
| `bench.py` | Pure-AWE and pure-PEM benchmark runs supporting a hybrid-advantage comparison that was withdrawn. |
| `ablation.py` | First ablation, run under unequal budgets. Replaced by `abl_cont.py`, `f1_grid.py`, `f23_check.py`. |
| `sweep.py` | Demand-scaling sweep around a superseded design. Excluded from the paper. |
| `maturity.py` | Mature-period demand experiment, not evaluated on the final 20-draw basis. Excluded. |
| `mechanism.py` | PEM-fraction mechanism sweep around a superseded design, tied to the withdrawn hybrid-advantage claim. Excluded. |
| `tornado.py` | Ten-parameter tornado around a superseded design. Excluded. |
| `figures_final.py`, `figures_extra.py`, `refigs.py`, `newfigs.py` | Earlier figure scripts that drew superseded designs and values. Replaced by `make_all_figures.py` and `make_pareto_convergence.py`. |

These scripts import modules from the repository root and read checkpoints that are not
shipped, so most will not run as-is. That is intentional.
