# Results basis map

Every primary number in the manuscript must carry one of the bases below. Do not
mix bases within a table, figure, abstract sentence, or conclusion sentence
without labeling them.

## Bases

| Label | Definition | Draws |
|---|---|---|
| **PRIMARY** | Fresh demand-bootstrap realizations, seeds 500-519. Not used in optimization, local refinement, or screening. | 20 |
| **SCREENING** | Common screening realizations, seeds 200-207. Used to select the design. | 8 |
| **HOLDOUT** | 2023 renewable chronology with independent demand draws. | 4 |
| **ABLATION** | Refined F0-F3 protocol: F1 grid, F2 and F3 NSGA-III plus local polish. | see caveat |
| **ILLUSTRATIVE** | One identified demand realization, shown as an example only. | 1 |

## Assignment

| Result | Basis | Source |
|---|---|---|
| LCOH mean, SD, SE, CI | PRIMARY | `validation_20_fresh_demand` |
| Service loss mean, q95, max | PRIMARY | `validation_20_fresh_demand` |
| Curtailment mean | PRIMARY | `validation_20_fresh_demand` |
| LCOH decomposition (4 components) | PRIMARY | `lcoh_decomposition`, sums to mean LCOH |
| Hydrogen flow balance | PRIMARY | `T19_hydrogen_balance.csv` |
| Storage behaviour (median inventory, time above 90%, time below 25%) | PRIMARY | `storage_behaviour`, `T26` |
| Selected design capacities | archive | `design_recommended` |
| Screening-tail q95 | SCREENING | `screening_vs_validation` |
| Held-out 2023 per-draw values | HOLDOUT | `holdout_2023_four_draws`, `T18` |
| Co-sizing decomposition | ABLATION | `ablation` |
| Dispatch trajectory figure | ILLUSTRATIVE | `T23_dispatch_window.csv` |

## Secondary, supplementary material only

| Result | Why secondary |
|---|---|
| Hours at the 5% HSS floor, 46.6 +/- 51.25, range 0-147 | CV 1.10; 8 of 20 draws never reach the floor |
| Longest low-inventory episode, 13.8-43.6 days | Varies with a descriptive 15-40% threshold (`T24`) |
| Floor-tolerance sensitivity | Stable, but qualifies a secondary metric (`T25`) |
| Deepest drawdown, ~187 kg over ~1 day | Single realization |
| Parameter sensitivity (`T20`) | Six-draw basis, not PRIMARY. Label explicitly. |

## Reporting rule

Report a statistic across the parameter it depends on, or do not report it **as a
system property**. A single-draw trajectory may be shown as an illustration
provided its conditional basis is stated.

## Excluded permanently

Old benchmark table (T03), demand sweep (T06), x_PEM mechanism sweep (T12),
tornado (T11, T21), near-optimal envelope from the superseded design (T16), and
every artifact from the chunk-restarted pipeline.

## Withdrawn strings

The manuscript source, tables, captions, figures, supplement, and compiled PDF
must contain zero occurrences of:

- 19.3746 USD/kg
- 5.00% curtailment
- 36.23% rule-based
- 11.2851 USD/kg
- 34.95% curtailment
- 10.45 co-sizing
- 4.82 interaction
- zero LPSP / zero unserved demand
- 10.56 pooled design
- hybrid advantage 0.49-1.40 USD/kg
- authoritative production run

## Contamination gate

Run against the regenerated source and the extracted PDF text:

```
19\.3746 | 5\.00% | 36\.23 | 11\.2851 | 34\.95 | 10\.45 | 4\.82 |
10\.56 | zero LPSP | zero unserved | authoritative production run |
0\.49 to 1\.40 | 128 hours | 22\.7 day
```

Note the last two: `128 hours` and `22.7 day` are withdrawn as system properties
and may appear only in supplementary variability discussion.
