# Reliability Qualified Co-Sizing of Renewable Hydrogen Supply Systems: Screening Uncertainty and Independent Validation

Code and data for a multi-objective capacity-planning study of a renewable
hydrogen refueling station on Jeju Island, Republic of Korea.

The model jointly sizes six capacities: photovoltaic rating, wind rating, total
electrolyzer rating, the proton exchange membrane (PEM) share of that rating,
battery energy, and hydrogen storage mass and evaluates every candidate over an
uninterrupted 26,304 hour renewable chronology (2020–2022). NSGA-III minimizes
three objectives: levelized cost of hydrogen (LCOH), mass-based loss of power
supply probability (LPSP), and renewable curtailment.

Every result reported in the accompanying manuscript can be regenerated from this
repository, and a verification script compares a fresh run with the reference
values.

---

## Contents

- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Verification](#verification)
- [Model summary](#model-summary)
- [Evaluation bases](#evaluation-bases)
- [Data](#data)
- [Diagnostics](#diagnostics)
- [Superseded code](#superseded-code)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Key results

Selected design, frozen after screening and evaluated on 20 demand realizations
that played no part in search or selection:

| Quantity | Value |
|---|---|
| Photovoltaic | 1846.8 kW |
| Wind | 668.0 kW |
| Total electrolysis | 1028.7 kW (PEM share 0.40) |
| Battery energy | 55.2 kWh |
| Hydrogen storage | 1855.8 kg |
| LCOH, mean ± SD (n = 20) | 11.7184 ± 0.1185 USD/kg |
| Service loss, mean / empirical 95th percentile | 0.140 % / 0.421 % |
| Curtailment, mean | 21.24 % |
| Held-out 2023 renewable year, 4 demand draws | 11.51–12.03 USD/kg, zero unserved mass |

Co-sizing ablation (refined, under a 1 % selection condition):

| Case | Free variables | LCOH (USD/kg) |
|---|---|---|
| F0 | none | 21.3595 |
| F1 | solar, wind | 18.7569 |
| F2 | electrolysis, PEM share, battery, storage | 17.2520 |
| F3 | all six | 10.4339 |

The service loss 95th percentile of the selected design was 0.0467 % on the eight
realizations used to select it and 0.4207 % on the 20 independent realizations.
A reliability statistic computed on the realizations that select a design is part
of the selection procedure and is not independent reliability evidence.

The LCOH boundary covers generation, electrolysis, and storage. Compression,
precooling, dispensing, land, and financing fees are outside it, so the values
are production costs per kilogram served, not station retail prices.

---

## Repository layout

```
.
├── config.py                 bounds, techno economic parameters, optimizer settings
├── data_v2.py                data loading and seven day block bootstrap
├── dispatch_v2.py            hourly chronological dispatch
├── objectives_v2.py          LCOH, LPSP, curtailment, degradation, replacement
├── optimizer_v2.py           NSGA-III problem, hypervolume, entropy TOPSIS
│
├── convert_kpx.py            step 0  raw portal files -> processed profiles
├── run_continuous.py         step 1  state preserving NSGA-III search
├── eps_sweep.py              step 2  multi-start searches (recorded outputs)
├── final_select.py           step 3  archive, screening, selection, validation
├── abl_cont.py               step 4  ablation searches (recorded outputs)
├── f1_grid.py                step 5  deterministic F1 grid refinement
├── f23_check.py              step 6  F2 and F3 local refinement
├── polish_val.py             step 7  polish overfitting check
├── equiv.py                  pause/resume equivalence test
│
├── make_all_figures.py       manuscript figures from tables/ and data/
├── make_pareto_convergence.py  Pareto front and convergence figures
├── RESULTS_LOCKED.json       locked numerical results
├── ENVIRONMENT.json          recorded software environment
│
├── data/                     processed profiles, demand workbook, raw/ portal files
├── checkpoints/              search outputs and pipeline results
├── tables/                   result tables read by the figure scripts
├── figures/                  generated figures, PNG + PDF, with metadata sidecars
├── verification/             reference values and verify_reproduction.py
├── diagnostics/              warm-up and PEM overload sensitivity studies
├── superseded/               intermediate scripts not used for reported results
└── docs/                     reporting basis map
```

---

## Installation

Python 3.12.3 was used.

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones that reproduced every result
exactly. Other versions of pymoo or NumPy can change the optimizer trajectory,
so later results may differ slightly.

---

## Reproducing the results

All commands run from the repository root.

**Step 0 — processed data (optional).** The processed profiles are already in
`data/`. To rebuild them from the original portal files:

```bash
python3 convert_kpx.py
```

The output is byte identical to the shipped files.

**Step 1 — search.** The seed 42 run uses 182 designs for 100 generations. The
full optimizer state is serialized after each call, so the run can be split into
chunks and the result is identical to one uninterrupted run:

```bash
python3 run_continuous.py 42 25 100
python3 run_continuous.py 42 25 100
python3 run_continuous.py 42 25 100
python3 run_continuous.py 42 25 100
```

Arguments are seed, generations per call, and total generations. On one CPU core
a generation takes about 9 s, so the full search takes about 15 minutes. But depending 
on the CPU and the system, it can vary.

**Steps 3, 5, 6, 7 — selection and ablation.**

```bash
python3 final_select.py      # about 2.5 min
python3 f1_grid.py           # about 2.5 min
python3 f23_check.py         # about 1 min
python3 polish_val.py        # about 3 min
```

`eps_sweep.py` (step 2) and `abl_cont.py` (step 4) are full multi seed searches.
Their outputs are included in `checkpoints/` and read by the later steps, so they
do not need to be rerun to reproduce the reported results.

**Figures.**

```bash
python3 make_pareto_convergence.py
python3 make_all_figures.py
```

Each figure is written as PNG and PDF with a `.meta.json` sidecar recording its
source files, evaluation basis, number of realizations, and SHA-256 hashes.

---

## Verification

```bash
python3 verification/verify_reproduction.py
```

The script compares the files in `checkpoints/` with the reference copies in
`verification/reference/`. In the recorded environment every comparison returns
a maximum absolute difference of exactly 0.0:

| Result | Script | Difference |
|---|---|---|
| Search trace, 100 generations | `run_continuous.py` | 0.0 |
| Selected design, validation, hold-out | `final_select.py` | 0.0 |
| F1 grid refinement | `f1_grid.py` | 0.0 |
| F0–F3 ablation decomposition | `f23_check.py` | 0.0 |
| Polish overfitting and seed-7 replication | `polish_val.py` | 0.0 |

`equiv.py` separately checks that a 12 generation run and a 6 + 6 run with
serialization in between produce identical objectives and design variables.

---

## Model summary

**Decision vector.** `x = [P_PV, P_W, P_EL, x_PEM, E_B, M_H]`, with installed
alkaline (AWE) and PEM ratings `P_A = (1 − x_PEM) P_EL` and `P_P = x_PEM P_EL`.

| Variable | Lower | Upper |
|---|---|---|
| Photovoltaic | 500 kW | 8000 kW |
| Wind | 200 kW | 3000 kW |
| Total electrolysis | 100 kW | 3000 kW |
| PEM share | 0 | 1 |
| Battery energy | 50 kWh | 3000 kWh |
| Hydrogen storage | 50 kg | 2000 kg |

**Dispatch, per hour.**

1. Renewable power is the plant rating times the normalized fleet profile.
2. A production cap limits electrolysis to demand plus available storage
   headroom, converted to power with the smaller specific consumption, so the
   cap is conservative.
3. AWE is dispatched first between 20 % and 100 % of its rating; PEM takes the
   remainder between 5 % and 120 % of its rating.
4. Production is recomputed with the actual technology-specific consumption and
   checked against the cap after allocation.
5. The battery charges or discharges within 10–90 % state of charge at up to
   0.5C; remaining renewable energy is curtailed.
6. Hydrogen serves demand directly; deficits draw on storage above its 5 % floor;
   surplus enters storage after a 1 % transfer loss.

Each evaluation runs the full chronology once as an unrecorded warm-up to set
the storage states, then again to record metrics.

**Objectives.** LCOH in USD per kilogram served, mass based service loss, and
curtailment as a share of available renewable energy. Stack replacement follows
an accumulated voltage loss model with a harsher degradation rate below 80 % load
for AWE and above rated power for PEM.

**Optimizer.** NSGA-III with 91 Das–Dennis reference directions (p = 12),
population 182, 100 generations, seed 42. Each candidate is evaluated on two
bootstrap demand realizations.

All parameters are in `config.py`.

---

## Evaluation bases

Separate sets of bootstrap demand realizations are used for each purpose, and no
reported value combines them.

| Basis | Demand seeds | Count | Purpose |
|---|---|---|---|
| SEARCH | 0–1 | 2 | objective evaluation inside NSGA-III |
| SCREENING | 200–207 | 8 | design selection, F1 grid, F2/F3 refinement |
| PRIMARY | 500–519 | 20 | independent performance estimate |
| HOLDOUT | 99–102 | 4 | 2023 renewable chronology |
| Polish check | 400–407 | 8 | fresh set for the refinement overfitting test |

`docs/results_basis_map.md` assigns every reported number to its basis.

---

## Data

**Renewable generation.** Hourly Jeju solar and wind fleet generation from the
Korea Power Exchange, downloaded from the Public Data Portal of the Republic of
Korea (dataset 15127502). The original files are in `data/raw/` (CP949 encoding).
`convert_kpx.py` removes negative values, subtracts a nighttime solar offset, and
divides each year by the 99.9th percentile of that year's hourly output. The
result is a normalized index, not a verified regional capacity factor: the
denominator is a statistical proxy for installed fleet capacity.

**Hydrogen demand.** Daily sales at the Jeju-do Hamdeok Green Hydrogen Refueling
Station from Korea Gas Technology Corporation, also from the Public Data Portal
(dataset 15150250). 255 days from 1st November 2024 to 15 July 2025 are used; 
there is one day record missing in it. Demand chronologies are
built by resampling seven day blocks, and each day's total is spread evenly over
24 hours.

The renewable (2020–2023) and demand (2024–2025) records are not contemporaneous.

Redistribution of the portal files follows the Public Data Portal's current terms.

---

## Diagnostics

`diagnostics/warmup_sensitivity/` evaluates the frozen design with zero, one,
three, and five full horizon warm-up passes, and with an iterative fixed point
criterion, to test whether the single production warm-up pass is sufficient.

`diagnostics/pem_overload/` re-evaluates the frozen design with the PEM ceiling at
100 %, 110 %, and 120 % of rating. Removing overload changes mean LCOH by
−0.060 USD/kg, about half the across draw standard deviation, and raises the
service loss 95th percentile from 0.421 % to 0.585 %. The design is not
re-optimized in this test.
---

## Superseded code

`superseded/` contains intermediate scripts kept for provenance. None produced a
reported result, and several produced values that were later withdrawn. In
particular, `run_chunk.py` rebuilt the optimizer every chunk and discarded its
internal state, and `archive.py` builds an archive that includes those restarted
runs. See `superseded/README.md`.

---

## Citation

If you use this code, please cite the accompanying article (details to be added
on publication):

> M. A. Ejaz, S. S. A. Naqvi, M. A. Khan, D.-H. Kim. Chronological Co-Sizing and Independent Reliability Validation of
Renewable Hydrogen Refueling Systems. Department of Computer Engineering, Jeju National
> University.


## Contact

Muhammad Adnan Ejaz (1st Author), adnan.ejaz@stu.jejunu.ac.kr
Do-Hyeun Kim (corresponding author), kimdh@jejunu.ac.kr

