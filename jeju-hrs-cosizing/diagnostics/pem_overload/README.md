# PEM overload sensitivity

## Question

The dispatch model permits PEM operation up to 120% of its installed rating.
Does the reported performance depend on that allowance?

## Method

The frozen selected design is re-evaluated on the 20 primary validation
realizations (seeds 500-519) with `PEM_MAX_LOAD_FRAC` set to 1.00, 1.10 and
1.20. The design is **not** re-optimized.

Run with:

    python3 pem_overload.py

## Result

| PEM cap | LCOH (USD/kg) | SD | Service loss q95 (%) | Curtailment (%) |
|---|---|---|---|---|
| 1.00 | 11.6585 | 0.1137 | 0.5846 | 21.44 |
| 1.10 | 11.7233 | 0.1163 | 0.4664 | 21.32 |
| 1.20 (reported) | 11.7184 | 0.1185 | 0.4207 | 21.24 |

Removing overload entirely changes mean LCOH by -0.060 USD/kg, about half the
across-draw standard deviation of 0.119, so the cost result is not
distinguishable from sampling variation. The q95 service loss rises from 0.421%
to 0.585%, a 39% relative increase that remains well inside the 1% engineering
target.

LCOH is marginally *lower* at the 1.00 cap because less hydrogen is served
(65,923 against 66,012 kg/yr) while PEM also spends fewer hours in the harsh
degradation regime, so the replacement term falls faster than the denominator.

## Scope

This is re-evaluation of a fixed design, not re-optimization. A design optimized
under a 1.00 cap could select different capacities. That test requires a new
production run and is listed in the remaining-work section.
