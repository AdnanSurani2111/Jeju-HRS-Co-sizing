# Code release update

This release addresses the computational recommendations in the four supplied
assessment notes. It contains the complete previous code/data/checkpoint release,
plus three-objective front plots, complete capacity/decision exports, and a
separate deterministic replay tool for recovering convergence histories.

## Changes

- Added `campaign_diagnostics.py`: per-run three-objective nondominance, six pairwise
  panels, frozen-selection markers, numeric objective export, all capacities,
  stable audit-vector IDs, and explicit not-applicable audit entries for no selection.
- Added `replay_diagnostics.py`: separate per-generation population storage,
  fixed diagnostic hypervolume settings, outside-box counts, exact final-state
  equivalence gate, and a plot only after successful equivalence.
- Updated README, GitHub instructions, ignore rules, changelog and release hashes.
- Kept the frozen model, protocol, selections, screening and audit bytes intact.

## Verification and limits

Stored-result verification and diagnostics generation are executed for this release.
New scripts are syntax checked. Replay optimization has not been executed; it is
an optional costly computation, not completed evidence. Original historical
hypervolume curves cannot be reconstructed from scalar-minimum traces alone.
The new plots use actual final objective matrices, without inferring coordinates
from PDFs. Existing numerical-model and audit verification scripts are included;
this update does not claim another execution of all physical simulations.

These are code and diagnostic deliverables. The supplied recommendations about
manuscript organization, literature review, and supplement restructuring require
corresponding document edits and are not represented as completed here.
