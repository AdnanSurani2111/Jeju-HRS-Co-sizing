"""verify_reproduction.py - compare regenerated results against the reference values.

Run from the repository root after regenerating results (see README.md):

    python3 verification/verify_reproduction.py

Each check compares a freshly written file in checkpoints/ with the copy kept in
verification/reference/. Every quantity is expected to match exactly (difference
0.0). A non-zero difference indicates a changed environment or code.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REF = os.path.join(HERE, 'reference')
NEW = os.path.join(CODE, 'checkpoints')


def load(folder, name):
    return json.load(open(os.path.join(folder, name)))


def max_diff(a, b):
    """Largest absolute difference over nested dicts, lists and numbers."""
    if isinstance(b, dict):
        return max((max_diff(a[k], b[k]) for k in b), default=0.0)
    if isinstance(b, list):
        if b and isinstance(b[0], dict):
            if len(a) != len(b):
                return float('inf')
            return max((max_diff(x, y) for x, y in zip(a, b)), default=0.0)
        va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return float('inf') if va.shape != vb.shape else float(np.abs(va - vb).max(initial=0.0))
    if isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float('inf')


CHECKS = [
    ('cont_hv_42.json', 'Search trace, seed 42, 100 generations', 'run_continuous.py 42 <n> 100'),
    ('FINAL.json', 'Selected design and 20-draw / hold-out validation', 'final_select.py'),
    ('f1_grid.json', 'F1 renewable-only grid refinement', 'f1_grid.py'),
    ('ablation_final.json', 'F0-F3 ablation decomposition', 'f23_check.py'),
    ('polish_val.json', 'F3 polish overfitting and seed-7 replication', 'polish_val.py'),
]

failures = 0
print(f"{'result':52s} {'max |difference|':>18s}  status")
for fname, label, script in CHECKS:
    if not os.path.exists(os.path.join(NEW, fname)):
        print(f"{label:52s} {'-':>18s}  NOT REGENERATED (run {script})")
        failures += 1
        continue
    d = max_diff(load(NEW, fname), load(REF, fname))
    ok = d == 0.0
    failures += 0 if ok else 1
    print(f"{label:52s} {d:18.3e}  {'EXACT' if ok else 'DIFFERS'}")

print()
print('ALL RESULTS REPRODUCED EXACTLY' if failures == 0 else f'{failures} check(s) did not reproduce')
sys.exit(1 if failures else 0)
