"""Optional costly deterministic replay, isolated from frozen campaign records.
Run: OMP_NUM_THREADS=8 python replay_diagnostics.py --case F3 --seed 7
Only use curves when equivalence.json reports PASS.
"""
from pathlib import Path
import argparse,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'full_search'))
from verify_release import verify

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',choices=['F2','F3'],required=True);ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    verify()
    from run_campaign import CapacityProblem,expand,P
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.indicators.hv import HV
    if args.seed not in P['optimizer_seeds']:ap.error('Seed must belong to recorded protocol')
    out=ROOT/'diagnostic_replays'/f'{args.case}_{args.seed}'
    if out.exists():ap.error('Replay output already exists; preserve or move it before a new replay')
    out.mkdir(parents=True)
    # Post hoc diagnostic region, common to every replay: USD/kg / 40,
    # raw loss fraction and raw curtailment fraction. Outside-box points contribute zero.
    scale=np.array([40.,1.,1.]);ref=np.array([1.,1.,1.]);indicator=HV(ref_point=ref)
    a=NSGA3(pop_size=P['population'],ref_dirs=get_reference_directions('das-dennis',3,n_partitions=12))
    a.setup(CapacityProblem(args.case),termination=('n_gen',P['generations']),seed=args.seed,verbose=False,save_history=False)
    history=[]
    while a.has_next():
        a.next();F=a.pop.get('F');n=int(a.n_gen-1)
        scaled=F/scale;inside=np.all(scaled<ref,axis=1)
        history.append(dict(generation=n,n_eval=int(a.evaluator.n_eval),current_population_hypervolume=float(indicator(scaled[inside])) if inside.any() else 0.,outside_reference_box=int((~inside).sum())))
        np.savez_compressed(out/f'population_{n:03d}.npz',X=expand(a.pop.get('X'),args.case),F=F,n_eval=a.evaluator.n_eval)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if n%10==0:print(args.case,args.seed,n,flush=True)
    stored=np.load(ROOT/'full_search'/f'{args.case}_{args.seed}'/'final.npz',allow_pickle=False)
    checks=dict(X_exact=bool(np.array_equal(expand(a.pop.get('X'),args.case),stored['X'])),F_exact=bool(np.array_equal(a.pop.get('F'),stored['F'])),budget_exact=int(a.evaluator.n_eval)==int(stored['n_eval']))
    passed=all(checks.values());checks['status']='PASS' if passed else 'FAIL'
    (out/'equivalence.json').write_text(json.dumps(checks,indent=2))
    (out/'hypervolume_method.json').write_text(json.dumps(dict(scale=scale.tolist(),scaled_reference_point=ref.tolist(),raw_reference_point=[40.,1.,1.],quantity='Current population, every generation',outside_points='Excluded from reference-box contribution',status='Post hoc diagnostic; not a selection rule',limitations='Does not establish global convergence; inspect excluded counts'),indent=2))
    if not passed:raise RuntimeError('Replay differs from archived final state; do not use curves as historical diagnostics')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([r['n_eval'] for r in history],[r['current_population_hypervolume'] for r in history]);plt.xlabel('Candidate objective evaluations');plt.ylabel('Current-population hypervolume');plt.title(f'{args.case}, seed {args.seed}: verified replay');plt.tight_layout();plt.savefig(out/'hypervolume.pdf');plt.close()
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
