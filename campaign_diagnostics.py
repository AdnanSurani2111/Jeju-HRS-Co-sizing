"""Read-only three-objective diagnostics and complete decision export.
Run from repository root: python campaign_diagnostics.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_release import verify
ROOT=Path(__file__).resolve().parent

def main():
    verify()
    src=ROOT/'full_search'; out=ROOT/'campaign_diagnostics';out.mkdir(exist_ok=True)
    protocol=json.loads((src/'protocol.json').read_text())
    report=json.loads((src/'audit_results.json').read_text())
    rows=report['decisions'];seeds=protocol['optimizer_seeds']
    labels=['Mean training cost (USD/kg)','Mean training mass loss (%)','Mean training curtailment (%)']
    fig,axes=plt.subplots(3,2,figsize=(11,12),layout='constrained')
    export=[]
    for col,case in enumerate(['F3','F2']):
        for j,seed in enumerate(seeds):
            z=np.load(src/f'{case}_{seed}'/'final.npz',allow_pickle=False)
            F=z['F']; X=z['X'];nd=np.array([i for i,f in enumerate(F) if not np.any(np.all(F<=f,axis=1) & np.any(F<f,axis=1))],dtype=int)
            selected=sorted({r['selected_index'] for r in rows if r['case']==case and r['optimizer_seed']==seed and r['selected_index'] is not None})
            shown=F*np.array([1,100,100])
            for i,(f,x) in enumerate(zip(F,X)):
                export.append(dict(case=case,optimizer_seed=seed,population_index=i,nondominated_3_objectives=i in nd,frozen_selection=i in selected,**dict(zip(['training_cost','training_loss_fraction','training_curtailment_fraction'],f)),**dict(zip(['pv_kw','wind_kw','electrolyzer_kw','pem_fraction','battery_kwh','hss_kg'],x))))
            for row,(a,b) in enumerate([(0,1),(0,2),(2,1)]):
                ax=axes[row,col]
                ax.scatter(shown[:,a],shown[:,b],s=8,color='.85',alpha=.45)
                ax.scatter(shown[nd,a],shown[nd,b],s=15,color=f'C{j}',label=f'Seed {seed}')
                if selected:ax.scatter(shown[selected,a],shown[selected,b],s=100,marker='*',color=f'C{j}',edgecolor='black',linewidth=.6)
                ax.set(xlabel=labels[a],ylabel=labels[b]);ax.grid(alpha=.15)
        axes[0,col].set_title('Joint sizing (F3)' if case=='F3' else 'Restricted sizing (F2)')
        axes[0,col].legend(fontsize=8)
    fig.suptitle('Three-objective final-population front approximations\nStars: frozen selections; training objectives do not imply audit qualification',fontsize=12)
    for ext in ['pdf','svg','png']:fig.savefig(out/f'pareto_approximations.{ext}',dpi=180)
    plt.close(fig);pd.DataFrame(export).to_csv(out/'final_population_objectives.csv',index=False)
    audit=np.load(src/'audit_metrics.npz',allow_pickle=False)['X']
    decisions=[]
    for r in rows:
        d=dict(r)
        if r['selected_index'] is None:
            d.update(design_id='',selection_status='No selection',audit_status='Not applicable',familywise_status='Not applicable')
        else:
            i=int(r['audit_index']);d.update(design_id=f'D{i+1:02d}',selection_status='Selected',audit_status='Pass' if r['audit_pass'] else 'Fail',familywise_status='Pass' if r['audit_familywise_pass'] else 'Fail')
            d.update(dict(zip(['pv_kw','wind_kw','electrolyzer_kw','pem_fraction','battery_kwh','hss_kg'],audit[i])))
        d.pop('x',None);decisions.append(d)
    pd.DataFrame(decisions).to_csv(out/'all_30_decisions_and_capacities.csv',index=False)
    summary=[]
    for case in ['F3','F2']:
        for seed in seeds:
            rr=[r for r in rows if r['case']==case and r['optimizer_seed']==seed];sel=[r for r in rr if r['selected_index'] is not None]
            summary.append(dict(case=case,optimizer_seed=seed,attempted=3,selected=len(sel),audit_pass_among_returned=sum(r['audit_pass'] for r in sel) if sel else 'Not applicable',familywise_pass_among_returned=sum(r['audit_familywise_pass'] for r in sel) if sel else 'Not applicable',qualified_decisions=sum(r['audit_familywise_pass'] for r in sel)))
    pd.DataFrame(summary).to_csv(out/'search_outcomes.csv',index=False)
    (out/'diagnostic_metadata.json').write_text(json.dumps(dict(nondominance='All three objectives within each final population; no pooling or reselection',stars='Frozen selections including any training-dominated selections',hypervolume_history='Unavailable in historical trace: only scalar minima and final population retained',protocol_unchanged=True),indent=2))
    print(out)
if __name__=='__main__':main()
