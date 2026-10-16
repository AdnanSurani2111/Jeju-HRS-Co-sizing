"""Read-only integrity and numerical audit of the recorded complete-search campaign."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent
F=ROOT/'full_search'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def upper(k,n,alpha=.05):
 k=np.asarray(k);return np.where(k<n,beta.ppf(1-alpha,k+1,n-k),1.)
def verify():
 p=json.loads((F/'protocol.json').read_text());ph=digest(F/'protocol.json');assert ph==(F/'protocol.sha256').read_text().strip()
 for name,h in p['source_sha256'].items():assert digest(ROOT/name)==h, f'Changed historical model/input: {name}'
 # Bind every distributed record, including screen files without embedded hashes,
 # to the release manifest. This is a corruption/mismatch check, not a signature.
 manifest=json.loads((ROOT/'release_manifest.json').read_text())
 for name,h in manifest.items():assert digest(ROOT/name)==h,f'Release mismatch: {name}'
 sh=digest(F/'selections_frozen.json');assert sh==(F/'selections_frozen.sha256').read_text().strip()
 frozen=json.loads((F/'selections_frozen.json').read_text());assert frozen['protocol_sha256']==ph
 rows=frozen['decisions'];assert len(rows)==30
 keys={(r['case'],r['optimizer_seed'],r['screen_bank']) for r in rows};assert len(keys)==30
 reproduced=[]
 for seed in p['optimizer_seeds']:
  for case in ['F3','F2']:
   folder=F/f'{case}_{seed}';a=np.load(folder/'final.npz',allow_pickle=False);X=a['X'];obj=a['F']
   assert X.shape==(182,6) and obj.shape==(182,3) and np.isfinite(obj).all()
   assert int(a['optimizer_seed'])==seed and int(a['n_eval'])==18200 and int(a['generations'])==100
   assert str(a['protocol_sha256'])==ph
   assert np.all(X>=np.array(p['bounds']['lower'])-1e-9) and np.all(X<=np.array(p['bounds']['upper'])+1e-9)
   if case=='F2':assert np.all(X[:,:2]==[5500.,1400.])
   c=json.loads((folder/'completed.json').read_text());assert c['n_eval']==18200 and c['generations']==100 and c['protocol_sha256']==ph
   trace=json.loads((folder/'trace.json').read_text());assert len(trace)==100
   assert [r['generation'] for r in trace]==list(range(1,101)) and [r['n_eval'] for r in trace]==list(range(182,18201,182))
   for bank,seeds in p['screening_banks'].items():
    z=np.load(folder/f'{bank}_screen.npz',allow_pickle=False);cost=z['cost'];m=z['metrics'];assert cost.shape==(182,400) and m.shape==(182,400,10)
    assert np.array_equal(z['seeds'],seeds)
    mask=np.isfinite(cost);assert np.array_equal(np.isfinite(m).all(2),mask)
    assert np.all(np.diff(mask.astype(int),axis=1)<=0),'Computed entries must form a prefix'
    k=(m[:,:,0]>.001).sum(1);complete=mask.all(1);assert np.all(k[~complete]>=5)
    eligible=np.flatnonzero(upper(k,400)<=.025);assert np.all(complete[eligible])
    assert np.nanmax(np.abs(m[:,:,7]))<1e-6
    row=next(r for r in rows if (r['case'],r['optimizer_seed'],r['screen_bank'])==(case,seed,bank))
    assert row['n_passing']==len(eligible)
    idx=int(eligible[np.argmin(cost[eligible].mean(1))]) if len(eligible) else None
    assert row['selected_index']==idx
    if idx is not None:
     assert np.array_equal(row['x'],X[idx]);assert np.isclose(row['screen_cost'],cost[idx].mean(),rtol=0,atol=1e-10)
     assert row['screen_k']==int(k[idx]) and np.isclose(row['screen_upper'],upper(k[idx],400))
    reproduced.append(row)
 selected=[r for r in rows if r['selected_index'] is not None];X,inv=np.unique(np.array([r['x'] for r in selected]),axis=0,return_inverse=True)
 a=np.load(F/'audit_metrics.npz',allow_pickle=False);assert np.array_equal(a['X'],X) and np.array_equal(a['seeds'],p['audit_demand_seeds'])
 assert str(a['protocol_sha256'])==ph and str(a['selections_sha256'])==sh
 assert a['cost'].shape==(len(X),2000) and a['metrics'].shape==(len(X),2000,10)
 assert np.isfinite(a['cost']).all() and np.isfinite(a['metrics']).all() and np.max(np.abs(a['metrics'][:,:,7]))<1e-6
 report=json.loads((F/'audit_results.json').read_text());assert report['protocol_sha256']==ph and report['selections_sha256']==sh
 for r,i in zip(selected,inv):
  q=next(v for v in report['decisions'] if (v['case'],v['optimizer_seed'],v['screen_bank'])==(r['case'],r['optimizer_seed'],r['screen_bank']))
  k=int((a['metrics'][i,:,0]>.001).sum());u=float(upper(k,2000));uf=float(upper(k,2000,.05/30))
  assert q['audit_k']==k and q['audit_n']==2000 and q['audit_index']==i
  assert np.isclose(q['audit_upper'],u) and np.isclose(q['audit_familywise_upper'],uf)
  assert q['audit_pass']==(u<=.05) and q['audit_familywise_pass']==(uf<=.05)
  assert np.isclose(q['audit_cost'],a['cost'][i].mean(),rtol=0,atol=1e-10)
 for pair in report['pairs']:
  rr=[v for v in report['decisions'] if v['optimizer_seed']==pair['optimizer_seed'] and v['screen_bank']==pair['screen_bank']]
  if all(v['selected_index'] is not None for v in rr):
   costs={v['case']:v['audit_cost'] for v in rr};assert np.isclose(pair['cost_reduction_percent'],100*(1-costs['F3']/costs['F2']))
 result={'status':'PASS','complete_optimizer_runs':10,'candidate_objective_evaluations':182000,'screening_decisions':30,'selected_decisions':len(selected),'distinct_audited_vectors':len(X),'audit_scenarios':2000,'protocol_sha256':ph,'scope':'Stored records, hashes, selection certificates and audit statistics; no new optimization or external validation.'}
 return result
if __name__=='__main__':print(json.dumps(verify(),indent=2))
