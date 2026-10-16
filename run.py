"""Safe entry point; historical results are never overwritten by this wrapper."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','build','recheck-audit','check-model','figures']);args=p.parse_args()
os.chdir(ROOT)
if args.command=='build':
 subprocess.run(['g++','-O3','-fPIC','-shared','-fopenmp','revision/fast_dispatch.cpp','-o','revision/fast_dispatch.so'],check=True)
else:
 from verify_release import verify
 import json
 print(json.dumps(verify(),indent=2),flush=True)
 if args.command=='recheck-audit':
  import numpy as np
  sys.path.insert(0,str(ROOT/'full_search'));from core import evaluate,demands
  old=np.load(ROOT/'full_search/audit_metrics.npz');cost,metrics=evaluate(old['X'],demands(old['seeds']))
  assert np.allclose(cost,old['cost'],rtol=0,atol=1e-9) and np.allclose(metrics,old['metrics'],rtol=0,atol=1e-7)
  print('PASS: all 10 vectors x 2000 recorded audit scenarios reproduced. Historical outputs untouched.')
 elif args.command=='check-model':
  import numpy as np
  sys.path.insert(0,str(ROOT/'full_search'));from core import evaluate,demands,PS,PW
  from objectives_v2 import evaluate_population
  X=np.array([[1956.9303445,763.4634311,1018.9559527,.5765450827,637.1814738,1951.4978884],[5500,1400,800,.4,500,2000],[500,200,100,1,50,50]])
  d=demands([60000]);cost,m=evaluate(X,d);r=evaluate_population(X,PS,PW,d[0])
  for key,i in [('lpsp',0),('h2_served',1),('curtailment',2),('awe_deg',3),('pem_deg',4),('awe_cycles',5),('pem_cycles',6)]:assert np.allclose(m[:,0,i],r[key],rtol=0,atol=1e-6),key
  assert np.allclose(cost[:,0],r['lcoh'],rtol=0,atol=1e-6);print('PASS: accelerated dispatcher matches original Python on three capacity vectors.')
 elif args.command=='figures':subprocess.run([sys.executable,'make_updated_figures.py'],check=True)
