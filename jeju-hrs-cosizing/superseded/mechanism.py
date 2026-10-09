"""Why is the optimal split interior? Sweep x_PEM with all other variables held at
the recommended design and decompose the resulting cost curve."""
import numpy as np, pandas as pd, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import config as C, data_v2 as D
from objectives_v2 import evaluate_population
plt.rcParams.update({'font.size':9,'figure.dpi':200,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':.3})
re=D.load_re(); ps,pw=re['train']; dh,_=D.demand_series(len(ps),seed=0)
rec=np.array(json.load(open('checkpoints/results_numbers.json'))['recommended'].values().__iter__().__next__()) if False else None
r=json.load(open('checkpoints/results_numbers.json'))['recommended']
base=np.array([r['P_solar'],r['P_wind'],r['P_elz'],r['x_PEM'],r['E_BESS'],r['M_HSS']])
xs=np.linspace(0.0,1.0,21)
X=np.tile(base,(len(xs),1)); X[:,3]=xs
m=evaluate_population(X,ps,pw,dh)
T=pd.DataFrame(dict(x_PEM=xs,LCOH=m['lcoh'],LPSP=m['lpsp'],CURT=m['curtailment'],
                    capex=m['ann_capex'],om=m['ann_om'],repl=m['ann_repl'],cyc=m['ann_cyc'],
                    h2=m['h2_served'],awe_h=m['awe_hours'],pem_h=m['pem_hours'],
                    awe_cyc=m['awe_cycles'],pem_cyc=m['pem_cycles'],
                    e_awe=m['e_awe'],e_pem=m['e_pem']))
T.round(4).to_csv('tables/T12_xpem_mechanism.csv',index=False)
i=int(np.argmin(T.LCOH.values))
print(f"minimum at x_PEM = {xs[i]:.2f}, LCOH {T.LCOH[i]:.3f}")
print(T[['x_PEM','LCOH','LPSP','CURT','capex','repl','h2','awe_h','pem_h','awe_cyc']].iloc[::4].round(3).to_string(index=False))
print(f"\nendpoints: x=0 -> LCOH {T.LCOH.iloc[0]:.2f} (LPSP {T.LPSP.iloc[0]:.4f}), "
      f"x=1 -> LCOH {T.LCOH.iloc[-1]:.2f} (LPSP {T.LPSP.iloc[-1]:.4f})")
print(f"penalty vs interior optimum: left {T.LCOH.iloc[0]-T.LCOH[i]:+.2f}, right {T.LCOH.iloc[-1]-T.LCOH[i]:+.2f}")

fig,ax=plt.subplots(1,3,figsize=(9.8,2.9))
ax[0].plot(xs,T.LCOH,'o-',color='tab:red'); ax[0].axvline(xs[i],ls='--',c='grey',lw=.8)
ax[0].set_xlabel('$x_{PEM}$'); ax[0].set_ylabel('LCOH (USD/kg)')
ax[0].set_title(f'(a) Cost curve, minimum at {xs[i]:.2f}',fontsize=8)
ax[1].plot(xs,T.h2/1000,'o-',color='tab:blue',label='H$_2$ served')
ax[1].set_xlabel('$x_{PEM}$'); ax[1].set_ylabel('H$_2$ served (t/yr)')
a2=ax[1].twinx(); a2.plot(xs,100*T.LPSP,'s-',color='tab:orange'); a2.set_ylabel('LPSP (%)',color='tab:orange')
a2.grid(False); ax[1].set_title('(b) Left arm: AWE minimum load\ncosts delivered hydrogen',fontsize=8)
ax[2].plot(xs,T.capex,'o-',label='CAPEX'); ax[2].plot(xs,T.repl,'s-',label='Replacement')
ax[2].plot(xs,T.om,'^-',label='O&M'); ax[2].plot(xs,T.cyc,'v-',label='Cyclic')
ax[2].set_xlabel('$x_{PEM}$'); ax[2].set_ylabel('USD/kg'); ax[2].legend(fontsize=6.5)
ax[2].set_title('(c) Right arm: PEM capital and\nreplacement cost',fontsize=8)
for e in ('png','pdf'): fig.savefig(f'figures/F14_mechanism.{e}')
print('figure F14 written')
