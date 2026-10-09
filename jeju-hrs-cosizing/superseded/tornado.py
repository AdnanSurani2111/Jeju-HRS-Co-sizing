"""Tornado over techno-economic parameters. Reports LCOH of the matched-reliability
hybrid and AWE-only designs and, critically, whether the sign of their difference
survives plausible parameter variation."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import config as C, data_v2 as D
import objectives_v2 as OB
plt.rcParams.update({'font.size':9,'figure.dpi':200,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':.3})

re=D.load_re(); ps,pw=re['train']; dh,_=D.demand_series(len(ps),seed=0)
def pick(tech,eps=0.01):
    F=np.load(f'checkpoints/eps_F_{tech}.npy'); X=np.load(f'checkpoints/eps_X_{tech}.npy')
    ok=np.where(F[:,1]<=eps+1e-12)[0]; return X[ok[np.argmin(F[ok,0])]]
HYB, AWE = pick('hybrid'), pick('awe')

def lcoh(x,**kw):
    return float(OB.evaluate_population(np.asarray(x)[None,:],ps,pw,dh,**kw)['lcoh'][0])

PARAMS=[('AWE CAPEX','AWE_CAPEX_PER_KW',0.3),('PEM CAPEX','PEM_CAPEX_PER_KW',0.3),
        ('AWE SEC','AWE_SEC_KWH_PER_KG',0.10),('PEM SEC','PEM_SEC_KWH_PER_KG',0.10),
        ('AWE stack life','AWE_V_REF_UV',0.3),('PEM stack life','PEM_V_REF_UV',0.3),
        ('Solar CAPEX','SOLAR_CAPEX_PER_KW',0.3),('Wind CAPEX','WIND_CAPEX_PER_KW',0.3),
        ('HSS CAPEX','HSS_CAPEX_PER_KG',0.3),('BESS CAPEX','BESS_CAPEX_PER_KWH',0.3)]
rows=[]
base_h, base_a = lcoh(HYB), lcoh(AWE)
for name,attr,frac in PARAMS:
    orig=getattr(C,attr)
    if attr in ('AWE_REPLACE_THRESHOLD_UV','PEM_REPLACE_THRESHOLD_UV'): pass
    out={}
    for sgn,tag in ((1-frac,'low'),(1+frac,'high')):
        setattr(C,attr,orig*sgn)
        if attr=='AWE_V_REF_UV': C.AWE_REPLACE_THRESHOLD_UV=0.25*C.AWE_V_REF_UV
        if attr=='PEM_V_REF_UV': C.PEM_REPLACE_THRESHOLD_UV=0.25*C.PEM_V_REF_UV
        out[tag]=(lcoh(HYB),lcoh(AWE))
    setattr(C,attr,orig)
    C.AWE_REPLACE_THRESHOLD_UV=0.25*C.AWE_V_REF_UV; C.PEM_REPLACE_THRESHOLD_UV=0.25*C.PEM_V_REF_UV
    rows.append(dict(Parameter=name,pct=int(100*frac),
                     hyb_low=out['low'][0],hyb_high=out['high'][0],
                     awe_low=out['low'][1],awe_high=out['high'][1],
                     d_low=out['low'][0]-out['low'][1],d_high=out['high'][0]-out['high'][1]))
for dr in (0.04,0.08):
    rows.append(dict(Parameter=f'Discount {dr:.0%}',pct=0,hyb_low=lcoh(HYB,discount_rate=dr),
                     hyb_high=np.nan,awe_low=lcoh(AWE,discount_rate=dr),awe_high=np.nan,
                     d_low=lcoh(HYB,discount_rate=dr)-lcoh(AWE,discount_rate=dr),d_high=np.nan))
T=pd.DataFrame(rows)
T['swing']=(T[['hyb_low','hyb_high']].max(1)-T[['hyb_low','hyb_high']].min(1))
T.round(4).to_csv('tables/T11_tornado.csv',index=False)
print(f'base LCOH  hybrid {base_h:.3f}  AWE-only {base_a:.3f}  delta {base_h-base_a:+.3f}')
print(T[['Parameter','hyb_low','hyb_high','d_low','d_high']].round(3).to_string(index=False))
dd=T[['d_low','d_high']].values.ravel(); dd=dd[~np.isnan(dd)]
print(f'\ndelta(hybrid-AWE) range over all variations: {dd.min():+.3f} to {dd.max():+.3f}')
print('sign flips:', bool((dd>0).any() and (dd<0).any()))

Tt=T[T.pct>0].copy(); Tt=Tt.sort_values('swing')
fig,ax=plt.subplots(figsize=(6.0,3.4))
y=np.arange(len(Tt))
for i,(_,r) in enumerate(Tt.iterrows()):
    ax.barh(i,r.hyb_high-base_h,left=base_h,color='indianred',height=.6)
    ax.barh(i,r.hyb_low-base_h,left=base_h,color='steelblue',height=.6)
ax.axvline(base_h,color='k',lw=1)
ax.set_yticks(y); ax.set_yticklabels([f"{r.Parameter} ±{r.pct}%" for _,r in Tt.iterrows()],fontsize=7)
ax.set_xlabel('LCOH (USD/kg)'); ax.set_title('Tornado: recommended hybrid design',fontsize=9)
for e in ('png','pdf'): fig.savefig(f'figures/F13_tornado.{e}')
print('figure F13 written')
