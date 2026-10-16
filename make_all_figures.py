"""Publication figures built only from the locked result bases and final tables."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
TAB = ROOT / "tables"
DATA = ROOT / "data"
LOCK = json.loads((ROOT / "RESULTS_LOCKED.json").read_text())
NAVY, BLUE, TEAL, GOLD, RED, GREY = "#17365D", "#2E75B6", "#2A9D8F", "#E9A23B", "#C94C4C", "#667085"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,
                     "axes.labelsize":9,"legend.fontsize":8,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.dpi":150,"savefig.dpi":300})
manifest=[]

def save(fig, fid, basis, sources, n=0, placement="main text", note=""):
    fig.tight_layout()
    pdf=FIG/f"{fid}.pdf"; png=FIG/f"{fid}.png"
    fig.savefig(pdf,bbox_inches="tight"); fig.savefig(png,bbox_inches="tight")
    plt.close(fig)
    meta={"figure_id":fid,"basis":basis,"number_of_realizations":n,
          "source_files":sources,"placement":placement,"note":note,
          "generated_at":datetime.now(timezone.utc).isoformat(),"script":"make_all_figures.py",
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "pdf_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
          "png_sha256":hashlib.sha256(png.read_bytes()).hexdigest()}
    (FIG/f"{fid}.meta.json").write_text(json.dumps(meta,indent=2))
    manifest.append(meta)

def box(ax,xy,w,h,title,sub,color=BLUE):
    p=FancyBboxPatch(xy,w,h,boxstyle="round,pad=0.012,rounding_size=0.018",
                     transform=ax.transAxes,facecolor="#F5F8FC",edgecolor=color,lw=1.5)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h*.62,title,ha="center",va="center",weight="bold",color=NAVY,transform=ax.transAxes)
    ax.text(xy[0]+w/2,xy[1]+h*.30,sub,ha="center",va="center",fontsize=8,color=GREY,transform=ax.transAxes)

# 1. Data-to-validation architecture.
fig,ax=plt.subplots(figsize=(7.2,4.0));ax.axis("off")
box(ax,(.03,.62),.25,.23,"Public data","KPX hourly generation\nHamdeok daily sales",TEAL)
box(ax,(.375,.62),.25,.23,"Scenario construction","2020--2022 chronology\n7-day demand blocks",BLUE)
box(ax,(.72,.62),.25,.23,"Six-variable design","PV, wind, AWE/PEM\nBESS and HSS",GOLD)
box(ax,(.20,.13),.26,.23,"Chronological dispatch","26,304 hours\nmass and energy balances",BLUE)
box(ax,(.57,.13),.26,.23,"Independent evaluation","20 fresh demand draws\n2023 chronology: 4 draws",RED)
for a,b in [((.28,.735),(.375,.735)),((.625,.735),(.72,.735)),((.845,.62),(.46,.36)),((.46,.245),(.57,.245))]:
    ax.annotate("",xy=b,xytext=a,xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=1.8,color=NAVY))
ax.text(.5,.96,"Study design and evidence separation",ha="center",va="center",fontsize=13,weight="bold",color=NAVY,transform=ax.transAxes)
save(fig,"F01_study_design","SCHEMATIC",["data_v2.py","optimizer_v2.py","results_basis_map.md"],note="No numerical result encoded")

# 2. Renewable indices and demand characterization.
S=pd.read_csv(DATA/"Jeju_Solar_Generation_2020_2023_KPX.csv")
W=pd.read_csv(DATA/"Jeju_Wind_Generation_2020_2023_KPX.csv")
H=[f"H{i:02d}" for i in range(24)]
rs=pd.read_csv(TAB/"T01_re_statistics.csv")
d=pd.read_excel(DATA/"Jeju_H2_Demand_Data_Hamdeok_Station_Selling_Data.xlsx")
d["Sale Date"]=pd.to_datetime(d["Sale Date"])
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2))
x=np.arange(4);w=.35
axs[0,0].bar(x-w/2,rs.solar_cf,w,label="Solar",color=GOLD);axs[0,0].bar(x+w/2,rs.wind_cf,w,label="Wind",color=BLUE)
axs[0,0].set_xticks(x,rs.Year);axs[0,0].set_ylabel("Normalized mean output (%)");axs[0,0].legend(frameon=False,ncol=2);axs[0,0].set_title("(a) Annual normalized generation index",loc="left")
solar24=S[S.Year.isin([2020,2021,2022])][H].mean()/5500*100
wind24=W[W.Year.isin([2020,2021,2022])][H].mean()/1400*100
axs[0,1].plot(range(24),solar24,color=GOLD,lw=2,label="Solar");axs[0,1].plot(range(24),wind24,color=BLUE,lw=2,label="Wind")
axs[0,1].set_xlabel("Hour");axs[0,1].set_ylabel("Mean normalized output (%)");axs[0,1].set_title("(b) Mean hourly profiles, 2020--2022",loc="left");axs[0,1].legend(frameon=False)
axs[1,0].plot(d["Sale Date"],d["H2 Sold (kg)"],color=TEAL,lw=1);axs[1,0].axhline(d["H2 Sold (kg)"].iloc[1:].mean(),color=NAVY,ls="--",lw=1)
axs[1,0].set_ylabel("Daily sales (kg)");axs[1,0].set_title("(c) Hamdeok daily hydrogen sales",loc="left");axs[1,0].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axs[1,1].hist(d["H2 Sold (kg)"].iloc[1:],bins=18,color=TEAL,edgecolor="white");axs[1,1].axvline(d["H2 Sold (kg)"].iloc[1:].mean(),color=NAVY,ls="--",label="Mean 179.2 kg/day")
axs[1,1].set_xlabel("Daily sales (kg)");axs[1,1].set_ylabel("Days");axs[1,1].set_title("(d) Retained demand distribution",loc="left");axs[1,1].legend(frameon=False)
save(fig,"F02_data_characterization","DATA",["T01_re_statistics.csv","T02_demand_statistics.csv","KPX processed profiles","Hamdeok portal workbook"],n=255)

# 3. Optimization and validation protocol.
fig,ax=plt.subplots(figsize=(7.2,3.5));ax.axis("off")
items=[(.02,"Initialize","NSGA-III population"),(.215,"Simulate","hourly dispatch"),(.41,"Evaluate","LCOH, LPSP, curtailment"),(.605,"Select","q95 <= 0.1% on 8 draws"),(.80,"Validate","20 fresh + 4 holdout")]
for x0,t,su in items: box(ax,(x0,.38),.16,.28,t,su,TEAL if t in ["Initialize","Simulate"] else BLUE if t=="Evaluate" else GOLD if t=="Select" else RED)
for x0 in [.18,.375,.57,.765]:ax.annotate("",xy=(x0+.035,.52),xytext=(x0,.52),xycoords="axes fraction",arrowprops=dict(arrowstyle="->",lw=1.8,color=NAVY))
ax.annotate("new generation",xy=(.10,.36),xytext=(.49,.16),xycoords="axes fraction",ha="center",fontsize=8,color=GREY,arrowprops=dict(arrowstyle="->",color=GREY,connectionstyle="arc3,rad=-.25"))
ax.text(.5,.90,"Search, selection and independent evaluation are separate stages",ha="center",weight="bold",fontsize=12,color=NAVY,transform=ax.transAxes)
save(fig,"F03_optimization_protocol","SCHEMATIC",["optimizer_v2.py","final_select.py","RESULTS_LOCKED.json"],note="Selection uses 8 draws; validation uses unused draws")

# 4. Selected capacities.
x=LOCK["design_recommended"]; vals=[x["P_solar"],x["P_wind"],x["P_elz"]*(1-x["x_PEM"]),x["P_elz"]*x["x_PEM"]]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.35,1]})
b=axs[0].bar(["PV","Wind","AWE","PEM"],np.array(vals)/1000,color=[GOLD,BLUE,TEAL,NAVY]);axs[0].set_ylabel("Electrical rating (MW)");axs[0].set_ylim(0,2.1);axs[0].set_title("(a) Conversion-side capacities",loc="left")
for q,v in zip(b,np.array(vals)/1000):axs[0].text(q.get_x()+q.get_width()/2,v+.05,f"{v:.2f}",ha="center")
b=axs[1].bar(["BESS\n(MWh)","HSS\n(t H$_2$)"],[x["E_BESS"]/1000,x["M_HSS"]/1000],color=[BLUE,TEAL]);axs[1].set_ylim(0,2.1);axs[1].set_title("(b) Storage capacities",loc="left")
for q,v in zip(b,[x["E_BESS"]/1000,x["M_HSS"]/1000]):axs[1].text(q.get_x()+q.get_width()/2,v+.05,f"{v:.3f}" if v<.1 else f"{v:.2f}",ha="center")
save(fig,"F04_selected_design","SCREENING",["RESULTS_LOCKED.json:design_recommended"],n=8)

# 5. Twenty-draw validation distributions.
v=pd.read_csv(TAB/"T17_final_validation_20.csv")
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2)); specs=[("lcoh","LCOH (USD/kg)",NAVY),('lpsp','Service loss (%)',RED),('curtailment','Curtailment (%)',GOLD),('h2_served','Hydrogen served (t/year)',TEAL)]
for ax,(c,lab,col) in zip(axs.flat,specs):
    y=v[c].values.copy();
    if c in ['lpsp','curtailment']: y*=100
    if c=='h2_served': y/=1000
    ax.scatter(np.arange(1,21),y,color=col,s=24,zorder=3);ax.axhline(y.mean(),color=GREY,ls='--',lw=1,label=f"Mean {y.mean():.3f}")
    ax.set_xlabel("Fresh demand realization");ax.set_ylabel(lab);ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
axs[0,0].set_title("(a) Cost dispersion",loc='left');axs[0,1].set_title("(b) Mass-based service loss",loc='left');axs[1,0].set_title("(c) Renewable curtailment",loc='left');axs[1,1].set_title("(d) Annual hydrogen service",loc='left')
save(fig,"F05_validation_20draw","PRIMARY",["T17_final_validation_20.csv"],n=20)

# 6. Cost and capital structure.
c=LOCK['lcoh_decomposition'];comp=['Annualized capital','O&M','Replacement','Starts'];cv=[c['capex'],c['om'],c['replacement'],c['cyclic']]
cap=v.iloc[0][['capex_re','capex_awe','capex_pem','capex_bess','capex_hss']].values/1e6
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));b=axs[0].barh(comp[::-1],cv[::-1],color=[GOLD,TEAL,TEAL,NAVY]);axs[0].set_xlabel("USD/kg served");axs[0].set_title("(a) 20-draw mean LCOH composition",loc='left')
for q,z in zip(b,cv[::-1]):axs[0].text(z+.07,q.get_y()+q.get_height()/2,f"{z:.2f}",va='center')
labels=['Renewables','AWE','PEM','BESS','HSS'];axs[1].pie(cap,labels=labels,autopct=lambda p:f'{p:.1f}%' if p>3 else '',colors=[GOLD,TEAL,NAVY,BLUE,'#7DB8A8'],startangle=90,textprops={'fontsize':8});axs[1].set_title("(b) Initial capital allocation (5.97 MUSD)",loc='left')
save(fig,"F06_cost_structure","PRIMARY",["RESULTS_LOCKED.json:lcoh_decomposition","T17_final_validation_20.csv"],n=20)

# 7. Hydrogen Sankey-like balance.
h=LOCK['hydrogen_balance_kg_per_yr'];fig,ax=plt.subplots(figsize=(7.2,3.5));ax.axis('off')
nodes=[(.03,.43,.19,.25,'Produced',h['produced']),(.31,.65,.22,.22,'Direct service',h['served_directly']),(.31,.18,.22,.22,'Gross to HSS',h['routed_to_hss_gross']),(.62,.18,.18,.22,'From HSS',h['served_from_hss']),(.79,.58,.18,.25,'Total served',h['total_served'])]
for X,Y,Wd,Ht,t,z in nodes:
    box(ax,(X,Y),Wd,Ht,t,f"{z/1000:.2f} t/year",TEAL if 'HSS' in t else BLUE)
for a,b,l in [((.22,.57),(.31,.75),''),((.22,.50),(.31,.29),''),((.53,.29),(.62,.29),'1% transfer loss'),((.80,.29),(.86,.58),''),((.53,.75),(.79,.70),'')]:
    ax.annotate(l,xy=b,xytext=a,xycoords='axes fraction',fontsize=8,color=GREY,arrowprops=dict(arrowstyle='->',lw=2,color=NAVY))
ax.text(.5,.94,"Annual hydrogen balance: 20-draw mean",ha='center',weight='bold',fontsize=12,color=NAVY,transform=ax.transAxes)
ax.text(.5,.04,"Discarded hydrogen approx. 0; cyclic net HSS inventory change = 0 in reported trajectories",ha='center',fontsize=8,color=GREY,transform=ax.transAxes)
save(fig,"F07_hydrogen_balance","PRIMARY",["T19_hydrogen_balance.csv","RESULTS_LOCKED.json:hydrogen_balance_kg_per_yr"],n=20)

# 8. Operating metrics by draw.
o=pd.read_csv(TAB/'T29_operating_20draw.csv');fig,axs=plt.subplots(1,3,figsize=(7.2,3.1))
axs[0].boxplot([o.awe_hours,o.pem_hours],tick_labels=['AWE','PEM'],patch_artist=True,boxprops={'facecolor':'#D5E8D4'});axs[0].set_ylabel('Operating hours/year');axs[0].set_title('(a) Utilization',loc='left')
axs[1].boxplot([o.awe_life,o.pem_life],tick_labels=['AWE','PEM'],patch_artist=True,boxprops={'facecolor':'#DAE8FC'});axs[1].set_ylabel('Modeled stack life (years)');axs[1].set_title('(b) Replacement interval',loc='left')
axs[2].boxplot([o.bess_cycles],tick_labels=['BESS'],patch_artist=True,boxprops={'facecolor':'#FFE6CC'});axs[2].set_ylabel('Equivalent cycles/year');axs[2].set_title('(c) Battery cycling',loc='left')
save(fig,"F08_operating_metrics","PRIMARY",["T29_operating_20draw.csv"],n=20)

# 9. HSS behavior across draws.
st=pd.read_csv(TAB/'T26_storage_behaviour_by_draw.csv');fig,axs=plt.subplots(1,3,figsize=(7.2,3.1))
for ax,c,lab,col in [(axs[0],'median_pct','Median inventory (%)',TEAL),(axs[1],'pct_above90','Hours above 90% (%)',BLUE),(axs[2],'pct_below25','Hours below 25% (%)',RED)]:
    ax.bar(st.seed.astype(str),st[c],color=col);ax.axhline(st[c].mean(),color=NAVY,ls='--',lw=1);ax.set_xlabel('Demand seed');ax.set_ylabel(lab);ax.tick_params(axis='x',rotation=90,labelsize=6)
axs[0].set_title('(a) Central occupancy',loc='left');axs[1].set_title('(b) High inventory',loc='left');axs[2].set_title('(c) Low inventory',loc='left')
save(fig,"F09_storage_behavior","PRIMARY",["T26_storage_behaviour_by_draw.csv"],n=20)

# 10. Screening vs validation tails, including low-loss controls.
tv=pd.read_csv(TAB/'T28_tail_estimator_lowloss.csv');s=LOCK['screening_vs_validation'];fig,axs=plt.subplots(1,2,figsize=(7.2,3.2))
axs[0].bar(['Screening\n8 draws','Validation\n20 draws'],[100*s['q95_screen_8draws'],100*s['q95_validation_20draws']],color=[BLUE,RED]);axs[0].axhline(.1,color=NAVY,ls='--',label='0.1% selection cut');axs[0].set_ylabel('Empirical service-loss q95 (%)');axs[0].legend(frameon=False);axs[0].set_title('(a) Selected design',loc='left')
axs[1].scatter(100*tv.q95_screen,100*tv.q95_valid,color=np.where(tv.zero_screen,RED,TEAL),s=30,alpha=.8);mx=100*max(tv.q95_screen.max(),tv.q95_valid.max());axs[1].plot([0,mx],[0,mx],ls='--',color=GREY);axs[1].set_xlabel('Screening q95 (%)');axs[1].set_ylabel('Validation q95 (%)');axs[1].set_title('(b) 27 low-loss designs',loc='left')
save(fig,"F10_tail_validation","SCREENING+PRIMARY",["T28_tail_estimator_lowloss.csv","RESULTS_LOCKED.json:screening_vs_validation"],n=27)

# 11. Subsample estimator stability from stored five-number summary.
q=pd.read_csv(TAB/'T27_q95_subsample_stability.csv',index_col=0);stats={'label':'q95 from 8 draws','whislo':100*q.loc['min','q95_8draw'],'q1':100*q.loc['25%','q95_8draw'],'med':100*q.loc['50%','q95_8draw'],'q3':100*q.loc['75%','q95_8draw'],'whishi':100*q.loc['max','q95_8draw'],'mean':100*q.loc['mean','q95_8draw'],'fliers':[]}
fig,ax=plt.subplots(figsize=(5.4,3.2));ax.bxp([stats],showmeans=True,meanline=True,patch_artist=True,boxprops={'facecolor':'#D5E8D4'},medianprops={'color':NAVY});ax.axhline(100*s['q95_validation_20draws'],color=RED,ls='--',label='20-draw reference q95');ax.set_ylabel('Service-loss q95 (%)');ax.set_title('Distribution of 5,000 eight-draw subsample estimates',loc='left');ax.legend(frameon=False)
save(fig,"F11_q95_subsample_stability","PRIMARY-SUBSAMPLE",["T27_q95_subsample_stability.csv","T17_final_validation_20.csv"],n=5000,placement='supplementary')

# 12. 2023 holdout draw comparison.
ho=pd.read_csv(TAB/'T18_final_holdout_2023.csv');fig,axs=plt.subplots(2,2,figsize=(7.2,5.0));draw=np.arange(1,5)
for ax,c,scale,lab,col in [(axs[0,0],'lcoh',1,'LCOH (USD/kg)',NAVY),(axs[0,1],'curtailment',100,'Curtailment (%)',GOLD),(axs[1,0],'h2_served',1/1000,'H2 served (t/year)',TEAL),(axs[1,1],'lpsp',100,'Service loss (%)',RED)]:
    y=ho[c]*scale;ax.bar(draw,y,color=col);ax.set_xticks(draw);ax.set_xlabel('2023 demand draw');ax.set_ylabel(lab)
axs[0,0].set_title('(a) Cost',loc='left');axs[0,1].set_title('(b) Curtailment',loc='left');axs[1,0].set_title('(c) Annual service',loc='left');axs[1,1].set_title('(d) Unserved mass',loc='left')
save(fig,"F12_holdout_2023","HOLDOUT",["T18_final_holdout_2023.csv"],n=4)

# 13. Refined ablation results.
a=LOCK['ablation'];names=['F0\nfixed','F1\nrenewable','F2\ndownstream','F3\njoint'];cost=[a['F0'],a['F1_grid'],a['F2_polished'],a['F3_polished']]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));b=axs[0].bar(names,cost,color=[GREY,BLUE,TEAL,GOLD]);axs[0].set_ylabel('LCOH (USD/kg)');axs[0].set_title('(a) Optimized case cost',loc='left')
for z,qv in zip(b,cost):axs[0].text(z.get_x()+z.get_width()/2,qv+.3,f'{qv:.2f}',ha='center')
effects=[a['benefit_renewable_only'],a['benefit_downstream_only'],a['interaction']];labs=['Renewable\nonly','Downstream\nonly','Residual\nnonadditivity'];b=axs[1].bar(labs,effects,color=[BLUE,TEAL,RED]);axs[1].set_ylabel('Reduction from F0 (USD/kg)');axs[1].set_title('(b) Joint-sizing decomposition',loc='left')
for z,qv in zip(b,effects):axs[1].text(z.get_x()+z.get_width()/2,qv+.08,f'{qv:.2f}',ha='center')
save(fig,"F13_ablation","ABLATION",["RESULTS_LOCKED.json:ablation"],note='F3 repeated over two seeds; F1 and F2 once under final treatment')

# 14. Six-draw parameter sensitivity.
se=pd.read_csv(TAB/'T20_sensitivity_final.csv');base=se.loc[se.Case.isin(['Discount rate 6%','Degradation factor 1.0']),'LCOH'].mean();se['delta']=se.LCOH-base
fig,ax=plt.subplots(figsize=(6.5,3.3));cols=np.where(se.delta>=0,RED,TEAL);ax.barh(se.Case,se.delta,color=cols);ax.axvline(0,color=NAVY,lw=1);ax.set_xlabel('Change in LCOH relative to 11.6765 USD/kg (USD/kg)');ax.set_title('Local parameter sensitivity at the selected design',loc='left')
save(fig,"F14_parameter_sensitivity","SIX-DRAW SENSITIVITY",["T20_sensitivity_final.csv"],n=6,placement='supplementary')

# 15. Illustrative low-inventory dispatch window.
dw=pd.read_csv(TAB/'T23_dispatch_window.csv',parse_dates=['datetime']);fig,axs=plt.subplots(3,1,figsize=(7.2,6.0),sharex=True)
axs[0].plot(dw.datetime,dw.p_re,color=GOLD,label='Renewable');axs[0].plot(dw.datetime,dw.p_awe,color=TEAL,label='AWE');axs[0].plot(dw.datetime,dw.p_pem,color=NAVY,label='PEM');axs[0].set_ylabel('Power (kW)');axs[0].legend(frameon=False,ncol=3);axs[0].set_title('(a) Renewable availability and electrolyzer allocation',loc='left')
axs[1].plot(dw.datetime,dw.h2_prod,color=BLUE,label='Produced');axs[1].plot(dw.datetime,dw.h2_dem,color=RED,ls='--',label='Demand');axs[1].plot(dw.datetime,dw.h2_served,color=TEAL,label='Served');axs[1].set_ylabel('Hydrogen (kg/h)');axs[1].legend(frameon=False,ncol=3);axs[1].set_title('(b) Hydrogen service',loc='left')
axs[2].plot(dw.datetime,100*dw.soc,color=BLUE,label='BESS SOC');axs[2].set_ylabel('BESS SOC (%)');ax2=axs[2].twinx();ax2.plot(dw.datetime,dw.hss,color=TEAL,label='HSS');ax2.set_ylabel('HSS inventory (kg)');axs[2].set_title('(c) Storage states',loc='left');axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
save(fig,"F15_dispatch_window","ILLUSTRATIVE",["T23_dispatch_window.csv"],n=1,placement='supplementary',note='One identified demand realization; not a system property')

(FIG/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))

# Separate graphical abstract for journal upload.
fig,ax=plt.subplots(figsize=(11.1,4.43));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.text(.035,.91,'Island hydrogen refueling: co-size, dispatch, then validate',fontsize=19,weight='bold',color=NAVY)
box(ax,(.03,.49),.20,.25,'Public case data','Jeju hourly renewables\nHamdeok daily sales',TEAL)
box(ax,(.29,.49),.20,.25,'Joint capacity search','PV + wind + AWE/PEM\nBESS + HSS',BLUE)
box(ax,(.55,.49),.20,.25,'Selected design','1.85 MW PV; 0.67 MW wind\n1.03 MW EL; 1.86 t HSS',GOLD)
box(ax,(.81,.49),.16,.25,'Fresh validation','20 unused\ndemand draws',RED)
for x0 in [.23,.49,.75]:ax.annotate('',xy=(x0+.06,.615),xytext=(x0+.01,.615),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,color=NAVY))
ax.text(.035,.24,'11.7184 USD/kg',fontsize=16,weight='bold',color=NAVY)
ax.text(.035,.14,'20-draw mean LCOH',fontsize=10,color=GREY)
ax.text(.30,.24,'35.38%',fontsize=16,weight='bold',color=TEAL)
ax.text(.30,.14,'served H2 routed through HSS',fontsize=10,color=GREY)
ax.text(.56,.24,'0.0467%  ->  0.4207%',fontsize=16,weight='bold',color=RED)
ax.text(.56,.14,'service-loss q95: screening vs. fresh validation',fontsize=10,color=GREY)
fig.tight_layout();fig.savefig(ROOT/'graphical_abstract.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'graphical_abstract.pdf',bbox_inches='tight');plt.close(fig)
print(f"Generated {len(manifest)} figures")
