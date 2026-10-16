"""Convert genuine KPX Jeju files into the per-unit hourly profiles the model needs.

Fleet generation is normalised by the installed capacity OF THAT PERIOD, then
scaled to the dedicated plant. Normalising all years by one fixed capacity
embeds fleet expansion into what the model reads as weather.
"""
import pandas as pd, numpy as np, os
import os
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'raw')    # original Public Data Portal files (CP949)
OUT = os.path.join(HERE, 'data')           # processed per-unit profiles
H = [f'H{i:02d}' for i in range(24)]

def load(p):
    df = pd.read_csv(p, encoding='cp949')
    df.columns = ['date'] + H
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    return df

def per_unit(df, tag):
    v = df[H].values.copy()
    neg = (v < 0).sum()
    if neg: v = np.clip(v, 0, None)
    if tag == 'solar':                      # strip night-time sensor offset
        night = np.median(v[:, [0,1,2,3,22,23]])
        v = np.clip(v - night, 0, None)
    # capacity proxy per year: p99.9 of hourly fleet output that year
    pu = np.zeros_like(v)
    caps = {}
    for y in sorted(df.Year.unique()):
        m = (df.Year == y).values
        cap = np.percentile(v[m], 99.9)
        caps[y] = cap
        pu[m] = v[m] / cap
    return np.clip(pu, 0, 1.2), caps, neg

def build(src, tag, rated_kw, out):
    df = load(src)
    pu, caps, neg = per_unit(df, tag)
    df = df[df.Year.isin([2020,2021,2022,2023])].reset_index(drop=True)
    pu = pu[df.index] if len(pu) != len(df) else pu
    d = load(src); mask = d.Year.isin([2020,2021,2022,2023]).values
    pu2, _, _ = per_unit(d, tag); pu2 = pu2[mask]
    kw = pu2 * rated_kw
    o = pd.DataFrame(kw, columns=H)
    o.insert(0, 'Year', d[mask]['Year'].values)
    o.insert(1, 'Month', d[mask]['Month'].values)
    o.insert(2, 'Day', d[mask]['date'].dt.day.values)
    o['Daily_Total_kWh'] = kw.sum(axis=1)
    o.to_csv(out, index=False)
    print(f"{tag:6s} rows={len(o)}  negatives_clipped={neg}  capacity proxy by year:"
          f" {{{', '.join(f'{y}: {c:.0f} MW' for y,c in caps.items() if y>2019)}}}")
    for y in (2020,2021,2022,2023):
        m = o.Year == y
        print(f"        {y} CF = {100*o[m][H].values.mean()/rated_kw:5.2f}%")
    return o

os.makedirs(OUT, exist_ok=True)
S = build(os.path.join(RAW, '제주지역_태양광_한시간단위실적_2019-12-01_2023-12-31_.csv'),
          'solar', 5500.0, os.path.join(OUT, 'Jeju_Solar_Generation_2020_2023_KPX.csv'))
W = build(os.path.join(RAW, '제주지역_풍력_한시간단위실적_2019-12-01_2023-12-31_.csv'),
          'wind', 1400.0, os.path.join(OUT, 'Jeju_Wind_Generation_2020_2023_KPX.csv'))

s = S[S.Year.isin([2020,2021,2022])][H].values.flatten()
w = W[W.Year.isin([2020,2021,2022])][H].values.flatten()
print(f"\nsolar-wind hourly correlation (2020-2022): {np.corrcoef(s,w)[0,1]:+.4f}")
print(f"longest run of combined RE below 10% of rated: "
      f"{max((len(list(g)) for k,g in __import__('itertools').groupby((s+w) < 0.10*6900) if k), default=0)} hours")
