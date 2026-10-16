"""
data_v2.py - real data only.

There is deliberately no synthetic-profile generator in this module and no
write-back of generated files into the raw-data filenames. If the inputs are
missing the pipeline fails loudly instead of fabricating a substitute.
"""
import os
import numpy as np
import pandas as pd
import config as C


def load_re():
    """Per-unit hourly solar and wind, chronological, split train/test."""
    sp = os.path.join(C.DATA_DIR, 'Jeju_Solar_Generation_2020_2023_KPX.csv')
    wp = os.path.join(C.DATA_DIR, 'Jeju_Wind_Generation_2020_2023_KPX.csv')
    for p in (sp, wp):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required KPX input missing: {p}\n"
                "Run convert_kpx.py on the raw data.go.kr download (dataset 15127502). "
                "This pipeline does not generate substitute data.")
    S = pd.read_csv(sp)
    W = pd.read_csv(wp)
    out = {}
    for tag, yrs in (('train', C.TRAIN_YEARS), ('test', [C.TEST_YEAR])):
        ms = S.Year.isin(yrs).values
        mw = W.Year.isin(yrs).values
        s = S.loc[ms, C.HOUR_COLS].values.flatten() / C.REF_SOLAR_KW
        w = W.loc[mw, C.HOUR_COLS].values.flatten() / C.REF_WIND_KW
        n = min(len(s), len(w))
        out[tag] = (s[:n].astype(float), w[:n].astype(float))
    return out


def load_demand_records():
    """Measured daily dispensing, Hamdeok green hydrogen station."""
    p = os.path.join(C.DATA_DIR, C.DEMAND_XLSX)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required demand input missing: {p}")
    d = pd.read_excel(p, header=0, skiprows=1,
                      names=['sn', 'station', 'loc', 'date', 'kg', 'comm', 'pass'])
    d = d.dropna(subset=['kg']).copy()
    d['date'] = pd.to_datetime(d['date'])
    return d.sort_values('date').reset_index(drop=True)


def demand_series(n_hours, seed=0, scale=1.0, flat=None, basis='full'):
    """
    Hourly demand, kg/h.

    Daily totals are block-bootstrapped from the measured record so that the
    weekly pattern and the day-to-day persistence (lag-1 rho = +0.30) survive.
    Within a day the requirement is spread uniformly: the source data is daily,
    so no intra-day arrival structure is claimed.
    """
    n_days = int(np.ceil(n_hours / 24))
    if flat is not None:
        daily = np.full(n_days, float(flat))
    else:
        d = load_demand_records()
        if basis == 'mature':
            # last three months of the record: the station after ramp-up
            cut = d.date.max() - pd.Timedelta(days=92)
            d = d[d.date >= cut]
        k = d.kg.values.astype(float)
        rng = np.random.default_rng(seed)
        B = C.DEMAND_BLOCK_DAYS
        starts = rng.integers(0, len(k) - B, size=n_days // B + 1)
        daily = np.concatenate([k[i:i + B] for i in starts])[:n_days]
    daily = daily * float(scale)
    return np.repeat(daily / 24.0, 24)[:n_hours], daily


def re_statistics():
    """Normalized generation indices based on the configured reference ratings."""
    S = pd.read_csv(os.path.join(C.DATA_DIR, 'Jeju_Solar_Generation_2020_2023_KPX.csv'))
    W = pd.read_csv(os.path.join(C.DATA_DIR, 'Jeju_Wind_Generation_2020_2023_KPX.csv'))
    rows = []
    for y in (2020, 2021, 2022, 2023):
        s = S.loc[S.Year == y, C.HOUR_COLS].values
        w = W.loc[W.Year == y, C.HOUR_COLS].values
        rows.append(dict(Year=y,
                         solar_cf=100 * s.mean() / C.REF_SOLAR_KW,
                         wind_cf=100 * w.mean() / C.REF_WIND_KW,
                         role='Training' if y in C.TRAIN_YEARS else 'Held-out test'))
    return pd.DataFrame(rows)
