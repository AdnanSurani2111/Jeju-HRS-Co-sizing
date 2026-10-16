"""
config.py - Hybrid AWE+PEM HRS sizing, corrected model (v2)

Changes from v1, each traceable to an identified defect:
  - Renewable capacities are DECISION VARIABLES, not fixed at 5.5/1.4 MW.
  - Hydrogen demand is the measured Hamdeok dispensing record, not a flat 200 kg/d.
  - Capacity factors are computed against year-specific fleet capacity.
  - PEM nominal degradation rate is now reachable (was dead code in v1).
  - Electrolyzer setpoints are capped by demand + storage headroom, so surplus
    hydrogen can no longer be manufactured and silently discarded.

All CAPEX are installed-system costs in 2024 USD.
"""
import os

# --- decision variable bounds ------------------------------------------------
# x = [P_solar, P_wind, P_elz, x_PEM, E_BESS, M_HSS]
P_SOLAR_MIN_KW, P_SOLAR_MAX_KW = 500.0, 8000.0
P_WIND_MIN_KW,  P_WIND_MAX_KW  = 200.0, 3000.0
P_ELZ_MIN_KW,   P_ELZ_MAX_KW   = 100.0, 3000.0
X_PEM_MIN,      X_PEM_MAX      = 0.0,   1.0
E_BESS_MIN_KWH, E_BESS_MAX_KWH = 50.0,  3000.0
M_HSS_MIN_KG,   M_HSS_MAX_KG   = 50.0,  2000.0

VAR_NAMES  = ['P_solar', 'P_wind', 'P_elz', 'x_PEM', 'E_BESS', 'M_HSS']
VAR_LOWER  = [P_SOLAR_MIN_KW, P_WIND_MIN_KW, P_ELZ_MIN_KW, X_PEM_MIN, E_BESS_MIN_KWH, M_HSS_MIN_KG]
VAR_UPPER  = [P_SOLAR_MAX_KW, P_WIND_MAX_KW, P_ELZ_MAX_KW, X_PEM_MAX, E_BESS_MAX_KWH, M_HSS_MAX_KG]

# --- reference per-unit plant used to build the normalised profiles ----------
# The CSVs in data/ are already scaled to these ratings; the simulator rescales
# to whatever capacity the optimiser proposes.
REF_SOLAR_KW = 5500.0
REF_WIND_KW  = 1400.0

# --- AWE ---------------------------------------------------------------------
AWE_CAPEX_PER_KW   = 700.0
AWE_SEC_KWH_PER_KG = 50.0
AWE_MIN_LOAD_FRAC  = 0.20
AWE_MAX_LOAD_FRAC  = 1.00
AWE_DEG_NOM_UV_H   = 0.5
AWE_DEG_HARSH_UV_H = 2.0
AWE_NOM_LOAD_FLOOR = 0.80    # at or above this load AWE ages at the nominal rate
AWE_V_REF_UV       = 160_000.0
AWE_CYCLE_COST     = 50.0
AWE_OM_FRAC        = 0.03

# --- PEM ---------------------------------------------------------------------
PEM_CAPEX_PER_KW   = 1500.0
PEM_SEC_KWH_PER_KG = 55.0
PEM_MIN_LOAD_FRAC  = 0.05
PEM_MAX_LOAD_FRAC  = 1.20
PEM_OVERLOAD_FRAC  = 1.00    # above rated counts as overload for ageing
PEM_DEG_NOM_UV_H   = 2.0
PEM_DEG_HARSH_UV_H = 4.0
PEM_V_REF_UV       = 560_000.0
PEM_CYCLE_COST     = 30.0
PEM_OM_FRAC        = 0.03

AWE_REPLACE_THRESHOLD_UV = 0.25 * AWE_V_REF_UV    # 40,000 uV
PEM_REPLACE_THRESHOLD_UV = 0.25 * PEM_V_REF_UV    # 140,000 uV

# --- BESS --------------------------------------------------------------------
BESS_CAPEX_PER_KWH = 300.0
BESS_OM_FRAC = 0.02
BESS_LIFE_YR = 10
BESS_ETA_CH = BESS_ETA_DIS = 0.95
BESS_SOC_MIN, BESS_SOC_MAX, BESS_SOC_INIT = 0.10, 0.90, 0.50
BESS_C_RATE = 0.5

# --- HSS ---------------------------------------------------------------------
HSS_CAPEX_PER_KG = 800.0
HSS_OM_FRAC = 0.015
HSS_ETA = 0.99
HSS_MIN_FRAC, HSS_MAX_FRAC, HSS_INIT_FRAC = 0.05, 1.00, 0.50

# --- renewable plant ---------------------------------------------------------
SOLAR_CAPEX_PER_KW = 1200.0
WIND_CAPEX_PER_KW  = 1800.0
RE_OM_FRAC = 0.015

# --- economics ---------------------------------------------------------------
DISCOUNT_RATE = 0.06
PROJECT_LIFE_YR = 20

# --- demand ------------------------------------------------------------------
DEMAND_XLSX = 'Jeju_H2_Demand_Data_Hamdeok_Station_Selling_Data.xlsx'
N_DEMAND_REALISATIONS = 2      # bootstrap draws averaged inside each evaluation
DEMAND_BLOCK_DAYS = 7          # block length preserves weekly + persistence structure

# --- optimisation ------------------------------------------------------------
N_OBJ = 3
NSGA3_POP = 182
NSGA3_GEN = 100          # production budget; run_continuous.py accepts an
                         # explicit target as argv[3] and the reported run used 100
NSGA3_SEED = 42
DAS_DENNIS_P = 12
HV_REF = [40.0, 0.30, 0.80]

TRAIN_YEARS = [2020, 2021, 2022]
TEST_YEAR = 2023
HOUR_COLS = [f'H{h:02d}' for h in range(24)]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
TABLE_DIR = os.path.join(BASE_DIR, 'tables')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')
for _d in (FIG_DIR, TABLE_DIR, CKPT_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)
