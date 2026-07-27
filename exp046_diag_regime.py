# ═══════════════════════════════════════════════════════
# exp-046 사전진단: 레짐별(풍속·시간대·pA·CF구간) GBM vs DL 상대 우위
#
# 목적: 전역 w=0.70 대신 조건부 w(레짐별)를 학습할 여지가 있는지 1분 진단.
#   레짐별 최적 w가 0.70 주변에서 평탄하면 조건부 블렌딩 축은 사망 → 본 실험 생략.
#   (exp-044 D블록에서 그룹별 상수 w는 이미 평탄 확인 — 여기선 그룹 외 축 확인)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp046_diag_regime.py
# ═══════════════════════════════════════════════════════
import pickle
import sys

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import load_gbm_cache
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]

cache, _ = load_gbm_cache()
cache = cache.rename(columns={"gbm_cf": "cf_g132"})
ext = pd.read_parquet(EXT_GBM_CACHE)
m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
cache["cf_gspd"] = m["cf_gspd"].to_numpy()
dl = pd.read_parquet(E39_CACHE)
m = cache[KEY].merge(dl[KEY + ["dl_q60"]], on=KEY, how="left")
cache["dl_q60"] = m["dl_q60"].to_numpy()

caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
acf = cache["target"].to_numpy() / caps
ev = (cache["target"].to_numpy() >= caps * 0.10) & np.isfinite(acf)

df = pd.DataFrame({
    "ws": cache["ws"].to_numpy(),
    "pA": cache["pA"].to_numpy(),
    "hour": pd.to_datetime(cache["forecast_kst_dtm"]).dt.hour,
    "acf": acf,
    "g": cache["cf_gspd"].to_numpy(),
    "d": cache["dl_q60"].to_numpy(),
})[ev].copy()

# 레짐별 최적 w: w* = argmin_w E|w·g + (1−w)·d − y| 를 그리드로
WGRID = np.arange(0.0, 1.01, 0.05)


def opt_w(sub):
    errs = [np.abs(w * sub["g"] + (1 - w) * sub["d"] - sub["acf"]).mean() for w in WGRID]
    i = int(np.argmin(errs))
    return WGRID[i], errs[i], np.abs(0.70 * sub["g"] + 0.30 * sub["d"] - sub["acf"]).mean(), len(sub)


def report(name, key_series, bins=None, labels=None):
    df2 = df.copy()
    df2["bin"] = pd.cut(key_series, bins, labels=labels) if bins is not None else key_series
    print(f"\n--- {name} ---")
    print(f"{'bin':<12}{'n':>7}{'w*':>6}{'MAE(w*)':>9}{'MAE(.70)':>9}{'개선':>8}")
    for b, sub in df2.groupby("bin", observed=True):
        if len(sub) < 100:
            continue
        w, e_opt, e_70, n = opt_w(sub)
        print(f"{str(b):<12}{n:>7}{w:>6.2f}{e_opt:>9.4f}{e_70:>9.4f}{e_70-e_opt:>8.4f}")


w_all, e_all, e70_all, n_all = opt_w(df)
print(f"전역 (평가시간 n={n_all}): w*={w_all:.2f}, MAE(w*)={e_all:.4f}, MAE(0.70)={e70_all:.4f}")

report("풍속 구간", df["ws"], bins=[0, 4, 6, 8, 10, 13, 99], labels=["<4", "4-6", "6-8", "8-10", "10-13", "13+"])
report("시간대", pd.cut(df["hour"], [-1, 5, 11, 17, 23], labels=["0-5", "6-11", "12-17", "18-23"]))
report("pA (정격확률)", df["pA"], bins=[-0.01, 0.05, 0.27, 0.6, 1.01], labels=["<.05", ".05-.27", ".27-.6", ".6+"])
report("실측 CF 구간", df["acf"], bins=[0.1, 0.3, 0.5, 0.7, 0.9, 1.01], labels=[".1-.3", ".3-.5", ".5-.7", ".7-.9", ".9+"])
