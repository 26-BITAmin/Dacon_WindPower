# ═══════════════════════════════════════════════════════
# exp-038 피처: JMA GSM/MSM 다격자 → 격자 편차(SPD 형식, spd와 동형)
#
# jma_grid_prevruns.csv(모델별 9격자, previous_day2)에서 변수별로:
#   공간평균 + 격자별 (값 − 공간평균)  → DL이 격자 간 고차 구조를 직접 추출.
#   레벨(공간평균)은 exp-020 단일점(jgsm_ws 등)과 겹치나, 격자 편차는 신규 공간 정보.
# 변수: ws, wd(sin/cos), t2m, sp. 풍향은 sin/cos로 분해 후 편차.
# 리크 없음: 격자 좌표 시간 불변, previous_day2(전날 13:00 이전 발행).
# ═══════════════════════════════════════════════════════
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp005_qopt import DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = DATA_DIR / "external" / "jma_grid_prevruns.csv"
JGRID_PATH = DATA_DIR / "external" / "exp038_jma_grid_spd.parquet"


def _pivot(sub, col, master_idx):
    """시각×격자 행렬. 격자 결측은 그 시각 공간평균으로, 시각 전체 결측은 NaN(하류 imputer)."""
    M = sub.pivot_table(index="forecast_kst_dtm", columns="grid_idx", values=col, aggfunc="mean")
    M = M.reindex(index=master_idx).sort_index(axis=1)
    row_mean = M.mean(axis=1)
    M = M.apply(lambda c: c.fillna(row_mean))
    return M


def build_jgrid(force=False):
    """반환: (df[forecast_kst_dtm + 편차컬럼], 컬럼목록). GSM/MSM 각 블록 컬럼 접두 jg_/jm_."""
    if not force and JGRID_PATH.exists():
        out = pd.read_parquet(JGRID_PATH)
        return out, [c for c in out.columns if c != "forecast_kst_dtm"]

    t0 = time.time()
    raw = pd.read_csv(SRC, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    master_idx = pd.Index(np.sort(raw["forecast_kst_dtm"].unique()), name="forecast_kst_dtm")
    out = pd.DataFrame(index=master_idx)

    for model, pref in [("jma_gsm", "jg"), ("jma_msm", "jm")]:
        sub = raw[raw["model"] == model].copy()
        wd = np.deg2rad(sub["wind_direction_10m"])
        sub["wdsin"], sub["wdcos"] = np.sin(wd), np.cos(wd)
        var_map = {"ws": "wind_speed_10m", "wdsin": "wdsin", "wdcos": "wdcos",
                   "t2": "temperature_2m", "sp": "surface_pressure"}
        gids = np.sort(sub["grid_idx"].unique())
        for name, col in var_map.items():
            M = _pivot(sub, col, master_idx)
            V = M.to_numpy()
            mean = np.nanmean(V, axis=1)
            out[f"{pref}_{name}_mean"] = mean
            for j, g in enumerate(gids):
                out[f"{pref}_{name}_g{int(g)}"] = V[:, j] - mean
    out = out.reset_index()
    out.to_parquet(JGRID_PATH, index=False)
    cols = [c for c in out.columns if c != "forecast_kst_dtm"]
    print(f"  JMA 격자편차 {len(cols)}개 피처, {len(out)}행 ({time.time()-t0:.0f}s)", flush=True)
    return out, cols


if __name__ == "__main__":
    out, cols = build_jgrid(force="--force" in sys.argv)
    print(f"JMA 다격자 편차: {len(cols)}개 피처, {len(out)}행")
    print(f"기간: {out['forecast_kst_dtm'].min()} ~ {out['forecast_kst_dtm'].max()}")
    gc = [c for c in cols if c.startswith("jg_")]
    mc = [c for c in cols if c.startswith("jm_")]
    print(f"GSM 블록 {len(gc)}개 / MSM 블록 {len(mc)}개")
    print(f"결측: {out[cols].isna().sum().sum()} (GSM은 2022 전기간, MSM은 2022 결측 → imputer 처리)")
    print("컬럼 예:", cols[:5], "...", cols[-3:])
