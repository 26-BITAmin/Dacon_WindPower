# ═══════════════════════════════════════════════════════
# exp-035: 원시 다격자(multi-grid) 공간 입력 빌더
#
# 배경: 챔피언 GBM 피처(132개)는 LDAPS 16격자 중 **1개(그룹별 최근접)**, GFS 9격자 중
#   **1개(세 그룹 공통 grid 5)**만 사용한다(exp005_qopt.build_datasets). exp-012 S블록이
#   ws 집계 5개를 더했을 뿐, 나머지 격자의 공간 구조(경도/위도 방향 경사 = 이류·전선 정보)는
#   전부 버려져 있다. exp-019에서 GBM에 다격자 집계를 넣었을 때는 이득이 노이즈 수준이었으나,
#   상관 높은 다수 입력은 GBM(축정렬 분할)보다 NN이 다루기 쉬운 형태 → DL 파트너 입력으로 재시도.
#
# 두 표현을 만든다 (둘 다 시각 단위, 그룹 무관 — 격자 좌표가 시간 불변이라 리크 없음):
#   SPG (경사 축약, 변수당 4개): 공간평균 / ∂v/∂lat / ∂v/∂lon / 잔차 std
#        → 최소제곱 평면 적합. 격자 좌표가 상수라 유사역행렬 1회 계산 후 행렬곱.
#   SPD (전체 편차, 변수당 격자수+1개): 공간평균 + 격자별 (값 − 공간평균)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp035_grid_features.py   (parquet 캐시 생성)
# ═══════════════════════════════════════════════════════
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp005_qopt import TRAIN_DIR, TEST_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_DIR = Path(__file__).parent / "open (1)" / "external"
SPG_PATH = CACHE_DIR / "exp035_spatial_spg.parquet"
SPD_PATH = CACHE_DIR / "exp035_spatial_spd.parquet"

LD_VARS = {
    "ws10":  lambda d: np.hypot(d["heightAboveGround_10_10u"], d["heightAboveGround_10_10v"]),
    "u10":   lambda d: d["heightAboveGround_10_10u"],
    "v10":   lambda d: d["heightAboveGround_10_10v"],
    "ws50x": lambda d: np.hypot(d["heightAboveGround_50_50MUmax"], d["heightAboveGround_50_50MVmax"]),
    "wsbl":  lambda d: np.hypot(d["heightAboveGround_5_XBLWS"], d["heightAboveGround_5_YBLWS"]),
    "t2":    lambda d: d["heightAboveGround_2_t"],
    "sp":    lambda d: d["surface_0_sp"],
    "blh":   lambda d: d["etc_0_blh"],
}
LD_SRC = ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
          "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax",
          "heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS",
          "heightAboveGround_2_t", "surface_0_sp", "etc_0_blh"]

GF_VARS = {
    "ws10":  lambda d: np.hypot(d["heightAboveGround_10_10u"], d["heightAboveGround_10_10v"]),
    "ws100": lambda d: np.hypot(d["heightAboveGround_100_100u"], d["heightAboveGround_100_100v"]),
    "u100":  lambda d: d["heightAboveGround_100_100u"],
    "v100":  lambda d: d["heightAboveGround_100_100v"],
    "gust":  lambda d: d["surface_0_gust"],
    "ws850": lambda d: np.hypot(d["isobaricInhPa_850_u"], d["isobaricInhPa_850_v"]),
    "t850":  lambda d: d["isobaricInhPa_850_t"],
    "prmsl": lambda d: d["meanSea_0_prmsl"],
}
GF_SRC = ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
          "heightAboveGround_100_100u", "heightAboveGround_100_100v",
          "surface_0_gust", "isobaricInhPa_850_u", "isobaricInhPa_850_v",
          "isobaricInhPa_850_t", "meanSea_0_prmsl"]

# exp-036후속(spd2): GFS 확장 변수 — 허브고도 80m 풍·상층풍(700/500hPa)·종관 지위고도·경계층 풍.
# 전부 격자 편차로만 사용(레벨은 GBM 132가 이미 가짐, 편차=공간 구조가 직교 신호).
GF_VARS_EXT = {
    "u80":   lambda d: d["heightAboveGround_80_u"],
    "v80":   lambda d: d["heightAboveGround_80_v"],
    "u700":  lambda d: d["isobaricInhPa_700_u"],
    "v700":  lambda d: d["isobaricInhPa_700_v"],
    "u500":  lambda d: d["isobaricInhPa_500_u"],
    "v500":  lambda d: d["isobaricInhPa_500_v"],
    "gh500": lambda d: d["isobaricInhPa_500_gh"],
    "pblu":  lambda d: d["planetaryBoundaryLayer_0_u"],
    "pblv":  lambda d: d["planetaryBoundaryLayer_0_v"],
}
GF_SRC_EXT = ["heightAboveGround_80_u", "heightAboveGround_80_v",
              "isobaricInhPa_700_u", "isobaricInhPa_700_v",
              "isobaricInhPa_500_u", "isobaricInhPa_500_v", "isobaricInhPa_500_gh",
              "planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"]
GEXT_PATH = CACHE_DIR / "exp035_spatial_gext.parquet"


def _load(paths, src_cols):
    """train/test CSV를 이어붙여 (forecast_kst_dtm, grid_id, 원본 변수) 반환"""
    frames = []
    for p in paths:
        d = pd.read_csv(p, encoding="utf-8-sig",
                        usecols=["forecast_kst_dtm", "grid_id", "latitude", "longitude", *src_cols])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df


def _pivot(df, fn, master_idx, gid):
    """시각×격자 행렬. 변수별로 결측 시각/격자가 다를 수 있어 공통 축으로 reindex 후
    (a) 격자 결측은 그 시각 공간평균, (b) 시각 전체 결측은 그대로 NaN(하류 imputer 처리)."""
    v = df[["forecast_kst_dtm", "grid_id"]].copy()
    v["val"] = np.asarray(fn(df), dtype=np.float64)
    M = v.pivot_table(index="forecast_kst_dtm", columns="grid_id", values="val", aggfunc="mean")
    M = M.reindex(index=master_idx, columns=gid)
    row_mean = M.mean(axis=1)
    M = M.apply(lambda c: c.fillna(row_mean))
    return M


def build_spatial(prefix, paths, src_cols, var_defs):
    df = _load(paths, src_cols)
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates().sort_values("grid_id")
    gid = grids["grid_id"].to_numpy()
    # 평면 적합 설계행렬: [1, Δlat(km), Δlon(km)] — 좌표가 시간 불변이라 1회 계산
    lat = grids["latitude"].to_numpy(); lon = grids["longitude"].to_numpy()
    A = np.column_stack([np.ones(len(gid)),
                         (lat - lat.mean()) * 111.0,
                         (lon - lon.mean()) * 111.0 * np.cos(np.deg2rad(lat.mean()))])
    Apinv = np.linalg.pinv(A)                      # (3, n_grid)

    master_idx = pd.Index(np.sort(df["forecast_kst_dtm"].unique()), name="forecast_kst_dtm")
    spg_cols, spd_cols = {}, {}
    idx = master_idx
    for name, fn in var_defs.items():
        M = _pivot(df, fn, master_idx, gid)
        assert list(M.columns) == list(gid), f"{prefix}/{name} 격자 순서 불일치"
        V = M.to_numpy()                            # (T, n_grid)
        coef = V @ Apinv.T                          # (T, 3) = [절편, d/dlat, d/dlon]
        resid = V - coef @ A.T
        mean = V.mean(axis=1)
        spg_cols[f"sp{prefix}_{name}_mean"] = mean
        spg_cols[f"sp{prefix}_{name}_dlat"] = coef[:, 1]
        spg_cols[f"sp{prefix}_{name}_dlon"] = coef[:, 2]
        spg_cols[f"sp{prefix}_{name}_rstd"] = resid.std(axis=1)
        spd_cols[f"sd{prefix}_{name}_mean"] = mean
        for j, g in enumerate(gid):
            spd_cols[f"sd{prefix}_{name}_g{int(g)}"] = V[:, j] - mean
    spg = pd.DataFrame(spg_cols, index=idx).reset_index()
    spd = pd.DataFrame(spd_cols, index=idx).reset_index()
    return spg, spd


def build_all(force=False):
    """캐시가 있으면 읽고, 없으면 생성. 반환: (spg_df, spd_df, spg_cols, spd_cols)"""
    if not force and SPG_PATH.exists() and SPD_PATH.exists():
        spg = pd.read_parquet(SPG_PATH); spd = pd.read_parquet(SPD_PATH)
    else:
        t0 = time.time()
        ld_spg, ld_spd = build_spatial("L", [TRAIN_DIR / "ldaps_train.csv", TEST_DIR / "ldaps_test.csv"],
                                       LD_SRC, LD_VARS)
        print(f"  LDAPS 완료 ({time.time()-t0:.0f}s) SPG {ld_spg.shape} SPD {ld_spd.shape}", flush=True)
        gf_spg, gf_spd = build_spatial("G", [TRAIN_DIR / "gfs_train.csv", TEST_DIR / "gfs_test.csv"],
                                       GF_SRC, GF_VARS)
        print(f"  GFS 완료 ({time.time()-t0:.0f}s) SPG {gf_spg.shape} SPD {gf_spd.shape}", flush=True)
        spg = ld_spg.merge(gf_spg, on="forecast_kst_dtm", how="outer")
        spd = ld_spd.merge(gf_spd, on="forecast_kst_dtm", how="outer")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        spg.to_parquet(SPG_PATH, index=False); spd.to_parquet(SPD_PATH, index=False)
        print(f"  캐시 저장 완료 ({time.time()-t0:.0f}s)")
    spg_cols = [c for c in spg.columns if c != "forecast_kst_dtm"]
    spd_cols = [c for c in spd.columns if c != "forecast_kst_dtm"]
    return spg, spd, spg_cols, spd_cols


def build_gext(force=False):
    """spd2용 GFS 확장 변수의 격자 편차(SPD 형식)만 생성·캐시. 반환: (df, 컬럼목록)"""
    if not force and GEXT_PATH.exists():
        gext = pd.read_parquet(GEXT_PATH)
    else:
        t0 = time.time()
        _, gext = build_spatial("Gx", [TRAIN_DIR / "gfs_train.csv", TEST_DIR / "gfs_test.csv"],
                                GF_SRC_EXT, GF_VARS_EXT)   # (spg, spd) 중 spd만
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        gext.to_parquet(GEXT_PATH, index=False)
        print(f"  gext 캐시 저장 ({time.time()-t0:.0f}s) {gext.shape}", flush=True)
    cols = [c for c in gext.columns if c != "forecast_kst_dtm"]
    return gext, cols


if __name__ == "__main__":
    spg, spd, sc, dc = build_all(force="--force" in sys.argv)
    gext, gc = build_gext(force="--force" in sys.argv)
    print(f"\nGEXT(GFS 확장 편차) {len(gc)}개 피처, {len(gext)}행, 결측 {gext[gc].isna().sum().sum()}")
    print(f"\nSPG(경사 축약) {len(sc)}개 피처, {len(spg)}행")
    print(f"SPD(전체 편차) {len(dc)}개 피처, {len(spd)}행")
    print(f"기간: {spg['forecast_kst_dtm'].min()} ~ {spg['forecast_kst_dtm'].max()}")
    print(f"결측: SPG {spg[sc].isna().sum().sum()} / SPD {spd[dc].isna().sum().sum()}")
    print("\n--- SPG 샘플 통계 (LDAPS ws10 / GFS ws100) ---")
    show = [c for c in sc if "L_ws10_" in c or "G_ws100_" in c]
    print(spg[show].describe().T.round(4).to_string())
