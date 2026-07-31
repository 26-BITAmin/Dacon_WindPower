# ═══════════════════════════════════════════════════════
# exp-073 제출: enfo ens3 프로브 — 챔피언 + 앙상블 최소 블록 3피처 (352→355)
#
#   챔피언(v_a40s93p27_hpo_w65_g3s97, LB 0.64973) 대비 변경점 **한 개뿐**:
#     GBM 입력에 ens3 = {ens_ws, ens_spread, ens_gap} 추가 (20-seed 재학습).
#   DL·pA·편향맵·신PP·w0.65·그룹별 스케일(0.98, 0.98, 0.97) 전부 동일 (캐시 재사용).
#   게이트: ens3_w65 ΔCV +0.0003/Δhold +0.0009 — 매그니튜드 미달 프로브
#   (사용자 승인 2026-07-31: 평균 이동 중립·corr 0.9996으로 하방 캡, exp-054 선례).
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp073_submit.py
# ═══════════════════════════════════════════════════════
import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR, make_sample_weight,
                         apply_rolling_smoothing, apply_bias_correction_v2, prepare_matrices)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp049_clean import merge_avail, avail_factor
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp056_hpo import RESULT_JSON
from exp073_gate import build_ens_frame, ENS3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
ENS_TEST = CACHE_DIR / "exp073_test_gbm_ens3_g20.parquet"
CHAMP_FILE = "v_a40s93p27_hpo_w65_g3s97_submit.csv"
W_GBM = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    best = json.loads(RESULT_JSON.read_text())

    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    ens = build_ens_frame()
    for name in ("train_df", "test_df"):
        df = locals()[name]
        df = df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
        df = df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
        df["ens_gap"] = df["ens_ws"] - df["ec_ws"]
        if name == "train_df":
            train_df = df
        else:
            test_df = df
    use_cols = list(cols) + spd_cols + EC_COLS + ENS3      # 355
    train_df = merge_avail(train_df)
    cov = test_df["ens_ws"].notna().mean()
    print(f"피처 {len(use_cols)} | test ens 커버리지 {cov:.3f}", flush=True)
    assert cov > 0.99, "test ens 커버리지 부족"

    if ENS_TEST.exists():
        gp = pd.read_parquet(ENS_TEST)
        print(f"ens3 test 캐시 로드: {ENS_TEST.name}", flush=True)
    else:
        rows = []
        for group in [1, 2, 3]:
            tg = time.time()
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
            te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            X_tr, X_te = prepare_matrices(tr_g, te_g, use_cols)
            y_tr = tr_g["target"].to_numpy()
            w_tr = make_sample_weight(y_tr, capacity, low_weight=0.05) * \
                avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            pl = np.mean([lgb.LGBMRegressor(n_estimators=best["lgb_n"], objective="mae",
                          random_state=s, n_jobs=args.jobs, verbose=-1, **best["lgb"])
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
                          for s in range(args.gbm_seeds)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=best["xgb_n"], objective="reg:absoluteerror",
                          tree_method="hist", random_state=s, n_jobs=args.jobs, verbosity=0, **best["xgb"])
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
                          for s in range(args.gbm_seeds)], axis=0)
            rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                      "kpx_group": group,
                                      "gbm_cf_ens3": (0.5 * pl + 0.5 * px) / capacity}))
            print(f"  g{group} ens3 {args.gbm_seeds}-seed 완료 ({time.time()-tg:.0f}s)", flush=True)
        gp = pd.concat(rows, ignore_index=True)
        gp.to_parquet(ENS_TEST, index=False)
        print(f"ens3 test 캐시 저장 → {ENS_TEST.name}", flush=True)

    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_ens3"].notna().all()

    parts = []
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (W_GBM * g["gbm_cf_ens3"].to_numpy() + (1 - W_GBM) * g["dl_cf"].to_numpy()) * capacity,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], capacity * SNAP_TO)
        p = np.clip(p, capacity * CLIP_LO, capacity)
        p = np.clip(p * SCALES[group], 0, capacity)
        parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{group}", "pred": p}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")
    assert sub.shape[0] == sample_submission.shape[0]
    assert sub[TARGET_COLS].isna().sum().sum() == 0
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all()
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all()
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0

    out = "v_a40s93p27_ens3_w65_g3s97_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    shift = (sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()).mean()
    print(f"Saved {out} | vs 챔피언(g3s97) 평균 |차이| {diff.mean():.1f}kWh "
          f"({diff.mean()/21600*100:.3f}%cap) | 평균 이동 {shift:+.1f}kWh", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
