# ═══════════════════════════════════════════════════════
# exp-056 제출: HPO GBM (Optuna 우승 파라미터) — hpo×dC_w65 프로브
#
#   보드 챔피언(v_a40s93p27_ec_w65) 대비 변경점 **한 개뿐**:
#     GBM 하이퍼파라미터 — LGBM 500트리·lr0.03·31잎·d6 → 448트리·lr0.026·37잎·d12·
#     ff0.57·bf0.69·λ2 0.05 / XGB 500·0.03·d6 → 585트리·lr0.025·d4·mcw29·cs0.40·α1.34
#   피처(352)·정제 가중(cln03)·DL·pA·편향맵·신PP·w0.65 전부 동일 (캐시 재사용).
#   게이트: GBM 단독 +0.0024/+0.0024 (양 게이트 매그니튜드), 블렌드 w65 +0.0016/+0.0006
#   (두 지표 동반 상승, 3중 미통과 프로브 — 사용자 승인 2026-07-28).
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp056_submit.py
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
HPO_TEST = CACHE_DIR / "exp056_test_gbm_hpo_g20.parquet"
CHAMP_FILE = "v_a40s93p27_ec_w65_submit.csv"
W_GBM = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    best = json.loads(RESULT_JSON.read_text())
    print(f"HPO 파라미터: LGB {best['lgb_n']}트리 / XGB {best['xgb_n']}트리", flush=True)

    # === 데이터 (exp-054 제출과 동일 352피처) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    for df in (train_df, test_df):
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
    use_cols = list(cols) + spd_cols + EC_COLS             # 352
    train_df = merge_avail(train_df)

    # === HPO GBM 20-seed 학습 → test 예측 ===
    if HPO_TEST.exists() and not args.refresh:
        gp = pd.read_parquet(HPO_TEST)
        print(f"HPO test 캐시 로드: {HPO_TEST.name}", flush=True)
    else:
        rows = []
        for group in [1, 2, 3]:
            tg = time.time()
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
            te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            X_tr, X_te = prepare_matrices(tr_g, te_g, use_cols)
            y_tr = tr_g["target"].to_numpy()
            w_tr = make_sample_weight(y_tr, capacity, low_weight=0.05)
            w_tr = w_tr * avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            pl = np.mean([lgb.LGBMRegressor(n_estimators=best["lgb_n"], objective="mae",
                          random_state=s, n_jobs=args.jobs, verbose=-1, **best["lgb"])
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=best["xgb_n"], objective="reg:absoluteerror",
                          tree_method="hist", random_state=s, n_jobs=args.jobs, verbosity=0, **best["xgb"])
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
            rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                      "kpx_group": group,
                                      "gbm_cf_hpo": (0.5 * pl + 0.5 * px) / capacity}))
            print(f"  kpx_group_{group} HPO {args.gbm_seeds}-seed 완료 ({time.time()-tg:.0f}s)", flush=True)
        gp = pd.concat(rows, ignore_index=True)
        gp.to_parquet(HPO_TEST, index=False)
        print(f"HPO test 캐시 저장 → {HPO_TEST.name}", flush=True)

    # === 무변경 성분 로드 + 블렌드·후처리 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_hpo"].notna().all()

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    parts = []
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2((W_GBM * g["gbm_cf_hpo"].to_numpy() + (1 - W_GBM) * g["dl_cf"].to_numpy()) * capacity,
                                     g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], capacity * SNAP_TO)
        parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{group}",
                                   "pred": np.clip(p, capacity * CLIP_LO, capacity)}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")

    assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
    assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all(), "forecast_id 불일치"
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all(), "dtm 불일치"
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0, f"{col} 범위 이탈"

    out = "v_a40s93p27_hpo_w65_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    n93 = [(sub[c] >= CAPACITY_KWH[c] * 0.93).sum() for c in TARGET_COLS]
    print(f"Saved {out} | vs ec_w65 평균 |차이| {diff.mean():.1f}kWh ({diff.mean()/21600*100:.2f}%cap) | "
          f"스냅(≥0.93cap) {n93} | 그룹평균 " + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
