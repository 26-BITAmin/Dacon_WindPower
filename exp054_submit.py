# ═══════════════════════════════════════════════════════
# exp-054 제출: ECMWF 10m 피처 4개를 GBM base에 추가 (프로브 제출)
#
#   챔피언(v_a40s93p27_cln_w65) 대비 변경점 **한 개뿐**:
#     GBM 입력 348 → 352 (ec_ws·ec_wdsin·ec_wdcos·ec_dis, D-2 12z ECMWF open-data)
#   DL(정제 q60 5-seed)·정격분류기·편향맵·후처리(신PP)·w0.65는 exp-049 제출 캐시 재사용
#   → GBM 20-seed만 재학습. 게이트: gEC×dC_w65 ΔCV +0.0007 / Δhold +0.0011 (3중 미통과,
#   전 셀 양 게이트 양수 + 데이터 축 증폭 선례로 프로브 — 사용자 승인 2026-07-28).
#
#   ⚠ 이 스크립트는 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp054_submit.py --wgbm 0.65
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR, make_sample_weight,
                         apply_rolling_smoothing, apply_bias_correction_v2,
                         lgbm_params, prepare_matrices)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp049_clean import merge_avail, avail_factor
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCHEME = "cln03"
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
GBM_EC_TEST = CACHE_DIR / "exp054_test_gbm_ec_g20.parquet"
CHAMP_FILE = "v_a40s93p27_cln_w65_submit.csv"          # 현 LB 챔피언 (비교용)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--wgbm", type=str, default="0.65")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    w_list = [float(w) for w in args.wgbm.split(",") if w]
    t0 = time.time()

    # === 데이터 (exp-049 제출과 동일 + EC 4피처) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS          # 132
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    for df in (train_df, test_df):
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
    gbm_cols = list(cols) + spd_cols + EC_COLS             # 352
    train_df = merge_avail(train_df)
    cov_te = test_df["ec_ws"].notna().mean()
    print(f"GBM 피처 {len(gbm_cols)} | test EC 커버리지 {cov_te*100:.1f}%", flush=True)

    # === 무변경 성분: exp-049 캐시 (DL·pA·ws·편향맵) ===
    assert E49_PRED.exists() and E49_BIAS.exists(), "exp-049 제출 캐시 필요"
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    print(f"exp-049 캐시 로드: {pred.shape} (dl_cf·pA·ws·편향맵 재사용)", flush=True)

    # === 변경점: GBM(352) 20-seed 재학습 ===
    if GBM_EC_TEST.exists() and not args.refresh:
        gec = pd.read_parquet(GBM_EC_TEST)
        print(f"EC GBM test 캐시 로드: {GBM_EC_TEST.name}", flush=True)
    else:
        rows = []
        for group in [1, 2, 3]:
            tg = time.time()
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
            te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            X_tr, X_te = prepare_matrices(tr_g, te_g, gbm_cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
            w_tr = w_tr * avail_factor(tr_g["avail_strict"].to_numpy(), SCHEME)
            pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s))
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                          subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                          objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
                          for s in range(args.gbm_seeds)], axis=0)
            rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                      "kpx_group": group,
                                      "gbm_cf_ec": (0.5 * pl + 0.5 * px) / capacity}))
            print(f"  kpx_group_{group} GBM {args.gbm_seeds}-seed 완료 ({time.time()-tg:.0f}s)", flush=True)
        gec = pd.concat(rows, ignore_index=True)
        gec.to_parquet(GBM_EC_TEST, index=False)
        print(f"EC GBM test 캐시 저장 → {GBM_EC_TEST.name}", flush=True)

    pred = pred.merge(gec, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_ec"].notna().all(), "EC GBM 정렬 실패"
    corr = np.corrcoef(pred["gbm_cf_ec"], pred["gbm_cf"])[0, 1]
    print(f"신구 GBM CF 상관: {corr:.4f}", flush=True)

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    for w in w_list:
        parts = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            g = pred.loc[pred["kpx_group"] == group]
            pA = g["pA"].to_numpy()
            p = apply_bias_correction_v2((w * g["gbm_cf_ec"].to_numpy() + (1 - w) * g["dl_cf"].to_numpy()) * capacity,
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

        out = f"v_a40s93p27_ec_w{int(round(w*100))}_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        rated = [(sub[c] >= CAPACITY_KWH[c] * 0.94).sum() for c in TARGET_COLS]
        print(f"[w{w:.2f}] Saved {out} | vs 챔피언(cln_w65) 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.2f}%cap) | 정격시간 {rated} | 그룹평균 "
              + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
