# ═══════════════════════════════════════════════════════
# exp-055 제출: EC⊕TB(터빈 분해) 50:50 멤버 혼합 프로브 (mix50_w65)
#
#   보드 챔피언(v_a40s93p27_ec_w65) 대비 변경점 **한 개뿐**:
#     GBM base = cf_ec → 0.5·cf_ec + 0.5·cf_tb (터빈 분해 합산, 352피처·cln 정지 다운웨이트)
#   DL(정제 q60 5-seed)·pA·편향맵·신PP·w0.65 전부 캐시 재사용.
#   게이트: mix50_w65 ΔCV +0.0015(FiCR +0.0035) / Δhold −0.0009 — 이중 게이트 미통과
#   프로브 (홀드아웃의 데이터 규모 맹점(exp-035) 논거, 사용자 승인 2026-07-28).
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp055_submit.py
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR,
                         apply_rolling_smoothing, apply_bias_correction_v2,
                         lgbm_params, prepare_matrices)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp055_turbine import build_turbine_hours, calib_ratios, TCAP

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
EC_TEST = CACHE_DIR / "exp054_test_gbm_ec_g20.parquet"
TB_TEST = CACHE_DIR / "exp055_test_gbm_tb_g20.parquet"
CHAMP_FILE = "v_a40s93p27_ec_w65_submit.csv"
W_GBM = 0.65
MIX = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

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

    tbh = build_turbine_hours()
    tbh["forecast_kst_dtm"] = pd.to_datetime(tbh["forecast_kst_dtm"])
    calib = calib_ratios(tbh, train_df)
    print(f"피처 {len(use_cols)} | 보정 비율 " + ", ".join(f"g{g}={r:.4f}" for g, r in calib.items()), flush=True)

    # === TB 멤버: 터빈 풀링 20-seed 전체 학습 → test 예측 ===
    if TB_TEST.exists() and not args.refresh:
        tbp = pd.read_parquet(TB_TEST)
        print(f"TB test 캐시 로드: {TB_TEST.name}", flush=True)
    else:
        rows = []
        for group in [1, 2, 3]:
            tg = time.time()
            gcap = CAPACITY_KWH[f"kpx_group_{group}"]
            tw = train_df.loc[train_df["kpx_group"] == group]
            te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            tbg = tbh.loc[(tbh["group"] == group) & tbh["energy"].notna()]
            tr_t = tbg.merge(tw, on="forecast_kst_dtm", how="inner")
            tbs = sorted(tbg["tb"].unique())
            oh_cols = [f"tb_{t}" for t in tbs]
            for t in tbs:
                tr_t[f"tb_{t}"] = (tr_t["tb"] == t).astype(float)
            te_rep = pd.concat([te_g.assign(**{f"tb_{t2}": float(t2 == t) for t2 in tbs})
                                for t in tbs], ignore_index=True)
            X_tr, X_te = prepare_matrices(tr_t, te_rep, use_cols + oh_cols)
            cf_t = (tr_t["energy"] / tr_t["tcap"]).to_numpy()
            w_tr = np.where(cf_t < 0.10, 0.05, 1.0) * np.where(tr_t["dead"].to_numpy(), 0.05, 1.0)
            pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s))
                          .fit(X_tr, cf_t, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                          n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror")
                          .fit(X_tr, cf_t, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
            pred_t = np.clip(0.5 * pl + 0.5 * px, 0, 1).reshape(len(tbs), len(te_g))
            tcaps = np.array([TCAP[t[:6]] for t in tbs])[:, None]
            g_cf = (pred_t * tcaps).sum(axis=0) * calib[group] / gcap
            rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                      "kpx_group": group, "gbm_cf_tb": g_cf}))
            print(f"  kpx_group_{group}: 학습행 {len(tr_t)} | TB {args.gbm_seeds}-seed 완료 "
                  f"({time.time()-tg:.0f}s)", flush=True)
        tbp = pd.concat(rows, ignore_index=True)
        tbp.to_parquet(TB_TEST, index=False)
        print(f"TB test 캐시 저장 → {TB_TEST.name}", flush=True)

    # === 무변경 성분 로드 + 혼합·후처리 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    gec = pd.read_parquet(EC_TEST)
    pred = pred.merge(gec, on=["forecast_id", "kpx_group"], how="left") \
               .merge(tbp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_ec"].notna().all() and pred["gbm_cf_tb"].notna().all()
    corr = np.corrcoef(pred["gbm_cf_ec"], pred["gbm_cf_tb"])[0, 1]
    print(f"EC↔TB test CF 상관: {corr:.4f}", flush=True)

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    parts = []
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        base_cf = MIX * g["gbm_cf_ec"].to_numpy() + (1 - MIX) * g["gbm_cf_tb"].to_numpy()
        p = apply_bias_correction_v2((W_GBM * base_cf + (1 - W_GBM) * g["dl_cf"].to_numpy()) * capacity,
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

    out = "v_a40s93p27_ectb50_w65_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    n93 = [(sub[c] >= CAPACITY_KWH[c] * 0.93).sum() for c in TARGET_COLS]
    print(f"Saved {out} | vs ec_w65 평균 |차이| {diff.mean():.1f}kWh ({diff.mean()/21600*100:.2f}%cap) | "
          f"스냅(≥0.93cap) {n93} | 그룹평균 " + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
