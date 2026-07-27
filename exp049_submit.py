# ═══════════════════════════════════════════════════════
# exp-049 제출: 가용성 정제 학습 — GBM(gspd 348·20-seed) + DL(q60·5-seed) 완전 정제 블렌드
#
#   챔피언(v_a40s93p27_gspd_q60_w65) 대비 변경점 **한 개뿐**:
#     학습 sample weight × 가용성 팩터 (터빈 정지 시간 ×0.05, exp-049 cln03)
#   피처(348)·DL 손실(q60)·용량·정격분류기(132)·편향맵(132 coarse)·후처리(신PP
#   c16·al40·sn93·pt27) 전부 챔피언 그대로. 게이트: ΔCV +0.0053 / Δhold +0.0055 (cln×cln_w70).
#
#   test 예측 캐시: gbm_cf·dl_cf·pA·편향맵은 w와 무관 → 1회 학습 후 parquet 캐시.
#   ⚠ 이 스크립트는 파일만 만든다. **제출 슬롯 소비는 사용자 판단.**
# 실행: C:\miniconda3\python.exe -X utf8 exp049_submit.py --wgbm 0.70,0.65
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, DATA_DIR,
    make_sample_weight, apply_rolling_smoothing, apply_bias_correction_v2,
    lgbm_params, prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp033_dl import predict, GIDX
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import DL_CFG, CACHE_DIR
from exp039_dl_diverse import train_one, make_loss
from exp049_clean import merge_avail, avail_factor, make_xy_clean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 신PP (현 챔피언 파일과 동일)
ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCHEME = "cln03"
CHAMP_FILE = "v_a40s93p27_gspd_q60_w65_submit.csv"   # 현 LB 챔피언 (비교용)


def build_test_predictions(args, train_df, test_df, cols, ext_cols):
    """w와 무관한 부분 — 정제 DL CF · 정제 GBM CF · 정격확률 pA · 편향맵."""
    t0 = time.time()
    # === DL(q60·정제): 전체 train 학습 → test CF 예측 ===
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(train_df, ext_cols, imp, sca, SCHEME, fit=True)
    Xte = sca.transform(imp.transform(test_df[ext_cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gte = test_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    dl_preds = []
    for s in range(args.dl_seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        dl_preds.append(predict(m, Xte, gte))
        print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
    test_df = test_df.copy()
    test_df["dl_cf"] = np.mean(dl_preds, axis=0)

    # === 편향맵(132, 무변경) · 정제 GBM(348) · 정격분류기(132, 무변경) ===
    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols, max_date, 500, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    tr_pool = train_df.loc[train_df["target"].notna()]
    pooled_clf, rows = [], []

    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        X_tr, X_te = prepare_matrices(tr_g, te_g, ext_cols)          # 348피처
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
        w_tr = w_tr * avail_factor(tr_g["avail_strict"].to_numpy(), SCHEME)   # ← 유일한 변경점
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s))
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                      objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
                      for s in range(args.gbm_seeds)], axis=0)
        gbm_cf = (0.5 * pl + 0.5 * px) / capacity

        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols)   # 132피처 (무변경)
        if not pooled_clf:
            caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
            for s in range(N_CLF_SEEDS):
                c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=s,
                    n_jobs=args.jobs, verbose=-1)
                c.fit(X_pool_tr, yA); pooled_clf.append(c)
        pA = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in pooled_clf], axis=0)

        rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                  "kpx_group": group,
                                  "ws": te_g["ldaps_ws_10m"].to_numpy(),
                                  "gbm_cf": gbm_cf, "dl_cf": te_g["dl_cf"].to_numpy(), "pA": pA}))
        print(f"kpx_group_{group}: DL vs GBM CF 상관 {np.corrcoef(gbm_cf, te_g['dl_cf'].to_numpy())[0,1]:.3f} | "
              f"스냅 {int((pA>=SNAP_THR).sum())} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True), coarse_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--wgbm", type=str, default="0.70,0.65")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    w_list = [float(w) for w in args.wgbm.split(",") if w]
    tag = f"{SCHEME}_g{args.gbm_seeds}_d{args.dl_seeds}"
    PRED_CACHE = CACHE_DIR / f"exp049_test_pred_{tag}.parquet"
    BIAS_PKL = CACHE_DIR / f"exp049_test_bias_{tag}.pkl"

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS          # 132
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ext_cols = list(cols) + spd_cols                       # 348
    train_df = merge_avail(train_df)
    n_lab = train_df["target"].notna().sum()
    n_low = (train_df["target"].notna() & (train_df["avail_strict"].fillna(1.0) < 0.999)).sum()
    print(f"학습행 {n_lab} 중 저가용 다운웨이트 {n_low} ({n_low/n_lab*100:.1f}%)", flush=True)

    if PRED_CACHE.exists() and BIAS_PKL.exists() and not args.refresh:
        pred = pd.read_parquet(PRED_CACHE)
        with open(BIAS_PKL, "rb") as f:
            coarse_map = pickle.load(f)
        print(f"test 예측 캐시 로드: {PRED_CACHE.name} {pred.shape} — 재학습 없음", flush=True)
    else:
        print(f"정제 학습({SCHEME}) | GBM {args.gbm_seeds}-seed, DL {args.dl_seeds}-seed | 피처 {len(ext_cols)}", flush=True)
        pred, coarse_map = build_test_predictions(args, train_df, test_df, cols, ext_cols)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(PRED_CACHE, index=False)
        with open(BIAS_PKL, "wb") as f:
            pickle.dump(coarse_map, f)
        print(f"test 예측 캐시 저장 → {PRED_CACHE.name}", flush=True)

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    for w in w_list:
        parts = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            g = pred.loc[pred["kpx_group"] == group]
            pA = g["pA"].to_numpy()
            p = apply_bias_correction_v2((w * g["gbm_cf"].to_numpy() + (1 - w) * g["dl_cf"].to_numpy()) * capacity,
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

        out = f"v_a40s93p27_cln_w{int(round(w*100))}_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        print(f"[w{w:.2f}] Saved {out} | vs 챔피언(신PP w65) 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.2f}%cap) | 그룹평균 "
              + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
