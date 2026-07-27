# ═══════════════════════════════════════════════════════
# exp-042 제출: GBM(spd 348피처·20-seed) + DL(q60 손실·spd 348·5-seed) 블렌드
#
#   챔피언 대비 변경점 **두 개뿐**:
#     (1) GBM 입력 132 → 132+격자편차216 = 348   (exp-040: base MAE 0.1209→0.1189)
#     (2) DL 손실 Huber → 비대칭 Huber q60(과소예측 ×1.2/과대 ×0.8)  (exp-039)
#   DL 용량(384×3·ep120)·정격분류기(132피처)·편향맵(132피처 coarse)·clip 0.16·α 0.50·
#   스냅 0.30/0.94는 전부 챔피언 그대로.
#   ※ 편향맵을 확장피처로 재추정하지 않는 이유: exp-040에서 챔피언 coarse 맵(cB)이 전 칸 우세.
#
#   LB 실적 (2026-07-27):
#     w0.65  **0.64649** (역대 최고, dlspd 0.64128 대비 +0.00521)  ← 현 챔피언
#     w0.70  0.64575 (dlspd 대비 +0.00447, w0.65에는 −0.00074)
#     → w를 낮출수록 FiCR↑·NMAE 불변. 게이트 홀드아웃 능선 정점은 w0.55(+0.0025).
#
#   **test 예측 캐시**: gbm_cf·dl_cf·pA·편향맵은 전부 w와 무관 → 1회 학습 후 parquet 캐시.
#   이후 다른 w 파일 생성은 즉시(재학습 없음).
#
#   ⚠ 이 스크립트는 파일만 만든다. **제출 슬롯 소비는 사용자 판단.**
# 실행: C:\miniconda3\python.exe -X utf8 exp042_submit.py --wgbm 0.60,0.55
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
from exp039_dl_diverse import make_xy, train_one, make_loss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94
CHAMP_FILE = "v_c16a50_gspd_q60_w65_submit.csv"   # 현 LB 챔피언 (비교용)


def build_test_predictions(args, train_df, test_df, cols, ext_cols):
    """w와 무관한 부분 전부 — DL CF · GBM CF · 정격확률 pA · 편향맵."""
    t0 = time.time()
    # === DL(q60): 전체 train 학습 → test CF 예측 ===
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy(train_df, ext_cols, imp, sca, "std", fit=True)
    Xte = sca.transform(imp.transform(test_df[ext_cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gte = test_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss(args.loss)
    dl_preds = []
    for s in range(args.dl_seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        dl_preds.append(predict(m, Xte, gte))
        print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
    test_df = test_df.copy()
    test_df["dl_cf"] = np.mean(dl_preds, axis=0)

    # === 편향맵(132) · GBM(348) · 정격분류기(132) ===
    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols, max_date, 500, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    tr_pool = train_df.loc[train_df["target"].notna()]
    pooled_clf, rows = [], []

    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        X_tr, X_te = prepare_matrices(tr_g, te_g, ext_cols)          # ← 348피처
        y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s))
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                      objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te)
                      for s in range(args.gbm_seeds)], axis=0)
        gbm_cf = (0.5 * pl + 0.5 * px) / capacity

        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols)   # ← 132피처 (챔피언 유지)
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
    ap.add_argument("--loss", type=str, default="q60")
    ap.add_argument("--wgbm", type=str, default="0.60,0.55", help="쉼표 구분 — 예측 1회로 여러 가중 파일 생성")
    ap.add_argument("--refresh", action="store_true", help="test 예측 캐시 무시하고 재학습")
    args = ap.parse_args()
    w_list = [float(w) for w in args.wgbm.split(",") if w]
    tag = f"{args.loss}_g{args.gbm_seeds}_d{args.dl_seeds}"
    PRED_CACHE = CACHE_DIR / f"exp042_test_pred_{tag}.parquet"
    BIAS_PKL = CACHE_DIR / f"exp042_test_bias_{tag}.pkl"

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS          # 132 — 분류기·편향맵은 이걸 유지
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ext_cols = list(cols) + spd_cols                       # 348 — GBM·DL 공통 입력

    if PRED_CACHE.exists() and BIAS_PKL.exists() and not args.refresh:
        pred = pd.read_parquet(PRED_CACHE)
        with open(BIAS_PKL, "rb") as f:
            coarse_map = pickle.load(f)
        print(f"test 예측 캐시 로드: {PRED_CACHE.name} {pred.shape} — 재학습 없음", flush=True)
    else:
        print(f"GBM 피처 {len(ext_cols)}개(챔피언 132 → +{len(spd_cols)}) / DL 손실 {args.loss} | "
              f"GBM {args.gbm_seeds}-seed, DL {args.dl_seeds}-seed", flush=True)
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

        out = f"v_c16a50_gspd_q60_w{int(round(w*100))}_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        print(f"[w{w:.2f}] Saved {out} | vs 챔피언(w65) 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.2f}%cap) | 그룹평균 "
              + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
