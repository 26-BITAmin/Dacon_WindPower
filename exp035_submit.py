# ═══════════════════════════════════════════════════════
# exp-035 제출: GBM(20-seed) + **원시 다격자 입력 DL**(spd) 블렌드 70:30
#
#   현 챔피언 v_c16a50_dlblend(exp-033) 대비 변경점은 **DL 입력뿐** — 132피처에
#   원시 격자 편차 216개(LDAPS 16격자·GFS 9격자, 변수별 공간평균 + 격자별 편차) 추가.
#   GBM·후처리·블렌드 가중(0.70)·DL 용량(384×3·ep120)은 전부 챔피언 그대로.
#   이중 게이트(exp035_log / exp035_r2_log): 챔피언 대비 ΔCV +0.0012 / Δ홀드아웃 +0.0022,
#   1-NMAE·FiCR 동반 상승, DL↔GBM 오차 상관 0.851→0.818.
#
#   ⚠ 이 스크립트는 **자동 실행하지 않는다.** 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp035_submit.py [--jobs N] [--wgbm 0.7]
# ═══════════════════════════════════════════════════════
import argparse
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
from exp033_dl import train_model, predict, GIDX
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import make_xy

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--wgbm", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="v_c16a50_dlspd_submit.csv")
    args = ap.parse_args()

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS          # GBM 입력 (챔피언과 동일 132)

    # DL 전용 입력: 132 + 원시 격자 편차 216
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    dl_cols = list(cols) + spd_cols
    print(f"GBM 피처 {len(cols)}개 / DL 피처 {len(dl_cols)}개 | 블렌드 GBM {args.wgbm:.2f}/DL {1-args.wgbm:.2f} | "
          f"GBM {args.gbm_seeds}-seed, DL {args.dl_seeds}-seed(ep{args.epochs}, {args.width}x{args.depth})", flush=True)

    # === DL: 전체 train 학습 → test CF 예측 ===
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy(train_df, dl_cols, imp, sca, fit=True)
    Xte = sca.transform(imp.transform(test_df[dl_cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gte = test_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    dl_preds = []
    for s in range(args.dl_seeds):
        model = train_model(Xtr, gtr, ytr, wtr, width=args.width, depth=args.depth, p_drop=0.15,
                            lr=1e-3, wd=1e-4, epochs=args.epochs, bs=1024, seed=s, verbose=(s == 0))
        dl_preds.append(predict(model, Xte, gte))
    test_df = test_df.copy()
    test_df["dl_cf"] = np.mean(dl_preds, axis=0)
    print(f"DL 학습 완료 ({time.time()-t0:.0f}s)", flush=True)

    # === GBM base + 편향맵 + 스냅분류기 (챔피언과 동일) ===
    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols, max_date, 500, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    tr_pool = train_df.loc[train_df["target"].notna()]
    pooled_clf = []
    predictions_final = pd.DataFrame(index=test_df.index)

    for group in [1, 2, 3]:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        X_tr, X_te = prepare_matrices(tr_g, te_g, cols)
        y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s)).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                      objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        gbm_cf = (0.5 * pl + 0.5 * px) / capacity
        dl_cf = te_g["dl_cf"].to_numpy()
        blend_cf = args.wgbm * gbm_cf + (1 - args.wgbm) * dl_cf

        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols)
        if not pooled_clf:
            caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
            for s in range(N_CLF_SEEDS):
                c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbose=-1)
                c.fit(X_pool_tr, yA); pooled_clf.append(c)
        pA = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in pooled_clf], axis=0)

        pred = apply_bias_correction_v2(blend_cf * capacity, te_g["ldaps_ws_10m"].to_numpy(), group, coarse_map)
        pred = apply_rolling_smoothing(pd.Series(pred, index=te_g.index), window=3, alpha=ALPHA).to_numpy().copy()
        pred[pA >= SNAP_THR] = np.maximum(pred[pA >= SNAP_THR], capacity * SNAP_TO)
        pred = np.clip(pred, capacity * CLIP_LO, capacity)
        predictions_final.loc[te_g.index, target_col] = pred
        print(f"{target_col}: DL vs GBM CF 상관 {np.corrcoef(gbm_cf, dl_cf)[0,1]:.3f} | 스냅 {int((pA>=SNAP_THR).sum())} ({time.time()-t0:.0f}s)", flush=True)

    test_pred_long = test_df[["forecast_id", "forecast_kst_dtm", "kpx_group"]].copy()
    for group in [1, 2, 3]:
        mask = test_df["kpx_group"] == group
        test_pred_long.loc[mask, "pred"] = predictions_final.loc[mask, f"kpx_group_{group}"]
    submission = test_pred_long.pivot(index=["forecast_id", "forecast_kst_dtm"], columns="kpx_group", values="pred")
    submission.columns = [f"kpx_group_{int(c)}" for c in submission.columns]
    submission = submission.reset_index()
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    submission = sample_submission[["forecast_id"]].merge(submission, on="forecast_id", how="left")

    assert submission.shape[0] == sample_submission.shape[0]
    assert submission[TARGET_COLS].isna().sum().sum() == 0
    assert (submission["forecast_id"] == sample_submission["forecast_id"]).all()
    for col in TARGET_COLS:
        assert submission[col].max() <= CAPACITY_KWH[col] and submission[col].min() >= 0

    output_path = DATA_DIR / args.out
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / "v_c16a50_dlblend_submit.csv")
    diff = np.abs(submission[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    print(f"vs 챔피언(v_c16a50_dlblend): 평균 |차이| {diff.mean():.1f}kWh, 변경 {(diff>1).sum()}/{diff.size}칸")
    print(f"Saved: {output_path}\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
