# ═══════════════════════════════════════════════════════
# exp-034 제출: GBM(20-seed) 0.65 + DL 앙상블(384×3 ⊕ 512×4) 0.35
#   현 챔피언 v_c16a50_dlblend(GBM 0.7 + DL_384 0.3) 대비: DL를 앙상블(384+512)로,
#   DL 비중 0.30→0.35. 이중게이트(exp034): ens_65 CV +0.0025 / 홀드아웃 +0.0024
#   (가장 대칭·홀드아웃 최고). 앙상블 파트너가 더 견고 → 높은 DL 비중 안전.
# 실행: C:\miniconda3\python.exe -X utf8 exp034_submit.py [--jobs N]
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

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR,
                         make_sample_weight, apply_rolling_smoothing, apply_bias_correction_v2,
                         lgbm_params, prepare_matrices)
from exp009_rated_snap import (estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp033_dl import make_xy, train_model, predict, GIDX

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94
W_GBM = 0.65
DL_CONFIGS = [dict(width=384, depth=3, p_drop=0.15, epochs=120),
              dict(width=512, depth=4, p_drop=0.20, epochs=200)]


def dl_ensemble_predict(train_df, test_df, cols, seeds):
    imp, sca = SimpleImputer(strategy="median"), StandardScaler()
    Xtr, gtr, ytr, wtr, _, _ = make_xy(train_df, cols, imp, sca, fit=True)
    Xte = sca.transform(imp.transform(test_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gte = test_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cfg_means = []
    for ci, cfg in enumerate(DL_CONFIGS):
        preds = []
        for s in range(seeds):
            m = train_model(Xtr, gtr, ytr, wtr, width=cfg["width"], depth=cfg["depth"], p_drop=cfg["p_drop"],
                            lr=1e-3, wd=1e-4, epochs=cfg["epochs"], bs=1024, seed=s, verbose=(s == 0))
            preds.append(predict(m, Xte, gte))
        cfg_means.append(np.mean(preds, axis=0))
        print(f"  DL cfg{ci} ({cfg['width']}x{cfg['depth']}) {seeds}-seed 완료", flush=True)
    return 0.5 * (cfg_means[0] + cfg_means[1])   # 앙상블


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gbm_seeds", type=int, default=20)
    ap.add_argument("--dl_seeds", type=int, default=4)
    ap.add_argument("--out", type=str, default="v_c16a50_dlens65_submit.csv")
    args = ap.parse_args()

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    print(f"피처 {len(cols)}개 | GBM {W_GBM:.2f} + DL앙상블(384×3⊕512×4) {1-W_GBM:.2f} | "
          f"GBM {args.gbm_seeds}-seed, DL {args.dl_seeds}-seed/cfg", flush=True)

    test_df = test_df.copy()
    test_df["dl_cf"] = dl_ensemble_predict(train_df, test_df, cols, args.dl_seeds)
    print(f"DL 앙상블 완료 ({time.time()-t0:.0f}s)", flush=True)

    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols, max_date, 500, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    tr_pool = train_df.loc[train_df["target"].notna()]
    pooled_clf = []
    predictions_final = pd.DataFrame(index=test_df.index)

    for group in [1, 2, 3]:
        tc = f"kpx_group_{group}"; cap = CAPACITY_KWH[tc]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        X_tr, X_te = prepare_matrices(tr_g, te_g, cols)
        y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), cap, low_weight=0.05)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s)).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
              objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in range(args.gbm_seeds)], axis=0)
        gbm_cf = (0.5 * pl + 0.5 * px) / cap
        blend_cf = W_GBM * gbm_cf + (1 - W_GBM) * te_g["dl_cf"].to_numpy()

        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols)
        if not pooled_clf:
            cp = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            yA = (tr_pool["target"].to_numpy() >= cp * RATED_CF).astype(int)
            for s in range(N_CLF_SEEDS):
                c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbose=-1)
                c.fit(X_pool_tr, yA); pooled_clf.append(c)
        pA = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in pooled_clf], axis=0)

        pred = apply_bias_correction_v2(blend_cf * cap, te_g["ldaps_ws_10m"].to_numpy(), group, coarse_map)
        pred = apply_rolling_smoothing(pd.Series(pred, index=te_g.index), window=3, alpha=ALPHA).to_numpy().copy()
        pred[pA >= SNAP_THR] = np.maximum(pred[pA >= SNAP_THR], cap * SNAP_TO)
        pred = np.clip(pred, cap * CLIP_LO, cap)
        predictions_final.loc[te_g.index, tc] = pred
        print(f"{tc}: 완료 ({time.time()-t0:.0f}s)", flush=True)

    tpl = test_df[["forecast_id", "forecast_kst_dtm", "kpx_group"]].copy()
    for group in [1, 2, 3]:
        tpl.loc[test_df["kpx_group"] == group, "pred"] = predictions_final.loc[test_df["kpx_group"] == group, f"kpx_group_{group}"]
    sub = tpl.pivot(index=["forecast_id", "forecast_kst_dtm"], columns="kpx_group", values="pred")
    sub.columns = [f"kpx_group_{int(c)}" for c in sub.columns]
    sub = sub.reset_index()
    sub["forecast_kst_dtm"] = pd.to_datetime(sub["forecast_kst_dtm"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    sub = sample_submission[["forecast_id"]].merge(sub, on="forecast_id", how="left")
    assert sub.shape[0] == sample_submission.shape[0] and sub[TARGET_COLS].isna().sum().sum() == 0
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all()
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0
    out = DATA_DIR / args.out
    sub.to_csv(out, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / "v_c16a50_dlblend_submit.csv")
    d = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    print(f"vs 현챔피언(dlblend): 평균|차이| {d.mean():.1f}kWh, 변경 {(d>1).sum()}/{d.size}칸")
    print(f"Saved: {out}\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
