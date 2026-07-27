# ═══════════════════════════════════════════════════════
# exp-017 제출: v13 base(exp-015) + pooled 정격 분류기(exp-016) 결합
#   → v_v13_snap30P_submit.csv
# 공통 기준 exp-012 LB(0.62844) 대비 독립 검증된 두 변경점의 합성:
#   - v13 SCADA 피처 (exp-015: LB +0.0007) — 회귀 base만 v13, 편향보정은 v10TPS(동일)
#   - pooled 정격 분류기 (exp-016: LB +0.0005) — exp-016과 동일하게 v10TPS 피처로 학습
# 새로 검증 안 된 조각 없음. 스냅 설정은 LB 검증된 snap30(p0.3/0.94) 유지.
# 실행: C:\miniconda3\python.exe -X utf8 exp017_submit.py [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

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
from exp015_scada_pc import (
    SCADA_COLS, build_scada_hourly, fit_scada_maps, apply_scada_features,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_P = 0.30
SNAP_TO = 0.94


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    cols_tps = col_sets["v10TPS"]
    cols_v13 = cols_tps + SCADA_COLS
    scada_h = build_scada_hourly()
    print(f"피처: 회귀 v13 {len(cols_v13)}개 / 분류기·편향보정 v10TPS {len(cols_tps)}개", flush=True)
    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols_tps, max_date, args.trees, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    print(f"편향 계수 추정 완료 ({time.time() - t0:.0f}s)", flush=True)

    tr_pool = train_df.loc[train_df["target"].notna()]
    pooled_clf = []

    predictions_final = pd.DataFrame(index=test_df.index)
    for group in [1, 2, 3]:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        # pooled 정격 분류기 (exp-016과 동일: v10TPS 피처, SCADA 미포함)
        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols_tps)
        if not pooled_clf:
            caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            y_pool_rated = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
            for seed in range(N_CLF_SEEDS):
                clf = lgb.LGBMClassifier(
                    n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                    num_leaves=31, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, n_jobs=args.jobs, verbose=-1,
                )
                clf.fit(X_pool_tr, y_pool_rated)
                pooled_clf.append(clf)
        p_rated = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in pooled_clf], axis=0)

        # 회귀 base (exp-015와 동일: v13 = v10TPS + SCADA 파워커브)
        maps = fit_scada_maps(scada_h[group], tr_g, max_date)
        tr_g = apply_scada_features(tr_g, maps)
        te_g = apply_scada_features(te_g, maps)

        X_tr, X_te = prepare_matrices(tr_g, te_g, cols_v13)
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)

        preds = []
        for seed in range(args.seeds):
            m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds.append(m.predict(X_te))
        pred_lgbm = np.mean(preds, axis=0)
        preds = []
        for seed in range(args.seeds):
            m = xgbm.XGBRegressor(
                n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            preds.append(m.predict(X_te))
        pred = 0.5 * pred_lgbm + 0.5 * np.mean(preds, axis=0)

        pred = apply_bias_correction_v2(pred, te_g["ldaps_ws_10m"].to_numpy(), group, coarse_map)
        pred = apply_rolling_smoothing(pd.Series(pred, index=te_g.index), window=3, alpha=0.7).to_numpy().copy()
        snap_mask = p_rated >= SNAP_P
        pred[snap_mask] = np.maximum(pred[snap_mask], capacity * SNAP_TO)
        pred = np.clip(pred, capacity * 0.10, capacity)

        predictions_final.loc[te_g.index, target_col] = pred
        print(f"{target_col}: train {len(tr_g)}행, 스냅 발동 {int(snap_mask.sum())}/{len(te_g)}시간 ({time.time() - t0:.0f}s)", flush=True)

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
        assert submission[col].max() <= CAPACITY_KWH[col]
        assert submission[col].min() >= 0

    output_path = DATA_DIR / "v_v13_snap30P_submit.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
    print(f"총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
