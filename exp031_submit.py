# ═══════════════════════════════════════════════════════
# exp-031 제출: 챔피언(c16_a50, 20-seed) + 2단 정격 스냅(tier B)
#   현 챔피언 v_c16a50_s20 대비 변경점: tier B 추가 하나뿐.
#   tier B = P(actual≥0.70cap) pooled 분류기 ≥ thr_B → max(pred, target_B·cap)
#   진단(exp-030): CF 0.7~0.9 밴드가 FiCR 손실 33%·71% 과소예측인데 기존 스냅
#     (P≥0.90cap)이 놓침. 이중 게이트: tB_050_082 CV +0.0054 / 홀드아웃 +0.0066
#     (exp031_log.txt), 4 fold 전부 +, fold_std 0.0040(챔피언 0.0089보다 낮음).
# 실행: C:\miniconda3\python.exe -X utf8 exp031_submit.py [--jobs N] [--seeds 20]
#       [--thrb 0.50] [--tgtb 0.82] [--out 파일명]
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
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO = 0.50, 0.16
SNAP_A_THR, SNAP_A_TO = 0.30, 0.94
RATED_CF_B = 0.70


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--thrb", type=float, default=0.50)   # tier B 임계
    parser.add_argument("--tgtb", type=float, default=0.82)   # tier B 목표 CF
    parser.add_argument("--out", type=str, default="v_c16a50_tB5082_s20_submit.csv")
    args = parser.parse_args()

    t0 = time.time()
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma)
    test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    print(f"피처 {len(cols)}개 | clip{CLIP_LO}/α{ALPHA}/snapA p{SNAP_A_THR}→{SNAP_A_TO} "
          f"+ tierB(P≥{RATED_CF_B}cap ≥{args.thrb}→{args.tgtb}cap) | seeds={args.seeds}", flush=True)

    max_date = train_df["forecast_kst_dtm"].max()
    bias_data = estimate_bias_data(train_df, cols, max_date, args.trees, args.jobs)
    coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
    print(f"편향 계수 추정 완료 ({time.time() - t0:.0f}s)", flush=True)

    tr_pool = train_df.loc[train_df["target"].notna()]
    clf_A, clf_B = [], []

    predictions_final = pd.DataFrame(index=test_df.index)
    fire_report = {}
    for group in [1, 2, 3]:
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        X_tr, X_te = prepare_matrices(tr_g, te_g, cols)
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

        X_pool_tr, X_pool_te = build_pooled_matrices(tr_pool, te_g, cols)
        if not clf_A:
            caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
            yB = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF_B).astype(int)
            for seed in range(N_CLF_SEEDS):
                for ylab, store in [(yA, clf_A), (yB, clf_B)]:
                    clf = lgb.LGBMClassifier(
                        n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                        num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=args.jobs, verbose=-1,
                    )
                    clf.fit(X_pool_tr, ylab)
                    store.append(clf)
        pA = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in clf_A], axis=0)
        pB = np.mean([c.predict_proba(X_pool_te)[:, 1] for c in clf_B], axis=0)

        pred = apply_bias_correction_v2(pred, te_g["ldaps_ws_10m"].to_numpy(), group, coarse_map)
        pred = apply_rolling_smoothing(pd.Series(pred, index=te_g.index), window=3, alpha=ALPHA).to_numpy().copy()
        maskA = pA >= SNAP_A_THR
        pred[maskA] = np.maximum(pred[maskA], capacity * SNAP_A_TO)
        maskB = pB >= args.thrb
        raisedB = maskB & (pred < capacity * args.tgtb)
        pred[maskB] = np.maximum(pred[maskB], capacity * args.tgtb)
        pred = np.clip(pred, capacity * CLIP_LO, capacity)

        predictions_final.loc[te_g.index, target_col] = pred
        fire_report[group] = (int(maskA.sum()), int(raisedB.sum()))
        print(f"{target_col}: train {len(tr_g)}행 | snapA {int(maskA.sum())} | tierB 상향 {int(raisedB.sum())}/{len(te_g)}시간 ({time.time() - t0:.0f}s)", flush=True)

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

    output_path = DATA_DIR / args.out
    submission.to_csv(output_path, index=False, encoding="utf-8-sig")
    champ_path = DATA_DIR / "v_c16a50_s20_submit.csv"
    if champ_path.exists():
        champ = pd.read_csv(champ_path)
        diff = np.abs(submission[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        print(f"vs 챔피언(v_c16a50_s20): 평균 |차이| {diff.mean():.1f}kWh, 변경 시간 {(diff > 1).sum()}/{diff.size}칸")
    print(f"Saved: {output_path}")
    print(f"총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
