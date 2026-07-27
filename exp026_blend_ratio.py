# ═══════════════════════════════════════════════════════
# exp-026: LGBM:XGB 블렌딩 비율 스윕 (챔피언 기준, 이중 게이트)
#
# 배경: 현행 50:50은 exp-003/004(v10, JMA 이전)에서 정한 값. exp-004는 70:30
#   전역 하나만 시험했고, 챔피언 피처셋·그룹별 차등 비율은 미검증.
#   base 학습은 fold당 1회, 비율은 캐시 재사용 → 저비용.
# 변형: w(LGBM 가중) ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × {nosnap, snapP}
#   그룹별 선호는 group_scores 분해로 관찰 (그룹별 차등 변형은 양 게이트에서
#   일관 패턴이 보일 때만 후속 구성 — CV 사후선택 방지)
# 실행: C:\miniconda3\python.exe -X utf8 exp026_blend_ratio.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, FOLDS_Q,
    metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp019_multigrid import HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94
RATIOS = {"w30": 0.3, "w40": 0.4, "w50": 0.5, "w60": 0.6, "w70": 0.7}
POST_MODES = ["nosnap", "snapP"]


def run_ratio_sweep(folds, train_df, cols, args, tag):
    names = [f"{r}_{pm}" for r in RATIOS for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}

        pooled_clf = []
        for group in [1, 2, 3]:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

            va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
            answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")
            if len(tr_g) == 0:
                for v in names:
                    pred_dfs[v][target_col] = 0.0
                continue

            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)

            preds = []
            for seed in range(args.seeds):
                m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            pred_lgbm = np.mean(preds, axis=0)
            preds = []
            for seed in range(args.seeds):
                m = xgbm.XGBRegressor(
                    n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
                )
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            pred_xgb = np.mean(preds, axis=0)

            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
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
            p_rated = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in pooled_clf], axis=0)
            snap_mask = p_rated >= SNAP_THR

            ws_va = va_g["ldaps_ws_10m"].to_numpy()
            actual_va = va_g["target"].to_numpy()
            for rname, w in RATIOS.items():
                blend = w * pred_lgbm + (1 - w) * pred_xgb
                base = apply_bias_correction_v2(blend, ws_va, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()
                for pm in POST_MODES:
                    pv = base.copy()
                    if pm == "snapP":
                        pv[snap_mask] = np.maximum(pv[snap_mask], capacity * SNAP_CF_VAL)
                    pv = np.clip(pv, capacity * 0.10, capacity)
                    v = f"{rname}_{pm}"
                    va_pred = va_g[["forecast_kst_dtm"]].copy()
                    va_pred[target_col] = pv
                    pred_dfs[v] = pred_dfs[v].merge(va_pred, on="forecast_kst_dtm", how="left")
                    nmae_g, ficr_g, _ = metric_single_group(actual_va, pv, capacity)
                    group_scores.append({"fold": fold["name"], "group": group, "variant": v,
                                         "1-NMAE": 1 - nmae_g, "FICR": ficr_g})

        answer_f = answer_df[TARGET_COLS].fillna(0.0)
        all_answer.append(answer_f)
        fold_line = {}
        for v in names:
            pred_f = pred_dfs[v][TARGET_COLS].fillna(0.0)
            all_pred[v].append(pred_f)
            s, _, _ = metric(answer_f, pred_f)
            fold_scores[v].append(s)
            fold_line[v] = s
        top = sorted(fold_line.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"[{tag}:{fold['name']}] ({time.time() - tf:.0f}s) w50_snapP={fold_line['w50_snapP']:.4f} | top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in names:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]),
                     "fold_std": np.std(fold_scores[v], ddof=1) if len(folds) > 1 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(group_scores)


def print_block(summary):
    piv = summary.assign(r=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print(piv.pivot(index="r", columns="pm", values="pooled_Score").round(4).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    folds = FOLDS_Q
    if args.quick:
        args.seeds, args.trees = 1, 150
        folds = FOLDS_Q[-1:]
        print("### QUICK MODE ###")

    t0 = time.time()
    train_df, _, col_sets, _ = build_v11()
    jma = build_jma_frame()
    train_df = train_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["jgsm_dis"] = (train_df["jgsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df["jmsm_dis"] = (train_df["jmsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    print(f"챔피언 피처 {len(cols)}개, 비율 {list(RATIOS.values())} × {POST_MODES}")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    summary, gs = run_ratio_sweep(folds, train_df, cols, args, "CV")
    print("\n================ 4-fold CV (pooled) ================")
    print_block(summary)
    print("\n--- 전체 (내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))
    print("\n--- 그룹×비율 (snapP, fold 평균 Score = 0.5·(1-NMAE)+0.5·FiCR) ---")
    sel = gs[gs["variant"].str.endswith("_snapP")].copy()
    sel["score_g"] = 0.5 * sel["1-NMAE"] + 0.5 * sel["FICR"]
    print(sel.groupby(["group", "variant"])["score_g"].mean().round(4).unstack().to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hA, hgs = run_ratio_sweep([HOLD_FOLD], train_df, cols, args, "HOLD")
        print_block(hA)
        print("\n--- 그룹×비율 (snapP, holdout) ---")
        hsel = hgs[hgs["variant"].str.endswith("_snapP")].copy()
        hsel["score_g"] = 0.5 * hsel["1-NMAE"] + 0.5 * hsel["FICR"]
        print(hsel.groupby(["group", "variant"])["score_g"].mean().round(4).unstack().to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
