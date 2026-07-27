# ═══════════════════════════════════════════════════════
# exp-024: 그룹3 전이 재시험 — pooled CF base를 JMA 챔피언 위에서 이중 게이트 판정
#
# 배경: exp-011의 pooled CF 전이는 CV +0.0034였으나 LB 악화 주범이던 snap15 설정과
#   묶여 기각됨 (exp-010 변경점 분리: 악화 대부분은 snap15 쪽). 순수 전이 효과는
#   LB-검증된 snap30 설정에서 미판정 상태. 그룹3은 여전히 최약체(FiCR 0.349 vs
#   그룹2 0.440). 챔피언 피처(TPSJgJm)로 이중 게이트 재판정.
# 변형: base 3종 (control / g3pool / g3mix=그룹3만 50:50) × 스냅 3종 (nosnap/snapG/snapP)
#   스냅 설정은 snap30 (p0.3/0.94). g3mix 믹싱은 각 소스 편향보정+rolling 후 (exp-011 동일).
# 검증: 공식 4-fold CV + 1년 홀드아웃 (이중 게이트)
# 실행: C:\miniconda3\python.exe -X utf8 exp024_g3_transfer2.py [--quick] [--jobs N]
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
from exp011_g3_transfer import build_pooled_matrices, estimate_bias_data_pooled
from exp012_feature_v11 import build_v11
from exp019_multigrid import HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94
BASE_SOURCES = ["control", "g3pool", "g3mix"]
SNAP_MODES = ["nosnap", "snapG", "snapP"]


def run_transfer(folds, train_df, cols, args, tag):
    names = [f"{b}_{s}" for b in BASE_SOURCES for s in SNAP_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
        bias_data_p = estimate_bias_data_pooled(train_df, cols, fold["train_end"], args.trees, args.jobs)
        pooled_map = bias_map_from_data(bias_data_p, {g: COARSE_BINS for g in bias_data_p})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        tr_pool = tr_all.loc[tr_all["target"].notna()].copy()
        caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
        tr_pool["cf"] = tr_pool["target"] / caps_pool
        w_pool = np.where(tr_pool["cf"].to_numpy() >= 0.10, 1.0, 0.05)
        y_pool_rated = (tr_pool["cf"].to_numpy() >= RATED_CF).astype(int)

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}

        pooled_lgbm, pooled_xgb, pooled_clf = [], [], []
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

            # (1) 그룹별 base + 그룹별 분류기
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
            blend_group = 0.5 * pred_lgbm + 0.5 * np.mean(preds, axis=0)

            y_rated = (y_tr.to_numpy() >= capacity * RATED_CF).astype(int)
            probas = []
            for seed in range(N_CLF_SEEDS):
                clf = lgb.LGBMClassifier(
                    n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                    num_leaves=31, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, n_jobs=args.jobs, verbose=-1,
                )
                clf.fit(X_tr, y_rated)
                probas.append(clf.predict_proba(X_va)[:, 1])
            p_rated_g = np.mean(probas, axis=0)

            # (2) pooled CF base + pooled 분류기 (첫 그룹에서 한 번만 학습)
            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not pooled_lgbm:
                for seed in range(args.seeds):
                    m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
                    m.fit(X_pool_tr, tr_pool["cf"], sample_weight=w_pool)
                    pooled_lgbm.append(m)
                for seed in range(args.seeds):
                    m = xgbm.XGBRegressor(
                        n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
                    )
                    m.fit(X_pool_tr, tr_pool["cf"], sample_weight=w_pool)
                    pooled_xgb.append(m)
                for seed in range(N_CLF_SEEDS):
                    clf = lgb.LGBMClassifier(
                        n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                        num_leaves=31, min_child_samples=20,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=args.jobs, verbose=-1,
                    )
                    clf.fit(X_pool_tr, y_pool_rated)
                    pooled_clf.append(clf)
            cf_lgbm = np.mean([m.predict(X_pool_va) for m in pooled_lgbm], axis=0)
            cf_xgb = np.mean([m.predict(X_pool_va) for m in pooled_xgb], axis=0)
            blend_pool = (0.5 * cf_lgbm + 0.5 * cf_xgb) * capacity
            p_rated_p = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in pooled_clf], axis=0)

            ws_va = va_g["ldaps_ws_10m"].to_numpy()

            def finalize(pred, bmap):
                p = apply_bias_correction_v2(pred, ws_va, group, bmap)
                return apply_rolling_smoothing(pd.Series(p, index=va_g.index), window=3, alpha=0.7).to_numpy()

            base_group = finalize(blend_group, coarse_map)
            base_pool = finalize(blend_pool, pooled_map)
            base_by_source = {
                "control": base_group,
                "g3pool": base_pool if group == 3 else base_group,
                "g3mix": 0.5 * base_group + 0.5 * base_pool if group == 3 else base_group,
            }
            p_by_mode = {"snapG": p_rated_g, "snapP": p_rated_p}

            actual_va = va_g["target"].to_numpy()
            for b in BASE_SOURCES:
                for s in SNAP_MODES:
                    v = f"{b}_{s}"
                    pv = base_by_source[b].copy()
                    if s != "nosnap":
                        mask = p_by_mode[s] >= SNAP_THR
                        pv[mask] = np.maximum(pv[mask], capacity * SNAP_CF_VAL)
                    pv = np.clip(pv, capacity * 0.10, capacity)
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
        print(f"[{tag}:{fold['name']}] ({time.time() - tf:.0f}s) top4: "
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
    piv = summary.assign(b=summary["variant"].str.rsplit("_", n=1).str[0],
                         s=summary["variant"].str.rsplit("_", n=1).str[1])
    print(piv.pivot(index="b", columns="s", values="pooled_Score").round(4).to_string())


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
    print(f"챔피언 피처 {len(cols)}개, base 3종 × 스냅 3종 (snap30)")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    summary, gs = run_transfer(folds, train_df, cols, args, "CV")
    print("\n================ 4-fold CV (pooled) ================")
    print_block(summary)
    print("\n--- 전체 (내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))
    print("\n--- 그룹3 fold 평균 (1-NMAE / FiCR) ---")
    g3 = gs[gs["group"] == 3].groupby("variant")[["1-NMAE", "FICR"]].mean().round(4)
    print(g3.to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hA, _ = run_transfer([HOLD_FOLD], train_df, cols, args, "HOLD")
        print_block(hA)

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
