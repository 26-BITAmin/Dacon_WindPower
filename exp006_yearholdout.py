# ═══════════════════════════════════════════════════════
# exp-006 진단: 1년 홀드아웃 (train ≤2023 → 2024년 전체 검증)
#
# 목적: exp-005(분위수 최적화)가 CV(다음 분기 예측)에서는 이겼지만 LB(1년 앞 예측)에서
#       진 원인을 좁히기 위한 진단. 실제 제출과 같은 "모델 고정 + 1년 예측" 설정에서
#       - 그룹별(1/2/3) 성능             → "그룹2가 약하다" 기억 검증
#       - 모델별(LGBM/XGB/블렌딩) 성능    → "두 번째 모델이 약하다" 해석 검증
#       - 분기별(Q1~Q4) control↔qopt 격차 → "학습 시점에서 멀어지면 캘리브레이션 붕괴" 가설 검증
#
# 실행: python -X utf8 exp006_yearholdout.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, SEED, QUANTILES,
    build_datasets, metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    estimate_bias_correction_adaptive, optimize_submission, prepare_matrices,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_END = pd.Timestamp("2024-01-01 00:00:00")
VALID_START = pd.Timestamp("2024-01-01 01:00:00")
VALID_END = pd.Timestamp("2025-01-01 00:00:00")

VARIANTS = ["lgbm_only", "xgb_only", "control", "qopt_s07", "qopt_s10", "qopt_s15"]
SPREADS = {"qopt_s07": 0.7, "qopt_s10": 1.0, "qopt_s15": 1.5}


def post_control(pred, va_g, group, bias_map, capacity):
    pred = apply_bias_correction_v2(pred, va_g["ldaps_ws_10m"].to_numpy(), group, bias_map)
    pred_series = pd.Series(pred, index=va_g.index)
    pred_series = apply_rolling_smoothing(pred_series, window=3, alpha=0.7)
    return pred_series.to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.trees = 1, 150
        print("### QUICK MODE ###")

    t0 = time.time()
    train_df, _, weather_cols, _ = build_datasets()
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n")

    bias_estimates = estimate_bias_correction_adaptive(
        train_df, weather_cols, TRAIN_END, args.trees, args.jobs)
    bias_map = {b["group"]: b["bias_by_bin"] for b in bias_estimates}

    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= TRAIN_END].reset_index(drop=True)
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= VALID_START)
                          & (train_df["forecast_kst_dtm"] <= VALID_END)].reset_index(drop=True)

    va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
    answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
    pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in VARIANTS}
    group_rows = []

    for group in [1, 2, 3]:
        tg = time.time()
        target_col = f"kpx_group_{group}"
        capacity = CAPACITY_KWH[target_col]
        tr_g = tr_all.loc[tr_all["kpx_group"] == group]
        tr_g = tr_g.loc[tr_g["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

        va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
        answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")

        X_tr, X_va = prepare_matrices(tr_g, va_g, weather_cols)
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
        ybar = float(y_tr[y_tr >= capacity * 0.10].mean())

        lgbm_preds = []
        for seed in range(args.seeds):
            m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            lgbm_preds.append(m.predict(X_va))
        pred_lgbm = np.mean(lgbm_preds, axis=0)

        xgb_preds = []
        for seed in range(args.seeds):
            m = xgbm.XGBRegressor(
                n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            xgb_preds.append(m.predict(X_va))
        pred_xgb = np.mean(xgb_preds, axis=0)
        pred_blend = 0.5 * pred_lgbm + 0.5 * pred_xgb

        q_preds = []
        for q in QUANTILES:
            m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, SEED, objective="quantile", alpha=q))
            m.fit(X_tr, y_tr, sample_weight=w_tr)
            q_preds.append(m.predict(X_va))
        quantile_preds = np.column_stack(q_preds)

        variant_preds = {}
        for name, raw in [("lgbm_only", pred_lgbm), ("xgb_only", pred_xgb), ("control", pred_blend)]:
            variant_preds[name] = np.clip(post_control(raw, va_g, group, bias_map, capacity),
                                          capacity * 0.10, capacity)
        pred_base = post_control(pred_blend, va_g, group, bias_map, capacity)
        for name, scale in SPREADS.items():
            variant_preds[name] = optimize_submission(pred_base, quantile_preds, capacity, ybar,
                                                      spread_scale=scale)

        actual_va = va_g["target"].to_numpy()
        for v in VARIANTS:
            va_pred = va_g[["forecast_kst_dtm"]].copy()
            va_pred[target_col] = variant_preds[v]
            pred_dfs[v] = pred_dfs[v].merge(va_pred, on="forecast_kst_dtm", how="left")
            nmae_g, ficr_g, valid_n = metric_single_group(actual_va, variant_preds[v], capacity)
            group_rows.append({"group": group, "variant": v, "1-NMAE": 1 - nmae_g,
                               "FICR": ficr_g, "score_g": 0.5 * (1 - nmae_g) + 0.5 * ficr_g,
                               "valid_n": valid_n})
        print(f"그룹{group} 완료 ({time.time() - tg:.0f}s, train {len(tr_g)}행 / valid {len(va_g)}행)", flush=True)

    quarters = pd.to_datetime(answer_df["forecast_kst_dtm"]).dt.quarter

    print("\n================ 1년 홀드아웃 (train ≤2023 → 2024 전체) ================")
    print("주의: 그룹3은 학습 데이터가 2023년 1년뿐 (실제 제출 학습셋보다 1년 적음)\n")

    print("--- 전체 (2024년 1년, 3그룹 pooled) ---")
    answer_f = answer_df[TARGET_COLS].fillna(0.0)
    for v in VARIANTS:
        pred_f = pred_dfs[v][TARGET_COLS].fillna(0.0)
        s, n, f = metric(answer_f, pred_f)
        print(f"{v:10s} | Score={s:.4f} | 1-NMAE={n:.4f} | FICR={f:.4f}")

    print("\n--- 그룹별 (변수 = 그룹별 0.5·(1-NMAE)+0.5·FiCR) ---")
    gr = pd.DataFrame(group_rows)
    pivot = gr.pivot(index="variant", columns="group", values="score_g").loc[VARIANTS].round(4)
    print(pivot.to_string())
    print("\n   1-NMAE:")
    print(gr.pivot(index="variant", columns="group", values="1-NMAE").loc[VARIANTS].round(4).to_string())
    print("\n   FiCR:")
    print(gr.pivot(index="variant", columns="group", values="FICR").loc[VARIANTS].round(4).to_string())

    print("\n--- 분기별 Score (3그룹 pooled, 모델은 2023년까지로 고정) ---")
    rows = []
    for q in [1, 2, 3, 4]:
        qmask = (quarters == q).to_numpy()
        row = {"quarter": f"2024Q{q}"}
        for v in VARIANTS:
            s, n, f = metric(answer_f.loc[qmask], pred_dfs[v][TARGET_COLS].fillna(0.0).loc[qmask])
            row[v] = round(s, 4)
        rows.append(row)
    qdf = pd.DataFrame(rows).set_index("quarter")
    qdf["qopt_s10-control"] = (qdf["qopt_s10"] - qdf["control"]).round(4)
    print(qdf.to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
