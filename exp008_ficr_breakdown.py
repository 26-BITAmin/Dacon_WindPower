# ═══════════════════════════════════════════════════════
# exp-008 진단: 공식 4-fold CV에서 FiCR이 정확히 어느 지점(조건)에서 떨어지는지 분해
#
# exp-003(현재 LB 최선, control 파이프라인)과 동일한 설정으로 FOLDS_Q 4-fold를 돌리되,
# 예측마다 fold/그룹/풍속/시간대/월 메타데이터를 같이 저장 → pooled 오차를 여러 축으로 분해.
# quantile 모델은 생략(속도) — control 예측만 필요.
#
# 실행: python -X utf8 exp008_ficr_breakdown.py [--quick] [--jobs N]
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
    build_datasets, metric, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    estimate_bias_correction_adaptive, prepare_matrices,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_control_cv_with_meta(train_df, weather_cols, folds, args):
    rows = []
    for fold in folds:
        t0 = time.time()
        bias_estimates = estimate_bias_correction_adaptive(
            train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
        bias_map = {b["group"]: b["bias_by_bin"] for b in bias_estimates}

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        for group in [1, 2, 3]:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            if len(tr_g) == 0 or len(va_g) == 0:
                continue

            X_tr, X_va = prepare_matrices(tr_g, va_g, weather_cols)
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
            pred = 0.5 * pred_lgbm + 0.5 * np.mean(preds, axis=0)

            pred = apply_bias_correction_v2(pred, va_g["ldaps_ws_10m"].to_numpy(), group, bias_map)
            pred = apply_rolling_smoothing(pd.Series(pred, index=va_g.index), window=3, alpha=0.7).to_numpy()
            pred = np.clip(pred, capacity * 0.10, capacity)

            dt = pd.to_datetime(va_g["forecast_kst_dtm"])
            rows.append(pd.DataFrame({
                "fold": fold["name"], "group": group, "capacity": capacity,
                "actual": va_g["target"].to_numpy(), "pred": pred,
                "ws": va_g["ldaps_ws_10m"].to_numpy(),
                "hour": dt.dt.hour.to_numpy(), "month": dt.dt.month.to_numpy(),
                "disagreement": va_g["ws_disagreement"].to_numpy(),
            }))
        print(f"[{fold['name']}] 완료 ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True)


def band(err):
    return np.select([err <= 0.06, err <= 0.08], ["<=6%", "6-8%"], default=">8%")


def summarize(df, by_cols, label):
    """actual 가중 FiCR과 밴드 분포를 by_cols 기준으로 집계"""
    df = df.copy()
    df["price"] = np.select([df["error_rate"] <= 0.06, df["error_rate"] <= 0.08], [4.0, 3.0], default=0.0)
    df["actual_price"] = df["actual"] * df["price"]

    def agg(g):
        ficr = g["actual_price"].sum() / (g["actual"].sum() * 4.0)
        nmae = g["error_rate"].mean()
        band_share = g["band"].value_counts(normalize=True).reindex(["<=6%", "6-8%", ">8%"]).fillna(0)
        return pd.Series({
            "n": len(g), "actual_sum_MWh": g["actual"].sum() / 1000,
            "FiCR": ficr, "1-NMAE": 1 - nmae,
            "share_<=6%": band_share["<=6%"], "share_6-8%": band_share["6-8%"], "share_>8%": band_share[">8%"],
        })

    out = df.groupby(by_cols, observed=True).apply(agg, include_groups=False)
    print(f"\n--- {label} ---")
    print(out.round(4).to_string())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    folds = FOLDS_Q
    if args.quick:
        args.seeds, args.trees = 1, 150
        folds = FOLDS_Q[-1:]
        print("### QUICK MODE ###")

    t0 = time.time()
    train_df, _, weather_cols, _ = build_datasets()
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    raw = run_control_cv_with_meta(train_df, weather_cols, folds, args)
    raw = raw[raw["actual"] >= raw["capacity"] * 0.10].copy()  # 평가 대상만
    raw["error_rate"] = np.abs(raw["pred"] - raw["actual"]) / raw["capacity"]
    raw["signed_error"] = (raw["pred"] - raw["actual"]) / raw["capacity"]
    raw["band"] = band(raw["error_rate"])
    raw["cf"] = raw["actual"] / raw["capacity"]

    # 전체 pooled 점검 (metric()과 일치하는지 확인용)
    ans = pd.DataFrame({"forecast_kst_dtm": range(len(raw))})  # dummy, metric()은 별도 재계산
    overall_ficr = (raw["actual"] * np.select([raw["error_rate"] <= 0.06, raw["error_rate"] <= 0.08], [4.0, 3.0], default=0.0)).sum() / (raw["actual"].sum() * 4.0)
    print(f"\n전체 pooled FiCR(그룹 단순평균 아님, 데이터 전체 가중): {overall_ficr:.4f}  ※ metric()의 그룹평균 FiCR(0.3723)과는 정의가 달라 직접비교 주의")

    summarize(raw, ["fold"], "1. Fold(분기)별")
    summarize(raw, ["group"], "2. 그룹별 (재확인)")
    raw["ws_bin"] = pd.cut(raw["ws"], bins=[0, 3, 5, 7, 9, 11, 13, 15, 100])
    summarize(raw, ["ws_bin"], "3. 풍속 구간별 (0.5m/s→coarse)")
    summarize(raw, ["hour"], "4. 시간대(hour)별")
    summarize(raw, ["month"], "5. 월별")
    raw["cf_bin"] = pd.cut(raw["cf"], bins=[0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
    summarize(raw, ["cf_bin"], "6. 발전량(CF, 이용률) 구간별 — FiCR이 actual가중이므로 핵심")
    raw["disagree_bin"] = pd.qcut(raw["disagreement"], 4, duplicates="drop")
    summarize(raw, ["disagree_bin"], "7. LDAPS-GFS 예보 불일치도 4분위별")

    print("\n--- 8. 그룹×풍속구간 교차 (그룹3 강풍 문제가 CV 전체에서도 보이는지) ---")
    cross = summarize(raw, ["group", "ws_bin"], "그룹×풍속구간")

    print("\n--- 9. FiCR 손실 기여도: (4원 - 실제단가)×actual 합계, 축별 상위 손실 구간 ---")
    raw["lost_revenue"] = raw["actual"] * (4.0 - np.select([raw["error_rate"] <= 0.06, raw["error_rate"] <= 0.08], [4.0, 3.0], default=0.0))
    total_lost = raw["lost_revenue"].sum()
    for col, label in [("group", "그룹"), ("ws_bin", "풍속구간"), ("month", "월"), ("hour", "시간대")]:
        share = raw.groupby(col, observed=True)["lost_revenue"].sum() / total_lost
        print(f"\n  {label}별 손실 기여 비율:")
        print("    " + share.round(3).sort_values(ascending=False).to_string().replace("\n", "\n    "))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
