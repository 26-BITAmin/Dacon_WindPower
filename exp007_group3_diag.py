# ═══════════════════════════════════════════════════════
# exp-007 진단: 그룹3 예측 성능이 왜/얼마나 떨어지는지 분해
#
# A. 2024 홀드아웃(현행 control 파이프라인)에서 그룹별 오차 분해
#    - FiCR 밴드 적중률(≤6% / 6~8% / >8%), 풍속 구간·월별 signed error
# B. 학습량 통제: 그룹1·2를 그룹3과 동일하게 2023년 1년치로만 학습했을 때 하락폭
#    → 그룹3 열세가 "라벨 1년 부족" 때문인지 분리
# C. 모델 무관 데이터 비교: 예보 풍속↔발전량 관계의 조건부 산포 (2023~2024 공통 구간)
#
# 실행: python -X utf8 exp007_group3_diag.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, SEED,
    build_datasets, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    estimate_bias_correction_adaptive, prepare_matrices,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_END = pd.Timestamp("2024-01-01 00:00:00")
VALID_START = pd.Timestamp("2024-01-01 01:00:00")
VALID_END = pd.Timestamp("2025-01-01 00:00:00")
YEAR_2023 = pd.Timestamp("2023-01-01 01:00:00")


def run_control_holdout(train_df, weather_cols, args, groups, train_start=None):
    """control 파이프라인(블렌딩+편향보정+rolling+clip)으로 2024년 홀드아웃 예측.
    train_start를 주면 학습 구간 시작을 자름 (학습량 통제 실험용)."""
    bias_src = train_df
    if train_start is not None:
        bias_src = train_df.loc[train_df["forecast_kst_dtm"] >= train_start]
    bias_estimates = estimate_bias_correction_adaptive(
        bias_src, weather_cols, TRAIN_END, args.trees, args.jobs)
    bias_map = {b["group"]: b["bias_by_bin"] for b in bias_estimates}

    out = {}
    for group in groups:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = train_df.loc[(train_df["kpx_group"] == group)
                            & (train_df["forecast_kst_dtm"] <= TRAIN_END)]
        if train_start is not None:
            tr_g = tr_g.loc[tr_g["forecast_kst_dtm"] >= train_start]
        tr_g = tr_g.loc[tr_g["target"].notna()].copy()
        va_g = train_df.loc[(train_df["kpx_group"] == group)
                            & (train_df["forecast_kst_dtm"] >= VALID_START)
                            & (train_df["forecast_kst_dtm"] <= VALID_END)].sort_values("forecast_kst_dtm").copy()

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
        out[group] = (pred, va_g)
    return out


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
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    # ── A. 표준 학습 홀드아웃 + 오차 분해 ──
    full = run_control_holdout(train_df, weather_cols, args, groups=[1, 2, 3])
    print(f"[A] 표준 학습 완료 ({time.time() - t0:.0f}s)", flush=True)

    print("\n=========== A. 2024 홀드아웃 그룹별 오차 분해 (control 파이프라인) ===========")
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        pred, va_g = full[group]
        actual = va_g["target"].to_numpy()
        nmae, ficr, valid_n = metric_single_group(actual, pred, capacity)
        score_g = 0.5 * (1 - nmae) + 0.5 * ficr

        valid = actual >= capacity * 0.10
        err = np.abs(pred[valid] - actual[valid]) / capacity
        signed = (pred[valid] - actual[valid]) / capacity
        a_v = actual[valid]
        b6, b8 = err <= 0.06, (err > 0.06) & (err <= 0.08)
        print(f"\n--- 그룹{group} | score_g={score_g:.4f} | 1-NMAE={1-nmae:.4f} | FiCR={ficr:.4f} | 평가시간 {valid_n} ({valid.mean()*100:.0f}%) ---")
        print(f"  밴드 적중(시간 비율):    ≤6% {b6.mean()*100:5.1f}% | 6~8% {b8.mean()*100:4.1f}% | >8% {(~(b6|b8)).mean()*100:5.1f}%")
        w = a_v / a_v.sum()
        print(f"  밴드 적중(actual 가중):  ≤6% {(w*b6).sum()*100:5.1f}% | 6~8% {(w*b8).sum()*100:4.1f}% | >8% {(w*~(b6|b8)).sum()*100:5.1f}%")
        near = ((err > 0.06) & (err <= 0.10)).mean()
        print(f"  경계 근처(6~10%) 시간 비율: {near*100:.1f}%  | 평가시간 error 중앙값 {np.median(err)*100:.1f}%")

        ws = va_g["ldaps_ws_10m"].to_numpy()[valid]
        dfb = pd.DataFrame({"ws_bin": pd.cut(ws, [0, 5, 8, 11, 13, 100]),
                            "signed": signed, "abs": err, "n": 1})
        agg = dfb.groupby("ws_bin", observed=True).agg(n=("n", "size"),
                                                       signed_err=("signed", "mean"),
                                                       abs_err=("abs", "mean")).round(3)
        print("  풍속 구간별 (signed +면 과대예측):")
        print(agg.to_string().replace("\n", "\n    "))

        mon = pd.to_datetime(va_g["forecast_kst_dtm"]).dt.month.to_numpy()[valid]
        dfm = pd.DataFrame({"month": mon, "signed": signed, "abs": err})
        aggm = dfm.groupby("month").agg(abs_err=("abs", "mean"), signed_err=("signed", "mean")).round(3)
        print("  월별 abs/signed error:")
        print("    " + aggm.T.to_string().replace("\n", "\n    "))

    # ── B. 학습량 통제: 그룹1·2를 2023년 1년치로만 학습 ──
    print(f"\n[B] 1년 학습 통제 실행 중... ({time.time() - t0:.0f}s)", flush=True)
    lim = run_control_holdout(train_df, weather_cols, args, groups=[1, 2], train_start=YEAR_2023)
    print("\n=========== B. 학습량 통제 (전부 2023년 1년치 학습 → 2024 검증) ===========")
    print("그룹3은 원래 1년치뿐이므로 A와 동일값")
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        pred_f, va_g = full[group]
        nm_f, fi_f, _ = metric_single_group(va_g["target"].to_numpy(), pred_f, capacity)
        s_f = 0.5 * (1 - nm_f) + 0.5 * fi_f
        if group in lim:
            pred_l, va_l = lim[group]
            nm_l, fi_l, _ = metric_single_group(va_l["target"].to_numpy(), pred_l, capacity)
            s_l = 0.5 * (1 - nm_l) + 0.5 * fi_l
            print(f"그룹{group}: 2년학습 {s_f:.4f} → 1년학습 {s_l:.4f} (Δ {s_l-s_f:+.4f}) | 1년학습 1-NMAE {1-nm_l:.4f} FiCR {fi_l:.4f}")
        else:
            print(f"그룹{group}: 1년학습 {s_f:.4f} (원래 1년) | 1-NMAE {1-nm_f:.4f} FiCR {fi_f:.4f}")

    # ── C. 모델 무관: 예보 풍속↔발전량 조건부 산포 (2023~2024 공통 라벨 구간) ──
    print("\n=========== C. 예보 풍속↔발전량 관계 산포 (2023~2024, 모델 무관) ===========")
    common = train_df.loc[(train_df["forecast_kst_dtm"] >= YEAR_2023) & train_df["target"].notna()].copy()
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = common.loc[common["kpx_group"] == group, ["forecast_kst_dtm", "target", "ldaps_ws_10m"]].dropna().copy()
        cf = g["target"] / capacity
        g["ws_bin"] = pd.cut(g["ldaps_ws_10m"], bins=np.arange(0, 26, 0.5))
        binned_mean = g.groupby("ws_bin", observed=True)["target"].transform("mean")
        resid_mae = float(np.mean(np.abs(g["target"] - binned_mean)) / capacity)
        valid_share = float((g["target"] >= capacity * 0.10).mean())
        # 평가 대상 시간만으로 같은 지표
        gv = g.loc[g["target"] >= capacity * 0.10].copy()
        binned_mean_v = gv.groupby("ws_bin", observed=True)["target"].transform("mean")
        resid_mae_v = float(np.mean(np.abs(gv["target"] - binned_mean_v)) / capacity)
        print(f"그룹{group}: 평균CF {cf.mean():.3f} | ≥10%cap 시간 {valid_share*100:.1f}% | "
              f"파워커브 잔차 MAE/cap 전체 {resid_mae:.4f} / 평가시간만 {resid_mae_v:.4f} | n={len(g)}")

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
