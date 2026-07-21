# ═══════════════════════════════════════════════════════
# exp-009: 정격 근처 스냅 + 그룹3 세분화 편향보정 (exp-008 진단의 공격 지점)
#
# exp-008에서 확인된 FiCR 최대 취약점 두 곳을 겨냥:
#   A. 정격 스냅: CF 0.9+ 구간(FiCR 0.237, 0원 밴드 74%)
#      - 그룹·fold별 LGBM 이진 분류기로 P(actual ≥ 0.9cap) 추정
#      - p ≥ 임계값이면 제출값을 max(기존예측, 0.94cap)로 상향 (임계값 0.5/0.6/0.7)
#   B. 그룹3 편향보정 세분화: 그룹3의 7~13m/s 전반 실패(0원 밴드 70%+)
#      - 그룹3만 ws bin [0,11,13,100] → [0,5,7,9,11,13,100]
#   C. A(p0.6) + B 결합
#
# 검증: 공식 4-fold CV (FOLDS_Q), control(exp-003) 동일 조건 비교
# 실행: python -X utf8 exp009_rated_snap.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, SEED, FOLDS_Q,
    build_datasets, metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    calendar_features, prepare_matrices,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COARSE_BINS = [0, 11, 13, 100]
FINE_BINS = [0, 5, 7, 9, 11, 13, 100]
SNAP_CF = 0.94
RATED_CF = 0.90
N_CLF_SEEDS = 3

VARIANTS = ["control", "snap_p30", "snap_p40", "snap_p50", "snap_p60", "g3fine", "combo"]
SNAP_THR = {"snap_p30": 0.3, "snap_p40": 0.4, "snap_p50": 0.5, "snap_p60": 0.6}
COMBO_THR = 0.4


def estimate_bias_data(train_df, weather_cols, tr_end, n_estimators, n_jobs,
                       low_weight=0.05, holdout_ratio=0.25, min_fit_days=90):
    """estimate_bias_correction_adaptive와 동일한 held-out 예측을 만들되,
    bin 집계 전의 원시 (ws, signed_error_rate)를 반환 → 여러 bin 구성에 재사용."""
    out = {}
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g_data = train_df[(train_df["kpx_group"] == group) & (train_df["forecast_kst_dtm"] <= tr_end)]
        g_data = g_data[g_data["target"].notna()]
        if len(g_data) == 0:
            continue
        first_date = g_data["forecast_kst_dtm"].min()
        available_days = max((tr_end - first_date).days, 1)
        holdout_days = min(int(available_days * holdout_ratio), 180)
        if available_days - holdout_days < min_fit_days:
            holdout_days = max(0, available_days - min_fit_days)
        tr_sub_end = tr_end - pd.Timedelta(days=holdout_days)
        fit_g = g_data[g_data["forecast_kst_dtm"] <= tr_sub_end]
        check_g = g_data[g_data["forecast_kst_dtm"] > tr_sub_end]
        if len(fit_g) == 0 or len(check_g) == 0:
            continue

        fit_feat = pd.concat([calendar_features(fit_g["forecast_kst_dtm"]).reset_index(drop=True),
                              fit_g[weather_cols].reset_index(drop=True)], axis=1)
        check_feat = pd.concat([calendar_features(check_g["forecast_kst_dtm"]).reset_index(drop=True),
                                check_g[weather_cols].reset_index(drop=True)], axis=1)
        imputer = SimpleImputer(strategy="median")
        X_fit = pd.DataFrame(imputer.fit_transform(fit_feat), columns=fit_feat.columns)
        X_check = pd.DataFrame(imputer.transform(check_feat), columns=check_feat.columns)
        w_fit = make_sample_weight(fit_g["target"].to_numpy(), capacity, low_weight=low_weight)

        model = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, SEED))
        model.fit(X_fit, fit_g["target"], sample_weight=w_fit)
        pred_check = model.predict(X_check)

        check_g = check_g.copy()
        check_g["signed_error_rate"] = (pred_check - check_g["target"]) / capacity
        check_g = check_g.loc[check_g["target"] >= capacity * 0.10]
        if len(check_g) == 0:
            continue
        out[group] = check_g[["ldaps_ws_10m", "signed_error_rate"]]
    return out


def bias_map_from_data(bias_data, bins_by_group):
    bias_map = {}
    for group, df in bias_data.items():
        bins = bins_by_group[group]
        ws_bin = pd.cut(df["ldaps_ws_10m"], bins=bins)
        bias_map[group] = df.groupby(ws_bin, observed=True)["signed_error_rate"].mean().to_dict()
    return bias_map


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

    fold_scores = {v: [] for v in VARIANTS}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in VARIANTS}
    snap_stats = []

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
        fine_map = bias_map_from_data(bias_data, {1: COARSE_BINS, 2: COARSE_BINS, 3: FINE_BINS})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in VARIANTS}

        for group in [1, 2, 3]:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

            va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
            answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")
            if len(tr_g) == 0:
                for v in VARIANTS:
                    pred_dfs[v][target_col] = 0.0
                continue

            X_tr, X_va = prepare_matrices(tr_g, va_g, weather_cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)

            # ── base 블렌딩 (control과 동일) ──
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
            pred_blend = 0.5 * pred_lgbm + 0.5 * np.mean(preds, axis=0)

            # ── 정격 도달 확률 분류기 ──
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
            p_rated = np.mean(probas, axis=0)

            ws_va = va_g["ldaps_ws_10m"].to_numpy()

            def finalize(pred_corr):
                s = apply_rolling_smoothing(pd.Series(pred_corr, index=va_g.index), window=3, alpha=0.7)
                return s.to_numpy()

            base_coarse = finalize(apply_bias_correction_v2(pred_blend, ws_va, group, coarse_map))
            base_fine = finalize(apply_bias_correction_v2(pred_blend, ws_va, group, fine_map))

            def snap(pred_base, thr):
                mask = p_rated >= thr
                out = pred_base.copy()
                out[mask] = np.maximum(out[mask], capacity * SNAP_CF)
                return out, int(mask.sum())

            variant_preds = {"control": base_coarse, "g3fine": base_fine}
            for name, thr in SNAP_THR.items():
                variant_preds[name], n_snap = snap(base_coarse, thr)
                if name == "snap_p40":
                    actual = va_g["target"].to_numpy()
                    rated_true = actual >= capacity * RATED_CF
                    mask = p_rated >= thr
                    snap_stats.append({
                        "fold": fold["name"], "group": group, "n_snap": n_snap,
                        "n_rated_true": int(rated_true.sum()),
                        "precision": float(rated_true[mask].mean()) if mask.sum() else np.nan,
                        "recall": float(mask[rated_true].mean()) if rated_true.sum() else np.nan,
                    })
            variant_preds["combo"], _ = snap(base_fine, COMBO_THR)

            actual_va = va_g["target"].to_numpy()
            for v in VARIANTS:
                pv = np.clip(variant_preds[v], capacity * 0.10, capacity)
                variant_preds[v] = pv
                va_pred = va_g[["forecast_kst_dtm"]].copy()
                va_pred[target_col] = pv
                pred_dfs[v] = pred_dfs[v].merge(va_pred, on="forecast_kst_dtm", how="left")
                nmae_g, ficr_g, _ = metric_single_group(actual_va, pv, capacity)
                group_scores.append({"fold": fold["name"], "group": group, "variant": v,
                                     "1-NMAE": 1 - nmae_g, "FICR": ficr_g})

        answer_f = answer_df[TARGET_COLS].fillna(0.0)
        all_answer.append(answer_f)
        line = [f"[{fold['name']}] ({time.time() - tf:.0f}s)"]
        for v in VARIANTS:
            pred_f = pred_dfs[v][TARGET_COLS].fillna(0.0)
            all_pred[v].append(pred_f)
            s, n, f = metric(answer_f, pred_f)
            fold_scores[v].append(s)
            line.append(f"{v}: S={s:.4f} N={n:.4f} F={f:.4f}")
        print("\n  ".join(line), flush=True)

    print("\n================ 결과 요약 (공식 4-fold) ================")
    pooled_answer = pd.concat(all_answer, ignore_index=True)
    for v in VARIANTS:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        fm = np.mean(fold_scores[v])
        print(f"{v:9s} | pooled Score={s:.4f} (1-NMAE={n:.4f}, FICR={f:.4f}) | fold평균 {fm:.4f}")

    gs = pd.DataFrame(group_scores)
    print("\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    pv = gs.groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4)
    print(pv.to_string())

    print("\n--- 정격 분류기 품질 (snap_p40 기준, fold×그룹) ---")
    ss = pd.DataFrame(snap_stats)
    print(ss.round(3).to_string(index=False))
    print("\n(precision: 스냅한 시간 중 실제 정격(≥0.9cap) 비율 / recall: 실제 정격 시간 중 스냅한 비율)")

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
