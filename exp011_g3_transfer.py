# ═══════════════════════════════════════════════════════
# exp-011: 그룹3 전이학습 — 전 그룹 pooled CF 모델 + pooled 정격 분류기
#
# 배경 (exp-007/008/009 진단):
#   - 그룹3 FiCR 0.31로 최약체. 라벨이 1년 적고, 7m/s+ 전반에서 0원 밴드 70%+.
#   - 정격 스냅 분류기의 그룹3 recall ≈ 0 → 스냅 이득이 그룹3만 비껴감.
# 가설:
#   - 타깃을 CF(=발전량/설비용량)로 정규화하면 그룹 간 스케일이 맞으므로,
#     3그룹 데이터를 합쳐 학습(kpx_group을 피처로)하면 그룹1·2의 2년 라벨이
#     그룹3(1년)로 전이되어 그룹3 예측이 개선될 수 있다.
#   - 정격 분류기도 pooled로 학습하면 그룹3 정격 시간 검출력이 생길 수 있다.
# 변형:
#   base 소스: control(그룹별) / g3pool(그룹3만 pooled) / g3mix(그룹3 50:50) / allpool(전 그룹 pooled)
#   snap: none / 그룹별 clf / pooled clf  (스냅 설정은 exp-010 최적값 사용)
# 실행: python -X utf8 exp011_g3_transfer.py [--quick] [--jobs N]
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
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# exp-010 스윕 최적값 (p15_cf92, pooled 0.6362)
BEST_THR = 0.15
BEST_CF = 0.92

BASE_SOURCES = ["control", "g3pool", "g3mix", "allpool"]
SNAP_MODES = ["nosnap", "snapG", "snapP"]  # 없음 / 그룹별 clf / pooled clf


def build_pooled_matrices(tr_pool, va_g, weather_cols):
    """pooled 학습셋(전 그룹)과 한 그룹의 검증셋 → kpx_group 피처 포함 행렬"""
    tr_feat = pd.concat([calendar_features(tr_pool["forecast_kst_dtm"]).reset_index(drop=True),
                         tr_pool[weather_cols].reset_index(drop=True),
                         tr_pool[["kpx_group"]].reset_index(drop=True)], axis=1)
    va_feat = pd.concat([calendar_features(va_g["forecast_kst_dtm"]).reset_index(drop=True),
                         va_g[weather_cols].reset_index(drop=True),
                         va_g[["kpx_group"]].reset_index(drop=True)], axis=1)
    imputer = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imputer.fit_transform(tr_feat), columns=tr_feat.columns)
    X_va = pd.DataFrame(imputer.transform(va_feat), columns=va_feat.columns)
    return X_tr, X_va


def estimate_bias_data_pooled(train_df, weather_cols, tr_end, n_estimators, n_jobs,
                              low_weight=0.05, holdout_ratio=0.25, min_fit_days=90):
    """pooled CF 모델의 held-out signed error — 그룹별 bias map 재사용을 위해 그룹별 반환"""
    data = train_df[(train_df["forecast_kst_dtm"] <= tr_end) & train_df["target"].notna()].copy()
    if len(data) == 0:
        return {}
    caps = data["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    data["cf"] = data["target"] / caps

    first_date = data["forecast_kst_dtm"].min()
    available_days = max((tr_end - first_date).days, 1)
    holdout_days = min(int(available_days * holdout_ratio), 180)
    if available_days - holdout_days < min_fit_days:
        holdout_days = max(0, available_days - min_fit_days)
    tr_sub_end = tr_end - pd.Timedelta(days=holdout_days)
    fit_d = data[data["forecast_kst_dtm"] <= tr_sub_end]
    check_d = data[data["forecast_kst_dtm"] > tr_sub_end]
    if len(fit_d) == 0 or len(check_d) == 0:
        return {}

    X_fit, X_check = build_pooled_matrices(fit_d, check_d, weather_cols)
    w_fit = np.where(fit_d["cf"].to_numpy() >= 0.10, 1.0, low_weight)
    model = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, SEED))
    model.fit(X_fit, fit_d["cf"], sample_weight=w_fit)
    pred_check = model.predict(X_check)

    check_d = check_d.copy()
    check_d["signed_error_rate"] = pred_check - check_d["cf"]  # CF 스케일 = error_rate 스케일
    check_d = check_d.loc[check_d["cf"] >= 0.10]
    out = {}
    for group in [1, 2, 3]:
        sub = check_d[check_d["kpx_group"] == group]
        if len(sub):
            out[group] = sub[["ldaps_ws_10m", "signed_error_rate"]]
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
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)", flush=True)
    print(f"스냅 설정: thr={BEST_THR}, snap_cf={BEST_CF}\n", flush=True)

    names = [f"{b}_{s}" for b in BASE_SOURCES for s in SNAP_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    clf_stats = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        # 그룹별 편향 보정 (exp-009와 동일) + pooled 모델용 편향 보정
        bias_data = estimate_bias_data(train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
        bias_data_p = estimate_bias_data_pooled(train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
        pooled_map = bias_map_from_data(bias_data_p, {g: COARSE_BINS for g in bias_data_p})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        # ── pooled 학습셋 (전 그룹, CF 타깃) ──
        tr_pool = tr_all.loc[tr_all["target"].notna()].copy()
        caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
        tr_pool["cf"] = tr_pool["target"] / caps_pool
        w_pool = np.where(tr_pool["cf"].to_numpy() >= 0.10, 1.0, 0.05)
        y_pool_rated = (tr_pool["cf"].to_numpy() >= RATED_CF).astype(int)

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}

        # pooled 회귀/분류기는 그룹 루프 밖에서 한 번만 학습
        pooled_lgbm, pooled_xgb, pooled_clf = [], [], []
        X_pool = None

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

            # ── (1) 그룹별 base + 그룹별 분류기 (exp-009 동일) ──
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

            # ── (2) pooled base + pooled 분류기 ──
            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, weather_cols)
            if not pooled_lgbm:  # 학습은 첫 그룹에서 한 번만
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

            # ── 후처리: 편향보정(각자 소스의 bias map) + rolling ──
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
                "allpool": base_pool,
            }
            p_by_mode = {"snapG": p_rated_g, "snapP": p_rated_p}

            actual_va = va_g["target"].to_numpy()
            rated_true = actual_va >= capacity * RATED_CF
            for mode, p in p_by_mode.items():
                mask = p >= BEST_THR
                clf_stats.append({
                    "fold": fold["name"], "group": group, "clf": mode,
                    "n_snap": int(mask.sum()), "n_rated_true": int(rated_true.sum()),
                    "precision": float(rated_true[mask].mean()) if mask.sum() else np.nan,
                    "recall": float(mask[rated_true].mean()) if rated_true.sum() else np.nan,
                })

            for b in BASE_SOURCES:
                for s in SNAP_MODES:
                    v = f"{b}_{s}"
                    pv = base_by_source[b].copy()
                    if s != "nosnap":
                        mask = p_by_mode[s] >= BEST_THR
                        pv[mask] = np.maximum(pv[mask], capacity * BEST_CF)
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
            s, n, f = metric(answer_f, pred_f)
            fold_scores[v].append(s)
            fold_line[v] = s
        top = sorted(fold_line.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) control_snapG={fold_line['control_snapG']:.4f} | top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    print("\n================ 결과 요약 (공식 4-fold, pooled Score 내림차순) ================")
    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in names:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]), "fold_std": np.std(fold_scores[v], ddof=1)})
    summary = pd.DataFrame(rows).sort_values("pooled_Score", ascending=False)
    print(summary.round(4).to_string(index=False))

    gs = pd.DataFrame(group_scores)
    print("\n--- 그룹3 fold 평균 (1-NMAE / FiCR), base 소스별 ---")
    g3 = gs[gs["group"] == 3].groupby("variant")[["1-NMAE", "FICR"]].mean().round(4)
    print(g3.to_string())

    print("\n--- 정격 분류기 비교 (그룹별 clf vs pooled clf, fold 합) ---")
    cs = pd.DataFrame(clf_stats)
    agg = cs.groupby(["clf", "group"]).agg(
        n_snap=("n_snap", "sum"), n_rated_true=("n_rated_true", "sum"),
        precision=("precision", "mean"), recall=("recall", "mean")).round(3)
    print(agg.to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
