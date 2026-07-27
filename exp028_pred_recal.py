# ═══════════════════════════════════════════════════════
# exp-028: 절단(truncation) 인지 재보정 — 예측값 기준 재보정 맵
#
# 배경/동기 (exp-027 clip 진단에서 파생):
#   평가는 actual ≥ 0.10·cap 시간만 대상. 모델은 전 시간으로 학습돼 무조건부
#   E[actual|X]를 예측하지만, 지표가 보는 건 절단된 부분모집단이다.
#   → 최적 제출값은 E[actual | X, actual ≥ 0.10cap] 쪽이고, 이는 무조건부 예측보다
#     항상 크며 특히 저예측 구간에서 격차가 크다.
#   exp-027 진단(quick): 0.15cap 미만으로 예측한 평가시간의 실측 평균 CF 0.164.
#   현행 후처리에서 이 효과를 다루는 건 하한 clip(0.10cap) 하나뿐 — 계단 하나짜리
#   조악한 근사다. 현행 편향보정은 ws bin 기준이라 축이 다르다.
#
# 변경점: held-out 구간에서 (예측 CF, 실측 CF | actual ≥ 0.10cap)를 모아
#   **예측값 bin별 가산 보정 맵**을 만들어 적용. ws 기준 보정과의 관계는
#   교체(replace) / 병용(stack) 두 가지로 분리 측정.
#
# 판정: CV(4-fold) + 1년 홀드아웃 이중 게이트. 챔피언은 exp-020 구성.
# 실행: C:\miniconda3\python.exe -X utf8 exp028_pred_recal.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm
from sklearn.impute import SimpleImputer

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, FOLDS_Q, SEED,
    metric, metric_single_group, make_sample_weight, calendar_features,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp019_multigrid import HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94
CLIP_LO = 0.10

# 예측 CF bin 구성 (저구간을 촘촘히 — 절단 효과가 집중되는 곳)
PRED_BINS_COARSE = [-1, 0.15, 0.30, 0.60, 2.0]
PRED_BINS_MED = [-1, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 2.0]

# 변형: (recal 사용 여부, bins, λ, ws보정 병용 여부)
VARIANTS = {
    "champ":        (False, None,             0.0, True),
    "rc_co_l10":    (True,  PRED_BINS_COARSE, 1.0, True),
    "rc_co_l05":    (True,  PRED_BINS_COARSE, 0.5, True),
    "rc_md_l10":    (True,  PRED_BINS_MED,    1.0, True),
    "rc_md_l05":    (True,  PRED_BINS_MED,    0.5, True),
    "rc_md_l10_only": (True, PRED_BINS_MED,   1.0, False),   # ws보정 제거 (교체)
    "rc_co_l10_only": (True, PRED_BINS_COARSE, 1.0, False),
}
NAMES = list(VARIANTS)


def estimate_recal_data(train_df, weather_cols, tr_end, n_estimators, n_jobs,
                        low_weight=0.05, holdout_ratio=0.25, min_fit_days=90):
    """estimate_bias_data와 동일한 held-out 분할·모델이되,
    ws뿐 아니라 **예측 CF**와 실측 CF도 함께 반환 (평가 대상 시간만)."""
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
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_fit = pd.DataFrame(imputer.fit_transform(fit_feat), columns=fit_feat.columns)
        X_check = pd.DataFrame(imputer.transform(check_feat), columns=check_feat.columns)
        w_fit = make_sample_weight(fit_g["target"].to_numpy(), capacity, low_weight=low_weight)

        model = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, SEED))
        model.fit(X_fit, fit_g["target"], sample_weight=w_fit)
        pred_check = model.predict(X_check)

        check_g = check_g.copy()
        check_g["pred_cf"] = pred_check / capacity
        check_g["actual_cf"] = check_g["target"] / capacity
        check_g["signed_error_rate"] = (pred_check - check_g["target"]) / capacity
        check_g = check_g.loc[check_g["target"] >= capacity * 0.10]   # 평가 대상 시간만
        if len(check_g) == 0:
            continue
        out[group] = check_g[["ldaps_ws_10m", "pred_cf", "actual_cf", "signed_error_rate"]]
    return out


def recal_map_from_data(recal_data, bins, ws_map=None):
    """예측 CF bin별 가산 보정량(= mean(pred_cf − actual_cf))을 추정.
    ws_map이 주어지면 ws보정을 먼저 적용한 뒤의 잔차 기준으로 추정 (병용 시 이중보정 방지)."""
    out = {}
    for group, df in recal_data.items():
        p = df["pred_cf"].to_numpy().copy()
        if ws_map is not None:
            ws = df["ldaps_ws_10m"].to_numpy()
            p = apply_bias_correction_v2(p, ws, group, ws_map)
        resid = p - df["actual_cf"].to_numpy()          # +면 과대예측
        pb = pd.cut(pd.Series(p), bins=bins)
        out[group] = pd.Series(resid).groupby(pb, observed=True).mean().to_dict()
    return out


def apply_recal(pred_cf, group, recal_map, lam):
    if group not in recal_map:
        return pred_cf
    out = pred_cf.copy()
    for interval, delta in recal_map[group].items():
        if not np.isfinite(delta):
            continue
        m = (pred_cf > interval.left) & (pred_cf <= interval.right)
        out[m] = pred_cf[m] - lam * delta
    return out


def run_sweep(folds, train_df, cols, args, tag):
    fold_scores = {v: [] for v in NAMES}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in NAMES}
    map_dump = []

    for fold in folds:
        tf = time.time()
        recal_data = estimate_recal_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        ws_map = bias_map_from_data({g: d[["ldaps_ws_10m", "signed_error_rate"]] for g, d in recal_data.items()},
                                    {g: COARSE_BINS for g in recal_data})
        # ws보정 병용/미병용 각각에 맞는 재보정 맵
        rmaps = {}
        for bname, bins in [("co", PRED_BINS_COARSE), ("md", PRED_BINS_MED)]:
            rmaps[(bname, True)] = recal_map_from_data(recal_data, bins, ws_map=ws_map)
            rmaps[(bname, False)] = recal_map_from_data(recal_data, bins, ws_map=None)
        for g, m in rmaps[("md", True)].items():
            for iv, d in m.items():
                map_dump.append({"fold": fold["name"], "group": g, "pred_cf_bin": str(iv), "delta_cf": d})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in NAMES}

        pooled_clf = []
        for group in [1, 2, 3]:
            target_col = f"kpx_group_{group}"
            capacity = CAPACITY_KWH[target_col]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

            va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
            answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")
            if len(tr_g) == 0:
                for v in NAMES:
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
            blend = 0.5 * pred_lgbm + 0.5 * pred_xgb

            for vname, (use_rc, bins, lam, use_ws) in VARIANTS.items():
                pv = blend.copy()
                if use_ws:
                    pv = apply_bias_correction_v2(pv, ws_va, group, ws_map)
                if use_rc:
                    bname = "co" if bins is PRED_BINS_COARSE else "md"
                    rmap = rmaps[(bname, use_ws)]
                    pv = apply_recal(pv / capacity, group, rmap, lam) * capacity
                pv = apply_rolling_smoothing(pd.Series(pv, index=va_g.index), 3, 0.7).to_numpy()
                pv = np.array(pv, dtype=float)
                pv[snap_mask] = np.maximum(pv[snap_mask], capacity * SNAP_CF_VAL)
                pv = np.clip(pv, capacity * CLIP_LO, capacity)

                va_pred = va_g[["forecast_kst_dtm"]].copy()
                va_pred[target_col] = pv
                pred_dfs[vname] = pred_dfs[vname].merge(va_pred, on="forecast_kst_dtm", how="left")
                nmae_g, ficr_g, _ = metric_single_group(actual_va, pv, capacity)
                group_scores.append({"fold": fold["name"], "group": group, "variant": vname,
                                     "1-NMAE": 1 - nmae_g, "FICR": ficr_g})

        answer_f = answer_df[TARGET_COLS].fillna(0.0)
        all_answer.append(answer_f)
        fold_line = {}
        for v in NAMES:
            pred_f = pred_dfs[v][TARGET_COLS].fillna(0.0)
            all_pred[v].append(pred_f)
            s, _, _ = metric(answer_f, pred_f)
            fold_scores[v].append(s)
            fold_line[v] = s
        top = sorted(fold_line.items(), key=lambda kv: kv[1], reverse=True)[:4]
        print(f"[{tag}:{fold['name']}] ({time.time() - tf:.0f}s) champ={fold_line['champ']:.4f} | top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in NAMES:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]),
                     "fold_std": np.std(fold_scores[v], ddof=1) if len(folds) > 1 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(group_scores), pd.DataFrame(map_dump)


def show(summary, tag):
    base = float(summary.loc[summary["variant"] == "champ", "pooled_Score"].iloc[0])
    out = summary.copy()
    out["Δchamp"] = out["pooled_Score"] - base
    print(f"\n--- {tag} (champ={base:.4f}, 내림차순) ---")
    print(out.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))
    return out.set_index("variant")["Δchamp"]


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
    print(f"챔피언 피처 {len(cols)}개, 재보정 변형 {len(NAMES)}종")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    cv, gs, md = run_sweep(folds, train_df, cols, args, "CV")
    d_cv = show(cv, "4-fold CV (pooled)")

    print("\n--- 추정된 재보정량 (md bins, ws보정 병용 기준, fold 평균 delta_cf; +면 과대예측 → 하향) ---")
    print(md.groupby(["group", "pred_cf_bin"])["delta_cf"].mean().round(4).unstack(0).to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hd, _, _ = run_sweep([HOLD_FOLD], train_df, cols, args, "HOLD")
        d_h = show(hd, "홀드아웃 2024 전체")

        print("\n================ 이중 게이트 판정 (Δchamp) ================")
        gate = pd.DataFrame({"ΔCV": d_cv, "Δholdout": d_h}).drop(index="champ")
        gate["both_pos"] = (gate["ΔCV"] > 0) & (gate["Δholdout"] > 0)
        print(gate.sort_values("ΔCV", ascending=False).round(4).to_string())
        passed = gate[gate["both_pos"]]
        print("\n>>> 양 게이트 통과: " + (", ".join(passed.index) if len(passed) else "없음"))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
