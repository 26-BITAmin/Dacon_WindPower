# ═══════════════════════════════════════════════════════
# exp-016: 발전량 비례 sample weight (COMPETITION.md 전략 3) + pooled 분류기 snap30 결합
#
# 배경 (MODEL.md):
#   - FiCR은 actual 가중 → 고발전 시간의 정확도가 지표에 곱으로 들어감.
#     현행 가중은 10%cap 이분법(0.05/1.0)뿐 — 발전량 비례 가중은 미시도 (전략 3).
#   - exp-008: CF 0.9+ 구간이 FiCR 최악(0.237, 0원 밴드 74%) → 고CF 상향 가중 후보.
#   - exp-011: pooled 정격 분류기는 snap15 설정에서 CV +0.0018 (control_snapP).
#     LB 악화(-0.0040)는 스냅 설정(p15/0.92) 쪽이 주범으로 분리됨(exp-010 관찰)
#     → LB 검증된 snap30(p0.3/0.94)과 pooled 분류기의 결합은 미시도.
#
# 변형: 가중 5종 × 후처리 3종 = 15
#   가중(회귀만, 분류기·편향보정 홀드아웃 모델은 각 가중 동일 적용):
#     wbin  = 현행 (10%cap 미만 0.05, 이상 1.0) ← control
#     wlin1 = 이상 구간 1 + cf          (cf = target/capacity)
#     wlin2 = 이상 구간 1 + 2·cf
#     wsq   = 이상 구간 1 + 3·cf²       (정격 근처 집중)
#     wcf   = max(cf, 0.05)             (순수 비례 — FiCR 정렬, NMAE 리스크)
#   후처리:
#     nosnap / snap30G(그룹별 분류기, p0.3·0.94) / snap30P(pooled 분류기, p0.3·0.94)
#
# 고정: v10TPS 피처(122), LGBM+XGB 50:50, 3-seed(exp-013: 3≈5 seed),
#       coarse 편향보정(가중별 재추정) + rolling(3, 0.7), 공식 4-fold CV
# 벤치마크: v10TPS_snap30 3-seed 0.6318 (exp-013 control)
# 실행: C:\miniconda3\python.exe -X utf8 exp016_weight_scheme.py [--quick] [--jobs N]
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
    metric, metric_single_group, calendar_features,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94  # exp-009 LB 검증 설정
POST_MODES = ["nosnap", "snap30G", "snap30P"]


# ───────────────────────────────────────────────
# 가중 스킴 (y: 발전량 kWh, cap: 설비용량 kWh)
# ───────────────────────────────────────────────
def _cf(y, cap):
    return np.clip(y / cap, 0.0, 1.0)


WEIGHT_SCHEMES = {
    "wbin": lambda y, cap: np.where(y >= 0.10 * cap, 1.0, 0.05),
    "wlin1": lambda y, cap: np.where(y >= 0.10 * cap, 1.0 + _cf(y, cap), 0.05),
    "wlin2": lambda y, cap: np.where(y >= 0.10 * cap, 1.0 + 2.0 * _cf(y, cap), 0.05),
    "wsq": lambda y, cap: np.where(y >= 0.10 * cap, 1.0 + 3.0 * _cf(y, cap) ** 2, 0.05),
    "wcf": lambda y, cap: np.maximum(_cf(y, cap), 0.05),
}


# ───────────────────────────────────────────────
# 편향보정: 가중 스킴별 held-out signed error (exp-009 방식, 행렬 준비 공유)
# ───────────────────────────────────────────────
def estimate_bias_data_by_scheme(train_df, weather_cols, tr_end, n_estimators, n_jobs,
                                 holdout_ratio=0.25, min_fit_days=90):
    out = {ws: {} for ws in WEIGHT_SCHEMES}
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
        y_fit = fit_g["target"].to_numpy()

        eval_mask = check_g["target"].to_numpy() >= capacity * 0.10
        for ws, fn in WEIGHT_SCHEMES.items():
            model = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, SEED))
            model.fit(X_fit, fit_g["target"], sample_weight=fn(y_fit, capacity))
            pred_check = model.predict(X_check)
            df = pd.DataFrame({
                "ldaps_ws_10m": check_g["ldaps_ws_10m"].to_numpy(),
                "signed_error_rate": (pred_check - check_g["target"].to_numpy()) / capacity,
            })[eval_mask]
            if len(df):
                out[ws][group] = df
    return out


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
    cols = col_sets["v10TPS"]
    print(f"v10TPS: {len(cols)}개 피처, 가중 {len(WEIGHT_SCHEMES)}종 × 후처리 {len(POST_MODES)}종")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    names = [f"{ws}_{pm}" for ws in WEIGHT_SCHEMES for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    snap_stats = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]

        bias_data = estimate_bias_data_by_scheme(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_maps = {ws: bias_map_from_data(bias_data[ws], {g: COARSE_BINS for g in bias_data[ws]})
                       for ws in WEIGHT_SCHEMES}

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
            y_np = y_tr.to_numpy()

            # ── 정격 분류기 (가중 스킴 간 공유) ──
            # 그룹별
            y_rated = (y_np >= capacity * RATED_CF).astype(int)
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
            p_rated_G = np.mean(probas, axis=0)

            # pooled (kpx_group 피처 포함, 학습은 첫 그룹에서 한 번만)
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
            p_rated_P = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in pooled_clf], axis=0)

            actual_va = va_g["target"].to_numpy()
            rated_va = actual_va >= capacity * RATED_CF
            for pm, p_rated in [("snap30G", p_rated_G), ("snap30P", p_rated_P)]:
                fire = p_rated >= SNAP_THR
                snap_stats.append({
                    "fold": fold["name"], "group": group, "clf": pm,
                    "n_fire": int(fire.sum()), "n_rated": int(rated_va.sum()),
                    "precision": float(rated_va[fire].mean()) if fire.any() else np.nan,
                    "recall": float(fire[rated_va].mean()) if rated_va.any() else np.nan,
                })

            # ── 가중 스킴별 회귀 ──
            ws_va = va_g["ldaps_ws_10m"].to_numpy()
            for ws, fn in WEIGHT_SCHEMES.items():
                w_tr = fn(y_np, capacity)
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

                base = apply_bias_correction_v2(pred_blend, ws_va, group, coarse_maps[ws])
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()

                for pm, p_rated in [("nosnap", None), ("snap30G", p_rated_G), ("snap30P", p_rated_P)]:
                    pv = base.copy()
                    if p_rated is not None:
                        mask = p_rated >= SNAP_THR
                        pv[mask] = np.maximum(pv[mask], capacity * SNAP_CF_VAL)
                    pv = np.clip(pv, capacity * 0.10, capacity)
                    v = f"{ws}_{pm}"
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
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) wbin_snap30G={fold_line['wbin_snap30G']:.4f} | top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    print("\n================ 결과 요약 (공식 4-fold, pooled) ================")
    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in names:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]), "fold_std": np.std(fold_scores[v], ddof=1)})
    summary = pd.DataFrame(rows)
    piv = summary.assign(ws=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print("\n--- 가중 × 후처리 (pooled Score) ---")
    print(piv.pivot(index="ws", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    best = summary.sort_values("pooled_Score", ascending=False).iloc[0]["variant"]
    gs = pd.DataFrame(group_scores)
    show = sorted({best, "wbin_nosnap", "wbin_snap30G", best.rsplit("_", 1)[0] + "_nosnap"})
    print(f"\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    print(gs[gs["variant"].isin(show)].groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4).to_string())

    ss = pd.DataFrame(snap_stats)
    print(f"\n--- 정격 분류기 (fold 평균, p≥{SNAP_THR}) ---")
    print(ss.groupby(["clf", "group"])[["n_fire", "n_rated", "precision", "recall"]].mean().round(3).to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
