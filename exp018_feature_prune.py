# ═══════════════════════════════════════════════════════
# exp-018: v10TPS 피처 프루닝 (importance 기반 top-k 선별)
#
# 배경 (MODEL.md):
#   - 원본 변수는 이미 전부 사용 중 (build_group_weather가 전 컬럼 반입) → "발굴" 여지 없음
#   - exp-014: 피처 추가(T2)가 CV 전 지표 하락 → 122개 안에 노이즈 피처 존재 정황
#   - exp-013: 모델 용량 축 수렴 → 남은 base 개선 축은 피처 구성
#
# 선정 프로토콜 (리크 방지):
#   모든 fold의 train_end 이하인 2024-01-01 이전 데이터만으로 그룹별 LGBM(3-seed)
#   gain importance 산출 → 모델별 합=1로 정규화 후 그룹·시드 평균 → 단일 랭킹.
#   검증 구간(2024년)은 어느 fold에서도 선정에 관여하지 않음.
#
# 변형: v10TPS(122, control) / top100 / top80 / top60 / top45
# 후처리: nosnap / snap30G(그룹별 분류기) / snap30P(pooled 분류기) — 모두 p0.3/0.94
# 분류기·편향보정도 각 변형의 피처로 학습 (exp-012 방식)
# 실행: C:\miniconda3\python.exe -X utf8 exp018_feature_prune.py [--quick] [--jobs N]
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
    metric, metric_single_group, calendar_features, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94
POST_MODES = ["nosnap", "snap30G", "snap30P"]
TOPK_LIST = [100, 80, 60, 45]
RANK_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")  # 모든 fold train_end의 최소값


def rank_features(train_df, cols, n_estimators, n_jobs, n_seeds=3):
    """2024년 이전 데이터만으로 그룹별 LGBM gain importance → 정규화 평균 랭킹"""
    sub = train_df[(train_df["forecast_kst_dtm"] <= RANK_CUTOFF) & train_df["target"].notna()]
    scores = pd.Series(0.0, index=cols)
    n_models = 0
    for group in [1, 2, 3]:
        g = sub[sub["kpx_group"] == group]
        if len(g) == 0:
            continue
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        feat = pd.concat([calendar_features(g["forecast_kst_dtm"]).reset_index(drop=True),
                          g[cols].reset_index(drop=True)], axis=1)
        X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(feat), columns=feat.columns)
        w = make_sample_weight(g["target"].to_numpy(), capacity, low_weight=0.05)
        for seed in range(n_seeds):
            m = lgb.LGBMRegressor(**lgbm_params(n_estimators, n_jobs, seed))
            m.fit(X, g["target"], sample_weight=w)
            imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
            imp = imp / imp.sum()
            scores = scores.add(imp.reindex(cols).fillna(0.0), fill_value=0.0)
            n_models += 1
    return (scores / n_models).sort_values(ascending=False)


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
    cols_full = col_sets["v10TPS"]

    ranking = rank_features(train_df, cols_full, args.trees, args.jobs)
    print(f"\n--- importance 랭킹 (2024년 이전 데이터, 상위 20) ---")
    print(ranking.head(20).round(5).to_string())
    print(f"\n--- 하위 20 (프루닝 후보) ---")
    print(ranking.tail(20).round(6).to_string())

    variants_cols = {"full122": cols_full}
    for k in TOPK_LIST:
        variants_cols[f"top{k}"] = ranking.head(k).index.tolist()
    print(f"\n변형: " + ", ".join(f"{n}({len(c)})" for n, c in variants_cols.items()))
    print(f"데이터·랭킹 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    names = [f"{fv}_{pm}" for fv in variants_cols for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}
        answer_done = False

        for fv, cols in variants_cols.items():
            bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
            coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

            pooled_clf = []
            for group in [1, 2, 3]:
                target_col = f"kpx_group_{group}"
                capacity = CAPACITY_KWH[target_col]
                tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
                va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()

                if not answer_done:
                    va_answer = va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": target_col})
                    answer_df = answer_df.merge(va_answer, on="forecast_kst_dtm", how="left")
                if len(tr_g) == 0:
                    for pm in POST_MODES:
                        pred_dfs[f"{fv}_{pm}"][target_col] = 0.0
                    continue

                X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
                y_tr = tr_g["target"]
                y_np = y_tr.to_numpy()
                w_tr = make_sample_weight(y_np, capacity, low_weight=0.05)

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

                # 그룹별 분류기
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

                # pooled 분류기 (변형 피처 + kpx_group, 학습은 첫 그룹에서 한 번만)
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

                ws_va = va_g["ldaps_ws_10m"].to_numpy()
                base = apply_bias_correction_v2(pred_blend, ws_va, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()

                actual_va = va_g["target"].to_numpy()
                for pm, p_rated in [("nosnap", None), ("snap30G", p_rated_G), ("snap30P", p_rated_P)]:
                    pv = base.copy()
                    if p_rated is not None:
                        mask = p_rated >= SNAP_THR
                        pv[mask] = np.maximum(pv[mask], capacity * SNAP_CF_VAL)
                    pv = np.clip(pv, capacity * 0.10, capacity)
                    v = f"{fv}_{pm}"
                    va_pred = va_g[["forecast_kst_dtm"]].copy()
                    va_pred[target_col] = pv
                    pred_dfs[v] = pred_dfs[v].merge(va_pred, on="forecast_kst_dtm", how="left")
                    nmae_g, ficr_g, _ = metric_single_group(actual_va, pv, capacity)
                    group_scores.append({"fold": fold["name"], "group": group, "variant": v,
                                         "1-NMAE": 1 - nmae_g, "FICR": ficr_g})
            answer_done = True

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
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) full122_snap30P={fold_line['full122_snap30P']:.4f} | top4: "
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
    piv = summary.assign(fv=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print("\n--- 피처 수 × 후처리 (pooled Score) ---")
    print(piv.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    best = summary.sort_values("pooled_Score", ascending=False).iloc[0]["variant"]
    gs = pd.DataFrame(group_scores)
    show = sorted({best, "full122_nosnap", "full122_snap30P", best.rsplit("_", 1)[0] + "_nosnap"})
    print(f"\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    print(gs[gs["variant"].isin(show)].groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4).to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
