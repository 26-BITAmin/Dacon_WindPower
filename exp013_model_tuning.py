# ═══════════════════════════════════════════════════════
# exp-013: 모델 용량·구성 실험 (1-NMAE 개선 트랙, 피처는 exp-012의 v10TPS 고정)
#
# 배경: 상위권 1-NMAE 0.88대 vs 우리 0.870~0.871 (COMPETITION.md 2026-07-21 갱신).
#       현 파이프라인은 500 trees / lr 0.03 고정, early stopping 없음 (exp-006 관찰).
# 변형 (패밀리 예측 캐시 → 조합은 추가 비용 없음):
#   families: lgb500, xgb500, cat500, lgb1500(lr0.01), xgb1500(lr0.01)
#   blends:   control(=lgb500+xgb500 50:50, exp-012 v10TPS 재현)
#             cat3(3-way 500) / big(lgb1500+xgb1500) / bigcat(lgb1500+xgb1500+cat500)
#   단독 패밀리 점수도 함께 출력 (어느 패밀리가 강한지 확인)
# 후처리: nosnap / snap30(p0.3·0.94, LB 검증 설정)만. 시드 3개 (스윕 모드).
# 실행: python -X utf8 exp013_model_tuning.py [--quick] [--jobs N] [--seeds N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm
from catboost import CatBoostRegressor

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, FOLDS_Q,
    metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp012_feature_v11 import build_v11

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FEATURE_SET = "v10TPS"
SNAP_THR, SNAP_TO = 0.30, 0.94

# family → (kind, n_estimators, learning_rate)
FAMILIES = {
    "lgb500": ("lgb", 500, 0.03),
    "xgb500": ("xgb", 500, 0.03),
    "cat500": ("cat", 500, 0.03),
    "lgb1500": ("lgb", 1500, 0.01),
    "xgb1500": ("xgb", 1500, 0.01),
}
BLENDS = {
    "control": ["lgb500", "xgb500"],
    "cat3": ["lgb500", "xgb500", "cat500"],
    "big": ["lgb1500", "xgb1500"],
    "bigcat": ["lgb1500", "xgb1500", "cat500"],
}
POST_MODES = ["nosnap", "snap30"]


def fit_family(kind, trees, lr, seed, jobs, X_tr, y_tr, w_tr, X_va):
    if kind == "lgb":
        m = lgb.LGBMRegressor(**lgbm_params(trees, jobs, seed))
        m.set_params(learning_rate=lr)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    elif kind == "xgb":
        m = xgbm.XGBRegressor(
            n_estimators=trees, max_depth=6, learning_rate=lr,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=jobs, verbosity=0, objective="reg:absoluteerror",
        )
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        m = CatBoostRegressor(
            iterations=trees, depth=6, learning_rate=lr, loss_function="MAE",
            bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
            random_seed=seed, thread_count=jobs, verbose=0, allow_writing_files=False,
        )
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict(X_va)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trees-scale", type=float, default=1.0)
    args = parser.parse_args()
    folds = FOLDS_Q
    if args.quick:
        args.seeds = 1
        folds = FOLDS_Q[-1:]
        print("### QUICK MODE ###")

    t0 = time.time()
    train_df, _, col_sets, _ = build_v11()
    cols = col_sets[FEATURE_SET]
    print(f"피처: {FEATURE_SET} {len(cols)}개, 시드 {args.seeds}개\n", flush=True)

    names = [f"{b}_{pm}" for b in list(BLENDS) + list(FAMILIES) for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], 500, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}

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

            fam_pred = {}
            for fam, (kind, trees, lr) in FAMILIES.items():
                trees_eff = max(int(trees * args.trees_scale), 50)
                preds = [fit_family(kind, trees_eff, lr, seed, args.jobs, X_tr, y_tr, w_tr, X_va)
                         for seed in range(args.seeds)]
                fam_pred[fam] = np.mean(preds, axis=0)

            y_rated = (y_tr.to_numpy() >= capacity * RATED_CF).astype(int)
            probas = []
            for seed in range(N_CLF_SEEDS):
                clf = lgb.LGBMClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.03,
                    num_leaves=31, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, n_jobs=args.jobs, verbose=-1,
                )
                clf.fit(X_tr, y_rated)
                probas.append(clf.predict_proba(X_va)[:, 1])
            p_rated = np.mean(probas, axis=0)

            ws_va = va_g["ldaps_ws_10m"].to_numpy()
            actual_va = va_g["target"].to_numpy()

            sources = {**{b: np.mean([fam_pred[f] for f in fams], axis=0) for b, fams in BLENDS.items()},
                       **fam_pred}
            for src_name, pred_blend in sources.items():
                base = apply_bias_correction_v2(pred_blend, ws_va, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()
                for pm in POST_MODES:
                    pv = base.copy()
                    if pm == "snap30":
                        mask = p_rated >= SNAP_THR
                        pv[mask] = np.maximum(pv[mask], capacity * SNAP_TO)
                    pv = np.clip(pv, capacity * 0.10, capacity)
                    v = f"{src_name}_{pm}"
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
        top = sorted(((v, s) for v, s in fold_line.items() if v.endswith("snap30")),
                     key=lambda kv: kv[1], reverse=True)[:4]
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) control_snap30={fold_line['control_snap30']:.4f} | "
              f"snap30 top4: " + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    print("\n================ 결과 요약 (공식 4-fold, pooled) ================")
    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in names:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]), "fold_std": np.std(fold_scores[v], ddof=1)})
    summary = pd.DataFrame(rows)
    piv = summary.assign(src=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print("\n--- 모델 소스 × 후처리 (pooled Score) ---")
    print(piv.pivot(index="src", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 모델 소스 × 후처리 (1-NMAE) ---")
    print(piv.pivot(index="src", columns="pm", values="1-NMAE").round(4).to_string())
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
