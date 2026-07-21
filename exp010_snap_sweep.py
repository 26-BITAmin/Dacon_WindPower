# ═══════════════════════════════════════════════════════
# exp-010: 정격 스냅 파라미터 스윕 (exp-009 후속)
#
# exp-009 관찰: 임계값을 낮출수록 단조 개선 (p60→p30), p30 미만은 미실험.
#              그룹3은 분류기 recall≈0이라 스냅이 거의 발동 안 함.
# 이번 실험:
#   A. 전역 임계값 확장: p ∈ {0.30, 0.25, 0.20, 0.15, 0.10}
#   B. 스냅 목표 CF: {0.92, 0.94, 0.96} (exp-009는 0.94 고정)
#   C. 그룹3 전용 저임계값: 그룹1·2는 p0.30 유지, 그룹3만 p ∈ {0.10, 0.05}
#
# base(블렌딩+편향보정+rolling)와 분류기는 변형 간 공유 → 비용은 exp-009 1회분.
# 검증: 공식 4-fold CV. control = exp-009 snap_p30 (p0.30, CF 0.94)
# 실행: python -X utf8 exp010_snap_sweep.py [--quick] [--jobs N]
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
    build_datasets, metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# variant → ({group: thr}, snap_cf)
VARIANTS = {"nosnap": (None, None)}
for thr in [0.30, 0.25, 0.20, 0.15, 0.10]:
    for cf in [0.92, 0.94, 0.96]:
        VARIANTS[f"p{int(thr*100):02d}_cf{int(cf*100)}"] = ({1: thr, 2: thr, 3: thr}, cf)
for g3thr in [0.10, 0.05]:
    VARIANTS[f"g3p{int(g3thr*100):02d}"] = ({1: 0.30, 2: 0.30, 3: g3thr}, 0.94)
CONTROL = "p30_cf94"


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

    names = list(VARIANTS)
    fold_scores = {v: [] for v in names}
    group_scores = []
    snap_counts = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, weather_cols, fold["train_end"], args.trees, args.jobs)
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

            X_tr, X_va = prepare_matrices(tr_g, va_g, weather_cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)

            # ── base 블렌딩 (exp-009 control과 동일) ──
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

            # ── 정격 도달 확률 분류기 (exp-009와 동일) ──
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
            base = apply_bias_correction_v2(pred_blend, ws_va, group, coarse_map)
            base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()

            actual_va = va_g["target"].to_numpy()
            rated_true = actual_va >= capacity * RATED_CF
            for v, (thr_map, snap_cf) in VARIANTS.items():
                pv = base.copy()
                if thr_map is not None:
                    mask = p_rated >= thr_map[group]
                    pv[mask] = np.maximum(pv[mask], capacity * snap_cf)
                    snap_counts.append({
                        "fold": fold["name"], "group": group, "variant": v,
                        "n_snap": int(mask.sum()), "n_rated_true": int(rated_true.sum()),
                        "precision": float(rated_true[mask].mean()) if mask.sum() else np.nan,
                        "recall": float(mask[rated_true].mean()) if rated_true.sum() else np.nan,
                    })
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
            fold_line[v] = (s, n, f)
        top = sorted(fold_line.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) "
              f"control S={fold_line[CONTROL][0]:.4f} | top5: "
              + ", ".join(f"{v}={s[0]:.4f}" for v, s in top), flush=True)

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

    best = summary.iloc[0]["variant"]
    gs = pd.DataFrame(group_scores)
    show = [v for v in [best, CONTROL, "nosnap"] if v in names]
    print(f"\n--- 그룹별 fold 평균 (1-NMAE / FiCR) — best({best}) vs control vs nosnap ---")
    pv = gs[gs["variant"].isin(show)].groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4)
    print(pv.to_string())

    sc = pd.DataFrame(snap_counts)
    if len(sc):
        print(f"\n--- 스냅 발동/품질 (fold 합, 변형별·그룹별) — best({best}) vs control ---")
        sel = sc[sc["variant"].isin([best, CONTROL])]
        agg = sel.groupby(["variant", "group"]).agg(
            n_snap=("n_snap", "sum"), n_rated_true=("n_rated_true", "sum"),
            precision=("precision", "mean"), recall=("recall", "mean")).round(3)
        print(agg.to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
