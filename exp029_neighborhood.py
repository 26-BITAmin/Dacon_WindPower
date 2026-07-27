# ═══════════════════════════════════════════════════════
# exp-029: 신 챔피언(exp-027 c16_a50) 주변 정밀 후처리 스윕
#
# 배경: exp-027로 clip 0.16·α0.50가 LB 경신(0.63523). 이 새 baseline 위에서
#   아직 안 건드린 축을 이중 게이트로 판정:
#     - 스냅값 (0.92~0.95; 챔피언 0.94) — exp-027 stage2에서 0.92가 CV/hold 소폭↑
#     - 스냅 임계 p (0.25~0.35; 챔피언 0.30) — 신 base에서 미검증
#     - α 미세 (0.40~0.55; 챔피언 0.50)
#     - **그룹별 차등 clip** (g3만 0.14~0.20; 신규 축. g3 clip 발동률 35.7%로 최고)
#   baseline "cham" = c16_a50 (clip 0.16 all, α0.50, snap p0.30→0.94).
#   Δcham > 0 이 "현 LB 챔피언보다 나음"을 뜻함. 양 게이트 통과분만 제출 후보.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp029_neighborhood.py [--quick] [--jobs N]
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
    metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp019_multigrid import HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 변형 스펙: (alpha, snap_thr, snap_to, clip_by_group)
#   clip_by_group: {1: c1, 2: c2, 3: c3}
BASE_CLIP = {1: 0.16, 2: 0.16, 3: 0.16}
VARIANTS = {
    "cham":       (0.50, 0.30, 0.94, BASE_CLIP),           # = exp-027 c16_a50 (신 LB 챔피언)
    # --- A. 스냅값 ---
    "sv_092":     (0.50, 0.30, 0.92, BASE_CLIP),
    "sv_093":     (0.50, 0.30, 0.93, BASE_CLIP),
    "sv_095":     (0.50, 0.30, 0.95, BASE_CLIP),
    # --- B. 스냅 임계 p ---
    "sp_025":     (0.50, 0.25, 0.94, BASE_CLIP),
    "sp_035":     (0.50, 0.35, 0.94, BASE_CLIP),
    "sp_020":     (0.50, 0.20, 0.94, BASE_CLIP),
    # --- C. α 미세 ---
    "a_040":      (0.40, 0.30, 0.94, BASE_CLIP),
    "a_045":      (0.45, 0.30, 0.94, BASE_CLIP),
    "a_055":      (0.55, 0.30, 0.94, BASE_CLIP),
    # --- D. 그룹별 차등 clip (g3만 변경) ---
    "g3c_014":    (0.50, 0.30, 0.94, {1: 0.16, 2: 0.16, 3: 0.14}),
    "g3c_018":    (0.50, 0.30, 0.94, {1: 0.16, 2: 0.16, 3: 0.18}),
    "g3c_020":    (0.50, 0.30, 0.94, {1: 0.16, 2: 0.16, 3: 0.20}),
    # --- E. 전역 clip 미세 (0.16 주변) ---
    "cl_017":     (0.50, 0.30, 0.94, {g: 0.17 for g in [1, 2, 3]}),
    "cl_015":     (0.50, 0.30, 0.94, {g: 0.15 for g in [1, 2, 3]}),
    # --- F. 사전등록 조합 (stage2 최상위 c16_a50_92 재확인 + g별) ---
    "cmb_092_017": (0.50, 0.30, 0.92, {g: 0.17 for g in [1, 2, 3]}),
}
NAMES = list(VARIANTS)


def run_sweep(folds, train_df, cols, args, tag):
    fold_scores = {v: [] for v in NAMES}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in NAMES}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

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

            ws_va = va_g["ldaps_ws_10m"].to_numpy()
            actual_va = va_g["target"].to_numpy()
            blend = 0.5 * pred_lgbm + 0.5 * pred_xgb
            base_bias = apply_bias_correction_v2(blend, ws_va, group, coarse_map)

            for vname, (alpha, sthr, sto, clipg) in VARIANTS.items():
                pv = apply_rolling_smoothing(pd.Series(base_bias, index=va_g.index), 3, alpha).to_numpy()
                pv = np.array(pv, dtype=float)
                snap_mask = p_rated >= sthr
                pv[snap_mask] = np.maximum(pv[snap_mask], capacity * sto)
                pv = np.clip(pv, capacity * clipg[group], capacity)

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
        print(f"[{tag}:{fold['name']}] ({time.time() - tf:.0f}s) cham={fold_line['cham']:.4f} | top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in NAMES:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]),
                     "fold_std": np.std(fold_scores[v], ddof=1) if len(folds) > 1 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(group_scores)


def show(summary, tag):
    base = float(summary.loc[summary["variant"] == "cham", "pooled_Score"].iloc[0])
    out = summary.copy()
    out["Δcham"] = out["pooled_Score"] - base
    print(f"\n--- {tag} (cham={base:.4f}, 내림차순) ---")
    print(out.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))
    return out.set_index("variant")["Δcham"]


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
    print(f"신 챔피언(c16_a50) 주변 {len(NAMES)}변형, 피처 {len(cols)}개")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    cv, gs = run_sweep(folds, train_df, cols, args, "CV")
    d_cv = show(cv, "4-fold CV (pooled)")
    print("\n--- 그룹별 score_g (snap 적용, fold 평균) 상위 변형 비교 ---")
    sel = gs.copy()
    sel["score_g"] = 0.5 * sel["1-NMAE"] + 0.5 * sel["FICR"]
    piv = sel.groupby(["group", "variant"])["score_g"].mean().unstack(0).round(4)
    print(piv.loc[["cham", "g3c_014", "g3c_018", "g3c_020", "sv_092", "cl_017"]].to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hd, _ = run_sweep([HOLD_FOLD], train_df, cols, args, "HOLD")
        d_h = show(hd, "홀드아웃 2024 전체")

        print("\n================ 이중 게이트 판정 (Δcham, cham=c16_a50=현 LB 챔피언) ================")
        gate = pd.DataFrame({"ΔCV": d_cv, "Δholdout": d_h}).drop(index="cham")
        gate["both_pos"] = (gate["ΔCV"] > 0) & (gate["Δholdout"] > 0)
        print(gate.sort_values("ΔCV", ascending=False).round(4).to_string())
        passed = gate[gate["both_pos"]]
        print("\n>>> 양 게이트 통과 (현 챔피언 초과): " + (", ".join(passed.index) if len(passed) else "없음"))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
