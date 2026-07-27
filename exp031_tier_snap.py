# ═══════════════════════════════════════════════════════
# exp-031: 2단 정격 스냅 — CF 0.7~0.9 밴드 공략 (FiCR 집중)
#
# 진단 근거 (exp-030, CV·홀드아웃 일치):
#   - FiCR 손실의 33%가 CF 0.7~0.9 밴드에 있고, 이 밴드 near-miss의 71%가 과소예측.
#   - 그런데 현 스냅은 P(actual≥0.90cap)≥0.30만 발동 → 0.7~0.9 밴드를 통째로 놓침.
#   - 전체 near-miss는 방향성 없음(52% 과대) → 무조건 nudge는 무효(exp-028 실패 재현).
#     따라서 "분류기로 고발전 시간을 식별해 위로만" 미는 스냅式이 유일하게 유효.
# 변경점: 기존 tier A(P≥0.90cap 분류기 → max(pred, 0.94cap))에 더해
#   tier B(P≥0.70cap 분류기 → max(pred, target_B)) 추가. threshold_B·target_B 스윕.
#   두 tier 모두 max 연산이라 순서 무관. 분류기는 pooled(exp-016 방식).
#
# 판정: 4-fold CV + 1년 홀드아웃 이중 게이트. baseline "cham"=c16_a50(현 LB 챔피언).
#   Δcham>0 이 "현 챔피언 초과". fold 편중(exp-029 sp_020 교훈)·분류기 정밀도 모니터.
# 실행: C:\miniconda3\python.exe -X utf8 exp031_tier_snap.py [--quick] [--jobs N]
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

ALPHA, CLIP_LO = 0.50, 0.16
SNAP_A_THR, SNAP_A_TO = 0.30, 0.94   # tier A (기존)
RATED_CF_B = 0.70                     # tier B 분류기 타깃 CF

# 변형: (thr_B, target_B) — None이면 tier B 없음(=cham)
VARIANTS = {
    "cham":        None,
    "tB_040_074":  (0.40, 0.74),
    "tB_040_078":  (0.40, 0.78),
    "tB_050_078":  (0.50, 0.78),
    "tB_050_082":  (0.50, 0.82),
    "tB_060_078":  (0.60, 0.78),
    "tB_060_082":  (0.60, 0.82),
    "tB_070_080":  (0.70, 0.80),
    "tB_070_084":  (0.70, 0.84),
    "tB_050_074":  (0.50, 0.74),
    "tB_035_076":  (0.35, 0.76),
}
NAMES = list(VARIANTS)


def run_sweep(folds, train_df, cols, args, tag):
    fold_scores = {v: [] for v in NAMES}
    group_scores = []
    clf_stats = []
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

        clf_A, clf_B = [], []
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
            if not clf_A:
                caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
                yB = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF_B).astype(int)
                for seed in range(N_CLF_SEEDS):
                    for ylab, store in [(yA, clf_A), (yB, clf_B)]:
                        clf = lgb.LGBMClassifier(
                            n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                            num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, n_jobs=args.jobs, verbose=-1,
                        )
                        clf.fit(X_pool_tr, ylab)
                        store.append(clf)
            pA = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf_A], axis=0)
            pB = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf_B], axis=0)

            ws = va_g["ldaps_ws_10m"].to_numpy()
            actual = va_g["target"].to_numpy()
            blend = 0.5 * pred_lgbm + 0.5 * pred_xgb
            base = apply_bias_correction_v2(blend, ws, group, coarse_map)
            base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), 3, ALPHA).to_numpy()
            base = np.array(base, dtype=float)
            maskA = pA >= SNAP_A_THR

            for vname, spec in VARIANTS.items():
                pv = base.copy()
                pv[maskA] = np.maximum(pv[maskA], capacity * SNAP_A_TO)
                if spec is not None:
                    thrB, tgtB = spec
                    maskB = pB >= thrB
                    pv[maskB] = np.maximum(pv[maskB], capacity * tgtB)
                    # tier B 발동 시간의 실측 진단 (첫 그룹만 기록)
                    if len(clf_stats) < 3 * len(folds) or True:
                        fired = maskB & (base < capacity * tgtB)  # 실제로 올린 시간
                        if fired.sum() and vname == "tB_050_078":
                            ev = actual >= capacity * 0.10
                            clf_stats.append({"fold": fold["name"], "group": group,
                                              "fired": int(fired.sum()),
                                              "of_which_ge07": float((actual[fired] >= capacity * 0.70).mean()),
                                              "mean_actual_cf": float(np.mean(actual[fired]) / capacity)})
                pv = np.clip(pv, capacity * CLIP_LO, capacity)

                va_pred = va_g[["forecast_kst_dtm"]].copy()
                va_pred[target_col] = pv
                pred_dfs[vname] = pred_dfs[vname].merge(va_pred, on="forecast_kst_dtm", how="left")
                nmae_g, ficr_g, _ = metric_single_group(actual, pv, capacity)
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
    return pd.DataFrame(rows), pd.DataFrame(group_scores), pd.DataFrame(clf_stats)


def show(summary, tag):
    base = float(summary.loc[summary["variant"] == "cham", "pooled_Score"].iloc[0])
    fic = float(summary.loc[summary["variant"] == "cham", "FICR"].iloc[0])
    out = summary.copy()
    out["Δcham"] = out["pooled_Score"] - base
    out["ΔFiCR"] = out["FICR"] - fic
    print(f"\n--- {tag} (cham Score={base:.4f} FiCR={fic:.4f}, 내림차순) ---")
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
    print(f"2단 스냅 {len(NAMES)}변형 (tier B = P≥{RATED_CF_B}cap 분류기), 피처 {len(cols)}개")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    cv, gs, cs = run_sweep(folds, train_df, cols, args, "CV")
    d_cv = show(cv, "4-fold CV (pooled)")
    if len(cs):
        print("\n--- tier B(tB_050_078) 발동 진단: 올린 시간의 실측 ---")
        print(cs.groupby("group").agg(fired=("fired", "sum"),
                                      P_actual_ge07=("of_which_ge07", "mean"),
                                      mean_actual_cf=("mean_actual_cf", "mean")).round(3).to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 ================")
        hd, _, _ = run_sweep([HOLD_FOLD], train_df, cols, args, "HOLD")
        d_h = show(hd, "홀드아웃 2024 전체")

        print("\n================ 이중 게이트 판정 (Δcham=c16_a50) ================")
        gate = pd.DataFrame({"ΔCV": d_cv, "Δholdout": d_h}).drop(index="cham")
        gate["both_pos"] = (gate["ΔCV"] > 0) & (gate["Δholdout"] > 0)
        print(gate.sort_values("ΔCV", ascending=False).round(4).to_string())
        passed = gate[gate["both_pos"]]
        print("\n>>> 양 게이트 통과: " + (", ".join(passed.index) if len(passed) else "없음"))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
