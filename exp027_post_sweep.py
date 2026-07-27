# ═══════════════════════════════════════════════════════
# exp-027: 미탐색 후처리 축 스윕 — rolling smoothing (w, α) × 하한 clip × 스냅 모양
#
# 배경: 챔피언(exp-020 TPSJgJm)이 8연속 방어(exp-021~026). 남은 미탐색 축을 점검하니
#   후처리 3요소 중 두 개가 exp-002(v10 시절) 이후 한 번도 스윕된 적 없음:
#     - rolling smoothing (window=3, alpha=0.7) — 고정값, 근거 미기록
#     - 하한 clip = 0.10·cap — "평가 대상은 actual ≥ 0.10cap이므로 그 밑은 손해"라는
#       논리로 정해졌으나, 최적 바닥은 P(actual | 모델이 매우 낮게 예측)의 조건부
#       분포에 달려 있음. 0.10이 최적이라는 보장 없음.
#   스냅은 exp-009/010에서 스윕됐으나 (a) p0.15·JMA 이전 base 기준이고
#   (b) "확률 가중 소프트 스냅"·"2단 스냅" 모양은 미시도.
#
# 설계: 챔피언 주위 one-at-a-time 스윕 (축 간 교차 조합 없음 — CV 사후선택 위험 억제).
#   base(블렌딩+편향보정)는 fold당 1회 학습 후 캐시, 후처리만 교체 → 저비용.
#   판정: CV(4-fold) + 1년 홀드아웃 이중 게이트 양쪽에서 챔피언 초과해야 후보.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp027_post_sweep.py [--quick] [--jobs N]
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

SNAP_THR, SNAP_CF_VAL = 0.30, 0.94

# 변형 정의: (smooth=(window, alpha) 또는 None, snap=모드, clip_lo=하한 CF)
#   snap 모드: "hard:<cf>" | "soft" | "tier" | "off"
CHAMP = ((3, 0.7), f"hard:{SNAP_CF_VAL}", 0.10)
VARIANTS = {
    "champ":      CHAMP,
    # --- A. rolling smoothing 축 ---
    "sm_off":     (None,       CHAMP[1], CHAMP[2]),
    "sm_w3a50":   ((3, 0.50),  CHAMP[1], CHAMP[2]),
    "sm_w3a85":   ((3, 0.85),  CHAMP[1], CHAMP[2]),
    "sm_w5a70":   ((5, 0.70),  CHAMP[1], CHAMP[2]),
    "sm_w5a85":   ((5, 0.85),  CHAMP[1], CHAMP[2]),
    # --- B. 하한 clip 축 (quick 진단: 0.15cap 미만 예측 시간의 실측 평균 CF 0.164) ---
    "cl_012":     (CHAMP[0], CHAMP[1], 0.12),
    "cl_014":     (CHAMP[0], CHAMP[1], 0.14),
    "cl_015":     (CHAMP[0], CHAMP[1], 0.15),
    "cl_016":     (CHAMP[0], CHAMP[1], 0.16),
    "cl_018":     (CHAMP[0], CHAMP[1], 0.18),
    "cl_020":     (CHAMP[0], CHAMP[1], 0.20),
    "cl_022":     (CHAMP[0], CHAMP[1], 0.22),
    # --- C. 스냅 모양 축 (soft:ramp lo→hi 로 계단 완만화) ---
    "sn_off":     (CHAMP[0], "off",       CHAMP[2]),
    "sn_092":     (CHAMP[0], "hard:0.92", CHAMP[2]),
    "sn_096":     (CHAMP[0], "hard:0.96", CHAMP[2]),
    "sn_soft":    (CHAMP[0], "soft:0.15:0.60", CHAMP[2]),
    "sn_soft2":   (CHAMP[0], "soft:0.20:0.50", CHAMP[2]),
    "sn_soft3":   (CHAMP[0], "soft:0.10:0.70", CHAMP[2]),
    "sn_tier":    (CHAMP[0], "tier",      CHAMP[2]),
    # --- D. 사전등록 조합 (축별 승자 가산 여부 확인용, 사후선택 아님) ---
    "cmb_c15_sf": (CHAMP[0], "soft:0.15:0.60", 0.15),
    "cmb_c18_sf": (CHAMP[0], "soft:0.15:0.60", 0.18),
}

# stage 2: stage 1에서 이중 게이트를 통과한 축(clip, smoothing, snap 0.92)의 조합.
#   stage 1은 one-at-a-time이라 축 간 가산성이 미검증 → 조합만 따로 확인.
STAGE2 = {
    "champ":      CHAMP,
    "c16":        ((3, 0.70), "hard:0.94", 0.16),   # stage1 단독 승자 (기준선)
    "c17":        ((3, 0.70), "hard:0.94", 0.17),
    "c16_a50":    ((3, 0.50), "hard:0.94", 0.16),
    "c16_a40":    ((3, 0.40), "hard:0.94", 0.16),
    "c15_a50":    ((3, 0.50), "hard:0.94", 0.15),
    "c17_a50":    ((3, 0.50), "hard:0.94", 0.17),
    "c18_a50":    ((3, 0.50), "hard:0.94", 0.18),
    "c16_w5":     ((5, 0.70), "hard:0.94", 0.16),
    "c16_w5a50":  ((5, 0.50), "hard:0.94", 0.16),
    "c16_92":     ((3, 0.70), "hard:0.92", 0.16),
    "c16_a50_92": ((3, 0.50), "hard:0.92", 0.16),
}
NAMES = list(VARIANTS)


def apply_snap(pv, p_rated, capacity, mode):
    """스냅 모양별 적용. pv를 아래로 내리는 일은 없음 (전부 max 기반)."""
    if mode == "off":
        return pv
    if mode.startswith("hard:"):
        cf = float(mode.split(":")[1])
        m = p_rated >= SNAP_THR
        pv[m] = np.maximum(pv[m], capacity * cf)
        return pv
    if mode.startswith("soft:"):
        # 하드 임계(p=0.30 계단)를 p lo→hi 선형 램프로 완만화
        _, lo, hi = mode.split(":")
        lo, hi = float(lo), float(hi)
        w = np.clip((p_rated - lo) / (hi - lo), 0.0, 1.0)
        return np.maximum(pv, pv + w * (capacity * SNAP_CF_VAL - pv))
    if mode == "tier":
        m1 = p_rated >= SNAP_THR
        pv[m1] = np.maximum(pv[m1], capacity * SNAP_CF_VAL)
        m2 = p_rated >= 0.70
        pv[m2] = np.maximum(pv[m2], capacity * 0.97)
        return pv
    raise ValueError(mode)


def run_sweep(folds, train_df, cols, args, tag):
    fold_scores = {v: [] for v in NAMES}
    group_scores = []
    clip_stats = []
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

            # base = 블렌딩 + 편향보정 (스무딩/스냅/clip 이전)
            blend = 0.5 * pred_lgbm + 0.5 * pred_xgb
            base_raw = apply_bias_correction_v2(blend, ws_va, group, coarse_map)

            # clip 축 진단: 평가 대상 시간 중 스무딩 후 예측이 낮은 구간의 실측 분포
            sm_champ = apply_rolling_smoothing(pd.Series(base_raw, index=va_g.index), 3, 0.7).to_numpy()
            ev = actual_va >= capacity * 0.10
            for lo in [0.05, 0.10, 0.15, 0.20]:
                m = ev & (sm_champ < capacity * lo)
                clip_stats.append({"fold": fold["name"], "group": group, "below_cf": lo,
                                   "n": int(m.sum()),
                                   "mean_actual_cf": float(np.mean(actual_va[m]) / capacity) if m.sum() else np.nan,
                                   "median_actual_cf": float(np.median(actual_va[m]) / capacity) if m.sum() else np.nan})

            for vname, (sm, snap_mode, clip_lo) in VARIANTS.items():
                pv = base_raw.copy()
                if sm is not None:
                    pv = apply_rolling_smoothing(pd.Series(pv, index=va_g.index), sm[0], sm[1]).to_numpy()
                pv = np.array(pv, dtype=float)  # to_numpy()가 읽기전용 뷰를 줄 수 있음
                pv = apply_snap(pv, p_rated, capacity, snap_mode)
                pv = np.clip(pv, capacity * clip_lo, capacity)

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
    return pd.DataFrame(rows), pd.DataFrame(group_scores), pd.DataFrame(clip_stats)


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
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    if args.stage == 2:
        global VARIANTS, NAMES
        VARIANTS = STAGE2
        NAMES = list(VARIANTS)
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
    print(f"챔피언 피처 {len(cols)}개, 후처리 변형 {len(NAMES)}종 (one-at-a-time)")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    cv, gs, cs = run_sweep(folds, train_df, cols, args, "CV")
    d_cv = show(cv, "4-fold CV (pooled)")

    print("\n--- clip 축 진단: 스무딩 후 예측이 낮은 평가시간의 실측 CF ---")
    agg = cs.groupby(["group", "below_cf"]).agg(n=("n", "sum"),
                                                mean_actual_cf=("mean_actual_cf", "mean"),
                                                median_actual_cf=("median_actual_cf", "mean"))
    print(agg.round(3).to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hd, hgs, _ = run_sweep([HOLD_FOLD], train_df, cols, args, "HOLD")
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
