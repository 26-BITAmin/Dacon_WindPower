# ═══════════════════════════════════════════════════════
# exp-014: 시간 문맥 피처 확장 (T2) — exp-012에서 효과 확인된 T 블록의 심화
#
# v10TPS(122개) 위에 T2 블록 15개 추가:
#   - 긴 창: ws r12 mean/std (LDAPS 10m, GFS 100m)
#   - 전일 대비: d24 (같은 시각 전일 예보 대비 변화 — 전날 발표분이므로 리크 아님)
#   - 램프×풍속: |d1|·ws, |dm1|·ws (강풍에서의 램프가 오차 주범이라는 가설)
#   - 풍향 변화율: 1−cos(Δwd) 1h + 3h rolling (전선 통과·풍향 급변 프록시)
#   - 예보 불일치 rolling: ws_disagreement r3 mean/std
#   - 시어·돌풍폭 rolling: wind_shear_100_10 r3 mean, phy_gust_range r3 mean
#   - 하루 블록 min
# 변형: v10TPS(control) / v12(=v10TPS+T2). 후처리 nosnap/snap30. 시드 5 (exp-012와 동일 조건)
# 실행: python -X utf8 exp014_temporal2.py [--quick] [--jobs N]
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
from exp012_feature_v11 import build_v11

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_THR, SNAP_TO = 0.30, 0.94
POST_MODES = ["nosnap", "snap30"]


def add_temporal2(df):
    """T2: v11 df(=v10TPS 컬럼 보유) 위에서 계산. train/test 각각 호출."""
    df = df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
    for src, p in [("ldaps_ws_10m", "l10"), ("gfs_ws_100m", "g100")]:
        g = df.groupby("kpx_group")[src]
        df[f"t2_{p}_r12_mean"] = g.transform(lambda s: s.rolling(12, center=True, min_periods=1).mean())
        df[f"t2_{p}_r12_std"] = g.transform(lambda s: s.rolling(12, center=True, min_periods=1).std())
        df[f"t2_{p}_d24"] = g.transform(lambda s: s.diff(24))
    df["t2_rampxws_d1"] = df["tmp_l10_d1"].abs() * df["ldaps_ws_10m"]
    df["t2_rampxws_dm1"] = df["tmp_l10_dm1"].abs() * df["ldaps_ws_10m"]
    gcos = df.groupby("kpx_group")["ldaps_wd_cos_10m"]
    gsin = df.groupby("kpx_group")["ldaps_wd_sin_10m"]
    cos_prev, sin_prev = gcos.shift(1), gsin.shift(1)
    df["t2_wd_change"] = 1 - (df["ldaps_wd_cos_10m"] * cos_prev + df["ldaps_wd_sin_10m"] * sin_prev)
    df["t2_wd_change_r3"] = df.groupby("kpx_group")["t2_wd_change"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())
    gd = df.groupby("kpx_group")["ws_disagreement"]
    df["t2_disag_r3_mean"] = gd.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    df["t2_disag_r3_std"] = gd.transform(lambda s: s.rolling(3, center=True, min_periods=1).std())
    df["t2_shear_r3_mean"] = df.groupby("kpx_group")["gfs_wind_shear_100_10"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())
    df["t2_gustrange_r3_mean"] = df.groupby("kpx_group")["phy_gust_range"].transform(
        lambda s: s.rolling(3, center=True, min_periods=1).mean())
    day_key = (pd.to_datetime(df["forecast_kst_dtm"]) - pd.Timedelta(hours=1)).dt.date
    df["t2_day_min"] = df.groupby([df["kpx_group"], day_key])["ldaps_ws_10m"].transform("min")
    cols = [c for c in df.columns if c.startswith("t2_")]
    return df, cols


def build_v12():
    train_df, test_df, col_sets, sample_submission = build_v11()
    train_df, t2_cols = add_temporal2(train_df)
    test_df, _ = add_temporal2(test_df)
    col_sets = {"v10TPS": col_sets["v10TPS"], "v12": col_sets["v10TPS"] + t2_cols}
    return train_df, test_df, col_sets, sample_submission


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
    train_df, _, col_sets, _ = build_v12()
    for k, v in col_sets.items():
        print(f"{k}: {len(v)}개 피처")
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    names = [f"{fv}_{pm}" for fv in col_sets for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]

        va_time_index = sorted(va_all["forecast_kst_dtm"].unique())
        answer_df = pd.DataFrame({"forecast_kst_dtm": va_time_index})
        pred_dfs = {v: pd.DataFrame({"forecast_kst_dtm": va_time_index}) for v in names}
        answer_done = False

        for fv, cols in col_sets.items():
            bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
            coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

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
                pred_blend = 0.5 * pred_lgbm + 0.5 * np.mean(preds, axis=0)

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
                for pm in POST_MODES:
                    pv = base.copy()
                    if pm == "snap30":
                        mask = p_rated >= SNAP_THR
                        pv[mask] = np.maximum(pv[mask], capacity * SNAP_TO)
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
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) " +
              " | ".join(f"{v}={s:.4f}" for v, s in fold_line.items()), flush=True)

    print("\n================ 결과 요약 (공식 4-fold, pooled) ================")
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
    print("\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    pv = gs.groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4)
    print(pv.to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
