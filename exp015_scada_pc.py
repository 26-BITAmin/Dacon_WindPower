# ═══════════════════════════════════════════════════════
# exp-015: SCADA 파워커브 피처 (v13) — train 기간 SCADA로 실측 파워커브 학습
#
# SCADA 검증 (2026-07-21): 10분 power 단위는 kWh/10min (VESTAS 정격 600, UNISON 700),
#   VESTAS에 0.047% 센티널 이상치(±4e7) → clip 필요. 클리핑 후 시간 합산은
#   라벨과 corr 0.9966~0.9998로 일치. 그룹1=VESTAS 1~6, 그룹2=VESTAS 7~12, 그룹3=UNISON 1~5.
# 피처 (그룹별·fold별로 train_end 이전 SCADA만 사용해 리크 차단):
#   1. 캘리브레이션: 예보 ws(LDAPS 10m / GFS 100m) → SCADA 파크 평균 ws 선형 사상
#   2. 파워커브: SCADA 파크 ws(0.5m/s bin) → 파크 CF 곡선 (bin 평균, 결측 bin 보간)
#   3. 주입 5개: scada_est_ws_l, scada_est_ws_g(추정 허브풍속),
#               scada_cf_l, scada_cf_g(커브 통과 CF), scada_cf_mean
# 변형: v10TPS(control) / v13(=v10TPS+5개). 후처리 nosnap/snap30. 시드 5.
# 편향보정 추정 모델은 두 변형 모두 v10TPS 피처 사용 (base 모델 효과만 분리).
# 실행: python -X utf8 exp015_scada_pc.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, FOLDS_Q, TRAIN_DIR,
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
SCADA_COLS = ["scada_est_ws_l", "scada_est_ws_g", "scada_cf_l", "scada_cf_g", "scada_cf_mean"]
PC_BINS = np.arange(0.0, 30.5, 0.5)
PC_CENTERS = (PC_BINS[:-1] + PC_BINS[1:]) / 2
MIN_BIN_N = 20

GROUP_TURBINES = {
    1: ("vestas", [f"vestas_wtg{i:02d}" for i in range(1, 7)], 620),
    2: ("vestas", [f"vestas_wtg{i:02d}" for i in range(7, 13)], 620),
    3: ("unison", [f"unison_wtg{i:02d}" for i in range(1, 6)], 840),
}


def build_scada_hourly():
    """그룹별 시간 집계: park_ws(터빈 평균 풍속), park_cf(발전량/설비용량)"""
    raw = {
        "vestas": pd.read_csv(TRAIN_DIR / "scada_vestas_train.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]),
        "unison": pd.read_csv(TRAIN_DIR / "scada_unison_train.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"]),
    }
    out = {}
    for group, (maker, turbs, cap10) in GROUP_TURBINES.items():
        df = raw[maker]
        pcols = [f"{t}_power_kw10m" for t in turbs]
        wcols = [f"{t}_ws" for t in turbs]
        d = pd.DataFrame({
            "hour": df["kst_dtm"].dt.ceil("h"),
            "p": df[pcols].clip(-50, cap10).sum(axis=1),
            "ws": df[wcols].mean(axis=1),
        })
        h = d.groupby("hour").agg(energy=("p", "sum"), park_ws=("ws", "mean"), n=("p", "size"))
        h = h[h["n"] == 6].drop(columns="n")
        h["park_cf"] = h["energy"] / CAPACITY_KWH[f"kpx_group_{group}"]
        out[group] = h.reset_index().rename(columns={"hour": "forecast_kst_dtm"})
    return out


def fit_scada_maps(scada_h, g_train, tr_end):
    """fold-안전: train_end 이전 시간만으로 (캘리브레이션 계수, 파워커브) 학습"""
    sc = scada_h[scada_h["forecast_kst_dtm"] <= tr_end]
    j = g_train.merge(sc, on="forecast_kst_dtm", how="inner").dropna(
        subset=["park_ws", "park_cf", "ldaps_ws_10m", "gfs_ws_100m"])
    if len(j) < 500:
        return None
    calib = {}
    for src, key in [("ldaps_ws_10m", "l"), ("gfs_ws_100m", "g")]:
        a, b = np.polyfit(j[src], j["park_ws"], 1)
        calib[key] = (a, b)
    ws_bin = pd.cut(j["park_ws"], bins=PC_BINS, labels=False)
    grp = j.groupby(ws_bin)["park_cf"]
    cf_mean, cf_n = grp.mean(), grp.size()
    curve = np.full(len(PC_CENTERS), np.nan)
    for idx, val in cf_mean.items():
        if cf_n[idx] >= MIN_BIN_N:
            curve[int(idx)] = val
    valid = ~np.isnan(curve)
    curve = np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])
    return calib, curve


def apply_scada_features(df, maps):
    """df(한 그룹의 행들)에 SCADA 피처 5개 컬럼 추가"""
    df = df.copy()
    if maps is None:
        for c in SCADA_COLS:
            df[c] = np.nan
        return df
    calib, curve = maps
    est = {}
    for src, key in [("ldaps_ws_10m", "l"), ("gfs_ws_100m", "g")]:
        a, b = calib[key]
        est[key] = np.clip(a * df[src].to_numpy() + b, 0, PC_CENTERS[-1])
        df[f"scada_est_ws_{key}"] = est[key]
        df[f"scada_cf_{key}"] = np.interp(est[key], PC_CENTERS, curve)
    df["scada_cf_mean"] = (df["scada_cf_l"] + df["scada_cf_g"]) / 2
    return df


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
    train_df, _, col_sets, _ = build_v11()
    base_cols = col_sets["v10TPS"]
    scada_h = build_scada_hourly()
    for g, h in scada_h.items():
        print(f"그룹{g} SCADA 시간수: {len(h)} ({h['forecast_kst_dtm'].min().date()} ~ {h['forecast_kst_dtm'].max().date()})")
    variants = {"v10TPS": base_cols, "v13": base_cols + SCADA_COLS}
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    names = [f"{fv}_{pm}" for fv in variants for pm in POST_MODES]
    fold_scores = {v: [] for v in names}
    group_scores = []
    all_answer, all_pred = [], {v: [] for v in names}

    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, base_cols, fold["train_end"], args.trees, args.jobs)
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

            maps = fit_scada_maps(scada_h[group], tr_g, fold["train_end"])
            tr_g = apply_scada_features(tr_g, maps)
            va_g = apply_scada_features(va_g, maps)

            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
            y_rated = (y_tr.to_numpy() >= capacity * RATED_CF).astype(int)
            actual_va = va_g["target"].to_numpy()
            ws_va = va_g["ldaps_ws_10m"].to_numpy()

            for fv, cols in variants.items():
                X_tr, X_va = prepare_matrices(tr_g, va_g, cols)

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

                base = apply_bias_correction_v2(pred_blend, ws_va, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), window=3, alpha=0.7).to_numpy()

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
