# ═══════════════════════════════════════════════════════
# exp-012: v11 피처 엔지니어링 ablation (T=시간 문맥 / P=물리 / S=공간)
#
# v10(87개) 위에 세 블록을 더해 공식 4-fold CV로 ablation:
#   T. 시간 문맥: 예보가 다음날 24h 단위로 한 번에 제공되므로(동일 data_available)
#      이웃 시각 예보 사용은 리크 아님 → ws rolling mean/std(3h/7h), 램프(±1h/±3h),
#      하루 블록(01~24시) mean/max/std
#   P. 물리: 공기밀도 ρ=sp/(287.05·T2m), ws³, ρ·ws³(출력 프록시), 파워커브 포화 프록시,
#      LDAPS 50m 돌풍폭·거스트비, GFS gust factor, 850hPa 풍속·100m 시어
#   S. 공간: LDAPS 그룹별 최근접 4격자 ws_10m mean/std/range, GFS 9격자 ws_100m mean/std
#
# 변형: v10(control) / v10T / v10P / v10S / v10TPS
# 각 변형을 nosnap / snap30(=exp-009 LB 검증 설정 p0.3·0.94) / snap15(=exp-010 p0.15·0.92)
# 세 후처리로 평가 (분류기는 변형 피처로 학습, 후처리 간 공유 → 추가 비용 없음)
# 실행: python -X utf8 exp012_feature_v11.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, TARGET_COLS, FOLDS_Q, DATA_DIR, TRAIN_DIR, TEST_DIR,
    build_datasets, metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices, haversine, dms_to_decimal,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SNAP_CONFIGS = {"snap30": (0.30, 0.94), "snap15": (0.15, 0.92)}
POST_MODES = ["nosnap", "snap30", "snap15"]
N_SPATIAL_GRIDS = 4


# ───────────────────────────────────────────────
# 피처 블록
# ───────────────────────────────────────────────
def add_physics(df):
    """P: 기존 컬럼에서 행 단위로 계산 (train/test 동형)"""
    df = df.copy()
    rho = df["ldaps_surface_0_sp"] / (287.05 * df["ldaps_heightAboveGround_2_t"])
    df["phy_density"] = rho
    df["phy_ws10_cu"] = df["ldaps_ws_10m"] ** 3
    df["phy_ws100_cu"] = df["gfs_ws_100m"] ** 3
    df["phy_rho_ws100_cu"] = rho * df["phy_ws100_cu"]
    df["phy_pc100"] = np.clip(df["gfs_ws_100m"], 0, 12) ** 3 / 12 ** 3
    ws50max = np.hypot(df["ldaps_heightAboveGround_50_50MUmax"], df["ldaps_heightAboveGround_50_50MVmax"])
    ws50min = np.hypot(df["ldaps_heightAboveGround_50_50MUmin"], df["ldaps_heightAboveGround_50_50MVmin"])
    df["phy_ws50max"] = ws50max
    df["phy_gust_range"] = ws50max - ws50min
    df["phy_gust_over_ws"] = ws50max / (df["ldaps_ws_10m"] + 0.5)
    df["phy_gfs_gust_factor"] = df["gfs_surface_0_gust"] / (df["gfs_ws_10m"] + 0.5)
    ws850 = np.hypot(df["gfs_isobaricInhPa_850_u"], df["gfs_isobaricInhPa_850_v"])
    df["phy_ws850"] = ws850
    df["phy_shear_850_100"] = ws850 - df["gfs_ws_100m"]
    cols = [c for c in df.columns if c.startswith("phy_")]
    return df, cols


def add_temporal(df):
    """T: 그룹별 시계열 정렬 후 rolling/램프/하루 블록 집계 (train/test 각각 호출)"""
    df = df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
    for src in ["ldaps_ws_10m", "gfs_ws_100m"]:
        g = df.groupby("kpx_group")[src]
        p = "tmp_" + ("l10" if src.startswith("ldaps") else "g100")
        df[f"{p}_r3_mean"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        df[f"{p}_r3_std"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).std())
        df[f"{p}_r7_mean"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
        df[f"{p}_r7_std"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).std())
        df[f"{p}_d1"] = g.transform(lambda s: s.diff(1))
        df[f"{p}_dm1"] = g.transform(lambda s: s.diff(-1))
        df[f"{p}_d3"] = g.transform(lambda s: s.diff(3))
        df[f"{p}_dm3"] = g.transform(lambda s: s.diff(-3))
    day_key = (pd.to_datetime(df["forecast_kst_dtm"]) - pd.Timedelta(hours=1)).dt.date
    day_grp = df.groupby([df["kpx_group"], day_key])["ldaps_ws_10m"]
    df["tmp_day_mean"] = day_grp.transform("mean")
    df["tmp_day_max"] = day_grp.transform("max")
    df["tmp_day_std"] = day_grp.transform("std")
    cols = [c for c in df.columns if c.startswith("tmp_")]
    return df, cols


def build_spatial_frames():
    """S: 원본 격자 데이터에서 그룹별 최근접 k격자 ws 집계 (train/test 각각 반환)"""
    raw = pd.read_excel(DATA_DIR / "info.xlsx", sheet_name="info", header=None)
    info = raw.iloc[4:21].copy()
    info.columns = raw.iloc[3]
    info = info.reset_index(drop=True)
    info = info.rename(columns={"좌표(Google)": "coord", "KPX그룹": "kpx_group"})
    info["kpx_group"] = info["kpx_group"].ffill().astype(int)
    coords = info["coord"].apply(dms_to_decimal)
    info["lat"] = coords.apply(lambda x: x[0])
    info["lon"] = coords.apply(lambda x: x[1])
    centroids = info.groupby("kpx_group")[["lat", "lon"]].mean()

    out = {}
    for split, ld_path, gf_path in [("train", TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv"),
                                    ("test", TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")]:
        ld = pd.read_csv(ld_path, encoding="utf-8-sig",
                         usecols=["forecast_kst_dtm", "grid_id", "latitude", "longitude",
                                  "heightAboveGround_10_10u", "heightAboveGround_10_10v"])
        ld["forecast_kst_dtm"] = pd.to_datetime(ld["forecast_kst_dtm"])
        ld["ws"] = np.hypot(ld["heightAboveGround_10_10u"], ld["heightAboveGround_10_10v"])
        grids = ld[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")

        frames = []
        for group in [1, 2, 3]:
            d = haversine(centroids.loc[group, "lat"], centroids.loc[group, "lon"],
                          grids["latitude"], grids["longitude"])
            near = d.nsmallest(N_SPATIAL_GRIDS).index
            sub = ld[ld["grid_id"].isin(near)].groupby("forecast_kst_dtm")["ws"]
            agg = pd.DataFrame({
                "spa_l10_mean": sub.mean(), "spa_l10_std": sub.std(),
                "spa_l10_range": sub.max() - sub.min(),
            }).reset_index()
            agg["kpx_group"] = group
            frames.append(agg)
        ldaps_spa = pd.concat(frames, ignore_index=True)

        gf = pd.read_csv(gf_path, encoding="utf-8-sig",
                         usecols=["forecast_kst_dtm", "grid_id",
                                  "heightAboveGround_100_100u", "heightAboveGround_100_100v"])
        gf["forecast_kst_dtm"] = pd.to_datetime(gf["forecast_kst_dtm"])
        gf["ws100"] = np.hypot(gf["heightAboveGround_100_100u"], gf["heightAboveGround_100_100v"])
        sub = gf.groupby("forecast_kst_dtm")["ws100"]
        gfs_spa = pd.DataFrame({"spa_g100_mean": sub.mean(), "spa_g100_std": sub.std()}).reset_index()

        out[split] = (ldaps_spa, gfs_spa)
    return out


def build_v11():
    train_df, test_df, weather_cols_v10, sample_submission = build_datasets()
    spa = build_spatial_frames()
    dfs = {}
    for split, df in [("train", train_df), ("test", test_df)]:
        df, phy_cols = add_physics(df)
        df, tmp_cols = add_temporal(df)
        ldaps_spa, gfs_spa = spa[split]
        df = df.merge(ldaps_spa, on=["forecast_kst_dtm", "kpx_group"], how="left")
        df = df.merge(gfs_spa, on="forecast_kst_dtm", how="left")
        dfs[split] = df
    spa_cols = ["spa_l10_mean", "spa_l10_std", "spa_l10_range", "spa_g100_mean", "spa_g100_std"]
    col_sets = {
        "v10": weather_cols_v10,
        "v10T": weather_cols_v10 + tmp_cols,
        "v10P": weather_cols_v10 + phy_cols,
        "v10S": weather_cols_v10 + spa_cols,
        "v10TPS": weather_cols_v10 + tmp_cols + phy_cols + spa_cols,
    }
    return dfs["train"], dfs["test"], col_sets, sample_submission


# ───────────────────────────────────────────────
# CV
# ───────────────────────────────────────────────
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
                    if pm != "nosnap":
                        thr, cf = SNAP_CONFIGS[pm]
                        mask = p_rated >= thr
                        pv[mask] = np.maximum(pv[mask], capacity * cf)
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
        print(f"[{fold['name']}] ({time.time() - tf:.0f}s) v10_nosnap={fold_line['v10_nosnap']:.4f} "
              f"v10_snap30={fold_line['v10_snap30']:.4f} | top4: "
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
    print("\n--- 후처리별 피처 비교 (행=피처 변형, 열=후처리) ---")
    piv = summary.assign(fv=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print(piv.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    best = summary.sort_values("pooled_Score", ascending=False).iloc[0]["variant"]
    gs = pd.DataFrame(group_scores)
    show = sorted({best, "v10_nosnap", "v10_snap30", best.rsplit("_", 1)[0] + "_nosnap"})
    print(f"\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    pv2 = gs[gs["variant"].isin(show)].groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4)
    print(pv2.to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
