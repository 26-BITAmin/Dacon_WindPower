# ═══════════════════════════════════════════════════════
# exp-019: 다중 격자 확장 (새 정보 주입) + 1년 홀드아웃 게이트
#
# 배경 (MODEL.md):
#   - 역대 LB +0.001 이상 개선은 피처 블록 추가(exp-012)가 유일 → "새 정보" 축 확장
#   - 현재 메인 피처는 그룹당 최근접 1격자뿐. LDAPS 16격자 중 15개의 변수 대부분이
#     미사용 (S블록은 ws 집계 3개뿐). 격자별 개별값·IDW 보간은 미시도.
#   - exp-018: 2024 분기 CV만으로는 LB 일반화 판정 불가 → 이번부터 exp-006 방식
#     1년 홀드아웃(train ≤2023 → 2024 전체)을 게이트로 병행 리포트.
#
# 피처 블록:
#   Gl. LDAPS 다중 격자 9개: 그룹별 최근접 4격자 ws_10m 개별값(4) + 최근접 2격자
#       wd sin/cos(4) + IDW(1/d) 가중 ws(1)
#   Gg. GFS 다중 격자 9개: 최근접 4격자 ws_100m 개별값(4) + ws_850hPa 개별값(4) + IDW ws100(1)
# 변형: v10TPS(control) / +Gl / +Gg / +GlGg × 후처리 nosnap/snap30G/snap30P
# 실행: C:\miniconda3\python.exe -X utf8 exp019_multigrid.py [--quick] [--jobs N]
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
    metric, metric_single_group, make_sample_weight,
    apply_rolling_smoothing, apply_bias_correction_v2, lgbm_params,
    prepare_matrices, haversine, dms_to_decimal,
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
N_MG = 4

HOLD_FOLD = {"name": "hold_2024full", "train_end": pd.Timestamp("2024-01-01 00:00:00"),
             "valid_start": pd.Timestamp("2024-01-01 01:00:00"), "valid_end": pd.Timestamp("2025-01-01 00:00:00")}


def _centroids():
    raw = pd.read_excel(DATA_DIR / "info.xlsx", sheet_name="info", header=None)
    info = raw.iloc[4:21].copy()
    info.columns = raw.iloc[3]
    info = info.reset_index(drop=True)
    info = info.rename(columns={"좌표(Google)": "coord", "KPX그룹": "kpx_group"})
    info["kpx_group"] = info["kpx_group"].ffill().astype(int)
    coords = info["coord"].apply(dms_to_decimal)
    info["lat"] = coords.apply(lambda x: x[0])
    info["lon"] = coords.apply(lambda x: x[1])
    return info.groupby("kpx_group")[["lat", "lon"]].mean()


def build_multigrid_frames():
    """그룹별 최근접 4격자의 개별 풍속·풍향 + IDW 보간 (train/test 각각)"""
    centroids = _centroids()
    out = {}
    for split, ld_path, gf_path in [("train", TRAIN_DIR / "ldaps_train.csv", TRAIN_DIR / "gfs_train.csv"),
                                    ("test", TEST_DIR / "ldaps_test.csv", TEST_DIR / "gfs_test.csv")]:
        ld = pd.read_csv(ld_path, encoding="utf-8-sig",
                         usecols=["forecast_kst_dtm", "grid_id", "latitude", "longitude",
                                  "heightAboveGround_10_10u", "heightAboveGround_10_10v"])
        ld["forecast_kst_dtm"] = pd.to_datetime(ld["forecast_kst_dtm"])
        ld["ws"] = np.hypot(ld["heightAboveGround_10_10u"], ld["heightAboveGround_10_10v"])
        wd = np.arctan2(ld["heightAboveGround_10_10v"], ld["heightAboveGround_10_10u"])
        ld["wd_sin"], ld["wd_cos"] = np.sin(wd), np.cos(wd)
        ld_grids = ld[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")

        gf = pd.read_csv(gf_path, encoding="utf-8-sig",
                         usecols=["forecast_kst_dtm", "grid_id", "latitude", "longitude",
                                  "heightAboveGround_100_100u", "heightAboveGround_100_100v",
                                  "isobaricInhPa_850_u", "isobaricInhPa_850_v"])
        gf["forecast_kst_dtm"] = pd.to_datetime(gf["forecast_kst_dtm"])
        gf["ws100"] = np.hypot(gf["heightAboveGround_100_100u"], gf["heightAboveGround_100_100v"])
        gf["ws850"] = np.hypot(gf["isobaricInhPa_850_u"], gf["isobaricInhPa_850_v"])
        gf_grids = gf[["grid_id", "latitude", "longitude"]].drop_duplicates().set_index("grid_id")

        frames = []
        for group in [1, 2, 3]:
            clat, clon = centroids.loc[group, "lat"], centroids.loc[group, "lon"]

            d_ld = haversine(clat, clon, ld_grids["latitude"], ld_grids["longitude"]).nsmallest(N_MG)
            base = None
            idw_num, idw_den = None, 0.0
            for i, (gid, dist) in enumerate(d_ld.items()):
                sub = ld.loc[ld["grid_id"] == gid, ["forecast_kst_dtm", "ws", "wd_sin", "wd_cos"]].copy()
                ren = {"ws": f"mg_l10_ws_{i}"}
                if i < 2:
                    ren.update({"wd_sin": f"mg_l10_wdsin_{i}", "wd_cos": f"mg_l10_wdcos_{i}"})
                else:
                    sub = sub.drop(columns=["wd_sin", "wd_cos"])
                sub = sub.rename(columns=ren)
                base = sub if base is None else base.merge(sub, on="forecast_kst_dtm", how="outer")
                w = 1.0 / max(dist, 0.1)
                contrib = base[f"mg_l10_ws_{i}"] * w
                idw_num = contrib if idw_num is None else idw_num + contrib
                idw_den += w
            base["mg_l10_ws_idw"] = idw_num / idw_den

            d_gf = haversine(clat, clon, gf_grids["latitude"], gf_grids["longitude"]).nsmallest(N_MG)
            idw_num, idw_den = None, 0.0
            for i, (gid, dist) in enumerate(d_gf.items()):
                sub = gf.loc[gf["grid_id"] == gid, ["forecast_kst_dtm", "ws100", "ws850"]].copy()
                sub = sub.rename(columns={"ws100": f"mg_g100_ws_{i}", "ws850": f"mg_g850_ws_{i}"})
                base = base.merge(sub, on="forecast_kst_dtm", how="left")
                w = 1.0 / max(dist, 0.1)
                contrib = base[f"mg_g100_ws_{i}"] * w
                idw_num = contrib if idw_num is None else idw_num + contrib
                idw_den += w
            base["mg_g100_ws_idw"] = idw_num / idw_den

            base["kpx_group"] = group
            frames.append(base)
        out[split] = pd.concat(frames, ignore_index=True)
    return out


GL_COLS = [f"mg_l10_ws_{i}" for i in range(N_MG)] + \
          [f"mg_l10_wd{t}_{i}" for i in range(2) for t in ("sin", "cos")] + ["mg_l10_ws_idw"]
GG_COLS = [f"mg_g100_ws_{i}" for i in range(N_MG)] + [f"mg_g850_ws_{i}" for i in range(N_MG)] + ["mg_g100_ws_idw"]


def run_folds(folds, train_df, variants_cols, args, tag):
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
        print(f"[{tag}:{fold['name']}] ({time.time() - tf:.0f}s) top4: "
              + ", ".join(f"{v}={s:.4f}" for v, s in top), flush=True)

    pooled_answer = pd.concat(all_answer, ignore_index=True)
    rows = []
    for v in names:
        pooled_pred = pd.concat(all_pred[v], ignore_index=True)
        s, n, f = metric(pooled_answer, pooled_pred)
        rows.append({"variant": v, "pooled_Score": s, "1-NMAE": n, "FICR": f,
                     "fold_mean": np.mean(fold_scores[v]), "fold_std": np.std(fold_scores[v], ddof=1) if len(folds) > 1 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(group_scores)


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
    mg = build_multigrid_frames()
    train_df = train_df.merge(mg["train"], on=["forecast_kst_dtm", "kpx_group"], how="left")
    cols_tps = col_sets["v10TPS"]
    variants_cols = {
        "v10TPS": cols_tps,
        "TPSGl": cols_tps + GL_COLS,
        "TPSGg": cols_tps + GG_COLS,
        "TPSGlGg": cols_tps + GL_COLS + GG_COLS,
    }
    print("변형: " + ", ".join(f"{n}({len(c)})" for n, c in variants_cols.items()))
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    summary, gs = run_folds(folds, train_df, variants_cols, args, "CV")
    print("\n================ 4-fold CV (pooled) ================")
    piv = summary.assign(fv=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print("\n--- 변형 × 후처리 (pooled Score) ---")
    print(piv.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    best = summary.sort_values("pooled_Score", ascending=False).iloc[0]["variant"]
    show = sorted({best, "v10TPS_nosnap", "v10TPS_snap30P", best.rsplit("_", 1)[0] + "_nosnap"})
    print(f"\n--- 그룹별 fold 평균 (1-NMAE / FiCR) ---")
    print(gs[gs["variant"].isin(show)].groupby(["variant", "group"])[["1-NMAE", "FICR"]].mean().round(4).to_string())

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        h_summary, _ = run_folds([HOLD_FOLD], train_df, variants_cols, args, "HOLD")
        piv_h = h_summary.assign(fv=h_summary["variant"].str.rsplit("_", n=1).str[0],
                                 pm=h_summary["variant"].str.rsplit("_", n=1).str[1])
        print("\n--- 변형 × 후처리 (2024 전체 Score) ---")
        print(piv_h.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())
        print("\n--- 전체 (내림차순) ---")
        print(h_summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
