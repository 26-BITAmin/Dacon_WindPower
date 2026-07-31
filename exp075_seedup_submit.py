# ═══════════════════════════════════════════════════════
# exp-075 제출: 시드 확대 리빌드 — GBM 20→30시드·DL 5→15시드 (순수 분산 감소)
#
#   챔피언(v_a40s93p27_hpo_w65_g3s97, LB 0.64973) 대비 변경점 = 시드 수뿐.
#   피처·하이퍼·가중·PP·편향맵·pA·그룹별 스케일(0.98, 0.98, 0.97) 전부 동일.
#   기존 캐시(20시드 GBM 평균·5시드 DL 평균)에 신규 시드 평균을 가중 결합.
#   평균 이동 없는 mild 변경 — 역전 위험 프로파일 아님.
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: --stage dl / --stage gbm (병렬 가능: GPU vs CPU) → --stage build
# ═══════════════════════════════════════════════════════
import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR, make_sample_weight,
                         apply_rolling_smoothing, apply_bias_correction_v2, prepare_matrices)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR, DL_CFG
from exp049_clean import merge_avail, avail_factor, make_xy_clean
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp056_hpo import RESULT_JSON

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}      # exp-071 확정 그룹별 스케일
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
HPO_TEST = CACHE_DIR / "exp056_test_gbm_hpo_g20.parquet"
DL_NEW = CACHE_DIR / "exp075_test_dl_s5_14.parquet"
GBM_NEW = CACHE_DIR / "exp075_test_gbm_s20_29.parquet"
CHAMP_FILE = "v_a40s93p27_hpo_w65_g3s97_submit.csv"
W_GBM = 0.65


def base_frames():
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols132 = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    train_df = merge_avail(train_df)
    return train_df, test_df, cols132, spd_cols, sample_submission


def stage_dl(args):
    import torch
    from exp033_dl import GIDX, predict
    from exp039_dl_diverse import make_loss, train_one
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    train_df, test_df, cols132, spd_cols, _ = base_frames()
    ext_cols = list(cols132) + spd_cols                     # 348
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(train_df, ext_cols, imp, sca, "cln03", fit=True)
    Xte = sca.transform(imp.transform(test_df[ext_cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gte = test_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds = []
    for s in range(5, 5 + args.dl_add):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xte, gte))
        del m
        torch.cuda.empty_cache()
        print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
    out = test_df[["forecast_id", "kpx_group"]].copy()
    out["dl_cf_new"] = np.mean(preds, axis=0)
    out.to_parquet(DL_NEW, index=False)
    print(f"DL 신규 {args.dl_add}시드 저장 → {DL_NEW.name} ({(time.time()-t0)/60:.1f}분)", flush=True)


def stage_gbm(args):
    import lightgbm as lgb
    import xgboost as xgbm
    t0 = time.time()
    best = json.loads(RESULT_JSON.read_text())
    train_df, test_df, cols132, spd_cols, _ = base_frames()
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    for df in (train_df, test_df):
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
    use_cols = list(cols132) + spd_cols + EC_COLS           # 352
    rows = []
    for group in [1, 2, 3]:
        tg = time.time()
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = train_df.loc[(train_df["kpx_group"] == group) & train_df["target"].notna()].copy()
        te_g = test_df.loc[test_df["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        X_tr, X_te = prepare_matrices(tr_g, te_g, use_cols)
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        seeds = range(20, 20 + args.gbm_add)
        pl = np.mean([lgb.LGBMRegressor(n_estimators=best["lgb_n"], objective="mae",
                      random_state=s, n_jobs=args.jobs, verbose=-1, **best["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in seeds], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=best["xgb_n"], objective="reg:absoluteerror",
                      tree_method="hist", random_state=s, n_jobs=args.jobs, verbosity=0, **best["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_te) for s in seeds], axis=0)
        rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                  "kpx_group": group,
                                  "gbm_cf_new": (0.5 * pl + 0.5 * px) / cap}))
        print(f"  g{group} GBM 신규 {args.gbm_add}시드 완료 ({time.time()-tg:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(GBM_NEW, index=False)
    print(f"GBM 신규 저장 → {GBM_NEW.name} ({(time.time()-t0)/60:.1f}분)", flush=True)


def stage_build(args):
    t0 = time.time()
    _, _, _, sample_submission = build_v11()
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    gp = pd.read_parquet(HPO_TEST)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left")
    pred = pred.merge(pd.read_parquet(DL_NEW), on=["forecast_id", "kpx_group"], how="left")
    pred = pred.merge(pd.read_parquet(GBM_NEW), on=["forecast_id", "kpx_group"], how="left")
    for c in ("gbm_cf_hpo", "dl_cf_new", "gbm_cf_new"):
        assert pred[c].notna().all(), c
    # 가중 결합: 기존 평균 × 기존 시드 수 + 신규 평균 × 신규 시드 수
    pred["dl15"] = (5 * pred["dl_cf"] + args.dl_add * pred["dl_cf_new"]) / (5 + args.dl_add)
    pred["gbm30"] = (20 * pred["gbm_cf_hpo"] + args.gbm_add * pred["gbm_cf_new"]) / (20 + args.gbm_add)
    for a, b in (("dl15", "dl_cf"), ("gbm30", "gbm_cf_hpo")):
        d = (pred[a] - pred[b]).abs().mean() * 100
        s = (pred[a] - pred[b]).mean() * 100
        print(f"  {a} vs {b}: 평균 |차이| {d:.3f}%cap | 평균 이동 {s:+.4f}%cap", flush=True)

    parts = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (W_GBM * g["gbm30"].to_numpy() + (1 - W_GBM) * g["dl15"].to_numpy()) * cap,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], cap * SNAP_TO)
        p = np.clip(p, cap * CLIP_LO, cap)
        p = np.clip(p * SCALES[group], 0, cap)              # 그룹별 스케일 (exp-071)
        parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{group}", "pred": p}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")
    assert sub.shape[0] == sample_submission.shape[0]
    assert sub[TARGET_COLS].isna().sum().sum() == 0
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all()
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all()
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0

    out = "v_a40s93p27_hpo30d15_w65_g3s97_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    print(f"Saved {out} | vs 챔피언(g3s97) 평균 |차이| {diff.mean():.1f}kWh "
          f"({diff.mean()/21600*100:.3f}%cap) ({(time.time()-t0)/60:.1f}분)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dl", "gbm", "build"], required=True)
    ap.add_argument("--dl_add", type=int, default=10)
    ap.add_argument("--gbm_add", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    {"dl": stage_dl, "gbm": stage_gbm, "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()
