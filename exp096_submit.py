# ═══════════════════════════════════════════════════════
# exp-096 제출: ASOS 관측편차를 DL 입력에 넣은 신 DL 멤버 (변경점 1개)
#
#   챔피언(v_a40s93p27_ens3_w65_g3s97, LB 0.64987) 대비 변경점 **DL 입력만**:
#     spd(348) -> spd + obs_mean + obs_dev_{stn}  (exp-035와 동일한 변경 유형)
#   GBM(ens3 20-seed 캐시)·pA·편향맵·신PP·그룹 스케일(0.98/0.98/0.97) 전부 동일.
#   DL만 5-seed 전체 재학습 (exp-058 제출 경로 미러링).
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp096_submit.py [--dl_seeds 5] [--wgbm 0.65]
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR,
                         apply_rolling_smoothing, apply_bias_correction_v2)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp033_dl import predict, GIDX
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR, DL_CFG
from exp039_dl_diverse import make_loss, train_one
from exp049_clean import merge_avail
from exp050_lambda_sweep import make_xy_lam
from exp096_dl_obs import build_obs_dev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
LAM = 0.05
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"   # pA·ws·dl_cf(구)
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
ENS_TEST = CACHE_DIR / "exp073_test_gbm_ens3_g20.parquet"        # 챔피언 GBM (ens3 20-seed)
DLOBS_TEST = CACHE_DIR / "exp096_test_dl_obs_d5.parquet"
CHAMP_FILE = "v_a40s93p27_ens3_w65_g3s97_submit.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--wgbm", type=float, default=0.65)
    ap.add_argument("--min-cov", type=float, default=0.98)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # === 데이터 (DL 입력 = spd 348 + 관측편차) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    vcols = list(cols) + spd_cols                          # 348
    train_df = merge_avail(train_df)

    # 관측 편차 — train·test 공통 빌더 (지점 집합을 train 기준으로 고정)
    obs_tr, ocols = build_obs_dev(train_df, args.min_cov)
    obs_te, ocols_te = build_obs_dev(test_df, args.min_cov)
    assert ocols == ocols_te, "train/test 지점 집합 불일치 — min-cov 재확인 필요"
    for name in ("train_df", "test_df"):
        df = locals()[name]
        df["tday"] = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
        df = df.merge(obs_tr if name == "train_df" else obs_te, on="tday", how="left")
        if name == "train_df":
            train_df = df
        else:
            test_df = df
    dl_cols = vcols + ocols
    cov_te = test_df["obs_mean"].notna().mean()
    print(f"DL 입력 {len(vcols)} -> {len(dl_cols)} | test 관측 커버리지 {cov_te:.3f}", flush=True)
    assert cov_te > 0.99, "test 관측 커버리지 부족"

    # === 신 DL 5-seed 전체 학습 -> test 예측 ===
    if DLOBS_TEST.exists() and not args.refresh:
        dlp = pd.read_parquet(DLOBS_TEST)
        print(f"DL test 캐시 로드: {DLOBS_TEST.name}", flush=True)
    else:
        import torch
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sca = StandardScaler()
        Xtr, gtr, ytr, wtr = make_xy_lam(train_df, dl_cols, imp, sca, LAM, fit=True)
        te_sorted = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        Xte = sca.transform(imp.transform(te_sorted[dl_cols].to_numpy(dtype=np.float32))).astype(np.float32)
        gte = te_sorted["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        loss_fn = make_loss("q60")
        preds = []
        for s in range(args.dl_seeds):
            m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
            preds.append(predict(m, Xte, gte))
            del m
            torch.cuda.empty_cache()
            print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
        dlp = pd.DataFrame({"forecast_id": te_sorted["forecast_id"].to_numpy(),
                            "kpx_group": te_sorted["kpx_group"].to_numpy(),
                            "dl_cf_obs": np.mean(preds, axis=0)})
        dlp.to_parquet(DLOBS_TEST, index=False)
        print(f"DL test 캐시 저장 -> {DLOBS_TEST.name}", flush=True)

    # === 무변경 성분 로드 + 블렌드·후처리 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    gp = pd.read_parquet(ENS_TEST)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left") \
               .merge(dlp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_ens3"].notna().all() and pred["dl_cf_obs"].notna().all()
    corr = np.corrcoef(pred["dl_cf_obs"], pred["dl_cf"])[0, 1]
    shift_dl = (pred["dl_cf_obs"] - pred["dl_cf"]).mean() * 100
    print(f"test DL: corr(신, 구 dl_cf) {corr:.4f} | 평균 이동 {shift_dl:+.3f}%cap", flush=True)

    w = args.wgbm
    parts = []
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (w * g["gbm_cf_ens3"].to_numpy() + (1 - w) * g["dl_cf_obs"].to_numpy()) * capacity,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], capacity * SNAP_TO)
        p = np.clip(p, capacity * CLIP_LO, capacity)
        p = np.clip(p * SCALES[group], 0, capacity)
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

    out = f"v_a40s93p27_ens3_dlobs_w{int(w*100)}_g3s97_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    shift = (sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()).mean()
    print(f"Saved {out} | vs 챔피언 평균 |차이| {diff.mean():.1f}kWh "
          f"({diff.mean()/21600*100:.3f}%cap) | 평균 이동 {shift:+.1f}kWh", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
