# ═══════════════════════════════════════════════════════
# exp-058 제출: 균형 제약 DL HPO (512×5·drop0.31·ep120·lr3.3e-4) — hpoG×hpob_w70
#
#   보드 챔피언(v_a40s93p27_hpo_w65) 대비 변경점 **두 개**:
#     ① DL 하이퍼: 384×3·drop0.15·lr1e-3 → 512×5·drop0.31·lr3.3e-4·wd9.5e-6 (균형 제약 우승)
#     ② 블렌드 w: 0.65 → 0.70 (게이트에서 Δn 양수인 셀)
#   GBM(hpo 20-seed)·pA·편향맵·신PP는 exp-056 제출 캐시 재사용 → DL 5-seed만 재학습.
#   게이트: w70 ΔCV +0.0013 / Δhold +0.0016 / Δn +0.0001 / Δf +0.0025 — 전 w 매그니튜드
#   통과·평탄 능선. Q3 −0.0037 (exp-042 전례도 Q3 음수 후 LB +0.0052).
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp058_submit.py --wgbm 0.70,0.65
# ═══════════════════════════════════════════════════════
import argparse
import json
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
from exp035_raw_dl import CACHE_DIR
from exp039_dl_diverse import make_loss, train_one
from exp049_clean import merge_avail
from exp050_lambda_sweep import make_xy_lam

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"   # pA·ws·dl_cf(구) 재사용
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
HPO_TEST = CACHE_DIR / "exp056_test_gbm_hpo_g20.parquet"          # GBM hpo 20-seed
BAL_JSON = CACHE_DIR / "exp058_dlhpo_bal.json"
DLB_TEST = CACHE_DIR / "exp058_test_dl_hpob_d5.parquet"
CHAMP_FILE = "v_a40s93p27_hpo_w65_submit.csv"
LAM = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--wgbm", type=str, default="0.70,0.65")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    w_list = [float(w) for w in args.wgbm.split(",") if w]
    t0 = time.time()
    cfg = json.loads(BAL_JSON.read_text())
    print(f"DL 균형 우승 cfg: {cfg}", flush=True)

    # === 데이터 (DL 입력 = spd 348, EC 미포함 — 게이트와 동일) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    vcols = list(cols) + spd_cols                          # 348
    train_df = merge_avail(train_df)

    # === 균형 DL 5-seed 전체 학습 → test 예측 ===
    if DLB_TEST.exists() and not args.refresh:
        dlp = pd.read_parquet(DLB_TEST)
        print(f"DL test 캐시 로드: {DLB_TEST.name}", flush=True)
    else:
        import torch
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sca = StandardScaler()
        Xtr, gtr, ytr, wtr = make_xy_lam(train_df, vcols, imp, sca, LAM, fit=True)
        te_sorted = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        Xte = sca.transform(imp.transform(te_sorted[vcols].to_numpy(dtype=np.float32))).astype(np.float32)
        gte = te_sorted["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        loss_fn = make_loss("q60")
        preds = []
        for s in range(args.dl_seeds):
            m = train_one(Xtr, gtr, ytr, wtr, cfg, loss_fn, s)
            preds.append(predict(m, Xte, gte))
            del m
            torch.cuda.empty_cache()
            print(f"  DL seed {s} 완료 ({time.time()-t0:.0f}s)", flush=True)
        dlp = pd.DataFrame({"forecast_id": te_sorted["forecast_id"].to_numpy(),
                            "kpx_group": te_sorted["kpx_group"].to_numpy(),
                            "dl_cf_bal": np.mean(preds, axis=0)})
        dlp.to_parquet(DLB_TEST, index=False)
        print(f"DL test 캐시 저장 → {DLB_TEST.name}", flush=True)

    # === 무변경 성분 로드 + 블렌드·후처리 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    gp = pd.read_parquet(HPO_TEST)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left") \
               .merge(dlp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_hpo"].notna().all() and pred["dl_cf_bal"].notna().all()
    corr = np.corrcoef(pred["dl_cf_bal"], pred["dl_cf"])[0, 1]
    print(f"신구 DL CF 상관: {corr:.4f}", flush=True)

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    for w in w_list:
        parts = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            g = pred.loc[pred["kpx_group"] == group]
            pA = g["pA"].to_numpy()
            p = apply_bias_correction_v2((w * g["gbm_cf_hpo"].to_numpy() + (1 - w) * g["dl_cf_bal"].to_numpy()) * capacity,
                                         g["ws"].to_numpy(), group, coarse_map)
            p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
            p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], capacity * SNAP_TO)
            parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                       "col": f"kpx_group_{group}",
                                       "pred": np.clip(p, capacity * CLIP_LO, capacity)}))
        sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
        sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")

        assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
        assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
        assert (sub["forecast_id"] == sample_submission["forecast_id"]).all(), "forecast_id 불일치"
        assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all(), "dtm 불일치"
        for col in TARGET_COLS:
            assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0, f"{col} 범위 이탈"

        out = f"v_a40s93p27_dlb_w{int(round(w*100))}_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        n93 = [(sub[c] >= CAPACITY_KWH[c] * 0.93).sum() for c in TARGET_COLS]
        print(f"[w{w:.2f}] Saved {out} | vs hpo_w65 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.2f}%cap) | 스냅 {n93} | 그룹평균 "
              + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
