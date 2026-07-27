# ═══════════════════════════════════════════════════════
# exp-043 제출 파일 생성: exp-042 test 예측 캐시 + 새 후처리 (재학습 0)
#
#   base는 LB 검증된 exp-042 제출(gspd GBM 20-seed + q60 DL 5-seed)과 **완전 동일**.
#   변경점은 후처리 3개뿐 (exp-043에서 이중 게이트 +0.0033/+0.0030, 두 지표 동반 상승):
#     alpha    0.50 → 0.40
#     snap_to  0.94 → 0.93
#     snap_thr 0.30 → 0.27
#   clip 0.16·편향맵·분류기·블렌드 성분은 그대로.
#
#   전제: exp042_submit.py가 저장한 test 예측 캐시 필요
#     open (1)/external/exp042_test_pred_q60_g20_d5.parquet
#     open (1)/external/exp042_test_bias_q60_g20_d5.pkl
#
#   ⚠ 이 스크립트는 파일만 만든다. **제출 슬롯 소비는 사용자 판단.**
# 실행: C:\miniconda3\python.exe -X utf8 exp043_submit.py [--wgbm 0.70,0.65]
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR,
                         apply_rolling_smoothing, apply_bias_correction_v2)
from exp035_raw_dl import CACHE_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# exp-043 최적 후처리 (c16·al40·sn93·pt27)
ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
TAG = "q60_g20_d5"
CHAMP_FILE = "v_c16a50_gspd_q60_w65_submit.csv"   # 현 LB 챔피언 (비교용)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wgbm", type=str, default="0.70,0.65")
    args = ap.parse_args()
    w_list = [float(w) for w in args.wgbm.split(",") if w]
    t0 = time.time()

    pred_path = CACHE_DIR / f"exp042_test_pred_{TAG}.parquet"
    bias_path = CACHE_DIR / f"exp042_test_bias_{TAG}.pkl"
    if not pred_path.exists() or not bias_path.exists():
        print(f"캐시 없음: {pred_path.name} / {bias_path.name}")
        print("  exp042_submit.py를 먼저 실행하세요 (캐시 생성 포함).")
        return
    pred = pd.read_parquet(pred_path)
    with open(bias_path, "rb") as f:
        coarse_map = pickle.load(f)
    print(f"test 예측 캐시 로드: {pred.shape} | 후처리 al{int(ALPHA*100)}·sn{int(SNAP_TO*100)}·pt{int(SNAP_THR*100)}·cl{int(CLIP_LO*100)}")

    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")

    for w in w_list:
        parts = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            g = pred.loc[pred["kpx_group"] == group]
            pA = g["pA"].to_numpy()
            p = apply_bias_correction_v2((w * g["gbm_cf"].to_numpy() + (1 - w) * g["dl_cf"].to_numpy()) * capacity,
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

        out = f"v_a40s93p27_gspd_q60_w{int(round(w*100))}_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
        n_chg = int((diff > 0.5).sum())
        print(f"[w{w:.2f}] Saved {out} | vs 챔피언(w65·구PP) 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.2f}%cap), 변경 {n_chg}/{diff.size}칸")
    print(f"\n총 소요: {(time.time()-t0):.0f}초")


if __name__ == "__main__":
    main()
