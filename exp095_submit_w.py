# ═══════════════════════════════════════════════════════
# exp-095 제출: 블렌드 가중 w 하향 프로브 — 게이트 불가측 축 (LB 직접 측정)
#
#   챔피언(v_a40s93p27_ens3_w65_g3s97, LB 0.64987) 대비 **변경점 단 1개: w만**.
#   GBM(ens3 20-seed)·DL(cln03 5-seed)·pA·편향맵·신PP·그룹 스케일 전부 동일 캐시.
#   재학습 0 — 캐시 재조합만.
#
# 근거: exp-083이 w 0.65→0.80에서 **FiCR −0.00289**(게이트 추정 −0.0007의 ~4배)를 냈다.
#   뒤집어 읽으면 2025에서 DL의 FiCR 기여가 게이트 평가보다 훨씬 크다는 뜻 →
#   2025 최적 w가 0.65보다 **아래**일 가능성. 스케일 축(exp-059/064/071)과 같은
#   "게이트가 원리적으로 못 재는 축"이고, 그 축이 이 프로젝트 최대 이득처였다.
#
# ⚠ 주의 2가지
#   - w↓ = 상향편향 DL 지분 확대 = **예측 평균 상승**. "평균을 올리는 변경은 2025 역전"이
#     판별 규칙이므로 **출력되는 평균 이동을 보고 제출 여부를 판단**할 것.
#   - exp-090 실측 LB 노이즈 1σ ≈ 0.0006 — w0.60의 기대 효과(선형 외삽 시 ~+0.0004)는
#     1σ 안이라 **측정이 안 될 수 있다**. w0.55는 ~+0.0008로 측정 가능 구간이나 레버가 큼.
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp095_submit_w.py --w 0.60 0.55
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
from exp035_raw_dl import CACHE_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
ENS_TEST = CACHE_DIR / "exp073_test_gbm_ens3_g20.parquet"
CHAMP_FILE = "v_a40s93p27_ens3_w65_g3s97_submit.csv"


def build(pred, coarse_map, sample_submission, w):
    parts = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (w * g["gbm_cf_ens3"].to_numpy() + (1 - w) * g["dl_cf"].to_numpy()) * cap,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], cap * SNAP_TO)
        p = np.clip(p, cap * CLIP_LO, cap)
        p = np.clip(p * SCALES[group], 0, cap)
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
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, nargs="+", default=[0.60])
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-095 블렌드 가중 w 하향 프로브 (재학습 0) ###", flush=True)

    _, _, _, sample_submission = build_v11()
    gp = pd.read_parquet(ENS_TEST)
    pred = pd.read_parquet(E49_PRED).merge(gp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_ens3"].notna().all() and pred["dl_cf"].notna().all()
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)

    d = pred["dl_cf"].to_numpy() - pred["gbm_cf_ens3"].to_numpy()
    print(f"\n[2025 test] DL − GBM 평균 차이 {d.mean()*100:+.3f}%cap "
          f"(DL이 위쪽 = w↓ 시 평균 상승) | corr {np.corrcoef(pred['dl_cf'], pred['gbm_cf_ens3'])[0,1]:.4f}",
          flush=True)

    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    # 재현 대조 — w0.65가 챔피언 파일과 일치해야 파이프라인 무결
    chk = build(pred, coarse_map, sample_submission, 0.65)
    dmax = np.abs(chk[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()).max()
    print(f"[무결성] w0.65 재현 vs 챔피언 파일 최대 차이 {dmax:.6f} kWh "
          f"({'일치' if dmax < 1e-6 else '★불일치 — 중단 권고'})", flush=True)

    print(f"\n{'w':>6}{'평균 이동':>14}{'평균|차이|':>14}  파일", flush=True)
    for w in args.w:
        sub = build(pred, coarse_map, sample_submission, w)
        arr = sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()
        out = f"v_a40s93p27_ens3_w{int(round(w*100))}_g3s97_submit.csv"
        sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
            DATA_DIR / out, index=False, encoding="utf-8-sig")
        print(f"{w:>6.2f}{arr.mean()/21600*100:>+13.3f}%{np.abs(arr).mean()/21600*100:>13.3f}%  {out}",
              flush=True)
        for i, col in enumerate(TARGET_COLS):
            s = (sub[col].to_numpy() - champ[col].to_numpy()).mean() / CAPACITY_KWH[col] * 100
            print(f"        {col}: {s:+.3f}%cap", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
