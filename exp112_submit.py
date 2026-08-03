# ═══════════════════════════════════════════════════════
# exp-112 (B단계) 제출 빌드: dl_hpo_L0 — 레벨 제거 무제약 DL 파트너
#
#   기준 제출물 `v_a40s93p27_hpo_w65_g3s97`(LB 0.64973) 대비 변경점 **한 개뿐**:
#     DL 교체 dl_cln03(384×3·drop0.15) → **dl_hpo_L0** =
#       exp-058 무제약 Optuna 우승(640×4·drop0.349·ep80·lr4.3e-4) 5-seed 재학습 후
#       **그룹별 평균을 챔피언 DL의 테스트 평균에 정렬**(상수 차감, 라벨 미사용).
#   GBM(hpo 20-seed)·pA·편향맵·신PP·w0.65·그룹 스케일(0.98/0.98/0.97) 전부 캐시 재사용.
#
#   근거: exp-100(전 쌍 교차) → exp-101(레벨/정보 분리) → exp-102(28멤버 통제)
#     → exp-103(Q4 제외 +0.0014 유지) → exp-112A(보드 base 검증).
#     최종 게이트: hpo×hpo_L0_w65 ΔCV +0.0025/Δhold +0.0019/Δn −0.0006/평균이동 ≈0.
#     exp-058 원본(dl_hpob)의 LB 역전 사유(상향 이동)는 레벨 정렬로 소거 —
#     여기서는 무제약 우승자를 쓰되 **테스트 분포에서 평균이동 0을 강제**한다.
#
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 exp112_submit.py
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
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"   # pA·ws·dl_cf(챔피언 DL)
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
HPO_TEST = CACHE_DIR / "exp056_test_gbm_hpo_g20.parquet"          # GBM hpo 20-seed
CFG_JSON = CACHE_DIR / "exp058_dlhpo_best.json"                   # 무제약 우승 (bal 아님!)
DLH_TEST = CACHE_DIR / "exp112_test_dl_hpo_d5.parquet"
CHAMP_FILE = "v_a40s93p27_hpo_w65_g3s97_submit.csv"               # 동일 base 기준 제출물
BOARD_FILE = "v_a40s93p27_ens3_w65_g3s97_submit.csv"              # 보드 챔피언 (참고)
# 2024 검증 캐시에서의 그룹별 이동량 (exp-101, 참고 출력용)
TRAIN_SHIFT = {1: 0.0225, 2: 0.0237, 3: 0.0205}
W_GBM = 0.65
LAM = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl_seeds", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=str, default="v_a40s93p27_hpoL0_w65_g3s97_submit.csv")
    args = ap.parse_args()
    t0 = time.time()
    cfg = json.loads(CFG_JSON.read_text())
    print(f"### exp-112B 제출 빌드 — dl_hpo_L0 ###", flush=True)
    print(f"DL 무제약 우승 cfg: {cfg}", flush=True)

    # === 데이터 (DL 입력 = spd 348 — exp-058 게이트와 동일) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    vcols = list(cols) + spd_cols                          # 348
    train_df = merge_avail(train_df)

    # === 무제약 DL 5-seed 전체 학습 → test 예측 ===
    if DLH_TEST.exists() and not args.refresh:
        dlp = pd.read_parquet(DLH_TEST)
        print(f"DL test 캐시 로드: {DLH_TEST.name}", flush=True)
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
                            "dl_cf_hpo": np.mean(preds, axis=0)})
        dlp.to_parquet(DLH_TEST, index=False)
        print(f"DL test 캐시 저장 → {DLH_TEST.name}", flush=True)

    # === 성분 로드 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    gp = pd.read_parquet(HPO_TEST)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left") \
               .merge(dlp, on=["forecast_id", "kpx_group"], how="left")
    assert pred["gbm_cf_hpo"].notna().all() and pred["dl_cf_hpo"].notna().all()

    # === 레벨 정렬 (핵심 단계, 라벨 미사용) ===
    # dl_hpo_L0 = dl_hpo − (그룹별 테스트 평균 차이 vs 챔피언 DL)
    # → 2025 분포에서 그룹별 평균이동 0을 강제. 참고로 2024 검증 이동량과 병기.
    print(f"\n[레벨 정렬] 그룹별 이동량 (테스트 기준 / 2024 검증 기준)", flush=True)
    pred["dl_cf_hpoL0"] = np.nan
    for group in [1, 2, 3]:
        m = pred["kpx_group"] == group
        shift = pred.loc[m, "dl_cf_hpo"].mean() - pred.loc[m, "dl_cf"].mean()
        pred.loc[m, "dl_cf_hpoL0"] = pred.loc[m, "dl_cf_hpo"] - shift
        print(f"  g{group}: test {shift:+.4f} CF ({shift*100:+.2f}%cap) | "
              f"2024검증 {TRAIN_SHIFT[group]:+.4f}", flush=True)
    assert pred["dl_cf_hpoL0"].notna().all()

    # === 블렌드 → PP → 그룹 스케일 → 제출 ===
    parts = []
    for group in [1, 2, 3]:
        capacity = CAPACITY_KWH[f"kpx_group_{group}"]
        g = pred.loc[pred["kpx_group"] == group]
        pA = g["pA"].to_numpy()
        p = apply_bias_correction_v2(
            (W_GBM * g["gbm_cf_hpo"].to_numpy() + (1 - W_GBM) * g["dl_cf_hpoL0"].to_numpy()) * capacity,
            g["ws"].to_numpy(), group, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], capacity * SNAP_TO)
        p = np.clip(p, capacity * CLIP_LO, capacity)
        p = np.clip(p * SCALES[group], 0, capacity)
        parts.append(pd.DataFrame({"forecast_id": g["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{group}", "pred": p}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")

    assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
    assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all(), "forecast_id 불일치"
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all(), "dtm 불일치"
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0, f"{col} 범위 이탈"

    out = args.out
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")

    # 기존 이름의 파일이 있으면 내용 대조 (동시 세션 덮어쓰기 검증)
    for prev_name in ("v_a40s93p27_hpoL0_w65_g3s97_submit.csv",
                      "v_a40s93p27_ens3_hpoL0_w65_g3s97_submit.csv"):
        p = DATA_DIR / prev_name
        if p.exists() and prev_name != out:
            prev = pd.read_csv(p, encoding="utf-8-sig")
            md = np.abs(sub[TARGET_COLS].to_numpy() - prev[TARGET_COLS].to_numpy()).max()
            print(f"[대조] {prev_name}: 최대 |차이| {md:.6f} kWh "
                  f"→ {'동일 내용' if md < 1e-6 else '다른 내용'}", flush=True)

    for ref_name, label in [(CHAMP_FILE, "동일base 챔피언(hpo_g3s97)"),
                            (BOARD_FILE, "보드 챔피언(ens3_g3s97)")]:
        ref = pd.read_csv(DATA_DIR / ref_name, encoding="utf-8-sig")
        diff = np.abs(sub[TARGET_COLS].to_numpy() - ref[TARGET_COLS].to_numpy())
        shift = (sub[TARGET_COLS].to_numpy() - ref[TARGET_COLS].to_numpy()).mean()
        print(f"vs {label}: 평균 |차이| {diff.mean():.1f}kWh "
              f"({diff.mean()/21600*100:.3f}%cap) | 평균 이동 {shift:+.1f}kWh", flush=True)
    n93 = [(sub[c] >= CAPACITY_KWH[c] * 0.93 * SCALES[i+1]).sum() for i, c in enumerate(TARGET_COLS)]
    print(f"Saved {out} | 스냅 도달 {n93} | 그룹평균 "
          + ", ".join(f"{c[-1]}={sub[c].mean():.0f}" for c in TARGET_COLS), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
