# ═══════════════════════════════════════════════════════
# exp-020: 외부 예보 피처 — JMA GSM/MSM (Open-Meteo Previous Runs, previous_day2)
#
# 배경:
#   - 역대 LB +0.001 이상 개선은 피처 블록 추가(exp-012)가 유일 → 모델 다양성 주입
#   - 현 파이프라인 예보 소스는 KMA LDAPS + NCEP GFS 2개. JMA GSM(~20km, 2022~)과
#     MSM(5km, 2023~)은 규정 적합(전날 13:00 이전 발행)인 유일한 전 기간 신규 소스
#   - 신호 확인: MSM ws↔그룹1 발전량 corr 0.676, GSM 0.567, GSM-MSM 상호 corr 0.622
#   - 데이터: exp020_fetch_jma.py → open (1)/external/jma_prevruns.csv (CC-BY 4.0)
#
# 블록:
#   Jg. GSM 6개: ws, wd sin/cos, t2m, sp, |ws−ldaps_ws_10m|
#   Jm. MSM 4개: ws, wd sin/cos, |ws−ldaps_ws_10m|  (2022년은 결측 → median impute)
# 변형: v10TPS / TPSJg / TPSJm / TPSJgJm × nosnap/snap30G/snap30P
# 검증: 공식 4-fold CV + 1년 홀드아웃 게이트 (exp-019 프레임 재사용)
# 실행: C:\miniconda3\python.exe -X utf8 exp020_jma_features.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import DATA_DIR, FOLDS_Q
from exp012_feature_v11 import build_v11
from exp019_multigrid import run_folds, HOLD_FOLD

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_jma_frame():
    j = pd.read_csv(DATA_DIR / "external" / "jma_prevruns.csv", encoding="utf-8-sig",
                    parse_dates=["forecast_kst_dtm"])
    frames = []
    for model, tag in [("jma_gsm", "jgsm"), ("jma_msm", "jmsm")]:
        sub = j[j["model"] == model].copy()
        wd_rad = np.deg2rad(sub["wind_direction_10m_d2"])
        out = pd.DataFrame({
            "forecast_kst_dtm": sub["forecast_kst_dtm"],
            "kpx_group": sub["kpx_group"],
            f"{tag}_ws": sub["wind_speed_10m_d2"],
            f"{tag}_wdsin": np.sin(wd_rad),
            f"{tag}_wdcos": np.cos(wd_rad),
        })
        if model == "jma_gsm":
            out[f"{tag}_t2m"] = sub["temperature_2m_d2"]
            out[f"{tag}_sp"] = sub["surface_pressure_d2"]
        frames.append(out)
    merged = frames[0].merge(frames[1], on=["forecast_kst_dtm", "kpx_group"], how="outer")
    return merged


JG_COLS = ["jgsm_ws", "jgsm_wdsin", "jgsm_wdcos", "jgsm_t2m", "jgsm_sp", "jgsm_dis"]
JM_COLS = ["jmsm_ws", "jmsm_wdsin", "jmsm_wdcos", "jmsm_dis"]


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
    jma = build_jma_frame()
    train_df = train_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["jgsm_dis"] = (train_df["jgsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df["jmsm_dis"] = (train_df["jmsm_ws"] - train_df["ldaps_ws_10m"]).abs()

    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)[["jgsm_ws", "jmsm_ws"]].apply(lambda d: d.notna().mean().round(3))
    print("연도별 JMA 커버리지:\n", cov.to_string())

    cols_tps = col_sets["v10TPS"]
    variants_cols = {
        "v10TPS": cols_tps,
        "TPSJg": cols_tps + JG_COLS,
        "TPSJm": cols_tps + JM_COLS,
        "TPSJgJm": cols_tps + JG_COLS + JM_COLS,
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
