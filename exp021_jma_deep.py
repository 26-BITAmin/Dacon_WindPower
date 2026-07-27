# ═══════════════════════════════════════════════════════
# exp-021: JMA 심화 — 시간 문맥(Jt) / MSM 이웃 격자(Jn) / GSM 예보 추세(Jd)
#
# 배경: exp-020(JMA GSM/MSM 기본 블록)이 CV +0.0043 / 홀드아웃 +0.0046 / LB +0.0044로
#   세션 첫 1:1 전이 달성 → 검증된 소스의 심화 확장.
#   - Jt. JMA ws에 T블록식 시간 문맥 (exp-012에서 LDAPS ws에 유효했던 레시피):
#         jgsm_ws·jmsm_ws 각각 r3 mean/std, r7 mean, 램프 d1/dm1 → 10개
#   - Jn. MSM 이웃 4점 (N/S/E/W, exp021_fetch_jma2.py): 개별 4 + 5점(중심 포함)
#         mean/std → 6개
#   - Jd. GSM 예보 추세: ws_d3 + 수정폭(d2−d3) → 2개 (MSM은 day3 미제공 확인)
# 변형: 챔피언 TPSJgJm(132, control) / +Jt / +Jn / +Jd / +all × nosnap/snap30G/snap30P
# 검증: 공식 4-fold CV + 1년 홀드아웃 게이트 (이중 게이트 통과 시에만 제출 후보)
# 실행: C:\miniconda3\python.exe -X utf8 exp021_jma_deep.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import DATA_DIR, FOLDS_Q
from exp012_feature_v11 import build_v11
from exp019_multigrid import run_folds, HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def add_jma_deep(df):
    """exp-020 기본 JMA 병합 위에 Jt/Jn/Jd 블록 추가 (train/test 동형)"""
    # Jn. MSM 이웃 격자
    nb = pd.read_csv(DATA_DIR / "external" / "jma_msm_neighbors.csv", encoding="utf-8-sig",
                     parse_dates=["forecast_kst_dtm"])
    nb_w = nb.pivot_table(index=["forecast_kst_dtm", "kpx_group"], columns="dir", values="ws_d2").reset_index()
    nb_w.columns = ["forecast_kst_dtm", "kpx_group"] + [f"jn_ws_{c}" for c in nb_w.columns[2:]]
    df = df.merge(nb_w, on=["forecast_kst_dtm", "kpx_group"], how="left")
    five = df[["jmsm_ws", "jn_ws_E", "jn_ws_N", "jn_ws_S", "jn_ws_W"]]
    df["jn_mean"] = five.mean(axis=1)
    df["jn_std"] = five.std(axis=1)

    # Jd. GSM day3 추세
    d3 = pd.read_csv(DATA_DIR / "external" / "jma_day3.csv", encoding="utf-8-sig",
                     parse_dates=["forecast_kst_dtm"])
    d3g = d3[d3["model"] == "jma_gsm"].rename(columns={"ws_d3": "jd_gsm_ws_d3"})[
        ["forecast_kst_dtm", "kpx_group", "jd_gsm_ws_d3"]]
    df = df.merge(d3g, on=["forecast_kst_dtm", "kpx_group"], how="left")
    df["jd_gsm_rev"] = df["jgsm_ws"] - df["jd_gsm_ws_d3"]

    # Jt. JMA ws 시간 문맥 (그룹별 시계열 정렬 후 rolling/램프)
    df = df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
    for src, p in [("jgsm_ws", "jt_g"), ("jmsm_ws", "jt_m")]:
        g = df.groupby("kpx_group")[src]
        df[f"{p}_r3_mean"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        df[f"{p}_r3_std"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).std())
        df[f"{p}_r7_mean"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
        df[f"{p}_d1"] = g.transform(lambda s: s.diff(1))
        df[f"{p}_dm1"] = g.transform(lambda s: s.diff(-1))
    return df


JT_COLS = [f"jt_{m}_{c}" for m in ("g", "m") for c in ("r3_mean", "r3_std", "r7_mean", "d1", "dm1")]
JN_COLS = ["jn_ws_E", "jn_ws_N", "jn_ws_S", "jn_ws_W", "jn_mean", "jn_std"]
JD_COLS = ["jd_gsm_ws_d3", "jd_gsm_rev"]


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
    train_df = add_jma_deep(train_df)

    champ = col_sets["v10TPS"] + JG_COLS + JM_COLS
    variants_cols = {
        "champ": champ,
        "champJt": champ + JT_COLS,
        "champJn": champ + JN_COLS,
        "champJd": champ + JD_COLS,
        "champAll": champ + JT_COLS + JN_COLS + JD_COLS,
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
    show = sorted({best, "champ_nosnap", "champ_snap30P", best.rsplit("_", 1)[0] + "_nosnap"})
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
