# ═══════════════════════════════════════════════════════
# exp-023: v13 SCADA 파워커브 피처를 JMA 챔피언 위에서 재검증
#
# 배경: exp-015에서 v10TPS 기준 CV −0.0007 / LB +0.0007로 애매했던 SCADA 피처 5개를
#   새 챔피언(TPSJgJm, LB 0.63336) 위에서 이중 게이트로 재판정.
#   리크 방지: 캘리브레이션·파워커브 맵을 ≤2024-01-01(전 fold train의 공통 부분집합)
#   데이터로 1회 적합 → 전 행에 정적 적용 (run_folds 프레임과 호환).
# 변형: champ(132) / champV13(137) × nosnap/snap30G/snap30P
# 검증: 4-fold CV + 1년 홀드아웃 (표준 이중 게이트 — SCADA는 전 기간 소스)
# 실행: C:\miniconda3\python.exe -X utf8 exp023_scada_on_champ.py [--quick] [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import pandas as pd

from exp005_qopt import FOLDS_Q
from exp012_feature_v11 import build_v11
from exp015_scada_pc import SCADA_COLS, build_scada_hourly, fit_scada_maps, apply_scada_features
from exp019_multigrid import run_folds, HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAP_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")


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

    scada_h = build_scada_hourly()
    parts = []
    for group in [1, 2, 3]:
        g_df = train_df[train_df["kpx_group"] == group].copy()
        g_pre = g_df[g_df["forecast_kst_dtm"] <= MAP_CUTOFF]
        maps = fit_scada_maps(scada_h[group], g_pre, MAP_CUTOFF)
        g_df = apply_scada_features(g_df, maps)
        print(f"group{group}: SCADA 맵 {'적합' if maps else '불가(NaN)'}, "
              f"cf_mean 비결측 {g_df['scada_cf_mean'].notna().mean():.3f}")
        parts.append(g_df)
    train_df = pd.concat(parts, ignore_index=True)

    champ = col_sets["v10TPS"] + JG_COLS + JM_COLS
    variants_cols = {"champ": champ, "champV13": champ + SCADA_COLS}
    print("변형: " + ", ".join(f"{n}({len(c)})" for n, c in variants_cols.items()))
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    summary, gs = run_folds(folds, train_df, variants_cols, args, "CV")
    print("\n================ 4-fold CV (pooled) ================")
    piv = summary.assign(fv=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print(piv.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())
    print("\n--- 전체 (내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    if not args.quick:
        print("\n================ 1년 홀드아웃 게이트 (train ≤2023 → 2024 전체) ================")
        hA, _ = run_folds([HOLD_FOLD], train_df, variants_cols, args, "HOLD")
        piv_h = hA.assign(fv=hA["variant"].str.rsplit("_", n=1).str[0],
                          pm=hA["variant"].str.rsplit("_", n=1).str[1])
        print(piv_h.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
