# ═══════════════════════════════════════════════════════
# exp-022: 2024+ 부분 커버 외부 예보 — ECMWF IFS 0.25° / ICON global
#
# 배경: 전 기간(2022~) prev-runs 소스는 JMA GSM이 유일함을 재확인(2026-07-23 프로브).
#   ECMWF·ICON은 2024-01부터만 존재 → 2022-23은 결측(median impute).
#   평가 문제: 표준 홀드아웃(train ≤2023)은 이 피처의 학습 데이터가 전무 → 측정 불가.
#   보강: ① 표준 4-fold CV (Q3·Q4는 train에 2024 데이터 3~9개월 포함 → 부분 증거)
#         ② 표준 1년 홀드아웃 (기록용 — 무효과가 정상)
#         ③ 신규 2024-H2 게이트: train ≤2024-06 → 2024 하반기 검증 (피처 6개월 학습)
#   판정: Q3·Q4 fold와 H2 게이트가 일관되게 +이면 채택 후보 (실제 제출 학습셋은
#   2024년 12개월을 포함하므로 게이트들은 이득을 과소평가하는 방향).
#
# 블록: Ec(ECMWF ws·wd sin/cos·|ws−ldaps| 4개) / Ic(ICON 동일 4개)
# 변형: 챔피언 TPSJgJm(132) / +Ec / +Ic / +EcIc × nosnap/snap30G/snap30P
# 실행: C:\miniconda3\python.exe -X utf8 exp022_ecmwf_icon.py [--quick] [--jobs N]
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

# 2022-23 전결측 피처(ec_*/ic_*)가 있는 fold에서 SimpleImputer가 컬럼을 드롭해
# shape 불일치로 죽는 문제 → 전결측 컬럼을 0으로 유지하도록 공유 모듈에 패치
from sklearn.impute import SimpleImputer as _SI


class _KeepEmptyImputer(_SI):
    def __init__(self, **kw):
        kw.setdefault("keep_empty_features", True)
        super().__init__(**kw)


import exp005_qopt as _m5
import exp009_rated_snap as _m9
import exp011_g3_transfer as _m11

_m5.SimpleImputer = _KeepEmptyImputer
_m9.SimpleImputer = _KeepEmptyImputer
_m11.SimpleImputer = _KeepEmptyImputer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

H2_FOLD = {"name": "hold_2024H2", "train_end": pd.Timestamp("2024-07-01 00:00:00"),
           "valid_start": pd.Timestamp("2024-07-01 01:00:00"), "valid_end": pd.Timestamp("2025-01-01 00:00:00")}

EC_COLS = ["ec_ws", "ec_wdsin", "ec_wdcos", "ec_dis"]
IC_COLS = ["ic_ws", "ic_wdsin", "ic_wdcos", "ic_dis"]


def add_ecmwf_icon(df):
    e = pd.read_csv(DATA_DIR / "external" / "ecmwf_icon_prevruns.csv", encoding="utf-8-sig",
                    parse_dates=["forecast_kst_dtm"])
    for model, tag in [("ecmwf_ifs025", "ec"), ("icon_global", "ic")]:
        sub = e[e["model"] == model].copy()
        wd_rad = np.deg2rad(sub["wd_d2"])
        out = pd.DataFrame({
            "forecast_kst_dtm": sub["forecast_kst_dtm"], "kpx_group": sub["kpx_group"],
            f"{tag}_ws": sub["ws_d2"],
            f"{tag}_wdsin": np.sin(wd_rad), f"{tag}_wdcos": np.cos(wd_rad),
        })
        df = df.merge(out, on=["forecast_kst_dtm", "kpx_group"], how="left")
        df[f"{tag}_dis"] = (df[f"{tag}_ws"] - df["ldaps_ws_10m"]).abs()
    return df


def print_block(title, summary):
    print(f"\n--- {title} ---")
    piv = summary.assign(fv=summary["variant"].str.rsplit("_", n=1).str[0],
                         pm=summary["variant"].str.rsplit("_", n=1).str[1])
    print(piv.pivot(index="fv", columns="pm", values="pooled_Score").round(4).to_string())


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
    train_df = add_ecmwf_icon(train_df)

    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)[["ec_ws", "ic_ws"]].apply(
        lambda d: d.notna().mean().round(3))
    print("연도별 커버리지:\n", cov.to_string())

    champ = col_sets["v10TPS"] + JG_COLS + JM_COLS
    variants_cols = {
        "champ": champ,
        "champEc": champ + EC_COLS,
        "champIc": champ + IC_COLS,
        "champEcIc": champ + EC_COLS + IC_COLS,
    }
    print("변형: " + ", ".join(f"{n}({len(c)})" for n, c in variants_cols.items()))
    print(f"데이터 준비 완료 ({time.time() - t0:.0f}s)\n", flush=True)

    summary, gs = run_folds(folds, train_df, variants_cols, args, "CV")
    print("\n================ 4-fold CV (pooled) ================")
    print_block("변형 × 후처리 (pooled Score)", summary)
    print("\n--- 전체 (pooled Score 내림차순) ---")
    print(summary.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    if not args.quick:
        print("\n================ 게이트 A: 1년 홀드아웃 (train ≤2023 → 2024 전체, 기록용) ================")
        hA, _ = run_folds([HOLD_FOLD], train_df, variants_cols, args, "HOLD")
        print_block("변형 × 후처리 (2024 전체)", hA)

        print("\n================ 게이트 B: 2024-H2 (train ≤2024-06 → 2024 하반기) ================")
        hB, _ = run_folds([H2_FOLD], train_df, variants_cols, args, "H2")
        print_block("변형 × 후처리 (2024-H2)", hB)
        print("\n--- 전체 (내림차순) ---")
        print(hB.sort_values("pooled_Score", ascending=False).round(4).to_string(index=False))

    print(f"\n총 소요: {(time.time() - t0) / 60:.1f}분")


if __name__ == "__main__":
    main()
