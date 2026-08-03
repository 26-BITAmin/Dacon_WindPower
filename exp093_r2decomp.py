# ═══════════════════════════════════════════════════════
# exp-093: 진단 — exp-091/092의 R²가 왜 음수인가 (분해)
#
# R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²  에서 ȳ = **평가 구간의 실제 평균**.
# 음수 = 모델이 "그 구간 평균만 찍기"보다 못하다는 뜻.
#
# 후보 원인 3가지를 수치로 분리한다:
#   (A) 타깃 쏠림 — avail_strict의 90%가 ≈1.0이라 분모(분산)가 애초에 작다
#   (B) 부분집합 평가 — 학습은 전체 행, 평가는 **실측 ≥10%cap 행만**.
#       부분집합의 평균·분산이 전체와 다르면 "평균 찍기" 기준선이 갑자기 강해진다
#   (C) 편향(수준 이동) — 예측 평균이 실제 평균과 어긋난 몫
#
# MSE = 편향² + 잔차분산 으로 쪼개 (C)의 몫을 직접 본다.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp093_r2decomp.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, prepare_matrices
from exp035_raw_dl import FOLD_LIST, HOLD_FOLD
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CV_FOLDS = [(fn, f) for fn, f in FOLD_LIST if fn != HOLD_FOLD["name"]]


def main():
    import lightgbm as lgb
    t0 = time.time()
    print("### exp-093 R² 음수 분해 ###", flush=True)
    train_df, use_cols = build_train_frame()
    df = train_df.loc[train_df["avail_strict"].notna()].copy()
    cols = use_cols + ["kpx_group"]

    y_all = df["avail_strict"].to_numpy()
    print(f"\n(A) 타깃 쏠림: 전체 n={len(y_all):,} | 평균 {y_all.mean():.4f} | "
          f"표준편차 {y_all.std():.4f} | ≥0.999 비율 {(y_all >= 0.999).mean():.3f}", flush=True)
    print(f"    → 분산 {y_all.var():.5f}. R² 분모가 이 크기다 (작을수록 뚫기 쉬움)\n", flush=True)

    print(f"{'fold':<12}{'n_ev':>7}{'ȳ(평가)':>10}{'sd(평가)':>10}{'ȳ(학습)':>10}"
          f"{'예측평균':>10}{'편향':>9}{'MSE':>10}{'편향²몫':>9}{'R²':>9}", flush=True)
    for fname, fold in CV_FOLDS:
        tr = df.loc[df["forecast_kst_dtm"] <= fold["train_end"]]
        va = df.loc[(df["forecast_kst_dtm"] >= fold["valid_start"]) &
                    (df["forecast_kst_dtm"] <= fold["valid_end"])]
        X_tr, X_va = prepare_matrices(tr, va, cols)
        reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                                random_state=0, n_jobs=8, verbose=-1).fit(
            X_tr, tr["avail_strict"].to_numpy())
        p = reg.predict(X_va)
        cap = va["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        ev = (va["target"].to_numpy() >= cap * 0.10) & va["target"].notna().to_numpy()

        y, ph = va["avail_strict"].to_numpy()[ev], p[ev]
        bias = ph.mean() - y.mean()
        mse = ((y - ph) ** 2).mean()
        sstot = ((y - y.mean()) ** 2).mean()          # = 평가구간 분산
        r2 = 1 - mse / sstot
        print(f"{fname.replace('fold_',''):<12}{ev.sum():>7,}{y.mean():>10.4f}{y.std():>10.4f}"
              f"{tr['avail_strict'].mean():>10.4f}{ph.mean():>10.4f}{bias:>+9.4f}"
              f"{mse:>10.5f}{bias**2/mse:>8.0%}{r2:>+9.4f}", flush=True)

    # (B) 부분집합 효과: 같은 fold에서 전체 vs 평가행의 기준선 비교
    print("\n(B) 부분집합 효과 — 같은 fold, 전체행 vs 평가행의 타깃 분포", flush=True)
    print(f"{'fold':<12}{'전체 ȳ':>10}{'전체 sd':>10}{'평가 ȳ':>10}{'평가 sd':>10}"
          f"{'분산비':>9}", flush=True)
    for fname, fold in CV_FOLDS:
        va = df.loc[(df["forecast_kst_dtm"] >= fold["valid_start"]) &
                    (df["forecast_kst_dtm"] <= fold["valid_end"])]
        cap = va["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        ev = (va["target"].to_numpy() >= cap * 0.10) & va["target"].notna().to_numpy()
        a, b = va["avail_strict"].to_numpy(), va["avail_strict"].to_numpy()[ev]
        print(f"{fname.replace('fold_',''):<12}{a.mean():>10.4f}{a.std():>10.4f}"
              f"{b.mean():>10.4f}{b.std():>10.4f}{b.var()/a.var():>9.2f}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
