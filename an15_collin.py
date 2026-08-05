# ═══════════════════════════════════════════════════════
# AN-15: 다중공선성·과적합 진단 — 5소스 풍속 OLS (EMOS 하이브리드의 취약 지점)
#
# 사용자 지적: EMOS 계열(선형 후처리)은 다중공선성·과적합에 취약할 수 있다.
# 구조 분석: 본체(파라미터 5~9개/수천 표본)는 안전, 취약 지점은
#   ① 5소스 풍속 OLS (소스 간 고상관), ② 소표본 적합(계절판 — 이미 ES 악화 관찰).
#
# 측정:
#   - 소스 간 상관행렬·설계행렬 조건수·VIF
#   - 폴드별 OLS 계수 (부호 반전·불안정 확인)
#   - 검증 MAE: OLS vs Ridge(α 격자) vs 비음수(NNLS) vs 균등평균(보정 후)
#     → 릿지/제약이 이기면 "공선성이 예측까지 해침", 동률이면 "계수만 불안정"
#
# 실행: C:\miniconda3\python.exe -X utf8 an15_collin.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = ["gfs_ws_100m", "ldaps_ws_10m", "ec_ws", "jmsm_ws", "jgsm_ws"]


def vif(X):
    c = np.corrcoef(X, rowvar=False)
    return np.diag(np.linalg.inv(c))


def main():
    t0 = time.time()
    print("### AN-15 다중공선성·과적합 진단 (5소스 풍속 OLS) ###", flush=True)
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    for g in [1, 2, 3]:
        sc = scada_h[g][["forecast_kst_dtm", "park_ws"]]
        train_df = train_df.merge(
            sc.assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"], how="left",
            suffixes=("", "_dup"))
        if "park_ws_dup" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_dup"))
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    from scipy.optimize import nnls
    from sklearn.linear_model import Ridge
    folds = dict(FOLD_LIST)

    # ── 1) 상관·조건수·VIF (전체 학습기간, 그룹별)
    print("\n[1] 소스 간 상관·조건수·VIF (park_ws 존재 행)", flush=True)
    for g in [1, 2, 3]:
        d = train_df[(train_df["kpx_group"] == g) & train_df["park_ws"].notna()]
        X = d[SRC].ffill().bfill().to_numpy()
        c = np.corrcoef(X, rowvar=False)
        off = c[np.triu_indices(5, 1)]
        Xs = (X - X.mean(0)) / X.std(0)
        cond = np.linalg.cond(np.c_[Xs, np.ones(len(Xs))])
        print(f"  g{g}: 소스상관 min/mean/max {off.min():.3f}/{off.mean():.3f}/"
              f"{off.max():.3f} | 조건수 {cond:.1f} | "
              f"VIF {np.round(vif(X), 1)}", flush=True)

    # ── 2) 폴드별 OLS 계수 안정성 + 검증 MAE 비교
    print("\n[2] 폴드별 계수·검증 MAE (OLS vs Ridge vs NNLS vs 균등평균)", flush=True)
    recs, coef_rows = [], []
    for fn in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tr_all, va_all = slice_fold(train_df, folds[fn])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g]
            va_g = va_all.loc[va_all["kpx_group"] == g]
            trm = tr_g["park_ws"].notna()
            vam = va_g["park_ws"].notna()
            Xtr = tr_g.loc[trm, SRC].ffill().bfill().to_numpy()
            ytr = tr_g.loc[trm, "park_ws"].to_numpy()
            Xva = va_g.loc[vam, SRC].ffill().bfill().to_numpy()
            yva = va_g.loc[vam, "park_ws"].to_numpy()

            A = np.c_[Xtr, np.ones(len(Xtr))]
            w_ols, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            coef_rows.append(dict(fold=fn, group=g,
                                  **dict(zip(SRC, np.round(w_ols[:5], 3)))))
            p_ols = np.c_[Xva, np.ones(len(Xva))] @ w_ols

            best_r = None
            for al in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                r = Ridge(alpha=al).fit(Xtr, ytr)
                p = r.predict(Xva)
                m = np.abs(p - yva).mean()
                if best_r is None or m < best_r[1]:
                    best_r = (al, m)

            w_nn, _ = nnls(A, ytr)
            p_nn = np.c_[Xva, np.ones(len(Xva))] @ w_nn

            # 균등평균: 소스별 단순 선형보정 후 평균 (공선성 원천 차단)
            p_eq = np.zeros(len(yva))
            for j in range(5):
                a, b = np.polyfit(Xtr[:, j], ytr, 1)
                p_eq += a * Xva[:, j] + b
            p_eq /= 5

            recs.append(dict(fold=fn, group=g,
                             ols=np.abs(p_ols - yva).mean(),
                             ridge=best_r[1], ridge_alpha=best_r[0],
                             nnls=np.abs(p_nn - yva).mean(),
                             eqmean=np.abs(p_eq - yva).mean(),
                             neg_coef=(w_ols[:5] < 0).sum()))
    cdf = pd.DataFrame(coef_rows)
    print("\n  OLS 계수 (폴드×그룹):", flush=True)
    print(cdf.to_string(index=False), flush=True)
    df = pd.DataFrame(recs)
    print(f"\n  음수 계수 발생: {df['neg_coef'].sum()}건 / "
          f"{len(df)*5}계수", flush=True)
    agg = df[["ols", "ridge", "nnls", "eqmean"]].mean()
    print("\n  검증 MAE (4폴드·3그룹 평균):", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"  릿지 최적 α 분포: {sorted(df['ridge_alpha'].unique())}", flush=True)

    # ── 3) 소표본 과적합 곡선: 학습행 수를 줄여가며 OLS vs Ridge 격차
    print("\n[3] 학습 표본 축소 시 OLS vs Ridge(α=10) 검증 MAE (Q4, 그룹 평균)",
          flush=True)
    from sklearn.linear_model import Ridge as R2
    for frac in [1.0, 0.25, 0.1, 0.03, 0.01]:
        o_, r_ = [], []
        for g in [1, 2, 3]:
            tr_all, va_all = slice_fold(train_df, folds["fold_2024Q4"])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g]
            va_g = va_all.loc[va_all["kpx_group"] == g]
            trm = tr_g["park_ws"].notna()
            Xtr = tr_g.loc[trm, SRC].ffill().bfill().to_numpy()
            ytr = tr_g.loc[trm, "park_ws"].to_numpy()
            n = max(30, int(len(ytr) * frac))
            Xtr, ytr = Xtr[-n:], ytr[-n:]                  # 최근 n행
            vam = va_g["park_ws"].notna()
            Xva = va_g.loc[vam, SRC].ffill().bfill().to_numpy()
            yva = va_g.loc[vam, "park_ws"].to_numpy()
            w, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(n)], ytr, rcond=None)
            o_.append(np.abs(np.c_[Xva, np.ones(len(Xva))] @ w - yva).mean())
            r_.append(np.abs(R2(alpha=10.0).fit(Xtr, ytr).predict(Xva) - yva).mean())
        print(f"  n={frac*100:5.0f}% : OLS {np.mean(o_):.4f} | Ridge {np.mean(r_):.4f}"
              f" | Δ {np.mean(o_)-np.mean(r_):+.4f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
