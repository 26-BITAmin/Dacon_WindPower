# ═══════════════════════════════════════════════════════
# exp-077 진단: 현 챔피언(352피처·HPO·cln03) base 잔차의 예측 가능성 재검정
#   — DeLTa(트리 규칙 기반 LLM 보정, NeurIPS 2025) 착수 전 go/no-go
#
# 근거: DeLTa의 보정 규칙은 결국 기존 피처의 함수 → 이득 상한 = 잔차 예측가능성.
#   exp-032가 132피처 시절 "잔차 직교(R² −0.095·AUC 0.66) = 보정 편향 전무"를 확정했으나
#   이후 base가 크게 변함(352피처·GBM HPO·가용성 가중·EC) — 재측정 필요.
#
# 방법 (exp-032 방법론, 현 캐시 재사용으로 재학습 0):
#   - cf_hpo OOF 예측(4 CV fold pooled)에서 잔차 r = target/cap − cf_hpo
#   - (a) signed 잔차 회귀 OOF R² (LGBM, GroupKFold by 월 — 누수 방지)
#   - (b) big-error(평가시간 error_rate>0.15) 분류 OOF AUC
#   - (c) 잔차 모델 상위 중요 피처 (go 시 DeLTa 규칙 생성의 입력)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp077_delta_diag.py [--jobs 8]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, r2_score

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp049_clean import KEY
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-077 진단: 현 챔피언 base 잔차 예측가능성 (DeLTa go/no-go) ###", flush=True)

    train_df, use_cols = build_train_frame()
    cache, _ = load_cache_full()
    cv = cache[cache["fold"] != HOLD_FOLD["name"]].copy()
    cv = cv[cv["target"].notna()]
    m = cv.merge(train_df.rename(columns={"kpx_group": "group"})[["group", "forecast_kst_dtm"] + use_cols],
                 on=["group", "forecast_kst_dtm"], how="left")
    assert len(m) == len(cv), "병합 행수 불일치"
    cap = m["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    resid = m["target"].to_numpy() / cap - m["cf_hpo"].to_numpy()
    X = m[use_cols].to_numpy(dtype=np.float32)
    month_grp = m["forecast_kst_dtm"].dt.strftime("%Y-%m").to_numpy()
    print(f"CV 라벨행 {len(m)} | 잔차 std {resid.std():.4f} | 평균 {resid.mean():+.4f}", flush=True)

    gkf = GroupKFold(n_splits=5)
    reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                            colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                            random_state=42, n_jobs=args.jobs, verbose=-1)
    pred_r = cross_val_predict(reg, X, resid, cv=gkf, groups=month_grp)
    r2 = r2_score(resid, pred_r)
    print(f"\n(a) signed 잔차 회귀 OOF R² = {r2:+.4f}  (exp-032 당시 −0.095)", flush=True)

    # (b) 평가시간(실측 ≥10%cap) big-error 분류
    ev = m["target"].to_numpy() >= 0.10 * cap
    err_rate = np.abs(m["cf_hpo"].to_numpy() * cap - m["target"].to_numpy()) / np.maximum(m["target"].to_numpy(), 1)
    big = (err_rate > 0.15).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                             colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                             random_state=42, n_jobs=args.jobs, verbose=-1)
    p_big = cross_val_predict(clf, X[ev], big[ev], cv=gkf, groups=month_grp[ev],
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(big[ev], p_big)
    print(f"(b) big-error(>15%) 분류 OOF AUC = {auc:.4f}  (평가행 {ev.sum()}, "
          f"양성비 {big[ev].mean():.3f}, exp-032 당시 0.656)", flush=True)

    reg.fit(X, resid)
    imp = pd.Series(reg.feature_importances_, index=use_cols).sort_values(ascending=False)
    print("\n(c) 잔차 모델 상위 15 피처:", flush=True)
    for name, v in imp.head(15).items():
        print(f"  {name:<40}{v:>6}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
