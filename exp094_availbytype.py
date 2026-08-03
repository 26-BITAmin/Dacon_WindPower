# ═══════════════════════════════════════════════════════
# exp-094: 정지 예측 — 터빈 종류(제조사)별 분리 (exp-091/092 후속)
#
# 문제 제기 (사용자): "터빈 종류가 2가지인데 구분해서 회귀한 거임?"
#   확인 결과 **종류 = 그룹과 일대일**이다:
#     VESTAS 12기 (3.6MW, 슬롯 600kWh) → g1(1~6호) · g2(7~12호)
#     UNISON  5기 (4.2MW, 슬롯 700kWh) → g3
#   exp-091 1차는 kpx_group 자체가 없었고, 2차·exp-092는 넣었으나 **단일 pooled 모델**이었다.
#   본 예측 파이프라인은 그룹별 개별 모델 3개를 쓰는데 진단만 pooled였다.
#
# 왜 중요할 수 있나: 제조사가 다르면 컷아웃 풍속·제어 로직·착빙 민감도가 다르고,
#   AVAILABILITY.md상 정지 시기도 다르다 (g1·g2 = 2024-01/02, g3 = 2023-10·2024-12).
#   희소 사건(양성 ~10%)에서 "group==3이면 다른 규칙"을 트리가 스스로 배우기는 어렵다.
#
# 비교: 동일 폴드·동일 평가행에서
#   pooled : 전 그룹 1개 모델(+kpx_group) → 그룹별로 나눠 AUC
#   bytype : 그룹별 개별 모델 3개        → 같은 행에서 AUC
#
# 실행: C:\miniconda3\python.exe -X utf8 exp094_availbytype.py
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

THR = 0.9
CV_FOLDS = [(fn, f) for fn, f in FOLD_LIST if fn != HOLD_FOLD["name"]]
TYPE = {1: "VESTAS", 2: "VESTAS", 3: "UNISON"}


def fit_predict(tr, va, cols, seed=0, want_imp=False):
    import lightgbm as lgb
    X_tr, X_va = prepare_matrices(tr, va, cols)
    y = (tr["avail_strict"] < THR).astype(int).to_numpy()
    if y.sum() < 30 or y.sum() == len(y):
        return (None, None) if want_imp else None
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                             random_state=seed, n_jobs=8, verbose=-1).fit(X_tr, y)
    p = clf.predict_proba(X_va)[:, 1]
    if want_imp:
        return p, pd.Series(clf.feature_importances_, index=X_tr.columns)
    return p


def main():
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    print("### exp-094 정지 예측: 터빈 종류별 분리 ###", flush=True)
    train_df, use_cols = build_train_frame()
    df = train_df.loc[train_df["avail_strict"].notna()].copy()
    df["cap"] = df["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])

    print("\n그룹·종류별 타깃 기저율 (avail_strict < 0.9):", flush=True)
    for g in [1, 2, 3]:
        s = df.loc[df["kpx_group"] == g]
        ev = (s["target"].to_numpy() >= s["cap"].to_numpy() * 0.10) & s["target"].notna().to_numpy()
        print(f"  g{g} ({TYPE[g]:<6}) n={len(s):,} 양성 {(s['avail_strict']<THR).mean():.3f} | "
              f"평가행 n={ev.sum():,} 양성 {(s['avail_strict'].to_numpy()[ev]<THR).mean():.3f} | "
              f"라벨시작 {s['forecast_kst_dtm'].min():%Y-%m}", flush=True)

    # ── 무결성: g3는 SCADA(UNISON)가 2023부터인데 avail_strict가 2022에도 있나? ──
    print("\n[무결성] 연도×그룹 avail_strict 커버리지 / 평균 / <0.9 비율 / ==1.0 비율:", flush=True)
    df["_yr"] = df["forecast_kst_dtm"].dt.year
    for g in [1, 2, 3]:
        for yr in sorted(df["_yr"].unique()):
            s = df.loc[(df["kpx_group"] == g) & (df["_yr"] == yr), "avail_strict"]
            if len(s) == 0:
                continue
            print(f"   g{g} {yr}: n={len(s):>6,} 평균 {s.mean():.4f} "
                  f"<0.9 {(s < THR).mean():.3f} ==1.0 {(s >= 0.999).mean():.3f}", flush=True)

    rec, imp_g = [], {g: [] for g in [1, 2, 3]}
    for fname, fold in CV_FOLDS:
        tr_all = df.loc[df["forecast_kst_dtm"] <= fold["train_end"]]
        va_all = df.loc[(df["forecast_kst_dtm"] >= fold["valid_start"]) &
                        (df["forecast_kst_dtm"] <= fold["valid_end"])]
        p_pool = fit_predict(tr_all, va_all, use_cols + ["kpx_group"])
        for g in [1, 2, 3]:
            m = (va_all["kpx_group"] == g).to_numpy()
            va_g, tr_g = va_all.loc[m], tr_all.loc[tr_all["kpx_group"] == g]
            ev = ((va_g["target"].to_numpy() >= va_g["cap"].to_numpy() * 0.10) &
                  va_g["target"].notna().to_numpy())
            y = (va_g["avail_strict"] < THR).astype(int).to_numpy()
            if ev.sum() < 50 or y[ev].std() == 0:
                continue
            p_by, imp = fit_predict(tr_g, va_g, use_cols, want_imp=True)   # 그룹 전용 모델
            if imp is not None:
                imp_g[g].append(imp)
            a_pool = roc_auc_score(y[ev], p_pool[m][ev])
            a_by = roc_auc_score(y[ev], p_by[ev]) if p_by is not None else np.nan
            rec.append(dict(fold=fname.replace("fold_", ""), g=g, n_ev=int(ev.sum()),
                            pos_ev=y[ev].mean(), pooled=a_pool, bytype=a_by,
                            n_tr=len(tr_g), pos_tr=(tr_g["avail_strict"] < THR).mean()))

    r = pd.DataFrame(rec)
    folds = list(dict.fromkeys(r["fold"]))

    def matrix(col, fmt="{:.4f}"):
        print(f"\n  {col} — 행=그룹, 열=fold", flush=True)
        print(f"    {'':<14}" + "".join(f"{f:>10}" for f in folds) + f"{'평균':>10}")
        for g in [1, 2, 3]:
            s = r[r["g"] == g].set_index("fold")
            vals = [s.loc[f, col] if f in s.index else np.nan for f in folds]
            print(f"    {'g'+str(g)+' '+TYPE[g]:<14}" +
                  "".join(fmt.format(v).rjust(10) for v in vals) +
                  fmt.format(np.nanmean(vals)).rjust(10), flush=True)

    print("\n" + "=" * 78)
    print("■ 그룹 × fold 상세")
    matrix("bytype")
    matrix("pooled")
    matrix("pos_ev", "{:.3f}")
    matrix("n_ev", "{:,.0f}")

    print("\n" + "=" * 78)
    print(f"{'':<16}{'pooled':>10}{'bytype':>10}{'Δ':>10}{'bytype 범위':>22}")
    for g in [1, 2, 3]:
        s = r[r["g"] == g]
        if len(s):
            print(f"{'g'+str(g)+' '+TYPE[g]:<16}{s['pooled'].mean():>10.4f}"
                  f"{s['bytype'].mean():>10.4f}{s['bytype'].mean()-s['pooled'].mean():>+10.4f}"
                  f"{s['bytype'].min():>12.3f}~{s['bytype'].max():<9.3f}")
    print(f"{'전체 평균':<16}{r['pooled'].mean():>10.4f}{r['bytype'].mean():>10.4f}"
          f"{r['bytype'].mean()-r['pooled'].mean():>+10.4f}")

    print("\n" + "=" * 78)
    print("■ 그룹 전용 모델이 실제로 쓰는 피처 (4-fold 평균 importance, 상위 12)")
    for g in [1, 2, 3]:
        if not imp_g[g]:
            continue
        s = pd.concat(imp_g[g], axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n  g{g} ({TYPE[g]}):", flush=True)
        for k, v in s.head(12).items():
            print(f"    {k:<44}{v:8.1f}", flush=True)
    # 세 그룹이 같은 피처를 쓰는가
    tops = {g: set(pd.concat(imp_g[g], axis=1).mean(axis=1)
                   .sort_values(ascending=False).head(20).index)
            for g in [1, 2, 3] if imp_g[g]}
    if len(tops) == 3:
        print(f"\n  상위 20 교집합: 3그룹 공통 {len(tops[1] & tops[2] & tops[3])}개 | "
              f"g1∩g2(둘 다 VESTAS) {len(tops[1] & tops[2])}개 | "
              f"g1∩g3(다른 제조사) {len(tops[1] & tops[3])}개 | "
              f"g2∩g3 {len(tops[2] & tops[3])}개", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
