# ═══════════════════════════════════════════════════════
# exp-091: 진단 — 터빈 정지(가용성)는 예보 피처로 예측 가능한가? (go/no-go)
#
# 왜 지금:
#   로그의 "정지는 예보 불가"는 **직접 측정된 적이 없다.** 근거로 인용되는 두 실험은
#     exp-053: 라벨을 avail로 나눠 복원 → 역방향 악화 (예측 불가를 **전제**하고 우회)
#     PC-02 R2: 정지행을 학습에서 제외 → 상향 편향 (같은 전제)
#   둘 다 "예측 가능한가"를 잰 게 아니라 "예측 못 한다고 치고" 다룬 것이다.
#   반면 AVAILABILITY.md의 월별 표는 정지가 **에피소드형·겨울 집중**임을 보여준다
#   (2024-01/02 세 그룹 동시 24~39%). 착빙·고풍속 컷아웃 등은 기상 구동이라
#   원리적으로 예보 피처에 신호가 있을 수 있다.
#
# 이 실험은 **진단 전용**이다. 학습 파이프라인·제출물은 건드리지 않는다.
#   (exp-032/077의 잔차 예측가능성 go/no-go와 동일 프로토콜)
#
# 측정:
#   (A) 분류: avail_strict < 0.9  ("cln03이 0.05로 죽이는 시간") OOF AUC
#   (B) 회귀: avail_strict 자체 OOF R²
#   (C) **평가 시간대 한정** — 실측 ≥ 10%cap 인 행만. AVAILABILITY.md 한계 §5가 지적했듯
#       전면 정지 시간은 애초에 채점 제외라, 점수에 실제로 전이되는 것은 **부분 정지**다.
#       여기서의 예측 가능성이 진짜 상한이다.
#   (D) 상위 피처 — 착빙 가설(기온·습도)이 뜨는지 확인
#
# 판정 (사전 등록):
#   평가시간 AUC ≥ 0.75  → GO (활용 방법 별도 설계)
#   0.65 ~ 0.75          → 약신호. exp-077의 big-error AUC 0.663과 같은 급 = 보류
#   < 0.65               → NO-GO. "정지는 예보 불가"를 **측정으로** 확정하고 축을 닫는다
#
# 실행: C:\miniconda3\python.exe -X utf8 exp091_availpred.py
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

THR = 0.9          # cln03이 0.05로 다운웨이트하는 문턱
CV_FOLDS = [(fn, f) for fn, f in FOLD_LIST if fn != HOLD_FOLD["name"]]


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, r2_score

    t0 = time.time()
    print("### exp-091 진단: 터빈 정지 예측 가능성 (go/no-go) ###", flush=True)
    train_df, use_cols = build_train_frame()
    df = train_df.loc[train_df["avail_strict"].notna()].copy()
    df["cap"] = df["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    print(f"가용성 라벨 보유 행 {len(df):,} | avail_strict 분포: "
          f"평균 {df['avail_strict'].mean():.3f} | "
          f"<0.9 비율 {(df['avail_strict'] < THR).mean():.3f} | "
          f"=1.0 비율 {(df['avail_strict'] >= 0.999).mean():.3f}", flush=True)
    print(f"입력 {len(use_cols)}피처 — {time.time()-t0:.0f}s\n", flush=True)

    rows, imps = [], []
    for fname, fold in CV_FOLDS:
        tf = time.time()
        tr = df.loc[df["forecast_kst_dtm"] <= fold["train_end"]]
        va = df.loc[(df["forecast_kst_dtm"] >= fold["valid_start"]) &
                    (df["forecast_kst_dtm"] <= fold["valid_end"])]
        if len(va) == 0 or (tr["avail_strict"] < THR).sum() < 50:
            continue
        # kpx_group 명시 주입: 352 중 216(격자 편차)이 시각 기준 병합이라 세 그룹이
        # 동일 값 → 그룹 식별 단서가 약함. 그룹별 정지 에피소드 시기가 다르므로 필수.
        X_tr, X_va = prepare_matrices(tr, va, use_cols + ["kpx_group"])
        y_cls_tr = (tr["avail_strict"] < THR).astype(int).to_numpy()
        y_cls_va = (va["avail_strict"] < THR).astype(int).to_numpy()
        y_reg_tr = tr["avail_strict"].to_numpy()
        y_reg_va = va["avail_strict"].to_numpy()

        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                                 random_state=0, n_jobs=8, verbose=-1).fit(X_tr, y_cls_tr)
        reg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                min_child_samples=40, subsample=0.8, colsample_bytree=0.6,
                                random_state=0, n_jobs=8, verbose=-1).fit(X_tr, y_reg_tr)
        p_cls = clf.predict_proba(X_va)[:, 1]
        p_reg = reg.predict(X_va)

        # (C) 평가 시간대 한정 — 채점 대상이 되는 행만
        ev = (va["target"].to_numpy() >= va["cap"].to_numpy() * 0.10) & va["target"].notna().to_numpy()
        rec = dict(fold=fname, n=len(va), pos=y_cls_va.mean(),
                   auc_all=roc_auc_score(y_cls_va, p_cls) if y_cls_va.std() > 0 else np.nan,
                   r2_all=r2_score(y_reg_va, p_reg),
                   n_ev=int(ev.sum()), pos_ev=y_cls_va[ev].mean() if ev.any() else np.nan,
                   auc_ev=(roc_auc_score(y_cls_va[ev], p_cls[ev])
                           if ev.any() and y_cls_va[ev].std() > 0 else np.nan),
                   r2_ev=r2_score(y_reg_va[ev], p_reg[ev]) if ev.any() else np.nan)
        rows.append(rec)
        imps.append(pd.Series(clf.feature_importances_, index=X_tr.columns))
        print(f"  [{fname}] n={rec['n']:,} 양성{rec['pos']:.3f} | "
              f"전체 AUC {rec['auc_all']:.4f} R² {rec['r2_all']:+.4f} || "
              f"평가시간 n={rec['n_ev']:,} 양성{rec['pos_ev']:.3f} "
              f"**AUC {rec['auc_ev']:.4f}** R² {rec['r2_ev']:+.4f} ({time.time()-tf:.0f}s)",
              flush=True)

    r = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"평균  전체 AUC {r['auc_all'].mean():.4f} / R² {r['r2_all'].mean():+.4f}")
    print(f"     **평가시간 AUC {r['auc_ev'].mean():.4f} / R² {r['r2_ev'].mean():+.4f}**")

    a = r["auc_ev"].mean()
    verdict = ("GO — 활용 설계 착수 가치 있음" if a >= 0.75 else
               ("약신호(보류) — exp-077 big-error AUC 0.663과 같은 급" if a >= 0.65 else
                "NO-GO — '정지는 예보 불가'를 측정으로 확정"))
    print(f"\n판정 (사전 등록 기준): **{verdict}**")

    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\n상위 15 피처 (착빙 가설 = 기온·습도 계열이 뜨는지):")
    for k, v in imp.head(15).items():
        print(f"    {k:<42} {v:8.1f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
