# ═══════════════════════════════════════════════════════
# exp-032: base 오차의 예측 가능성 검정 — base 트랙 go/no-go
#
# 질문: 챔피언이 크게 틀리는 시간(|err|>15%, FiCR 손실 55%)을 피처로 예측할 수 있나?
#   - 예측 가능(AUC 높음) → base가 구조를 놓침(underfit) → 새 모델/피처/상호작용 여지.
#   - 예측 불가(AUC~0.5-0.6) → 오차가 입력(예보) 한계의 비가역 노이즈 → base 트랙 소진.
#   또한 signed 오차(잔차) 회귀 R²로 "보정 가능한 편향"이 남았는지 확인
#   (exp-028은 예측CF-bin 보정이 실패 = 잔차가 예측값엔 안 실림. 여기선 전 피처로 재검).
#
# 방법: fold별로 챔피언 base 학습→valid 예측→eval 시간의 (X, |err|>0.15, signed_err) 수집.
#   pooled 후 (a) big-error 분류기 OOF AUC, (b) signed-err 회귀 OOF R², (c) 중요 피처.
#   누수 방지: 진단 모델은 base와 다른 fold 분할(GroupKFold by 월)로 OOF 평가.
# 실행: C:\miniconda3\python.exe -X utf8 exp032_error_predictability.py [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score, r2_score

from exp005_qopt import (
    CAPACITY_KWH, FOLDS_Q,
    make_sample_weight, apply_rolling_smoothing, apply_bias_correction_v2,
    lgbm_params, prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS,
)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO = 0.50, 0.16
SNAP_A_THR, SNAP_A_TO = 0.30, 0.94
BIG = 0.15


def collect(folds, train_df, cols, args):
    Xs, big, signed, meta = [], [], [], []
    from exp011_g3_transfer import build_pooled_matrices
    from exp009_rated_snap import RATED_CF, N_CLF_SEEDS
    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all, va_all = train_df.loc[tr_mask], train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]
        clf_A = []
        for group in [1, 2, 3]:
            capacity = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            if len(tr_g) == 0 or len(va_g) == 0:
                continue
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"]
            w_tr = make_sample_weight(y_tr.to_numpy(), capacity, low_weight=0.05)
            preds = []
            for seed in range(args.seeds):
                m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed)); m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            pl = np.mean(preds, axis=0)
            preds = []
            for seed in range(args.seeds):
                m = xgbm.XGBRegressor(n_estimators=args.trees, max_depth=6, learning_rate=0.03, min_child_weight=5,
                                      subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=args.jobs,
                                      verbosity=0, objective="reg:absoluteerror"); m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            px = np.mean(preds, axis=0)
            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not clf_A:
                caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                yA = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
                for seed in range(N_CLF_SEEDS):
                    c = lgb.LGBMClassifier(n_estimators=args.trees, max_depth=6, learning_rate=0.03, num_leaves=31,
                                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                           n_jobs=args.jobs, verbose=-1); c.fit(X_pool_tr, yA); clf_A.append(c)
            pA = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf_A], axis=0)
            ws = va_g["ldaps_ws_10m"].to_numpy(); actual = va_g["target"].to_numpy()
            pv = apply_bias_correction_v2(0.5 * pl + 0.5 * px, ws, group, coarse_map)
            pv = apply_rolling_smoothing(pd.Series(pv, index=va_g.index), 3, ALPHA).to_numpy().copy()
            pv[pA >= SNAP_A_THR] = np.maximum(pv[pA >= SNAP_A_THR], capacity * SNAP_A_TO)
            pv = np.clip(pv, capacity * CLIP_LO, capacity)
            ev = actual >= capacity * 0.10
            err = np.abs(pv - actual) / capacity
            sgn = (pv - actual) / capacity
            Xs.append(X_va.loc[ev].assign(group=group, pred_cf=(pv/capacity)[ev], month=pd.to_datetime(va_g["forecast_kst_dtm"]).dt.month.to_numpy()[ev]))
            big.append((err[ev] > BIG).astype(int)); signed.append(sgn[ev])
            meta.append(pd.to_datetime(va_g["forecast_kst_dtm"]).dt.month.to_numpy()[ev])
        print(f"  fold {fold['name']} 수집 ({time.time()-tf:.0f}s)", flush=True)
    X = pd.concat(Xs, ignore_index=True)
    return X, np.concatenate(big), np.concatenate(signed), np.concatenate(meta)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--trees", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    train_df, _, col_sets, _ = build_v11()
    jma = build_jma_frame()
    train_df = train_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["jgsm_dis"] = (train_df["jgsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df["jmsm_dis"] = (train_df["jmsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    print(f"피처 {len(cols)}개 수집 시작\n", flush=True)
    X, big, signed, month = collect(FOLDS_Q, train_df, cols, args)
    print(f"\n수집 완료: {len(X)} 평가시간, big-error 비율 {big.mean():.3f}")

    feat = [c for c in X.columns if c != "month"]
    gkf = GroupKFold(n_splits=5)
    # (a) big-error 분류기 OOF AUC
    clf = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.03, num_leaves=31,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8, n_jobs=args.jobs, verbose=-1)
    oof_p = cross_val_predict(clf, X[feat], big, cv=gkf, groups=month, method="predict_proba", n_jobs=1)[:, 1]
    auc = roc_auc_score(big, oof_p)
    # (b) signed-err 회귀 OOF R²
    reg = lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.03, num_leaves=31,
                            min_child_samples=40, subsample=0.8, colsample_bytree=0.8, n_jobs=args.jobs, verbose=-1)
    oof_s = cross_val_predict(reg, X[feat], signed, cv=gkf, groups=month, n_jobs=1)
    r2 = r2_score(signed, oof_s)
    print(f"\n{'='*56}")
    print(f"(a) big-error(|err|>{BIG}) 예측 OOF AUC = {auc:.4f}")
    print(f"    (0.5=무작위/비가역, 0.7+=구조 존재)")
    print(f"(b) signed 잔차 회귀 OOF R² = {r2:.4f}")
    print(f"    (0=보정 가능한 편향 없음, >0.1=잔차에 구조 남음)")
    # (c) 중요 피처
    clf.fit(X[feat], big)
    imp = pd.Series(clf.feature_importances_, index=feat).sort_values(ascending=False).head(12)
    print(f"\nbig-error 예측 상위 피처:\n{imp.to_string()}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
