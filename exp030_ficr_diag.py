# ═══════════════════════════════════════════════════════
# exp-030: 현 챔피언(c16_a50) FiCR 손실 구조 재진단
#
# 목적: exp-008(FiCR 분해)은 JMA·clip·스냅튜닝 이전 파이프라인 기준. 현 챔피언에서
#   FiCR을 어디서·왜 잃는지 다시 측정해 다음 레버를 정한다. 특히:
#     1. 오차율 분포: 6/8% 경계 근처(고칠 수 있는) vs 20%+(가망 없는) 질량 분리
#     2. **방향성 분해**: "near-miss"(오차 6~14%) 시간이 과대/과소 어느 쪽으로 쏠렸나
#        — 한 방향이면 nudge로 밴드 진입 가능, 50/50이면 base 정확도만이 답
#     3. 세그먼트(그룹·CF·ws·시간대·월)별 회복 가능 FiCR 손실 순위
#   순수 진단 — 제출 없음. 4-fold CV + 1년 홀드아웃 두 뷰.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp030_ficr_diag.py [--jobs N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm

from exp005_qopt import (
    CAPACITY_KWH, FOLDS_Q,
    make_sample_weight, apply_rolling_smoothing, apply_bias_correction_v2,
    lgbm_params, prepare_matrices,
)
from exp009_rated_snap import (
    estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS,
)
from exp011_g3_transfer import build_pooled_matrices
from exp012_feature_v11 import build_v11
from exp019_multigrid import HOLD_FOLD
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 챔피언 후처리 (c16_a50)
ALPHA, SNAP_THR, SNAP_TO, CLIP_LO = 0.50, 0.30, 0.94, 0.16


def collect_champion(folds, train_df, cols, args, tag):
    """챔피언 파이프라인을 돌려 평가 대상 시간의 per-hour 레코드를 수집."""
    recs = []
    for fold in folds:
        tf = time.time()
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], args.trees, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all = train_df.loc[tr_mask]
        va_all = train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]
        pooled_clf = []

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
                m = lgb.LGBMRegressor(**lgbm_params(args.trees, args.jobs, seed))
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            pred_lgbm = np.mean(preds, axis=0)
            preds = []
            for seed in range(args.seeds):
                m = xgbm.XGBRegressor(
                    n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror",
                )
                m.fit(X_tr, y_tr, sample_weight=w_tr)
                preds.append(m.predict(X_va))
            pred_xgb = np.mean(preds, axis=0)

            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not pooled_clf:
                caps_pool = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                y_pool_rated = (tr_pool["target"].to_numpy() >= caps_pool * RATED_CF).astype(int)
                for seed in range(N_CLF_SEEDS):
                    clf = lgb.LGBMClassifier(
                        n_estimators=args.trees, max_depth=6, learning_rate=0.03,
                        num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=args.jobs, verbose=-1,
                    )
                    clf.fit(X_pool_tr, y_pool_rated)
                    pooled_clf.append(clf)
            p_rated = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in pooled_clf], axis=0)

            ws = va_g["ldaps_ws_10m"].to_numpy()
            actual = va_g["target"].to_numpy()
            blend = 0.5 * pred_lgbm + 0.5 * pred_xgb
            pv = apply_bias_correction_v2(blend, ws, group, coarse_map)
            pv = apply_rolling_smoothing(pd.Series(pv, index=va_g.index), 3, ALPHA).to_numpy()
            pv = np.array(pv, dtype=float)
            snap_mask = p_rated >= SNAP_THR
            pv[snap_mask] = np.maximum(pv[snap_mask], capacity * SNAP_TO)
            pv = np.clip(pv, capacity * CLIP_LO, capacity)

            ev = actual >= capacity * 0.10   # 평가 대상
            dt = pd.to_datetime(va_g["forecast_kst_dtm"].to_numpy())
            rec = pd.DataFrame({
                "group": group, "actual": actual[ev], "pred": pv[ev], "cap": capacity,
                "ws": ws[ev], "p_rated": p_rated[ev], "snapped": snap_mask[ev],
                "hour": dt.hour.to_numpy()[ev], "month": dt.month.to_numpy()[ev],
            })
            rec["cf"] = rec["actual"] / capacity
            rec["err"] = np.abs(rec["pred"] - rec["actual"]) / capacity
            rec["signed"] = (rec["pred"] - rec["actual"]) / capacity   # +면 과대예측
            recs.append(rec)
        print(f"[{tag}:{fold['name']}] 수집 ({time.time()-tf:.0f}s)", flush=True)
    return pd.concat(recs, ignore_index=True)


def unit_price(err):
    return np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)


def analyze(df, tag):
    print(f"\n{'='*62}\n[{tag}] 평가 시간 {len(df)}개, 발전량 합 {df['actual'].sum()/1e6:.1f}GWh")
    up = unit_price(df["err"].to_numpy())
    full = df["actual"].to_numpy() * 4.0
    got = df["actual"].to_numpy() * up
    ficr = got.sum() / full.sum()
    print(f"FiCR (pooled, actual가중) = {ficr:.4f}  |  손실 = {1-ficr:.4f}")

    # 1. 오차율 분포 (시간 수 + 발전량가중 FiCR손실 기여)
    print("\n--- 1. 오차율 구간별: 시간%, 발전량%, FiCR손실 기여% ---")
    bins = [0, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 10]
    labs = ["≤6%(만점)", "6-8%", "8-10%", "10-12%", "12-15%", "15-20%", "20-30%", "30%+"]
    df = df.copy()
    df["ebin"] = pd.cut(df["err"], bins=bins, labels=labs)
    loss = df["actual"].to_numpy() * (4.0 - up)   # 만점 대비 잃은 정산금
    df["loss"] = loss
    g = df.groupby("ebin", observed=True).agg(n=("err", "size"), gen=("actual", "sum"), loss=("loss", "sum"))
    g["시간%"] = (g["n"] / g["n"].sum() * 100).round(1)
    g["발전량%"] = (g["gen"] / g["gen"].sum() * 100).round(1)
    g["손실기여%"] = (g["loss"] / g["loss"].sum() * 100).round(1)
    print(g[["시간%", "발전량%", "손실기여%"]].to_string())

    # 2. 회복 가능성: near-miss(6~14%) 시간의 방향성
    print("\n--- 2. near-miss(오차 6~14%) 방향성 — 과대(+)/과소(−) 쏠림 ---")
    nm = df[(df["err"] > 0.06) & (df["err"] <= 0.14)].copy()
    over = (nm["signed"] > 0).sum()
    tot = len(nm)
    print(f"near-miss {tot}시간 (발전량 {nm['actual'].sum()/1e6:.1f}GWh, 손실기여 {nm['loss'].sum()/df['loss'].sum()*100:.0f}%)")
    print(f"  과대예측(pred>actual) {over}시간 ({over/tot*100:.0f}%) / 과소 {tot-over}시간 ({(tot-over)/tot*100:.0f}%)")
    print(f"  signed error 평균 {nm['signed'].mean():+.3f} (0에 가까울수록 방향성 없음)")

    # 3. 세그먼트별 near-miss 방향성 + 손실기여 (nudge 표적 탐색)
    for axis, binner in [
        ("group", lambda d: d["group"]),
        ("CF", lambda d: pd.cut(d["cf"], [0.1, 0.3, 0.5, 0.7, 0.9, 1.01],
                                labels=["0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9+"])),
        ("ws", lambda d: pd.cut(d["ws"], [0, 5, 7, 9, 11, 13, 100],
                                labels=["0-5", "5-7", "7-9", "9-11", "11-13", "13+"])),
    ]:
        nm2 = nm.copy(); nm2["seg"] = binner(nm2)
        alld = df.copy(); alld["seg"] = binner(alld)
        agg = nm2.groupby("seg", observed=True).agg(
            near_miss_n=("err", "size"), signed_mean=("signed", "mean"),
            over_pct=("signed", lambda s: (s > 0).mean() * 100))
        agg["손실기여%"] = alld.groupby("seg", observed=True)["loss"].sum() / df["loss"].sum() * 100
        print(f"\n--- 3.{axis}: near-miss 방향성 × 전체 손실기여 ---")
        print(agg.round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()

    t0 = time.time()
    train_df, _, col_sets, _ = build_v11()
    jma = build_jma_frame()
    train_df = train_df.merge(jma, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["jgsm_dis"] = (train_df["jgsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df["jmsm_dis"] = (train_df["jmsm_ws"] - train_df["ldaps_ws_10m"]).abs()
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    print(f"챔피언 피처 {len(cols)}개, 후처리 clip{CLIP_LO}/α{ALPHA}/snap p{SNAP_THR}→{SNAP_TO}")
    print(f"데이터 준비 완료 ({time.time()-t0:.0f}s)\n", flush=True)

    cv = collect_champion(FOLDS_Q, train_df, cols, args, "CV")
    analyze(cv, "4-fold CV")
    hold = collect_champion([HOLD_FOLD], train_df, cols, args, "HOLD")
    analyze(hold, "1년 홀드아웃 2024")

    # 저장 (후속 실험 재사용)
    out = "exp030_champ_records.parquet"
    try:
        cv.assign(view="cv").to_parquet(out, index=False)
        print(f"\n레코드 저장: {out}")
    except Exception as e:
        print(f"\n(parquet 저장 스킵: {e})")
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
