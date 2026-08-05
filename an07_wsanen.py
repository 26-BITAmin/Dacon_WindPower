# ═══════════════════════════════════════════════════════
# AN-07: AnEn 본래 용도 — "풍속 자체"를 잘 예측하기
#
# 예측 대상: SCADA park_ws (터빈 평균 나셀 풍속, m/s)
# 비교 사다리 (전부 fold-safe):
#   raw    : 원시 예보 풍속 5소스 (gfs100, ldaps10, ec, jmsm, jgsm)
#   lin    : 소스별 선형 보정 park_ws ≈ a·ws + b (train으로 적합)
#   mls    : 5소스 OLS 결합
#   gbm    : LGB 1-seed (352피처, e56 하이퍼) — ML 상한 참고점
#   anen   : Delle Monache AnEn (hub 예측자 · K/분위수 스윕 · 기본/AN-05 가중)
# 평가: 2024 4개 분기 폴드, 그룹별 + pooled MAE·RMSE (m/s)
#
# 실행: C:\miniconda3\python.exe -X utf8 an07_wsanen.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an01_anen import PRED_SETS, anen_dists, lagged_block
from an06_final import PSET_G
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_SOURCES = ["gfs_ws_100m", "ldaps_ws_10m", "ec_ws", "jmsm_ws", "jgsm_ws"]
HW = 3
K_MAX = 300


def ws_analogs(tr_g, va_g, pset):
    """예측 대상만 park_ws로 바꾼 AnEn (an02 build_analogs2 변형)."""
    cols = [c for c, _ in pset]
    w = np.array([wt for _, wt in pset])
    A = lagged_block(tr_g, cols)
    F = lagged_block(va_g, cols)
    ok = tr_g["park_ws"].notna().to_numpy()
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    D = anen_dists(F, A[ok], sig, w)
    h_tr = tr_g.loc[ok, "forecast_kst_dtm"].dt.hour.to_numpy()
    h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()
    dh = np.abs(h_va[:, None] - h_tr[None, :])
    D = np.where(np.minimum(dh, 24 - dh) <= HW, D, np.inf)
    k = min(K_MAX, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    rows = np.arange(D.shape[0])[:, None]
    dsub = D[rows, idx]
    order = np.argsort(dsub, axis=1)
    idx = idx[rows, order]
    return tr_g.loc[ok, "park_ws"].to_numpy()[idx]     # [Nva, K_MAX] 아날로그 실측 풍속


def main():
    t0 = time.time()
    print("### AN-07 풍속 예측 AnEn (대상 = SCADA park_ws) ###", flush=True)
    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    for g in [1, 2, 3]:
        sc = scada_h[g][["forecast_kst_dtm", "park_ws"]]
        train_df = train_df.merge(
            sc.assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"], how="left",
            suffixes=("", "_dup"))
        if "park_ws_dup" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_dup"))
    print(f"입력 준비 완료 | park_ws 커버 "
          f"{train_df['park_ws'].notna().mean()*100:.1f}% — {time.time()-t0:.0f}s",
          flush=True)
    import json
    import lightgbm as lgb
    e56 = json.loads(E56_JSON.read_text())

    folds = [(fn, f) for fn, f in FOLD_LIST if fn.startswith("fold_")]
    recs = []
    for fname, fold in folds:
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            m = va_g["park_ws"].notna().to_numpy()
            if m.sum() < 100:
                print(f"  [{fname} g{g}] 실측 풍속 {m.sum()}시간 — 건너뜀", flush=True)
                continue
            y = va_g["park_ws"].to_numpy()

            def add(model, pred):
                e = pred[m] - y[m]
                recs.append(dict(fold=fname, group=g, model=model, n=int(m.sum()),
                                 mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                                 signed=e.mean()))

            trm = tr_g["park_ws"].notna().to_numpy()
            va_src = va_g[RAW_SOURCES].ffill().bfill()   # 2024 검증측 결측 보간
            for s in RAW_SOURCES:                        # raw + 선형 보정 (관측 쌍만 적합)
                add(f"raw_{s}", va_src[s].to_numpy())
                sel = trm & tr_g[s].notna().to_numpy()
                a, b = np.polyfit(tr_g.loc[sel, s], tr_g.loc[sel, "park_ws"], 1)
                add(f"lin_{s}", a * va_src[s].to_numpy() + b)
            sel = trm & tr_g[RAW_SOURCES].notna().all(axis=1).to_numpy()
            Xtr = tr_g.loc[sel, RAW_SOURCES].to_numpy()  # 다소스 OLS (완전행만)
            Xva = va_src.to_numpy()
            coef, *_ = np.linalg.lstsq(
                np.c_[Xtr, np.ones(len(Xtr))], tr_g.loc[sel, "park_ws"], rcond=None)
            add("mls_ols5", Xva @ coef[:-1] + coef[-1])

            mdl = lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                                    random_state=0, n_jobs=8, verbose=-1, **e56["lgb"])
            mdl.fit(tr_g.loc[trm, use_cols], tr_g.loc[trm, "park_ws"])
            add("gbm_lgb352", mdl.predict(va_g[use_cols]))

            for ptag, pset in (("hub", PRED_SETS["hub"]), ("tuned", PSET_G[g])):
                ana = ws_analogs(tr_g, va_g, pset)
                for K in [20, 50, 100, 200]:
                    add(f"anen_{ptag}_K{K}", ana[:, :K].mean(1))
                add(f"anen_{ptag}_K200q50", np.median(ana[:, :200], axis=1))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    agg = (df.groupby("model")[["mae", "rmse", "signed"]].mean()
             .sort_values("mae"))
    print("\n=== 4폴드 평균 (그룹 평균, m/s) ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:+.4f}"), flush=True)
    print("\n=== 그룹별 MAE (4폴드 평균) ===", flush=True)
    piv = df.pivot_table(index="model", columns="group", values="mae").sort_values(1)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    df.to_csv("an07_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n상세 → an07_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
