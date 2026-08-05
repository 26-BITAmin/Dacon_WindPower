# ═══════════════════════════════════════════════════════
# AN-09: 공정 비교 — 같은 입력정보에서 AnEn vs GBM
#
# AN-08: GBM(352피처) 1.0385 < AnEn(10예측자, 최적가중) 1.2070.
# 질문: 격차가 "함수족"인가 "입력정보량"인가?
#   → GBM을 AnEn과 동일한 10컬럼(hub 예측자)만으로 학습해 비교.
#   (AnEn의 ±1h 시간창과 조건을 맞추기 위해 ±1h 시프트 포함판(30컬럼)도 병행)
#
# 실행: C:\miniconda3\python.exe -X utf8 an09_fair.py
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from an07_wsanen import ws_analogs
from an08_wsopt import wsset
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BEST_W = {1: (2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
          2: (1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5),
          3: (1.0, 2.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0)}
HUB10 = ["gfs_ws_100m", "gfs_wd_sin_100m", "gfs_wd_cos_100m", "ldaps_ws_10m",
         "ec_ws", "jmsm_ws", "jgsm_ws", "gfs_wind_shear_100_10",
         "ldaps_heightAboveGround_2_t", "ldaps_surface_0_sp"]


def main():
    t0 = time.time()
    print("### AN-09 공정 비교 (동일 입력 AnEn vs GBM) ###", flush=True)
    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    for g in [1, 2, 3]:
        sc = scada_h[g][["forecast_kst_dtm", "park_ws"]]
        train_df = train_df.merge(
            sc.assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"], how="left",
            suffixes=("", "_dup"))
        if "park_ws_dup" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_dup"))
    # ±1h 시프트 컬럼 (그룹별 시계열 내에서)
    lagcols = []
    train_df = train_df.sort_values(["kpx_group", "forecast_kst_dtm"]).reset_index(drop=True)
    for c in HUB10:
        for sh, tagn in ((-1, "m1"), (1, "p1")):
            cn = f"{c}_{tagn}"
            train_df[cn] = train_df.groupby("kpx_group")[c].shift(sh)
            lagcols.append(cn)
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    import lightgbm as lgb
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        fold = folds[fname]
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            m = va_g["park_ws"].notna().to_numpy()
            y = va_g["park_ws"].to_numpy()
            trm = tr_g["park_ws"].notna().to_numpy()

            def add(model, p):
                e = p[m] - y[m]
                recs.append(dict(fold=fname, group=g, model=model,
                                 mae=np.abs(e).mean(), rmse=np.sqrt((e**2).mean())))

            for tag, cols in (("gbm10", HUB10), ("gbm30", HUB10 + lagcols)):
                mdl = lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                                        random_state=0, n_jobs=8, verbose=-1,
                                        **e56["lgb"])
                mdl.fit(tr_g.loc[trm, cols], tr_g.loc[trm, "park_ws"])
                add(tag, mdl.predict(va_g[cols]))
            ana = ws_analogs(tr_g, va_g, wsset(*BEST_W[g]))
            add("anen_K50", ana[:, :50].mean(1))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    agg = df.groupby("model")[["mae", "rmse"]].mean().sort_values("mae")
    print("\n=== 4폴드 평균 (m/s) — 참조: GBM352 1.0385 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    piv = df.pivot_table(index="model", columns="group", values="mae")
    print("\n=== 그룹별 MAE ===", flush=True)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
