# ═══════════════════════════════════════════════════════
# AN-17: 풍향 종목에 GBM 출전 — 공정 대조
#
# 사용자 질문 "EMOS 개조판이 GBM보다도 좋았나?"에 답하려면 풍향/벡터 종목에
# GBM이 출전한 적이 없다는 공백을 메워야 한다.
# 구성: LGBM(352피처, e56 하이퍼, 1시드)을 u_obs·v_obs에 성분별 학습 →
#       예측 벡터의 풍향 MAE·벡터 RMSE·|V| 풍속 MAE를 AN-13과 동일 마스크로.
#
# 실행(루트에서): C:\miniconda3\python.exe -X utf8 "open (1)/testing/an17_gbmvec.py"
# ═══════════════════════════════════════════════════════
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from an12_emos import build_scada_vec
from exp035_raw_dl import FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    print("### AN-17 GBM 벡터(u,v) 대조 — 풍향 종목 공정 비교 ###", flush=True)
    train_df, use_cols = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    import lightgbm as lgb
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, folds[fname])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g]
            va_g = va_all.loc[va_all["kpx_group"] == g]
            trm = tr_g["u_obs"].notna()
            vam = va_g["u_obs"].notna().to_numpy()
            pred = {}
            for comp in ["u_obs", "v_obs"]:
                mdl = lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                                        random_state=0, n_jobs=8, verbose=-1,
                                        **e56["lgb"])
                mdl.fit(tr_g.loc[trm, use_cols], tr_g.loc[trm, comp])
                pred[comp] = mdl.predict(va_g[use_cols])
            P = np.c_[pred["u_obs"], pred["v_obs"]][vam]
            Y = np.c_[va_g["u_obs"], va_g["v_obs"]][vam]
            y_ws = va_g["park_ws"].to_numpy()[vam]
            dir_mae = np.abs((np.rad2deg(np.arctan2(P[:, 1], P[:, 0])
                              - np.arctan2(Y[:, 1], Y[:, 0])) + 180) % 360 - 180).mean()
            recs.append(dict(
                fold=fname, group=g, dir_mae=dir_mae,
                vec_rmse=np.sqrt(((P - Y) ** 2).sum(1).mean()),
                ws_mae_norm=np.abs(np.linalg.norm(P, axis=1) - y_ws).mean()))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    agg = df[["dir_mae", "vec_rmse", "ws_mae_norm"]].mean()
    print("\n=== GBM 벡터(u,v 성분별 352피처) — 4폴드·3그룹 평균 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\n참조 (동일 마스크·지표): EMOS premix(AN-13) 풍향 21.1°·vec_rmse 2.76 · "
          "원시 5멤버 28.6° | GBM 풍속 전용판(AN-08) 1.039", flush=True)
    print("그룹별 풍향 MAE:", flush=True)
    print(df.pivot_table(index="fold", columns="group", values="dir_mae")
          .to_string(float_format=lambda x: f"{x:.1f}"), flush=True)
    df.to_csv(ROOT / "open (1)" / "testing" / "an17_results.csv", index=False,
              encoding="utf-8-sig")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
