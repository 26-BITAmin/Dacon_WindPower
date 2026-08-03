# ═══════════════════════════════════════════════════════
# PC-18a: 트랙 GBM 입력에 GEFS 3피처 — 데이터 다양화 (피처 채널, 완전성 확인)
#
# 근거: 사용자 "데이터 여러 개 다양하게 써봤어?" — GEFS는 트랙 미시도.
#   예상은 중립(메인 exp-089 corr 0.9994 + PC-10에서 벽의 타깃 무관성 확인)이나
#   실측으로 닫는다. 구조는 pc10_ens3.py 미러 (ens3 → gefs3).
#
# 실행: C:\miniconda3\python.exe -X utf8 pc18a_gefs.py   (~14분)
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame
from exp089_gefs_features import build_gefs_frame
from pc01_pcinv import VARIANTS, group_fold_data, fit_predict_cf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GEFS3 = ["gefs_ws", "gefs_spread", "gefs_gap"]
CACHE_P = CACHE_DIR / "exp082_gbm_pcgefs.parquet"
VARIANTS["pcgefs"] = dict(q=None)


def main():
    t0 = time.time()
    print("### PC-18a: 트랙 GBM + GEFS3 (355피처, 3-seed) ###", flush=True)
    train_df, use_cols = build_train_frame()
    gf = build_gefs_frame()
    train_df = train_df.merge(gf, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["gefs_gap"] = train_df["gefs_ws"] - train_df["gfs_ws_10m"]
    cols = list(use_cols) + GEFS3
    print(f"피처 {len(cols)} | train gefs 커버리지 {train_df['gefs_ws'].notna().mean():.3f}", flush=True)
    scada_h = build_scada_hourly()
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])

    if "cf_pcgefs" not in merge_cached_cols(cache, CACHE_P, ["cf_pcgefs"]):
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            rows = []
            for group in (1, 2, 3):
                d = group_fold_data(train_df, scada_h, fold, group, cols, None)
                if d is None:
                    continue
                cf, _ = fit_predict_cf(d, "pcgefs", e56, 3, 8)
                rows.append(pd.DataFrame({
                    "fold": fold["name"], "group": group,
                    "forecast_kst_dtm": d["va_g"]["forecast_kst_dtm"].to_numpy(),
                    "cf_pcgefs": cf}))
            preds.append(pd.concat(rows, ignore_index=True))
            print(f"  [pcgefs:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_pcgefs"].notna().all()
        cache["cf_pcgefs"] = m["cf_pcgefs"].to_numpy()
        save_cols(cache, CACHE_P, ["cf_pcgefs"])
        print(f"  캐시 저장 → {CACHE_P.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    r = ev("cf_pcgefs", "dl_cln03", 0.65, PP_NEW)
    s = ev("cf_pcgefs", None, 1.0, PP_NEW)
    si = ev("cf_pcinv", None, 1.0, PP_NEW)
    mm = cache["target"].notna()
    print(f"\nv1 {v1['cv'][0]:.4f}/{v1['hold'][0]:.4f} (기대 0.6530/0.6512)", flush=True)
    print(f"pcgefs×dC_w65: ΔCV {r['cv'][0]-v1['cv'][0]:+.4f} / Δhold {r['hold'][0]-v1['hold'][0]:+.4f}", flush=True)
    print(f"pcgefs solo Δ(pcinv solo): {s['cv'][0]-si['cv'][0]:+.4f}/{s['hold'][0]-si['hold'][0]:+.4f}", flush=True)
    print(f"corr(cf_pcgefs, cf_pcinv) = "
          f"{np.corrcoef(cache.loc[mm,'cf_pcgefs'], cache.loc[mm,'cf_pcinv'])[0,1]:.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
