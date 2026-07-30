# ═══════════════════════════════════════════════════════
# exp-062: 그룹별 트리 수 — exp-056 공유 파라미터 유지, 조기종료가 요구한 그룹별 n만 반영
#
# 배경: exp-056 게이트는 그룹 공유 트리 수(median×1.1 = LGB 448/XGB 585)를 강제했으나
#   탐색의 조기종료는 그룹별 [407, 983, 88](LGB)·[532, 1448, 91](XGB)을 요구 —
#   그룹2는 절반만 학습, 그룹3은 5배 과학습. exp-060(그룹별 파라미터 = Q4 과적합)과 달리
#   파라미터는 일반화 검증된 공유 우승을 유지하고 트리 수(1차원·조기종료 유도)만 그룹별화.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp062_ntrees_pg.py
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, merge_avail
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import (train_gate_fold_pg, load_cache_full, make_ev, print_table)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GBM_NTG_CACHE = CACHE_DIR / "exp062_gbm_ntg.parquet"
# exp-056 탐색 best trial의 그룹별 조기종료 iters × 1.1
LGB_N = {1: 448, 2: 1081, 3: 97}
XGB_N = {1: 585, 2: 1593, 3: 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-062 그룹별 트리 수 (공유 파라미터) ###", flush=True)

    e56 = json.loads(E56_JSON.read_text())
    pg = {str(g): dict(lgb=e56["lgb"], lgb_n=LGB_N[g], xgb=e56["xgb"], xgb_n=XGB_N[g])
          for g in [1, 2, 3]}
    for g in [1, 2, 3]:
        print(f"  g{g}: LGB {LGB_N[g]}트리 (현행 448) | XGB {XGB_N[g]}트리 (현행 585)", flush=True)

    cache, champ_bias = load_cache_full()
    have = merge_cached_cols(cache, GBM_NTG_CACHE, ["cf_pg"])
    if "cf_pg" not in have:
        train_df, cols, colsets = build_inputs(["spd"])
        ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
        train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
        train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
        train_df = merge_avail(train_df)
        use_cols = colsets["spd"] + EC_COLS
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(train_gate_fold_pg(train_df, use_cols, fold, args.seeds, args.jobs, pg))
            print(f"  [ntg:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_pg"].notna().all(), "정렬 실패"
        cache["cf_pg"] = m["cf_pg"].to_numpy()
        save_cols(cache, GBM_NTG_CACHE, ["cf_pg"])
        print(f"  캐시 저장 → {GBM_NTG_CACHE.name}", flush=True)
    cache = cache.rename(columns={"cf_pg": "cf_ntg"})

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"ntg_w100": ev("cf_ntg", None, 1.0, PP_NEW),
           "hpo_w100": ev("cf_hpo", None, 1.0, PP_NEW)}
    for w in (0.70, 0.65, 0.60):
        res[f"ntg×dC_w{int(w*100)}"] = ev("cf_ntg", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    d_cv = res["ntg_w100"]["cv"][0] - res["hpo_w100"]["cv"][0]
    d_hd = res["ntg_w100"]["hold"][0] - res["hpo_w100"]["hold"][0]
    print(f"\nGBM 단독 델타 (ntg−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
