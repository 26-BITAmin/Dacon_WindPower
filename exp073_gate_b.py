# ═══════════════════════════════════════════════════════
# exp-073b: enfo 피처 수 사다리 마지막 칸 — 1-피처 변형(cf_ens1 = +ens_spread만)
#   + w 사다리 확장 {0.55~0.75} (cf_ens5/cf_ens3는 캐시 재사용, 재학습은 ens1만)
#
# 근거: 본 게이트에서 피처 수 단조 기울기 관측 (ens5(+5) < ens3(+3)) →
#   사다리 끝점 = 기존 정보와 유일하게 안 겹치는 신호 유형(스프레드) 1개만.
#   ens_ws는 ec_ws와 정보 중복(corr(cf_ens3, cf_hpo)=0.9996이 정황).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp073_gate_b.py
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp073_gate import ENS_CACHE, prepare_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALL_COLS = ["cf_ens5", "cf_ens3", "cf_ens1"]


def main():
    t0 = time.time()
    print("### exp-073b: 1-피처 변형(ens_spread) + w 사다리 ###", flush=True)
    train_df, use_cols = prepare_frame()
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    have = merge_cached_cols(cache, ENS_CACHE, ALL_COLS)
    if "cf_ens1" not in have:
        cols = use_cols + ["ens_spread"]
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, 3, 8,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            preds.append(p.rename(columns={"cf_hpo": "cf_ens1"}))
            print(f"  [cf_ens1:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_ens1"].notna().all(), "정렬 실패"
        cache["cf_ens1"] = m["cf_ens1"].to_numpy()
        have.append("cf_ens1")
        save_cols(cache, ENS_CACHE, have)
        print(f"  캐시 저장 → {ENS_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    c = np.corrcoef(cache.loc[mm, "cf_ens1"], cache.loc[mm, "cf_hpo"])[0, 1]
    shift = (cache["cf_ens1"].to_numpy() - cache["cf_hpo"].to_numpy()).mean() * 100
    print(f"  corr(cf_ens1, cf_hpo) = {c:.4f} | 평균 이동 {shift:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}  "
          f"(기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW),
           "ens1_solo": ev("cf_ens1", None, 1.0, PP_NEW)}
    for w in (0.55, 0.60, 0.65, 0.70, 0.75):
        for tag, col in (("ens1", "cf_ens1"), ("ens3", "cf_ens3"), ("ens5", "cf_ens5")):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    d_cv = res["ens1_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
    d_hd = res["ens1_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
    print(f"GBM 단독 델타 (ens1−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
