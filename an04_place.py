# ═══════════════════════════════════════════════════════
# AN-04: AnEn 배치 실험 — "어느 자리에 섞는가"
#
# AN-03 결과: GBM(cf) 자리 혼합은 t90에서도 −0.0005~−0.0009, fold 일관 X.
# 남은 배치 2곳:
#   (a) DL 자리: dl_mix = λ·dl_cln03 + (1−λ)·cf_an_q60 — q60 상향 프로파일이
#       DL(FiCR 엔진)과 동형이므로 여기가 자연스러운 자리라는 가설
#   (b) post-PP 파일 혼합: μ·PP(챔피언) + (1−μ)·PP(AnEn측) — 보드 챔피언(mix70)이
#       나온 검증된 경로. 게이트 합성과 위험 구조가 다름 (양 끝점 내삽)
#
# 실행: C:\miniconda3\python.exe -X utf8 an04_place.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AN_COLS = ["cf_an_m100", "cf_an_m200", "cf_an_q60"]


def pp_tidy(cache, champ_bias, blend, pp):
    """blend(cf 공간, 전체 캐시 정렬) → PP 적용 제출값 tidy."""
    parts = []
    for (fname, group), sub in build_slices(cache):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]]
                     .assign(pred=pred))
    return pd.concat(parts, ignore_index=True)


def score_mix(tidy_a, tidy_b, mu, hold):
    pred = mu * tidy_a["pred"].to_numpy() + (1 - mu) * tidy_b["pred"].to_numpy()
    t = tidy_a.assign(pred=pred)
    return dict(cv=score_from_tidy(t[t["fold"] != hold], "pred"),
                hold=score_from_tidy(t[t["fold"] == hold], "pred"),
                folds={fn: score_from_tidy(t[t["fold"] == fn], "pred")[0]
                       for fn, _ in FOLD_LIST if fn != hold})


def main():
    t0 = time.time()
    print("### AN-04 AnEn 배치 실험 (DL 자리 / post-PP 파일 혼합) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert all(c in merge_cached_cols(cache, CACHE_DIR / "an03_gate.parquet", AN_COLS)
               for c in AN_COLS)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}", flush=True)

    res = {}
    # (a) DL 자리 혼합
    dlv = cache["dl_cln03"].to_numpy()
    for src in ["cf_an_q60", "cf_an_m100"]:
        sv = cache[src].to_numpy()
        for lam in [0.9, 0.8, 0.7, 0.5]:
            col = f"dlmx_{src[-3:]}_{int(lam*100)}"
            cache[col] = lam * dlv + (1 - lam) * sv
            res[f"dl자리_{src[6:]}_l{int(lam*100)}"] = ev("cf_hpo", col, 0.65, PP_NEW)

    # (b) post-PP 파일 혼합
    hold = HOLD_FOLD["name"]
    champ_blend = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * dlv
    tidy_champ = pp_tidy(cache, champ_bias, champ_blend, PP_NEW)
    rep = score_from_tidy(tidy_champ[tidy_champ["fold"] != hold], "pred")
    print(f"post-PP 챔피언 재현 대조: CV {rep[0]:.4f} (ev와 일치해야)", flush=True)
    sides = {
        "q60solo": cache["cf_an_q60"].to_numpy(),
        "anblend": 0.65 * cache["cf_an_m100"].to_numpy()
                   + 0.35 * cache["cf_an_q60"].to_numpy(),
        "anXdl": 0.65 * cache["cf_an_m100"].to_numpy() + 0.35 * dlv,
    }
    for sname, sv in sides.items():
        tidy_s = pp_tidy(cache, champ_bias, sv, PP_NEW)
        for mu in [0.9, 0.8, 0.7]:
            res[f"ppmix_{sname}_m{int(mu*100)}"] = score_mix(tidy_champ, tidy_s, mu, hold)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
