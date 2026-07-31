# ═══════════════════════════════════════════════════════
# exp-081: 후처리-블렌드 합성 순서 — pp(블렌드) vs 블렌드(pp(멤버)) (캐시 전용, 재학습 0)
#
# 배경: 파이프라인은 항상 CF 블렌드 → 편향보정 → smoothing → snap → clip 순서.
#   snap(max)·clip은 비선형이라 순서 교환이 동치가 아님 — 특히 blend(max_g, max_d) ≥
#   max(blend) (Jensen)이라 스냅 구간에서 상향 = FiCR 방향. 순서 축은 exp-027/043/044/
#   067 전부에서 미탐색 (전부 "블렌드 후 PP" 고정).
# 변형: 멤버별로 전체 PP(동일 champ_bias·PP_NEW) 적용 후 kWh 수준 가중 평균.
#   mixed 변형 = 편향보정·smoothing은 블렌드 후(현행), snap·clip만 멤버별 → 평균은
#   snap이 pA 공유라 동치에 가까워 생략, 전체 순서 교환만 측정.
# 판정: 표준 이중 게이트, Δ = 보드 챔피언(hpo×dC_w65, 재현 기대 CV 0.6556/0.6535).
# 실행: C:\miniconda3\python.exe -X utf8 exp081_pporder.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    print("### exp-081 후처리-블렌드 합성 순서 (캐시 전용) ###", flush=True)
    cache, champ_bias = load_cache_full()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    slices = build_slices(cache)
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn, _ in FOLD_LIST if fn != hold]
    gv = cache["cf_hpo"].to_numpy()
    dv = cache["dl_cln03"].to_numpy()

    def ev_ppfirst(w):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            ws, pA = sub["ws"].to_numpy(), sub["pA"].to_numpy()
            pg = postprocess_flex(gv[sub.index], cap, ws, pA, group, champ_bias[fname], **PP_NEW)
            pdl = postprocess_flex(dv[sub.index], cap, ws, pA, group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]]
                         .assign(pred=w * pg + (1 - w) * pdl))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0]
                           for fn in cv_folds})

    res = {}
    for w in (0.65, 0.70):
        res[f"ppfirst_w{int(w*100)}"] = ev_ppfirst(w)
    # 반반 절충: pp(블렌드)와 블렌드(pp(멤버))의 산술 평균
    def ev_half(w):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            ws, pA = sub["ws"].to_numpy(), sub["pA"].to_numpy()
            bl = w * gv[sub.index] + (1 - w) * dv[sub.index]
            p_late = postprocess_flex(bl, cap, ws, pA, group, champ_bias[fname], **PP_NEW)
            pg = postprocess_flex(gv[sub.index], cap, ws, pA, group, champ_bias[fname], **PP_NEW)
            pdl = postprocess_flex(dv[sub.index], cap, ws, pA, group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]]
                         .assign(pred=0.5 * p_late + 0.5 * (w * pg + (1 - w) * pdl)))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0]
                           for fn in cv_folds})
    res["pphalf_w65"] = ev_half(0.65)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
