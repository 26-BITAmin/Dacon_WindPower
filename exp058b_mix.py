# exp-058b: dl_hpo(상향 편향 +0.062)를 dl_cln03과 혼합해 희석 — 캐시 채점만 (재학습 0)
import sys
import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, PP_NEW,
                          postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import merge_cached_cols
from exp056_hpo import GBM_HPO_CACHE
from exp058_dl_hpo import DL_HPO_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cache, champ_bias = load_eval_frame()
assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
assert merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
assert merge_cached_cols(cache, DL_HPO_CACHE, ["dl_hpo"])

fold_names = [fn for fn, _ in FOLD_LIST]
hold = HOLD_FOLD["name"]
cv_folds = [fn for fn in fold_names if fn != hold]
slices = build_slices(cache)


def ev(dl_arr, w):
    blend = w * cache["cf_hpo"].to_numpy() + (1 - w) * dl_arr
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})


dC = cache["dl_cln03"].to_numpy()
dH = cache["dl_hpo"].to_numpy()
r_ref = ev(dC, 0.65)
print(f"ref(hpo×dC_w65): CV {r_ref['cv'][0]:.4f} (n {r_ref['cv'][1]:.4f} f {r_ref['cv'][2]:.4f}) | "
      f"HOLD {r_ref['hold'][0]:.4f}  (기대 0.6556/0.6535)")
res = {}
for mr in (0.25, 0.5, 0.75, 1.0):
    dl = (1 - mr) * dC + mr * dH
    for w in (0.70, 0.65):
        res[f"mixH{int(mr*100)}_w{int(w*100)}"] = ev(dl, w)
ch_cv, ch_hd, ch_n, ch_f = r_ref["cv"][0], r_ref["hold"][0], r_ref["cv"][1], r_ref["cv"][2]
print(f"{'변형':<14}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}  fold별 ΔCV")
for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
    fd = " ".join(f"{r['folds'][fn]-r_ref['folds'][fn]:+.4f}" for fn in cv_folds)
    print(f"{name:<14}{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}"
          f"{r['cv'][1]-ch_n:>+9.4f}{r['cv'][2]-ch_f:>+9.4f}  {fd}")
