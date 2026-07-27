# exp-049b: 정제 이득의 후처리 독립성 체크 — 구PP(c16·a50)에서도 이득 유지되는가 (채점만)
# 실행: C:\miniconda3\python.exe -X utf8 exp049b_ppcheck.py
import sys

import pandas as pd

from exp049_clean import (load_eval_frame, postprocess_flex, PP_CHAMP, PP_NEW, KEY,
                          GBM_CLEAN_CACHE)
from exp035_raw_dl import score_from_tidy, FOLD_LIST
from exp019_multigrid import HOLD_FOLD
from exp005_qopt import CAPACITY_KWH
from exp039_dl_diverse import build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cache, champ_bias = load_eval_frame()
prev = pd.read_parquet(GBM_CLEAN_CACHE)
m = cache[KEY].merge(prev[KEY + ["cf_cln03"]], on=KEY, how="left")
assert m["cf_cln03"].notna().all()
cache["cf_cln03"] = m["cf_cln03"].to_numpy()

hold = HOLD_FOLD["name"]
cv_folds = [fn for fn, _ in FOLD_LIST if fn != hold]
slices = build_slices(cache)


def ev(gbm_col, w, pp):
    blend = w * cache[gbm_col].to_numpy() + (1 - w) * cache["dl_q60"].to_numpy()
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    return (score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
            score_from_tidy(tidy[tidy["fold"] == hold], "pred"))


print(f"{'구성':<28}{'CV':>8}{'HOLD':>8}{'1-NMAE':>9}{'FiCR':>8}")
for pp_name, pp in [("구PP(c16·a50)", PP_CHAMP), ("신PP(al40·sn93·pt27)", PP_NEW)]:
    for gc, label in [("cf_gspd", "gspd(dirty)"), ("cf_cln03", "cln(clean)")]:
        cv, hd = ev(gc, 0.70, pp)
        print(f"{label+'·q60·w70·'+pp_name:<28}{cv[0]:>8.4f}{hd[0]:>8.4f}{cv[1]:>9.4f}{cv[2]:>8.4f}")
print("\n→ 두 PP 모두에서 clean−dirty 격차가 비슷하면 이득은 PP 독립(견고).")
