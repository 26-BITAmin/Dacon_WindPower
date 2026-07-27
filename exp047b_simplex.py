# exp-047b: 3멤버(gspd·q60·seq) 가중 심플렉스 미세 스윕 (사전등록 8칸, 재학습 0)
# 근거: exp-047에서 q60+seq_w65(seq:q60=1:1)가 세션 유일의 두 지표 동반 상승.
#   seq 비중을 낮춘 심플렉스 칸은 미탐색. 판정: 3중 조건 (ref 대비).
# 실행: C:\miniconda3\python.exe -X utf8 exp047b_simplex.py
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import load_gbm_cache, score_from_tidy, FOLD_LIST
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE, build_slices
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE
from exp047_seq_dl import DL_PRED_CACHE as SEQ_CACHE, postprocess_flex, PP_NEW, PP_CHAMP

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]

t0 = time.time()
print("### exp-047b: 3멤버 심플렉스 (사전등록 8칸) ###", flush=True)
cache, champ_bias = load_gbm_cache()
cache = cache.rename(columns={"gbm_cf": "cf_g132"})
for src, cols in [(EXT_GBM_CACHE, ["cf_gspd"]), (E39_CACHE, ["dl_q60", "dl_hub"]), (SEQ_CACHE, ["dl_seq"])]:
    df = pd.read_parquet(src)
    for c in cols:
        m = cache[KEY].merge(df[KEY + [c]], on=KEY, how="left")
        assert m[c].notna().all(), c
        cache[c] = m[c].to_numpy()

fold_names = [fn for fn, _ in FOLD_LIST]
hold = HOLD_FOLD["name"]
cv_folds = [fn for fn in fold_names if fn != hold]
slices = build_slices(cache)
gspd = cache["cf_gspd"].to_numpy()
q60 = cache["dl_q60"].to_numpy()
seq = cache["dl_seq"].to_numpy()


def evaluate(blend, pp):
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})


r_champ = evaluate(0.70 * cache["cf_g132"].to_numpy() + 0.30 * cache["dl_hub"].to_numpy(), PP_CHAMP)
r_ref = evaluate(0.70 * gspd + 0.30 * q60, PP_NEW)
c_cv, c_hd = r_champ["cv"][0], r_champ["hold"][0]
ref_cv, ref_hd = r_ref["cv"][0], r_ref["hold"][0]
ref_n, ref_f = r_ref["cv"][1], r_ref["cv"][2]
print(f"챔피언(구PP): CV {c_cv:.4f} | HOLD {c_hd:.4f}")
print(f"ref(gspd·q60·w70·신PP): CV {ref_cv:.4f} | HOLD {ref_hd:.4f}")

CELLS = [(wg, ws_) for wg in (0.65, 0.70) for ws_ in (0.05, 0.10, 0.15, 0.20)]
res = {}
for wg, ws_ in CELLS:
    wq = round(1 - wg - ws_, 2)
    res[f"g{int(wg*100)}q{int(wq*100)}s{int(ws_*100)}"] = evaluate(wg * gspd + wq * q60 + ws_ * seq, PP_NEW)

print("\n" + "=" * 100)
print(f"{'변형':<14}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}{'Δn':>8}{'Δf':>8}  fold별 ΔCV / 판정")
for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ref_cv, kv[1]["hold"][0] - ref_hd)):
    dcv, dhd = r["cv"][0] - ref_cv, r["hold"][0] - ref_hd
    dn, df_ = r["cv"][1] - ref_n, r["cv"][2] - ref_f
    fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
    mag = dcv > 0.001 and dhd > 0.001
    cons = all(x > -0.0002 for x in fold_d)
    both = dn > 0 and df_ > 0
    verdict = ("★3중충족" if (mag and cons and both) else
               ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
    print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}{dn:>+8.4f}{df_:>+8.4f}  "
          + " ".join(f"{x:+.4f}" for x in fold_d) + f"  {verdict}")

print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
