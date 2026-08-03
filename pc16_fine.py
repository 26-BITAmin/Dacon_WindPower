# ═══════════════════════════════════════════════════════
# PC-16 (R16): PC-14 정점 미세 격자 — (pcden, pchpo, dlvc, dC) 0.025 스텝
#
# 판정: Δ = v1. 후보 = 양 게이트 +0.001 · Δn≥−0.0001 · 폴드 3/4.
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc16_fine.py
# ═══════════════════════════════════════════════════════
import sys
import time
from itertools import product

import numpy as np

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEMBERS = ["cf_pcden", "cf_pchpo", "dl_dlvc", "dl_cln03"]


def main():
    t0 = time.time()
    print("### PC-16: PC-14 정점 미세 격자 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("exp085_gbm_pchpo.parquet", ["cf_pchpo"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"]),
                  ("exp082_gbm_pcinv.parquet", ["cf_pcinv"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()

    reg = {}
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            reg[(fname, group)] = (fwd, sel,
                                   {c: inv(cache.loc[sel, c].to_numpy()) for c in MEMBERS})
    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"준비 ({time.time()-t0:.0f}s) | v1 {v1_cv:.4f}/{v1_hd:.4f} | "
          f"챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    res = {}
    for a, p, b in product(np.arange(0.625, 0.776, 0.025),
                           (0.0, 0.05),
                           np.arange(0.15, 0.276, 0.025)):
        c = 1 - a - p - b
        if c < -1e-9 or c > 0.20:
            continue
        wts = tuple(round(x, 3) for x in (a, p, b, c))
        label = "f_" + "_".join(str(int(x*1000)) for x in wts)
        vals = np.full(len(cache), np.nan)
        for (fname, group), (fwd, sel, v) in reg.items():
            vals[sel] = fwd(sum(w * v[m] for w, m in zip(wts, MEMBERS)))
        assert not np.isnan(vals).any()
        cache[label] = vals
        res[wts] = ev(label, None, 1.0, PP_NEW)
    print(f"격자 {len(res)}셀 채점 완료", flush=True)

    print(f"\n{'(pcden,pchpo,dlvc,dC)':<28}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}"
          f"{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    ranked = sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                     kv[1]["hold"][0] - v1_hd))
    for wts, r in ranked[:10]:
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{str(wts):<28}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)
    best, r = ranked[0]
    print(f"\n[정점] {best}: 챔피언 대비 {r['cv'][0]-champ['cv'][0]:+.4f}/"
          f"{r['hold'][0]-champ['hold'][0]:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
