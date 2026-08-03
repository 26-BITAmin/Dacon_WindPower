# ═══════════════════════════════════════════════════════
# exp-112 (A단계): 제출 구성 검증 — 보드 챔피언 base(cf_ens3) × dl_hpo_L0
#
# exp-100~103은 게이트 표준 챔피언(cf_hpo × dl_cln03) 대비로 dl_hpo_L0를 확립했다.
# 실제 보드 챔피언은 GBM이 cf_ens3(355피처)이므로, 제출 전에 **정확히 그 base**로
# 같은 판정을 반복한다. 판정 Δ = 보드 챔피언 구성 ens3×dC_w65 (동일 런).
# 변경점 1개(DL 교체)만 — w는 0.65 유지가 기본, 사다리는 참고.
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 exp112_ens3_l0.py
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
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]
SEARCH_FOLD = "fold_2024Q4"
REF_DL = "dl_cln03"
WS = (0.60, 0.65, 0.70, 0.75)


def delevel_lofo(cache, src, ref, out):
    mm = cache["target"].notna().to_numpy()
    fold, grp = cache["fold"].to_numpy(), cache["group"].to_numpy()
    d = cache[src].to_numpy() - cache[ref].to_numpy()
    vals = cache[src].to_numpy().copy()
    for f in np.unique(fold):
        for g in np.unique(grp):
            other = mm & (fold != f) & (grp == g)
            if other.sum():
                vals[(fold == f) & (grp == g)] -= d[other].mean()
    cache[out] = vals


def main():
    t0 = time.time()
    print("### exp-112A: 보드 챔피언 base(cf_ens3) × dl_hpo_L0 검증 ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp073_gbm_ens.parquet", ["cf_ens3"])
    assert merge_cached_cols(cache, CACHE_DIR / "exp058_dl_hpo.parquet", ["dl_hpo"])
    delevel_lofo(cache, "dl_hpo", REF_DL, "dl_hpo_L0")

    ev = make_ev(cache, champ_bias)
    board = ev("cf_ens3", REF_DL, 0.65, PP_NEW)      # 보드 챔피언 구성 재현
    gate = ev("cf_hpo", REF_DL, 0.65, PP_NEW)        # 게이트 표준 챔피언 (참고)
    b_cv, b_hd = board["cv"][0], board["hold"][0]
    print(f"\n보드 챔피언(ens3×dC_w65) 재현: CV {b_cv:.4f} | HOLD {b_hd:.4f}"
          f"  (exp-089 기록 0.6559/0.6544)", flush=True)
    print(f"게이트 챔피언(hpo×dC_w65) 재현: CV {gate['cv'][0]:.4f} | HOLD {gate['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)

    print(f"\n{'셀':<22}{'CV':>8}{'Δ보드':>9}{'HOLD':>8}{'Δ보드':>9}{'Δn':>9}{'Δf':>9}"
          f"  fold(Δ보드)", flush=True)
    res = {}
    for w in WS:
        for d in ("dl_hpo_L0", "dl_hpo"):
            r = ev("cf_ens3", d, w, PP_NEW)
            res[(d, w)] = r
            fd = " ".join(f"{r['folds'][fn]-board['folds'][fn]:+.4f}" for fn in CV_FOLDS)
            print(f"ens3x{d[3:]:<12}w{int(w*100):<4}{r['cv'][0]:>8.4f}{r['cv'][0]-b_cv:>+9.4f}"
                  f"{r['hold'][0]:>8.4f}{r['hold'][0]-b_hd:>+9.4f}"
                  f"{r['cv'][1]-board['cv'][1]:>+9.4f}{r['cv'][2]-board['cv'][2]:>+9.4f}  {fd}",
                  flush=True)

    # Q4(선택 폴드) 제외 pooled — 제출 셀 한정
    slices = build_slices(cache)
    def tidy_of(gcol, dcol, w):
        blend = w * cache[gcol].to_numpy() + (1 - w) * cache[dcol].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return pd.concat(parts, ignore_index=True)

    noq4 = lambda t: (t["fold"] != HOLD) & (t["fold"] != SEARCH_FOLD)
    tb = tidy_of("cf_ens3", REF_DL, 0.65)
    tc = tidy_of("cf_ens3", "dl_hpo_L0", 0.65)
    sb, sc_ = score_from_tidy(tb[noq4(tb)], "pred"), score_from_tidy(tc[noq4(tc)], "pred")
    print(f"\n[Q4 제외 3폴드 pooled] 보드 챔피언 {sb[0]:.4f} → 후보 {sc_[0]:.4f} "
          f"(Δ {sc_[0]-sb[0]:+.4f} | Δn {sc_[1]-sb[1]:+.4f} | Δf {sc_[2]-sb[2]:+.4f})", flush=True)

    mm = cache["target"].notna()
    bm = (0.65 * cache.loc[mm, "cf_ens3"] + 0.35 * cache.loc[mm, REF_DL]).mean()
    cm = (0.65 * cache.loc[mm, "cf_ens3"] + 0.35 * cache.loc[mm, "dl_hpo_L0"]).mean()
    print(f"블렌드 평균 이동(후보 − 보드): {(cm-bm)*100:+.3f}%cap", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
