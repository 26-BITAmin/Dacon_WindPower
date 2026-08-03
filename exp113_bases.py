# ═══════════════════════════════════════════════════════
# exp-113: dl_hpo_L0의 GBM 파트너 선별 — 대안 base 쌍의 Q4 제외 재판정
#
# exp-100에서 dl_hpo와의 쌍 상위에 gfens·gf3·pcaA16·rc4·xgb base가 올랐고,
# exp-101에서 gfens×hpo_L0_w60이 +0.0026/+0.0031로 hpo×hpo_L0와 비등했다.
# 제출은 변경점 1개(hpo base 유지)로 확정했지만, 검증 기록으로서
# "대안 base가 Q4 제외에서도 hpo를 이기는가"를 닫는다.
# 주의: 대안 base는 각각 이미 폐쇄된 축의 산물(gfens/gf3=GEFS, pcaA16=PCA,
# rc4=최근성 가중, xgb=단일 계열) — 이기더라도 변경점 2개 묶음이라 제출 대상 아님.
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 exp113_bases.py
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
SOURCES = {
    "exp089_gbm_gfens.parquet": ["cf_gfens"],
    "exp089_gbm_gf3.parquet": ["cf_gf3"],
    "exp088_gbm_pcaA16.parquet": ["cf_pcaA16"],
    "exp066_gbm_rc4.parquet": ["cf_rc4"],
    "exp065_gbm_members.parquet": ["cf_xgb"],
    "exp073_gbm_ens.parquet": ["cf_ens3"],
    "exp058_dl_hpo.parquet": ["dl_hpo"],
}
BASES = ["cf_hpo", "cf_ens3", "cf_gfens", "cf_gf3", "cf_pcaA16", "cf_rc4", "cf_xgb"]
WS = (0.60, 0.65, 0.70)


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
    print("### exp-113: dl_hpo_L0의 GBM 파트너 선별 (Q4 제외 포함) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for fname, cols in SOURCES.items():
        merge_cached_cols(cache, CACHE_DIR / fname, [c for c in cols if c not in cache.columns])
    delevel_lofo(cache, "dl_hpo", REF_DL, "dl_hpo_L0")

    ev = make_ev(cache, champ_bias)
    champ = ev("cf_hpo", REF_DL, 0.65, PP_NEW)
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    print(f"챔피언 재현: CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

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
    tb = tidy_of("cf_hpo", REF_DL, 0.65)
    b_noq4 = score_from_tidy(tb[noq4(tb)], "pred")[0]

    print(f"\n{'base':<10}{'w':>5}{'ΔCV':>9}{'Δhold':>9}{'ΔCV−Q4':>10}{'Δn':>9}{'Δf':>9}"
          f"  fold(Δ챔)", flush=True)
    rows = []
    for b in BASES:
        if b not in cache.columns:
            print(f"{b:<10}  [캐시 없음]", flush=True)
            continue
        for w in WS:
            r = ev(b, "dl_hpo_L0", w, PP_NEW)
            t = tidy_of(b, "dl_hpo_L0", w)
            nq = score_from_tidy(t[noq4(t)], "pred")[0] - b_noq4
            fd = " ".join(f"{r['folds'][fn]-champ['folds'][fn]:+.4f}" for fn in CV_FOLDS)
            rows.append((b, w, r["cv"][0] - ch_cv, r["hold"][0] - ch_hd, nq,
                         r["cv"][1] - champ["cv"][1], r["cv"][2] - champ["cv"][2], fd))
    for b, w, dcv, dhd, nq, dn, df_, fd in sorted(rows, key=lambda x: -x[4]):
        print(f"{b.replace('cf_',''):<10}{w:>5.2f}{dcv:>+9.4f}{dhd:>+9.4f}{nq:>+10.4f}"
              f"{dn:>+9.4f}{df_:>+9.4f}  {fd}", flush=True)

    print(f"\n(판정 참고: 제출 채택 셀 = hpo×hpo_L0_w65. 대안 base는 전부 변경점 2개 묶음)", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
