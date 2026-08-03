# ═══════════════════════════════════════════════════════
# exp-115: 안전 프로파일 재스캔 — Δn≥0 · Δf>0 · 제출값 평균 비상향
#
# exp-112 LB 역전(−0.00237)의 교훈을 필터로 만든 첫 스캔.
#   역전군 공통 프로파일 = Δn 음수 + Δf 큰 양수 (exp-031/055/058/076/112).
#   따라서 여기서는 처음부터 반대 조건을 요구한다:
#     ① ΔCV > +0.001  ② Δhold > 0  ③ **Δn ≥ 0**  ④ Δf > 0
#     ⑤ 4폴드 일관(전부 > −0.0002)  ⑥ **후처리 후 평균 이동 ≤ +0.05%cap** (exp-114 교훈)
#   dl_hpo·dl_hpob(및 파생)는 LB 역전 확정 계열이라 제외.
#
# 스캔 대상 (재학습 0):
#   A. GBM 전 멤버 × DL 생존 멤버 × w {0.65, 0.70}
#   B. DL 혼합: dl_cln03 ⊕ 파트너 (frac 0.25/0.50) × w 0.65 — 체계 스캔 미실시 축
#
# 실행: C:\miniconda3\python.exe -X utf8 exp115_safescan.py
# ═══════════════════════════════════════════════════════
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SKIP_PAT = re.compile(r"(_partial|test_|^kma_|^tmp_)")
DEAD_DL = {"dl_hpo", "dl_hpob"}                  # LB 역전 확정 계열
HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]
SEARCH_FOLD = "fold_2024Q4"
WS = (0.65, 0.70)
MIX_FRACS = (0.25, 0.50)


def attach_all(cache):
    for path in sorted(Path(CACHE_DIR).glob("*.parquet")):
        if SKIP_PAT.search(path.name):
            continue
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            continue
        want = [c for c in names if c.startswith(("cf_", "dl_")) and c not in cache.columns]
        if want:
            merge_cached_cols(cache, path, want)


def main():
    t0 = time.time()
    print("### exp-115: 안전 프로파일 재스캔 (Δn≥0 · 평균 비상향) ###", flush=True)
    cache, champ_bias = load_cache_full()
    attach_all(cache)
    gbms = [c for c in cache.columns if c.startswith("cf_")]
    dls = [c for c in cache.columns if c.startswith("dl_") and c not in DEAD_DL]
    print(f"GBM {len(gbms)} × DL {len(dls)}(사망 계열 제외) × w {WS}", flush=True)

    # DL 혼합 컬럼 (cln03 앵커)
    mix_cols = []
    for d in dls:
        if d == "dl_cln03":
            continue
        for f in MIX_FRACS:
            col = f"mx_{d[3:]}_{int(f*100)}"
            cache[col] = f * cache[d].to_numpy() + (1 - f) * cache["dl_cln03"].to_numpy()
            mix_cols.append(col)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    ch_n, ch_f = ref["cv"][1], ref["cv"][2]
    print(f"챔피언 재현: CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    # ── 1차: 점수 필터 ①~⑤ ─────────────────────────────
    cells = [(g, d, w) for g in gbms for d in dls for w in WS]
    cells += [("cf_hpo", m, 0.65) for m in mix_cols]
    print(f"총 {len(cells)}셀 스캔 시작", flush=True)
    surv, done = [], 0
    for g, d, w in cells:
        r = ev(g, d, w, PP_NEW)
        done += 1
        if done % 300 == 0:
            print(f"  ...{done}/{len(cells)} ({(time.time()-t0)/60:.1f}분, 생존 {len(surv)})", flush=True)
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        if not (dcv > 0.001 and dhd > 0 and dn_ >= 0 and df_ > 0):
            continue
        if not all(r["folds"][fn] - ref["folds"][fn] > -0.0002 for fn in CV_FOLDS):
            continue
        surv.append((g, d, w, r))
    print(f"\n[1차 필터 ①~⑤] 생존 {len(surv)}셀 / {len(cells)}", flush=True)

    # ── 2차: 후처리 후 평균 이동 ⑥ + Q4 제외 ────────────
    slices = build_slices(cache)
    def tidy_of(gcol, dcol, w):
        blend = w * cache[gcol].to_numpy() + (1 - w) * cache[dcol].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]]
                         .assign(pred=pred, cap=cap))
        return pd.concat(parts, ignore_index=True)

    tb = tidy_of("cf_hpo", "dl_cln03", 0.65)
    noq4 = lambda t: (t["fold"] != HOLD) & (t["fold"] != SEARCH_FOLD)
    b_noq4 = score_from_tidy(tb[noq4(tb)], "pred")[0]
    b_shift_ref = (tb["pred"] / tb["cap"]).mean()

    if surv:
        print(f"\n{'셀':<30}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}"
              f"{'PP후이동%cap':>13}{'ΔCV−Q4':>10}  판정", flush=True)
    final = []
    for g, d, w, r in sorted(surv, key=lambda x: -(x[3]["cv"][0])):
        t = tidy_of(g, d, w)
        shift = ((t["pred"] / t["cap"]).mean() - b_shift_ref) * 100
        nq = score_from_tidy(t[noq4(t)], "pred")[0] - b_noq4
        ok = shift <= 0.05
        name = f"{g.replace('cf_','')}x{d.replace('dl_','')}_w{int(w*100)}"
        print(f"{name:<30}{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}"
              f"{r['cv'][1]-ch_n:>+9.4f}{r['cv'][2]-ch_f:>+9.4f}{shift:>+13.3f}"
              f"{nq:>+10.4f}  {'통과' if ok else '이동 초과'}", flush=True)
        if ok:
            final.append((name, r, shift, nq))

    print(f"\n[최종] 전 6조건 통과: {len(final)}셀", flush=True)
    if not final:
        print("  → 안전 프로파일을 만족하는 셀 없음 — 캐시 재조합 공간에서 이 기준은 공집합.", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
