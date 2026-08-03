# ═══════════════════════════════════════════════════════
# PC-06 (R6): 트랙 base × 전 DL 멤버 완전 교차 — DL 파트너 축의 잔여 전부
#
# 근거: 트랙의 DL 파트너 축은 dl_cln03(PC-01~04)과 dlv/dlvc(PC-05)만 측정됨.
#   메인 라인 exp-100(전 멤버 쌍 교차)이 같은 방법으로 dl_hpo(_L0) 축을 찾았으므로
#   (exp-101~103에서 레벨/정보 분리·Q4 통제까지 통과), 트랙 base에도 전 DL을 교차한다.
#
# 판정 (트랙 규칙): Δ = 트랙 v1 (pcinv × dl_cln03 × w0.65 + PP_NEW, CV 0.6530/HOLD 0.6512).
#   채택 = ΔCV > +0.001 · Δhold > 0 · 4폴드 전부 양수. corr는 판정에 쓰지 않는다(사용자 지시).
#   챔피언 초과를 주장하는 셀은 **동일 DL 교체를 챔피언 base에 적용한 통제**를 병기(PC-04 규칙).
#
# 재학습 0 · 캐시 전용.
# 실행: C:\miniconda3\python.exe -X utf8 pc06_pairgrid.py
# ═══════════════════════════════════════════════════════
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp035_raw_dl import CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SKIP_PAT = re.compile(r"(_partial|test_|^kma_|^tmp_)")
BASES = ["cf_pcinv", "cf_pchpo"]          # 트랙 base (PC-01 / PC-03)
WS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
REF_DL = "dl_cln03"


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


def delevel_lofo(cache, src, ref, out):
    """(fold, group)별 이동량을 그 폴드 제외 나머지에서 추정해 제거 (exp-102와 동일, 라벨 미사용)"""
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
    print("### PC-06: 트랙 base × 전 DL 멤버 교차 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    attach_all(cache)
    for src in ("dl_hpo", "dl_hpob"):
        delevel_lofo(cache, src, REF_DL, f"{src}_L0")

    dls = [c for c in cache.columns if c.startswith("dl_")]
    missing = [b for b in BASES if b not in cache.columns]
    if missing:
        print(f"  [중단] 트랙 base 캐시 누락: {missing}", flush=True)
        return
    print(f"트랙 base {BASES} × DL {len(dls)}개 × w {len(WS)}단 = "
          f"{len(BASES)*len(dls)*len(WS)}셀", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", REF_DL, 0.65, PP_NEW)
    champ = ev("cf_hpo", REF_DL, 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\n트랙 v1 재현: CV {v1_cv:.4f} | HOLD {v1_hd:.4f}  (기대 0.6530/0.6512)", flush=True)
    print(f"챔피언 재현:  CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    res = {}
    for b in BASES:
        for d in dls:
            for w in WS:
                res[(b, d, w)] = ev(b, d, w, PP_NEW)

    # ── v1 대비 상위 표 ──────────────────────────────────
    ranked = sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                     kv[1]["hold"][0] - v1_hd))
    print(f"\n--- v1 대비 상위 20 (판정축 = min(ΔCV, Δhold)) ---", flush=True)
    print(f"{'base':<10}{'DL':<13}{'w':>5}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}"
          f"{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for (b, d, w), r in ranked[:20]:
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{b.replace('cf_',''):<10}{d.replace('dl_',''):<13}{w:>5.2f}"
              f"{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    # ── 채택 판정 (트랙 규칙) ────────────────────────────
    strict = []
    for (b, d, w), r in res.items():
        dcv, dhd = r["cv"][0] - v1_cv, r["hold"][0] - v1_hd
        if dcv > 0.001 and dhd > 0 and \
           all(r["folds"][fn] - v1["folds"][fn] > 0 for fn in cv_folds):
            strict.append(((b, d, w), dcv, dhd, r))
    print(f"\n[채택 판정 — v1 대비] 전 {len(res)}셀 중 "
          f"ΔCV>+0.001 · Δhold>0 · 4폴드 전부 양수: {len(strict)}셀", flush=True)

    # ── 챔피언 대비 + 통제 (PC-04 규칙) ──────────────────
    show = strict if strict else [(k, r["cv"][0] - v1_cv, r["hold"][0] - v1_hd, r)
                                  for k, r in ranked[:8]]
    print(f"\n[챔피언 대비 + 통제] 각 셀의 (a) 챔피언 대비 Δ, "
          f"(b) 동일 DL을 챔피언 base에 적용한 통제 Δ(챔피언 대비)", flush=True)
    print(f"{'셀':<32}{'Δ챔CV':>9}{'Δ챔hd':>9}{'통제ΔCV':>10}{'통제Δhd':>10}", flush=True)
    for (b, d, w), dcv, dhd, r in sorted(show, key=lambda x: -x[1])[:12]:
        ctrl = res.get(("cf_hpo", d, w)) or ev("cf_hpo", d, w, PP_NEW)
        name = f"{b.replace('cf_','')}x{d.replace('dl_','')}_w{int(w*100)}"
        print(f"{name:<32}{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}"
              f"{ctrl['cv'][0]-ch_cv:>+10.4f}{ctrl['hold'][0]-ch_hd:>+10.4f}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
