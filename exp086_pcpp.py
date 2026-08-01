# ═══════════════════════════════════════════════════════
# exp-086: 파워커브 트랙 R4 — **트랙 전용 후처리 재최적화** (캐시 전용, 재학습 0)
#
# 트랙 규칙 (사용자 지시): 판정 기준 = 이 파이프라인의 이전 버전(v1 = pcinv×dC_w65 + PP_NEW).
#
# R4 가설: 트랙은 지금까지 **챔피언용으로 튜닝된 PP_NEW**(clip_lo 0.16 · alpha 0.40 ·
#   snap_thr 0.27 · snap_to 0.93)를 그대로 써 왔음. 그런데 이 파이프라인의 출력 분포는
#   챔피언과 구조적으로 다름 — exp-082b 실측으로 raw base max **0.918**(챔피언 0.986),
#   p99 0.881(0.909). 즉 **snap_to 0.93은 base 최댓값보다 높은 값**이라 같은 파라미터가
#   전혀 다르게 작동함. clip 하한·평활도 분포가 다르면 최적점이 이동.
#   선례: exp-043이 base 교체(gspd·q60) 후 후처리를 그 base 전용으로 재최적화해
#   **+0.0033/+0.0030**을 얻음 — 트랙에서 가장 값싸고(재학습 0) 전례가 강한 축.
#
# 구성: 1-D 사다리(각 파라미터·w) → 상위 축 2-D 그리드. 후보 base 2개(v1 pcinv / R3 pchpo).
#   선택 낙관 방어 = 홀드아웃·fold 일관 동시 확인 (exp-043/067과 동일 프로토콜).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp086_pcpp.py
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np

from exp035_raw_dl import CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

V1_CACHE = CACHE_DIR / "exp082_gbm_pcinv.parquet"
R3_CACHE = CACHE_DIR / "exp085_gbm_pchpo.parquet"


def pp(**kw):
    p = dict(PP_NEW)
    p.update(kw)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cf_pcinv", choices=["cf_pcinv", "cf_pchpo"])
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-086 파워커브 트랙 R4: 트랙 전용 후처리 재최적화 ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, V1_CACHE, ["cf_pcinv"]), "v1 캐시 없음"
    have_r3 = "cf_pchpo" in merge_cached_cols(cache, R3_CACHE, ["cf_pchpo"])
    ev = make_ev(cache, champ_bias)

    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    base = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)          # 트랙 기준 v1
    print(f"[참고] 챔피언: CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}", flush=True)
    print(f"[기준] v1:     CV {base['cv'][0]:.4f} | HOLD {base['hold'][0]:.4f}"
          f"  (기대 0.6530/0.6512)", flush=True)

    b = args.base
    if b == "cf_pchpo" and not have_r3:
        raise SystemExit("R3 캐시 없음")
    print(f"\n--- 스윕 base = {b} ---", flush=True)

    res = {}
    # 1-D 사다리
    for v in (0.10, 0.13, 0.16, 0.19, 0.22):
        res[f"clip{int(v*100)}"] = ev(b, "dl_cln03", 0.65, pp(clip_lo=v))
    for v in (0.25, 0.40, 0.55, 0.70, 1.00):
        res[f"alpha{int(v*100)}"] = ev(b, "dl_cln03", 0.65, pp(alpha=v))
    for v in (0.15, 0.21, 0.27, 0.33, 0.40):
        res[f"thr{int(v*100)}"] = ev(b, "dl_cln03", 0.65, pp(snap_thr=v))
    # snap_to: base max 0.918 근방을 촘촘히 (챔피언 0.93은 base max보다 위)
    for v in (0.85, 0.88, 0.90, 0.93, 0.96):
        res[f"snapto{int(v*100)}"] = ev(b, "dl_cln03", 0.65, pp(snap_to=v))
    for w in (0.55, 0.60, 0.65, 0.70, 0.75):
        res[f"w{int(w*100)}"] = ev(b, "dl_cln03", w, PP_NEW)
    print_table(res, base)
    print(f"\n[1-D 완료 {(time.time()-t0)/60:.1f}분] — 상위 축으로 2-D 정련", flush=True)

    # 2-D: snap(thr × to) — 분포 차이가 가장 큰 축
    res2 = {}
    for thr in (0.15, 0.21, 0.27, 0.33):
        for to in (0.85, 0.88, 0.90, 0.93):
            res2[f"t{int(thr*100)}s{int(to*100)}"] = ev(
                b, "dl_cln03", 0.65, pp(snap_thr=thr, snap_to=to))
    print_table(res2, base)

    # 최고 셀 3개를 w와 교차
    top = sorted(res2.items(), key=lambda kv: -min(kv[1]["cv"][0] - base["cv"][0],
                                                   kv[1]["hold"][0] - base["hold"][0]))[:3]
    res3 = {}
    for name, _ in top:
        thr = int(name[1:name.index("s")]) / 100
        to = int(name[name.index("s") + 1:]) / 100
        for w in (0.60, 0.65, 0.70):
            for cl in (0.13, 0.16, 0.19):
                res3[f"{name}_w{int(w*100)}c{int(cl*100)}"] = ev(
                    b, "dl_cln03", w, pp(snap_thr=thr, snap_to=to, clip_lo=cl))
    print_table(res3, base)

    allr = {**res, **res2, **res3}
    bestn, bestr = max(allr.items(),
                       key=lambda kv: min(kv[1]["cv"][0] - base["cv"][0],
                                          kv[1]["hold"][0] - base["hold"][0]))
    print(f"\n트랙 최고: {bestn} — CV {bestr['cv'][0]:.4f} (v1 대비 "
          f"{bestr['cv'][0]-base['cv'][0]:+.4f}) | HOLD {bestr['hold'][0]:.4f} "
          f"({bestr['hold'][0]-base['hold'][0]:+.4f})")
    print(f"챔피언 대비: CV {bestr['cv'][0]-champ['cv'][0]:+.4f} | "
          f"HOLD {bestr['hold'][0]-champ['hold'][0]:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
