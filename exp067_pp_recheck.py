# ═══════════════════════════════════════════════════════
# exp-067: 후처리 파라미터 능선 재확인 (캐시 전용) — 신PP(exp-043)의 현 base 재검
#
# 배경: PP_NEW(clip_lo 0.16, alpha 0.40, snap_thr 0.27, snap_to 0.93)는 exp-043에서
#   gspd·q60 base 위에 최적화된 값. 이후 base가 exp-049(정제)·054(EC)·056(HPO)로
#   누적 +0.008 이동 — 능선 정점이 유지되는지 재확인 (exp-063의 하이퍼 능선 재확인과
#   동일 성격의 브래킷). 캐시 전용이라 재학습 0.
#   주의: 같은 CV로 선택·판정하는 스윕이므로 선택 낙관 존재 (exp-043과 동일 프로토콜,
#   홀드아웃 게이트·fold 일관이 방어선). LB 전이는 후처리 증발 선례(exp-043) 참조.
#
# 구성: 1-D 사다리 (clip_lo·alpha) + 2-D 그리드 (snap_thr × snap_to), w=0.65 고정.
#   상위 셀은 w0.70 재확인. 판정 Δ = 보드 챔피언(hpo×dC_w65 + PP_NEW).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp067_pp_recheck.py
# ═══════════════════════════════════════════════════════
import sys
import time

from exp049_clean import PP_NEW
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    print("### exp-067 후처리 능선 재확인 (캐시 전용) ###", flush=True)
    print(f"현행 PP_NEW = {PP_NEW}", flush=True)
    cache, champ_bias = load_cache_full()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    res = {}
    for cl in (0.10, 0.13, 0.19, 0.22):
        res[f"clip{int(cl*100)}"] = ev("cf_hpo", "dl_cln03", 0.65, {**PP_NEW, "clip_lo": cl})
    for al in (0.25, 0.30, 0.35, 0.45, 0.50, 0.60):
        res[f"a{int(al*100)}"] = ev("cf_hpo", "dl_cln03", 0.65, {**PP_NEW, "alpha": al})
    for thr in (0.21, 0.24, 0.27, 0.30, 0.33):
        for to in (0.91, 0.92, 0.93, 0.94):
            if (thr, to) == (PP_NEW["snap_thr"], PP_NEW["snap_to"]):
                continue
            res[f"p{int(thr*100)}s{int(to*100)}"] = ev(
                "cf_hpo", "dl_cln03", 0.65, {**PP_NEW, "snap_thr": thr, "snap_to": to})
    print_table(res, ref)

    top = sorted(res.items(),
                 key=lambda kv: -min(kv[1]["cv"][0] - ref["cv"][0],
                                     kv[1]["hold"][0] - ref["hold"][0]))[:3]
    print(f"\n상위 3셀 (min(ΔCV, Δhd) 기준): {[k for k, _ in top]}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
