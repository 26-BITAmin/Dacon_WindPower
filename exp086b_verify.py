# ═══════════════════════════════════════════════════════
# exp-086b: R4 결과 검증 — "트랙 고유 이득"인가, 그냥 PP 설정 효과인가 (캐시 전용)
#
# R4에서 t15s93_w60c16(snap_thr 0.27→0.15)이 챔피언 대비 CV +0.0018/HOLD +0.0013.
# 그러나 위험 신호 3건: (a) Δn −0.0017~28 · Δf +0.0101 = exp-031 LB 역전 프로파일,
# (b) 이득이 Q1 편중, (c) 선택을 CV·홀드 동시 최대화로 해 홀드가 독립 검증이 아님.
#
# 검증 3종:
#   A. **통제**: 같은 PP(thr 0.15 등)를 **챔피언 base(cf_hpo)에도** 적용.
#      챔피언도 같이 오르면 트랙 고유 이득이 아니라 PP 설정 효과 (기각 사유).
#      ※ exp-067은 챔피언에서 thr 0.21까지만 스윕했고 NMAE 음수로 기각한 이력.
#   B. **선택 분리**: CV(4폴드)만으로 선택 → 그 셀의 홀드아웃을 독립 확인.
#   C. **fold 분해**: 챔피언 대비 폴드별 Δ — Q1 편중 여부 확인.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp086b_verify.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD_FOLD["name"]]


def pp(**kw):
    p = dict(PP_NEW)
    p.update(kw)
    return p


def main():
    t0 = time.time()
    print("### exp-086b R4 검증: 트랙 고유 이득 vs PP 설정 효과 ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    ev = make_ev(cache, champ_bias)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 기준: CV {champ['cv'][0]:.4f} (N {champ['cv'][1]:.4f} F {champ['cv'][2]:.4f}) "
          f"| HOLD {champ['hold'][0]:.4f}\n", flush=True)

    # ── A. 통제: 동일 PP를 챔피언 base에도 적용 ──
    print("[A] 통제 — snap_thr 사다리를 두 base에 동일 적용 (w는 각자 최적 근방)")
    print(f"{'base':<10}{'thr':>6}{'w':>6}{'CV':>9}{'Δ챔피언':>10}{'HOLD':>9}{'Δ챔피언':>10}"
          f"{'1-NMAE':>9}{'Δn':>9}{'FiCR':>9}{'Δf':>9}")
    rows = {}
    for b, w in (("cf_hpo", 0.65), ("cf_pcinv", 0.60)):
        for thr in (0.27, 0.21, 0.15, 0.10):
            r = ev(b, "dl_cln03", w, pp(snap_thr=thr))
            rows[(b, thr)] = r
            print(f"{b:<10}{thr:>6.2f}{w:>6.2f}{r['cv'][0]:>9.4f}"
                  f"{r['cv'][0]-champ['cv'][0]:>+10.4f}{r['hold'][0]:>9.4f}"
                  f"{r['hold'][0]-champ['hold'][0]:>+10.4f}{r['cv'][1]:>9.4f}"
                  f"{r['cv'][1]-champ['cv'][1]:>+9.4f}{r['cv'][2]:>9.4f}"
                  f"{r['cv'][2]-champ['cv'][2]:>+9.4f}", flush=True)
    g_hpo = rows[("cf_hpo", 0.15)]["cv"][0] - rows[("cf_hpo", 0.27)]["cv"][0]
    g_pc = rows[("cf_pcinv", 0.15)]["cv"][0] - rows[("cf_pcinv", 0.27)]["cv"][0]
    print(f"\n  thr 0.27→0.15 이득: 챔피언 base {g_hpo:+.4f} / 트랙 base {g_pc:+.4f}")
    print(f"  → 트랙 고유분 = {g_pc - g_hpo:+.4f} "
          f"({'트랙 고유 이득 있음' if g_pc - g_hpo > 0.001 else '대부분 PP 설정 효과'})")

    # ── B. 선택 분리: CV만으로 선택 → 홀드 독립 확인 ──
    print("\n[B] 선택 분리 — CV(4폴드)만으로 최적 선택 후 홀드아웃 독립 확인")
    cands = {}
    for thr in (0.10, 0.15, 0.21, 0.27):
        for to in (0.88, 0.90, 0.93):
            for w in (0.55, 0.60, 0.65, 0.70):
                for cl in (0.13, 0.16, 0.19):
                    cands[(thr, to, w, cl)] = ev("cf_pcinv", "dl_cln03", w,
                                                 pp(snap_thr=thr, snap_to=to, clip_lo=cl))
    best_cv = max(cands.items(), key=lambda kv: kv[1]["cv"][0])
    k, r = best_cv
    print(f"  CV 단독 선택: thr {k[0]} · to {k[1]} · w {k[2]} · clip {k[3]}")
    print(f"    CV {r['cv'][0]:.4f} (Δ챔피언 {r['cv'][0]-champ['cv'][0]:+.4f}) | "
          f"**독립 홀드** {r['hold'][0]:.4f} (Δ챔피언 {r['hold'][0]-champ['hold'][0]:+.4f})")
    print(f"    1-NMAE {r['cv'][1]:.4f} (Δn {r['cv'][1]-champ['cv'][1]:+.4f}) | "
          f"FiCR {r['cv'][2]:.4f} (Δf {r['cv'][2]-champ['cv'][2]:+.4f})")

    # ── C. fold 분해 (챔피언 대비) ──
    print("\n[C] fold 분해 — 챔피언 대비 폴드별 Δ (이득 편중 확인)")
    print(f"{'변형':<26}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in CV_FOLDS)
          + f"{'최소폴드':>10}")
    show = {"t15s93_w60c16": ev("cf_pcinv", "dl_cln03", 0.60,
                               pp(snap_thr=0.15, snap_to=0.93, clip_lo=0.16)),
            f"CV선택 {k}": r,
            "champ_thr15(통제)": rows[("cf_hpo", 0.15)]}
    for name, rr in show.items():
        ds = [rr["folds"][fn] - champ["folds"][fn] for fn in CV_FOLDS]
        print(f"{name:<26}" + "".join(f"{d:>+10.4f}" for d in ds) + f"{min(ds):>+10.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
