# ═══════════════════════════════════════════════════════
# exp-086: 소스 분할 3멤버 GBM — cf_hpo ⊕ cf_noL ⊕ cf_noG (캐시 전용, 재학습 0)
#
# exp-084가 "무작위 부분집합은 안 되고 **소스 정렬**이 원인"을 확정했으므로 다음 라운드는
#   소스 구조를 더 쓰는 쪽. exp-083에서 noG(GFS 계열 제거)는 **단독 −0.0170으로 열등**해
#   2멤버 혼합에서는 전 셀 음수였지만, corr(cf_noG, cf_hpo) = **0.9815로 세 멤버 중 최저**
#   (noL 0.9893)다. 지분을 크게 낮추면(≤15%) 정확도 손실보다 탈상관 기여가 클 수 있다 —
#   exp-084에서 관측된 "corr 낮은 멤버만 이득" 규칙의 직접 시험.
#
# 격자: (hpo, noL, noG) 볼록결합 × DL 블렌드 w. noG 지분 0 행이 exp-083b 재현 대조.
#   + noL 지분 미세화(0.34~0.50)로 능선 정점 확인.
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 후처리·편향맵·DL 전부 챔피언 고정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp086_three.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp035_raw_dl import CACHE_DIR, FOLD_LIST, HOLD_FOLD
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

B_NOL = [0.34, 0.42, 0.50]          # noL 지분
C_NOG = [0.00, 0.05, 0.10, 0.15]    # noG 지분
WS = [0.75, 0.80, 0.85]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD_FOLD["name"]]


def main():
    t0 = time.time()
    print("### exp-086 소스 분할 3멤버 GBM (캐시 전용) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for tag in ("noL", "noG"):
        assert f"cf_{tag}" in merge_cached_cols(
            cache, CACHE_DIR / f"exp083_gbm_{tag}.parquet", [f"cf_{tag}"]), f"{tag} 캐시 없음"

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)

    res, grid = {}, {}
    for b in B_NOL:
        for c in C_NOG:
            a = 1.0 - b - c
            col = f"m3_{int(b*100)}_{int(c*100)}"
            cache[col] = (a * cache["cf_hpo"] + b * cache["cf_noL"] + c * cache["cf_noG"])
            for w in WS:
                name = f"L{int(b*100)}G{int(c*100)}_w{int(w*100)}"
                r = ev(col, "dl_cln03", w, PP_NEW)
                res[name] = r
                grid[(b, c, w)] = r
    print_table(res, ref)

    print("\n--- noG 지분의 순효과 (같은 noL 지분·w에서 c=0 대비) ---", flush=True)
    print(f"{'':<12}" + "".join(f"{'c=' + str(int(c*100)) + '%':>20}" for c in C_NOG))
    for b in B_NOL:
        for w in WS:
            row = f"L{int(b*100)}_w{int(w*100):<7}"
            base = grid[(b, 0.0, w)]
            for c in C_NOG:
                r = grid[(b, c, w)]
                row += (f"{r['cv'][0]-base['cv'][0]:>+10.4f}"
                        f"{r['hold'][0]-base['hold'][0]:>+10.4f}")
            print(row, flush=True)

    best = max(grid, key=lambda k: min(grid[k]["cv"][0] - ref["cv"][0],
                                       grid[k]["hold"][0] - ref["hold"][0]))
    r = grid[best]
    print(f"\n양게이트 최소값 최대: noL {best[0]:.2f} / noG {best[1]:.2f} / w {best[2]:.2f}"
          f"  →  ΔCV {r['cv'][0]-ref['cv'][0]:+.4f} / Δhold {r['hold'][0]-ref['hold'][0]:+.4f}",
          flush=True)
    print(f"   fold Δ: " + " ".join(
        f"{fn.replace('fold_','')} {r['folds'][fn]-ref['folds'][fn]:+.4f}" for fn in CV_FOLDS))
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
