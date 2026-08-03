# ═══════════════════════════════════════════════════════
# exp-083c: (지분 m, 블렌드 w) 선택의 LOFO 안정성 검정 — 캐시 전용
#
# exp-083b가 t58_w80에서 ΔCV +0.0021 / Δhold +0.0013(양쪽 매그니튜드 통과·두 지표 동반)을
#   냈다. 그러나 이 (m, w)는 4폴드 CV 격자에서 고른 값이므로 **선택 자체의 과적합**을
#   배제해야 한다 — exp-046(예측CF 조건부 w)·exp-061(그룹별 w)이 오라클은 양수인데
#   LOFO에서 음수로 뒤집혀 기각된 전례.
#
# 검정 2종:
#   (A) LOFO: 각 폴드 f의 (m, w)를 **나머지 3폴드에서만** 선택 → f에서 평가.
#       폴드마다 선택이 튀면(=exp-046 패턴) LOFO 이득이 소멸한다.
#   (B) 홀드아웃 전이: 4폴드 CV에서 고른 (m, w)를 **독립 홀드아웃**에서 평가.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp083c_lofo.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp035_raw_dl import CACHE_DIR, FOLD_LIST, HOLD_FOLD
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SHARES = [(1.0, "t100"), (5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67"),
          (0.58, "t58"), (0.50, "t50"), (0.42, "t42")]
WS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD_FOLD["name"]]


def main():
    t0 = time.time()
    print("### exp-083c (m, w) 선택 LOFO 안정성 검정 ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert "cf_noL" in merge_cached_cols(cache, CACHE_DIR / "exp083_gbm_noL.parquet",
                                         ["cf_noL"]), "exp-083 noL 캐시 없음"
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}", flush=True)

    grid = {}
    for m, mt in SHARES:
        col = f"mix_{mt}"
        cache[col] = m * cache["cf_hpo"] + (1 - m) * cache["cf_noL"]
        for w in WS:
            grid[(mt, w)] = ev(col, "dl_cln03", w, PP_NEW)

    # ── (A) LOFO ──
    print("\n--- (A) LOFO: 나머지 3폴드에서 선택 → 해당 폴드에서 평가 ---", flush=True)
    tot_sel, tot_ref = 0.0, 0.0
    for f in CV_FOLDS:
        others = [x for x in CV_FOLDS if x != f]
        key = max(grid, key=lambda k: np.mean([grid[k]["folds"][o] for o in others]))
        s_sel, s_ref = grid[key]["folds"][f], ref["folds"][f]
        tot_sel += s_sel; tot_ref += s_ref
        print(f"  {f}: 선택 {key[0]}_w{int(key[1]*100):d}  →  {s_sel:.4f} "
              f"(챔피언 {s_ref:.4f}, Δ {s_sel-s_ref:+.4f})", flush=True)
    print(f"  LOFO 평균 Δ = {(tot_sel-tot_ref)/len(CV_FOLDS):+.4f}", flush=True)

    # ── (B) CV 선택 → 홀드아웃 전이 ──
    print("\n--- (B) 4폴드 CV에서 선택 → 독립 홀드아웃 평가 ---", flush=True)
    best = max(grid, key=lambda k: grid[k]["cv"][0])
    r = grid[best]
    print(f"  CV 최적: {best[0]}_w{int(best[1]*100)}  CV {r['cv'][0]:.4f} "
          f"(Δ {r['cv'][0]-ref['cv'][0]:+.4f}) | HOLD {r['hold'][0]:.4f} "
          f"(Δ {r['hold'][0]-ref['hold'][0]:+.4f})", flush=True)
    both = max(grid, key=lambda k: min(grid[k]["cv"][0] - ref["cv"][0],
                                       grid[k]["hold"][0] - ref["hold"][0]))
    r2 = grid[both]
    print(f"  양게이트 최소값 최대: {both[0]}_w{int(both[1]*100)}  "
          f"ΔCV {r2['cv'][0]-ref['cv'][0]:+.4f} / Δhold {r2['hold'][0]-ref['hold'][0]:+.4f}",
          flush=True)

    # ── 능선 평탄성: 선택 오차의 비용 ──
    print("\n--- 능선 평탄성 (Δhold, m×w) ---", flush=True)
    print(f"{'':<7}" + "".join(f"{'w'+str(int(w*100)):>9}" for w in WS))
    for m, mt in SHARES:
        print(f"{mt:<7}" + "".join(
            f"{grid[(mt, w)]['hold'][0]-ref['hold'][0]:>+9.4f}" for w in WS), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
