# ═══════════════════════════════════════════════════════
# exp-083b: noL(LDAPS 제거 멤버) 지분·블렌드 w 사다리 확장 — 캐시 전용, 재학습 0
#
# exp-083 결과: noL은 corr(cf_hpo) 0.9893(0.99 미만 = 다양성 실재)·평균 이동 +0.041%cap
#   (중립)·단독 손실 −0.0022뿐. 최고 셀 noLt67_w70 / noLt75_w70이 ΔCV **+0.0010** /
#   Δhold +0.0004 로 **CV 매그니튜드 충족·홀드 미달**. 두 지표(Δn·Δf) 동반 양수.
#   사다리 끝(t67)과 w 끝(0.70)이 **둘 다 경계에 걸려** 정점이 표 밖에 있을 개연성 →
#   지분을 더 키우고(t58/t50/t42) w를 더 올려(0.75/0.80) 능선 정점을 확인한다.
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 캐시(exp083_gbm_noL.parquet)만 사용.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp083b_ladder.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp035_raw_dl import CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (혼합 계수 m = cf_hpo 지분, 태그)
SHARES = [(0.75, "t75"), (2 / 3, "t67"), (0.58, "t58"), (0.50, "t50"), (0.42, "t42")]
WS = [0.75, 0.80, 0.85, 0.90, 1.00]


def main():
    t0 = time.time()
    print("### exp-083b noL 지분×w 사다리 확장 (캐시 전용) ###", flush=True)
    cache, champ_bias = load_cache_full()
    got = merge_cached_cols(cache, CACHE_DIR / "exp083_gbm_noL.parquet", ["cf_noL"])
    assert "cf_noL" in got, "exp-083 noL 캐시 없음"

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)

    res = {}
    for m, mt in SHARES:
        col = f"mix_{mt}"
        cache[col] = m * cache["cf_hpo"] + (1 - m) * cache["cf_noL"]
        for w in WS:
            res[f"{mt}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    # 대조: 챔피언 base를 같은 w로 올렸을 때 (w 상향 자체의 효과 분리)
    for w in WS:
        if w != 0.65:
            res[f"[대조]hpo_w{int(w*100)}"] = ev("cf_hpo", "dl_cln03", w, PP_NEW)
    print_table(res, ref)

    print("\n--- 지분×w 격자 (ΔCV / Δhold) ---", flush=True)
    print(f"{'':<8}" + "".join(f"{'w'+str(int(w*100)):>18}" for w in WS))
    for m, mt in SHARES:
        row = f"{mt:<8}"
        for w in WS:
            r = res[f"{mt}_w{int(w*100)}"]
            row += f"{r['cv'][0]-ref['cv'][0]:>+9.4f}{r['hold'][0]-ref['hold'][0]:>+9.4f}"
        print(row, flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
