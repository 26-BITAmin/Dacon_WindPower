# ═══════════════════════════════════════════════════════
# exp-080: 비대칭 시차 커널 — 예보 타이밍 편향 보정 (캐시 전용, 재학습 0)
#
# 배경:
#   - exp-078 진단: 평가시간 오차 분산의 75.3%가 일중 모양(타이밍) 오차.
#   - 현행 PP의 rolling(3, α, center=True)은 좌우 대칭 — 시간축 방향성(예보가
#     체계적으로 이르거나 늦는 편향)을 표현 불가. exp-027(clip·smoothing·snap 모양)·
#     exp-044(윈도우)·exp-067(파라미터 능선) 전부 대칭 커널만 스윕 — 비대칭은 미탐색.
#   - 가설: NWP 전선 통과 타이밍에 체계 편향이 있으면 pred'_t = (1−γ)·pred_t +
#     γ·pred_{t∓δ}가 이득. 없으면 전 셀 음수(γ만큼 자기 평활 희석)로 축 폐쇄.
#
# 보너스 축 (같은 캐시): 블렌드 함수형 — 산술(현행) vs 기하 가중평균.
#   AM-GM 갭이 멤버 불일치 구간에서 하향으로 작동 — exp-044 캐시 교차(전부 산술)의
#   잔여 미탐색 형태.
#
# 판정: 표준 이중 게이트, Δ = 보드 챔피언(hpo×dC_w65, 재현 기대 CV 0.6556/HOLD 0.6535).
# 실행: C:\miniconda3\python.exe -X utf8 exp080_lagkernel.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np

from exp049_clean import PP_NEW
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def lag_kernel(cache, src, delta, gamma):
    """시간 정렬 시퀀스에서 (1−γ)·v_t + γ·v_{t−δ}. δ=+1 → 프로파일을 뒤로(늦게) 이동."""
    out = np.full(len(cache), np.nan)
    for (fn, g), sub in cache.groupby(["fold", "group"], sort=False):
        s = sub.sort_values("forecast_kst_dtm")
        v = s[src].to_numpy()
        sh = np.roll(v, delta)
        if delta > 0:
            sh[:delta] = v[:delta]
        else:
            sh[delta:] = v[delta:]
        out[s.index] = (1 - gamma) * v + gamma * sh
    assert not np.isnan(out).any()
    return out


def main():
    t0 = time.time()
    print("### exp-080 비대칭 시차 커널 + 블렌드 함수형 (캐시 전용) ###", flush=True)
    cache, champ_bias = load_cache_full()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    g = cache["cf_hpo"].to_numpy()
    d = cache["dl_cln03"].to_numpy()
    cache["bl65"] = 0.65 * g + 0.35 * d
    ident = ev("bl65", None, 1.0, PP_NEW)
    assert abs(ident["cv"][0] - ref["cv"][0]) < 1e-12, "블렌드 열 재현 불일치"

    res = {}
    # A. 시차 커널: δ=+1(늦게)·−1(이르게) × γ, δ=±2 저강도
    for delta, gamma in [(+1, 0.10), (+1, 0.20), (+1, 0.30),
                         (-1, 0.10), (-1, 0.20), (-1, 0.30),
                         (+2, 0.15), (-2, 0.15)]:
        tag = f"lag{'p' if delta > 0 else 'm'}{abs(delta)}_g{int(gamma*100)}"
        cache[tag] = lag_kernel(cache, "bl65", delta, gamma)
        res[tag] = ev(tag, None, 1.0, PP_NEW)

    # B. 기하 가중평균 블렌드 (eps 하한 — DL 음수 방지)
    eps = 1e-4
    lg_, ld_ = np.log(np.clip(g, eps, None)), np.log(np.clip(d, eps, None))
    for w in (0.65, 0.70):
        tag = f"geo_w{int(w*100)}"
        cache[tag] = np.exp(w * lg_ + (1 - w) * ld_)
        res[tag] = ev(tag, None, 1.0, PP_NEW)
        shift = (cache[tag].to_numpy() - (w * g + (1 - w) * d)).mean() * 100
        print(f"  {tag}: 평균 이동(vs 산술) {shift:+.3f}%cap", flush=True)

    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
