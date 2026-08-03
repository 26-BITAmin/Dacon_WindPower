# ═══════════════════════════════════════════════════════
# exp-088: exp-069(TabPFN) 재평가 — CV 주기준 + 격자 확장 (캐시 전용, 재학습 0)
#
# 왜 다시 보나:
#   (1) **판정 기준 정정.** 사용자 지시(2026-08-02): 홀드아웃은 CV와 **완전히 동일한
#       26,352행**을 덮고 차이는 학습 윈도우뿐이라 독립 확인이 아니다. exp-069의 기각
#       근거였던 "Δhold −0.0006·pfn 단독 −0.0083 = 시간 감쇠"는 홀드아웃을 독립
#       시험으로 본 해석이었다. CV 주기준으로 보면 최고 셀은 ΔCV **+0.0007**로
#       매그니튜드 미달일 뿐 명확한 기각이 아니다.
#   (2) **격자가 좁았다.** exp-069는 지분 {t90,t80,t67} × w{0.65,0.70} 6칸만 훑음.
#       exp-083에서 정점이 초기 격자 **바깥**(t58×w80)에 있었던 전례가 오늘 나왔다.
#
# cf_pfn = exp-069 5폴드 캐시(구 패키지 v2·ctx8000·draws2·n_est4) 그대로 재사용.
# 판정: **ΔCV(챔피언 대비) 주기준**, Δhold·corr·fold는 보조 서술.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp088_pfn_recheck.py
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

SHARES = [(1.00, "t100"), (0.95, "t95"), (0.90, "t90"), (0.85, "t85"), (0.80, "t80"),
          (0.75, "t75"), (2 / 3, "t67"), (0.58, "t58"), (0.50, "t50")]
WS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD_FOLD["name"]]


def main():
    t0 = time.time()
    print("### exp-088 exp-069(TabPFN) 재평가 — CV 주기준·격자 확장 ###", flush=True)
    cache, champ_bias = load_cache_full()
    got = merge_cached_cols(cache, CACHE_DIR / "exp069_gbm_pfn.parquet", ["cf_pfn"])
    assert "cf_pfn" in got, "exp-069 캐시 없음"

    mm = cache["target"].notna().to_numpy()
    v, h, d = (cache["cf_pfn"].to_numpy(), cache["cf_hpo"].to_numpy(),
               cache["dl_cln03"].to_numpy())
    print(f"[참고] corr(cf_pfn, cf_hpo) {np.corrcoef(v[mm], h[mm])[0,1]:.4f} | "
          f"corr(cf_pfn, dl_cln03) {np.corrcoef(v[mm], d[mm])[0,1]:.4f} | "
          f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | [참고]HOLD {ref['hold'][0]:.4f}\n", flush=True)

    grid = {}
    for m_, mt in SHARES:
        cache[f"mx_{mt}"] = m_ * cache["cf_hpo"] + (1 - m_) * cache["cf_pfn"]
        for w in WS:
            grid[(mt, w)] = ev(f"mx_{mt}", "dl_cln03", w, PP_NEW)

    print("--- ΔCV 격자 (주기준) ---", flush=True)
    print(f"{'':<7}" + "".join(f"{'w'+str(int(w*100)):>10}" for w in WS))
    for m_, mt in SHARES:
        print(f"{mt:<7}" + "".join(
            f"{grid[(mt, w)]['cv'][0]-ref['cv'][0]:>+10.4f}" for w in WS), flush=True)

    print("\n--- [보조] Δhold 격자 ---", flush=True)
    print(f"{'':<7}" + "".join(f"{'w'+str(int(w*100)):>10}" for w in WS))
    for m_, mt in SHARES:
        print(f"{mt:<7}" + "".join(
            f"{grid[(mt, w)]['hold'][0]-ref['hold'][0]:>+10.4f}" for w in WS), flush=True)

    best = max(grid, key=lambda k: grid[k]["cv"][0])
    r = grid[best]
    dn, df = r["cv"][1] - ref["cv"][1], r["cv"][2] - ref["cv"][2]
    print(f"\nCV 최적: {best[0]}_w{int(best[1]*100)}  "
          f"ΔCV **{r['cv'][0]-ref['cv'][0]:+.4f}**  "
          f"(Δ1-NMAE {dn:+.4f} / ΔFiCR {df:+.4f})  "
          f"[보조] Δhold {r['hold'][0]-ref['hold'][0]:+.4f}", flush=True)
    print("  fold Δ: " + "  ".join(
        f"{fn.replace('fold_','')} {r['folds'][fn]-ref['folds'][fn]:+.4f}" for fn in CV_FOLDS))

    # LOFO — 선택 자체의 과적합 검정 (CV 기준)
    tot = 0.0
    print("\n--- LOFO (나머지 3폴드에서 (지분,w) 선택 → 해당 폴드 평가) ---", flush=True)
    for f in CV_FOLDS:
        others = [x for x in CV_FOLDS if x != f]
        k = max(grid, key=lambda kk: np.mean([grid[kk]["folds"][o] for o in others]))
        dd = grid[k]["folds"][f] - ref["folds"][f]
        tot += dd
        print(f"  {f}: 선택 {k[0]}_w{int(k[1]*100)} → Δ {dd:+.4f}", flush=True)
    print(f"  LOFO 평균 Δ = {tot/len(CV_FOLDS):+.4f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
