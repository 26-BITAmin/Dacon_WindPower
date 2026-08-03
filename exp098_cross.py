# ═══════════════════════════════════════════════════════
# exp-098: ASOS GBM 멤버 × ASOS DL 멤버 **교차 채점** (재학습 0, 캐시 전용)
#
# 근거 (사용자 지적): "DL만 바꾸면 어쩌라는 거냐 — DL 바꾸면 GBM도 영향받는다".
#   이 프로젝트 역대 최대 상승 exp-042(+0.0052)가 정확히 그 발견이었음 —
#   exp-040(GBM 격자편차: NMAE만 개선)과 exp-039(DL q60: FiCR만 개선)는
#   **각자 단독으론 총점 상쇄로 무개선**이었으나 **교차하니 양쪽 유지**.
#   즉 멤버는 따로 판정하면 안 되고 쌍으로 봐야 함.
#
#   exp-082(ASOS→GBM, 단독 판정 기각)와 exp-096/097(ASOS→DL, 단독 판정 기각)은
#   전부 **상대 멤버를 챔피언으로 고정한 채** 판정했음 → 교차 축이 미측정으로 남음.
#   두 실험의 캐시가 모두 존재하므로 재학습 0으로 전 조합 채점 가능.
#
# 격자: GBM {cf_hpo(챔피언), cf_obs3, cf_obs1, cf_obsraw}
#     × DL  {dl_cln03(챔피언), dl_obs, dl_big, dl_thin}   (있는 것만)
#     × w   {0.55, 0.60, 0.65, 0.70, 0.75}
#   판정 Δ = 보드 챔피언(cf_hpo × dl_cln03 × w0.65).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp098_cross.py
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

SOURCES = {
    CACHE_DIR / "exp082_gbm_obs.parquet": ["cf_obs3", "cf_obs1", "cf_obsraw"],
    CACHE_DIR / "exp096_dl_obs.parquet": ["dl_obs"],
    CACHE_DIR / "exp097_dl_shape.parquet": ["dl_big", "dl_thin"],
}
WS = (0.55, 0.60, 0.65, 0.70, 0.75)


def main():
    t0 = time.time()
    print("### exp-098: GBM x DL 교차 채점 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for path, cols in SOURCES.items():
        if not path.exists():
            print(f"  [건너뜀] {path.name} 없음", flush=True)
            continue
        got = merge_cached_cols(cache, path, cols)
        print(f"  [로드] {path.name}: {got}", flush=True)

    gbms = [c for c in ["cf_hpo", "cf_obs3", "cf_obs1", "cf_obsraw"] if c in cache.columns]
    dls = [c for c in ["dl_cln03", "dl_obs", "dl_big", "dl_thin"] if c in cache.columns]
    print(f"\n격자: GBM {len(gbms)} x DL {len(dls)} x w {len(WS)} = "
          f"{len(gbms)*len(dls)*len(WS)}셀", flush=True)

    mm = cache["target"].notna()
    print("\n[멤버 간 상관 — 교차 이득의 전제]", flush=True)
    for g in gbms:
        row = " ".join(f"{d}:{np.corrcoef(cache.loc[mm, g], cache.loc[mm, d])[0,1]:.4f}"
                       for d in dls)
        print(f"  {g:<12} {row}", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} FiCR {ref['cv'][2]:.4f})"
          f" | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    res = {}
    for g in gbms:
        for d in dls:
            for w in WS:
                res[f"{g.replace('cf_','')}x{d.replace('dl_','')}_w{int(w*100)}"] = \
                    ev(g, d, w, PP_NEW)
    # 상위 18셀만 표로 (전 셀은 아래 요약)
    top = dict(sorted(res.items(),
                      key=lambda kv: -min(kv[1]["cv"][0] - ref["cv"][0],
                                          kv[1]["hold"][0] - ref["hold"][0]))[:18])
    print_table(top, ref)

    # 이중 게이트 통과 셀 요약
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    passed = [(k, v["cv"][0] - ch_cv, v["hold"][0] - ch_hd) for k, v in res.items()
              if v["cv"][0] - ch_cv > 0.001 and v["hold"][0] - ch_hd > 0.001]
    both_pos = [(k, v["cv"][0] - ch_cv, v["hold"][0] - ch_hd) for k, v in res.items()
                if v["cv"][0] > ch_cv and v["hold"][0] > ch_hd]
    print(f"\n[요약] 전 {len(res)}셀 중 — 양 게이트 양수 {len(both_pos)}셀 / "
          f"매그니튜드(±0.001) 통과 {len(passed)}셀", flush=True)
    for k, a, b in sorted(both_pos, key=lambda x: -min(x[1], x[2]))[:8]:
        print(f"    {k:<28} dCV {a:+.4f} / dhold {b:+.4f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
