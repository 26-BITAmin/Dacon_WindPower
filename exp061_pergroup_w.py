# ═══════════════════════════════════════════════════════
# exp-061: 그룹별 블렌드 가중 w_g (LOFO) + DL 멤버 혼합 (dl_cln03 ⊕ dl_hpob) — 캐시 전용
#
# 배경: 블렌드 가중 w=0.65는 전 그룹 공유. metric()은 그룹별 NMAE·FiCR의 평균이라
#   그룹 간 완전 가산적 → w_g 선택은 그룹별 독립 최적화가 정확하고,
#   선택은 exp-046 프로토콜(LOFO — 다른 폴드에서 선택, 해당 폴드에 적용)로 정직 평가.
#   그룹3(라벨 1년 적음)의 GBM:DL 균형이 다를 가설. 재학습 0.
# B블록: exp-058 균형 DL(dl_hpob)은 전량 교체로 LB 역전했으나 dl_cln03과의
#   예측 혼합(부분 지분)은 미측정 — 캐시로 사다리만 측정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp061_pergroup_w.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, metric_single_group
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import KEY, PP_NEW, postprocess_flex, load_eval_frame
from exp050_lambda_sweep import merge_cached_cols
from exp054_gate import GBM_EC_CACHE
from exp056_hpo import GBM_HPO_CACHE
from exp049_clean import GBM_CLEAN_CACHE, DL_CLEAN_CACHE
from exp060_pergroup_hpo import make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DL_HPOB_CACHE = CACHE_DIR / "exp058_dl_hpob.parquet"
W_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
HOLD = HOLD_FOLD["name"]


def main():
    t0 = time.time()
    print("### exp-061 그룹별 블렌드 가중 (LOFO) + DL 혼합 ###", flush=True)
    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
    assert merge_cached_cols(cache, DL_HPOB_CACHE, ["dl_hpob"])
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)

    # ── A블록: 그룹별 w (LOFO) ──
    slices = build_slices(cache)
    cv_folds = [fn for fn, _ in FOLD_LIST if fn != HOLD]
    gbm = cache["cf_hpo"].to_numpy()
    dl = cache["dl_cln03"].to_numpy()
    # (fold, group, w) → dict(pred, target, dtm)
    store = {}
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        for w in W_GRID:
            blend = w * gbm[sub.index] + (1 - w) * dl[sub.index]
            pred = postprocess_flex(blend, cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
            store[(fname, group, w)] = dict(
                pred=pred, target=sub["target"].to_numpy(),
                dtm=sub["forecast_kst_dtm"].to_numpy(), cap=cap)

    def contrib(folds, group, w):
        """폴드 집합 pooled 그룹 기여 점수 0.5*(1-nmae)+0.5*ficr"""
        a = np.concatenate([store[(f, group, w)]["target"] for f in folds])
        p = np.concatenate([store[(f, group, w)]["pred"] for f in folds])
        m = ~np.isnan(a)
        nmae, ficr, n = metric_single_group(a[m], p[m], store[(folds[0], group, w)]["cap"])
        return 0.5 * (1 - nmae) + 0.5 * ficr

    print("\n--- 그룹별 w 능선 (4-fold pooled 기여 점수, ×1000 상대 w65) ---")
    print(f"{'group':<7}" + "".join(f"{w:>8.2f}" for w in W_GRID) + "   best_w")
    for g in [1, 2, 3]:
        base = contrib(cv_folds, g, 0.65)
        vals = [contrib(cv_folds, g, w) for w in W_GRID]
        best_w = W_GRID[int(np.argmax(vals))]
        print(f"g{g:<6}" + "".join(f"{(v-base)*1000:>+8.2f}" for v in vals) + f"{best_w:>9.2f}")
    print("  (홀드아웃 능선 — 참고용, 선택에 미사용)")
    for g in [1, 2, 3]:
        base = contrib([HOLD], g, 0.65)
        vals = [contrib([HOLD], g, w) for w in W_GRID]
        best_w = W_GRID[int(np.argmax(vals))]
        print(f"g{g:<6}" + "".join(f"{(v-base)*1000:>+8.2f}" for v in vals) + f"{best_w:>9.2f}")

    def assemble(choice):
        """choice: (fold, group) → w. tidy 반환"""
        parts = []
        for (fname, group), _ in slices:
            w = choice[(fname, group)]
            s = store[(fname, group, w)]
            parts.append(pd.DataFrame({"fold": fname, "group": group,
                                       "forecast_kst_dtm": s["dtm"], "target": s["target"],
                                       "pred": s["pred"]}))
        return pd.concat(parts, ignore_index=True)

    # LOFO 선택: CV 폴드 f는 다른 3폴드에서, 홀드는 4폴드 전체에서 선택
    choice, picks = {}, {}
    for f in cv_folds:
        sel = [x for x in cv_folds if x != f]
        for g in [1, 2, 3]:
            vals = [contrib(sel, g, w) for w in W_GRID]
            choice[(f, g)] = W_GRID[int(np.argmax(vals))]
    for g in [1, 2, 3]:
        vals = [contrib(cv_folds, g, w) for w in W_GRID]
        choice[(HOLD, g)] = W_GRID[int(np.argmax(vals))]
        picks[g] = choice[(HOLD, g)]
    print("\nLOFO 선택 w (fold별×그룹):")
    for f in cv_folds + [HOLD]:
        print(f"  {f}: " + ", ".join(f"g{g}={choice[(f, g)]:.2f}" for g in [1, 2, 3]))

    tidy = assemble(choice)
    cv_s = score_from_tidy(tidy[tidy["fold"] != HOLD], "pred")
    hd_s = score_from_tidy(tidy[tidy["fold"] == HOLD], "pred")
    fold_d = [score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] - ref["folds"][fn] for fn in cv_folds]
    print(f"\n[A] LOFO 그룹별 w: CV {cv_s[0]:.4f} (Δ {cv_s[0]-ref['cv'][0]:+.4f}) | "
          f"HOLD {hd_s[0]:.4f} (Δ {hd_s[0]-ref['hold'][0]:+.4f}) | "
          f"1-NMAE Δ {cv_s[1]-ref['cv'][1]:+.4f} FiCR Δ {cv_s[2]-ref['cv'][2]:+.4f}")
    print("  fold별 Δ: " + ", ".join(f"{fn.replace('fold_','')} {d:+.4f}" for fn, d in zip(cv_folds, fold_d)))

    # 오라클 (같은 폴드에서 선택 — 상한 참고)
    ochoice = {}
    for f in cv_folds + [HOLD]:
        for g in [1, 2, 3]:
            vals = [contrib([f], g, w) for w in W_GRID]
            ochoice[(f, g)] = W_GRID[int(np.argmax(vals))]
    otidy = assemble(ochoice)
    ocv = score_from_tidy(otidy[otidy["fold"] != HOLD], "pred")
    print(f"  (오라클 상한: CV {ocv[0]:.4f} Δ {ocv[0]-ref['cv'][0]:+.4f})")

    # ── B블록: DL 멤버 혼합 사다리 ──
    res = {}
    for m in (0.75, 0.50):
        col = f"dlmix{int(m*100)}"
        cache[col] = m * cache["dl_cln03"].to_numpy() + (1 - m) * cache["dl_hpob"].to_numpy()
        for w in (0.70, 0.65):
            res[f"{col}_w{int(w*100)}"] = ev("cf_hpo", col, w, PP_NEW)
    res["hpob100_w70"] = ev("cf_hpo", "dl_hpob", 0.70, PP_NEW)   # exp-058 재현점
    print_table(res, ref)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
