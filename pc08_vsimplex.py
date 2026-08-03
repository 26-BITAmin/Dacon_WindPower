# ═══════════════════════════════════════════════════════
# PC-08 (R8): 3멤버 v-공간 심플렉스 — v̂_pc ⊕ PC⁻¹(dlvc) ⊕ PC⁻¹(dl_cln03)
#
# 근거: PC-07에서 v-공간 블렌드가 (i) 트랙 쌍에는 소폭 양수(+0.0005~+0.0010),
#   (ii) 챔피언 쌍에는 −0.0024로 해로움 = 효과가 트랙 고유임이 통제로 확인됐다.
#   PC-05의 dlvc(v-타깃 DL)와 PC-07의 v-공간 블렌드는 각각 소폭 양수였는데
#   두 v-네이티브 부품을 **함께** 쓰는 3멤버 조합은 미측정.
#
# 판정 (트랙 규칙): Δ = v1. 다중비교 방어 = LOFO (남은 3폴드에서 가중 선택 →
#   빠진 폴드 적용) — exp-046/061/106/109의 "오라클 양수/LOFO 음수" 함정 검사.
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc08_vsimplex.py
# ═══════════════════════════════════════════════════════
import sys
import time
from itertools import product

import numpy as np

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 심플렉스 격자: a = v̂_pc 지분, b = dlvc 지분, c = dl_cln03 지분 (a+b+c=1)
GRID = [(round(a, 2), round(b, 2), round(1 - a - b, 2))
        for a, b in product(np.arange(0.45, 0.80, 0.05), np.arange(0.0, 0.55, 0.05))
        if 1 - a - b >= -1e-9 and 1 - a - b < 0.56]


def main():
    t0 = time.time()
    print("### PC-08: 3멤버 v-공간 심플렉스 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()

    # fold×그룹 v-공간 좌표 사전 계산
    vs = {}                       # (콜럼) → v 배열
    fwds = {}                     # (fold, group) → fwd
    sel_of = {}
    for c in ("cf_pcinv", "dl_dlvc", "dl_cln03"):
        vs[c] = np.full(len(cache), np.nan)
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            fwds[(fname, group)] = fwd
            sel_of[(fname, group)] = sel
            for c in vs:
                vs[c][sel] = inv(cache.loc[sel, c].to_numpy())
    print(f"v-좌표 준비 완료 ({time.time()-t0:.0f}s) | 격자 {len(GRID)}점", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"트랙 v1 재현: CV {v1_cv:.4f} | HOLD {v1_hd:.4f}  (기대 0.6530/0.6512)", flush=True)

    def score_weights(a, b, c):
        col = f"vs_{int(a*100)}_{int(b*100)}_{int(c*100)}"
        if col not in cache.columns:
            vals = np.full(len(cache), np.nan)
            vmix = a * vs["cf_pcinv"] + b * vs["dl_dlvc"] + c * vs["dl_cln03"]
            for key, fwd in fwds.items():
                sel = sel_of[key]
                vals[sel] = fwd(vmix[sel])
            assert not np.isnan(vals).any()
            cache[col] = vals
        return ev(col, None, 1.0, PP_NEW)

    res = {w: score_weights(*w) for w in GRID}

    ranked = sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                     kv[1]["hold"][0] - v1_hd))
    print(f"\n--- 상위 12 (판정축 = min(ΔCV, Δhold) vs v1) ---", flush=True)
    print(f"{'(a,b,c)':<18}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for w, r in ranked[:12]:
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{str(w):<18}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    # ── LOFO: 남은 3폴드 합산 점수로 가중 선택 → 빠진 폴드 적용 ──
    lofo_sum = 0.0
    picks = {}
    for fn in cv_folds:
        others = [f for f in cv_folds if f != fn]
        best_w = max(GRID, key=lambda w: sum(res[w]["folds"][f] for f in others))
        picks[fn] = best_w
        lofo_sum += res[best_w]["folds"][fn] - v1["folds"][fn]
    oracle = sum(max(res[w]["folds"][fn] for w in GRID) - v1["folds"][fn] for fn in cv_folds)
    print(f"\n[LOFO] 폴드별 선택: " + " ".join(f"{fn.replace('fold_','')}→{picks[fn]}" for fn in cv_folds), flush=True)
    print(f"[LOFO] Δ합(4폴드) {lofo_sum:+.4f} | 오라클 상한 {oracle:+.4f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
