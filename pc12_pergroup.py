# ═══════════════════════════════════════════════════════
# PC-12 (R12): 그룹별 심플렉스 가중 — PC-11 정점의 그룹 분해 (LOFO 방어)
#
# 근거: PC-11 정점 (0.60, 0.10, 0.20, 0.10)은 전 그룹 공유 가중.
#   그룹별 최적이 다를 수 있으나, 메인 라인의 그룹별 적응은 3연속 기각
#   (exp-060/061/062 — 탐색 폴드 편중)이므로 **LOFO를 판정 주기준**으로 삼는다:
#   그룹 g의 가중은 남은 3폴드 pooled 점수로 선택 → 빠진 폴드에 적용.
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc12_pergroup.py
# ═══════════════════════════════════════════════════════
import sys
import time

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

MEMBERS = ["cf_pcinv", "cf_pchpo", "dl_dlvc", "dl_cln03"]
PEAK = (0.60, 0.10, 0.20, 0.10)
CANDS = [PEAK,
         (0.70, 0.00, 0.15, 0.15),
         (0.50, 0.20, 0.15, 0.15),
         (0.60, 0.10, 0.15, 0.15),
         (0.60, 0.10, 0.25, 0.05),
         (0.55, 0.15, 0.20, 0.10),
         (0.65, 0.05, 0.20, 0.10)]


def main():
    t0 = time.time()
    print("### PC-12: 그룹별 심플렉스 (LOFO 주기준) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "exp085_gbm_pchpo.parquet", ["cf_pchpo"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()

    # (fold, group)별 후보 블렌드 값 사전 계산
    blend_vals = {}                       # (cand, fold, group) → 값 배열(sel 행)
    sel_of = {}
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            sel_of[(fname, group)] = sel
            v = {c: inv(cache.loc[sel, c].to_numpy()) for c in MEMBERS}
            for cand in CANDS:
                blend_vals[(cand, fname, group)] = fwd(sum(w * v[c] for w, c in zip(cand, MEMBERS)))
    print(f"블렌드 사전 계산 완료 ({time.time()-t0:.0f}s)", flush=True)

    def assemble(assign):
        """assign: {group: cand} → 전 행 열 생성"""
        vals = np.full(len(cache), np.nan)
        for (fname, _f) , in []: pass
        for fname, _ in FOLD_LIST:
            for group in (1, 2, 3):
                key = (fname, group)
                if key not in sel_of:
                    continue
                vals[sel_of[key]] = blend_vals[(assign[group], fname, group)]
        assert not np.isnan(vals).any()
        return vals

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"트랙 v1: CV {v1_cv:.4f}/{v1_hd:.4f} (기대 0.6530/0.6512)", flush=True)

    # 단일 스왑 열: 그룹 g만 cand, 나머지는 PEAK
    swap = {}
    for g in (1, 2, 3):
        for cand in CANDS:
            assign = {1: PEAK, 2: PEAK, 3: PEAK}
            assign[g] = cand
            col = f"sw_g{g}_{CANDS.index(cand)}"
            cache[col] = assemble(assign)
            swap[(g, cand)] = ev(col, None, 1.0, PP_NEW)

    peak_r = swap[(1, PEAK)]              # 전 그룹 PEAK와 동일
    print(f"전 그룹 PEAK: CV {peak_r['cv'][0]:.4f} ({peak_r['cv'][0]-v1_cv:+.4f}) / "
          f"HOLD {peak_r['hold'][0]:.4f} ({peak_r['hold'][0]-v1_hd:+.4f})", flush=True)

    # ── LOFO: 폴드 f 제외 3폴드 합으로 그룹별 후보 선택 → f 적용 ──
    lofo_total, fixed_total = 0.0, 0.0
    print(f"\n{'폴드':<10}{'선택 (g1/g2/g3 idx)':<22}{'LOFO Δ':>9}{'PEAK Δ':>9}", flush=True)
    for fn in cv_folds:
        others = [f for f in cv_folds if f != fn]
        assign = {}
        for g in (1, 2, 3):
            assign[g] = max(CANDS, key=lambda c: sum(swap[(g, c)]["folds"][f] for f in others))
        col = f"lofo_{fn}"
        cache[col] = assemble(assign)
        r = ev(col, None, 1.0, PP_NEW)
        d_lofo = r["folds"][fn] - v1["folds"][fn]
        d_fix = peak_r["folds"][fn] - v1["folds"][fn]
        lofo_total += d_lofo
        fixed_total += d_fix
        print(f"{fn.replace('fold_',''):<10}"
              f"{'/'.join(str(CANDS.index(assign[g])) for g in (1,2,3)):<22}"
              f"{d_lofo:>+9.4f}{d_fix:>+9.4f}", flush=True)
    print(f"\n[판정] LOFO Δ합 {lofo_total:+.4f} vs 고정 PEAK Δ합 {fixed_total:+.4f} "
          f"→ 그룹별 적응 순가치 {lofo_total-fixed_total:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
