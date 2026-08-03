# ═══════════════════════════════════════════════════════
# PC-11 (R11): v-심플렉스 멤버 확장 — GBM측 2멤버(+pchpo) · DL측 3멤버(+dlv)
#
# 근거: PC-08 심플렉스는 {v̂_pcinv, PC⁻¹(dlvc), PC⁻¹(dl_cln03)} 3멤버뿐.
#   트랙 캐시에는 v-공간에 얹을 수 있는 클린 멤버가 2개 더 있다 —
#   cf_pchpo(PC-03 풍속 타깃 HPO base, PC-06에서 pcinv와 다른 거동 확인)와
#   dl_dlv(t-공간 q60 DL). 멤버 확장은 미시도.
#
# 탐색 구조 (격자 폭발 방지):
#   S1: GBM측 분할 — (a1·v_pcinv + a2·v_pchpo), a1+a2=0.70, DL측 (0.15,0.15) 고정
#   S2: DL측 분할 — GBM측은 S1 최선, (b_dlvc, b_dlv, c) 합=0.30 격자
#   S3: 정점 근방 미세 격자
#   판정 = v1 대비 + LOFO(남은 3폴드 선택 → 빠진 폴드 적용).
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc11_simplex2.py
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

MEMBERS = ["cf_pcinv", "cf_pchpo", "dl_dlvc", "dl_dlv", "dl_cln03"]


def main():
    t0 = time.time()
    print("### PC-11: v-심플렉스 멤버 확장 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "exp085_gbm_pchpo.parquet", ["cf_pchpo"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlv.parquet", ["dl_dlv"])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()

    # v-좌표 사전 계산
    vs = {c: np.full(len(cache), np.nan) for c in MEMBERS}
    fwds, sel_of = {}, {}
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
            for c in MEMBERS:
                vs[c][sel] = inv(cache.loc[sel, c].to_numpy())
    print(f"v-좌표 준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"트랙 v1: CV {v1_cv:.4f}/{v1_hd:.4f} (기대 0.6530/0.6512) | "
          f"챔피언: {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    scored = {}
    def score(wts):
        key = tuple(round(w, 3) for w in wts)
        if key in scored:
            return scored[key]
        col = "vsim_" + "_".join(str(int(w * 1000)) for w in key)
        vmix = sum(w * vs[c] for w, c in zip(key, MEMBERS))
        vals = np.full(len(cache), np.nan)
        for k, fwd in fwds.items():
            vals[sel_of[k]] = fwd(vmix[sel_of[k]])
        assert not np.isnan(vals).any()
        cache[col] = vals
        scored[key] = ev(col, None, 1.0, PP_NEW)
        return scored[key]

    def report(title, keys):
        print(f"\n--- {title} ---", flush=True)
        print(f"{'(pcinv,pchpo,dlvc,dlv,dC)':<30}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}"
              f"{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
        for k in sorted(keys, key=lambda k: -min(scored[k]["cv"][0] - v1_cv,
                                                 scored[k]["hold"][0] - v1_hd))[:10]:
            r = scored[k]
            fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
            print(f"{str(k):<30}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
                  f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
                  f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    # ── S1: GBM측 분할 (DL측 0.15/0/0.15 고정 = PC-08 정점 대응) ──
    s1 = []
    for a1 in (0.70, 0.60, 0.50, 0.40, 0.35, 0.30):
        a2 = 0.70 - a1
        k = tuple(round(x, 3) for x in (a1, a2, 0.15, 0.0, 0.15))
        score(k); s1.append(k)
    report("S1: GBM측 pcinv↔pchpo 분할", s1)
    best_s1 = max(s1, key=lambda k: min(scored[k]["cv"][0] - v1_cv,
                                        scored[k]["hold"][0] - v1_hd))
    a1, a2 = best_s1[0], best_s1[1]

    # ── S2: DL측 3멤버 분할 (GBM측 = S1 최선) ──
    s2 = []
    for b1, b2 in product(np.arange(0.0, 0.35, 0.05), repeat=2):
        c = 0.30 - b1 - b2
        if c < -1e-9 or c > 0.301:
            continue
        k = tuple(round(x, 3) for x in (a1, a2, b1, b2, c))
        score(k); s2.append(k)
    report(f"S2: DL측 분할 (GBM측 {a1:.2f}/{a2:.2f})", s2)
    best_s2 = max(s2, key=lambda k: min(scored[k]["cv"][0] - v1_cv,
                                        scored[k]["hold"][0] - v1_hd))

    # ── S3: 정점 근방 미세 (총 GBM 지분 ±0.05) ──
    s3 = []
    g_tot = best_s2[0] + best_s2[1]
    d_tot = 1 - g_tot
    for gt in (g_tot - 0.05, g_tot, g_tot + 0.05):
        sc_g = gt / g_tot if g_tot else 0
        sc_d = (1 - gt) / d_tot if d_tot else 0
        k = tuple(round(x, 3) for x in (best_s2[0]*sc_g, best_s2[1]*sc_g,
                                        best_s2[2]*sc_d, best_s2[3]*sc_d, best_s2[4]*sc_d))
        score(k); s3.append(k)
    report("S3: 총 지분 미세 조정", s3)

    # ── LOFO (전 채점 셀 대상) ──
    all_keys = list(scored.keys())
    lofo_sum, picks = 0.0, {}
    for fn in cv_folds:
        others = [f for f in cv_folds if f != fn]
        bw = max(all_keys, key=lambda k: sum(scored[k]["folds"][f] for f in others))
        picks[fn] = bw
        lofo_sum += scored[bw]["folds"][fn] - v1["folds"][fn]
    print(f"\n[LOFO] Δ합(4폴드) {lofo_sum:+.4f} | 폴드별 선택: "
          + " ".join(f"{fn.replace('fold_','')}→{picks[fn]}" for fn in cv_folds), flush=True)

    best = max(all_keys, key=lambda k: min(scored[k]["cv"][0] - v1_cv,
                                           scored[k]["hold"][0] - v1_hd))
    r = scored[best]
    print(f"\n[정점] {best}: CV {r['cv'][0]:.4f} ({r['cv'][0]-v1_cv:+.4f}) / "
          f"HOLD {r['hold'][0]:.4f} ({r['hold'][0]-v1_hd:+.4f}) | "
          f"챔피언 대비 {r['cv'][0]-champ['cv'][0]:+.4f}/{r['hold'][0]-champ['hold'][0]:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
