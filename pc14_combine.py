# ═══════════════════════════════════════════════════════
# PC-14 (R14): pcden × v-심플렉스 조합 정리 — 밀도 커브와 멤버 확장의 합성
#
# 근거: PC-13에서 pcden(밀도 커브 GBM)의 CF 블렌드가 +0.0012/+0.0011에
#   Δn 양수·폴드 준일관(Q3 −0.0001뿐)로 트랙 최상 프로파일. PC-11 심플렉스는
#   +0.0014/+0.0014. 두 개선의 합성 방식이 3개 미측정:
#   A. 원곡선 v-심플렉스에서 pcinv→pcden 멤버 치환 (+ 가중 국소 재탐색)
#   B. pcden CF 블렌드의 w 사다리 (0.65만 측정됨)
#   C. 밀도곡선 v-심플렉스(vs_den)의 가중 재탐색 (PC-11 가중을 그대로 썼었음)
#
# 판정: Δ = v1, 채택 기준 = 양 게이트 +0.001 + 폴드 3/4 이상 + Δn 비음수.
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc14_combine.py
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
from pc13_denscurve import add_density, build_curve_norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEMBERS = ["cf_pcden", "cf_pchpo", "dl_dlvc", "dl_cln03"]


def main():
    t0 = time.time()
    print("### PC-14: pcden × v-심플렉스 조합 정리 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cols in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                    ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                    ("exp085_gbm_pchpo.parquet", ["cf_pchpo"]),
                    ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        assert merge_cached_cols(cache, CACHE_DIR / f, [c for c in cols if c not in cache.columns]) or True

    train_df, _ = build_train_frame()
    train_df = add_density(train_df)
    scada_h = build_scada_hourly()

    # 두 곡선 체계의 v-좌표·fwd 사전 계산
    reg = {}                                   # ("org"|"den", fold, group) → (fwd, sel, {col: v})
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            for kind, builder in (("org", lambda: build_curve(scada_h[group], tr_g, fold["train_end"], None)),
                                  ("den", lambda: build_curve_norm(scada_h[group], tr_g, fold["train_end"]))):
                curve = builder()
                if curve is None:
                    continue
                fwd, inv, _, _ = make_transform(curve)
                v = {c: inv(cache.loc[sel, c].to_numpy())
                     for c in MEMBERS + ["cf_pcinv"]}
                reg[(kind, fname, group)] = (fwd, sel, v)
    print(f"곡선 2체계 준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"트랙 v1: {v1_cv:.4f}/{v1_hd:.4f} (기대 0.6530/0.6512) | "
          f"챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    res = {}
    def simplex(kind, wts, label):
        vals = np.full(len(cache), np.nan)
        for (k, fname, group), (fwd, sel, v) in reg.items():
            if k != kind:
                continue
            vals[sel] = fwd(sum(w * v[c] for w, c in zip(wts, MEMBERS)))
        assert not np.isnan(vals).any()
        cache[label] = vals
        res[label] = ev(label, None, 1.0, PP_NEW)

    # A/C: 두 곡선 체계 × 가중 국소 격자 (PC-11 정점 근방)
    grid = [(0.60, 0.10, 0.20, 0.10), (0.65, 0.05, 0.20, 0.10), (0.55, 0.15, 0.20, 0.10),
            (0.60, 0.10, 0.15, 0.15), (0.60, 0.10, 0.25, 0.05), (0.65, 0.10, 0.15, 0.10),
            (0.55, 0.10, 0.25, 0.10), (0.70, 0.00, 0.20, 0.10)]
    for kind in ("org", "den"):
        for wts in grid:
            simplex(kind, wts, f"sx_{kind}_" + "_".join(str(int(w*100)) for w in wts))

    # B: pcden CF 블렌드 w 사다리
    for w in (0.55, 0.60, 0.65, 0.70, 0.75):
        res[f"cfden_w{int(w*100)}"] = ev("cf_pcden", "dl_cln03", w, PP_NEW)

    # ── 표 ──
    print(f"\n{'셀':<26}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd))[:16]:
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<26}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    # 채택 판정
    ok = []
    for name, r in res.items():
        dcv, dhd = r["cv"][0] - v1_cv, r["hold"][0] - v1_hd
        dn = r["cv"][1] - v1["cv"][1]
        npos = sum(r["folds"][fn] - v1["folds"][fn] > -0.0002 for fn in cv_folds)
        if dcv > 0.001 and dhd > 0.001 and dn >= -0.0001 and npos >= 3:
            ok.append((name, dcv, dhd, dn, r))
    print(f"\n[채택 후보] 양 게이트 +0.001 · Δn≥−0.0001 · 폴드 3/4: {len(ok)}셀", flush=True)
    for name, dcv, dhd, dn, r in sorted(ok, key=lambda x: -min(x[1], x[2]))[:8]:
        print(f"  {name:<26} ΔCV {dcv:+.4f} / Δhd {dhd:+.4f} | Δn {dn:+.4f} | "
              f"챔피언 대비 {r['cv'][0]-champ['cv'][0]:+.4f}/{r['hold'][0]-champ['hold'][0]:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
