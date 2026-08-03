# ═══════════════════════════════════════════════════════
# PC-15 (R15): dlvc의 밀도 곡선판 `dlvcd` — PC-13의 DL측 대칭 완성
#
# 근거: PC-13에서 밀도 정규화 커브가 GBM측(pcden) CF 블렌드를 +0.0012/+0.0011로
#   올렸고(Δn 양수·폴드 준일관), 정점 조합(PC-14 sx_org_70_0_20_10)의 DL 멤버
#   dlvc(지분 0.20)는 여전히 **원곡선 타깃**으로 학습된 상태다. 밀도 곡선을
#   타깃·손실 곡선에 일관 적용한 dlvcd가 대칭 완성.
#
# 구성: pc05_dlv의 dlvc 학습 하네스 재사용, 곡선만 build_curve_norm(밀도 정규화).
#   손실 = CF 공간 q60 (torch 미분가능 곡선 경유), 3-seed, 폴드별 partial 캐시.
# 판정: Δ = v1. 채점 = PC-14 정점 가중에서 dlvc→dlvcd 치환 (곡선 2체계).
#
# 실행: C:\miniconda3\python.exe -X utf8 pc15_dlvcden.py   (GPU ~20분)
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform
from pc05_dlv import (build_frames, va_frame, dl_fold_predict_v, fold_curves)
from pc13_denscurve import add_density, build_curve_norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COL = "dl_dlvcd"
CPATH = CACHE_DIR / "pc15_dlvcd.parquet"
PPATH = CACHE_DIR / "pc15_dlvcd_partial.parquet"
MEMBERS = ["cf_pcden", "cf_pchpo", "dl_dlvcd", "dl_cln03"]
GRID = [(0.70, 0.00, 0.20, 0.10), (0.60, 0.10, 0.20, 0.10), (0.65, 0.05, 0.20, 0.10),
        (0.70, 0.00, 0.25, 0.05), (0.65, 0.00, 0.20, 0.15)]


def fold_curves_den(train_df, scada_h, fold):
    """pc05 fold_curves와 동일 구조, 곡선만 밀도 정규화판."""
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    out = {}
    for g in (1, 2, 3):
        tr_g = tr_all.loc[(tr_all["kpx_group"] == g) & tr_all["target"].notna()]
        if len(tr_g) == 0:
            continue
        curve = build_curve_norm(scada_h[g], tr_g, fold["train_end"])
        if curve is None:
            continue
        fwd, inv, _, rng = make_transform(curve)
        out[g] = dict(fwd=fwd, inv=inv, v_lo=float(inv(rng[0])), v_hi=float(inv(rng[1])),
                      curve=np.maximum.accumulate(np.asarray(curve, float)))
    return out


def main():
    t0 = time.time()
    print("### PC-15: dlvc 밀도 곡선판 (dlvcd, 3-seed) ###", flush=True)
    from exp015_scada_pc import build_scada_hourly
    train_df, cols = build_frames()
    for need in ("ldaps_heightAboveGround_2_t", "ldaps_surface_0_sp"):
        assert need in train_df.columns, f"밀도 계산 컬럼 없음: {need}"
    train_df = add_density(train_df)
    scada_h = build_scada_hourly()
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("exp085_gbm_pchpo.parquet", ["cf_pchpo"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])

    # ── dlvcd 학습 (pc05 gate 흐름 미러, 곡선만 밀도판) ──
    if COL not in merge_cached_cols(cache, CPATH, [COL]):
        out = np.full(len(cache), np.nan)
        if PPATH.exists():
            prev = pd.read_parquet(PPATH)
            mm = cache[KEY].merge(prev[KEY + [COL]], on=KEY, how="left")
            out = mm[COL].to_numpy()
            done = [fn for fn, _ in FOLD_LIST
                    if not np.isnan(out[(cache["fold"] == fn).to_numpy()]).any()]
            print(f"  partial 재개: 완료 폴드 {done}", flush=True)
        for fname, fold in FOLD_LIST:
            sel, va = va_frame(cache, train_df, fname)
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            curves = fold_curves_den(train_df, scada_h, fold)
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            out[sel] = dl_fold_predict_v(tr_all, va, cols, 3, "dlvc", curves)
            cache[COL] = out
            cache[KEY + [COL]].to_parquet(PPATH, index=False)
            print(f"  [dlvcd:{fname}] {time.time()-tf:.0f}s (partial 저장)", flush=True)
        cache[COL] = out
        save_cols(cache, CPATH, [COL])
        print(f"  캐시 저장 → {CPATH.name}", flush=True)

    # ── 채점: PC-14 정점 가중 근방, 곡선 2체계 조립 ──────
    gbm_frame, _ = build_train_frame()
    gbm_frame = add_density(gbm_frame)
    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\n트랙 v1: {v1_cv:.4f}/{v1_hd:.4f} (기대 0.6530/0.6512) | "
          f"챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    res = {}
    for kind in ("org", "den"):
        # 조립 곡선 준비
        reg = {}
        for fname, fold in FOLD_LIST:
            tr_all, _ = slice_fold(gbm_frame, fold)
            for group in (1, 2, 3):
                tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
                curve = (build_curve(scada_h[group], tr_g, fold["train_end"], None)
                         if kind == "org" else
                         build_curve_norm(scada_h[group], tr_g, fold["train_end"]))
                if curve is None:
                    continue
                fwd, inv, _, _ = make_transform(curve)
                sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
                reg[(fname, group)] = (fwd, sel,
                                       {c: inv(cache.loc[sel, c].to_numpy()) for c in MEMBERS})
        for wts in GRID:
            label = f"sx{kind[0]}_" + "_".join(str(int(w*100)) for w in wts)
            vals = np.full(len(cache), np.nan)
            for (fname, group), (fwd, sel, v) in reg.items():
                vals[sel] = fwd(sum(w * v[c] for w, c in zip(wts, MEMBERS)))
            assert not np.isnan(vals).any()
            cache[label] = vals
            res[label] = ev(label, None, 1.0, PP_NEW)
    res["cfden×dlvcd_w65"] = ev("cf_pcden", COL, 0.65, PP_NEW)
    res["cfden×dlvcd_w75"] = ev("cf_pcden", COL, 0.75, PP_NEW)

    print(f"\n{'셀':<22}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd)):
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<22}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)
    best = max(res.items(), key=lambda kv: min(kv[1]["cv"][0] - v1_cv, kv[1]["hold"][0] - v1_hd))
    r = best[1]
    print(f"\n[정점] {best[0]}: 챔피언 대비 {r['cv'][0]-champ['cv'][0]:+.4f}/"
          f"{r['hold'][0]-champ['hold'][0]:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
