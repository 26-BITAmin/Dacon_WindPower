# ═══════════════════════════════════════════════════════
# exp-043b: sn_92 세밀 탐색 및 fold-vs-champion 분해
#
# 배경: exp-043에서 snap_to=0.92(sn_92)가 챔피언(0.6448) 대비
#   ΔCV +0.0019 / Δhold +0.0018 — 양 게이트 ±0.001 매그니튜드 통과.
#   그러나 fold_ds는 기준(gspd·q60·c16·a50) 대비만 출력했고,
#   챔피언(g132·hub) 대비 fold-vs-champion은 알 수 없었다.
#   이 스크립트에서:
#     (a) fold-vs-champion 분해를 명시적으로 출력
#     (b) snap_to ∈ {0.90, 0.91, 0.92, 0.93, 0.94} 세밀 탐색
#     (c) snap_thr ∈ {0.22, 0.25, 0.28, 0.30, 0.32} 세밀 탐색
#     (d) 조합: snap_to×snap_thr 최선 격자 탐색
#
# 실행: C:\miniconda3\python.exe -X utf8 exp043b_snap_detail.py
# ═══════════════════════════════════════════════════════
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import load_gbm_cache, score_from_tidy, FOLD_LIST
from exp039_dl_diverse import DL_PRED_CACHE, build_slices
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE, BIAS_CACHE as EXT_BIAS_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
CHAMP_PP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def load_all():
    champ, champ_bias = load_gbm_cache()
    cache = champ.rename(columns={"gbm_cf": "cf_g132"})
    ext = pd.read_parquet(EXT_GBM_CACHE)
    with open(EXT_BIAS_CACHE, "rb") as f:
        ext_bias = pickle.load(f)
    for c in [c for c in ext.columns if c.startswith("cf_")]:
        m = cache[KEY].merge(ext[KEY + [c]], on=KEY, how="left")
        if m[c].notna().all():
            cache[c] = m[c].to_numpy()
    dl = pd.read_parquet(DL_PRED_CACHE)
    for c in ["dl_hub", "dl_q60"]:
        if c in dl.columns:
            m = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
            if m[c].notna().all():
                cache[c] = m[c].to_numpy()
    return cache, champ_bias


def evaluate_pp(cache, slices, bias_maps, blend_cf, cv_folds, hold, pp):
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        idx = sub.index
        cf = blend_cf[idx]
        pred = postprocess_flex(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                                group, bias_maps[fname], **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    sc_cv   = score_from_tidy(tidy[tidy["fold"] != hold], "pred")
    sc_hold = score_from_tidy(tidy[tidy["fold"] == hold], "pred")
    folds   = {fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds}
    return sc_cv, sc_hold, folds


def main():
    t0 = time.time()
    print("### exp-043b: sn_92 세밀 탐색 + fold-vs-champion 분해 ###", flush=True)

    cache, champ_bias = load_all()
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    # 블렌드
    blend_g132_hub = 0.70 * cache["cf_g132"].to_numpy() + 0.30 * cache["dl_hub"].to_numpy()
    blend_gspd_q60 = 0.70 * cache["cf_gspd"].to_numpy() + 0.30 * cache["dl_q60"].to_numpy()

    # 챔피언 (g132·hub·w0.70·c16·a50)
    sc_champ_cv, sc_champ_hold, folds_champ = evaluate_pp(
        cache, slices, champ_bias, blend_g132_hub, cv_folds, hold, CHAMP_PP)
    champ_cv   = sc_champ_cv[0]
    champ_hold = sc_champ_hold[0]
    print(f"챔피언: CV {champ_cv:.4f} | HOLD {champ_hold:.4f}")
    print(f"  fold별: " + " / ".join(f"{fn[-2:]}={folds_champ[fn]:.4f}" for fn in cv_folds))

    # 스윕 정의
    # (A) snap_to 세밀 + snap_thr 고정 c16·a50
    snap_to_range   = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    snap_thr_range  = [0.22, 0.25, 0.27, 0.30, 0.32, 0.35]

    # 전체 그리드 (snap_to × snap_thr)
    grid = [(st, sh) for st in snap_to_range for sh in snap_thr_range]
    print(f"\n{len(grid)}칸 snap 그리드 채점 중...", flush=True)

    results = {}
    for (snap_to, snap_thr) in grid:
        pp = CHAMP_PP | {"snap_to": snap_to, "snap_thr": snap_thr}
        name = f"sn{int(snap_to*100):02d}_pt{int(snap_thr*100):02d}"
        sc_cv, sc_hold, folds = evaluate_pp(
            cache, slices, champ_bias, blend_gspd_q60, cv_folds, hold, pp)
        results[name] = dict(cv=sc_cv[0], hold=sc_hold[0], folds=folds,
                              nmae=sc_cv[1], ficr=sc_cv[2],
                              snap_to=snap_to, snap_thr=snap_thr)

    print(f"채점 완료 ({time.time()-t0:.1f}s)\n")

    # 리포트 — fold-vs-champion 분해 포함
    print("=" * 120)
    print(f"{'변형':<22}{'CV':>8}{'ΔCV':>8}{'HOLD':>8}{'Δhold':>8}"
          f"{'1-NMAE':>8}{'FiCR':>8}"
          + "".join(f"{'Q'+fn[-1]:>8}" for fn in cv_folds)
          + "  fold일관  판정")
    print("=" * 120)

    rows = []
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["cv"]):
        dcv  = r["cv"] - champ_cv
        dhold = r["hold"] - champ_hold
        fold_ds = [r["folds"][fn] - folds_champ[fn] for fn in cv_folds]
        cons = all(x > -0.0005 for x in fold_ds)   # 완화: -0.0005 이상 (exp-035 기준)
        cons_strict = all(x > -0.0002 for x in fold_ds)
        mag = dcv > 0.001 and dhold > 0.001
        verdict = ""
        if mag and cons_strict:
            verdict = "★완전통과"
        elif mag and cons:
            verdict = "★매그니튜드"
        elif mag:
            verdict = "매그니튜드"
        elif dcv > 0 and dhold > 0:
            verdict = "양게이트"
        print(f"{name:<22}{r['cv']:>8.4f}{dcv:>+8.4f}{r['hold']:>8.4f}{dhold:>+8.4f}"
              f"{r['nmae']:>8.4f}{r['ficr']:>8.4f}"
              + "".join(f"{x:>+8.4f}" for x in fold_ds)
              + f"  {'O(완화)' if cons else 'X':>9}  {verdict}")
        rows.append(dict(name=name, dcv=dcv, dhold=dhold, fold_ds=fold_ds,
                        cons=cons, cons_strict=cons_strict, mag=mag,
                        snap_to=r["snap_to"], snap_thr=r["snap_thr"],
                        cv=r["cv"], hold=r["hold"], nmae=r["nmae"], ficr=r["ficr"]))

    # 최선 후보 상세
    best = sorted([r for r in rows if r["mag"]], key=lambda r: -min(r["dcv"], r["dhold"]))
    if best:
        print(f"\n★ 매그니튜드 통과 후보 ({len(best)}개):")
        for r in best[:5]:
            print(f"  {r['name']}: ΔCV {r['dcv']:+.4f} / Δhold {r['dhold']:+.4f}"
                  f" | 1-NMAE {r['nmae']:.4f} | FiCR {r['ficr']:.4f}"
                  f" | fold: " + " ".join(f"{x:+.4f}" for x in r["fold_ds"])
                  + f" {'fold일관' if r['cons'] else 'fold불일치'}")

    # snap_to 축별 최선 (snap_thr 고정 0.30)
    print(f"\n--- snap_to 축 (snap_thr=0.30 고정) ---")
    print(f"{'snap_to':>9}{'CV':>8}{'ΔCV':>8}{'HOLD':>8}{'Δhold':>8}{'1-NMAE':>8}{'FiCR':>8}")
    for st in snap_to_range:
        name = f"sn{int(st*100):02d}_pt30"
        r = results[name]
        print(f"{st:>9.2f}{r['cv']:>8.4f}{r['cv']-champ_cv:>+8.4f}"
              f"{r['hold']:>8.4f}{r['hold']-champ_hold:>+8.4f}"
              f"{r['nmae']:>8.4f}{r['ficr']:>8.4f}")

    # snap_thr 축별 최선 (snap_to 고정 0.92)
    print(f"\n--- snap_thr 축 (snap_to=0.92 고정) ---")
    print(f"{'snap_thr':>9}{'CV':>8}{'ΔCV':>8}{'HOLD':>8}{'Δhold':>8}{'1-NMAE':>8}{'FiCR':>8}")
    for sh in snap_thr_range:
        name = f"sn92_pt{int(sh*100):02d}"
        r = results[name]
        print(f"{sh:>9.2f}{r['cv']:>8.4f}{r['cv']-champ_cv:>+8.4f}"
              f"{r['hold']:>8.4f}{r['hold']-champ_hold:>+8.4f}"
              f"{r['nmae']:>8.4f}{r['ficr']:>8.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
