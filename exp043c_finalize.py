# exp-043c: sn93·pt27 위에서 clip_lo × alpha 세밀 탐색
#
# 배경: exp-043b에서 sn93_pt27 (snap_to=0.93, snap_thr=0.27)이
#   챔피언 대비 ΔCV +0.0031 / Δhold +0.0030으로 최선.
#   clip_lo=0.16, alpha=0.50은 exp-043 기준이지만 sn93·pt27 기저에서
#   다른 clip/alpha가 더 나을 수 있다.
#   → clip_lo × alpha 세밀 격자 스윕 후 최종 구성 확정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp043c_finalize.py
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

# 최선 snap 파라미터 (exp-043b 결과)
BEST_SNAP_TO  = 0.93
BEST_SNAP_THR = 0.27
CLIP_RANGE  = [0.12, 0.14, 0.16, 0.18]
ALPHA_RANGE = [0.30, 0.40, 0.50, 0.70]


def pp_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map, clip_lo, alpha, snap_thr, snap_to):
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


def eval_pp(cache, slices, bias_maps, blend_cf, cv_folds, hold, pp):
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = pp_flex(blend_cf[sub.index], cap,
                       sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                       group, bias_maps[fname], **pp)
        parts.append(sub[["fold","group","forecast_kst_dtm","target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    sc_cv   = score_from_tidy(tidy[tidy["fold"] != hold], "pred")
    sc_hold = score_from_tidy(tidy[tidy["fold"] == hold], "pred")
    folds   = {fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds}
    return sc_cv, sc_hold, folds


def main():
    t0 = time.time()
    print("### exp-043c: sn93·pt27 위에서 clip × alpha 세밀 탐색 ###", flush=True)

    cache, champ_bias = load_all()
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    blend_g132_hub = 0.70 * cache["cf_g132"].to_numpy() + 0.30 * cache["dl_hub"].to_numpy()
    blend_gspd_q60 = 0.70 * cache["cf_gspd"].to_numpy() + 0.30 * cache["dl_q60"].to_numpy()

    # 챔피언 기준
    pp_champ = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)
    sc_c_cv, sc_c_hold, folds_champ = eval_pp(cache, slices, champ_bias, blend_g132_hub,
                                               cv_folds, hold, pp_champ)
    champ_cv, champ_hold = sc_c_cv[0], sc_c_hold[0]
    print(f"챔피언: CV {champ_cv:.4f} | HOLD {champ_hold:.4f}")

    # sn93·pt27 기준 (exp-043b 최선)
    pp_ref = dict(clip_lo=0.16, alpha=0.50, snap_thr=BEST_SNAP_THR, snap_to=BEST_SNAP_TO)
    sc_ref_cv, sc_ref_hold, folds_ref = eval_pp(cache, slices, champ_bias, blend_gspd_q60,
                                                 cv_folds, hold, pp_ref)
    ref_cv, ref_hold = sc_ref_cv[0], sc_ref_hold[0]
    print(f"기준(sn93·pt27): CV {ref_cv:.4f} (Δchamp {ref_cv-champ_cv:+.4f}) | "
          f"HOLD {ref_hold:.4f} (Δchamp {ref_hold-champ_hold:+.4f})")

    # clip × alpha 그리드
    grid = [(c, a) for c in CLIP_RANGE for a in ALPHA_RANGE]
    print(f"\n{len(grid)}칸 clip×alpha 채점...", flush=True)
    results = {}
    for (cl, al) in grid:
        pp = pp_ref | {"clip_lo": cl, "alpha": al}
        name = f"cl{int(cl*100):02d}_al{int(al*100):02d}"
        sc_cv, sc_hold, folds = eval_pp(cache, slices, champ_bias, blend_gspd_q60,
                                         cv_folds, hold, pp)
        results[name] = dict(cv=sc_cv[0], hold=sc_hold[0], folds=folds,
                              nmae=sc_cv[1], ficr=sc_cv[2], clip=cl, alpha=al)
    print(f"완료 ({time.time()-t0:.1f}s)\n")

    print("=" * 100)
    print(f"{'변형':<20}{'CV':>8}{'ΔCV_champ':>10}{'HOLD':>8}{'Δhold_champ':>12}"
          f"{'1-NMAE':>8}{'FiCR':>8}"
          + "".join(f"{'Q'+fn[-1]:>8}" for fn in cv_folds)
          + "  판정")
    print("=" * 100)
    rows = []
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1]["cv"] + kv[1]["hold"])):
        dcv  = r["cv"]  - champ_cv
        dhold = r["hold"] - champ_hold
        fold_ds = [r["folds"][fn] - folds_champ[fn] for fn in cv_folds]
        cons = all(x > -0.0005 for x in fold_ds)
        mag = dcv > 0.001 and dhold > 0.001
        vs_ref = r["cv"] - ref_cv
        verdict = ""
        if mag and vs_ref > 0.001:
            verdict = "★ref초과"
        elif mag and cons:
            verdict = "★매그·fold"
        elif mag:
            verdict = "★매그니튜드"
        elif dcv > 0 and dhold > 0:
            verdict = "양게이트"
        print(f"{name:<20}{r['cv']:>8.4f}{dcv:>+10.4f}{r['hold']:>8.4f}{dhold:>+12.4f}"
              f"{r['nmae']:>8.4f}{r['ficr']:>8.4f}"
              + "".join(f"{x:>+8.4f}" for x in fold_ds)
              + f"  {verdict}")
        rows.append(dict(name=name, dcv=dcv, dhold=dhold, fold_ds=fold_ds,
                        vs_ref=vs_ref, cons=cons, mag=mag, **r))

    # alpha 고정 cl16 비교
    print(f"\n--- clip=0.16 고정, alpha 축 (sn93·pt27) ---")
    for al in ALPHA_RANGE:
        name = f"cl16_al{int(al*100):02d}"
        if name in results:
            r = results[name]
            print(f"  alpha={al:.2f}: CV {r['cv']:.4f} (Δchamp {r['cv']-champ_cv:+.4f} / Δref {r['cv']-ref_cv:+.4f})")

    # clip 고정 al50 비교
    print(f"\n--- alpha=0.50 고정, clip 축 (sn93·pt27) ---")
    for cl in CLIP_RANGE:
        name = f"cl{int(cl*100):02d}_al50"
        if name in results:
            r = results[name]
            print(f"  clip={cl:.2f}: CV {r['cv']:.4f} (Δchamp {r['cv']-champ_cv:+.4f} / Δref {r['cv']-ref_cv:+.4f})")

    best = sorted(rows, key=lambda r: -min(r["dcv"], r["dhold"]))[:3]
    print(f"\n★ 최선 3개 (min-gate 기준):")
    for r in best:
        print(f"  {r['name']}: ΔCV_champ {r['dcv']:+.4f} / Δhold_champ {r['dhold']:+.4f}"
              f" | vs sn93·pt27 Δ{r['vs_ref']:+.4f}"
              f" | fold: " + " ".join(f"{x:+.4f}" for x in r["fold_ds"])
              + f" {'fold일관' if r['cons'] else 'fold불일치'}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
