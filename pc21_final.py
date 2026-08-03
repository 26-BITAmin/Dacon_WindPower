# ═══════════════════════════════════════════════════════
# PC-21 (R21): ens2d 정점 정밀 + 견고성 — δ·가중 미세 격자 + LOFO + Q4 제외
#
# 근거: PC-20에서 ens2d(스프레드 항 캘리브레이션) 0.075 지분이 챔피언 동률
#   (CV −0.0000 / HOLD +0.0001, Δn·Δf 동반 양수). 정점 확정과 견고성 검사:
#   ① δ {0.05~0.125} × 기저 가중 지터, ② LOFO(가중 선택 안정성),
#   ③ Q4 제외 pooled (가중이 전 폴드로 선택된 데 대한 통제).
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc21_final.py
# ═══════════════════════════════════════════════════════
import sys
import time
from itertools import product

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp015_scada_pc import build_scada_hourly
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from exp073_gate import build_ens_frame
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]


def main():
    t0 = time.time()
    print("### PC-21: ens2d 정점 정밀 + 견고성 ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    ens = build_ens_frame()
    src = ens.rename(columns={"kpx_group": "group"})[
        ["group", "forecast_kst_dtm", "ens_ws", "ens_spread"]]
    m = cache[["group", "forecast_kst_dtm"]].merge(src, on=["group", "forecast_kst_dtm"], how="left")
    cache["ens_ws"], cache["ens_spread"] = m["ens_ws"].to_numpy(), m["ens_spread"].to_numpy()
    train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")

    MEM = ["cf_pcden", "dl_dlvc", "dl_cln03"]
    reg = {}
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            v = {c: inv(cache.loc[sel, c].to_numpy()) for c in MEM}
            v_base = sum(v.values()) / len(v)
            v_t_tr = inv((tr_g["target"] / cap).to_numpy())
            ws_tr, sp_tr = tr_g["ens_ws"].to_numpy(), tr_g["ens_spread"].to_numpy()
            ok = ~np.isnan(ws_tr) & ~np.isnan(sp_tr) & ~np.isnan(v_t_tr)
            A = np.column_stack([ws_tr[ok], sp_tr[ok], np.ones(ok.sum())])
            coef, *_ = np.linalg.lstsq(A, v_t_tr[ok], rcond=None)
            ws_ev = cache.loc[sel, "ens_ws"].to_numpy()
            sp_ev = cache.loc[sel, "ens_spread"].to_numpy()
            v_cal = np.where(np.isnan(ws_ev) | np.isnan(sp_ev), np.nan,
                             coef[0] * ws_ev + coef[1] * sp_ev + coef[2])
            v_cal = np.where(np.isnan(v_cal), v_base, v_cal)
            reg[(fname, group)] = (fwd, sel, v, v_cal)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    print(f"v1 {v1_cv:.4f}/{v1_hd:.4f} | 챔피언 {ch_cv:.4f}/{ch_hd:.4f}", flush=True)

    res = {}
    def score(key):
        # key = (pcden, dlvc, dC, ens2d)
        if key in res:
            return res[key]
        label = "z_" + "_".join(str(int(x * 1000)) for x in key)
        vals = np.full(len(cache), np.nan)
        for (fname, group), (fwd, sel, v, v_cal) in reg.items():
            vals[sel] = fwd(key[0] * v["cf_pcden"] + key[1] * v["dl_dlvc"]
                            + key[2] * v["dl_cln03"] + key[3] * v_cal)
        assert not np.isnan(vals).any()
        cache[label] = vals
        res[key] = (label, ev(label, None, 1.0, PP_NEW))
        return res[key]

    for pd_, dv, en in product((0.70, 0.725, 0.75, 0.775),
                               (0.10, 0.125, 0.15, 0.175),
                               (0.05, 0.075, 0.10, 0.125)):
        dc = round(1 - pd_ - dv - en, 4)
        if dc < -1e-9 or dc > 0.125:
            continue
        score((pd_, dv, dc, en))
    print(f"격자 {len(res)}셀", flush=True)

    ranked = sorted(res.items(), key=lambda kv: -min(kv[1][1]["cv"][0] - v1_cv,
                                                     kv[1][1]["hold"][0] - v1_hd))
    print(f"\n{'(pcden,dlvc,dC,ens2d)':<28}{'CV':>8}{'HOLD':>8}{'Δn':>9}{'Δf':>9}"
          f"{'Δ챔CV':>9}{'Δ챔hd':>9}  fold(Δv1)", flush=True)
    for key, (label, r) in ranked[:10]:
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in CV_FOLDS)
        print(f"{str(key):<28}{r['cv'][0]:>8.4f}{r['hold'][0]:>8.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}"
              f"{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}  {fd}", flush=True)

    # LOFO
    keys = list(res.keys())
    lofo, picks = 0.0, {}
    for fn in CV_FOLDS:
        others = [f for f in CV_FOLDS if f != fn]
        bw = max(keys, key=lambda k: sum(res[k][1]["folds"][f] for f in others))
        picks[fn] = bw
        lofo += res[bw][1]["folds"][fn] - v1["folds"][fn]
    print(f"\n[LOFO] Δ합 {lofo:+.4f} | 선택: "
          + " ".join(f"{fn.replace('fold_','')}→{picks[fn]}" for fn in CV_FOLDS), flush=True)

    # Q4 제외 pooled — 정점 vs 챔피언
    best_key, (best_label, best_r) = ranked[0]
    slices = build_slices(cache)
    def noq4_of(col):
        parts = []
        for (fname, group), sub in slices:
            if fname in (HOLD, "fold_2024Q4"):
                continue
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(cache[col].to_numpy()[sub.index], cap,
                                    sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                                    group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return score_from_tidy(pd.concat(parts, ignore_index=True), "pred")
    cache["champ_bl"] = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    nb, nc = noq4_of("champ_bl"), noq4_of(best_label)
    print(f"[Q4 제외 pooled] 챔피언 {nb[0]:.4f} → 정점 {nc[0]:.4f} (Δ {nc[0]-nb[0]:+.4f} | "
          f"Δn {nc[1]-nb[1]:+.4f} | Δf {nc[2]-nb[2]:+.4f})", flush=True)
    print(f"[정점] {best_key}: CV {best_r['cv'][0]:.4f} ({best_r['cv'][0]-ch_cv:+.4f}) / "
          f"HOLD {best_r['hold'][0]:.4f} ({best_r['hold'][0]-ch_hd:+.4f})", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
