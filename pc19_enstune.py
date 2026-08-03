# ═══════════════════════════════════════════════════════
# PC-19 (R19): ens_ws 멤버 정밀 조정 — δ 미세 + 4멤버 가중 공동 재탐색
#
# 근거: PC-18b에서 캘리브레이션된 ens_ws의 5% 지분이 양 게이트를 동반 개선
#   (+0.0020/+0.0017, Δn +0.0009). δ 격자가 {0, 5, 10}뿐이었고 기저 심플렉스
#   가중은 ens 멤버 없던 시절(PC-16) 정점 고정이었다 → 공동 재탐색.
#
# 격자: (pcden, dlvc, dC, ens) 합=1, pcden {0.70~0.80}, dlvc {0.10~0.20},
#   ens {0.025~0.10}, dC = 나머지 {0~0.15}. + LOFO.
# 판정: Δ = v1 · 챔피언 대비 병기. 재학습 0.
# 실행: C:\miniconda3\python.exe -X utf8 pc19_enstune.py
# ═══════════════════════════════════════════════════════
import sys
import time
from itertools import product

import numpy as np

from exp005_qopt import CAPACITY_KWH
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from exp073_gate import build_ens_frame
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.time()
    print("### PC-19: ens_ws 멤버 정밀 조정 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    ens = build_ens_frame()
    m = cache[["group", "forecast_kst_dtm"]].merge(
        ens.rename(columns={"kpx_group": "group"})[["group", "forecast_kst_dtm", "ens_ws"]],
        on=["group", "forecast_kst_dtm"], how="left")
    cache["ens_ws"] = m["ens_ws"].to_numpy()
    train_df = train_df.merge(ens[["forecast_kst_dtm", "kpx_group", "ens_ws"]],
                              on=["forecast_kst_dtm", "kpx_group"], how="left")

    # fold×그룹 사전 계산: v-좌표·캘리브레이션 v_cal·fwd
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
            v_t_tr = inv((tr_g["target"] / cap).to_numpy())
            ws_tr = tr_g["ens_ws"].to_numpy()
            ok = ~np.isnan(ws_tr) & ~np.isnan(v_t_tr)
            a, b = np.polyfit(ws_tr[ok], v_t_tr[ok], 1)
            ws_ev = cache.loc[sel, "ens_ws"].to_numpy()
            v_base = sum(vv for vv in v.values()) / len(v)
            v_cal = np.where(np.isnan(ws_ev), np.nan, a * ws_ev + b)
            v_cal = np.where(np.isnan(v_cal), v_base, v_cal)
            reg[(fname, group)] = (fwd, sel, v, v_cal)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"v1 {v1_cv:.4f}/{v1_hd:.4f} | 챔피언 {ch_cv:.4f}/{ch_hd:.4f}", flush=True)

    res = {}
    def score(wts):
        # wts = (pcden, dlvc, dC, ens)
        key = tuple(round(x, 4) for x in wts)
        if key in res:
            return res[key]
        label = "e_" + "_".join(str(int(x * 1000)) for x in key)
        vals = np.full(len(cache), np.nan)
        for (fname, group), (fwd, sel, v, v_cal) in reg.items():
            vmix = (key[0] * v["cf_pcden"] + key[1] * v["dl_dlvc"]
                    + key[2] * v["dl_cln03"] + key[3] * v_cal)
            vals[sel] = fwd(vmix)
        assert not np.isnan(vals).any()
        cache[label] = vals
        res[key] = ev(label, None, 1.0, PP_NEW)
        return res[key]

    for pd_, dv, en in product((0.70, 0.725, 0.75, 0.775, 0.80),
                               (0.10, 0.125, 0.15, 0.175),
                               (0.025, 0.05, 0.075, 0.10)):
        dc = 1 - pd_ - dv - en
        if dc < -1e-9 or dc > 0.15:
            continue
        score((pd_, dv, dc, en))
    print(f"격자 {len(res)}셀 채점", flush=True)

    ranked = sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                     kv[1]["hold"][0] - v1_hd))
    print(f"\n{'(pcden,dlvc,dC,ens)':<26}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}"
          f"{'Δn':>9}{'Δf':>9}{'Δ챔CV':>9}{'Δ챔hd':>9}", flush=True)
    for wts, r in ranked[:12]:
        print(f"{str(wts):<26}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}"
              f"{r['cv'][0]-ch_cv:>+9.4f}{r['hold'][0]-ch_hd:>+9.4f}", flush=True)

    # LOFO
    keys = list(res.keys())
    lofo, picks = 0.0, {}
    for fn in cv_folds:
        others = [f for f in cv_folds if f != fn]
        bw = max(keys, key=lambda k: sum(res[k]["folds"][f] for f in others))
        picks[fn] = bw
        lofo += res[bw]["folds"][fn] - v1["folds"][fn]
    print(f"\n[LOFO] Δ합 {lofo:+.4f} | 선택: "
          + " ".join(f"{fn.replace('fold_','')}→{picks[fn]}" for fn in cv_folds), flush=True)
    best, r = ranked[0]
    fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
    print(f"[정점] {best}: fold {fd} | 챔피언 대비 {r['cv'][0]-ch_cv:+.4f}/{r['hold'][0]-ch_hd:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
