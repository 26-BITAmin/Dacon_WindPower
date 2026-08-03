# ═══════════════════════════════════════════════════════
# PC-22 (R22): 견고 구성 탐색 — 3중 잣대 (CV · HOLD · Q4 제외) 동시 충족 셀
#
# 근거: PC-21에서 정점이 Q4 편중(Q4 제외 시 챔피언 −0.0049)으로 제출 부적격.
#   Q3 열세의 유력 기여 = dlvc (PC-05부터 Q3 음수) → dlvc 0 셀 포함 재격자.
#   선택 기준을 "4폴드 CV 최대"에서 **min(Δ챔CV, Δ챔hold, Δ챔noQ4) 최대**로 교체.
#   cfden×dC CF 블렌드(폴드 일관 최상)의 v-공간 대응(dlvc 없는 3멤버)도 포함.
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc22_robust.py
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
    print("### PC-22: 견고 구성 탐색 (3중 잣대) ###", flush=True)
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
            reg[(fname, group)] = (fwd, sel, v, np.where(np.isnan(v_cal), v_base, v_cal))
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]

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
    ch_nq = noq4_of("champ_bl")[0]
    print(f"챔피언: CV {ch_cv:.4f} / HOLD {ch_hd:.4f} / noQ4 {ch_nq:.4f}", flush=True)

    rows = []
    for pd_, dv, en in product((0.50, 0.55, 0.60, 0.65, 0.70, 0.75),
                               (0.0, 0.05, 0.10, 0.15),
                               (0.0, 0.05, 0.075, 0.10)):
        dc = round(1 - pd_ - dv - en, 4)
        if dc < -1e-9 or dc > 0.45:
            continue
        key = (pd_, dv, dc, en)
        label = "r_" + "_".join(str(int(x * 1000)) for x in key)
        vals = np.full(len(cache), np.nan)
        for (fname, group), (fwd, sel, v, v_cal) in reg.items():
            vals[sel] = fwd(key[0] * v["cf_pcden"] + key[1] * v["dl_dlvc"]
                            + key[2] * v["dl_cln03"] + key[3] * v_cal)
        assert not np.isnan(vals).any()
        cache[label] = vals
        r = ev(label, None, 1.0, PP_NEW)
        nq = noq4_of(label)
        rows.append(dict(key=key, label=label, r=r,
                         dcv=r["cv"][0] - ch_cv, dhd=r["hold"][0] - ch_hd,
                         dnq=nq[0] - ch_nq, dn=r["cv"][1] - champ["cv"][1]))
    print(f"격자 {len(rows)}셀 (3중 채점)", flush=True)

    rows.sort(key=lambda x: -min(x["dcv"], x["dhd"], x["dnq"]))
    print(f"\n{'(pcden,dlvc,dC,ens2d)':<26}{'Δ챔CV':>9}{'Δ챔hd':>9}{'Δ챔noQ4':>10}{'Δn':>9}"
          f"  fold(Δv1)", flush=True)
    for x in rows[:14]:
        fd = " ".join(f"{x['r']['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in CV_FOLDS)
        print(f"{str(x['key']):<26}{x['dcv']:>+9.4f}{x['dhd']:>+9.4f}{x['dnq']:>+10.4f}"
              f"{x['dn']:>+9.4f}  {fd}", flush=True)

    ok = [x for x in rows if x["dcv"] >= 0 and x["dhd"] >= 0 and x["dnq"] >= -0.0005 and x["dn"] >= 0]
    print(f"\n[3중 충족 (Δ챔CV≥0 · Δ챔hd≥0 · Δ챔noQ4≥−0.0005 · Δn≥0)]: {len(ok)}셀", flush=True)
    for x in ok[:6]:
        print(f"  {x['key']}: {x['dcv']:+.4f}/{x['dhd']:+.4f}/noQ4 {x['dnq']:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
