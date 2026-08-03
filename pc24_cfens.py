# ═══════════════════════════════════════════════════════
# PC-24 (R24): CF-공간 성분별 합성 — 천장 캡을 ens 성분에만 국한
#
# 근거: PC-22/23에서 v-공간 조립의 noQ4 적자 −0.005가 95셀 불변 —
#   v-공간 조립은 **전체 블렌드**를 커브 천장(fwd ≤ 곡선 최대)에 가둔다.
#   반면 CF-공간 블렌드 cfden×dC는 폴드 일관 최상(+0.0005/+0.0024/−0.0001/+0.0019).
#   ens 멤버를 CF-공간에서 합성하면(cf_ens = PC(v_cal)) 천장 캡이 ens 성분에만
#   걸리고 dl_cln03의 고CF 출력은 보존된다 — 미시도 합성 계열.
#
# 격자: a·cf_pcden + d·dl_dlvc + c·dl_cln03 + e·PC(v_cal), 3중 잣대.
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc24_cfens.py
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
    print("### PC-24: CF-공간 성분별 합성 (ens만 커브 통과) ###", flush=True)
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

    # cf_enscal = PC(v_cal) — ens 성분의 CF 사상 (결측은 블렌드에서 cf_pcden으로 대체)
    cf_enscal = np.full(len(cache), np.nan)
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
            v_t_tr = inv((tr_g["target"] / cap).to_numpy())
            ws_tr, sp_tr = tr_g["ens_ws"].to_numpy(), tr_g["ens_spread"].to_numpy()
            ok = ~np.isnan(ws_tr) & ~np.isnan(sp_tr) & ~np.isnan(v_t_tr)
            A = np.column_stack([ws_tr[ok], sp_tr[ok], np.ones(ok.sum())])
            coef, *_ = np.linalg.lstsq(A, v_t_tr[ok], rcond=None)
            ws_ev = cache.loc[sel, "ens_ws"].to_numpy()
            sp_ev = cache.loc[sel, "ens_spread"].to_numpy()
            v_cal = coef[0] * ws_ev + coef[1] * sp_ev + coef[2]
            cf_enscal[sel] = np.where(np.isnan(v_cal), np.nan, fwd(np.nan_to_num(v_cal)))
    # 결측 → cf_pcden 대체 (지분 효과 중립화)
    nanm = np.isnan(cf_enscal)
    cf_enscal[nanm] = cache["cf_pcden"].to_numpy()[nanm]
    cache["cf_enscal"] = cf_enscal
    print(f"cf_enscal 준비 (결측 대체 {nanm.mean()*100:.1f}%) ({time.time()-t0:.0f}s)", flush=True)

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
    for a, d, e in product((0.55, 0.60, 0.65, 0.70),
                           (0.0, 0.05, 0.10),
                           (0.0, 0.05, 0.075, 0.10, 0.15)):
        c = round(1 - a - d - e, 4)
        if c < 0.15 or c > 0.45:
            continue
        key = (a, d, c, e)
        label = "cf_" + "_".join(str(int(x * 1000)) for x in key)
        blend = (a * cache["cf_pcden"].to_numpy() + d * cache["dl_dlvc"].to_numpy()
                 + c * cache["dl_cln03"].to_numpy() + e * cache["cf_enscal"].to_numpy())
        cache[label] = blend
        r = ev(label, None, 1.0, PP_NEW)
        nq = noq4_of(label)
        rows.append(dict(key=key, label=label, r=r,
                         dcv=r["cv"][0] - ch_cv, dhd=r["hold"][0] - ch_hd,
                         dnq=nq[0] - ch_nq, dn=r["cv"][1] - champ["cv"][1]))
    print(f"격자 {len(rows)}셀 (3중 채점)", flush=True)

    rows.sort(key=lambda x: -min(x["dcv"], x["dhd"], x["dnq"]))
    print(f"\n{'(pcden,dlvc,dC,ensCF)':<26}{'Δ챔CV':>9}{'Δ챔hd':>9}{'Δ챔noQ4':>10}{'Δn':>9}"
          f"  fold(Δv1)", flush=True)
    for x in rows[:14]:
        fd = " ".join(f"{x['r']['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in CV_FOLDS)
        print(f"{str(x['key']):<26}{x['dcv']:>+9.4f}{x['dhd']:>+9.4f}{x['dnq']:>+10.4f}"
              f"{x['dn']:>+9.4f}  {fd}", flush=True)

    ok = [x for x in rows if x["dcv"] >= 0 and x["dhd"] >= 0 and x["dnq"] >= -0.0005 and x["dn"] >= 0]
    print(f"\n[3중 충족]: {len(ok)}셀", flush=True)
    for x in ok[:6]:
        print(f"  {x['key']}: {x['dcv']:+.4f}/{x['dhd']:+.4f}/noQ4 {x['dnq']:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
