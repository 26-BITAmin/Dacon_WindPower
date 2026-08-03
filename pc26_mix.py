# ═══════════════════════════════════════════════════════
# PC-26 (R26): 챔피언 ⊕ 신판 트랙(ens2d) 혼합 — LB 상보 프로파일의 결합
#
# 근거 (LB 실측 2026-08-03): 챔피언 0.64987 (n 0.8767 / f 0.4230) vs
#   트랙 파일 0.64959 (n 0.8781 / f 0.4211) — 점수 동급(Δ −0.00028 < 1σ)인데
#   **지표 프로파일이 상보적** (트랙이 NMAE 엔진, 챔피언이 FiCR 엔진).
#   exp-042의 "다른 절반 담당" 구조. PC-09의 혼합 전셀 음수는 구판 트랙 기준 —
#   ens2d 신판으로 재측정. 혼합은 **후처리 이후**(파일 평균과 동일 연산)와
#   **후처리 이전**(블렌드 평균) 두 방식 모두.
#
# 판정: 3중 잣대 (Δ챔CV · Δ챔hold · Δ챔noQ4) — 통과 시 파일 혼합 즉시 생성 가능.
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc26_mix.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp015_scada_pc import build_scada_hourly
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import CACHE_DIR, FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame, slice_fold
from exp073_gate import build_ens_frame
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]
W = dict(pcden=0.75, dlvc=0.15, dC=0.025, ens=0.075)
LAMS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def main():
    t0 = time.time()
    print("### PC-26: 챔피언 ⊕ 신판 트랙 혼합 (재학습 0) ###", flush=True)
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

    # 트랙 블렌드(pre-PP CF) 재구성 — R20 정점
    trk = np.full(len(cache), np.nan)
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
            v_pc = inv(cache.loc[sel, "cf_pcden"].to_numpy())
            v_dv = inv(cache.loc[sel, "dl_dlvc"].to_numpy())
            v_dc = inv(cache.loc[sel, "dl_cln03"].to_numpy())
            v_base = (W["pcden"] * v_pc + W["dlvc"] * v_dv + W["dC"] * v_dc) / \
                (W["pcden"] + W["dlvc"] + W["dC"])
            ws_ev = cache.loc[sel, "ens_ws"].to_numpy()
            sp_ev = cache.loc[sel, "ens_spread"].to_numpy()
            v_cal = coef[0] * ws_ev + coef[1] * sp_ev + coef[2]
            v_cal = np.where(np.isnan(v_cal), v_base, v_cal)
            trk[sel] = fwd(W["pcden"] * v_pc + W["dlvc"] * v_dv
                           + W["dC"] * v_dc + W["ens"] * v_cal)
    assert not np.isnan(trk).any()
    cache["trk_new"] = trk
    cache["champ_bl"] = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    print(f"트랙 블렌드 재구성 완료 ({time.time()-t0:.0f}s)", flush=True)

    # 채점기: pre-PP 혼합은 블렌드를 섞고 PP 1회 / post-PP 혼합은 PP 후 예측 평균
    slices = build_slices(cache)
    def pp_of(colvals, fname, group, sub):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        return postprocess_flex(colvals[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)

    def tidy_of_mix(lam, mode):
        parts = []
        ch, tk = cache["champ_bl"].to_numpy(), cache["trk_new"].to_numpy()
        for (fname, group), sub in slices:
            if mode == "pre":
                pred = pp_of(lam * ch + (1 - lam) * tk, fname, group, sub)
            else:
                pred = lam * pp_of(ch, fname, group, sub) + (1 - lam) * pp_of(tk, fname, group, sub)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return pd.concat(parts, ignore_index=True)

    def views(t):
        cv_m = t["fold"] != HOLD
        nq_m = cv_m & (t["fold"] != "fold_2024Q4")
        hd_m = t["fold"] == HOLD
        return (score_from_tidy(t[cv_m], "pred"), score_from_tidy(t[hd_m], "pred"),
                score_from_tidy(t[nq_m], "pred"))

    tb = tidy_of_mix(1.0, "pre")            # λ=1 = 챔피언
    cb, hb, nb = views(tb)
    print(f"챔피언: CV {cb[0]:.4f} / HOLD {hb[0]:.4f} / noQ4 {nb[0]:.4f}", flush=True)

    print(f"\n{'모드':<6}{'λ':>5}{'Δ챔CV':>9}{'Δ챔hd':>9}{'Δ챔noQ4':>10}{'Δn':>9}{'Δf':>9}", flush=True)
    ok_cells = []
    for mode in ("post", "pre"):
        for lam in LAMS:
            t = tidy_of_mix(lam, mode)
            cv, hd, nq = views(t)
            dcv, dhd, dnq = cv[0] - cb[0], hd[0] - hb[0], nq[0] - nb[0]
            dn, df_ = cv[1] - cb[1], cv[2] - cb[2]
            mark = " ←3중" if (dcv > 0 and dhd > 0 and dnq > -0.0005 and dn >= 0) else ""
            if mark:
                ok_cells.append((mode, lam, dcv, dhd, dnq))
            print(f"{mode:<6}{lam:>5.2f}{dcv:>+9.4f}{dhd:>+9.4f}{dnq:>+10.4f}"
                  f"{dn:>+9.4f}{df_:>+9.4f}{mark}", flush=True)

    print(f"\n[3중 충족] {len(ok_cells)}셀: {ok_cells}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
