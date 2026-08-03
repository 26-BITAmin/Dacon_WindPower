# ═══════════════════════════════════════════════════════
# PC-09 (R9): 챔피언 블렌드 ⊕ 트랙 블렌드 — 2단 앙상블 (트랙의 실전 가치 측정)
#
# 근거: PC-01 게이트는 GBM 멤버 혼합(cf_hpo ⊕ cf_pcinv)만 측정해 전부 음수였다.
#   그 후 트랙에 v-네이티브 부품 2개가 생겼다 — dlvc(PC-05)·v-공간 블렌드(PC-07/08).
#   "개선된 트랙 블렌드 전체"를 챔피언 블렌드와 섞는 축은 미측정.
#   최종 = λ·(0.65 cf_hpo + 0.35 dl_cln03) + (1−λ)·track,  track ∈
#     {v1 (CF공간), PC-08 정점 (0.70, 0.15, 0.15) v-공간}
#
# 판정: 챔피언 초과가 목적이므로 Δ = 챔피언 (hpo×dC_w65). Q4 제외 pooled 병기.
#   프로파일 체크(사용자 규칙): Δn 비음수 필수. λ는 폴드 일관성으로 방어.
#
# 재학습 0. 실행: C:\miniconda3\python.exe -X utf8 pc09_ensemble.py
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
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
CV_FOLDS = [fn for fn, _ in FOLD_LIST if fn != HOLD]
SEARCH_FOLD = "fold_2024Q4"
LAMS = (0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
SIMPLEX = (0.70, 0.15, 0.15)          # PC-08 정점 (v̂_pc, dlvc, dl_cln03)


def main():
    t0 = time.time()
    print("### PC-09: 챔피언 ⊕ 트랙 2단 앙상블 (재학습 0) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()

    # 트랙 블렌드 2종을 열로 생성
    champ_blend = 0.65 * cache["cf_hpo"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    track_cf = 0.65 * cache["cf_pcinv"].to_numpy() + 0.35 * cache["dl_cln03"].to_numpy()
    track_vs = np.full(len(cache), np.nan)
    a, b, c = SIMPLEX
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            vmix = (a * inv(cache.loc[sel, "cf_pcinv"].to_numpy())
                    + b * inv(cache.loc[sel, "dl_dlvc"].to_numpy())
                    + c * inv(cache.loc[sel, "dl_cln03"].to_numpy()))
            track_vs[sel] = fwd(vmix)
    assert not np.isnan(track_vs).any()
    cache["trk_cf"] = track_cf
    cache["trk_vs"] = track_vs
    mm = cache["target"].notna()
    print(f"corr(champ_blend, trk_cf) {np.corrcoef(champ_blend[mm], track_cf[mm])[0,1]:.4f} | "
          f"corr(champ_blend, trk_vs) {np.corrcoef(champ_blend[mm], track_vs[mm])[0,1]:.4f}", flush=True)

    for lam in LAMS:
        cache[f"ens_cf_{int(lam*100)}"] = lam * champ_blend + (1 - lam) * track_cf
        cache[f"ens_vs_{int(lam*100)}"] = lam * champ_blend + (1 - lam) * track_vs

    ev = make_ev(cache, champ_bias)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    ch_cv, ch_hd = champ["cv"][0], champ["hold"][0]
    print(f"챔피언 재현: CV {ch_cv:.4f} | HOLD {ch_hd:.4f}  (기대 0.6556/0.6535)", flush=True)

    # Q4 제외 pooled용 tidy
    slices = build_slices(cache)
    def noq4_score(col):
        parts = []
        for (fname, group), sub in slices:
            if fname in (HOLD, SEARCH_FOLD):
                continue
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(cache[col].to_numpy()[sub.index], cap,
                                    sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                                    group, champ_bias[fname], **PP_NEW)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        return score_from_tidy(pd.concat(parts, ignore_index=True), "pred")[0]

    base_noq4 = None
    res = {}
    for kind in ("cf", "vs"):
        for lam in LAMS:
            res[f"ens_{kind}_{int(lam*100)}"] = ev(f"ens_{kind}_{int(lam*100)}", None, 1.0, PP_NEW)
    res["trk_cf(=v1)"] = ev("trk_cf", None, 1.0, PP_NEW)
    res["trk_vs(PC08)"] = ev("trk_vs", None, 1.0, PP_NEW)

    # 챔피언 noq4 기준
    cache["champ_bl"] = champ_blend
    base_noq4 = noq4_score("champ_bl")

    print(f"\n{'셀':<16}{'CV':>8}{'Δ챔':>9}{'HOLD':>8}{'Δ챔':>9}{'ΔCV−Q4':>10}{'Δn':>9}{'Δf':>9}"
          f"  fold(Δ챔)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv,
                                                           kv[1]["hold"][0] - ch_hd)):
        col = name if name.startswith("ens_") else ("trk_cf" if "cf" in name else "trk_vs")
        nq = noq4_score(col) - base_noq4
        fd = " ".join(f"{r['folds'][fn]-champ['folds'][fn]:+.4f}" for fn in CV_FOLDS)
        print(f"{name:<16}{r['cv'][0]:>8.4f}{r['cv'][0]-ch_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-ch_hd:>+9.4f}{nq:>+10.4f}"
              f"{r['cv'][1]-champ['cv'][1]:>+9.4f}{r['cv'][2]-champ['cv'][2]:>+9.4f}  {fd}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
