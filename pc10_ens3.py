# ═══════════════════════════════════════════════════════
# PC-10 (R10): 트랙 GBM 입력에 enfo ens3 추가 (352 → 355) — 입력 축의 마지막 잔여
#
# 근거: 메인 라인은 exp-073에서 ens3 3피처로 +0.0003(CV)·LB +0.00014를 얻었는데
#   트랙 GBM(pcinv 타깃)은 여전히 352피처다. 트랙 7부품 중 입력만 메인 라인보다
#   한 세대 뒤 — 마지막 미시도 라운드.
#
# 판정 (트랙 규칙): Δ = v1. PC-08 정점 구성(v-심플렉스 0.70/0.15/0.15)에
#   cf_pcinv → cf_pcens3 치환한 셀도 함께 채점.
#
# 실행: C:\miniconda3\python.exe -X utf8 pc10_ens3.py   (GBM 3-seed 재학습 ~15분)
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from exp073_gate import build_ens_frame
from pc01_pcinv import build_curve, make_transform, group_fold_data, fit_predict_cf, VARIANTS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ENS3 = ["ens_ws", "ens_spread", "ens_gap"]
SIMPLEX = (0.70, 0.15, 0.15)
CACHE_P = CACHE_DIR / "exp082_gbm_pcens3.parquet"
VARIANTS["pcens3"] = dict(q=None)          # pcinv와 동일 규약, 입력만 355


def main():
    t0 = time.time()
    print("### PC-10: 트랙 GBM + ens3 (355피처, 3-seed) ###", flush=True)
    train_df, use_cols = build_train_frame()
    ens = build_ens_frame()
    train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ens_gap"] = train_df["ens_ws"] - train_df["ec_ws"]
    cols = list(use_cols) + ENS3
    cov = train_df["ens_ws"].notna().mean()
    print(f"피처 {len(cols)} | train ens 커버리지 {cov:.3f}", flush=True)
    scada_h = build_scada_hourly()
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    # ── cf_pcens3 학습 (캐시 우선) ───────────────────────
    if "cf_pcens3" not in merge_cached_cols(cache, CACHE_P, ["cf_pcens3"]):
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            rows = []
            for group in (1, 2, 3):
                d = group_fold_data(train_df, scada_h, fold, group, cols, None)
                if d is None:
                    continue
                cf, _ = fit_predict_cf(d, "pcens3", e56, 3, 8)
                rows.append(pd.DataFrame({
                    "fold": fold["name"], "group": group,
                    "forecast_kst_dtm": d["va_g"]["forecast_kst_dtm"].to_numpy(),
                    "cf_pcens3": cf}))
            preds.append(pd.concat(rows, ignore_index=True))
            print(f"  [pcens3:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_pcens3"].notna().all()
        cache["cf_pcens3"] = m["cf_pcens3"].to_numpy()
        save_cols(cache, CACHE_P, ["cf_pcens3"])
        print(f"  캐시 저장 → {CACHE_P.name}", flush=True)

    # ── v-심플렉스 열 생성 (pcinv판 · pcens3판) ──────────
    a, b, c = SIMPLEX
    for gcol, out in (("cf_pcinv", "vs_old"), ("cf_pcens3", "vs_new")):
        vals = np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            tr_all, _ = slice_fold(train_df, fold)
            for group in (1, 2, 3):
                tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
                curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
                if curve is None:
                    continue
                fwd, inv, _, _ = make_transform(curve)
                sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
                vmix = (a * inv(cache.loc[sel, gcol].to_numpy())
                        + b * inv(cache.loc[sel, "dl_dlvc"].to_numpy())
                        + c * inv(cache.loc[sel, "dl_cln03"].to_numpy()))
                vals[sel] = fwd(vmix)
        assert not np.isnan(vals).any()
        cache[out] = vals

    # ── 채점 ─────────────────────────────────────────────
    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\n트랙 v1 재현: CV {v1_cv:.4f} | HOLD {v1_hd:.4f}  (기대 0.6530/0.6512)", flush=True)
    print(f"챔피언 재현:  CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}", flush=True)

    cells = {
        "pcens3xdC_w65": ev("cf_pcens3", "dl_cln03", 0.65, PP_NEW),
        "vsimplex_old": ev("vs_old", None, 1.0, PP_NEW),
        "vsimplex_new": ev("vs_new", None, 1.0, PP_NEW),
        "pcens3_solo": ev("cf_pcens3", None, 1.0, PP_NEW),
        "pcinv_solo": ev("cf_pcinv", None, 1.0, PP_NEW),
    }
    print(f"\n{'셀':<16}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in cells.items():
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<16}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    mm = cache["target"].notna()
    print(f"\ncorr(cf_pcens3, cf_pcinv) = "
          f"{np.corrcoef(cache.loc[mm,'cf_pcens3'], cache.loc[mm,'cf_pcinv'])[0,1]:.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
