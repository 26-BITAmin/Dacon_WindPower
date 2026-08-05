# ═══════════════════════════════════════════════════════
# AN-06: 최종 게이트 — AN-05 그룹별 최적 가중의 전해상도 4-fold 평가
#
# AN-05 결과 (Q1~3 탐색, Q4 검증):
#   g1: wd 1.0, shear .5, t .5, sp 1.0          (Q4 +0.0008 — 불안정)
#   g2: gfs100 ×2, wd 1.0, shear .5, t 0, sp 1.0 (Q4 −0.0006)
#   g3: wd 1.0, shear .5, t 0, sp .5            (Q4 −0.0022)
# 공통 신호: 풍향 0.5→1.0 상향, 기온 하향.
#
# 구성: anw_m100(평균 K100) · anw_q60(q60 K500), filter·hw3
# 사다리: 단독 / cf자리 t90·t83 / DL자리 l90·l80 — AN-03/04 최선 지점 재측정
#
# 실행: C:\miniconda3\python.exe -X utf8 an06_final.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an02_clean import build_analogs2, variants_from, predict
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# AN-05 그룹별 최적 가중 → pset (풍향은 sin/cos 각 w/2)
def gset(gfs100, wd, shear, t, sp):
    return [("gfs_ws_100m", gfs100), ("gfs_wd_sin_100m", wd / 2),
            ("gfs_wd_cos_100m", wd / 2), ("ldaps_ws_10m", 1.0), ("ec_ws", 1.0),
            ("jmsm_ws", 1.0), ("jgsm_ws", 1.0), ("gfs_wind_shear_100_10", shear),
            ("ldaps_heightAboveGround_2_t", t), ("ldaps_surface_0_sp", sp)]

PSET_G = {1: gset(1.0, 1.0, 0.5, 0.5, 1.0),
          2: gset(2.0, 1.0, 0.5, 0.0, 1.0),
          3: gset(1.0, 1.0, 0.5, 0.0, 0.5)}
COLS = ["cf_anw_m100", "cf_anw_q60"]


def main():
    t0 = time.time()
    print("### AN-06 최종 게이트 (그룹별 최적 가중) ###", flush=True)
    cache, champ_bias = load_cache_full()
    cpath = CACHE_DIR / "an06_final.parquet"
    if not all(c in merge_cached_cols(cache, cpath, COLS) for c in COLS):
        train_df, _ = build_train_frame()
        print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
        preds = {c: [] for c in COLS}
        for fname, fold in FOLD_LIST:
            tf = time.time()
            for group in [1, 2, 3]:
                ana = build_analogs2(train_df, fold, group, PSET_G[group], 3)
                tabs = variants_from(ana)
                tm = ana["va_g"]["forecast_kst_dtm"].to_numpy()
                base = dict(fold=fold["name"], group=group)
                preds["cf_anw_m100"].append(pd.DataFrame(
                    {**base, "forecast_kst_dtm": tm,
                     "cf_anw_m100": predict(tabs["filter"], 100, "mean")}))
                preds["cf_anw_q60"].append(pd.DataFrame(
                    {**base, "forecast_kst_dtm": tm,
                     "cf_anw_q60": predict(tabs["filter"], 500, "q60")}))
            print(f"  [{fname}] {time.time()-tf:.0f}s", flush=True)
        for c in COLS:
            p = pd.concat(preds[c], ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[c].notna().all(), f"정렬 실패: {c}"
            cache[c] = m[c].to_numpy()
        save_cols(cache, cpath, COLS)
        print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}", flush=True)
    hpov = cache["cf_hpo"].to_numpy()
    dlv = cache["dl_cln03"].to_numpy()
    res = {}
    for tag in ["anw_m100", "anw_q60"]:
        col = f"cf_{tag}"
        print(f"  corr({col}, cf_hpo) = {corr_on_eval(cache, col, 'cf_hpo'):.4f} | "
              f"평균 이동 {(cache[col].to_numpy()-hpov).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for m_, mt in ((0.9, "t90"), (5 / 6, "t83")):
            mix = f"mx_{tag}_{mt}"
            cache[mix] = m_ * hpov + (1 - m_) * cache[col].to_numpy()
            res[f"{tag}{mt}_w65"] = ev(mix, "dl_cln03", 0.65, PP_NEW)
        for lam in (0.9, 0.8):
            dcol = f"dlmx_{tag}_{int(lam*100)}"
            cache[dcol] = lam * dlv + (1 - lam) * cache[col].to_numpy()
            res[f"{tag}_DL자리l{int(lam*100)}"] = ev("cf_hpo", dcol, 0.65, PP_NEW)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
