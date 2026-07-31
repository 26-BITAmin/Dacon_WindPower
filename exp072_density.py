# ═══════════════════════════════════════════════════════
# exp-072: 공기 밀도 보정 물리 피처 — 발전량 P ∝ ρv³의 ρ 축 (제공 데이터만, 수집 0)
#
# 배경: 파워커브는 표준밀도(1.225) 기준인데 실제 출력은 ρ에 비례 — 계절·기압 변동으로
#   ±5~8% 차이. 제공 데이터에 2m기온·지표기압이 있는데 밀도 파생 피처는 미구성
#   (v13 SCADA 파워커브(exp-015)는 ws만 사용, exp-023에서 JMA에 흡수 판정).
#   FiCR 병목 = base 오차 폭(exp-057) → 물리 보정은 순수 정확도형 후보.
#
# 피처 4개 (GBM에만 — exp-054 선례, DL 무변경):
#   rho      = ldaps_sp / (287.05 × T2m[K])     (자동 K 변환)
#   veff_l10 = ldaps_ws_10m × (rho/1.225)^(1/3)
#   veff_g100= gfs_ws_100m × (rho/1.225)^(1/3)
#   lapse850 = gfs_850_t − ldaps_2m_t           (안정도 프록시)
#
# 게이트: 챔피언 GBM 파라미터·cln03 가중·3-seed·4-fold+홀드아웃 (356피처),
#   dl_cln03 블렌드 w{0.65, 0.70}. 판정 Δ = 보드 챔피언(hpo×dC_w65).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp072_density.py
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RHO_CACHE = CACHE_DIR / "exp072_gbm_rho.parquet"
RHO_COLS = ["rho", "veff_l10", "veff_g100", "lapse850"]


def add_density(df):
    t2 = df["ldaps_heightAboveGround_2_t"].to_numpy(dtype=float)
    if np.nanmedian(t2) < 200:          # 섭씨로 저장된 경우 자동 변환
        t2 = t2 + 273.15
    t850 = df["gfs_isobaricInhPa_850_t"].to_numpy(dtype=float)
    if np.nanmedian(t850) < 200:
        t850 = t850 + 273.15
    sp = df["ldaps_surface_0_sp"].to_numpy(dtype=float)
    if np.nanmedian(sp) < 20000:        # hPa로 저장된 경우 Pa 변환
        sp = sp * 100.0
    rho = sp / (287.05 * t2)
    f = np.cbrt(rho / 1.225)
    df["rho"] = rho
    df["veff_l10"] = df["ldaps_ws_10m"].to_numpy() * f
    df["veff_g100"] = df["gfs_ws_100m"].to_numpy() * f
    df["lapse850"] = t850 - t2
    print(f"rho: mean {np.nanmean(rho):.3f} (기대 1.15~1.30) | 밀도계수 f 범위 "
          f"{np.nanmin(f):.3f}~{np.nanmax(f):.3f}", flush=True)
    return df


def main():
    t0 = time.time()
    print("### exp-072 공기 밀도 보정 피처 (GBM) ###", flush=True)
    train_df, use_cols = build_train_frame()
    train_df = add_density(train_df)
    cols = use_cols + RHO_COLS
    print(f"입력 준비 완료 ({len(cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    have = merge_cached_cols(cache, RHO_CACHE, ["cf_rho"])
    if "cf_rho" not in have:
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            p = train_gate_fold(train_df, cols, fold, 3, 8,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            preds.append(p.rename(columns={"cf_hpo": "cf_rho"}))
            print(f"  [rho:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_rho"].notna().all(), "정렬 실패"
        cache["cf_rho"] = m["cf_rho"].to_numpy()
        save_cols(cache, RHO_CACHE, ["cf_rho"])
        print(f"  캐시 저장 → {RHO_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    print(f"  corr(cf_rho, cf_hpo) = "
          f"{np.corrcoef(cache.loc[mm, 'cf_rho'], cache.loc[mm, 'cf_hpo'])[0, 1]:.4f}", flush=True)
    shift = (cache["cf_rho"].to_numpy() - cache["cf_hpo"].to_numpy()).mean() * 100
    print(f"  평균 이동 (cf_rho − cf_hpo): {shift:+.3f}%cap", flush=True)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"rho_solo": ev("cf_rho", None, 1.0, PP_NEW),
           "hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for w in (0.65, 0.70):
        res[f"rho_w{int(w*100)}"] = ev("cf_rho", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    d_cv = res["rho_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
    d_hd = res["rho_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
    print(f"\nGBM 단독 델타 (rho−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
