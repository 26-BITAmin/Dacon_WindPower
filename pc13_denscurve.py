# ═══════════════════════════════════════════════════════
# PC-13 (R13): 밀도 보정 파워커브 — IEC 61400-12 관행의 커브 적용
#
# 근거: 트랙 커브는 SCADA 나셀 풍속 vs 출력의 bin 평균 — **공기 밀도가 다른 시간을
#   한 bin에 섞는다** (P ∝ ρv³, ρ 연간 변동 ±7%). IEC 표준은 풍속을
#   v_norm = v·(ρ/ρ0)^(1/3)로 정규화해 커브를 만든다. 정규화 커브는 산포가 줄어
#   역변환 타깃 v_t = PC_norm⁻¹(y/cap)의 노이즈가 감소할 수 있다.
#   커브 구축 브래킷(PC-01: bin 평균 vs 분위수)은 밀도 축을 덮지 않았다.
#   ρ는 exp-072 방식 재사용: ldaps_sp / (287.05·T2m[K]) — 예보값이라 fold-safe·테스트 재현 가능.
#
# 구성: 변형 `pcden` — 커브 bin 축을 v_norm으로, 타깃도 같은 정규화 공간.
#   예측 시 CF = PC_norm(v̂_norm). GBM 355 아님(352 유지, PC-10에서 ens3 중립 확인).
#   나머지(하이퍼 e56·cln03 가중·3-seed) v1과 동일.
# 판정: Δ = v1. 채점 셀 = CF 블렌드(×dC_w65) + PC-11 정점 심플렉스 치환.
#
# 실행: C:\miniconda3\python.exe -X utf8 pc13_denscurve.py   (~15분)
# ═══════════════════════════════════════════════════════
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp015_scada_pc import PC_CENTERS, build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import (PC_BINS_L, MIN_BIN_N_L, VARIANTS, make_transform,
                        build_curve, fit_predict_cf)

VARIANTS["pcden"] = dict(q=None)          # fit_predict_cf가 참조하는 변형 등록

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_P = CACHE_DIR / "exp082_gbm_pcden.parquet"
RHO0 = 1.225
SIMPLEX = dict(pcden=0.60, pchpo=0.10, dlvc=0.20, dC=0.10)   # PC-11 정점의 pcinv→pcden 치환


def add_density(df):
    t2 = df["ldaps_heightAboveGround_2_t"].to_numpy(dtype=float)
    if np.nanmean(t2) < 200:
        t2 = t2 + 273.15
    sp = df["ldaps_surface_0_sp"].to_numpy(dtype=float)
    rho = sp / (287.05 * t2)
    df["dens_f"] = np.cbrt(rho / RHO0)          # v_norm = v × dens_f
    return df


def build_curve_norm(scada_h_g, g_train, tr_end):
    """밀도 정규화 bin 평균 커브 — g_train의 dens_f를 SCADA 시간에 병합해 v_norm으로 bin."""
    sc = scada_h_g[scada_h_g["forecast_kst_dtm"] <= tr_end]
    j = g_train[["forecast_kst_dtm", "dens_f"]].merge(sc, on="forecast_kst_dtm",
                                                      how="inner").dropna(
        subset=["park_ws", "park_cf", "dens_f"])
    if len(j) < 500:
        return None
    v_norm = j["park_ws"] * j["dens_f"]
    ws_bin = pd.cut(v_norm, bins=PC_BINS_L, labels=False)
    grp = j.groupby(ws_bin)["park_cf"]
    mv, cnt = grp.mean(), grp.size()
    curve = np.full(len(PC_CENTERS), np.nan)
    for idx, val in mv.items():
        if cnt[idx] >= MIN_BIN_N_L:
            curve[int(idx)] = val
    valid = ~np.isnan(curve)
    if valid.sum() < 5:
        return None
    return np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])


def main():
    t0 = time.time()
    print("### PC-13: 밀도 보정 파워커브 (3-seed 재학습) ###", flush=True)
    train_df, use_cols = build_train_frame()
    train_df = add_density(train_df)
    print(f"dens_f 범위 {np.nanmin(train_df['dens_f']):.3f}~{np.nanmax(train_df['dens_f']):.3f} "
          f"(커버리지 {train_df['dens_f'].notna().mean():.3f})", flush=True)
    scada_h = build_scada_hourly()
    e56 = json.loads(E56_JSON.read_text())

    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    assert merge_cached_cols(cache, CACHE_DIR / "exp085_gbm_pchpo.parquet", ["cf_pchpo"])
    assert merge_cached_cols(cache, CACHE_DIR / "pc05_dlvc.parquet", ["dl_dlvc"])

    # ── cf_pcden 학습 ────────────────────────────────────
    if "cf_pcden" not in merge_cached_cols(cache, CACHE_P, ["cf_pcden"]):
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            rows = []
            for group in (1, 2, 3):
                cap = CAPACITY_KWH[f"kpx_group_{group}"]
                tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
                va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
                curve = build_curve_norm(scada_h[group], tr_g, fold["train_end"])
                if curve is None:
                    continue
                fwd, inv, slope_at, rng = make_transform(curve)
                X_tr, X_va = prepare_matrices(tr_g, va_g, use_cols)
                y_tr = tr_g["target"].to_numpy()
                w_base = make_sample_weight(y_tr, cap, low_weight=0.05) * \
                    avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
                d = dict(X_tr=X_tr, X_va=X_va, y_tr=y_tr, w_base=w_base, va_g=va_g,
                         cap=cap, fwd=fwd, inv=inv, slope_at=slope_at, rng=rng)
                cf, _ = fit_predict_cf(d, "pcden", e56, 3, 8)
                rows.append(pd.DataFrame({
                    "fold": fold["name"], "group": group,
                    "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                    "cf_pcden": cf}))
            preds.append(pd.concat(rows, ignore_index=True))
            print(f"  [pcden:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_pcden"].notna().all()
        cache["cf_pcden"] = m["cf_pcden"].to_numpy()
        save_cols(cache, CACHE_P, ["cf_pcden"])
        print(f"  캐시 저장 → {CACHE_P.name}", flush=True)

    # ── v-심플렉스 (PC-11 정점의 pcinv→pcden 치환, 커브는 pcden 정규화 커브) ──
    vals = np.full(len(cache), np.nan)
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(train_df, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve_norm(scada_h[group], tr_g, fold["train_end"])
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            vmix = (SIMPLEX["pcden"] * inv(cache.loc[sel, "cf_pcden"].to_numpy())
                    + SIMPLEX["pchpo"] * inv(cache.loc[sel, "cf_pchpo"].to_numpy())
                    + SIMPLEX["dlvc"] * inv(cache.loc[sel, "dl_dlvc"].to_numpy())
                    + SIMPLEX["dC"] * inv(cache.loc[sel, "dl_cln03"].to_numpy()))
            vals[sel] = fwd(vmix)
    assert not np.isnan(vals).any()
    cache["vs_den"] = vals

    # ── 채점 ─────────────────────────────────────────────
    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\n트랙 v1: CV {v1_cv:.4f}/{v1_hd:.4f} (기대 0.6530/0.6512) | "
          f"챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    cells = {
        "pcdenxdC_w65": ev("cf_pcden", "dl_cln03", 0.65, PP_NEW),
        "vs_den(PC11치환)": ev("vs_den", None, 1.0, PP_NEW),
        "pcden_solo": ev("cf_pcden", None, 1.0, PP_NEW),
        "pcinv_solo": ev("cf_pcinv", None, 1.0, PP_NEW),
    }
    print(f"\n{'셀':<18}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in cells.items():
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<18}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)

    mm = cache["target"].notna()
    print(f"\ncorr(cf_pcden, cf_pcinv) = "
          f"{np.corrcoef(cache.loc[mm,'cf_pcden'], cache.loc[mm,'cf_pcinv'])[0,1]:.4f} | "
          f"corr(cf_pcden, cf_hpo) = "
          f"{np.corrcoef(cache.loc[mm,'cf_pcden'], cache.loc[mm,'cf_hpo'])[0,1]:.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
