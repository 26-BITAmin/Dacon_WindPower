# ═══════════════════════════════════════════════════════
# PC-25 제출 빌드: 트랙 최고점 구성 (R20 ens2d) — 테스트측 전체 파이프라인
#
#   구성 (게이트 검증 R20/R21): v-공간 심플렉스, 원곡선 조립
#     CF = PC_org(0.75·PC_org⁻¹(cf_pcden) + 0.15·PC_org⁻¹(dl_dlvc)
#              + 0.025·PC_org⁻¹(dl_cln03) + 0.075·v_cal)
#     cf_pcden = 밀도 정규화 커브 타깃 GBM (352피처, e56 하이퍼, cln03 가중, 3-seed)
#     dl_dlvc  = v-타깃 DL (CF-공간 q60, 384×3, 3-seed)
#     v_cal    = ECMWF 앙상블 (ens_ws, ens_spread) → v_t 선형회귀 (전체 train, 그룹별)
#   후처리: 챔피언 PP_NEW + 편향맵 + pA 스냅 + 그룹 스케일 (0.98/0.98/0.97) 전부 재사용.
#
#   게이트 성적: CV 0.6556 / HOLD 0.6537 = 챔피언 동률 (Δn +0.0013 / Δf +0.0039)
#   ⚠ 알려진 리스크 (R21/R22 실측): 이득이 Q4 편중 — Q4 제외 pooled 챔피언 −0.0049.
#   ⚠ 파일만 만든다. 제출 슬롯 소비는 사용자 판단.
# 실행: C:\miniconda3\python.exe -X utf8 pc25_submit.py   (~15분, GBM CPU + DL GPU)
# ═══════════════════════════════════════════════════════
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, DATA_DIR, make_sample_weight,
                         apply_rolling_smoothing, apply_bias_correction_v2, prepare_matrices)
from exp012_feature_v11 import build_v11
from exp015_scada_pc import build_scada_hourly
from exp020_jma_features import build_jma_frame, JG_COLS, JM_COLS
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp049_clean import merge_avail, avail_factor
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp056_hpo import RESULT_JSON
from exp073_gate import build_ens_frame
from pc01_pcinv import VARIANTS, build_curve, make_transform, fit_predict_cf
from pc13_denscurve import add_density, build_curve_norm
import pc05_dlv as p5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.40, 0.16, 0.27, 0.93
SCALES = {1: 0.98, 2: 0.98, 3: 0.97}
W = dict(pcden=0.75, dlvc=0.15, dC=0.025, ens=0.075)
E49_PRED = CACHE_DIR / "exp049_test_pred_cln03_g20_d5.parquet"   # pA·ws·dl_cf
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
PCDEN_TEST = CACHE_DIR / "pc25_test_gbm_pcden.parquet"
DLVC_TEST = CACHE_DIR / "pc25_test_dl_dlvc.parquet"
CHAMP_FILE = "v_a40s93p27_ens3_w65_g3s97_submit.csv"
VARIANTS.setdefault("pcden", dict(q=None))


def main():
    t0 = time.time()
    print("### PC-25 제출 빌드 — 트랙 ens2d 구성 ###", flush=True)
    e56 = json.loads(RESULT_JSON.read_text())

    # === 피처 프레임 (챔피언 352 = v10TPS + JMA + spd + EC) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma); test_df = add_jma(test_df, jma)
    cols = col_sets["v10TPS"] + JG_COLS + JM_COLS
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    for name in ("train_df", "test_df"):
        df = locals()[name]
        df = df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
        if name == "train_df":
            train_df = df
        else:
            test_df = df
    gbm_cols = list(cols) + spd_cols + EC_COLS          # 352
    dl_cols = list(cols) + spd_cols                     # 348
    train_df = merge_avail(train_df)
    train_df = add_density(train_df)
    scada_h = build_scada_hourly()
    ens = build_ens_frame()
    train_df = train_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
    test_df = test_df.merge(ens, on=["forecast_kst_dtm", "kpx_group"], how="left")
    print(f"프레임 준비 완료 ({time.time()-t0:.0f}s) | test ens 커버리지 "
          f"{test_df['ens_ws'].notna().mean():.3f}", flush=True)

    tr_end = train_df["forecast_kst_dtm"].max()

    # === 그룹별 커브 (전체 train) ===
    curves_org, curves_den = {}, {}
    for g in (1, 2, 3):
        tr_g = train_df.loc[(train_df["kpx_group"] == g) & train_df["target"].notna()]
        c_org = build_curve(scada_h[g], tr_g, tr_end, None)
        c_den = build_curve_norm(scada_h[g], tr_g, tr_end)
        assert c_org is not None and c_den is not None
        curves_org[g] = make_transform(c_org)
        curves_den[g] = make_transform(c_den)

    # === ① cf_pcden 테스트 예측 (3-seed) ===
    if PCDEN_TEST.exists():
        gp = pd.read_parquet(PCDEN_TEST)
        print(f"pcden test 캐시 로드: {PCDEN_TEST.name}", flush=True)
    else:
        rows = []
        for g in (1, 2, 3):
            tg = time.time()
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            tr_g = train_df.loc[(train_df["kpx_group"] == g) & train_df["target"].notna()].copy()
            te_g = test_df.loc[test_df["kpx_group"] == g].sort_values("forecast_kst_dtm").copy()
            fwd, inv, slope_at, rng = curves_den[g]
            X_tr, X_te = prepare_matrices(tr_g, te_g, gbm_cols)
            y_tr = tr_g["target"].to_numpy()
            w_base = make_sample_weight(y_tr, cap, low_weight=0.05) * \
                avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            d = dict(X_tr=X_tr, X_va=X_te, y_tr=y_tr, w_base=w_base, cap=cap,
                     fwd=fwd, inv=inv, slope_at=slope_at, rng=rng)
            cf, _ = fit_predict_cf(d, "pcden", e56, 3, 8)
            rows.append(pd.DataFrame({"forecast_id": te_g["forecast_id"].to_numpy(),
                                      "kpx_group": g, "cf_pcden": cf}))
            print(f"  g{g} pcden 완료 ({time.time()-tg:.0f}s)", flush=True)
        gp = pd.concat(rows, ignore_index=True)
        gp.to_parquet(PCDEN_TEST, index=False)
        print(f"pcden test 캐시 저장 → {PCDEN_TEST.name}", flush=True)

    # === ② dl_dlvc 테스트 예측 (3-seed, 원곡선 v-타깃) ===
    if DLVC_TEST.exists():
        dv = pd.read_parquet(DLVC_TEST)
        print(f"dlvc test 캐시 로드: {DLVC_TEST.name}", flush=True)
    else:
        tg = time.time()
        curves_p5 = {}
        for g in (1, 2, 3):
            fwd, inv, _, rng = curves_org[g]
            curves_p5[g] = dict(fwd=fwd, inv=inv, v_lo=float(inv(rng[0])),
                                v_hi=float(inv(rng[1])))
        te_sorted = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        pred = p5.dl_fold_predict_v(train_df, te_sorted, dl_cols, 3, "dlvc", curves_p5)
        dv = pd.DataFrame({"forecast_id": te_sorted["forecast_id"].to_numpy(),
                           "kpx_group": te_sorted["kpx_group"].to_numpy(),
                           "dl_dlvc": pred})
        dv.to_parquet(DLVC_TEST, index=False)
        print(f"dlvc test 완료·저장 ({time.time()-tg:.0f}s) → {DLVC_TEST.name}", flush=True)

    # === ③ ens2d 캘리브레이션 (전체 train, 그룹별) ===
    coefs = {}
    for g in (1, 2, 3):
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        _, inv, _, _ = curves_org[g]
        tr_g = train_df.loc[(train_df["kpx_group"] == g) & train_df["target"].notna()]
        v_t = inv((tr_g["target"] / cap).to_numpy())
        ws, sp = tr_g["ens_ws"].to_numpy(), tr_g["ens_spread"].to_numpy()
        ok = ~np.isnan(ws) & ~np.isnan(sp) & ~np.isnan(v_t)
        A = np.column_stack([ws[ok], sp[ok], np.ones(ok.sum())])
        coefs[g], *_ = np.linalg.lstsq(A, v_t[ok], rcond=None)
        print(f"  ens2d g{g}: a={coefs[g][0]:.3f} c_spr={coefs[g][1]:+.3f} "
              f"b={coefs[g][2]:+.2f} (n={ok.sum()})", flush=True)

    # === ④ 조립 → PP → 스케일 → 제출 ===
    pred = pd.read_parquet(E49_PRED)
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    pred = pred.merge(gp, on=["forecast_id", "kpx_group"], how="left") \
               .merge(dv, on=["forecast_id", "kpx_group"], how="left") \
               .merge(test_df[["forecast_id", "kpx_group", "ens_ws", "ens_spread"]],
                      on=["forecast_id", "kpx_group"], how="left")
    assert pred["cf_pcden"].notna().all() and pred["dl_dlvc"].notna().all()

    parts = []
    for g in (1, 2, 3):
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        fwd, inv, _, _ = curves_org[g]
        sub = pred.loc[pred["kpx_group"] == g]
        v_pc = inv(sub["cf_pcden"].to_numpy())
        v_dv = inv(sub["dl_dlvc"].to_numpy())
        v_dc = inv(sub["dl_cf"].to_numpy())
        a, c_spr, b = coefs[g]
        v_cal = a * sub["ens_ws"].to_numpy() + c_spr * sub["ens_spread"].to_numpy() + b
        v_base = (W["pcden"] * v_pc + W["dlvc"] * v_dv + W["dC"] * v_dc) / \
            (W["pcden"] + W["dlvc"] + W["dC"])
        v_cal = np.where(np.isnan(v_cal), v_base, v_cal)
        nan_share = np.isnan(a * sub["ens_ws"].to_numpy()).mean()
        vmix = (W["pcden"] * v_pc + W["dlvc"] * v_dv + W["dC"] * v_dc + W["ens"] * v_cal)
        cf_blend = fwd(vmix)
        pA = sub["pA"].to_numpy()
        p = apply_bias_correction_v2(cf_blend * cap, sub["ws"].to_numpy(), g, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p[pA >= SNAP_THR] = np.maximum(p[pA >= SNAP_THR], cap * SNAP_TO)
        p = np.clip(p, cap * CLIP_LO, cap)
        p = np.clip(p * SCALES[g], 0, cap)
        parts.append(pd.DataFrame({"forecast_id": sub["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{g}", "pred": p}))
        print(f"  g{g}: ens 결측 대체율 {nan_share*100:.1f}%", flush=True)

    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id", how="left")
    assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
    assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all()
    assert (sub["forecast_kst_dtm"] == sample_submission["forecast_kst_dtm"]).all()
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0

    out = "v_pctrk_ens2d_g3s97_submit.csv"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / out, index=False, encoding="utf-8-sig")
    champ = pd.read_csv(DATA_DIR / CHAMP_FILE, encoding="utf-8-sig")
    diff = np.abs(sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy())
    shift = (sub[TARGET_COLS].to_numpy() - champ[TARGET_COLS].to_numpy()).mean()
    print(f"\nSaved {out}", flush=True)
    print(f"vs 보드 챔피언(ens3_g3s97): 평균 |차이| {diff.mean():.1f}kWh "
          f"({diff.mean()/21600*100:.3f}%cap) | **평균 이동 {shift:+.1f}kWh "
          f"({shift/21600*100:+.3f}%cap)** ← 판별 규칙 점검 지점", flush=True)
    for i, c in enumerate(TARGET_COLS):
        gsh = (sub[c] - champ[c]).mean() / CAPACITY_KWH[c] * 100
        print(f"  {c}: 이동 {gsh:+.3f}%cap | 그룹평균 {sub[c].mean():.0f}", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
