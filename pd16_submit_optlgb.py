# ═══════════════════════════════════════════════════════
# pd16 제출 빌더: 트랙 D anws_hub_optlgb — PD-14 신 기준 제출판 (PD-16)
#
# 사용자 지시 (2026-08-12): "optlgb판 제출 파일 만들어봐"
#
# PD-10 빌더와의 변경점 = 잔차 GBM 하나:
#   LGB e56(448) + XGB e56(585) 50:50  →  optlgb 단독 (PD-14 Optuna best,
#   633트리·잎36·min_child141·mae) · 3시드 (게이트 동일)
# DL base·AnEn 멤버는 PD-10 캐시 그대로 재사용 (변경 없음 — 재학습 0회):
#   pd10_anen_members_traintest.parquet · pd10_dltrain_anwshub.parquet ·
#   pd10_test_dl_anwshub_d3.parquet
# PP = 챔피언 최종 편향맵 → rolling α0.40 → clip 0.16 → 스케일 .98/.98/.97 (스냅 OFF)
#
# 게이트(트랙, PD-14): 제출 작동점 CV 0.6499 (기준 anws_hub +0.0012 ·
#   Δn +0.0009 · Δf +0.0014) / HOLD 0.6401
#
# 산출: open (1)/v_pdtrk_anwshub_optlgb_submit.csv
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd16_submit_optlgb.py"
# ═══════════════════════════════════════════════════════
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "wind_input"))
sys.path.insert(0, str(HERE.parents[1] / "anen_input"))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

from an14_featsel import entry_cols
from an14b_final import W14
from exp005_qopt import (CAPACITY_KWH, DATA_DIR, TARGET_COLS, make_sample_weight,
                         prepare_matrices, apply_bias_correction_v2,
                         apply_rolling_smoothing)
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame
from exp020_submit import add_jma
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp049_clean import avail_factor, merge_avail
from exp054_gate import EC_WIDE, build_ec_frame

from wi01_bivgate import SCALES

from pd00_inputs import add_calendar, load_pools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["kpx_group", "forecast_kst_dtm"]
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
CHAMP_MAIN = "v_a40nosnap_ens3es_dles_w65_g3s97_submit.csv"
PD10_CSV = "v_pdtrk_anwshub_submit.csv"
AN_MEM_TT = HERE.parent / "pd10_anen_members_traintest.parquet"
DL_TEST = HERE.parent / "pd10_test_dl_anwshub_d3.parquet"
DL_TRAIN = HERE.parent / "pd10_dltrain_anwshub.parquet"
OPT_JSON = HERE.parent / "pd14_opt.json"
FIN_TEST = HERE.parent / "pd16_test_final_anwshub_optlgb.parquet"
OUT_MAIN = "v_pdtrk_anwshub_optlgb_submit.csv"
SEEDS = 3
MEM_WS = [f"mws{k}" for k in range(30)]
CLIP_LO, ALPHA = 0.16, 0.40


def main():
    t0 = time.time()
    print("### pd16 제출 빌더: anws_hub_optlgb (변경점 = 잔차 GBM = optlgb 단독) ###",
          flush=True)
    for f in (E49_BIAS, AN_MEM_TT, DL_TEST, DL_TRAIN, OPT_JSON):
        assert f.exists(), f"필수 캐시 부재: {f.name}"
    dl_pool, gbm_pool, env_cols = load_pools()
    opt = json.loads(OPT_JSON.read_text())["lgb"]["params"]
    print(f"optlgb: {opt}", flush=True)

    # === 프레임 (PD-10과 동일 경로: 348 + EC4) ===
    train_df, test_df, col_sets, sample_submission = build_v11()
    jma = build_jma_frame()
    train_df = add_jma(train_df, jma)
    test_df = add_jma(test_df, jma)
    _, spd, _, spd_cols = build_spatial_all()
    train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=KEY, how="left")
    test_df = test_df.merge(ec, on=KEY, how="left")
    for df in (train_df, test_df):
        df["ec_dis"] = (df["ec_ws"] - df["ldaps_ws_10m"]).abs()
    train_df = merge_avail(train_df)
    train_df = add_calendar(train_df)
    test_df = add_calendar(test_df)
    need = sorted({c for g in (1, 2, 3) for e in W14[g] for c in entry_cols(e)})
    gmiss = [c for c in gbm_pool if c not in test_df.columns]
    assert not gmiss, f"test GBM 풀 부재 {gmiss[:5]}"
    print(f"프레임 train {train_df.shape} / test {test_df.shape} — "
          f"{time.time()-t0:.0f}s", flush=True)

    # === AnEn 멤버 (PD-10 캐시) ===
    mem = pd.read_parquet(AN_MEM_TT)
    for k in range(30):
        mem[f"mws{k}"] = np.hypot(mem[f"m{k}u"], mem[f"m{k}v"])

    def with_mem(df):
        m = df.merge(mem[KEY + MEM_WS], on=KEY, how="left")
        assert len(m) == len(df), "멤버 병합 행 증식"
        return m

    train_df = with_mem(train_df)
    test_df = with_mem(test_df)
    assert test_df["mws0"].notna().all(), "test 멤버 결측"
    lab = train_df.loc[train_df["target"].notna()].copy()

    # === DL 예측 (PD-10 캐시: train 교차적합 + test) ===
    prev = pd.read_parquet(DL_TRAIN)
    lab = lab.merge(prev[KEY + ["pd_dl_cf"]], on=KEY, how="left")
    assert lab["pd_dl_cf"].notna().all(), "DL 학습행 캐시 커버 미달"
    te = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).reset_index(drop=True)
    dlp = pd.read_parquet(DL_TEST)
    te = te.merge(dlp.rename(columns={"dl_anws_hub": "pd_dl_cf"}),
                  on=["forecast_id", "kpx_group"], how="left")
    assert te["pd_dl_cf"].notna().all()
    print(f"DL 캐시 재사용 완료 (train {len(lab):,} / test {len(te):,}) — "
          f"{time.time()-t0:.0f}s", flush=True)

    # === GBM 잔차 최종: optlgb 단독 · 전 라벨행 → 2025 ===
    import lightgbm as lgb
    gbm_cols = gbm_pool + ["pd_dl_cf"]
    if FIN_TEST.exists():
        fin = pd.read_parquet(FIN_TEST)
        print(f"최종 예측 캐시 로드: {FIN_TEST.name}", flush=True)
    else:
        rows = []
        for g in [1, 2, 3]:
            tg = time.time()
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            tr_g = lab.loc[lab["kpx_group"] == g]
            te_g = te.loc[te["kpx_group"] == g]
            X_tr, X_te = prepare_matrices(tr_g, te_g, gbm_cols)
            y_res = tr_g["target"].to_numpy() - tr_g["pd_dl_cf"].to_numpy() * cap
            w_tr = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
                avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            resid = np.mean([lgb.LGBMRegressor(random_state=s, n_jobs=8, verbose=-1,
                                               device="gpu", **opt)
                             .fit(X_tr, y_res, sample_weight=w_tr).predict(X_te)
                             for s in range(SEEDS)], axis=0)
            rows.append(pd.DataFrame({
                "forecast_id": te_g["forecast_id"].to_numpy(), "kpx_group": g,
                "ws": te_g["ldaps_ws_10m"].to_numpy(),
                "cf_final": te_g["pd_dl_cf"].to_numpy() + resid / cap,
                "cf_dl": te_g["pd_dl_cf"].to_numpy()}))
            print(f"  g{g} optlgb 잔차 완료 ({time.time()-tg:.0f}s | "
                  f"|잔차| 평균 {np.abs(resid).mean()/cap*100:.2f}%cap)", flush=True)
        fin = pd.concat(rows, ignore_index=True)
        fin.to_parquet(FIN_TEST, index=False)

    # === PP + 제출 조립 (PD-10과 동일) ===
    with open(E49_BIAS, "rb") as f:
        coarse_map = pickle.load(f)
    parts = []
    for g in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        fg = fin.loc[fin["kpx_group"] == g]
        p = apply_bias_correction_v2(fg["cf_final"].to_numpy() * cap,
                                     fg["ws"].to_numpy(), g, coarse_map)
        p = apply_rolling_smoothing(pd.Series(p), window=3, alpha=ALPHA).to_numpy().copy()
        p = np.clip(p, cap * CLIP_LO, cap)
        p = np.clip(p * SCALES[g], 0, cap)
        parts.append(pd.DataFrame({"forecast_id": fg["forecast_id"].to_numpy(),
                                   "col": f"kpx_group_{g}", "pred": p}))
    sub = pd.concat(parts).pivot(index="forecast_id", columns="col", values="pred").reset_index()
    sub = sample_submission[["forecast_id", "forecast_kst_dtm"]].merge(sub, on="forecast_id",
                                                                       how="left")
    assert sub.shape[0] == sample_submission.shape[0], "행 수 불일치"
    assert sub[TARGET_COLS].isna().sum().sum() == 0, "NaN 존재"
    assert (sub["forecast_id"] == sample_submission["forecast_id"]).all()
    for col in TARGET_COLS:
        assert sub[col].max() <= CAPACITY_KWH[col] and sub[col].min() >= 0, f"{col} 범위 이탈"
    sub[["forecast_id", "forecast_kst_dtm"] + TARGET_COLS].to_csv(
        DATA_DIR / OUT_MAIN, index=False, encoding="utf-8-sig")

    # === 제출물 진단 (챔피언·PD-10 대비) ===
    print(f"\n[제출] {OUT_MAIN}", flush=True)
    for nm, path in (("챔피언", CHAMP_MAIN), ("PD-10 anws_hub", PD10_CSV)):
        ref = pd.read_csv(DATA_DIR / path, encoding="utf-8-sig")
        tot = 0.0
        print(f"-- vs {nm} --", flush=True)
        for col in TARGET_COLS:
            cap = CAPACITY_KWH[col]
            d = sub[col].to_numpy() - ref[col].to_numpy()
            r = np.corrcoef(sub[col], ref[col])[0, 1]
            tot += d.mean() / cap * 100 / 3
            print(f"  {col}: 이동 {d.mean()/cap*100:+.3f}%cap | 평균 |차이| "
                  f"{np.abs(d).mean()/cap*100:.3f}%cap | corr {r:.4f}", flush=True)
        print(f"  → 평균 이동 {tot:+.3f}%cap", flush=True)
    print(f"\n※ 게이트(트랙): 제출 작동점 CV 0.6499 (anws_hub +0.0012 · Δn +0.0009 · "
          f"Δf +0.0014) / HOLD 0.6401", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
