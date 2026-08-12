# ═══════════════════════════════════════════════════════
# pd10 제출 빌더: 트랙 D anws_hub — 첫 제출판 (PD-10)
#
# 사용자 지시 (2026-08-11): "우선 anws_hub 제출 파일을 만들어보자"
#
# 구성 (트랙 기준 = PD-03 anws_hub, 게이트 제출 작동점 CV 0.6487 / HOLD 0.6414):
#   1단 DL base : AnEn K=30 멤버 ws 30열 → MTMLP 384×3 · hub 손실 · lw 0.05 ·
#                 ES(val 블록 = 2024 3/6/9/12월 11~20일 · patience 15 · best 복원)
#   2단 GBM 잔차: 타깃 y − DL·cap · 입력 = DL 출력 + GBM 풀 269 + calendar ·
#                 LGB e56(448) + XGB e56(585) 50:50 · 챔피언 가중 · GPU
#   PP          : 챔피언 최종 편향맵(E49_BIAS) → rolling α0.40 → clip 0.16 →
#                 그룹 스케일 0.98/0.98/0.97 → clip [0, cap] (스냅 OFF)
#
# ⚠ 시드 = DL·GBM 모두 3 (게이트와 동일) — 챔피언 관례(DL 5)와 다름.
#   근거 = exp-145(게이트 3시드 vs 제출 5시드 불일치가 게이트 증거를 무효화).
#   트랙 D 게이트 전 라운드가 3시드라 제출도 3시드로 게이트 증거를 보존한다.
#
# AnEn 멤버 traintest (av08 관례): train행 = 전 학습창 시간중앙 2분할 교차적합 /
#   test행 = 전체 풀 → pd10_anen_members_traintest.parquet (60열, 자산)
#
# 산출:
#   open (1)/v_pdtrk_anwshub_submit.csv (+ 캐시 pd10_*.parquet 보존)
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd10_submit_anwshub.py"
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
import torch

from an12_emos import build_scada_vec
from an14_featsel import entry_cols
from an14b_final import W14
from exp005_qopt import (CAPACITY_KWH, DATA_DIR, TARGET_COLS, make_sample_weight,
                         prepare_matrices, apply_bias_correction_v2,
                         apply_rolling_smoothing)
from exp012_feature_v11 import build_v11
from exp020_jma_features import JG_COLS, JM_COLS, build_jma_frame
from exp020_submit import add_jma
from exp033_dl import GIDX, predict
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import CACHE_DIR
from exp039_dl_diverse import DL_CFG, make_loss
from exp049_clean import avail_factor, make_xy_clean, merge_avail
from exp054_gate import EC_WIDE, build_ec_frame
from exp056_hpo import RESULT_JSON as E56_JSON

from wi01_bivgate import SCALES
from wi09_earlystop import train_one_es, val_mask

from av00_anenfeat import anen_block

from pd00_inputs import CALENDAR9, add_calendar, load_pools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["kpx_group", "forecast_kst_dtm"]
E49_BIAS = CACHE_DIR / "exp049_test_bias_cln03_g20_d5.pkl"
CHAMP_MAIN = "v_a40nosnap_ens3es_dles_w65_g3s97_submit.csv"
AN_MEM_TT = HERE.parent / "pd10_anen_members_traintest.parquet"
DL_TEST = HERE.parent / "pd10_test_dl_anwshub_d3.parquet"
DL_TRAIN = HERE.parent / "pd10_dltrain_anwshub.parquet"
FIN_TEST = HERE.parent / "pd10_test_final_anwshub.parquet"
OUT_MAIN = "v_pdtrk_anwshub_submit.csv"
SEEDS = 3
MEM_WS = [f"mws{k}" for k in range(30)]
CLIP_LO, ALPHA = 0.16, 0.40


def fold_predict_hub(tr_df, va_df, cols, seeds, tag):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    vm = val_mask(tr_df)
    fit_df = tr_df.loc[~vm]
    val_df = tr_df.loc[vm & tr_df["target"].notna()]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(fit_df, cols, imp, sca, "cln03", fit=True)
    Xv, gv, yv, wv = make_xy_clean(val_df, cols, imp, sca, "cln03", fit=False)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("hub")
    preds, eps = [], []
    for s in range(seeds):
        m, bep = train_one_es(Xtr, gtr, ytr, wtr, Xv, gv, yv, wv, DL_CFG, loss_fn, s, True)
        preds.append(predict(m, Xva, gva))
        eps.append(bep)
        del m
        torch.cuda.empty_cache()
    print(f"    {tag} best-epoch: {eps} (val {len(val_df):,}행)", flush=True)
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    print("### pd10 제출 빌더: 트랙 D anws_hub (DL·GBM 3시드 = 게이트 동일) ###", flush=True)
    assert E49_BIAS.exists(), "챔피언 최종 편향맵 부재"
    dl_pool, gbm_pool, env_cols = load_pools()
    e56 = json.loads(E56_JSON.read_text())

    # === 프레임 (챔피언 제출 경로: 348 + EC4) ===
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
    for nm, df in (("train", train_df), ("test", test_df)):
        miss = [c for c in need if c not in df.columns]
        assert not miss, f"{nm}: AnEn 입력 부재 {miss}"
    gmiss = [c for c in gbm_pool if c not in test_df.columns]
    assert not gmiss, f"test GBM 풀 부재 {gmiss[:5]}"
    print(f"프레임 train {train_df.shape} / test {test_df.shape} — "
          f"{time.time()-t0:.0f}s", flush=True)

    # === AnEn 멤버: train 교차적합 + test 전체 풀 (av08 관례 + 멤버 저장) ===
    if AN_MEM_TT.exists():
        mem = pd.read_parquet(AN_MEM_TT)
        print(f"AnEn 멤버 캐시 로드: {mem.shape}", flush=True)
    else:
        vec = build_scada_vec()
        for g in (1, 2, 3):
            train_df = train_df.merge(vec[g].assign(kpx_group=g), on=KEY, how="left",
                                      suffixes=("", "_d"))
            for c in ["park_ws", "u_obs", "v_obs"]:
                if f"{c}_d" in train_df:
                    train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
        obs = train_df[train_df["u_obs"].notna()]
        mid = obs["forecast_kst_dtm"].quantile(0.5)
        A, B = obs[obs["forecast_kst_dtm"] <= mid], obs[obs["forecast_kst_dtm"] > mid]
        tr_a = train_df[train_df["forecast_kst_dtm"] <= mid]
        tr_b = train_df[train_df["forecast_kst_dtm"] > mid]
        blocks = []
        for pool, query, tag in ((A, tr_b, "cf(A→후반)"), (B, tr_a, "cf(B→전반)"),
                                 (obs, test_df, "test(전체→2025)")):
            M = anen_block(pool, query, tag)
            d = query[KEY].reset_index(drop=True).copy()
            for k in range(30):
                d[f"m{k}u"] = M[:, k, 0]
                d[f"m{k}v"] = M[:, k, 1]
            blocks.append(d)
            print(f"  {tag}: 풀 {pool['u_obs'].notna().sum():,} → 예측 {len(query):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        mem = pd.concat(blocks, ignore_index=True)
        assert not mem.duplicated(KEY).any(), "행 중복"
        mem.to_parquet(AN_MEM_TT, index=False)
        print(f"AnEn 멤버 저장: {AN_MEM_TT.name} {mem.shape}", flush=True)
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
    print(f"멤버 커버 train(라벨행) {lab['mws0'].notna().mean():.4f} / test 1.0000",
          flush=True)

    # === DL train행 교차적합 (잔차 타깃용) ===
    if DL_TRAIN.exists():
        prev = pd.read_parquet(DL_TRAIN)
        lab = lab.merge(prev[KEY + ["pd_dl_cf"]], on=KEY, how="left")
        print(f"DL 학습행 예측 캐시 로드 ({lab['pd_dl_cf'].notna().sum():,}행)", flush=True)
    else:
        lab["pd_dl_cf"] = np.nan
        # 전역 시간중앙 2분할 (av08 AnEn 분할과 동일 관례).
        # 그룹별 분할은 전체 학습창에서 g3 전반부가 2024 초까지 걸쳐
        # ES val 블록(마지막 해 3/6/9/12월)이 빈 집합이 됨 — 1차 실행 실측.
        mid = lab["forecast_kst_dtm"].quantile(0.5)
        h0 = lab["forecast_kst_dtm"] <= mid
        for fit_m, q_m, tg in ((h0, ~h0, "cf01"), (~h0, h0, "cf10")):
            p = fold_predict_hub(lab.loc[fit_m], lab.loc[q_m], MEM_WS, SEEDS, tg)
            lab.loc[q_m, "pd_dl_cf"] = p
        lab[KEY + ["pd_dl_cf"]].to_parquet(DL_TRAIN, index=False)
    assert lab["pd_dl_cf"].notna().all()

    # === DL 최종: 전 학습창 → 2025 ===
    te = test_df.sort_values(["kpx_group", "forecast_kst_dtm"]).reset_index(drop=True)
    if DL_TEST.exists():
        dlp = pd.read_parquet(DL_TEST)
        print(f"DL test 캐시 로드: {DL_TEST.name}", flush=True)
    else:
        dl_te = fold_predict_hub(train_df, te, MEM_WS, SEEDS, "final:test")
        dlp = pd.DataFrame({"forecast_id": te["forecast_id"].to_numpy(),
                            "kpx_group": te["kpx_group"].to_numpy(),
                            "dl_anws_hub": dl_te})
        dlp.to_parquet(DL_TEST, index=False)
    te = te.merge(dlp.rename(columns={"dl_anws_hub": "pd_dl_cf"}),
                  on=["forecast_id", "kpx_group"], how="left")
    assert te["pd_dl_cf"].notna().all()

    # === GBM 잔차 최종: 전 라벨행 → 2025 ===
    import lightgbm as lgb
    import xgboost as xgbm
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
            pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                                            random_state=s, n_jobs=8, verbose=-1,
                                            device="gpu", **e56["lgb"])
                          .fit(X_tr, y_res, sample_weight=w_tr).predict(X_te)
                          for s in range(SEEDS)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"],
                                            objective="reg:absoluteerror",
                                            tree_method="hist", device="cuda",
                                            random_state=s, n_jobs=8, verbosity=0,
                                            **e56["xgb"])
                          .fit(X_tr, y_res, sample_weight=w_tr).predict(X_te)
                          for s in range(SEEDS)], axis=0)
            resid = 0.5 * pl + 0.5 * px
            rows.append(pd.DataFrame({
                "forecast_id": te_g["forecast_id"].to_numpy(), "kpx_group": g,
                "ws": te_g["ldaps_ws_10m"].to_numpy(),
                "cf_final": te_g["pd_dl_cf"].to_numpy() + resid / cap,
                "cf_dl": te_g["pd_dl_cf"].to_numpy()}))
            print(f"  g{g} 잔차 GBM 완료 ({time.time()-tg:.0f}s | "
                  f"|잔차| 평균 {np.abs(resid).mean()/cap*100:.2f}%cap)", flush=True)
        fin = pd.concat(rows, ignore_index=True)
        fin.to_parquet(FIN_TEST, index=False)

    # === PP + 제출 조립 ===
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

    # === 제출물 진단 (챔피언 대비) ===
    champ = pd.read_csv(DATA_DIR / CHAMP_MAIN, encoding="utf-8-sig")
    print(f"\n[제출] {OUT_MAIN}", flush=True)
    tot = 0.0
    for col in TARGET_COLS:
        cap = CAPACITY_KWH[col]
        d = sub[col].to_numpy() - champ[col].to_numpy()
        r = np.corrcoef(sub[col], champ[col])[0, 1]
        tot += d.mean() / cap * 100 / 3
        print(f"  {col}: 이동 {d.mean()/cap*100:+.3f}%cap | 평균 |차이| "
              f"{np.abs(d).mean()/cap*100:.3f}%cap | corr {r:.4f}", flush=True)
    print(f"  → 제출값 평균 이동 {tot:+.3f}%cap (챔피언 대비)", flush=True)
    print(f"\n※ 게이트(트랙): 제출 작동점 CV 0.6487 / HOLD 0.6414 | 참고 챔피언 행 대비 "
          f"+0.0042 (Δn −0.0034 / Δf +0.0118 — 2024 수치, 트랙 내부 기준)", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
