# ═══════════════════════════════════════════════════════
# exp-055: 터빈 단위 분해 모델 — SCADA 중간 타깃 (그룹 라벨 → 터빈별 발전량 합산)
#
# 배경 (2026-07-28, 사용자 합의):
#   - LB 111위, 1위와 격차 0.0257의 88%가 FiCR (우리 0.4209 vs 1위 0.4663).
#   - 그룹 라벨은 "정지 섞인 합"이라 정격 어깨(CF 0.7~0.9 = FiCR 손실 33%)가 뭉개짐.
#     터빈 단위에서는 정지가 행 단위로 식별·제거되고 파워커브가 날카로움.
#   - 주최측 명시 용도: "SCADA는 터빈 파워커브 학습·중간 타깃(터빈별 발전량)" (COMPETITION.md).
#
# 검증된 전제 (이 세션 사전 검증):
#   - power_kw10m = 10분 슬롯당 발전량(kWh). 정격 슬롯 V126 600·U136 700.
#   - 시간 귀속 = ceil (hour-ending): corr(터빈합, 라벨) 0.9998(g1)/0.9965(g3).
#   - 라벨/터빈합 보정 비율: g1 ~0.986(변전 손실), g3 ~1.000 → 그룹별 재추정해 적용.
#   - 글리치: 슬롯 상한 ×1.1 초과 → NaN, 음수 → 0.
#
# 구성: 그룹 내 터빈 풀링(6/6/5기, one-hot id — exp-011 전 그룹 풀링 실패 전례로 그룹 경계
#   유지). 타깃 = 터빈 CF, weight = 저CF 0.05 규칙 × (dead ? 0.05 : 1.0). 피처 = 352
#   (gspd 348 + EC 4, 그룹 공통) + 달력 + 터빈 one-hot. LGBM/XGB 50:50 3-seed.
#   그룹 예측 = Σ(터빈 CF × 터빈용량) × calib_g. DL·pA·편향맵·신PP 고정.
# 평가: champ49(cln×cln_w70) + cf_ec(352 그룹 GBM, 캐시) 대비 — ΔvsEC가 분해 효과 격리.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp055_turbine.py --jobs 8 --seeds 3
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, lgbm_params, prepare_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (AVAIL_PARQUET, GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY,
                          PP_CHAMP, PP_NEW, merge_avail, postprocess_flex, load_eval_frame)
from exp049_diag import load_scada, GROUP_TURBINES
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS, GBM_EC_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TB_HOURS_CACHE = CACHE_DIR / "exp055_turbine_hours.parquet"
GBM_TB_CACHE = CACHE_DIR / "exp055_gbm_tb.parquet"
TCAP = {"vestas": 3600.0, "unison": 4200.0}     # kWh/h (V126 3.6MW·U136 4.2MW)
CAPSLOT = {"vestas": 660.0, "unison": 770.0}    # 10분 슬롯 상한 (정격/6 ×1.1)


def build_turbine_hours():
    """터빈×시간 발전량(kWh)·dead 플래그 (ceil 귀속, 글리치 정제, 5슬롯+ 스케일 보정)"""
    if TB_HOURS_CACHE.exists():
        return pd.read_parquet(TB_HOURS_CACHE)
    sc = load_scada()
    hour = sc["kst_dtm"].dt.ceil("h")
    rows = []
    for g, turbines in GROUP_TURBINES.items():
        w_cols = [f"{mk}_wtg{i:02d}_ws" for mk, i in turbines]
        med_ws = sc[w_cols].median(axis=1)
        for mk, i in turbines:
            pc, wc = f"{mk}_wtg{i:02d}_power_kw10m", f"{mk}_wtg{i:02d}_ws"
            p = sc[pc].astype(float).clip(lower=0)
            p = p.where(p <= CAPSLOT[mk], np.nan)
            n = p.notna().groupby(hour).sum()
            e = p.groupby(hour).sum(min_count=1)
            e_h = (e * (6.0 / n.replace(0, np.nan))).where(n >= 5, np.nan)
            ws_eff = sc[wc].fillna(med_ws)
            dead_slot = (p.isna() | (p <= 1.0)) & (ws_eff >= 5.0)
            dead_h = dead_slot.groupby(hour).sum() >= 5
            rows.append(pd.DataFrame({
                "forecast_kst_dtm": e_h.index, "group": g, "tb": f"{mk}{i:02d}",
                "tcap": TCAP[mk], "energy": e_h.to_numpy(), "dead": dead_h.reindex(e_h.index).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(TB_HOURS_CACHE, index=False)
    return out


def calib_ratios(tbh, train_df):
    """그룹별 라벨/터빈합 보정 비율 (전 터빈 생존·완전 시간만)"""
    lab = train_df.loc[train_df["target"].notna(), ["forecast_kst_dtm", "kpx_group", "target"]] \
                  .rename(columns={"kpx_group": "group"})
    ratios = {}
    for g in [1, 2, 3]:
        sub = tbh[tbh["group"] == g]
        piv_e = sub.pivot_table(index="forecast_kst_dtm", columns="tb", values="energy")
        piv_d = sub.pivot_table(index="forecast_kst_dtm", columns="tb", values="dead")
        ok = piv_e.notna().all(axis=1) & (piv_d < 0.5).all(axis=1)
        ssum = piv_e.loc[ok].sum(axis=1)
        m = lab[lab["group"] == g].set_index("forecast_kst_dtm")["target"].to_frame().join(
            ssum.rename("ss"), how="inner").dropna()
        m = m[m["ss"] > 0]
        ratios[g] = float((m["ss"] * m["target"]).sum() / (m["ss"] ** 2).sum())
    return ratios


def train_tb_fold(train_df, tbh, cols, fold, seeds, jobs, calib):
    """폴드 하나: 그룹별 터빈 풀링 학습 → 합산 그룹 CF 예측"""
    import lightgbm as lgb
    import xgboost as xgbm
    tr_w = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_w = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                        (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    out = []
    for group in [1, 2, 3]:
        gcap = CAPACITY_KWH[f"kpx_group_{group}"]
        tw = tr_w.loc[tr_w["kpx_group"] == group]
        va_g = va_w.loc[va_w["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        tbg = tbh.loc[(tbh["group"] == group) & tbh["energy"].notna()]
        tr_t = tbg.merge(tw, left_on="forecast_kst_dtm", right_on="forecast_kst_dtm", how="inner")
        if len(tr_t) == 0 or len(va_g) == 0:
            continue
        tbs = sorted(tbg["tb"].unique())
        oh_cols = [f"tb_{t}" for t in tbs]
        for t in tbs:
            tr_t[f"tb_{t}"] = (tr_t["tb"] == t).astype(float)
        va_rep = pd.concat([va_g.assign(**{f"tb_{t2}": float(t2 == t) for t2 in tbs}, _tb=t)
                            for t in tbs], ignore_index=True)
        use_cols = list(cols) + oh_cols
        X_tr, X_va = prepare_matrices(tr_t, va_rep, use_cols)
        cf_t = (tr_t["energy"] / tr_t["tcap"]).to_numpy()
        w_tr = np.where(cf_t < 0.10, 0.05, 1.0) * np.where(tr_t["dead"].to_numpy(), 0.05, 1.0)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, jobs, s))
                      .fit(X_tr, cf_t, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                      n_jobs=jobs, verbosity=0, objective="reg:absoluteerror")
                      .fit(X_tr, cf_t, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        pred_t = np.clip(0.5 * pl + 0.5 * px, 0, 1).reshape(len(tbs), len(va_g))
        tcaps = np.array([TCAP[t[:6]] for t in tbs])[:, None]
        g_kwh = (pred_t * tcaps).sum(axis=0) * calib[group]
        out.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                 "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                 "cf_tb": g_kwh / gcap}))
    return pd.concat(out, ignore_index=True)


def evaluate(cache, champ_bias):
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def ev(gbm_col, dl_col, w, pp):
        blend = cache[gbm_col].to_numpy() if (dl_col is None or w >= 1.0) else \
            w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    r_ref = ev("cf_gspd", "dl_q60", 0.70, PP_NEW)
    r_ch = ev("cf_cln03", "dl_cln03", 0.70, PP_NEW)
    r_ec = ev("cf_ec", "dl_cln03", 0.65, PP_NEW)
    print(f"\nref(gspd·q60·w70·신PP):     CV {r_ref['cv'][0]:.4f} | HOLD {r_ref['hold'][0]:.4f}  (기대 0.6481/0.6463)")
    print(f"champ49(cln×cln_w70·신PP):  CV {r_ch['cv'][0]:.4f} | HOLD {r_ch['hold'][0]:.4f}  (기대 0.6534/0.6518)")
    print(f"보드챔피언(ec×dC_w65):       CV {r_ec['cv'][0]:.4f} (1-NMAE {r_ec['cv'][1]:.4f} FiCR {r_ec['cv'][2]:.4f}) | "
          f"HOLD {r_ec['hold'][0]:.4f}  (기대 0.6541/0.6529)")

    # 터빈합을 GBM 멤버로: 단독·DL 블렌드·EC GBM과 멤버 혼합
    cache["cf_ectb"] = 0.5 * cache["cf_ec"].to_numpy() + 0.5 * cache["cf_tb"].to_numpy()
    res = {"gTB_w100": ev("cf_tb", None, 1.0, PP_NEW)}
    for w in (0.75, 0.70, 0.65, 0.60):
        res[f"gTB×dC_w{int(w*100)}"] = ev("cf_tb", "dl_cln03", w, PP_NEW)
        res[f"mixEC×dC_w{int(w*100)}"] = ev("cf_ectb", "dl_cln03", w, PP_NEW)

    ch_cv, ch_hd = r_ec["cv"][0], r_ec["hold"][0]
    ch_n, ch_f = r_ec["cv"][1], r_ec["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<16}{'CV':>8}{'ΔCVec':>9}{'HOLD':>8}{'Δhdec':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=보드챔피언 ec_w65 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ec["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<16}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both))
    print(f"\n--- fold별 ΔCV (ec_w65 대비) ---")
    print(f"{'변형':<16}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관")
    for r in rows:
        print(f"{r['name']:<16}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}")

    # 기전: 정격 어깨(실측 CF 0.7~0.9)·만가용 signed error
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    tmp = cache.copy()
    tmp["forecast_kst_dtm"] = pd.to_datetime(tmp["forecast_kst_dtm"])
    tmp = tmp.merge(av[["forecast_kst_dtm", "group", "avail_strict"]], on=["forecast_kst_dtm", "group"], how="left")
    caps = tmp["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = tmp["target"].to_numpy() / caps
    evm = (tmp["target"].to_numpy() >= caps * 0.10) & tmp["avail_strict"].notna().to_numpy()
    full = evm & (tmp["avail_strict"].to_numpy() >= 0.999)
    shoulder = evm & (acf >= 0.7) & (acf < 0.9)
    print("\n--- 기전 (보정 전 원시 CF): 만가용/정격어깨 signed + MAE + corr_DL ---")
    print(f"{'모델':<10}{'만가용':>10}{'어깨(0.7~0.9)':>14}{'MAE':>8}{'corr_dl':>9}")
    e_d = tmp["dl_cln03"].to_numpy() - acf
    for c in ["cf_ec", "cf_tb", "cf_ectb"]:
        e = tmp[c].to_numpy() - acf
        cr = np.corrcoef(e[evm], e_d[evm])[0, 1]
        print(f"{c:<10}{e[full].mean():>+10.4f}{e[shoulder].mean():>+14.4f}{np.abs(e[evm]).mean():>8.4f}{cr:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-055 터빈 단위 분해 모델 ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
    use_cols = vcols + EC_COLS                       # 352

    tbh = build_turbine_hours()
    tbh["forecast_kst_dtm"] = pd.to_datetime(tbh["forecast_kst_dtm"])
    nlab = tbh["energy"].notna().sum()
    print(f"터빈-시간 행: {len(tbh)} (에너지 유효 {nlab}) | dead 비율 "
          f"{tbh.loc[tbh['energy'].notna(), 'dead'].mean()*100:.1f}%", flush=True)
    calib = calib_ratios(tbh, train_df)
    print(f"보정 비율 (라벨/터빈합): " + ", ".join(f"그룹{g} {r:.4f}" for g, r in calib.items()), flush=True)

    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"]), "exp-049 GBM 캐시 필요"
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"]), "exp-049 DL 캐시 필요"
    assert merge_cached_cols(cache, GBM_EC_CACHE, ["cf_ec"]), "exp-054 EC GBM 캐시 필요"

    have = merge_cached_cols(cache, GBM_TB_CACHE, ["cf_tb"])
    if "cf_tb" not in have:
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(train_tb_fold(train_df, tbh, use_cols, fold, args.seeds, args.jobs, calib))
            print(f"  [tb:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_tb"].notna().all(), "정렬 실패"
        cache["cf_tb"] = m["cf_tb"].to_numpy()
        save_cols(cache, GBM_TB_CACHE, ["cf_tb"])
        print(f"  캐시 저장 → {GBM_TB_CACHE.name}", flush=True)

    evaluate(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
