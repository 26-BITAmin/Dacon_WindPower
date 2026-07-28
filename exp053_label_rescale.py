# ═══════════════════════════════════════════════════════
# exp-053: 가용성 기반 학습 라벨 재구성 (rescale — 다운웨이트의 반대 방향)
#
# 근거:
#   - exp-049 진단: 라벨 = 날씨발전량 × 가용률 (ws 12~14에서 avail 1.0 CF 0.846 /
#     0.7~0.9 0.703 / <0.7 0.399 — 정비례). 챔피언은 정지 시간(11.2%)을 weight ×0.05로
#     사실상 폐기. exp-050에서 λ 완전 단조(0.05 정점) = "버리는 강도"는 소진.
#   - 미실험 축: 같은 기전을 역이용해 y' = min(y/avail, cap)로 만가용 발전량을 **복원**
#     하고 full weight로 학습 — 11.2% 데이터를 정보로 되살림.
#   - 노이즈 관리: avail < 0.5(복원 배율 2배+)는 복원하지 않고 기존 ×0.05 다운웨이트 유지.
#     헤지 변형 rscd(복원 행 weight 0.5)로 avail 추정 노이즈 민감도 브래킷.
#
# 변형: rsc(복원 행 w1.0) / rscd(복원 행 w0.5) — GBM·DL(q60 τ0.60) 각각 학습.
# 고정: 피처 spd 348·모델·신PP·챔피언 편향맵 = champ49와 동일 (변경점 = 학습 라벨·weight만).
# 평가: GBM {cln03, rsc, rscd} × DL {cln03, rsc, rscd} × w {0.70, 0.65} 그리드,
#   champ49(cln×cln_w70) 대비 3중 조건. 검증 라벨은 원본 그대로(재구성은 학습행만).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp053_label_rescale.py --jobs 8 --seeds 3
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, lgbm_params, prepare_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, DL_CFG, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices, make_loss, train_one
from exp049_clean import (AVAIL_PARQUET, GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY,
                          PP_CHAMP, PP_NEW, merge_avail, postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import merge_cached_cols, save_cols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = ["rsc", "rscd"]          # rsc: 복원 행 w1.0 / rscd: 복원 행 w0.5
GBM_RSC_CACHE = CACHE_DIR / "exp053_gbm_rsc.parquet"
DL_RSC_CACHE = CACHE_DIR / "exp053_dl_rsc.parquet"
AV_FLOOR = 0.5                       # 이 미만은 복원하지 않고 ×0.05 다운웨이트 유지


def rescale_y_w(y, cap, avail, variant):
    """학습 라벨 복원 + 가용성 weight 팩터. 반환: (y', weight_factor)"""
    a = np.nan_to_num(np.asarray(avail, dtype=float), nan=1.0)
    y = np.asarray(y, dtype=float)
    y2 = y.copy()
    wf = np.ones(len(y))
    m_rs = (a < 0.999) & (a >= AV_FLOOR)          # 복원 대상
    y2[m_rs] = np.minimum(y[m_rs] / a[m_rs], cap)
    if variant == "rscd":
        wf[m_rs] = 0.5
    wf[a < AV_FLOOR] = 0.05                        # 복원 배율 과대 구간은 기존 폐기 처리
    return y2, wf


# ── GBM ──
def train_gbm_fold_rsc(train_df, cols, fold, seeds, jobs, variant):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y2, wf = rescale_y_w(tr_g["target"].to_numpy(), cap, tr_g["avail_strict"].to_numpy(), variant)
        w_tr = make_sample_weight(y2, cap, low_weight=0.05) * wf
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, jobs, s))
                      .fit(X_tr, y2, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                      n_jobs=jobs, verbosity=0, objective="reg:absoluteerror")
                      .fit(X_tr, y2, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  f"cf_{variant}": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


# ── DL (q60 τ0.60 고정 — 변경점 격리) ──
def make_xy_rsc(df, cols, imputer, scaler, variant, fit=False):
    from exp033_dl import GIDX
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    # CF 공간에서 복원 (cap=1.0 — cf/avail을 1.0으로 clip)
    cf = (sub["target"].to_numpy() / caps).astype(np.float64)
    cf2, wf = rescale_y_w(cf, 1.0, sub["avail_strict"].to_numpy(), variant)
    w = np.where(cf2 < 0.10, 0.05, 1.0).astype(np.float32) * wf.astype(np.float32)
    return Xs.astype(np.float32), g, cf2.astype(np.float32), w


def dl_fold_predict_rsc(tr_df, va_df, cols, seeds, variant):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_rsc(tr_df, cols, imp, sca, variant, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def evaluate_grid(cache, champ_bias):
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

    r_old = ev("cf_g132", "dl_hub", 0.70, PP_CHAMP)
    r_ref = ev("cf_gspd", "dl_q60", 0.70, PP_NEW)
    r_ch = ev("cf_cln03", "dl_cln03", 0.70, PP_NEW)
    print(f"\n구챔피언(g132·hub·w70·구PP): CV {r_old['cv'][0]:.4f} | HOLD {r_old['hold'][0]:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):     CV {r_ref['cv'][0]:.4f} | HOLD {r_ref['hold'][0]:.4f}  (기대 0.6481/0.6463)")
    print(f"champ49(cln×cln_w70·신PP):  CV {r_ch['cv'][0]:.4f} (1-NMAE {r_ch['cv'][1]:.4f} FiCR {r_ch['cv'][2]:.4f}) | "
          f"HOLD {r_ch['hold'][0]:.4f}  (기대 0.6534/0.6518)")

    gbm_map = {"gC": "cf_cln03", "gR": "cf_rsc", "gD": "cf_rscd"}
    dl_map = {"dC": "dl_cln03", "dR": "dl_rsc", "dD": "dl_rscd"}
    res = {}
    for gn, gc in gbm_map.items():
        res[f"{gn}_w100"] = ev(gc, None, 1.0, PP_NEW)
        for dn, dc in dl_map.items():
            for w in (0.70, 0.65):
                res[f"{gn}×{dn}_w{int(w*100)}"] = ev(gc, dc, w, PP_NEW)

    ch_cv, ch_hd = r_ch["cv"][0], r_ch["hold"][0]
    ch_n, ch_f = r_ch["cv"][1], r_ch["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVch':>9}{'HOLD':>8}{'Δhdch':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=champ49 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ch["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))
    print(f"\n--- fold별 ΔCV (champ49 대비, 상위 12) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"]))[:12]:
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # 기전: 만가용/저가용 signed error + MAE
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    tmp = cache.copy()
    tmp["forecast_kst_dtm"] = pd.to_datetime(tmp["forecast_kst_dtm"])
    tmp = tmp.merge(av[["forecast_kst_dtm", "group", "avail_strict"]], on=["forecast_kst_dtm", "group"], how="left")
    caps = tmp["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = tmp["target"].to_numpy() / caps
    evm = (tmp["target"].to_numpy() >= caps * 0.10) & tmp["avail_strict"].notna().to_numpy()
    full = evm & (tmp["avail_strict"].to_numpy() >= 0.999)
    lowa = evm & (tmp["avail_strict"].to_numpy() < 0.9)
    print("\n--- 기전 (보정 전 원시 CF, signed err) ---")
    print(f"{'모델':<10}{'만가용 signed':>14}{'저가용(<0.9) signed':>20}{'전체 MAE':>10}")
    for c in ["cf_cln03", "cf_rsc", "cf_rscd", "dl_cln03", "dl_rsc", "dl_rscd"]:
        if c not in tmp.columns:
            continue
        e = tmp[c].to_numpy() - acf
        print(f"{c:<10}{e[full].mean():>+14.4f}{e[lowa].mean():>+20.4f}{np.abs(e[evm]).mean():>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-053 라벨 재구성 — 변형 {VARIANTS}, floor {AV_FLOOR} ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    lab = train_df["target"].notna()
    a = train_df["avail_strict"].fillna(1.0)
    n_rs = (lab & (a < 0.999) & (a >= AV_FLOOR)).sum()
    n_lo = (lab & (a < AV_FLOOR)).sum()
    print(f"학습행 {lab.sum()} 중 복원 대상 {n_rs} ({n_rs/lab.sum()*100:.1f}%) / "
          f"floor 미만(×0.05 유지) {n_lo} ({n_lo/lab.sum()*100:.1f}%)", flush=True)

    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"]), "exp-049 GBM 캐시 필요"
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"]), "exp-049 DL 캐시 필요"

    # GBM
    have_g = merge_cached_cols(cache, GBM_RSC_CACHE, [f"cf_{v}" for v in VARIANTS])
    for v in VARIANTS:
        col = f"cf_{v}"
        if col in have_g:
            print(f"[gbm:{v}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(train_gbm_fold_rsc(train_df, vcols, fold, args.seeds, args.jobs, v))
            print(f"  [gbm:{v}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), f"{v} 정렬 실패"
        cache[col] = m[col].to_numpy()
        have_g.append(col)
        save_cols(cache, GBM_RSC_CACHE, have_g)
        print(f"  캐시 저장 → {GBM_RSC_CACHE.name}", flush=True)

    # DL
    have_d = merge_cached_cols(cache, DL_RSC_CACHE, [f"dl_{v}" for v in VARIANTS])
    for v in VARIANTS:
        col = f"dl_{v}"
        if col in have_d:
            print(f"[dl:{v}] 캐시 재사용", flush=True)
            continue
        out = np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys)
            out[sel] = dl_fold_predict_rsc(tr_all, va, vcols, args.seeds, v)
            print(f"  [dl:{v}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        have_d.append(col)
        save_cols(cache, DL_RSC_CACHE, have_d)
        print(f"  캐시 저장 → {DL_RSC_CACHE.name}", flush=True)

    evaluate_grid(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
