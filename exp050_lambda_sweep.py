# ═══════════════════════════════════════════════════════
# exp-050: 가용성 다운웨이트 강도 λ 스윕 (exp-049 후속, MODEL.md "남은 조정 여지")
#
# exp-049는 "터빈 정지 시간 sample weight ×0.05" 단일 극단만 검증했다.
#   - 게이트 ΔCV +0.0053 / Δhold +0.0055 (cln×cln_w70) → LB +0.00058 (~11% 전이)
#   - LB에서 1-NMAE −0.0027 반전 · FiCR +0.0039 — 정제가 2025 실제 정지 시간에 과대예측.
# → λ ∈ {0.15, 0.30, 0.50} 부분 정제로 NMAE 비용·FiCR 이득 사이 내부 최적점 탐색.
#
# 변경점 격리: exp-049와 동일하게 "학습 weight × 가용성 팩터" 하나만 다름.
#   피처(spd 348)·모델(LGBM/XGB 50:50, q60 MTMLP)·신PP(c16·al40·sn93·pt27)·챔피언 편향맵 고정.
# 평가: 4-fold CV + 1년 홀드아웃 이중 게이트.
#   ref     = gspd×q60_w70 (dirty, exp-043 구성. 재현 기대 CV 0.6481 / HOLD 0.6463)
#   champ49 = cln03×cln03_w70 (λ0.05, 현 챔피언. 재현 기대 CV 0.6534 / HOLD 0.6518)
# 판정: champ49 대비 3중 조건 (양 게이트 +0.001 / fold 일관 / 두 지표 동반 상승).
# 그리드: GBM λ {1.0, 0.50, 0.30, 0.15, 0.05} × DL λ {동일 5종} × w {0.70, 0.65}
#   — 대각(동일 λ 쌍)이 사전등록 주 비교축, 비대각은 캐시 공짜 탐색(다중비교 유의).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp050_lambda_sweep.py --jobs 8 --seeds 3
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LAMS = {"l50": 0.50, "l30": 0.30, "l15": 0.15}   # 신규 학습 대상
GBM_LAM_CACHE = CACHE_DIR / "exp050_gbm_lam.parquet"
DL_LAM_CACHE = CACHE_DIR / "exp050_dl_lam.parquet"


def lam_factor(av_vals, lam):
    """가용률 <1.0(터빈 1기+ 정지) 시간의 weight 배율 = λ. 결측(SCADA 없음)은 1.0."""
    a = np.nan_to_num(np.asarray(av_vals, dtype=float), nan=1.0)
    return np.where(a < 0.999, lam, 1.0)


# ── GBM 학습 (exp049_clean.train_gbm_fold_clean에서 scheme→λ만 교체) ──
def train_gbm_fold_lam(train_df, cols, fold, seeds, jobs, lam, tag):
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
        y_tr = tr_g["target"]
        w_tr = make_sample_weight(y_tr.to_numpy(), cap, low_weight=0.05)
        w_tr = w_tr * lam_factor(tr_g["avail_strict"].to_numpy(), lam)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, jobs, s))
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                      n_jobs=jobs, verbosity=0, objective="reg:absoluteerror")
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  f"cf_{tag}": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


# ── DL 학습 (exp049_clean.make_xy_clean/dl_fold_predict_clean에서 λ만 교체) ──
def make_xy_lam(df, cols, imputer, scaler, lam, fit=False):
    from exp033_dl import GIDX
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    w = (w * lam_factor(sub["avail_strict"].to_numpy(), lam)).astype(np.float32)
    return Xs.astype(np.float32), g, cf, w


def dl_fold_predict_lam(tr_df, va_df, cols, seeds, lam):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_lam(tr_df, cols, imp, sca, lam, fit=True)
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


# ── 캐시 유틸 ──
def merge_cached_cols(cache, path, wanted):
    """path parquet에서 wanted 중 존재·완전한 컬럼을 cache에 붙이고 확보 목록 반환"""
    got = []
    if path.exists():
        prev = pd.read_parquet(path)
        for c in wanted:
            if c in prev.columns:
                m = cache[KEY].merge(prev[KEY + [c]], on=KEY, how="left")
                if m[c].notna().all():
                    cache[c] = m[c].to_numpy()
                    got.append(c)
    return got


def save_cols(cache, path, cols):
    cache[KEY + cols].to_parquet(path, index=False)


# ── 평가 ──
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

    gbm_map = {"g100": "cf_gspd", "g50": "cf_l50", "g30": "cf_l30", "g15": "cf_l15", "g05": "cf_cln03"}
    dl_map = {"d100": "dl_q60", "d50": "dl_l50", "d30": "dl_l30", "d15": "dl_l15", "d05": "dl_cln03"}
    res = {}
    for gn, gc in gbm_map.items():
        res[f"{gn}_w100"] = ev(gc, None, 1.0, PP_NEW)
        for dn, dc in dl_map.items():
            for w in (0.70, 0.65):
                res[f"{gn}×{dn}_w{int(w*100)}"] = ev(gc, dc, w, PP_NEW)

    ch_cv, ch_hd = r_ch["cv"][0], r_ch["hold"][0]
    ch_n, ch_f = r_ch["cv"][1], r_ch["cv"][2]
    diag = {f"g{k}×d{k}_w{w}" for k in ("100", "50", "30", "15", "05") for w in (70, 65)}
    print("\n" + "=" * 116)
    print(f"{'변형':<18}{'CV':>8}{'ΔCVch':>9}{'HOLD':>8}{'Δhdch':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=champ49 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ch["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        tagd = " [대각]" if name in diag else ""
        print(f"{name:<18}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}{tagd}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))
    print(f"\n--- fold별 ΔCV (champ49 대비, 상위 12) ---")
    print(f"{'변형':<18}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"]))[:12]:
        print(f"{r['name']:<18}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # 기전: λ 사다리별 만가용/저가용 signed error (보정 전 원시 CF)
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
    print("\n--- 기전: λ 사다리 signed error (보정 전 원시 CF) ---")
    print(f"{'모델':<10}{'만가용 signed':>14}{'저가용(<0.9) signed':>20}{'전체 MAE':>10}")
    for c in ["cf_gspd", "cf_l50", "cf_l30", "cf_l15", "cf_cln03",
              "dl_q60", "dl_l50", "dl_l30", "dl_l15", "dl_cln03"]:
        if c not in tmp.columns:
            continue
        e = tmp[c].to_numpy() - acf
        print(f"{c:<10}{e[full].mean():>+14.4f}{e[lowa].mean():>+20.4f}{np.abs(e[evm]).mean():>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-dl", action="store_true", help="GBM 스윕만 (진단용)")
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-050 λ 스윕 — λ={list(LAMS.values())} ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    n_lab = train_df["target"].notna().sum()
    n_low = (train_df["target"].notna() & (train_df["avail_strict"].fillna(1.0) < 0.999)).sum()
    print(f"학습행 {n_lab} 중 저가용(<1.0) {n_low} ({n_low/n_lab*100:.1f}%)", flush=True)

    cache, champ_bias = load_eval_frame()
    got49g = merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    got49d = merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert got49g and got49d, "exp-049 캐시(cln03) 필요"

    # GBM λ 스윕
    gbm_cols = [f"cf_{t}" for t in LAMS]
    have_g = merge_cached_cols(cache, GBM_LAM_CACHE, gbm_cols)
    for tag, lam in LAMS.items():
        col = f"cf_{tag}"
        if col in have_g:
            print(f"[gbm:{tag}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(train_gbm_fold_lam(train_df, vcols, fold, args.seeds, args.jobs, lam, tag))
            print(f"  [gbm:{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), f"{tag} 정렬 실패"
        cache[col] = m[col].to_numpy()
        have_g.append(col)
        save_cols(cache, GBM_LAM_CACHE, have_g)
        print(f"  캐시 저장 → {GBM_LAM_CACHE.name} ({len(have_g)}/{len(gbm_cols)})", flush=True)

    # DL λ 스윕
    if not args.skip_dl:
        dl_cols = [f"dl_{t}" for t in LAMS]
        have_d = merge_cached_cols(cache, DL_LAM_CACHE, dl_cols)
        for tag, lam in LAMS.items():
            col = f"dl_{tag}"
            if col in have_d:
                print(f"[dl:{tag}] 캐시 재사용", flush=True)
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
                out[sel] = dl_fold_predict_lam(tr_all, va, vcols, args.seeds, lam)
                print(f"  [dl:{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
            cache[col] = out
            have_d.append(col)
            save_cols(cache, DL_LAM_CACHE, have_d)
            print(f"  캐시 저장 → {DL_LAM_CACHE.name} ({len(have_d)}/{len(dl_cols)})", flush=True)

    evaluate_grid(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
