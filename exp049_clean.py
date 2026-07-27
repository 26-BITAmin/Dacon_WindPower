# ═══════════════════════════════════════════════════════
# exp-049 본실험: SCADA 가용성 기반 학습 라벨 정제 (sample weight 다운웨이트)
#
# 진단(exp049_diag) 결과:
#   - 라벨 = 날씨발전량 × 가용률 (ws 12~14에서 avail 1.0 CF 0.846 / 0.7~0.9 0.703 / <0.7 0.399)
#   - corr(signed_err, avail) = −0.228: avail<0.7 시간 +0.286 과대예측, 만가용(87%) −0.015 과소예측
#   → 저가용 시간이 학습을 오염시켜 "만가용 발전량"을 체계적으로 낮게 배움.
#
# 개입: 학습 sample weight에 가용률 факт만 곱함 (다른 모든 것 동일 = 변경점 격리).
#   cln03 (soft): avail<0.9 → ×0.05 / 0.9~<1.0 → ×0.3 / 1.0 → ×1.0
#   cln00 (hard): avail<1.0 → ×0.05
#   avail 결측(SCADA 없음)은 ×1.0 (증거 없으면 건드리지 않음).
#
# 스테이지:
#   --stage gbm : 정제 GBM(gspd 348, 3-seed) 학습·캐시 → GBM 단독 + 기존 q60 DL과 블렌드 평가
#   --stage dl  : (승자 스킴) 정제 q60 DL(3-seed) 학습·캐시 → 완전 정제 블렌드 평가
#
# 평가: 신PP(c16·al40·sn93·pt27) + 챔피언 편향맵(cB — exp-040/043에서 전 칸 우세) 고정.
#   ref = gspd(dirty)·q60(dirty)·w0.70·신PP (재현 기대 CV 0.6481 / HOLD 0.6463).
#   ⚠ 편향맵이 dirty 모델 기준이라 clean 모델에 불리한 보수적 비교임을 명시.
# 판정: 3중 조건 (양 게이트 +0.001 / fold 일관 / 두 지표 동반 상승).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp049_clean.py --stage gbm
#       C:\miniconda3\python.exe -X utf8 exp049_clean.py --stage dl --scheme cln03
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import (CAPACITY_KWH, DATA_DIR, make_sample_weight, lgbm_params,
                         prepare_matrices, apply_rolling_smoothing, apply_bias_correction_v2)
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import (build_inputs, load_gbm_cache, score_from_tidy,
                           FOLD_LIST, DL_CFG, CACHE_DIR)
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE, build_slices, make_loss, train_one
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
AVAIL_PARQUET = DATA_DIR / "external" / "exp049_avail.parquet"
GBM_CLEAN_CACHE = CACHE_DIR / "exp049_gbm_clean.parquet"
DL_CLEAN_CACHE = CACHE_DIR / "exp049_dl_clean.parquet"

PP_CHAMP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)
PP_NEW   = dict(clip_lo=0.16, alpha=0.40, snap_thr=0.27, snap_to=0.93)
SCHEMES = ["cln03", "cln00"]


def avail_factor(av_vals, scheme):
    a = np.nan_to_num(av_vals.astype(float), nan=1.0)
    if scheme == "cln03":
        return np.where(a < 0.9, 0.05, np.where(a < 0.999, 0.3, 1.0))
    if scheme == "cln00":
        return np.where(a < 0.999, 0.05, 1.0)
    raise ValueError(scheme)


def merge_avail(df):
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    out = df.merge(av[["forecast_kst_dtm", "group", "avail_strict"]]
                   .rename(columns={"group": "kpx_group"}),
                   on=["forecast_kst_dtm", "kpx_group"], how="left")
    assert len(out) == len(df)
    return out


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


# ── GBM 스테이지 ──
def train_gbm_fold_clean(train_df, cols, fold, seeds, jobs, scheme):
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
        w_tr = w_tr * avail_factor(tr_g["avail_strict"].to_numpy(), scheme)
        pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, jobs, s))
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                      n_jobs=jobs, verbosity=0, objective="reg:absoluteerror")
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  f"cf_{scheme}": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


# ── DL 스테이지 ──
def make_xy_clean(df, cols, imputer, scaler, scheme, fit=False):
    from exp033_dl import GIDX
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    w = (w * avail_factor(sub["avail_strict"].to_numpy(), scheme)).astype(np.float32)
    return Xs.astype(np.float32), g, cf, w


def dl_fold_predict_clean(tr_df, va_df, cols, seeds, scheme):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(tr_df, cols, imp, sca, scheme, fit=True)
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


def load_eval_frame():
    cache, champ_bias = load_gbm_cache()
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})
    ext = pd.read_parquet(EXT_GBM_CACHE)
    m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
    assert m["cf_gspd"].notna().all()
    cache["cf_gspd"] = m["cf_gspd"].to_numpy()
    dl = pd.read_parquet(E39_CACHE)
    for c in ["dl_q60", "dl_hub"]:
        mm = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
        assert mm[c].notna().all()
        cache[c] = mm[c].to_numpy()
    return cache, champ_bias


def evaluate_all(cache, champ_bias, gbm_cols, dl_cols, ws, ref_pair):
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

    r_champ = ev("cf_g132", "dl_hub", 0.70, PP_CHAMP)
    r_ref = ev(ref_pair[0], ref_pair[1], 0.70, PP_NEW)
    res = {}
    for gc in gbm_cols:
        res[f"{gc[3:]}_w100"] = ev(gc, None, 1.0, PP_NEW)
        for dc in dl_cols:
            for w in ws:
                res[f"{gc[3:]}×{dc[3:]}_w{int(w*100)}"] = ev(gc, dc, w, PP_NEW)

    c_cv, c_hd = r_champ["cv"][0], r_champ["hold"][0]
    ref_cv, ref_hd = r_ref["cv"][0], r_ref["hold"][0]
    ref_n, ref_f = r_ref["cv"][1], r_ref["cv"][2]
    print(f"\n챔피언(g132·hub·w70·구PP): CV {c_cv:.4f} | HOLD {c_hd:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):   CV {ref_cv:.4f} (1-NMAE {ref_n:.4f} FiCR {ref_f:.4f}) | "
          f"HOLD {ref_hd:.4f}  (기대 0.6481/0.6463)")
    print("\n" + "=" * 108)
    print(f"{'변형':<20}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ref_cv, kv[1]["hold"][0] - ref_hd)):
        dcv, dhd = r["cv"][0] - ref_cv, r["hold"][0] - ref_hd
        dn, df_ = r["cv"][1] - ref_n, r["cv"][2] - ref_f
        fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<20}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))
    print(f"\n--- fold별 ΔCV (ref 대비) ---")
    print(f"{'변형':<20}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"]))[:10]:
        print(f"{r['name']:<20}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # 기전 확인: 만가용 시간 signed error가 줄었는가
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
    print("\n--- 기전 확인 (보정 전 원시 CF, signed err) ---")
    print(f"{'모델':<12}{'만가용 signed':>14}{'저가용(<0.9) signed':>20}{'전체 MAE':>10}")
    for c in gbm_cols + dl_cols:
        e = tmp[c].to_numpy() - acf
        print(f"{c:<12}{e[full].mean():>+14.4f}{e[lowa].mean():>+20.4f}{np.abs(e[evm]).mean():>10.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gbm", "dl"], default="gbm")
    ap.add_argument("--scheme", type=str, default="cln03", help="dl 스테이지용 승자 스킴")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-049 정제 학습 — stage={args.stage} ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    n_lab = train_df["target"].notna().sum()
    n_low = (train_df["target"].notna() & (train_df["avail_strict"].fillna(1.0) < 0.999)).sum()
    print(f"학습행 {n_lab} 중 저가용(<1.0) {n_low} ({n_low/n_lab*100:.1f}%)", flush=True)

    cache, champ_bias = load_eval_frame()

    if args.stage == "gbm":
        have = {}
        if GBM_CLEAN_CACHE.exists() and not args.refresh:
            prev = pd.read_parquet(GBM_CLEAN_CACHE)
            for c in [c for c in prev.columns if c.startswith("cf_")]:
                m = cache[KEY].merge(prev[KEY + [c]], on=KEY, how="left")
                if m[c].notna().all():
                    have[c] = m[c].to_numpy()
        for scheme in SCHEMES:
            col = f"cf_{scheme}"
            if col in have:
                cache[col] = have[col]
                continue
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                p = train_gbm_fold_clean(train_df, vcols, fold, args.seeds, args.jobs, scheme)
                preds.append(p)
                print(f"  [gbm:{scheme}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"{scheme} 정렬 실패"
            cache[col] = m[col].to_numpy()
            have[col] = cache[col].to_numpy()
            cache[KEY + list(have)].to_parquet(GBM_CLEAN_CACHE, index=False)
            print(f"  캐시 저장 → {GBM_CLEAN_CACHE.name}", flush=True)
        evaluate_all(cache, champ_bias,
                     gbm_cols=["cf_gspd", "cf_cln03", "cf_cln00"],
                     dl_cols=["dl_q60"], ws=[0.70, 0.65],
                     ref_pair=("cf_gspd", "dl_q60"))

    else:  # dl 스테이지
        prev = pd.read_parquet(GBM_CLEAN_CACHE)
        for c in [c for c in prev.columns if c.startswith("cf_")]:
            m = cache[KEY].merge(prev[KEY + [c]], on=KEY, how="left")
            assert m[c].notna().all()
            cache[c] = m[c].to_numpy()
        col = f"dl_{args.scheme}"
        done = False
        if DL_CLEAN_CACHE.exists() and not args.refresh:
            prevd = pd.read_parquet(DL_CLEAN_CACHE)
            if col in prevd.columns:
                m = cache[KEY].merge(prevd[KEY + [col]], on=KEY, how="left")
                if m[col].notna().all():
                    cache[col] = m[col].to_numpy(); done = True
        if not done:
            out = np.full(len(cache), np.nan)
            for fname, fold in FOLD_LIST:
                tf = time.time()
                tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
                sel = (cache["fold"] == fname).to_numpy()
                keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
                va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                                right_on=["kpx_group", "forecast_kst_dtm"], how="left")
                assert len(va) == len(keys)
                out[sel] = dl_fold_predict_clean(tr_all, va, vcols, args.seeds, args.scheme)
                print(f"  [dl:{args.scheme}:{fname}] {time.time()-tf:.0f}s", flush=True)
            cache[col] = out
            cache[KEY + [col]].to_parquet(DL_CLEAN_CACHE, index=False)
            print(f"  캐시 저장 → {DL_CLEAN_CACHE.name}", flush=True)
        evaluate_all(cache, champ_bias,
                     gbm_cols=["cf_gspd", f"cf_{args.scheme}"],
                     dl_cols=["dl_q60", col], ws=[0.70, 0.65],
                     ref_pair=("cf_gspd", "dl_q60"))

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
