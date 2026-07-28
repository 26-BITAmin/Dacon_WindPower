# ═══════════════════════════════════════════════════════
# exp-052: 정격 스냅 분류기(pA) 가용성 정제 + dirty·clean DL 혼합 (캐시 무료)
#
# 근거:
#   - 현행 pA는 pooled LGBM이 yA=(target ≥ 0.9cap)를 **가중치 없이 dirty 라벨**로 학습.
#     정지 시간은 실측이 정격에 못 미치므로 음성 예제로 들어가 "이 날씨 → 가끔 비정격"을
#     학습 = exp-049의 스모킹 건(corr(signed_err, avail) −0.228)과 동일 기전이 FiCR 직결
#     장치에 잔존. exp-041(피처 추가·AUC 포화)과 달리 이번엔 **학습 분포**를 바꿈.
#   - 개입: 분류기 sample weight에 가용성 팩터(정지 시간 ×λ) 적용. λ ∈ {0.05, 0.30}.
#   - 캘리브레이션 시프트(전반적 p 상향)와 판별 개선을 분리하기 위해 snap_thr {0.27,
#     0.30, 0.33} 교차 (0.27이 champ49 값).
#   - B블록 (재학습 0): DL 혼합 0.5·q60(dirty)+0.5·cln03 등 — exp-050 그리드는 λ 재학습
#     중간값만 봤고 두 극단의 **예측 평균**은 미탐색 (exp-044 B는 dirty 조건부).
#
# 고정: cln×cln GBM·DL(λ0.05)·w0.70·신PP·챔피언 편향맵 = champ49 그대로. pA만 교체.
# 판정: champ49 대비 3중 조건. 실행: C:\miniconda3\python.exe -X utf8 exp052_snap_clf_clean.py
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp009_rated_snap import RATED_CF, N_CLF_SEEDS
from exp011_g3_transfer import build_pooled_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (AVAIL_PARQUET, GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY,
                          PP_CHAMP, PP_NEW, merge_avail, postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import lam_factor, merge_cached_cols, save_cols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CLF_LAMS = {"p05": 0.05, "p30": 0.30}
PA_CACHE = CACHE_DIR / "exp052_pa_clean.parquet"
THRS = [0.27, 0.30, 0.33]


def train_pa_fold(train_df, cols, fold, jobs, lam, tag):
    """cache_gbm의 분류기 블록 재현 + 가용성 weight만 추가."""
    import lightgbm as lgb
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    tr_pool = tr_all.loc[tr_all["target"].notna()]
    rows, clf = [], []
    for group in [1, 2, 3]:
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(va_g) == 0:
            continue
        X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
        if not clf:
            cp = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
            yA = (tr_pool["target"].to_numpy() >= cp * RATED_CF).astype(int)
            wA = lam_factor(tr_pool["avail_strict"].to_numpy(), lam)
            for s in range(N_CLF_SEEDS):
                c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                       random_state=s, n_jobs=jobs, verbose=-1)
                c.fit(X_pool_tr, yA, sample_weight=wA)
                clf.append(c)
        pa = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  f"pA_{tag}": pa}))
    return pd.concat(rows, ignore_index=True)


def evaluate(cache, champ_bias):
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def ev(gbm_col, dl_col, w, pp, pa_col="pA"):
        blend = cache[gbm_col].to_numpy() if (dl_col is None or w >= 1.0) else \
            w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub[pa_col].to_numpy(), group, champ_bias[fname], **pp)
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

    res = {}
    # A블록: pA 변형 × snap_thr
    for pa_tag, pa_col in [("pAch", "pA"), ("p05", "pA_p05"), ("p30", "pA_p30")]:
        for thr in THRS:
            pp = dict(PP_NEW, snap_thr=thr)
            res[f"A_{pa_tag}_t{int(thr*100)}"] = ev("cf_cln03", "dl_cln03", 0.70, pp, pa_col)
    # B블록: dirty·clean DL 예측 혼합 (champ pA·신PP)
    for mix_tag, a in [("m25", 0.25), ("m50", 0.50), ("m75", 0.75)]:
        cache[f"dl_{mix_tag}"] = a * cache["dl_q60"].to_numpy() + (1 - a) * cache["dl_cln03"].to_numpy()
        for w in (0.70, 0.65):
            res[f"B_{mix_tag}_w{int(w*100)}"] = ev("cf_cln03", f"dl_{mix_tag}", w, PP_NEW)

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
    print(f"\n--- fold별 ΔCV (champ49 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"])):
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # 진단: 분류기 품질 (라벨 있는 행)
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    tmp = cache.copy()
    tmp["forecast_kst_dtm"] = pd.to_datetime(tmp["forecast_kst_dtm"])
    tmp = tmp.merge(av[["forecast_kst_dtm", "group", "avail_strict"]], on=["forecast_kst_dtm", "group"], how="left")
    caps = tmp["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    lab = tmp["target"].notna().to_numpy()
    yA = (tmp["target"].to_numpy() >= caps * RATED_CF)
    full = lab & (tmp["avail_strict"].fillna(1.0).to_numpy() >= 0.999)
    from sklearn.metrics import roc_auc_score
    print("\n--- 분류기 진단 (라벨 있는 행, thr는 발동 기준) ---")
    print(f"{'pA':<8}{'AUC':>8}{'AUC(만가용)':>12} | thr=0.27: {'발동':>6}{'정밀':>7}{'재현':>7}{'재현(만가용)':>13}")
    for col in ["pA", "pA_p05", "pA_p30"]:
        p = tmp[col].to_numpy()
        auc = roc_auc_score(yA[lab], p[lab])
        auc_f = roc_auc_score(yA[full], p[full])
        act = (p >= 0.27) & lab
        prec = yA[act].mean() if act.any() else float("nan")
        rec = (act & yA)[lab].sum() / max(yA[lab].sum(), 1)
        rec_f = (act & yA)[full].sum() / max(yA[full].sum(), 1)
        print(f"{col:<8}{auc:>8.4f}{auc_f:>12.4f} | {act.sum():>10}{prec:>7.3f}{rec:>7.3f}{rec_f:>13.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-052 스냅 분류기 정제 — λ={list(CLF_LAMS.values())} ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)

    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"]), "exp-049 GBM 캐시 필요"
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"]), "exp-049 DL 캐시 필요"

    pa_cols = [f"pA_{t}" for t in CLF_LAMS]
    have = merge_cached_cols(cache, PA_CACHE, pa_cols)
    for tag, lam in CLF_LAMS.items():
        col = f"pA_{tag}"
        if col in have:
            print(f"[clf:{tag}] 캐시 재사용", flush=True)
            continue
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(train_pa_fold(train_df, cols, fold, args.jobs, lam, tag))
            print(f"  [clf:{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m[col].notna().all(), f"{tag} 정렬 실패"
        cache[col] = m[col].to_numpy()
        have.append(col)
        save_cols(cache, PA_CACHE, have)
        print(f"  캐시 저장 → {PA_CACHE.name} ({len(have)}/{len(pa_cols)})", flush=True)

    evaluate(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
