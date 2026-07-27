# ═══════════════════════════════════════════════════════
# exp-041: 정격 스냅 분류기에 격자 편차(spd) 투입 — 후처리 장치 쪽 정보 주입
#
# 배경: 후처리의 정격 스냅(tier A)은 pooled LGBM 분류기로 P(actual ≥ 0.90cap) ≥ 0.30인
#   시간을 max(pred, 0.94cap)로 올린다. exp-027의 snap_off가 −0.0063(후처리 중 최대 단일 레버)
#   이므로 이 분류기의 정확도는 base 정확도만큼 값이 크다. 그런데 **분류기는 아직 132피처**만 본다.
#   exp-030: CF 0.9+ 구간은 98% 과소예측 → 이 밴드의 재현율은 FiCR로 거의 직결.
#   exp-035에서 spd 편차가 실제 신호를 갖는 것이 실증됐고(DL LB +0.00508), 분류기는 미투입.
#
# 변경점 분리: base 블렌드(gbm132 + DLspd, w0.70)·clip·α·스냅 임계/목표 전부 챔피언 고정.
#   **pA(정격 확률) 벡터만** 교체한다.
#
# 변형:
#   p132 : 챔피언 (exp-035 캐시의 pA — 재학습 불요, 결정적)
#   p348 : pooled LGBM 분류기, 132 + spd 편차 216
#   pdl  : 멀티태스크 MLP 분류기(BCE), 132 + spd 216 — 격자 구조 추출은 NN이 낫다는
#          exp-035 관찰(spg 실패·spd 성공)의 분류기 버전
#   pmix : p132·p348·pdl 평균 (확률 앙상블)
#
# 판정: 이중 게이트(4-fold CV + 1년 홀드아웃), 기준 = 챔피언 pA. 분류기 품질(AUC·임계 0.30
#   에서의 정밀/재현·발동수)도 별개 지표로 병기.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp041_snap_clf_spd.py --variants p348,pdl
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH
from exp009_rated_snap import RATED_CF, N_CLF_SEEDS
from exp011_g3_transfer import build_pooled_matrices
from exp019_multigrid import HOLD_FOLD
from exp033_dl import MTMLP, GIDX, DEVICE
from exp035_raw_dl import (build_inputs, load_gbm_cache, postprocess, score_from_tidy,
                           FOLD_LIST, DL_CFG, CACHE_DIR)
from exp039_dl_diverse import DL_PRED_CACHE, build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PA_CACHE = CACHE_DIR / "exp041_pa.parquet"
SNAP_THR, DL_COL, W_GBM = 0.30, "dl_hub", 0.70


# ───────────────────────────────────────────────
# 분류기
# ───────────────────────────────────────────────
def clf_lgb(tr_pool, va_g_list, cols, jobs):
    """pooled LGBM 분류기 (챔피언과 동일 하이퍼파라미터, 피처만 확장)"""
    import lightgbm as lgb
    caps = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    y = (tr_pool["target"].to_numpy() >= caps * RATED_CF).astype(int)
    models, Xva_list = [], []
    for va_g in va_g_list:
        X_tr, X_va = build_pooled_matrices(tr_pool, va_g, cols)
        Xva_list.append(X_va)
        if not models:
            for s in range(N_CLF_SEEDS):
                m = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                                       min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                       random_state=s, n_jobs=jobs, verbose=-1)
                m.fit(X_tr, y); models.append(m)
    return [np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0) for X in Xva_list]


def clf_dl(tr_pool, va_g_list, cols, seeds):
    """멀티태스크 MLP 분류기 (BCE). 회귀 DL과 동일 아키텍처·에폭."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr = sca.fit_transform(imp.fit_transform(tr_pool[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gtr = tr_pool["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    caps = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    ytr = (tr_pool["target"].to_numpy() >= caps * RATED_CF).astype(np.float32)
    cfg, bs = DL_CFG, 1024
    outs = [np.zeros(len(v)) for v in va_g_list]
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        model = MTMLP(Xtr.shape[1], cfg["width"], cfg["depth"], cfg["p_drop"]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.tensor(Xtr, device=DEVICE); gt = torch.tensor(gtr, device=DEVICE)
        yt = torch.tensor(ytr, device=DEVICE)
        n = Xt.shape[0]
        for _ in range(cfg["epochs"]):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(model(Xt[idx], gt[idx]), yt[idx])
                loss.backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            for j, va in enumerate(va_g_list):
                Xv = sca.transform(imp.transform(va[cols].to_numpy(dtype=np.float32))).astype(np.float32)
                gv = va["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
                logits = []
                for i in range(0, len(Xv), 8192):
                    logits.append(model(torch.tensor(Xv[i:i+8192], device=DEVICE),
                                        torch.tensor(gv[i:i+8192], device=DEVICE)).cpu().numpy())
                outs[j] += 1.0 / seeds * (1.0 / (1.0 + np.exp(-np.concatenate(logits))))
        del model
        torch.cuda.empty_cache()
    return outs


def run(args):
    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    champ, bias_maps = load_gbm_cache()
    folds = FOLD_LIST if not args.quick else [FOLD_LIST[3], FOLD_LIST[4]]
    fold_names = [fn for fn, _ in folds]
    cache = champ[champ["fold"].isin(fold_names)].reset_index(drop=True).rename(columns={"pA": "pa_p132"})
    key = ["fold", "group", "forecast_kst_dtm"]

    dl = pd.read_parquet(DL_PRED_CACHE)
    m = cache[key].merge(dl[key + [DL_COL]], on=key, how="left")
    assert m[DL_COL].notna().all(), "DL 예측 캐시에 정렬 실패 — exp-039를 먼저 실행"
    cache[DL_COL] = m[DL_COL].to_numpy()
    print(f"입력 spd={len(vcols)}개 | 캐시 {cache.shape}", flush=True)

    if PA_CACHE.exists() and not args.refresh and not args.quick:
        prev = pd.read_parquet(PA_CACHE)
        for c in [c for c in prev.columns if c.startswith("pa_")]:
            mm = cache[key].merge(prev[key + [c]], on=key, how="left")
            if mm[c].notna().all():
                cache[c] = mm[c].to_numpy()
        print(f"pA 캐시 로드: {[c for c in cache.columns if c.startswith('pa_')]}", flush=True)

    for v in args.variants:
        if f"pa_{v}" in cache.columns:
            continue
        out = np.full(len(cache), np.nan)
        for fname, fold in folds:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            tr_pool = tr_all.loc[tr_all["target"].notna()]
            sel_masks, va_list, groups = [], [], []
            for g in [1, 2, 3]:
                sel = ((cache["fold"] == fname) & (cache["group"] == g)).to_numpy()
                if not sel.any():
                    continue
                keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
                va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                                right_on=["kpx_group", "forecast_kst_dtm"], how="left")
                assert len(va) == len(keys), f"{v}/{fname}/g{g} 정렬 실패"
                sel_masks.append(sel); va_list.append(va); groups.append(g)
            probs = (clf_lgb(tr_pool, va_list, vcols, args.jobs) if v == "p348"
                     else clf_dl(tr_pool, va_list, vcols, args.seeds))
            for sel, p in zip(sel_masks, probs):
                out[sel] = p
            print(f"  [clf:{v}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[f"pa_{v}"] = out
        if not args.quick:
            cache[key + [c for c in cache.columns if c.startswith("pa_") and c != "pa_p132"]] \
                .to_parquet(PA_CACHE, index=False)

    if "pmix" in args.mixes:
        members = [f"pa_{v}" for v in ["p132"] + list(args.variants)]
        cache["pa_pmix"] = np.mean([cache[c].to_numpy() for c in members], axis=0)

    # ── 평가 ──
    cv_folds = [fn for fn in fold_names if fn != HOLD_FOLD["name"]]
    slices = build_slices(cache)
    hold = HOLD_FOLD["name"]
    pa_cols = [c for c in cache.columns if c.startswith("pa_")]

    def score(pa_col):
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            cf = W_GBM * sub["gbm_cf"].to_numpy() + (1 - W_GBM) * sub[DL_COL].to_numpy()
            pred = postprocess(cf, cap, sub["ws"].to_numpy(), sub[pa_col].to_numpy(), group, bias_maps[fname])
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    res = {c[3:]: score(c) for c in pa_cols}
    b = res["p132"]
    b_cv, b_hd = b["cv"][0], b["hold"][0]
    print("\n" + "=" * 84)
    print(f"기준 챔피언 pA(p132): CV {b_cv:.4f} (1-NMAE {b['cv'][1]:.4f} FiCR {b['cv'][2]:.4f}) | HOLD {b_hd:.4f}"
          f"   ※ exp-035 spd_w70 = 0.6448/0.6434")
    print("=" * 84)
    print(f"{'분류기':<10}{'CV':>8}{'ΔCV':>9}{'HOLD':>8}{'Δhold':>9}{'1-NMAE':>9}{'FiCR':>8}  게이트")
    for name, r in res.items():
        dcv, dhd = r["cv"][0] - b_cv, r["hold"][0] - b_hd
        print(f"{name:<10}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{r['cv'][2]:>8.4f}  " + ("통과" if (dcv > 0 and dhd > 0) else ""))

    # ── 분류기 품질 (임계 0.30, 평가대상 무관 전 시각) ──
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    ytrue = (cache["target"].to_numpy() >= caps * RATED_CF)
    ok = np.isfinite(cache["target"].to_numpy())
    print("\n--- 정격 분류기 품질 (라벨 있는 시각, 임계 0.30) ---")
    rows = []
    for c in pa_cols:
        p = cache[c].to_numpy()
        fire = (p >= SNAP_THR) & ok
        tp = (fire & ytrue).sum()
        rows.append({"clf": c[3:], "발동": int(fire.sum()), "정밀": round(tp / max(fire.sum(), 1), 3),
                     "재현": round(tp / max((ytrue & ok).sum(), 1), 3),
                     "AUC": round(float(_auc(ytrue[ok], p[ok])), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- fold별 ΔCV (기준 p132 대비) ---")
    print(f"{'분류기':<10}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds))
    for name, r in res.items():
        if name == "p132":
            continue
        print(f"{name:<10}" + "".join(f"{r['folds'][fn]-b['folds'][fn]:>+10.4f}" for fn in cv_folds))
    return res


def _auc(y, p):
    o = np.argsort(p)
    r = np.empty(len(p), float); r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="p348,pdl")
    ap.add_argument("--mixes", type=str, default="pmix")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    args.mixes = [v for v in args.mixes.split(",") if v]
    if args.quick:
        args.seeds = 1
        print("### QUICK 배관 점검 모드 ###")
    t0 = time.time()
    print(f"### exp-041 (분류기={args.variants}, seeds={args.seeds}, device={DEVICE}) ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
