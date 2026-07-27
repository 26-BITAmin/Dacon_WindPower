# ═══════════════════════════════════════════════════════
# exp-039: DL 학습기 다양성 (손실·가중 = 추정량 교체) — 탈상관 축 재개
#
# 배경: 블렌드 이득의 기전은 **오차 탈상관**이고 exp-035에서 별개 지표로 실증됨
#   (corr 0.851→0.818, 세 변형의 corr 순서 = 점수 순서). 그런데 입력 축은 소진:
#   exp-037(GFS 변수 확장) corr 0.8184→0.8169(무변화), exp-038(JMA 다격자) 0.8254(상승).
#   한편 지금까지 만든 **모든 DL은 동일 추정량** — MTMLP + weighted Huber(δ0.05) on CF.
#   δ0.05는 전형 오차(CF 0.12) 대비 작아 사실상 L1이고, GBM도 MAE 목적(lgb mae /
#   xgb reg:absoluteerror) → 둘 다 **조건부 중앙값 근처**를 추정한다. exp-034의 DL
#   앙상블(384⊕512)이 sub-노이즈였던 것도 용량만 달라 추정량이 같았기 때문.
#   → 미탐색 축: **손실을 바꿔 다른 조건부 통계량을 추정**시키면 오차 구조가 구조적으로
#   달라져 탈상관이 커질 수 있다. 입력·용량·가중 축과 독립.
#
# 변형 (spd 입력 348 고정, MLP 384×3 ep120 = exp-033/035 국소 최적 고정):
#   hub : weighted Huber δ0.05        (= exp-035 spd 재현. 런 무결성 기준)
#   l2  : weighted MSE                 → 조건부 평균 (중앙값과 다른 통계량)
#   aw  : Huber + actual 비례 가중      → FiCR 정렬(고발전 시간 강조). GBM에서는 exp-016
#                                        기각됐으나 DL·블렌드 파트너로는 미시도
#   q60 : 비대칭 Huber (과소예측 ×1.2 / 과대 ×0.8) → 상측 expectile.
#         exp-030의 "CF 0.7~0.9 71% 과소예측"을 blanket nudge(exp-031 LB 역전) 대신
#         **학습된 비대칭**으로 접근하고, 30% 블렌드 지분으로만 반영 = mild
#
# 혼합: DL 파트너를 단일 변형 대신 여러 변형 평균으로도 평가(단·쌍·삼중·전체).
#   변형 간 추정량이 다르므로 exp-034의 동일추정량 앙상블과 성질이 다름. w는 후처리라 무비용.
#
# 비용: GBM base/분류기/편향맵은 exp-035 캐시 재사용 → DL 학습 비용만.
#   DL 예측은 parquet 캐시(변형별) → 후속 블렌드·후처리 실험은 재학습 없이 무비용.
# 판정: 이중 게이트(4-fold CV + 1년 홀드아웃), 기준 hub_w70(=spd_w70) 대비 동일 런 Δ,
#   ±0.001 매그니튜드 (cv-baseline-drift 규칙: 절대 CV 교차비교 금지).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp039_dl_diverse.py
#       C:\miniconda3\python.exe -X utf8 exp039_dl_diverse.py --variants hub,l2 --seeds 1 --quick
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp033_dl import MTMLP, predict, GIDX, DEVICE
from exp035_raw_dl import (build_inputs, load_gbm_cache, postprocess, score_from_tidy,
                           FOLD_LIST, DL_CFG, CACHE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DL_PRED_CACHE = CACHE_DIR / "exp039_dl_preds.parquet"

# 변형 = (손실 종류, 샘플가중 방식). 입력·아키텍처·에폭은 전부 동일.
VARIANTS = {
    "hub": ("hub", "std"),   # exp-035 spd 재현 (기준)
    "l2":  ("l2",  "std"),
    "aw":  ("hub", "act"),
    "q60": ("q60", "std"),
}
REF = "hub"
BLEND_WS = [0.70, 0.65, 0.60]
DELTA = 0.05      # Huber 임계 (exp-033 값)
TAU = 0.60        # q60 비대칭 계수

# 혼합 조합 (추정량이 다른 변형끼리 평균)
MIXES = [("hub", "l2"), ("hub", "aw"), ("hub", "q60"), ("l2", "aw"), ("l2", "q60"),
         ("aw", "q60"), ("hub", "l2", "aw"), ("hub", "l2", "q60"), ("hub", "aw", "q60"),
         ("l2", "aw", "q60"), ("hub", "l2", "aw", "q60")]


# ───────────────────────────────────────────────
# 손실 (exp-033 huber_weighted와 hub는 수식 동일 → 재현 보장)
# ───────────────────────────────────────────────
def _huber(e):
    a = torch.abs(e)
    return torch.where(a <= DELTA, 0.5 * e * e, DELTA * (a - 0.5 * DELTA))


def make_loss(kind):
    if kind == "hub":
        def f(pred, tgt, w):
            return (_huber(pred - tgt) * w).sum() / w.sum()
    elif kind == "l2":
        def f(pred, tgt, w):
            e = pred - tgt
            return (0.5 * e * e * w).sum() / w.sum()
    elif kind == "q60":
        def f(pred, tgt, w):
            e = pred - tgt
            # e<0 = 과소예측 → 2τ=1.2배, 과대예측 → 2(1−τ)=0.8배
            asym = torch.where(e < 0, 2.0 * TAU, 2.0 * (1.0 - TAU))
            return (_huber(e) * asym * w).sum() / w.sum()
    else:
        raise ValueError(kind)
    return f


def sample_weight(cf, mode):
    """std = 현행(저발전 0.05 / 그 외 1.0). act = 평가구간을 actual(CF) 비례로 (스케일 정규화)."""
    if mode == "std":
        return np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    ev = cf >= 0.10
    scale = cf[ev].mean() if ev.any() else 1.0
    return np.where(ev, cf / scale, 0.05).astype(np.float32)


# ───────────────────────────────────────────────
# 학습 (exp-033 train_model과 RNG 소비 순서 동일 — 손실만 인자화)
# ───────────────────────────────────────────────
def make_xy(df, cols, imputer, scaler, wmode, fit=False):
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    return Xs.astype(np.float32), g, cf, sample_weight(cf, wmode)


def train_one(Xtr, gtr, ytr, wtr, cfg, loss_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MTMLP(Xtr.shape[1], cfg["width"], cfg["depth"], cfg["p_drop"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    Xt = torch.tensor(Xtr, device=DEVICE); gt = torch.tensor(gtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE); wt = torch.tensor(wtr, device=DEVICE)
    n, bs = Xt.shape[0], 1024
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward(); opt.step()
        sched.step()
    return model


def dl_fold_predict(tr_df, va_df, cols, seeds, kind, wmode):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy(tr_df, cols, imp, sca, wmode, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss(kind)
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


# ───────────────────────────────────────────────
# 평가
# ───────────────────────────────────────────────
def build_slices(cache):
    """(fold, group)별 시각 정렬 슬라이스 — exp-035와 동일 순서(재현 보장)."""
    return [((fn, int(g)), sub.sort_values("forecast_kst_dtm"))
            for (fn, g), sub in cache.groupby(["fold", "group"], sort=False)]


def evaluate(slices, bias_maps, dl_col, w, cv_folds):
    parts = []
    for (fname, group), sub in slices:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        gbm = sub["gbm_cf"].to_numpy()
        cf = gbm if dl_col is None else w * gbm + (1 - w) * sub[dl_col].to_numpy()
        pred = postprocess(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(), group, bias_maps[fname])
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    hold = HOLD_FOLD["name"]
    cv = score_from_tidy(tidy[tidy["fold"] != hold], "pred")
    hd = score_from_tidy(tidy[tidy["fold"] == hold], "pred")
    folds = {fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds}
    return dict(cv=cv, hold=hd, folds=folds)


def run(args):
    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    cache, bias_maps = load_gbm_cache()
    print(f"입력 spd={len(vcols)}개 / GBM 캐시 {cache.shape}", flush=True)

    folds = FOLD_LIST if not args.quick else [(FOLD_LIST[3][0], FOLD_LIST[3][1]), (FOLD_LIST[4][0], FOLD_LIST[4][1])]
    fold_names = [fn for fn, _ in folds]
    cache = cache[cache["fold"].isin(fold_names)].reset_index(drop=True)

    # ── DL 예측 (변형별 parquet 캐시) ──
    have = {}
    if DL_PRED_CACHE.exists() and not args.refresh:
        prev = pd.read_parquet(DL_PRED_CACHE)
        key = ["fold", "group", "forecast_kst_dtm"]
        for c in [c for c in prev.columns if c.startswith("dl_")]:
            merged = cache[key].merge(prev[key + [c]], on=key, how="left")
            if merged[c].notna().all():
                have[c[3:]] = merged[c].to_numpy()
        if have:
            print(f"DL 예측 캐시 로드: {sorted(have)}", flush=True)

    for v in args.variants:
        if v in have:
            continue
        kind, wmode = VARIANTS[v]
        out = np.full(len(cache), np.nan)
        for fname, fold in folds:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), f"{v}/{fname} 검증행 정렬 실패 ({len(va)} vs {len(keys)})"
            out[sel] = dl_fold_predict(tr_all, va, vcols, args.seeds, kind, wmode)
            print(f"  [dl:{v}({kind}/{wmode}):{fname}] {time.time()-tf:.0f}s", flush=True)
        have[v] = out

    for v in args.variants:
        cache[f"dl_{v}"] = have[v]
    if not args.quick:
        key = ["fold", "group", "forecast_kst_dtm"]
        keep = key + [f"dl_{v}" for v in args.variants]
        cache[keep].to_parquet(DL_PRED_CACHE, index=False)
        print(f"DL 예측 캐시 저장 → {DL_PRED_CACHE.name}", flush=True)

    # ── 혼합 컬럼 ──
    mixes = [m for m in MIXES if set(m) <= set(args.variants)] if not args.quick else []
    for m in mixes:
        cache["dl_" + "+".join(m)] = np.mean([cache[f"dl_{v}"].to_numpy() for v in m], axis=0)

    # ── 점수 ──
    cv_folds = [fn for fn in fold_names if fn != HOLD_FOLD["name"]]
    slices = build_slices(cache)
    results = {"gbm": evaluate(slices, bias_maps, None, 1.0, cv_folds)}
    cand = list(args.variants) + ["+".join(m) for m in mixes]
    for c in cand:
        for w in BLEND_WS:
            results[f"{c}_w{int(w*100):02d}"] = evaluate(slices, bias_maps, f"dl_{c}", w, cv_folds)
    if args.dl_only:
        for c in args.variants:
            results[f"{c}_w00"] = evaluate(slices, bias_maps, f"dl_{c}", 0.0, cv_folds)

    # ── 탈상관 지표 ──
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    ev = (cache["target"].to_numpy() >= caps * 0.10) & np.isfinite(acf)
    eg = cache["gbm_cf"].to_numpy() - acf
    errs = {v: cache[f"dl_{v}"].to_numpy() - acf for v in args.variants}
    corr_rows = [{"variant": v, "corr_vs_gbm": float(np.corrcoef(eg[ev], e[ev])[0, 1]),
                  "dl_mae": float(np.abs(e[ev]).mean()),
                  "signed": float(e[ev].mean())} for v, e in errs.items()]
    corr_rows.append({"variant": "gbm", "corr_vs_gbm": 1.0,
                      "dl_mae": float(np.abs(eg[ev]).mean()), "signed": float(eg[ev].mean())})

    # ── 리포트 ──
    g_cv, g_hd = results["gbm"]["cv"][0], results["gbm"]["hold"][0]
    ref_name = f"{REF}_w70"
    r_cv = results[ref_name]["cv"][0] if ref_name in results else None
    r_hd = results[ref_name]["hold"][0] if ref_name in results else None
    print("\n" + "=" * 92)
    print(f"기준 gbm: CV {g_cv:.4f} (1-NMAE {results['gbm']['cv'][1]:.4f} FiCR {results['gbm']['cv'][2]:.4f}) | HOLD {g_hd:.4f}")
    if r_cv is not None:
        print(f"기준 {ref_name}(=exp-035 spd_w70): CV {r_cv:.4f} (Δgbm {r_cv-g_cv:+.4f}) | "
              f"HOLD {r_hd:.4f} (Δgbm {r_hd-g_hd:+.4f})   ※ 로그값 +0.0040/+0.0042와 일치해야 재현 OK")
    print("=" * 92)
    hdr = f"{'변형':<22}{'CV':>8}{'Δgbm':>9}{'HOLD':>8}{'Δgbm':>9}{'1-NMAE':>9}{'FiCR':>8}"
    if r_cv is not None:
        hdr += f"{'ΔrefCV':>9}{'Δrefhold':>10}  게이트"
    print(hdr)
    for name in results:
        if name == "gbm":
            continue
        r = results[name]
        dcv, dhd = r["cv"][0] - g_cv, r["hold"][0] - g_hd
        line = (f"{name:<22}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
                f"{r['cv'][1]:>9.4f}{r['cv'][2]:>8.4f}")
        if r_cv is not None:
            rc, rh = r["cv"][0] - r_cv, r["hold"][0] - r_hd
            gate = "통과" if (rc > 0 and rh > 0) else ("  ·" if max(rc, rh) > 0 else "   ")
            line += f"{rc:>+9.4f}{rh:>+10.4f}  {gate}"
        print(line)

    print("\n--- DL 단독 품질 / GBM 오차 상관 (평가대상 시간, 낮을수록 탈상관) ---")
    print(pd.DataFrame(corr_rows).round(4).to_string(index=False))
    if len(args.variants) > 1:
        print("\n--- DL 변형 간 오차 상관 ---")
        M = pd.DataFrame({a: {b: round(float(np.corrcoef(errs[a][ev], errs[b][ev])[0, 1]), 4)
                              for b in args.variants} for a in args.variants})
        print(M.to_string())

    print(f"\n--- fold별 ΔCV (기준 {ref_name} 대비) ---")
    if r_cv is not None:
        print(f"{'변형':<22}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds))
        for name in results:
            if name in ("gbm", ref_name):
                continue
            print(f"{name:<22}" + "".join(
                f"{results[name]['folds'][fn]-results[ref_name]['folds'][fn]:>+10.4f}" for fn in cv_folds))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=str, default="hub,l2,aw,q60")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dl-only", action="store_true", help="DL 단독(w=0) 칸도 평가")
    ap.add_argument("--refresh", action="store_true", help="DL 예측 캐시 무시하고 재학습")
    ap.add_argument("--quick", action="store_true", help="Q4+홀드아웃 배관 점검 (캐시 저장 안 함)")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    unknown = [v for v in args.variants if v not in VARIANTS]
    assert not unknown, f"미정의 변형: {unknown}"
    if args.quick:
        args.seeds = 1
        print("### QUICK 배관 점검 모드 (판정용 아님) ###")
    t0 = time.time()
    print(f"### exp-039 (변형={args.variants}, seeds={args.seeds}, device={DEVICE}) ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
