# ═══════════════════════════════════════════════════════
# exp-076: FiCR 밴드 목적함수 정련 부스팅 — "종류가 다른 레버" (훈련 단계 FiCR 직접 겨냥)
#
# 배경: 1위 격차 ~0.022의 ~90%가 FiCR (0.423 vs 0.466). exp-057이 죽인 것은
#   "고정 분포의 사후 재배치"였고, 이것은 훈련 목적함수로 분포 자체를 밴드에 맞게
#   바꾸는 시도 — 원리가 다름. COMPETITION.md 전략 4·5번("경계 근처 질량을 6% 안으로
#   미는 목적함수")의 첫 구현.
#
# 손실 (metric 정합 surrogate): credit(e) = 0.75·σ((b2−|e|)/τ) + 0.25·σ((b1−|e|)/τ)
#   (b1=0.06cap→4원, b2=0.08cap→3원=0.75크레딧, τ=0.015cap 온도).
#   L = (1−λ)·w_mae·|e|/cap − λ·w_band·credit(e),
#   w_mae = 챔피언 가중(저CF 0.05 × cln03) / w_band = [actual ≥ 0.1cap]·(actual/cap)·cln03
#   (FiCR의 actual 가중·eval 시간 구조 그대로).
#
# 구조 (순수 밴드는 0-초기화에서 그래디언트 소실 → 2단):
#   1단 = 챔피언 LGB(MAE, e56 하이퍼, 448트리) 그대로 학습
#   2단 = 그 예측을 init_score로 밴드 목적함수 정련 부스터 (150트리, lr 0.01)
#   변형: λ ∈ {1.0(순수 밴드), 0.5(혼합)} — cf_bd100 / cf_bd50
#
# 게이트: 3-seed·4-fold+홀드아웃. 멤버 배치는 사다리로 측정 (교차 구조 훼손 검출):
#   bd 단독 / 챔피언GBM⊕bd {75:25, 50:50} / × dl_cln03 w{0.65, 0.70}.
#   판정 Δ = 보드 챔피언 구성(hpo×dC_w65, 기대 0.6556/0.6535). 진단 = 밴드 적중률 직접 측정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp076_bandobj.py --stage smoke → gate
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor, merge_avail
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BND_CACHE = CACHE_DIR / "exp076_gbm_bnd.parquet"
TAU_F, N2, LR2 = 0.015, 150, 0.01


def band_objective(y, cap, w_mae, w_band, lam):
    """LGB 네이티브 fobj — 가중은 클로저에서 직접 반영 (dataset weight 미사용)"""
    b1, b2, tau = 0.06 * cap, 0.08 * cap, TAU_F * cap

    def fobj(preds, _train_data):
        e = preds - y
        ae = np.abs(e)
        s1 = 1.0 / (1.0 + np.exp(-np.clip((b1 - ae) / tau, -30, 30)))
        s2 = 1.0 / (1.0 + np.exp(-np.clip((b2 - ae) / tau, -30, 30)))
        dcredit = (0.25 * s1 * (1 - s1) + 0.75 * s2 * (1 - s2)) / tau   # d(credit)/d|e| 의 −값
        g_band = np.sign(e) * dcredit * cap                              # CF 단위 정규화
        g_mae = np.sign(e) / 1.0
        grad = (1 - lam) * w_mae * g_mae + lam * w_band * g_band
        hess = np.full_like(grad, 1.0)
        return grad, hess
    return fobj


def band_hits(pred, y, cap, mask):
    """actual 가중 밴드 적중 크레딧 (FiCR 분자/분모 그대로, 그룹 내)"""
    er = np.abs(pred[mask] - y[mask]) / cap
    a = y[mask]
    price = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
    return float((a * price).sum() / (a * 4.0).sum())


VARIANTS = {"bdn": dict(c=0.07, sd=0.025, n2=300, lr2=0.02),   # 6~8% 경계 협폭 타겟
            "bdw": dict(c=0.08, sd=0.050, n2=300, lr2=0.02)}   # 4~12% 광폭 타겟


def train_fold_band(train_df, cols, fold, seeds, jobs, e56, vc, colname):
    """1단 = 챔피언 LGB(MAE). 2단 = 경계 가중 MAE 정련 부스팅:
    w2 = [actual-가중 eval 시간] × Gaussian(er_init; c, sd) — 초기 오차가 밴드 경계
    근처인 행만 강하게 당김. (er_init은 in-sample이라 과소평가 소지 — 기록)"""
    import lightgbm as lgb
    tr_all, va_all = slice_fold(train_df, fold)
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        av = avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        w_mae = make_sample_weight(y_tr, cap, low_weight=0.05) * av
        w_band = (y_tr >= 0.1 * cap).astype(float) * (y_tr / cap) * av
        preds, inits = [], []
        for s in range(seeds):
            m1 = lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s,
                                   n_jobs=jobs, verbose=-1, **e56["lgb"])
            m1.fit(X_tr, y_tr, sample_weight=w_mae)
            init_tr = m1.predict(X_tr)
            init_va = m1.predict(X_va)
            er = np.abs(init_tr - y_tr) / cap
            bump = np.exp(-0.5 * ((er - vc["c"]) / vc["sd"]) ** 2)
            w2 = w_band * (bump + 0.05)
            w2 *= w_mae.sum() / max(w2.sum(), 1e-9)
            p2 = dict(e56["lgb"])
            p2["learning_rate"] = vc["lr2"]
            m2 = lgb.LGBMRegressor(n_estimators=vc["n2"], objective="mae", random_state=s,
                                   n_jobs=jobs, verbose=-1, **p2)
            m2.fit(X_tr, y_tr, sample_weight=w2, init_score=init_tr)
            preds.append(init_va + m2.predict(X_va))
            inits.append(init_va)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  colname: np.mean(preds, axis=0) / cap,
                                  "_init": np.mean(inits, axis=0) / cap}))
    return pd.concat(rows, ignore_index=True)


def stage_smoke(train_df, use_cols, e56, jobs):
    fold = dict(FOLD_LIST)["fold_2024Q4"]
    for tag, vc in VARIANTS.items():
        t0 = time.time()
        p = train_fold_band(train_df, use_cols, fold, 1, jobs, e56, vc, "cf_bd")
        _, va_all = slice_fold(train_df, fold)
        for group in [1, 2, 3]:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm")
            y = va_g["target"].to_numpy()
            sub = p.loc[p["group"] == group]
            ok = ~np.isnan(y)
            mask = ok & (y >= 0.1 * cap)
            pred_bd = sub["cf_bd"].to_numpy() * cap
            pred_in = sub["_init"].to_numpy() * cap
            fic_bd = band_hits(pred_bd, y, cap, mask)
            fic_in = band_hits(pred_in, y, cap, mask)
            mae_bd = np.abs(pred_bd[mask] - y[mask]).mean() / cap
            mae_in = np.abs(pred_in[mask] - y[mask]).mean() / cap
            print(f"  {tag} g{group}: FiCR형 크레딧 {fic_in:.4f}→{fic_bd:.4f} "
                  f"({fic_bd-fic_in:+.4f}) | eval MAE {mae_in:.4f}→{mae_bd:.4f} "
                  f"({mae_bd-mae_in:+.4f})", flush=True)
        print(f"  [{tag} smoke {time.time()-t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-076 FiCR 밴드 목적함수 정련 부스팅 ###", flush=True)
    train_df, use_cols = build_train_frame()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())

    if args.stage == "smoke":
        stage_smoke(train_df, use_cols, e56, args.jobs)
        print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
        return

    cache, champ_bias = load_cache_full()
    for tag, col in (("bdn", "cf_bd100"), ("bdw", "cf_bd50")):
        have = merge_cached_cols(cache, BND_CACHE, [col])
        if col not in have:
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                p = train_fold_band(train_df, use_cols, fold, args.seeds, args.jobs, e56, VARIANTS[tag], col)
                preds.append(p.drop(columns="_init"))
                print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, BND_CACHE,
                      [c for c in ("cf_bd100", "cf_bd50") if c in cache.columns])
            print(f"  캐시 저장 → {BND_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    for col in ("cf_bd100", "cf_bd50"):
        c = np.corrcoef(cache.loc[mm, col], cache.loc[mm, "cf_hpo"])[0, 1]
        sh = (cache[col] - cache["cf_hpo"]).mean() * 100
        print(f"  {col}: corr(cf_hpo) {c:.4f} | 평균 이동 {sh:+.3f}%cap", flush=True)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    hpov = cache["cf_hpo"].to_numpy()
    for col, tag in (("cf_bd100", "bd100"), ("cf_bd50", "bd50")):
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for mix, mt in ((0.75, "m75"), (0.5, "m50")):
            mc = f"{tag}{mt}"
            cache[mc] = mix * hpov + (1 - mix) * cache[col].to_numpy()
            res[f"{mc}_w65"] = ev(mc, "dl_cln03", 0.65, PP_NEW)
        res[f"{tag}_w65"] = ev(col, "dl_cln03", 0.65, PP_NEW)
        res[f"{tag}_w70"] = ev(col, "dl_cln03", 0.70, PP_NEW)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
