# ═══════════════════════════════════════════════════════
# exp-117: CatBoost를 **exp-056 프로토콜로 튜닝**한 뒤 심플렉스 재탐색
#
# exp-115/116의 미해결 한계. CatBoost는 **일반값 하이퍼**로 들어갔고 현직 LGB/XGB는
#   exp-056 HPO를 받은 상태였다 — 동일 조건 비교가 아니었다.
#   그리고 실패 원인이 **평가행 signed −4.00%**(현직 +0.27%)로 특정됐는데,
#   exp-056의 목적함수 `eval_mae_cf`는 **평가행(실측 ≥10%cap)에서만 MAE를 잰다**
#   (`exp056_hpo.py:70` — `m = eval_w > 0`). 즉 CatBoost에 없던 게 정확히 그 압력이다.
#   → 같은 압력을 주면 −4% 편향이 잡히는지 본다.
#
# 단계:
#   A(search) — Q4 폴드·1-seed·Optuna N트라이얼, 목적 = `eval_mae_cf`(평가행 MAE).
#               트라이얼 0 = exp-115 우승 cat_d(일반값)로 **개선폭을 직접 측정**.
#               트라이얼마다 **평가행 signed**를 user_attr로 기록 — 이게 진짜 판정 대상.
#   B(gate)   — 우승 cfg로 전 폴드 학습 → `cf_cat2` 캐시.
#   C(simplex)— (LGB, XGB, **cat2**) 심플렉스 91셀 × w{0.65,0.70} 재탐색 + 구 cf_cat 대조.
#               양수 셀이 나오면 **LOFO 필수**(exp-116에서 91셀 최댓값 +0.0006 → LOFO −0.0008).
#
# ⚠ 사전 확률: exp-116 심플렉스는 CatBoost 지분에 **단조 악화**였고 상위 8칸이 전부 Cat=0이었다.
#   튜닝으로 signed가 0 근처로 오지 않으면 결과는 안 바뀐다. signed를 먼저 볼 것.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp117_cat_hpo.py [--trials 20] [--stage all]
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import build_search_data, eval_mae_cf, SEARCH_FOLD
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold, fold_group_data, cache_train_col
from exp115_catboost import fit_cat, CONFIGS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BEST_JSON = CACHE_DIR / "exp117_cat_best.json"
CAT2_CACHE = CACHE_DIR / "exp117_gbm_cat2.parquet"
STEP = 12
WS = (0.65, 0.70)
NAMED = {(6, 6, 0): "LX(현직)", (6, 0, 6): "LC(XGB교체)", (0, 6, 6): "XC(LGB교체)",
         (0, 0, 12): "C2(단독)", (10, 0, 2): "Cat17%", (4, 4, 4): "균등"}
TMP = "cf_tmp"


def signed_on_eval(data, preds):
    """평가행 signed 편차(%cap) — exp-115가 특정한 실패 지표."""
    out = []
    for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
        m = eval_w > 0
        out.append(float((preds[g][m] - y_va[m]).mean() / cap * 100))
    return out


def search(data, n_trials, jobs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            iterations=trial.suggest_int("iterations", 500, 2500, step=250),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
            rsm=trial.suggest_float("rsm", 0.4, 1.0),
            random_strength=trial.suggest_float("random_strength", 0.5, 5.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            bootstrap_type="Bernoulli",
        )
        preds = {}
        for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
            preds[g] = fit_cat(X_tr, y_tr, w_tr, p, 0, jobs).predict(X_va)
        sg = signed_on_eval(data, preds)
        trial.set_user_attr("signed", sg)
        return eval_mae_cf(data, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    base = dict(CONFIGS["cat_d"])                     # exp-115 우승(일반값) = 트라이얼 0
    study.enqueue_trial({k: base[k] for k in ("depth", "learning_rate", "iterations",
                                              "l2_leaf_reg", "rsm")}
                        | {"random_strength": 1.0, "subsample": 0.8})
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    b0, bb = study.trials[0], study.best_trial
    print(f"\n[탐색 완료 {(time.time()-t0)/60:.1f}분]", flush=True)
    print(f"  기준(exp-115 일반값) 평가행 MAE {b0.value:.5f} | signed "
          + " / ".join(f"{x:+.2f}%" for x in b0.user_attrs["signed"]), flush=True)
    print(f"  최적 #{bb.number}         평가행 MAE {bb.value:.5f} | signed "
          + " / ".join(f"{x:+.2f}%" for x in bb.user_attrs["signed"])
          + f"  (ΔMAE {b0.value-bb.value:+.5f})", flush=True)
    print(f"\n  상위 5 트라이얼:", flush=True)
    for t in sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:5]:
        print(f"    #{t.number} MAE {t.value:.5f} signed "
              + "/".join(f"{x:+.1f}" for x in t.user_attrs["signed"])
              + f"  {t.params}", flush=True)
    p = dict(bb.params); p["bootstrap_type"] = "Bernoulli"
    BEST_JSON.write_text(json.dumps(p, indent=1))
    print(f"\n  저장 → {BEST_JSON.name}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stage", choices=["search", "all"], default="all")
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-117: CatBoost HPO (exp-056 프로토콜) + 심플렉스 재탐색 ###", flush=True)
    train_df, cols = build_train_frame()
    print(f"입력 {len(cols)}피처", flush=True)

    # ── A. 탐색 ──
    if BEST_JSON.exists():
        p = json.loads(BEST_JSON.read_text())
        print(f"\n[A] 캐시된 최적 cfg 재사용: {p}", flush=True)
    else:
        print(f"\n═══ A. Optuna 탐색 ({SEARCH_FOLD}, 1-seed, {args.trials}트라이얼) ═══", flush=True)
        data = build_search_data(train_df, cols)
        p = search(data, args.trials, args.jobs)
    if args.stage == "search":
        return

    # ── B. 전 폴드 학습 ──
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp065_gbm_members.parquet", ["cf_lgb", "cf_xgb"])
    merge_cached_cols(cache, CACHE_DIR / "exp115_gbm_cat.parquet", ["cf_cat"])
    n_seeds = 1 if p.get("rsm", 1.0) >= 0.999 and p.get("subsample", 1.0) >= 0.999 else args.seeds

    def trainer(tdf, use_cols, fold):
        tr_all, va_all = slice_fold(tdf, fold)
        rows = []
        for group in [1, 2, 3]:
            d = fold_group_data(tr_all, va_all, group, use_cols)
            if d is None:
                continue
            X_tr, y_tr, w_tr, X_va, va_g, cap = d
            pc = np.mean([fit_cat(X_tr, y_tr, w_tr, p, s, args.jobs).predict(X_va)
                          for s in range(n_seeds)], axis=0)
            rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                      "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                      "cf_cat2": pc / cap}))
        return pd.concat(rows, ignore_index=True)

    print(f"\n═══ B. 전 폴드 학습 ({n_seeds}-seed) ═══", flush=True)
    cache_train_col(cache, train_df, cols, CAT2_CACHE, ["cf_cat2"], trainer)

    # ── 프로파일: signed가 잡혔는가 ──
    lab = cache["target"].notna().to_numpy()
    cap = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / cap
    evm = lab & (acf >= 0.10)
    print(f"\n═══ C. 멤버 프로파일 (평가행 {int(evm.sum()):,}행) ═══", flush=True)
    print(f"  {'멤버':<10}{'평가행 MAE':>12}{'평가행 signed':>15}{'corr(cf_hpo)':>14}", flush=True)
    for c in ("cf_lgb", "cf_xgb", "cf_hpo", "cf_cat", "cf_cat2"):
        if c not in cache.columns:
            continue
        e = cache[c].to_numpy() - acf
        r = np.corrcoef(cache.loc[evm, c], cache.loc[evm, "cf_hpo"])[0, 1]
        print(f"  {c:<10}{np.abs(e[evm]).mean():>12.5f}{e[evm].mean()*100:>+14.3f}%{r:>14.4f}",
              flush=True)

    # ── D. 심플렉스 재탐색 ──
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n═══ D. 심플렉스 (LGB, XGB, cat2) ═══", flush=True)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}", flush=True)
    L, X, C = (cache[c].to_numpy() for c in ("cf_lgb", "cf_xgb", "cf_cat2"))
    rows = []
    for i in range(STEP + 1):
        for j in range(STEP + 1 - i):
            k = STEP - i - j
            cache[TMP] = (i * L + j * X + k * C) / STEP
            for w in WS:
                r = ev(TMP, "dl_cln03", w, PP_NEW)
                rows.append(dict(i=i, j=j, k=k, w=w, cv=r["cv"][0], hold=r["hold"][0],
                                 n=r["cv"][1], f=r["cv"][2],
                                 folds=[r["folds"][fn] for fn in sorted(r["folds"])]))
    df = pd.DataFrame(rows)
    df["dcv"] = df["cv"] - ref["cv"][0]; df["dhd"] = df["hold"] - ref["hold"][0]
    df["dn"] = df["n"] - ref["cv"][1];   df["df_"] = df["f"] - ref["cv"][2]

    print(f"\n  {'셀':<14}{'w':>6}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}", flush=True)
    for key, nm in NAMED.items():
        for w in WS:
            s = df[(df.i == key[0]) & (df.j == key[1]) & (df.k == key[2]) & (df.w == w)]
            if len(s):
                r = s.iloc[0]
                print(f"  {nm:<14}{w:>6.2f}{r.dcv:>+9.4f}{r.dhd:>+9.4f}{r.dn:>+9.4f}{r.df_:>+9.4f}",
                      flush=True)
    print(f"\n  ─ 상위 10 (전 91셀) ─", flush=True)
    print(f"  {'LGB/XGB/C2':<14}{'w':>6}{'ΔCV':>9}{'Δhold':>9}{'Δn':>9}{'Δf':>9}", flush=True)
    for r in df.sort_values("dcv", ascending=False).head(10).itertuples():
        print(f"  {f'{r.i}/{r.j}/{r.k}':<14}{r.w:>6.2f}{r.dcv:>+9.4f}{r.dhd:>+9.4f}"
              f"{r.dn:>+9.4f}{r.df_:>+9.4f}", flush=True)

    best = df.sort_values("dcv", ascending=False).iloc[0]
    print(f"\n═══ 판정 ═══", flush=True)
    print(f"  최댓값 {best.i}/{best.j}/{best.k} @ w{best.w:.2f}: ΔCV {best.dcv:+.4f} / "
          f"Δhold {best.dhd:+.4f}", flush=True)
    if best.dcv <= 0:
        print("  → **전 셀 음수. 튜닝해도 CatBoost 축은 닫힘.**", flush=True)
    else:
        folds = sorted(ref["folds"])
        tot = 0.0
        for fi, fn in enumerate(folds):
            o = df.assign(o=[np.mean([f for x, f in enumerate(r.folds) if x != fi])
                             for r in df.itertuples()])
            pick = o.sort_values("o", ascending=False).iloc[0]
            tot += pick.folds[fi] - ref["folds"][fn]
            print(f"    [{fn}] 타폴드 선택 {pick.i}/{pick.j}/{pick.k}@w{pick.w:.2f} "
                  f"→ Δ {pick.folds[fi]-ref['folds'][fn]:+.4f}", flush=True)
        print(f"  **LOFO 평균 Δ = {tot/len(folds):+.4f}**  "
              f"(exp-116: 최댓값 +0.0006 → LOFO −0.0008)", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
