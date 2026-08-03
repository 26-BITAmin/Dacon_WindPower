# ═══════════════════════════════════════════════════════
# exp-115: GBM 제3 계열로 **CatBoost** — exp-065에서 측정 불가로 남은 유일한 칸
#
# 배경. exp-065가 "GBM 계열 내 다양성" 축을 닫았지만 **CatBoost만 미측정**이었다 —
#   당시 이 머신에서 DLL 로드가 애플리케이션 제어 정책에 차단됐기 때문(경험적 기각 아님).
#   2026-08-03 확인 결과 **import 정상(1.2.10)** → 그 칸을 지금 닫는다.
#
# 사전 확률은 낮다. exp-065 실측:
#   LGB · XGB · HGB **상호 corr 0.997~0.999** — 구현이 달라도 같은 352피처 위에서 수렴.
#   HGB는 Q4 MAE 0.11045(LGB 0.10779·XGB 0.10875 대비 열세) · corr 0.9985 · 혼합 8셀 전부 음수.
#
# 그래도 0은 아닌 이유:
#   HGB는 XGB와 같은 **level-wise**라 수렴이 당연했다. CatBoost는 **oblivious(대칭) 트리 +
#   ordered boosting**으로 GBM 계열 안에서 함수 클래스가 구조적으로 가장 다르다.
#   (exp-013의 CatBoost 기각은 132피처·무HPO 조건부 결론 — exp-056이 그 유형을 반증한 전례)
#
# 판정 순서 — **corr이 조기 사망 신호다**:
#   ① 단독 강도: Q4 MAE가 LGB/XGB와 같은 자릿수인가 (HGB는 여기서 이미 탈락)
#   ② **corr(cf_cat, cf_hpo)**: >0.998이면 HGB와 같은 결론 → 즉시 종료.
#      0.99 아래면 그때부터 혼합 사다리를 볼 가치가 있다. (참고: GBM↔DL은 0.9635)
#   ③ 혼합 사다리 cf_hpo ⊕ cf_cat × w{0.65, 0.70}
#
# ⚠ 변경 종류가 "모델 형태"다 — 7/27 이후 이 범주의 LB 전이 실적은 5연속 음수
#   (exp-055 −0.0025 · 058 −0.0029 · 083 −0.00124 · 112 −0.00237/−0.00333).
#   게이트를 통과해도 그 사실을 판정에 함께 넣을 것.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp115_catboost.py [--stage all] [--seeds 3]
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import (build_train_frame, slice_fold, fold_group_data,
                         cache_train_col, corr_on_eval)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CAT_CACHE = CACHE_DIR / "exp115_gbm_cat.parquet"
BENCH_FOLD = "fold_2024Q4"                       # exp-056/065와 동일 탐색 폴드
CORR_KILL = 0.998                                # HGB(0.9985)와 같은 영역이면 종료

# 대칭 트리 깊이는 leaf 수 = 2^depth. 챔피언 능선이 LGB(깊게)·XGB(얕게) 양쪽에 걸쳐 있어
# 얕음·중간·깊음 세 방향으로 괄호한다.
CONFIGS = {
    "cat_s": dict(depth=4, learning_rate=0.08, iterations=2000, l2_leaf_reg=1.0, rsm=1.0),
    "cat_m": dict(depth=6, learning_rate=0.05, iterations=1500, l2_leaf_reg=3.0, rsm=1.0),
    "cat_d": dict(depth=8, learning_rate=0.05, iterations=1000, l2_leaf_reg=5.0, rsm=0.8),
}


def fit_cat(X_tr, y_tr, w_tr, p, seed, jobs):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(loss_function="MAE", random_seed=seed, thread_count=jobs,
                          verbose=0, allow_writing_files=False, **p)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def bench(train_df, cols, jobs):
    """탐색 폴드에서 구성 3종 + 현직 2종의 단독 MAE 비교 (1-seed)."""
    import lightgbm as lgb
    import xgboost as xgbm
    e56 = json.loads(E56_JSON.read_text())
    fold = dict(FOLD_LIST)[BENCH_FOLD]
    tr_all, va_all = slice_fold(train_df, fold)
    out = {k: [] for k in list(CONFIGS) + ["lgb(현직)", "xgb(현직)"]}
    for group in [1, 2, 3]:
        d = fold_group_data(tr_all, va_all, group, cols)
        if d is None:
            continue
        X_tr, y_tr, w_tr, X_va, va_g, cap = d
        yv = va_g["target"].to_numpy()
        m = ~np.isnan(yv)
        def mae(pred):
            return float(np.abs(pred[m] - yv[m]).mean() / cap)
        out["lgb(현직)"].append(mae(lgb.LGBMRegressor(
            n_estimators=e56["lgb_n"], objective="mae", random_state=0, n_jobs=jobs,
            verbose=-1, **e56["lgb"]).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va)))
        out["xgb(현직)"].append(mae(xgbm.XGBRegressor(
            n_estimators=e56["xgb_n"], objective="reg:absoluteerror", tree_method="hist",
            random_state=0, n_jobs=jobs, verbosity=0,
            **e56["xgb"]).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va)))
        for name, p in CONFIGS.items():
            t = time.time()
            out[name].append(mae(fit_cat(X_tr, y_tr, w_tr, p, 0, jobs).predict(X_va)))
            print(f"    [{name} g{group}] {time.time()-t:.0f}s", flush=True)
    print(f"\n  {'구성':<12}{'Q4 MAE(그룹평균)':>18}   g1 / g2 / g3", flush=True)
    res = {}
    for k, v in out.items():
        res[k] = float(np.mean(v))
        print(f"  {k:<12}{np.mean(v):>18.5f}   " + " / ".join(f"{x:.5f}" for x in v), flush=True)
    best = min(CONFIGS, key=lambda k: res[k])
    print(f"\n  → CatBoost 최선 = {best} (MAE {res[best]:.5f}) | "
          f"현직 LGB {res['lgb(현직)']:.5f} · XGB {res['xgb(현직)']:.5f}", flush=True)
    print(f"  (참고: exp-065 HGB는 0.11045로 여기서 이미 탈락했다)", flush=True)
    return best, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bench", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--config", type=str, default=None, help="bench 건너뛰고 지정")
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-115: GBM 제3 계열 CatBoost (exp-065 미측정 칸) ###", flush=True)
    import catboost
    print(f"catboost {catboost.__version__}", flush=True)

    train_df, cols = build_train_frame()
    print(f"입력 {len(cols)}피처 | 학습행 {int(train_df['target'].notna().sum()):,}", flush=True)

    best = args.config
    if best is None:
        print(f"\n═══ ① 단독 강도 ({BENCH_FOLD}, 1-seed) ═══", flush=True)
        best, _ = bench(train_df, cols, args.jobs)
    if args.stage == "bench":
        return

    cache, champ_bias = load_cache_full()
    p = CONFIGS[best]
    n_seeds = 1 if p.get("rsm", 1.0) >= 0.999 else args.seeds     # rsm=1이면 시드 분산 미미

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
                                      "cf_cat": pc / cap}))
        return pd.concat(rows, ignore_index=True)

    print(f"\n═══ ② 전 폴드 학습 ({best}, {n_seeds}-seed) ═══", flush=True)
    cache_train_col(cache, train_df, cols, CAT_CACHE, ["cf_cat"], trainer)

    print(f"\n═══ ③ 조기 사망 신호 — 상관 ═══", flush=True)
    r_hpo = corr_on_eval(cache, "cf_cat", "cf_hpo")
    r_dl = corr_on_eval(cache, "cf_cat", "dl_cln03")
    mm = cache["target"].notna().to_numpy()
    shift = (cache["cf_cat"].to_numpy() - cache["cf_hpo"].to_numpy())[mm].mean() * 100
    print(f"  corr(cf_cat, cf_hpo)   = {r_hpo:.4f}   ← 판정선 {CORR_KILL}", flush=True)
    print(f"  corr(cf_cat, dl_cln03) = {r_dl:.4f}   (참고: cf_hpo↔dl_cln03 = 0.9635)", flush=True)
    print(f"  평균 이동 vs cf_hpo    = {shift:+.3f}%cap", flush=True)
    print(f"  [비교] exp-065 HGB = 0.9985 · exp-089 GEFS = 0.9994 · exp-088 PCA = 0.9996", flush=True)
    verdict = "희석 영역 — HGB와 같은 결론" if r_hpo >= CORR_KILL else \
              ("경계" if r_hpo >= 0.99 else "★ 탈상관 — 사다리 볼 가치 있음")
    print(f"  → {verdict}", flush=True)

    print(f"\n═══ ④ 혼합 사다리 ═══", flush=True)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f} (기대 0.6556/0.6535)",
          flush=True)
    for t in (5/6, 3/4, 2/3, 1/2):
        cache[f"cf_mix{int(t*100)}"] = t * cache["cf_hpo"] + (1 - t) * cache["cf_cat"]
    res = {"cat_solo": ev("cf_cat", None, 1.0, PP_NEW),
           "hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for t in (5/6, 3/4, 2/3, 1/2):
        for w in (0.65, 0.70):
            res[f"mix{int(t*100)}_w{int(w*100)}"] = ev(f"cf_mix{int(t*100)}", "dl_cln03", w, PP_NEW)
    for w in (0.65, 0.70):
        res[f"cat_w{int(w*100)}"] = ev("cf_cat", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print(f"\nGBM 단독 ΔCV: cat {res['cat_solo']['cv'][0]-res['hpo_solo']['cv'][0]:+.4f} "
          f"(exp-065 HGB는 −0.0020)", flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
