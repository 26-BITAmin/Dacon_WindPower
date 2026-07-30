# ═══════════════════════════════════════════════════════
# exp-063: 2-폴드 목적함수 GBM HPO — 공유 파라미터 유지, 선택 기준만 다폴드화
#
# 배경: exp-060(그룹별 파라미터)·exp-062(그룹별 트리 수) 모두 "Q4 단일 폴드 선택"이
#   Q4 전용해로 과적합 (Q4 +0.005~7 / 타 폴드 −0.003~10). exp-056(공유 파라미터)만
#   일반화 — 공유 제약이 폴드 전문가를 걸러낸 정황. 남은 개선 경로는 제약을 풀지 않고
#   선택 기준을 강화하는 것: 목적함수 = Q3(챔피언 약점 폴드)+Q4 평균 eval-hour CF MAE.
#   트라이얼 0 = exp-056 우승(현직) → Δ가 "다폴드 기준에서의 추가 개선"을 직접 측정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp063_hpo2fold.py --stage all
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
from exp056_hpo import (BASELINE_LGB, BASELINE_XGB, RESULT_JSON as E56_JSON,
                        train_gate_fold)
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp063_h2_best.json"
GBM_H2_CACHE = CACHE_DIR / "exp063_gbm_h2.parquet"
SEARCH_FOLDS = ["fold_2024Q3", "fold_2024Q4"]


def build_search_data_mf(train_df, cols):
    """폴드×그룹별 (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap)"""
    fl = dict(FOLD_LIST)
    data = {}
    for fn in SEARCH_FOLDS:
        fold = fl[fn]
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                              (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
        for group in [1, 2, 3]:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[(va_all["kpx_group"] == group) & va_all["target"].notna()] \
                         .sort_values("forecast_kst_dtm").copy()
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"].to_numpy()
            w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
                avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            y_va = va_g["target"].to_numpy()
            eval_w = (y_va >= cap * 0.10).astype(float)
            data[(fn, group)] = (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap)
    return data


def run_trial_lgb(data, p, jobs):
    import lightgbm as lgb
    fold_maes, iters = [], []
    for fn in SEARCH_FOLDS:
        maes = []
        for g in [1, 2, 3]:
            X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap = data[(fn, g)]
            m = lgb.LGBMRegressor(n_estimators=3000, objective="mae", random_state=0,
                                  n_jobs=jobs, verbose=-1, **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], eval_sample_weight=[eval_w],
                  eval_metric="mae", callbacks=[lgb.early_stopping(150, verbose=False)])
            pred = m.predict(X_va)
            mm = eval_w > 0
            maes.append(np.abs(pred[mm] - y_va[mm]).mean() / cap)
            iters.append(int(m.best_iteration_ or 3000))
        fold_maes.append(float(np.mean(maes)))
    return float(np.mean(fold_maes)), iters


def run_trial_xgb(data, p, jobs):
    import xgboost as xgbm
    fold_maes, iters = [], []
    for fn in SEARCH_FOLDS:
        maes = []
        for g in [1, 2, 3]:
            X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap = data[(fn, g)]
            m = xgbm.XGBRegressor(n_estimators=3000, objective="reg:absoluteerror",
                                  tree_method="hist", random_state=0, n_jobs=jobs,
                                  verbosity=0, early_stopping_rounds=150,
                                  eval_metric="mae", **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], sample_weight_eval_set=[eval_w], verbose=False)
            pred = m.predict(X_va)
            mm = eval_w > 0
            maes.append(np.abs(pred[mm] - y_va[mm]).mean() / cap)
            iters.append(int(m.best_iteration + 1))
        fold_maes.append(float(np.mean(maes)))
    return float(np.mean(fold_maes)), iters


def search(data, n_lgb, n_xgb, jobs, e56):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj_lgb(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 160, log=True),
            max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8, 12]),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=1,
            lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        )
        v, iters = run_trial_lgb(data, p, jobs)
        trial.set_user_attr("iters", iters)
        return v

    def obj_xgb(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        v, iters = run_trial_xgb(data, p, jobs)
        trial.set_user_attr("iters", iters)
        return v

    out = {}
    for name, obj, n, inc, base in (("lgb", obj_lgb, n_lgb, e56["lgb"], BASELINE_LGB),
                                    ("xgb", obj_xgb, n_xgb, e56["xgb"], BASELINE_XGB)):
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=63))
        st.enqueue_trial(inc)     # 트라이얼 0 = exp-056 우승 (현직)
        st.enqueue_trial(base)    # 트라이얼 1 = 구 베이스라인
        st.optimize(obj, n_trials=n, show_progress_bar=False)
        incv, best = st.trials[0], st.best_trial
        print(f"[{name.upper()}] 현직(2폴드) {incv.value:.5f} → best {best.value:.5f} "
              f"(#{best.number}) Δ {incv.value-best.value:+.5f}", flush=True)
        for t in sorted(st.trials, key=lambda t: t.value)[:3]:
            print(f"  #{t.number} {t.value:.5f} {t.params}", flush=True)
        out[name] = best.params
        out[f"{name}_n"] = int(round(np.median(best.user_attrs["iters"]) * 1.1))
        out[f"{name}_delta"] = incv.value - best.value
        out[f"{name}_best_trial"] = best.number
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "gate", "all"], default="all")
    ap.add_argument("--trials-lgb", type=int, default=18)
    ap.add_argument("--trials-xgb", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-063 2-폴드 목적함수 GBM HPO ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df = merge_avail(train_df)
    use_cols = colsets["spd"] + EC_COLS
    e56 = json.loads(E56_JSON.read_text())

    if args.stage in ("search", "all"):
        data = build_search_data_mf(train_df, use_cols)
        print(f"탐색 데이터 준비 ({'+'.join(SEARCH_FOLDS)}) — {time.time()-t0:.0f}s", flush=True)
        best = search(data, args.trials_lgb, args.trials_xgb, args.jobs, e56)
        RESULT_JSON.write_text(json.dumps(best, indent=1))
        print(f"우승 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        best = json.loads(RESULT_JSON.read_text())
        if best.get("lgb_best_trial") == 0 and best.get("xgb_best_trial") == 0:
            print("\n두 탐색 모두 현직(트라이얼 0)이 우승 — 게이트 불필요, 챔피언 유지로 종료.", flush=True)
            print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
            return
        print(f"게이트: LGB {best['lgb_n']}트리 | XGB {best['xgb_n']}트리", flush=True)
        cache, champ_bias = load_cache_full()
        have = merge_cached_cols(cache, GBM_H2_CACHE, ["cf_h2"])
        if "cf_h2" not in have:
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                p = train_gate_fold(train_df, use_cols, fold, args.seeds, args.jobs,
                                    best["lgb"], best["lgb_n"], best["xgb"], best["xgb_n"])
                preds.append(p.rename(columns={"cf_hpo": "cf_h2"}))
                print(f"  [h2:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m["cf_h2"].notna().all(), "정렬 실패"
            cache["cf_h2"] = m["cf_h2"].to_numpy()
            save_cols(cache, GBM_H2_CACHE, ["cf_h2"])
            print(f"  캐시 저장 → {GBM_H2_CACHE.name}", flush=True)
        ev = make_ev(cache, champ_bias)
        ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
              f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
        res = {"h2_w100": ev("cf_h2", None, 1.0, PP_NEW),
               "hpo_w100": ev("cf_hpo", None, 1.0, PP_NEW)}
        for w in (0.70, 0.65, 0.60):
            res[f"h2×dC_w{int(w*100)}"] = ev("cf_h2", "dl_cln03", w, PP_NEW)
        print_table(res, ref)
        d_cv = res["h2_w100"]["cv"][0] - res["hpo_w100"]["cv"][0]
        d_hd = res["h2_w100"]["hold"][0] - res["hpo_w100"]["hold"][0]
        print(f"\nGBM 단독 델타 (h2−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
