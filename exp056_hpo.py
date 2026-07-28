# ═══════════════════════════════════════════════════════
# exp-056: GBM 하이퍼파라미터 최적화 (Optuna) — exp-002 이래 고정된 기본기 재탐색
#
# 배경: LGBM(500트리·lr0.03·31잎·depth6)·XGB(500·0.03·depth6)는 87피처 시절 설정.
#   현재 352피처·정제 가중인데 용량 재탐색은 exp-013(구식 132피처) 한 번뿐,
#   early stopping은 전 실험 미사용 (exp-006 기록). 상위권과의 1-NMAE 격차 0.006.
#
# 구조 (탐색·판정 분리):
#   A. LGBM 탐색: Q4 폴드(최대 학습량), 그룹별 학습(kWh 타깃·cln03 가중), n_est 3000 +
#      early stopping 150 (검증 가중 = 평가시간만). 목적함수 = 3그룹 평균 eval-hour CF MAE.
#      트라이얼 0 = 현행 파라미터 강제 삽입 (개선 폭 직접 측정).
#   B. XGB 탐색: 동일 구조.
#   C. 게이트: 우승 파라미터(트리 수 = 탐색 best_iter 중앙값 ×1.1 고정 — 제출 시 검증셋
#      없음)로 3-seed·4-fold+홀드아웃, dl_cln03 블렌드, 신PP. Δ = 보드 챔피언(ec×dC_w65).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp056_hpo.py --stage all --trials-lgb 30 --trials-xgb 25
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY, PP_NEW,
                          avail_factor, merge_avail, postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp054_gate import build_ec_frame, EC_WIDE, EC_COLS, GBM_EC_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp056_hpo_best.json"
GBM_HPO_CACHE = CACHE_DIR / "exp056_gbm_hpo.parquet"
SEARCH_FOLD = "fold_2024Q4"
BASELINE_LGB = dict(learning_rate=0.03, num_leaves=31, max_depth=6, min_child_samples=20,
                    feature_fraction=0.8, bagging_fraction=0.8,
                    lambda_l1=1e-4, lambda_l2=1e-4)      # log 공간 하한 = 사실상 0 (현행)
BASELINE_XGB = dict(learning_rate=0.03, max_depth=6, min_child_weight=5.0,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-4, reg_lambda=1.0)


def build_search_data(train_df, cols):
    """Q4 폴드 그룹별 (X_tr, y_tr, w_tr, X_va, y_va, eval_w) — 1회 준비 후 전 트라이얼 공유"""
    fold = dict(FOLD_LIST)[SEARCH_FOLD]
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    data = {}
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
        data[group] = (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap)
    return data


def eval_mae_cf(data, preds):
    maes = []
    for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
        m = eval_w > 0
        maes.append(np.abs(preds[g][m] - y_va[m]).mean() / cap)
    return float(np.mean(maes))


def search_lgb(data, n_trials, jobs):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
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
        preds, iters = {}, []
        for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
            m = lgb.LGBMRegressor(n_estimators=3000, objective="mae", random_state=0,
                                  n_jobs=jobs, verbose=-1, **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], eval_sample_weight=[eval_w],
                  eval_metric="mae", callbacks=[lgb.early_stopping(150, verbose=False)])
            preds[g] = m.predict(X_va)
            iters.append(int(m.best_iteration_ or 3000))
        trial.set_user_attr("iters", iters)
        return eval_mae_cf(data, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(BASELINE_LGB)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    base = study.trials[0]
    best = study.best_trial
    print(f"[LGB] baseline MAE {base.value:.5f} (iters {base.user_attrs['iters']}) → "
          f"best MAE {best.value:.5f} (trial {best.number}, iters {best.user_attrs['iters']}) "
          f"Δ {base.value-best.value:+.5f}", flush=True)
    for t in sorted(study.trials, key=lambda t: t.value)[:5]:
        print(f"  #{t.number} {t.value:.5f} {t.params}", flush=True)
    return best.params, int(np.median(best.user_attrs["iters"]))


def search_xgb(data, n_trials, jobs):
    import xgboost as xgbm
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        preds, iters = {}, []
        for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
            m = xgbm.XGBRegressor(n_estimators=3000, objective="reg:absoluteerror",
                                  tree_method="hist", random_state=0, n_jobs=jobs,
                                  verbosity=0, early_stopping_rounds=150,
                                  eval_metric="mae", **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr,
                  eval_set=[(X_va, y_va)], sample_weight_eval_set=[eval_w], verbose=False)
            preds[g] = m.predict(X_va)
            iters.append(int(m.best_iteration + 1))
        trial.set_user_attr("iters", iters)
        return eval_mae_cf(data, preds)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(BASELINE_XGB)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    base = study.trials[0]
    best = study.best_trial
    print(f"[XGB] baseline MAE {base.value:.5f} (iters {base.user_attrs['iters']}) → "
          f"best MAE {best.value:.5f} (trial {best.number}, iters {best.user_attrs['iters']}) "
          f"Δ {base.value-best.value:+.5f}", flush=True)
    for t in sorted(study.trials, key=lambda t: t.value)[:5]:
        print(f"  #{t.number} {t.value:.5f} {t.params}", flush=True)
    return best.params, int(np.median(best.user_attrs["iters"]))


def train_gate_fold(train_df, cols, fold, seeds, jobs, lgb_p, lgb_n, xgb_p, xgb_n):
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
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        pl = np.mean([lgb.LGBMRegressor(n_estimators=lgb_n, objective="mae", random_state=s,
                                        n_jobs=jobs, verbose=-1, **lgb_p)
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=xgb_n, objective="reg:absoluteerror",
                                        tree_method="hist", random_state=s, n_jobs=jobs,
                                        verbosity=0, **xgb_p)
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_hpo": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


def evaluate(cache, champ_bias):
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

    r_ec = ev("cf_ec", "dl_cln03", 0.65, PP_NEW)
    print(f"\n보드챔피언(ec×dC_w65): CV {r_ec['cv'][0]:.4f} (1-NMAE {r_ec['cv'][1]:.4f} FiCR {r_ec['cv'][2]:.4f}) | "
          f"HOLD {r_ec['hold'][0]:.4f}  (기대 0.6541/0.6529)")
    res = {"ec_w100": ev("cf_ec", None, 1.0, PP_NEW),
           "hpo_w100": ev("cf_hpo", None, 1.0, PP_NEW)}
    for w in (0.75, 0.70, 0.65, 0.60):
        res[f"hpo×dC_w{int(w*100)}"] = ev("cf_hpo", "dl_cln03", w, PP_NEW)

    ch_cv, ch_hd = r_ec["cv"][0], r_ec["hold"][0]
    ch_n, ch_f = r_ec["cv"][1], r_ec["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVec':>9}{'HOLD':>8}{'Δhdec':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=보드챔피언 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ec["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons))
    d_solo_cv = res["hpo_w100"]["cv"][0] - res["ec_w100"]["cv"][0]
    d_solo_hd = res["hpo_w100"]["hold"][0] - res["ec_w100"]["hold"][0]
    print(f"\nGBM 단독 델타 (hpo−ec): ΔCV {d_solo_cv:+.4f} / Δhold {d_solo_hd:+.4f}")
    print(f"\n--- fold별 ΔCV (보드챔피언 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관")
    for r in rows:
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "gate", "all"], default="all")
    ap.add_argument("--trials-lgb", type=int, default=30)
    ap.add_argument("--trials-xgb", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-056 GBM HPO ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df = merge_avail(train_df)
    use_cols = vcols + EC_COLS

    if args.stage in ("search", "all"):
        data = build_search_data(train_df, use_cols)
        print(f"탐색 데이터 준비 ({SEARCH_FOLD}) — {time.time()-t0:.0f}s", flush=True)
        lgb_p, lgb_n = search_lgb(data, args.trials_lgb, args.jobs)
        xgb_p, xgb_n = search_xgb(data, args.trials_xgb, args.jobs)
        RESULT_JSON.write_text(json.dumps(dict(
            lgb=lgb_p, lgb_n=int(round(lgb_n * 1.1)),
            xgb=xgb_p, xgb_n=int(round(xgb_n * 1.1))), indent=1))
        print(f"우승 파라미터 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        best = json.loads(RESULT_JSON.read_text())
        print(f"게이트: LGB {best['lgb_n']}트리 {best['lgb']} | XGB {best['xgb_n']}트리 {best['xgb']}", flush=True)
        cache, champ_bias = load_eval_frame()
        assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
        assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
        assert merge_cached_cols(cache, GBM_EC_CACHE, ["cf_ec"])
        have = merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
        if "cf_hpo" not in have:
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                preds.append(train_gate_fold(train_df, use_cols, fold, args.seeds, args.jobs,
                                             best["lgb"], best["lgb_n"], best["xgb"], best["xgb_n"]))
                print(f"  [hpo:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m["cf_hpo"].notna().all(), "정렬 실패"
            cache["cf_hpo"] = m["cf_hpo"].to_numpy()
            save_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
            print(f"  캐시 저장 → {GBM_HPO_CACHE.name}", flush=True)
        evaluate(cache, champ_bias)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
