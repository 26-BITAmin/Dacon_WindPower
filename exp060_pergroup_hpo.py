# ═══════════════════════════════════════════════════════
# exp-060: 그룹별 GBM 하이퍼파라미터 최적화 — exp-056(공유 파라미터)의 그룹 분해
#
# 배경: exp-056은 3그룹이 하나의 파라미터를 공유(목적함수 = 3그룹 평균 MAE).
#   그러나 베이스라인 조기종료 최적 트리 수가 그룹별 154/461/72로 극단 편차 —
#   그룹별 데이터 양(그룹3은 1년 적음)·지형·터빈(V126 vs U136)이 다르므로
#   그룹별 독립 탐색이 공유 최적보다 나쁠 수 없는 구조 (탐색 노이즈 제외).
#
# 블록:
#   A. mix (캐시 전용): GBM 파라미터 앙상블 — cf_hpo ⊕ {cf_ec, cf_cln03} 혼합 사다리.
#      서로 다른 하이퍼의 GBM 평균 = 공짜 분산 감소 후보.
#   B. search: 그룹별 Optuna (Q4 폴드, exp-056과 동일 프로토콜·공간,
#      트라이얼 0 = exp-056 우승(현직), 1 = 구 베이스라인).
#   C. gate: 그룹별 우승 파라미터 → 3-seed·4-fold+홀드아웃, dl_cln03 블렌드, 신PP.
#      Δ = 보드 챔피언 구성(hpo×dC_w65, 재현 기대 CV 0.6556 / HOLD 0.6535).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp060_pergroup_hpo.py --stage all
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
from exp056_hpo import (BASELINE_LGB, BASELINE_XGB, GBM_HPO_CACHE, SEARCH_FOLD,
                        RESULT_JSON as E56_JSON, build_search_data)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp060_pg_best.json"
GBM_PG_CACHE = CACHE_DIR / "exp060_gbm_pg.parquet"


# ── A. 캐시 전용: GBM 파라미터 앙상블 혼합 ──
def stage_mix(cache, champ_bias):
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n[mix] 챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)
    ev_rows = cache["target"].notna()
    for other in ("cf_ec", "cf_cln03"):
        c = np.corrcoef(cache.loc[ev_rows, "cf_hpo"], cache.loc[ev_rows, other])[0, 1]
        print(f"  corr(cf_hpo, {other}) = {c:.4f}", flush=True)
    res = {}
    for other, tag in (("cf_ec", "ec"), ("cf_cln03", "cln")):
        for m in (0.75, 0.50):
            col = f"mix_{tag}{int(m*100)}"
            cache[col] = m * cache["cf_hpo"].to_numpy() + (1 - m) * cache[other].to_numpy()
            for w in (0.70, 0.65):
                res[f"{col}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)


# ── B. 그룹별 탐색 ──
def search_group_lgb(data, g, n_trials, jobs, incumbent):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap = data[g]
    m_ev = eval_w > 0

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
        m = lgb.LGBMRegressor(n_estimators=3000, objective="mae", random_state=0,
                              n_jobs=jobs, verbose=-1, **p)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)], eval_sample_weight=[eval_w],
              eval_metric="mae", callbacks=[lgb.early_stopping(150, verbose=False)])
        trial.set_user_attr("iters", int(m.best_iteration_ or 3000))
        return float(np.abs(m.predict(X_va)[m_ev] - y_va[m_ev]).mean() / cap)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42 + g))
    study.enqueue_trial(incumbent)      # 트라이얼 0 = exp-056 우승 (현직)
    study.enqueue_trial(BASELINE_LGB)   # 트라이얼 1 = 구 베이스라인
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    inc, best = study.trials[0], study.best_trial
    print(f"[LGB g{g}] 현직 {inc.value:.5f} (iter {inc.user_attrs['iters']}) → "
          f"best {best.value:.5f} (#{best.number}, iter {best.user_attrs['iters']}) "
          f"Δ {inc.value-best.value:+.5f}", flush=True)
    print(f"  {best.params}", flush=True)
    return best.params, int(round(best.user_attrs["iters"] * 1.1))


def search_group_xgb(data, g, n_trials, jobs, incumbent):
    import xgboost as xgbm
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap = data[g]
    m_ev = eval_w > 0

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
        m = xgbm.XGBRegressor(n_estimators=3000, objective="reg:absoluteerror",
                              tree_method="hist", random_state=0, n_jobs=jobs,
                              verbosity=0, early_stopping_rounds=150,
                              eval_metric="mae", **p)
        m.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)], sample_weight_eval_set=[eval_w], verbose=False)
        trial.set_user_attr("iters", int(m.best_iteration + 1))
        return float(np.abs(m.predict(X_va)[m_ev] - y_va[m_ev]).mean() / cap)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=142 + g))
    study.enqueue_trial(incumbent)
    study.enqueue_trial(BASELINE_XGB)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    inc, best = study.trials[0], study.best_trial
    print(f"[XGB g{g}] 현직 {inc.value:.5f} (iter {inc.user_attrs['iters']}) → "
          f"best {best.value:.5f} (#{best.number}, iter {best.user_attrs['iters']}) "
          f"Δ {inc.value-best.value:+.5f}", flush=True)
    print(f"  {best.params}", flush=True)
    return best.params, int(round(best.user_attrs["iters"] * 1.1))


# ── C. 게이트 ──
def train_gate_fold_pg(train_df, cols, fold, seeds, jobs, pg):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        gp = pg[str(group)]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        pl = np.mean([lgb.LGBMRegressor(n_estimators=gp["lgb_n"], objective="mae", random_state=s,
                                        n_jobs=jobs, verbose=-1, **gp["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=gp["xgb_n"], objective="reg:absoluteerror",
                                        tree_method="hist", random_state=s, n_jobs=jobs,
                                        verbosity=0, **gp["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_pg": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


# ── 공통 평가 ──
def make_ev(cache, champ_bias):
    slices = build_slices(cache)
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn, _ in FOLD_LIST if fn != hold]

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
    return ev


def print_table(res, ref):
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn, _ in FOLD_LIST if fn != hold]
    ch_cv, ch_hd = ref["cv"][0], ref["hold"][0]
    ch_n, ch_f = ref["cv"][1], ref["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<16}{'CV':>8}{'ΔCV':>9}{'HOLD':>8}{'Δhd':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=hpo×dC_w65 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - ref["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<16}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons))
    print(f"\n--- fold별 ΔCV (hpo×dC_w65 대비) ---")
    print(f"{'변형':<16}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관")
    for r in rows:
        print(f"{r['name']:<16}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}")


def load_cache_full():
    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert merge_cached_cols(cache, GBM_EC_CACHE, ["cf_ec"])
    assert merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
    return cache, champ_bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mix", "search", "gate", "all"], default="all")
    ap.add_argument("--trials-lgb", type=int, default=22)
    ap.add_argument("--trials-xgb", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-060 그룹별 GBM HPO ###", flush=True)

    if args.stage in ("mix", "all"):
        cache, champ_bias = load_cache_full()
        stage_mix(cache, champ_bias)
        print(f"[mix 완료 {(time.time()-t0)/60:.1f}분]", flush=True)

    if args.stage in ("search", "gate", "all"):
        train_df, cols, colsets = build_inputs(["spd"])
        ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
        train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
        train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
        train_df = merge_avail(train_df)
        use_cols = colsets["spd"] + EC_COLS

    if args.stage in ("search", "all"):
        e56 = json.loads(E56_JSON.read_text())
        data = build_search_data(train_df, use_cols)
        print(f"탐색 데이터 준비 ({SEARCH_FOLD}) — {time.time()-t0:.0f}s", flush=True)
        pg = {}
        for g in [1, 2, 3]:
            lgb_p, lgb_n = search_group_lgb(data, g, args.trials_lgb, args.jobs, e56["lgb"])
            xgb_p, xgb_n = search_group_xgb(data, g, args.trials_xgb, args.jobs, e56["xgb"])
            pg[str(g)] = dict(lgb=lgb_p, lgb_n=lgb_n, xgb=xgb_p, xgb_n=xgb_n)
            print(f"  [g{g} 탐색 완료 {(time.time()-t0)/60:.1f}분]", flush=True)
        RESULT_JSON.write_text(json.dumps(pg, indent=1))
        print(f"그룹별 우승 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        pg = json.loads(RESULT_JSON.read_text())
        for g in [1, 2, 3]:
            print(f"게이트 g{g}: LGB {pg[str(g)]['lgb_n']}트리 | XGB {pg[str(g)]['xgb_n']}트리", flush=True)
        cache, champ_bias = load_cache_full()
        have = merge_cached_cols(cache, GBM_PG_CACHE, ["cf_pg"])
        if "cf_pg" not in have:
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                preds.append(train_gate_fold_pg(train_df, use_cols, fold, args.seeds, args.jobs, pg))
                print(f"  [pg:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m["cf_pg"].notna().all(), "정렬 실패"
            cache["cf_pg"] = m["cf_pg"].to_numpy()
            save_cols(cache, GBM_PG_CACHE, ["cf_pg"])
            print(f"  캐시 저장 → {GBM_PG_CACHE.name}", flush=True)
        ev = make_ev(cache, champ_bias)
        ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
              f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
        cache["cf_pgmix"] = 0.5 * cache["cf_pg"].to_numpy() + 0.5 * cache["cf_hpo"].to_numpy()
        res = {"pg_w100": ev("cf_pg", None, 1.0, PP_NEW),
               "hpo_w100": ev("cf_hpo", None, 1.0, PP_NEW)}
        for w in (0.70, 0.65, 0.60):
            res[f"pg×dC_w{int(w*100)}"] = ev("cf_pg", "dl_cln03", w, PP_NEW)
            res[f"pgmix×dC_w{int(w*100)}"] = ev("cf_pgmix", "dl_cln03", w, PP_NEW)
        print_table(res, ref)
        d_cv = res["pg_w100"]["cv"][0] - res["hpo_w100"]["cv"][0]
        d_hd = res["pg_w100"]["hold"][0] - res["hpo_w100"]["hold"][0]
        print(f"\nGBM 단독 델타 (pg−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
