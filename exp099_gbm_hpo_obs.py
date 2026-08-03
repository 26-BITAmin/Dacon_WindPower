# ═══════════════════════════════════════════════════════
# exp-099: GBM 하이퍼 재탐색 — **목적함수 = 파트너 DL과의 블렌드 Score** (공동 최적화)
#
# 사용자 지적 2건 반영:
#   (1) "상관계수는 참고용, validation으로 판정" → 판정은 CV·홀드 두 게이트로만.
#   (2) "GBM x DL이라면서 DL만 돌리는 게 아니다" → 초판(exp-099 v1)은 목적함수가
#       GBM 단독 MAE였음(exp-056 하네스 그대로) = 파트너 무시. **폐기하고 재작성.**
#       여기서는 exp-058 선례(목적함수 = Q4 블렌드 Score)를 GBM 쪽에 적용 —
#       각 트라이얼의 GBM 예측을 **고정된 파트너 DL과 블렌드 → 신PP → 채점**.
#
# 파트너 DL을 바꿔가며 두 번 탐색해서 "파트너가 바뀌면 최적 GBM도 바뀌는가"를 직접 측정:
#   --partner cln03  : 챔피언 DL 파트너 (대조)
#   --partner obs    : ASOS DL 파트너 (exp-096, 새 멤버)
#
# 피처셋: 챔피언 352(spd+EC) + ASOS obs3. 탐색 폴드 = Q4 (exp-056/058과 동일).
# 실행: C:\miniconda3\python.exe -X utf8 exp099_gbm_hpo_obs.py --partner obs [--trials-lgb 20]
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import KEY, PP_NEW, postprocess_flex, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp082_gate import prepare_frame, OBS3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SEARCH_FOLD = "fold_2024Q4"
W_GBM = 0.65
PARTNERS = {"cln03": "dl_cln03", "obs": "dl_obs"}


def prep_search(train_df, cache, cols):
    """Q4 폴드: 그룹별 (X_tr, y_tr, w_tr, X_va) + cache 정렬 인덱스"""
    fold = dict(FOLD_LIST)[SEARCH_FOLD]
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    sel = (cache["fold"] == SEARCH_FOLD).to_numpy()
    data, order = {}, []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        keys = cache.loc[sel & (cache["group"] == group), ["group", "forecast_kst_dtm"]]
        va_g = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                          right_on=["kpx_group", "forecast_kst_dtm"], how="left")
        assert len(va_g) == len(keys), "정렬 실패"
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        data[group] = (X_tr, y_tr, w_tr, X_va, cap)
        order.append(keys.assign(row=np.arange(len(keys))))
    return data, sel


def score_blend(cache, champ_bias, sel, gbm_cf_by_group, dl_col):
    """트라이얼 GBM 예측(그룹별 CF)을 파트너 DL과 블렌드 → 신PP → Q4 채점"""
    sub = cache.loc[sel].reset_index(drop=True).copy()
    gcf = np.empty(len(sub))
    for group, arr in gbm_cf_by_group.items():
        gcf[(sub["group"] == group).to_numpy()] = arr
    blend = W_GBM * gcf + (1 - W_GBM) * sub[dl_col].to_numpy()
    parts = []
    for (fname, group), s in build_slices(sub):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[s.index], cap, s["ws"].to_numpy(), s["pA"].to_numpy(),
                                group, champ_bias[fname], **PP_NEW)
        parts.append(s[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    return score_from_tidy(pd.concat(parts, ignore_index=True), "pred")[0]


def search(data, cache, champ_bias, sel, dl_col, n_lgb, n_xgb, jobs):
    import lightgbm as lgb
    import xgboost as xgbm
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run_lgb(p, n):
        return {g: np.mean([lgb.LGBMRegressor(n_estimators=n, objective="mae", random_state=s,
                                              n_jobs=jobs, verbose=-1, **p)
                            .fit(d[0], d[1], sample_weight=d[2]).predict(d[3])
                            for s in range(1)], axis=0) / d[4] for g, d in data.items()}

    def run_xgb(p, n):
        return {g: np.mean([xgbm.XGBRegressor(n_estimators=n, objective="reg:absoluteerror",
                                              tree_method="hist", random_state=s, n_jobs=jobs,
                                              verbosity=0, **p)
                            .fit(d[0], d[1], sample_weight=d[2]).predict(d[3])
                            for s in range(1)], axis=0) / d[4] for g, d in data.items()}

    best = {}
    for tag, n_trials, runner, space in (
        ("lgb", n_lgb, run_lgb, lambda t: dict(
            num_leaves=t.suggest_int("num_leaves", 31, 255, log=True),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.10, log=True),
            min_child_samples=t.suggest_int("min_child_samples", 10, 80),
            feature_fraction=t.suggest_float("feature_fraction", 0.4, 0.95),
            bagging_fraction=t.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=t.suggest_int("bagging_freq", 0, 5),
            lambda_l2=t.suggest_float("lambda_l2", 1e-3, 10.0, log=True))),
        ("xgb", n_xgb, run_xgb, lambda t: dict(
            max_depth=t.suggest_int("max_depth", 4, 12),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.10, log=True),
            min_child_weight=t.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 0.95),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))),
    ):
        n_est = 800 if tag == "lgb" else 700

        def objective(trial):
            p = space(trial)
            preds = runner(p, n_est)
            return score_blend(cache, champ_bias, sel, preds, dl_col)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        print(f"  [{tag}] best Q4 블렌드 Score {study.best_value:.5f} "
              f"(trial {study.best_trial.number})", flush=True)
        for t in sorted(study.trials, key=lambda t: -(t.value or 0))[:3]:
            print(f"    #{t.number} {t.value:.5f} {t.params}", flush=True)
        best[tag] = study.best_params
        best[f"{tag}_n"] = n_est
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partner", choices=list(PARTNERS), default="obs")
    ap.add_argument("--trials-lgb", type=int, default=20)
    ap.add_argument("--trials-xgb", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stage", choices=["search", "gate", "all"], default="all")
    args = ap.parse_args()
    t0 = time.time()
    dl_col = PARTNERS[args.partner]
    tag = f"hpoJ{args.partner}"
    RESULT_JSON = CACHE_DIR / f"exp099_{tag}_best.json"
    GBM_CACHE = CACHE_DIR / f"exp099_gbm_{tag}.parquet"
    print(f"### exp-099: GBM 하이퍼 공동 최적화 (파트너 = {dl_col}) ###", flush=True)

    train_df, use_cols, _ = prepare_frame(5)
    cols = use_cols + OBS3
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp096_dl_obs.parquet", ["dl_obs"]), \
        "exp-096 DL 캐시 필요"
    print(f"탐색 피처 {len(cols)}개 | 목적함수 = Q4 블렌드 Score (w_gbm {W_GBM})", flush=True)

    if args.stage in ("search", "all"):
        data, sel = prep_search(train_df, cache, cols)
        base = score_blend(cache, champ_bias, sel,
                           {g: cache.loc[sel & (cache["group"] == g), "cf_hpo"].to_numpy()
                            for g in [1, 2, 3]}, dl_col)
        print(f"  기준선(챔피언 GBM x {dl_col}) Q4 Score {base:.5f}", flush=True)
        best = search(data, cache, champ_bias, sel, dl_col,
                      args.trials_lgb, args.trials_xgb, args.jobs)
        RESULT_JSON.write_text(json.dumps(best))
        print(f"우승 파라미터 저장 -> {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        best = json.loads(RESULT_JSON.read_text())
        col = f"cf_{tag}"
        if not merge_cached_cols(cache, GBM_CACHE, [col]):
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                p = train_gate_fold(train_df, cols, fold, args.seeds, args.jobs,
                                    best["lgb"], best["lgb_n"], best["xgb"], best["xgb_n"])
                preds.append(p.rename(columns={"cf_hpo": col}))
                print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), "정렬 실패"
            cache[col] = m[col].to_numpy()
            save_cols(cache, GBM_CACHE, [col])
            print(f"  캐시 저장 -> {GBM_CACHE.name}", flush=True)

        ev = make_ev(cache, champ_bias)
        ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f} "
              f"(기대 0.6556/0.6535)", flush=True)
        res = {}
        for g in ("cf_hpo", col):
            for d in ("dl_cln03", "dl_obs"):
                for w in (0.60, 0.65, 0.70):
                    res[f"{g.replace('cf_','')}x{d.replace('dl_','')}_w{int(w*100)}"] = \
                        ev(g, d, w, PP_NEW)
        print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
