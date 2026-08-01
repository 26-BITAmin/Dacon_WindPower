# ═══════════════════════════════════════════════════════
# exp-085: 파워커브 트랙 R3 — **풍속 타깃 전용 GBM HPO**
#
# 트랙 규칙 (사용자 지시): 판정 기준 = 이 파이프라인의 이전 버전(pcinv v1).
#
# R3 가설: v1~R2는 챔피언의 GBM 하이퍼(exp-056)를 그대로 썼는데, 그 값은 **kWh 타깃**
#   기준으로 탐색된 것. 이 파이프라인의 타깃은 유효 풍속(평균 6.42 m/s, 분포·스케일·
#   꼬리가 전혀 다름) → 하이퍼가 정합하지 않을 개연성이 큼.
#   선례: exp-056이 "87피처 시절 하이퍼가 실병목"을 실증하며 base측 최대 이득
#   (+0.0024/+0.0024, 완전 대칭)을 냈음. 같은 유형의 미정합이 타깃 축에서 반복될 수 있음.
#
# 프로토콜 개선 2건:
#   (a) 목적함수 = **Q2+Q4 2폴드 평균** (exp-079/082에서 Q4 단독 선택이 3연속 과대평가
#       — exp-063의 2폴드 HPO 선례).
#   (b) 목적함수 척도 = **CF 공간 MAE** (풍속 공간이 아니라 정방향 변환 후) — 실제
#       점수가 매겨지는 공간에서 선택.
#   트라이얼 0 = 현행 e56 하이퍼 강제 삽입 → 개선 폭 직접 측정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp085_pchpo.py --stage search --trials-lgb 20 --trials-xgb 16
#       C:\miniconda3\python.exe -X utf8 exp085_pchpo.py --stage gate
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, BASELINE_LGB, BASELINE_XGB
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval, slice_fold
from exp082_pcinv import make_transform, build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp085_pchpo_best.json"
V1_CACHE = CACHE_DIR / "exp082_gbm_pcinv.parquet"
HPO_CACHE = CACHE_DIR / "exp085_gbm_pchpo.parquet"
SEARCH_FOLDS = ["fold_2024Q2", "fold_2024Q4"]


def build_search_data(train_df, scada_h, cols):
    """Q2+Q4 × 3그룹 — 트라이얼 간 공유. v1 구성(평균 커브·전 시간·cln03) 고정."""
    data = {}
    for fname in SEARCH_FOLDS:
        fold = dict(FOLD_LIST)[fname]
        tr_all, va_all = slice_fold(train_df, fold)
        for group in [1, 2, 3]:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) &
                              tr_all["target"].notna()].copy()
            va_g = va_all.loc[(va_all["kpx_group"] == group) &
                              va_all["target"].notna()].sort_values("forecast_kst_dtm").copy()
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            fwd, inv, _, _ = make_transform(curve)
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"].to_numpy()
            w = make_sample_weight(y_tr, cap, low_weight=0.05) * \
                avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
            y_va = va_g["target"].to_numpy()
            data[(fname, group)] = dict(
                X_tr=X_tr, v_tr=inv(y_tr / cap), w=w, X_va=X_va,
                v_va=inv(y_va / cap), y_cf=y_va / cap,
                ev=(y_va >= cap * 0.10), fwd=fwd, cap=cap)
    return data


def cf_mae(data, preds):
    """CF 공간 평가시간 MAE — 폴드×그룹 평균 (점수가 매겨지는 공간에서 선택)"""
    out = []
    for k, d in data.items():
        m = d["ev"]
        out.append(np.abs(d["fwd"](preds[k])[m] - d["y_cf"][m]).mean())
    return float(np.mean(out))


def search(kind, data, n_trials, jobs, baseline):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if kind == "lgb":
        import lightgbm as lgb
    else:
        import xgboost as xgbm

    def suggest(trial):
        if kind == "lgb":
            return dict(
                learning_rate=trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
                num_leaves=trial.suggest_int("num_leaves", 16, 160, log=True),
                max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8, 12]),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
                feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
                bagging_freq=1,
                lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
                lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True))
        return dict(
            learning_rate=trial.suggest_float("learning_rate", 0.008, 0.06, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))

    def objective(trial):
        p = suggest(trial)
        tt = time.time()
        preds, iters = {}, []
        for k, d in data.items():
            if kind == "lgb":
                m = lgb.LGBMRegressor(n_estimators=3000, objective="mae", random_state=0,
                                      n_jobs=jobs, verbose=-1, **p)
                m.fit(d["X_tr"], d["v_tr"], sample_weight=d["w"],
                      eval_set=[(d["X_va"], d["v_va"])],
                      eval_sample_weight=[d["ev"].astype(float)], eval_metric="mae",
                      callbacks=[lgb.early_stopping(150, verbose=False)])
                iters.append(int(m.best_iteration_ or 3000))
            else:
                m = xgbm.XGBRegressor(n_estimators=3000, objective="reg:absoluteerror",
                                      tree_method="hist", random_state=0, n_jobs=jobs,
                                      verbosity=0, early_stopping_rounds=150,
                                      eval_metric="mae", **p)
                m.fit(d["X_tr"], d["v_tr"], sample_weight=d["w"],
                      eval_set=[(d["X_va"], d["v_va"])],
                      sample_weight_eval_set=[d["ev"].astype(float)], verbose=False)
                iters.append(int(m.best_iteration + 1))
            preds[k] = m.predict(d["X_va"])
        trial.set_user_attr("iters", iters)
        v = cf_mae(data, preds)
        print(f"  [{kind}#{trial.number}] CF-MAE {v:.5f} ({time.time()-tt:.0f}s) "
              f"iters~{int(np.median(iters))}", flush=True)
        return v

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=85))
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    mir, best = study.trials[0], study.best_trial
    print(f"[{kind.upper()}] 미러(현행 e56) {mir.value:.5f} → best {best.value:.5f} "
          f"(#{best.number}) Δ {mir.value-best.value:+.5f}", flush=True)
    for t in sorted(study.trials, key=lambda t: t.value)[:5]:
        print(f"  #{t.number} {t.value:.5f} {t.params}", flush=True)
    return best.params, int(np.median(best.user_attrs["iters"])), mir.value, best.value


def train_fold(train_df, scada_h, cols, fold, seeds, jobs, best):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all, va_all = slice_fold(train_df, fold)
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
        fwd, inv, _, _ = make_transform(curve)
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        v_t = inv(y_tr / cap)
        w = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        pl = np.mean([lgb.LGBMRegressor(n_estimators=best["lgb_n"], objective="mae",
                                        random_state=s, n_jobs=jobs, verbose=-1, **best["lgb"])
                      .fit(X_tr, v_t, sample_weight=w).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=best["xgb_n"], objective="reg:absoluteerror",
                                        tree_method="hist", random_state=s, n_jobs=jobs,
                                        verbosity=0, **best["xgb"])
                      .fit(X_tr, v_t, sample_weight=w).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_pchpo": fwd(0.5 * pl + 0.5 * px)}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "gate", "all"], default="all")
    ap.add_argument("--trials-lgb", type=int, default=20)
    ap.add_argument("--trials-xgb", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-085 파워커브 트랙 R3: 풍속 타깃 전용 GBM HPO ###", flush=True)

    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)

    if args.stage in ("search", "all"):
        data = build_search_data(train_df, scada_h, use_cols)
        print(f"탐색 데이터 준비 ({'+'.join(SEARCH_FOLDS)} × 3그룹 = {len(data)}칸) "
              f"— {time.time()-t0:.0f}s", flush=True)
        e56 = json.loads(E56_JSON.read_text())
        lgb_p, lgb_n, lm, lb = search("lgb", data, args.trials_lgb, args.jobs, e56["lgb"])
        xgb_p, xgb_n, xm, xb = search("xgb", data, args.trials_xgb, args.jobs, e56["xgb"])
        RESULT_JSON.write_text(json.dumps(dict(
            lgb=lgb_p, lgb_n=int(round(lgb_n * 1.1)),
            xgb=xgb_p, xgb_n=int(round(xgb_n * 1.1)),
            mae_mirror_lgb=lm, mae_best_lgb=lb,
            mae_mirror_xgb=xm, mae_best_xgb=xb), indent=1))
        print(f"우승 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        best = json.loads(RESULT_JSON.read_text())
        print(f"게이트: LGB {best['lgb_n']}트리 | XGB {best['xgb_n']}트리", flush=True)
        cache, champ_bias = load_cache_full()
        assert merge_cached_cols(cache, V1_CACHE, ["cf_pcinv"]), "v1 캐시 없음"
        if "cf_pchpo" not in merge_cached_cols(cache, HPO_CACHE, ["cf_pchpo"]):
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                preds.append(train_fold(train_df, scada_h, use_cols, fold,
                                        args.seeds, args.jobs, best))
                print(f"  [pchpo:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m["cf_pchpo"].notna().all(), "정렬 실패"
            cache["cf_pchpo"] = m["cf_pchpo"].to_numpy()
            save_cols(cache, HPO_CACHE, ["cf_pchpo"])
            print(f"  캐시 저장 → {HPO_CACHE.name}", flush=True)

        ev = make_ev(cache, champ_bias)
        champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        base = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
        print(f"\n[참고] 챔피언(hpo×dC_w65): CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}"
              f"  (기대 0.6556/0.6535)", flush=True)
        print(f"[기준] v1(pcinv×dC_w65):   CV {base['cv'][0]:.4f} | HOLD {base['hold'][0]:.4f}"
              f"  (기대 0.6530/0.6512)", flush=True)
        for b in ("cf_pcinv", "cf_hpo", "dl_cln03"):
            print(f"  corr(cf_pchpo, {b}) = {corr_on_eval(cache, 'cf_pchpo', b):.4f}", flush=True)
        print(f"  평균 이동 (pchpo − pcinv): "
              f"{(cache['cf_pchpo'].to_numpy()-cache['cf_pcinv'].to_numpy()).mean()*100:+.3f}%cap",
              flush=True)
        res = {"v1_solo": ev("cf_pcinv", None, 1.0, PP_NEW),
               "pchpo_solo": ev("cf_pchpo", None, 1.0, PP_NEW)}
        for w in (0.65, 0.70):
            res[f"pchpo_w{int(w*100)}"] = ev("cf_pchpo", "dl_cln03", w, PP_NEW)
        # 챔피언 GBM과의 혼합 (트랙이 제3 멤버로 기여하는지)
        hpov, pcv = cache["cf_hpo"].to_numpy(), cache["cf_pchpo"].to_numpy()
        for m_, mt in ((0.75, "t75"), (2 / 3, "t67"), (0.5, "t50")):
            cache[f"mx{mt}"] = m_ * hpov + (1 - m_) * pcv
            res[f"mix{mt}_w65"] = ev(f"mx{mt}", "dl_cln03", 0.65, PP_NEW)
        print_table(res, base)
        print("\n※ Δ는 **v1 대비** (트랙 내부 판정). 챔피언 대비는 위 참고 줄과 비교.")
        print(f"   챔피언 대비 최고 셀: "
              f"{max((r['cv'][0]-champ['cv'][0], n) for n, r in res.items())}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
