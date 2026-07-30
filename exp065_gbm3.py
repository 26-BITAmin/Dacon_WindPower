# ═══════════════════════════════════════════════════════
# exp-065: GBM base 구성 재검토 — LGB:XGB 비율 재스윕(신 하이퍼) + 제3 계열 HGB 추가
#
# 배경:
#   - exp-026(50:50 정점)은 122피처·구 하이퍼 조건부 결론. exp-056 신 하이퍼는
#     LGB(깊게 12잎37)·XGB(얕게 4)로 반대 방향 수렴 — 비율 재측정 근거.
#   - exp-060 관찰: "남은 GBM 다양성 후보는 다른 모델 계열뿐". CatBoost는 이 머신에서
#     DLL 로드 차단(애플리케이션 제어 정책) → sklearn HistGradientBoosting으로 대체.
#     exp-013(CatBoost 이득 없음)은 132피처·무HPO 조건부 결론 — exp-056이 반증한 유형.
#
# 블록:
#   A. members: 챔피언 파라미터로 LGB·XGB 3-seed 예측을 분리 캐시(cf_lgb/cf_xgb).
#      50:50 재조합 = cf_hpo 재현 대조(순수 재현 확인).
#   B. ratio (캐시 전용): 비율 사다리 r ∈ {0.35..0.70} × dl w{0.65, 0.70}.
#   C. search: HGB Optuna (Q4, exp-056 프로토콜). HGB는 외부 eval-set 조기종료
#      미지원이라 max_iter를 탐색 차원에 포함. 트라이얼 0 = LGB 챔피언 미러,
#      트라이얼 1 = 구 베이스라인 아날로그.
#   D. gate: 우승 HGB 3-seed·4-fold+홀드아웃(cf_hgb) → 3계열 혼합 사다리
#      cf_hpo⊕cf_hgb {5/6, 3/4, 2/3(=삼등분), 1/2} × dl_cln03 w{0.65, 0.70}.
#   판정 Δ = 보드 챔피언 구성(hpo×dC_w65, 재현 기대 CV 0.6556 / HOLD 0.6535).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp065_gbm3.py --stage all
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
from exp056_hpo import RESULT_JSON as E56_JSON, build_search_data, eval_mae_cf
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp065_hgb_best.json"
MEMBERS_CACHE = CACHE_DIR / "exp065_gbm_members.parquet"
HGB_CACHE = CACHE_DIR / "exp065_gbm_hgb.parquet"

# 트라이얼 0: exp-056 LGB 우승의 HGB 미러 / 트라이얼 1: 구 베이스라인 아날로그
MIRROR_HGB = dict(learning_rate=0.0257, max_iter=448, max_leaf_nodes=37, max_depth=12,
                  min_samples_leaf=45, l2_regularization=0.051, max_features=0.568)
BASE_HGB = dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=31, max_depth=6,
                min_samples_leaf=20, l2_regularization=1e-4, max_features=0.8)


def slice_fold(train_df, fold):
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                          (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
    return tr_all, va_all


def fold_group_data(tr_all, va_all, group, cols):
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
    if len(tr_g) == 0 or len(va_g) == 0:
        return None
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_tr = tr_g["target"].to_numpy()
    w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    return X_tr, y_tr, w_tr, X_va, va_g, cap


# ── A. members: 챔피언 LGB·XGB 분리 예측 ──
def train_members_fold(train_df, cols, fold, seeds, jobs, e56):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all, va_all = slice_fold(train_df, fold)
    rows = []
    for group in [1, 2, 3]:
        d = fold_group_data(tr_all, va_all, group, cols)
        if d is None:
            continue
        X_tr, y_tr, w_tr, X_va, va_g, cap = d
        pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s,
                                        n_jobs=jobs, verbose=-1, **e56["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                                        tree_method="hist", random_state=s, n_jobs=jobs,
                                        verbosity=0, **e56["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_lgb": pl / cap, "cf_xgb": px / cap}))
    return pd.concat(rows, ignore_index=True)


# ── C/D. HGB ──
def hgb_params(p):
    q = dict(p)
    if q.get("max_depth") == 0:
        q["max_depth"] = None
    return q


def fit_hgb(X_tr, y_tr, w_tr, p, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    m = HGB(loss="absolute_error", random_state=seed, early_stopping=False, **hgb_params(p))
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def search_hgb(data, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
            max_iter=trial.suggest_int("max_iter", 250, 1600, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 16, 128, log=True),
            max_depth=trial.suggest_categorical("max_depth", [0, 6, 8, 12]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
        )
        tt = time.time()
        preds = {}
        for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
            preds[g] = fit_hgb(X_tr, y_tr, w_tr, p, 0).predict(X_va)
        v = eval_mae_cf(data, preds)
        print(f"  #{trial.number} MAE {v:.5f} ({time.time()-tt:.0f}s) {trial.params}", flush=True)
        return v

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=65))
    study.enqueue_trial({**MIRROR_HGB, "max_depth": 12})
    study.enqueue_trial({**BASE_HGB, "max_depth": 6})
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    mir, best = study.trials[0], study.best_trial
    print(f"[HGB] 미러(트라이얼0) {mir.value:.5f} → best {best.value:.5f} "
          f"(#{best.number}) Δ {mir.value-best.value:+.5f}", flush=True)
    for t in sorted(study.trials, key=lambda t: t.value)[:5]:
        print(f"  #{t.number} {t.value:.5f} {t.params}", flush=True)
    return best.params, best.number, mir.value, best.value


def train_gate_fold_hgb(train_df, cols, fold, seeds, p):
    tr_all, va_all = slice_fold(train_df, fold)
    n_seeds = 1 if p.get("max_features", 1.0) >= 0.999 else seeds
    rows = []
    for group in [1, 2, 3]:
        d = fold_group_data(tr_all, va_all, group, cols)
        if d is None:
            continue
        X_tr, y_tr, w_tr, X_va, va_g, cap = d
        ph = np.mean([fit_hgb(X_tr, y_tr, w_tr, p, s).predict(X_va)
                      for s in range(n_seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_hgb": ph / cap}))
    return pd.concat(rows, ignore_index=True)


def cache_train_col(cache, train_df, use_cols, cache_path, colnames, trainer):
    have = merge_cached_cols(cache, cache_path, colnames)
    if all(c in have for c in colnames):
        return
    preds = []
    for fname, fold in FOLD_LIST:
        tf = time.time()
        preds.append(trainer(train_df, use_cols, fold))
        print(f"  [{'/'.join(colnames)}:{fname}] {time.time()-tf:.0f}s", flush=True)
    p = pd.concat(preds, ignore_index=True)
    m = cache[KEY].merge(p, on=KEY, how="left")
    for c in colnames:
        assert m[c].notna().all(), f"정렬 실패: {c}"
        cache[c] = m[c].to_numpy()
    save_cols(cache, cache_path, colnames)
    print(f"  캐시 저장 → {cache_path.name}", flush=True)


def corr_on_eval(cache, a, b):
    m = cache["target"].notna()
    return float(np.corrcoef(cache.loc[m, a], cache.loc[m, b])[0, 1])


def build_train_frame():
    train_df, cols, colsets = build_inputs(["spd"])
    ec = build_ec_frame(EC_WIDE, "u10", "v10", "ec")
    train_df = train_df.merge(ec, on=["forecast_kst_dtm", "kpx_group"], how="left")
    train_df["ec_dis"] = (train_df["ec_ws"] - train_df["ldaps_ws_10m"]).abs()
    train_df = merge_avail(train_df)
    return train_df, colsets["spd"] + EC_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bench", "members", "ratio", "search", "gate", "all"],
                    default="all")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-065 GBM base 구성 재검토 (비율 + HGB 제3 계열) ###", flush=True)

    train_df = use_cols = None
    if args.stage in ("bench", "members", "search", "gate", "all"):
        train_df, use_cols = build_train_frame()
        print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())

    if args.stage == "bench":
        data = build_search_data(train_df, use_cols)
        for tag, p in (("mirror", MIRROR_HGB), ("base", BASE_HGB)):
            tt = time.time()
            preds = {g: fit_hgb(d[0], d[1], d[2], p, 0).predict(d[3]) for g, d in data.items()}
            print(f"[bench:{tag}] 3그룹 fit+pred {time.time()-tt:.0f}s | "
                  f"MAE {eval_mae_cf(data, preds):.5f}", flush=True)
        print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
        return

    if args.stage in ("members", "all"):
        cache, champ_bias = load_cache_full()
        cache_train_col(cache, train_df, use_cols, MEMBERS_CACHE, ["cf_lgb", "cf_xgb"],
                        lambda td, uc, f: train_members_fold(td, uc, f, args.seeds, args.jobs, e56))
        print(f"[members 완료 {(time.time()-t0)/60:.1f}분]", flush=True)

    if args.stage in ("ratio", "all"):
        cache, champ_bias = load_cache_full()
        assert merge_cached_cols(cache, MEMBERS_CACHE, ["cf_lgb", "cf_xgb"])
        lgbv, xgbv = cache["cf_lgb"].to_numpy(), cache["cf_xgb"].to_numpy()
        mix50 = 0.5 * lgbv + 0.5 * xgbv
        diff = np.abs(mix50 - cache["cf_hpo"].to_numpy())
        print(f"\n[ratio] 재현 대조: max|50:50 − cf_hpo| = {diff.max():.6f} "
              f"(mean {diff.mean():.6f}) | corr(cf_lgb, cf_xgb) = "
              f"{corr_on_eval(cache, 'cf_lgb', 'cf_xgb'):.4f}", flush=True)
        ev = make_ev(cache, champ_bias)
        ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        print(f"챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
              f"  (기대 0.6556/0.6535)", flush=True)
        res = {}
        for r in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            col = f"mxr{int(r*100)}"
            cache[col] = r * lgbv + (1 - r) * xgbv
            res[f"r{int(r*100)}_solo"] = ev(col, None, 1.0, PP_NEW)
            res[f"r{int(r*100)}_w65"] = ev(col, "dl_cln03", 0.65, PP_NEW)
        for r in (0.40, 0.50, 0.60):
            res[f"r{int(r*100)}_w70"] = ev(f"mxr{int(r*100)}", "dl_cln03", 0.70, PP_NEW)
        print_table(res, ref)
        print(f"[ratio 완료 {(time.time()-t0)/60:.1f}분]", flush=True)

    if args.stage in ("search", "all"):
        data = build_search_data(train_df, use_cols)
        print(f"탐색 데이터 준비 — {time.time()-t0:.0f}s", flush=True)
        params, best_no, v_mir, v_best = search_hgb(data, args.trials)
        RESULT_JSON.write_text(json.dumps(dict(
            hgb=params, best_trial=best_no, mae_mirror=v_mir, mae_best=v_best), indent=1))
        print(f"우승 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        best = json.loads(RESULT_JSON.read_text())
        print(f"게이트: HGB {best['hgb']}", flush=True)
        cache, champ_bias = load_cache_full()
        assert merge_cached_cols(cache, MEMBERS_CACHE, ["cf_lgb", "cf_xgb"])
        cache_train_col(cache, train_df, use_cols, HGB_CACHE, ["cf_hgb"],
                        lambda td, uc, f: train_gate_fold_hgb(td, uc, f, args.seeds, best["hgb"]))
        for a, b in (("cf_hgb", "cf_hpo"), ("cf_hgb", "cf_lgb"), ("cf_hgb", "cf_xgb"),
                     ("cf_hgb", "dl_cln03"), ("cf_hpo", "dl_cln03")):
            print(f"  corr({a}, {b}) = {corr_on_eval(cache, a, b):.4f}", flush=True)
        ev = make_ev(cache, champ_bias)
        ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
        print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
              f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
        res = {"hgb_solo": ev("cf_hgb", None, 1.0, PP_NEW),
               "hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
        hpov, hgbv = cache["cf_hpo"].to_numpy(), cache["cf_hgb"].to_numpy()
        for m, tag in ((5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67"), (0.5, "t50")):
            col = f"tri{tag}"
            cache[col] = m * hpov + (1 - m) * hgbv
            for w in (0.65, 0.70):
                res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
        print_table(res, ref)
        d_cv = res["hgb_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res["hgb_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"\nGBM 단독 델타 (hgb−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
