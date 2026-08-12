# ═══════════════════════════════════════════════════════
# pd14: 트랙 D 8라운드 — 잔차 GBM 구성 변경 (PD-14)
#
# 배경: PD-10 LB 분해에서 손실의 79%가 NMAE = 잔차 GBM 담당.
#   현 잔차 GBM은 직접 타깃용 e56 하이퍼(LGB 448 + XGB 585 50:50)를 그대로 차용 —
#   잔차 타깃은 분포·스케일이 달라 계열/하이퍼 재검 근거 명확 (사용자 지시 2026-08-12).
#
# 변경점 = 잔차 GBM만. DL base = PD-03 anws_hub 캐시 그대로 재사용
#   (pd03_dltrain.parquet 학습행 교차적합 · pd03_preds.parquet pdb_anws_hub 검증행).
#
# 1단계 --stage fam: 계열 교체 스윕 (각 3시드, 가중·타깃·피처 동일)
#   lgb  LGB e56 단독            xgb  XGB e56 단독
#   cat  CatBoost MAE (GPU)      hgb  sklearn HGB absolute_error (CPU)
#   ctrl = 0.5·lgb + 0.5·xgb (후처리 도출 = PD-03 잔차 스테이지 재현 — draw 검증)
# 2단계 --stage opt --fam X: Optuna (1시드·CV 4폴드·제출 작동점 목적) → best 3시드 재적합
# 채점 --stage eval: 단독 + 혼합(쌍/삼/사 균등) 전수, 기준 = PD-03 anws_hub 저장분
#
# 실행(루트에서):
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd14_resgbm.py" --stage fam
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd14_resgbm.py" --stage eval
#   C:\miniconda3\python.exe -X utf8 "open (1)/testing/curve_track/pd14_resgbm.py" --stage opt --fam lgb --trials 40
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "wind_input"))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import FOLD_LIST, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import avail_factor, postprocess_flex
from exp050_lambda_sweep import save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full
from exp073_gate import ENS_CACHE

from wi01_bivgate import NOSNAP, SCALES

from pd00_inputs import load_frame_cached, load_pools
from pd01_gate import fill_from

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HOLD = HOLD_FOLD["name"]
KEY = ["kpx_group", "forecast_kst_dtm"]
DL_TRAIN = HERE.parent / "pd03_dltrain.parquet"
DL_PREDS = HERE.parent / "pd03_preds.parquet"


def store_of(tag):
    return HERE.parent / (f"pd14_preds_{tag}.parquet" if tag else "pd14_preds.parquet")


def fill_all_stores(cache, cols):
    """모든 pd14_preds*.parquet에서 취합 (프로세스별 태그 저장 파일 대응)."""
    for f in sorted(HERE.parent.glob("pd14_preds*.parquet")):
        fill_from(cache, f, cols)
CSV_OUT = HERE.parent / "pd14_results.csv"
OPT_JSON = HERE.parent / "pd14_opt.json"
FAMS = ("lgb", "xgb", "cat", "hgb")


def fit_family(fam, tr_g, va_g, cols, e56, params=None, seeds=3, jobs=8):
    """잔차 회귀 1계열. 반환 = va 잔차 예측(kWh). params=None이면 기본 구성."""
    cap = CAPACITY_KWH[f"kpx_group_{int(tr_g['kpx_group'].iloc[0])}"]
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_res = tr_g["target"].to_numpy() - tr_g["pd_dl_cf"].to_numpy() * cap
    w_tr = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    preds = []
    for s in range(seeds):
        if fam == "lgb":
            import lightgbm as lgb
            kw = params or dict(n_estimators=e56["lgb_n"], objective="mae", **e56["lgb"])
            m = lgb.LGBMRegressor(random_state=s, n_jobs=jobs, verbose=-1,
                                  device="gpu", **kw)
        elif fam == "xgb":
            import xgboost as xgbm
            kw = params or dict(n_estimators=e56["xgb_n"],
                                objective="reg:absoluteerror", **e56["xgb"])
            m = xgbm.XGBRegressor(tree_method="hist", device="cuda", random_state=s,
                                  n_jobs=jobs, verbosity=0, **kw)
        elif fam == "cat":
            from catboost import CatBoostRegressor
            kw = params or dict(iterations=800, learning_rate=0.06, depth=8,
                                loss_function="MAE")
            m = CatBoostRegressor(random_seed=s, verbose=0, task_type="GPU",
                                  devices="0", allow_writing_files=False, **kw)
        elif fam == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            kw = params or dict(max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
                                loss="absolute_error")
            m = HistGradientBoostingRegressor(random_state=s, **kw)
        m.fit(X_tr, y_res, sample_weight=w_tr)
        preds.append(np.asarray(m.predict(X_va), dtype=float))
    return np.mean(preds, axis=0)


def load_all():
    """공통 데이터 적재: 프레임·풀·캐시·DL base(hub) 예측."""
    dl_pool, gbm_pool, env_cols = load_pools()
    e56 = json.loads(E56_JSON.read_text())
    train_df = load_frame_cached()
    gbm_cols = gbm_pool + ["pd_dl_cf"]
    cache, champ_bias = load_cache_full()
    fill_from(cache, ENS_CACHE, ["cf_ens3"])
    for f in sorted((HERE.parents[1] / "wind_input").glob("wi09_dl_es*.parquet")):
        fill_from(cache, f, ["dl_es"])
    fill_from(cache, DL_PREDS, ["pdd_anws_hub", "pdb_anws_hub"])
    assert cache["cf_ens3"].notna().all() and cache["dl_es"].notna().all()
    assert cache["pdb_anws_hub"].notna().all(), "PD-03 base 캐시 미완"
    dltrain = pd.read_parquet(DL_TRAIN)
    dltrain = dltrain[dltrain["cell"] == "anws_hub"]
    return train_df, gbm_cols, e56, cache, champ_bias, dltrain


def fold_frames(train_df, cache, dltrain, fname, fold):
    """폴드별 (lab, vf): lab = 학습행+pd_dl_cf(교차적합), vf = 검증행+pd_dl_cf(base)."""
    dtr = dltrain.loc[dltrain["fold"] == fname, KEY + ["pd_dl_cf"]]
    lab = train_df.loc[(train_df["forecast_kst_dtm"] <= fold["train_end"])
                       & train_df["target"].notna()]
    lab = lab.merge(dtr, on=KEY, how="inner")
    assert len(lab) == len(dtr), f"dltrain 병합 {len(lab)} != {len(dtr)}"
    sel = (cache["fold"] == fname).to_numpy()
    keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
    vf = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                    right_on=KEY, how="left")
    assert len(vf) == len(keys), "정렬 실패"
    vf["pd_dl_cf"] = cache.loc[sel, "pdb_anws_hub"].to_numpy()
    return lab, vf, sel


def run_families(args, data):
    train_df, gbm_cols, e56, cache, champ_bias, dltrain = data
    fams = [f for f in args.fams.split(",") if f]
    assert all(f in FAMS for f in fams), fams
    store = store_of(args.tag)
    cols_new = [f"pdr12_{f}" for f in fams]
    fill_all_stores(cache, cols_new)
    for fname, fold in FOLD_LIST:
        need = [f for f in fams
                if f"pdr12_{f}" not in cache.columns
                or cache.loc[cache["fold"] == fname, f"pdr12_{f}"].isna().any()]
        if not need:
            continue
        lab, vf, sel = fold_frames(train_df, cache, dltrain, fname, fold)
        for fam in need:
            col = f"pdr12_{fam}"
            if col not in cache.columns:
                cache[col] = np.nan
            tf = time.time()
            res_va = np.full(len(vf), np.nan)
            for g in [1, 2, 3]:
                tr_g = lab.loc[lab["kpx_group"] == g]
                vm = (vf["kpx_group"] == g).to_numpy()
                cap = CAPACITY_KWH[f"kpx_group_{g}"]
                res_va[vm] = fit_family(fam, tr_g, vf.loc[vm], gbm_cols, e56,
                                        seeds=args.seeds) / cap
            v = cache[col].to_numpy()
            v[sel] = res_va
            cache[col] = v
            save_cols(cache, store, [c for c in cols_new if c in cache.columns])
            print(f"  [{fam}:{fname}] {time.time()-tf:.0f}s "
                  f"(|잔차| 평균 {np.nanmean(np.abs(res_va))*100:.2f}%cap)", flush=True)


def make_scorers(cache, champ_bias):
    slices = build_slices(cache)

    def pp_of(raw, scaled):
        out = np.full(len(cache), np.nan)
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            p = postprocess_flex(raw[sub.index], cap, sub["ws"].to_numpy(),
                                 sub["pA"].to_numpy(), group, champ_bias[fname],
                                 **NOSNAP)
            out[sub.index] = np.clip(p * SCALES[group], 0, cap) if scaled else p
        return out

    def sc_of(pred):
        t = pd.DataFrame({"fold": cache["fold"], "group": cache["group"],
                          "forecast_kst_dtm": cache["forecast_kst_dtm"],
                          "target": cache["target"], "pred": pred})
        c = t[t["fold"] != HOLD]
        return score_from_tidy(c, "pred"), score_from_tidy(t[t["fold"] == HOLD], "pred")

    return pp_of, sc_of


def run_eval(args, data):
    from itertools import combinations
    train_df, gbm_cols, e56, cache, champ_bias, dltrain = data
    opt_cols = [f"pdr12_opt{f}" for f in FAMS]
    fill_all_stores(cache, [f"pdr12_{f}" for f in FAMS] + opt_cols)
    miss = [f for f in FAMS if f"pdr12_{f}" not in cache.columns
            or cache[f"pdr12_{f}"].isna().any()]
    assert not miss, f"미완성 계열: {miss}"
    pp_of, sc_of = make_scorers(cache, champ_bias)
    base = cache["pdb_anws_hub"].to_numpy()
    res = {f: cache[f"pdr12_{f}"].to_numpy() for f in FAMS}
    variants = {f: res[f] for f in FAMS}
    variants["ctrl(lgb+xgb)"] = 0.5 * res["lgb"] + 0.5 * res["xgb"]
    for r in (2, 3, 4):
        for combo in combinations(FAMS, r):
            if combo == ("lgb", "xgb"):
                continue
            variants["+".join(combo)] = np.mean([res[f] for f in combo], axis=0)
    opt_res = {c.replace("pdr12_", ""): cache[c].to_numpy() for c in opt_cols
               if c in cache.columns and cache[c].notna().all()}
    variants.update(opt_res)
    if "optxgb" in opt_res:
        variants["lgb+optxgb"] = 0.5 * res["lgb"] + 0.5 * opt_res["optxgb"]
    if "optlgb" in opt_res:
        variants["optlgb+xgb"] = 0.5 * opt_res["optlgb"] + 0.5 * res["xgb"]
    if {"optlgb", "optxgb"} <= opt_res.keys():
        variants["optlgb+optxgb"] = 0.5 * opt_res["optlgb"] + 0.5 * opt_res["optxgb"]
    mm = cache["target"].notna().to_numpy()
    champ_raw = 0.65 * cache["cf_ens3"].to_numpy() + 0.35 * cache["dl_es"].to_numpy()
    rows = []
    for scaled in (False, True):
        tag = "제출 작동점 ×.98/.98/.97" if scaled else "미스케일"
        print("\n" + "=" * 100, flush=True)
        print(f"═══ {tag} | 기준 = PD-03 anws_hub 저장분 (챔피언 행은 참고 전용) ═══",
              flush=True)
        cref = sc_of(pp_of(champ_raw, scaled))
        bref = sc_of(pp_of(cache["pdd_anws_hub"].to_numpy(), scaled))
        print(f"  {'구성':<22}{'CV':>9}{'Δhub':>9}{'Δn':>9}{'Δf':>9}{'HOLD':>9}"
              f"{'corr(champ)':>12}", flush=True)
        print(f"  {'(참고) champ w65':<22}{cref[0][0]:>9.4f}"
              f"{cref[0][0]-bref[0][0]:>+9.4f}{cref[0][1]-bref[0][1]:>+9.4f}"
              f"{cref[0][2]-bref[0][2]:>+9.4f}{cref[1][0]:>9.4f}", flush=True)
        print(f"  {'anws_hub (PD-03 저장)':<22}{bref[0][0]:>9.4f}{'—':>9}{'—':>9}"
              f"{'—':>9}{bref[1][0]:>9.4f}", flush=True)
        for name, rv in variants.items():
            s, sh = sc_of(pp_of(base + rv, scaled)), None
            s, sh = s[0], s[1]
            r = np.corrcoef((base + rv)[mm], champ_raw[mm])[0, 1]
            print(f"  {name:<22}{s[0]:>9.4f}{s[0]-bref[0][0]:>+9.4f}"
                  f"{s[1]-bref[0][1]:>+9.4f}{s[2]-bref[0][2]:>+9.4f}"
                  f"{sh[0]:>9.4f}{r:>12.4f}", flush=True)
            rows.append(dict(scaled=scaled, name=name, cv=s[0], hold=sh[0],
                             d_hub=s[0]-bref[0][0], d_n=s[1]-bref[0][1],
                             d_f=s[2]-bref[0][2], corr_champ=r))
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n상세 → {CSV_OUT.name}")


def suggest_params(fam, tr):
    if fam == "lgb":
        obj = tr.suggest_categorical("objective", ["mae", "huber"])
        p = dict(objective=obj,
                 n_estimators=tr.suggest_int("n_estimators", 200, 1400),
                 learning_rate=tr.suggest_float("learning_rate", 0.01, 0.15, log=True),
                 num_leaves=tr.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=tr.suggest_int("min_child_samples", 5, 200, log=True),
                 feature_fraction=tr.suggest_float("feature_fraction", 0.3, 1.0),
                 bagging_fraction=tr.suggest_float("bagging_fraction", 0.5, 1.0),
                 bagging_freq=tr.suggest_int("bagging_freq", 0, 10),
                 lambda_l1=tr.suggest_float("lambda_l1", 1e-4, 10, log=True),
                 lambda_l2=tr.suggest_float("lambda_l2", 1e-4, 10, log=True))
        if obj == "huber":
            p["alpha"] = tr.suggest_float("huber_alpha", 0.5, 5.0, log=True)
        return p
    if fam == "xgb":
        return dict(objective="reg:absoluteerror",
                    n_estimators=tr.suggest_int("n_estimators", 200, 1400),
                    learning_rate=tr.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    max_depth=tr.suggest_int("max_depth", 4, 12),
                    min_child_weight=tr.suggest_float("min_child_weight", 0.5, 50, log=True),
                    subsample=tr.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=tr.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_alpha=tr.suggest_float("reg_alpha", 1e-4, 10, log=True),
                    reg_lambda=tr.suggest_float("reg_lambda", 1e-4, 10, log=True))
    if fam == "cat":
        return dict(loss_function="MAE",
                    iterations=tr.suggest_int("iterations", 300, 1500),
                    learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    depth=tr.suggest_int("depth", 4, 10),
                    l2_leaf_reg=tr.suggest_float("l2_leaf_reg", 0.5, 30, log=True))
    if fam == "hgb":
        return dict(loss="absolute_error",
                    max_iter=tr.suggest_int("max_iter", 200, 1200),
                    learning_rate=tr.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    max_leaf_nodes=tr.suggest_int("max_leaf_nodes", 15, 255, log=True),
                    min_samples_leaf=tr.suggest_int("min_samples_leaf", 5, 200, log=True),
                    l2_regularization=tr.suggest_float("l2_regularization", 1e-4, 10,
                                                       log=True))


def run_opt(args, data):
    import optuna
    train_df, gbm_cols, e56, cache, champ_bias, dltrain = data
    fam = args.fam
    assert fam in FAMS
    pp_of, sc_of = make_scorers(cache, champ_bias)
    base = cache["pdb_anws_hub"].to_numpy()
    cv_folds = [(fn, fd) for fn, fd in FOLD_LIST if fn != HOLD]
    frames = {fn: fold_frames(train_df, cache, dltrain, fn, fd)
              for fn, fd in cv_folds}
    cap_of = {g: CAPACITY_KWH[f"kpx_group_{g}"] for g in [1, 2, 3]}
    hub_cv = sc_of(pp_of(cache["pdd_anws_hub"].to_numpy(), True))[0][0]
    jpath = str(HERE.parent / "pd14_optuna_journal.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            jpath, lock_obj=optuna.storages.journal.JournalFileOpenLock(jpath)))

    def objective(tr):
        params = suggest_params(fam, tr)
        pred = base.copy()
        for fn, (lab, vf, sel) in frames.items():
            res_va = np.full(len(vf), np.nan)
            for g in [1, 2, 3]:
                tr_g = lab.loc[lab["kpx_group"] == g]
                vm = (vf["kpx_group"] == g).to_numpy()
                res_va[vm] = fit_family(fam, tr_g, vf.loc[vm], gbm_cols, e56,
                                        params=params, seeds=1) / cap_of[g]
            pred[np.where(sel)[0]] = base[sel] + res_va
        cv = sc_of(pp_of(pred, True))[0][0]
        print(f"  trial {tr.number:>3} cv {cv:.4f} (Δhub {cv-hub_cv:+.4f}) {params}",
              flush=True)
        return cv

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=f"pd14_{fam}", storage=storage,
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=args.wseed))
    if args.trials > 0:
        study.optimize(objective, n_trials=args.trials)
    if not args.refit:
        print(f"워커 종료 (누적 trial {len(study.trials)}) — refit은 --refit 호출에서",
              flush=True)
        return
    best = study.best_trial
    print(f"\nbest trial {best.number}: cv {best.value:.4f} "
          f"(Δhub {best.value-hub_cv:+.4f}) [누적 {len(study.trials)}]\n{best.params}",
          flush=True)
    saved = json.loads(OPT_JSON.read_text()) if OPT_JSON.exists() else {}
    saved[fam] = dict(cv_1seed=best.value, d_hub=best.value - hub_cv,
                      params=best.params)
    OPT_JSON.write_text(json.dumps(saved, indent=1))

    # best 구성 3시드 재적합 (전 5폴드) → pdr12_opt{fam} (전용 저장 파일)
    params = suggest_params(fam, optuna.trial.FixedTrial(best.params))
    col = f"pdr12_opt{fam}"
    store = store_of(f"opt{fam}")
    cache[col] = np.nan
    for fname, fold in FOLD_LIST:
        lab, vf, sel = fold_frames(train_df, cache, dltrain, fname, fold)
        res_va = np.full(len(vf), np.nan)
        for g in [1, 2, 3]:
            tr_g = lab.loc[lab["kpx_group"] == g]
            vm = (vf["kpx_group"] == g).to_numpy()
            res_va[vm] = fit_family(fam, tr_g, vf.loc[vm], gbm_cols, e56,
                                    params=params, seeds=args.seeds) / cap_of[g]
        v = cache[col].to_numpy()
        v[sel] = res_va
        cache[col] = v
        save_cols(cache, store, [col])
        print(f"  [opt{fam}:{fname}] 완료", flush=True)
    for scaled in (False, True):
        s = sc_of(pp_of(base + cache[col].to_numpy(), scaled))
        b = sc_of(pp_of(cache["pdd_anws_hub"].to_numpy(), scaled))
        print(f"[opt{fam} {'제출' if scaled else '미스케일'}] CV {s[0][0]:.4f} "
              f"(Δhub {s[0][0]-b[0][0]:+.4f} · Δn {s[0][1]-b[0][1]:+.4f} · "
              f"Δf {s[0][2]-b[0][2]:+.4f}) HOLD {s[1][0]:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fam", "eval", "opt"], default="fam")
    ap.add_argument("--fams", type=str, default=",".join(FAMS))
    ap.add_argument("--fam", type=str, default="lgb")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--wseed", type=int, default=0)
    ap.add_argument("--refit", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"### pd14: 잔차 GBM 구성 변경 — stage={args.stage} ###", flush=True)
    data = load_all()
    print(f"적재 완료 — {time.time()-t0:.0f}s", flush=True)
    if args.stage == "fam":
        run_families(args, data)
    elif args.stage == "eval":
        run_eval(args, data)
    else:
        run_opt(args, data)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
