# ═══════════════════════════════════════════════════════
# exp-058: DL 하이퍼파라미터 Optuna — exp-033 이래 고정(384×3·drop0.15·ep120·lr1e-3) 재탐색
#
# 근거: exp-056에서 "구시절 고정 하이퍼 = 실병목"이 GBM에서 실증(+0.0024 대칭, LB +0.00015).
#   DL도 같은 상태 — exp-033 수동 탐색(132피처 시절) 이후 348입력·정제 가중으로 바뀌었는데
#   용량 재검은 exp-034(384⊕512 앙상블)뿐.
# 핵심 설계: 목적함수 = **Q4 블렌드 Score** (0.65·cf_hpo + 0.35·DL트라이얼 → 신PP → 채점).
#   DL 단독 MAE가 아님 — "더 정확한 단독 모델 ≠ 더 좋은 파트너" 3회 실증(exp-039/047/051) 반영.
# 구조: exp-056 프로토콜 미러 — Q4 탐색(1-seed, 트라이얼 0 = 현행) → 우승 3-seed 이중 게이트.
#   손실 q60(τ0.60)·입력 spd 348·정제 가중(λ0.05) 고정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp058_dl_hpo.py --stage all --trials 25
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices, make_loss, train_one
from exp049_clean import (GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY, PP_NEW,
                          merge_avail, postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import make_xy_lam, merge_cached_cols, save_cols
from exp056_hpo import GBM_HPO_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULT_JSON = CACHE_DIR / "exp058_dlhpo_best.json"
DL_HPO_CACHE = CACHE_DIR / "exp058_dl_hpo.parquet"
SEARCH_FOLD = "fold_2024Q4"
BASELINE = dict(width=384, depth=3, p_drop=0.15, epochs=120, lr=1e-3, wd=1e-4)
W_GBM = 0.65
LAM = 0.05


def prep_fold_dl(train_df, cache, vcols, fname):
    """폴드 하나의 DL 학습·검증 텐서 + cache 행 인덱스 (1회 준비, 전 트라이얼 공유)"""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import GIDX
    fold = dict(FOLD_LIST)[fname]
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_lam(tr_all, vcols, imp, sca, LAM, fit=True)
    sel = (cache["fold"] == fname).to_numpy()
    keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
    va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                    right_on=["kpx_group", "forecast_kst_dtm"], how="left")
    assert len(va) == len(keys)
    Xva = sca.transform(imp.transform(va[vcols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    return Xtr, gtr, ytr, wtr, Xva, gva, sel


def score_q4(cache, champ_bias, dl_arr, sel):
    """Q4 행에 dl_arr을 넣고 블렌드→신PP→채점"""
    sub_cache = cache.loc[sel].reset_index(drop=True)
    sub_cache["dl_trial"] = dl_arr
    blend = W_GBM * sub_cache["cf_hpo"].to_numpy() + (1 - W_GBM) * sub_cache["dl_trial"].to_numpy()
    parts = []
    for (fname, group), sub in build_slices(sub_cache):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                                group, champ_bias[fname], **PP_NEW)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    tidy = pd.concat(parts, ignore_index=True)
    return score_from_tidy(tidy, "pred")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "gate", "all"], default="all")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--balanced", action="store_true",
                    help="NMAE 균형 제약: nmae가 기준 −0.0003 아래로 내려가면 부족분 ×5 페널티")
    args = ap.parse_args()
    global RESULT_JSON, DL_HPO_CACHE
    dl_col = "dl_hpo"
    if args.balanced:
        RESULT_JSON = RESULT_JSON.with_name("exp058_dlhpo_bal.json")
        DL_HPO_CACHE = DL_HPO_CACHE.with_name("exp058_dl_hpob.parquet")
        dl_col = "dl_hpob"
    t0 = time.time()
    print(f"### exp-058 DL HPO (목적함수 = Q4 블렌드 Score{', NMAE 균형 제약' if args.balanced else ''}) ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)
    cache, champ_bias = load_eval_frame()
    assert merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    assert merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert merge_cached_cols(cache, GBM_HPO_CACHE, ["cf_hpo"])
    loss_fn = make_loss("q60")

    if args.stage in ("search", "all"):
        import optuna
        import torch
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        Xtr, gtr, ytr, wtr, Xva, gva, sel = prep_fold_dl(train_df, cache, vcols, SEARCH_FOLD)
        base_dl = cache.loc[sel, "dl_cln03"].to_numpy()
        s_base = score_q4(cache, champ_bias, base_dl, sel)
        n_base = s_base[1]
        print(f"탐색 준비 완료 — Q4 기준(현행 dl_cln03 3-seed): Score {s_base[0]:.4f} (n {n_base:.4f})", flush=True)

        def objective(trial):
            cfg = dict(
                width=trial.suggest_categorical("width", [256, 384, 512, 640]),
                depth=trial.suggest_int("depth", 2, 5),
                p_drop=trial.suggest_float("p_drop", 0.0, 0.35),
                epochs=trial.suggest_categorical("epochs", [80, 120, 160, 200]),
                lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
                wd=trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            )
            m = train_one(Xtr, gtr, ytr, wtr, cfg, loss_fn, 0)
            from exp033_dl import predict
            dl = predict(m, Xva, gva)
            del m
            torch.cuda.empty_cache()
            s = score_q4(cache, champ_bias, dl, sel)
            trial.set_user_attr("nmae", float(s[1])); trial.set_user_attr("ficr", float(s[2]))
            if args.balanced:
                return s[0] - 5.0 * max(0.0, (n_base - 0.0003) - s[1])
            return s[0]

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial(BASELINE)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        base_t, best = study.trials[0], study.best_trial
        print(f"[DL] baseline(1-seed) {base_t.value:.4f} → best {best.value:.4f} "
              f"(trial {best.number}, Δ {best.value-base_t.value:+.4f})", flush=True)
        for t in sorted(study.trials, key=lambda t: -t.value)[:5]:
            print(f"  #{t.number} {t.value:.4f} (n {t.user_attrs['nmae']:.4f} f {t.user_attrs['ficr']:.4f}) {t.params}", flush=True)
        RESULT_JSON.write_text(json.dumps(best.params, indent=1))
        print(f"우승 저장 → {RESULT_JSON.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    if args.stage in ("gate", "all"):
        import torch
        from exp033_dl import predict
        best = json.loads(RESULT_JSON.read_text())
        print(f"게이트: {best}", flush=True)
        have = merge_cached_cols(cache, DL_HPO_CACHE, [dl_col])
        if dl_col not in have:
            out = np.full(len(cache), np.nan)
            for fname, fold in FOLD_LIST:
                tf = time.time()
                Xtr, gtr, ytr, wtr, Xva, gva, sel = prep_fold_dl(train_df, cache, vcols, fname)
                preds = []
                for s in range(args.seeds):
                    m = train_one(Xtr, gtr, ytr, wtr, best, loss_fn, s)
                    preds.append(predict(m, Xva, gva))
                    del m
                    torch.cuda.empty_cache()
                out[sel] = np.mean(preds, axis=0)
                print(f"  [dlhpo:{fname}] {time.time()-tf:.0f}s", flush=True)
            cache[dl_col] = out
            save_cols(cache, DL_HPO_CACHE, [dl_col])
            print(f"  캐시 저장 → {DL_HPO_CACHE.name}", flush=True)

        # 평가
        fold_names = [fn for fn, _ in FOLD_LIST]
        hold = HOLD_FOLD["name"]
        cv_folds = [fn for fn in fold_names if fn != hold]
        slices = build_slices(cache)

        def ev(gbm_col, dl_col, w):
            blend = w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy()
            parts = []
            for (fname, group), sub in slices:
                cap = CAPACITY_KWH[f"kpx_group_{group}"]
                pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                        sub["pA"].to_numpy(), group, champ_bias[fname], **PP_NEW)
                parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
            tidy = pd.concat(parts, ignore_index=True)
            return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                        hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                        folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

        r_ref = ev("cf_hpo", "dl_cln03", 0.65)
        print(f"\n보드챔피언(hpo×dC_w65): CV {r_ref['cv'][0]:.4f} (1-NMAE {r_ref['cv'][1]:.4f} "
              f"FiCR {r_ref['cv'][2]:.4f}) | HOLD {r_ref['hold'][0]:.4f}  (기대 0.6556/0.6535)")
        res = {}
        for w in (0.70, 0.65, 0.60, 0.55):
            res[f"hpoG×{dl_col[3:]}_w{int(w*100)}"] = ev("cf_hpo", dl_col, w)
        ch_cv, ch_hd = r_ref["cv"][0], r_ref["hold"][0]
        ch_n, ch_f = r_ref["cv"][1], r_ref["cv"][2]
        print("\n" + "=" * 110)
        print(f"{'변형':<16}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}"
              f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정")
        for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
            dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
            dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
            fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
            mag = dcv > 0.001 and dhd > 0.001
            cons = all(x > -0.0002 for x in fold_d)
            both = dn_ > 0 and df_ > 0
            verdict = ("★3중충족" if (mag and cons and both) else
                       ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
            print(f"{name:<16}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
                  f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        print(f"\n--- fold별 ΔCV (ref 대비) ---")
        print(f"{'변형':<16}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds))
        for name, r in res.items():
            fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
            print(f"{name:<16}" + "".join(f"{x:>+10.4f}" for x in fold_d))
        # DL 단독 진단
        evm = cache["target"].notna().to_numpy()
        caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        acf = cache["target"].to_numpy() / caps
        ev2 = evm & (cache["target"].to_numpy() >= caps * 0.10)
        e_g = cache["cf_hpo"].to_numpy() - acf
        print("\n--- DL 단독 (평가시간): MAE | signed | corr_gbm ---")
        for c in ["dl_cln03", dl_col]:
            e = cache[c].to_numpy() - acf
            print(f"  {c}: {np.abs(e[ev2]).mean():.4f} | {e[ev2].mean():+.4f} | "
                  f"{np.corrcoef(e_g[ev2], e[ev2])[0, 1]:.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
