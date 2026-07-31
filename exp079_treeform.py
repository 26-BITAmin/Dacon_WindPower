# ═══════════════════════════════════════════════════════
# exp-079: LGB 트리 형태 축 — HPO 공간 밖의 부스터 구조 변형 3종
#
# 배경:
#   - exp-056/063 HPO는 표준 GBDT 하이퍼(lr·잎·깊이·정칙화·샘플링 비율)만 탐색.
#     부스터 "형태" 축(linear_tree·DART)과 bagging 활성 스위치는 공간 밖.
#   - (a) bagf: exp-056 탐색 objective는 bagging_freq=1을 하드코딩해 행 bagging(0.69)을
#     켠 채 최적화했으나, 저장 JSON에는 suggest 파라미터만 담겨 게이트/챔피언 학습은
#     bagging_freq 부재 = 행 bagging 비활성 (LightGBM은 freq=0이면 fraction 무시).
#     탐색-배포 불일치 — 정합 회복이 유일 변경.
#   - (b) linl: linear_tree=True — 잎 상수 → 잎 선형모델 (조각 상수 → 조각 선형).
#     지배 오차가 일중 모양(exp-078 75.3%)이라 매끄러운 반응함수가 계단 아티팩트를
#     줄일 후보. 구현 교체(HGB corr 0.9985)와 달리 함수족 자체가 바뀜.
#   - (c) dart: boosting_type=dart — 트리 드롭아웃으로 앞 트리 지배 완화.
#   전부 GBM측 정확도형 개입 (exp-049/056에서 대칭 전이 실증된 유형). DL 무변경.
#
# 프로토콜 (exp-065 HGB 블록 준용):
#   smoke: Q4 폴드·1-seed·고정 트리 수(배포 미러) — MAE·signed·corr·시간 → go/no-go
#   gate:  통과 변형만 3-seed·4-fold+홀드아웃 LGB만 재학습 (XGB는 exp-065 멤버 캐시
#          cf_xgb 재사용) → 50:50 재조합 = cf_{tag}. 판정 Δ = 보드 챔피언(hpo×dC_w65,
#          재현 기대 CV 0.6556 / HOLD 0.6535). 반half 혼합(변형25·현행25·XGB50)도 측정.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp079_treeform.py --stage smoke
#       C:\miniconda3\python.exe -X utf8 exp079_treeform.py --stage gate --variants bagf,linl
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, build_search_data, eval_mae_cf
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import (MEMBERS_CACHE, build_train_frame, corr_on_eval,
                         fold_group_data, slice_fold)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# extra = 챔피언 LGB 파라미터 위에 덮는 변경분 (그 외 전부 동일)
VARIANTS = {
    "bagf":  dict(extra=dict(bagging_freq=1)),
    "linl":  dict(extra=dict(linear_tree=True)),
    "linl1": dict(extra=dict(linear_tree=True, linear_lambda=1.0)),
    "dart":  dict(extra=dict(boosting_type="dart", drop_rate=0.1, skip_drop=0.5)),
    "dart2": dict(extra=dict(boosting_type="dart", drop_rate=0.1, skip_drop=0.5), n_mult=2),
}


def fit_lgb_variant(X_tr, y_tr, w_tr, e56, seed, jobs, extra=None, n_mult=1):
    import lightgbm as lgb
    p = dict(e56["lgb"])
    if extra:
        p.update(extra)
    m = lgb.LGBMRegressor(n_estimators=int(round(e56["lgb_n"] * n_mult)), objective="mae",
                          random_state=seed, n_jobs=jobs, verbose=-1, **p)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def stage_smoke(train_df, use_cols, e56, jobs):
    data = build_search_data(train_df, use_cols)
    print(f"[smoke] Q4 탐색 데이터 준비 완료", flush=True)
    base_preds = {}

    def run(tag, extra=None, n_mult=1):
        tt = time.time()
        preds, fails = {}, None
        for g, (X_tr, y_tr, w_tr, X_va, y_va, eval_w, cap) in data.items():
            try:
                preds[g] = fit_lgb_variant(X_tr, y_tr, w_tr, e56, 0, jobs, extra, n_mult)\
                    .predict(X_va)
            except Exception as e:
                fails = f"{type(e).__name__}: {e}"
                break
        if fails:
            print(f"[smoke:{tag}] 실행 불가 — {fails}", flush=True)
            return None
        mae = eval_mae_cf(data, preds)
        sgn = np.mean([((preds[g] - d[4])[d[5] > 0] / d[6]).mean()
                       for g, d in data.items()])
        line = f"[smoke:{tag}] MAE {mae:.5f} | signed {sgn:+.4f} | {time.time()-tt:.0f}s"
        if base_preds:
            cat_v = np.concatenate([preds[g] for g in sorted(preds)])
            cat_b = np.concatenate([base_preds[g] for g in sorted(base_preds)])
            line += f" | corr(vs mirror) {np.corrcoef(cat_v, cat_b)[0, 1]:.4f}"
        print(line, flush=True)
        return preds

    base_preds_ = run("mirror")           # 배포 재현 기준점
    if base_preds_ is None:
        raise SystemExit("mirror 실패 — 환경 이상")
    base_preds.update(base_preds_)
    for tag, spec in VARIANTS.items():
        run(tag, spec.get("extra"), spec.get("n_mult", 1))


def train_variant_fold(train_df, cols, fold, seeds, jobs, e56, spec, colname):
    tr_all, va_all = slice_fold(train_df, fold)
    rows = []
    for group in [1, 2, 3]:
        d = fold_group_data(tr_all, va_all, group, cols)
        if d is None:
            continue
        X_tr, y_tr, w_tr, X_va, va_g, cap = d
        pv = np.mean([fit_lgb_variant(X_tr, y_tr, w_tr, e56, s, jobs,
                                      spec.get("extra"), spec.get("n_mult", 1)).predict(X_va)
                      for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  colname: pv / cap}))
    return pd.concat(rows, ignore_index=True)


def stage_gate(train_df, use_cols, e56, jobs, seeds, tags):
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, MEMBERS_CACHE, ["cf_lgb", "cf_xgb"])
    for tag in tags:
        colname = f"lg_{tag}"
        cpath = CACHE_DIR / f"exp079_gbm_{tag}.parquet"
        have = merge_cached_cols(cache, cpath, [colname])
        if colname not in have:
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                preds.append(train_variant_fold(train_df, use_cols, fold, seeds, jobs,
                                                e56, VARIANTS[tag], colname))
                print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[colname].notna().all(), f"정렬 실패: {colname}"
            cache[colname] = m[colname].to_numpy()
            save_cols(cache, cpath, [colname])
            print(f"  캐시 저장 → {cpath.name}", flush=True)

    xgbv, lgbv = cache["cf_xgb"].to_numpy(), cache["cf_lgb"].to_numpy()
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag in tags:
        colname = f"lg_{tag}"
        lv = cache[colname].to_numpy()
        for a, b in ((colname, "cf_lgb"), (colname, "cf_hpo"), (colname, "dl_cln03")):
            print(f"  corr({a}, {b}) = {corr_on_eval(cache, a, b):.4f}", flush=True)
        print(f"  평균 이동 ({colname} − cf_lgb): "
              f"{(lv - lgbv).mean()*100:+.3f}%cap", flush=True)
        cache[f"g_{tag}"] = 0.5 * lv + 0.5 * xgbv                     # 변형 LGB 전량 교체
        cache[f"gh_{tag}"] = 0.25 * lv + 0.25 * lgbv + 0.5 * xgbv     # 반half 혼합
        res[f"{tag}_solo"] = ev(f"g_{tag}", None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(f"g_{tag}", "dl_cln03", w, PP_NEW)
        res[f"{tag}h_w65"] = ev(f"gh_{tag}", "dl_cln03", 0.65, PP_NEW)
    print_table(res, ref)
    for tag in tags:
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"GBM 단독 델타 ({tag}−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--variants", default="bagf")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-079 LGB 트리 형태 축 (bagf / linear_tree / DART) ###", flush=True)

    train_df, use_cols = build_train_frame()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())

    if args.stage == "smoke":
        stage_smoke(train_df, use_cols, e56, args.jobs)
    else:
        tags = [t.strip() for t in args.variants.split(",") if t.strip()]
        for t in tags:
            assert t in VARIANTS, f"미정의 변형: {t}"
        stage_gate(train_df, use_cols, e56, args.jobs, args.seeds, tags)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
