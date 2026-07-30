# ═══════════════════════════════════════════════════════
# exp-066: 학습 샘플 최근성(시대) 가중 — GBM base, half-life 사다리
#
# 배경:
#   - exp-055: 터빈 단위 신호에 시간 비지속 성분 실증 (검증 시점이 학습에서 멀수록
#     단조 감쇠·역전). exp-059/064: 2025 전역 하향 드리프트 실증 (LB 프로브).
#   - 그런데 학습은 2022~2024를 동일 가중으로 사용 중 — 오래된 표본의 다운웨이트는
#     미시도 축 (exp-016은 발전량 비례, exp-049/050은 가용성 가중 — 시간축 아님).
#   - 순수 정확도형(입력정보·평균 이동 없음) 후보. GBM만 변경, DL(dl_cln03) 고정
#     (exp-054/056 선례).
#
# 구성:
#   w_tr = [저CF 0.05 × avail cln03] × 0.5^(age_years / hl), age = train_end − t.
#   총 가중 질량은 그룹×폴드별로 챔피언과 동일하게 재정규화 (모양만 변경 —
#   XGB min_child_weight 등 가중 총량 상호작용 배제).
#   사다리 hl ∈ {1, 2, 4}년 → 각각 3-seed·4-fold+홀드아웃 게이트 (챔피언 파라미터),
#   dl_cln03 블렌드 w{0.65, 0.70} + GBM 단독. 판정 Δ = 보드 챔피언(hpo×dC_w65).
#   상향 이동 검출: mean(cf_rc − cf_hpo) %cap 보고 (판별 규칙 — 평균을 올리는
#   변경은 게이트 통과와 무관하게 2025 역전).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp066_recency.py --hl 1 2 4
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_fold_recency(train_df, cols, fold, seeds, jobs, e56, hl_years, colname):
    import lightgbm as lgb
    import xgboost as xgbm
    tr_all, va_all = slice_fold(train_df, fold)
    train_end = pd.Timestamp(fold["train_end"])
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
        y_tr = tr_g["target"].to_numpy()
        w_base = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        age = (train_end - tr_g["forecast_kst_dtm"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
        tf = 0.5 ** (age / hl_years)
        w_tr = w_base * tf
        w_tr *= w_base.sum() / w_tr.sum()   # 총 질량 = 챔피언과 동일 (모양만 변경)
        pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s,
                                        n_jobs=jobs, verbose=-1, **e56["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                                        tree_method="hist", random_state=s, n_jobs=jobs,
                                        verbosity=0, **e56["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  colname: (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hl", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-066 최근성 가중 GBM (half-life 사다리) ###", flush=True)

    train_df, use_cols = build_train_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()

    tags = []
    for hl in args.hl:
        tag = f"rc{str(hl).replace('.0', '').replace('.', 'p')}"
        col = f"cf_{tag}"
        tags.append((hl, tag, col))
        cpath = CACHE_DIR / f"exp066_gbm_{tag}.parquet"
        have = merge_cached_cols(cache, cpath, [col])
        if col not in have:
            print(f"[hl={hl}y] 게이트 학습 시작", flush=True)
            preds = []
            for fname, fold in FOLD_LIST:
                tf_ = time.time()
                preds.append(train_fold_recency(train_df, use_cols, fold, args.seeds,
                                                args.jobs, e56, hl, col))
                print(f"  [{tag}:{fname}] {time.time()-tf_:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    hpov = cache["cf_hpo"].to_numpy()
    for hl, tag, col in tags:
        shift = (cache[col].to_numpy() - hpov).mean() * 100
        print(f"  평균 이동 (cf_{tag} − cf_hpo): {shift:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for hl, tag, col in tags:
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for hl, tag, col in tags:
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"GBM 단독 델타 ({tag}−hpo): ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
