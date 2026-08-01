# ═══════════════════════════════════════════════════════
# exp-084: 파워커브 역변환 파이프라인 **자체 개발 트랙** — 라운드 1: 가용성 정합 커브
#
# 트랙 규칙 (사용자 지시): 새 파이프라인은 챔피언과 즉시 비교해 기각하지 않는다.
#   **판정 기준 = 이 파이프라인의 이전 버전(pcinv v1)**. 챔피언은 참고 표시만.
#
# 라운드 1 가설 (exp-082b 진단 기반):
#   v1은 SCADA 파워커브를 "전 시간 bin 평균"으로 적합 — 터빈이 정지한 시간이 섞여
#   곡선이 아래로 눌림. 그 결과 (a) 천장 0.88/0.90/0.82 → CF 0.9+ 밴드 −0.051 붕괴,
#   (b) 몸통(CF 0.3~0.7)도 소폭 열세(+0.0018~30).
#   v1b(분위수 커브)는 천장은 고쳤으나 상향 편향(signed +0.006~7)·탈상관 소실(0.9986)
#   = 증상 치료였음.
#   → 올바른 수정: **가동률이 온전한 시간만으로 커브 적합** (exp049_avail.parquet의
#     시간×그룹 avail_strict 사용). 편향 주입 없이 천장·몸통 동시 교정.
#
# 변형: pcA (avail_strict == 1.0 시간만) / pcA95 (avail_strict >= 0.95)
#   그 외 전부 v1과 동일 (챔피언 GBM 하이퍼·cln03 가중·352피처·타깃 변환 방식).
#
# 프로토콜 개선: 스모크를 **Q2+Q4 2폴드**로 (exp-079/082에서 Q4 단독 스모크가 전체
#   CV를 3연속 과대평가한 관측 반영 — Q2/Q3가 항상 반대 부호였음).
#
# 실행: C:\miniconda3\python.exe -X utf8 pc02_curve.py --stage smoke
#       C:\miniconda3\python.exe -X utf8 pc02_curve.py --stage gate --variants pcA
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp015_scada_pc import PC_CENTERS, build_scada_hourly
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval, slice_fold
from pc01_pcinv import make_transform

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

AVAIL_PQ = CACHE_DIR / "exp049_avail.parquet"
V1_CACHE = CACHE_DIR / "exp082_gbm_pcinv.parquet"      # 트랙 베이스라인 (v1)
PC_BINS = np.arange(0.0, 30.5, 0.5)
MIN_BIN_N = 20
SMOKE_FOLDS = ["fold_2024Q2", "fold_2024Q4"]

# R1: 가용성 정합 커브 (avail_thr) — 커브 top 0.88→0.93이나 Q4 악화
# R2 진단 2건에 대한 대응:
#   (a) 커브 top이 건강 시간만 써도 0.93에서 멈춤 = "조건부 평균"의 구조적 성질.
#       bin 평균 커브를 **정방향**으로 쓰는 한 고출력 꼬리(실측 0.947)에 도달 불가.
#       → fwd_u: 역방향(타깃)은 평균 커브, **정방향(예측)만 상위 커브로 상향**.
#         프로젝트에서 검증된 "보정된 크기의 상향 편향"(DL q60, exp-039/042) 기전과 동형.
#   (b) 커브를 건강 시간으로 올렸는데 학습 라벨엔 정지 시간이 ×0.05로 남아 타깃 불일치
#       확대 (λ=0.05는 평균 커브 기준 최적값, exp-049/050) → drop_low: 완전 제외로 정합.
VARIANTS = {
    "pcA":    dict(avail_thr=1.0),
    "pcA95":  dict(avail_thr=0.95),
    "pcA0":   dict(avail_thr=1.0, drop_low=True),
    "pcAu30": dict(avail_thr=1.0, fwd_u=0.30),
    "pcAu60": dict(avail_thr=1.0, fwd_u=0.60),
    "pcU30":  dict(avail_thr=0.0, fwd_u=0.30),      # 평균 커브(v1) + 정방향만 상향
}


def load_avail():
    a = pd.read_parquet(AVAIL_PQ)
    return {g: sub[["forecast_kst_dtm", "avail_strict"]]
            for g, sub in a.groupby("group", sort=False)}


def _bin_curve(j, agg):
    ws_bin = pd.cut(j["park_ws"], bins=PC_BINS, labels=False)
    grp = j.groupby(ws_bin)["park_cf"]
    vals, cnt = agg(grp), grp.size()
    curve = np.full(len(PC_CENTERS), np.nan)
    for idx, val in vals.items():
        if cnt[idx] >= MIN_BIN_N:
            curve[int(idx)] = val
    valid = ~np.isnan(curve)
    if valid.sum() < 5:
        return None
    return np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])


def build_curve_avail(scada_g, avail_g, g_train, tr_end, thr, fwd_u=0.0):
    """fold-safe 파워커브. 반환 (curve_inv, curve_fwd, n).
    curve_inv = bin 평균 (타깃 역변환용 — 라벨 분포와 정합)
    curve_fwd = (1−u)·평균 + u·bin p90 (예측 정방향용 — u>0이면 고출력 꼬리 도달 가능)"""
    sc = scada_g[scada_g["forecast_kst_dtm"] <= tr_end]
    if thr > 0:
        sc = sc.merge(avail_g, on="forecast_kst_dtm", how="left")
        sc = sc[sc["avail_strict"].fillna(0.0) >= thr]
    j = g_train.merge(sc, on="forecast_kst_dtm", how="inner").dropna(
        subset=["park_ws", "park_cf"])
    if len(j) < 500:
        return None, None, 0
    c_mean = _bin_curve(j, lambda g: g.mean())
    if c_mean is None:
        return None, None, len(j)
    if fwd_u <= 0:
        return c_mean, c_mean, len(j)
    c_hi = _bin_curve(j, lambda g: g.quantile(0.90))
    if c_hi is None:
        return c_mean, c_mean, len(j)
    return c_mean, (1 - fwd_u) * c_mean + fwd_u * c_hi, len(j)


def group_fold_data(train_df, scada_h, avail, fold, group, cols, spec):
    tr_all, va_all = slice_fold(train_df, fold)
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
    if len(tr_g) == 0 or len(va_g) == 0:
        return None
    if spec.get("drop_low"):                       # 커브와 학습 라벨의 가용성 정합
        tr_g = tr_g.loc[tr_g["avail_strict"] >= 1.0].copy()
        if len(tr_g) < 500:
            return None
    c_inv, c_fwd, n_fit = build_curve_avail(scada_h[group], avail[group], tr_g,
                                            fold["train_end"], spec.get("avail_thr", 1.0),
                                            spec.get("fwd_u", 0.0))
    if c_inv is None:
        return None
    _, inv, _, rng = make_transform(c_inv)         # 역변환 = 평균 커브 (라벨 정합)
    fwd, _, _, rng_f = make_transform(c_fwd)       # 정방향 = 상향 커브 (꼬리 도달)
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_tr = tr_g["target"].to_numpy()
    w = make_sample_weight(y_tr, cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    return dict(X_tr=X_tr, X_va=X_va, y_tr=y_tr, w=w, va_g=va_g, cap=cap,
                fwd=fwd, inv=inv, rng=rng, rng_f=rng_f, n_fit=n_fit, n_tr=len(tr_g))


def fit_predict_cf(d, e56, seeds, jobs):
    import lightgbm as lgb
    import xgboost as xgbm
    v_t = d["inv"](d["y_tr"] / d["cap"])
    pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s,
                                    n_jobs=jobs, verbose=-1, **e56["lgb"])
                  .fit(d["X_tr"], v_t, sample_weight=d["w"]).predict(d["X_va"])
                  for s in range(seeds)], axis=0)
    px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                                    tree_method="hist", random_state=s, n_jobs=jobs,
                                    verbosity=0, **e56["xgb"])
                  .fit(d["X_tr"], v_t, sample_weight=d["w"]).predict(d["X_va"])
                  for s in range(seeds)], axis=0)
    return d["fwd"](0.5 * pl + 0.5 * px)


def stage_smoke(train_df, use_cols, scada_h, avail, e56, jobs, tags):
    print(f"[smoke] {'+'.join(f.replace('fold_','') for f in SMOKE_FOLDS)} 2폴드·1-seed — "
          f"판정 기준 = pcinv v1 (챔피언은 참고)", flush=True)
    cache, _ = load_cache_full()
    assert merge_cached_cols(cache, V1_CACHE, ["cf_pcinv"]), "v1 캐시 없음"
    acc = {v: {} for v in tags}
    ref = {}
    for fname in SMOKE_FOLDS:
        fold = dict(FOLD_LIST)[fname]
        for group in [1, 2, 3]:
            sub = cache[(cache["fold"] == fname) & (cache["group"] == group)]
            for v in tags:
                d = group_fold_data(train_df, scada_h, avail, fold, group, use_cols,
                                    VARIANTS[v])
                if d is None:
                    print(f"  {fname}/g{group}/{v}: 커브 적합 실패", flush=True)
                    continue
                va = d["va_g"]
                m = va["target"].notna().to_numpy() & \
                    (va["target"].to_numpy() >= d["cap"] * 0.10)
                y_cf = va["target"].to_numpy() / d["cap"]
                tt = time.time()
                cf = fit_predict_cf(d, e56, 1, jobs)
                acc[v].setdefault(fname, []).append(
                    dict(mae=np.abs(cf[m] - y_cf[m]).mean(),
                         signed=(cf[m] - y_cf[m]).mean(), cf=cf[m], top=cf.max()))
                print(f"  {fname}/g{group}/{v}: inv커브top {d['rng'][1]:.3f} "
                      f"/ fwd커브top {d['rng_f'][1]:.3f} (적합 {d['n_fit']}행 "
                      f"· 학습 {d['n_tr']}행) | 예측max {cf.max():.3f} | "
                      f"{time.time()-tt:.0f}s", flush=True)
                if v == tags[0]:
                    j = va[["forecast_kst_dtm"]].merge(
                        sub[["forecast_kst_dtm", "cf_pcinv", "cf_hpo"]],
                        on="forecast_kst_dtm", how="left")
                    ref.setdefault(fname, []).append(dict(
                        v1=j["cf_pcinv"].to_numpy()[m], hpo=j["cf_hpo"].to_numpy()[m],
                        y=y_cf[m],
                        mae_v1=np.abs(j["cf_pcinv"].to_numpy()[m] - y_cf[m]).mean(),
                        mae_hpo=np.abs(j["cf_hpo"].to_numpy()[m] - y_cf[m]).mean()))

    print(f"\n{'변형':<10}{'폴드':<10}{'MAE':>9}{'Δv1':>10}{'Δ챔피언':>10}"
          f"{'signed':>9}{'corr(v1)':>10}{'예측top':>9}")
    for fname in SMOKE_FOLDS:
        r = ref[fname]
        mv1 = np.mean([x["mae_v1"] for x in r])
        mhp = np.mean([x["mae_hpo"] for x in r])
        print(f"{'(v1)':<10}{fname.replace('fold_',''):<10}{mv1:>9.5f}{0.0:>+10.5f}"
              f"{mv1-mhp:>+10.5f}")
        for v in tags:
            rows = acc[v][fname]
            mae = np.mean([x["mae"] for x in rows])
            cat = np.concatenate([x["cf"] for x in rows])
            v1cat = np.concatenate([x["v1"] for x in r])
            print(f"{v:<10}{fname.replace('fold_',''):<10}{mae:>9.5f}{mae-mv1:>+10.5f}"
                  f"{mae-mhp:>+10.5f}{np.mean([x['signed'] for x in rows]):>+9.4f}"
                  f"{np.corrcoef(cat, v1cat)[0,1]:>10.4f}"
                  f"{np.mean([x['top'] for x in rows]):>9.3f}")
    print("\n라운드1 통과 기준: **2폴드 모두** Δv1 < 0 (v1 대비 MAE 개선)", flush=True)


def stage_gate(train_df, use_cols, scada_h, avail, e56, jobs, seeds, tags):
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, V1_CACHE, ["cf_pcinv"]), "v1 캐시 없음"
    for tag in tags:
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp084_gbm_{tag}.parquet"
        if col not in merge_cached_cols(cache, cpath, [col]):
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                rows = []
                for group in [1, 2, 3]:
                    d = group_fold_data(train_df, scada_h, avail, fold, group, use_cols,
                                        VARIANTS[tag])
                    if d is None:
                        continue
                    rows.append(pd.DataFrame({
                        "fold": fold["name"], "group": group,
                        "forecast_kst_dtm": d["va_g"]["forecast_kst_dtm"].to_numpy(),
                        col: fit_predict_cf(d, e56, seeds, jobs)}))
                preds.append(pd.concat(rows, ignore_index=True))
                print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    base = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)     # ← 트랙 판정 기준 (v1)
    print(f"\n[참고] 챔피언(hpo×dC_w65): CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)
    print(f"[기준] v1(pcinv×dC_w65):   CV {base['cv'][0]:.4f} | HOLD {base['hold'][0]:.4f}"
          f"  (기대 0.6530/0.6512)", flush=True)
    res = {"v1_solo": ev("cf_pcinv", None, 1.0, PP_NEW)}
    for tag in tags:
        col = f"cf_{tag}"
        for b in ("cf_pcinv", "cf_hpo", "dl_cln03"):
            print(f"  corr({col}, {b}) = {corr_on_eval(cache, col, b):.4f}", flush=True)
        print(f"  평균 이동 ({col} − cf_pcinv): "
              f"{(cache[col].to_numpy()-cache['cf_pcinv'].to_numpy()).mean()*100:+.3f}%cap",
              flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, base)      # Δ는 v1 기준
    print("\n※ 위 Δ는 **v1(pcinv) 대비** — 트랙 내부 개선 판정. 챔피언 대비는 참고 줄 사용.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--variants", default="pcA,pcA95")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-084 파워커브 트랙 R1: 가용성 정합 커브 ###", flush=True)

    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    avail = load_avail()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    for t in tags:
        assert t in VARIANTS, f"미정의 변형: {t}"

    if args.stage == "smoke":
        stage_smoke(train_df, use_cols, scada_h, avail, e56, args.jobs, tags)
    else:
        stage_gate(train_df, use_cols, scada_h, avail, e56, args.jobs, args.seeds, tags)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
