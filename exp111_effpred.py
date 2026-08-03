# ═══════════════════════════════════════════════════════
# exp-111: 유효출력률 `eff`의 **예보 기반 예측 가능성** + eff_hat 투입
#
# 왜 다시 재는가 (exp-091/092/094와 무엇이 다른가):
#   기존 정지 예측의 결론 "그룹 내부 AUC 0.60 = 사실상 불가"는 타깃이 `avail_strict`였다.
#   그런데 exp-110/EDA 12절에서 **`avail_strict`가 그룹3에서 오라벨**임이 확인됐다
#   ("완전가동" 시간의 66%가 실제로는 eff<0.8). **틀린 타깃으로 잰 예측 불가 판정**이므로
#   올바른 타깃(`eff`)으로 재측정할 근거가 있다. 재제안 금지는 옛 타깃에 대한 것.
#
# 오늘 발견한 신호가 실제로 잡히는지도 같이 본다 (EDA 11절):
#   - 예보풍속 8~12 m/s에서 g3 완전가동률 55~61% (g1 78~82 / g2 85~87)
#   - 저온+고습(T<2℃, RH≥90)에서 세 그룹 62~70%로 하락, 경로는 그룹별로 다름
#   - 시간대 패턴도 그룹별 반대
#
# 단계:
#   A. eff 예측 가능성 — 폴드 안전 OOF 회귀. 전체 352피처 vs **오늘 발견 신호 6개만**(`sig6`).
#      보고: R², AUC(eff<0.8), 그룹별. 상위 피처.
#   B. eff_hat 을 챔피언 GBM 입력에 추가(353피처) → 게이트.
#      ※ 사전 예상: **중립 개연성 높음** — GBM이 이미 ws·T·RH를 보고 있어 그 함수인 eff_hat은
#        신규 정보가 아니다(exp-072 물리파생 ±0.0000 · exp-088 PCA corr 0.9996의 유형).
#        그럼에도 사용자 지시로 측정한다. A의 R²이 낮으면 B는 자동 무의미.
#
# 판정 Δ = 보드 챔피언(hpo×dC_w65). 주기준 CV 4-fold (홀드아웃·corr은 참고).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp111_effpred.py --stage A
#       C:\miniconda3\python.exe -X utf8 exp111_effpred.py --stage B
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold
from exp110_effweight import EFF_PARQUET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EFFHAT_PARQUET = CACHE_DIR / "exp099_effhat.parquet"
# 오늘(EDA 11절) 발견한 신호만으로 구성한 최소 세트
SIG6 = ["ldaps_ws_10m", "ldaps_heightAboveGround_2_t", "ldaps_heightAboveGround_2_r",
        "gfs_ws_100m", "kpx_group", "hour_sin"]


def add_frame():
    train_df, cols = build_train_frame()
    eff = pd.read_parquet(EFF_PARQUET)[["forecast_kst_dtm", "group", "eff"]] \
        .rename(columns={"group": "kpx_group"})
    train_df = train_df.merge(eff, on=["forecast_kst_dtm", "kpx_group"], how="left")
    h = (train_df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.hour
    train_df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    train_df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return train_df, cols


def fit_eff_oof(train_df, cols, seeds, jobs, tag):
    """폴드 안전 OOF eff 예측 (검증 구간만 채움). 반환: eff_hat 컬럼이 붙은 사본."""
    import lightgbm as lgb
    out = np.full(len(train_df), np.nan)
    idx = pd.Series(np.arange(len(train_df)), index=train_df.index)
    for fname, fold in FOLD_LIST:
        tr_all, va_all = slice_fold(train_df, fold)
        tr = tr_all.loc[tr_all["eff"].notna()]
        if len(tr) == 0:
            continue
        X = tr[cols].to_numpy(dtype=np.float32)
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                              random_state=0, n_jobs=jobs, verbose=-1)
        m.fit(X, tr["eff"].to_numpy())
        p = m.predict(va_all[cols].to_numpy(dtype=np.float32))
        out[idx.loc[va_all.index].to_numpy()] = np.clip(p, 0, 1.2)
        if fname == FOLD_LIST[0][0]:
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            print(f"  [{tag}] 상위 피처: " + ", ".join(imp.head(8).index), flush=True)
    return out


def stage_a(train_df, cols, seeds, jobs):
    from sklearn.metrics import roc_auc_score
    print(f"eff 라벨행 {train_df['eff'].notna().sum():,} / 전체 {len(train_df):,}", flush=True)
    res = {}
    for tag, cs in (("all352", cols), ("sig6", SIG6)):
        t0 = time.time()
        res[tag] = fit_eff_oof(train_df, cs, seeds, jobs, tag)
        print(f"  [{tag}] {len(cs)}피처 · {time.time()-t0:.0f}s", flush=True)
    d = train_df.assign(**{f"hat_{k}": v for k, v in res.items()})
    d = d[d["eff"].notna()]
    print(f"\n{'세트':<8}{'그룹':<6}{'n':>8}{'R²':>9}{'AUC(eff<0.8)':>15}{'상관':>8}")
    for tag in res:
        for g in ([1, 2, 3, "전체"]):
            s = d if g == "전체" else d[d["kpx_group"] == g]
            y, p = s["eff"].to_numpy(), s[f"hat_{tag}"].to_numpy()
            ok = ~np.isnan(p)
            y, p = y[ok], p[ok]
            r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            lab = (y < 0.8).astype(int)
            auc = roc_auc_score(lab, -p) if 0 < lab.mean() < 1 else np.nan
            print(f"{tag:<8}{str(g):<6}{len(y):>8,}{r2:>9.3f}{auc:>15.3f}"
                  f"{np.corrcoef(y, p)[0,1]:>8.3f}")
    keep = train_df[["forecast_kst_dtm", "kpx_group"]].copy()
    keep["eff_hat"] = res["all352"]
    keep.to_parquet(EFFHAT_PARQUET, index=False)
    print(f"\neff_hat 저장 → {EFFHAT_PARQUET.name}", flush=True)
    print("\n※ 판정: R²이 0.2 미만이거나 AUC가 0.75 미만이면 B단계는 무의미 "
          "(exp-091/094의 0.60 대비 개선 여부가 관건).", flush=True)


def _eff_model(tr, cols, jobs, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                          min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, n_jobs=jobs, verbose=-1)
    m.fit(tr[cols].to_numpy(dtype=np.float32), tr["eff"].to_numpy())
    return m


def fill_effhat(tr_all, va_all, cols, jobs, kfold=4):
    """학습창은 **교차적합**(in-sample 낙관 제거), 검증창은 학습창 전체 적합으로 채운다.
    eff_hat을 피처로 쓰려면 학습 행에도 값이 있어야 하는데, OOF는 검증창만 채우므로 별도 필요."""
    tr = tr_all.copy()
    lab = tr["eff"].notna().to_numpy()
    hat = np.full(len(tr), np.nan)
    if lab.sum() >= 200:
        order = np.argsort(tr["forecast_kst_dtm"].to_numpy())   # 시간 블록 K분할
        blocks = np.array_split(order, kfold)
        for k in range(kfold):
            hold = np.zeros(len(tr), bool); hold[blocks[k]] = True
            sub = tr.loc[lab & ~hold]
            if len(sub) < 100:
                continue
            m = _eff_model(sub, cols, jobs, seed=k)
            hat[hold] = np.clip(m.predict(tr.loc[hold, cols].to_numpy(dtype=np.float32)), 0, 1.2)
        full = _eff_model(tr.loc[lab], cols, jobs)
        va_hat = np.clip(full.predict(va_all[cols].to_numpy(dtype=np.float32)), 0, 1.2)
    else:
        va_hat = np.full(len(va_all), np.nan)
    med = np.nanmedian(hat) if np.isfinite(hat).any() else 0.85
    tr["eff_hat"] = np.where(np.isnan(hat), med, hat)
    va = va_all.copy()
    va["eff_hat"] = np.where(np.isnan(va_hat), med, va_hat)
    return tr, va


def train_fold_effhat(tr_all, va_all, use, seeds, jobs, e56):
    import lightgbm as lgb
    import xgboost as xgbm
    from exp005_qopt import make_sample_weight
    from exp049_clean import avail_factor
    rows = []
    for group in [1, 2, 3]:
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        X_tr, X_va = prepare_matrices(tr_g, va_g, use)
        y_tr = tr_g["target"].to_numpy()
        w_tr = make_sample_weight(y_tr, cap, low_weight=0.05) * \
            avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
        pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                      random_state=s, n_jobs=jobs, verbose=-1, **e56["lgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                      tree_method="hist", random_state=s, n_jobs=jobs, verbosity=0, **e56["xgb"])
                      .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(seeds)], axis=0)
        rows.append(pd.DataFrame({"group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_effhat": (0.5 * pl + 0.5 * px) / cap}))
    return pd.concat(rows, ignore_index=True)


def stage_b(train_df, cols, seeds, jobs):
    use = list(cols) + ["eff_hat"]
    print(f"[B] {len(use)}피처 (eff_hat 추가) — 학습창은 시간블록 4겹 교차적합으로 채움",
          flush=True)
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()
    col, cpath = "cf_effhat", CACHE_DIR / "exp099_gbm_effhat.parquet"
    ppath = CACHE_DIR / "exp099_gbm_effhat_partial.parquet"
    if col not in merge_cached_cols(cache, cpath, [col]):
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            tr_all, va_all = fill_effhat(tr_all, va_all, cols, jobs)
            p = train_fold_effhat(tr_all, va_all, use, seeds, jobs, e56)
            m = cache.loc[sel, ["group", "forecast_kst_dtm"]].merge(
                p, on=["group", "forecast_kst_dtm"], how="left")
            assert m[col].notna().all(), f"정렬 실패 {fname}"
            out[sel] = m[col].to_numpy()
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [effhat:{fname}] {time.time()-tf:.0f}s "
                  f"(eff_hat 평균 tr {tr_all['eff_hat'].mean():.3f} / va {va_all['eff_hat'].mean():.3f})",
                  flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])
    mm = cache["target"].notna().to_numpy()
    v, h = cache[col].to_numpy(), cache["cf_hpo"].to_numpy()
    print(f"\ncorr(cf_effhat, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} (참고) | "
          f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW),
           "effhat_solo": ev(col, None, 1.0, PP_NEW)}
    for w in (0.65, 0.70):
        res[f"effhat_w{int(w*100)}"] = ev(col, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print(f"GBM 단독 ΔCV: {res['effhat_solo']['cv'][0]-res['hpo_solo']['cv'][0]:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], default="A")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-111 eff 예측 가능성 (stage {args.stage}) ###", flush=True)
    train_df, cols = add_frame()
    (stage_a if args.stage == "A" else stage_b)(train_df, cols, args.seeds, args.jobs)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
