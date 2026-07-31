# ═══════════════════════════════════════════════════════
# exp-082: 파워커브 역변환 타깃 파이프라인 — "무엇을 예측하는가" 축 (새 함수족)
#
# 배경:
#   - exp-065: GBM 3구현(LGB·XGB·HGB) 상호 corr 0.997~0.999 = 같은 정보 위에서
#     같은 함수족을 쓰면 구현을 바꿔도 예측이 수렴. exp-069 TabPFN만 0.9839로
#     처음 0.99를 깼으나 시간 감쇠(홀드 −0.0083)로 기각.
#   - exp-032/077의 "잔차 직교 = base near-optimal"은 **현 함수족 조건부** 결론.
#     타깃 자체를 바꾸면 다른 능선일 수 있음 (exp-056이 "하이퍼 조건부 결론"을
#     반증했던 것과 같은 유형).
#
# 변경점: 발전량 직접 회귀 → **유효 풍속 회귀 + 물리 곡선 복원**
#   학습 타깃  v_t = PC_g^{-1}(y/cap)      (PC = SCADA 실측 파워커브, fold-safe)
#   예측       CF  = PC_g(v̂)
#   → 3승 비선형을 데이터가 아니라 실측 물리곡선이 담당. 모델은 매끄러운 풍속 공간만 학습.
#
# 변형 2개:
#   pcinv : 풍속 공간 MAE (균등)
#   pcw   : 국소 기울기 dPC/dv 가중 — 풍속 MAE ≈ 발전량 MAE로 되돌림
#           (exp-008: 손실의 57%가 파워커브 급경사 5~9m/s 구간 → 균등 MAE는 이 구간
#            과소 중시. 가중 총질량은 그룹×폴드별 재정규화 — exp-066 선례)
#
# 프로토콜: smoke = Q4·1-seed → CF공간 MAE·signed·corr(vs cf_hpo) go/no-go
#           gate  = 3-seed·4-fold+홀드아웃 → 제3 base 멤버 혼합 사다리 × dl_cln03
#           판정 Δ = 보드 챔피언(hpo×dC_w65, 재현 기대 CV 0.6556 / HOLD 0.6535)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp082_pcinv.py --stage smoke
#       C:\miniconda3\python.exe -X utf8 exp082_pcinv.py --stage gate --variants pcinv,pcw
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp015_scada_pc import PC_CENTERS, build_scada_hourly, fit_scada_maps
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, corr_on_eval, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# q=None → bin 평균 커브(1차) / q=분위수 → 정지 시간에 눌리지 않은 "건강한 터빈" 커브
# (exp-082b 진단: 평균 커브는 꼭대기가 0.88/0.90/0.82로 눌려 CF 0.9+ 밴드 진입 실패)
VARIANTS = {"pcinv": dict(q=None), "pcw": dict(q=None, slope_w=True),
            "pcq80": dict(q=0.80), "pcq90": dict(q=0.90)}
CF_TAIL = 0.98        # 곡선 최댓값의 98%까지만 역변환 (평탄 꼭대기의 발산 구간 제외)
PC_BINS_L = np.arange(0.0, 30.5, 0.5)
MIN_BIN_N_L = 20


def build_curve(scada_h_g, g_train, tr_end, q):
    """fold-safe 파워커브. q=None이면 bin 평균(exp-015 원형), 아니면 bin 상위 분위수."""
    if q is None:
        maps = fit_scada_maps(scada_h_g, g_train, tr_end)
        return None if maps is None else maps[1]
    sc = scada_h_g[scada_h_g["forecast_kst_dtm"] <= tr_end]
    j = g_train.merge(sc, on="forecast_kst_dtm", how="inner").dropna(
        subset=["park_ws", "park_cf"])
    if len(j) < 500:
        return None
    ws_bin = pd.cut(j["park_ws"], bins=PC_BINS_L, labels=False)
    grp = j.groupby(ws_bin)["park_cf"]
    qv, cnt = grp.quantile(q), grp.size()
    curve = np.full(len(PC_CENTERS), np.nan)
    for idx, val in qv.items():
        if cnt[idx] >= MIN_BIN_N_L:
            curve[int(idx)] = val
    valid = ~np.isnan(curve)
    if valid.sum() < 5:
        return None
    return np.interp(np.arange(len(curve)), np.flatnonzero(valid), curve[valid])


def make_transform(curve):
    """SCADA 파워커브 → (정방향 PC, 역방향 PC^-1, 기울기) — 단조화 후 엄격 증가 보장."""
    c = np.maximum.accumulate(np.asarray(curve, dtype=float))
    c = c + np.arange(len(c)) * 1e-9                      # 엄격 증가 (np.interp 요구)
    cf_lo, cf_hi = c[0], c[0] + (c[-1] - c[0]) * CF_TAIL
    slope = np.gradient(c, PC_CENTERS)

    def fwd(v):                                            # 풍속 → CF
        return np.interp(np.clip(v, PC_CENTERS[0], PC_CENTERS[-1]), PC_CENTERS, c)

    def inv(cf):                                           # CF → 풍속
        return np.interp(np.clip(cf, cf_lo, cf_hi), c, PC_CENTERS)

    def slope_at(v):
        return np.interp(np.clip(v, PC_CENTERS[0], PC_CENTERS[-1]), PC_CENTERS, slope)

    return fwd, inv, slope_at, (cf_lo, cf_hi)


def group_fold_data(train_df, scada_h, fold, group, cols, q=None):
    """폴드×그룹: 행렬 + 파워커브 변환기 (커브는 train_end 이전 SCADA만 사용 = fold-safe)"""
    tr_all, va_all = slice_fold(train_df, fold)
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
    if len(tr_g) == 0 or len(va_g) == 0:
        return None
    curve = build_curve(scada_h[group], tr_g, fold["train_end"], q)
    if curve is None:
        return None
    fwd, inv, slope_at, rng = make_transform(curve)
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_tr = tr_g["target"].to_numpy()
    w_base = make_sample_weight(y_tr, cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    return dict(X_tr=X_tr, X_va=X_va, y_tr=y_tr, w_base=w_base, va_g=va_g, cap=cap,
                fwd=fwd, inv=inv, slope_at=slope_at, rng=rng)


def fit_predict_cf(d, variant, e56, seeds, jobs):
    """풍속 공간 학습 → 예측 → 파워커브로 CF 복원."""
    import lightgbm as lgb
    import xgboost as xgbm
    v_t = d["inv"](d["y_tr"] / d["cap"])
    w = d["w_base"]
    if VARIANTS[variant].get("slope_w"):
        s = np.clip(d["slope_at"](v_t), 1e-3, None)
        w = w * (s / s.mean())                            # 총질량 재정규화
    pl = np.mean([lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae", random_state=s_,
                                    n_jobs=jobs, verbose=-1, **e56["lgb"])
                  .fit(d["X_tr"], v_t, sample_weight=w).predict(d["X_va"])
                  for s_ in range(seeds)], axis=0)
    px = np.mean([xgbm.XGBRegressor(n_estimators=e56["xgb_n"], objective="reg:absoluteerror",
                                    tree_method="hist", random_state=s_, n_jobs=jobs,
                                    verbosity=0, **e56["xgb"])
                  .fit(d["X_tr"], v_t, sample_weight=w).predict(d["X_va"])
                  for s_ in range(seeds)], axis=0)
    return d["fwd"](0.5 * pl + 0.5 * px), v_t


def stage_smoke(train_df, use_cols, scada_h, e56, jobs, tags):
    fold = dict(FOLD_LIST)["fold_2024Q4"]
    print("[smoke] Q4 폴드·1-seed — CF 공간 MAE 비교 (기준 = 챔피언 캐시 cf_hpo)", flush=True)
    cache, _ = load_cache_full()
    out = {v: [] for v in tags}
    ref_rows = []
    for group in [1, 2, 3]:
        sub = cache[(cache["fold"] == fold["name"]) & (cache["group"] == group)]
        for v in tags:
            d = group_fold_data(train_df, scada_h, fold, group, use_cols,
                                VARIANTS[v].get("q"))
            if d is None:
                print(f"  그룹{group}/{v}: SCADA 커브 적합 실패 — 건너뜀", flush=True)
                continue
            va = d["va_g"]
            m = va["target"].notna().to_numpy() & (va["target"].to_numpy() >= d["cap"] * 0.10)
            y_cf = va["target"].to_numpy() / d["cap"]
            tt = time.time()
            cf, v_t = fit_predict_cf(d, v, e56, 1, jobs)
            out[v].append(dict(group=group, mae=np.abs(cf[m] - y_cf[m]).mean(),
                               signed=(cf[m] - y_cf[m]).mean(), cf=cf[m],
                               vt_mean=v_t.mean(), sec=time.time() - tt))
            print(f"  그룹{group}/{v}: 커브 CF범위 {d['rng'][0]:.3f}~{d['rng'][1]:.3f} "
                  f"| 예측 max {cf.max():.3f} | 평가시간 {m.sum()}", flush=True)
            if v == tags[0]:
                j = va[["forecast_kst_dtm"]].merge(sub[["forecast_kst_dtm", "cf_hpo"]],
                                                   on="forecast_kst_dtm", how="left")
                ref_rows.append(dict(group=group, cf=j["cf_hpo"].to_numpy()[m],
                                     mae=np.abs(j["cf_hpo"].to_numpy()[m] - y_cf[m]).mean()))
    ref_mae = np.mean([r["mae"] for r in ref_rows])
    ref_cat = np.concatenate([r["cf"] for r in ref_rows])
    print(f"\n[smoke] 기준 cf_hpo(3-seed 캐시) MAE {ref_mae:.5f}", flush=True)
    for v in tags:
        mae = np.mean([r["mae"] for r in out[v]])
        sgn = np.mean([r["signed"] for r in out[v]])
        cat = np.concatenate([r["cf"] for r in out[v]])
        ok = np.isfinite(cat) & np.isfinite(ref_cat)
        print(f"[smoke:{v}] MAE {mae:.5f} (Δ {mae-ref_mae:+.5f}) | signed {sgn:+.4f} | "
              f"corr(vs cf_hpo) {np.corrcoef(cat[ok], ref_cat[ok])[0,1]:.4f} | "
              f"타깃풍속 평균 {np.mean([r['vt_mean'] for r in out[v]]):.2f} m/s | "
              f"{np.mean([r['sec'] for r in out[v]]):.0f}s/그룹", flush=True)
    print("\ngo/no-go 기준: corr < 0.99 (실다양성) ∧ MAE ≤ 파트너급(+0.005 이내)", flush=True)


def stage_gate(train_df, use_cols, scada_h, e56, jobs, seeds, tags):
    cache, champ_bias = load_cache_full()
    for tag in tags:
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp082_gbm_{tag}.parquet"
        if col not in merge_cached_cols(cache, cpath, [col]):
            preds = []
            for fname, fold in FOLD_LIST:
                tf = time.time()
                rows = []
                for group in [1, 2, 3]:
                    d = group_fold_data(train_df, scada_h, fold, group, use_cols,
                                        VARIANTS[tag].get("q"))
                    if d is None:
                        continue
                    cf, _ = fit_predict_cf(d, tag, e56, seeds, jobs)
                    rows.append(pd.DataFrame({
                        "fold": fold["name"], "group": group,
                        "forecast_kst_dtm": d["va_g"]["forecast_kst_dtm"].to_numpy(),
                        col: cf}))
                preds.append(pd.concat(rows, ignore_index=True))
                print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
            p = pd.concat(preds, ignore_index=True)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    hpov = cache["cf_hpo"].to_numpy()
    for tag in tags:
        col = f"cf_{tag}"
        for b in ("cf_hpo", "dl_cln03"):
            print(f"  corr({col}, {b}) = {corr_on_eval(cache, col, b):.4f}", flush=True)
        print(f"  평균 이동 ({col} − cf_hpo): "
              f"{(cache[col].to_numpy() - hpov).mean()*100:+.3f}%cap", flush=True)
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        res[f"{tag}_w65"] = ev(col, "dl_cln03", 0.65, PP_NEW)
        for m_, mt in ((5 / 6, "t83"), (0.75, "t75"), (2 / 3, "t67"), (0.5, "t50")):
            mix = f"mx_{tag}_{mt}"
            cache[mix] = m_ * hpov + (1 - m_) * cache[col].to_numpy()
            for w in (0.65, 0.70):
                res[f"{tag}{mt}_w{int(w*100)}"] = ev(mix, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    for tag in tags:
        print(f"GBM 단독 델타 ({tag}−hpo): "
              f"ΔCV {res[f'{tag}_solo']['cv'][0]-res['hpo_solo']['cv'][0]:+.4f} / "
              f"Δhold {res[f'{tag}_solo']['hold'][0]-res['hpo_solo']['hold'][0]:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--variants", default="pcq80,pcq90")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-082 파워커브 역변환 타깃 파이프라인 ###", flush=True)

    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    print(f"입력 준비 완료 ({len(use_cols)}피처) | SCADA 시간수 "
          f"{ {g: len(h) for g, h in scada_h.items()} } — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())

    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    for t in tags:
        assert t in VARIANTS, f"미정의 변형: {t}"
    if args.stage == "smoke":
        stage_smoke(train_df, use_cols, scada_h, e56, args.jobs, tags)
    else:
        stage_gate(train_df, use_cols, scada_h, e56, args.jobs, args.seeds, tags)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
