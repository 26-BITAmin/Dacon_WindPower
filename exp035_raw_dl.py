# ═══════════════════════════════════════════════════════
# exp-035: 원시 다격자·시퀀스 입력 DL — GBM 탈상관 파트너 강화
#
# 배경: exp-033이 GBM+DL(동일 132피처) 블렌드로 LB 경신(0.63620), exp-034에서 DL 용량·
#   앙상블·가중 축은 전부 sub-노이즈로 소진. 메모 기준 **남은 유일한 DL 레버는 "원시 입력"** —
#   GBM 피처가 버린 LDAPS 16격자/GFS 9격자의 공간 구조와 예보 시계열 형상.
#   exp-032(잔차 비가역)는 **132피처 조건부** 결론이므로, 새 입력 축에는 적용되지 않는다.
#
# 변형 (DL 입력만 다름, 블렌드·후처리는 챔피언 고정):
#   base : 132 (exp-033 챔피언 DL 재현 — 동일 런 기준선)
#   spg  : 132 + 공간 경사 축약 64 (변수당 평균/∂lat/∂lon/잔차std)
#   spd  : 132 + 전체 격자 편차 216
#   seq  : 132 + 예보블록 내 시계열 창 48 (±1~4h, 6채널; 블록 밖은 현재값 패딩 = 리크 없음)
#
# 비용 설계: GBM base·정격분류기·편향맵은 DL 변형과 무관 → fold당 1회 계산 후 캐시(parquet).
#   이후 DL 변형은 DL 학습 비용만 든다.
# 판정: 이중 게이트(4-fold CV + 1년 홀드아웃), 동일 런 Δgbm 및 base 대비 우위.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp035_raw_dl.py --stage cache
#       C:\miniconda3\python.exe -X utf8 exp035_raw_dl.py --stage dl --variants base,spg,spd,seq
# ═══════════════════════════════════════════════════════
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, FOLDS_Q, metric,
                         make_sample_weight, apply_rolling_smoothing,
                         apply_bias_correction_v2, lgbm_params, prepare_matrices)
from exp009_rated_snap import (estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS)
from exp011_g3_transfer import build_pooled_matrices
from exp019_multigrid import HOLD_FOLD
from exp033_dl import load_data, train_model, predict, GIDX
from exp035_grid_features import build_all as build_spatial_all, build_gext

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CACHE_DIR = Path(__file__).parent / "open (1)" / "external"
GBM_CACHE = CACHE_DIR / "exp035_gbm_cache.parquet"
BIAS_CACHE = CACHE_DIR / "exp035_bias_maps.pkl"

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94   # 챔피언 c16_a50 후처리
DL_CFG = dict(width=384, depth=3, p_drop=0.15, epochs=120, lr=1e-3, wd=1e-4)  # exp-033 국소 최적
# 변형별 DL 용량 오버라이드 (입력이 커진 변형에 384×3이 부족한지 검정)
VAR_CFG = {"spd512":   dict(width=512, depth=4, p_drop=0.20, epochs=200, lr=1e-3, wd=1e-4),
           "spd2_512": dict(width=512, depth=4, p_drop=0.20, epochs=200, lr=1e-3, wd=1e-4)}
BLEND_WS = [0.70, 0.65, 0.0]   # w축은 exp-036에서 0.70 정점 확인 — 확인차 0.65만 병기

SEQ_CH = {"lws": "ldaps_ws_10m", "gws": "gfs_ws_100m", "jg": "jgsm_ws",
          "jm": "jmsm_ws", "t2": "ldaps_heightAboveGround_2_t", "gst": "gfs_surface_0_gust"}
SEQ_OFF = [-4, -3, -2, -1, 1, 2, 3, 4]

# fold 목록 (CV 4개 + 홀드아웃). --quick이면 main에서 축소.
FOLD_LIST = [(f["name"], f) for f in FOLDS_Q] + [(HOLD_FOLD["name"], HOLD_FOLD)]


# ───────────────────────────────────────────────
# 입력 구성
# ───────────────────────────────────────────────
def add_seq_features(df):
    """예보 블록(전날 13시 발표분 = 01~24시) **내부**로 제한한 시계열 창.
    블록 경계 밖은 현재값으로 패딩 → 다음날 발표분을 당기지 않으므로 리크 없음."""
    df = df.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
    df["_blk"] = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.date
    g = df.groupby(["kpx_group", "_blk"], sort=False)
    cols = []
    for tag, ch in SEQ_CH.items():
        base = df[ch]
        for o in SEQ_OFF:
            name = f"sq_{tag}_{o:+d}"
            df[name] = g[ch].shift(-o).fillna(base)   # o>0 = 미래 시각
            cols.append(name)
    return df.drop(columns=["_blk"]), cols


def build_inputs(variants):
    """train_df에 변형별 입력 컬럼을 붙이고 {변형: 컬럼목록} 반환"""
    train_df, _, cols, _ = load_data()
    colsets = {"base": list(cols)}
    SPD_KIN = {"spd", "spd512", "spdL", "spdG", "spd2", "spd2_512", "spdj"}  # 전부 SPD 컬럼을 쓰는 변형
    if ({"spg"} | SPD_KIN) & set(variants):
        spg, spd, spg_cols, spd_cols = build_spatial_all()
        if "spg" in variants:
            train_df = train_df.merge(spg, on="forecast_kst_dtm", how="left")
            colsets["spg"] = list(cols) + spg_cols
        if SPD_KIN & set(variants):
            train_df = train_df.merge(spd, on="forecast_kst_dtm", how="left")
            colsets["spd"] = list(cols) + spd_cols
            colsets["spd512"] = list(cols) + spd_cols     # 입력 동일, 용량만 다름
            colsets["spdL"] = list(cols) + [c for c in spd_cols if c.startswith("sdL_")]
            colsets["spdG"] = list(cols) + [c for c in spd_cols if c.startswith("sdG_")]
            if {"spd2", "spd2_512"} & set(variants):      # spd + GFS 확장 편차(gext)
                gext, gext_cols = build_gext()
                train_df = train_df.merge(gext, on="forecast_kst_dtm", how="left")
                colsets["spd2"] = list(cols) + spd_cols + gext_cols
                colsets["spd2_512"] = list(cols) + spd_cols + gext_cols
    # JMA 다격자 편차 (exp-038) — 단독(jgrid) 및 spd와 누적(spdj)
    if {"jgrid", "spdj"} & set(variants):
        from exp038_jma_grid_features import build_jgrid
        jg, jg_cols = build_jgrid()
        train_df = train_df.merge(jg, on="forecast_kst_dtm", how="left")
        colsets["jgrid"] = list(cols) + jg_cols
        colsets["spdj"] = colsets.get("spd", list(cols)) + jg_cols
    if "seq" in variants:
        train_df, seq_cols = add_seq_features(train_df)
        colsets["seq"] = list(cols) + seq_cols
    if "spgseq" in variants:
        colsets["spgseq"] = colsets.get("spg", list(cols)) + [c for c in colsets.get("seq", []) if c.startswith("sq_")]
    return train_df, cols, colsets


# ───────────────────────────────────────────────
# 1단계: GBM base 캐시 (DL 변형과 무관 → 1회만)
# ───────────────────────────────────────────────
def cache_gbm(train_df, cols, args):
    import lightgbm as lgb
    import xgboost as xgbm

    rows, bias_maps = [], {}
    for fname, fold in FOLD_LIST:
        tf = time.time()
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                              (train_df["forecast_kst_dtm"] <= fold["valid_end"])]
        tr_pool = tr_all.loc[tr_all["target"].notna()]
        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], 500, args.jobs)
        bias_maps[fname] = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})

        clf = []
        for group in [1, 2, 3]:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            if len(tr_g) == 0 or len(va_g) == 0:
                continue
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), cap, low_weight=0.05)
            pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s))
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(args.gbm_seeds)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=s,
                          n_jobs=args.jobs, verbosity=0, objective="reg:absoluteerror")
                          .fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(args.gbm_seeds)], axis=0)
            gbm_cf = (0.5 * pl + 0.5 * px) / cap

            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not clf:
                cp = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                yA = (tr_pool["target"].to_numpy() >= cp * RATED_CF).astype(int)
                for s in range(N_CLF_SEEDS):
                    c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                           random_state=s, n_jobs=args.jobs, verbose=-1)
                    c.fit(X_pool_tr, yA); clf.append(c)
            pA = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf], axis=0)

            rows.append(pd.DataFrame({
                "fold": fname, "group": group,
                "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                "target": va_g["target"].to_numpy(),
                "ws": va_g["ldaps_ws_10m"].to_numpy(),
                "gbm_cf": gbm_cf, "pA": pA,
            }))
        print(f"  [cache:{fname}] {time.time()-tf:.0f}s", flush=True)

    out = pd.concat(rows, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GBM_CACHE, index=False)
    with open(BIAS_CACHE, "wb") as f:
        pickle.dump(bias_maps, f)
    print(f"GBM 캐시 저장: {out.shape} → {GBM_CACHE.name}")
    return out, bias_maps


def load_gbm_cache():
    cache = pd.read_parquet(GBM_CACHE)
    with open(BIAS_CACHE, "rb") as f:
        bias_maps = pickle.load(f)
    return cache, bias_maps


# ───────────────────────────────────────────────
# 2단계: DL 학습 + 블렌드 평가
# ───────────────────────────────────────────────
def make_xy(df, cols, imputer, scaler, fit=False):
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    return Xs.astype(np.float32), g, cf, w


def dl_fold_predict(tr_df, va_df, cols, seeds, cfg=None):
    cfg = cfg or DL_CFG
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy(tr_df, cols, imp, sca, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    preds = []
    for s in range(seeds):
        m = train_model(Xtr, gtr, ytr, wtr, width=cfg["width"], depth=cfg["depth"],
                        p_drop=cfg["p_drop"], lr=cfg["lr"], wd=cfg["wd"],
                        epochs=cfg["epochs"], bs=1024, seed=s, verbose=False)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def postprocess(cf, cap, ws, pA, group, bias_map):
    base = apply_bias_correction_v2(cf * cap, ws, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, ALPHA).to_numpy().copy()
    base[pA >= SNAP_THR] = np.maximum(base[pA >= SNAP_THR], cap * SNAP_TO)
    return np.clip(base, cap * CLIP_LO, cap)


def score_from_tidy(tidy, pred_col):
    """tidy(fold,group,dtm,target,pred) → 공식 metric (fold pooled)"""
    ans, prd = [], []
    for fname, sub in tidy.groupby("fold", sort=False):
        # 전 그룹 라벨 결측인 시각은 target 피벗에서 통째로 빠지므로 공통 시각축으로 reindex
        idx = np.sort(sub["forecast_kst_dtm"].unique())
        out = []
        for val in ("target", pred_col):
            m = sub.pivot_table(index="forecast_kst_dtm", columns="group", values=val)
            m.columns = [f"kpx_group_{int(c)}" for c in m.columns]
            out.append(m.reindex(index=idx, columns=TARGET_COLS).fillna(0.0))
        ans.append(out[0]); prd.append(out[1])
    return metric(pd.concat(ans, ignore_index=True), pd.concat(prd, ignore_index=True))


def run(args):
    global GBM_CACHE, BIAS_CACHE, FOLD_LIST
    train_df, cols, colsets = build_inputs(args.variants)
    print(f"입력 구성: " + " / ".join(f"{v}={len(colsets[v])}개" for v in args.variants), flush=True)

    if not GBM_CACHE.exists() or args.stage == "cache":
        print("### GBM base 캐시 생성 ###", flush=True)
        cache, bias_maps = cache_gbm(train_df, cols, args)
        if args.stage == "cache":
            return
    else:
        cache, bias_maps = load_gbm_cache()
        print(f"GBM 캐시 로드: {cache.shape}", flush=True)

    folds = FOLD_LIST
    cv_folds = [(fn, f) for fn, f in FOLD_LIST if fn != HOLD_FOLD["name"]]
    dl_cols = {}      # (변형, fold) → DL CF 예측 (cache 행 순서)
    corr_rows = []

    for v in args.variants:
        vcols = colsets[v]
        for fname, fold in folds:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = cache["fold"] == fname
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), f"{v}/{fname} 검증행 정렬 실패 ({len(va)} vs {len(keys)})"
            dl_cols[(v, fname)] = dl_fold_predict(tr_all, va, vcols, args.seeds, VAR_CFG.get(v))
            print(f"  [dl:{v}:{fname}] {time.time()-tf:.0f}s", flush=True)

    # 블렌드 → 후처리 → 점수
    results = {}
    for v in args.variants:
        cache[f"dl_{v}"] = np.nan
        for fname, _ in folds:
            cache.loc[cache["fold"] == fname, f"dl_{v}"] = dl_cols[(v, fname)]

    variant_names = ["gbm"] + [f"{v}_w{int(w*100):02d}" for v in args.variants for w in BLEND_WS]
    for name in variant_names:
        parts = []
        for (fname, group), sub in cache.groupby(["fold", "group"], sort=False):
            sub = sub.sort_values("forecast_kst_dtm")
            cap = CAPACITY_KWH[f"kpx_group_{int(group)}"]
            if name == "gbm":
                cf = sub["gbm_cf"].to_numpy()
            else:
                v, wtag = name.rsplit("_w", 1)
                w = int(wtag) / 100
                cf = w * sub["gbm_cf"].to_numpy() + (1 - w) * sub[f"dl_{v}"].to_numpy()
            pred = postprocess(cf, cap, sub["ws"].to_numpy(), sub["pA"].to_numpy(),
                               int(group), bias_maps[fname])
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        cv = score_from_tidy(tidy[tidy["fold"] != HOLD_FOLD["name"]], "pred")
        hd = score_from_tidy(tidy[tidy["fold"] == HOLD_FOLD["name"]], "pred")
        fold_cv = {fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn, _ in cv_folds}
        results[name] = dict(cv=cv, hold=hd, folds=fold_cv)

    # DL↔GBM 오차 상관 / DL 단독 MAE
    ev = cache["target"] >= cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]) * 0.10
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    eg = cache["gbm_cf"].to_numpy() - acf
    for v in args.variants:
        ed = cache[f"dl_{v}"].to_numpy() - acf
        m = ev.to_numpy() & np.isfinite(ed) & np.isfinite(eg)
        corr_rows.append({"variant": v, "corr": float(np.corrcoef(eg[m], ed[m])[0, 1]),
                          "dl_mae": float(np.abs(ed[m]).mean()), "gbm_mae": float(np.abs(eg[m]).mean())})

    # ── 리포트 ──
    g_cv, g_hd = results["gbm"]["cv"][0], results["gbm"]["hold"][0]
    print("\n" + "=" * 78)
    print(f"기준 gbm: CV {g_cv:.4f} (1-NMAE {results['gbm']['cv'][1]:.4f} FiCR {results['gbm']['cv'][2]:.4f}) | "
          f"HOLD {g_hd:.4f}")
    print("=" * 78)
    print(f"{'변형':<12}{'CV':>8}{'ΔCV':>9}{'HOLD':>8}{'Δhold':>9}{'1-NMAE':>9}{'FiCR':>8}  게이트")
    for name in variant_names:
        if name == "gbm":
            continue
        r = results[name]
        dcv, dhd = r["cv"][0] - g_cv, r["hold"][0] - g_hd
        gate = "통과" if (dcv > 0 and dhd > 0) else "  -"
        print(f"{name:<12}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{r['cv'][2]:>8.4f}  {gate}")

    print("\n--- fold별 ΔCV(gbm 대비) ---")
    fold_names = [fn for fn, _ in cv_folds]
    print(f"{'변형':<12}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in fold_names))
    for name in variant_names:
        if name == "gbm":
            continue
        print(f"{name:<12}" + "".join(f"{results[name]['folds'][fn]-results['gbm']['folds'][fn]:>+10.4f}"
                                      for fn in fold_names))

    print("\n--- DL 단독 품질 / GBM 오차 탈상관 (평가대상 시간) ---")
    print(pd.DataFrame(corr_rows).round(4).to_string(index=False))

    # 데이터량 → 최적 w 추세 (게이트 과소평가 우회 — 제출 w 외삽 근거)
    ws_only = [w for w in BLEND_WS if w > 0]
    for v in args.variants:
        rows = [(fn, f["train_end"], {w: results[f"{v}_w{int(w*100):02d}"]["folds"][fn] for w in ws_only})
                for fn, f in cv_folds]
        rows.append((HOLD_FOLD["name"], HOLD_FOLD["train_end"],
                     {w: results[f"{v}_w{int(w*100):02d}"]["hold"][0] for w in ws_only}))
        rows.sort(key=lambda r: r[1])   # 학습 데이터량 오름차순
        print(f"\n--- {v}: fold별 최적 w_gbm (train_end↑ = 데이터량↑ 순, 제출은 train≤2025-01) ---")
        print(f"{'fold':<16}{'train≤':<12}{'최적w':>7}   " + " ".join(f"w{int(w*100)}" for w in ws_only))
        for fn, te, sc in rows:
            bw = max(sc, key=sc.get)
            star = {w: ("*" if w == bw else " ") for w in ws_only}
            print(f"{fn:<16}{str(te)[:10]:<12}{bw:>7.2f}   "
                  + " ".join(f"{sc[w]:.4f}{star[w]}" for w in ws_only))

    # 기준 변형 대비 우위 (동일 런 Δ만 유효 — cv-baseline-drift 규칙)
    ref = f"{args.ref}_w70"
    if ref in results:
        rr = results[ref]
        print(f"\n--- 기준 {ref} 대비 (동일 런) ---")
        for name in variant_names:
            if name in ("gbm", ref):
                continue
            r = results[name]
            print(f"  {name:<12} ΔCV {r['cv'][0]-rr['cv'][0]:+.4f} | Δhold {r['hold'][0]-rr['hold'][0]:+.4f}")
    return results


def main():
    global GBM_CACHE, BIAS_CACHE, FOLD_LIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "dl"], default="dl")
    ap.add_argument("--variants", type=str, default="base,spg,spd,seq")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gbm-seeds", type=int, default=3)
    ap.add_argument("--ref", type=str, default="base", help="최종 비교표의 기준 변형")
    ap.add_argument("--quick", action="store_true", help="Q4+홀드아웃·1시드 배관 점검 (별도 캐시)")
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    if args.quick:
        args.seeds, args.gbm_seeds = 1, 1
        FOLD_LIST = [(FOLDS_Q[-1]["name"], FOLDS_Q[-1]), (HOLD_FOLD["name"], HOLD_FOLD)]
        GBM_CACHE = CACHE_DIR / "exp035_gbm_cache_quick.parquet"
        BIAS_CACHE = CACHE_DIR / "exp035_bias_maps_quick.pkl"
        print("### QUICK 배관 점검 모드 (판정용 아님) ###")
    t0 = time.time()
    print(f"### exp-035 (stage={args.stage}, 변형={args.variants}, DL seeds={args.seeds}, "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}) ###", flush=True)
    run(args)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
