# ═══════════════════════════════════════════════════════
# exp-034: DL 앙상블 파트너 — GBM + (DL_384 ⊕ DL_512) 블렌드
#
# 배경(exp-033): GBM+DL_384 블렌드가 LB 경신(0.63620). DL_384/DL_512 각각 좋은
#   파트너(둘 다 게이트 통과)이고 용량이 달라 오차가 다름 → 둘을 평균한 DL 앙상블은
#   더 견고한 탈상관 파트너. GBM+DL 원리를 DL 내부에 한 번 더 적용. mild 변경.
# 판정: 단일 DL 블렌드(현 챔피언) vs DL 앙상블 블렌드를 동일 게이트에서 직접 비교.
# 실행: C:\miniconda3\python.exe -X utf8 exp034_dl_ensemble.py [--seeds N]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgbm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import (CAPACITY_KWH, TARGET_COLS, FOLDS_Q, metric,
                         make_sample_weight, apply_rolling_smoothing,
                         apply_bias_correction_v2, lgbm_params, prepare_matrices)
from exp009_rated_snap import (estimate_bias_data, bias_map_from_data, COARSE_BINS, RATED_CF, N_CLF_SEEDS)
from exp011_g3_transfer import build_pooled_matrices
from exp019_multigrid import HOLD_FOLD
from exp033_dl import load_data, make_xy, train_model, predict, GIDX

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALPHA, CLIP_LO, SNAP_THR, SNAP_TO = 0.50, 0.16, 0.30, 0.94
DL_CONFIGS = [dict(width=384, depth=3, p_drop=0.15, epochs=120),
              dict(width=512, depth=4, p_drop=0.20, epochs=200)]
# 블렌드 변형: 파트너 종류 × w
VARIANTS = {  # name: (partner, w_gbm)
    "gbm":       ("none", 1.0),
    "dl384_70":  ("dl0", 0.7),
    "dl512_70":  ("dl1", 0.7),
    "ens_70":    ("ens", 0.7),
    "ens_65":    ("ens", 0.65),
    "ens_60":    ("ens", 0.60),
}


def dl_predict(tr_df, va_df, cols, cfg, seeds):
    imp, sca = SimpleImputer(strategy="median"), StandardScaler()
    Xtr, gtr, ytr, wtr, _, _ = make_xy(tr_df, cols, imp, sca, fit=True)
    preds = []
    for s in range(seeds):
        m = train_model(Xtr, gtr, ytr, wtr, width=cfg["width"], depth=cfg["depth"],
                        p_drop=cfg["p_drop"], lr=1e-3, wd=1e-4, epochs=cfg["epochs"], bs=1024, seed=s, verbose=False)
        Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32)))
        gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        preds.append(predict(m, Xva.astype(np.float32), gva))
    return np.mean(preds, axis=0)


def run(folds, tag, args):
    train_df, _, cols, _ = load_data()
    rows = {v: {"ans": [], "pred": []} for v in VARIANTS}
    for fold in folds:
        tf = time.time()
        tr_mask = train_df["forecast_kst_dtm"] <= fold["train_end"]
        va_mask = (train_df["forecast_kst_dtm"] >= fold["valid_start"]) & (train_df["forecast_kst_dtm"] <= fold["valid_end"])
        tr_all, va_all = train_df.loc[tr_mask], train_df.loc[va_mask]
        tr_pool = tr_all.loc[tr_all["target"].notna()]
        va_sorted = va_all.sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        dl_all = [dl_predict(tr_all, va_sorted, cols, cfg, args.seeds) for cfg in DL_CONFIGS]
        va_sorted["dl0"] = dl_all[0]; va_sorted["dl1"] = dl_all[1]
        va_sorted["ens"] = 0.5 * (dl_all[0] + dl_all[1])

        bias_data = estimate_bias_data(train_df, cols, fold["train_end"], 500, args.jobs)
        coarse_map = bias_map_from_data(bias_data, {g: COARSE_BINS for g in bias_data})
        va_time = sorted(va_all["forecast_kst_dtm"].unique())
        ans = pd.DataFrame({"forecast_kst_dtm": va_time})
        pdf = {v: pd.DataFrame({"forecast_kst_dtm": va_time}) for v in VARIANTS}
        clf = []
        for group in [1, 2, 3]:
            tc = f"kpx_group_{group}"; cap = CAPACITY_KWH[tc]
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
            va_g = va_sorted.loc[va_sorted["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
            ans = ans.merge(va_g[["forecast_kst_dtm", "target"]].rename(columns={"target": tc}), on="forecast_kst_dtm", how="left")
            if len(tr_g) == 0:
                for v in VARIANTS: pdf[v][tc] = 0.0
                continue
            X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
            y_tr = tr_g["target"]; w_tr = make_sample_weight(y_tr.to_numpy(), cap, low_weight=0.05)
            pl = np.mean([lgb.LGBMRegressor(**lgbm_params(500, args.jobs, s)).fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(3)], axis=0)
            px = np.mean([xgbm.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbosity=0,
                objective="reg:absoluteerror").fit(X_tr, y_tr, sample_weight=w_tr).predict(X_va) for s in range(3)], axis=0)
            gbm_cf = (0.5 * pl + 0.5 * px) / cap
            X_pool_tr, X_pool_va = build_pooled_matrices(tr_pool, va_g, cols)
            if not clf:
                cp = tr_pool["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
                yA = (tr_pool["target"].to_numpy() >= cp * RATED_CF).astype(int)
                for s in range(N_CLF_SEEDS):
                    c = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, num_leaves=31,
                        min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=args.jobs, verbose=-1)
                    c.fit(X_pool_tr, yA); clf.append(c)
            pA = np.mean([c.predict_proba(X_pool_va)[:, 1] for c in clf], axis=0)
            ws = va_g["ldaps_ws_10m"].to_numpy()
            parts = {"dl0": va_g["dl0"].to_numpy(), "dl1": va_g["dl1"].to_numpy(), "ens": va_g["ens"].to_numpy()}
            for v, (partner, w) in VARIANTS.items():
                cf = gbm_cf if partner == "none" else (w * gbm_cf + (1 - w) * parts[partner])
                base = apply_bias_correction_v2(cf * cap, ws, group, coarse_map)
                base = apply_rolling_smoothing(pd.Series(base, index=va_g.index), 3, ALPHA).to_numpy().copy()
                base[pA >= SNAP_THR] = np.maximum(base[pA >= SNAP_THR], cap * SNAP_TO)
                base = np.clip(base, cap * CLIP_LO, cap)
                vp = va_g[["forecast_kst_dtm"]].copy(); vp[tc] = base
                pdf[v] = pdf[v].merge(vp, on="forecast_kst_dtm", how="left")
        ans_f = ans[TARGET_COLS].fillna(0.0)
        for v in VARIANTS:
            rows[v]["ans"].append(ans_f); rows[v]["pred"].append(pdf[v][TARGET_COLS].fillna(0.0))
        line = {v: metric(ans_f, pdf[v][TARGET_COLS].fillna(0.0))[0] for v in VARIANTS}
        print(f"[{tag}:{fold['name']}] ({time.time()-tf:.0f}s) " + " ".join(f"{v}={line[v]:.4f}" for v in VARIANTS), flush=True)
    res = {}
    print(f"\n--- {tag} pooled ---")
    for v in VARIANTS:
        A = pd.concat(rows[v]["ans"], ignore_index=True); P = pd.concat(rows[v]["pred"], ignore_index=True)
        s, n, f = metric(A, P); res[v] = s
        tag2 = "(기준)" if v == "gbm" else f"Δgbm {s-res['gbm']:+.4f}"
        print(f"  {v:10s}: {s:.4f} | 1-NMAE {n:.4f} | FiCR {f:.4f}  {tag2}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    cv = run(FOLDS_Q, "CV", args)
    hd = run([HOLD_FOLD], "HOLD", args)
    print("\n=== 이중 게이트 (Δgbm) ===")
    for v in cv:
        if v != "gbm":
            print(f"  {v:10s}: ΔCV {cv[v]-cv['gbm']:+.4f} | Δhold {hd[v]-hd['gbm']:+.4f} | "
                  f"both+ {(cv[v]-cv['gbm']>0) and (hd[v]-hd['gbm']>0)}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
