# ═══════════════════════════════════════════════════════
# AN-14b: 최종 확정 런 — AN-14 최적 피처·가중 + HW=±4h, 전해상도 4폴드 K{30,50,100}
# 실행: C:\miniconda3\python.exe -X utf8 an14b_final.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an14_featsel import ENTRIES, entry_cols, lag_block
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

W14 = {
    1: {"gfs_ws_100m": 2.0, "ldaps_ws_10m": 2.0, "ec_ws": 1.0, "jgsm_ws": 1.0,
        "wd_pair": 1.0, "gfs_wind_shear_100_10": 0.5,
        "ldaps_heightAboveGround_2_t": 0.5, "ldaps_surface_0_sp": 0.5,
        "phy_ws850": 2.0, "sdL_ws10_mean": 2.0, "phy_gfs_gust_factor": 2.0},
    2: {"gfs_ws_100m": 0.5, "ldaps_ws_10m": 2.0, "ec_ws": 1.0, "jmsm_ws": 1.0,
        "jgsm_ws": 0.5, "wd_pair": 0.5, "gfs_wind_shear_100_10": 0.5,
        "ldaps_heightAboveGround_2_t": 0.5, "ldaps_surface_0_sp": 0.5,
        "phy_ws850": 2.0, "phy_gfs_gust_factor": 1.0},
    3: {"gfs_ws_100m": 0.5, "ldaps_ws_10m": 2.0, "ec_ws": 0.5, "jmsm_ws": 1.0,
        "jgsm_ws": 0.5, "wd_pair": 0.5, "gfs_wind_shear_100_10": 0.5,
        "ldaps_heightAboveGround_2_t": 0.5,
        "phy_ws850": 2.0, "phy_ws50max": 0.5, "phy_gfs_gust_factor": 1.0},
}
KS = [30, 50, 100]
HW = 4


def analogs_full(tr_g, va_g, w_map, twin=1, hw=HW, kmax=100):
    ent = [(e, w) for e, w in w_map.items() if w > 0]
    cols = sum((entry_cols(e) for e, _ in ent), [])
    A = lag_block(tr_g, cols, twin)
    F = lag_block(va_g, cols, twin)
    ok = tr_g["park_ws"].notna().to_numpy()
    vam = va_g["park_ws"].notna().to_numpy()
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    vi_, ti_ = np.flatnonzero(vam), np.flatnonzero(ok)
    D = np.zeros((len(vi_), len(ti_)), dtype=np.float32)
    ci = 0
    for e, w in ent:
        nc = len(entry_cols(e))
        for k in range(nc):
            j = ci + k
            sq = np.zeros_like(D, dtype=np.float64)
            for L in range(A.shape[2]):
                d = F[vi_, j, L][:, None] - A[ti_, j, L][None, :]
                sq += d * d
            D += (w * (0.5 if nc == 2 else 1.0)
                  / max(sig[j], 1e-9) * np.sqrt(sq)).astype(np.float32)
        ci += nc
    h_tr = tr_g["forecast_kst_dtm"].dt.hour.to_numpy()[ti_]
    h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()[vi_]
    dh = np.abs(h_va[:, None] - h_tr[None, :])
    D[np.minimum(dh, 24 - dh) > hw] = np.inf
    idx = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
    rows = np.arange(len(vi_))[:, None]
    order = np.argsort(D[rows, idx], axis=1)
    y_tr = tr_g["park_ws"].to_numpy()[ti_]
    return y_tr[idx[rows, order]], va_g["park_ws"].to_numpy()[vi_], \
        va_g.loc[vam, "forecast_kst_dtm"].to_numpy()


def main():
    t0 = time.time()
    print("### AN-14b 최종 확정 런 (HW=±4h) ###", flush=True)
    train_df, _ = build_train_frame()
    scada_h = build_scada_hourly()
    for g in [1, 2, 3]:
        sc = scada_h[g][["forecast_kst_dtm", "park_ws"]]
        train_df = train_df.merge(
            sc.assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"], how="left",
            suffixes=("", "_dup"))
        if "park_ws_dup" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_dup"))
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    folds = dict(FOLD_LIST)

    recs, pred_rows = [], []
    for fn in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, folds[fn])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            ana, y, tm = analogs_full(tr_g, va_g, W14[g])
            for K in KS:
                recs.append(dict(fold=fn, group=g, K=K,
                                 mae=np.abs(ana[:, :K].mean(1) - y).mean()))
            pred_rows.append(pd.DataFrame(
                {"fold": fn, "group": g, "forecast_kst_dtm": tm, "ws_obs": y,
                 "ws_anen14": ana[:, :50].mean(1)}))
            print(f"  [{fn} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    print("\n=== K별 4폴드·그룹 평균 MAE — 참조: AN-08 1.2070 · GBM352 1.0385 ===",
          flush=True)
    print(df.groupby("K")["mae"].mean().to_string(
        float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\n그룹별 (K=50):", flush=True)
    print(df[df["K"] == 50].pivot_table(index="fold", columns="group", values="mae")
          .to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    out = pd.concat(pred_rows, ignore_index=True)
    from exp035_raw_dl import CACHE_DIR
    out.to_parquet(CACHE_DIR / "an14_ws_preds.parquet", index=False)
    print(f"\n예측 저장 → an14_ws_preds.parquet ({len(out)}행) | "
          f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
