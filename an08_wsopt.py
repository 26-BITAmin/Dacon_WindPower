# ═══════════════════════════════════════════════════════
# AN-08: 풍속 AnEn 개선 — 가중 재최적화(풍속 타깃) + GBM 혼합
#
# AN-07 결과 (4폴드 평균 MAE, m/s):
#   원시 예보 2.04~4.43 → 선형보정 1.64 → OLS5 1.50 → AnEn(hub,K50) 1.217 → GBM 1.039
# 남은 레버:
#   1) AN-05 가중은 발전량(CF) 목적이었음 → 풍속 타깃으로 재탐색 (Q1~Q3 최적화)
#   2) AnEn ⊕ GBM 혼합 — 함수족이 달라 상보 기대
#   3) 예측 parquet 저장 (후속 사용)
#
# 실행: C:\miniconda3\python.exe -X utf8 an08_wsopt.py
# ═══════════════════════════════════════════════════════
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd

from an01_anen import lagged_block
from an07_wsanen import RAW_SOURCES, ws_analogs
from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SPACE = [("gfs_ws_100m", (1.0, 2.0)),
         ("ldaps_ws_10m", (0.5, 1.0, 2.0)), ("ec_ws", (0.5, 1.0)),
         ("jmsm_ws", (0.5, 1.0)), ("jgsm_ws", (0.0, 0.5, 1.0)),
         ("wd_pair", (0.0, 0.5, 1.0)),
         ("gfs_wind_shear_100_10", (0.0, 0.5)),
         ("ldaps_heightAboveGround_2_t", (0.0, 0.5)),
         ("ldaps_surface_0_sp", (0.0, 0.5, 1.0))]
K_SEARCH = 50
HW = 3


def pervar_ws(train_df, fold, group, sub_tr=2, sub_va=2):
    """an05 pervar_dists의 풍속 타깃판 — y = park_ws, 마스크 없음."""
    cols = []
    for c, _ in SPACE:
        cols += ["gfs_wd_sin_100m", "gfs_wd_cos_100m"] if c == "wd_pair" else [c]
    tr_all, va_all = slice_fold(train_df, fold)
    tr_g = tr_all.loc[tr_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    A = lagged_block(tr_g, cols)
    F = lagged_block(va_g, cols)
    ok = tr_g["park_ws"].notna().to_numpy()
    tr_idx = np.flatnonzero(ok)[::sub_tr]
    va_ok = va_g["park_ws"].notna().to_numpy()
    va_idx = np.flatnonzero(va_ok)[::sub_va]
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    h_tr = tr_g["forecast_kst_dtm"].dt.hour.to_numpy()[tr_idx]
    h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()[va_idx]
    dh = np.abs(h_va[:, None] - h_tr[None, :])
    hour_bad = np.minimum(dh, 24 - dh) > HW
    blocks, ci = [], 0
    for c, _ in SPACE:
        ncomp = 2 if c == "wd_pair" else 1
        acc = np.zeros((len(va_idx), len(tr_idx)), dtype=np.float64)
        for k in range(ncomp):
            j = ci + k
            sq = np.zeros_like(acc)
            for L in range(A.shape[2]):
                diff = F[va_idx, j, L][:, None] - A[tr_idx, j, L][None, :]
                sq += diff * diff
            acc += (0.5 if ncomp == 2 else 1.0) / max(sig[j], 1e-9) * np.sqrt(sq)
        blocks.append(acc.astype(np.float32))
        ci += ncomp
    return (np.stack(blocks), tr_g["park_ws"].to_numpy()[tr_idx],
            va_g["park_ws"].to_numpy()[va_idx], hour_bad)


def eval_w(stack, y_tr, y_va, hour_bad, w):
    D = np.tensordot(np.asarray(w, dtype=np.float32), stack, axes=1)
    D[hour_bad] = np.inf
    k = min(K_SEARCH, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    return np.abs(y_tr[idx].mean(1) - y_va).mean()


def wsset(gfs100, ldaps, ec, jmsm, jgsm, wd, shear, t, sp):
    return [("gfs_ws_100m", gfs100), ("gfs_wd_sin_100m", wd / 2),
            ("gfs_wd_cos_100m", wd / 2), ("ldaps_ws_10m", ldaps), ("ec_ws", ec),
            ("jmsm_ws", jmsm), ("jgsm_ws", jgsm), ("gfs_wind_shear_100_10", shear),
            ("ldaps_heightAboveGround_2_t", t), ("ldaps_surface_0_sp", sp)]


def main():
    t0 = time.time()
    print("### AN-08 풍속 AnEn 가중 재최적화 + GBM 혼합 ###", flush=True)
    train_df, use_cols = build_train_frame()
    scada_h = build_scada_hourly()
    for g in [1, 2, 3]:
        sc = scada_h[g][["forecast_kst_dtm", "park_ws"]]
        train_df = train_df.merge(
            sc.assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"], how="left",
            suffixes=("", "_dup"))
        if "park_ws_dup" in train_df:
            train_df["park_ws"] = train_df["park_ws"].fillna(train_df.pop("park_ws_dup"))
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    e56 = json.loads(E56_JSON.read_text())
    import lightgbm as lgb

    folds = dict(FOLD_LIST)
    print("[1단계] 가중 좌표하강 (Q1~Q3 합산, 2패스, 서브샘플 1/3)", flush=True)
    base_w = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5)   # AN-07 hub 대응
    best = {}
    for g in [1, 2, 3]:
        tt = time.time()
        packs = [pervar_ws(train_df, folds[fn], g, sub_tr=3, sub_va=3)
                 for fn in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3"]]

        def obj(w):
            return sum(eval_w(*p[:3], p[3], w) for p in packs) / len(packs)

        cur = list(base_w)
        cur_v = obj(cur)
        base_v = cur_v
        for _ in range(2):                                   # 2패스 좌표하강
            for vi, (_, levels) in enumerate(SPACE):
                if len(levels) == 1:
                    continue
                for lv in levels:
                    if lv == cur[vi]:
                        continue
                    cand = cur.copy()
                    cand[vi] = lv
                    v = obj(cand)
                    if v < cur_v - 1e-6:
                        cur, cur_v = cand, v
        del packs
        best[g] = tuple(cur)
        print(f"[그룹{g}] 최적 {dict(zip([c for c, _ in SPACE], best[g]))} | "
              f"Q1~3 {cur_v:.4f} vs hub형 {base_v:.4f} (Δ {cur_v-base_v:+.4f}) "
              f"— {time.time()-tt:.0f}s", flush=True)

    print("\n[2단계] 전해상도 4폴드 — 최적 AnEn·GBM·혼합", flush=True)
    recs, pred_rows = [], []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        fold = folds[fname]
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, fold)
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            m = va_g["park_ws"].notna().to_numpy()
            y = va_g["park_ws"].to_numpy()
            trm = tr_g["park_ws"].notna().to_numpy()
            mdl = lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective="mae",
                                    random_state=0, n_jobs=8, verbose=-1, **e56["lgb"])
            mdl.fit(tr_g.loc[trm, use_cols], tr_g.loc[trm, "park_ws"])
            p_gbm = mdl.predict(va_g[use_cols])
            ana = ws_analogs(tr_g, va_g, wsset(*best[g]))
            preds = {"gbm": p_gbm}
            for K in [30, 50, 100]:
                preds[f"anen_K{K}"] = ana[:, :K].mean(1)
            for lam in [0.7, 0.6, 0.5, 0.4, 0.3]:
                preds[f"mix_l{int(lam*100)}"] = lam * p_gbm + (1 - lam) * preds["anen_K50"]
            for name, p in preds.items():
                e = p[m] - y[m]
                recs.append(dict(fold=fname, group=g, model=name,
                                 mae=np.abs(e).mean(), rmse=np.sqrt((e**2).mean()),
                                 signed=e.mean()))
            pred_rows.append(va_g[["forecast_kst_dtm"]].assign(
                fold=fname, group=g, ws_obs=y, ws_gbm=p_gbm,
                ws_anen=preds["anen_K50"], ws_mix=preds["mix_l60"]))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    agg = df.groupby("model")[["mae", "rmse", "signed"]].mean().sort_values("mae")
    print("\n=== 4폴드 평균 (그룹 평균, m/s) — AN-07 참조: GBM 1.0385 / AnEn(hub,K50) 1.2174 ===",
          flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:+.4f}"), flush=True)
    piv = df.pivot_table(index="model", columns="group", values="mae").sort_values(1)
    print("\n=== 그룹별 MAE ===", flush=True)
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    out = pd.concat(pred_rows, ignore_index=True)
    out.to_parquet(CACHE_DIR / "an08_ws_preds.parquet", index=False)
    print(f"\n예측 저장 → an08_ws_preds.parquet ({len(out)}행) | "
          f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
