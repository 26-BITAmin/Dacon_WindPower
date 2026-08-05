# ═══════════════════════════════════════════════════════
# AN-14: 풍속 AnEn — 피처 선택·시간창 하이퍼의 "제대로 된" 탐색
#
# 배경(사용자 지적): GBM은 352피처를 받았는데 AnEn은 수동 10예측자에 갇혀
#   있었고, AN-05/08 가중 탐색은 그 10개의 재가중만 가능했다(추가 불가).
#   시간창(TWIN ±1h)·시각창(HW ±3h)도 미스윕.
#
# 이번 탐색 (전부 Q1~Q3 최적화 / Q4 검증 분리 유지):
#   A) 피처: 현 10종 + 후보 12종(풍속 계열·거스트·850hPa·공간평균·불일치도)
#      → 그룹별 탐욕 이동 (각 변수 레벨 {0, .5, 1, 2}, 추가·제거·재가중 통합)
#   B) 시간창: TWIN {0,1,2} × HW {2,3,4} 미니그리드 (A의 최종 가중으로)
#   C) 최종 구성 전해상도 4폴드 검증 vs AN-08(1.2070)·GBM352(1.0385)
#
# 실행: C:\miniconda3\python.exe -X utf8 an14_featsel.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 현 10예측자 (AN-08 최적 가중을 초기값으로) + 후보 12
BASE = ["gfs_ws_100m", "ldaps_ws_10m", "ec_ws", "jmsm_ws", "jgsm_ws", "wd_pair",
        "gfs_wind_shear_100_10", "ldaps_heightAboveGround_2_t", "ldaps_surface_0_sp"]
CAND = ["gfs_ws_80m", "gfs_ws_pbl", "gfs_surface_0_gust", "gfs_wind_shear_80_10",
        "phy_ws850", "phy_ws50max", "ws_disagreement", "sdL_ws10_mean",
        "sdL_wsbl_mean", "sdG_ws100_mean", "sdG_gust_mean", "phy_gfs_gust_factor"]
ENTRIES = BASE + CAND
INIT_W = {1: dict(zip(BASE, (2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5))),
          2: dict(zip(BASE, (1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5))),
          3: dict(zip(BASE, (1.0, 2.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0)))}
LEVELS = (0.0, 0.5, 1.0, 2.0)
K_EVAL = 50


def entry_cols(e):
    return ["gfs_wd_sin_100m", "gfs_wd_cos_100m"] if e == "wd_pair" else [e]


def lag_block(g_df, cols, twin):
    arr = g_df[cols].ffill().bfill().to_numpy(dtype=np.float64)
    N, V = arr.shape
    L = 2 * twin + 1
    out = np.empty((N, V, L))
    for k, j in enumerate(range(-twin, twin + 1)):
        sh = np.roll(arr, -j, axis=0)
        if j < 0:
            sh[:(-j)] = arr[0]
        elif j > 0:
            sh[-j:] = arr[-1]
        out[:, :, k] = sh
    return out


def build_blocks(tr_g, va_g, entries, twin, hw, sub_tr=4, sub_va=3):
    """엔트리별 σ-정규화 거리 블록 [E, Nva', Ntr'] + y·시각마스크 (서브샘플)."""
    cols = sum((entry_cols(e) for e in entries), [])
    A = lag_block(tr_g, cols, twin)
    F = lag_block(va_g, cols, twin)
    ok = tr_g["park_ws"].notna().to_numpy()
    tr_idx = np.flatnonzero(ok)[::sub_tr]
    va_ok = va_g["park_ws"].notna().to_numpy()
    va_idx = np.flatnonzero(va_ok)[::sub_va]
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    h_tr = tr_g["forecast_kst_dtm"].dt.hour.to_numpy()[tr_idx]
    h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()[va_idx]
    dh = np.abs(h_va[:, None] - h_tr[None, :])
    hbad = np.minimum(dh, 24 - dh) > hw
    blocks, ci = [], 0
    for e in entries:
        nc = len(entry_cols(e))
        acc = np.zeros((len(va_idx), len(tr_idx)))
        for k in range(nc):
            j = ci + k
            sq = np.zeros_like(acc)
            for L in range(A.shape[2]):
                d = F[va_idx, j, L][:, None] - A[tr_idx, j, L][None, :]
                sq += d * d
            acc += (0.5 if nc == 2 else 1.0) / max(sig[j], 1e-9) * np.sqrt(sq)
        blocks.append(acc.astype(np.float32))
        ci += nc
    return (np.stack(blocks), tr_g["park_ws"].to_numpy()[tr_idx],
            va_g["park_ws"].to_numpy()[va_idx], hbad)


def eval_w(pack, w_vec):
    stack, y_tr, y_va, hbad = pack
    D = np.tensordot(np.asarray(w_vec, dtype=np.float32), stack, axes=1)
    D[hbad] = np.inf
    k = min(K_EVAL, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    return np.abs(y_tr[idx].mean(1) - y_va).mean()


def greedy(packs, w0, max_moves=30):
    cur = w0.copy()
    cur_v = np.mean([eval_w(p, cur) for p in packs])
    hist = [cur_v]
    for _ in range(max_moves):
        best_gain, best_move = 0.0, None
        for vi in range(len(cur)):
            for lv in LEVELS:
                if lv == cur[vi]:
                    continue
                cand = cur.copy()
                cand[vi] = lv
                v = np.mean([eval_w(p, cand) for p in packs])
                if cur_v - v > best_gain + 1e-5:
                    best_gain, best_move = cur_v - v, (vi, lv, v)
        if best_move is None:
            break
        vi, lv, v = best_move
        cur[vi], cur_v = lv, v
        hist.append(cur_v)
    return cur, cur_v, hist


def main():
    t0 = time.time()
    print("### AN-14 피처 선택 + 시간창 스윕 (풍속 AnEn) ###", flush=True)
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
    SEARCH = ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3"]

    def fold_frames(fn, g):
        tr_all, va_all = slice_fold(train_df, folds[fn])
        return (tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                    "forecast_kst_dtm").reset_index(drop=True),
                va_all.loc[va_all["kpx_group"] == g].sort_values(
                    "forecast_kst_dtm").reset_index(drop=True))

    # ── A) 피처 탐욕 탐색
    best_w = {}
    for g in [1, 2, 3]:
        tt = time.time()
        packs = []
        for fn in SEARCH:
            tr_g, va_g = fold_frames(fn, g)
            packs.append(build_blocks(tr_g, va_g, ENTRIES, twin=1, hw=3))
        w0 = np.array([INIT_W[g].get(e, 0.0) for e in ENTRIES])
        v0 = np.mean([eval_w(p, w0) for p in packs])
        w_opt, v_opt, hist = greedy(packs, w0)
        best_w[g] = w_opt
        del packs
        print(f"[그룹{g}] Q1~3 MAE {v0:.4f} → {v_opt:.4f} (Δ {v_opt-v0:+.4f}, "
              f"{len(hist)-1}회 이동) — {time.time()-tt:.0f}s", flush=True)
        for e, w in zip(ENTRIES, w_opt):
            mark = "" if e in BASE else "  ← 신규"
            if w > 0 or e in BASE:
                print(f"    {e:<28}{w:.1f}{mark if w > 0 else ''}", flush=True)

    # ── B) 시간창 미니그리드 (최종 가중 고정)
    print("\n[B] TWIN×HW 그리드 (Q1~3 평균, 그룹 평균)", flush=True)
    grid = [(1, 3), (0, 3), (2, 3), (1, 2), (1, 4)]
    for twin, hw in grid:
        vals = []
        for g in [1, 2, 3]:
            for fn in SEARCH:
                tr_g, va_g = fold_frames(fn, g)
                pack = build_blocks(tr_g, va_g, ENTRIES, twin=twin, hw=hw)
                vals.append(eval_w(pack, best_w[g]))
                del pack
        print(f"  TWIN=±{twin}h HW=±{hw}h : {np.mean(vals):.4f}", flush=True)

    # ── C) 전해상도 4폴드 검증 — 가중>0 엔트리만 거리 직접 누적 (블록 미저장)
    print("\n[C] 전해상도 4폴드 (K=50) — 참조: AN-08 AnEn 1.2070 · GBM352 1.0385",
          flush=True)

    def eval_full(tr_g, va_g, w_map, twin=1, hw=3):
        ent = [(e, w) for e, w in w_map.items() if w > 0]
        cols = sum((entry_cols(e) for e, _ in ent), [])
        A = lag_block(tr_g, cols, twin)
        F = lag_block(va_g, cols, twin)
        ok = tr_g["park_ws"].notna().to_numpy()
        vam = va_g["park_ws"].notna().to_numpy()
        sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
        D = np.zeros((vam.sum(), ok.sum()), dtype=np.float32)
        vi_ = np.flatnonzero(vam)
        ti_ = np.flatnonzero(ok)
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
        idx = np.argpartition(D, K_EVAL - 1, axis=1)[:, :K_EVAL]
        y_tr = tr_g["park_ws"].to_numpy()[ti_]
        y_va = va_g["park_ws"].to_numpy()[vi_]
        return np.abs(y_tr[idx].mean(1) - y_va).mean()

    maes = []
    for fn in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tr_g, va_g = fold_frames(fn, g)
            mae = eval_full(tr_g, va_g, dict(zip(ENTRIES, best_w[g])))
            maes.append(dict(fold=fn, group=g, mae=mae))
    df = pd.DataFrame(maes)
    print(df.pivot_table(index="fold", columns="group", values="mae")
          .to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\n4폴드·그룹 평균: {df['mae'].mean():.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
