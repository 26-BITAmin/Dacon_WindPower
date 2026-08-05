# ═══════════════════════════════════════════════════════
# AN-13: 벡터 EMOS 개선 — 멤버 사전보정 + 계절 적합 + 속도 마진 하이브리드
#
# AN-12: EMOS가 원시 앙상블은 크게 이기지만(ES 3.70→2.08) 풍속 마진이 약함
#        (E|V| 1.87 vs 단변량 보정 1.50~1.64). |E[V]|의 구조적 과소평가 확인.
# 레버 3개:
#   premix : 멤버별 2×2 회전-스케일 보정 후 앙상블 구성 (고도 편향 제거)
#   +seas  : 모든 적합을 검증월 ±1 달력월 훈련행으로 (계절 드리프트)
#   +hyb   : 벡터 분포는 EMOS(rot), 샘플 속도 스케일을 5소스 속도 OLS에 정합
#
# 실행: C:\miniconda3\python.exe -X utf8 an13_premix.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an12_emos import (build_scada_vec, member_uv, fit_emos, sample_biv,
                       energy_score)
from an10_paper import crps_ens
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_WS = ["gfs_ws_100m", "ldaps_ws_10m", "ec_ws", "jmsm_ws", "jgsm_ws"]


def calib_members(M_tr, y_tr, M_va):
    """멤버별 2×2 + 절편 벡터 보정 (train 적합 → 양쪽 적용)."""
    out_tr = np.empty_like(M_tr)
    out_va = np.empty_like(M_va)
    for m in range(M_tr.shape[1]):
        A = np.c_[M_tr[:, m, :], np.ones(len(M_tr))]
        B, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        out_tr[:, m, :] = A @ B
        out_va[:, m, :] = np.c_[M_va[:, m, :], np.ones(len(M_va))] @ B
    return out_tr, out_va


def month_mask(dt_ser, month):
    mo = dt_ser.dt.month
    return (mo == month) | (mo == month % 12 + 1) | (mo == (month - 2) % 12 + 1)


def run_variant(tr_g, va_g, trm, vam, y_tr_vec, y_vec, y_ws, seasonal, hybrid):
    """premix EMOS(rot) 한 변형 실행 → 지표 dict."""
    M_tr_all, M_va_all = member_uv(tr_g), member_uv(va_g)
    mu_va = np.full((len(va_g), 2), np.nan)
    sig_va = np.full((len(va_g), 2), np.nan)
    r_va = np.full(len(va_g), np.nan)
    spd_cal = np.full(len(va_g), np.nan)
    months = sorted(va_g["forecast_kst_dtm"].dt.month.unique()) if seasonal else [None]
    for mo in months:
        sel_tr = trm & (month_mask(tr_g["forecast_kst_dtm"], mo).to_numpy()
                        if mo else np.ones(len(tr_g), bool))
        sel_va = (va_g["forecast_kst_dtm"].dt.month.to_numpy() == mo) \
            if mo else np.ones(len(va_g), bool)
        if sel_tr.sum() < 500:
            sel_tr = trm
        # y_tr_vec은 trm=True 행과 정렬 → sel_tr(⊆trm) 행의 y 위치 변환
        y_sub = y_tr_vec[np.flatnonzero(trm).searchsorted(np.flatnonzero(sel_tr))]
        Mc_tr, Mc_va = calib_members(M_tr_all[sel_tr], y_sub, M_va_all[sel_va])
        mk = lambda M: dict(mu=M.mean(1), s2=M.var(1),
                            theta=np.arctan2(M.mean(1)[:, 1], M.mean(1)[:, 0]))
        tr_d = {**mk(Mc_tr), "y": y_sub}
        va_d = mk(Mc_va)
        mu, sig, r = fit_emos(tr_d, va_d, "rot")
        mu_va[sel_va], sig_va[sel_va], r_va[sel_va] = mu, sig, r
        if hybrid:  # 속도 OLS (같은 훈련 부분집합)
            Xtr = tr_g.loc[sel_tr, SRC_WS].to_numpy()
            coef, *_ = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))],
                                       tr_g.loc[sel_tr, "park_ws"], rcond=None)
            spd_cal[sel_va] = np.c_[va_g.loc[sel_va, SRC_WS],
                                    np.ones(sel_va.sum())] @ coef
    samp = sample_biv(mu_va[vam], sig_va[vam], r_va[vam])
    if hybrid:  # 샘플 속도 스케일 정합 (방향·상대산포 유지)
        cur = np.linalg.norm(samp, axis=2).mean(1)
        samp = samp * np.clip(spd_cal[vam] / np.maximum(cur, 1e-6), 0.2, 5.0)[:, None, None]
    spd = np.linalg.norm(samp, axis=2)
    ang = np.abs((np.rad2deg(np.arctan2(mu_va[vam, 1], mu_va[vam, 0])
                             - np.arctan2(y_vec[:, 1], y_vec[:, 0])) + 180) % 360 - 180)
    return dict(ws_mae=np.abs(spd.mean(1) - y_ws).mean(),
                ws_crps=crps_ens(spd, y_ws).mean(),
                dir_mae=ang.mean(),
                vec_rmse=np.sqrt(((mu_va[vam] - y_vec) ** 2).sum(1).mean()),
                es=energy_score(samp, y_vec).mean())


def main():
    t0 = time.time()
    print("### AN-13 premix + 계절 + 속도 하이브리드 ###", flush=True)
    train_df, _ = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tt = time.time()
            tr_all, va_all = slice_fold(train_df, folds[fname])
            tr_g = tr_all.loc[tr_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            va_g = va_all.loc[va_all["kpx_group"] == g].sort_values(
                "forecast_kst_dtm").reset_index(drop=True)
            for d_ in (tr_g, va_g):
                for c in ["ec_ws", "ec_wdcos", "ec_wdsin", "jmsm_ws", "jmsm_wdcos",
                          "jmsm_wdsin", "jgsm_ws", "jgsm_wdcos", "jgsm_wdsin"]:
                    d_[c] = d_[c].ffill().bfill()
            trm = tr_g["u_obs"].notna().to_numpy()
            vam = va_g["u_obs"].notna().to_numpy()
            if trm.sum() < 1000 or vam.sum() < 100:
                continue
            y_tr_vec = np.c_[tr_g.loc[trm, "u_obs"], tr_g.loc[trm, "v_obs"]]
            y_vec = np.c_[va_g["u_obs"], va_g["v_obs"]][vam]
            y_ws = va_g["park_ws"].to_numpy()[vam]
            for tag, seas, hyb in (("premix", False, False),
                                   ("premix_seas", True, False),
                                   ("premix_hyb", True, True)):
                r = run_variant(tr_g, va_g, trm, vam, y_tr_vec, y_vec, y_ws, seas, hyb)
                recs.append(dict(fold=fname, group=g, model=tag, **r))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    cols = ["ws_mae", "ws_crps", "dir_mae", "vec_rmse", "es"]
    agg = df.groupby("model")[cols].mean().sort_values("es")
    print("\n=== 4폴드·3그룹 평균 — AN-12 참조: emos_rot ws1.876/dir23.1/ES2.084 · "
          "raw 3.066/28.6/3.705 | AnEn ws_crps 0.875 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    df.to_csv("an13_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n상세 → an13_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
