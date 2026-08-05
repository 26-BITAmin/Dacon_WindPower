# ═══════════════════════════════════════════════════════
# AN-12: Schuhen–Thorarinsdottir–Gneiting 2012 (MWR 140, 3204–3219) 적용
#        "Ensemble Model Output Statistics for Wind Vectors" — 이변량 벡터 후처리
#
# 논문 구성 (우리 대응):
#   - 예측분포 = (u,v) 이변량 정규
#   - 평균: 앙상블 평균의 성분별 선형 편향보정  μ_u = a_u + b_u·ū
#   - 분산: 앙상블 분산의 선형함수            σ²_u = c_u + d_u·s²_u
#   - ★ u–v 오차 상관 = 예보 평균 풍향의 삼각함수  r(θ) = k0 + k1·cosθ + k2·sinθ
#   - "앙상블" = 다소스 벡터 5멤버 (gfs100 · ldaps10 · ec10 · jmsm · jgsm)
#   - 실측 = SCADA 나셀 벡터 (park_ws + 터빈 풍향 원형평균, 기상 관례)
#
# 변형: emos_uni(r=0, 성분 독립 대조) / emos_biv(논문) / emos_rot(평균을 2×2 행렬로 — 확장)
# 평가: 풍속 MAE(E|V|·|μ|) · 풍향 MAE · 벡터 RMSE · Energy Score · 풍속 CRPS
#       대조 = 원시 앙상블 · AnEn(ws, tuned K30) · (참고 GBM352 1.039, AN-08)
#
# ⚠ 풍향 관례: 나셀 = 기상(FROM), 예보 u,v = 수학 벡터. 시작 전 대응률 검증 출력.
#
# 실행: C:\miniconda3\python.exe -X utf8 an12_emos.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an07_wsanen import ws_analogs
from an08_wsopt import wsset
from an10_paper import crps_ens
from exp015_scada_pc import GROUP_TURBINES, TRAIN_DIR
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BEST_W = {1: (2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5),
          2: (1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5),
          3: (1.0, 2.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0)}
N_SAMP = 32
RNG = np.random.default_rng(0)


def build_scada_vec():
    """그룹별 시간 집계: park_ws + 나셀 풍향(기상 관례) 원형평균 → 실측 u,v."""
    raw = {
        "vestas": pd.read_csv(TRAIN_DIR / "scada_vestas_train.csv",
                              encoding="utf-8-sig", parse_dates=["kst_dtm"]),
        "unison": pd.read_csv(TRAIN_DIR / "scada_unison_train.csv",
                              encoding="utf-8-sig", parse_dates=["kst_dtm"]),
    }
    out = {}
    for group, (maker, turbs, cap10) in GROUP_TURBINES.items():
        df = raw[maker]
        wcols = [f"{t}_ws" for t in turbs]
        dcols = [f"{t}_wd" for t in turbs]
        th = np.deg2rad(df[dcols].to_numpy(dtype=float))
        # 기상 관례 FROM θ → 벡터 u = −sinθ, v = −cosθ (터빈 평균은 단위벡터 평균)
        ub = np.nanmean(-np.sin(th), axis=1)
        vb = np.nanmean(-np.cos(th), axis=1)
        d = pd.DataFrame({
            "hour": df["kst_dtm"].dt.ceil("h"),
            "ws": df[wcols].mean(axis=1), "ub": ub, "vb": vb})
        h = d.groupby("hour").agg(park_ws=("ws", "mean"), ub=("ub", "mean"),
                                  vb=("vb", "mean"), n=("ws", "size"))
        h = h[h["n"] == 6].drop(columns="n")
        norm = np.hypot(h["ub"], h["vb"]).replace(0, np.nan)
        h["u_obs"] = h["park_ws"] * h["ub"] / norm
        h["v_obs"] = h["park_ws"] * h["vb"] / norm
        out[group] = h.reset_index().rename(columns={"hour": "forecast_kst_dtm"})[
            ["forecast_kst_dtm", "park_ws", "u_obs", "v_obs"]]
    return out


def member_uv(df):
    """다소스 벡터 5멤버 [N, 5, 2]. wd_sin/cos는 수학 관례 각의 sin/cos."""
    mems = []
    mems.append(np.stack([df["gfs_heightAboveGround_100_100u"],
                          df["gfs_heightAboveGround_100_100v"]], -1))
    mems.append(np.stack([df["ldaps_heightAboveGround_10_10u"],
                          df["ldaps_heightAboveGround_10_10v"]], -1))
    for ws, cs, sn in (("ec_ws", "ec_wdcos", "ec_wdsin"),
                       ("jmsm_ws", "jmsm_wdcos", "jmsm_wdsin"),
                       ("jgsm_ws", "jgsm_wdcos", "jgsm_wdsin")):
        mems.append(np.stack([df[ws] * df[cs], df[ws] * df[sn]], -1))
    return np.stack(mems, 1).astype(float)                 # [N, 5, 2]


def harmonic_corr(theta, eu, ev, nbins=12, min_n=50):
    """섹터별 경험 상관 → r(θ) = k0 + k1 cosθ + k2 sinθ 적합. 반환 = 계수."""
    bins = (theta // (2 * np.pi / nbins)).astype(int) % nbins
    centers, corrs, wts = [], [], []
    for b in range(nbins):
        m = bins == b
        if m.sum() >= min_n:
            c = np.corrcoef(eu[m], ev[m])[0, 1]
            if np.isfinite(c):
                centers.append((b + 0.5) * 2 * np.pi / nbins)
                corrs.append(c)
                wts.append(m.sum())
    if len(centers) < 4:
        return np.array([np.corrcoef(eu, ev)[0, 1], 0.0, 0.0])
    A = np.c_[np.ones(len(centers)), np.cos(centers), np.sin(centers)]
    W = np.sqrt(np.asarray(wts))
    k, *_ = np.linalg.lstsq(A * W[:, None], np.asarray(corrs) * W, rcond=None)
    return k


def fit_emos(tr, va, mode):
    """tr/va: dict(mu[N,2], s2[N,2], theta[N], y[N,2](tr만)). 반환 = 예측 μ, Σ 성분."""
    y, mu, s2 = tr["y"], tr["mu"], tr["s2"]
    if mode == "rot":                                       # 평균 = 2×2 행렬 + 절편
        A = np.c_[mu, np.ones(len(mu))]
        B, *_ = np.linalg.lstsq(A, y, rcond=None)
        mu_tr, mu_va = A @ B, np.c_[va["mu"], np.ones(len(va["mu"]))] @ B
    else:                                                   # 논문: 성분별 선형
        mu_tr = np.empty_like(mu)
        mu_va = np.empty_like(va["mu"])
        for j in range(2):
            a, b = np.polyfit(mu[:, j], y[:, j], 1)
            mu_tr[:, j] = a * mu[:, j] + b
            mu_va[:, j] = a * va["mu"][:, j] + b
    e = y - mu_tr
    sig_va = np.empty_like(mu_va)
    for j in range(2):                                      # σ² = c + d·s² (OLS, 양수 클립)
        d, c = np.polyfit(s2[:, j], e[:, j] ** 2, 1)
        v = np.clip(c + max(d, 0.0) * va["s2"][:, j], 0.05, None)
        sig_va[:, j] = np.sqrt(v)
    if mode == "uni":
        r_va = np.zeros(len(mu_va))
    else:                                                   # ★ r(θ) 삼각함수 (논문)
        k = harmonic_corr(tr["theta"], e[:, 0], e[:, 1])
        r_va = np.clip(k[0] + k[1] * np.cos(va["theta"]) + k[2] * np.sin(va["theta"]),
                       -0.95, 0.95)
    return mu_va, sig_va, r_va


def sample_biv(mu, sig, r, n=N_SAMP):
    """이변량 정규 샘플 [N, n, 2]."""
    z1 = RNG.standard_normal((len(mu), n))
    z2 = RNG.standard_normal((len(mu), n))
    x = mu[:, None, 0] + sig[:, None, 0] * z1
    yv = mu[:, None, 1] + sig[:, None, 1] * (r[:, None] * z1
                                             + np.sqrt(1 - r[:, None] ** 2) * z2)
    return np.stack([x, yv], -1)


def energy_score(samp, y):
    """ES: mean||X−y|| − 0.5·mean||X−X'||. samp [N,M,2], y [N,2]"""
    t1 = np.linalg.norm(samp - y[:, None, :], axis=2).mean(1)
    d = samp[:, :, None, :] - samp[:, None, :, :]
    t2 = np.linalg.norm(d, axis=3).mean((1, 2))
    return t1 - 0.5 * t2


def main():
    t0 = time.time()
    print("### AN-12 이변량 벡터 EMOS (Schuhen et al. 2012) ###", flush=True)
    train_df, _ = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    print(f"입력 준비 완료 | 벡터 커버 {train_df['u_obs'].notna().mean()*100:.1f}% "
          f"— {time.time()-t0:.0f}s", flush=True)

    # ── 관례 검증: 예보(gfs100) 기상 풍향 vs 나셀 풍향 ±30° 대응률
    for g in [1, 2, 3]:
        d = train_df[(train_df["kpx_group"] == g) & train_df["u_obs"].notna()]
        met_f = (270 - np.rad2deg(np.arctan2(
            d["gfs_heightAboveGround_100_100v"], d["gfs_heightAboveGround_100_100u"]))) % 360
        met_o = (270 - np.rad2deg(np.arctan2(d["v_obs"], d["u_obs"]))) % 360
        diff = np.abs((met_f - met_o + 180) % 360 - 180)
        print(f"  [관례검증 g{g}] gfs100 vs 나셀 ±30° 대응 {np.mean(diff <= 30)*100:.1f}%"
              f" (중앙 오차 {np.median(diff):.0f}°)", flush=True)

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
            M_tr, M_va = member_uv(tr_g), member_uv(va_g)
            mk = lambda M: dict(mu=M.mean(1), s2=M.var(1),
                                theta=np.arctan2(M.mean(1)[:, 1], M.mean(1)[:, 0]))
            tr = {**mk(M_tr[trm]), "y": np.c_[tr_g.loc[trm, "u_obs"],
                                              tr_g.loc[trm, "v_obs"]]}
            va = mk(M_va)
            y_vec = np.c_[va_g["u_obs"], va_g["v_obs"]][vam]
            y_ws = va_g["park_ws"].to_numpy()[vam]

            def add(model, **kw):
                recs.append(dict(fold=fname, group=g, model=model, **kw))

            # 원시 앙상블 대조
            raw_mu = va["mu"][vam]
            raw_ws = np.linalg.norm(M_va[vam], axis=2).mean(1)
            ang = lambda p, o: np.abs((np.rad2deg(np.arctan2(p[:, 1], p[:, 0])
                                       - np.arctan2(o[:, 1], o[:, 0])) + 180) % 360 - 180)
            add("raw_ens", ws_mae=np.abs(raw_ws - y_ws).mean(),
                dir_mae=ang(raw_mu, y_vec).mean(),
                vec_rmse=np.sqrt(((raw_mu - y_vec) ** 2).sum(1).mean()),
                es=energy_score(M_va[vam], y_vec).mean(),
                ws_crps=crps_ens(np.linalg.norm(M_va[vam], axis=2), y_ws).mean())

            for mode in ["uni", "biv", "rot"]:
                mu_va, sig_va, r_va = fit_emos(tr, va, mode)
                samp = sample_biv(mu_va[vam], sig_va[vam], r_va[vam])
                spd = np.linalg.norm(samp, axis=2)
                add(f"emos_{mode}",
                    ws_mae=np.abs(spd.mean(1) - y_ws).mean(),
                    ws_mae_mu=np.abs(np.linalg.norm(mu_va[vam], axis=1) - y_ws).mean(),
                    dir_mae=ang(mu_va[vam], y_vec).mean(),
                    vec_rmse=np.sqrt(((mu_va[vam] - y_vec) ** 2).sum(1).mean()),
                    es=energy_score(samp, y_vec).mean(),
                    ws_crps=crps_ens(spd, y_ws).mean())

            # AnEn(ws) 대조 — 풍속만 (벡터 아님)
            ana = ws_analogs(tr_g, va_g, wsset(*BEST_W[g]))[vam]
            add("anen_K30", ws_mae=np.abs(ana[:, :30].mean(1) - y_ws).mean(),
                ws_crps=crps_ens(ana[:, :30], y_ws).mean())
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    cols = ["ws_mae", "ws_mae_mu", "ws_crps", "dir_mae", "vec_rmse", "es"]
    agg = df.groupby("model")[cols].mean().sort_values("ws_mae")
    print("\n=== 4폴드·3그룹 평균 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\n항목: ws_mae = E|V| 풍속 MAE (m/s) · ws_mae_mu = |μ| 사용시 · "
          "dir_mae = 풍향 MAE(°) · es = Energy Score (벡터 확률평가) | "
          "참고: AN-08 GBM352 풍속 MAE 1.039 · AnEn 1.207", flush=True)
    df.to_csv("an12_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n상세 → an12_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
