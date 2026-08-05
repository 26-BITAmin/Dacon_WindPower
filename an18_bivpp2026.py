# ═══════════════════════════════════════════════════════
# AN-18: Buchner et al. 2026 (QJRMS, arXiv 2601.21401) "Bivariate Postprocessing
#        of Wind Vectors" — 논문의 실제 신형 2종 구현
#
# (그동안 구현한 Schuhen 2012 이변량 EMOS는 이 논문의 "베이스라인"이었음)
#   BIV-DRN     : 은닉 1층 MLP + 그룹 임베딩 → (μu, μv, σu, σv, ρ), 이변량 정규
#                 NLL 손실, 넷 5개 앙상블(파라미터 평균) — 논문 1위 방법
#   GBM-파라미터 : BIV-EMOS-GB의 실용 대응 — 평균은 LGBM(mae), σ는 교차적합
#                 잔차의 log분산 LGBM, ρ는 그룹 상수(교차적합 표준화 잔차 상관)
#
# 입력(C⁺ 대응 21피처): enfo 16멤버 통계 8(u/v 평균·sd, ws 평균·sd, control u/v)
#   + 5모델 벡터 10(각 모델 u,v) + 기온·기압·비습 3
# 실측: 나셀 벡터. 평가: AN-12~17과 동일 지표 + LogS.
#
# 실행(루트에서): C:\miniconda3\python.exe -X utf8 "open (1)/testing/an18_bivpp2026.py"
# ═══════════════════════════════════════════════════════
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from an10_paper import crps_ens
from an12_emos import build_scada_vec, member_uv, energy_score
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp056_hpo import RESULT_JSON as E56_JSON
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEM_CACHE = CACHE_DIR / "an16_enfo_members.parquet"
THERMO = ["ldaps_heightAboveGround_2_t", "ldaps_surface_0_sp",
          "ldaps_heightAboveGround_2_q"]
N_SAMP = 32
RNG = np.random.default_rng(0)


def build_features(train_df):
    """행 단위 21피처 + 실측 벡터. enfo 커버 행만."""
    mem = pd.read_parquet(MEM_CACHE)
    uw = mem.pivot_table(index=["forecast_kst_dtm", "group"], columns="member",
                         values="u")
    vw = mem.pivot_table(index=["forecast_kst_dtm", "group"], columns="member",
                         values="v")
    U, V = uw.to_numpy(), vw.to_numpy()
    ws = np.hypot(U, V)
    ef = pd.DataFrame({
        "enfo_u_mean": np.nanmean(U, 1), "enfo_v_mean": np.nanmean(V, 1),
        "enfo_u_sd": np.nanstd(U, 1), "enfo_v_sd": np.nanstd(V, 1),
        "enfo_ws_mean": np.nanmean(ws, 1), "enfo_ws_sd": np.nanstd(ws, 1),
        "enfo_u_ctl": U[:, 0], "enfo_v_ctl": V[:, 0]}, index=uw.index).reset_index()
    ef = ef.rename(columns={"group": "kpx_group"})
    df = train_df.merge(ef, on=["forecast_kst_dtm", "kpx_group"], how="inner")
    M = member_uv(df)                                        # [N, 5, 2] 다모델 벡터
    for i, s in enumerate(["gfs100", "ldaps10", "ec", "jmsm", "jgsm"]):
        df[f"m_{s}_u"] = M[:, i, 0]
        df[f"m_{s}_v"] = M[:, i, 1]
    feats = [c for c in df.columns if c.startswith(("enfo_", "m_"))] + THERMO
    df[feats] = df[feats].ffill().bfill()
    return df, feats


def nll_np(y, mu, sig, rho):
    zu = (y[:, 0] - mu[:, 0]) / sig[:, 0]
    zv = (y[:, 1] - mu[:, 1]) / sig[:, 1]
    om = 1 - rho ** 2
    return (np.log(sig[:, 0]) + np.log(sig[:, 1]) + 0.5 * np.log(om)
            + (zu ** 2 + zv ** 2 - 2 * rho * zu * zv) / (2 * om)
            + np.log(2 * np.pi))


def fit_drn(Xtr, gtr, Ytr, Xva, gva, n_nets=5, hidden=64, emb=4,
            epochs=150, patience=10):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_val = max(500, int(len(Xtr) * 0.15))
    Xt, Yt, gt = Xtr[:-n_val], Ytr[:-n_val], gtr[:-n_val]
    Xv, Yv, gv = Xtr[-n_val:], Ytr[-n_val:], gtr[-n_val:]

    class DRN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(3, emb)
            self.h = nn.Linear(Xtr.shape[1] + emb, hidden)
            self.out = nn.Linear(hidden, 5)

        def forward(self, x, g):
            z = torch.relu(self.h(torch.cat([x, self.emb(g)], -1)))
            o = self.out(z)
            mu = o[:, :2]
            sig = nn.functional.softplus(o[:, 2:4]) + 1e-3
            rho = torch.tanh(o[:, 4]) * 0.99
            return mu, sig, rho

    def nll_t(y, mu, sig, rho):
        zu = (y[:, 0] - mu[:, 0]) / sig[:, 0]
        zv = (y[:, 1] - mu[:, 1]) / sig[:, 1]
        om = 1 - rho ** 2
        return (torch.log(sig[:, 0]) + torch.log(sig[:, 1]) + 0.5 * torch.log(om)
                + (zu ** 2 + zv ** 2 - 2 * rho * zu * zv) / (2 * om)).mean()

    tens = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    Xt_, Yt_, gt_ = tens(Xt), tens(Yt), tens(gt, torch.long)
    Xv_, Yv_, gv_ = tens(Xv), tens(Yv), tens(gv, torch.long)
    Xva_, gva_ = tens(Xva), tens(gva, torch.long)
    params = []
    for seed in range(n_nets):
        torch.manual_seed(seed)
        net = DRN().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, best_state, bad = np.inf, None, 0
        idx = torch.randperm(len(Xt_), device=dev)
        for ep in range(epochs):
            net.train()
            for s in range(0, len(idx), 1024):
                b = idx[s:s + 1024]
                opt.zero_grad()
                loss = nll_t(Yt_[b], *net(Xt_[b], gt_[b]))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(nll_t(Yv_, *net(Xv_, gv_)))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            mu, sig, rho = net(Xva_, gva_)
        params.append((mu.cpu().numpy(), sig.cpu().numpy(), rho.cpu().numpy()))
    mu = np.mean([p[0] for p in params], 0)
    sig = np.mean([p[1] for p in params], 0)
    rho = np.mean([p[2] for p in params], 0)
    return mu, sig, np.clip(rho, -0.95, 0.95)


def fit_gbmparam(Xtr, Ytr, Xva, e56):
    """평균 LGBM ×2 + 교차적합 잔차로 σ LGBM ×2 · ρ 상수."""
    import lightgbm as lgb

    def mk(obj="mae"):
        return lgb.LGBMRegressor(n_estimators=e56["lgb_n"], objective=obj,
                                 random_state=0, n_jobs=8, verbose=-1, **e56["lgb"])
    half = len(Xtr) // 2
    res = np.empty_like(Ytr)
    for a, b in ((slice(0, half), slice(half, None)),
                 (slice(half, None), slice(0, half))):
        for j in range(2):
            m = mk().fit(Xtr[a], Ytr[a][:, j])
            res[b, j] = Ytr[b][:, j] - m.predict(Xtr[b])
    mu_va = np.stack([mk().fit(Xtr, Ytr[:, j]).predict(Xva) for j in range(2)], -1)
    sig_va = np.stack([np.sqrt(np.clip(np.exp(
        mk("l2").fit(Xtr, np.log(res[:, j] ** 2 + 1e-3)).predict(Xva)), 0.05, None))
        for j in range(2)], -1)
    sd = res.std(0)
    rho = float(np.clip(np.corrcoef((res / sd).T)[0, 1], -0.95, 0.95))
    return mu_va, sig_va, np.full(len(Xva), rho)


def main():
    t0 = time.time()
    print("### AN-18 Buchner 2026 — BIV-DRN + GBM-파라미터 ###", flush=True)
    train_df, _ = build_train_frame()
    vec = build_scada_vec()
    for g in [1, 2, 3]:
        train_df = train_df.merge(
            vec[g].assign(kpx_group=g), on=["forecast_kst_dtm", "kpx_group"],
            how="left", suffixes=("", "_d"))
        for c in ["park_ws", "u_obs", "v_obs"]:
            if f"{c}_d" in train_df:
                train_df[c] = train_df[c].fillna(train_df.pop(f"{c}_d"))
    df, feats = build_features(train_df)
    print(f"피처 프레임 {df.shape} | 피처 {len(feats)}개 — {time.time()-t0:.0f}s",
          flush=True)
    import json
    e56 = json.loads(E56_JSON.read_text())
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        tt = time.time()
        tr_all, va_all = slice_fold(df, folds[fname])
        tr = tr_all[tr_all["u_obs"].notna()].sort_values(
            ["forecast_kst_dtm", "kpx_group"]).reset_index(drop=True)
        va = va_all.sort_values(["forecast_kst_dtm", "kpx_group"]).reset_index(drop=True)
        vam = va["u_obs"].notna().to_numpy()
        mu_ = tr[feats].mean()
        sd_ = tr[feats].std().replace(0, 1)
        Xtr = ((tr[feats] - mu_) / sd_).to_numpy(dtype=np.float32)
        Xva = ((va[feats] - mu_) / sd_).to_numpy(dtype=np.float32)
        Ytr = tr[["u_obs", "v_obs"]].to_numpy(dtype=np.float32)
        gtr = tr["kpx_group"].to_numpy() - 1
        gva = va["kpx_group"].to_numpy() - 1
        y_vec = va[["u_obs", "v_obs"]].to_numpy()[vam]
        y_ws = va["park_ws"].to_numpy()[vam]

        outs = {"drn": fit_drn(Xtr, gtr, Ytr, Xva, gva)}
        gm_mu = np.full((len(va), 2), np.nan)
        gm_sig = np.full((len(va), 2), np.nan)
        gm_rho = np.full(len(va), np.nan)
        for g in [1, 2, 3]:
            tm = gtr == g - 1
            vm = gva == g - 1
            m, s, r = fit_gbmparam(Xtr[tm], Ytr[tm], Xva[vm], e56)
            gm_mu[vm], gm_sig[vm], gm_rho[vm] = m, s, r
        outs["gbmparam"] = (gm_mu, gm_sig, gm_rho)

        for name, (mu, sig, rho) in outs.items():
            z1 = RNG.standard_normal((vam.sum(), N_SAMP))
            z2 = RNG.standard_normal((vam.sum(), N_SAMP))
            r_ = rho[vam][:, None]
            su = mu[vam][:, None, 0] + sig[vam][:, None, 0] * z1
            sv = mu[vam][:, None, 1] + sig[vam][:, None, 1] * (
                r_ * z1 + np.sqrt(1 - r_ ** 2) * z2)
            samp = np.stack([su, sv], -1)
            spd = np.linalg.norm(samp, axis=2)
            ang = np.abs((np.rad2deg(np.arctan2(mu[vam, 1], mu[vam, 0])
                          - np.arctan2(y_vec[:, 1], y_vec[:, 0])) + 180) % 360 - 180)
            recs.append(dict(
                fold=fname, model=name,
                ws_mae=np.abs(spd.mean(1) - y_ws).mean(),
                ws_crps=crps_ens(spd, y_ws).mean(),
                dir_mae=ang.mean(),
                vec_rmse=np.sqrt(((mu[vam] - y_vec) ** 2).sum(1).mean()),
                es=energy_score(samp, y_vec).mean(),
                logs=nll_np(y_vec, mu[vam], sig[vam], rho[vam]).mean()))
        print(f"  [{fname}] {time.time()-tt:.0f}s", flush=True)

    out = pd.DataFrame(recs)
    agg = out.groupby("model")[["ws_mae", "ws_crps", "dir_mae", "vec_rmse", "es",
                                "logs"]].mean().sort_values("es")
    print("\n=== 4폴드 평균 — 참조: EMOS premix_hyb ws1.50/dir21.4°/ES1.73 · "
          "GBM 점벡터 dir17.2°/vecRMSE2.13 · AnEn ws_crps 0.875 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    out.to_csv(ROOT / "open (1)" / "testing" / "an18_results.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n상세 → testing/an18_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
