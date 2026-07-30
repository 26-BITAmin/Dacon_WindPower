# ═══════════════════════════════════════════════════════
# exp-068: DL 멤버 아키텍처 업그레이드 — TabM(파라미터 공유 앙상블) + PLR 수치 임베딩
#
# 배경 (LG Aimers 3강 접점):
#   - TabM (Gorishniy+, ICLR 2025): BatchEnsemble식 k-멤버 공유 MLP — 단일 MLP·딥 앙상블
#     모두 능가 보고. exp-047(GRU)의 실패 원인이 "GBM 상관↑·상향 편향 소실"(파트너 역할
#     훼손)이었는데, TabM은 같은 MLP 계열이라 멤버 역할 보존 가능성.
#   - PLR (Gorishniy+, NeurIPS 2022): 수치 피처의 주기 임베딩 — "임베딩이 백본만큼 중요".
#     현 MTMLP는 스칼라 입력 그대로.
#
# 변경 최소화 원칙: 손실(q60 τ0.60·δ0.05)·가중(저CF 0.05 × cln03)·입력(spd 348)·
#   학습 레시피(DL_CFG: 384×3·drop0.15·ep120·lr1e-3·wd1e-4·bs1024)·3-seed 전부 고정,
#   아키텍처만 교체. TabM의 BN은 (B·k) 평탄화 공유 적용(k=1이면 MTMLP와 동일 구조).
#   PLR: n_freq 16·d_emb 8·freq_init 0.1 (rtdl 기본 근방, 1회 고정 — 미세튜닝 없음).
#
# 변형: tabm (k=8) / plr / tabmplr (k=8 + PLR)
# 게이트: 변형별 3-seed·4-fold+홀드아웃 캐시(exp068_dl_{tag}.parquet) →
#   cf_hpo × dl_new 블렌드 w{0.60, 0.65, 0.70}. 판정 Δ = 보드 챔피언(hpo×dC_w65).
#   진단: DL 단독 eval-hour CF MAE·signed·corr(cf_hpo)·corr(dl_cln03)·평균 이동
#   (판별 규칙 — 상향 이동 동반 변경은 게이트 통과와 무관하게 2025 역전).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp068_tabm_plr.py --variants tabm,plr,tabmplr
# ═══════════════════════════════════════════════════════
import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH
from exp033_dl import GIDX, DEVICE
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, DL_CFG
from exp039_dl_diverse import make_loss
from exp049_clean import KEY, PP_NEW, avail_factor, merge_avail
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = {"tabm": dict(k=8, plr=False), "plr": dict(k=1, plr=True),
            "tabmplr": dict(k=8, plr=True)}
PLR_FREQ, PLR_DEMB, PLR_SIGMA = 16, 8, 0.1


class PLREmbed(nn.Module):
    """피처별 주기 임베딩: x_j → ReLU(Linear_j([sin, cos](2π c_j x_j)))"""
    def __init__(self, n_feat, n_freq=PLR_FREQ, d_emb=PLR_DEMB, sigma=PLR_SIGMA):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n_feat, n_freq) * sigma)
        self.W = nn.Parameter(torch.randn(n_feat, 2 * n_freq, d_emb) / math.sqrt(2 * n_freq))
        self.b = nn.Parameter(torch.zeros(n_feat, d_emb))
        self.out_dim = n_feat * d_emb

    def forward(self, x):                      # (B, n_feat)
        ang = 2 * math.pi * x.unsqueeze(-1) * self.c
        e = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (B, n_feat, 2F)
        z = torch.relu(torch.einsum("bnf,nfd->bnd", e, self.W) + self.b)
        return z.flatten(1)                    # (B, n_feat*d_emb)


class BELinear(nn.Module):
    """BatchEnsemble Linear: 공유 W + 멤버별 입/출력 스케일 r, s와 bias (TabM 방식)"""
    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        if k > 1:   # TabM 초기화: 랜덤 ±1 부호
            self.r = nn.Parameter((torch.randint(0, 2, (k, d_in)) * 2 - 1).float())
            self.s = nn.Parameter((torch.randint(0, 2, (k, d_out)) * 2 - 1).float())
        else:
            self.r = nn.Parameter(torch.ones(k, d_in))
            self.s = nn.Parameter(torch.ones(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):                      # (B, k, d_in)
        return self.lin(x * self.r) * self.s + self.b


class TabMMLP(nn.Module):
    """MTMLP의 TabM화: 공유 trunk(BatchEnsemble) + 그룹×멤버별 head. k=1·PLR off면 MTMLP 등가."""
    def __init__(self, n_feat, width, depth, p_drop, k=8, emb=8, use_plr=False):
        super().__init__()
        self.k = k
        self.gemb = nn.Embedding(3, emb)
        self.plr = PLREmbed(n_feat) if use_plr else None
        d_in = (self.plr.out_dim if use_plr else n_feat) + emb
        self.blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(BELinear(d_in, width, k))
            self.bns.append(nn.BatchNorm1d(width))
            d_in = width
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_drop)
        self.headW = nn.Parameter(torch.randn(3, k, width) / math.sqrt(width))
        self.headb = nn.Parameter(torch.zeros(3, k))

    def forward(self, x, g):                   # x (B, n_feat), g (B,)
        h = self.plr(x) if self.plr is not None else x
        h = torch.cat([h, self.gemb(g)], dim=1)
        h = h.unsqueeze(1).expand(-1, self.k, -1)          # (B, k, d_in)
        B = h.shape[0]
        for blk, bn in zip(self.blocks, self.bns):
            h = blk(h)
            h = bn(h.reshape(B * self.k, -1)).reshape(B, self.k, -1)
            h = self.drop(self.act(h))
        out = torch.zeros(B, self.k, device=h.device)
        for gi in range(3):
            m = g == gi
            if m.any():
                out[m] = torch.einsum("bkw,kw->bk", h[m], self.headW[gi]) + self.headb[gi]
        return out                              # (B, k)


def train_one_v(Xtr, gtr, ytr, wtr, cfg, loss_fn, seed, k, use_plr):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TabMMLP(Xtr.shape[1], cfg["width"], cfg["depth"], cfg["p_drop"],
                    k=k, use_plr=use_plr).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    Xt = torch.tensor(Xtr, device=DEVICE); gt = torch.tensor(gtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE); wt = torch.tensor(wtr, device=DEVICE)
    n, bs = Xt.shape[0], 1024
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Xt[idx], gt[idx])                       # (b, k)
            kk = pred.shape[1]
            loss = loss_fn(pred.flatten(),
                           yt[idx].repeat_interleave(kk), wt[idx].repeat_interleave(kk))
            loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_v(model, X, g, bs=8192):
    model.eval()
    out = []
    Xt = torch.tensor(X, device=DEVICE); gt = torch.tensor(g, device=DEVICE)
    for i in range(0, Xt.shape[0], bs):
        out.append(model(Xt[i:i + bs], gt[i:i + bs]).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def make_xy_clean(df, cols, imputer, scaler, fit=False):
    sub = df.loc[df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    Xraw = sub[cols].to_numpy(dtype=np.float32)
    Xs = scaler.fit_transform(imputer.fit_transform(Xraw)) if fit else scaler.transform(imputer.transform(Xraw))
    g = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    w = np.where(cf < 0.10, 0.05, 1.0).astype(np.float32)
    w = (w * avail_factor(sub["avail_strict"].to_numpy(), "cln03")).astype(np.float32)
    return Xs.astype(np.float32), g, cf, w


def dl_fold_predict_v(tr_df, va_df, cols, seeds, k, use_plr):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(tr_df, cols, imp, sca, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds = []
    for s in range(seeds):
        m = train_one_v(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s, k, use_plr)
        preds.append(predict_v(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def train_variant(train_df, cols, tag, seeds):
    """폴드 단위 중간 저장(_partial) — 프로세스 중단 시 완료 폴드 재사용."""
    vc = VARIANTS[tag]
    ppath = CACHE_DIR / f"exp068_dl_{tag}_partial.parquet"
    preds = [pd.read_parquet(ppath)] if ppath.exists() else []
    done = set(preds[0]["fold"].unique()) if preds else set()
    if done:
        print(f"  (partial 재사용: {sorted(done)})", flush=True)
    for fname, fold in FOLD_LIST:
        if fname in done:
            continue
        tf = time.time()
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                              (train_df["forecast_kst_dtm"] <= fold["valid_end"])] \
                         .sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        p = dl_fold_predict_v(tr_all, va_all, cols, seeds, vc["k"], vc["plr"])
        preds.append(pd.DataFrame({"fold": fold["name"], "group": va_all["kpx_group"].to_numpy(),
                                   "forecast_kst_dtm": va_all["forecast_kst_dtm"].to_numpy(),
                                   f"dl_{tag}": p}))
        pd.concat(preds, ignore_index=True).to_parquet(ppath, index=False)
        print(f"  [dl_{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
    return pd.concat(preds, ignore_index=True)


def diag(cache, col):
    m = cache["target"].notna()
    caps = cache.loc[m, "group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache.loc[m, "target"].to_numpy() / caps
    ev = acf >= 0.10
    err = cache.loc[m, col].to_numpy() - acf
    c_gbm = np.corrcoef(cache.loc[m, col], cache.loc[m, "cf_hpo"])[0, 1]
    c_dl = np.corrcoef(cache.loc[m, col], cache.loc[m, "dl_cln03"])[0, 1]
    shift = (cache[col].to_numpy() - cache["dl_cln03"].to_numpy()).mean() * 100
    print(f"  {col}: eval MAE {np.abs(err[ev]).mean():.4f} | signed {err[ev].mean():+.4f} | "
          f"corr(cf_hpo) {c_gbm:.4f} | corr(dl_cln03) {c_dl:.4f} | 평균이동 vs dC {shift:+.3f}%cap",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="tabm,plr,tabmplr")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tags = [t.strip() for t in args.variants.split(",")]
    t0 = time.time()
    print(f"### exp-068 TabM + PLR (device={DEVICE}) ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)
    use_cols = colsets["spd"]
    print(f"입력 준비 완료 (DL 입력 {len(use_cols)}) — {time.time()-t0:.0f}s", flush=True)

    cache, champ_bias = load_cache_full()
    for tag in tags:
        col = f"dl_{tag}"
        cpath = CACHE_DIR / f"exp068_dl_{tag}.parquet"
        have = merge_cached_cols(cache, cpath, [col])
        if col not in have:
            print(f"[{tag}] 학습 시작 (k={VARIANTS[tag]['k']}, plr={VARIANTS[tag]['plr']})", flush=True)
            p = train_variant(train_df, use_cols, tag, args.seeds)
            m = cache[KEY].merge(p, on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패: {col}"
            cache[col] = m[col].to_numpy()
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name} ({(time.time()-t0)/60:.1f}분)", flush=True)

    print("\n--- 진단 (dl_cln03 기준) ---", flush=True)
    diag(cache, "dl_cln03")
    for tag in tags:
        diag(cache, f"dl_{tag}")

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {}
    for tag in tags:
        for w in (0.60, 0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev("cf_hpo", f"dl_{tag}", w, PP_NEW)
    print_table(res, ref)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
