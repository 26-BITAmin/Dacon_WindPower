# ═══════════════════════════════════════════════════════
# exp-070: DL 멤버 SSL 사전학습 — DAE / VIME형 MFR / SCARF (LG Aimers 4강 축)
#
# 설계 원칙 (exp-068 교훈): 아키텍처(MTMLP)·손실(q60)·가중(cln03)·입력(spd 348)·
#   레시피(DL_CFG)·3-seed 전부 고정 — **초기화만** 사전학습 가중치로 교체.
#   근거: exp-068에서 편향 프로파일이 아키텍처 의존(시드=초기화에는 안정)임이 관측됨
#   → 초기화 교체는 파트너 역할(상향 편향·탈상관) 보존 개연성이 높은 유일한 DL 개입.
#
# 사전학습 데이터: 2022~2024 학습 전 행(라벨 NaN 행 포함) + 2025 테스트 행 —
#   **피처만 사용** (라벨·SCADA 미사용). 테스트 피처는 예측 시점 제공 데이터라 적법,
#   배포 시에도 동일하게 사전학습 가능 → 게이트·제출 조건 일치. 2025 분포 친숙화가
#   시간 감쇠(exp-055/069) 반대 방향 효과 후보.
#   전처리: 전 행 global median impute + standard scale (파인튜닝은 폴드별 스케일러 —
#   분포 불일치는 2차 효과로 수용, 기록).
#
# 기법 (인코더 = MTMLP trunk + 그룹 임베딩, 사전학습 시드 42 고정):
#   dae   : swap-noise 15% → 전 셀 복원 MSE
#   mfr   : swap-noise 마스킹 25% → 마스크 셀 복원 MSE + 마스크 판별 BCE (1:1)
#   scarf : 30% 오염 뷰 vs 원본 InfoNCE (proj 128, τ0.1, 대칭)
#   사전학습: bs1024·AdamW lr1e-3 wd1e-5·cosine·40ep
#
# 파인튜닝·게이트: 변형별 3-seed·4-fold+홀드아웃 (시드별 head 초기화·배치 순서는
#   기존과 동일하게 시드 의존 유지, trunk·gemb만 사전학습값 로드). 폴드 단위 partial
#   캐시. 판정 Δ = 보드 챔피언(hpo×dC_w65, 기대 0.6556/0.6535), 진단 = exp-068 동일.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp070_ssl.py --variants dae,mfr,scarf
# ═══════════════════════════════════════════════════════
import argparse
import copy
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH
from exp012_feature_v11 import build_v11
from exp020_jma_features import build_jma_frame
from exp020_submit import add_jma
from exp033_dl import MTMLP, predict, GIDX, DEVICE
from exp035_grid_features import build_all as build_spatial_all
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, DL_CFG
from exp039_dl_diverse import make_loss
from exp049_clean import KEY, PP_NEW, merge_avail
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp068_tabm_plr import make_xy_clean, diag

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = ["dae", "mfr", "scarf"]
PRE_CFG = dict(epochs=40, bs=1024, lr=1e-3, wd=1e-5, seed=42)
P_CORR = {"dae": 0.15, "mfr": 0.25, "scarf": 0.30}


# ── 사전학습 데이터 (train 전 행 + test 행, 피처만) ──
def build_pretrain_matrix(train_df, use_cols):
    _, test_df, _, _ = build_v11()
    jma = build_jma_frame()
    test_df = add_jma(test_df, jma)
    _, spd, _, _ = build_spatial_all()
    test_df = test_df.merge(spd, on="forecast_kst_dtm", how="left")
    missing = [c for c in use_cols if c not in test_df.columns]
    assert not missing, f"테스트 피처 누락: {missing[:5]}"
    frames = [train_df[["kpx_group"] + use_cols], test_df[["kpx_group"] + use_cols]]
    allf = pd.concat(frames, ignore_index=True)
    imp, sca = SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    X = sca.fit_transform(imp.fit_transform(allf[use_cols].to_numpy(dtype=np.float32)))
    g = allf["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    print(f"사전학습 행렬: train {len(train_df)} + test {len(test_df)} = {len(allf)}행 × {len(use_cols)}피처",
          flush=True)
    return X.astype(np.float32), g


class SSLModel(nn.Module):
    """MTMLP의 emb+trunk를 인코더로 공유, 기법별 head 부착"""
    def __init__(self, n_feat, kind):
        super().__init__()
        self.core = MTMLP(n_feat, DL_CFG["width"], DL_CFG["depth"], DL_CFG["p_drop"])
        self.kind = kind
        w = DL_CFG["width"]
        if kind in ("dae", "mfr"):
            self.dec = nn.Linear(w, n_feat)
        if kind == "mfr":
            self.mask_head = nn.Linear(w, n_feat)
        if kind == "scarf":
            self.proj = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 128))

    def encode(self, x, g):
        return self.core.trunk(torch.cat([x, self.core.emb(g)], dim=1))


def swap_corrupt(Xb, X_all, p):
    """셀별 확률 p로 전체 데이터 임의 행의 같은 열 값으로 치환 (marginal swap)"""
    B, Fn = Xb.shape
    m = torch.rand(B, Fn, device=Xb.device) < p
    ridx = torch.randint(0, X_all.shape[0], (B, Fn), device=Xb.device)
    swap = X_all[ridx, torch.arange(Fn, device=Xb.device).expand(B, Fn)]
    return torch.where(m, swap, Xb), m.float()


def pretrain(kind, X, g, use_cols):
    torch.manual_seed(PRE_CFG["seed"]); np.random.seed(PRE_CFG["seed"])
    model = SSLModel(len(use_cols), kind).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=PRE_CFG["lr"], weight_decay=PRE_CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRE_CFG["epochs"])
    Xt = torch.tensor(X, device=DEVICE); gt = torch.tensor(g, device=DEVICE)
    n, bs, p = Xt.shape[0], PRE_CFG["bs"], P_CORR[kind]
    t0 = time.time()
    for ep in range(PRE_CFG["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = cnt = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, gb = Xt[idx], gt[idx]
            xc, m = swap_corrupt(xb, Xt, p)
            opt.zero_grad()
            if kind == "dae":
                loss = F.mse_loss(model.dec(model.encode(xc, gb)), xb)
            elif kind == "mfr":
                h = model.encode(xc, gb)
                rec = ((model.dec(h) - xb) ** 2 * m).sum() / m.sum().clamp(min=1)
                mk = F.binary_cross_entropy_with_logits(model.mask_head(h), m)
                loss = rec + mk
            else:  # scarf
                z1 = F.normalize(model.proj(model.encode(xb, gb)), dim=1)
                z2 = F.normalize(model.proj(model.encode(xc, gb)), dim=1)
                logits = z1 @ z2.T / 0.1
                lab = torch.arange(z1.shape[0], device=DEVICE)
                loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            loss.backward(); opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        sched.step()
        if ep % 10 == 0 or ep == PRE_CFG["epochs"] - 1:
            print(f"    [{kind}] ep{ep:3d} loss {tot/cnt:.5f}", flush=True)
    enc_state = {k: v for k, v in model.core.state_dict().items()
                 if k.startswith("trunk.") or k.startswith("emb.")}
    path = CACHE_DIR / f"exp070_enc_{kind}.pt"
    torch.save(enc_state, path)
    print(f"  [{kind}] 사전학습 완료 {time.time()-t0:.0f}s → {path.name}", flush=True)
    return enc_state


def train_one_init(Xtr, gtr, ytr, wtr, loss_fn, seed, enc_state):
    """exp-039 train_one과 동일 — 초기화만 enc_state(trunk·emb) 로드"""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MTMLP(Xtr.shape[1], DL_CFG["width"], DL_CFG["depth"], DL_CFG["p_drop"]).to(DEVICE)
    model.load_state_dict(enc_state, strict=False)   # heads는 시드별 초기화 유지
    opt = torch.optim.AdamW(model.parameters(), lr=DL_CFG["lr"], weight_decay=DL_CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DL_CFG["epochs"])
    Xt = torch.tensor(Xtr, device=DEVICE); gt = torch.tensor(gtr, device=DEVICE)
    yt = torch.tensor(ytr, device=DEVICE); wt = torch.tensor(wtr, device=DEVICE)
    n, bs = Xt.shape[0], 1024
    for _ in range(DL_CFG["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx], gt[idx]), yt[idx], wt[idx])
            loss.backward(); opt.step()
        sched.step()
    return model


def train_variant(train_df, cols, tag, seeds, enc_state):
    ppath = CACHE_DIR / f"exp070_dl_{tag}_partial.parquet"
    preds = [pd.read_parquet(ppath)] if ppath.exists() else []
    done = set(preds[0]["fold"].unique()) if preds else set()
    if done:
        print(f"  (partial 재사용: {sorted(done)})", flush=True)
    loss_fn = make_loss("q60")
    for fname, fold in FOLD_LIST:
        if fname in done:
            continue
        tf = time.time()
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        va_all = train_df.loc[(train_df["forecast_kst_dtm"] >= fold["valid_start"]) &
                              (train_df["forecast_kst_dtm"] <= fold["valid_end"])] \
                         .sort_values(["kpx_group", "forecast_kst_dtm"]).copy()
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sca = StandardScaler()
        Xtr, gtr, ytr, wtr = make_xy_clean(tr_all, cols, imp, sca, fit=True)
        Xva = sca.transform(imp.transform(va_all[cols].to_numpy(dtype=np.float32))).astype(np.float32)
        gva = va_all["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        ps = []
        for s in range(seeds):
            m = train_one_init(Xtr, gtr, ytr, wtr, loss_fn, s, enc_state)
            ps.append(predict(m, Xva, gva))
            del m
            torch.cuda.empty_cache()
        preds.append(pd.DataFrame({"fold": fold["name"], "group": va_all["kpx_group"].to_numpy(),
                                   "forecast_kst_dtm": va_all["forecast_kst_dtm"].to_numpy(),
                                   f"dl_{tag}": np.mean(ps, axis=0)}))
        pd.concat(preds, ignore_index=True).to_parquet(ppath, index=False)
        print(f"  [dl_{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
    return pd.concat(preds, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="dae,mfr,scarf")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tags = [t.strip() for t in args.variants.split(",")]
    t0 = time.time()
    print(f"### exp-070 SSL 사전학습 (device={DEVICE}) ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)
    use_cols = colsets["spd"]
    print(f"입력 준비 완료 (DL 입력 {len(use_cols)}) — {time.time()-t0:.0f}s", flush=True)

    # 사전학습 (인코더 캐시 재사용)
    enc_states = {}
    need = [t for t in tags
            if not (CACHE_DIR / f"exp070_enc_{t}.pt").exists()
            or not (CACHE_DIR / f"exp070_dl_{t}.parquet").exists()]
    Xp = gp = None
    for tag in tags:
        epath = CACHE_DIR / f"exp070_enc_{tag}.pt"
        if epath.exists():
            enc_states[tag] = torch.load(epath, map_location=DEVICE)
            print(f"  [{tag}] 인코더 캐시 로드", flush=True)
        else:
            if Xp is None:
                Xp, gp = build_pretrain_matrix(train_df, use_cols)
            enc_states[tag] = pretrain(tag, Xp, gp, use_cols)

    cache, champ_bias = load_cache_full()
    for tag in tags:
        col = f"dl_{tag}"
        cpath = CACHE_DIR / f"exp070_dl_{tag}.parquet"
        have = merge_cached_cols(cache, cpath, [col])
        if col not in have:
            print(f"[{tag}] 파인튜닝 게이트 시작", flush=True)
            p = train_variant(train_df, use_cols, tag, args.seeds, enc_states[tag])
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
