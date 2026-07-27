# ═══════════════════════════════════════════════════════
# exp-047: 시퀀스 DL (24h 일블록 BiGRU) — 아키텍처 다양성 축
#
# 배경: 지금까지 모든 DL이 행 단위 MTMLP (exp-033~045). 손실 축(exp-039)·τ 축(exp-045)·
#   입력 축(exp-037/038)·조건부 w(exp-046)는 소진. 남은 미탐색 축 = **아키텍처 귀납 편향**.
#   예보는 전날 13시에 다음날 24h가 한 번에 제공되므로(리크 없음) 하루 24h 블록을 통째로
#   보는 시퀀스 모델은 램프·전선 통과 같은 시간 구조를 행 단위 MLP와 다르게 학습한다
#   → 오차 탈상관(exp-035 실증 기전)의 마지막 미탐색 경로.
#   변경점 격리: 손실은 q60(τ=0.60, 현 최선 성분) 고정, 입력 spd 348 고정 — 아키텍처만 변경.
#
# 아키텍처: Linear proj(348+emb→256)+LN+SiLU → BiGRU(hid 128×2) → LN+SiLU → 그룹별 head.
#   일블록 (B, 24, F), 결측 슬롯은 0(스케일 후 평균) 입력 + 손실 마스크.
#
# 평가: gspd base(exp-040 캐시) + 신PP(c16·al40·sn93·pt27).
#   (a) 대체 파트너: gspd+seq, w{0.75,0.70,0.65}
#   (b) 3멤버: gspd + 0.5(mlp_q60+seq), w{0.75,0.70,0.65}
#   ref = gspd·q60(MLP)·w0.70·신PP (재현 기대 CV 0.6481 / HOLD 0.6463)
#   진단: seq 단독 MAE·signed, corr(gspd err, seq err), corr(mlp_q60 err, seq err)
# 판정: 3중 조건 (양 게이트 +0.001 / fold 일관 / 두 지표 동반 상승)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp047_seq_dl.py
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from exp005_qopt import CAPACITY_KWH, apply_rolling_smoothing, apply_bias_correction_v2
from exp019_multigrid import HOLD_FOLD
from exp033_dl import DEVICE, GIDX
from exp035_raw_dl import (build_inputs, load_gbm_cache, score_from_tidy,
                           FOLD_LIST, CACHE_DIR)
from exp039_dl_diverse import DL_PRED_CACHE as E39_CACHE, build_slices, _huber
from exp040_gbm_spd import GBM_CACHE as EXT_GBM_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEY = ["fold", "group", "forecast_kst_dtm"]
DL_PRED_CACHE = CACHE_DIR / "exp047_dl_preds.parquet"
TAU = 0.60
SEQ_CFG = dict(proj=256, hid=128, p_drop=0.15, epochs=120, lr=1e-3, wd=1e-4, bs=64)

PP_CHAMP = dict(clip_lo=0.16, alpha=0.50, snap_thr=0.30, snap_to=0.94)
PP_NEW   = dict(clip_lo=0.16, alpha=0.40, snap_thr=0.27, snap_to=0.93)
BLEND_WS = [0.75, 0.70, 0.65]


class SeqGRU(nn.Module):
    def __init__(self, n_feat, emb=8, proj=256, hid=128, p_drop=0.15):
        super().__init__()
        self.emb = nn.Embedding(3, emb)
        self.proj = nn.Sequential(nn.Linear(n_feat + emb, proj), nn.LayerNorm(proj),
                                  nn.SiLU(), nn.Dropout(p_drop))
        self.gru = nn.GRU(proj, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.post = nn.Sequential(nn.LayerNorm(2 * hid), nn.SiLU(), nn.Dropout(p_drop))
        self.heads = nn.ModuleList([nn.Linear(2 * hid, 1) for _ in range(3)])

    def forward(self, x, g):          # x: (B,24,F), g: (B,)
        ge = self.emb(g).unsqueeze(1).expand(-1, x.shape[1], -1)
        h = self.proj(torch.cat([x, ge], dim=2))
        h, _ = self.gru(h)
        h = self.post(h)
        out = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        for gi in range(3):
            m = g == gi
            if m.any():
                out[m] = self.heads[gi](h[m]).squeeze(2)
        return out


def loss_q60(pred, tgt, w):
    e = pred - tgt
    asym = torch.where(e < 0, 2.0 * TAU, 2.0 * (1.0 - TAU))
    return (_huber(e) * asym * w).sum() / w.sum()


def build_blocks(df, cols, imp, sca, fit=False):
    """long rows → (X(n,24,F), g(n), Y(n,24), W(n,24), row_bi, row_hr).
    fit=True면 imputer/scaler를 target 보유 행으로 적합."""
    sub = df.reset_index(drop=True)
    dt = pd.to_datetime(sub["forecast_kst_dtm"])
    dates = dt.dt.normalize()
    hours = dt.dt.hour.to_numpy()
    if fit:
        fit_rows = sub["target"].notna().to_numpy()
        imp.fit(sub.loc[fit_rows, cols].to_numpy(dtype=np.float32))
        sca.fit(imp.transform(sub.loc[fit_rows, cols].to_numpy(dtype=np.float32)))
    Xs = sca.transform(imp.transform(sub[cols].to_numpy(dtype=np.float32))).astype(np.float32)

    blk = pd.DataFrame({"g": sub["kpx_group"].to_numpy(), "d": dates})
    keys = blk.drop_duplicates().reset_index(drop=True)
    bid_map = {(g, d): i for i, (g, d) in enumerate(zip(keys["g"], keys["d"]))}
    bi = np.array([bid_map[(g, d)] for g, d in zip(blk["g"], blk["d"])], dtype=np.int64)
    n = len(keys)

    X = np.zeros((n, 24, len(cols)), dtype=np.float32)
    Y = np.zeros((n, 24), dtype=np.float32)
    W = np.zeros((n, 24), dtype=np.float32)
    X[bi, hours] = Xs
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    cf = (sub["target"].to_numpy() / caps).astype(np.float32)
    has_y = sub["target"].notna().to_numpy()
    Y[bi[has_y], hours[has_y]] = cf[has_y]
    W[bi[has_y], hours[has_y]] = np.where(cf[has_y] < 0.10, 0.05, 1.0)
    g_blk = keys["g"].map(GIDX).to_numpy().astype(np.int64)
    return X, g_blk, Y, W, bi, hours


def train_seq(X, g, Y, W, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SeqGRU(X.shape[2], proj=cfg["proj"], hid=cfg["hid"], p_drop=cfg["p_drop"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    Xt = torch.tensor(X, device=DEVICE); gt = torch.tensor(g, device=DEVICE)
    Yt = torch.tensor(Y, device=DEVICE); Wt = torch.tensor(W, device=DEVICE)
    n, bs = Xt.shape[0], cfg["bs"]
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_q60(model(Xt[idx], gt[idx]), Yt[idx], Wt[idx])
            loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_seq(model, X, g, bs=256):
    model.eval()
    out = []
    Xt = torch.tensor(X, device=DEVICE); gt = torch.tensor(g, device=DEVICE)
    for i in range(0, Xt.shape[0], bs):
        out.append(model(Xt[i:i + bs], gt[i:i + bs]).cpu().numpy())
    return np.concatenate(out)


def seq_fold_predict(tr_df, va_df, cols, seeds, cfg):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, Ytr, Wtr, _, _ = build_blocks(tr_df, cols, imp, sca, fit=True)
    Xva, gva, _, _, bi_va, hr_va = build_blocks(va_df, cols, imp, sca, fit=False)
    preds = []
    for s in range(seeds):
        m = train_seq(Xtr, gtr, Ytr, Wtr, cfg, s)
        P = predict_seq(m, Xva, gva)       # (n_blk, 24)
        preds.append(P[bi_va, hr_va])      # 행 순서 복원
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def postprocess_flex(cf_seq, cap, ws_seq, pA_seq, group, bias_map,
                     clip_lo, alpha, snap_thr, snap_to):
    base = apply_bias_correction_v2(cf_seq * cap, ws_seq, group, bias_map)
    base = apply_rolling_smoothing(pd.Series(base), 3, alpha).to_numpy().copy()
    base[pA_seq >= snap_thr] = np.maximum(base[pA_seq >= snap_thr], cap * snap_to)
    return np.clip(base, cap * clip_lo, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-047: 24h 일블록 BiGRU (q60 손실, device={DEVICE}) ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    cache, champ_bias = load_gbm_cache()
    cache = cache.rename(columns={"gbm_cf": "cf_g132"})
    print(f"입력 spd={len(vcols)}개 / GBM 캐시 {cache.shape}", flush=True)

    ext = pd.read_parquet(EXT_GBM_CACHE)
    m = cache[KEY].merge(ext[KEY + ["cf_gspd"]], on=KEY, how="left")
    assert m["cf_gspd"].notna().all()
    cache["cf_gspd"] = m["cf_gspd"].to_numpy()
    dl = pd.read_parquet(E39_CACHE)
    for c in ["dl_q60", "dl_hub"]:
        mm = cache[KEY].merge(dl[KEY + [c]], on=KEY, how="left")
        assert mm[c].notna().all()
        cache[c] = mm[c].to_numpy()

    folds = FOLD_LIST
    fold_names = [fn for fn, _ in folds]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]

    # ── 시퀀스 DL 학습 (parquet 캐시) ──
    seq_pred = None
    if DL_PRED_CACHE.exists() and not args.refresh:
        prev = pd.read_parquet(DL_PRED_CACHE)
        mm = cache[KEY].merge(prev[KEY + ["dl_seq"]], on=KEY, how="left")
        if mm["dl_seq"].notna().all():
            seq_pred = mm["dl_seq"].to_numpy()
            print("exp-047 seq 캐시 로드", flush=True)

    if seq_pred is None:
        out = np.full(len(cache), np.nan)
        for fname, fold in folds:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), f"{fname} 검증행 정렬 실패"
            out[sel] = seq_fold_predict(tr_all, va, vcols, args.seeds, SEQ_CFG)
            print(f"  [seq:{fname}] {time.time()-tf:.0f}s", flush=True)
        seq_pred = out
        cache["dl_seq"] = seq_pred
        cache[KEY + ["dl_seq"]].to_parquet(DL_PRED_CACHE, index=False)
        print(f"seq 캐시 저장 → {DL_PRED_CACHE.name}", flush=True)
    cache["dl_seq"] = seq_pred
    cache["dl_q60+seq"] = 0.5 * (cache["dl_q60"].to_numpy() + seq_pred)

    # ── 평가 ──
    slices = build_slices(cache)
    gspd = cache["cf_gspd"].to_numpy()

    def evaluate(dl_col, w, pp, gbm_col=None):
        gbm = cache[gbm_col].to_numpy() if gbm_col else gspd
        blend = w * gbm + (1 - w) * cache[dl_col].to_numpy()
        parts = []
        for (fname, group), sub in slices:
            cap = CAPACITY_KWH[f"kpx_group_{group}"]
            pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                    sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
            parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
        tidy = pd.concat(parts, ignore_index=True)
        return dict(cv=score_from_tidy(tidy[tidy["fold"] != hold], "pred"),
                    hold=score_from_tidy(tidy[tidy["fold"] == hold], "pred"),
                    folds={fn: score_from_tidy(tidy[tidy["fold"] == fn], "pred")[0] for fn in cv_folds})

    r_champ = evaluate("dl_hub", 0.70, PP_CHAMP, gbm_col="cf_g132")
    r_ref   = evaluate("dl_q60", 0.70, PP_NEW)
    c_cv, c_hd = r_champ["cv"][0], r_champ["hold"][0]
    ref_cv, ref_hd = r_ref["cv"][0], r_ref["hold"][0]
    ref_n, ref_f = r_ref["cv"][1], r_ref["cv"][2]
    print(f"\n챔피언(구PP): CV {c_cv:.4f} | HOLD {c_hd:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP): CV {ref_cv:.4f} (1-NMAE {ref_n:.4f} FiCR {ref_f:.4f}) | "
          f"HOLD {ref_hd:.4f}  (기대 0.6481/0.6463)")

    res = {}
    for cnd in ["seq", "q60+seq"]:
        for w in BLEND_WS:
            res[f"{cnd}_w{int(w*100)}"] = evaluate(f"dl_{cnd}", w, PP_NEW)

    print("\n" + "=" * 108)
    print(f"{'변형':<14}{'CV':>8}{'ΔCVref':>9}{'HOLD':>8}{'Δhdref':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}{'ΔCVch':>9}{'Δhdch':>9}  판정")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ref_cv, kv[1]["hold"][0] - ref_hd)):
        dcv, dhd = r["cv"][0] - ref_cv, r["hold"][0] - ref_hd
        dn, df_ = r["cv"][1] - ref_n, r["cv"][2] - ref_f
        fold_d = [r["folds"][fn] - r_ref["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<14}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}"
              f"{r['cv'][0]-c_cv:>+9.4f}{r['hold'][0]-c_hd:>+9.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))

    print(f"\n--- fold별 ΔCV (ref 대비) ---")
    print(f"{'변형':<14}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"])):
        print(f"{r['name']:<14}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # ── 진단 ──
    caps = cache["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = cache["target"].to_numpy() / caps
    ev = (cache["target"].to_numpy() >= caps * 0.10) & np.isfinite(acf)
    eg = gspd - acf
    em = cache["dl_q60"].to_numpy() - acf
    es = cache["dl_seq"].to_numpy() - acf
    print("\n--- 진단 (평가대상 시간) ---")
    print(f"  MAE(CF): gspd {np.abs(eg[ev]).mean():.4f} / mlp_q60 {np.abs(em[ev]).mean():.4f} / "
          f"seq_q60 {np.abs(es[ev]).mean():.4f}")
    print(f"  signed:  gspd {eg[ev].mean():+.4f} / mlp_q60 {em[ev].mean():+.4f} / seq_q60 {es[ev].mean():+.4f}")
    print(f"  corr(gspd, mlp) {np.corrcoef(eg[ev], em[ev])[0,1]:.4f} | "
          f"corr(gspd, seq) {np.corrcoef(eg[ev], es[ev])[0,1]:.4f} | "
          f"corr(mlp, seq) {np.corrcoef(em[ev], es[ev])[0,1]:.4f}")

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
