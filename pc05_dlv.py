# ═══════════════════════════════════════════════════════
# PC-05: 파워커브 트랙 R5 — **v-공간 DL 파트너** (트랙을 실제로 독립시키는 라운드)
#
# 트랙 규칙: 판정 기준 = 트랙 v1(cf_pcinv × dl_cln03 w65 + PP_NEW, CV 0.6530 / HOLD 0.6512).
#   챔피언(0.6556/0.6535)은 참고. 챔피언 초과 주장 시 **동일 변경을 챔피언에도 적용한 통제 필수**.
#
# 배경: PC-01~04는 파이프라인 7부품 중 **타깃 1개만** 바뀐 상태였음 — DL 파트너는
#   챔피언의 `dl_cln03`(kWh/CF 공간·q60)을 그대로 차용. corr(cf_pcinv, cf_hpo)=0.9974가
#   4라운드 내내 안 떨어진 직접 원인. DL까지 v-공간으로 옮겨야 별도 파이프라인 자격.
#
# 변경점: 기존 DL(spd 348 원시 다격자·MTMLP 384×3·q60 expectile·cln03 가중)의 **타깃만**
#   교체. 그룹별 파워커브(fold-safe, PC-01과 동일 변환)로
#     학습 타깃  t = (PC_g⁻¹(cf) − v_lo) / (v_hi − v_lo)   ∈ [0,1]
#     예측       cf = PC_g(v_lo + t̂·(v_hi − v_lo))
#   ※ [0,1] 정규화 필수 — 손실이 Huber(δ=0.05)라 m/s 원단위면 δ가 무의미해짐(사실상 L1).
#
# 변형 2개 (q60 비대칭을 어느 공간에서 걸 것인가):
#   dlv  : t-공간에서 q60 (구현 단순, 커브 볼록성 때문에 CF 공간 비대칭과 등가 아님)
#   dlvc : **미분가능 커브**를 torch로 통과시켜 **CF 공간에서 q60** (손실 기하는 챔피언과
#          동일하게 두고 학습 표현만 매끄러운 v-공간 — 원리적으로 가장 정합)
#
# 스모크: Q2+Q4 2폴드·1-seed (Q4 단독은 3연속 과대평가 실적) → DL 단독 CF-MAE·signed·corr
# 게이트: 5폴드·3-seed → 트랙 블렌드(cf_pcinv × dl_new) 사다리, 판정 Δ = v1
#
# 실행: C:\miniconda3\python.exe -X utf8 pc05_dlv.py --stage smoke
#       C:\miniconda3\python.exe -X utf8 pc05_dlv.py --stage gate --variants dlvc
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp015_scada_pc import PC_CENTERS, build_scada_hourly
from exp033_dl import DEVICE, GIDX, MTMLP, predict
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, DL_CFG
from exp039_dl_diverse import DELTA, TAU, _huber, train_one
from exp049_clean import KEY, PP_NEW, avail_factor, merge_avail
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import corr_on_eval
from pc01_pcinv import make_transform, build_curve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

V1_CACHE = CACHE_DIR / "exp082_gbm_pcinv.parquet"
SMOKE_FOLDS = ["fold_2024Q2", "fold_2024Q4"]
VARIANTS = ["dlv", "dlvc"]


# ── 그룹별 커브 (fold-safe) ──
def fold_curves(train_df, scada_h, fold):
    """그룹 → (fwd, inv, v_lo, v_hi, curve_array). train_end 이전 SCADA만 사용."""
    tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
    out = {}
    for g in [1, 2, 3]:
        tr_g = tr_all.loc[(tr_all["kpx_group"] == g) & tr_all["target"].notna()]
        if len(tr_g) == 0:
            continue
        curve = build_curve(scada_h[g], tr_g, fold["train_end"], None)
        if curve is None:
            continue
        fwd, inv, _, rng = make_transform(curve)
        v_lo, v_hi = float(inv(rng[0])), float(inv(rng[1]))
        out[g] = dict(fwd=fwd, inv=inv, v_lo=v_lo, v_hi=v_hi,
                      curve=np.maximum.accumulate(np.asarray(curve, float)))
    return out


def cf_to_t(cf, gcol, curves):
    """행별 그룹 커브로 CF → 정규화 유효풍속 t ∈ [0,1]"""
    t = np.empty(len(cf), dtype=np.float32)
    for g, c in curves.items():
        m = gcol == g
        if m.any():
            t[m] = (c["inv"](cf[m]) - c["v_lo"]) / (c["v_hi"] - c["v_lo"])
    return np.clip(t, 0.0, 1.0)


def t_to_cf(t, gcol, curves):
    out = np.empty(len(t), dtype=np.float64)
    for g, c in curves.items():
        m = gcol == g
        if m.any():
            out[m] = c["fwd"](c["v_lo"] + np.clip(t[m], 0.0, 1.0) * (c["v_hi"] - c["v_lo"]))
    return out


# ── 미분가능 커브 (dlvc용): t → cf, torch 구간선형 보간 ──
def make_torch_curve(curves, device):
    import torch
    xs, ys = [], []
    for gi in range(3):
        c = curves.get(gi + 1)
        if c is None:                       # 결측 그룹은 항등에 가까운 더미 (실사용 없음)
            xs.append(np.linspace(0, 1, len(PC_CENTERS)))
            ys.append(np.linspace(0, 1, len(PC_CENTERS)))
            continue
        v = c["v_lo"] + np.linspace(0, 1, len(PC_CENTERS)) * (c["v_hi"] - c["v_lo"])
        xs.append(np.linspace(0, 1, len(PC_CENTERS)))
        ys.append(c["fwd"](v))
    X = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)   # [3, B]
    Y = torch.tensor(np.stack(ys), dtype=torch.float32, device=device)
    step = 1.0 / (X.shape[1] - 1)

    def fwd_t(t, g):
        tc = t.clamp(0.0, 1.0)
        pos = (tc / step).clamp(0, X.shape[1] - 1 - 1e-6)
        i0 = pos.floor().long()
        frac = pos - i0.float()
        y0 = Y[g, i0]
        y1 = Y[g, (i0 + 1).clamp(max=X.shape[1] - 1)]
        return y0 + frac * (y1 - y0)        # t에 대해 미분가능

    return fwd_t


def make_loss_v(kind, fwd_t):
    import torch

    if kind == "dlv":                       # t-공간 q60
        def f(pred, tgt, w, g=None):
            e = pred - tgt
            asym = torch.where(e < 0, 2.0 * TAU, 2.0 * (1.0 - TAU))
            return (_huber(e) * asym * w).sum() / w.sum()
    else:                                   # dlvc — CF 공간 q60 (커브 통과 후)
        def f(pred, tgt, w, g=None):
            e = fwd_t(pred, g) - fwd_t(tgt, g)
            asym = torch.where(e < 0, 2.0 * TAU, 2.0 * (1.0 - TAU))
            return (_huber(e) * asym * w).sum() / w.sum()
    return f


def train_one_v(Xtr, gtr, ytr, wtr, cfg, loss_fn, seed):
    """exp-039 train_one과 동일 구조 — 손실에 group 인자를 넘기는 것만 다름."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    model = MTMLP(Xtr.shape[1], cfg["width"], cfg["depth"], cfg["p_drop"]).to(DEVICE)
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
            loss = loss_fn(model(Xt[idx], gt[idx]), yt[idx], wt[idx], gt[idx])
            loss.backward(); opt.step()
        sched.step()
    return model


def dl_fold_predict_v(tr_df, va_df, cols, seeds, kind, curves):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    sub = tr_df.loc[tr_df["target"].notna()]
    caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    cf = (sub["target"].to_numpy() / caps).astype(np.float64)
    gcol = sub["kpx_group"].to_numpy().astype(int)

    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr = sca.fit_transform(imp.fit_transform(sub[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gtr = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    ytr = cf_to_t(cf, gcol, curves)                                   # ← 타깃 교체
    wtr = (np.where(cf < 0.10, 0.05, 1.0) *
           avail_factor(sub["avail_strict"].to_numpy(), "cln03")).astype(np.float32)

    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    gva_raw = va_df["kpx_group"].to_numpy().astype(int)

    fwd_t = make_torch_curve(curves, DEVICE)
    loss_fn = make_loss_v(kind, fwd_t)
    preds = []
    for s in range(seeds):
        m = train_one_v(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(t_to_cf(predict(m, Xva, gva), gva_raw, curves))  # ← 커브로 CF 복원
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def build_frames():
    train_df, cols, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)
    return train_df, colsets["spd"]


def va_frame(cache, train_df, fname):
    sel = (cache["fold"] == fname).to_numpy()
    keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
    va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                    right_on=["kpx_group", "forecast_kst_dtm"], how="left")
    assert len(va) == len(keys)
    return sel, va


def stage_smoke(train_df, cols, scada_h, cache, tags, seeds):
    print(f"[smoke] {'+'.join(f.replace('fold_','') for f in SMOKE_FOLDS)} 2폴드·{seeds}-seed "
          f"— DL 단독 CF-MAE (기준 = 챔피언 DL dl_cln03)", flush=True)
    for fname in SMOKE_FOLDS:
        fold = dict(FOLD_LIST)[fname]
        curves = fold_curves(train_df, scada_h, fold)
        print(f"  {fname}: 커브 " + " / ".join(
            f"g{g} v[{c['v_lo']:.2f},{c['v_hi']:.2f}]" for g, c in curves.items()), flush=True)
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        sel, va = va_frame(cache, train_df, fname)
        y = cache.loc[sel, "target"].to_numpy()
        caps = cache.loc[sel, "group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        m = ~np.isnan(y) & (y >= caps * 0.10)
        ycf = y / caps
        ref = cache.loc[sel, "dl_cln03"].to_numpy()
        print(f"    [기준 dl_cln03] MAE {np.abs(ref[m]-ycf[m]).mean():.5f} "
              f"| signed {(ref[m]-ycf[m]).mean():+.4f}", flush=True)
        for tag in tags:
            tt = time.time()
            p = dl_fold_predict_v(tr_all, va, cols, seeds, tag, curves)
            print(f"    [{tag}] MAE {np.abs(p[m]-ycf[m]).mean():.5f} "
                  f"(Δ기준 {np.abs(p[m]-ycf[m]).mean()-np.abs(ref[m]-ycf[m]).mean():+.5f}) "
                  f"| signed {(p[m]-ycf[m]).mean():+.4f} "
                  f"| corr(dl_cln03) {np.corrcoef(p[m], ref[m])[0,1]:.4f} "
                  f"| pred max {p.max():.3f} | {time.time()-tt:.0f}s", flush=True)
    print("\n통과 기준: **2폴드 모두** MAE Δ기준 ≤ +0.002 ∧ signed 양수 유지(파트너 역할 보존)",
          flush=True)


def stage_gate(train_df, cols, scada_h, cache, champ_bias, tags, seeds):
    for tag in tags:
        col = f"dl_{tag}"
        cpath = CACHE_DIR / f"pc05_{tag}.parquet"
        ppath = CACHE_DIR / f"pc05_{tag}_partial.parquet"
        if col not in merge_cached_cols(cache, cpath, [col]):
            # 폴드 단위 partial 캐시 — 중단돼도 완료 폴드는 보존 (exp-068 선례)
            out = np.full(len(cache), np.nan)
            if ppath.exists():
                prev = pd.read_parquet(ppath)
                mm = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")
                out = mm[col].to_numpy()
                done = [fn for fn, _ in FOLD_LIST
                        if not np.isnan(out[(cache["fold"] == fn).to_numpy()]).any()]
                print(f"  partial 재개: 완료 폴드 {done}", flush=True)
            for fname, fold in FOLD_LIST:
                sel, va = va_frame(cache, train_df, fname)
                if not np.isnan(out[sel]).any():
                    continue
                tf = time.time()
                curves = fold_curves(train_df, scada_h, fold)
                tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
                out[sel] = dl_fold_predict_v(tr_all, va, cols, seeds, tag, curves)
                cache[col] = out
                cache[KEY + [col]].to_parquet(ppath, index=False)      # ← 폴드마다 저장
                print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s (partial 저장)", flush=True)
            cache[col] = out
            save_cols(cache, cpath, [col])
            print(f"  캐시 저장 → {cpath.name}", flush=True)

    ev = make_ev(cache, champ_bias)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    print(f"\n[참고] 챔피언: CV {champ['cv'][0]:.4f} | HOLD {champ['hold'][0]:.4f}", flush=True)
    print(f"[기준] 트랙 v1: CV {v1['cv'][0]:.4f} | HOLD {v1['hold'][0]:.4f}"
          f"  (기대 0.6530/0.6512)", flush=True)
    res = {}
    for tag in tags:
        col = f"dl_{tag}"
        for b in ("dl_cln03", "cf_pcinv", "cf_hpo"):
            print(f"  corr({col}, {b}) = {corr_on_eval(cache, col, b):.4f}", flush=True)
        for w in (0.60, 0.65, 0.70):
            res[f"pc×{tag}_w{int(w*100)}"] = ev("cf_pcinv", col, w, PP_NEW)
        # 통제: 같은 DL을 **챔피언 base**에도 붙여봄 (트랙 고유분 분리)
        res[f"[통제]hpo×{tag}_w65"] = ev("cf_hpo", col, 0.65, PP_NEW)
    print_table(res, v1)
    print("\n※ Δ는 트랙 v1 대비. [통제] 행이 같이 오르면 DL 교체 효과는 트랙 고유가 아님.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--variants", default="dlv,dlvc")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke-seeds", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    print("### PC-05 v-공간 DL 파트너 ###", flush=True)
    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    for t in tags:
        assert t in VARIANTS, f"미정의 변형: {t}"

    train_df, cols = build_frames()
    scada_h = build_scada_hourly()
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, V1_CACHE, ["cf_pcinv"]), "PC-01 캐시 없음"
    print(f"입력 준비 완료 (DL {len(cols)}피처) — {time.time()-t0:.0f}s", flush=True)

    if args.stage == "smoke":
        stage_smoke(train_df, cols, scada_h, cache, tags, args.smoke_seeds)
    else:
        stage_gate(train_df, cols, scada_h, cache, champ_bias, tags, args.seeds)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
