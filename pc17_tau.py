# ═══════════════════════════════════════════════════════
# PC-17 (R17): dlvc τ 스윕 — v-공간 DL의 비대칭 계수 (트랙 마지막 잔여 레버)
#
# 근거: dlvc는 CF-공간 q60(τ=0.60)로 학습된 상태. 메인 라인 τ 스윕(exp-045/051)은
#   dl_cln03(kWh 공간) 대상이었고 v-타깃 DL의 τ는 미측정. 심플렉스 dlvc 지분 0.15~0.20.
# 구성: τ ∈ {0.55, 0.65} 각 3-seed (pc05 하네스 재사용, 손실만 τ 파라미터화).
# 판정: PC-16 정점 (0.775, 0, 0.15, 0.075)에서 dlvc 치환 + 가중 지터.
#
# 실행: C:\miniconda3\python.exe -X utf8 pc17_tau.py   (GPU ~50분)
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp015_scada_pc import build_scada_hourly
from exp035_raw_dl import CACHE_DIR, FOLD_LIST
from exp039_dl_diverse import _huber
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev
from exp065_gbm3 import build_train_frame, slice_fold
from pc01_pcinv import build_curve, make_transform
import pc05_dlv as p5

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TAUS = (0.65, 0.55)
MEMBERS_BASE = ["cf_pcden", "cf_pchpo"]
PEAK = (0.775, 0.0, 0.15, 0.075)          # (pcden, pchpo, DL, dC)


def make_loss_tau(fwd_t, tau):
    import torch

    def f(pred, tgt, w, g=None):
        e = fwd_t(pred, g) - fwd_t(tgt, g)
        asym = torch.where(e < 0, 2.0 * tau, 2.0 * (1.0 - tau))
        return (_huber(e) * asym * w).sum() / w.sum()
    return f


def train_variant(tag, tau, train_df, cols, scada_h, cache):
    col = f"dl_{tag}"
    cpath = CACHE_DIR / f"pc17_{tag}.parquet"
    ppath = CACHE_DIR / f"pc17_{tag}_partial.parquet"
    if col in merge_cached_cols(cache, cpath, [col]):
        return col
    out = np.full(len(cache), np.nan)
    if ppath.exists():
        prev = pd.read_parquet(ppath)
        mm = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")
        out = mm[col].to_numpy()
    # dl_fold_predict_v는 make_loss_v(kind)를 내부 호출하므로 여기서는 직접 미러
    from exp033_dl import predict, GIDX
    from exp005_qopt import CAPACITY_KWH
    from exp049_clean import avail_factor
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    for fname, fold in FOLD_LIST:
        sel, va = p5.va_frame(cache, train_df, fname)
        if not np.isnan(out[sel]).any():
            continue
        tf = time.time()
        curves = p5.fold_curves(train_df, scada_h, fold)
        tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
        sub = tr_all.loc[tr_all["target"].notna()]
        caps = sub["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
        cf = (sub["target"].to_numpy() / caps).astype(np.float64)
        gcol = sub["kpx_group"].to_numpy().astype(int)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(sub[cols].to_numpy(dtype=np.float32))).astype(np.float32)
        gtr = sub["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        ytr = p5.cf_to_t(cf, gcol, curves)
        wtr = (np.where(cf < 0.10, 0.05, 1.0) *
               avail_factor(sub["avail_strict"].to_numpy(), "cln03")).astype(np.float32)
        Xva = sca.transform(imp.transform(va[cols].to_numpy(dtype=np.float32))).astype(np.float32)
        gva = va["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
        gva_raw = va["kpx_group"].to_numpy().astype(int)
        fwd_t = p5.make_torch_curve(curves, p5.DEVICE)
        loss_fn = make_loss_tau(fwd_t, tau)
        preds = []
        for s in range(3):
            m = p5.train_one_v(Xtr, gtr, ytr, wtr, p5.DL_CFG, loss_fn, s)
            preds.append(p5.t_to_cf(predict(m, Xva, gva), gva_raw, curves))
            del m
            torch.cuda.empty_cache()
        out[sel] = np.mean(preds, axis=0)
        cache[col] = out
        cache[KEY + [col]].to_parquet(ppath, index=False)
        print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s (partial 저장)", flush=True)
    cache[col] = out
    save_cols(cache, cpath, [col])
    print(f"  캐시 저장 → {cpath.name}", flush=True)
    return col


def main():
    t0 = time.time()
    print("### PC-17: dlvc τ 스윕 (0.65 / 0.55, 3-seed) ###", flush=True)
    train_df, cols = p5.build_frames()
    scada_h = build_scada_hourly()
    cache, champ_bias = load_cache_full()
    for f, cc in [("exp082_gbm_pcinv.parquet", ["cf_pcinv"]),
                  ("exp082_gbm_pcden.parquet", ["cf_pcden"]),
                  ("exp085_gbm_pchpo.parquet", ["cf_pchpo"]),
                  ("pc05_dlvc.parquet", ["dl_dlvc"])]:
        merge_cached_cols(cache, CACHE_DIR / f, [c for c in cc if c not in cache.columns])

    dl_cols = {0.60: "dl_dlvc"}
    for tau in TAUS:
        tag = f"dlvc{int(tau*100)}"
        dl_cols[tau] = train_variant(tag, tau, train_df, cols, scada_h, cache)

    # ── 채점: PC-16 정점에서 DL 치환 (+ DL 지분 지터) ──
    gbm_frame, _ = build_train_frame()
    reg = {}
    for fname, fold in FOLD_LIST:
        tr_all, _ = slice_fold(gbm_frame, fold)
        for group in (1, 2, 3):
            tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()]
            curve = build_curve(scada_h[group], tr_g, fold["train_end"], None)
            if curve is None:
                continue
            fwd, inv, _, _ = make_transform(curve)
            sel = ((cache["fold"] == fname) & (cache["group"] == group)).to_numpy()
            allm = MEMBERS_BASE + list(dl_cols.values()) + ["dl_cln03"]
            reg[(fname, group)] = (fwd, sel, {c: inv(cache.loc[sel, c].to_numpy()) for c in allm})

    ev = make_ev(cache, champ_bias)
    v1 = ev("cf_pcinv", "dl_cln03", 0.65, PP_NEW)
    champ = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    v1_cv, v1_hd = v1["cv"][0], v1["hold"][0]
    cv_folds = list(v1["folds"].keys())
    print(f"\nv1 {v1_cv:.4f}/{v1_hd:.4f} | 챔피언 {champ['cv'][0]:.4f}/{champ['hold'][0]:.4f}", flush=True)

    res = {}
    for tau, dcol in sorted(dl_cols.items()):
        for wts in [PEAK, (0.75, 0.0, 0.175, 0.075), (0.75, 0.0, 0.15, 0.10),
                    (0.70, 0.0, 0.20, 0.10)]:
            label = f"t{int(tau*100)}_" + "_".join(str(int(x*1000)) for x in wts)
            vals = np.full(len(cache), np.nan)
            mem = ["cf_pcden", "cf_pchpo", dcol, "dl_cln03"]
            for (fname, group), (fwd, sel, v) in reg.items():
                vals[sel] = fwd(sum(w * v[c] for w, c in zip(wts, mem)))
            assert not np.isnan(vals).any()
            cache[label] = vals
            res[label] = ev(label, None, 1.0, PP_NEW)

    print(f"\n{'셀':<24}{'CV':>8}{'Δv1':>9}{'HOLD':>8}{'Δv1':>9}{'Δn':>9}{'Δf':>9}  fold(Δv1)", flush=True)
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - v1_cv,
                                                           kv[1]["hold"][0] - v1_hd)):
        fd = " ".join(f"{r['folds'][fn]-v1['folds'][fn]:+.4f}" for fn in cv_folds)
        print(f"{name:<24}{r['cv'][0]:>8.4f}{r['cv'][0]-v1_cv:>+9.4f}"
              f"{r['hold'][0]:>8.4f}{r['hold'][0]-v1_hd:>+9.4f}"
              f"{r['cv'][1]-v1['cv'][1]:>+9.4f}{r['cv'][2]-v1['cv'][2]:>+9.4f}  {fd}", flush=True)
    best = max(res.items(), key=lambda kv: min(kv[1]["cv"][0] - v1_cv, kv[1]["hold"][0] - v1_hd))
    print(f"\n[정점] {best[0]}: 챔피언 대비 {best[1]['cv'][0]-champ['cv'][0]:+.4f}/"
          f"{best[1]['hold'][0]-champ['hold'][0]:+.4f}", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
