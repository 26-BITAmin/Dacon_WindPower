# ═══════════════════════════════════════════════════════
# exp-051: 정제 라벨 아래 q60 비대칭 계수(τ) 재스윕
#
# 근거 사슬:
#   - exp-045: τ=0.60이 능선 정점 — 단 **dirty 라벨 조건부** 결론.
#   - exp-049: 정제 후 q60 DL의 만가용 signed +0.0096 (상향 과잉) — "q60 유효성의 원천이
#     FiCR 정렬이 아니라 가용성 아티팩트 보상이었을 정황".
#   - exp-050: DL λ 사다리 signed −0.0019(dirty)→+0.0007(λ.5)→+0.0096(λ.05) — 정제가
#     비대칭의 보상 대상을 제거함을 사다리로 실증. 부호 전환 관측.
#   → 정제 라벨(λ0.05) 아래서는 최적 τ가 0.60보다 낮을 가설. τ ∈ {0.50, 0.55} 브래킷.
#     τ0.50 = 대칭 Huber (비대칭 완전 제거 = cleaned hub).
#
# 변경점 격리: DL 손실의 τ만 변경. GBM(cf_cln03)·가중 정제(λ0.05)·입력(spd 348)·
#   용량(384×3 ep120 3-seed)·신PP·챔피언 편향맵 전부 champ49와 동일.
# 평가: champ49(cln×cln_w70) 대비 3중 조건. w {0.80~0.55} 확장 스윕 동시 수행
#   (exp-049는 w70/65만 검증 — 정제 쌍 아래 w 능선 미탐색).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp051_tau_clean.py --jobs 8 --seeds 3
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp035_raw_dl import build_inputs, FOLD_LIST, DL_CFG, CACHE_DIR, score_from_tidy
from exp039_dl_diverse import build_slices, train_one
from exp045_tau_sweep import make_loss_tau
from exp049_clean import (AVAIL_PARQUET, GBM_CLEAN_CACHE, DL_CLEAN_CACHE, KEY,
                          PP_CHAMP, PP_NEW, merge_avail, postprocess_flex, load_eval_frame)
from exp050_lambda_sweep import make_xy_lam, merge_cached_cols, save_cols

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TAUS = {"c55": 0.55, "c50": 0.50}   # 신규 학습 (정제 λ0.05 고정, τ만 변경)
DL_TAU_CACHE = CACHE_DIR / "exp051_dl_tauclean.parquet"
BLEND_WS = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
LAM = 0.05


def dl_fold_predict_tau_clean(tr_df, va_df, cols, seeds, tau):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_lam(tr_df, cols, imp, sca, LAM, fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss_tau(tau)
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, DL_CFG, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def evaluate(cache, champ_bias):
    fold_names = [fn for fn, _ in FOLD_LIST]
    hold = HOLD_FOLD["name"]
    cv_folds = [fn for fn in fold_names if fn != hold]
    slices = build_slices(cache)

    def ev(gbm_col, dl_col, w, pp):
        blend = cache[gbm_col].to_numpy() if (dl_col is None or w >= 1.0) else \
            w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy()
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

    r_old = ev("cf_g132", "dl_hub", 0.70, PP_CHAMP)
    r_ref = ev("cf_gspd", "dl_q60", 0.70, PP_NEW)
    r_ch = ev("cf_cln03", "dl_cln03", 0.70, PP_NEW)
    print(f"\n구챔피언(g132·hub·w70·구PP): CV {r_old['cv'][0]:.4f} | HOLD {r_old['hold'][0]:.4f}  (기대 0.6448/0.6434)")
    print(f"ref(gspd·q60·w70·신PP):     CV {r_ref['cv'][0]:.4f} | HOLD {r_ref['hold'][0]:.4f}  (기대 0.6481/0.6463)")
    print(f"champ49(cln×cln_w70·신PP):  CV {r_ch['cv'][0]:.4f} (1-NMAE {r_ch['cv'][1]:.4f} FiCR {r_ch['cv'][2]:.4f}) | "
          f"HOLD {r_ch['hold'][0]:.4f}  (기대 0.6534/0.6518)")

    dl_map = {"t60": "dl_cln03", "t55": "dl_c55", "t50": "dl_c50"}
    res = {}
    for dn, dc in dl_map.items():
        for w in BLEND_WS:
            res[f"{dn}_w{int(w*100)}"] = ev("cf_cln03", dc, w, PP_NEW)

    ch_cv, ch_hd = r_ch["cv"][0], r_ch["hold"][0]
    ch_n, ch_f = r_ch["cv"][1], r_ch["cv"][2]
    print("\n" + "=" * 112)
    print(f"{'변형':<12}{'CV':>8}{'ΔCVch':>9}{'HOLD':>8}{'Δhdch':>9}"
          f"{'1-NMAE':>9}{'Δn':>8}{'FiCR':>8}{'Δf':>8}  판정 (Δ=champ49 대비)")
    rows = []
    for name, r in sorted(res.items(), key=lambda kv: -min(kv[1]["cv"][0] - ch_cv, kv[1]["hold"][0] - ch_hd)):
        dcv, dhd = r["cv"][0] - ch_cv, r["hold"][0] - ch_hd
        dn_, df_ = r["cv"][1] - ch_n, r["cv"][2] - ch_f
        fold_d = [r["folds"][fn] - r_ch["folds"][fn] for fn in cv_folds]
        mag = dcv > 0.001 and dhd > 0.001
        cons = all(x > -0.0002 for x in fold_d)
        both = dn_ > 0 and df_ > 0
        verdict = ("★3중충족" if (mag and cons and both) else
                   ("통과(매그니튜드)" if mag else ("양게이트" if (dcv > 0 and dhd > 0) else "")))
        print(f"{name:<12}{r['cv'][0]:>8.4f}{dcv:>+9.4f}{r['hold'][0]:>8.4f}{dhd:>+9.4f}"
              f"{r['cv'][1]:>9.4f}{dn_:>+8.4f}{r['cv'][2]:>8.4f}{df_:>+8.4f}  {verdict}")
        rows.append(dict(name=name, folds=fold_d, cons=cons, both=both, dcv=dcv, dhd=dhd))
    print(f"\n--- fold별 ΔCV (champ49 대비) ---")
    print(f"{'변형':<12}" + "".join(f"{fn.replace('fold_',''):>10}" for fn in cv_folds) + "  fold일관 두지표")
    for r in sorted(rows, key=lambda r: -min(r["dcv"], r["dhd"])):
        print(f"{r['name']:<12}" + "".join(f"{x:>+10.4f}" for x in r["folds"])
              + f"{'  O' if r['cons'] else '  X':>8}{'  O' if r['both'] else '  X':>7}")

    # 기전: τ 사다리 (정제 DL) signed error
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    tmp = cache.copy()
    tmp["forecast_kst_dtm"] = pd.to_datetime(tmp["forecast_kst_dtm"])
    tmp = tmp.merge(av[["forecast_kst_dtm", "group", "avail_strict"]], on=["forecast_kst_dtm", "group"], how="left")
    caps = tmp["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = tmp["target"].to_numpy() / caps
    evm = (tmp["target"].to_numpy() >= caps * 0.10) & tmp["avail_strict"].notna().to_numpy()
    full = evm & (tmp["avail_strict"].to_numpy() >= 0.999)
    print("\n--- 기전: 정제 DL τ 사다리 (보정 전 원시 CF) + corr vs cln GBM ---")
    print(f"{'모델':<10}{'만가용 signed':>14}{'전체 MAE':>10}{'corr_gbm':>10}")
    e_g = tmp["cf_cln03"].to_numpy() - acf
    for c in ["dl_q60", "dl_cln03", "dl_c55", "dl_c50"]:
        if c not in tmp.columns:
            continue
        e = tmp[c].to_numpy() - acf
        cr = np.corrcoef(e_g[evm], e[evm])[0, 1]
        print(f"{c:<10}{e[full].mean():>+14.4f}{np.abs(e[evm]).mean():>10.4f}{cr:>10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    print(f"### exp-051 정제 라벨 τ 재스윕 — τ={list(TAUS.values())} (λ={LAM} 고정) ###", flush=True)

    train_df, cols, colsets = build_inputs(["spd"])
    vcols = colsets["spd"]
    train_df = merge_avail(train_df)

    cache, champ_bias = load_eval_frame()
    got_g = merge_cached_cols(cache, GBM_CLEAN_CACHE, ["cf_cln03"])
    got_d = merge_cached_cols(cache, DL_CLEAN_CACHE, ["dl_cln03"])
    assert got_g and got_d, "exp-049 캐시(cln03) 필요"

    dl_cols = [f"dl_{t}" for t in TAUS]
    have = merge_cached_cols(cache, DL_TAU_CACHE, dl_cols)
    for tag, tau in TAUS.items():
        col = f"dl_{tag}"
        if col in have:
            print(f"[dl:{tag}] 캐시 재사용", flush=True)
            continue
        out = np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys)
            out[sel] = dl_fold_predict_tau_clean(tr_all, va, vcols, args.seeds, tau)
            print(f"  [dl:{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        have.append(col)
        save_cols(cache, DL_TAU_CACHE, have)
        print(f"  캐시 저장 → {DL_TAU_CACHE.name} ({len(have)}/{len(dl_cols)})", flush=True)

    evaluate(cache, champ_bias)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
