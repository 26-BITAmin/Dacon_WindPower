# ═══════════════════════════════════════════════════════
# exp-097: exp-096 실패 원인 판별 — DL 용량 부족인가, 노이즈 차원인가 (2x2 대조)
#
# 배경: exp-096에서 ASOS 편차 93개를 DL 입력에 넣자 DL 단독이 −0.0065 붕괴.
#   두 가설이 경합:
#     (A) 용량 부족 — 입력이 348→441로 25% 커졌는데 DL_CFG(384x3)는 348 시절 최적.
#         선례 exp-056: "87피처 시절 GBM 하이퍼가 352피처에서 실병목" 실증(+0.0024).
#     (B) 노이즈 차원 — 추가 93열이 전부 **일 단위 상수**(tday merge, 24시간 동일).
#         exp-078: 손실의 75.3%가 일중 모양 성분 → 일 상수는 그 성분에 정보 0.
#
# 판별 (챔피언 DL = 348 + 384x3, exp-096 = 441 + 384x3 이 이미 있음):
#   big  = 441 + 512x4   → (A)가 맞으면 exp-096보다 개선
#   thin = 354 + 384x3   → (B)가 맞으면 exp-096보다 개선 (ASOS를 6열로 축소)
#   ※ thin의 6열 = obs_mean + 상위 5지점(100/105/104/216/217) 편차
#
# 실행: C:\miniconda3\python.exe -X utf8 exp097_dl_shape.py [--seeds 3]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR, DL_CFG
from exp049_clean import PP_NEW, merge_avail, dl_fold_predict_clean
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp096_dl_obs import build_obs_dev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SHAPE_CACHE = CACHE_DIR / "exp097_dl_shape.parquet"
TOP5 = [100, 105, 104, 216, 217]      # exp082_inventory 실측 상위 5지점
BIG_CFG = dict(DL_CFG, width=512, depth=4)


def dl_fold_predict_cfg(tr_df, va_df, cols, seeds, cfg):
    """dl_fold_predict_clean과 동일하되 DL_CFG를 인자로 (용량 변형용)."""
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from exp033_dl import predict, GIDX
    from exp039_dl_diverse import make_loss, train_one
    from exp049_clean import make_xy_clean
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xtr, gtr, ytr, wtr = make_xy_clean(tr_df, cols, imp, sca, "cln03", fit=True)
    Xva = sca.transform(imp.transform(va_df[cols].to_numpy(dtype=np.float32))).astype(np.float32)
    gva = va_df["kpx_group"].map(GIDX).to_numpy().astype(np.int64)
    loss_fn = make_loss("q60")
    preds = []
    for s in range(seeds):
        m = train_one(Xtr, gtr, ytr, wtr, cfg, loss_fn, s)
        preds.append(predict(m, Xva, gva))
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-cov", type=float, default=0.98)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-097: DL 용량(A) vs 노이즈 차원(B) 판별 ###", flush=True)

    train_df, _, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)
    vcols = colsets["spd"]
    obs, ocols = build_obs_dev(train_df, args.min_cov)
    train_df["tday"] = (train_df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    train_df = train_df.merge(obs, on="tday", how="left")
    thin_cols = ["obs_mean"] + [f"obs_dev_{s}" for s in TOP5 if f"obs_dev_{s}" in ocols]
    print(f"입력: big = {len(vcols)+len(ocols)} (512x4) / thin = {len(vcols)+len(thin_cols)} (384x3)",
          flush=True)

    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp096_dl_obs.parquet", ["dl_obs"]), \
        "exp-096 캐시 필요"
    specs = {"dl_big": (vcols + ocols, BIG_CFG), "dl_thin": (vcols + thin_cols, DL_CFG)}
    have = merge_cached_cols(cache, SHAPE_CACHE, list(specs))
    for col, (cols_, cfg) in specs.items():
        if col in have:
            print(f"[{col}] 캐시 재사용", flush=True)
            continue
        out = np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), "정렬 실패"
            out[sel] = (dl_fold_predict_cfg(tr_all, va, cols_, args.seeds, cfg)
                        if cfg is not DL_CFG else
                        dl_fold_predict_clean(tr_all, va, cols_, args.seeds, "cln03"))
            print(f"  [{col}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        have.append(col)
        save_cols(cache, SHAPE_CACHE, have)
        print(f"  캐시 저장 -> {SHAPE_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    print("\n[GBM과의 상관 — 블렌드 이득의 관건 축]", flush=True)
    for c in ("dl_cln03", "dl_obs", "dl_big", "dl_thin"):
        r = np.corrcoef(cache.loc[mm, c], cache.loc[mm, "cf_hpo"])[0, 1]
        sh = (cache[c].to_numpy() - cache["dl_cln03"].to_numpy()).mean() * 100
        print(f"  corr({c}, cf_hpo) = {r:.4f} | 평균 이동 {sh:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f} (기대 0.6556/0.6535)",
          flush=True)
    res = {}
    for tag in ("dl_cln03", "dl_obs", "dl_big", "dl_thin"):
        res[f"{tag}_solo"] = ev(tag, None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev("cf_hpo", tag, w, PP_NEW)
    print_table(res, ref)
    base = res["dl_obs_solo"]["cv"][0]
    print("\n[판별] DL 단독 CV — exp-096(dl_obs) 대비")
    for tag in ("dl_big", "dl_thin"):
        d = res[f"{tag}_solo"]["cv"][0] - base
        print(f"  {tag}: {d:+.4f}  ({'개선' if d > 0.001 else '무개선'})")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
