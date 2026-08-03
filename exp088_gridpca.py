# ═══════════════════════════════════════════════════════
# exp-088: 격자 편차 PCA 임베딩 — 트리에 "선형결합"을 쥐여준다
#
# 배경 (기전 논거):
#   트리는 축 정렬 분할(x_i > thr)만 한다. 그런데 352피처의 **74%(261개)가 격자 편차**로,
#   같은 기상장을 여러 격자에서 본 강상관 변수들이다. 여기서 의미 있는 신호는 대개
#   **선형결합**(공간 경도·방향 성분)이라 트리는 분할을 수십 개 쌓아야 근사한다.
#   - exp-035: 같은 편차를 **DL**(첫 층이 선형결합)에 주니 +0.0051 (역대 최대)
#   - exp-040: 같은 편차를 **GBM**에 주니 단독 +0.0012 — 같은 정보에서 훨씬 덜 뽑아냄
#   - exp-079: `linear_tree`(잎 선형모델 = 트리에 선형 능력 부여)는 **붕괴**(MAE 0.25)
#     → "트리에게 선형결합을 준다"는 목표의 **다른 경로**가 PCA 임베딩.
#
# 반대 증거 (사전 기록): exp-072의 물리 파생 피처는 `+0.0000/+0.0001` 완전 중립.
#   단 그건 **단일 변수의 비선형 스칼라 변환**(ρ, veff)이라 트리가 원래 표현 가능한 형태였다.
#   PCA 성분은 **다변수 선형결합**이라 표현 난이도가 다르다 — 이 구분이 실재하는지가 실험 내용.
#
# 변형 4개 (GBM만 — DL·후처리·분류기·블렌드 w 전부 챔피언 고정):
#   nospd   :  91 = 352 − 격자편차 261            (PCA 없는 통제. exp-040 되돌리기)
#   pcaR16  :  91 + 16 PC                          (편차를 압축으로 **대체**)
#   pcaA16  : 352 + 16 PC                          (원시 + 압축 **병기** — 본 가설)
#   pcaA32  : 352 + 32 PC
#
# ⚠ 누수 방지: PCA는 **폴드마다 train 구간(≤ train_end) 행에서만 적합** 후 전체 변환.
#   (imputer·scaler도 동일 구간에서 적합)
#
# 게이트: 챔피언 GBM 하이퍼(e56)·cln03 가중·3-seed·4-fold+홀드아웃,
#   dl_cln03 블렌드 w{0.65, 0.70} + 단독. 판정 Δ = 보드 챔피언(hpo×dC_w65).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp088_gridpca.py
# ═══════════════════════════════════════════════════════
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp056_hpo import RESULT_JSON as E56_JSON, train_gate_fold
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

K_MAX = 32
# 변형 → (기저 컬럼셋, PC 개수)
VARIANTS = {"nospd":  ("base", 0),
            "pcaR16": ("base", 16),
            "pcaA16": ("full", 16),
            "pcaA32": ("full", 32)}


def split_cols(use_cols):
    spd = [c for c in use_cols if c.startswith("sdL_") or c.startswith("sdG_")]
    base = [c for c in use_cols if c not in set(spd)]
    return base, spd


def fold_pca(train_df, spd_cols, train_end, k, report=False):
    """train 구간에서만 적합 → 전체 변환. 반환: (n_rows, k) 성분."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    X = train_df[spd_cols].to_numpy(dtype=np.float64)
    fit_mask = (train_df["forecast_kst_dtm"] <= train_end).to_numpy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sca = StandardScaler()
    Xf = sca.fit_transform(imp.fit_transform(X[fit_mask]))
    pca = PCA(n_components=k, random_state=0).fit(Xf)
    if report:
        ev = pca.explained_variance_ratio_
        print(f"    PCA 설명분산: PC1 {ev[0]:.3f} | 누적 8 {ev[:8].sum():.3f} "
              f"| 16 {ev[:16].sum():.3f} | {k} {ev.sum():.3f}", flush=True)
    return pca.transform(sca.transform(imp.transform(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="nospd,pcaR16,pcaA16,pcaA32")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    tags = [t.strip() for t in args.variants.split(",") if t.strip()]
    for t in tags:
        assert t in VARIANTS, f"미정의 변형: {t}"

    t0 = time.time()
    print("### exp-088 격자 편차 PCA 임베딩 (GBM) ###", flush=True)
    train_df, use_cols = build_train_frame()
    base_cols, spd_cols = split_cols(use_cols)
    print(f"입력 준비 완료 — 전체 {len(use_cols)} = 기저 {len(base_cols)} + 격자편차 "
          f"{len(spd_cols)} ({len(spd_cols)/len(use_cols)*100:.0f}%) — {time.time()-t0:.0f}s",
          flush=True)
    pc_cols = [f"pc_{i:02d}" for i in range(K_MAX)]
    e56 = json.loads(E56_JSON.read_text())
    cache, champ_bias = load_cache_full()

    for tag in tags:
        base_kind, k = VARIANTS[tag]
        col = f"cf_{tag}"
        cpath = CACHE_DIR / f"exp088_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp088_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            continue
        cols_fixed = base_cols if base_kind == "base" else use_cols
        print(f"\n[{tag}] {len(cols_fixed)} + PC {k} = {len(cols_fixed)+k}피처", flush=True)
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
        for fi, (fname, fold) in enumerate(FOLD_LIST):
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            if k > 0:                        # 폴드마다 재적합 (누수 방지)
                Z = fold_pca(train_df, spd_cols, fold["train_end"], k, report=(fi == 0))
                for i in range(k):
                    train_df[pc_cols[i]] = Z[:, i]
            cols = cols_fixed + pc_cols[:k]
            p = train_gate_fold(train_df, cols, fold, args.seeds, args.jobs,
                                e56["lgb"], e56["lgb_n"], e56["xgb"], e56["xgb_n"])
            m = cache.loc[sel, KEY].merge(p.rename(columns={"cf_hpo": col}), on=KEY, how="left")
            assert m[col].notna().all(), f"정렬 실패 {fname}"
            out[sel] = m[col].to_numpy()
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])

    mm = cache["target"].notna().to_numpy()
    print("", flush=True)
    for tag in tags:
        v, h = cache[f"cf_{tag}"].to_numpy(), cache["cf_hpo"].to_numpy()
        print(f"  corr(cf_{tag:<7}, cf_hpo) = {np.corrcoef(v[mm], h[mm])[0,1]:.4f} | "
              f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)
    res = {"hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for tag in tags:
        res[f"{tag}_solo"] = ev(f"cf_{tag}", None, 1.0, PP_NEW)
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(f"cf_{tag}", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print("\n--- GBM 단독 델타 (변형 − hpo) ---")
    for tag in tags:
        d_cv = res[f"{tag}_solo"]["cv"][0] - res["hpo_solo"]["cv"][0]
        d_hd = res[f"{tag}_solo"]["hold"][0] - res["hpo_solo"]["hold"][0]
        print(f"  {tag:<8} ΔCV {d_cv:+.4f} / Δhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
