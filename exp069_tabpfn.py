# ═══════════════════════════════════════════════════════
# exp-069: TabPFN v2 회귀 — "새 추정 계열" 제3 base 멤버 타당성 (LG Aimers 6강 접점)
#
# 배경: exp-065에서 히스토그램 GBM 구현 교체(HGB, corr 0.9985)로는 다양성 축이 안
#   열림을 실증 — 남은 새 계열은 ICL 트랜스포머. TabPFN 2.5(패키지 8.x)는 브라우저
#   로그인+라이선스 동의 게이트라 자동 실행 불가 → 게이트 없는 v2(tabpfn 2.x)로 측정.
#   ⚠ 규정: v2 가중치 = Prior Labs License(Apache 2.0 기반·공개 다운로드). 대회의
#   "공개·합법" 기준 부합으로 보이나 사전학습 가중치 허용 여부는 제출 전 확인 필요.
#
# 한계 대응: v2 공식 한계 1만 행 < 그룹당 학습 1.5~2.4만 행 → 컨텍스트 서브샘플
#   앙상블 (draws회, 학습 가중 비례 확률 샘플링 — TabPFN은 sample_weight 미지원이라
#   가중을 샘플링 확률로 근사. 저CF 0.05·정지시간 cln03 다운웨이트 반영).
#
# 단계:
#   --stage q4  : Q4 폴드 진단 — 그룹별 solo eval-hour CF MAE·corr(cf_hpo)·corr(dl_cln03).
#                 go/no-go: corr(cf_hpo) < 0.99 이고 MAE가 파트너급(≤ dl ~0.127)일 것.
#   --stage gate: (go일 때만) 5폴드 전체 캐시(cf_pfn) → 3계열 혼합 게이트.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp069_tabpfn.py --stage q4
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd
import truststore
truststore.inject_into_ssl()

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, avail_factor
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PFN_CACHE = CACHE_DIR / "exp069_gbm_pfn.parquet"
CTX = 8000          # 컨텍스트 행수 (v2 공식 한계 1만 이하)
DRAWS = 2           # 서브샘플 앙상블 횟수 (4→2: 1차 실행에서 g1 3시간+ 미완 — 비용 절반)
N_EST = 4           # TabPFN 내부 앙상블 (기본 8→4: 비용 절반)


def pfn_fold_group(tr_g, va_g, cols, cap, draws, seed0=0):
    from tabpfn import TabPFNRegressor
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_tr = (tr_g["target"].to_numpy() / cap).astype(np.float32)
    w = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    prob = w / w.sum()
    Xv = X_va.to_numpy(dtype=np.float32)
    preds = []
    for d in range(draws):
        td = time.time()
        rng = np.random.RandomState(seed0 + d)
        n = min(CTX, len(X_tr))
        idx = rng.choice(len(X_tr), size=n, replace=False, p=prob)
        m = TabPFNRegressor(device="cuda", random_state=seed0 + d, n_estimators=N_EST)
        m.fit(X_tr.to_numpy(dtype=np.float32)[idx], y_tr[idx])
        preds.append(m.predict(Xv))
        del m
        print(f"      draw{d}: {time.time()-td:.0f}s", flush=True)
    return np.mean(preds, axis=0)


def run_fold(train_df, cols, fold, draws):
    """그룹 단위 중간 저장(_partial) — 중단 시 완료 그룹 재사용."""
    tr_all, va_all = slice_fold(train_df, fold)
    ppath = CACHE_DIR / f"exp069_pfn_{fold['name']}_partial.parquet"
    rows = [pd.read_parquet(ppath)] if ppath.exists() else []
    done = set(rows[0]["group"].unique()) if rows else set()
    if done:
        print(f"    (partial 재사용: g{sorted(done)})", flush=True)
    for group in [1, 2, 3]:
        if group in done:
            continue
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        tr_g = tr_all.loc[(tr_all["kpx_group"] == group) & tr_all["target"].notna()].copy()
        va_g = va_all.loc[va_all["kpx_group"] == group].sort_values("forecast_kst_dtm").copy()
        if len(tr_g) == 0 or len(va_g) == 0:
            continue
        tg = time.time()
        p = pfn_fold_group(tr_g, va_g, cols, cap, draws)
        rows.append(pd.DataFrame({"fold": fold["name"], "group": group,
                                  "forecast_kst_dtm": va_g["forecast_kst_dtm"].to_numpy(),
                                  "cf_pfn": p}))
        pd.concat(rows, ignore_index=True).to_parquet(ppath, index=False)
        print(f"    g{group}: ctx {min(CTX, len(tr_g))}/{len(tr_g)}행 × {draws}draws "
              f"→ {time.time()-tg:.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def diag_q4(cache, col="cf_pfn_q4"):
    sub = cache.loc[(cache["fold"] == "fold_2024Q4") & cache["target"].notna() &
                    cache[col].notna()].copy()
    caps = sub["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    acf = sub["target"].to_numpy() / caps
    ev = acf >= 0.10
    for ref in ("cf_hpo", "dl_cln03"):
        c = np.corrcoef(sub.loc[:, col], sub.loc[:, ref])[0, 1]
        print(f"  corr({col}, {ref}) = {c:.4f}", flush=True)
    err = sub[col].to_numpy() - acf
    err_h = sub["cf_hpo"].to_numpy() - acf
    print(f"  Q4 eval MAE: pfn {np.abs(err[ev]).mean():.4f} vs cf_hpo {np.abs(err_h[ev]).mean():.4f} "
          f"| pfn signed {err[ev].mean():+.4f}", flush=True)
    for g in (1, 2, 3):
        mg = (sub["group"] == g).to_numpy() & ev
        print(f"    g{g}: pfn MAE {np.abs(err[mg]).mean():.4f} vs hpo {np.abs(err_h[mg]).mean():.4f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["q4", "gate"], default="q4")
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-069 TabPFN v2 회귀 (제3 계열 타당성) ###", flush=True)

    train_df, use_cols = build_train_frame()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)
    cache, champ_bias = load_cache_full()

    if args.stage == "q4":
        fold = dict(FOLD_LIST)["fold_2024Q4"]
        print(f"[Q4 진단] draws={args.draws}, ctx={CTX}", flush=True)
        p = run_fold(train_df, use_cols, fold, args.draws)
        m = cache[KEY].merge(p.rename(columns={"cf_pfn": "cf_pfn_q4"}), on=KEY, how="left")
        cache["cf_pfn_q4"] = m["cf_pfn_q4"].to_numpy()
        diag_q4(cache)
        print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
        return

    # gate: 5폴드 전체
    have = merge_cached_cols(cache, PFN_CACHE, ["cf_pfn"])
    if "cf_pfn" not in have:
        preds = []
        for fname, fold in FOLD_LIST:
            tf = time.time()
            preds.append(run_fold(train_df, use_cols, fold, args.draws))
            print(f"  [pfn:{fname}] {time.time()-tf:.0f}s", flush=True)
        p = pd.concat(preds, ignore_index=True)
        m = cache[KEY].merge(p, on=KEY, how="left")
        assert m["cf_pfn"].notna().all(), "정렬 실패"
        cache["cf_pfn"] = m["cf_pfn"].to_numpy()
        save_cols(cache, PFN_CACHE, ["cf_pfn"])
        print(f"  캐시 저장 → {PFN_CACHE.name}", flush=True)
    mm = cache["target"].notna()
    for ref in ("cf_hpo", "dl_cln03"):
        print(f"  corr(cf_pfn, {ref}) = "
              f"{np.corrcoef(cache.loc[mm, 'cf_pfn'], cache.loc[mm, ref])[0, 1]:.4f}", flush=True)
    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo×dC_w65) 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)
    hpov, pfnv = cache["cf_hpo"].to_numpy(), cache["cf_pfn"].to_numpy()
    res = {"pfn_solo": ev("cf_pfn", None, 1.0, PP_NEW),
           "hpo_solo": ev("cf_hpo", None, 1.0, PP_NEW)}
    for m_, tag in ((0.9, "t90"), (0.8, "t80"), (2 / 3, "t67")):
        cache[f"tri{tag}"] = m_ * hpov + (1 - m_) * pfnv
        for w in (0.65, 0.70):
            res[f"{tag}_w{int(w*100)}"] = ev(f"tri{tag}", "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
