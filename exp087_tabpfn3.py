# ═══════════════════════════════════════════════════════
# exp-087: TabPFN 재시험 — exp-069의 핸디캡 2개 중 "컨텍스트 절단"만 분리 해제
#
# 배경 (exp-069, tabpfn 2.2.1 = V2 모델):
#   corr(cf_pfn, cf_hpo) 0.9839 = 시도한 모든 base측 후보 중 최저(진짜 다양성),
#   Q4 단독 MAE 0.1113 vs GBM 0.1098로 대등. 그런데 **홀드아웃 −0.0083** 급락하고
#   지분에 단조 악화 → 기각. 원인은 MODEL.md에 "미논의"로 남아 있고, 구조적 핸디캡
#   2개만 사실로 확인돼 있었다:
#     (H1) v2 하드캡 1만 행 < 그룹당 학습 1.5~2.4만 행 → **8,000행 서브샘플**(데이터 1/3)
#     (H2) sample_weight 미지원 → 저CF 0.05·cln03 가중을 **샘플링 확률로 근사**
#
# 패키지 8.2.0에서 확인된 사실 (2026-08-02):
#   - `MAX_NUMBER_OF_SAMPLES` 하드캡 상수가 **사라짐** → H1은 해제 가능
#   - `fit()`에 **sample_weight 여전히 없음** → H2는 그대로
#   - ModelVersion = V2 / V2_5 / V2_6 / V3 선택 가능
#
# ⚠ 라이선스: V2 가중치만 Prior Labs License(Apache 2.0+표기). **V2.5/V2.6/V3는
#   non-commercial** 이고 브라우저 라이선스 동의 + TABPFN_TOKEN 필요.
#   → 기본은 V2. V3는 사용자가 동의·토큰을 넣은 경우에만 `--version v3`.
#
# 변경점 분리 (exp-083 역전의 교훈 — 한 번에 하나만):
#   V2 가중치·동일 가중근사·동일 폴드를 고정하고 **컨텍스트 크기만** 사다리로 변경.
#   ctx 8000 = exp-069 재현 대조 / ctx 20000 = H1 해제.
#   가중은 **복원추출**로 유지 (exp-069는 비복원 8,000 — 크기를 키우려면 복원이 필요).
#
# 스모크: Q2+Q4 2폴드 (Q4 단독은 3연속 과대평가 실적)
# 실행: C:\miniconda3\python.exe -X utf8 exp087_tabpfn3.py --stage smoke
# ═══════════════════════════════════════════════════════
import argparse
import os
import sys
import time

import numpy as np
import truststore
truststore.inject_into_ssl()

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST, CACHE_DIR
from exp049_clean import avail_factor
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SMOKE_FOLDS = ["fold_2024Q2", "fold_2024Q4"]
N_EST = 4          # exp-069와 동일 (기본 8 → 비용 절반)


def make_reg(version, seed, n_est):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    # fit_with_cache: 컨텍스트 KV를 1회만 만들고 test 행은 조건부 독립으로 값싸게 처리.
    #   기본값(fit_preprocessors)에서 수동 청크 predict를 하면 청크마다 컨텍스트를
    #   재인코딩 → 실측 그룹당 13분+ (GPU 26W·메모리 대역폭 0% = 레이턴시 바운드).
    kw = dict(device="cuda", random_state=seed, n_estimators=n_est,
              auto_scale_n_estimators=False, ignore_pretraining_limits=True,
              memory_saving_mode="auto", fit_mode="fit_with_cache",
              inference_precision="autocast")   # bf16 텐서코어 경로 강제
    #   "auto"에서는 전력 27W·메모리 대역폭 0%로 fp32 비텐서코어 경로 정황
    # ⚠ n_preprocessing_jobs>1 금지: joblib 워커가 임시폴더 경로를 ascii로 인코딩하는데
    #   이 PC 경로에 한글(사용자명 강원우)이 있어 UnicodeEncodeError로 죽는다.
    #   CPU는 0.01코어로 놀고 있어 병렬화 이득도 없음.
    if version == "v2":
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2, **kw)
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3, **kw)


def predict_batched(m, Xv, bs):
    """8GB VRAM 대응 — test 행을 청크로 나눠 예측. OOM이면 청크를 반으로 줄여 재시도.
    (신 패키지는 test 전체를 한 번에 올려 2,184행에서 12.5GiB를 요구 → 필수)"""
    import torch
    from tabpfn.errors import TabPFNCUDAOutOfMemoryError
    while True:
        try:
            out = [m.predict(Xv[i:i + bs]) for i in range(0, len(Xv), bs)]
            return np.concatenate(out), bs
        except (TabPFNCUDAOutOfMemoryError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            if bs <= 32:
                raise
            bs //= 2
            print(f"      (OOM → test 청크 {bs}로 축소 재시도)", flush=True)


def pfn_group(tr_g, va_g, cols, cap, ctx, version, draws, bs, seed0=0):
    """exp-069와 동일한 가중근사(샘플링 확률). 크기만 ctx로 조절 — 복원추출."""
    import torch
    X_tr, X_va = prepare_matrices(tr_g, va_g, cols)
    y_tr = (tr_g["target"].to_numpy() / cap).astype(np.float32)
    w = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    prob = w / w.sum()
    Xtr = X_tr.to_numpy(dtype=np.float32)
    Xv = X_va.to_numpy(dtype=np.float32)
    preds = []
    for d in range(draws):
        rng = np.random.RandomState(seed0 + d)
        n = min(ctx, len(Xtr))
        idx = rng.choice(len(Xtr), size=n, replace=True, p=prob)
        m = make_reg(version, seed0 + d, N_EST)
        m.fit(Xtr[idx], y_tr[idx])
        p, bs = predict_batched(m, Xv, bs)
        preds.append(p)
        del m
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0), len(Xtr), bs


def stage_gate(train_df, use_cols, cache, champ_bias, ctxs, version, draws, bs):
    """5폴드(CV4+홀드아웃) 전체 캐시 → 혼합 사다리. 폴드 단위 partial 저장으로 중단 내성."""
    import pandas as pd
    from exp049_clean import KEY
    from exp050_lambda_sweep import merge_cached_cols, save_cols
    from exp060_pergroup_hpo import make_ev, print_table
    from exp049_clean import PP_NEW

    tags = []
    for ctx in ctxs:
        tag = f"pfn{ctx // 1000}"
        col = f"cf_{tag}"
        tags.append((tag, col, ctx))
        cpath = CACHE_DIR / f"exp087_gbm_{tag}.parquet"
        ppath = CACHE_DIR / f"exp087_gbm_{tag}_partial.parquet"
        if col in merge_cached_cols(cache, cpath, [col]):
            print(f"[{tag}] 캐시 로드 완료", flush=True)
            continue
        out = np.full(len(cache), np.nan)
        if ppath.exists():
            prev = pd.read_parquet(ppath)
            out = cache[KEY].merge(prev[KEY + [col]], on=KEY, how="left")[col].to_numpy()
            done = [fn for fn, _ in FOLD_LIST
                    if not np.isnan(out[(cache["fold"] == fn).to_numpy()]).any()]
            print(f"[{tag}] partial 재개 — 완료 폴드 {done}", flush=True)
        for fname, fold in FOLD_LIST:
            sel = (cache["fold"] == fname).to_numpy()
            if not np.isnan(out[sel]).any():
                continue
            tf = time.time()
            tr_all, _ = slice_fold(train_df, fold)
            for g in [1, 2, 3]:
                cap = CAPACITY_KWH[f"kpx_group_{g}"]
                tr_g = tr_all.loc[(tr_all["kpx_group"] == g) & tr_all["target"].notna()].copy()
                # 검증행은 **캐시 키 순서 그대로** 구성 (PC-05 va_frame 방식) — 정렬 가정 제거
                idx = np.where(sel & (cache["group"].to_numpy() == g))[0]
                if len(tr_g) == 0 or len(idx) == 0:
                    continue
                keys = cache.iloc[idx][["group", "forecast_kst_dtm"]]
                va_g = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                                  right_on=["kpx_group", "forecast_kst_dtm"], how="left")
                assert len(va_g) == len(idx), f"검증행 정렬 실패 {fname} g{g}"
                tg = time.time()
                p, _, bs = pfn_group(tr_g, va_g, use_cols, cap, ctx, version, draws, bs)
                out[idx] = p
                print(f"    [{tag}:{fname}:g{g}] {time.time()-tg:.0f}s "
                      f"(학습 {len(tr_g):,} → 검증 {len(idx):,})", flush=True)
            cache[col] = out
            cache[KEY + [col]].to_parquet(ppath, index=False)
            print(f"  [{tag}:{fname}] {time.time()-tf:.0f}s (partial 저장)", flush=True)
        cache[col] = out
        save_cols(cache, cpath, [col])
        print(f"  캐시 저장 → exp087_gbm_{tag}.parquet", flush=True)

    mm = cache["target"].notna().to_numpy()
    print("", flush=True)
    for tag, col, ctx in tags:
        v, h, d = (cache[col].to_numpy(), cache["cf_hpo"].to_numpy(),
                   cache["dl_cln03"].to_numpy())
        print(f"  corr({col}, cf_hpo) {np.corrcoef(v[mm], h[mm])[0,1]:.4f} | "
              f"corr({col}, dl_cln03) {np.corrcoef(v[mm], d[mm])[0,1]:.4f} | "
              f"평균 이동 {(v-h).mean()*100:+.3f}%cap", flush=True)
    if len(tags) == 2:
        a, b = cache[tags[0][1]].to_numpy(), cache[tags[1][1]].to_numpy()
        print(f"  corr({tags[0][1]}, {tags[1][1]}) = {np.corrcoef(a[mm], b[mm])[0,1]:.4f}",
              flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언 재현: CV {ref['cv'][0]:.4f} | HOLD {ref['hold'][0]:.4f}"
          f"  (기대 0.6556/0.6535)", flush=True)
    res = {}
    for tag, col, ctx in tags:
        res[f"{tag}_solo"] = ev(col, None, 1.0, PP_NEW)
        for m_, mt in ((0.90, "t90"), (0.80, "t80"), (2 / 3, "t67")):
            mix = f"mx_{tag}_{mt}"
            cache[mix] = m_ * cache["cf_hpo"] + (1 - m_) * cache[col]
            for w in (0.65, 0.70):
                res[f"{tag}{mt}_w{int(w*100)}"] = ev(mix, "dl_cln03", w, PP_NEW)
    print_table(res, ref)
    print("\nexp-069 대조(ctx 8000·구 패키지·draws2): 최고 t80_w70 ΔCV +0.0007 / Δhold −0.0006")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["smoke", "gate"], default="smoke")
    ap.add_argument("--version", choices=["v2", "v3"], default="v2")
    ap.add_argument("--ctx", default="8000,20000")
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512, help="test 예측 청크 (OOM 시 자동 축소)")
    ap.add_argument("--folds", default="fold_2024Q4",
                    help="exp-069 기준값이 Q4라 기본은 Q4 단독 진단 (쉼표로 추가)")
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctx.split(",")]
    bs = args.batch
    folds_to_run = [f.strip() for f in args.folds.split(",") if f.strip()]

    if args.version == "v3" and not os.environ.get("TABPFN_TOKEN"):
        sys.exit("V3는 TABPFN_TOKEN 환경변수가 필요합니다 "
                 "(ux.priorlabs.ai에서 라이선스 동의 후 발급).")

    t0 = time.time()
    print(f"### exp-087 TabPFN {args.version.upper()} 컨텍스트 사다리 "
          f"(ctx {ctxs}, draws {args.draws}, n_est {N_EST}) ###", flush=True)
    train_df, use_cols = build_train_frame()
    cache, champ_bias = load_cache_full()
    print(f"입력 준비 완료 ({len(use_cols)}피처) — {time.time()-t0:.0f}s", flush=True)

    if args.stage == "gate":
        stage_gate(train_df, use_cols, cache, champ_bias, ctxs,
                   args.version, args.draws, bs)
        print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
        return

    for fname in folds_to_run:
        fold = dict(FOLD_LIST)[fname]
        tr_all, va_all = slice_fold(train_df, fold)
        sel = (cache["fold"] == fname).to_numpy()
        sub = cache.loc[sel]
        print(f"\n── {fname} ──", flush=True)
        for ctx in ctxs:
            rows, ns = [], []
            tc = time.time()
            for g in [1, 2, 3]:
                cap = CAPACITY_KWH[f"kpx_group_{g}"]
                tr_g = tr_all.loc[(tr_all["kpx_group"] == g) & tr_all["target"].notna()].copy()
                va_g = va_all.loc[va_all["kpx_group"] == g].sort_values("forecast_kst_dtm").copy()
                if len(tr_g) == 0 or len(va_g) == 0:
                    continue
                p, n, bs = pfn_group(tr_g, va_g, use_cols, cap, ctx,
                                     args.version, args.draws, bs)
                ns.append(n)
                sg = sub.loc[sub["group"] == g].sort_values("forecast_kst_dtm")
                assert len(sg) == len(p)
                rows.append((sg, p, cap))
            # 평가시간(실측 ≥10%cap) 기준 집계
            P, H, D, Y = [], [], [], []
            for sg, p, cap in rows:
                y = sg["target"].to_numpy()
                m = ~np.isnan(y) & (y >= cap * 0.10)
                P.append(p[m]); Y.append(y[m] / cap)
                H.append(sg["cf_hpo"].to_numpy()[m]); D.append(sg["dl_cln03"].to_numpy()[m])
            P, H, D, Y = map(np.concatenate, (P, H, D, Y))
            print(f"  [ctx {ctx:>5} / 학습행 {max(ns):,}] "
                  f"MAE {np.abs(P-Y).mean():.5f} "
                  f"(GBM {np.abs(H-Y).mean():.5f} / DL {np.abs(D-Y).mean():.5f}) | "
                  f"signed {(P-Y).mean():+.4f} | "
                  f"corr(cf_hpo) {np.corrcoef(P, H)[0,1]:.4f} | "
                  f"corr(dl) {np.corrcoef(P, D)[0,1]:.4f} | {time.time()-tc:.0f}s", flush=True)
    print(f"\n판정 기준: corr(cf_hpo) < 0.99 유지 ∧ ctx 확대로 MAE 개선 "
          f"(= H1이 exp-069 실패의 요인이었다는 증거)")
    print("exp-069 Q4 기준값(대조): pfn MAE 0.1113 / cf_hpo 0.1098 / corr(cf_hpo) 0.9898")
    print("주의: Q4는 exp-056·063 하이퍼 탐색 폴드라 낙관 편향 실적 있음 — "
          "여기서 GO여도 결론 아님(exp-069도 Q4 GO 후 5폴드에서 기각).")
    print(f"총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
