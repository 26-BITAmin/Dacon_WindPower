# exp-087 속도 프로브: fit_mode / 청크 크기 A/B (그룹1·Q1·ctx 8000 고정, 1 draw)
#   증상: GPU util 100%인데 전력 26W·메모리 대역폭 0%·CPU 0.01코어 = 레이턴시 바운드.
#   가설: 기본 fit_mode="fit_preprocessors"에서 수동 청크 predict를 하면 청크마다
#         컨텍스트를 재인코딩 → fit_with_cache면 KV 캐시 1회 + 값싼 test 어텐션.
# 실행: C:\miniconda3\python.exe -X utf8 exp087_speedprobe.py
import sys, time
import numpy as np
import truststore
truststore.inject_into_ssl()

from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST
from exp049_clean import avail_factor
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CTX, N_EST, G = 8000, 4, 1


def run(cfg_name, fit_mode, bs, n_prep, Xtr, y, prob, Xv):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Xtr), size=min(CTX, len(Xtr)), replace=True, p=prob)
    t0 = time.time()
    m = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cuda", random_state=0, n_estimators=N_EST,
        auto_scale_n_estimators=False, ignore_pretraining_limits=True,
        memory_saving_mode="auto", fit_mode=fit_mode, n_preprocessing_jobs=n_prep)
    m.fit(Xtr[idx], y[idx])
    t_fit = time.time() - t0
    t1 = time.time()
    out = [m.predict(Xv[i:i + bs]) for i in range(0, len(Xv), bs)]
    p = np.concatenate(out)
    t_pred = time.time() - t1
    del m
    torch.cuda.empty_cache()
    print(f"  {cfg_name:<34} fit {t_fit:6.1f}s | predict {t_pred:7.1f}s | "
          f"합 {t_fit+t_pred:7.1f}s | 청크 {len(range(0, len(Xv), bs))}회", flush=True)
    return p, t_fit + t_pred


def main():
    t0 = time.time()
    print("### exp-087 속도 프로브 (그룹1 · Q1 · ctx 8000 · 1 draw) ###", flush=True)
    train_df, use_cols = build_train_frame()
    fold = dict(FOLD_LIST)["fold_2024Q1"]
    tr_all, va_all = slice_fold(train_df, fold)
    cap = CAPACITY_KWH[f"kpx_group_{G}"]
    tr_g = tr_all.loc[(tr_all["kpx_group"] == G) & tr_all["target"].notna()].copy()
    va_g = va_all.loc[va_all["kpx_group"] == G].sort_values("forecast_kst_dtm").copy()
    X_tr, X_va = prepare_matrices(tr_g, va_g, use_cols)
    y = (tr_g["target"].to_numpy() / cap).astype(np.float32)
    w = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
        avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
    prob = w / w.sum()
    Xtr = X_tr.to_numpy(dtype=np.float32)
    Xv = X_va.to_numpy(dtype=np.float32)
    print(f"학습행 {len(Xtr):,} | 검증행 {len(Xv):,} | 피처 {Xtr.shape[1]} "
          f"— 준비 {time.time()-t0:.0f}s\n", flush=True)

    res = {}
    ref, res["A"] = run("A) 현행: preprocessors + bs512", "fit_preprocessors", 512, 1,
                        Xtr, y, prob, Xv)
    for name, fm, bs, npj in [
        ("B) fit_with_cache + bs512", "fit_with_cache", 512, 1),
        ("C) fit_with_cache + bs2048", "fit_with_cache", 2048, 1),
        ("D) fit_with_cache + bs2048 + prep4", "fit_with_cache", 2048, 4),
    ]:
        try:
            p, t = run(name, fm, bs, npj, Xtr, y, prob, Xv)
            res[name[0]] = t
            print(f"     └ 예측 일치도 vs A: max|Δ| {np.abs(p-ref).max():.2e} "
                  f"| corr {np.corrcoef(p, ref)[0,1]:.6f}", flush=True)
        except Exception as e:
            print(f"  {name:<34} 실패: {type(e).__name__} {str(e)[:90]}", flush=True)

    if "A" in res:
        print("\n--- 배속 (A 기준) ---")
        for k in sorted(res):
            if k != "A":
                print(f"  {k}: {res['A']/res[k]:.2f}배")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
