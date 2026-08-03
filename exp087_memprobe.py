# 메모리 제약이 병목인지 2분 판별: ctx를 줄이고 memory_saving_mode를 끄면 빨라지는가
import sys, time
import numpy as np
import truststore; truststore.inject_into_ssl()
from exp005_qopt import CAPACITY_KWH, make_sample_weight, prepare_matrices
from exp035_raw_dl import FOLD_LIST
from exp049_clean import avail_factor
from exp065_gbm3 import build_train_frame, slice_fold
sys.stdout.reconfigure(encoding="utf-8")


def timed(name, ctx, msm, n_est, Xtr, y, prob, Xv):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    idx = np.random.RandomState(0).choice(len(Xtr), size=min(ctx, len(Xtr)), replace=True, p=prob)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        m = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cuda", random_state=0, n_estimators=n_est,
            auto_scale_n_estimators=False, ignore_pretraining_limits=True,
            memory_saving_mode=msm, fit_mode="fit_with_cache",
            inference_precision="autocast")
        m.fit(Xtr[idx], y[idx]); tf = time.time() - t0
        t1 = time.time(); m.predict(Xv[:512]); tp = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {name:<38} fit {tf:6.1f}s | predict512 {tp:6.1f}s | peak {peak:.2f}GB", flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {name:<38} 실패: {type(e).__name__} {str(e)[:70]}", flush=True)
        torch.cuda.empty_cache()


t0 = time.time()
train_df, use_cols = build_train_frame()
fold = dict(FOLD_LIST)["fold_2024Q1"]
tr_all, va_all = slice_fold(train_df, fold)
cap = CAPACITY_KWH["kpx_group_1"]
tr_g = tr_all.loc[(tr_all["kpx_group"] == 1) & tr_all["target"].notna()].copy()
va_g = va_all.loc[va_all["kpx_group"] == 1].sort_values("forecast_kst_dtm").copy()
X_tr, X_va = prepare_matrices(tr_g, va_g, use_cols)
y = (tr_g["target"].to_numpy() / cap).astype(np.float32)
w = make_sample_weight(tr_g["target"].to_numpy(), cap, low_weight=0.05) * \
    avail_factor(tr_g["avail_strict"].to_numpy(), "cln03")
prob = w / w.sum()
Xtr = X_tr.to_numpy(dtype=np.float32); Xv = X_va.to_numpy(dtype=np.float32)
print(f"준비 {time.time()-t0:.0f}s | 학습 {len(Xtr):,} | 검증 {len(Xv):,}\n", flush=True)

timed("ctx1000 n_est1 msm=False", 1000, False, 1, Xtr, y, prob, Xv)
timed("ctx2000 n_est1 msm=False", 2000, False, 1, Xtr, y, prob, Xv)
timed("ctx4000 n_est1 msm=False", 4000, False, 1, Xtr, y, prob, Xv)
timed("ctx4000 n_est1 msm=auto ", 4000, "auto", 1, Xtr, y, prob, Xv)
timed("ctx8000 n_est1 msm=auto ", 8000, "auto", 1, Xtr, y, prob, Xv)
print(f"\n총 {(time.time()-t0)/60:.1f}분")
