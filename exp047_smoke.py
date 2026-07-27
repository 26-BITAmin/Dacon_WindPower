# exp-047 스모크 테스트: 블록 구성·학습·예측 경로 검증 (1 fold, 1 seed, 2 epoch)
import sys
import numpy as np

import exp047_seq_dl as E

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

train_df, cols, colsets = E.build_inputs(["spd"])
vcols = colsets["spd"]
cache, _ = E.load_gbm_cache()

fname, fold = E.FOLD_LIST[0]
tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
sel = (cache["fold"] == fname).to_numpy()
keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                right_on=["kpx_group", "forecast_kst_dtm"], how="left")
print(f"fold {fname}: tr {len(tr_all)} rows / va {len(va)} rows")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
imp = SimpleImputer(strategy="median", keep_empty_features=True)
sca = StandardScaler()
Xtr, gtr, Ytr, Wtr, bi_tr, hr_tr = E.build_blocks(tr_all, vcols, imp, sca, fit=True)
print(f"train 블록: X {Xtr.shape}, g {gtr.shape}, Y {Ytr.shape}, W>0 슬롯 {int((Wtr>0).sum())}"
      f" (target 행 {int(tr_all['target'].notna().sum())})")
assert (Wtr > 0).sum() == tr_all["target"].notna().sum(), "손실 마스크 수 불일치"
assert np.isfinite(Xtr).all() and np.isfinite(Ytr).all(), "NaN 존재"

Xva, gva, _, _, bi_va, hr_va = E.build_blocks(va, vcols, imp, sca, fit=False)
print(f"va 블록: X {Xva.shape} → 행 복원 {len(bi_va)}")
assert len(bi_va) == len(va)

cfg = E.SEQ_CFG | {"epochs": 2}
m = E.train_seq(Xtr, gtr, Ytr, Wtr, cfg, 0)
P = E.predict_seq(m, Xva, gva)
pred_rows = P[bi_va, hr_va]
print(f"예측: {pred_rows.shape}, 범위 [{pred_rows.min():.3f}, {pred_rows.max():.3f}], NaN {np.isnan(pred_rows).sum()}")
assert np.isfinite(pred_rows).all()

# 2 epoch 기준 조잡한 MAE (버그면 0.3+ / 정상 수렴 시작이면 0.1대)
caps = va["kpx_group"].map(lambda g: E.CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
acf = va["target"].to_numpy() / caps
ok = np.isfinite(acf) & (acf >= 0.10)
print(f"va MAE(CF, 2ep, 평가시간): {np.abs(pred_rows[ok] - acf[ok]).mean():.4f}")
print("스모크 통과")
