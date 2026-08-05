# AnEn 트랙 사전 검사: 예측자 후보 컬럼 확인
import sys
import time

import numpy as np

from exp065_gbm3 import build_train_frame
from exp035_raw_dl import FOLD_LIST

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
train_df, use_cols = build_train_frame()
print(f"frame {train_df.shape} | use_cols {len(use_cols)} | {time.time()-t0:.0f}s")

import re
pats = ["ws", "wd", "t2m", "temp", "sp", "press", "shear", "jma", "msm", "gsm",
        "ec_", "enfo", "spread", "pbl", "dis"]
lvl = [c for c in use_cols if not re.search(r"(_g\d+$|_d\d+$|dev|grid)", c)]
print(f"\n레벨성 컬럼 후보 {len(lvl)}개:")
for c in lvl:
    print(" ", c)

print("\nFOLD_LIST:")
for name, f in FOLD_LIST:
    print(" ", name, {k: str(v) for k, v in f.items() if k != "name"})

g = train_df[train_df["kpx_group"] == 1]
print("\n그룹1 행수:", len(g), "| 기간:", g["forecast_kst_dtm"].min(), "~",
      g["forecast_kst_dtm"].max())
print("target notna:", g["target"].notna().sum())
