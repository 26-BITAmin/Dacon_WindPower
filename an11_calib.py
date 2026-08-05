# ═══════════════════════════════════════════════════════
# AN-11: AnEn 확률론 보정 — K 스윕 + 스프레드 팽창
#
# AN-10: CRPSS(기후학) +0.67·(챔피언GBM) +0.24로 논문 재현 성공.
#        단 outlier 0.209 (이상 0.095) = 과소산포 (spread/RMSE 0.90).
# 레버 2개:
#   1) K ∈ {10, 20, 30, 50, 100} — 멤버 수와 CRPS·보정의 트레이드
#   2) 멤버 팽창 x → mean + α(x−mean), α ∈ {1.0, 1.15, 1.3} (K=20)
#
# 실행: C:\miniconda3\python.exe -X utf8 an11_calib.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an02_clean import build_analogs2, topk
from an06_final import PSET_G
from an10_paper import PSET_PAPER, crps_ens
from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST
from exp065_gbm3 import build_train_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KS = [10, 20, 30, 50, 100]
ALPHAS = [1.0, 1.15, 1.3]


def stats(mem, y, lab):
    K = mem.shape[1]
    crps = crps_ens(mem[lab], y[lab]).mean()
    ranks = (mem[lab] < y[lab][:, None]).sum(1)
    outl = ((ranks == 0) | (ranks == K)).mean()
    err = np.abs(mem.mean(1) - y)
    sr = mem.std(1)[lab].mean() / max(np.sqrt((err[lab] ** 2).mean()), 1e-9)
    return crps, outl, sr, (2.0 / (K + 1))


def main():
    t0 = time.time()
    print("### AN-11 K 스윕 + 스프레드 팽창 ###", flush=True)
    train_df, _ = build_train_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    folds = dict(FOLD_LIST)

    recs = []
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        for g in [1, 2, 3]:
            tt = time.time()
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            for vtag, pset in (("paper_f", PSET_PAPER), ("tuned_f", PSET_G[g])):
                ana = build_analogs2(train_df, folds[fname], g, pset, hour_win=0)
                D = np.where(ana["av"][None, :] >= 0.999, ana["D"], np.inf)
                idx, _ = topk(D, max(KS))
                allmem = ana["y_cf"][idx]
                va = ana["va_g"]
                lab = va["target"].notna().to_numpy()
                y = va["target"].to_numpy() / cap
                for K in KS:
                    mem = allmem[:, :K]
                    crps, outl, sr, ideal = stats(mem, y, lab)
                    recs.append(dict(variant=vtag, K=K, alpha=1.0, fold=fname,
                                     group=g, crps=crps, outlier=outl, sr=sr,
                                     ideal_out=ideal))
                for a in ALPHAS[1:]:
                    mem = allmem[:, :20]
                    mm = mem.mean(1, keepdims=True)
                    mem_i = np.clip(mm + a * (mem - mm), 0, 1)
                    crps, outl, sr, ideal = stats(mem_i, y, lab)
                    recs.append(dict(variant=vtag, K=20, alpha=a, fold=fname,
                                     group=g, crps=crps, outlier=outl, sr=sr,
                                     ideal_out=ideal))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    agg = (df.groupby(["variant", "K", "alpha"])[["crps", "outlier", "sr", "ideal_out"]]
             .mean().sort_values(["variant", "alpha", "K"]))
    print("\n=== 4폴드·3그룹 평균 ===", flush=True)
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    print("\n판독: CRPS 낮을수록 · outlier는 ideal_out에 가까울수록 · sr(스프레드/RMSE) 1에 "
          "가까울수록 좋음", flush=True)
    df.to_csv("an11_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n상세 → an11_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
