# ═══════════════════════════════════════════════════════
# AN-02: AnEn + 가용성 정제 아날로그 — AN-01 후속
#
# AN-01 결과: 최선 hub/hw3/K200 = MAE +0.0255 열세, corr 0.926 (실다양성),
#             signed −0.061 (강한 과소예측)
# 가설: 아날로그 멤버(과거 실측 CF)에 터빈 정지·저가용 시간의 눌린 라벨이 섞임
#       → 챔피언 라인이 cln03 가중으로 잡은 것과 같은 원인.
#
# 변경 3가지:
#   1) avail: none | filter(avail≥0.999만 후보) | wfac(멤버를 avail_factor로 가중)
#   2) K 확대: K_MAX 500 (AN-01에서 K 단조 개선)
#   3) 예측자: 멤버 평균 대신 분위수 q∈{0.55, 0.60} (FiCR용 상향, DL q60과 동형)
#
# 실행: C:\miniconda3\python.exe -X utf8 an02_clean.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an01_anen import PRED_SETS, anen_dists, lagged_block
from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST
from exp049_clean import avail_factor
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

K_MAX = 500
TWIN = 1


def build_analogs2(train_df, fold, group, pset, hour_win=3):
    """AN-01과 동일하되 후보의 avail_strict·CF·거리 전부 반환 (avail 처리는 밖에서)."""
    cols = [c for c, _ in pset]
    w = np.array([wt for _, wt in pset])
    cap = CAPACITY_KWH[f"kpx_group_{group}"]
    tr_all, va_all = slice_fold(train_df, fold)
    tr_g = tr_all.loc[(tr_all["kpx_group"] == group)].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    va_g = va_all.loc[va_all["kpx_group"] == group].sort_values(
        "forecast_kst_dtm").reset_index(drop=True)
    A = lagged_block(tr_g, cols)
    F = lagged_block(va_g, cols)
    ok = tr_g["target"].notna().to_numpy()
    sig = np.nanstd(tr_g[cols].to_numpy(dtype=np.float64), axis=0)
    D = anen_dists(F, A[ok], sig, w)
    if hour_win is not None:
        h_tr = tr_g.loc[ok, "forecast_kst_dtm"].dt.hour.to_numpy()
        h_va = va_g["forecast_kst_dtm"].dt.hour.to_numpy()
        dh = np.abs(h_va[:, None] - h_tr[None, :])
        dh = np.minimum(dh, 24 - dh)
        D = np.where(dh <= hour_win, D, np.inf)
    return dict(va_g=va_g, cap=cap, D=D,
                y_cf=(tr_g.loc[ok, "target"].to_numpy() / cap),
                av=np.nan_to_num(tr_g.loc[ok, "avail_strict"].to_numpy(dtype=float),
                                 nan=1.0))


def topk(D, kmax):
    k = min(kmax, D.shape[1])
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    rows = np.arange(D.shape[0])[:, None]
    dsub = D[rows, idx]
    order = np.argsort(dsub, axis=1)
    return idx[rows, order], dsub[rows, order]


def variants_from(ana):
    """avail 모드별 상위 K_MAX 아날로그 테이블 생성."""
    out = {}
    D, y, av = ana["D"], ana["y_cf"], ana["av"]
    # none
    idx, d = topk(D, K_MAX)
    out["none"] = (y[idx], np.ones_like(idx, dtype=float), d)
    # filter: 만가용 후보만
    Df = np.where(av[None, :] >= 0.999, D, np.inf)
    idx, d = topk(Df, K_MAX)
    out["filter"] = (y[idx], np.ones_like(idx, dtype=float), d)
    # wfac: cln03 가중
    idx, d = topk(D, K_MAX)
    out["wfac"] = (y[idx], avail_factor(av[idx], "cln03"), d)
    return out


def predict(tbl, K, mode):
    cf, wm, _ = tbl
    cf, wm = cf[:, :K], wm[:, :K]
    if mode == "mean":
        return (cf * wm).sum(1) / np.maximum(wm.sum(1), 1e-12)
    q = float(mode[1:]) / 100.0                     # "q55" → 0.55 (가중 미지원 → 균등)
    return np.quantile(cf, q, axis=1)


def main():
    t0 = time.time()
    print("### AN-02 AnEn 가용성 정제 (Q4 스모크) ###", flush=True)
    train_df, _ = build_train_frame()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    fold = dict(FOLD_LIST)["fold_2024Q4"]
    cache, _ = load_cache_full()

    ev, tabs = {}, {}
    for g in [1, 2, 3]:
        sub = cache[(cache["fold"] == fold["name"]) & (cache["group"] == g)]
        tt = time.time()
        ana = build_analogs2(train_df, fold, g, PRED_SETS["hub"], 3)
        tabs[g] = variants_from(ana)
        va = ana["va_g"]
        m = va["target"].notna().to_numpy() & \
            (va["target"].to_numpy() >= ana["cap"] * 0.10)
        j = va[["forecast_kst_dtm"]].merge(
            sub[["forecast_kst_dtm", "cf_hpo"]], on="forecast_kst_dtm", how="left")
        ev[g] = dict(m=m, y=(va["target"].to_numpy() / ana["cap"]),
                     hpo=j["cf_hpo"].to_numpy())
        print(f"  그룹{g} 거리+테이블 {time.time()-tt:.0f}s", flush=True)
    ref_mae = np.mean([np.abs(ev[g]["hpo"][ev[g]["m"]] - ev[g]["y"][ev[g]["m"]]).mean()
                       for g in [1, 2, 3]])
    print(f"기준 cf_hpo MAE {ref_mae:.5f}  (AN-01 최선 0.13584)", flush=True)

    rows = []
    for avm in ["none", "filter", "wfac"]:
        for K in [100, 200, 300, 500]:
            modes = ["mean", "q55", "q60"] if avm == "filter" else ["mean"]
            for pm in modes:
                maes, sgns, cats, refc = [], [], [], []
                for g in [1, 2, 3]:
                    cf = predict(tabs[g][avm], K, pm)
                    m, y = ev[g]["m"], ev[g]["y"]
                    maes.append(np.abs(cf[m] - y[m]).mean())
                    sgns.append((cf[m] - y[m]).mean())
                    cats.append(cf[m])
                    refc.append(ev[g]["hpo"][m])
                cat, rc = np.concatenate(cats), np.concatenate(refc)
                okm = np.isfinite(cat) & np.isfinite(rc)
                rows.append(dict(avail=avm, K=K, pred=pm, mae=np.mean(maes),
                                 signed=np.mean(sgns),
                                 corr=np.corrcoef(cat[okm], rc[okm])[0, 1]))
    df = pd.DataFrame(rows).sort_values("mae")
    df["d_mae"] = df["mae"] - ref_mae
    with pd.option_context("display.width", 130):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.5f}"), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
