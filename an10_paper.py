# ═══════════════════════════════════════════════════════
# AN-10: Alessandrini et al. 2015 (Renewable Energy 76, 768-781) 충실 재현
#        "A novel application of an analog ensemble for short-term wind power forecasting"
#
# 논문 구성 (우리 셋업 대응):
#   - 예측 대상: 풍력 발전량 직접 (정규화 = CF)                → target/cap
#   - 예측자: 단일 결정론 NWP의 ws10 · ws100 · wd100          → GFS 트리오
#   - 아날로그 탐색: 동일 리드타임만 (day-ahead → 같은 시각)   → hour_win=0
#   - 멤버 K=20, 시간창 ±1 리드, 가중 균등
#   - 평가: "확률론" — CRPS · rank histogram · spread-skill  ← 이번에 처음 하는 축
#           + 앙상블 평균의 결정론 MAE
# 대조: 기후학 앙상블(같은 시각의 과거 실측 전부) · 챔피언 GBM(cf_hpo, 결정론)
# 변형: paper(원형) / +avail filter(AN-02 교훈) / tuned(AN-06 hub 최적가중)
#
# 실행: C:\miniconda3\python.exe -X utf8 an10_paper.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from an02_clean import build_analogs2, topk
from an06_final import PSET_G
from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import FOLD_LIST
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame, slice_fold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PSET_PAPER = [("gfs_ws_10m", 1.0), ("gfs_ws_100m", 1.0),
              ("gfs_wd_sin_100m", 0.5), ("gfs_wd_cos_100m", 0.5)]
K = 20


def crps_ens(members, y):
    """CRPS (empirical ensemble): E|X−y| − 0.5·E|X−X'|.  members [N,K], y [N]"""
    t1 = np.abs(members - y[:, None]).mean(1)
    t2 = np.abs(members[:, :, None] - members[:, None, :]).mean((1, 2))
    return t1 - 0.5 * t2


def crps_clim_terms(vals):
    """기후학 앙상블(고정 멤버 집합)의 내부항 0.5·E|X−X'| 와 E|X−y| 계산기."""
    s = np.sort(vals)
    n = len(s)
    cum = np.concatenate([[0.0], np.cumsum(s)])
    i = np.arange(1, n + 1)
    inner = ((2 * i - n - 1) * s).sum() / (n * n)          # E|X−X'|

    def e_abs(y):                                           # E|X−y| (벡터 y)
        pos = np.searchsorted(s, y)
        left = y * pos - cum[pos]
        right = (cum[n] - cum[pos]) - y * (n - pos)
        return (left + right) / n
    return inner, e_abs


def main():
    t0 = time.time()
    print("### AN-10 Alessandrini 2015 재현 — 발전량 AnEn 확률론 검증 ###", flush=True)
    train_df, _ = build_train_frame()
    cache, _ = load_cache_full()
    print(f"입력 준비 완료 — {time.time()-t0:.0f}s", flush=True)
    folds = dict(FOLD_LIST)

    variants = [("paper", PSET_PAPER, "none"), ("paper_f", PSET_PAPER, "filter"),
                ("tuned_f", None, "filter")]           # tuned = 그룹별 AN-06 가중
    rows, rankhist = [], {v[0]: np.zeros(K + 1, dtype=int) for v in variants}
    for fname in ["fold_2024Q1", "fold_2024Q2", "fold_2024Q3", "fold_2024Q4"]:
        fold = folds[fname]
        for g in [1, 2, 3]:
            tt = time.time()
            cap = CAPACITY_KWH[f"kpx_group_{g}"]
            sub = cache[(cache["fold"] == fname) & (cache["group"] == g)]
            # 기후학 앙상블: 같은 시각의 학습기간 실측 CF (그룹별)
            tr_all, _ = slice_fold(train_df, fold)
            tr_g = tr_all.loc[(tr_all["kpx_group"] == g)
                              & tr_all["target"].notna()]
            clim = {h: crps_clim_terms((s["target"] / cap).to_numpy())
                    for h, s in tr_g.groupby(tr_g["forecast_kst_dtm"].dt.hour)}

            for vtag, pset, avm in variants:
                ana = build_analogs2(train_df, fold, g,
                                     pset if pset is not None else PSET_G[g],
                                     hour_win=0)
                D, y_cf, av = ana["D"], ana["y_cf"], ana["av"]
                if avm == "filter":
                    D = np.where(av[None, :] >= 0.999, D, np.inf)
                idx, _ = topk(D, K)
                mem = y_cf[idx]                                    # [Nva, 20]
                va = ana["va_g"]
                lab = va["target"].notna().to_numpy()
                y = (va["target"].to_numpy() / cap)
                emask = lab & (va["target"].to_numpy() >= cap * 0.10)
                hrs = va["forecast_kst_dtm"].dt.hour.to_numpy()
                j = va[["forecast_kst_dtm"]].merge(
                    sub[["forecast_kst_dtm", "cf_hpo"]],
                    on="forecast_kst_dtm", how="left")
                hpo = j["cf_hpo"].to_numpy()

                crps_a = crps_ens(mem[lab], y[lab])
                cl_in = np.array([clim[h][0] for h in hrs[lab]])
                cl_e = np.array([clim[h][1](np.array([yy]))[0]
                                 for h, yy in zip(hrs[lab], y[lab])])
                crps_c = cl_e - 0.5 * cl_in
                mean_p = mem.mean(1)
                spread = mem.std(1)
                err = np.abs(mean_p - y)
                ranks = (mem[lab] < y[lab][:, None]).sum(1)
                rankhist[vtag] += np.bincount(ranks, minlength=K + 1)
                rows.append(dict(
                    variant=vtag, fold=fname, group=g,
                    mae_all=err[lab].mean(), mae_ev=err[emask].mean(),
                    mae_hpo=np.abs(hpo - y)[emask].mean(),
                    crps=crps_a.mean(), crps_clim=crps_c.mean(),
                    crps_hpo=np.abs(hpo - y)[lab].mean(),
                    ss_corr=np.corrcoef(spread[lab], err[lab])[0, 1],
                    spread_rmse=spread[lab].mean()
                    / max(np.sqrt((err[lab] ** 2).mean()), 1e-9),
                    outlier=((ranks == 0) | (ranks == K)).mean()))
            print(f"  [{fname} g{g}] {time.time()-tt:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("variant")[["mae_all", "mae_ev", "mae_hpo", "crps", "crps_clim",
                                 "crps_hpo", "ss_corr", "spread_rmse",
                                 "outlier"]].mean()
    agg["crpss_clim"] = 1 - agg["crps"] / agg["crps_clim"]     # skill score vs 기후학
    agg["crpss_hpo"] = 1 - agg["crps"] / agg["crps_hpo"]       # vs 결정론 챔피언 GBM
    print("\n=== 4폴드·3그룹 평균 (CF 단위) ===", flush=True)
    with pd.option_context("display.width", 160):
        print(agg.to_string(float_format=lambda x: f"{x:+.4f}"), flush=True)
    print("\n항목: mae_ev/mae_hpo = 평가마스크 앙상블평균 vs 챔피언GBM | "
          "crps* = 전체 라벨행 | ss_corr = spread-skill 상관 | "
          "spread_rmse = 스프레드/RMSE (1이면 완벽 정합) | outlier = 실측이 "
          "앙상블 밖(rank 0 또는 20) 비율 (이상적 = 2/21 = 0.095)", flush=True)
    print("\n=== Rank histogram (21칸을 5구간 집계, 비율) — 평평해야 보정 양호 ===",
          flush=True)
    for vtag, h in rankhist.items():
        p = h / h.sum()
        bins = [p[:4].sum(), p[4:9].sum(), p[9:12].sum(), p[12:17].sum(), p[17:].sum()]
        print(f"  {vtag:<8}: " + " ".join(f"{b:.3f}" for b in bins)
              + f"   (균등이면 {4/21:.3f} {5/21:.3f} {3/21:.3f} {5/21:.3f} {4/21:.3f})",
              flush=True)
    df.to_csv("an10_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n상세 → an10_results.csv | 총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
