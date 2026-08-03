# ═══════════════════════════════════════════════════════
# exp-096: ASOS 관측을 **DL 멤버 입력**에 투입 — 투입 위치 축 (exp-082의 후속)
#
# 근거 (사용자 질문에서 확정된 패턴):
#   역대 큰 상승은 전부 DL 쪽 개입 또는 멤버 간 역할 분담이었음 —
#   exp-035(DL 입력 교체) +0.0051 / exp-042(GBM x DL 교차) +0.0052 / exp-033(DL 신설).
#   반면 GBM 입력 추가는 exp-054·072·073·082·089 전부 노이즈~기각.
#   **결정적 증거 exp-040**: exp-035와 동일 정보(격자편차 216)를 GBM에 주자
#   base 단독은 +0.0012인데 블렌드 총점 ±0 (DL과 중복 -> 탈상관 0.818->0.840 소실).
#   -> exp-082(ASOS를 GBM에)가 기각됐다고 해서 ASOS 자체가 무용하다는 결론은 못 냄.
#
# exp-035 교훈 반영: "수제 축약은 실패, 원시 편차만 성공 — 추출을 NN에 맡길 것".
#   exp-082는 97지점을 상위5 평균 스칼라로 뭉갰음 -> 여기서는 **지점별 원시 편차**를 준다.
#   피처: obs_mean(창 공간평균 1) + obs_dev_{stn}(지점별 값 − 공간평균, N개)
#   창 = [D-2 12:00, D-1 12:00] (규정: 관측시각 <= D-1 12:00, exp-082와 동일)
#
# 변경점 격리: GBM(cf_hpo)·후처리(PP_NEW)·가중(cln03)·손실(q60)·시드 전부 챔피언 그대로,
#   **DL 입력만** spd(348) -> spd + 관측편차. exp-035와 동일한 변경 유형.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp096_dl_obs.py [--seeds 3] [--min-cov 0.98]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import DATA_DIR
from exp035_raw_dl import build_inputs, FOLD_LIST, CACHE_DIR
from exp049_clean import KEY, PP_NEW, merge_avail, dl_fold_predict_clean
from exp050_lambda_sweep import merge_cached_cols, save_cols
from exp060_pergroup_hpo import load_cache_full, make_ev, print_table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ASOS_PARQUET = DATA_DIR / "external" / "kma_asos.parquet"
DL_OBS_CACHE = CACHE_DIR / "exp096_dl_obs.parquet"


def build_obs_dev(train_df, min_cov):
    """지점별 원시 편차 (exp-035 spd 형태). 창 = [D-2 12:00, D-1 12:00]."""
    a = pd.read_parquet(ASOS_PARQUET)
    a["ws"] = pd.to_numeric(a["ws"], errors="coerce")
    piv = a.pivot_table(index="tm", columns="stn", values="ws").sort_index()
    cov = piv.notna().mean()
    keep = cov[cov >= min_cov].index.tolist()
    piv = piv[keep]
    print(f"ASOS 지점 {len(cov)}개 중 커버리지 >= {min_cov:.0%} 인 {len(keep)}개 사용", flush=True)

    idx = piv.index.to_numpy()
    vals = piv.to_numpy(dtype=np.float32)          # (시각, 지점)
    days = pd.DatetimeIndex(sorted(
        (train_df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize().unique()))
    rows, dcol = [], []
    for d in days:
        d64 = np.datetime64(d)
        m = (idx >= d64 - np.timedelta64(36, "h")) & (idx <= d64 - np.timedelta64(12, "h"))
        if m.sum() < 6:
            continue
        v = np.nanmean(vals[m], axis=0)             # 지점별 창 평균
        if not np.isfinite(v).any():
            continue
        mu = float(np.nanmean(v))
        rows.append(np.concatenate([[mu], v - mu]))
        dcol.append(d)
    cols = ["obs_mean"] + [f"obs_dev_{int(s)}" for s in keep]
    out = pd.DataFrame(rows, columns=cols)
    out["tday"] = dcol
    print(f"관측 편차 피처: {len(out)}일 x {len(cols)}개", flush=True)
    return out, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--min-cov", type=float, default=0.98)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-096: ASOS 관측 -> DL 입력 (투입 위치 축) ###", flush=True)

    train_df, _, colsets = build_inputs(["spd"])
    train_df = merge_avail(train_df)
    vcols = colsets["spd"]
    obs, ocols = build_obs_dev(train_df, args.min_cov)
    train_df["tday"] = (train_df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    train_df = train_df.merge(obs, on="tday", how="left")
    dl_cols = vcols + ocols
    cov = train_df.groupby(train_df["forecast_kst_dtm"].dt.year)["obs_mean"] \
                  .apply(lambda d: d.notna().mean().round(3))
    print(f"DL 입력 {len(vcols)} -> {len(dl_cols)}개 | 연도별 커버리지:\n{cov.to_string()}", flush=True)

    cache, champ_bias = load_cache_full()
    col = "dl_obs"
    if not merge_cached_cols(cache, DL_OBS_CACHE, [col]):
        out = np.full(len(cache), np.nan)
        for fname, fold in FOLD_LIST:
            tf = time.time()
            tr_all = train_df.loc[train_df["forecast_kst_dtm"] <= fold["train_end"]]
            sel = (cache["fold"] == fname).to_numpy()
            keys = cache.loc[sel, ["group", "forecast_kst_dtm"]]
            va = keys.merge(train_df, left_on=["group", "forecast_kst_dtm"],
                            right_on=["kpx_group", "forecast_kst_dtm"], how="left")
            assert len(va) == len(keys), "정렬 실패"
            out[sel] = dl_fold_predict_clean(tr_all, va, dl_cols, args.seeds, "cln03")
            print(f"  [dl_obs:{fname}] {time.time()-tf:.0f}s", flush=True)
        cache[col] = out
        save_cols(cache, DL_OBS_CACHE, [col])
        print(f"  캐시 저장 -> {DL_OBS_CACHE.name}", flush=True)

    mm = cache["target"].notna()
    c_dl = np.corrcoef(cache.loc[mm, col], cache.loc[mm, "dl_cln03"])[0, 1]
    c_gbm = np.corrcoef(cache.loc[mm, col], cache.loc[mm, "cf_hpo"])[0, 1]
    c_ref = np.corrcoef(cache.loc[mm, "dl_cln03"], cache.loc[mm, "cf_hpo"])[0, 1]
    shift = (cache[col].to_numpy() - cache["dl_cln03"].to_numpy()).mean() * 100
    print(f"\ncorr(dl_obs, dl_cln03) = {c_dl:.4f} | corr(dl_obs, cf_hpo) = {c_gbm:.4f} "
          f"(챔피언 DL은 {c_ref:.4f}) | 평균 이동 {shift:+.3f}%cap", flush=True)

    ev = make_ev(cache, champ_bias)
    ref = ev("cf_hpo", "dl_cln03", 0.65, PP_NEW)
    print(f"\n챔피언(hpo x dC_w65) 재현: CV {ref['cv'][0]:.4f} (1-NMAE {ref['cv'][1]:.4f} "
          f"FiCR {ref['cv'][2]:.4f}) | HOLD {ref['hold'][0]:.4f}  (기대 0.6556/0.6535)", flush=True)
    res = {"dlobs_solo": ev(col, None, 1.0, PP_NEW),
           "dlC_solo": ev("dl_cln03", None, 1.0, PP_NEW)}
    for w in (0.55, 0.60, 0.65, 0.70, 0.75):
        res[f"dlobs_w{int(w*100)}"] = ev("cf_hpo", col, w, PP_NEW)
    # DL 멤버 혼합(챔피언 DL과 신 DL 반반)도 한 칸
    cache["dl_mix"] = 0.5 * cache["dl_cln03"].to_numpy() + 0.5 * cache[col].to_numpy()
    for w in (0.65, 0.70):
        res[f"dlmix_w{int(w*100)}"] = ev("cf_hpo", "dl_mix", w, PP_NEW)
    print_table(res, ref)
    d_cv = res["dlobs_solo"]["cv"][0] - res["dlC_solo"]["cv"][0]
    d_hd = res["dlobs_solo"]["hold"][0] - res["dlC_solo"]["hold"][0]
    print(f"DL 단독 델타 (dl_obs - dl_cln03): dCV {d_cv:+.4f} / dhold {d_hd:+.4f}")
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
