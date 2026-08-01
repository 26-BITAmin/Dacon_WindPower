# ═══════════════════════════════════════════════════════
# exp-082b: pcinv 실패 지점 진단 — "커브 천장" 가설 검증 (캐시 전용, 학습 0)
#
# 관측: pcinv는 1-NMAE 거의 중립(−0.0007)인데 FiCR만 −0.0074 붕괴.
# 가설: SCADA 파워커브를 bin **평균**으로 만들어 꼭대기가 0.880/0.905/0.816에 눌림
#   → base가 원리적으로 그 위를 예측 못함 → exp-008이 지목한 CF 0.9+ 구간(0원 밴드
#   74.4%)에서 밴드 진입 실패.
# 검정: 실측 CF 구간별로 (a) 두 base의 평균 예측 CF, (b) FiCR 손실 기여를 분해.
#   가설 참이면 고CF 구간에서만 pcinv가 낮고, 저·중 구간은 동급 이상이어야 함.
#
# 실행: C:\miniconda3\python.exe -X utf8 pc01b_diag.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full
from exp035_raw_dl import CACHE_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("### exp-082b pcinv 실패 지점 진단 (캐시 전용) ###", flush=True)
    cache, champ_bias = load_cache_full()
    assert merge_cached_cols(cache, CACHE_DIR / "exp082_gbm_pcinv.parquet", ["cf_pcinv"])
    hold = HOLD_FOLD["name"]

    parts = []
    for (fname, group), sub in build_slices(cache):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        ws, pA = sub["ws"].to_numpy(), sub["pA"].to_numpy()
        out = sub[["fold", "group", "forecast_kst_dtm", "target"]].copy()
        for tag, col in (("champ", "cf_hpo"), ("pcinv", "cf_pcinv")):
            bl = 0.65 * cache[col].to_numpy()[sub.index] + \
                 0.35 * cache["dl_cln03"].to_numpy()[sub.index]
            out[tag] = postprocess_flex(bl, cap, ws, pA, group,
                                        champ_bias[fname], **PP_NEW) / cap
        out["acf"] = out["target"] / cap
        parts.append(out)
    d = pd.concat(parts, ignore_index=True)
    d = d[(d["fold"] != hold) & d["target"].notna() & (d["acf"] >= 0.10)]
    print(f"평가시간 {len(d)}행\n", flush=True)

    bins = [0.10, 0.30, 0.50, 0.70, 0.90, 1.01]
    d["bin"] = pd.cut(d["acf"], bins=bins, right=False)
    print(f"{'실측CF구간':<14}{'n':>7}{'실측평균':>9}{'champ':>9}{'pcinv':>9}{'Δpred':>9}"
          f"{'champ오차':>10}{'pcinv오차':>10}{'Δ오차':>9}")
    for b, s in d.groupby("bin", observed=True):
        ec = np.abs(s["champ"] - s["acf"]).mean()
        ep = np.abs(s["pcinv"] - s["acf"]).mean()
        print(f"{str(b):<14}{len(s):>7}{s['acf'].mean():>9.3f}{s['champ'].mean():>9.3f}"
              f"{s['pcinv'].mean():>9.3f}{s['pcinv'].mean()-s['champ'].mean():>+9.3f}"
              f"{ec:>10.4f}{ep:>10.4f}{ep-ec:>+9.4f}")

    # 밴드 진입률 (FiCR 직결): 오차율 = |pred−act|/act
    print(f"\n{'실측CF구간':<14}{'champ≤6%':>10}{'pcinv≤6%':>10}{'champ≤8%':>10}"
          f"{'pcinv≤8%':>10}  (actual 가중)")
    for b, s in d.groupby("bin", observed=True):
        w = s["acf"].to_numpy()
        rc = np.abs(s["champ"] - s["acf"]).to_numpy() / s["acf"].to_numpy()
        rp = np.abs(s["pcinv"] - s["acf"]).to_numpy() / s["acf"].to_numpy()
        f = lambda r, t: (w * (r <= t)).sum() / w.sum()
        print(f"{str(b):<14}{f(rc,0.06):>10.3f}{f(rp,0.06):>10.3f}"
              f"{f(rc,0.08):>10.3f}{f(rp,0.08):>10.3f}")

    # 커브 천장 도달 여부: pcinv raw base(후처리 전)의 상한 분포
    print("\n[raw base 상한] 후처리 전 cf 값의 분위수")
    mm = cache["target"].notna()
    for col in ("cf_hpo", "cf_pcinv"):
        v = cache.loc[mm, col].to_numpy()
        print(f"  {col:<10} p50 {np.percentile(v,50):.3f} p95 {np.percentile(v,95):.3f} "
              f"p99 {np.percentile(v,99):.3f} max {v.max():.3f} | "
              f">0.85 비율 {(v>0.85).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
