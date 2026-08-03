# ═══════════════════════════════════════════════════════
# exp-090: Public LB의 표본 노이즈 규모 측정 — "우리 갱신폭은 신호인가 동전던지기인가"
#
# 문제:
#   최근 LB 갱신폭이 +0.00014(exp-073) / +0.00015(exp-056) / +0.00056(exp-071) / −0.00124(exp-083)다.
#   그런데 **Public은 8,760시간의 40%만 채점**하고 최종 순위는 **Private 100%**다.
#   이 크기의 차이가 표본 추출 노이즈보다 큰지 **한 번도 확인한 적이 없다.**
#   작다면 지금까지의 "챔피언 갱신" 몇 건은 동전던지기였고, 최종 제출물 선택 기준도 바뀐다.
#
# 방법 (2025 라벨이 없으므로 **2024 홀드아웃 1년치**를 대역으로 사용):
#   두 후보의 후처리 예측을 만들고, 시각을 40/60으로 반복 분할해
#     Δ_public(40%) 과 Δ_private(60%) 의 분포를 본다.
#   (a) iid-시각 분할  (b) **일 단위 블록** 분할 — 실제 사전 샘플링이 블록형이면
#       유효 표본이 줄어 노이즈가 커지므로 양쪽으로 괄호를 친다.
#
# 산출:
#   - 40% 표본에서 Δ의 표준편차 (= LB 한 번 읽었을 때의 오차 막대)
#   - **부호 일치율**: Public이 이겼을 때 Private도 이길 확률
#   - 관측된 우리 갱신폭들을 이 척도에 대입
#
# 무결성: 직접 구현한 metric이 기존 score_from_tidy와 일치하는지 먼저 검증.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp090_lbnoise.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import CACHE_DIR, HOLD_FOLD, score_from_tidy
from exp039_dl_diverse import build_slices
from exp049_clean import PP_NEW, postprocess_flex
from exp050_lambda_sweep import merge_cached_cols
from exp060_pergroup_hpo import load_cache_full

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

B = 2000
PUBLIC_FRAC = 0.40
RNG = np.random.default_rng(12345)


def ev_tidy(cache, champ_bias, gbm_col, dl_col, w, pp):
    """후처리까지 끝난 long-format 예측 (fold, group, dtm, target, pred)."""
    blend = (cache[gbm_col].to_numpy() if dl_col is None else
             w * cache[gbm_col].to_numpy() + (1 - w) * cache[dl_col].to_numpy())
    parts = []
    for (fname, group), sub in build_slices(cache):
        cap = CAPACITY_KWH[f"kpx_group_{group}"]
        pred = postprocess_flex(blend[sub.index], cap, sub["ws"].to_numpy(),
                                sub["pA"].to_numpy(), group, champ_bias[fname], **pp)
        parts.append(sub[["fold", "group", "forecast_kst_dtm", "target"]].assign(pred=pred))
    return pd.concat(parts, ignore_index=True)


def prep_arrays(tidy):
    """평가 대상 행만 남기고 그룹별 배열로 정리 (시각 인덱스 포함)."""
    t = tidy.loc[tidy["target"].notna()].copy()
    caps = t["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    t = t.loc[(t["target"].to_numpy() >= caps * 0.10)].copy()      # 공식 평가 마스크
    t["cap"] = t["group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"])
    return t


def score_subset(t, pred_col, keep):
    """keep = 시각 boolean(행 단위). 공식 metric을 직접 계산."""
    sub = t.loc[keep]
    n_, f_ = [], []
    for g, s in sub.groupby("group", sort=True):
        cap = CAPACITY_KWH[f"kpx_group_{int(g)}"]
        a = s["target"].to_numpy()
        er = np.abs(s[pred_col].to_numpy() - a) / cap
        n_.append(er.mean())
        price = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
        f_.append((a * price).sum() / (a * 4.0).sum())
    nm, fi = 1.0 - np.mean(n_), np.mean(f_)
    return 0.5 * nm + 0.5 * fi, nm, fi


def main():
    t0 = time.time()
    print("### exp-090 Public LB 표본 노이즈 규모 ###", flush=True)
    cache, champ_bias = load_cache_full()
    merge_cached_cols(cache, CACHE_DIR / "exp083_gbm_noL.parquet", ["cf_noL"])
    cache["mix58"] = 0.58 * cache["cf_hpo"] + 0.42 * cache["cf_noL"]

    PAIRS = [("챔피언 vs exp-083(t58_w80)", ("cf_hpo", 0.65), ("mix58", 0.80)),
             ("챔피언 vs w0.70만",          ("cf_hpo", 0.65), ("cf_hpo", 0.70))]

    hold = HOLD_FOLD["name"]
    for label, (ga, wa), (gb, wb) in PAIRS:
        print(f"\n{'='*78}\n[{label}]", flush=True)
        ta = ev_tidy(cache, champ_bias, ga, "dl_cln03", wa, PP_NEW)
        tb = ev_tidy(cache, champ_bias, gb, "dl_cln03", wb, PP_NEW)
        # 홀드아웃(2024 1년) — 2025 test와 같은 "1년 전체" 형태
        ta = ta[ta["fold"] == hold]; tb = tb[tb["fold"] == hold]
        merged = ta.rename(columns={"pred": "pA_"}).merge(
            tb[["group", "forecast_kst_dtm", "pred"]].rename(columns={"pred": "pB_"}),
            on=["group", "forecast_kst_dtm"], how="inner")
        t = prep_arrays(merged)
        print(f"  평가 대상 행 {len(t):,} | 고유 시각 {t['forecast_kst_dtm'].nunique():,}", flush=True)

        # 무결성: 전체 점수가 기존 구현과 일치하는지
        full_a = score_from_tidy(ta, "pred")[0]
        my_a = score_subset(t, "pA_", np.ones(len(t), bool))[0]
        print(f"  무결성 검사: score_from_tidy {full_a:.6f} vs 직접구현 {my_a:.6f} "
              f"(차 {abs(full_a-my_a):.2e})", flush=True)

        d_full = (score_subset(t, "pB_", np.ones(len(t), bool))[0] -
                  score_subset(t, "pA_", np.ones(len(t), bool))[0])
        print(f"  **전체(100%) Δ = {d_full:+.5f}**", flush=True)

        times = t["forecast_kst_dtm"].to_numpy()
        uniq = np.sort(pd.unique(times))
        days = pd.Series(t["forecast_kst_dtm"]).dt.floor("D").to_numpy()
        uday = np.sort(pd.unique(days))

        for mode, keys, allkeys in (("iid-시각", times, uniq), ("일-블록", days, uday)):
            dp, dv, sp = [], [], []
            n_pub = int(round(len(allkeys) * PUBLIC_FRAC))
            for _ in range(B):
                pub = set(RNG.choice(allkeys, n_pub, replace=False).tolist())
                m = np.fromiter((k in pub for k in keys), bool, len(keys))
                a_pub = score_subset(t, "pA_", m)[0]; b_pub = score_subset(t, "pB_", m)[0]
                a_prv = score_subset(t, "pA_", ~m)[0]; b_prv = score_subset(t, "pB_", ~m)[0]
                dp.append(b_pub - a_pub); dv.append(b_prv - a_prv); sp.append(a_pub)
            dp, dv, sp = np.array(dp), np.array(dv), np.array(sp)
            agree = np.mean(np.sign(dp) == np.sign(dv))
            print(f"  ── {mode} 분할 (Public {PUBLIC_FRAC:.0%}, B={B}) ──", flush=True)
            print(f"     Δ_public  평균 {dp.mean():+.5f} | **SD {dp.std():.5f}** "
                  f"| 95% [{np.percentile(dp,2.5):+.5f}, {np.percentile(dp,97.5):+.5f}]", flush=True)
            print(f"     Δ_private 평균 {dv.mean():+.5f} | SD {dv.std():.5f}", flush=True)
            print(f"     **Public/Private 부호 일치율 {agree:.1%}**  "
                  f"| 챔피언 절대점수 SD {sp.std():.5f}", flush=True)

    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
