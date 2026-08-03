# ═══════════════════════════════════════════════════════
# exp-109 진단: 그룹3 전용 · 예측 CF 창(window) 국소 보정의 오라클 상한과 안정성
#
# 사용자 제안: "그룹 1·2는 그대로 두고 3만 0.5~0.8 구간에 뭔가 하면 되지 않나".
#   → 논쟁 대신 직접 측정한다. 단 조건 변수는 **예측 CF**여야 한다(실측 CF는 예측 시점에 미관측,
#     EDA 10.5의 대각선 적중률 44%가 그 희석의 정량판).
#
# 측정: 그룹3의 예측 CF가 창 W 안인 행에만 가산 시프트 δ(%cap)를 적용 → 공식 점수 재계산.
#   (곱셈이 아니라 가산 — 스칼라 곱 프로브 금지 지시 준수. 제출 아님, 계산만.)
#   ① 창별·δ별 ΔScore (오라클 상한: 사후 최적 δ를 알고 있다고 가정한 값)
#   ② **폴드별 최적 δ의 안정성** — exp-046/061을 죽인 것이 바로 이 불안정성이므로 핵심 판정 기준
#   ③ LOFO: 나머지 3폴드에서 고른 δ를 해당 폴드에 적용 (실제로 쓸 수 있는 값)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp109_g3window.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import HOLD_FOLD
from exp043_post_resweep import postprocess_flex
from exp049_clean import PP_NEW, AVAIL_PARQUET
from exp060_pergroup_hpo import load_cache_full, build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WINDOWS = [(0.5, 0.8), (0.5, 1.01), (0.7, 1.01), (0.8, 1.01), (0.6, 0.9), (0.0, 1.01)]
DELTAS = [-0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13]


def base_frame():
    cache, cb = load_cache_full()
    cache["blend"] = 0.65 * cache["cf_hpo"] + 0.35 * cache["dl_cln03"]
    parts = []
    for (fn, g), sub in build_slices(cache):
        if fn == HOLD_FOLD["name"]:
            continue
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        p = postprocess_flex(cache["blend"].to_numpy()[sub.index], cap, sub["ws"].to_numpy(),
                             sub["pA"].to_numpy(), g, cb[fn], **PP_NEW)
        parts.append(sub[["fold", "group", "target", "forecast_kst_dtm"]]
                     .assign(pred=p, cap=cap))
    t = pd.concat(parts, ignore_index=True)
    t = t[t["target"].notna()].copy()
    t["pcf"] = t["pred"] / t["cap"]
    return t


def score(t):
    """공식 지표 — 그룹별 계산 후 평균. 평가 대상(실측 ≥10%cap)만."""
    ns, fs = [], []
    for g, s in t.groupby("group"):
        cap = s["cap"].iloc[0]
        m = s["target"] >= cap * 0.10
        if not m.any():
            continue
        er = (s.loc[m, "pred"] - s.loc[m, "target"]).abs() / cap
        ns.append(er.mean())
        up = np.select([er <= 0.06, er <= 0.08], [4.0, 3.0], default=0.0)
        fs.append((s.loc[m, "target"] * up).sum() / (s.loc[m, "target"] * 4).sum())
    n, f = np.mean(ns), np.mean(fs)
    return 0.5 * (1 - n) + 0.5 * f, 1 - n, f


def apply_shift(t, lo, hi, d):
    out = t.copy()
    sel = (out["group"] == 3) & (out["pcf"] >= lo) & (out["pcf"] < hi)
    out.loc[sel, "pred"] = np.clip(out.loc[sel, "pred"] + d * out.loc[sel, "cap"],
                                   0, out.loc[sel, "cap"])
    return out, int(sel.sum())


def main():
    t = base_frame()
    b, bn, bf = score(t)
    folds = sorted(t["fold"].unique())
    print(f"기준(챔피언 후처리 블렌드): Score {b:.4f} | 1-NMAE {bn:.4f} | FiCR {bf:.4f}", flush=True)

    # 창 안 행의 실태
    print("\n=== 그룹3 예측 CF 창별 실태 ===", flush=True)
    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    tv = t.merge(av[["forecast_kst_dtm", "group", "avail_strict"]],
                 on=["forecast_kst_dtm", "group"], how="left")
    for lo, hi in WINDOWS[:5]:
        s = tv[(tv.group == 3) & (tv.pcf >= lo) & (tv.pcf < hi)]
        e = (s.pred - s.target) / s.cap
        m = s.target >= s.cap * 0.10
        print(f"  예측CF [{lo:.1f},{hi:.2f}) n={len(s):5d} | 평가행 {m.sum():5d} | "
              f"평균오차 {e[m].mean()*100:+6.2f}%cap | 과소8%초과 {(e[m]<-0.08).mean()*100:5.1f}% | "
              f"완전가동 {(s.avail_strict>=0.999).mean()*100:5.1f}%", flush=True)

    print("\n=== 창 × δ 오라클 ΔScore (사후 최적 가정) ===", flush=True)
    print(f"{'창':<14}" + "".join(f"{d*100:>+8.0f}%" for d in DELTAS), flush=True)
    best = {}
    for lo, hi in WINDOWS:
        row, sc = f"[{lo:.1f},{hi:.2f})".ljust(14), []
        for d in DELTAS:
            s, _ = apply_shift(t, lo, hi, d)
            sc.append(score(s)[0] - b)
        best[(lo, hi)] = sc
        row += "".join(f"{x:>+9.4f}" for x in sc)
        print(row, flush=True)

    # 폴드별 최적 δ 안정성 + LOFO
    print("\n=== 폴드별 최적 δ (안정성) 와 LOFO ===", flush=True)
    for lo, hi in WINDOWS[:4]:
        per = {}
        for fn in folds:
            tf = t[t.fold == fn]
            bf_ = score(tf)[0]
            per[fn] = [score(apply_shift(tf, lo, hi, d)[0])[0] - bf_ for d in DELTAS]
        argbest = {fn: DELTAS[int(np.argmax(v))] for fn, v in per.items()}
        lofo = 0.0
        for fn in folds:
            others = [x for x in folds if x != fn]
            mean_other = [np.mean([per[o][i] for o in others]) for i in range(len(DELTAS))]
            pick = int(np.argmax(mean_other))
            lofo += per[fn][pick]
        print(f"  [{lo:.1f},{hi:.2f}) 폴드별 최적 δ = "
              + " / ".join(f"{fn.replace('fold_2024','')}:{argbest[fn]*100:+.0f}%" for fn in folds)
              + f"  | LOFO 평균 Δ = {lofo/len(folds):+.4f}", flush=True)


if __name__ == "__main__":
    main()
