# ═══════════════════════════════════════════════════════
# exp-111C: 예측 유효출력률 `eff_hat` 기반 **연속 후처리 보정** (캐시 전용, 재학습 0)
#
# A단계 결과: eff는 예보 피처로 실제 예측된다 — 그룹 내부 AUC 0.758~0.798 · R² 0.39~0.49.
#   (exp-091/094의 avail_strict 기준 0.60은 오라벨 타깃 탓이었음이 확인됨)
#
# 앞서 잰 손익분기 AUC 0.93은 **이분 스위치 + 방향이 반대인 두 배율**의 값이다.
#   여기서는 eff_hat 분위 구간별 **연속(계단) 배율**을 쓴다 — 훨씬 완만하고 방향이 단일하다.
#
# 핵심 통제: base 모델은 이미 ws·기온·습도를 입력으로 보므로 **평균적 열화는 이미 학습돼 있다.**
#   따라서 여기서 잡히는 것은 그 위의 **잔여 성분**뿐이고, 잔여가 없으면 적합 배율이 1.0 근처로
#   나와 이득이 0이 된다 — 그 자체가 판정이다.
#
# 판정: **LOFO**(나머지 3폴드에서 배율 선택 → 해당 폴드 적용). exp-046/061/097을 죽인 기준.
#   오라클(사후 최적)은 천장 표시용.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp111c_effhat_pp.py
# ═══════════════════════════════════════════════════════
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import HOLD_FOLD, CACHE_DIR
from exp043_post_resweep import postprocess_flex
from exp049_clean import PP_NEW
from exp060_pergroup_hpo import load_cache_full, build_slices

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NBIN = 5
KS = np.arange(0.80, 1.21, 0.025)


def score(t):
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
    n, f = 1 - np.mean(ns), np.mean(fs)
    return 0.5 * n + 0.5 * f, n, f


def build():
    cache, cb = load_cache_full()
    cache["blend"] = 0.65 * cache["cf_hpo"] + 0.35 * cache["dl_cln03"]
    parts = []
    for (fn, g), sub in build_slices(cache):
        if fn == HOLD_FOLD["name"]:
            continue
        cap = CAPACITY_KWH[f"kpx_group_{g}"]
        p = postprocess_flex(cache["blend"].to_numpy()[sub.index], cap, sub["ws"].to_numpy(),
                             sub["pA"].to_numpy(), g, cb[fn], **PP_NEW)
        parts.append(sub[["fold", "group", "target", "forecast_kst_dtm"]].assign(pred=p, cap=cap))
    t = pd.concat(parts, ignore_index=True)
    t = t[t["target"].notna()].copy()
    eh = pd.read_parquet(CACHE_DIR / "exp099_effhat.parquet") \
        .rename(columns={"kpx_group": "group"})
    t = t.merge(eh, on=["forecast_kst_dtm", "group"], how="left")
    return t


def apply_k(t, kmap):
    o = t.copy()
    for (g, b), k in kmap.items():
        s = (o["group"] == g) & (o["bin"] == b)
        o.loc[s, "pred"] = np.clip(o.loc[s, "pred"] * k, 0, o.loc[s, "cap"])
    return o


def fit(t):
    """(그룹, eff_hat 분위구간)별 최적 배율 — 좌표별 탐색."""
    km = {}
    for g in [1, 2, 3]:
        for b in range(NBIN):
            sc = [score(apply_k(t, {(g, b): k}))[0] for k in KS]
            km[(g, b)] = KS[int(np.argmax(sc))]
    return km


def main():
    t0 = time.time()
    print("### exp-111C eff_hat 연속 후처리 보정 ###", flush=True)
    t = build()
    print(f"평가행 {len(t):,} | eff_hat 결측 {t['eff_hat'].isna().mean():.3f}", flush=True)
    t["eff_hat"] = t["eff_hat"].fillna(t["eff_hat"].median())
    t["bin"] = t.groupby("group")["eff_hat"].transform(
        lambda s: pd.qcut(s, NBIN, labels=False, duplicates="drop"))
    b, bn, bf = score(t)
    print(f"기준 Score {b:.4f} (1-NMAE {bn:.4f} FiCR {bf:.4f})\n", flush=True)

    print("=== eff_hat 구간별 실태 (평가 대상) ===", flush=True)
    ev = t[t["target"] >= t["cap"] * 0.10].copy()
    ev["e"] = (ev["pred"] - ev["target"]) / ev["cap"]
    r = ev.groupby(["group", "bin"]).agg(n=("e", "size"), eff_hat=("eff_hat", "mean"),
                                         평균오차=("e", lambda s: s.mean() * 100)).round(3)
    print(r.to_string(), flush=True)

    km = fit(t)
    oa = score(apply_k(t, km))
    print(f"\n오라클(사후 최적 배율): Score {oa[0]:.4f}  Δ {oa[0]-b:+.4f}", flush=True)
    print("  배율: " + " | ".join(
        f"g{g}b{bn_}={km[(g,bn_)]:.3f}" for g in [1, 2, 3] for bn_ in range(NBIN)), flush=True)

    folds = sorted(t["fold"].unique())
    tot, picks = 0.0, {}
    for fn in folds:
        oth, cur = t[t.fold != fn], t[t.fold == fn]
        k = fit(oth)
        picks[fn] = k
        tot += score(apply_k(cur, k))[0] - score(cur)[0]
    print(f"\n**LOFO 평균 Δ = {tot/len(folds):+.4f}**", flush=True)
    print("  폴드별 선택(그룹3): " + " | ".join(
        f"{fn.replace('fold_2024','')}:" + ",".join(f"{picks[fn][(3,b)]:.2f}" for b in range(NBIN))
        for fn in folds), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
