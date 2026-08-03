# ═══════════════════════════════════════════════════════
# exp-106 진단: 예보 풍향 섹터가 챔피언 오차를 움직이는가
#
# exp-105 D에서 확인: 그룹3 터빈마다 **최악 정지 섹터가 전부 다름**
#   (wtg01 90°에서 55.4% · wtg02 315° · wtg03 0° · wtg04 45° · wtg05 270°)
#   = wind sector management(후류·난류 섹터) 서명. 빈번하고 **풍향은 예보에 있다.**
#
# 그런데 exp-105은 **나셀 풍향**(SCADA, 예측 시점 미관측) 기준이었다.
#   실행 가능하려면 **예보 풍향**으로 같은 구조가 보여야 한다 — 그것부터 확인한다.
#   (예측 CF로 조건을 걸었을 때 부호가 뒤집혔던 exp-109의 교훈: 조건 변수는 반드시 예측 시점 관측 가능한 것)
#
# 단계:
#   ① 예보 풍향 섹터 × 그룹 → 가동률·챔피언 오차·FiCR 손실
#   ② 나셀 풍향과 예보 풍향의 대응(회전·산포) 확인 — 섹터가 흐려지는 정도
#   ③ **LOFO**: 섹터별 배율을 나머지 3폴드에서 골라 해당 폴드 적용 (exp-109과 동일 판정)
#
# 실행: C:\miniconda3\python.exe -X utf8 exp102_wdsector.py
# ═══════════════════════════════════════════════════════
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from exp005_qopt import CAPACITY_KWH
from exp035_raw_dl import HOLD_FOLD, CACHE_DIR
from exp043_post_resweep import postprocess_flex
from exp049_clean import PP_NEW, AVAIL_PARQUET
from exp060_pergroup_hpo import load_cache_full, build_slices
from exp065_gbm3 import build_train_frame
from exp110_effweight import EFF_PARQUET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NSEC = 8
KS = np.arange(0.85, 1.16, 0.025)


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
    return 0.5 * (1 - np.mean(ns)) + 0.5 * np.mean(fs)


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

    tf, _ = build_train_frame()
    # 피처는 wd_rad = arctan2(v, u) = **수학 관례**(동 기준 반시계, 바람이 가는 방향).
    # SCADA 나셀 풍향은 **기상 관례**(북 기준 시계, 바람이 오는 방향) → 변환 필요.
    #   wd_met = (270 − wd_math) mod 360.  거울상+회전이라 단순 오프셋이 아니다.
    wd = (270.0 - np.degrees(np.arctan2(tf["ldaps_wd_sin_10m"], tf["ldaps_wd_cos_10m"]))) % 360
    w = tf[["forecast_kst_dtm", "kpx_group"]].copy()
    w["fwd"] = wd.to_numpy()
    w["fws"] = tf["ldaps_ws_10m"].to_numpy()
    t = t.merge(w.rename(columns={"kpx_group": "group"}), on=["forecast_kst_dtm", "group"], how="left")

    av = pd.read_parquet(AVAIL_PARQUET)
    av["forecast_kst_dtm"] = pd.to_datetime(av["forecast_kst_dtm"])
    t = t.merge(av[["forecast_kst_dtm", "group", "avail_strict"]], on=["forecast_kst_dtm", "group"],
                how="left")
    eff = pd.read_parquet(EFF_PARQUET)[["forecast_kst_dtm", "group", "eff"]]
    t = t.merge(eff, on=["forecast_kst_dtm", "group"], how="left")
    t["sec"] = (t["fwd"] // (360 / NSEC) % NSEC).astype("Int64")
    t["e"] = (t["pred"] - t["target"]) / t["cap"]
    t["단가"] = np.select([t["e"].abs() <= 0.06, t["e"].abs() <= 0.08], [4.0, 3.0], default=0.0)
    return t


def main():
    t0 = time.time()
    print("### exp-106 예보 풍향 섹터 진단 ###", flush=True)
    t = build()
    ev = t[t["target"] >= t["cap"] * 0.10].copy()
    print(f"평가행 {len(ev):,} | 예보풍향 결측 {ev['fwd'].isna().mean():.3f}", flush=True)

    print("\n═══ ① 예보 풍향 섹터 × 그룹 ═══", flush=True)
    for g in [1, 2, 3]:
        s = ev[ev.group == g]
        den = (s["target"] * 4).sum()
        r = s.groupby("sec", observed=False).apply(lambda x: pd.Series({
            "n": len(x),
            "완전가동%": (x["avail_strict"] >= 0.999).mean() * 100,
            "eff평균": x["eff"].mean(),
            "평균오차%": x["e"].mean() * 100,
            "MAE": x["e"].abs().mean(),
            "FiCR손실": (x["target"] * (4 - x["단가"])).sum() / den,
        }))
        r.index = [f"{int(i)*45}-{int(i)*45+45}°" for i in r.index]
        print(f"--- 그룹 {g} ---"); print(r.round(3).to_string()); print(flush=True)

    print("═══ ② 나셀 풍향 vs 예보 풍향 대응 ═══", flush=True)
    sl = pd.read_parquet(CACHE_DIR / "exp105_slots.parquet")
    sl = sl.dropna(subset=["wd"]).groupby(["group", "hour"]).apply(
        lambda x: np.degrees(np.arctan2(np.sin(np.radians(x.wd)).mean(),
                                        np.cos(np.radians(x.wd)).mean())) % 360).rename("nwd").reset_index()
    j = t.merge(sl.rename(columns={"hour": "forecast_kst_dtm"}), on=["group", "forecast_kst_dtm"],
                how="inner").dropna(subset=["fwd", "nwd"])
    d = (j["nwd"] - j["fwd"] + 180) % 360 - 180
    print(f"  n={len(j):,} | 각도차(나셀−예보) 중앙 {np.median(d):+.1f}° | "
          f"|차이|<45° 비율 {(np.abs(d) < 45).mean()*100:.1f}% | "
          f"같은 섹터 비율 {(((j.nwd//45) % 8) == ((j.fwd//45) % 8)).mean()*100:.1f}%", flush=True)

    print("\n═══ ③ 섹터별 배율의 LOFO 전이 ═══", flush=True)
    folds = sorted(t["fold"].unique())

    def ap(x, km):
        o = x.copy()
        for (g, s), k in km.items():
            m = (o["group"] == g) & (o["sec"] == s)
            o.loc[m, "pred"] = np.clip(o.loc[m, "pred"] * k, 0, o.loc[m, "cap"])
        return o

    def fit(x):
        km = {}
        for g in [1, 2, 3]:
            for s in range(NSEC):
                sc = [score(ap(x, {(g, s): k})) for k in KS]
                km[(g, s)] = KS[int(np.argmax(sc))]
        return km

    b = score(t)
    km_all = fit(t)
    print(f"  기준 {b:.4f} | 오라클(사후 최적) {score(ap(t, km_all)):.4f} "
          f"(Δ {score(ap(t, km_all))-b:+.4f})", flush=True)
    tot = 0.0
    picks = {}
    for fn in folds:
        oth, cur = t[t.fold != fn], t[t.fold == fn]
        k = fit(oth)
        picks[fn] = k
        tot += score(ap(cur, k)) - score(cur)
    print(f"  **LOFO 평균 Δ = {tot/len(folds):+.4f}**", flush=True)
    print("  그룹3 섹터별 선택: " + " | ".join(
        f"{fn.replace('fold_2024','')}:" + ",".join(f"{picks[fn][(3,s)]:.2f}" for s in range(NSEC))
        for fn in folds), flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
