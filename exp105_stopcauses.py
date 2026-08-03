# ═══════════════════════════════════════════════════════
# exp-105 진단: 터빈 정지·감발 원인의 데이터 서명 조사
#
# 사용자 제공 도메인 지식(2026-08-03)에서 **검증 가능한 서명**만 추려 SCADA에서 확인한다.
#   A. 고풍속 컷아웃 — V126 22.5 / U136 22 / V105-3.3 25 m/s, 재기동은 더 낮음(히스테리시스)
#   B. 저온 정지 — 표준 하한 −20℃ (저온 패키지 −30℃)
#   C. **착빙 시 나셀 풍속계 결빙** — 계측 ws가 실제보다 낮게 → "같은 출력에 낮은 풍속"
#      = 겉보기 **과다생산(r>1)** 아티팩트. ★ 이게 사실이면 exp-110의 `eff` 지표가
#        착빙 구간에서 열화를 **과소**하게 잡는다 (분모 ref가 같이 내려가므로).
#   D. Wind sector management — 특정 풍향에서 후류·난류로 감발/정지 (터빈별로 다름)
#   E. 박쥐 저감 — cut-in 인위 상향, **늦여름~가을 야간**에 집중 (사례: 야간의 31~39%)
#   F. 그림자 깜빡임 — 일출·일몰 시간대 프로그램 정지 (태양 방위각·시각 결정론)
#   G. 고장 다운타임의 두터운 꼬리 — 소형 75%건/5%시간 vs 대형 25%건/95%시간
#   H. 계통 정전·curtailment — 전 터빈 동시 정지, 바람 좋을 때 더 잦음
#
# 슬롯 단위 정지 정의: `power <= 1 kW` (exp-049 dead 정의에서 ws≥5 조건만 뺌 —
#   저풍속 정지(E)도 봐야 하므로).
#
# 실행: C:\miniconda3\python.exe -X utf8 exp101_stopcauses.py --stage build
#       C:\miniconda3\python.exe -X utf8 exp101_stopcauses.py --stage A  (A~H)
# ═══════════════════════════════════════════════════════
import argparse
import sys

import numpy as np
import pandas as pd

from exp005_qopt import TRAIN_DIR
from exp015_scada_pc import GROUP_TURBINES
from exp035_raw_dl import CACHE_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SLOT = CACHE_DIR / "exp105_slots.parquet"


def build():
    """터빈 슬롯 × (power, ws, wd) + 시간별 기상(T, RH, 예보ws) 병합."""
    raw = {m: pd.read_csv(TRAIN_DIR / f"scada_{m}_train.csv", encoding="utf-8-sig",
                          parse_dates=["kst_dtm"]) for m in ("vestas", "unison")}
    longs = []
    for g, (maker, turbs, _) in GROUP_TURBINES.items():
        df = raw[maker]
        for t in turbs:
            s = df[["kst_dtm", f"{t}_power_kw10m", f"{t}_ws", f"{t}_wd"]].copy()
            s.columns = ["dtm", "power", "ws", "wd"]
            s["turbine"], s["group"], s["maker"] = t, g, maker
            longs.append(s)
    a = pd.concat(longs, ignore_index=True)
    a["hour"] = a["dtm"].dt.ceil("h")

    import warnings
    warnings.filterwarnings("ignore")
    from exp065_gbm3 import build_train_frame
    tf, _ = build_train_frame()
    w = tf.loc[tf["kpx_group"] == 1, ["forecast_kst_dtm", "ldaps_heightAboveGround_2_t",
                                      "ldaps_heightAboveGround_2_r", "ldaps_ws_10m"]].copy()
    w.columns = ["hour", "T", "RH", "fws"]
    if w["T"].median() > 200:
        w["T"] = w["T"] - 273.15
    a = a.merge(w, on="hour", how="left")
    a["stop"] = a["power"] <= 1.0
    a.to_parquet(SLOT, index=False)
    print(f"저장 → {SLOT.name} | {len(a):,}슬롯 | 기간 {a.dtm.min().date()} ~ {a.dtm.max().date()}",
          flush=True)
    print(f"  결측: power {a.power.isna().mean():.3f} ws {a.ws.isna().mean():.3f} "
          f"wd {a.wd.isna().mean():.3f} T {a['T'].isna().mean():.3f}", flush=True)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build")
    args = ap.parse_args()
    if args.stage == "build":
        build()
        return
    a = pd.read_parquet(SLOT)
    a = a[~((a.group == 3) & (a.dtm.dt.year == 2022))]      # g3 2022 SCADA 없음
    a = a.dropna(subset=["power", "ws"])
    ok = a[~a.stop]

    print("═══ A. 고풍속 컷아웃 ═══", flush=True)
    print(f"나셀 ws 분위수: " + " ".join(
        f"p{int(q*100)}={a.ws.quantile(q):.1f}" for q in (.5, .9, .99, .999, 1.0)), flush=True)
    hb = pd.cut(a.ws, [0, 3, 12, 18, 20, 22, 25, 99],
                labels=["<3", "3-12", "12-18", "18-20", "20-22", "22-25", "25+"])
    r = a.groupby([hb, "maker"], observed=False).agg(
        n=("stop", "size"), 정지율=("stop", lambda s: s.mean() * 100),
        평균출력=("power", "mean")).round(1)
    print(r.to_string(), flush=True)

    print("\n═══ B. 저온 정지 ═══", flush=True)
    tb = pd.cut(a["T"], [-99, -20, -15, -10, -5, 0, 99],
                labels=["<-20", "-20~-15", "-15~-10", "-10~-5", "-5~0", "0+"])
    m = a.ws >= 5                                            # 바람은 충분한데 서 있나
    print(a[m].groupby([tb, "maker"], observed=False).agg(
        n=("stop", "size"), 정지율=("stop", lambda s: s.mean() * 100)).round(1).to_string(), flush=True)

    print("\n═══ C. 착빙 시 겉보기 과다생산 (풍속계 결빙 서명) ═══", flush=True)
    o = ok[(ok.ws >= 5) & (ok.ws < 14)].copy()
    o["wsb"] = (o.ws * 2).round() / 2
    ref = o.groupby(["turbine", "wsb"]).power.quantile(0.85).rename("ref").reset_index()
    o = o.merge(ref, on=["turbine", "wsb"]); o = o[o.ref > 50]
    o["r"] = o.power / o.ref
    o["env"] = np.where(o["T"] < 2, np.where(o["RH"] >= 90, "저온+고습", "저온+건조"), "온난")
    print(o.groupby(["maker", "env"], observed=False).agg(
        n=("r", "size"), r중앙=("r", "median"), r_1p05초과=("r", lambda s: (s > 1.05).mean() * 100),
        r_0p6미만=("r", lambda s: (s < 0.6).mean() * 100),
        나셀ws평균=("ws", "mean"), 예보ws평균=("fws", "mean")).round(3).to_string(), flush=True)

    print("\n═══ D. 풍향 섹터별 정지율 (ws 5~15 한정) ═══", flush=True)
    d = a[(a.ws >= 5) & (a.ws < 15) & a.wd.notna()].copy()
    d["sec"] = (d.wd // 45 % 8).astype(int)
    piv = d.pivot_table(index="sec", columns="group", values="stop",
                        aggfunc=lambda s: s.mean() * 100).round(1)
    piv.index = [f"{s*45}-{s*45+45}°" for s in piv.index]
    print(piv.to_string(), flush=True)
    print("  터빈별 최대-최소 섹터 정지율 격차 (그룹3):", flush=True)
    for t in sorted(d[d.group == 3].turbine.unique()):
        v = d[d.turbine == t].groupby("sec").stop.mean() * 100
        print(f"    {t}: {v.min():.1f}% ~ {v.max():.1f}%  (최대 섹터 {int(v.idxmax())*45}°)", flush=True)

    print("\n═══ E. 야간 저풍속 정지 — 박쥐 저감 서명 ═══", flush=True)
    a2 = a.copy(); a2["h"] = a2.dtm.dt.hour; a2["mo"] = a2.dtm.dt.month
    a2["night"] = (a2.h >= 20) | (a2.h < 5)
    lo = a2[(a2.ws >= 3.0) & (a2.ws < 5.0)]                  # cut-in 근방
    print(lo.pivot_table(index="mo", columns=["group", "night"], values="stop",
                         aggfunc=lambda s: s.mean() * 100).round(1).to_string(), flush=True)

    print("\n═══ F. 일출·일몰 정지 (그림자 깜빡임) — 시각별 정지율, ws 5+ ═══", flush=True)
    hi = a2[a2.ws >= 5]
    print(hi.pivot_table(index="h", columns="group", values="stop",
                         aggfunc=lambda s: s.mean() * 100).round(1).T.to_string(), flush=True)

    print("\n═══ G. 정지 지속시간 분포 ═══", flush=True)
    a3 = a.sort_values(["turbine", "dtm"]).copy()
    a3["blk"] = (a3.stop != a3.groupby("turbine").stop.shift()).cumsum()
    runs = a3[a3.stop].groupby(["turbine", "blk"]).size().mul(10).rename("min").reset_index()
    tot = runs["min"].sum()
    for lo_, hi_, lab in [(0, 60, "<1시간"), (60, 360, "1~6시간"), (360, 1440, "6~24시간"),
                          (1440, 10080, "1~7일"), (10080, 10**9, "7일+")]:
        s = runs[(runs["min"] >= lo_) & (runs["min"] < hi_)]
        print(f"  {lab:<8} 건수 {len(s):>5} ({len(s)/len(runs)*100:>5.1f}%) | "
              f"총시간 {s['min'].sum()/60:>8.0f}h ({s['min'].sum()/tot*100:>5.1f}%)", flush=True)

    print("\n═══ H. 전 터빈 동시 정지 (계통·curtailment 서명) ═══", flush=True)
    g = a.groupby(["group", "hour"]).agg(stop_frac=("stop", "mean"), ws=("ws", "mean")).reset_index()
    allstop = g[g.stop_frac > 0.95]
    print(f"  그룹-시간 중 95%+ 정지: {len(allstop):,} / {len(g):,} ({len(allstop)/len(g)*100:.2f}%)",
          flush=True)
    print(f"  그중 나셀 ws ≥ 8 m/s (바람 좋은데 전부 정지): "
          f"{(allstop.ws >= 8).sum():,} ({(allstop.ws >= 8).mean()*100:.1f}%)", flush=True)
    print("  연-월별 상위 10:", flush=True)
    print(allstop.assign(ym=allstop.hour.dt.to_period("M")).groupby(["ym", "group"])
          .size().sort_values(ascending=False).head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
