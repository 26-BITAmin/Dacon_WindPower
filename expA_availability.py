# ═══════════════════════════════════════════════════════
# expA: 연도별 가동률 실측 — "연도 간 가동률 변동이 train 데이터 안에서 실재하는가"
#
# 배경 (2026-07-30, 사용자 요청): 2025 하향(s*=0.98)의 근거가 public score뿐이라는
#   지적. 데이터 자체에서 근거를 찾는 1단계 = train(2022~2024) 내부에서
#   연도별 가동률·동일풍속 출력이 실제로 변동했는지 직접 측정.
#
# 정의 (exp-049/055에서 검증된 것 재사용):
#   - dead_strict: 터빈 풍속(결측 시 그룹 중앙값) ≥ 5 m/s 인데 슬롯 출력 ≤ 1 kWh 또는 결측
#   - dead_lenient: 명시적 0/1 이하만 (결측 제외)
#   - 가동률 = 1 − (windy 슬롯 중 dead 비율)
#   - 매칭 풍속 출력비: 그룹 풀링 파워커브(0.5 m/s 빈 중앙값 CF, 전 연도)로
#     기대 CF를 만들고, 연도별 실측 CF 합 / 기대 CF 합. dead 포함(ratio_all)과
#     dead 제외(ratio_alive)를 나눠 가동률 효과를 분리.
#
# 실행: C:\miniconda3\python.exe -X utf8 expA_availability.py
# ═══════════════════════════════════════════════════════
import sys

import numpy as np
import pandas as pd

from exp049_diag import load_scada, GROUP_TURBINES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RATED_SLOT = {"vestas": 600.0, "unison": 700.0}   # kWh/10분 정격
GLITCH_CAP = {"vestas": 660.0, "unison": 770.0}   # 정격×1.1 초과 = 글리치 NaN
WS_BINS = np.arange(0.0, 30.5, 0.5)


def group_slots(sc, g):
    """그룹 g의 터빈×슬롯 long frame: year, month, ws_eff, cf, dead flags"""
    turbines = GROUP_TURBINES[g]
    w_cols = [f"{mk}_wtg{i:02d}_ws" for mk, i in turbines]
    med_ws = sc[w_cols].median(axis=1)
    frames = []
    for mk, i in turbines:
        pc, wc = f"{mk}_wtg{i:02d}_power_kw10m", f"{mk}_wtg{i:02d}_ws"
        p = sc[pc].astype(float).clip(lower=0)
        p = p.where(p <= GLITCH_CAP[mk], np.nan)
        ws_eff = sc[wc].fillna(med_ws)
        frames.append(pd.DataFrame({
            "year": sc["kst_dtm"].dt.year,
            "month": sc["kst_dtm"].dt.month,
            "tb": f"{mk}{i:02d}",
            "ws": ws_eff,
            "cf": p / RATED_SLOT[mk],
            "p_na": p.isna(),
            "p_low": (p <= 1.0).fillna(False),
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["ws"].notna()]
    df["windy"] = df["ws"] >= 5.0
    df["dead_strict"] = (df["p_na"] | df["p_low"]) & df["windy"]
    df["dead_lenient"] = df["p_low"] & df["windy"]
    return df


def year_table(df):
    """연도별 가동률 + 매칭 풍속 출력비"""
    valid = df["cf"].notna()
    bidx = np.digitize(df["ws"], WS_BINS)
    df = df.assign(bidx=bidx)
    curve = df.loc[valid].groupby("bidx")["cf"].median()
    df["exp_cf"] = df["bidx"].map(curve)

    rows = []
    for y, sub in df.groupby("year"):
        windy = sub["windy"]
        v = sub["cf"].notna()
        alive = v & ~sub["dead_strict"]
        rows.append({
            "year": y,
            "slots": len(sub),
            "windy_slots": int(windy.sum()),
            "ws_mean": sub["ws"].mean(),
            "avail_strict": 1 - sub.loc[windy, "dead_strict"].mean() if windy.any() else np.nan,
            "avail_lenient": 1 - sub.loc[windy, "dead_lenient"].mean() if windy.any() else np.nan,
            "ratio_all": sub.loc[v, "cf"].sum() / sub.loc[v, "exp_cf"].sum(),
            "ratio_alive": sub.loc[alive, "cf"].sum() / sub.loc[alive, "exp_cf"].sum(),
        })
    return pd.DataFrame(rows).set_index("year")


def monthly_dead(df):
    """(연,월)별 dead_strict 비율 — 하락이 특정 기간에 몰렸는지"""
    windy = df[df["windy"]]
    return windy.groupby(["year", "month"])["dead_strict"].mean().unstack("month")


def main():
    sc = load_scada()
    print(f"SCADA 기간: {sc['kst_dtm'].min()} ~ {sc['kst_dtm'].max()}  ({len(sc):,} 슬롯행)")
    for g in [1, 2, 3]:
        df = group_slots(sc, g)
        print(f"\n══════ 그룹 {g} ({df['tb'].nunique()}기) ══════")
        tab = year_table(df)
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(tab.to_string())
        print("-- (연,월)별 dead_strict 비율 --")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(monthly_dead(df).to_string())


if __name__ == "__main__":
    main()
