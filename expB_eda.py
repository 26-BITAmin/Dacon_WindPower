# ═══════════════════════════════════════════════════════
# expB: 주최측 제공 데이터 전수 EDA (읽기 전용, 모델 학습 0)
#
# 목적: 라벨·SCADA·예보(LDAPS/GFS)의 결측/평균/분위수/분산/min/max를
#   year · quarter · season · month · hour 축으로 정리하고,
#   AVAILABILITY.md의 가동률 정의를 같은 축에 병치한다.
#
# 시각 규약 (본 스크립트에서 실측 검증):
#   라벨 kst_dtm = 집계 구간 **종료** 시각 → SCADA 슬롯 00:10~01:00 이 라벨 01:00 에 대응.
#   (2023-06-01 01:00 검증: SCADA 6슬롯×6터빈 합 7,417 kWh vs 라벨 7,434 kWh)
#
# 가동률 정의: AVAILABILITY.md / expA_availability.py 와 동일 (exp-049에서 확립)
#   windy = 터빈 풍속(결측 시 그룹 중앙값) ≥ 5 m/s
#   dead_strict  = windy 인데 슬롯 출력 ≤ 1 kWh 또는 결측(글리치 NaN 포함)
#   dead_lenient = windy 인데 명시적으로 ≤ 1 kWh (결측 제외)
#   가동률 = 1 − (windy 터빈-슬롯 중 dead 비율)
#
# 실행: C:\miniconda3\python.exe -X utf8 expB_eda.py > expB_log.txt
#   (표준출력 = 리포트 본문, 동시에 EDA.md 로 저장)
# ═══════════════════════════════════════════════════════
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "open (1)"
OUT_MD = Path(__file__).resolve().parent / "EDA.md"

CAP = {1: 21600.0, 2: 21600.0, 3: 21000.0}          # 1시간 환산 설비용량 kWh
GROUP_TURBINES = {
    1: [("vestas", i) for i in range(1, 7)],
    2: [("vestas", i) for i in range(7, 13)],
    3: [("unison", i) for i in range(1, 6)],
}
RATED_SLOT = {"vestas": 600.0, "unison": 700.0}      # kWh / 10분 정격
GLITCH_CAP = {"vestas": 660.0, "unison": 770.0}      # 정격×1.1 초과 = 글리치

SEASON = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
          6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
QS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

_BUF = []


def w(s=""):
    _BUF.append(str(s))
    print(s)


def md(df, idx_name="", fmt="{:.4g}"):
    """DataFrame → 마크다운 표 문자열 (float 포맷 통일)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([idx_name] + cols) + " |",
             "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
    for i, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else fmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            else:
                cells.append("—" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join(lines)


def stats(s):
    """단일 시리즈 기술통계 dict (결측·평균·분산·분위수·min·max)."""
    s = pd.Series(s, dtype="float64")
    n = len(s)
    nna = int(s.isna().sum())
    v = s.dropna()
    d = {"n": n, "결측": nna, "결측%": 100 * nna / n if n else np.nan,
         "mean": v.mean(), "std": v.std(), "var": v.var(),
         "min": v.min()}
    for q in QS:
        d[f"q{int(q * 100):02d}"] = v.quantile(q) if len(v) else np.nan
    d["max"] = v.max() if len(v) else np.nan
    return d


def by_axis(df, axis, col, order=None):
    """축(axis)별 col 의 기술통계 표."""
    rows = {}
    keys = order if order is not None else sorted(df[axis].dropna().unique())
    for k in keys:
        sub = df.loc[df[axis] == k, col]
        if len(sub):
            rows[k] = stats(sub)
    return pd.DataFrame(rows).T


# ───────────────────────── 로더 ─────────────────────────
def add_calendar(df, tcol, interval_end=True):
    """달력 피처.

    `interval_end=True` 면 해당 시각이 집계구간의 **종료** 시각이라는 뜻이므로
    구간이 실제로 속한 시점(= t − 1시간)에서 연/월/분기/계절/시를 뽑는다.
    이렇게 하지 않으면 라벨 `2025-01-01 00:00`(실제 구간 2024-12-31 23~24시)이
    2025년으로 분류돼 연도 표에 n=1 짜리 유령 행이 생긴다.
    """
    t = df[tcol] - pd.Timedelta(hours=1) if interval_end else df[tcol]
    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["quarter"] = "Q" + t.dt.quarter.astype(str)
    df["season"] = t.dt.month.map(SEASON)
    df["hour"] = t.dt.hour
    return df


def load_labels():
    """라벨 long frame. **그룹별 유효기간(support) 밖 구간은 아예 제외**한다.

    group_3 은 2022년 라벨이 '결측'인 게 아니라 주최측이 제공하지 않은 구간이다
    (data_description.md §5: '2022년 구간은 빈칸'). 이를 결측으로 세면 group_3 의
    결측률이 33.3% 로 잡혀 다른 그룹과 비교가 불가능해진다. 따라서 각 그룹의
    support = [첫 유효 라벨 시각, 마지막 유효 라벨 시각] 로 잘라내고,
    그 **안에서만** 결측을 센다.
    """
    lab = pd.read_csv(DATA_DIR / "train" / "train_labels.csv",
                      encoding="utf-8-sig", parse_dates=["kst_dtm"])
    long = lab.melt(id_vars="kst_dtm", var_name="grp", value_name="kwh")
    long["group"] = long["grp"].str.replace("kpx_group_", "").astype(int)

    support, keep = {}, []
    for g, sub in long.groupby("group"):
        ok = sub.loc[sub["kwh"].notna(), "kst_dtm"]
        support[g] = (ok.min(), ok.max())
        keep.append(sub[(sub["kst_dtm"] >= ok.min()) & (sub["kst_dtm"] <= ok.max())])
    long = pd.concat(keep, ignore_index=True)

    long["cap"] = long["group"].map(CAP)
    long["cf"] = long["kwh"] / long["cap"]
    long["eval"] = long["kwh"] >= 0.10 * long["cap"]
    return lab, add_calendar(long.drop(columns="grp"), "kst_dtm"), support


def load_scada():
    """SCADA 병합 프레임 + **제작사별 실제 커버리지**.

    UNISON 파일은 2023년부터라, VESTAS 시간축에 outer merge 하면 2022년 행이
    전부 NaN 으로 생긴다. 이는 결측이 아니라 미제공 구간이므로 커버리지를 함께
    돌려주고 group_slots 에서 잘라낸다.
    """
    v = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv",
                    encoding="utf-8-sig", parse_dates=["kst_dtm"])
    u = pd.read_csv(DATA_DIR / "train" / "scada_unison_train.csv",
                    encoding="utf-8-sig", parse_dates=["kst_dtm"])
    cov = {"vestas": (v["kst_dtm"].min(), v["kst_dtm"].max()),
           "unison": (u["kst_dtm"].min(), u["kst_dtm"].max())}
    merged = v.merge(u, on="kst_dtm", how="outer").sort_values("kst_dtm").reset_index(drop=True)
    return merged, cov


def group_slots(sc, g, cov):
    """그룹 g 의 터빈×슬롯 long frame (expA_availability.py 와 동일 정의).

    해당 제작사 파일이 실제로 덮는 기간으로 먼저 자른다 — group_3(UNISON)의
    2022년은 merge 로 생긴 빈 행이지 결측이 아니다.
    """
    turbines = GROUP_TURBINES[g]
    mk0 = turbines[0][0]
    lo, hi = cov[mk0]
    sc = sc[(sc["kst_dtm"] >= lo) & (sc["kst_dtm"] <= hi)].reset_index(drop=True)
    w_cols = [f"{mk}_wtg{i:02d}_ws" for mk, i in turbines]
    med_ws = sc[w_cols].median(axis=1)
    # 라벨 시각 규약: 슬롯 00:10~01:00 → 라벨시각 01:00
    lab_hour = (sc["kst_dtm"] - pd.Timedelta(minutes=1)).dt.ceil("h")
    frames = []
    for mk, i in turbines:
        pc, wc = f"{mk}_wtg{i:02d}_power_kw10m", f"{mk}_wtg{i:02d}_ws"
        raw = sc[pc].astype(float)
        p = raw.clip(lower=0)
        p = p.where(p <= GLITCH_CAP[mk], np.nan)
        ws_eff = sc[wc].fillna(med_ws)
        frames.append(pd.DataFrame({
            "lab_hour": lab_hour,
            "tb": f"{mk}{i:02d}",
            "ws": ws_eff,
            "ws_raw_na": sc[wc].isna(),
            "kwh": p,
            "kwh_raw": raw,
            "glitch": raw > GLITCH_CAP[mk],
            "raw_na": raw.isna(),
            "neg": raw < 0,
            "p_na": p.isna(),
            "p_low": (p <= 1.0).fillna(False),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["windy"] = (df["ws"] >= 5.0).fillna(False)
    df["dead_strict"] = (df["p_na"] | df["p_low"]) & df["windy"]
    df["dead_lenient"] = df["p_low"] & df["windy"]
    return add_calendar(df, "lab_hour")


def avail_by(df, axis, order=None):
    """축별 가동률 표 (windy 슬롯 기준)."""
    keys = order if order is not None else sorted(df[axis].dropna().unique())
    rows = {}
    for k in keys:
        sub = df[df[axis] == k]
        wsub = sub[sub["windy"]]
        if not len(wsub):
            continue
        rows[k] = {
            "슬롯": len(sub),
            "windy슬롯": len(wsub),
            "windy%": 100 * len(wsub) / len(sub),
            "ws_mean": sub["ws"].mean(),
            "가동률_strict": 1 - wsub["dead_strict"].mean(),
            "가동률_lenient": 1 - wsub["dead_lenient"].mean(),
            "평균CF_windy": (wsub["kwh"] / RATED_SLOT[
                "unison" if wsub["tb"].iloc[0].startswith("u") else "vestas"]).mean(),
        }
    return pd.DataFrame(rows).T


def hourly_avail(df):
    """라벨 시각별 가동률 (windy 슬롯 없는 시간은 NaN)."""
    wsub = df[df["windy"]]
    g = wsub.groupby("lab_hour")["dead_strict"].agg(["mean", "size"])
    out = (1 - g["mean"]).rename("avail")
    return out.where(g["size"] >= 6)      # 최소 6 터빈-슬롯 있는 시간만


def load_fcst(kind, split):
    f = DATA_DIR / split / f"{kind}_{split}.csv"
    df = pd.read_csv(f, encoding="utf-8-sig",
                     parse_dates=["forecast_kst_dtm", "data_available_kst_dtm"])
    if kind == "ldaps":
        df["ws10"] = np.hypot(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        u50 = (df["heightAboveGround_50_50MUmax"] + df["heightAboveGround_50_50MUmin"]) / 2
        v50 = (df["heightAboveGround_50_50MVmax"] + df["heightAboveGround_50_50MVmin"]) / 2
        df["ws50"] = np.hypot(u50, v50)
        df["wsbl"] = np.hypot(df["heightAboveGround_5_XBLWS"], df["heightAboveGround_5_YBLWS"])
    else:
        df["ws10"] = np.hypot(df["heightAboveGround_10_10u"], df["heightAboveGround_10_10v"])
        df["ws80"] = np.hypot(df["heightAboveGround_80_u"], df["heightAboveGround_80_v"])
        df["ws100"] = np.hypot(df["heightAboveGround_100_100u"], df["heightAboveGround_100_100v"])
        df["wspbl"] = np.hypot(df["planetaryBoundaryLayer_0_u"], df["planetaryBoundaryLayer_0_v"])
    return add_calendar(df, "forecast_kst_dtm")


# ───────────────────────── 섹션 ─────────────────────────
def sec1_inventory(lab, sc, fc):
    w("## 1. 파일 인벤토리 · 시간축 무결성\n")
    rows = {}
    for name, df, tcol in [
        ("train_labels", lab, "kst_dtm"),
        ("scada(vestas+unison merged)", sc, "kst_dtm"),
        ("ldaps_train", fc[("ldaps", "train")], "forecast_kst_dtm"),
        ("gfs_train", fc[("gfs", "train")], "forecast_kst_dtm"),
        ("ldaps_test", fc[("ldaps", "test")], "forecast_kst_dtm"),
        ("gfs_test", fc[("gfs", "test")], "forecast_kst_dtm"),
    ]:
        t = df[tcol]
        uniq = t.nunique()
        rows[name] = {
            "행": len(df), "열": df.shape[1], "고유시각": uniq,
            "시각당행": len(df) / uniq,
            "시작": str(t.min()), "종료": str(t.max()),
            "중복행": int(df.duplicated().sum()),
        }
    w(md(pd.DataFrame(rows).T, "파일"))
    w()

    # 시간 격자 결손
    w("**시간 격자 결손 점검** (기대 시각 대비 실제 존재 시각)\n")
    chk = {}
    for name, df, tcol, freq in [
        ("train_labels", lab, "kst_dtm", "h"),
        ("scada", sc, "kst_dtm", "10min"),
        ("ldaps_train", fc[("ldaps", "train")], "forecast_kst_dtm", "h"),
        ("gfs_train", fc[("gfs", "train")], "forecast_kst_dtm", "h"),
        ("ldaps_test", fc[("ldaps", "test")], "forecast_kst_dtm", "h"),
        ("gfs_test", fc[("gfs", "test")], "forecast_kst_dtm", "h"),
    ]:
        t = pd.DatetimeIndex(df[tcol].unique()).sort_values()
        full = pd.date_range(t.min(), t.max(), freq=freq)
        chk[name] = {"기대시각": len(full), "실제시각": len(t),
                     "결손시각": len(full) - len(t),
                     "결손%": 100 * (len(full) - len(t)) / len(full)}
    w(md(pd.DataFrame(chk).T, "파일"))
    w()

    # 예보 리드타임
    w("**예보 리드타임** (`forecast_kst_dtm` − `data_available_kst_dtm`, 시간)\n")
    lt = {}
    for (k, s), df in fc.items():
        h = (df["forecast_kst_dtm"] - df["data_available_kst_dtm"]).dt.total_seconds() / 3600
        lt[f"{k}_{s}"] = {"min": h.min(), "q25": h.quantile(.25), "median": h.median(),
                          "q75": h.quantile(.75), "max": h.max(), "고유값수": h.nunique()}
    w(md(pd.DataFrame(lt).T, "파일"))
    w()


def sec2_labels(lab, long):
    w("## 2. 라벨(실제 발전량) 분포\n")
    w("### 2-1. 그룹 전체 기술통계 (kWh)\n")
    t = pd.DataFrame({f"group_{g}": stats(long.loc[long["group"] == g, "kwh"])
                      for g in [1, 2, 3]}).T
    t["설비용량"] = [CAP[g] for g in [1, 2, 3]]
    w(md(t, "그룹", "{:,.1f}"))
    w()
    w("### 2-2. 그룹 전체 기술통계 (CF = kWh / 설비용량)\n")
    w(md(pd.DataFrame({f"group_{g}": stats(long.loc[long["group"] == g, "cf"])
                       for g in [1, 2, 3]}).T, "그룹"))
    w()

    w("### 2-3. 결측·특이값 실태\n")
    rows = {}
    for g in [1, 2, 3]:
        s = long[long["group"] == g]
        v = s["kwh"]
        obs = v.dropna()          # 0% 등의 분모는 라벨이 존재하는 시간만
        rows[f"group_{g}"] = {
            "유효기간 시간": len(s), "결측": int(v.isna().sum()),
            "결측%": 100 * v.isna().mean(),
            "라벨有시간": len(obs),
            "정확히0": int((obs == 0).sum()), "0%(라벨有기준)": 100 * (obs == 0).mean(),
            "음수": int((v < 0).sum()),
            "용량초과": int((v > CAP[g]).sum()),
            "≥10%cap(평가대상)": int(s["eval"].sum()),
            "평가대상%(라벨有기준)": 100 * s["eval"].sum() / len(obs),
        }
    w(md(pd.DataFrame(rows).T, "그룹", "{:,.2f}"))
    w()
    w("**결측 구간 상세**\n")
    for g in [1, 2, 3]:
        s = long[long["group"] == g].sort_values("kst_dtm")
        na = s["kwh"].isna()
        if na.any():
            blocks = (na != na.shift()).cumsum()[na]
            seg = s.loc[na].groupby(blocks)["kst_dtm"].agg(["min", "max", "size"])
            w(f"- group_{g}: 결측 연속구간 {len(seg)}개 — "
              + "; ".join(f"{a} ~ {b} ({c:,}h)" for a, b, c in
                          seg.head(5).itertuples(index=False)))
        else:
            w(f"- group_{g}: 결측 없음")
    w()

    for axis, order, title in [("year", None, "연도"), ("quarter", None, "분기"),
                               ("season", SEASON_ORDER, "계절"), ("month", None, "월")]:
        w(f"### 2-4-{title}. {title}별 발전량 (kWh)\n")
        for g in [1, 2, 3]:
            sub = long[long["group"] == g]
            tb = by_axis(sub, axis, "kwh", order)
            if not len(tb):
                continue
            ev = sub.groupby(axis)["eval"].mean().reindex(tb.index) * 100
            tb["CF"] = tb["mean"] / CAP[g]
            tb["평가대상%"] = ev
            w(f"**group_{g}**\n")
            w(md(tb, title, "{:,.4g}"))
            w()

    w("### 2-5. 시간대(hour)별 발전량 CF\n")
    piv = long.pivot_table(index="hour", columns="group", values="cf", aggfunc="mean")
    piv.columns = [f"g{c}_CF평균" for c in piv.columns]
    piv2 = long.pivot_table(index="hour", columns="group", values="cf", aggfunc="std")
    for c in piv2.columns:
        piv[f"g{c}_CF표준편차"] = piv2[c]
    w(md(piv, "hour"))
    w()

    w("### 2-6. 그룹 간 상관 (동시간대 kWh)\n")
    wide = lab.set_index("kst_dtm")[["kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    w(md(wide.corr(), "corr"))
    w()


def sec3_avail(slots):
    w("## 3. 가동률 (SCADA 기반, AVAILABILITY.md 정의)\n")
    w("> windy = 터빈풍속(결측 시 그룹 중앙값) ≥ 5 m/s · "
      "dead_strict = windy인데 슬롯출력 ≤ 1 kWh 또는 결측 · "
      "가동률 = 1 − dead 비율. 판정 단위 = 터빈 1기 × 10분 슬롯 1개.\n")

    w("### 3-1. SCADA 원자료 품질 (글리치·결측·음수)\n")
    rows = {}
    for g, df in slots.items():
        rows[f"group_{g}"] = {
            "터빈수": df["tb"].nunique(), "터빈×슬롯": len(df),
            "power결측(원자료)": int(df["raw_na"].sum()),
            "power결측%": 100 * df["raw_na"].mean(),
            "글리치(>정격×1.1)": int(df["glitch"].sum()),
            "글리치%": 100 * df["glitch"].mean(),
            "음수": int(df["neg"].sum()),
            "ws결측(원자료)": int(df["ws_raw_na"].sum()),
            "ws결측%": 100 * df["ws_raw_na"].mean(),
        }
    w(md(pd.DataFrame(rows).T, "그룹", "{:,.3f}"))
    w()

    w("**음수 출력의 정체 — 기생소비 vs 글리치 분리**\n")
    rows = {}
    for g, df in slots.items():
        raw = df["kwh_raw"].dropna()
        neg = raw[raw < 0]
        cap = GLITCH_CAP["unison" if g == 3 else "vestas"]
        rows[f"group_{g}"] = {
            "음수셀": len(neg), "음수%": 100 * len(neg) / len(raw) if len(raw) else np.nan,
            "음수 중앙(kWh)": neg.median() if len(neg) else np.nan,
            "음수 q01": neg.quantile(.01) if len(neg) else np.nan,
            f"양수글리치(>{cap:.0f})": int((raw > cap).sum()),
            f"음수글리치(<-{cap:.0f})": int((raw < -cap).sum()),
            "최댓값": raw.max(), "최솟값": raw.min(),
        }
    w(md(pd.DataFrame(rows).T, "그룹", "{:,.4g}"))
    w("\n> 음수의 절대다수는 −1~−7 kWh = **무발전 시 터빈 자체 소비(기생소비)** 로 정상. "
      "그와 별개로 절댓값 5천만 kWh 급의 **부호 대칭 글리치**가 양·음 같은 개수로 존재한다. "
      "현행 정제 규칙(`clip(lower=0)` → 정격×1.1 초과 NaN)은 양수 글리치만 NaN 처리하고 "
      "음수 글리치는 0 으로 눌러 **dead 후보로 편입**시킨다. "
      "규모가 전체 셀의 0.03% 미만이라 §3-3 이하 수치에는 영향이 없으나, 알려진 아티팩트로 남긴다.\n")
    ua = slots[3]["kwh_raw"].dropna().values.astype(float)
    w(f"> UNISON(group_3)은 음수가 **0개**로 기록 규약 자체가 다르다. 대신 정격(700 kWh/슬롯) "
      f"초과가 {int((ua > 700).sum()):,}셀 있고 그중 {int((ua > 770).sum()):,}셀이 글리치 문턱을 넘으며 "
      f"**최댓값이 정확히 {ua.max():,.0f}** 이다. q99 는 {np.percentile(ua, 99):.0f} 로 대부분은 정격 부근에 "
      "붙어 있고 700→1000 구간만 얇은 꼬리를 이룬다. 이 꼬리의 성격은 **미논의**.\n")

    w("### 3-2. 터빈 풍속 기술통계 (m/s)\n")
    w(md(pd.DataFrame({f"group_{g}": stats(df["ws"]) for g, df in slots.items()}).T, "그룹"))
    w()

    for axis, order, title in [("year", None, "연도"), ("quarter", None, "분기"),
                               ("season", SEASON_ORDER, "계절"), ("month", None, "월"),
                               ("hour", None, "시간대")]:
        w(f"### 3-3-{title}. {title}별 가동률\n")
        for g, df in slots.items():
            tb = avail_by(df, axis, order)
            if not len(tb):
                continue
            w(f"**group_{g}**\n")
            w(md(tb, title, "{:,.4g}"))
            w()

    w("### 3-3b. 계절 효과의 강건성 — 2024-01/02 광역 에피소드를 빼도 겨울이 최저인가\n")
    rows = {}
    for g, df in slots.items():
        wd = df[df["windy"]]
        ex = wd[~((wd["year"] == 2024) & (wd["month"].isin([1, 2])))]
        a = 1 - wd.groupby("season")["dead_strict"].mean()
        b = 1 - ex.groupby("season")["dead_strict"].mean()
        for s in SEASON_ORDER:
            rows.setdefault(s, {})[f"g{g} 전체"] = a.get(s, np.nan)
            rows.setdefault(s, {})[f"g{g} 제외"] = b.get(s, np.nan)
    w(md(pd.DataFrame(rows).T.loc[SEASON_ORDER], "계절", "{:.3f}"))
    w("\n> 2024-01/02 를 빼면 겨울 가동률이 g1 0.881→0.930 · g2 0.884→0.937 · g3 0.813→0.848 로 "
      "올라오지만, **그래도 세 그룹 모두 겨울이 여전히 최저 계절**이다. "
      "즉 겨울 저가동은 2024 에피소드만의 산물이 아니라 상시 계절 패턴이고, "
      "에피소드가 그 위에 3.5~5pp 를 더 얹은 구조다. 원인은 **미논의**.\n")

    w("### 3-4. (연 × 월) 가동률 히트맵 — 정지의 에피소드 구조\n")
    for g, df in slots.items():
        wsub = df[df["windy"]]
        piv = 1 - wsub.pivot_table(index="year", columns="month",
                                   values="dead_strict", aggfunc="mean")
        w(f"**group_{g}** (값 = 가동률, 낮을수록 정지 많음)\n")
        w(md(piv, "연도", "{:.3f}"))
        w()

    w("### 3-5. (연 × 분기) 가동률\n")
    for g, df in slots.items():
        wsub = df[df["windy"]]
        piv = 1 - wsub.pivot_table(index="year", columns="quarter",
                                   values="dead_strict", aggfunc="mean")
        w(f"**group_{g}**\n")
        w(md(piv, "연도", "{:.3f}"))
        w()


def sec4_cross(long, slots):
    w("## 4. 라벨 × 가동률 결합 (시간 단위)\n")
    av = pd.concat([hourly_avail(df).rename(g) for g, df in slots.items()], axis=1)
    av = av.stack(dropna=True).rename("avail").reset_index()
    av.columns = ["kst_dtm", "group", "avail"]
    mg = long.merge(av, on=["kst_dtm", "group"], how="inner").dropna(subset=["kwh", "avail"])

    w("### 4-1. 가동률과 발전량 CF 의 관계\n")
    rows = {}
    for g in [1, 2, 3]:
        s = mg[mg["group"] == g]
        rows[f"group_{g}"] = {
            "매칭시간": len(s),
            "corr(가동률, CF)": s["avail"].corr(s["cf"]),
            "가동률 평균": s["avail"].mean(),
            "가동률=1.0 비율%": 100 * (s["avail"] >= 0.999).mean(),
            "가동률<0.5 비율%": 100 * (s["avail"] < 0.5).mean(),
        }
    w(md(pd.DataFrame(rows).T, "그룹"))
    w()

    w("### 4-2. 가동률 구간별 라벨 CF 와 평가대상 비율\n")
    bins = [-.01, .25, .5, .75, .9, .999, 1.01]
    names = ["0~25%", "25~50%", "50~75%", "75~90%", "90~99.9%", "만가동"]
    mg["av_bin"] = pd.cut(mg["avail"], bins, labels=names)
    for g in [1, 2, 3]:
        s = mg[mg["group"] == g]
        tb = s.groupby("av_bin", observed=True).agg(
            시간=("cf", "size"), CF평균=("cf", "mean"), CF중앙=("cf", "median"),
            CF표준편차=("cf", "std"), 평가대상비율=("eval", "mean"),
            터빈풍속대용=("avail", "size"))
        tb["시간%"] = 100 * tb["시간"] / len(s)
        tb["평가대상비율"] *= 100
        w(f"**group_{g}**\n")
        w(md(tb.drop(columns="터빈풍속대용"), "가동률구간", "{:,.4g}"))
        w()

    w("### 4-3. 연도별 — 라벨 CF · 가동률 · 평가대상 비율 한 표\n")
    for g in [1, 2, 3]:
        s = mg[mg["group"] == g]
        tb = s.groupby("year").agg(시간=("cf", "size"), CF평균=("cf", "mean"),
                                   CF중앙=("cf", "median"), 가동률=("avail", "mean"),
                                   평가대상비율=("eval", "mean"))
        tb["평가대상비율"] *= 100
        w(f"**group_{g}**\n")
        w(md(tb, "연도"))
        w()
    return mg


def sec5_fcst(fc):
    w("## 5. 기상 예보 데이터 (LDAPS / GFS)\n")

    w("### 5-1. 결측 실태 (전 컬럼)\n")
    for kind in ["ldaps", "gfs"]:
        rows = {}
        for split in ["train", "test"]:
            df = fc[(kind, split)]
            num = df.select_dtypes("number")
            na = num.isna().sum()
            rows[f"{split}_결측열수"] = int((na > 0).sum())
            rows[f"{split}_총셀"] = int(num.size)
            rows[f"{split}_총결측"] = int(na.sum())
        w(f"- **{kind}**: " + " · ".join(f"{k}={v:,}" for k, v in rows.items()))
        for split in ["train", "test"]:
            na = fc[(kind, split)].select_dtypes("number").isna().sum()
            bad = na[na > 0]
            if len(bad):
                w(f"  - {split} 결측 컬럼: " + ", ".join(f"`{c}`({n:,})" for c, n in bad.items()))
                df = fc[(kind, split)]
                hit = df[df[list(bad.index)].isna().any(axis=1)]
                per = hit.groupby("forecast_kst_dtm").agg(
                    격자수=("grid_id", "nunique"),
                    결측컬럼수=("grid_id", lambda s: 0))
                per["결측컬럼수"] = hit.groupby("forecast_kst_dtm").apply(
                    lambda d: int(d[list(bad.index)].isna().all(axis=0).sum()))
                w(f"  - {split} 결측 발생 시각 ({len(per)}개, 해당 시각은 **전 격자 통째로** 빠짐):")
                for t, r in per.iterrows():
                    w(f"    - `{t}` — 격자 {int(r['격자수'])}개 전부, 결측 컬럼 {int(r['결측컬럼수'])}개")
    w()

    w("### 5-2. 격자 정보\n")
    for kind in ["ldaps", "gfs"]:
        df = fc[(kind, "train")]
        te = fc[(kind, "test")]
        w(f"- **{kind}**: train 격자 {df['grid_id'].nunique()}개 / test {te['grid_id'].nunique()}개, "
          f"위도 {df['latitude'].min():.4f}~{df['latitude'].max():.4f}, "
          f"경도 {df['longitude'].min():.4f}~{df['longitude'].max():.4f}, "
          f"격자 좌표 train≡test: {set(zip(df['latitude'], df['longitude'])) == set(zip(te['latitude'], te['longitude']))}")
    w()

    w("### 5-3. 파생 풍속 기술통계 (전 격자·전 시각 pooled, m/s)\n")
    for kind, cols in [("ldaps", ["ws10", "ws50", "wsbl"]),
                       ("gfs", ["ws10", "ws80", "ws100", "wspbl", "surface_0_gust"])]:
        rows = {}
        for c in cols:
            for split in ["train", "test"]:
                rows[f"{c} [{split}]"] = stats(fc[(kind, split)][c])
        w(f"**{kind}**\n")
        w(md(pd.DataFrame(rows).T, "변수"))
        w()

    w("### 5-4. train(2022~2024) vs test(2025) 분포 시프트 — 전 수치 컬럼\n")
    w("> `d` = (test평균 − train평균) / train표준편차. |d| ≥ 0.10 이면 굵게 표시할 만한 이동.\n")
    for kind in ["ldaps", "gfs"]:
        tr, te = fc[(kind, "train")], fc[(kind, "test")]
        skip = {"year", "month", "hour", "grid_id", "latitude", "longitude"}
        cols = [c for c in tr.select_dtypes("number").columns
                if c in te.columns and c not in skip]
        rows = {}
        for c in cols:
            a, b = tr[c], te[c]
            sd = a.std()
            rows[c] = {"train_mean": a.mean(), "train_std": sd,
                       "test_mean": b.mean(), "test_std": b.std(),
                       "Δmean": b.mean() - a.mean(),
                       "d": (b.mean() - a.mean()) / sd if sd else np.nan}
        tb = pd.DataFrame(rows).T
        tb = tb.reindex(tb["d"].abs().sort_values(ascending=False).index)
        w(f"**{kind}** (|d| 내림차순)\n")
        w(md(tb, "컬럼"))
        w()

    w("### 5-5. 연도별 예보 풍속 (test 2025 포함) — 기상 조건 자체가 달랐는가\n")
    for kind, col in [("ldaps", "ws50"), ("gfs", "ws100")]:
        both = pd.concat([fc[(kind, "train")][["year", "season", "quarter", "month", col]],
                          fc[(kind, "test")][["year", "season", "quarter", "month", col]]])
        both = both[both["year"] <= 2025]
        w(f"**{kind} `{col}` — 연도별**\n")
        w(md(by_axis(both, "year", col), "연도"))
        w()
        w(f"**{kind} `{col}` — 연×계절 평균**\n")
        piv = both.pivot_table(index="year", columns="season", values=col, aggfunc="mean")
        piv = piv[[s for s in SEASON_ORDER if s in piv.columns]]
        w(md(piv, "연도", "{:.3f}"))
        w()
        w(f"**{kind} `{col}` — 연×월 평균**\n")
        w(md(both.pivot_table(index="year", columns="month", values=col, aggfunc="mean"),
             "연도", "{:.3f}"))
        w()

    w("### 5-6. 계절·분기별 예보 풍속 기술통계 (train+test pooled)\n")
    for kind, col in [("ldaps", "ws50"), ("gfs", "ws100")]:
        both = pd.concat([fc[(kind, "train")][["season", "quarter", col]],
                          fc[(kind, "test")][["season", "quarter", col]]])
        w(f"**{kind} `{col}` — 계절별**\n")
        w(md(by_axis(both, "season", col, SEASON_ORDER), "계절"))
        w()
        w(f"**{kind} `{col}` — 분기별**\n")
        w(md(by_axis(both, "quarter", col), "분기"))
        w()


def sec6_link(long, fc):
    w("## 6. 예보 ↔ 라벨 연결 (파워커브·평가지표 관점)\n")
    ld = fc[("ldaps", "train")].groupby("forecast_kst_dtm")["ws50"].mean().rename("ws50")
    gf = fc[("gfs", "train")].groupby("forecast_kst_dtm")["ws100"].mean().rename("ws100")
    site = pd.concat([ld, gf], axis=1)
    mg = long.merge(site, left_on="kst_dtm", right_index=True, how="inner").dropna(subset=["cf"])

    w("### 6-1. 예보 풍속 ↔ 발전량 상관\n")
    rows = {}
    for g in [1, 2, 3]:
        s = mg[mg["group"] == g]
        rows[f"group_{g}"] = {
            "n": len(s),
            "corr(ws50_LDAPS, CF)": s["ws50"].corr(s["cf"]),
            "corr(ws100_GFS, CF)": s["ws100"].corr(s["cf"]),
            "corr(ws50, ws100)": s["ws50"].corr(s["ws100"]),
            "spearman(ws100, CF)": s["ws100"].corr(s["cf"], method="spearman"),
        }
    w(md(pd.DataFrame(rows).T, "그룹"))
    w()

    w("### 6-2. GFS ws100 구간별 실측 CF (경험적 파워커브)\n")
    edges = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 40]
    mg["ws_bin"] = pd.cut(mg["ws100"], edges)
    for g in [1, 2, 3]:
        s = mg[mg["group"] == g]
        tb = s.groupby("ws_bin", observed=True).agg(
            시간=("cf", "size"), CF평균=("cf", "mean"), CF중앙=("cf", "median"),
            CF표준편차=("cf", "std"), CF_q05=("cf", lambda x: x.quantile(.05)),
            CF_q95=("cf", lambda x: x.quantile(.95)), 평가대상비율=("eval", "mean"))
        tb["평가대상비율"] *= 100
        w(f"**group_{g}**\n")
        w(md(tb, "ws100(m/s)", "{:,.4g}"))
        w()

    w("### 6-3. 평가 마스크(실측 ≥ 10%cap) 아래서의 라벨 분포\n")
    rows = {}
    for g in [1, 2, 3]:
        s = long[(long["group"] == g) & long["eval"]]
        allg = long[(long["group"] == g)].dropna(subset=["kwh"])
        rows[f"group_{g}"] = {
            "평가시간": len(s), "전체시간(라벨有)": len(allg),
            "평가시간%": 100 * len(s) / len(allg),
            "평가구간 에너지비중%": 100 * s["kwh"].sum() / allg["kwh"].sum(),
            "평가구간 CF평균": s["cf"].mean(), "평가구간 CF중앙": s["cf"].median(),
            "평가구간 CF표준편차": s["cf"].std(),
        }
    w(md(pd.DataFrame(rows).T, "그룹", "{:,.4g}"))
    w()

    w("### 6-4. FiCR 가중(actual 가중) 관점 — 발전량 상위 구간의 에너지 집중도\n")
    rows = {}
    for g in [1, 2, 3]:
        s = long[(long["group"] == g) & long["eval"]].sort_values("kwh", ascending=False)
        tot = s["kwh"].sum()
        rows[f"group_{g}"] = {f"상위{p}%시간 에너지비중%":
                              100 * s["kwh"].head(max(1, int(len(s) * p / 100))).sum() / tot
                              for p in [5, 10, 20, 30, 50]}
    w(md(pd.DataFrame(rows).T, "그룹", "{:.2f}"))
    w()

    w("### 6-5. 연·계절별 평가대상 시간 비율 (평가에 실제로 잡히는 양)\n")
    for g in [1, 2, 3]:
        s = long[long["group"] == g].dropna(subset=["kwh"])
        piv = s.pivot_table(index="year", columns="season", values="eval", aggfunc="mean") * 100
        piv = piv[[x for x in SEASON_ORDER if x in piv.columns]]
        w(f"**group_{g}** (단위 %)\n")
        w(md(piv, "연도", "{:.1f}"))
        w()


def sec7_summary(long, slots, fc):
    w("## 7. 요약 — 이 데이터에서 사실로 확인된 것\n")
    w("> 아래는 전부 제공 데이터에서 직접 계산한 값이다. **원인 해석은 포함하지 않는다** "
      "(CLAUDE.md 작업 규칙: 원인은 논의 전이면 `미논의`).\n")

    w("### 7-1. 무결성\n")
    w("- 시간 격자 결손 **0** — 라벨·SCADA·LDAPS·GFS·test 전부 기대 시각을 빠짐없이 채운다. 중복 행도 0.")
    w("- 예보 리드타임은 전 파일 동일하게 **12~35시간**(24개 고유값) — train/test 간 예보 조건 차이 없음.")
    w("- 격자 좌표는 train ≡ test 로 완전 동일 (LDAPS 16개 / GFS 9개).")
    w()

    w("### 7-2. 결측\n")
    w("> 아래 결측은 전부 **그룹별 유효기간 안에서** 센 것이다. "
      "group_3 의 2022년(미제공 구간)은 애초에 대상에서 빠져 있다.")
    for g in [1, 2, 3]:
        s = long[long["group"] == g]["kwh"]
        w(f"- **라벨 group_{g}**: 유효기간 {len(s):,}시간 중 결측 "
          f"**{int(s.isna().sum())}시간** ({100 * s.isna().mean():.2f}%)")
    w("- group_1/2 의 최장 결측은 **2022-10-24 09:00 ~ 10-27 18:00 (82시간)** 으로 두 그룹 공통 — "
      "단지 전체가 동시에 빠진 구간이다. group_3 은 3시간짜리 두 구간뿐.")
    w("- **예보**: LDAPS/GFS train 은 결측 0. **`ldaps_test` 만 3개 예보시각에서 전 16격자 통째 결측** "
      "(§5-1). 해당 시각도 제출 대상이므로 대체값 없이는 예측 불가 — GFS 단독 폴백이나 "
      "인접 시각 보간이 필요하다.")
    w("- **SCADA**: VESTAS(group_1·2) 는 power·ws·wd 전부 결측 0. "
      "UNISON(group_3) 은 커버리지(2023~2024) 안에서 power 2,214셀(0.42%) · ws 5,096셀(0.97%) 결측 — "
      "터빈별 140~1,466셀의 센서 드롭아웃이며 `ws` 가 `power` 보다 잦다(§9-3).")
    w()

    w("### 7-3. 라벨 분포\n")
    for g in [1, 2, 3]:
        s = long[long["group"] == g].dropna(subset=["kwh"])
        w(f"- group_{g}: CF 평균 {s['cf'].mean():.3f} · 중앙 {s['cf'].median():.3f} · "
          f"정확히 0 인 시간 {100 * (s['kwh'] == 0).mean():.1f}% · "
          f"평가대상(≥10%cap) {100 * s['eval'].mean():.1f}%")
    w("- 분포는 세 그룹 모두 **강한 우편향 + 0 질량**: 중앙값 CF 가 평균의 2/3 수준이고 q25 는 0.03 이하.")
    w("- 계절성이 크다 — Q4/Q1(겨울) CF 가 Q3(여름)의 약 2배. 평가대상 비율도 겨울 65~78% vs 여름 43~51%.")
    w()

    w("### 7-4. 가동률 (AVAILABILITY.md 재현)\n")
    for g, df in slots.items():
        tb = avail_by(df, "year")
        tb = tb[tb["windy슬롯"] > 1000]
        vals = " · ".join(f"{int(y)} {v:.3f}" for y, v in tb["가동률_strict"].items())
        w(f"- group_{g}: {vals}")
    w("- AVAILABILITY.md §3 의 수치와 **완전 일치** (독립 재계산으로 교차검증됨).")
    w("- 정지는 상시가 아니라 **에피소드형**: (연×월) 표에서 2024-01/02 에 세 그룹 동시 급락(§3-4).")
    w("- **새 발견 — 겨울이 최저 가동률 계절이다**: g1 0.881 · g2 0.884 · g3 0.813 으로 "
      "네 계절 중 꼴찌. 2024-01/02 에피소드를 제외해도 각 0.930 · 0.937 · 0.848 로 **여전히 최저**"
      "(§3-3b) — 에피소드가 만든 착시가 아니라 상시 계절 패턴이다.")
    w("- 이 겨울이 **동시에 풍속 최고(8.3~9.0 m/s)·CF 최고·평가대상 비율 최고(65~78%)** 계절이다. "
      "즉 점수 비중이 가장 큰 계절과 가동률이 가장 낮은 계절이 겹친다. 원인은 **미논의**.")
    w("- 가동률과 라벨 CF 의 **선형** 상관은 0.27~0.32 로 낮다. 다만 매칭 시간의 59~79% 가 "
      "가동률 = 1.0 한 점에 몰려 있어 선형 상관이 압축된 것이고, 구간별로 보면(§4-2) "
      "가동률 0~25% 구간 CF 는 0.003~0.026, 만가동 구간은 0.44~0.47 로 뚜렷이 갈린다.")
    w()

    w("### 7-5. 2025(test) 예보 분포 시프트 — AVAILABILITY.md §8-C 항목\n")
    for kind, col in [("ldaps", "ws50"), ("gfs", "ws100")]:
        tr, te = fc[(kind, "train")][col], fc[(kind, "test")][col]
        w(f"- **{kind} `{col}`**: train {tr.mean():.3f} → test {te.mean():.3f} m/s "
          f"(Δ +{te.mean() - tr.mean():.3f}, d = +{(te.mean() - tr.mean()) / tr.std():.3f})")
    w("- 2025 예보 풍속은 train 3년 어느 해보다도 높다 (LDAPS ws50 연평균: 7.40/7.04/7.11 → **7.66**).")
    w("- 시프트 상위 컬럼은 **전부 U(동서) 성분** — 10m/50m/80m/100m/PBL/850·700·500hPa 의 u 가 "
      "d ≈ +0.12~0.14 로 상위를 독점하고, v 성분은 d ≤ 0.06. 즉 **서풍 성분이 강해진 해로 예보**됐다.")
    w("- 온도·습도·복사·운량은 d ≤ 0.05 로 사실상 이동 없음 — 시프트는 바람 축에 국한된다.")
    w("- 월별로는 **1월(+0.3~1.7)·2월(+0.7~3.2)·11월** 에 집중.")
    w("- **함의(사실 진술까지만)**: 예측 입력의 풍속 평균이 올라가면 모델 출력 평균도 올라간다. "
      "현행 제출물에 곱해진 전역 스케일 ×0.98 은 이 시프트와 **독립적으로** LB 에서 캘리브레이션된 값이다. "
      "둘의 관계(가동률 하락 보정인지, 풍속 시프트로 인한 과대예측 보정인지)는 **미논의**.")
    w()

    w("### 7-6. 지표 관점\n")
    for g in [1, 2, 3]:
        s = long[(long["group"] == g) & long["eval"]]
        allg = long[long["group"] == g].dropna(subset=["kwh"])
        w(f"- group_{g}: 평가시간은 전체의 {100 * len(s) / len(allg):.1f}% 뿐이지만 "
          f"**총 발전량의 {100 * s['kwh'].sum() / allg['kwh'].sum():.1f}%** 를 담는다.")
    w("- FiCR 은 actual 가중이므로 실질 승부처는 이 구간 — 상위 10% 시간이 평가구간 에너지의 "
      "약 18~20%, 상위 30% 가 약 50% 를 차지한다(§6-4).")
    w("- 경험적 파워커브(§6-2)에서 CF 조건부 평균은 ws100 ≈ 11~12 m/s 에서 **0.70 안팎으로 포화**하고 "
      "1.0 에 도달하지 않는다. 같은 풍속 구간의 CF 표준편차가 0.15~0.25 로 크다는 사실만 확인되고, "
      "포화 수준이 이 값인 이유는 **미논의**.")
    w()


DERIVED = {"ldaps": ["ws10", "ws50", "wsbl"],
           "gfs": ["ws10", "ws80", "ws100", "wspbl"]}
CALENDAR = ["year", "month", "quarter", "season", "hour"]


def col_inventory(df, name, drop=()):
    """파일 1개의 컬럼별 notna / na 개수 표."""
    cols = [c for c in df.columns if c not in set(drop)]
    rows = {}
    for c in cols:
        s = df[c]
        nn = int(s.notna().sum())
        rows[c] = {"dtype": str(s.dtype), "n": len(s), "notna": nn,
                   "na": len(s) - nn, "na%": 100 * (len(s) - nn) / len(s) if len(s) else np.nan}
    tb = pd.DataFrame(rows).T
    nbad = int((tb["na"] > 0).sum())
    w(f"**{name}** — 전체 {len(cols)}개 컬럼 중 결측 있는 컬럼 **{nbad}개**\n")
    w(md(tb, "컬럼", "{:,.4g}"))
    w()
    return tb


def sec8_columns(fc, lab, long):
    w("## 8. 부록 — 파일별 전 컬럼 notna / na 개수\n")
    w("> `notna` = 값이 있는 셀 수, `na` = 결측 셀 수. 파생 컬럼(ws10 등)과 달력 컬럼은 "
      "제공 원본이 아니므로 제외하고, **주최측이 준 컬럼만** 싣는다.\n")

    lab = pd.read_csv(DATA_DIR / "train" / "train_labels.csv",
                      encoding="utf-8-sig", parse_dates=["kst_dtm"])
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    ves = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv",
                      encoding="utf-8-sig", parse_dates=["kst_dtm"])
    uni = pd.read_csv(DATA_DIR / "train" / "scada_unison_train.csv",
                      encoding="utf-8-sig", parse_dates=["kst_dtm"])

    summary = []
    for nm, df, drop in [("train_labels.csv", lab, ()),
                         ("sample_submission.csv", sub, ()),
                         ("scada_vestas_train.csv", ves, ()),
                         ("scada_unison_train.csv", uni, ())]:
        tb = col_inventory(df, nm, drop)
        summary.append((nm, len(tb), int((tb["na"] > 0).sum()), int(tb["na"].sum())))

    for kind in ["ldaps", "gfs"]:
        for split in ["train", "test"]:
            drop = DERIVED[kind] + CALENDAR
            tb = col_inventory(fc[(kind, split)], f"{kind}_{split}.csv", drop)
            summary.append((f"{kind}_{split}.csv", len(tb),
                            int((tb["na"] > 0).sum()), int(tb["na"].sum())))

    try:
        # 헤더가 4번째 행(0-indexed 3), 첫 열은 빈 열, 데이터는 터빈 17기
        info = pd.read_excel(DATA_DIR / "info.xlsx", header=3).dropna(axis=1, how="all")
        info = info.dropna(how="all")
        tb = col_inventory(info, "info.xlsx (header=3, 빈 열 제거)")
        w("> `info.xlsx` 는 상단에 제목 행이 있어 헤더가 4번째 행이다. "
          "`KPX그룹`·`그룹설비용량(MW)` 의 결측은 **엑셀 병합셀** 때문으로, 그룹의 첫 행에만 "
          "값이 있고 나머지는 비어 있다 — 실제 누락이 아니라 `ffill` 로 채우면 되는 형태다.\n")
        summary.append(("info.xlsx", len(tb), int((tb["na"] > 0).sum()), int(tb["na"].sum())))
    except Exception as e:                                    # openpyxl 부재 등
        w(f"- `info.xlsx` 읽기 실패: {type(e).__name__}: {e}\n")

    w("### 8-1. 파일별 총괄\n")
    s = pd.DataFrame(summary, columns=["파일", "컬럼수", "결측있는컬럼", "총결측셀"]).set_index("파일")
    w(md(s, "파일", "{:,.0f}"))
    w("\n> ⚠ 위 표는 **파일에 적힌 그대로**의 빈 셀 수다. "
      "`train_labels.csv` 의 8,973 중 8,760 은 group_3 의 2022년 = **미제공 구간이지 결측이 아니다**. "
      "유효기간 기준으로 다시 세면 아래와 같다.\n")
    lg = pd.DataFrame({
        f"group_{g}": {
            "파일상 빈 셀": int(lab[f"kpx_group_{g}"].isna().sum()),
            "미제공 구간(결측 아님)": int(lab[f"kpx_group_{g}"].isna().sum())
                                  - int(long[long["group"] == g]["kwh"].isna().sum()),
            "실제 결측": int(long[long["group"] == g]["kwh"].isna().sum()),
            "유효기간 시간": int((long["group"] == g).sum()),
        } for g in [1, 2, 3]}).T
    lg["실제 결측%"] = 100 * lg["실제 결측"] / lg["유효기간 시간"]
    w(md(lg, "라벨 컬럼", "{:,.4g}"))
    w()

    w("**읽는 법 — 결측의 성격이 파일마다 다르다.**\n")
    w("- `ldaps_train` · `gfs_train` · `gfs_test` · `scada_vestas_train` · `sample_submission`: "
      "**전 컬럼 결측 0**.")
    w("- `train_labels`: 유효기간 기준 실제 결측은 group_1 104 · group_2 103 · group_3 6 시간뿐 "
      "(각 0.4% / 0.4% / 0.03%). group_3 의 8,760 은 미제공 구간이라 제외.")
    w("- `ldaps_test`: 19개 컬럼 × 752셀 — 3개 예보시각이 전 격자에서 통째로 빠진 것(§5-1). "
      "제출 대상 시각이므로 처리가 필요하다.")
    w("- **`scada_unison_train`: 15개 컬럼 9,511셀은 진짜 원본 결측이다.** 터빈별 0.13~1.39% 로 "
      "센서 드롭아웃 수준이며, wtg01·02·05 의 `ws` 가 1.3~1.4% 로 가장 잦다. "
      "(이전 판에서 group_3 SCADA 결측을 33.6% 로 적었던 것은 UNISON 을 VESTAS 시간축에 "
      "outer merge 하며 2022년 빈 행 262,775셀이 생긴 탓이었고, 현재는 제작사 커버리지로 "
      "잘라내 §3-1 에서 제거됐다.)")
    w("- `info.xlsx`: 병합셀 아티팩트 (위 주석 참조), 실제 누락 아님.\n")


WDROP = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}


def group_grid_maps(fc):
    """터빈 좌표 → 그룹 centroid → 최근접 격자. 파이프라인(exp005_qopt)과 동일 함수 사용."""
    from exp005_qopt import dms_to_decimal, find_nearest_grid, haversine

    info = pd.read_excel(DATA_DIR / "info.xlsx", header=3).dropna(axis=1, how="all").dropna(how="all")
    info["KPX그룹"] = info["KPX그룹"].ffill().astype(int)
    ll = info["좌표(Google)"].map(dms_to_decimal)
    info["lat"] = [c[0] for c in ll]
    info["lon"] = [c[1] for c in ll]
    cent = info.groupby("KPX그룹")[["lat", "lon"]].mean().reset_index()
    cent.columns = ["group", "lat", "lon"]
    maps = {k: find_nearest_grid(fc[(k, "train")], cent, "group") for k in ["ldaps", "gfs"]}

    rows = {}
    for _, r in cent.iterrows():
        g = int(r["group"])
        d = {"터빈수": int((info["KPX그룹"] == g).sum()),
             "제작사": "/".join(sorted(info.loc[info["KPX그룹"] == g, "제작사"].unique())),
             "centroid_lat": r["lat"], "centroid_lon": r["lon"]}
        for k in ["ldaps", "gfs"]:
            gid = maps[k][g]
            gr = fc[(k, "train")].loc[fc[(k, "train")]["grid_id"] == gid,
                                      ["latitude", "longitude"]].iloc[0]
            d[f"{k}_grid"] = gid
            d[f"{k}_거리km"] = haversine(r["lat"], r["lon"], gr["latitude"], gr["longitude"])
        rows[f"group_{g}"] = d
    return maps, pd.DataFrame(rows).T, info


def sec9_group_features(fc, long):
    w("## 9. 그룹별 feature notna / na — 모델이 실제로 보는 단위\n")
    w("> §8 은 파일 단위 집계라 모델 관점과 어긋난다. 여기서는 **그룹 1·2·3 각각이 학습/추론에 "
      "쓰는 프레임**(그룹 대표 격자의 예보 + 해당 그룹 라벨 + 해당 그룹 터빈 SCADA) 기준으로 "
      "피처별 notna/na 를 센다.\n")

    maps, gtab, info = group_grid_maps(fc)
    w("### 9-1. 그룹 ↔ 격자 매핑 (파이프라인과 동일 산출)\n")
    w(md(gtab, "그룹", "{:,.4f}"))
    w("\n> LDAPS 는 그룹마다 다른 격자, GFS 는 0.25° 해상도라 세 그룹이 **같은 격자**를 공유한다 "
      "— 아래 표에서 GFS 행의 세 그룹 값이 동일한 이유.\n")

    for split, nhour in [("train", 26304), ("test", 8760)]:
        w(f"### 9-2-{split}. 예보 피처 ({split}, 그룹당 {nhour:,}시각)\n")
        rows = {}
        for kind in ["ldaps", "gfs"]:
            cols = [c for c in fc[(kind, split)].columns
                    if c not in WDROP and c not in DERIVED[kind] + CALENDAR]
            for c in cols:
                key = f"{kind}.{c}"
                for g in [1, 2, 3]:
                    sub = fc[(kind, split)]
                    sub = sub[sub["grid_id"] == maps[kind][g]]
                    rows.setdefault(key, {})[f"g{g} notna"] = int(sub[c].notna().sum())
                    rows[key][f"g{g} na"] = int(sub[c].isna().sum())
        if split == "train":
            for g in [1, 2, 3]:
                s = long[long["group"] == g]["kwh"]
                rows.setdefault("label.kpx_group_N", {})[f"g{g} notna"] = int(s.notna().sum())
                rows["label.kpx_group_N"][f"g{g} na"] = int(s.isna().sum())
        tb = pd.DataFrame(rows).T
        tb = tb[[f"g{g} {x}" for g in [1, 2, 3] for x in ["notna", "na"]]]
        w(md(tb, "feature", "{:,.0f}"))
        w()
        na_cols = [c for c in tb.columns if c.endswith(" na")]     # 'notna' 와 구분 (앞의 공백)
        bad = tb[(tb[na_cols] > 0).any(axis=1)]
        w(f"→ {split}: 전체 {len(tb)}개 피처 중 **결측 있는 피처 {len(bad)}개**"
          + (f" — {', '.join('`' + i + '`' for i in bad.index)}" if len(bad) else "") + "\n")

    w("### 9-3. SCADA 터빈 컬럼 (그룹별, 원본 파일 기준)\n")
    ves = pd.read_csv(DATA_DIR / "train" / "scada_vestas_train.csv", encoding="utf-8-sig")
    uni = pd.read_csv(DATA_DIR / "train" / "scada_unison_train.csv", encoding="utf-8-sig")
    src = {"vestas": ves, "unison": uni}
    for g, turbines in GROUP_TURBINES.items():
        rows = {}
        for mk, i in turbines:
            df = src[mk]
            n = len(df)
            d = {"파일행수": n}
            for meas in ["power_kw10m", "ws", "wd"]:
                c = f"{mk}_wtg{i:02d}_{meas}"
                nn = int(df[c].notna().sum())
                d[f"{meas} notna"] = nn
                d[f"{meas} na"] = n - nn
            rows[f"{mk}_wtg{i:02d}"] = d
        w(f"**group_{g}** ({turbines[0][0].upper()} {len(turbines)}기)\n")
        w(md(pd.DataFrame(rows).T, "터빈", "{:,.0f}"))
        w()
    w("> VESTAS(group_1·2)는 세 측정 모두 결측 0. UNISON(group_3)만 터빈별 140~1,466셀 결측이 있고 "
      "`ws` 가 `power` 보다 잦다. 단 UNISON 파일은 105,264행(2023~2024)이라 "
      "VESTAS 157,819행(2022~2024)과 기간 자체가 다르다.\n")


def main():
    w("# 데이콘 BARAM 2026 — 제공 데이터 EDA\n")
    w("> 생성: `expB_eda.py` · 대상 = 주최측 제공 원자료 전부 "
      "(train_labels / SCADA / LDAPS / GFS / sample_submission)\n")
    w("> 가동률 정의는 [AVAILABILITY.md](AVAILABILITY.md) 와 동일 "
      "(windy ≥ 5 m/s 슬롯 중 출력 ≤ 1 kWh 비율의 여집합).\n")
    w("> 시각 규약: 라벨 `kst_dtm` = 집계구간 **종료** 시각. "
      "SCADA 슬롯 00:10~01:00 → 라벨 01:00. (본문 §0 검증)\n")
    w("> **핵심 요약은 [§7](#7-요약--이-데이터에서-사실로-확인된-것)** — 본문 §0~§6 은 근거 표.\n")

    lab, long, support = load_labels()
    sc, cov = load_scada()
    fc = {(k, s): load_fcst(k, s) for k in ["ldaps", "gfs"] for s in ["train", "test"]}
    slots = {g: group_slots(sc, g, cov) for g in [1, 2, 3]}

    w("## 주의 — 이 리포트의 두 가지 집계 규약\n")
    w("**(1) 그룹별 유효기간(support) 밖은 결측으로 세지 않는다.** "
      "`kpx_group_3` 의 2022년은 주최측이 제공하지 않은 구간이지 결측이 아니다"
      "(`data_description.md` §5: \"2022년 구간은 빈칸\"). "
      "따라서 각 그룹을 아래 기간으로 자른 뒤 **그 안에서만** 결측·통계를 계산한다. "
      "SCADA 도 같은 이유로 제작사 파일의 실제 커버리지로 자른다.\n")
    rows = {}
    for g, (lo, hi) in support.items():
        n = int(((long["group"] == g)).sum())
        rows[f"group_{g}"] = {"유효기간 시작": str(lo), "유효기간 끝": str(hi),
                              "대상 시간": n}
    for mk, (lo, hi) in cov.items():
        rows[f"SCADA {mk}"] = {"유효기간 시작": str(lo), "유효기간 끝": str(hi), "대상 시간": np.nan}
    w(md(pd.DataFrame(rows).T, "대상", "{:,.0f}"))
    w()
    w("**(2) 달력 피처는 구간 종료 시각이 아니라 구간 자체에서 뽑는다.** "
      "라벨·예보 시각은 집계구간의 **종료** 시각이므로(§0), 연/월/분기/계절/시는 "
      "`t − 1시간` 기준으로 매긴다. 이렇게 하지 않으면 `2025-01-01 00:00`"
      "(= 2024-12-31 23~24시 구간)이 2025년으로 잡혀 연도 표에 n=1 짜리 유령 행이 생긴다. "
      "`hour` 도 구간 시작 기준이라 `hour=0` 은 00:00~01:00 구간을 뜻한다.\n")

    w("## 0. 시각 정렬 검증 — SCADA 합 vs 라벨\n")
    w("라벨 `kst_dtm` 이 집계구간의 **시작**인지 **종료**인지에 따라 슬롯 매핑이 한 칸 달라진다. "
      "그룹별 터빈 출력 합을 두 가정으로 각각 시간 집계해 라벨과 대조했다 "
      "(부호 대칭 글리치는 양쪽 모두 제거).\n")
    li = lab.set_index("kst_dtm")
    rows = {}
    for g, turbines in GROUP_TURBINES.items():
        cols = [f"{mk}_wtg{i:02d}_power_kw10m" for mk, i in turbines]
        mk0 = turbines[0][0]
        x = sc[cols].astype(float)
        x = x.clip(lower=0).where(x.abs() <= GLITCH_CAP[mk0])
        tot = x.sum(axis=1, min_count=1)
        for name, h in [("floor(구간 시작 가정)", sc["kst_dtm"].dt.floor("h")),
                        ("ceil(구간 종료 가정)", (sc["kst_dtm"] - pd.Timedelta(minutes=1)).dt.ceil("h"))]:
            agg = pd.DataFrame({"h": h, "tot": tot}).groupby("h")["tot"].sum(min_count=1)
            m = li[[f"kpx_group_{g}"]].join(agg.rename("scada")).dropna()
            rows[f"group_{g} · {name}"] = {
                "매칭시간": len(m), "corr": m["scada"].corr(m[f"kpx_group_{g}"]),
                "SCADA합/라벨합": m["scada"].sum() / m[f"kpx_group_{g}"].sum()}
    w(md(pd.DataFrame(rows).T, "그룹 · 가정"))
    w("\n→ 세 그룹 모두 **구간 종료(ceil) 가정에서 corr 0.996~0.9998**. 이 규약을 "
      "이후 모든 SCADA↔라벨 결합에 적용한다. 합계비가 1.01 안팎인 것은 SCADA 가 "
      "터빈 단자 기준, 라벨이 계통 인도 기준이라는 차이와 정합적이나 세부는 **미논의**.\n")

    sec1_inventory(lab, sc, fc)
    sec2_labels(lab, long)
    sec3_avail(slots)
    sec4_cross(long, slots)
    sec5_fcst(fc)
    sec6_link(long, fc)
    sec7_summary(long, slots, fc)
    sec8_columns(fc, lab, long)
    sec9_group_features(fc, long)

    OUT_MD.write_text("\n".join(_BUF), encoding="utf-8")
    print(f"\n[저장] {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
