# ═══════════════════════════════════════════════════════
# exp-082 정리: 확보한 ASOS 수집물 인벤토리 + 단일 parquet 통합
#
# 수집 중단(apihub IP 차단) 시점의 자산을 정리해 재개 시 바로 쓸 수 있는 형태로 남긴다.
#   - 월별 CSV 산재 → kma_asos.parquet 단일 파일
#   - 커버리지·지점·결측 요약, 발전단지 인근 지점 신호 강도
#
# 실행: C:\miniconda3\python.exe -X utf8 exp082_inventory.py
# ═══════════════════════════════════════════════════════
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH, DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ASOS_DIR = DATA_DIR / "external" / "kma_asos"
OUT = DATA_DIR / "external" / "kma_asos.parquet"


def main():
    t0 = time.time()
    print("### exp-082 정리: ASOS 수집물 인벤토리 ###", flush=True)
    files = sorted(ASOS_DIR.glob("*.csv"))
    print(f"월별 CSV {len(files)}개", flush=True)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tm"] = pd.to_datetime(df["tm"].astype(str), format="%Y%m%d%H%M")
    for c in ("ws", "ta", "wd", "gst_ws", "pa", "hm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(["tm", "stn"]).sort_values(["tm", "stn"])

    print(f"\n[전체] {len(df):,}행 · 지점 {df['stn'].nunique()}개 · "
          f"{df['tm'].min():%Y-%m-%d %H:%M} ~ {df['tm'].max():%Y-%m-%d %H:%M}", flush=True)
    hrs = df["tm"].nunique()
    print(f"고유 시각 {hrs:,}개 (= 소비한 API 콜 수)", flush=True)

    # 커버리지: 목표 구간 대비
    def rng(d0, d1):
        out, d = [], d0
        while d <= d1:
            out.append(d)
            d += timedelta(days=1)
        return out
    have_d = set(df["tm"].dt.date.unique())
    seg = {"train(2022-01~2024-12)": rng(date(2022, 1, 1), date(2024, 12, 31)),
           "test(2025)": rng(date(2025, 1, 1), date(2025, 12, 31))}
    print("\n[구간별 일 커버리지]", flush=True)
    for name, days in seg.items():
        n = sum(1 for d in days if d in have_d)
        print(f"  {name}: {n}/{len(days)}일 ({n/len(days):.1%})", flush=True)

    # 결측률
    print("\n[변수 결측률]", flush=True)
    for c in ("ws", "ta", "wd", "gst_ws", "pa", "hm"):
        if c in df.columns:
            print(f"  {c}: {df[c].isna().mean():.1%}", flush=True)

    # 시간당 지점 수 안정성
    per_h = df.groupby("tm")["stn"].nunique()
    print(f"\n시각당 지점 수: 중앙값 {per_h.median():.0f} · 최소 {per_h.min()} · 최대 {per_h.max()}",
          flush=True)

    # 발전단지 인근·고상관 지점 요약 (라벨 기간과 겹치는 부분)
    lab = pd.read_csv(DATA_DIR / "train" / "train_labels.csv", encoding="utf-8-sig")
    lab["kst_dtm"] = pd.to_datetime(lab["kst_dtm"])
    lab["date"] = (lab["kst_dtm"] - pd.Timedelta(hours=1)).dt.date
    cf = pd.concat([lab.groupby("date")[f"kpx_group_{g}"].mean() /
                    CAPACITY_KWH[f"kpx_group_{g}"] for g in (1, 2, 3)], axis=1).mean(axis=1)
    dws = df.assign(date=df["tm"].dt.date).groupby(["stn", "date"])["ws"].mean().unstack(0)
    common = dws.index.intersection(cf.dropna().index)
    corr = dws.loc[common].corrwith(cf.loc[common]).dropna().sort_values(ascending=False)
    print(f"\n[일평균 풍속 ↔ 일평균 CF 상관 상위 8지점] (겹치는 {len(common)}일)", flush=True)
    for stn, c in corr.head(8).items():
        n = int(dws[stn].loc[common].notna().sum())
        print(f"  stn {int(stn):>3}: corr {c:+.4f} (유효 {n}일)", flush=True)

    df.to_parquet(OUT, index=False)
    mb = OUT.stat().st_size / 1e6
    print(f"\n통합 저장 → {OUT.name} ({mb:.1f} MB, {len(df):,}행)", flush=True)
    print(f"총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
