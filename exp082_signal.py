# ═══════════════════════════════════════════════════════
# exp-082 사전 신호 선별: ASOS 관측 유도 변수 vs **챔피언 실제 잔차**
#
# 목적: 게이트(30분)를 돌리기 전에, 어떤 관측 신호 형태가 챔피언이 놓친 성분과
#   실제로 상관되는지 싸게(캐시 재사용, 학습 0) 선별. exp-072(밀도)처럼 신호가
#   없으면 게이트는 ±0.0000이 나오므로, 형태 선택을 먼저 하고 최선만 게이트로 보냄.
#
# 후보 신호 (전부 관측시각 ≤ D-1 12:00 준수):
#   gap24      : mean(obs_ws) − mean(fc_ws), 창 [D-2 12:00, D-1 12:00]
#   gap12      : 동일, 창 [D-1 00:00, D-1 12:00]  (더 최근)
#   ratio24    : mean(obs_ws) / mean(fc_ws)
#   gap_anom   : gap24 − (직전 30일 gap24 이동평균)   ← 느린 드리프트 제거
#   obs_ws24   : 관측 풍속 수준 자체 (지속성)
#
# ⚠ 선별 caveat: 잔차는 캐시(2024 폴드)에만 존재하므로 스크리닝도 2024에서 수행 —
#   약간 낙관 편향. 단 적합이 아니라 상관 스크린이고, 판정은 게이트(CV+홀드)가 함.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp082_signal.py [--topk 5]
# ═══════════════════════════════════════════════════════
import argparse
import sys
import time

import numpy as np
import pandas as pd

from exp005_qopt import CAPACITY_KWH
from exp019_multigrid import HOLD_FOLD
from exp060_pergroup_hpo import load_cache_full
from exp065_gbm3 import build_train_frame
from exp082_gate import load_asos, pick_stations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_signals(asos, train_df, stns):
    a = asos[asos["stn"].isin(stns)]
    obs_s = a.groupby("tm")["ws"].mean().sort_index()
    df = train_df[["forecast_kst_dtm", "kpx_group", "ldaps_ws_10m"]].copy()
    df["tday"] = (df["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    days = pd.DatetimeIndex(sorted(df["tday"].unique()))
    oi, ov = obs_s.index.to_numpy(), obs_s.to_numpy()
    rows = []
    for g, sub in df.groupby("kpx_group"):
        s = sub.set_index("forecast_kst_dtm")["ldaps_ws_10m"].sort_index()
        si, sv = s.index.to_numpy(), s.to_numpy()
        for d in days:
            d64 = np.datetime64(d)
            w24_0 = d64 - np.timedelta64(36, "h")
            w12_0 = d64 - np.timedelta64(24, "h")
            w_1 = d64 - np.timedelta64(12, "h")
            o24 = ov[(oi >= w24_0) & (oi <= w_1)]
            o12 = ov[(oi >= w12_0) & (oi <= w_1)]
            f24 = sv[(si >= w24_0) & (si <= w_1)]
            f12 = sv[(si >= w12_0) & (si <= w_1)]
            if len(o24) < 6 or len(f24) < 6:
                continue
            om, fm = o24.mean(), f24.mean()
            rows.append((d, g, om - fm,
                         (o12.mean() - f12.mean()) if (len(o12) >= 3 and len(f12) >= 3) else np.nan,
                         om / fm if fm > 0.1 else np.nan, om))
    out = pd.DataFrame(rows, columns=["tday", "kpx_group", "gap24", "gap12", "ratio24", "obs_ws24"])
    out = out.sort_values(["kpx_group", "tday"])
    out["gap_anom"] = out["gap24"] - out.groupby("kpx_group")["gap24"] \
                                       .transform(lambda x: x.shift(1).rolling(30, min_periods=5).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    print("### exp-082 사전 신호 선별: 관측 신호 vs 챔피언 잔차 ###", flush=True)

    train_df, _ = build_train_frame()
    asos = load_asos()
    stns = pick_stations(asos, train_df, args.topk)
    sig = build_signals(asos, train_df, stns)
    print(f"신호 프레임 {len(sig)}행", flush=True)

    cache, _ = load_cache_full()
    cv = cache[(cache["fold"] != HOLD_FOLD["name"]) & cache["target"].notna()].copy()
    cv["tday"] = (cv["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.normalize()
    cv = cv.rename(columns={"group": "kpx_group"})
    m = cv.merge(sig, on=["tday", "kpx_group"], how="left")
    assert len(m) == len(cv)
    cap = m["kpx_group"].map(lambda g: CAPACITY_KWH[f"kpx_group_{int(g)}"]).to_numpy()
    blend = 0.65 * m["cf_hpo"].to_numpy() + 0.35 * m["dl_cln03"].to_numpy()
    resid = m["target"].to_numpy() / cap - blend          # 챔피언 base 잔차 (후처리 전)
    ev = m["target"].to_numpy() >= 0.10 * cap
    cols = ["gap24", "gap12", "ratio24", "gap_anom", "obs_ws24"]
    print(f"\n커버리지: " + ", ".join(f"{c} {m[c].notna().mean():.2f}" for c in cols), flush=True)
    print(f"\n{'신호':<12}{'corr(잔차)':>12}{'corr(|잔차|)':>14}{'corr(CF)':>11}   (평가시간 n="
          f"{int((ev & m['gap24'].notna()).sum())})", flush=True)
    cf = m["target"].to_numpy() / cap
    for c in cols:
        ok = ev & m[c].notna().to_numpy()
        x = m.loc[ok, c].to_numpy()
        if len(x) < 100 or np.std(x) == 0:
            print(f"{c:<12}{'표본부족':>12}", flush=True)
            continue
        r1 = np.corrcoef(x, resid[ok])[0, 1]
        r2 = np.corrcoef(x, np.abs(resid[ok]))[0, 1]
        r3 = np.corrcoef(x, cf[ok])[0, 1]
        print(f"{c:<12}{r1:>+12.4f}{r2:>+14.4f}{r3:>+11.4f}", flush=True)

    # 참고: 그룹별 최고 후보 분해
    best = "gap24"
    print(f"\n그룹별 corr({best}, 잔차):", flush=True)
    for g in (1, 2, 3):
        ok = ev & (m["kpx_group"] == g).to_numpy() & m[best].notna().to_numpy()
        if ok.sum() > 100:
            print(f"  그룹{g}: {np.corrcoef(m.loc[ok, best], resid[ok])[0,1]:+.4f} (n={ok.sum()})",
                  flush=True)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
