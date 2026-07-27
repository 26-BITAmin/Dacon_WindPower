# ═══════════════════════════════════════════════════════
# exp-048 프로브: Open-Meteo Previous Runs API — 미사용 모델의 한국 커버리지 확인
#
# 목적: exp-020(JMA GSM/MSM)과 같은 방식으로 투입할 수 있는 **추가 예보 모델** 중
#   previous_day2가 2022년(최소 2023년)부터 한국 좌표에서 non-null인 모델 선별.
#   ECMWF IFS·ICON은 exp-022에서 2024+ 부분 커버로 기각 → 제외.
# 비용: ~40콜 (무료 티어 10,000콜/일).
# 실행: C:\miniconda3\python.exe -X utf8 exp048_probe_models.py
# ═══════════════════════════════════════════════════════
import json
import subprocess
import sys
import time

import pandas as pd

from exp019_multigrid import _centroids

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CANDIDATES = [
    "gem_seamless", "gem_global",
    "meteofrance_seamless", "meteofrance_arpege_world",
    "ukmo_seamless", "ukmo_global_deterministic_10km",
    "cma_grapes_global",
    "bom_access_global",
    "kma_seamless", "kma_gdps",
]
WINDOWS = [("2022-01-01", "2022-01-07"), ("2022-07-01", "2022-07-07"),
           ("2023-01-01", "2023-01-07"), ("2024-07-01", "2024-07-07")]
HOURLY = "wind_speed_10m_previous_day2,wind_speed_100m_previous_day2"


def fetch(url):
    r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time", "60", url],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip().startswith("{"):
        return {"error": f"curl rc={r.returncode}"}
    return json.loads(r.stdout)


def main():
    c = _centroids()
    lat, lon = round(float(c.loc[1, "lat"]), 4), round(float(c.loc[1, "lon"]), 4)
    print(f"프로브 좌표(그룹1 centroid): {lat}, {lon}\n")
    print(f"{'model':<32}{'window':<24}{'ws10_d2':>9}{'ws100_d2':>10}  비고")
    ok_models = {}
    for model in CANDIDATES:
        model_ok = []
        for s, e in WINDOWS:
            url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&models={model}"
                   f"&hourly={HOURLY}&wind_speed_unit=ms"
                   f"&start_date={s}&end_date={e}&timezone=Asia%2FSeoul")
            d = fetch(url)
            time.sleep(0.8)
            if "hourly" not in d:
                reason = d.get("reason", d.get("error", "unknown"))
                print(f"{model:<32}{s[:7]:<24}{'':>9}{'':>10}  ERROR: {str(reason)[:60]}")
                break
            h = d["hourly"]
            n = len(h.get("time", []))
            w10 = pd.Series(h.get("wind_speed_10m_previous_day2", [])).notna().sum()
            w100 = pd.Series(h.get("wind_speed_100m_previous_day2", [])).notna().sum()
            print(f"{model:<32}{s[:7]:<24}{w10:>7}/{n:<3}{w100:>8}/{n:<3}")
            model_ok.append((s[:7], int(w10), int(w100), n))
        if model_ok:
            ok_models[model] = model_ok
        print()

    print("=" * 80)
    print("요약 — 전 윈도우 ws10_d2 완전 커버 모델:")
    for m, wins in ok_models.items():
        full10 = all(w10 >= n - 2 for _, w10, _, n in wins)
        full100 = all(w100 >= n - 2 for _, _, w100, n in wins)
        cov = ", ".join(f"{w}:{w10}/{n}" for w, w10, _, n in wins)
        print(f"  {m:<32} 2022완전커버={'O' if full10 else 'X'} ws100={'O' if full100 else 'X'}  ({cov})")


if __name__ == "__main__":
    main()
