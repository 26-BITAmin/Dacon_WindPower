# exp-048 프로브 2차: gem/cma/bom 아카이브 시작 경계 탐색
# 실행: C:\miniconda3\python.exe -X utf8 exp048_probe2.py
import json
import subprocess
import sys
import time

import pandas as pd

from exp019_multigrid import _centroids

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ["gem_global", "cma_grapes_global", "bom_access_global"]
WINDOWS = [("2023-04-01", "2023-04-07"), ("2023-07-01", "2023-07-07"),
           ("2023-10-01", "2023-10-07"), ("2024-01-01", "2024-01-07"),
           ("2024-04-01", "2024-04-07")]


def fetch(url):
    r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time", "60", url],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip().startswith("{"):
        return {"error": f"curl rc={r.returncode}"}
    return json.loads(r.stdout)


c = _centroids()
lat, lon = round(float(c.loc[1, "lat"]), 4), round(float(c.loc[1, "lon"]), 4)
print(f"{'model':<20}" + "".join(f"{s[:7]:>10}" for s, _ in WINDOWS))
for model in MODELS:
    row = []
    for s, e in WINDOWS:
        url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&models={model}"
               f"&hourly=wind_speed_10m_previous_day2&wind_speed_unit=ms"
               f"&start_date={s}&end_date={e}&timezone=Asia%2FSeoul")
        d = fetch(url)
        time.sleep(0.8)
        if "hourly" not in d:
            row.append("ERR")
        else:
            h = d["hourly"]
            n = len(h.get("time", []))
            w = pd.Series(h.get("wind_speed_10m_previous_day2", [])).notna().sum()
            row.append(f"{w}/{n}")
    print(f"{model:<20}" + "".join(f"{x:>10}" for x in row))
