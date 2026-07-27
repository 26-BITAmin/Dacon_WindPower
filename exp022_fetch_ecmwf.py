# ═══════════════════════════════════════════════════════
# exp-022 데이터 수집: ECMWF IFS 0.25° + ICON global (previous_day2, 2024~2025)
# 2024+ 부분 커버 소스 — 2022-23은 결측(median impute). 규정·라이선스는 exp-020과 동일.
# 실행: C:\miniconda3\python.exe -X utf8 exp022_fetch_ecmwf.py
# ═══════════════════════════════════════════════════════
import json
import sys

import pandas as pd

from exp019_multigrid import _centroids
from exp005_qopt import DATA_DIR
from exp020_fetch_jma import fetch_json, BASE_URL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = DATA_DIR / "external"
MODELS = ["ecmwf_ifs025", "icon_global"]
YEARS = ["2024", "2025"]


def main():
    centroids = _centroids()
    frames = []
    n_calls = 0
    for group in [1, 2, 3]:
        lat = round(float(centroids.loc[group, "lat"]), 4)
        lon = round(float(centroids.loc[group, "lon"]), 4)
        for model in MODELS:
            for year in YEARS:
                url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&models={model}"
                       f"&hourly=wind_speed_10m_previous_day2,wind_direction_10m_previous_day2"
                       f"&wind_speed_unit=ms&start_date={year}-01-01&end_date={year}-12-31&timezone=Asia%2FSeoul")
                d = fetch_json(url)
                n_calls += 1
                h = d["hourly"]
                df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"]),
                                   "ws_d2": h["wind_speed_10m_previous_day2"],
                                   "wd_d2": h["wind_direction_10m_previous_day2"]})
                df["kpx_group"], df["model"] = group, model
                nn = df["ws_d2"].notna().sum()
                print(f"group{group} {model} {year}: {nn}/{len(df)}", flush=True)
                frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "ecmwf_icon_prevruns.csv", index=False, encoding="utf-8-sig")
    meta = {"note": "exp-022 수집: ECMWF IFS025 + ICON global, previous_day2, 2024~2025 (2022-23 결측)",
            "license": "CC-BY 4.0 (Open-Meteo)", "fetched_at": pd.Timestamp.now().isoformat(),
            "n_api_calls": n_calls}
    with open(OUT_DIR / "fetch_meta_exp022.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: ecmwf_icon_prevruns.csv ({len(out)}행), API {n_calls}회")


if __name__ == "__main__":
    main()
