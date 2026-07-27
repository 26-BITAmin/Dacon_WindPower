# ═══════════════════════════════════════════════════════
# exp-021 데이터 수집: JMA 심화 — MSM 이웃 격자 + previous_day3 (예보 추세용)
#   a. MSM 이웃 4점 (N/S/E/W, ±0.1° lat / ±0.125° lon) ws previous_day2, 2023~2025
#   b. GSM/MSM centroid ws previous_day3 (d2−d3 = 예보 수정폭 피처용)
# 규정·라이선스는 exp-020과 동일 (fetch_meta.json 참조, CC-BY 4.0)
# 실행: C:\miniconda3\python.exe -X utf8 exp021_fetch_jma2.py
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
OFFSETS = {"N": (0.10, 0.0), "S": (-0.10, 0.0), "E": (0.0, 0.125), "W": (0.0, -0.125)}


def main():
    centroids = _centroids()
    n_calls = 0

    # a. MSM 이웃 격자 (previous_day2)
    frames = []
    for group in [1, 2, 3]:
        clat = round(float(centroids.loc[group, "lat"]), 4)
        clon = round(float(centroids.loc[group, "lon"]), 4)
        for d, (dlat, dlon) in OFFSETS.items():
            for year in ["2023", "2024", "2025"]:
                url = (f"{BASE_URL}?latitude={clat + dlat}&longitude={clon + dlon}&models=jma_msm"
                       f"&hourly=wind_speed_10m_previous_day2&wind_speed_unit=ms"
                       f"&start_date={year}-01-01&end_date={year}-12-31&timezone=Asia%2FSeoul")
                dd = fetch_json(url)
                n_calls += 1
                h = dd["hourly"]
                df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"]),
                                   "ws_d2": h["wind_speed_10m_previous_day2"]})
                df["kpx_group"], df["dir"] = group, d
                frames.append(df)
        print(f"group{group} 이웃 4점 완료", flush=True)
    nb = pd.concat(frames, ignore_index=True)
    nb.to_csv(OUT_DIR / "jma_msm_neighbors.csv", index=False, encoding="utf-8-sig")
    print(f"이웃 격자 저장: {len(nb)}행, non-null {nb['ws_d2'].notna().mean():.3f}")

    # b. previous_day3 (centroid)
    frames = []
    for group in [1, 2, 3]:
        clat = round(float(centroids.loc[group, "lat"]), 4)
        clon = round(float(centroids.loc[group, "lon"]), 4)
        for model, years in [("jma_gsm", ["2022", "2023", "2024", "2025"]),
                             ("jma_msm", ["2023", "2024", "2025"])]:
            for year in years:
                url = (f"{BASE_URL}?latitude={clat}&longitude={clon}&models={model}"
                       f"&hourly=wind_speed_10m_previous_day3&wind_speed_unit=ms"
                       f"&start_date={year}-01-01&end_date={year}-12-31&timezone=Asia%2FSeoul")
                dd = fetch_json(url)
                n_calls += 1
                h = dd["hourly"]
                df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"]),
                                   "ws_d3": h["wind_speed_10m_previous_day3"]})
                df["kpx_group"], df["model"] = group, model
                frames.append(df)
        print(f"group{group} day3 완료", flush=True)
    d3 = pd.concat(frames, ignore_index=True)
    d3.to_csv(OUT_DIR / "jma_day3.csv", index=False, encoding="utf-8-sig")
    cov = d3.groupby("model")["ws_d3"].apply(lambda s: s.notna().mean().round(3))
    print(f"day3 저장: {len(d3)}행, non-null\n{cov.to_string()}")

    meta = {"note": "exp-021 추가 수집 (MSM 이웃 4점 d2, GSM/MSM centroid d3)",
            "offsets": OFFSETS, "fetched_at": pd.Timestamp.now().isoformat(), "n_api_calls": n_calls}
    with open(OUT_DIR / "fetch_meta_exp021.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"API 호출 {n_calls}회")


if __name__ == "__main__":
    main()
