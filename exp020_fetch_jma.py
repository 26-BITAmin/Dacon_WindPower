# ═══════════════════════════════════════════════════════
# exp-020 데이터 수집: Open-Meteo Previous Runs API — JMA GSM/MSM 예보
#
# 규정 적합성 (대회 규칙: 외부 데이터는 공개·라이선스 준수·예측기준시점 이전 생성):
#   - previous_day2 = 밸리드 시점 48시간 전 초기화 예보 → 전날 13:00(예측기준시점)
#     이전에 발행 완료. 재분석·실측 아님 (대회 제공 LDAPS/GFS와 동일 성격의 예보).
#   - previous_day1(24h 전)은 밸리드 당일 저녁 시간대에 대해 전날 13:00 이후 발행일 수
#     있어 기준 위반 소지 → 참고용으로만 저장, 피처로는 day2만 사용.
#   - 라이선스: CC-BY 4.0 (2026-07-22 https://open-meteo.com/en/terms 확인),
#     무료 티어 10,000콜/일. 본 수집은 ~30콜.
#   - 커버리지 (2026-07-22 프로브): jma_gsm 2022~2025 전 기간 / jma_msm 2023~2025
# 실행: C:\miniconda3\python.exe -X utf8 exp020_fetch_jma.py
# 출력: open (1)/external/jma_prevruns.csv + fetch_meta.json
# ═══════════════════════════════════════════════════════
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from exp019_multigrid import _centroids
from exp005_qopt import DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = DATA_DIR / "external"
OUT_DIR.mkdir(exist_ok=True)

MODELS = {"jma_gsm": ["2022", "2023", "2024", "2025"],
          "jma_msm": ["2023", "2024", "2025"]}
HOURLY_VARS = ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "surface_pressure"]
LEADS = ["previous_day2", "previous_day1"]
BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch_json(url, retries=3):
    for attempt in range(retries):
        r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "--max-time", "120", url],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            d = json.loads(r.stdout)
            if "hourly" in d:
                return d
            if "reason" in d:
                raise RuntimeError(f"API error: {d['reason']}")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch 실패: {url[:120]}")


def main():
    centroids = _centroids()
    hourly_param = ",".join(f"{v}_{lead}" for v in HOURLY_VARS for lead in LEADS)

    frames = []
    n_calls = 0
    for group in [1, 2, 3]:
        lat = round(float(centroids.loc[group, "lat"]), 4)
        lon = round(float(centroids.loc[group, "lon"]), 4)
        for model, years in MODELS.items():
            for year in years:
                url = (f"{BASE_URL}?latitude={lat}&longitude={lon}&models={model}"
                       f"&hourly={hourly_param}&wind_speed_unit=ms"
                       f"&start_date={year}-01-01&end_date={year}-12-31&timezone=Asia%2FSeoul")
                d = fetch_json(url)
                n_calls += 1
                h = d["hourly"]
                df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"])})
                for v in HOURLY_VARS:
                    for lead in LEADS:
                        short = {"previous_day2": "d2", "previous_day1": "d1"}[lead]
                        df[f"{v}_{short}"] = h.get(f"{v}_{lead}")
                df["kpx_group"] = group
                df["model"] = model
                nn = df[f"wind_speed_10m_d2"].notna().sum()
                print(f"group{group} {model} {year}: {nn}/{len(df)} non-null ws_d2 "
                      f"(grid {d['latitude']},{d['longitude']}, elev {d.get('elevation')})", flush=True)
                frames.append(df)
                time.sleep(1)

    out = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / "jma_prevruns.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta = {
        "source": "Open-Meteo Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api)",
        "license": "CC-BY 4.0 (https://open-meteo.com/en/terms, 2026-07-22 확인)",
        "fetched_at": pd.Timestamp.now().isoformat(),
        "models": MODELS,
        "variables": HOURLY_VARS,
        "leads": LEADS,
        "lead_compliance_note": "피처로는 previous_day2(밸리드 48h 전 예보)만 사용 — 예측기준시점(전날 13:00) 이전 발행 보장. previous_day1은 참고 저장.",
        "coordinates": {int(g): [round(float(centroids.loc[g, 'lat']), 4), round(float(centroids.loc[g, 'lon']), 4)] for g in [1, 2, 3]},
        "url_template": f"{BASE_URL}?latitude={{lat}}&longitude={{lon}}&models={{model}}&hourly={hourly_param}&wind_speed_unit=ms&start_date={{Y}}-01-01&end_date={{Y}}-12-31&timezone=Asia/Seoul",
        "n_api_calls": n_calls,
    }
    with open(OUT_DIR / "fetch_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path} ({len(out)}행), fetch_meta.json (API 호출 {n_calls}회)")


if __name__ == "__main__":
    main()
