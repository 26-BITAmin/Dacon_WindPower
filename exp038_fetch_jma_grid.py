# ═══════════════════════════════════════════════════════
# exp-038 데이터 수집: JMA GSM/MSM **다격자** 예보 (Open-Meteo Previous Runs)
#
# 배경: exp-020은 JMA를 그룹 centroid 1점씩만 받아 GBM 집계 피처로 씀(LB +0.0044).
#   exp-035에서 원시 다격자 편차(spd)가 DL 파트너로 크게 성공(LB +0.0051), exp-037에서
#   GFS 변수 확장은 정보 중복으로 무효. JMA는 **다른 예보 모델**이라 GFS/LDAPS와 정보가
#   덜 중복 → JMA를 다격자로 받아 DL 편차 입력에 넣으면 탈상관 확대 여지(두 성공의 교집합).
#
# 규정 적합성 (exp-020과 동일 프로토콜):
#   - previous_day2(밸리드 48h 전 초기화) = 전날 13:00(예측기준시점) 이전 발행 보장. 재분석 아님.
#   - previous_day1은 미수집(당일 저녁 위반 소지). CC-BY 4.0, 무료 10,000콜/일 중 63콜.
#   - 격자: 3그룹 중심 주변 3×3. GSM ±0.5°(0.5° 격자, span 1.0°≈110km 광역),
#     MSM ±0.1°(5km 격자, 조밀). 프로브(exp038_probe_grid)에서 각 9개 고유 격자 확인.
# 실행: C:\miniconda3\python.exe -X utf8 exp038_fetch_jma_grid.py
# 출력: open (1)/external/jma_grid_prevruns.csv + fetch_meta_exp038.json
# ═══════════════════════════════════════════════════════
import json
import subprocess
import sys
import time

import pandas as pd

from exp019_multigrid import _centroids
from exp005_qopt import DATA_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = DATA_DIR / "external"
OUT_DIR.mkdir(exist_ok=True)
BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

VARS = ["wind_speed_10m", "wind_direction_10m", "temperature_2m", "surface_pressure"]
GRID_CFG = {
    "jma_gsm": {"years": ["2022", "2023", "2024", "2025"], "offs": [-0.5, 0.0, 0.5]},
    "jma_msm": {"years": ["2023", "2024", "2025"],         "offs": [-0.1, 0.0, 0.1]},
}


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
    c = _centroids()
    clat = round(float(c[["lat"]].mean().iloc[0]), 4)
    clon = round(float(c[["lon"]].mean().iloc[0]), 4)
    hourly = ",".join(f"{v}_previous_day2" for v in VARS)

    frames, n_calls = [], 0
    grid_ids = {}   # model → {(glat,glon): idx}
    for model, cfg in GRID_CFG.items():
        grid_ids[model] = {}
        for dlat in cfg["offs"]:
            for dlon in cfg["offs"]:
                rlat, rlon = round(clat + dlat, 4), round(clon + dlon, 4)
                for year in cfg["years"]:
                    url = (f"{BASE}?latitude={rlat}&longitude={rlon}&models={model}"
                           f"&hourly={hourly}&wind_speed_unit=ms"
                           f"&start_date={year}-01-01&end_date={year}-12-31&timezone=Asia%2FSeoul")
                    d = fetch_json(url)
                    n_calls += 1
                    glat, glon = round(d["latitude"], 4), round(d["longitude"], 4)
                    gkey = (glat, glon)
                    if gkey not in grid_ids[model]:
                        grid_ids[model][gkey] = len(grid_ids[model])
                    gidx = grid_ids[model][gkey]
                    h = d["hourly"]
                    df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"])})
                    for v in VARS:
                        df[v] = h.get(f"{v}_previous_day2")
                    df["model"] = model
                    df["grid_idx"] = gidx
                    df["glat"], df["glon"] = glat, glon
                    frames.append(df)
                    time.sleep(0.5)
            print(f"{model} req({dlat:+.2f}) 완료, 누적 콜 {n_calls}", flush=True)
        print(f"{model}: 고유 격자 {len(grid_ids[model])}개 {list(grid_ids[model])}", flush=True)

    out = pd.concat(frames, ignore_index=True)
    # 같은 격자로 매핑된 중복 요청 제거 (격자·시각 유일)
    out = out.drop_duplicates(["forecast_kst_dtm", "model", "grid_idx"]).reset_index(drop=True)
    out_path = OUT_DIR / "jma_grid_prevruns.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta = {
        "source": "Open-Meteo Previous Runs API (https://open-meteo.com/en/docs/previous-runs-api)",
        "license": "CC-BY 4.0 (https://open-meteo.com/en/terms)",
        "fetched_at": pd.Timestamp.now().isoformat(),
        "purpose": "exp-038: JMA GSM/MSM 다격자 → DL 편차 입력 (exp-020 단일점의 다격자 확장)",
        "models": {m: cfg["years"] for m, cfg in GRID_CFG.items()},
        "variables": VARS,
        "lead": "previous_day2 only (전날 13:00 이전 발행 보장)",
        "grid_offsets": {m: cfg["offs"] for m, cfg in GRID_CFG.items()},
        "grids": {m: {f"{k[0]},{k[1]}": v for k, v in gd.items()} for m, gd in grid_ids.items()},
        "center": [clat, clon],
        "n_api_calls": n_calls,
    }
    with open(OUT_DIR / "fetch_meta_exp038.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path} ({len(out)}행), API 호출 {n_calls}회")
    print(f"기간: {out['forecast_kst_dtm'].min()} ~ {out['forecast_kst_dtm'].max()}")
    for m in GRID_CFG:
        sub = out[out["model"] == m]
        print(f"  {m}: {sub['grid_idx'].nunique()}격자, {len(sub)}행, ws 결측 {sub['wind_speed_10m'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
