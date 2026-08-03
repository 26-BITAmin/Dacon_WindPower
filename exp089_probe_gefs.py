# ═══════════════════════════════════════════════════════
# exp-089 프로브: NOAA GEFS(미국 앙상블) 아카이브 가용성 — 수집 착수 전 go/no-go
#
# 왜: exp-048이 Open-Meteo 경유 타 모델은 전부 2024-01 이후만 커버해 축을 닫았고,
#   기관 원본 축(ECMWF open-data S3)만 2023+ 확보돼 exp-054/073으로 이어졌다.
#   NOAA GEFS v12는 AWS 공개 버킷(noaa-gefs-pds)에 있고 **2020-09 이후 운영 아카이브**가
#   있다고 알려져 있어 2022~2025 전 구간 커버 가능성이 있다 — 사실이면 처음으로
#   **train 전 구간을 덮는 두 번째 앙상블 소스**가 된다.
#
# 규정: exp-054/073과 동일 논리 — 공개 아카이브 + 예측기준시점(D-1 13:00 KST) 이전 발표분만.
#   GEFS 00z는 발표가 +4h(=13:00 KST)로 경계선이라 **D-2 12z**를 쓴다(ECMWF와 동일 기준).
#
# 이 스크립트는 **다운로드하지 않는다.** HEAD로 존재·크기만 확인.
#
# 실행: C:\miniconda3\python.exe -X utf8 exp089_probe_gefs.py
# ═══════════════════════════════════════════════════════
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# 타깃일 D의 01~24시(KST)를 덮으려면 D-2 12z 기준 리드타임 ~27~50h
PROBE_DATES = ["20220115", "20220715", "20230115", "20230715",
               "20240115", "20240715", "20250115", "20250715", "20251215"]
# 후보 경로 (v12 구조). 0p50 pgrb2a에 10m u/v 포함.
PATTERNS = [
    ("pgrb2ap5 cf", "{b}/gefs.{d}/12/atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f030"),
    ("pgrb2ap5 p01", "{b}/gefs.{d}/12/atmos/pgrb2ap5/gep01.t12z.pgrb2a.0p50.f030"),
    ("pgrb2sp25 cf", "{b}/gefs.{d}/12/atmos/pgrb2sp25/gec00.t12z.pgrb2s.0p25.f030"),
]


def head(url, timeout=25):
    """curl -I. 반환: (http코드, content-length)"""
    try:
        r = subprocess.run(
            ["curl", "-sI", "--ssl-no-revoke", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return "timeout", 0
    code, size = "000", 0
    for line in r.stdout.splitlines():
        ls = line.strip().lower()
        if ls.startswith("http/"):
            code = line.split()[1]
        elif ls.startswith("content-length:"):
            size = int(line.split(":")[1].strip())
    return code, size


def main():
    t0 = time.time()
    print("### exp-089 프로브: NOAA GEFS 아카이브 가용성 (다운로드 없음) ###", flush=True)
    print(f"버킷 {BASE}\n", flush=True)

    for name, pat in PATTERNS:
        print(f"── {name} ──", flush=True)
        ok = 0
        for d in PROBE_DATES:
            url = pat.format(b=BASE, d=d)
            code, size = head(url)
            mark = "O" if code == "200" else "X"
            if code == "200":
                ok += 1
            print(f"  {mark} {d} 12z  http {code}  {size/1e6:.1f}MB", flush=True)
        print(f"  → {ok}/{len(PROBE_DATES)} 존재\n", flush=True)

    print(f"총 소요: {time.time()-t0:.0f}s")
    print("\n판정 기준: 2022·2023 날짜가 200이면 **train 전 구간 커버 = 축 개방**.")
    print("           2024 이후만 200이면 exp-022/048과 동일 조건 → 축 폐쇄.")


if __name__ == "__main__":
    main()
